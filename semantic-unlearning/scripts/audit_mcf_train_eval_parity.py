#!/usr/bin/env python3
"""Why do 26 of 50 MCF rewrites stay above epsilon even under the genie?

Under ground-truth routing, only 24 of the 50 canonical rewrite prompts reach a
complete-answer probability below 1e-6. The rewrites are TRAINING prompts, so
before calling the other 26 training-budget failures, each one has to be
traced through the four places training and evaluation can disagree:

  metric       Training drives exp(-mean answer-token NLL) -- the geometric-
               mean token probability -- below epsilon on the hardest view.
               The evaluator reports exp(-sum NLL), the complete-answer
               probability. For m answer tokens the complete probability is
               the geometric mean to the power m, so it is never larger. A row
               that met epsilon in training therefore CANNOT exceed epsilon at
               evaluation through the metric definition alone; if it does,
               something else differs. The audit reports both metrics on both
               tokenizations so this is checked, not assumed.

  tokens       Training jointly tokenizes prompt + " " + object with BOS
               (_encode_answer_example). The evaluator tokenizes the prompt
               with BOS and " " + answer separately, then concatenates. These
               usually agree for a space-prefixed BPE word, but not always, and
               any difference moves the scored context.

  boundary     Training intervenes at (first labeled answer token) - 1
               (batched_answer_nll); the evaluator at len(prompt ids) - 1.
               Equal iff the prompt token sequences have equal length.

  state/dtype  Training ran in float32 and restores the selected checkpoint
               before saving. If the saved artifact, rescored now with
               training's own tokens, no longer meets epsilon on a row the
               training report called passing, the saved rows do not reproduce
               the trained state. A float32 pass that fails in bfloat16 is a
               deployment-precision effect.

Each rewrite gets exactly one classification, checked in this order:

  passes_eval                 complete-answer probability < epsilon at eval
  training_reported_failing   the selected checkpoint's report lists this fact
                              as failing: an optimization or budget outcome,
                              to be reported as such in the paper
  artifact_not_reproducing    report says passing, but training-style scoring
                              of the saved artifact now fails
  tokenization_or_boundary    training-style passes, eval-style fails, and the
                              token sequences or boundary differ
  precision                   passes in float32, fails only in bfloat16
  unexplained                 none of the above; investigate before citing

Genie routing is used throughout (the gold row is forced), so routing cannot
contribute: V2 and the genie already agree on all 50 rewrites.

`--static-only` runs steps that need no model -- training-report extraction and
the token/boundary comparison -- in seconds on CPU.

Usage
-----
python -u scripts/audit_mcf_train_eval_parity.py \
  --run-dir outputs/mcf_fact_assoc_router_v2_seed1 \
  --output-dir outputs/mcf_fact_assoc_router_v2_seed1/parity_audit \
  --dtypes float32,bfloat16 --device cuda --local-files-only
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path


CANONICAL_ROLE = "canonical_rewrite"


# ------------------------------------------------------------- static part

def selected_gate(report):
    """The gate whose checkpoint was restored and saved."""
    best = int(report["best_step"])
    gates = [g for g in report.get("gates", []) if int(g.get("step", -1)) == best]
    if not gates:
        raise ValueError(f"No gate recorded at best_step={best}")
    return gates[-1]


def training_status(report):
    gate = selected_gate(report)
    train = gate["metrics"]["train"]
    return {
        "stop_reason": report.get("stop_reason"),
        "best_step": int(report["best_step"]),
        "globally_feasible_at_best": gate.get("globally_feasible"),
        "facts_passing": train.get("facts_passing"),
        "facts_total": train.get("facts_total"),
        "failing_fact_ids": list(train.get("failing_fact_ids", [])),
        "target_probability": float(train["target_probability"]),
        "metric_definition": train.get("metric_definition"),
    }


def split_training_example(example):
    """Recover prompt ids, answer ids and boundary exactly as training used them."""
    ids = [int(t) for t in example["input_ids"]]
    labels = [int(t) for t in example["labels"]]
    positions = [i for i, label in enumerate(labels) if label != -100]
    if not positions:
        raise ValueError(f"{example['id']} has no labeled answer tokens")
    if positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError(f"{example['id']} answer labels are not contiguous")
    boundary = positions[0]
    return {
        "full_ids": ids,
        "prompt_ids": ids[:boundary],
        "answer_ids": [ids[i] for i in positions],
        "boundary": boundary,
        "trailing_tokens_after_answer": len(ids) - 1 - positions[-1],
    }


def evaluator_tokens(tokenizer, prompt, answer):
    """The decomposition evaluator's tokenization (score_batch)."""
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(" " + str(answer).strip(), add_special_tokens=False)["input_ids"]
    return {
        "full_ids": [*prompt_ids, *answer_ids],
        "prompt_ids": list(prompt_ids),
        "answer_ids": list(answer_ids),
        "boundary": len(prompt_ids),
    }


def compare_tokens(train, evaluate, bos_id=None):
    first_diff = next(
        (i for i, (a, b) in enumerate(zip(train["full_ids"], evaluate["full_ids"])) if a != b),
        None,
    )
    if first_diff is None and len(train["full_ids"]) != len(evaluate["full_ids"]):
        first_diff = min(len(train["full_ids"]), len(evaluate["full_ids"]))
    return {
        "identical_full_sequence": train["full_ids"] == evaluate["full_ids"],
        "identical_prompt_ids": train["prompt_ids"] == evaluate["prompt_ids"],
        "identical_answer_ids": train["answer_ids"] == evaluate["answer_ids"],
        "train_boundary": train["boundary"],
        "eval_boundary": evaluate["boundary"],
        "boundary_equal": train["boundary"] == evaluate["boundary"],
        "train_answer_tokens": len(train["answer_ids"]),
        "eval_answer_tokens": len(evaluate["answer_ids"]),
        "first_differing_position": first_diff,
        "train_bos_count": (
            train["prompt_ids"].count(bos_id) if bos_id is not None else None
        ),
        "eval_bos_count": (
            evaluate["prompt_ids"].count(bos_id) if bos_id is not None else None
        ),
    }


# ---------------------------------------------------------- classification

def classify(fact_id, eps, status, scores, token_cmp, primary="float32"):
    """Assign exactly one explanation, in a fixed order of precedence.

    scores[dtype] = {"train_geo", "train_full", "eval_geo", "eval_full"}
    """
    primary_scores = scores.get(primary)
    if primary_scores is None:
        return "not_scored", "no model pass in the training dtype"
    if primary_scores["eval_full"] < eps:
        # Could still fail in another dtype; report that as a note.
        failing_elsewhere = [
            d for d, s in scores.items() if d != primary and s["eval_full"] >= eps
        ]
        if failing_elsewhere:
            return "precision", (
                f"passes in {primary}, fails in {', '.join(failing_elsewhere)}"
            )
        return "passes_eval", "complete-answer probability below epsilon"
    if fact_id in set(status["failing_fact_ids"]):
        return "training_reported_failing", (
            f"selected checkpoint (step {status['best_step']}, stop reason "
            f"{status['stop_reason']}) lists this fact as failing"
        )
    if primary_scores["train_geo"] >= eps:
        return "artifact_not_reproducing", (
            "training report says passing, but the saved rows rescored on "
            "training's own tokens no longer meet epsilon"
        )
    if not token_cmp["identical_full_sequence"] or not token_cmp["boundary_equal"]:
        return "tokenization_or_boundary", (
            f"training-style passes, eval-style fails; first differing token at "
            f"position {token_cmp['first_differing_position']}, boundaries "
            f"{token_cmp['train_boundary']} vs {token_cmp['eval_boundary']}"
        )
    return "unexplained", (
        "identical tokens and boundary, training-style passes and eval-style "
        "fails: investigate before citing"
    )


# -------------------------------------------------------------- model part

def answer_logprobs(model, ids, boundary, n_answer, device):
    """Log-probabilities of the answer tokens under a fixed request boundary."""
    import torch
    from torch.nn import functional as F

    sequence = torch.tensor([ids], dtype=torch.long, device=device)
    model.set_association_prefix_lengths([boundary])
    logits = model(
        input_ids=sequence, attention_mask=torch.ones_like(sequence), use_cache=False
    ).logits[0].float()
    log_probs = F.log_softmax(logits, dim=-1)
    values = []
    for offset in range(n_answer):
        position = boundary + offset
        values.append(float(log_probs[position - 1, ids[position]]))
    return values


def score_pair(model, device, train, evaluate):
    out = {}
    for label, tokens in (("train", train), ("eval", evaluate)):
        lp = answer_logprobs(
            model, tokens["full_ids"], tokens["boundary"], len(tokens["answer_ids"]), device
        )
        out[f"{label}_geo"] = math.exp(sum(lp) / len(lp))
        out[f"{label}_full"] = math.exp(sum(lp))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtypes", default="float32,bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    manifest = json.loads((run_dir / "association_manifest.json").read_text())
    report = json.loads((run_dir / "training_report.json").read_text())
    examples = json.loads((run_dir / "association_examples.json").read_text())
    status = training_status(report)
    eps = status["target_probability"]

    import torch
    artifact = torch.load(
        run_dir / "fact_association_embeddings.pt", map_location="cpu", weights_only=False
    )
    facts = list(artifact["facts"])
    row_of = {fact["id"]: index for index, fact in enumerate(facts)}

    canonical = {
        ex["fact_id"]: ex for ex in examples
        if ex.get("split") == "train" and (
            ex.get("group") == CANONICAL_ROLE or str(ex.get("id", "")).endswith(CANONICAL_ROLE)
        )
    }
    missing = [f["id"] for f in facts if f["id"] not in canonical]
    if missing:
        raise SystemExit(f"No canonical training example for {len(missing)} facts: {missing[:5]}")

    from transformers import AutoTokenizer

    model_path = Path(manifest["model_path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    for fact in facts:
        example = canonical[fact["id"]]
        train = split_training_example(example)
        evaluate = evaluator_tokens(tokenizer, example["prompt"], fact["object"])
        rows.append({
            "fact_id": fact["id"],
            "row": row_of[fact["id"]],
            "prompt": example["prompt"],
            "answer": fact["object"],
            "training_reported_failing": fact["id"] in set(status["failing_fact_ids"]),
            "tokens": compare_tokens(train, evaluate, tokenizer.bos_token_id),
            "_train": train,
            "_eval": evaluate,
            "scores": {},
        })

    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    if not args.static_only:
        from transformers import AutoModelForCausalLM

        from oracle_router_gate import ForcedRowBank
        from static_overlap_fact_association_embeddings import AssociationCausalLM

        for dtype in dtypes:
            base = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=getattr(torch, dtype),
                local_files_only=args.local_files_only,
                attn_implementation="eager",
            ).to(args.device).eval()
            base.requires_grad_(False)
            bank = ForcedRowBank(
                base, int(artifact["layer"]), artifact["rows"],
                artifact["subject_patterns"], facts,
            )
            model = AssociationCausalLM(base, bank).eval()
            with torch.no_grad():
                for row in rows:
                    bank.forced_row = row["row"]
                    genie = score_pair(model, args.device, row["_train"], row["_eval"])
                    bank.forced_row = None
                    base_scores = score_pair(model, args.device, row["_train"], row["_eval"])
                    row["scores"][dtype] = {
                        **genie,
                        "base_train_geo": base_scores["train_geo"],
                        "base_eval_full": base_scores["eval_full"],
                    }
            bank.close()
            del model, bank, base
            if args.device == "cuda":
                torch.cuda.empty_cache()

    primary = "float32" if "float32" in dtypes else dtypes[0]
    for row in rows:
        category, reason = classify(
            row["fact_id"], eps, status, row["scores"], row["tokens"], primary=primary
        )
        row["classification"] = category
        row["reason"] = reason
        del row["_train"], row["_eval"]

    counts = Counter(row["classification"] for row in rows)
    token_summary = {
        "identical_full_sequence": sum(r["tokens"]["identical_full_sequence"] for r in rows),
        "boundary_equal": sum(r["tokens"]["boundary_equal"] for r in rows),
        "identical_answer_ids": sum(r["tokens"]["identical_answer_ids"] for r in rows),
        "facts": len(rows),
    }
    audit = {
        "schema_version": "mcf_train_eval_parity_v1",
        "run_dir": str(run_dir),
        "epsilon": eps,
        "training": status,
        "token_parity": token_summary,
        "dtypes_scored": [] if args.static_only else dtypes,
        "primary_dtype": primary,
        "classification_counts": dict(counts),
        "rows": rows,
        "interpretation": (
            "training_reported_failing rows are an optimization/budget outcome "
            "and should be reported as a count. tokenization_or_boundary and "
            "artifact_not_reproducing rows are bugs to fix before any MCF number "
            "is cited. unexplained rows need manual inspection."
        ),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "mcf_train_eval_parity.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "status": "parity_audit_complete",
        "training": {k: status[k] for k in (
            "stop_reason", "best_step", "globally_feasible_at_best",
            "facts_passing", "facts_total",
        )},
        "token_parity": token_summary,
        "classification_counts": dict(counts),
        "output": str(output / "mcf_train_eval_parity.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
