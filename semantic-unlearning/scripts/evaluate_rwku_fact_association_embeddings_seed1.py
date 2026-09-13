#!/usr/bin/env python3
"""Evaluate the saved RWKU Batch-50 seed-1 fact-association residual bank.

Headline RWKU-style outputs:
- same-50 efficacy recovery/ROUGE-L plus sensitive-answer probability;
- content-disjoint Level-1/Level-2 generalization;
- held-out Level-2 deterministic paraphrase;
- native Level-3 adversarial recovery;
- native neighbor locality;
- runtime-aligned and legacy Wikidata PPL;
- route activity for every evaluated split.

This is evaluation of the explicit RWKU-Batch-50-v1 method extension, not a
claim that the native target-only RWKU training protocol was used unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import rwku_eval as rwku
from rwku_batch50 import build_batch_split
from rwku_fact_association_embeddings import (
    association_key_from_row,
    build_association_facts,
)
from mcf_zero_unlearn_official_eval import (
    dtype_from_str,
    load_official_ppl_text,
    official_perplexity,
    runtime_aligned_perplexity,
)
from static_overlap_fact_association_embeddings import load_artifact_into_model


def _counter_delta(after, before):
    return {
        "hook_calls": after["hook_calls"] - before["hook_calls"],
        "active_batch_rows": after["active_batch_rows"] - before["active_batch_rows"],
        "active_token_positions": (
            after["active_token_positions"] - before["active_token_positions"]
        ),
        "active_fact_counts": [
            a - b
            for a, b in zip(
                after["active_fact_counts"], before["active_fact_counts"]
            )
        ],
    }


@torch.no_grad()
def generate_fixed_boundary(model, bank, tokenizer, prompt, max_new_tokens=30):
    device = next(model.parameters()).device
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        return_token_type_ids=False,
    ).to(device)
    prompt_width = int(encoded["input_ids"].shape[1])
    generated = model.generate_uncached_fixed_boundary(
        encoded["input_ids"],
        encoded.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    route = list(bank.last_active_fact_indices[0])
    continuation = generated[0, prompt_width:]
    return tokenizer.decode(continuation, skip_special_tokens=True).strip(), route


@torch.no_grad()
def score_answer_fixed_boundary(model, tokenizer, prompt, answer):
    device = next(model.parameters()).device
    prompt_ids = rwku._token_ids(tokenizer, prompt, add_special_tokens=True)
    answer_ids = rwku._token_ids(
        tokenizer,
        rwku._normalized_completion(str(answer)),
        add_special_tokens=False,
    )
    if not prompt_ids or not answer_ids:
        raise ValueError("RWKU scoring received empty prompt/answer tokenization")
    sequence = torch.tensor(
        [[*prompt_ids, *answer_ids]], dtype=torch.long, device=device
    )
    attention = torch.ones_like(sequence)
    model.set_association_prefix_lengths([len(prompt_ids)])
    output = model(
        input_ids=sequence,
        attention_mask=attention,
        use_cache=False,
    )
    positions = torch.arange(
        len(prompt_ids) - 1,
        len(prompt_ids) + len(answer_ids) - 1,
        device=device,
    )
    logits = output.logits[0, positions, :].float()
    targets = torch.tensor(answer_ids, dtype=torch.long, device=device)
    log_probs = F.log_softmax(logits, dim=-1)
    target_lp = log_probs.gather(1, targets[:, None]).squeeze(1)
    predicted = logits.argmax(dim=-1)
    return {
        "sum_logprob": float(target_lp.sum().cpu()),
        "mean_logprob": float(target_lp.mean().cpu()),
        "geometric_probability": float(target_lp.mean().exp().cpu()),
        "first_token_probability": float(target_lp[0].exp().cpu()),
        "sensitive_token_top1_accuracy": float(
            100.0 * (predicted == targets).float().mean().cpu()
        ),
        "sensitive_correct_tokens": int((predicted == targets).sum().cpu()),
        "sensitive_total_tokens": len(answer_ids),
    }


def evaluate_rows(
    model,
    bank,
    tokenizer,
    rows,
    *,
    expected_source_to_row=None,
    score_answers=True,
    max_new_tokens=30,
):
    details = []
    for row in rows:
        prompt = rwku.format_qa_prompt(tokenizer, row)
        prediction, route = generate_fixed_boundary(
            model,
            bank,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
        )
        item = {
            "source_record_sha256": str(row.get("source_record_sha256", "")),
            "subject": str(row.get("subject", "")),
            "level": str(row.get("level", "")),
            "query": str(row["query"]),
            "answer": str(row["answer"]),
            "prediction": prediction or "NOANSWER",
            "recovery_success": rwku.recovery_success(
                prediction, str(row["answer"])
            ),
            "rouge_l_recall": rwku.rouge_l_recall(
                prediction, str(row["answer"])
            ),
            "active_fact_rows": route,
            "route_active": bool(route),
        }
        source_hash = str(row.get("source_record_sha256", ""))
        if expected_source_to_row is not None and source_hash in expected_source_to_row:
            expected = int(expected_source_to_row[source_hash])
            item["expected_association_row"] = expected
            item["route_correct"] = expected in route
        else:
            item["expected_association_row"] = None
            item["route_correct"] = None
        if score_answers:
            item.update(
                score_answer_fixed_boundary(
                    model, tokenizer, prompt, str(row["answer"])
                )
            )
        details.append(item)

    count = len(details)
    summary = {
        "count": count,
        "recovery_accuracy": (
            100.0 * sum(bool(x["recovery_success"]) for x in details) / count
            if count else None
        ),
        "rouge_l_recall": (
            100.0 * sum(float(x["rouge_l_recall"]) for x in details) / count
            if count else None
        ),
        "route_active_fraction": (
            sum(bool(x["route_active"]) for x in details) / count
            if count else None
        ),
    }
    expected = [
        bool(x["route_correct"])
        for x in details
        if x.get("route_correct") is not None
    ]
    summary["route_correct_fraction"] = (
        sum(expected) / len(expected) if expected else None
    )
    scored = [x for x in details if "geometric_probability" in x]
    if scored:
        summary.update(
            {
                "answer_geometric_probability": sum(
                    float(x["geometric_probability"]) for x in scored
                )
                / len(scored),
                "answer_first_token_probability": sum(
                    float(x["first_token_probability"]) for x in scored
                )
                / len(scored),
                "sensitive_token_top1_accuracy": sum(
                    int(x["sensitive_correct_tokens"]) for x in scored
                )
                * 100.0
                / sum(int(x["sensitive_total_tokens"]) for x in scored),
            }
        )
    return summary, details


def _native_rows(split, filename, level):
    rows = []
    for target_split in split["per_target"]:
        for source in target_split["evaluation_only"][filename]:
            row = dict(source)
            row["subject"] = str(row.get("subject") or target_split["subject"])
            row["rwku_target_seed"] = int(target_split["target_seed"])
            row["rwku_target_subject"] = str(target_split["subject"])
            row["level"] = str(level)
            rows.append(row)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--data-root", default="data/rwku")
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--skip-level3", action="store_true")
    p.add_argument("--skip-neighbors", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    manifest = json.loads((run_dir / "association_manifest.json").read_text())
    if int(manifest.get("seed", -1)) != 1:
        raise ValueError("Registered RWKU evaluator requires seed 1")
    if int(manifest.get("forget_train_count", -1)) != 50:
        raise ValueError("Registered RWKU evaluator requires 50 forget rows")

    artifact = torch.load(
        run_dir / "fact_association_embeddings.pt",
        map_location="cpu",
        weights_only=False,
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(manifest["model_path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    split = build_batch_split(
        data_root=Path(args.data_root).resolve(),
        batch_seed=1,
        allow_download=not args.no_download,
    )
    forget_rows = list(split["efficacy_forget"])
    expected_facts, record_to_fact_id, _ = build_association_facts(
        forget_rows, tokenizer
    )
    expected_keys = [str(f["association_key"]) for f in expected_facts]
    artifact_keys = [str(f.get("association_key")) for f in artifact["facts"]]
    if artifact_keys != expected_keys:
        raise RuntimeError(
            "Saved RWKU association bank no longer matches the frozen seed-1 Batch-50 split"
        )

    dtype = dtype_from_str(args.dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    base_model.requires_grad_(False)
    base_model.config.use_cache = False
    model, bank = load_artifact_into_model(base_model, artifact)
    model.eval()

    fact_to_row = {
        fact["id"]: index for index, fact in enumerate(expected_facts)
    }
    expected_source_to_row = {
        source_hash: fact_to_row[fact_id]
        for source_hash, fact_id in record_to_fact_id.items()
    }

    same50, same50_detail = evaluate_rows(
        model,
        bank,
        tokenizer,
        forget_rows,
        expected_source_to_row=expected_source_to_row,
        score_answers=True,
    )
    if same50["route_correct_fraction"] != 1.0:
        raise RuntimeError(
            "Reloaded RWKU bank failed exact same-50 routing: "
            f"{same50['route_correct_fraction']}"
        )

    heldout_l1, heldout_l1_detail = evaluate_rows(
        model, bank, tokenizer, split["heldout_level1"], score_answers=True
    )
    heldout_l2, heldout_l2_detail = evaluate_rows(
        model, bank, tokenizer, split["heldout_level2"], score_answers=True
    )
    paraphrase, paraphrase_detail = evaluate_rows(
        model, bank, tokenizer, split["heldout_paraphrase"], score_answers=True
    )

    level3 = level3_detail = None
    if not args.skip_level3:
        level3_rows = _native_rows(split, "forget_level3.json", 3)
        level3, level3_detail = evaluate_rows(
            model,
            bank,
            tokenizer,
            level3_rows,
            score_answers=False,
        )

    neighbors = neighbor_detail = None
    if not args.skip_neighbors:
        neighbor_rows = [
            *_native_rows(split, "neighbor_level1.json", 1),
            *_native_rows(split, "neighbor_level2.json", 2),
        ]
        neighbors, neighbor_detail = evaluate_rows(
            model,
            bank,
            tokenizer,
            neighbor_rows,
            score_answers=False,
        )

    legacy_ppl = runtime_ppl = runtime_ppl_route_activity = None
    if not args.skip_ppl:
        ppl_text = load_official_ppl_text(args.wikidata_dir)
        if ppl_text is not None:
            legacy_ppl = official_perplexity(
                model, tokenizer, ppl_text, next(model.parameters()).device,
                max_input_length=100,
            )
            before = bank.counters()
            runtime = runtime_aligned_perplexity(
                model, tokenizer, ppl_text, next(model.parameters()).device,
                max_input_length=100,
            )
            after = bank.counters()
            runtime_ppl = runtime["ppl"]
            runtime_ppl_route_activity = _counter_delta(after, before)

    result = {
        "method": "FactAssociationBank",
        "dataset": "RWKU",
        "protocol_id": manifest["protocol_id"],
        "protocol_status": manifest["protocol_status"],
        "seed": 1,
        "target_seeds": split["manifest"]["target_seeds"],
        "subjects": [item["subject"] for item in split["manifest"]["targets"]],
        "forget_train_count": 50,
        "unique_forget_association_count": len(expected_facts),
        "architecture": {
            "layer": int(artifact["layer"]),
            "trainable_vectors": len(artifact["facts"]),
            "base_weights_edited": False,
            "input_embeddings_edited": False,
            "lm_head_edited": False,
            "tokenizer_extended": False,
            "fact_id_injection_used": False,
            "boundary": "original formatted RWKU request boundary",
        },
        "same_50_efficacy": same50,
        "heldout_level1": heldout_l1,
        "heldout_level2": heldout_l2,
        "heldout_level2_paraphrase": paraphrase,
        "adversarial_level3": level3,
        "neighbors": neighbors,
        "legacy_PPL": legacy_ppl,
        "runtime_aligned_PPL": runtime_ppl,
        "runtime_aligned_PPL_route_activity": runtime_ppl_route_activity,
        "details": {
            "same_50_efficacy": same50_detail,
            "heldout_level1": heldout_l1_detail,
            "heldout_level2": heldout_l2_detail,
            "heldout_level2_paraphrase": paraphrase_detail,
            "adversarial_level3": level3_detail,
            "neighbors": neighbor_detail,
        },
        "runtime_counters": bank.counters(),
    }

    out = (
        Path(args.out).resolve()
        if args.out
        else run_dir / "official_rwku_batch50_eval.json"
    )
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "same50_recovery": same50["recovery_accuracy"],
                "same50_sensitive_token_top1_accuracy": same50.get(
                    "sensitive_token_top1_accuracy"
                ),
                "same50_route_correct_fraction": same50[
                    "route_correct_fraction"
                ],
                "heldout_l1_recovery": heldout_l1["recovery_accuracy"],
                "heldout_l2_recovery": heldout_l2["recovery_accuracy"],
                "heldout_paraphrase_recovery": paraphrase[
                    "recovery_accuracy"
                ],
                "level3_recovery": (
                    None if level3 is None else level3["recovery_accuracy"]
                ),
                "neighbor_recovery": (
                    None
                    if neighbors is None
                    else neighbors["recovery_accuracy"]
                ),
                "neighbor_route_active_fraction": (
                    None
                    if neighbors is None
                    else neighbors["route_active_fraction"]
                ),
                "runtime_aligned_PPL": runtime_ppl,
                "runtime_aligned_PPL_route_activity": runtime_ppl_route_activity,
                "legacy_PPL": legacy_ppl,
                "out": str(out),
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
