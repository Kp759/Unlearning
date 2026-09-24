#!/usr/bin/env python3
"""Step 0: separate routing failures from actuation failures before fixing either.

This produces G1, G2 and G3 in a single pass, because they share the same
forward machinery and only differ in how the results are grouped.

  G3  activation rate on every evaluation group. Spe and PPL are currently
      byte-identical across base, V1 and V2, which is consistent with either
      "the residual preserves locality" or "the router never fired there". The
      second is just the inactive-path identity restated and proves nothing.
      This measurement decides which claim the paper can make.

  G1  the oracle arm: ground-truth association supplied, so routing is perfect
      by construction and what remains is the actuator's ceiling. The V2-to-
      oracle gap is the total value of all router work. A small gap means most
      router effort is wasted, and that is the single most useful thing to
      learn on day one. The random arm anchors the other end: suppression that
      survives random routing among eligible candidates is relation-generic,
      not fact-specific.

  G2  attribution. Every forget-set failure is labelled by cross-tabulating
      whether the router fired correctly against whether the answer was
      suppressed:

        routed + suppressed      working as intended
        routed + not suppressed  ACTUATION failure -- the residual did not
                                 generalize to this phrasing
        not routed + suppressed  suppressed for an unrelated reason; the base
                                 model was already weak here, so this case
                                 should not be counted as a success
        not routed + not         ROUTING failure -- the fix is candidate
        suppressed               generation or scoring, not the residual

      These two failure modes need opposite fixes, which is why a pooled
      failure rate cannot direct any next step.

Scoring is teacher-forced sensitive-answer probability under the fixed request
boundary, matching the training objective's accounting. It is deliberately not
the official Eff/Gen pipeline: this script attributes failures, and the
official evaluator remains the source of headline numbers.

Usage
-----
python -u scripts/evaluate_router_decomposition.py \
  --run-dir outputs/<run> --mcf-path data/multi_counterfact.json \
  --output-dir outputs/<run>/decomposition \
  --arms v2,oracle,subject_only,random --device cuda
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from mcf_sampling import sample_official_mcf_records
from linear_router import load_router_artifact
from oracle_router_gate import build_oracle_table, load_arm
from static_overlap_fact_association_embeddings import (
    load_artifact_into_model,
    make_subject_patterns,
)
from static_overlap_fact_association_v2_gate import (
    load_relation_prototype_artifact,
)
from static_overlap_natural_writer import mcf_facts


GROUPS = ("rewrite", "paraphrase", "neighborhood", "retain")


def _rewrite(record):
    return record.get("requested_rewrite", record)


def build_records_from_mcf(mcf_path, forget_num, retain_num, seed):
    """Group every evaluation prompt with its gold answer and owning fact.

    `rewrite` and `paraphrase` are forget prompts and should route.
    `neighborhood` shares the answer token by MCF construction but is a
    different subject, so it must NOT route -- it is the locality test.
    `retain` is a disjoint record set and must not route either.
    """
    data = json.loads(Path(mcf_path).read_text())
    forget_records, retain_records = sample_official_mcf_records(
        data, forget_num, retain_num, seed
    )
    facts = mcf_facts(forget_records, "forget")
    index_of = {fact["id"]: position for position, fact in enumerate(facts)}

    rows = []
    for fact, record in zip(facts, forget_records):
        rewrite = _rewrite(record)
        answer = fact["object"]
        rows.append({
            "group": "rewrite",
            "fact_id": fact["id"],
            "fact_index": index_of[fact["id"]],
            "prompt": fact["canonical_prompt"],
            "answer": answer,
            "should_route": True,
        })
        for prompt in record.get("paraphrase_prompts", []):
            rows.append({
                "group": "paraphrase",
                "fact_id": fact["id"],
                "fact_index": index_of[fact["id"]],
                "prompt": str(prompt),
                "answer": answer,
                "should_route": True,
            })
        for prompt in record.get("neighborhood_prompts", []):
            rows.append({
                "group": "neighborhood",
                "fact_id": fact["id"],
                "fact_index": index_of[fact["id"]],
                "prompt": str(prompt),
                "answer": answer,
                # Different subject, same answer token. Firing here is a
                # locality violation, so the gold label is "do not route".
                "should_route": False,
            })
    for record in retain_records:
        rewrite = _rewrite(record)
        target = rewrite["target_true"]
        answer = str(target["str"] if isinstance(target, dict) else target).strip()
        rows.append({
            "group": "retain",
            "fact_id": f"mcf_retain_{int(record['case_id'])}",
            "fact_index": None,
            "prompt": str(rewrite["prompt"]).format(rewrite["subject"]).strip(),
            "answer": answer,
            "should_route": False,
        })
    return facts, rows


@torch.no_grad()
def score_batch(model, bank, tokenizer, rows, device, batch_size=8):
    """Teacher-forced answer probability with the boundary at the request end.

    Returns, per row: the complete-answer probability, all-token top-1
    correctness, and the association the router selected (or None).
    """
    results = []
    for start in range(0, len(rows), int(batch_size)):
        window = rows[start:start + int(batch_size)]
        prompt_ids, full_ids, boundaries = [], [], []
        for row in window:
            p = tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]
            a = tokenizer(
                " " + str(row["answer"]).strip(), add_special_tokens=False
            )["input_ids"]
            if not a:
                raise ValueError(f"Empty answer tokenization: {row['answer']!r}")
            prompt_ids.append(p)
            full_ids.append(p + a)
            boundaries.append(len(p))
        width = max(len(ids) for ids in full_ids)
        padded = torch.full(
            (len(window), width), tokenizer.pad_token_id, dtype=torch.long
        )
        mask = torch.zeros((len(window), width), dtype=torch.long)
        for index, ids in enumerate(full_ids):
            padded[index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask[index, :len(ids)] = 1
        padded, mask = padded.to(device), mask.to(device)

        # The intervention position is the END OF THE REQUEST, not the end of
        # the padded sequence: the appended answer tokens must not move the
        # boundary or the route.
        model.set_association_prefix_lengths(boundaries)
        logits = model(input_ids=padded, attention_mask=mask, use_cache=False).logits
        log_probs = F.log_softmax(logits.float(), dim=-1)

        routes = list(bank.last_active_fact_indices)
        for index, row in enumerate(window):
            boundary = boundaries[index]
            answer_ids = full_ids[index][boundary:]
            total = 0.0
            correct = True
            for offset, token in enumerate(answer_ids):
                position = boundary + offset - 1
                step = log_probs[index, position]
                total += float(step[token])
                if int(step.argmax()) != int(token):
                    correct = False
            active = routes[index] if index < len(routes) else []
            results.append({
                **row,
                "answer_log_prob": total,
                "answer_prob": float(torch.tensor(total).exp()),
                "all_token_correct": bool(correct),
                "routed_index": int(active[0]) if active else None,
                "fired": bool(active),
            })
    return results


def attribute(rows, suppression_threshold):
    """G2: cross-tabulate correct routing against successful suppression."""
    table = defaultdict(int)
    examples = defaultdict(list)
    for row in rows:
        routed_correctly = (
            row["fired"] and row["routed_index"] == row["fact_index"]
        )
        suppressed = row["answer_prob"] < float(suppression_threshold)
        key = (
            "routed" if routed_correctly else "not_routed",
            "suppressed" if suppressed else "not_suppressed",
        )
        table["_".join(key)] += 1
        if len(examples["_".join(key)]) < 10:
            examples["_".join(key)].append({
                "prompt": row["prompt"],
                "answer": row["answer"],
                "answer_prob": row["answer_prob"],
                "routed_index": row["routed_index"],
                "gold_index": row["fact_index"],
            })
    total = sum(table.values()) or 1
    failures = table["routed_not_suppressed"] + table["not_routed_not_suppressed"]
    return {
        "counts": dict(table),
        "rates": {k: v / total for k, v in table.items()},
        "failure_count": failures,
        "actuation_failure_share": (
            table["routed_not_suppressed"] / failures if failures else None
        ),
        "routing_failure_share": (
            table["not_routed_not_suppressed"] / failures if failures else None
        ),
        "suppressed_without_routing": table["not_routed_suppressed"],
        "suppressed_without_routing_note": (
            "Suppressed while the router abstained: the base model was already "
            "weak on this prompt, so it is not evidence the method worked."
        ),
        "examples": dict(examples),
    }


def summarize(rows):
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    summary = {}
    for group, items in sorted(by_group.items()):
        count = len(items)
        should_route = items[0]["should_route"]
        summary[group] = {
            "count": count,
            "should_route": should_route,
            "activation_rate": sum(r["fired"] for r in items) / count,
            # Only defined where routing is the desired behaviour. On a
            # must-not-route group, "routed to the gold index" is a locality
            # violation, not a success, and reporting it as a correctness rate
            # would invert its meaning.
            "correct_route_rate": (
                sum(
                    r["fired"] and r["routed_index"] == r["fact_index"]
                    for r in items
                ) / count
                if should_route else None
            ),
            # On a group that must not route, activation IS the false-activation
            # rate. This is the number missing from the paper.
            "false_activation_rate": (
                None if should_route else sum(r["fired"] for r in items) / count
            ),
            "mean_answer_prob": sum(r["answer_prob"] for r in items) / count,
            "all_token_accuracy": sum(r["all_token_correct"] for r in items) / count,
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mcf-path", default="")
    parser.add_argument(
        "--records", default="", help="precomputed record JSON for non-MCF adapters"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arms", default="base,v2,oracle,subject_only,random")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--suppression-threshold", type=float, default=1e-6)
    parser.add_argument("--max-rows-per-group", type=int, default=0)
    args = parser.parse_args(argv)

    if not args.mcf_path and not args.records:
        raise SystemExit("Pass --mcf-path or --records")

    run_dir = Path(args.run_dir).resolve()
    artifact_path = run_dir / "fact_association_embeddings.pt"
    manifest = json.loads((run_dir / "association_manifest.json").read_text())
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = manifest["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.records:
        payload = json.loads(Path(args.records).read_text())
        facts, rows = payload["facts"], payload["rows"]
    else:
        facts, rows = build_records_from_mcf(
            args.mcf_path, args.forget_num, args.retain_num, args.seed
        )
    if args.max_rows_per_group:
        capped, seen = [], defaultdict(int)
        for row in rows:
            if seen[row["group"]] < int(args.max_rows_per_group):
                capped.append(row)
                seen[row["group"]] += 1
        rows = capped

    # The oracle table covers forget prompts only. Absence from the table is
    # the ground-truth "do not route" signal for neighborhood and retain.
    oracle_table, collisions = build_oracle_table(
        tokenizer,
        {
            row["prompt"]: row["fact_index"]
            for row in rows
            if row["should_route"] and row["fact_index"] is not None
        },
        add_special_tokens=True,
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    report = {
        "schema_version": "router_decomposition_v1",
        "run_dir": str(run_dir),
        "artifact_architecture": artifact.get("architecture"),
        "layer": int(artifact["layer"]),
        "suppression_threshold": args.suppression_threshold,
        "row_count": len(rows),
        "oracle_table_size": len(oracle_table),
        "oracle_table_collisions": len(collisions),
        "arms": {},
    }

    for arm in arms:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=getattr(torch, args.dtype),
            local_files_only=args.local_files_only,
            attn_implementation="eager",
        ).to(args.device)
        base_model.eval()
        base_model.requires_grad_(False)

        if arm == "base":
            # Zeroed rows through the shipped router: the router still runs and
            # is still counted, but the intervention is the identity. This
            # isolates activation from effect.
            neutral = dict(artifact)
            neutral["rows"] = torch.zeros_like(artifact["rows"])
            model, bank = _load_shipped(base_model, neutral)
        elif arm == "v2":
            model, bank = _load_shipped(base_model, artifact)
        else:
            model, bank = load_arm(
                base_model,
                artifact,
                arm,
                oracle_table=oracle_table if arm == "oracle" else None,
            )
        model.eval()

        scored = score_batch(
            model, bank, tokenizer, rows, args.device, batch_size=args.batch_size
        )
        forget_rows = [r for r in scored if r["should_route"]]
        report["arms"][arm] = {
            "groups": summarize(scored),
            "attribution": attribute(forget_rows, args.suppression_threshold),
            "counters": bank.counters(),
        }
        (output / f"rows_{arm}.json").write_text(
            json.dumps(scored, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(
            {"arm": arm, "groups": report["arms"][arm]["groups"]}, indent=2
        ), flush=True)

        bank.close()
        del model, bank, base_model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    if "v2" in report["arms"] and "oracle" in report["arms"]:
        v2 = report["arms"]["v2"]["groups"]
        oracle = report["arms"]["oracle"]["groups"]
        report["v2_to_oracle_gap"] = {
            group: {
                "v2_mean_answer_prob": v2[group]["mean_answer_prob"],
                "oracle_mean_answer_prob": oracle[group]["mean_answer_prob"],
                "gap": v2[group]["mean_answer_prob"] - oracle[group]["mean_answer_prob"],
            }
            for group in v2 if group in oracle
        }
        report["interpretation"] = (
            "A small v2-to-oracle gap on the paraphrase group means routing is "
            "not the bottleneck and router work has little headroom. A large "
            "gap means the opposite. Read it together with the attribution "
            "table before choosing what to fix."
        )

    (output / "router_decomposition.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(
        {
            "status": "decomposition_complete",
            "output": str(output / "router_decomposition.json"),
            "arms": arms,
            "v2_to_oracle_gap": report.get("v2_to_oracle_gap"),
        },
        indent=2,
    ))
    return 0


def _load_shipped(base_model, artifact):
    # The "v2" arm is whatever router the artifact ships: Router V2, V1, or the
    # learned linear router. Oracle/subject-only/random arms reuse its rows.
    return load_router_artifact(base_model, artifact)


if __name__ == "__main__":
    raise SystemExit(main())
