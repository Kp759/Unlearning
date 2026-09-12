#!/usr/bin/env python3
"""Train the MQuAKE seed-1 fact-association bank on locked direct facts only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from mquake_fact_association_embeddings import (
    BASE_PLAN,
    METHOD,
    association_key_from_record,
    build_association_facts,
    build_editor,
    build_exact_direct_token_cases,
    load_locked_visible_forget,
    train_direct_only,
)


@torch.no_grad()
def audit_direct_routes(
    model,
    bank,
    tokenizer,
    facts,
    records,
    case_to_fact_id,
    batch_size=16,
):
    """Require every original atomic record to route to its merged association."""
    fact_to_row = {fact["id"]: index for index, fact in enumerate(facts)}
    device = next(model.parameters()).device
    rows = []

    def contains_subsequence(tokens, pattern):
        width = len(pattern)
        if width == 0 or width > len(tokens):
            return False
        return any(
            tuple(tokens[i : i + width]) == tuple(pattern)
            for i in range(len(tokens) - width + 1)
        )

    prompts = [
        str(record["requested_rewrite"]["prompt"]).format(
            str(record["requested_rewrite"]["subject"])
        )
        for record in records
    ]
    for start in range(0, len(records), int(batch_size)):
        batch_records = records[start : start + int(batch_size)]
        batch_prompts = prompts[start : start + int(batch_size)]
        encoded = tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(device)
        model(**encoded, use_cache=False)
        routes = list(bank.last_active_fact_indices)

        input_rows = encoded["input_ids"].detach().cpu().tolist()
        mask_rows = encoded["attention_mask"].detach().cpu().bool().tolist()

        for record, prompt, route, token_row, mask_row in zip(
            batch_records,
            batch_prompts,
            routes,
            input_rows,
            mask_rows,
        ):
            case_id = int(record["case_id"])
            expected_fact_id = case_to_fact_id[case_id]
            expected_row = fact_to_row[expected_fact_id]
            rr = record["requested_rewrite"]
            prompt_tokens = [
                token for token, keep in zip(token_row, mask_row) if keep
            ]
            candidate_rows = [
                index
                for index, patterns in enumerate(bank.subject_patterns)
                if any(
                    contains_subsequence(prompt_tokens, pattern)
                    for pattern in patterns
                )
            ]
            candidate_facts = [
                {
                    "row": int(index),
                    "fact_id": facts[index]["id"],
                    "association_key": facts[index]["association_key"],
                    "subject": facts[index]["subject"],
                    "relation": facts[index]["relation"],
                    "object": facts[index]["object"],
                    "occurrence_case_ids": facts[index]["occurrence_case_ids"],
                    "canonical_prompts": facts[index]["canonical_prompts"],
                }
                for index in candidate_rows
            ]
            rows.append(
                {
                    "case_id": case_id,
                    "association_key": association_key_from_record(record),
                    "expected_fact_id": expected_fact_id,
                    "expected_row": expected_row,
                    "subject": str(rr["subject"]),
                    "relation": str(rr.get("relation_id")),
                    "object": str(rr["target_true"]["str"]),
                    "canonical_prompt": prompt,
                    "active_rows": route,
                    "candidate_rows_from_subject_scan": candidate_rows,
                    "candidate_facts_from_subject_scan": candidate_facts,
                    "subject_candidate_count": len(candidate_rows),
                    "correct_row_active": expected_row in route,
                    "wrong_row_active": any(
                        index != expected_row for index in route
                    ),
                }
            )

    failures = [row for row in rows if not row["correct_row_active"]]
    return {
        "atomic_record_count": len(records),
        "unique_association_count": len(facts),
        "correct_row_active_fraction": (
            sum(row["correct_row_active"] for row in rows) / len(rows)
        ),
        "any_row_active_fraction": (
            sum(bool(row["active_rows"]) for row in rows) / len(rows)
        ),
        "wrong_row_active_fraction": (
            sum(row["wrong_row_active"] for row in rows) / len(rows)
        ),
        "rows_with_multiple_subject_candidates": sum(
            row["subject_candidate_count"] > 1 for row in rows
        ),
        "failure_count": len(failures),
        "failures": failures,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--row-updates-per-fact", type=int, default=30)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--max-training-seconds", type=float, default=14400.0)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args(argv)

    if args.seed != 1 or args.forget_num != 50 or args.retain_num != 1000:
        raise ValueError(
            "Registered first MQuAKE transfer is fixed to seed=1, "
            "forget_num=50 instances, retain_num=1000 instances"
        )

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite MQuAKE run: {output}")
    output.mkdir(parents=True)

    visible_path = Path(args.training_visible).resolve()
    split_manifest_path = Path(args.split_manifest).resolve()
    split_manifest = json.loads(split_manifest_path.read_text())
    if int(split_manifest.get("seed", -1)) != args.seed:
        raise ValueError("Locked MQuAKE split seed does not match runner")

    sampling = split_manifest.get("sampling", {})
    if int(sampling.get("forget_num_instances", -1)) != args.forget_num:
        raise ValueError("Locked MQuAKE forget instance count does not match")
    if int(sampling.get("retain_num_instances", -1)) != args.retain_num:
        raise ValueError("Locked MQuAKE retain instance count does not match")

    records = load_locked_visible_forget(visible_path)
    expected_atomic = int(sampling.get("forget_atomic_fact_count", -1))
    if len(records) != expected_atomic:
        raise ValueError(
            f"Expected {expected_atomic} atomic forget facts, got {len(records)}"
        )
    facts, case_to_fact_id, dedup_diagnostics = build_association_facts(
        records
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(args.model_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    torch.manual_seed(args.seed)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    base_model.requires_grad_(False)
    base_model.config.use_cache = False

    plan = dict(BASE_PLAN)
    total_steps = (
        int(args.steps)
        if args.steps is not None
        else len(facts) * int(args.row_updates_per_fact)
    )
    if total_steps <= 0 or total_steps % len(facts):
        raise ValueError("--steps must be a positive multiple of unique association count")
    plan.update(
        {
            "seed": int(args.seed),
            "steps": total_steps,
            "check_every": len(facts),
            "row_updates_per_fact": total_steps // len(facts),
            "max_training_seconds": float(args.max_training_seconds),
        }
    )

    editor, bank, key_diagnostics = build_editor(
        base_model,
        tokenizer,
        facts,
        plan,
    )
    token_cases, llama_like = build_exact_direct_token_cases(
        records,
        facts,
        tokenizer,
        editor.model,
    )
    route_audit = audit_direct_routes(
        editor.model,
        bank,
        tokenizer,
        facts,
        records,
        case_to_fact_id,
    )
    route_preflight_path = output / "route_preflight.json"
    route_preflight_path.write_text(
        json.dumps(route_audit, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "route_preflight": str(route_preflight_path),
                "atomic_record_count": route_audit["atomic_record_count"],
                "unique_association_count": route_audit["unique_association_count"],
                "correct_row_active_fraction": route_audit[
                    "correct_row_active_fraction"
                ],
                "any_row_active_fraction": route_audit[
                    "any_row_active_fraction"
                ],
                "wrong_row_active_fraction": route_audit[
                    "wrong_row_active_fraction"
                ],
                "rows_with_multiple_subject_candidates": route_audit[
                    "rows_with_multiple_subject_candidates"
                ],
                "duplicate_records_collapsed": dedup_diagnostics[
                    "duplicate_records_collapsed"
                ],
                "duplicate_association_group_count": dedup_diagnostics[
                    "duplicate_association_group_count"
                ],
                "failure_count": route_audit["failure_count"],
                "first_failures": route_audit["failures"][:10],
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    if route_audit["correct_row_active_fraction"] != 1.0:
        raise RuntimeError(
            "Locked MQuAKE direct prompts do not route perfectly; refusing "
            f"training. Inspect {route_preflight_path}"
        )

    fact_to_row = {fact["id"]: index for index, fact in enumerate(facts)}
    manifest = {
        "method": METHOD,
        "dataset": "MQuAKE-CF-3k-v2",
        "protocol": (
            "ZeroUnlearn-style instance sampling; forget-only direct-fact training"
        ),
        "seed": 1,
        "forget_num_instances": 50,
        "retain_num_instances_final_evaluation": 1000,
        "forget_atomic_record_count": len(records),
        "unique_forget_association_count": len(facts),
        "model_path": str(model_path),
        "training_visible_path": str(visible_path),
        "split_manifest_path": str(split_manifest_path),
        "source_dataset": split_manifest.get("source_dataset"),
        "source_revision": split_manifest.get("source_revision"),
        "source_sha256": split_manifest.get("source_sha256"),
        "training_visible_sha256": split_manifest.get("training_visible_sha256"),
        "forget_source_indices": list(
            sampling.get("forget_source_indices", [])
        ),
        "retain_source_indices_final_evaluation": list(
            sampling.get("retain_source_indices", [])
        ),
        "forget_atomic_case_ids": list(
            sampling.get("forget_atomic_case_ids", [])
        ),
        "atomic_case_to_association_id": {
            str(case_id): fact_id
            for case_id, fact_id in case_to_fact_id.items()
        },
        "architecture": (
            "one independent residual vector per unique (subject, relation, target_true) association; "
            "frozen Llama; layer 19; one original-request-boundary intervention"
        ),
        "routing_policy": "hierarchical_subject_then_context_if_ambiguous",
        "layer": int(plan["layer"]),
        "trainable_vectors": len(facts),
        "association_deduplication": dedup_diagnostics,
        "base_parameters_trainable": 0,
        "tokenizer_extended": False,
        "lm_head_edited": False,
        "target_new_used": False,
        "unknown_or_replacement_target_used": False,
        "training_visible": [
            "direct requested_rewrite prompts from 50 sampled forget instances",
            "subjects",
            "relation_id provenance where present",
            "original sensitive target_true answers",
            "exact per-token teacher-forced rewrite contexts reconstructed from direct requests",
        ],
        "evaluation_only": [
            "1000 sampled retain instances",
            "atomic natural-language questions",
            "instance-level multi-hop questions",
            "benchmark counterfactual target_new",
            "Wikidata PPL text",
        ],
        "objective": (
            "absolute suppression of every direct sensitive answer token in the "
            "exact official MQuAKE Eff token contexts; maximum sensitive-token "
            "probability < 1e-6"
        ),
        "objective_metric_alignment": (
            "native MQuAKE Eff is sensitive target_true token argmax accuracy; "
            "a target-token probability below 1e-6 cannot remain vocabulary top-1"
        ),
        "official_evaluation_started": False,
        "route_audit": route_audit,
        "key_diagnostics": key_diagnostics,
        "plan": {
            **plan,
            "radius_schedule": [
                list(value) for value in plan["radius_schedule"]
            ],
        },
        "facts": facts,
    }
    (output / "association_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    (output / "training_token_cases.json").write_text(
        json.dumps([asdict(case) for case in token_cases], indent=2) + "\n"
    )

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_complete",
                    "optimization_started": False,
                    "forget_instances": 50,
                    "atomic_records": len(records),
                    "unique_associations": len(facts),
                    "duplicate_records_collapsed": dedup_diagnostics[
                        "duplicate_records_collapsed"
                    ],
                    "route_audit": route_audit,
                    "output_dir": str(output),
                },
                indent=2,
            )
        )
        return 0

    report = train_direct_only(
        editor,
        tokenizer,
        token_cases,
        fact_to_row,
        plan,
        output,
        llama_like=llama_like,
    )
    artifact = editor.artifact()
    artifact.update(
        {
            "method": METHOD,
            "dataset": "MQuAKE-CF-3k-v2",
            "seed": 1,
            "forget_num_instances": 50,
            "forget_atomic_record_count": len(records),
            "unique_forget_association_count": len(facts),
            "association_deduplication": dedup_diagnostics,
            "atomic_case_to_association_id": {
                str(case_id): fact_id
                for case_id, fact_id in case_to_fact_id.items()
            },
            "target_new_used": False,
            "unknown_or_replacement_target_used": False,
            "training_probe_scope": "direct requested_rewrite only",
            "atomic_questions_used_for_training_or_selection": False,
            "multihop_questions_used_for_training_or_selection": False,
            "retain_records_used_for_training_or_selection": False,
        }
    )
    torch.save(artifact, output / "fact_association_embeddings.pt")

    report["manifest"] = manifest
    report["runtime_counters"] = bank.counters()
    report["official_evaluation_started"] = False
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "mquake_fact_association_training_complete",
                "stop_reason": report["stop_reason"],
                "best_step": report["best_step"],
                "forget_instances": 50,
                "atomic_records": len(records),
                "unique_associations": len(facts),
                "final_training_metrics": report["final_training_metrics"],
                "artifact": str(output / "fact_association_embeddings.pt"),
                "official_evaluation_started": False,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
