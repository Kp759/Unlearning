#!/usr/bin/env python3
"""Train RWKU seed-1 fact-association residuals with Router V2."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from rwku_batch50 import (
    PROTOCOL_ID,
    EVALUATION_ONLY_FILES,
    materialize_batch_split,
)
from rwku_fact_association_embeddings import (
    BASE_PLAN,
    METHOD,
    association_key_from_row,
    build_association_facts,
    build_editor_v2,
    build_exact_direct_token_cases,
    train_direct_only,
)


METHOD_V2 = "fact_association_embeddings_rwku_batch50_router_v2"

@torch.no_grad()
def audit_direct_routes(model, bank, tokenizer, facts, rows, record_to_fact_id, batch_size=16):
    fact_to_row = {fact["id"]: index for index, fact in enumerate(facts)}
    device = next(model.parameters()).device
    output_rows = []

    def contains_subsequence(tokens, pattern):
        width = len(pattern)
        if width == 0 or width > len(tokens):
            return False
        return any(
            tuple(tokens[i : i + width]) == tuple(pattern)
            for i in range(len(tokens) - width + 1)
        )

    prompts = []
    import rwku_eval as rwku
    for row in rows:
        prompts.append(rwku.format_qa_prompt(tokenizer, row))

    for start in range(0, len(rows), int(batch_size)):
        batch_rows = rows[start : start + int(batch_size)]
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

        for row, prompt, route, token_row, mask_row in zip(
            batch_rows, batch_prompts, routes, input_rows, mask_rows
        ):
            source_hash = str(row["source_record_sha256"])
            expected_fact_id = record_to_fact_id[source_hash]
            expected_row = fact_to_row[expected_fact_id]
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
            output_rows.append(
                {
                    "source_record_sha256": source_hash,
                    "association_key": association_key_from_row(row),
                    "expected_fact_id": expected_fact_id,
                    "expected_row": expected_row,
                    "subject": str(row["subject"]),
                    "query": str(row["query"]),
                    "answer": str(row["answer"]),
                    "canonical_prompt": prompt,
                    "active_rows": route,
                    "candidate_rows_from_subject_scan": candidate_rows,
                    "subject_candidate_count": len(candidate_rows),
                    "correct_row_active": expected_row in route,
                    "wrong_row_active": any(i != expected_row for i in route),
                }
            )

    failures = [row for row in output_rows if not row["correct_row_active"]]
    return {
        "forget_row_count": len(rows),
        "unique_association_count": len(facts),
        "correct_row_active_fraction": (
            sum(row["correct_row_active"] for row in output_rows) / len(output_rows)
        ),
        "any_row_active_fraction": (
            sum(bool(row["active_rows"]) for row in output_rows) / len(output_rows)
        ),
        "wrong_row_active_fraction": (
            sum(row["wrong_row_active"] for row in output_rows) / len(output_rows)
        ),
        "rows_with_multiple_subject_candidates": sum(
            row["subject_candidate_count"] > 1 for row in output_rows
        ),
        "min_subject_candidate_count": min(
            row["subject_candidate_count"] for row in output_rows
        ),
        "max_subject_candidate_count": max(
            row["subject_candidate_count"] for row in output_rows
        ),
        "failure_count": len(failures),
        "failures": failures,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-root", default="data/rwku")
    p.add_argument("--split-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--row-updates-per-fact", type=int, default=30)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--max-training-seconds", type=float, default=14400.0)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args(argv)

    if args.seed != 1:
        raise ValueError("Registered first RWKU residual-bank transfer is fixed to seed=1")

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite RWKU run: {output}")
    output.mkdir(parents=True)

    split_dir = Path(args.split_dir).resolve()
    split = materialize_batch_split(
        data_root=Path(args.data_root).resolve(),
        output_dir=split_dir,
        batch_seed=args.seed,
        allow_download=not args.no_download,
    )
    rows = list(split["forget_train"])
    if len(rows) != 50:
        raise RuntimeError(f"{PROTOCOL_ID} seed 1 must expose exactly 50 training rows")

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

    facts, record_to_fact_id, dedup_diagnostics = build_association_facts(
        rows, tokenizer
    )

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
        raise ValueError("--steps must be a positive multiple of association count")
    plan.update(
        {
            "seed": int(args.seed),
            "steps": total_steps,
            "check_every": len(facts),
            "row_updates_per_fact": total_steps // len(facts),
            "max_training_seconds": float(args.max_training_seconds),
        }
    )

    editor, bank, key_diagnostics = build_editor_v2(
        base_model, tokenizer, facts, plan
    )
    token_cases = build_exact_direct_token_cases(rows, facts, tokenizer)
    route_audit = audit_direct_routes(
        editor.model,
        bank,
        tokenizer,
        facts,
        rows,
        record_to_fact_id,
    )
    route_preflight_path = output / "route_preflight.json"
    route_preflight_path.write_text(
        json.dumps(route_audit, indent=2, allow_nan=False) + "\n"
    )

    print(
        json.dumps(
            {
                "route_preflight": str(route_preflight_path),
                "forget_row_count": route_audit["forget_row_count"],
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
                "subject_candidate_count_range": [
                    route_audit["min_subject_candidate_count"],
                    route_audit["max_subject_candidate_count"],
                ],
                "duplicate_records_collapsed": dedup_diagnostics[
                    "duplicate_records_collapsed"
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
            "RWKU training prompts do not route perfectly; refusing optimization. "
            f"Inspect {route_preflight_path}"
        )

    fact_to_row = {fact["id"]: index for index, fact in enumerate(facts)}
    split_manifest_path = split_dir / "split_manifest.json"
    manifest = {
        "method": METHOD_V2,
        "dataset": "RWKU",
        "protocol_id": PROTOCOL_ID,
        "protocol_status": "probe_assisted_cross_benchmark_method_extension",
        "seed": 1,
        "target_seeds": split["manifest"]["target_seeds"],
        "subjects": [item["subject"] for item in split["manifest"]["targets"]],
        "forget_train_count": len(rows),
        "unique_forget_association_count": len(facts),
        "model_path": str(model_path),
        "data_root": str(Path(args.data_root).resolve()),
        "split_dir": str(split_dir),
        "split_manifest_path": str(split_manifest_path),
        "rwku_code_revision": split["manifest"]["rwku_code_revision"],
        "rwku_dataset_revision": split["manifest"]["rwku_dataset_revision"],
        "architecture": (
            "one independent residual vector per selected natural-input RWKU factual association; "
            "Router V2; frozen Llama; layer 19; original request-boundary intervention"
        ),
        "association_identity": (
            "normalized(subject), normalized(selected query), normalized(sensitive answer); "
            "record identity is provenance only and never a runtime input"
        ),
        "routing_policy": "subject_candidate_plus_direct_context_confirmation_v2",
        "unique_subject_bypass": False,
        "layer": int(plan["layer"]),
        "trainable_vectors": len(facts),
        "association_deduplication": dedup_diagnostics,
        "base_parameters_trainable": 0,
        "tokenizer_extended": False,
        "lm_head_edited": False,
        "replacement_target_used": False,
        "training_visible": [
            "only the exact 50 RWKU-Batch-50-v1 selected Level-1/Level-2 probes",
            "subject",
            "natural query/context",
            "original sensitive answer",
        ],
        "evaluation_only": [
            "held-out Level-1 and Level-2 probes",
            "held-out Level-2 deterministic paraphrases",
            *list(EVALUATION_ONLY_FILES),
            "Wikidata PPL text",
        ],
        "objective": (
            "absolute suppression of every sensitive answer token on the 50 "
            "selected training probes; maximum sensitive-token probability < 1e-6"
        ),
        "official_native_rwku_note": (
            "RWKU natively provides a target entity rather than a forget corpus; "
            "this Batch-50 experiment is explicitly a probe-assisted method extension"
        ),
        "official_evaluation_started": False,
        "router_v2_contract": {
            "subject_is_candidate_eligibility_only": True,
            "unique_subject_bypass": False,
            "positive_prototypes": "training-visible direct prompts only",
            "negative_controls": "same-subject competitors plus training-only subject-transplanted wrong contexts",
            "target_new_or_eval_probe_use": False
        },
        "route_audit": route_audit,
        "key_diagnostics": key_diagnostics,
        "plan": {
            **plan,
            "radius_schedule": [list(value) for value in plan["radius_schedule"]],
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
                    "forget_rows": len(rows),
                    "unique_associations": len(facts),
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
    )
    artifact = editor.artifact()
    artifact.update(
        {
            "method": METHOD_V2,
            "dataset": "RWKU",
            "protocol_id": PROTOCOL_ID,
            "seed": 1,
            "target_seeds": split["manifest"]["target_seeds"],
            "forget_train_count": len(rows),
            "unique_forget_association_count": len(facts),
            "association_deduplication": dedup_diagnostics,
            "source_record_to_association_id": record_to_fact_id,
            "heldout_rwku_probes_used_for_training_or_selection": False,
            "router_v2_unique_subject_bypass": False,
            "neighbor_mia_utility_used_for_training_or_selection": False,
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
                "status": "rwku_fact_association_router_v2_training_complete",
                "stop_reason": report["stop_reason"],
                "best_step": report["best_step"],
                "forget_rows": len(rows),
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
