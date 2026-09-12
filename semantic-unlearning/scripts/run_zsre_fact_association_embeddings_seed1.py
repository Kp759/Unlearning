#!/usr/bin/env python3
"""Train seed-1 / 50-fact association embeddings on locked ZsRE direct requests."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from zsre_fact_association_embeddings import (
    METHOD,
    PLAN,
    build_editor,
    build_exact_direct_token_cases,
    facts_from_locked_records,
    load_locked_visible_forget,
    train_direct_only,
)


@torch.no_grad()
def audit_direct_routes(model, bank, tokenizer, facts, batch_size=16):
    prompts = [fact["canonical_prompt"] for fact in facts]
    device = next(model.parameters()).device
    rows = []
    for start in range(0, len(facts), int(batch_size)):
        batch_facts = facts[start:start + int(batch_size)]
        batch_prompts = prompts[start:start + int(batch_size)]
        encoded = tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(device)
        model(**encoded, use_cache=False)
        routes = list(bank.last_active_fact_indices)
        for offset, (fact, route) in enumerate(zip(batch_facts, routes)):
            expected = start + offset
            rows.append({
                "fact_id": fact["id"],
                "case_id": fact["case_id"],
                "subject": fact["subject"],
                "expected_row": expected,
                "active_rows": route,
                "correct_row_active": expected in route,
                "wrong_row_active": any(index != expected for index in route),
            })
    return {
        "count": len(rows),
        "correct_row_active_fraction": sum(
            row["correct_row_active"] for row in rows
        ) / len(rows),
        "any_row_active_fraction": sum(
            bool(row["active_rows"]) for row in rows
        ) / len(rows),
        "wrong_row_active_fraction": sum(
            row["wrong_row_active"] for row in rows
        ) / len(rows),
        "failures": [
            row for row in rows if not row["correct_row_active"]
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-visible", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--max-training-seconds", type=float, default=3600.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    if args.seed != 1 or args.forget_num != 50:
        raise ValueError(
            "This registered first ZsRE transfer run is fixed to seed=1, forget_num=50"
        )

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite ZsRE run: {output}")
    output.mkdir(parents=True)

    visible_path = Path(args.training_visible).resolve()
    split_manifest_path = Path(args.split_manifest).resolve()
    split_manifest = json.loads(split_manifest_path.read_text())
    if int(split_manifest.get("seed", -1)) != args.seed:
        raise ValueError("Locked split manifest seed does not match runner")
    sampling = split_manifest.get("sampling", {})
    if int(sampling.get("forget_num", -1)) != args.forget_num:
        raise ValueError("Locked split manifest forget count does not match runner")
    roles = split_manifest.get("data_roles", {})
    if roles.get("target_new_visible") is not False:
        raise ValueError("ZsRE split must hide target_new")
    if roles.get("neutral_or_replacement_target_visible") is not False:
        raise ValueError("ZsRE split must hide neutral/replacement targets")

    records = load_locked_visible_forget(visible_path)
    if len(records) != args.forget_num:
        raise ValueError(
            f"Expected {args.forget_num} visible forget records, got {len(records)}"
        )
    facts = facts_from_locked_records(records)

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

    plan = dict(PLAN)
    plan.update({
        "seed": int(args.seed),
        "steps": int(args.steps),
        "max_training_seconds": float(args.max_training_seconds),
    })

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
    )
    if route_audit["correct_row_active_fraction"] != 1.0:
        raise RuntimeError(
            "Locked ZsRE direct prompts do not route perfectly; refusing training"
        )

    fact_to_row = {
        fact["id"]: index for index, fact in enumerate(facts)
    }
    manifest = {
        "method": METHOD,
        "dataset": "ZsRE",
        "protocol": "ZeroUnlearn-style locked prompt-level holdout",
        "seed": 1,
        "forget_num": 50,
        "retain_num_final_evaluation": 1000,
        "model_path": str(model_path),
        "training_visible_path": str(visible_path),
        "split_manifest_path": str(split_manifest_path),
        "source_dataset": split_manifest.get("source_dataset"),
        "source_sha256": split_manifest.get("source_sha256"),
        "training_visible_sha256": split_manifest.get("training_visible_sha256"),
        "forget_case_ids": list(sampling.get("forget_case_ids", [])),
        "retain_case_ids_final_evaluation": list(
            sampling.get("retain_case_ids", [])
        ),
        "architecture": (
            "50 independent fact-specific residual vectors; frozen Llama; "
            "layer 19; one original-request-boundary intervention"
        ),
        "routing_policy": "hierarchical_subject_then_context_if_ambiguous",
        "layer": int(plan["layer"]),
        "trainable_vectors": 50,
        "base_parameters_trainable": 0,
        "tokenizer_extended": False,
        "lm_head_edited": False,
        "target_new_used": False,
        "unknown_or_replacement_target_used": False,
        "training_visible": [
            "50 direct requested_rewrite prompts",
            "subjects",
            "original sensitive target_true answers",
            "exact per-token teacher-forced rewrite contexts reconstructed from direct requests",
        ],
        "evaluation_only": [
            "official rephrases",
            "official locality/neighborhood probes",
            "1000 sampled retain records",
            "Wikidata PPL text",
        ],
        "objective": (
            "absolute suppression of every direct sensitive answer token in the "
            "exact official rewrite-token contexts; maximum sensitive-token "
            "probability < 1e-6"
        ),
        "objective_metric_alignment": (
            "ZsRE Eff/Gen are greedy sensitive-token accuracies; a target-token "
            "probability below 1e-6 cannot remain vocabulary top-1"
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
    (output / "route_preflight.json").write_text(
        json.dumps(route_audit, indent=2, allow_nan=False) + "\n"
    )

    if args.preflight_only:
        print(json.dumps({
            "status": "preflight_complete",
            "optimization_started": False,
            "route_audit": route_audit,
            "output_dir": str(output),
        }, indent=2))
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
    artifact.update({
        "method": METHOD,
        "dataset": "ZsRE",
        "seed": 1,
        "forget_num": 50,
        "target_new_used": False,
        "unknown_or_replacement_target_used": False,
        "training_probe_scope": "direct requested_rewrite only",
        "official_rephrases_used_for_training_or_selection": False,
        "official_locality_used_for_training_or_selection": False,
        "retain_records_used_for_training_or_selection": False,
    })
    torch.save(artifact, output / "fact_association_embeddings.pt")

    report["manifest"] = manifest
    report["runtime_counters"] = bank.counters()
    report["official_evaluation_started"] = False
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "status": "zsre_fact_association_training_complete",
        "stop_reason": report["stop_reason"],
        "best_step": report["best_step"],
        "final_training_metrics": report["final_training_metrics"],
        "artifact": str(output / "fact_association_embeddings.pt"),
        "official_evaluation_started": False,
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
