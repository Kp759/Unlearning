#!/usr/bin/env python3
"""Train 50 independent natural-prompt fact-association embeddings on MCF."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from mcf_sampling import sample_official_mcf_records
from run_static_overlap_mlp_pilot import emit
from static_overlap_extended_tokens_v2 import routed_metrics, train_row_wise
from static_overlap_fact_association_embeddings import (
    METHOD,
    PLAN,
    FactAssociationBank,
    FactAssociationEditor,
    audit_runtime_routes,
    build_forget_examples,
    build_semantic_keys,
    make_subject_patterns,
    make_unknown_examples,
)
from static_overlap_natural_writer import mcf_facts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--layer", type=int, default=PLAN["layer"])
    parser.add_argument("--gate-slack", type=float, default=PLAN["gate_slack"])
    parser.add_argument(
        "--min-dev-route-recall",
        type=float,
        default=PLAN["min_development_route_recall"],
    )
    args = parser.parse_args(argv)

    if args.forget_num != 50 or args.seed != 1:
        raise ValueError("Registered experiment is fixed to forget_num=50, seed=1.")

    model_path = Path(args.model_path).resolve()
    mcf_path = Path(args.mcf_path).resolve()
    output = Path(args.output_dir).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory is missing: {model_path}")
    if not mcf_path.is_file():
        raise FileNotFoundError(f"MCF JSON is missing: {mcf_path}")
    output.mkdir(parents=True, exist_ok=False)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    model.requires_grad_(False)

    records = json.loads(mcf_path.read_text())
    forget_records, _ = sample_official_mcf_records(
        records,
        forget_num=args.forget_num,
        retain_num=0,
        seed=args.seed,
        strict=True,
    )
    facts = mcf_facts(forget_records, "forget")
    examples = build_forget_examples(facts, tokenizer, PLAN["max_length"])

    emit(
        phase="fact_association_data_ready",
        method=METHOD,
        forget_facts=len(facts),
        train_views=sum(e.split == "train" for e in examples),
        development_views=sum(e.split == "development" for e in examples),
        official_eff_canonical_training_visible=True,
        official_paraphrase_fields_used=False,
        official_neighborhood_fields_used=False,
        private_tokens=False,
        base_weights_trainable=False,
    )

    keys, thresholds, gate_diagnostics = build_semantic_keys(
        model=model,
        tokenizer=tokenizer,
        facts=facts,
        examples=examples,
        layer=args.layer,
        gate_slack=args.gate_slack,
        relation_negative_count=PLAN["relation_negative_count"],
    )
    (output / "gate_diagnostics.json").write_text(
        json.dumps(gate_diagnostics, indent=2, allow_nan=False) + "\n"
    )
    emit(
        phase="fact_association_gate_ready",
        layer=args.layer,
        development_correct_key_fraction=gate_diagnostics[
            "development_correct_key_fraction"
        ],
        development_any_key_active_fraction=gate_diagnostics[
            "development_any_key_active_fraction"
        ],
        mean_relation_negative_fire_fraction=(
            sum(
                row["relation_negative_fire_fraction"]
                for row in gate_diagnostics["per_fact"]
            )
            / len(gate_diagnostics["per_fact"])
        ),
    )

    subject_patterns = make_subject_patterns(tokenizer, facts)
    bank = FactAssociationBank(
        base_model=model,
        layer=args.layer,
        keys=keys,
        thresholds=thresholds,
        subject_patterns=subject_patterns,
        facts=facts,
    )
    editor = FactAssociationEditor(model, bank)

    # Zero rows must make the wrapper bit-exact even if its gate fires.
    neutral = tokenizer(
        "A neutral sentence about mathematics and weather.",
        return_tensors="pt",
    ).to(args.device)
    with torch.no_grad():
        bank.close()
        base_logits = model(**neutral, use_cache=False).logits.detach().clone()
        # Reattach a fresh bank because hooks cannot be re-enabled after removal.
        bank = FactAssociationBank(
            base_model=model,
            layer=args.layer,
            keys=keys,
            thresholds=thresholds,
            subject_patterns=subject_patterns,
            facts=facts,
        )
        editor = FactAssociationEditor(model, bank)
        wrapped_logits = editor.model(**neutral, use_cache=False).logits.detach()
        if not torch.equal(base_logits, wrapped_logits):
            raise ValueError("Zero association rows changed an unmatched base prompt")

    fact_to_row = {fact["id"]: index for index, fact in enumerate(facts)}
    route_audit = audit_runtime_routes(
        editor.model,
        bank,
        tokenizer,
        examples,
        fact_to_row,
    )
    (output / "runtime_route_audit.json").write_text(
        json.dumps(route_audit, indent=2, allow_nan=False) + "\n"
    )
    emit(
        phase="fact_association_runtime_route_audit",
        train=route_audit["train"],
        development=route_audit["development"],
    )
    if route_audit["train"]["correct_row_active_fraction"] < 1.0:
        raise RuntimeError(
            "Automatic association gate misses fitting prompts; refusing expensive training"
        )
    if (
        route_audit["development"]["correct_row_active_fraction"]
        < float(args.min_dev_route_recall)
    ):
        raise RuntimeError(
            "Automatic association gate development recall is below the "
            f"{args.min_dev_route_recall:.3f} preflight floor; refusing expensive training"
        )

    answer_map = {example.id: example for example in examples}
    unknown_map = make_unknown_examples(
        examples,
        tokenizer,
        PLAN["max_length"],
        PLAN["unknown_completion"],
    )
    plan = dict(PLAN)
    plan["layer"] = int(args.layer)
    plan["gate_slack"] = float(args.gate_slack)
    plan["min_development_route_recall"] = float(args.min_dev_route_recall)
    plan["radius_schedule"] = tuple(tuple(x) for x in PLAN["radius_schedule"])

    manifest = {
        "method": METHOD,
        "architecture": "50_independent_subject_relation_association_embeddings_v1",
        "model_path": str(model_path),
        "mcf_path": str(mcf_path),
        "sampling": {
            "forget_num": 50,
            "retain_num": 0,
            "seed": 1,
            "convention": "ZeroUnlearn/official MCF forget split",
        },
        "plan": {
            **plan,
            "radius_schedule": [list(x) for x in plan["radius_schedule"]],
        },
        "facts": facts,
        "forget_case_ids": [int(record["case_id"]) for record in forget_records],
        "trainable_vectors": len(bank.rows),
        "trainable_parameters": sum(row.numel() for row in bank.rows),
        "base_parameters_trainable": 0,
        "tokenizer_extended": False,
        "lm_head_edited": False,
        "requires_fact_id_token_injection": False,
        "runtime_trigger": (
            "exact subject-token eligibility AND frozen natural-context semantic key"
        ),
        "runtime_trigger_uses_object": False,
        "object_role": "suppression target only",
        "official_eff_canonical_training_visible": True,
        "official_paraphrase_fields_used": False,
        "official_neighborhood_fields_used": False,
        "development_used_for_gradients": False,
        "development_used_for_key_or_threshold_fitting": False,
        "unmatched_inputs_follow_exact_frozen_base_path": True,
        "runtime_route_audit": route_audit,
    }
    (output / "association_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    (output / "association_examples.json").write_text(
        json.dumps([asdict(e) for e in examples], indent=2, allow_nan=False) + "\n"
    )

    # V2.1's optimizer is intentionally reused: one Adam instance per vector,
    # worst-view suppression before the threshold, abstention only after lock.
    report = train_row_wise(
        editor=editor,
        original_examples=examples,
        routed_answer=answer_map,
        routed_unknown=unknown_map,
        fact_to_row=fact_to_row,
        plan=plan,
        output=output,
    )
    final_metrics = routed_metrics(
        editor.model,
        answer_map,
        unknown_map,
        plan["target_probability"],
    )

    # Unmatched natural text must remain exactly base after nonzero training.
    with torch.no_grad():
        edited_neutral = editor.model(**neutral, use_cache=False).logits.detach()
        if not torch.equal(base_logits, edited_neutral):
            raise ValueError("Unmatched natural prompt no longer follows exact base path")

    artifact = editor.artifact()
    torch.save(artifact, output / "fact_association_embeddings.pt")
    report.update(
        {
            "method": METHOD,
            "manifest": manifest,
            "gate_diagnostics": gate_diagnostics,
            "runtime_route_audit": route_audit,
            "final_metrics": final_metrics,
            "runtime_counters": bank.counters(),
            "unmatched_neutral_logits_exact_base_after_training": True,
            "official_evaluation_started": False,
        }
    )
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )

    emit(
        status="fact_association_embedding_training_complete",
        stop_reason=report["stop_reason"],
        best_step=report["best_step"],
        final_metrics=final_metrics,
        gate_diagnostics=str(output / "gate_diagnostics.json"),
        artifact=str(output / "fact_association_embeddings.pt"),
        official_evaluation_started=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
