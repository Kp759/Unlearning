#!/usr/bin/env python3
"""Matched four-arm V2 development experiment for fact-association embeddings.

A: subject-first gate + absolute suppression
B: subject-first gate + absolute + actual comparator-margin constraints
C: relation-prototype gate + absolute suppression
D: relation-prototype gate + absolute + actual comparator-margin constraints

Seed 1 is development-only because its official failures have already been
inspected. This runner never opens official paraphrase/neighborhood fields.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from mcf_sampling import sample_official_mcf_records
from run_static_overlap_mlp_pilot import emit
from static_overlap_extended_tokens_v2 import routed_metrics, train_row_wise
from static_overlap_fact_association_embeddings import (
    METHOD as V1_METHOD,
    PLAN as V1_PLAN,
    FactAssociationBank,
    FactAssociationEditor,
    audit_runtime_routes,
    build_forget_examples,
    build_semantic_keys,
    make_subject_patterns,
    make_unknown_examples,
    replace_completion,
    relation_negative_prompts,
)
from static_overlap_fact_association_v2_gate import (
    RelationPrototypeAssociationBank,
    build_relation_prototype_gate,
)
from static_overlap_fact_association_v2_optimizer import (
    constraint_metrics,
    train_row_wise_constraints,
)
from static_overlap_natural_writer import mcf_facts


ARMS = {
    "A": {"gate": "subject_first", "objective": "absolute"},
    "B": {"gate": "subject_first", "objective": "absolute_margin"},
    "C": {"gate": "relation_prototype", "objective": "absolute"},
    "D": {"gate": "relation_prototype", "objective": "absolute_margin"},
}


def requested_rewrite(record):
    rr = record["requested_rewrite"]
    if isinstance(rr, list):
        rr = rr[0]
    return rr


def target_new_by_fact(forget_records):
    result = {}
    for record in forget_records:
        rr = requested_rewrite(record)
        value = rr["target_new"]
        text = str(value["str"] if isinstance(value, dict) else value).strip()
        if not text:
            raise ValueError("MCF target_new is empty")
        result[f"mcf_forget_{int(record['case_id'])}"] = text
    return result


def make_comparator_examples(examples, target_new, tokenizer, max_length):
    result = {}
    for example in examples:
        candidate = replace_completion(
            example,
            " " + target_new[example.fact_id],
            tokenizer,
            max_length,
        )
        result[example.id] = replace(
            candidate,
            id=f"{example.id}:comparator",
            group=f"{example.group}:comparator",
        )
    return result


@torch.no_grad()
def audit_wrong_relation_routes(
    model, bank, tokenizer, facts, fact_to_row, count, batch_size=16
):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    prompts = []
    owners = []
    for fact in facts:
        current = relation_negative_prompts(fact, facts, count)
        prompts.extend(current)
        owners.extend([fact_to_row[fact["id"]]] * len(current))

    active = 0
    owner_active = 0
    wrong_active = 0
    rows = []
    for start in range(0, len(prompts), int(batch_size)):
        batch_prompts = prompts[start:start + int(batch_size)]
        batch_owners = owners[start:start + int(batch_size)]
        encoded = tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(device)
        model(**encoded, use_cache=False)
        routes = list(bank.last_active_fact_indices)
        for prompt, owner, route in zip(batch_prompts, batch_owners, routes):
            active += int(bool(route))
            owner_active += int(owner in route)
            wrong_active += int(any(index != owner for index in route))
            rows.append({
                "prompt": prompt,
                "expected_owner": owner,
                "active_rows": route,
            })
    total = len(rows)
    return {
        "count": total,
        "any_route_fraction": active / total,
        "expected_owner_route_fraction": owner_active / total,
        "wrong_owner_route_fraction": wrong_active / total,
        "examples": rows[:20],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--max-training-seconds", type=float, default=3600.0)
    parser.add_argument("--absolute-nll-buffer", type=float, default=0.1)
    parser.add_argument("--margin-target", type=float, default=0.1)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--prototype-u-slack", type=float, default=0.01)
    parser.add_argument("--prototype-d-slack", type=float, default=0.01)
    args = parser.parse_args(argv)

    if args.forget_num != 50 or args.seed != 1:
        raise ValueError(
            "The registered four-arm development comparison is fixed to "
            "forget_num=50, seed=1."
        )
    if args.absolute_nll_buffer < 0 or args.margin_target < 0:
        raise ValueError("Constraint buffers must be non-negative")

    arm = ARMS[args.arm]
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite arm output: {output}")
    output.mkdir(parents=True)
    model_path = Path(args.model_path).resolve()
    mcf_path = Path(args.mcf_path).resolve()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=args.local_files_only,
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
    comparator_targets = target_new_by_fact(forget_records)
    # target_new is extra comparator supervision for B/D and diagnostics for
    # A/C. It is explicitly recorded rather than described as forget-fact-only.
    for fact in facts:
        fact["comparator_target_new"] = comparator_targets[fact["id"]]

    plan = dict(V1_PLAN)
    plan.update({
        "layer": int(args.layer),
        "steps": int(args.steps),
        "max_training_seconds": float(args.max_training_seconds),
        "absolute_nll_buffer": float(args.absolute_nll_buffer),
        "margin_target": float(args.margin_target),
        "margin_weight": float(args.margin_weight),
        "log_phase": f"fact_association_v2_arm_{args.arm.lower()}",
    })
    plan["radius_schedule"] = tuple(
        tuple(value) for value in V1_PLAN["radius_schedule"]
    )

    examples = build_forget_examples(
        facts, tokenizer, int(plan["max_length"])
    )
    true_map = {example.id: example for example in examples}
    comparator_map = make_comparator_examples(
        examples,
        comparator_targets,
        tokenizer,
        int(plan["max_length"]),
    )
    unknown_map = make_unknown_examples(
        examples,
        tokenizer,
        int(plan["max_length"]),
        str(plan["unknown_completion"]),
    )
    subject_patterns = make_subject_patterns(tokenizer, facts)

    gate_diagnostics = {}
    if arm["gate"] == "subject_first":
        keys, thresholds, gate_diagnostics = build_semantic_keys(
            model=model,
            tokenizer=tokenizer,
            facts=facts,
            examples=examples,
            layer=args.layer,
            gate_slack=plan["gate_slack"],
            relation_negative_count=plan["relation_negative_count"],
        )
        bank = FactAssociationBank(
            base_model=model,
            layer=args.layer,
            keys=keys,
            thresholds=thresholds,
            subject_patterns=subject_patterns,
            facts=facts,
        )
    else:
        (
            positive_prototypes,
            negative_prototypes,
            alpha,
            tau,
            gate_diagnostics,
        ) = build_relation_prototype_gate(
            model=model,
            tokenizer=tokenizer,
            facts=facts,
            examples=examples,
            layer=args.layer,
            relation_negative_count=plan["relation_negative_count"],
            u_slack=args.prototype_u_slack,
            d_slack=args.prototype_d_slack,
        )
        bank = RelationPrototypeAssociationBank(
            base_model=model,
            layer=args.layer,
            positive_prototypes=positive_prototypes,
            negative_prototypes=negative_prototypes,
            alpha=alpha,
            tau=tau,
            subject_patterns=subject_patterns,
            facts=facts,
        )

    editor = FactAssociationEditor(model, bank)
    fact_to_row = {
        fact["id"]: index for index, fact in enumerate(facts)
    }
    route_audit = audit_runtime_routes(
        editor.model,
        bank,
        tokenizer,
        examples,
        fact_to_row,
    )
    wrong_relation_route_audit = audit_wrong_relation_routes(
        editor.model,
        bank,
        tokenizer,
        facts,
        fact_to_row,
        int(plan["relation_negative_count"]),
    )
    if route_audit["train"]["correct_row_active_fraction"] < 1.0:
        raise RuntimeError(
            "Arm gate misses training positives; stop before expensive training"
        )

    baseline_constraints = constraint_metrics(
        editor.model,
        true_map,
        comparator_map,
        unknown_map,
        target_probability=plan["target_probability"],
        absolute_nll_buffer=plan["absolute_nll_buffer"],
        margin_target=plan["margin_target"],
    )

    manifest = {
        "method": "static_overlap_fact_association_embeddings_v2_four_arm",
        "arm": args.arm,
        "arm_definition": arm,
        "architecture": (
            "50 independent hidden-state fact vectors; frozen Llama; "
            "layer 19; one original-request-boundary intervention"
        ),
        "model_path": str(model_path),
        "mcf_path": str(mcf_path),
        "sampling": {
            "forget_num": 50,
            "seed": 1,
            "status": (
                "DEVELOPMENT ONLY: seed-1 official failures were inspected "
                "after frozen V1 evaluation"
            ),
        },
        "official_paraphrases_used": False,
        "official_neighborhoods_used": False,
        "target_new_used_as_training_supervision": (
            arm["objective"] == "absolute_margin"
        ),
        "target_new_used_for_diagnostics": True,
        "comparator_role": (
            "moving edited-model comparator; detached only in proposal gradient "
            "for B/D; freshly recomputed for candidate acceptance"
        ),
        "gate": arm["gate"],
        "objective": arm["objective"],
        "one_position_intervention": True,
        "configured_layer": int(args.layer),
        "base_parameters_trainable": 0,
        "trainable_vectors": 50,
        "runtime_boundary_contract": (
            "prompt-only route; fixed original request boundary; answer suffix "
            "cannot change route"
        ),
        "cached_generation_supported": False,
        "plan": {
            **plan,
            "radius_schedule": [list(value) for value in plan["radius_schedule"]],
        },
        "forget_case_ids": [
            int(record["case_id"]) for record in forget_records
        ],
        "facts": facts,
        "route_audit": route_audit,
        "wrong_relation_route_audit": wrong_relation_route_audit,
        "gate_diagnostics": gate_diagnostics,
        "baseline_constraints": baseline_constraints,
    }
    (output / "association_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    (output / "association_examples.json").write_text(
        json.dumps([asdict(example) for example in examples], indent=2) + "\n"
    )
    (output / "comparator_examples.json").write_text(
        json.dumps(
            [asdict(comparator_map[example.id]) for example in examples],
            indent=2,
        ) + "\n"
    )
    (output / "gate_diagnostics.json").write_text(
        json.dumps(gate_diagnostics, indent=2, allow_nan=False) + "\n"
    )
    (output / "runtime_route_audit.json").write_text(
        json.dumps(route_audit, indent=2, allow_nan=False) + "\n"
    )
    (output / "wrong_relation_route_audit.json").write_text(
        json.dumps(
            wrong_relation_route_audit, indent=2, allow_nan=False
        ) + "\n"
    )

    emit(
        phase="fact_association_v2_arm_ready",
        arm=args.arm,
        gate=arm["gate"],
        objective=arm["objective"],
        train_route=route_audit["train"],
        development_route=route_audit["development"],
        wrong_relation_route=wrong_relation_route_audit,
        baseline_constraints=baseline_constraints,
    )

    if arm["objective"] == "absolute":
        report = train_row_wise(
            editor=editor,
            original_examples=examples,
            routed_answer=true_map,
            routed_unknown=unknown_map,
            fact_to_row=fact_to_row,
            plan=plan,
            output=output,
        )
    else:
        report = train_row_wise_constraints(
            editor=editor,
            original_examples=examples,
            routed_true=true_map,
            routed_comparator=comparator_map,
            routed_unknown=unknown_map,
            fact_to_row=fact_to_row,
            plan=plan,
            output=output,
        )

    absolute_metrics = routed_metrics(
        editor.model,
        true_map,
        unknown_map,
        plan["target_probability"],
    )
    final_constraints = constraint_metrics(
        editor.model,
        true_map,
        comparator_map,
        unknown_map,
        target_probability=plan["target_probability"],
        absolute_nll_buffer=plan["absolute_nll_buffer"],
        margin_target=plan["margin_target"],
    )
    artifact = editor.artifact()
    artifact["method"] = "static_overlap_fact_association_embeddings_v2_four_arm"
    artifact["v2_arm"] = args.arm
    artifact["v2_gate"] = arm["gate"]
    artifact["v2_objective"] = arm["objective"]
    artifact["development_seed"] = 1
    artifact["target_new_used_as_training_supervision"] = (
        arm["objective"] == "absolute_margin"
    )
    torch.save(artifact, output / "fact_association_embeddings.pt")

    report.update({
        "manifest": manifest,
        "absolute_metrics": absolute_metrics,
        "constraint_metrics": final_constraints,
        "runtime_counters": bank.counters(),
        "official_evaluation_started": False,
        "seed1_status": "development_after_V1_official_inspection",
    })
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    emit(
        status="fact_association_v2_arm_complete",
        arm=args.arm,
        stop_reason=report["stop_reason"],
        best_step=report["best_step"],
        absolute_metrics=absolute_metrics,
        constraint_metrics=final_constraints,
        artifact=str(output / "fact_association_embeddings.pt"),
        official_evaluation_started=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
