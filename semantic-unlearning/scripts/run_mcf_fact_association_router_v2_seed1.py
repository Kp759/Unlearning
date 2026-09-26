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
    block_output_capture,
    build_forget_examples,
    build_semantic_keys,
    make_subject_patterns,
    make_unknown_examples,
)
from static_overlap_natural_writer import mcf_facts
from fact_association_router_v2 import (
    build_direct_prompt_context_gate,
    prompt_map_from_examples,
)
from static_overlap_fact_association_v2_gate import (
    RelationPrototypeAssociationBank,
)

METHOD_V2 = "static_overlap_fact_association_embeddings_router_v2"


@torch.no_grad()
def boundary_norms(model, tokenizer, prompts, layers, batch_size=16):
    """L2 norm of each block's raw output at the final prompt token."""
    device = next(model.parameters()).device
    result = {int(layer): [] for layer in layers}
    for start in range(0, len(prompts), int(batch_size)):
        encoded = tokenizer(
            prompts[start:start + int(batch_size)],
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(device)
        captures = [block_output_capture(model, layer) for layer in result]
        try:
            model(**encoded, use_cache=False)
        finally:
            for _, handle in captures:
                handle.remove()
        mask = encoded["attention_mask"].bool()
        positions = (
            torch.arange(mask.shape[1], device=device)[None, :]
            .expand_as(mask)
            .masked_fill(~mask, -1)
            .max(dim=1)
            .values
        )
        index = torch.arange(mask.shape[0], device=device)
        for layer, (captured, _) in zip(result, captures):
            hidden = captured["hidden"].float()[index, positions]
            result[layer].extend(hidden.norm(dim=-1).cpu().tolist())
    return {layer: torch.tensor(values) for layer, values in result.items()}


def oracle_route_map(tokenizer, example_maps, fact_to_row):
    """{prompt-prefix tokens: row} for every training-visible prompt.

    Two spellings per example: the teacher-forced prefix the trainer binds
    (tokens before the first labelled position) and the bare tokenized prompt
    the route audit uses. A prefix owned by two facts is an error.
    """
    mapping = {}
    for examples in example_maps:
        for example in examples.values():
            row = fact_to_row[example.fact_id]
            first = next(
                i for i, label in enumerate(example.labels) if label != -100
            )
            keys = (
                tuple(example.input_ids[:first]),
                tuple(tokenizer(example.prompt)["input_ids"]),
            )
            for key in keys:
                if mapping.setdefault(key, row) != row:
                    raise ValueError(
                        f"Prompt prefix is shared by two facts: {example.prompt!r}"
                    )
    return mapping


def resolve_norm_scale(value, norms, layer, reference_layer):
    if str(value).strip().lower() == "auto":
        return float(norms[layer].median() / norms[reference_layer].median())
    scale = float(value)
    if not scale > 0:
        raise ValueError("--norm-scale must be positive or 'auto'")
    return scale


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
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--training-route",
        choices=("gate", "oracle"),
        default="gate",
        help=(
            "gate: rows are trained under Router V2 routing at --layer (shipped). "
            "oracle: rows are trained with ground-truth routing on the "
            "training-visible prompts, so the write layer can be varied without "
            "the read layer's quality deciding which prompts get a row. The "
            "saved artifact always routes by gate."
        ),
    )
    parser.add_argument(
        "--norm-scale",
        default="1",
        help=(
            "Multiply the learning rate and every trust radius by this factor. "
            "'auto' = median boundary-token norm at --layer divided by that at "
            "--norm-reference-layer, so the per-step edit is the same fraction "
            "of the residual stream at every layer."
        ),
    )
    parser.add_argument(
        "--norm-reference-layer", type=int, default=PLAN["layer"]
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
    block_count = len(model.model.layers)
    for name in ("layer", "norm_reference_layer"):
        if not 0 <= int(getattr(args, name)) < block_count:
            raise ValueError(f"--{name.replace('_', '-')} must lie in [0, {block_count - 1}]")

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
        method=METHOD_V2,
        forget_facts=len(facts),
        train_views=sum(e.split == "train" for e in examples),
        development_views=sum(e.split == "development" for e in examples),
        official_eff_canonical_training_visible=True,
        official_paraphrase_fields_used=False,
        official_neighborhood_fields_used=False,
        private_tokens=False,
        base_weights_trainable=False,
    )

    train_prompts = [e.prompt for e in examples if e.split == "train"]
    norms = boundary_norms(
        model, tokenizer, train_prompts, sorted({args.layer, args.norm_reference_layer})
    )
    norm_scale = resolve_norm_scale(
        args.norm_scale, norms, args.layer, args.norm_reference_layer
    )
    representation = {
        "layer": int(args.layer),
        "block_count": block_count,
        "relative_depth": round(args.layer / max(block_count - 1, 1), 4),
        "boundary_norm_median": float(norms[args.layer].median()),
        "boundary_norm_mean": float(norms[args.layer].mean()),
        "reference_layer": int(args.norm_reference_layer),
        "reference_boundary_norm_median": float(
            norms[args.norm_reference_layer].median()
        ),
        "norm_scale_argument": str(args.norm_scale),
        "norm_scale": norm_scale,
        "training_route": args.training_route,
        "hidden_source": "raw decoder-block output (pre final norm)",
    }
    emit(phase="layer_representation_ready", **representation)

    positive_prompts = prompt_map_from_examples(examples, split="train")
    (
        positive_prototypes,
        negative_prototypes,
        alpha,
        tau,
        gate_diagnostics,
    ) = build_direct_prompt_context_gate(
        model=model,
        tokenizer=tokenizer,
        facts=facts,
        positive_prompts_by_fact=positive_prompts,
        layer=args.layer,
        negative_count=12,
        margin_slack=0.02,
    )
    (output / "gate_diagnostics.json").write_text(
        json.dumps(gate_diagnostics, indent=2, allow_nan=False) + "\n"
    )
    emit(
        phase="fact_association_router_v2_gate_ready",
        layer=args.layer,
        unique_subject_bypass=False,
        mean_training_negative_fire_fraction=gate_diagnostics[
            "mean_training_negative_fire_fraction"
        ],
    )

    subject_patterns = make_subject_patterns(tokenizer, facts)

    def make_v2_bank():
        return RelationPrototypeAssociationBank(
            base_model=model,
            layer=args.layer,
            positive_prototypes=positive_prototypes,
            negative_prototypes=negative_prototypes,
            alpha=alpha,
            tau=tau,
            subject_patterns=subject_patterns,
            facts=facts,
            ambiguity_margin=0.02,
        )

    bank = make_v2_bank()
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
        bank = make_v2_bank()
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
    preflight = {
        "train_recall_complete": (
            route_audit["train"]["correct_row_active_fraction"] >= 1.0
        ),
        "development_recall_above_floor": (
            route_audit["development"]["correct_row_active_fraction"]
            >= float(args.min_dev_route_recall)
        ),
    }
    if args.training_route == "gate":
        if not preflight["train_recall_complete"]:
            raise RuntimeError(
                "Automatic association gate misses fitting prompts; refusing expensive training"
            )
        if not preflight["development_recall_above_floor"]:
            raise RuntimeError(
                "Automatic association gate development recall is below the "
                f"{args.min_dev_route_recall:.3f} preflight floor; refusing expensive training"
            )
    else:
        # The V2 gate's quality at this layer is a read-side result, recorded
        # rather than enforced: under oracle training it does not decide
        # which prompts receive a row, and the learned router is refit later.
        emit(phase="oracle_training_gate_preflight", **preflight)

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
    plan["radius_schedule"] = tuple(
        (float(upper), float(radius) * norm_scale)
        for upper, radius in PLAN["radius_schedule"]
    )
    plan["learning_rate"] = float(PLAN["learning_rate"]) * norm_scale
    plan["training_route"] = args.training_route
    plan["norm_scale"] = norm_scale

    manifest = {
        "method": METHOD_V2,
        "architecture": "relation_prototype_fact_association_bank_v2",
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
            "complete subject-token eligibility plus direct-context confirmation "
            "for every candidate"
        ),
        "routing_policy": "subject_candidate_plus_direct_context_confirmation_v2",
        "unique_subject_bypass": False,
        "subject_scan_scope": "prompt_prefix_only",
        "teacher_forced_suffix_can_affect_routing": False,
        "runtime_trigger_uses_object": False,
        "object_role": "suppression target only",
        "official_eff_canonical_training_visible": True,
        "official_paraphrase_fields_used": False,
        "official_neighborhood_fields_used": False,
        "development_used_for_gradients": False,
        "development_used_for_key_or_threshold_fitting": False,
        "router_v2_negative_controls": (
            "same-subject competitors plus training-only subject-transplanted "
            "wrong-context direct prompts"
        ),
        "router_v2_target_new_or_eval_probe_use": False,
        "unmatched_inputs_follow_exact_frozen_base_path": True,
        "runtime_route_audit": route_audit,
        "gate_preflight": preflight,
        "training_route": args.training_route,
        "oracle_routes_used_for": (
            "row training and checkpoint selection on training-visible prompts only"
            if args.training_route == "oracle" else None
        ),
        "layer_representation": representation,
    }
    (output / "association_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    (output / "association_examples.json").write_text(
        json.dumps([asdict(e) for e in examples], indent=2, allow_nan=False) + "\n"
    )

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "mcf_router_v2_preflight_complete",
                    "optimization_started": False,
                    "route_audit": route_audit,
                    "gate_diagnostics": gate_diagnostics,
                    "output_dir": str(output),
                },
                indent=2,
                allow_nan=False,
            )
        )
        return 0

    # V2.1's optimizer is intentionally reused: one Adam instance per vector,
    # worst-view suppression before the threshold, abstention only after lock.
    if args.training_route == "oracle":
        bank.set_oracle_routes(
            oracle_route_map(tokenizer, (answer_map, unknown_map), fact_to_row)
        )
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
    final_metrics_routing = args.training_route
    bank.set_oracle_routes(None)
    row_norms = bank.extra.detach().float().norm(dim=-1).cpu()
    representation["row_norm_median"] = float(row_norms.median())
    representation["row_norm_max"] = float(row_norms.max())
    representation["row_to_boundary_norm_ratio_median"] = float(
        row_norms.median() / norms[args.layer].median()
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
            "method": METHOD_V2,
            "manifest": manifest,
            "gate_diagnostics": gate_diagnostics,
            "runtime_route_audit": route_audit,
            "final_metrics": final_metrics,
            "final_metrics_routing": final_metrics_routing,
            "layer_representation": representation,
            "runtime_counters": bank.counters(),
            "unmatched_neutral_logits_exact_base_after_training": True,
            "official_evaluation_started": False,
        }
    )
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )

    emit(
        status="fact_association_router_v2_training_complete",
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
