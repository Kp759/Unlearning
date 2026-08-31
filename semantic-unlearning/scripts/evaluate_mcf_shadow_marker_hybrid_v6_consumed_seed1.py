#!/usr/bin/env python3
"""Consumed seed-1 replay of V6 routing around the frozen V3.6.2 mechanism."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

import evaluate_mcf_embedding_keyed_neuron_v3_6_2_official as v3_eval
import evaluate_mcf_exact_subject_target_sidecar_v5_seed1_development as v5_dev
import evaluate_mcf_normalization_preserving_sidecar_v6_consumed as v6_eval
import mcf_normalization_preserving_sidecar_v6_core as v6_core
import mcf_shadow_marker_hybrid_v6_core as hybrid_core
import mcf_zero_unlearn_official_eval as official
import scoped_span_edit as scoped
from mcf_sampling import sample_official_mcf_records


PROTOCOL = "mcf_shadow_marker_hybrid_v6_consumed_seed1_mechanistic_replay_v1"
ARM_ORDER = (
    "reconstructed_base",
    "routed_shadow_only",
    "outer_detector_without_writer",
    "outer_detector_with_writer",
    "outer_direct_without_writer",
    "outer_direct_with_writer",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--v3-6-2-training-run-dir", required=True)
    parser.add_argument("--stage1-writer-state", required=True)
    parser.add_argument("--v6-candidate-run-dir", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--wikidata-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--route-audit-batch-size", type=int, default=128)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.route_audit_batch_size != 128:
        parser.error("hybrid route-audit batch size is frozen at 128")
    return args


def _arm_switches(label: str) -> tuple[bool, bool, str]:
    table = {
        "reconstructed_base": (False, False, "outer_and_detector"),
        "routed_shadow_only": (True, False, "outer_and_detector"),
        "outer_detector_without_writer": (False, True, "outer_and_detector"),
        "outer_detector_with_writer": (True, True, "outer_and_detector"),
        "outer_direct_without_writer": (False, True, "outer_direct"),
        "outer_direct_with_writer": (True, True, "outer_direct"),
    }
    return table[label]


@torch.no_grad()
def evaluate_arm(
    label: str,
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    writer: hybrid_core.RoutedShadowEmbeddingDelta,
    bank: hybrid_core.OuterRoutedThresholdGatedActuatorBank,
    forget_records: Sequence[Mapping[str, Any]],
    retain_records: Sequence[Mapping[str, Any]],
    ppl_text: str,
) -> tuple[dict[str, Any], Any, Any]:
    writer_enabled, actuator_enabled, mode = _arm_switches(label)
    writer.enabled = writer_enabled
    bank.enabled = actuator_enabled
    bank.write_enabled = actuator_enabled
    bank.outer_gate_mode = mode
    model.eval()
    device = next(model.parameters()).device
    llama_like = official.is_llama_like(model, tokenizer)
    forget, forget_raw = official.evaluate_record_split(
        model, tokenizer, forget_records, device, llama_like, "forget"
    )
    retain, retain_raw = official.evaluate_record_split(
        model, tokenizer, retain_records, device, llama_like, "retain"
    )
    ppl = official.official_perplexity(model, tokenizer, ppl_text, device)
    return (
        {
            "switches": {
                "two_sided_outer_router_enabled": True,
                "shadow_embedding_writer_enabled": writer_enabled,
                "width16_actuator_enabled": actuator_enabled,
                "record_gate_mode": mode,
            },
            "forget": forget,
            "retain": retain,
            "PPL": ppl,
        },
        forget_raw,
        retain_raw,
    )


def _metric_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        float(left[key]) == float(right[key])
        for key in ("Eff", "Gen", "Spe", "minimum_rewrite_paraphrase_margin")
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model_path = Path(args.model_path).resolve()
    training_root = Path(args.v3_6_2_training_run_dir).resolve()
    stage1_path = Path(args.stage1_writer_state).resolve()
    v6_root = Path(args.v6_candidate_run_dir).resolve()
    mcf_path = Path(args.mcf_path).resolve()
    wikidata_dir = Path(args.wikidata_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    candidate_path = training_root / "method" / "v3_6_2_candidate_state.pt"
    v6_candidate_path = v6_root / "v6_normalization_preserving_sidecar.pt"
    v6_completion_path = v6_root / "completion.json"
    for path in (
        model_path,
        training_root,
        stage1_path,
        candidate_path,
        v6_candidate_path,
        v6_completion_path,
        mcf_path,
        wikidata_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    v3_state = torch.load(candidate_path, map_location="cpu", weights_only=False)
    if not isinstance(v3_state, Mapping):
        raise RuntimeError("V3.6.2 candidate is not a mapping")
    v3_eval.validate_candidate_structure(v3_state)
    lineage = v3_eval.validate_training_lineage(
        v3_state,
        training_root=training_root,
        stage1_path=stage1_path,
        candidate_path=candidate_path,
    )
    v6_state = torch.load(v6_candidate_path, map_location="cpu", weights_only=False)
    if not isinstance(v6_state, Mapping):
        raise RuntimeError("V6 candidate is not a mapping")
    v6_core.validate_candidate_state(v6_state)
    v6_completion = json.loads(v6_completion_path.read_text(encoding="utf-8"))
    if (
        int(v6_state["seed"]) != 1
        or v6_completion.get("passed") is not True
        or v6_completion.get("candidate_sha256") != sha256_file(v6_candidate_path)
        or v6_completion.get("eligible_for_consumed_development_replay") is not True
        or [int(item) for item in v3_state["case_ids"]]
        != [int(item) for item in v6_state["case_ids"]]
    ):
        raise RuntimeError("V3.6.2 and V6 seed-1 candidates are not bound")

    model, tokenizer, old_writer, old_bank, base_runtime = v3_eval._load_runtime(
        model_path=model_path,
        state=v3_state,
        expected_base_embedding_rows_sha256=str(
            lineage["base_selected_embedding_rows_sha256"]
        ),
    )
    old_bank.remove()
    old_writer.remove()
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if v6_core.tensor_sha256(input_layer.weight) != v6_state[
        "base_embedding_sha256"
    ] or v6_core.tensor_sha256(output_layer.weight) != v6_state["base_lm_head_sha256"]:
        raise RuntimeError("V6 route candidate differs from reconstructed Base")
    if scoped.build_subject_patterns(tokenizer, v6_state["subjects"]) != v6_state[
        "subject_patterns"
    ]:
        raise RuntimeError("V6 subject patterns do not reproduce in hybrid runtime")

    router = v6_core.TwoSidedEntityRelationRouter(
        input_layer,
        v6_state["subject_patterns"],
        subjects=v6_state["subjects"],
        relation_ids=v6_state["relation_ids"],
        frame_to_relation_ids=v6_state["frame_lexicon"]["frame_to_relation_ids"],
        tokenizer=tokenizer,
        model=model,
    )
    writer = hybrid_core.RoutedShadowEmbeddingDelta(
        input_layer,
        router,
        v3_state["selected_embedding_rows"],
        v3_state["embedding_delta"],
    )
    mlp = v3_eval.learner._resolve_swiglu_mlp(model, 27)
    bank = hybrid_core.OuterRoutedThresholdGatedActuatorBank(
        mlp,
        v3_state["actuator_neuron_ids"],
        v3_state["actuator_owner_indices"],
        outer_router=router,
        outer_gate_mode="outer_and_detector",
        detector_gate_rows=v3_state["detector_gate_rows"],
        detector_up_rows=v3_state["detector_up_rows"],
        detector_local_groups=v3_state["detector_local_groups"],
        detector_flat_signs=v3_state["detector_flat_signs"],
        off_boundary=float(v3_state["threshold_off_boundary"]),
        on_boundary=float(v3_state["threshold_on_boundary"]),
    )
    with torch.no_grad():
        bank.down_delta.copy_(
            v3_state["actuator_down_delta"].to(
                device=bank.down_delta.device,
                dtype=bank.down_delta.dtype,
            )
        )
    bank.down_delta.requires_grad_(False)
    bank.install(mlp)

    data = official.load_mcf(mcf_path)
    forget_raw_records, retain_raw_records = sample_official_mcf_records(
        data, 50, 1000, 1
    )
    forget_records = [official.normalize_record(dict(item)) for item in forget_raw_records]
    retain_records = [official.normalize_record(dict(item)) for item in retain_raw_records]
    data_bindings = [
        (
            str(record["requested_rewrite"]["subject"]),
            str(record["requested_rewrite"]["relation_id"]),
            str(record["requested_rewrite"]["target_new"]["str"]),
            str(record["requested_rewrite"]["target_true"]["str"]),
        )
        for record in forget_records
    ]
    state_bindings = list(
        zip(
            v6_state["subjects"],
            v6_state["relation_ids"],
            v6_state["target_new"],
            v6_state["target_true"],
        )
    )
    if data_bindings != state_bindings:
        raise RuntimeError("consumed seed-1 data differs from hybrid candidates")
    ppl_text = official.load_official_ppl_text(wikidata_dir)
    if not ppl_text:
        raise RuntimeError("PPL text is unavailable")

    output.mkdir(parents=True)
    versions_before = v3_eval.parameter_versions(model)
    input_hash_before = v6_core.tensor_sha256(input_layer.weight)
    output_hash_before = v6_core.tensor_sha256(output_layer.weight)
    actuator_hash_before = v3_eval.tensor_sha256(bank.down_delta)
    detector_gate_hash_before = v3_eval.tensor_sha256(bank.detector_gate_rows)
    detector_up_hash_before = v3_eval.tensor_sha256(bank.detector_up_rows)
    writer_hash_before = v3_eval.tensor_sha256(writer.delta)

    try:
        routing = v6_eval.route_audit(
            router,
            tokenizer,
            forget_records,
            retain_records,
            ppl_text=ppl_text,
            batch_size=int(args.route_audit_batch_size),
        )
        write_json(output / "route_audit.json", routing)
        if not routing["route_coverage_gate_passed"]:
            raise RuntimeError("hybrid V6 route coverage failed before replay")

        arms: Dict[str, Dict[str, Any]] = {}
        raw_by_arm: Dict[str, tuple[Any, Any]] = {}
        for label in ARM_ORDER:
            print(f"Hybrid consumed replay arm: {label}")
            summary, arm_forget_raw, arm_retain_raw = evaluate_arm(
                label,
                model=model,
                tokenizer=tokenizer,
                writer=writer,
                bank=bank,
                forget_records=forget_records,
                retain_records=retain_records,
                ppl_text=ppl_text,
            )
            arms[label] = summary
            raw_by_arm[label] = (arm_forget_raw, arm_retain_raw)
            write_json(output / "arms" / f"{label}_forget_raw.json", arm_forget_raw)
            write_json(output / "arms" / f"{label}_retain_raw.json", arm_retain_raw)
    finally:
        bank.remove()
        writer.remove()
        router.close()

    base = arms["reconstructed_base"]
    preservation: Dict[str, Any] = {}
    for label in ARM_ORDER[1:]:
        preservation[label] = v5_dev.exact_preservation_comparison(
            raw_by_arm["reconstructed_base"][0],
            raw_by_arm[label][0],
            raw_by_arm["reconstructed_base"][1],
            raw_by_arm[label][1],
            base_ppl=float(base["PPL"]),
            candidate_ppl=float(arms[label]["PPL"]),
        )

    mechanistic = {
        "outer_detector_writer_necessary_behaviorally": not _metric_equal(
            arms["outer_detector_with_writer"]["forget"],
            arms["outer_detector_without_writer"]["forget"],
        ),
        "outer_direct_writer_necessary_behaviorally": not _metric_equal(
            arms["outer_direct_with_writer"]["forget"],
            arms["outer_direct_without_writer"]["forget"],
        ),
        "detector_recall_limits_gen": float(
            arms["outer_direct_with_writer"]["forget"]["Gen"]
        )
        < float(arms["outer_detector_with_writer"]["forget"]["Gen"]),
        "outer_detector_forget": {
            "Eff": float(arms["outer_detector_with_writer"]["forget"]["Eff"]),
            "Gen": float(arms["outer_detector_with_writer"]["forget"]["Gen"]),
        },
        "outer_direct_forget": {
            "Eff": float(arms["outer_direct_with_writer"]["forget"]["Eff"]),
            "Gen": float(arms["outer_direct_with_writer"]["forget"]["Gen"]),
        },
        "interpretation_boundary": (
            "outer_direct success is evidence about frozen actuator capacity and "
            "detector recall; it is not evidence that the shadow writer is necessary"
        ),
    }
    integrity = {
        "base_parameter_versions_unchanged": v3_eval.parameter_versions_unchanged(
            versions_before, v3_eval.parameter_versions(model)
        )["passed"],
        "embedding_hash_unchanged": input_hash_before
        == v6_core.tensor_sha256(input_layer.weight),
        "lm_head_hash_unchanged": output_hash_before
        == v6_core.tensor_sha256(output_layer.weight),
        "writer_delta_unchanged": writer_hash_before
        == v3_eval.tensor_sha256(writer.delta),
        "detector_gate_unchanged": detector_gate_hash_before
        == v3_eval.tensor_sha256(bank.detector_gate_rows),
        "detector_up_unchanged": detector_up_hash_before
        == v3_eval.tensor_sha256(bank.detector_up_rows),
        "actuator_delta_unchanged": actuator_hash_before
        == v3_eval.tensor_sha256(bank.down_delta),
        "v3_candidate_bit_identical": sha256_file(candidate_path)
        == v3_eval.EXPECTED_CANDIDATE_SHA256,
        "v6_candidate_bit_identical": sha256_file(v6_candidate_path)
        == v6_completion["candidate_sha256"],
        "optimizer_constructed": False,
        "gradient_updates_performed": 0,
        "checkpoint_saved": False,
    }
    integrity["passed"] = all(
        bool(value)
        for key, value in integrity.items()
        if key not in ("optimizer_constructed", "gradient_updates_performed", "checkpoint_saved")
    )
    result = {
        "schema_version": 1,
        "kind": "mcf_shadow_marker_hybrid_v6_consumed_seed1_mechanistic_replay",
        "protocol": PROTOCOL,
        "completed": True,
        "evaluation_status": "consumed_development_not_blind_not_official",
        "seed": 1,
        "fixed_arm_order": list(ARM_ORDER),
        "architecture": {
            "outer_router": "V6 two-sided frozen entity/relation grammar",
            "shadow_writer": "exact frozen V6.2 embedding delta, routed and reversible",
            "learned_detector": "exact frozen V3.5.4 four-neuron-per-record detector",
            "actuator": "exact frozen V3.6.2 separate width-16 actuator",
            "base_parameters_mutated": False,
        },
        "arms": arms,
        "exact_preservation_vs_base": preservation,
        "route_audit": routing,
        "mechanistic_conclusions": mechanistic,
        "integrity": integrity,
        "lineage": {
            "v3_candidate_sha256": sha256_file(candidate_path),
            "v6_candidate_sha256": sha256_file(v6_candidate_path),
            "v3_training_acceptance": lineage["training_acceptance"],
            "base_runtime": base_runtime,
        },
        "used_for_checkpoint_selection_or_retry": False,
        "fresh_seed_claim_permitted": False,
        "claim_scope": "mechanistic_consumed_development_ablation",
    }
    write_json(output / "hybrid_replay.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
