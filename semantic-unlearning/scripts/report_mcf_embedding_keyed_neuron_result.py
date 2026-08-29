#!/usr/bin/env python3
"""Build the preregistered conditional-suppression mechanism report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


METRICS = ("Eff", "Gen", "Spe", "Spe_success")
EXPECTED_PROTOCOL = "mcf_embedding_keyed_sparse_neuron_suppression_v3"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--edited", required=True)
    parser.add_argument("--method-summary", required=True)
    parser.add_argument("--post-reload-acceptance", required=True)
    parser.add_argument("--mlp-only")
    parser.add_argument("--mlp-only-method-summary")
    parser.add_argument("--mlp-only-post-reload-acceptance")
    parser.add_argument("--mlp-only-retain-tail-audit")
    parser.add_argument("--writer-portability-audit")
    parser.add_argument("--retain-tail-audit")
    parser.add_argument("--latent-recovery-audit")
    parser.add_argument("--relearning-audit")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-abs-spe-delta", type=float, default=0.2)
    parser.add_argument("--max-abs-retain-eff-gen-delta", type=float, default=1.0)
    parser.add_argument("--max-ppl-percent-delta", type=float, default=5.0)
    parser.add_argument(
        "--allow-incomplete-mechanism-evidence",
        action="store_true",
        help="Create a diagnostic report without requiring all paper-level endpoints.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _load(path: str) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _load_optional(path: str | None) -> Dict[str, Any] | None:
    return _load(path) if path else None


def _metrics(payload: Mapping[str, Any], split: str) -> Dict[str, float]:
    value = payload.get(split)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"evaluation lacks {split!r} metrics")
    result = {metric: float(value[metric]) for metric in METRICS}
    if not all(math.isfinite(number) for number in result.values()):
        raise RuntimeError(f"non-finite {split} metric")
    return result


def _evaluation_comparison(
    base: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Dict[str, Any]:
    base_forget = _metrics(base, "forget")
    candidate_forget = _metrics(candidate, "forget")
    base_retain = _metrics(base, "retain")
    candidate_retain = _metrics(candidate, "retain")
    base_ppl = float(base["PPL"])
    candidate_ppl = float(candidate["PPL"])
    return {
        "forget": candidate_forget,
        "retain": candidate_retain,
        "PPL": candidate_ppl,
        "delta": {
            "forget": {
                key: candidate_forget[key] - base_forget[key] for key in METRICS
            },
            "retain": {
                key: candidate_retain[key] - base_retain[key] for key in METRICS
            },
            "PPL": candidate_ppl - base_ppl,
            "PPL_percent": 100.0
            * (candidate_ppl - base_ppl)
            / max(abs(base_ppl), 1e-12),
        },
    }


def _metric_envelope(
    comparison: Mapping[str, Any],
    *,
    max_abs_spe_delta: float,
    max_abs_retain_eff_gen_delta: float,
    max_ppl_percent_delta: float,
) -> Dict[str, bool]:
    forget = comparison["forget"]
    retain_delta = comparison["delta"]["retain"]
    forget_delta = comparison["delta"]["forget"]
    return {
        "Eff_zero": float(forget["Eff"]) == 0.0,
        "Gen_zero": float(forget["Gen"]) == 0.0,
        "forget_Spe_local": abs(float(forget_delta["Spe"])) <= float(max_abs_spe_delta),
        "retain_Spe_local": abs(float(retain_delta["Spe"])) <= float(max_abs_spe_delta),
        "retain_Eff_local": abs(float(retain_delta["Eff"]))
        <= float(max_abs_retain_eff_gen_delta),
        "retain_Gen_local": abs(float(retain_delta["Gen"]))
        <= float(max_abs_retain_eff_gen_delta),
        "PPL_local": abs(float(comparison["delta"]["PPL_percent"]))
        <= float(max_ppl_percent_delta),
    }


def _budget_match(
    proposed: Mapping[str, Any], control: Mapping[str, Any]
) -> Dict[str, Any]:
    first = proposed.get("optimization_budget")
    second = control.get("optimization_budget")
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        return {"passed": False, "reason": "optimization budget missing"}
    required = (
        "detector_response_mode",
        "dormant_fraction",
        "selection_stability_weight",
        "selection_positive_contexts",
        "selection_negative_contexts",
        "detector_steps",
        "detector_lr",
        "detector_record_batch",
        "detector_positive_contexts",
        "detector_negative_contexts",
        "detector_positive_floor",
        "detector_off_abs_max",
        "detector_negative_weight",
        "detector_cross_weight",
        "detector_consistency_weight",
        "detector_l2",
        "detector_relative_cap",
        "actuator_steps",
        "actuator_lr",
        "actuator_batch_size",
        "actuator_protected_batch",
        "actuator_writer_off_every",
        "actuator_relative_cap",
        "actuator_l2",
        "neurons_per_record",
        "selected_existing_mlp_neurons",
        "mlp_layer",
        "protected_prompt_count",
        "protected_kl_weight",
        "margin_weight",
        "reference_nll_weight",
        "reference_nll_tolerance",
        "forget_margin",
        "grad_clip",
        "kl_topk",
    )
    comparisons = {
        key: {
            "proposed": first.get(key),
            "mlp_only": second.get(key),
            "present": key in first and key in second,
            "equal": first.get(key) == second.get(key),
        }
        for key in required
    }
    return {
        "passed": all(row["present"] and row["equal"] for row in comparisons.values()),
        "fields": comparisons,
    }


def _evaluation_binding_match(
    *payloads: Mapping[str, Any],
) -> Dict[str, Any]:
    fields = ("dataset", "sample_mode", "seed", "unlearn_num", "retain_num")
    rows: Dict[str, Any] = {}
    for field in fields:
        values = [payload.get(field) for payload in payloads]
        rows[field] = {
            "values": values,
            "present": all(value is not None for value in values),
            "equal": len(set(values)) == 1
            if all(value is not None for value in values)
            else False,
        }
    for split in ("forget", "retain"):
        signatures = [
            _evaluation_split_signature(payload, split) for payload in payloads
        ]
        rows[f"{split}_sample_signature"] = {
            "values": signatures,
            "present": all(value is not None for value in signatures),
            "equal": len(set(signatures)) == 1
            if all(value is not None for value in signatures)
            else False,
        }
    return {
        "passed": all(row["present"] and row["equal"] for row in rows.values()),
        "fields": rows,
    }


def _evaluation_split_signature(payload: Mapping[str, Any], split: str) -> str | None:
    rows = payload.get(f"{split}_raw")
    if not isinstance(rows, list) or not rows:
        return None
    identities = []
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        rewrite = row.get("requested_rewrite")
        if not isinstance(rewrite, Mapping):
            return None
        identities.append(rewrite)
    encoded = json.dumps(
        identities, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _training_binding_match(
    proposed: Mapping[str, Any], control: Mapping[str, Any]
) -> Dict[str, Any]:
    first_firewall = proposed.get("data_firewall", {})
    second_firewall = control.get("data_firewall", {})
    values = {
        "seed": (proposed.get("seed"), control.get("seed")),
        "forget_num": (proposed.get("forget_num"), control.get("forget_num")),
        "training_visible_sha256": (
            first_firewall.get("training_visible_sha256"),
            second_firewall.get("training_visible_sha256"),
        ),
        "context_manifest_sha256": (
            first_firewall.get("context_manifest_sha256"),
            second_firewall.get("context_manifest_sha256"),
        ),
        "split_manifest_sha256": (
            first_firewall.get("split_manifest_sha256"),
            second_firewall.get("split_manifest_sha256"),
        ),
        "stage1_state_sha256": (
            first_firewall.get("stage1_state_sha256"),
            second_firewall.get("stage1_state_sha256"),
        ),
        "stage1_report_sha256": (
            first_firewall.get("stage1_report_sha256"),
            second_firewall.get("stage1_report_sha256"),
        ),
        "stage1_writer_log_sha256": (
            first_firewall.get("stage1_writer_log_sha256"),
            second_firewall.get("stage1_writer_log_sha256"),
        ),
        "clean_stage1_acceptance_sha256": (
            first_firewall.get("clean_stage1_acceptance_sha256"),
            second_firewall.get("clean_stage1_acceptance_sha256"),
        ),
        "clean_stage1_portability_sha256": (
            first_firewall.get("clean_stage1_portability_sha256"),
            second_firewall.get("clean_stage1_portability_sha256"),
        ),
        "experiment_registry_sha256": (
            first_firewall.get("experiment_registry_sha256"),
            second_firewall.get("experiment_registry_sha256"),
        ),
    }
    rows = {
        key: {
            "proposed": pair[0],
            "mlp_only": pair[1],
            "present": pair[0] is not None and pair[1] is not None,
            "equal": pair[0] == pair[1],
        }
        for key, pair in values.items()
    }
    return {
        "passed": all(row["present"] and row["equal"] for row in rows.values()),
        "fields": rows,
    }


def _audit_passed(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping) and "passed" in acceptance:
        return bool(acceptance["passed"])
    return bool(payload.get("passed"))


def _json_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _post_freeze_audit_binding(
    payload: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any],
    *,
    writer_mode: str,
    require_retain_coverage: bool = False,
) -> bool:
    if not isinstance(payload, Mapping):
        return False
    bound = bool(
        payload.get("dataset") == evaluation.get("dataset") == "MCF"
        and _json_int(payload.get("seed")) is not None
        and _json_int(payload.get("seed")) == _json_int(evaluation.get("seed"))
        and _json_int(payload.get("unlearn_num")) is not None
        and _json_int(payload.get("unlearn_num"))
        == _json_int(evaluation.get("unlearn_num"))
        and payload.get("sample_mode") == evaluation.get("sample_mode")
        and payload.get("writer_mode") == writer_mode
    )
    if not require_retain_coverage:
        return bound
    audit_retain = _json_int(payload.get("retain_num"))
    evaluation_retain = _json_int(evaluation.get("retain_num"))
    return bool(
        bound
        and audit_retain is not None
        and evaluation_retain is not None
        and audit_retain >= evaluation_retain
    )


def _retain_tail_complete(payload: Mapping[str, Any] | None) -> bool:
    """Require the registered audit type and its full 100k-prompt denominator."""
    if not isinstance(payload, Mapping):
        return False
    groups = payload.get("groups")
    overall = groups.get("all") if isinstance(groups, Mapping) else None
    acceptance = payload.get("acceptance")
    checks = acceptance.get("checks") if isinstance(acceptance, Mapping) else None
    required_checks = {
        "minimum_unique_prompt_count",
        "response_rate_at_most_24_over_13000",
        "response_wilson_upper_at_most_24_over_13000",
        "top1_change_rate_at_most_24_over_13000",
    }
    return bool(
        payload.get("kind")
        == "mcf_embedding_keyed_neuron_post_freeze_retain_tail_audit"
        and payload.get("unique_prompts") is True
        and payload.get("used_for_training_checkpoint_selection_or_retry") is False
        and isinstance(overall, Mapping)
        and (_json_int(overall.get("prompt_count")) or 0) >= 100_000
        and isinstance(checks, Mapping)
        and required_checks.issubset(checks)
        and all(isinstance(checks[name], bool) for name in required_checks)
        and isinstance(acceptance.get("passed"), bool)
    )


def _writer_portability_complete(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    by_type = payload.get("by_prompt_type")
    return bool(
        payload.get("kind") == "mcf_frozen_stage1_writer_official_portability_audit"
        and payload.get("threshold_binding_passed") is True
        and payload.get("decoder_loaded") is False
        and payload.get("writer_parameters_updated") is False
        and payload.get("used_for_training_checkpoint_selection_or_retry") is False
        and (_json_int(payload.get("prompt_count")) or 0) > 0
        and (_json_int(payload.get("record_count")) or 0) > 0
        and _json_int(payload.get("record_count"))
        == _json_int(payload.get("unlearn_num"))
        and isinstance(by_type, Mapping)
        and all(
            isinstance(by_type.get(prompt_type), Mapping)
            and (_json_int(by_type[prompt_type].get("prompt_count")) or 0) > 0
            for prompt_type in ("rewrite", "paraphrase")
        )
        and isinstance(payload.get("acceptance"), Mapping)
        and isinstance(payload["acceptance"].get("passed"), bool)
    )


def _latent_endpoint_complete(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    modes = payload.get("modes")
    return bool(
        payload.get("kind")
        == "mcf_embedding_keyed_neuron_post_freeze_latent_recovery_audit"
        and isinstance(payload.get("fact_recoverable"), bool)
        and (_json_int(payload.get("prompt_count")) or 0) > 0
        and (_json_int(payload.get("record_count")) or 0) > 0
        and _json_int(payload.get("record_count"))
        == _json_int(payload.get("unlearn_num"))
        and payload.get("used_for_training_checkpoint_selection_or_retry") is False
        and isinstance(modes, Mapping)
        and all(
            isinstance(modes.get(mode), Mapping)
            for mode in ("edited", "reconstructed_base")
        )
        and "final_model_output" in modes["edited"]
        and "final_model_output" in modes["reconstructed_base"]
        and payload.get("positive_control_passed") is True
    )


def _relearning_endpoint_complete(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    curve = payload.get("curve")
    attack = payload.get("attack")
    maximum_steps = attack.get("maximum_steps") if isinstance(attack, Mapping) else None
    observed_steps = set()
    if isinstance(curve, list):
        observed_steps = {
            step
            for row in curve
            if isinstance(row, Mapping)
            for step in [_json_int(row.get("step"))]
            if step is not None
        }
    return bool(
        payload.get("kind")
        == "mcf_embedding_keyed_neuron_post_freeze_relearning_attack"
        and isinstance(payload.get("fact_recoverable"), bool)
        and (_json_int(payload.get("record_count")) or 0) > 0
        and _json_int(payload.get("record_count"))
        == _json_int(payload.get("unlearn_num"))
        and payload.get("used_for_training_checkpoint_selection_or_retry") is False
        and isinstance(curve, list)
        and bool(curve)
        and _json_int(maximum_steps) is not None
        and 0 in observed_steps
        and maximum_steps in observed_steps
        and isinstance(payload.get("reconstructed_base_positive_control"), Mapping)
        and payload.get("positive_control_passed") is True
    )


def build_report(
    base: Mapping[str, Any],
    edited: Mapping[str, Any],
    method: Mapping[str, Any],
    reload: Mapping[str, Any],
    *,
    max_abs_spe_delta: float,
    max_abs_retain_eff_gen_delta: float,
    max_ppl_percent_delta: float,
    mlp_only: Mapping[str, Any] | None = None,
    mlp_only_method: Mapping[str, Any] | None = None,
    mlp_only_reload: Mapping[str, Any] | None = None,
    mlp_only_retain_tail: Mapping[str, Any] | None = None,
    writer_portability: Mapping[str, Any] | None = None,
    retain_tail: Mapping[str, Any] | None = None,
    latent_recovery: Mapping[str, Any] | None = None,
    relearning: Mapping[str, Any] | None = None,
    require_complete_mechanism_evidence: bool = False,
) -> Dict[str, Any]:
    base_summary = {
        "forget": _metrics(base, "forget"),
        "retain": _metrics(base, "retain"),
        "PPL": float(base["PPL"]),
    }
    proposed = _evaluation_comparison(base, edited)
    metric_checks = _metric_envelope(
        proposed,
        max_abs_spe_delta=max_abs_spe_delta,
        max_abs_retain_eff_gen_delta=max_abs_retain_eff_gen_delta,
        max_ppl_percent_delta=max_ppl_percent_delta,
    )
    method_acceptance = method.get("acceptance", {})
    causal = method.get("causal_component_ablation", {})
    architecture = method.get("architecture", {})
    firewall = method.get("data_firewall", {})
    data_access = (
        firewall.get("data_access", {}) if isinstance(firewall, Mapping) else {}
    )
    clean_stage1 = (
        firewall.get("clean_stage1_writer", {}) if isinstance(firewall, Mapping) else {}
    )
    clean_stage1_acceptance = (
        firewall.get("clean_stage1_acceptance", {})
        if isinstance(firewall, Mapping)
        else {}
    )
    checks: Dict[str, bool] = {
        **metric_checks,
        "locked_training_acceptance": bool(method_acceptance.get("passed")),
        "fresh_reload_acceptance": bool(reload.get("passed")),
        "within_checkpoint_writer_dependence": bool(causal.get("writer_is_necessary")),
        "within_checkpoint_neuron_dependence": bool(causal.get("decoder_is_necessary")),
        "detector_gate_passed": bool(method_acceptance.get("detector_gate_passed")),
        "LM_head_unchanged": architecture.get("lm_head_edited") is False
        and bool(method_acceptance.get("lm_head_bit_identical")),
        "no_router_or_sidecar": not any(
            bool(architecture.get(key))
            for key in (
                "runtime_string_matcher",
                "external_router",
                "retrieval_cache",
                "sidecar",
            )
        ),
        "no_evaluation_training_access": data_access
        == {
            "official_paraphrases_seen": 0,
            "official_neighborhoods_seen": 0,
            "benchmark_retain_seen": 0,
            "official_ppl_seen": False,
        },
        "clean_from_scratch_stage1_writer": bool(
            isinstance(clean_stage1, Mapping)
            and clean_stage1.get("from_scratch") is True
            and int(clean_stage1.get("writer_steps", 0)) == 1200
            and clean_stage1.get("positive_context_policy")
            == "relation_templates_only_v1"
            and int(clean_stage1.get("writer_log_event_count", 0)) > 0
        ),
        "clean_stage1_integrity_and_portability_accepted": bool(
            isinstance(clean_stage1_acceptance, Mapping)
            and clean_stage1_acceptance.get("kind")
            == "mcf_clean_stage1_writer_acceptance"
            and clean_stage1_acceptance.get("passed") is True
            and float(
                clean_stage1_acceptance.get("training_safe_portability", {}).get(
                    "amplitude_threshold", float("nan")
                )
            )
            == 4.5
            and float(
                clean_stage1_acceptance.get("training_safe_portability", {}).get(
                    "global_complete_fraction", 0.0
                )
            )
            >= 0.95
            and float(
                clean_stage1_acceptance.get("training_safe_portability", {}).get(
                    "minimum_record_complete_fraction", 0.0
                )
            )
            >= 0.80
        ),
    }

    matched_control: Dict[str, Any] = {"supplied": False}
    evaluation_binding = {"passed": False, "reason": "matched control missing"}
    if all(value is not None for value in (mlp_only, mlp_only_method, mlp_only_reload)):
        assert mlp_only is not None
        assert mlp_only_method is not None
        assert mlp_only_reload is not None
        control_comparison = _evaluation_comparison(base, mlp_only)
        control_envelope = _metric_envelope(
            control_comparison,
            max_abs_spe_delta=max_abs_spe_delta,
            max_abs_retain_eff_gen_delta=max_abs_retain_eff_gen_delta,
            max_ppl_percent_delta=max_ppl_percent_delta,
        )
        budget = _budget_match(method, mlp_only_method)
        training_binding = _training_binding_match(method, mlp_only_method)
        evaluation_binding = _evaluation_binding_match(base, edited, mlp_only)
        control_architecture = mlp_only_method.get("architecture", {})
        independently_retrained = bool(
            mlp_only_method.get("writer_mode") == "none"
            and control_architecture.get("input_embedding_rows_edited") == 0
            and mlp_only_method.get("experiment_label") == "mlp_only_retrained"
        )
        control_tail_complete = _retain_tail_complete(mlp_only_retain_tail)
        control_tail_bound = _post_freeze_audit_binding(
            mlp_only_retain_tail,
            mlp_only,
            writer_mode="none",
            require_retain_coverage=True,
        )
        control_tail_passed = bool(
            control_tail_complete
            and control_tail_bound
            and _audit_passed(mlp_only_retain_tail)
        )
        reload_verification_completed = bool(
            mlp_only_reload.get("verification_completed")
            or mlp_only_reload.get("passed")
        )
        control_meets_same_envelope = bool(
            all(control_envelope.values()) and control_tail_passed
        )
        comparison_is_valid = bool(
            independently_retrained
            and budget.get("passed")
            and training_binding.get("passed")
            and evaluation_binding.get("passed")
            and reload_verification_completed
            and control_tail_complete
            and control_tail_bound
        )
        matched_control = {
            "supplied": True,
            "independently_retrained_without_writer": independently_retrained,
            "budget_match": budget,
            "training_binding_match": training_binding,
            "evaluation_binding_match": evaluation_binding,
            "reload_verification_completed": reload_verification_completed,
            "comparison_is_valid": comparison_is_valid,
            "evaluation": control_comparison,
            "acceptance_envelope": control_envelope,
            "retain_tail_audit": dict(mlp_only_retain_tail or {}),
            "retain_tail_complete": control_tail_complete,
            "retain_tail_bound_to_official_split": control_tail_bound,
            "retain_tail_passed": control_tail_passed,
            "locked_training_acceptance": bool(
                mlp_only_method.get("acceptance", {}).get("passed")
            ),
            "fresh_reload_acceptance": bool(mlp_only_reload.get("passed")),
            "meets_same_forgetting_and_locality_envelope": control_meets_same_envelope,
            "embedding_key_necessity_falsified": bool(
                comparison_is_valid and control_meets_same_envelope
            ),
            "registered_control_supports_keyed_advantage": bool(
                comparison_is_valid and not control_meets_same_envelope
            ),
            "architecture_level_necessity_proven": False,
            "interpretation": (
                "A matching no-writer result falsifies embedding-key necessity. "
                "A weaker no-writer result supports an advantage only under this "
                "registered optimization budget; it is not a universal "
                "architecture-level impossibility proof."
            ),
        }

    evidence_checks = {
        "primary_protocol_is_v3": method.get("protocol") == EXPECTED_PROTOCOL,
        "matched_control_protocol_is_v3": bool(
            mlp_only_method is not None
            and mlp_only_method.get("protocol") == EXPECTED_PROTOCOL
        ),
        "matched_mlp_only_control_supplied": bool(matched_control["supplied"]),
        "matched_mlp_only_independent": bool(
            matched_control.get("independently_retrained_without_writer")
        ),
        "matched_optimization_budget": bool(
            matched_control.get("budget_match", {}).get("passed")
        ),
        "matched_training_artifacts": bool(
            matched_control.get("training_binding_match", {}).get("passed")
        ),
        "matched_official_evaluation_split": bool(evaluation_binding.get("passed")),
        "matched_control_reload_verification_completed": bool(
            matched_control.get("reload_verification_completed")
        ),
        "matched_control_retain_tail_complete": bool(
            matched_control.get("retain_tail_complete")
        ),
        "matched_control_retain_tail_bound": bool(
            matched_control.get("retain_tail_bound_to_official_split")
        ),
        "registered_control_supports_keyed_advantage": bool(
            matched_control.get("registered_control_supports_keyed_advantage")
        ),
        "writer_portability_complete": _writer_portability_complete(writer_portability),
        "writer_portability_bound": _post_freeze_audit_binding(
            writer_portability, edited, writer_mode="embedding_keyed"
        ),
        "writer_portability_passed": bool(
            _writer_portability_complete(writer_portability)
            and _post_freeze_audit_binding(
                writer_portability, edited, writer_mode="embedding_keyed"
            )
            and _audit_passed(writer_portability)
        ),
        "retain_tail_complete": _retain_tail_complete(retain_tail),
        "retain_tail_bound": _post_freeze_audit_binding(
            retain_tail,
            edited,
            writer_mode="embedding_keyed",
            require_retain_coverage=True,
        ),
        "retain_tail_passed": bool(
            _retain_tail_complete(retain_tail)
            and _post_freeze_audit_binding(
                retain_tail,
                edited,
                writer_mode="embedding_keyed",
                require_retain_coverage=True,
            )
            and _audit_passed(retain_tail)
        ),
        "latent_recovery_endpoint_complete": _latent_endpoint_complete(latent_recovery),
        "latent_recovery_endpoint_bound": _post_freeze_audit_binding(
            latent_recovery, edited, writer_mode="embedding_keyed"
        ),
        "relearning_endpoint_complete": _relearning_endpoint_complete(relearning),
        "relearning_endpoint_bound": _post_freeze_audit_binding(
            relearning, edited, writer_mode="embedding_keyed"
        ),
    }
    if require_complete_mechanism_evidence:
        checks.update(evidence_checks)

    primary_passed = all(checks.values())
    knowledge_recovered = bool(
        (latent_recovery or {}).get("fact_recoverable")
        or (relearning or {}).get("fact_recoverable")
    )
    if not primary_passed:
        status = "FAIL"
        supported_claim = "none; acceptance or required evidence is incomplete"
    else:
        status = "PASS"
        supported_claim = (
            "context-conditional factual suppression with measured locality on "
            "the evaluated distributions"
        )
    return {
        "schema_version": 3,
        "status": status,
        "supported_claim": supported_claim,
        "knowledge_removal_claim_allowed": False,
        "knowledge_recovered_by_diagnostic": knowledge_recovered,
        "target": {
            "Eff": 0.0,
            "Gen": 0.0,
            "max_abs_Spe_delta": max_abs_spe_delta,
            "max_abs_retain_Eff_Gen_delta": max_abs_retain_eff_gen_delta,
            "max_abs_PPL_percent_delta": max_ppl_percent_delta,
        },
        "base": base_summary,
        "edited": {
            "forget": proposed["forget"],
            "retain": proposed["retain"],
            "PPL": proposed["PPL"],
        },
        "delta": proposed["delta"],
        "checks": checks,
        "paper_level_evidence_checks": evidence_checks,
        "matched_mlp_only_control": matched_control,
        "writer_portability_audit": dict(writer_portability or {}),
        "retain_tail_audit": dict(retain_tail or {}),
        "latent_recovery_audit": dict(latent_recovery or {}),
        "relearning_audit": dict(relearning or {}),
        "architecture": architecture,
        "causal_component_ablation": causal,
        "post_reload_acceptance": dict(reload),
        "claim_boundary": (
            "The fitted-checkpoint 2x2 intervention is dependence evidence only. "
            "Even a passing report supports conditional suppression, not deletion."
        ),
    }


def write_report(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# MCF embedding-keyed sparse-neuron conditional suppression",
        "",
        f"Overall: **{report['status']}**",
        "",
        f"Supported claim: {report['supported_claim']}",
        "",
        "| Split | Metric | Base | Edited | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("forget", "retain"):
        for metric in METRICS:
            lines.append(
                f"| {split} | {metric} | {report['base'][split][metric]:.4f} | "
                f"{report['edited'][split][metric]:.4f} | "
                f"{report['delta'][split][metric]:+.4f} |"
            )
    lines.append(
        f"| utility | PPL | {report['base']['PPL']:.6f} | "
        f"{report['edited']['PPL']:.6f} | {report['delta']['PPL']:+.6f} "
        f"({report['delta']['PPL_percent']:+.3f}%) |"
    )
    lines.extend(["", "## Preregistered acceptance", ""])
    for key, passed in report["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{key}`")
    control = report.get("matched_mlp_only_control", {})
    lines.extend(["", "## Matched no-writer control", ""])
    if control.get("supplied"):
        lines.append(
            "- Meets same forgetting/locality envelope: "
            f"**{control['meets_same_forgetting_and_locality_envelope']}**"
        )
        lines.append(
            "- Identical 100k retain-tail criterion passed: "
            f"**{control['retain_tail_passed']}**"
        )
        lines.append(
            "- Registered control supports a budget-qualified keyed advantage: "
            f"**{control['registered_control_supports_keyed_advantage']}**"
        )
        lines.append(
            "- Architecture-level necessity proven: "
            f"**{control['architecture_level_necessity_proven']}**"
        )
    else:
        lines.append(
            "- Missing; no architecture-level necessity conclusion is allowed."
        )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(
        _load(args.base),
        _load(args.edited),
        _load(args.method_summary),
        _load(args.post_reload_acceptance),
        max_abs_spe_delta=float(args.max_abs_spe_delta),
        max_abs_retain_eff_gen_delta=float(args.max_abs_retain_eff_gen_delta),
        max_ppl_percent_delta=float(args.max_ppl_percent_delta),
        mlp_only=_load_optional(args.mlp_only),
        mlp_only_method=_load_optional(args.mlp_only_method_summary),
        mlp_only_reload=_load_optional(args.mlp_only_post_reload_acceptance),
        mlp_only_retain_tail=_load_optional(args.mlp_only_retain_tail_audit),
        writer_portability=_load_optional(args.writer_portability_audit),
        retain_tail=_load_optional(args.retain_tail_audit),
        latent_recovery=_load_optional(args.latent_recovery_audit),
        relearning=_load_optional(args.relearning_audit),
        require_complete_mechanism_evidence=not bool(
            args.allow_incomplete_mechanism_evidence
        ),
    )
    output_dir = Path(args.output_dir).resolve()
    write_report(report, output_dir)
    print(json.dumps(report, indent=2))
    print(f"comparison: {output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
