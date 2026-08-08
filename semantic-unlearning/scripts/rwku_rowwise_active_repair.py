#!/usr/bin/env python3
"""Protected row-wise active LM-head repair for RWKU generated facts.

This module is intentionally independent from :mod:`rwku_repair`.  The older
module implements the published sparse/global-scale path and must remain
bitwise compatible.  Here every editable output row receives an independent
scale and every proposed change is checked against the complete, disjoint
protection bank.
"""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Sequence,
    Tuple,
)

import torch


ACTIVE_SOURCE = "target_generated_entity_fact_views"
ROW_SCALE_CANDIDATES = (
    1.0,
    0.875,
    0.75,
    0.625,
    0.5,
    0.375,
    0.25,
    0.1875,
    0.125,
    0.09375,
    0.0625,
    0.046875,
    0.03125,
    0.015625,
    0.0078125,
    0.0,
)
FORBIDDEN_ACTIVE_MARKERS = (
    "forget_level1.json",
    "forget_level2.json",
    "forget_level3.json",
    "official_locked_eval",
    "official_evaluation",
    "seen_fact_unseen_prompt_eval",
    "unseen_fact_eval",
)
SPECIAL_TOKEN_NAMES = (
    "eos_token_id",
    "bos_token_id",
    "pad_token_id",
    "unk_token_id",
)


@dataclass(frozen=True)
class RowEligibility:
    token_id: int
    decoded_token_piece: str
    eligible: bool
    eligibility_class: str
    reasons: Tuple[str, ...]
    retain_document_frequency: float
    protected_overlap: bool


def tokenizer_special_ids(tokenizer: Any) -> Tuple[int, ...]:
    values = {
        int(value)
        for name in SPECIAL_TOKEN_NAMES
        for value in [getattr(tokenizer, name, None)]
        if isinstance(value, int) and value >= 0
    }
    values.update(
        int(value)
        for value in getattr(tokenizer, "all_special_ids", [])
        if isinstance(value, int) and value >= 0
    )
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(converter):
        eot = converter("<|eot_id|>")
        if (
            isinstance(eot, int)
            and eot >= 0
            and eot != getattr(tokenizer, "unk_token_id", None)
        ):
            values.add(int(eot))
    return tuple(sorted(values))


def decode_token_piece(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except TypeError:
        return str(tokenizer.decode([int(token_id)]))


def _only_punctuation_or_symbols(value: str) -> bool:
    return bool(value) and all(
        unicodedata.category(character).startswith(("P", "S")) for character in value
    )


def classify_output_row(
    token_id: int,
    tokenizer: Any,
    *,
    protected_row_ids: Iterable[int] = (),
    retain_document_frequency: float = 0.0,
    maximum_document_frequency: float = 0.01,
) -> RowEligibility:
    """Apply the shared training/repair content-bearing row policy."""

    token_id = int(token_id)
    piece = decode_token_piece(tokenizer, token_id)
    stripped = piece.strip()
    protected = token_id in {int(value) for value in protected_row_ids}
    reasons: List[str] = []
    if token_id in tokenizer_special_ids(tokenizer):
        reasons.append("tokenizer_special_row")
    if not stripped:
        reasons.append("whitespace_only")
    elif _only_punctuation_or_symbols(stripped):
        reasons.append("punctuation_only")
    elif all(character.isnumeric() for character in stripped):
        reasons.append("numeric_only")
    elif len(stripped) == 1 and stripped.isalnum():
        reasons.append("single_character_alphanumeric")
    if protected:
        reasons.append("protected_overlap")
    if float(retain_document_frequency) > float(maximum_document_frequency):
        reasons.append("high_retain_document_frequency")
    return RowEligibility(
        token_id=token_id,
        decoded_token_piece=piece,
        eligible=not reasons,
        eligibility_class="safe_content_row" if not reasons else "ineligible",
        reasons=tuple(reasons),
        retain_document_frequency=float(retain_document_frequency),
        protected_overlap=protected,
    )


def validate_active_points(
    points: Sequence[Mapping[str, Any]],
    *,
    training_bundle_path: Path,
    training_bundle_sha256: str,
) -> List[Dict[str, Any]]:
    """Reject evaluation-derived active points and verify full provenance."""

    required = {
        "fact_id",
        "view_id",
        "prompt_style",
        "answer_alias",
        "source_record_sha256",
        "training_bundle_sha256",
        "active_source",
    }
    result: List[Dict[str, Any]] = []
    expected_path = str(Path(training_bundle_path).resolve())
    for index, point in enumerate(points):
        missing = required - set(point)
        if missing:
            raise ValueError(
                f"Active point {index} lacks provenance: {sorted(missing)}"
            )
        if point.get("active_source") != ACTIVE_SOURCE:
            raise ValueError(
                "Target-only active_source must be "
                f"{ACTIVE_SOURCE}, got {point.get('active_source')!r}"
            )
        if point.get("training_bundle_sha256") != training_bundle_sha256:
            raise ValueError("Active point training-bundle hash mismatch")
        source_path = str(point.get("source_path", expected_path))
        lowered = source_path.casefold()
        if any(marker in lowered for marker in FORBIDDEN_ACTIVE_MARKERS):
            raise ValueError(
                "Official/evaluation RWKU row cannot become an active point"
            )
        if point.get("source_artifact_role") in {
            "official_locked_eval",
            "seen_fact_unseen_prompt_eval",
            "unseen_fact_eval",
        }:
            raise ValueError("Evaluation artifact cannot become an active point")
        if str(point.get("level", "generated")) in {"1", "2", "3"}:
            raise ValueError("Official Level-1/2/3 row cannot become an active point")
        if source_path and str(Path(source_path).resolve()) != expected_path:
            raise ValueError(
                "Active point is not bound to the generated training bundle"
            )
        result.append(dict(point))
    return result


def protection_gates_pass(metrics: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    """Fixed no-gradient protection gates shared by candidate and repair selection."""

    checks = {
        "full_retain_probability_ratio": 0.995
        <= float(metrics.get("full_retain_probability_ratio", float("nan")))
        <= 1.005,
        "geometric_retain_probability_ratio": 0.98
        <= float(metrics.get("geometric_retain_probability_ratio", float("nan")))
        <= 1.02,
        "mean_retain_kl": float(metrics.get("mean_retain_kl", float("inf"))) <= 0.01,
        "p95_retain_kl": float(metrics.get("p95_retain_kl", float("inf"))) <= 0.05,
        "retain_top1_agreement": float(metrics.get("retain_top1_agreement", -1.0))
        >= 0.99,
        "protected_answer_probability_ratio": float(
            metrics.get("protected_answer_probability_ratio", -1.0)
        )
        >= 0.999,
        "protected_selected_row_logit_drift": float(
            metrics.get("protected_selected_row_logit_drift", float("inf"))
        )
        <= 0.05,
        "protected_top1_changes": int(metrics.get("protected_top1_changes", 1)) == 0,
        "proxy_ppl": float(metrics.get("proxy_ppl", float("inf")))
        <= float(metrics.get("base_proxy_ppl", 0.0)) * 1.02,
        "nonselected_rows_equal_base": metrics.get("nonselected_rows_equal_base")
        is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return not failed, failed


def generated_forget_gates_pass(metrics: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    checks = {
        "direct_generation_recovery": float(
            metrics.get("direct_generation_recovery", math.inf)
        )
        == 0.0,
        "cloze_generation_recovery": float(
            metrics.get("cloze_generation_recovery", math.inf)
        )
        == 0.0,
        "paraphrase_generation_recovery": float(
            metrics.get("paraphrase_generation_recovery", math.inf)
        )
        == 0.0,
        "generated_geometric_answer_probability": float(
            metrics.get("generated_geometric_answer_probability", math.inf)
        )
        <= 0.01,
        "active_violation_count": int(metrics.get("active_violation_count", 1)) == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return not failed, failed


def apply_rowwise_delta(
    output_weight: torch.Tensor,
    immutable_weight: torch.Tensor,
    unscaled_deltas: Mapping[int, torch.Tensor],
    scales: Mapping[int, float],
) -> None:
    """Materialize one non-cumulative row-wise repair candidate."""

    with torch.no_grad():
        output_weight.copy_(
            immutable_weight.to(output_weight.device, output_weight.dtype)
        )
        for token_id, delta in unscaled_deltas.items():
            scale = float(scales.get(int(token_id), 0.0))
            if scale == 0.0:
                continue
            output_weight[int(token_id)].add_(
                delta.to(output_weight.device, output_weight.dtype),
                alpha=scale,
            )


def _next_lower_scale(scale: float, candidates: Sequence[float]) -> float:
    try:
        index = list(candidates).index(float(scale))
    except ValueError:
        return 0.0
    return float(candidates[min(index + 1, len(candidates) - 1)])


def select_rowwise_scales(
    row_ids: Sequence[int],
    *,
    evaluate: Callable[[Mapping[int, float]], Mapping[str, Any]],
    row_contributions: Mapping[int, Mapping[str, float]] | None = None,
    candidate_scales: Sequence[float] = ROW_SCALE_CANDIDATES,
) -> Dict[str, Any]:
    """Greedily choose independent row scales under complete fixed gates.

    ``evaluate`` must run both the full no-gradient protection bank and all
    generated sequence-level checks for the *combined* edit represented by the
    supplied scale mapping.
    """

    if not candidate_scales or float(candidate_scales[-1]) != 0.0:
        raise ValueError("Row-scale candidates must include mandatory all-zero last")
    contributions = row_contributions or {}

    def rank_key(row_id: int) -> Tuple[float, int]:
        values = contributions.get(int(row_id), {})
        efficacy = float(values.get("generated_efficacy_contribution", 0.0))
        drift = max(float(values.get("protected_drift_contribution", 0.0)), 1e-12)
        return (-(efficacy / drift), int(row_id))

    ordered = sorted({int(value) for value in row_ids}, key=rank_key)
    selected: Dict[int, float] = {row_id: 0.0 for row_id in ordered}
    trials: List[Dict[str, Any]] = []
    zero_metrics = dict(evaluate(dict(selected)))
    zero_protection, zero_protection_failed = protection_gates_pass(zero_metrics)
    zero_forget, zero_forget_failed = generated_forget_gates_pass(zero_metrics)
    trials.append(
        {
            "operation": "mandatory_all_zero_candidate",
            "scales": dict(selected),
            "protection_passed": zero_protection,
            "generated_forget_passed": zero_forget,
            "failed_protection_gates": zero_protection_failed,
            "failed_generated_gates": zero_forget_failed,
        }
    )
    for row_id in ordered:
        accepted = 0.0
        for scale in candidate_scales:
            proposed = {**selected, row_id: float(scale)}
            metrics = dict(evaluate(proposed))
            passed, failed = protection_gates_pass(metrics)
            trials.append(
                {
                    "operation": "row_scale_trial",
                    "row_id": row_id,
                    "scale": float(scale),
                    "protection_passed": passed,
                    "failed_protection_gates": failed,
                }
            )
            if passed:
                accepted = float(scale)
                break
        selected[row_id] = accepted

    combined_metrics = dict(evaluate(selected))
    protection_ok, protection_failed = protection_gates_pass(combined_metrics)
    forget_ok, forget_failed = generated_forget_gates_pass(combined_metrics)
    scale_back: List[Dict[str, Any]] = []
    # Interactions can violate a gate even if each greedy insertion passed.
    # Reduce the least efficient non-zero row one notch per pass.
    while (not protection_ok or not forget_ok) and any(selected.values()):
        nonzero = [row_id for row_id in reversed(ordered) if selected[row_id] > 0.0]
        if not nonzero:
            break
        row_id = nonzero[0]
        selected[row_id] = _next_lower_scale(selected[row_id], candidate_scales)
        combined_metrics = dict(evaluate(selected))
        protection_ok, protection_failed = protection_gates_pass(combined_metrics)
        forget_ok, forget_failed = generated_forget_gates_pass(combined_metrics)
        scale_back.append(
            {
                "row_id": row_id,
                "new_scale": selected[row_id],
                "protection_passed": protection_ok,
                "generated_forget_passed": forget_ok,
            }
        )

    selected_success = protection_ok and forget_ok
    if not selected_success:
        selected = {row_id: 0.0 for row_id in ordered}
        combined_metrics = zero_metrics
        protection_ok, protection_failed = protection_gates_pass(combined_metrics)
        forget_ok, forget_failed = generated_forget_gates_pass(combined_metrics)
        selected_success = protection_ok and forget_ok

    return {
        "method": "protected_row_wise_active_lm_head_repair",
        "active_source": ACTIVE_SOURCE,
        "candidate_scales": [float(value) for value in candidate_scales],
        "row_rank_order": ordered,
        "selected_scale_by_row": {str(key): value for key, value in selected.items()},
        "selected_success": bool(selected_success),
        "repair_applied": bool(selected_success and any(selected.values())),
        "combined_gate_results": {
            "protection_passed": protection_ok,
            "generated_forget_passed": forget_ok,
            "failed_protection_gates": protection_failed,
            "failed_generated_gates": forget_failed,
            "metrics": combined_metrics,
        },
        "row_contributions": {
            str(key): dict(value) for key, value in contributions.items()
        },
        "trials": trials,
        "combined_scale_back_pass": scale_back,
    }


def selected_delta_norm(
    unscaled_deltas: Mapping[int, torch.Tensor],
    scales: Mapping[int, float],
) -> float:
    total = 0.0
    for token_id, delta in unscaled_deltas.items():
        total += (
            float(delta.detach().float().pow(2).sum())
            * float(scales.get(int(token_id), 0.0)) ** 2
        )
    return math.sqrt(total)


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


__all__ = [
    "ACTIVE_SOURCE",
    "FORBIDDEN_ACTIVE_MARKERS",
    "ROW_SCALE_CANDIDATES",
    "RowEligibility",
    "apply_rowwise_delta",
    "classify_output_row",
    "decode_token_piece",
    "generated_forget_gates_pass",
    "protection_gates_pass",
    "select_rowwise_scales",
    "selected_delta_norm",
    "tokenizer_special_ids",
    "validate_active_points",
    "write_report",
]
