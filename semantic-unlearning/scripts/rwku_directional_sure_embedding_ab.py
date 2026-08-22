#!/usr/bin/env python3
"""Embedding-policy adapter for pure two-stage Directional SURE variants A/B.

Variant A freezes input embeddings completely while retaining all-non-special
sensitive LM-head rows and the existing directional GA->B_S / GD->B_P rule.

Variant B trains only the existing locked content-sensitive answer-token input
rows, while the LM head still trains every non-special sensitive answer row.
Embedding gradients are ordinary weighted GA+GD; no hidden-space basis is
applied directly to embedding gradients.

The underlying two-stage learner, utility budgets, data boundary, ranks, step
counts, and Level-2 residual-repair logic are otherwise unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import rwku_directional_sure_two_stage as two_stage
import rwku_directional_sure_v2 as v2
import rwku_directional_sure_v21 as v21
import rwku_setting5e_utility_controlled as sparse_rows
import sure_canonical_core as core

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
CANONICAL_TWO_STAGE_CONFIGURATION = (
    PROJECT_ROOT / "config" / "rwku" / "directional_sure_two_stage_seed0.json"
)

VARIANT_A = "A"
VARIANT_B = "B"
VALID_VARIANTS = (VARIANT_A, VARIANT_B)

_ORIGINAL_ALL_NON_SPECIAL_ROWS = v21.all_non_special_sensitive_rows
_ORIGINAL_SPARSE_CLASS = sparse_rows.SparseFP32RowDeltas
_ACTIVE_VARIANT: str | None = None
_CAPTURED_OUTPUT_ROWS: Tuple[int, ...] | None = None
_CAPTURED_INPUT_ROWS: Tuple[int, ...] | None = None
_CAPTURED_INPUT_AUDIT: Dict[str, Any] | None = None


def resolve_input_rows(
    variant: str,
    output_rows: Sequence[int],
    content_rows: Sequence[int],
) -> List[int]:
    """Return the controlled input-row policy for an A/B ablation."""
    variant = str(variant).upper()
    output = sorted({int(x) for x in output_rows})
    content = sorted({int(x) for x in content_rows})
    if variant == VARIANT_A:
        return []
    if variant == VARIANT_B:
        unexpected = sorted(set(content) - set(output))
        if unexpected:
            raise RuntimeError(
                "Variant-B content-safe input rows are outside all-sensitive output rows: "
                f"{unexpected[:10]}"
            )
        return content
    raise ValueError(f"Unsupported Directional SURE embedding variant: {variant!r}")


def _capturing_sensitive_row_selector(
    tokenizer: Any,
    cases: Sequence[core.SensitivePredictionCase],
    tids: Any,
    source_cfg: Mapping[str, Any],
    prompt_count: int,
):
    global _CAPTURED_OUTPUT_ROWS, _CAPTURED_INPUT_ROWS, _CAPTURED_INPUT_AUDIT
    if _ACTIVE_VARIANT not in VALID_VARIANTS:
        raise RuntimeError("A/B embedding policy was not installed")

    output_rows, output_audit = _ORIGINAL_ALL_NON_SPECIAL_ROWS(
        tokenizer, cases, tids, source_cfg, prompt_count
    )
    content_rows, content_audit = v2._content_sensitive_rows(
        tokenizer, cases, tids, source_cfg, prompt_count
    )
    input_rows = resolve_input_rows(_ACTIVE_VARIANT, output_rows, content_rows)

    _CAPTURED_OUTPUT_ROWS = tuple(int(x) for x in output_rows)
    _CAPTURED_INPUT_ROWS = tuple(int(x) for x in input_rows)
    _CAPTURED_INPUT_AUDIT = {
        "variant": _ACTIVE_VARIANT,
        "policy": (
            "input_embeddings_frozen"
            if _ACTIVE_VARIANT == VARIANT_A
            else "locked_content_sensitive_answer_rows_only"
        ),
        "selected_input_row_count": len(input_rows),
        "selected_input_row_ids": list(input_rows),
        "all_non_special_output_row_count": len(output_rows),
        "content_filter_candidate_row_count": len(content_rows),
        "content_filter_audit": content_audit,
        "embedding_gradient_directional_projection": False,
        "lm_head_all_non_special_sensitive_rows": True,
    }
    augmented = dict(output_audit)
    augmented["embedding_ablation"] = dict(_CAPTURED_INPUT_AUDIT)
    return output_rows, augmented


class PolicySparseFP32RowDeltas(_ORIGINAL_SPARSE_CLASS):
    """Use the A/B input policy while preserving all-sensitive output rows."""

    def __init__(
        self,
        model: Any,
        selected_input_rows: Sequence[int],
        selected_output_rows: Sequence[int],
    ) -> None:
        del selected_input_rows
        if _ACTIVE_VARIANT not in VALID_VARIANTS:
            raise RuntimeError("A/B embedding policy was not installed")
        if _CAPTURED_OUTPUT_ROWS is None or _CAPTURED_INPUT_ROWS is None:
            raise RuntimeError("Sensitive rows were not captured before sparse setup")
        actual_output = tuple(sorted({int(x) for x in selected_output_rows}))
        if actual_output != tuple(sorted(_CAPTURED_OUTPUT_ROWS)):
            raise RuntimeError("LM-head row set changed after A/B policy capture")
        super().__init__(
            model,
            selected_input_rows=list(_CAPTURED_INPUT_ROWS),
            selected_output_rows=list(_CAPTURED_OUTPUT_ROWS),
        )


def _assert_common_configuration_unchanged(
    cfg: Mapping[str, Any], canonical: Mapping[str, Any]
) -> None:
    for section in ("optimization", "stage2", "acceptance", "data_boundary"):
        if cfg.get(section) != canonical.get(section):
            raise ValueError(
                f"A/B ablation changed canonical two-stage section {section}"
            )
    fixed_identity = {
        "development_only": True,
        "posthoc_development_target": True,
        "official_rwku_metrics_observed_before_method_design": True,
        "seed": 0,
        "target_entity": "Stephen King",
        "target_entity_id": "rwku:1_Stephen_King",
        "neutral_target": "Unknown",
        "level3_representation_repair_enabled": False,
    }
    for key, expected in fixed_identity.items():
        if cfg.get(key) != expected:
            raise ValueError(f"A/B ablation changed locked identity field {key}")


def load_variant_configuration(
    path: Path,
    *,
    variant: str,
    expected_schema: str,
    expected_experiment_id: str,
) -> Dict[str, Any]:
    cfg = two_stage.read_json(path)
    canonical = two_stage.read_json(CANONICAL_TWO_STAGE_CONFIGURATION)
    if cfg.get("schema_version") != expected_schema:
        raise ValueError("Unsupported Directional SURE A/B configuration schema")
    if cfg.get("configuration_id") != expected_experiment_id:
        raise ValueError("Directional SURE A/B configuration ID mismatch")
    if cfg.get("embedding_ablation_variant") != variant:
        raise ValueError("Directional SURE A/B variant identity mismatch")
    _assert_common_configuration_unchanged(cfg, canonical)

    components = cfg.get("trainable_components", {})
    expected_input = variant == VARIANT_B
    expected_components = {
        "sensitive_input_embedding_rows": expected_input,
        "sensitive_untied_lm_head_rows": True,
        "non_sensitive_input_embedding_rows": False,
        "non_sensitive_lm_head_rows": False,
        "transformer_parameters": False,
        "mlp_parameters": False,
        "attention_parameters": False,
        "lora_parameters": False,
    }
    for key, expected in expected_components.items():
        if components.get(key) != expected:
            raise ValueError(f"A/B trainable component changed {key}")

    expected_policy = (
        "frozen"
        if variant == VARIANT_A
        else "locked_content_sensitive_answer_rows_only"
    )
    if cfg.get("input_embedding_policy") != expected_policy:
        raise ValueError("A/B input embedding policy changed")
    if cfg.get("lm_head_row_policy") != "all_non_special_sensitive_answer_rows":
        raise ValueError("A/B LM-head row policy changed")
    if cfg.get("embedding_gradient_policy") != "ordinary_weighted_GA_plus_GD_no_hidden_basis_projection":
        raise ValueError("A/B embedding gradient policy changed")
    if cfg.get("lm_head_gradient_policy") != "GA_to_sensitive_exclusive_basis_and_GD_to_protected_basis":
        raise ValueError("A/B LM-head gradient policy changed")
    return dict(cfg)


def install_variant(
    *,
    variant: str,
    schema: str,
    experiment_id: str,
    configuration_path: Path,
    learner_dir: str,
) -> None:
    """Install one controlled embedding policy into the two-stage learner."""
    global _ACTIVE_VARIANT, _CAPTURED_OUTPUT_ROWS, _CAPTURED_INPUT_ROWS
    global _CAPTURED_INPUT_AUDIT
    variant = str(variant).upper()
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Unsupported A/B variant: {variant}")
    _ACTIVE_VARIANT = variant
    _CAPTURED_OUTPUT_ROWS = None
    _CAPTURED_INPUT_ROWS = None
    _CAPTURED_INPUT_AUDIT = None

    two_stage.SCHEMA = str(schema)
    two_stage.EXPERIMENT_ID = str(experiment_id)
    two_stage.DEFAULT_CONFIGURATION = Path(configuration_path)
    two_stage.LEARNER_DIR = str(learner_dir)
    two_stage.load_configuration = lambda path: load_variant_configuration(
        Path(path),
        variant=variant,
        expected_schema=str(schema),
        expected_experiment_id=str(experiment_id),
    )

    # The base learner asks v2.1 for the LM-head row set. Capture both the full
    # output set and the controlled input subset at exactly that point.
    v21.all_non_special_sensitive_rows = _capturing_sensitive_row_selector
    # The base learner still passes the full sensitive set to both arguments;
    # this adapter substitutes the controlled input rows while preserving the
    # all-sensitive LM-head rows.
    sparse_rows.SparseFP32RowDeltas = PolicySparseFP32RowDeltas
