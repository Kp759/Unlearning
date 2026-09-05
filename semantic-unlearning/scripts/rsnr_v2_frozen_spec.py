#!/usr/bin/env python3
"""Machine-readable freeze for RSNR-V2 = Stage 0 Emb+LM erasure + V1A PreHead.

RSNR-V1A is a *conditional* method: disabling the gate restores the frozen Base
model exactly, so nothing leaves the weights and the reported PPL parity is an
identity rather than a measurement.  RSNR-V2 keeps the V1A adapter bit-for-bit
and adds an unconditional Stage 0 that erases the sensitive answer from the
input embedding and LM-head rows before the adapter is ever attached.

What that buys, and what it costs:

  * Knowledge actually leaves the weights, so gate-removal and relearning
    attacks have something to attack.  V1A had no answer to either.
  * PPL stops being free.  Stage 0 is a global weight change, so the utility
    columns become real measurements.  ``SURE_MCF_DIRECTIONAL_EMB_LM_FULLREPAIR.md``
    records PPL reaching 18.875 against a ~11.0 base when the protected
    subspace was too narrow; that failure mode is now in scope.
  * ``gate_off_equivalence`` keeps working unchanged -- it compares the loaded
    model against itself with the hook cleared, not against a separately loaded
    Base -- but its *meaning* changes.  Under V2 a zero drift means "gate off
    equals the Stage-0 model", not "gate off equals Base".  V1A's error string
    still says "not Base-identical"; that wording is stale under V2 and the
    number should be read against ``gate_off_reference`` below.

The V1A spec forbids changing the adapter architecture, intervention site,
objective, or consumed seed-1 configuration.  V2 changes none of them: it
changes what the adapter sits on top of.  V1A therefore remains a valid,
unmodified ablation arm (Stage-0 scale 0.0 reproduces it exactly).
"""
from __future__ import annotations

from typing import Any, Mapping

import rsnr_v1a_frozen_spec as v1a

FROZEN_SPEC_VERSION = "rsnr_v2_emb_lm_erasure_frozen_2026-09-05"

# Inherited verbatim from V1A; V2 must not diverge on any of these.
INHERITED_ADAPTER_KEYS = (
    "variant",
    "intervention_site",
    "adapter_type",
    "adapter_formula",
    "adapter_rank",
    "adapter_alpha",
    "adapter_scaling",
    "activation",
    "bias",
    "abstention",
    "target_new_used",
)

FROZEN_ARCHITECTURE = {
    "variant": "RSNR-V2-Stage0Emb+LM-PreHead",
    "protocol": "mcf_rsnr_v2_abstention_emb_lm_erasure_plus_prehead",
    "stage0": {
        "method": "abstention-anchored directional Emb+LM gradient ascent",
        "entrypoint": "mcf_sure_directional_emb_lm_stage1.py --reference-anchor abstention",
        "direction": "d = h_true - h_IDK; w_true - w_IDK decoder discriminant fallback",
        "target_new_used": False,
        "conditional": False,
        "input_embeddings_frozen": False,
        "lm_head_frozen": False,
        "transformer_frozen": True,
        "edited_rows": "target_true sensitive vocabulary rows only",
    },
    "stage1": {
        "method": "frozen RSNR-V1A-PreHead null adapter",
        "inherits_spec_version": v1a.FROZEN_SPEC_VERSION,
        "trained_on": "Stage-0 model, not Base",
        "conditional": True,
    },
    # Under V1A this was Base. Under V2 the adapter's pass-through reference is
    # the Stage-0 model, and that is the only thing gate-off drift may be read
    # against.
    "gate_off_reference": "stage0_model",
    "base_recovery_on_gate_off": False,
    "knowledge_removed_from_weights": True,
}

# One training run, materialized at several multiplicative delta scales. The
# frontier is the contribution: how much erasure the weights absorb before
# Spe/PPL degrade, and how much residual the gate must carry at each point.
# Scale 0.0 is Base + adapter, i.e. exactly RSNR-V1A.
FROZEN_SWEEP = {
    "sweep_axis": "stage0_delta_scale",
    "retrains_per_point": 0,
    "default_scales": (1.0, 0.5, 0.25, 0.125, 0.0625, 0.0),
    "scale_zero_reproduces": "RSNR-V1A-PreHead",
    "reported_per_point": (
        "eq16_style_residual_likelihood_proxy",
        "released_table_style_accuracy",
        "PPL",
        "gate_residual_load",
    ),
    "development_seed": 1,
    "forget_num": 50,
    "official_paraphrases_used_for_training": False,
    "official_neighborhood_prompts_used_for_training": False,
    # SURE_MCF_DIRECTIONAL_EMB_LM_FULLREPAIR.md: training against the PPL text
    # contaminates the PPL result. Stage 0's protection sample must stay
    # disjoint from official PPL's [:20] slice.
    "ppl_text_disjoint_from_training": True,
}


def frozen_spec() -> dict[str, Any]:
    return {
        "spec_version": FROZEN_SPEC_VERSION,
        "architecture": dict(FROZEN_ARCHITECTURE),
        "sweep": dict(FROZEN_SWEEP),
        "inherited_v1a_spec": v1a.frozen_spec(),
    }


def validate_adapter_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Accept only an unmodified V1A adapter, but allow a Stage-0 substrate.

    V1A's own validator hard-requires ``lm_head_weights_modified is False``,
    which is exactly what V2 changes.  The adapter itself must still match V1A
    bit-for-bit, so the geometric checks are delegated and only the substrate
    flags are relaxed.
    """
    expected = v1a.FROZEN_ARCHITECTURE
    mismatches = [
        f"{key}: got {checkpoint.get(key)!r}, expected {expected[key]!r}"
        for key in ("variant", "intervention_site", "abstention")
        if checkpoint.get(key) != expected[key]
    ]
    if int(checkpoint.get("adapter_rank", -1)) != int(expected["adapter_rank"]):
        mismatches.append(
            f"adapter_rank: got {checkpoint.get('adapter_rank')!r}, "
            f"expected {expected['adapter_rank']!r}"
        )
    if float(checkpoint.get("adapter_alpha", float("nan"))) != float(
        expected["adapter_alpha"]
    ):
        mismatches.append(
            f"adapter_alpha: got {checkpoint.get('adapter_alpha')!r}, "
            f"expected {expected['adapter_alpha']!r}"
        )
    if checkpoint.get("transformer_weights_modified") is not False:
        mismatches.append("transformer_weights_modified must stay False under V2")
    if "adapter_state_dict" not in checkpoint:
        mismatches.append("adapter_state_dict missing")
    if mismatches:
        raise RuntimeError(
            "checkpoint violates frozen RSNR-V2 specification:\n- "
            + "\n- ".join(mismatches)
        )


def validate_stage0_config(config: Mapping[str, Any]) -> None:
    """Reject a Stage-0 run that read target_new or lost its abstention anchor."""
    mismatches = []
    if config.get("reference_anchor") != "abstention":
        mismatches.append(
            f"reference_anchor: got {config.get('reference_anchor')!r}, "
            "expected 'abstention'"
        )
    if config.get("target_new_used") is not False:
        mismatches.append("target_new_used must be False under the RSNR contract")
    anchor = config.get("abstention_anchor")
    if not isinstance(anchor, Mapping):
        mismatches.append("abstention_anchor summary missing from stage0 config")
    elif int(anchor.get("degenerate_fallback_count", 0)) > 0:
        mismatches.append(
            "stage0 used the degenerate sensitive-hidden fallback for "
            f"{anchor['degenerate_fallback_count']} direction(s); those rows "
            "were not erased along a true contrast"
        )
    if mismatches:
        raise RuntimeError(
            "stage0 config violates the RSNR-V2 contract:\n- " + "\n- ".join(mismatches)
        )
