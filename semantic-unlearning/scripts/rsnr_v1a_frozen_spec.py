#!/usr/bin/env python3
"""Machine-readable freeze for the RSNR-V1A-PreHead method.

This file is the protocol boundary for all post-development experiments.  New
experiments may change routing/evaluation, but they must not change the null
adapter architecture, intervention site, objective, or consumed seed-1
configuration recorded here.
"""
from __future__ import annotations

from typing import Any, Mapping

FROZEN_SPEC_VERSION = "rsnr_v1a_prehead_frozen_2026-09-05"

FROZEN_ARCHITECTURE = {
    "variant": "RSNR-V1A-PreHead",
    "protocol": "mcf_rsnr_v1a_prehead_oracle_null_adapter",
    "intervention_site": "pre_lm_head_final_hidden_state",
    "adapter_type": "residual_bottleneck",
    "adapter_formula": "h' = h + gate * W_up(tanh(W_down h)) * (alpha/rank)",
    "adapter_rank": 16,
    "adapter_alpha": 16.0,
    "adapter_scaling": 1.0,
    "activation": "tanh",
    "bias": False,
    "base_model_frozen": True,
    "transformer_frozen": True,
    "final_norm_frozen": True,
    "lm_head_frozen": True,
    "input_embeddings_frozen": True,
    "abstention": "I don't know.",
    "target_new_used": False,
}

FROZEN_TRAINING = {
    "development_seed": 1,
    "forget_num": 50,
    "steps_max": 800,
    "case_batch_size": 4,
    "check_every": 25,
    "learning_rate": 2e-4,
    "weight_decay": 0.0,
    "abstention_weight": 1.0,
    "true_answer_unlikelihood_weight": 1.0,
    "anchor_weight": 1e-4,
    "grad_clip": 1.0,
    "minimum_abstain_vs_true_margin": 0.1,
    "minimum_true_logprob_drop": 2.0,
    "views_per_case": 5,
    "worst_of_views": True,
    "official_paraphrases_used_for_training": False,
    "official_neighborhood_prompts_used_for_training": False,
}


def frozen_spec() -> dict[str, Any]:
    return {
        "spec_version": FROZEN_SPEC_VERSION,
        "architecture": dict(FROZEN_ARCHITECTURE),
        "training": dict(FROZEN_TRAINING),
    }


def validate_adapter_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Reject a checkpoint that is not the frozen V1A-PreHead architecture."""
    expected = FROZEN_ARCHITECTURE
    checks = {
        "variant": checkpoint.get("variant"),
        "protocol": checkpoint.get("protocol"),
        "intervention_site": checkpoint.get("intervention_site"),
        "adapter_rank": int(checkpoint.get("adapter_rank", -1)),
        "adapter_alpha": float(checkpoint.get("adapter_alpha", float("nan"))),
        "abstention": checkpoint.get("abstention"),
        "transformer_weights_modified": checkpoint.get("transformer_weights_modified"),
        "lm_head_weights_modified": checkpoint.get("lm_head_weights_modified"),
    }
    required = {
        "variant": expected["variant"],
        "protocol": expected["protocol"],
        "intervention_site": expected["intervention_site"],
        "adapter_rank": expected["adapter_rank"],
        "adapter_alpha": expected["adapter_alpha"],
        "abstention": expected["abstention"],
        "transformer_weights_modified": False,
        "lm_head_weights_modified": False,
    }
    mismatches = [
        f"{key}: got {checks[key]!r}, expected {value!r}"
        for key, value in required.items()
        if checks[key] != value
    ]
    if "adapter_state_dict" not in checkpoint:
        mismatches.append("adapter_state_dict missing")
    if mismatches:
        raise RuntimeError(
            "checkpoint violates frozen RSNR-V1A-PreHead specification:\n- "
            + "\n- ".join(mismatches)
        )
