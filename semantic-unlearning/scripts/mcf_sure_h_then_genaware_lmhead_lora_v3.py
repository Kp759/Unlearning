#!/usr/bin/env python3
"""Gen-aware MCF Stage 2 v3 requiring semantically validated surrogates.

Optimization is unchanged from mcf_sure_h_then_genaware_lmhead_lora.py:
Stage-1 remains frozen and only residual sparse LM-head LoRA is trained.

This entrypoint strengthens only the data contract.  It first applies the v2
baseline-aware answer-occurrence validation, then refuses to train unless the
artifact was produced by the v3 semantic builder and every surrogate passed the
answer-blind dual-pass semantic validator.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import mcf_surrogate_semantic_validator as semantic
import mcf_sure_h_then_genaware_lmhead_lora as base
import mcf_sure_h_then_genaware_lmhead_lora_v2 as v2


SEMANTIC_BUILDER_PROTOCOL = "mcf_locked_direct_only_semantic_surrogates_v3"
_V2_LOAD = v2.load_surrogate_artifact


def load_surrogate_artifact(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    forget_num: int,
) -> Tuple[Dict[str, Any], List[List[str]]]:
    data, prompts = _V2_LOAD(
        path, records, seed=int(seed), forget_num=int(forget_num)
    )

    if data.get("builder_protocol") != SEMANTIC_BUILDER_PROTOCOL:
        raise RuntimeError(
            "Gen-aware v3 requires the semantically validated v3 surrogate builder"
        )
    sem = data.get("semantic_validation")
    if not isinstance(sem, Mapping) or not bool(sem.get("enabled", False)):
        raise RuntimeError("Surrogate artifact lacks enabled semantic validation")
    if sem.get("protocol") != semantic.VALIDATOR_PROTOCOL:
        raise RuntimeError("Unexpected surrogate semantic-validator protocol")
    if not bool(sem.get("dual_pass_consensus", False)):
        raise RuntimeError("Semantic validation must use dual-pass consensus")
    if not bool(sem.get("required_for_every_surrogate", False)):
        raise RuntimeError("Semantic validation was not required for every surrogate")

    forbidden_true = (
        "validator_received_target_true",
        "validator_received_target_new",
        "validator_received_official_paraphrases",
    )
    for key in forbidden_true:
        if bool(sem.get(key, True)):
            raise RuntimeError(f"Semantic validator data-access violation: {key}")

    generator = data.get("generator", {})
    if bool(generator.get("deterministic_wrapper_fallback_used", True)):
        raise RuntimeError("Gen-aware v3 forbids deterministic wrapper fallbacks")
    if bool(generator.get("generator_received_target_true", True)):
        raise RuntimeError("Generator received target_true")
    if bool(generator.get("generator_received_target_new", True)):
        raise RuntimeError("Generator received target_new")

    expected_per_record = int(data.get("surrogates_per_record", -1))
    if expected_per_record <= 0:
        raise RuntimeError("Invalid surrogate count in semantic artifact")
    for pos, row in enumerate(prompts):
        if len(row) != expected_per_record:
            raise RuntimeError(
                f"Record {pos} has {len(row)} surrogates; expected {expected_per_record}"
            )

    return data, prompts


# Patch only the artifact boundary. Training objective, active-set policy, LoRA,
# locality guards, Wikipedia guard, and scale selection remain the original
# Gen-aware implementation.
base.load_surrogate_artifact = load_surrogate_artifact


if __name__ == "__main__":
    base.main()
