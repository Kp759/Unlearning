#!/usr/bin/env python3
"""Gen-aware MCF Stage 2 v4 requiring calibrated relation-slot surrogates.

Optimization is unchanged from mcf_sure_h_then_genaware_lmhead_lora.py. This
entrypoint strengthens only the surrogate artifact contract:
  * baseline-aware answer-occurrence guard;
  * v4 relation-slot semantic builder;
  * answer-blind relation-slot + adversarial dual-pass validation;
  * exactly K validated surrogates per locked fact;
  * no deterministic wrapper fallback.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import mcf_surrogate_semantic_validator_v2 as semantic_v2
import mcf_sure_h_then_genaware_lmhead_lora as base
import mcf_sure_h_then_genaware_lmhead_lora_v2 as v2


SEMANTIC_BUILDER_PROTOCOL = "mcf_locked_direct_only_relation_slot_surrogates_v4"
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
            "Gen-aware v4 requires the calibrated relation-slot v4 surrogate builder"
        )
    sem = data.get("semantic_validation")
    if not isinstance(sem, Mapping) or not bool(sem.get("enabled", False)):
        raise RuntimeError("Surrogate artifact lacks enabled semantic validation")
    if sem.get("protocol") != semantic_v2.VALIDATOR_PROTOCOL:
        raise RuntimeError("Unexpected calibrated semantic-validator protocol")
    if not bool(sem.get("dual_pass_consensus", False)):
        raise RuntimeError("Semantic validation must use dual-pass consensus")
    if not bool(sem.get("required_for_every_surrogate", False)):
        raise RuntimeError("Semantic validation was not required for every surrogate")
    if not bool(sem.get("completion_fragments_explicitly_allowed", False)):
        raise RuntimeError("v4 artifact lacks completion-fragment calibration")

    for key in (
        "validator_received_target_true",
        "validator_received_target_new",
        "validator_received_official_paraphrases",
    ):
        if bool(sem.get(key, True)):
            raise RuntimeError(f"Semantic validator data-access violation: {key}")

    generator = data.get("generator", {})
    if bool(generator.get("deterministic_wrapper_fallback_used", True)):
        raise RuntimeError("Gen-aware v4 forbids deterministic wrapper fallbacks")
    if bool(generator.get("generator_received_target_true", True)):
        raise RuntimeError("Generator received target_true")
    if bool(generator.get("generator_received_target_new", True)):
        raise RuntimeError("Generator received target_new")

    expected = int(data.get("surrogates_per_record", -1))
    if expected <= 0:
        raise RuntimeError("Invalid surrogate count in v4 artifact")
    for pos, row in enumerate(prompts):
        if len(row) != expected:
            raise RuntimeError(
                f"Record {pos} has {len(row)} surrogates; expected {expected}"
            )
    return data, prompts


base.load_surrogate_artifact = load_surrogate_artifact


if __name__ == "__main__":
    base.main()
