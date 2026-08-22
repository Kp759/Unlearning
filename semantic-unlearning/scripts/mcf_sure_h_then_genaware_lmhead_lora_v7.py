#!/usr/bin/env python3
"""Gen-aware MCF Stage 2 consuming high-precision adapter-v7 prompt sets.

The SURE optimizer remains the common implementation.  This entrypoint changes
only the dataset-adapter contract and verifies that the v7 artifact includes the
answer-blind high-precision rejection guard.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import mcf_dataset_adapter_high_precision as hp
import mcf_sure_h_then_genaware_lmhead_lora as base
import mcf_sure_h_then_genaware_lmhead_lora_v6 as v6


ARTIFACT_PROTOCOL = "mcf_direct_only_robust_prompt_adapter_v7"
ADAPTER_PROTOCOL = "mcf_relation_slot_answer_type_semantic_high_precision_adapter_v2"


def load_surrogate_artifact(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    forget_num: int,
) -> Tuple[Dict[str, Any], List[List[str]]]:
    old_artifact = v6.ARTIFACT_PROTOCOL
    old_adapter = v6.ADAPTER_PROTOCOL
    try:
        v6.ARTIFACT_PROTOCOL = ARTIFACT_PROTOCOL
        v6.ADAPTER_PROTOCOL = ADAPTER_PROTOCOL
        data, prompts = v6.load_surrogate_artifact(
            path, records, seed=int(seed), forget_num=int(forget_num)
        )
    finally:
        v6.ARTIFACT_PROTOCOL = old_artifact
        v6.ADAPTER_PROTOCOL = old_adapter

    guard = data.get("high_precision_guard", {})
    if not isinstance(guard, Mapping) or not bool(guard.get("enabled", False)):
        raise RuntimeError("v7 artifact lacks enabled high-precision guard")
    if guard.get("protocol") != hp.PROTOCOL:
        raise RuntimeError("Unexpected v7 high-precision guard protocol")
    if not bool(guard.get("authoritative_for_rejection", False)):
        raise RuntimeError("v7 high-precision guard is not authoritative for rejection")
    for key in ("uses_target_true", "uses_target_new", "uses_official_paraphrases"):
        if bool(guard.get(key, True)):
            raise RuntimeError(f"v7 high-precision guard data-access violation: {key}")

    return data, prompts


base.load_surrogate_artifact = load_surrogate_artifact


if __name__ == "__main__":
    base.main()
