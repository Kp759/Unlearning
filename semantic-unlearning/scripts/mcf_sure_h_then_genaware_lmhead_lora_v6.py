#!/usr/bin/env python3
"""Gen-aware MCF Stage 2 consuming dataset-adapter robust prompt sets.

The SURE optimization remains exactly the common implementation in
``mcf_sure_h_then_genaware_lmhead_lora.py``.  This entrypoint changes only the
dataset-adapter boundary: each MCF fact may contribute either 3--8 validated
semantic surrogates or zero surrogates (intentional direct-only fallback).

No MCF-specific relation logic is present in the optimizer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import mcf_surrogate_answer_guard as answer_guard
import mcf_sure_h_then_genaware_lmhead_lora as base


ARTIFACT_PROTOCOL = "mcf_direct_only_robust_prompt_adapter_v6"
ADAPTER_PROTOCOL = "mcf_relation_slot_answer_type_semantic_adapter_v1"
RELATION_PROTOCOL = "mcf_answer_blind_relation_slot_adapter_v1"
SEMANTIC_PROTOCOL = "mcf_direct_only_relation_slot_boolean_consensus_v3"


def _norm(text: str) -> str:
    return " ".join(str(text).split()).strip().casefold()


def load_surrogate_artifact(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    forget_num: int,
) -> Tuple[Dict[str, Any], List[List[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", -1)) != 1:
        raise RuntimeError("Unsupported MCF robust-prompt adapter schema")
    if data.get("protocol") != ARTIFACT_PROTOCOL:
        raise RuntimeError("Gen-aware v6 requires the MCF robust-prompt adapter v6 artifact")
    if data.get("adapter_protocol") != ADAPTER_PROTOCOL:
        raise RuntimeError("Unexpected MCF dataset-adapter protocol")
    if int(data.get("seed", -1)) != int(seed):
        raise RuntimeError("MCF adapter seed mismatch")
    if int(data.get("forget_num", -1)) != int(forget_num):
        raise RuntimeError("MCF adapter forget count mismatch")

    min_surrogates = int(data.get("min_surrogates", -1))
    max_surrogates = int(data.get("max_surrogates", -1))
    if min_surrogates < 1 or max_surrogates < min_surrogates:
        raise RuntimeError("Invalid min/max surrogate contract")

    access = data.get("data_access", {})
    if int(access.get("official_paraphrase_seen", -1)) != 0:
        raise RuntimeError("Adapter reports official paraphrase access")
    if int(access.get("official_neighborhood_seen", -1)) != 0:
        raise RuntimeError("Adapter reports official neighborhood access")
    if int(access.get("benchmark_retain_seen", -1)) != 0:
        raise RuntimeError("Adapter reports benchmark retain access")
    if bool(access.get("official_PPL_seen", True)):
        raise RuntimeError("Adapter reports official PPL access")

    generator = data.get("generator", {})
    if bool(generator.get("generator_received_target_true", True)):
        raise RuntimeError("Adapter generator received target_true")
    if bool(generator.get("generator_received_target_new", True)):
        raise RuntimeError("Adapter generator received target_new")
    if bool(generator.get("deterministic_wrapper_fallback_used", True)):
        raise RuntimeError("Adapter used deterministic wrapper fallback")

    relation = data.get("relation_slot_adapter", {})
    if relation.get("protocol") != RELATION_PROTOCOL:
        raise RuntimeError("Unexpected relation-slot adapter protocol")
    if bool(relation.get("classifier_received_target_true", True)):
        raise RuntimeError("Relation-slot classifier received target_true")
    if bool(relation.get("classifier_received_target_new", True)):
        raise RuntimeError("Relation-slot classifier received target_new")

    semantic = data.get("semantic_validation", {})
    if semantic.get("protocol") != SEMANTIC_PROTOCOL:
        raise RuntimeError("Unexpected semantic-equivalence validator protocol")
    if bool(semantic.get("validator_received_target_true", True)):
        raise RuntimeError("Semantic validator received target_true")
    if bool(semantic.get("validator_received_target_new", True)):
        raise RuntimeError("Semantic validator received target_new")
    if bool(semantic.get("validator_received_official_paraphrases", True)):
        raise RuntimeError("Semantic validator received official paraphrases")

    rows = data.get("records")
    if not isinstance(rows, list) or len(rows) != int(forget_num):
        raise RuntimeError("Adapter record count mismatch")

    all_prompts: List[List[str]] = []
    for pos, (record, row) in enumerate(zip(records, rows)):
        if int(row.get("sampled_position", -1)) != pos:
            raise RuntimeError(f"Adapter sampled_position mismatch at record {pos}")
        expected_case = int(record.get("case_id", pos))
        if int(row.get("case_id", -1)) != expected_case:
            raise RuntimeError(f"Adapter case_id mismatch at record {pos}")

        rr = record["requested_rewrite"]
        subject = str(rr["subject"])
        direct_prompt = str(rr["prompt"]).format(subject)
        if str(row.get("subject", "")) != subject:
            raise RuntimeError(f"Adapter subject mismatch at record {pos}")
        if _norm(row.get("direct_prompt", "")) != _norm(direct_prompt):
            raise RuntimeError(f"Adapter direct prompt mismatch at record {pos}")

        prompts = row.get("surrogate_prompts")
        if not isinstance(prompts, list):
            raise RuntimeError(f"Adapter surrogate_prompts is not a list at record {pos}")
        status = str(row.get("augmentation_status", ""))
        count = len(prompts)
        declared_count = int(row.get("surrogate_count", -1))
        if declared_count != count:
            raise RuntimeError(f"Adapter surrogate count mismatch at record {pos}")
        if status == "direct_only":
            if count != 0:
                raise RuntimeError(f"direct_only record {pos} unexpectedly has surrogates")
        elif status == "robust_prompt_set":
            if not (min_surrogates <= count <= max_surrogates):
                raise RuntimeError(
                    f"robust_prompt_set record {pos} has {count} surrogates; "
                    f"expected {min_surrogates}..{max_surrogates}"
                )
        else:
            raise RuntimeError(f"Unknown augmentation_status at record {pos}: {status!r}")

        answers = [
            str(rr["target_true"]["str"]),
            str(rr["target_new"]["str"]),
        ]
        seen = {_norm(direct_prompt)}
        clean: List[str] = []
        for j, prompt in enumerate(prompts):
            prompt = " ".join(str(prompt).split()).strip()
            key = _norm(prompt)
            if not prompt or key in seen:
                raise RuntimeError(f"Empty/duplicate adapter surrogate at record {pos}, index {j}")
            if answer_guard.introduced_answer_occurrences(prompt, direct_prompt, answers):
                raise RuntimeError(
                    f"Adapter surrogate introduced an answer occurrence at record {pos}, index {j}"
                )
            seen.add(key)
            clean.append(prompt)
        all_prompts.append(clean)

    summary = data.get("adapter_summary", {})
    actual_augmented = sum(bool(x) for x in all_prompts)
    actual_direct_only = len(all_prompts) - actual_augmented
    actual_total = sum(len(x) for x in all_prompts)
    if int(summary.get("records_with_robust_prompt_sets", -1)) != actual_augmented:
        raise RuntimeError("Adapter summary augmented-record count mismatch")
    if int(summary.get("records_direct_only", -1)) != actual_direct_only:
        raise RuntimeError("Adapter summary direct-only count mismatch")
    if int(summary.get("surrogate_prompts_total", -1)) != actual_total:
        raise RuntimeError("Adapter summary surrogate total mismatch")

    return data, all_prompts


# Patch only the adapter loading boundary.  The common Gen-aware Stage-2 code
# handles the resulting variable-size surrogate lists without dataset-specific
# relation logic.
base.load_surrogate_artifact = load_surrogate_artifact


if __name__ == "__main__":
    base.main()
