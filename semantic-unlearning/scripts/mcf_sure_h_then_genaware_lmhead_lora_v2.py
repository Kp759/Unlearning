#!/usr/bin/env python3
"""Gen-aware MCF Stage 2 v2 with baseline-aware surrogate answer validation.

Training/optimization is exactly the implementation in
mcf_sure_h_then_genaware_lmhead_lora.py.  This entrypoint replaces only the
surrogate-artifact validator so an answer string already present in the locked
direct prompt is not falsely classified as leakage.  Additional answer mentions
introduced by a surrogate are still rejected.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import mcf_surrogate_answer_guard as answer_guard
import mcf_sure_h_then_genaware_lmhead_lora as base


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
        raise RuntimeError("Unsupported surrogate artifact schema")
    if data.get("protocol") != base.SURROGATE_PROTOCOL:
        raise RuntimeError("Unexpected surrogate artifact protocol")
    if int(data.get("seed", -1)) != int(seed):
        raise RuntimeError("Surrogate artifact seed mismatch")
    if int(data.get("forget_num", -1)) != int(forget_num):
        raise RuntimeError("Surrogate artifact forget count mismatch")

    access = data.get("data_access", {})
    if int(access.get("official_paraphrase_seen", -1)) != 0:
        raise RuntimeError("Surrogate artifact reports official paraphrase access")
    if int(access.get("official_neighborhood_seen", -1)) != 0:
        raise RuntimeError("Surrogate artifact reports official neighborhood access")
    if int(access.get("benchmark_retain_seen", -1)) != 0:
        raise RuntimeError("Surrogate artifact reports benchmark retain access")
    if bool(access.get("official_PPL_seen", True)):
        raise RuntimeError("Surrogate artifact reports official PPL access")

    rows = data.get("records")
    if not isinstance(rows, list) or len(rows) != int(forget_num):
        raise RuntimeError("Surrogate artifact records do not match forget count")

    all_prompts: List[List[str]] = []
    for pos, (record, row) in enumerate(zip(records, rows)):
        if int(row.get("sampled_position", -1)) != pos:
            raise RuntimeError(f"Surrogate sampled_position mismatch at {pos}")
        expected_case = int(record.get("case_id", pos))
        if int(row.get("case_id", -1)) != expected_case:
            raise RuntimeError(f"Surrogate case_id mismatch at {pos}")

        rr = record["requested_rewrite"]
        subject = str(rr["subject"])
        direct_prompt = str(rr["prompt"]).format(subject)
        if str(row.get("subject", "")) != subject:
            raise RuntimeError(f"Surrogate subject mismatch at {pos}")
        if _norm(row.get("direct_prompt", "")) != _norm(direct_prompt):
            raise RuntimeError(f"Surrogate direct prompt mismatch at {pos}")

        prompts = row.get("surrogate_prompts")
        if not isinstance(prompts, list) or not prompts:
            raise RuntimeError(f"No surrogate prompts at {pos}")

        answers = [
            str(rr["target_true"]["str"]),
            str(rr["target_new"]["str"]),
        ]
        direct_key = _norm(direct_prompt)
        seen = {direct_key}
        clean: List[str] = []
        for j, prompt in enumerate(prompts):
            prompt = " ".join(str(prompt).split()).strip()
            key = _norm(prompt)
            if not prompt or key in seen:
                raise RuntimeError(
                    f"Empty/duplicate surrogate at record {pos}, index {j}"
                )
            if answer_guard.introduced_answer_occurrences(
                prompt, direct_prompt, answers
            ):
                raise RuntimeError(
                    f"answer occurrence introduced into surrogate at record {pos}, index {j}"
                )
            seen.add(key)
            clean.append(prompt)
        all_prompts.append(clean)

    return data, all_prompts


# Patch only the artifact-validation boundary. All training behavior, active-set
# logic, LoRA parameterization, locality guards, and scale selection stay v1.
base.load_surrogate_artifact = load_surrogate_artifact


if __name__ == "__main__":
    base.main()
