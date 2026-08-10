#!/usr/bin/env python3
"""Build a ZeroUnlearn-style MCF view with evaluation probes locked away.

The source MultiCounterFact file is kept unchanged for final evaluation.  This
script writes a second, repair-visible copy in which paraphrase, neighborhood,
and generation prompts are empty.  The record order and requested_rewrite
objects are preserved, so `sample_official_mcf_records` selects exactly the
same forget/retain records from both files for every seed.

This is a prompt-level holdout protocol, not a fact-level train/test split:
the same sampled forget facts are supplied as deletion requests, while their
benchmark-provided paraphrases/neighborhood probes remain unavailable until
the final frozen-checkpoint evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from mcf_sampling import sample_official_mcf_records


EVALUATION_ONLY_FIELDS = (
    "paraphrase_prompts",
    "neighborhood_prompts",
    "generation_prompts",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def scrub_requested_rewrite(value: Any) -> Any:
    """Copy requested_rewrite while removing any nested evaluation probes."""
    if isinstance(value, dict):
        result = copy.deepcopy(value)
        for field in EVALUATION_ONLY_FIELDS:
            if field in result:
                result[field] = []
        return result
    if isinstance(value, list):
        return [scrub_requested_rewrite(item) for item in value]
    return copy.deepcopy(value)


def build_repair_visible_dataset(data: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    visible: List[Dict[str, Any]] = []
    for record in data:
        copied = copy.deepcopy(record)
        for field in EVALUATION_ONLY_FIELDS:
            copied[field] = []
        if "requested_rewrite" in copied:
            copied["requested_rewrite"] = scrub_requested_rewrite(
                copied["requested_rewrite"]
            )
        visible.append(copied)
    return visible


def _nested_probe_values(value: Any) -> Iterable[Tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in EVALUATION_ONLY_FIELDS:
                yield key, child
            yield from _nested_probe_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_probe_values(child)


def assert_repair_view_locked(data: Sequence[Dict[str, Any]]) -> None:
    for index, record in enumerate(data):
        for field in EVALUATION_ONLY_FIELDS:
            if record.get(field):
                raise AssertionError(
                    f"repair-visible record {index} still exposes {field}"
                )
        for field, value in _nested_probe_values(record.get("requested_rewrite")):
            if value:
                raise AssertionError(
                    f"repair-visible record {index} requested_rewrite still "
                    f"exposes {field}"
                )


def selected_indices(
    data: Sequence[Dict[str, Any]],
    forget_num: int,
    retain_num: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    index_by_identity = {id(record): index for index, record in enumerate(data)}
    forget, retain = sample_official_mcf_records(
        data,
        forget_num=forget_num,
        retain_num=retain_num,
        seed=seed,
        strict=True,
    )
    return (
        [index_by_identity[id(record)] for record in forget],
        [index_by_identity[id(record)] for record in retain],
    )


def case_ids(data: Sequence[Dict[str, Any]], indices: Sequence[int]) -> List[Any]:
    return [data[index].get("case_id", index) for index in indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="ZeroUnlearn few-shot paper uses seeds 1..10.",
    )
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.forget_num <= 0 or args.retain_num <= 0:
        raise ValueError("forget-num and retain-num must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")

    source_path = Path(args.mcf_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_bytes = source_path.read_bytes()
    data = json.loads(source_bytes)
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("MCF source must be a JSON list of objects")

    half = len(data) // 2
    if half < args.retain_num or len(data) - half < args.forget_num:
        raise ValueError(
            "ZeroUnlearn pool split is too small for requested sample sizes: "
            f"retain_pool={half}, forget_pool={len(data)-half}, "
            f"retain_num={args.retain_num}, forget_num={args.forget_num}"
        )

    repair_visible = build_repair_visible_dataset(data)
    assert_repair_view_locked(repair_visible)

    repair_path = output_dir / "repair_visible_mcf.json"
    repair_text = json.dumps(repair_visible, ensure_ascii=False, indent=2) + "\n"
    repair_path.write_text(repair_text, encoding="utf-8")

    per_seed: List[Dict[str, Any]] = []
    for seed in args.seeds:
        source_forget, source_retain = selected_indices(
            data, args.forget_num, args.retain_num, seed
        )
        repair_forget, repair_retain = selected_indices(
            repair_visible, args.forget_num, args.retain_num, seed
        )
        if source_forget != repair_forget or source_retain != repair_retain:
            raise AssertionError(
                f"sanitized repair view changed ZeroUnlearn selection for seed {seed}"
            )
        if set(source_forget) & set(source_retain):
            raise AssertionError(f"forget/retain overlap for seed {seed}")
        if any(index < half for index in source_forget):
            raise AssertionError(f"seed {seed} forget sample escaped second-half pool")
        if any(index >= half for index in source_retain):
            raise AssertionError(f"seed {seed} retain sample escaped first-half pool")

        per_seed.append(
            {
                "seed": seed,
                "forget_record_indices": source_forget,
                "retain_record_indices": source_retain,
                "forget_case_ids": case_ids(data, source_forget),
                "retain_case_ids": case_ids(data, source_retain),
            }
        )

    manifest = {
        "schema_version": 1,
        "protocol": "mcf_zerounlearn_locked_paraphrase",
        "source_dataset": str(source_path),
        "repair_visible_dataset": str(repair_path),
        "source_sha256": sha256_bytes(source_bytes),
        "repair_visible_sha256": sha256_bytes(repair_text.encode("utf-8")),
        "dataset_size": len(data),
        "pool_split": {
            "retain_pool": {"start": 0, "stop_exclusive": half, "size": half},
            "forget_pool": {
                "start": half,
                "stop_exclusive": len(data),
                "size": len(data) - half,
            },
        },
        "sampling": {
            "implementation": "sample_official_mcf_records",
            "order": "forget sample first, then retain sample from one seeded RNG",
            "forget_num": args.forget_num,
            "retain_num": args.retain_num,
            "seeds": args.seeds,
        },
        "data_roles": {
            "method_visible": ["requested_rewrite"],
            "evaluation_only": list(EVALUATION_ONLY_FIELDS),
            "final_evaluation_uses_original_source_file": True,
            "same_underlying_forget_facts_at_final_eval": True,
            "fact_level_holdout": False,
            "prompt_level_holdout": True,
        },
        "per_seed": per_seed,
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"repair-visible MCF: {repair_path}")
    print(f"split manifest: {manifest_path}")
    print(
        "locked fields: " + ", ".join(EVALUATION_ONLY_FIELDS)
        + "; final evaluation must use the original MCF file"
    )


if __name__ == "__main__":
    main()
