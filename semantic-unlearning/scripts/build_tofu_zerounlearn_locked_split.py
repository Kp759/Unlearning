#!/usr/bin/env python3
"""Build leakage-locked TOFU views for a ZeroUnlearn-style 50-shot protocol.

The unlearning stages receive only 50 sampled ``forget05`` question/answer
pairs.  Benchmark-provided paraphrases/perturbed answers, the remaining
``forget05`` facts, retain95, real-authors, and world-facts stay unavailable
until the final frozen-checkpoint evaluation.

Sampling intentionally mirrors ``tofu_eval.subset_samples``: each primary
split uses a fresh ``random.Random(seed).sample``.  Therefore the 50 direct
forget rows and 1,000 retain rows recorded here match the corresponding
standard TOFU evaluator subsets for the same seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

from datasets import load_dataset


DATASET_NAME = "locuslab/TOFU"
FORGET_SPLIT = "forget05"
FORGET_PERTURBED_SPLIT = "forget05_perturbed"
RETAIN_SPLIT = "retain95"
EVAL_ONLY_KEYS = {
    "paraphrased_question",
    "paraphrased_answer",
    "perturbed_answer",
    "perturbed_answers",
}


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _write_json(path: Path, payload: Any) -> str:
    text = _json_text(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _plain_row(row: Dict[str, Any], source_index: int) -> Dict[str, Any]:
    if "question" not in row or "answer" not in row:
        raise ValueError(f"TOFU row {source_index} lacks question/answer")
    return {
        "question": str(row["question"]),
        "answer": str(row["answer"]),
        "_source_index": int(source_index),
    }


def _paraphrase_row(
    direct_row: Dict[str, Any],
    perturbed_row: Dict[str, Any],
    source_index: int,
) -> Dict[str, Any]:
    paraphrased_question = perturbed_row.get("paraphrased_question")
    paraphrased_answer = perturbed_row.get("paraphrased_answer")
    if not paraphrased_question or not paraphrased_answer:
        raise ValueError(
            f"TOFU perturbed row {source_index} lacks paraphrased question/answer"
        )
    result = {
        "question": str(paraphrased_question),
        "answer": str(paraphrased_answer),
        "original_question": str(direct_row["question"]),
        "original_answer": str(direct_row["answer"]),
        "_source_index": int(source_index),
    }
    perturbed = perturbed_row.get(
        "perturbed_answer", perturbed_row.get("perturbed_answers")
    )
    if perturbed is not None:
        result["perturbed_answer"] = perturbed
    return result


def assert_perturbed_alignment(
    forget_rows: Sequence[Dict[str, Any]],
    perturbed_rows: Sequence[Dict[str, Any]],
) -> None:
    if len(forget_rows) != len(perturbed_rows):
        raise ValueError(
            "forget05 and forget05_perturbed have different lengths: "
            f"{len(forget_rows)} vs {len(perturbed_rows)}"
        )
    for index, (direct, perturbed) in enumerate(zip(forget_rows, perturbed_rows)):
        if str(direct.get("question")) != str(perturbed.get("question")):
            raise ValueError(f"TOFU perturbed question misalignment at index {index}")
        if str(direct.get("answer")) != str(perturbed.get("answer")):
            raise ValueError(f"TOFU perturbed answer misalignment at index {index}")


def assert_training_view_locked(rows: Sequence[Dict[str, Any]]) -> None:
    for position, row in enumerate(rows):
        extras = EVAL_ONLY_KEYS.intersection(row)
        if extras:
            raise AssertionError(
                f"training-visible row {position} exposes evaluation keys: {sorted(extras)}"
            )
        if set(row) - {"question", "answer", "_source_index"}:
            raise AssertionError(
                f"training-visible row {position} has unexpected fields: {sorted(row)}"
            )


def sample_indices(population_size: int, sample_size: int, seed: int) -> List[int]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > population_size:
        raise ValueError(
            f"requested {sample_size} rows from population of {population_size}"
        )
    return random.Random(seed).sample(range(population_size), sample_size)


def build_seed_split(
    forget_rows: Sequence[Dict[str, Any]],
    perturbed_rows: Sequence[Dict[str, Any]],
    retain_rows: Sequence[Dict[str, Any]],
    *,
    seed: int,
    forget_num: int,
    retain_num: int,
    output_dir: Path,
    dataset_revision: str | None,
) -> Dict[str, Any]:
    forget_indices = sample_indices(len(forget_rows), forget_num, seed)
    retain_indices = sample_indices(len(retain_rows), retain_num, seed)
    forget_index_set = set(forget_indices)
    heldout_indices = [
        index for index in range(len(forget_rows)) if index not in forget_index_set
    ]

    train_forget = [_plain_row(forget_rows[i], i) for i in forget_indices]
    assert_training_view_locked(train_forget)

    direct_eval = [_plain_row(forget_rows[i], i) for i in forget_indices]
    paraphrase_eval = [
        _paraphrase_row(forget_rows[i], perturbed_rows[i], i)
        for i in forget_indices
    ]
    heldout_direct = [_plain_row(forget_rows[i], i) for i in heldout_indices]
    heldout_paraphrase = [
        _paraphrase_row(forget_rows[i], perturbed_rows[i], i)
        for i in heldout_indices
    ]
    retain_eval = [_plain_row(retain_rows[i], i) for i in retain_indices]

    seed_dir = output_dir / f"seed{seed}"
    train_dir = seed_dir / "train_visible"
    eval_dir = seed_dir / "eval_only"
    paths = {
        "train_forget": train_dir / "forget.json",
        "forget_direct": eval_dir / "forget_direct.json",
        "forget_paraphrase": eval_dir / "forget_paraphrase.json",
        "heldout_direct": eval_dir / "heldout_direct.json",
        "heldout_paraphrase": eval_dir / "heldout_paraphrase.json",
        "retain": eval_dir / "retain.json",
    }
    payloads = {
        "train_forget": train_forget,
        "forget_direct": direct_eval,
        "forget_paraphrase": paraphrase_eval,
        "heldout_direct": heldout_direct,
        "heldout_paraphrase": heldout_paraphrase,
        "retain": retain_eval,
    }
    sha256 = {
        name: _write_json(paths[name], payloads[name]) for name in paths
    }

    if [row["_source_index"] for row in train_forget] != forget_indices:
        raise AssertionError("training view changed forget ordering")
    if [row["_source_index"] for row in direct_eval] != forget_indices:
        raise AssertionError("direct eval view changed forget ordering")
    if [row["_source_index"] for row in paraphrase_eval] != forget_indices:
        raise AssertionError("paraphrase eval view changed forget ordering")
    if set(forget_indices).intersection(heldout_indices):
        raise AssertionError("training and held-out forget indices overlap")

    manifest = {
        "schema_version": 1,
        "protocol": "tofu_zerounlearn_data_access_forget_only_locked",
        "dataset": DATASET_NAME,
        "dataset_revision": dataset_revision,
        "forget_split": FORGET_SPLIT,
        "retain_split": RETAIN_SPLIT,
        "seed": seed,
        "source_counts": {
            "forget05": len(forget_rows),
            "retain95": len(retain_rows),
        },
        "sampling": {
            "implementation": "random.Random(seed).sample with fresh RNG per split",
            "forget_num": forget_num,
            "retain_num": retain_num,
            "forget_source_indices": forget_indices,
            "retain_source_indices": retain_indices,
            "heldout_forget_source_indices": heldout_indices,
        },
        "data_access": {
            "stage1": {
                "forget_records": forget_num,
                "retain_records": 0,
                "visible_fields": ["question", "answer"],
                "paraphrases": 0,
                "perturbed_answers": 0,
            },
            "stage2": {
                "forget_records": forget_num,
                "retain_records": 0,
                "visible_fields": ["question", "answer"],
                "paraphrases": 0,
                "perturbed_answers": 0,
            },
            "final_evaluation_only": {
                "paired_forget_paraphrases": forget_num,
                "unseen_same_split_forget_facts": len(heldout_indices),
                "retain95_records": retain_num,
                "real_authors": "standard final evaluator only",
                "world_facts": "standard final evaluator only",
            },
        },
        "holdout": {
            "primary_generalization": "same selected forget QA facts, benchmark-provided paraphrased questions",
            "secondary_generalization": "remaining forget05 QA facts never exposed to unlearning",
            "prompt_level_holdout": True,
            "additional_fact_level_holdout": True,
        },
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "sha256": sha256,
    }
    _write_json(seed_dir / "split_manifest.json", manifest)
    return manifest


def _load_split(name: str, revision: str | None) -> List[Dict[str, Any]]:
    kwargs: Dict[str, Any] = {
        "path": DATASET_NAME,
        "name": name,
        "split": "train",
    }
    if revision:
        kwargs["revision"] = revision
    return [dict(row) for row in load_dataset(**kwargs)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(range(1, 11))
    )
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help="Optional immutable Hugging Face dataset revision.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading TOFU forget05 / forget05_perturbed / retain95")
    forget_rows = _load_split(FORGET_SPLIT, args.dataset_revision)
    perturbed_rows = _load_split(FORGET_PERTURBED_SPLIT, args.dataset_revision)
    retain_rows = _load_split(RETAIN_SPLIT, args.dataset_revision)
    assert_perturbed_alignment(forget_rows, perturbed_rows)

    if args.forget_num >= len(forget_rows):
        raise ValueError(
            "forget-num must leave at least one forget05 record unseen for the locked diagnostic"
        )
    if args.retain_num > len(retain_rows):
        raise ValueError(
            f"retain-num={args.retain_num} exceeds retain95 size {len(retain_rows)}"
        )

    manifests = [
        build_seed_split(
            forget_rows,
            perturbed_rows,
            retain_rows,
            seed=seed,
            forget_num=args.forget_num,
            retain_num=args.retain_num,
            output_dir=output_dir,
            dataset_revision=args.dataset_revision,
        )
        for seed in args.seeds
    ]

    root_manifest = {
        "schema_version": 1,
        "protocol": "tofu_zerounlearn_data_access_forget_only_locked",
        "dataset": DATASET_NAME,
        "dataset_revision": args.dataset_revision,
        "forget_split": FORGET_SPLIT,
        "retain_split": RETAIN_SPLIT,
        "seeds": args.seeds,
        "forget_num": args.forget_num,
        "retain_eval_num": args.retain_num,
        "stage1_retain_num": 0,
        "stage2_retain_num": 0,
        "seed_manifests": [
            str((output_dir / f"seed{item['seed']}" / "split_manifest.json").resolve())
            for item in manifests
        ],
    }
    root_path = output_dir / "split_manifest.json"
    _write_json(root_path, root_manifest)
    print(f"locked TOFU protocol: {root_path}")
    print(
        f"per seed: {args.forget_num} direct forget QA visible; "
        f"{args.retain_num} retain + paraphrases + remaining forget05 are evaluation-only"
    )


if __name__ == "__main__":
    main()
