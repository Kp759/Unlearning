#!/usr/bin/env python3
"""Build an author-balanced, leakage-locked TOFU unlearning protocol.

TOFU contains 20 QA pairs per fictitious author.  For the default forget05
protocol we treat the 200 rows as 10 contiguous 20-QA author blocks.  Per seed:

* choose 5 forget authors;
* expose 10 direct QAs per selected author to Stage 1 / Stage 2 (50 total);
* hold out the other 10 QAs per selected author for same-author evaluation
  (50 total);
* keep paraphrases, perturbed answers, retain95, real-authors, and world-facts
  unavailable until the final frozen-checkpoint evaluation;
* sample 1,000 retain95 QAs for final utility evaluation.

The training-visible JSON contains only question, answer, and source index.
Author-block membership is recorded only in the split manifest for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from datasets import load_dataset


DATASET_NAME = "locuslab/TOFU"
FORGET_SPLIT = "forget05"
FORGET_PERTURBED_SPLIT = "forget05_perturbed"
RETAIN_SPLIT = "retain95"
PROTOCOL = "tofu_author_balanced_forget_only_locked_v1"
DEFAULT_QAS_PER_AUTHOR = 20
DEFAULT_FORGET_AUTHORS = 5
DEFAULT_TRAIN_QAS_PER_AUTHOR = 10
DEFAULT_RETAIN_NUM = 1000
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


def build_author_blocks(population_size: int, qas_per_author: int) -> List[List[int]]:
    if qas_per_author <= 0:
        raise ValueError("qas-per-author must be positive")
    if population_size % qas_per_author != 0:
        raise ValueError(
            f"forget split size {population_size} is not divisible by "
            f"qas-per-author={qas_per_author}"
        )
    return [
        list(range(start, start + qas_per_author))
        for start in range(0, population_size, qas_per_author)
    ]


def sample_author_balanced_indices(
    population_size: int,
    *,
    seed: int,
    qas_per_author: int,
    forget_authors: int,
    train_qas_per_author: int,
) -> Tuple[List[int], List[int], List[int], Dict[int, Dict[str, List[int]]]]:
    blocks = build_author_blocks(population_size, qas_per_author)
    if forget_authors <= 0 or forget_authors > len(blocks):
        raise ValueError(
            f"forget-authors must be in [1,{len(blocks)}], got {forget_authors}"
        )
    if train_qas_per_author <= 0 or train_qas_per_author >= qas_per_author:
        raise ValueError(
            "train-qas-per-author must be positive and smaller than qas-per-author"
        )

    rng = random.Random(seed)
    selected_author_ids = rng.sample(range(len(blocks)), forget_authors)
    train_indices: List[int] = []
    heldout_indices: List[int] = []
    per_author: Dict[int, Dict[str, List[int]]] = {}

    for author_id in selected_author_ids:
        block = blocks[author_id]
        train = rng.sample(block, train_qas_per_author)
        train_set = set(train)
        heldout = [index for index in block if index not in train_set]
        train_indices.extend(train)
        heldout_indices.extend(heldout)
        per_author[author_id] = {
            "all_source_indices": list(block),
            "train_source_indices": list(train),
            "heldout_source_indices": list(heldout),
        }

    expected_train = forget_authors * train_qas_per_author
    expected_heldout = forget_authors * (qas_per_author - train_qas_per_author)
    if len(train_indices) != expected_train or len(set(train_indices)) != expected_train:
        raise AssertionError("author-balanced train sampling produced duplicate/missing rows")
    if len(heldout_indices) != expected_heldout or len(set(heldout_indices)) != expected_heldout:
        raise AssertionError("author-balanced holdout produced duplicate/missing rows")
    if set(train_indices).intersection(heldout_indices):
        raise AssertionError("training and same-author holdout indices overlap")

    return selected_author_ids, train_indices, heldout_indices, per_author


def build_seed_split(
    forget_rows: Sequence[Dict[str, Any]],
    perturbed_rows: Sequence[Dict[str, Any]],
    retain_rows: Sequence[Dict[str, Any]],
    *,
    seed: int,
    retain_num: int,
    output_dir: Path,
    dataset_revision: str | None,
    qas_per_author: int = DEFAULT_QAS_PER_AUTHOR,
    forget_authors: int = DEFAULT_FORGET_AUTHORS,
    train_qas_per_author: int = DEFAULT_TRAIN_QAS_PER_AUTHOR,
) -> Dict[str, Any]:
    (
        selected_author_ids,
        forget_indices,
        heldout_indices,
        per_author,
    ) = sample_author_balanced_indices(
        len(forget_rows),
        seed=seed,
        qas_per_author=qas_per_author,
        forget_authors=forget_authors,
        train_qas_per_author=train_qas_per_author,
    )
    retain_indices = sample_indices(len(retain_rows), retain_num, seed)

    train_forget = [_plain_row(forget_rows[i], i) for i in forget_indices]
    assert_training_view_locked(train_forget)

    # Efficacy on the exact 50 deletion requests seen during unlearning.
    direct_eval = [_plain_row(forget_rows[i], i) for i in forget_indices]
    paraphrase_eval = [
        _paraphrase_row(forget_rows[i], perturbed_rows[i], i)
        for i in forget_indices
    ]

    # Same-author generalization: the other 10 QAs for each selected author.
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
    if [row["_source_index"] for row in heldout_direct] != heldout_indices:
        raise AssertionError("same-author holdout view changed ordering")

    train_num = len(forget_indices)
    heldout_num = len(heldout_indices)
    all_author_ids = list(range(len(build_author_blocks(len(forget_rows), qas_per_author))))
    unselected_author_ids = [
        author_id for author_id in all_author_ids if author_id not in set(selected_author_ids)
    ]

    manifest = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "dataset": DATASET_NAME,
        "dataset_revision": dataset_revision,
        "forget_split": FORGET_SPLIT,
        "retain_split": RETAIN_SPLIT,
        "seed": seed,
        "source_counts": {
            "forget05": len(forget_rows),
            "retain95": len(retain_rows),
            "forget05_author_blocks": len(all_author_ids),
            "qas_per_author": qas_per_author,
        },
        "sampling": {
            "implementation": (
                "random.Random(seed): sample author blocks, then sample QAs within each "
                "selected author; retain95 uses a fresh random.Random(seed).sample"
            ),
            "forget_authors": forget_authors,
            "train_qas_per_author": train_qas_per_author,
            "heldout_qas_per_author": qas_per_author - train_qas_per_author,
            "train_forget_num": train_num,
            "same_author_heldout_num": heldout_num,
            "retain_num": retain_num,
            "selected_author_block_ids": selected_author_ids,
            "unselected_author_block_ids": unselected_author_ids,
            "per_selected_author": {
                str(author_id): details for author_id, details in per_author.items()
            },
            "forget_source_indices": forget_indices,
            "heldout_same_author_source_indices": heldout_indices,
            "retain_source_indices": retain_indices,
        },
        "data_access": {
            "stage1": {
                "forget_records": train_num,
                "forget_authors": forget_authors,
                "qas_per_author": train_qas_per_author,
                "retain_records": 0,
                "visible_fields": ["question", "answer"],
                "paraphrases": 0,
                "perturbed_answers": 0,
            },
            "stage2": {
                "forget_records": train_num,
                "forget_authors": forget_authors,
                "qas_per_author": train_qas_per_author,
                "retain_records": 0,
                "visible_fields": ["question", "answer"],
                "paraphrases": 0,
                "perturbed_answers": 0,
            },
            "final_evaluation_only": {
                "seen_forget_direct": train_num,
                "seen_forget_paraphrases": train_num,
                "same_author_unseen_direct": heldout_num,
                "same_author_unseen_paraphrases": heldout_num,
                "retain95_records": retain_num,
                "real_authors": "optional standard final evaluator only",
                "world_facts": "optional standard final evaluator only",
            },
        },
        "holdout": {
            "efficacy": "same 10 training QAs per selected author, evaluated after freezing",
            "prompt_generalization": "benchmark paraphrases of the 50 seen deletion QAs",
            "same_author_fact_generalization": (
                "other 10 QAs per selected author, never exposed to unlearning"
            ),
            "same_author_fact_generalization_paraphrase": (
                "benchmark paraphrases of the 50 unseen same-author QAs"
            ),
            "prompt_level_holdout": True,
            "same_author_fact_level_holdout": True,
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
    parser.add_argument("--forget-authors", type=int, default=DEFAULT_FORGET_AUTHORS)
    parser.add_argument("--qas-per-author", type=int, default=DEFAULT_QAS_PER_AUTHOR)
    parser.add_argument(
        "--train-qas-per-author", type=int, default=DEFAULT_TRAIN_QAS_PER_AUTHOR
    )
    parser.add_argument("--retain-num", type=int, default=DEFAULT_RETAIN_NUM)
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
    build_author_blocks(len(forget_rows), args.qas_per_author)

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
            retain_num=args.retain_num,
            output_dir=output_dir,
            dataset_revision=args.dataset_revision,
            qas_per_author=args.qas_per_author,
            forget_authors=args.forget_authors,
            train_qas_per_author=args.train_qas_per_author,
        )
        for seed in args.seeds
    ]

    train_num = args.forget_authors * args.train_qas_per_author
    heldout_num = args.forget_authors * (
        args.qas_per_author - args.train_qas_per_author
    )
    root_manifest = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "dataset": DATASET_NAME,
        "dataset_revision": args.dataset_revision,
        "forget_split": FORGET_SPLIT,
        "retain_split": RETAIN_SPLIT,
        "seeds": args.seeds,
        "forget_authors": args.forget_authors,
        "qas_per_author": args.qas_per_author,
        "train_qas_per_author": args.train_qas_per_author,
        "train_forget_num": train_num,
        "same_author_heldout_num": heldout_num,
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
    print(f"author-balanced locked TOFU protocol: {root_path}")
    print(
        f"per seed: {args.forget_authors} authors x {args.train_qas_per_author} "
        f"train QAs = {train_num}; {heldout_num} same-author QAs + "
        f"{args.retain_num} retain95 QAs are evaluation-only"
    )


if __name__ == "__main__":
    main()
