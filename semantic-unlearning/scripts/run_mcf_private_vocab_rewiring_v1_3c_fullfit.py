#!/usr/bin/env python3
"""V1.3c: fit all five training views for every MCF forget case.

This wrapper keeps V1.3's architecture, five-view corpus, worst-view margin,
semantic gates, relation-preserving retain objective, row cap, and leakage
firewall.  It changes optimization only:

* curriculum: 2 views for steps 1-150, 3 views for 151-300, all 5 thereafter;
* hard-case oversampling: after all-five evaluation margins are available,
  75% of each forget minibatch is drawn from cases still below margin 0.1;
* checkpoint/model selection always evaluates the worst margin across all 5
  registered training-only views.

No official paraphrase/neighborhood/generation/retain text is read here.
Seed 1 remains development only.
"""
from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path
import random as _stdlib_random
import sys
from typing import Any, Dict, Mapping, Sequence

import torch

import run_mcf_private_vocab_rewiring_v1_3_multiview as v13


_BASE_RANDOM_CLASS = _stdlib_random.Random
_FORGET_IDS: set[int] = set()
_RELATION_BY_CASE: dict[int, str] = {}
_LATEST_MARGIN_BY_CASE: dict[int, float] = {}
_EVAL_SEEN: set[int] = set()
_TRAIN_STEP = 0
_ACTIVE_TRAIN_VIEWS = 2
_HARD_MARGIN = 0.1
_HARD_FRACTION = 0.75


def curriculum_views(step: int) -> int:
    if int(step) <= 150:
        return 2
    if int(step) <= 300:
        return 3
    return 5


def _case_id(record: Mapping[str, Any]) -> int:
    return int(record["case_id"])


def _is_forget_population(population: Sequence[Any]) -> bool:
    if len(population) != len(_FORGET_IDS) or not population:
        return False
    try:
        ids = {_case_id(item) for item in population}
    except (TypeError, KeyError, ValueError):
        return False
    return ids == _FORGET_IDS


class AdaptiveForgetRandom(_BASE_RANDOM_CLASS):
    """Deterministic hard-case oversampling for the forget minibatch only."""

    def sample(self, population, k, *, counts=None):  # type: ignore[override]
        global _TRAIN_STEP, _ACTIVE_TRAIN_VIEWS
        if counts is not None:
            return super().sample(population, k, counts=counts)
        if not _is_forget_population(population):
            return super().sample(population, k)

        _TRAIN_STEP += 1
        _ACTIVE_TRAIN_VIEWS = curriculum_views(_TRAIN_STEP)

        hard = [
            row
            for row in population
            if _LATEST_MARGIN_BY_CASE.get(_case_id(row), float("inf")) < _HARD_MARGIN
        ]
        hard_target = min(int(k), int(math.ceil(float(k) * _HARD_FRACTION))) if hard else 0
        hard_n = min(len(hard), hard_target)
        chosen = super().sample(hard, hard_n) if hard_n else []
        chosen_ids = {_case_id(row) for row in chosen}
        remaining = [row for row in population if _case_id(row) not in chosen_ids]
        rest_n = int(k) - len(chosen)
        if rest_n:
            chosen.extend(super().sample(remaining, rest_n))
        self.shuffle(chosen)
        return chosen


def _view_records(record: Mapping[str, Any], count: int) -> list[Dict[str, Any]]:
    local = v13.view_records_for_case(record)
    if len(local) != 5:
        raise RuntimeError(f"V1.3c requires exactly 5 registered views, got {len(local)}")
    return local[: int(count)]


def curriculum_multiview_margin_batch(
    model: Any,
    prompt_tokenizer: Any,
    base_tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Train on curriculum views; evaluate/model-select on all five views."""
    training = torch.is_grad_enabled()
    use_views = int(_ACTIVE_TRAIN_VIEWS) if training else 5

    flat: list[Dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    for record in records:
        start = len(flat)
        local = _view_records(record, use_views)
        flat.extend(local)
        spans.append((start, len(flat)))

    values: list[torch.Tensor] = []
    for start in range(0, len(flat), int(v13._VIEW_CHUNK)):
        chunk = flat[start : start + int(v13._VIEW_CHUNK)]
        values.append(
            v13._ORIGINAL_MARGIN_BATCH(
                model,
                prompt_tokenizer,
                base_tokenizer,
                chunk,
                device=device,
            )
        )
    all_values = torch.cat(values, dim=0)
    worst = torch.stack([all_values[start:stop].min() for start, stop in spans])

    if not training:
        for record, value in zip(records, worst.detach().cpu().tolist()):
            cid = _case_id(record)
            _LATEST_MARGIN_BY_CASE[cid] = float(value)
            _EVAL_SEEN.add(cid)
        if _FORGET_IDS and _EVAL_SEEN.issuperset(_FORGET_IDS):
            failures = [
                (cid, _LATEST_MARGIN_BY_CASE[cid], _RELATION_BY_CASE.get(cid, "?"))
                for cid in sorted(_FORGET_IDS)
                if _LATEST_MARGIN_BY_CASE.get(cid, float("-inf")) < _HARD_MARGIN
            ]
            by_relation = Counter(rel for _, _, rel in failures)
            print(
                json.dumps(
                    {
                        "v1_3c_all5_diagnostic": {
                            "train_step": int(_TRAIN_STEP),
                            "failures": len(failures),
                            "passed": len(_FORGET_IDS) - len(failures),
                            "threshold": _HARD_MARGIN,
                            "failure_relations": dict(sorted(by_relation.items())),
                            "failing_cases": [
                                {
                                    "case_id": cid,
                                    "relation_id": rel,
                                    "worst_margin": margin,
                                }
                                for cid, margin, rel in sorted(failures, key=lambda x: x[1])
                            ],
                        }
                    },
                    indent=2,
                ),
                flush=True,
            )
            _EVAL_SEEN.clear()

    return worst


def _load_case_metadata(protocol_dir: Path) -> None:
    global _FORGET_IDS, _RELATION_BY_CASE
    rows = json.loads(
        (protocol_dir / "training_visible_forget_direct.json").read_text(encoding="utf-8")
    )
    _FORGET_IDS = {int(row["case_id"]) for row in rows}
    _RELATION_BY_CASE = {
        int(row["case_id"]): str(row["requested_rewrite"]["relation_id"])
        for row in rows
    }
    if len(_FORGET_IDS) != 50:
        raise RuntimeError(f"V1.3c expected 50 forget cases, found {len(_FORGET_IDS)}")


def main() -> None:
    argv = sys.argv[1:]
    try:
        protocol_dir = Path(argv[argv.index("--protocol-dir") + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise RuntimeError("V1.3c requires --protocol-dir") from exc
    _load_case_metadata(protocol_dir)

    corpus_env = os.environ.get("MCF_V13_VIEW_CORPUS")
    if not corpus_env:
        raise RuntimeError("MCF_V13_VIEW_CORPUS is required")
    view_map, meta = v13.load_view_corpus(Path(corpus_env).resolve())
    if int(meta.get("views_per_case", 0)) != 5 or len(view_map) != 50:
        raise RuntimeError("V1.3c requires the locked 50-case, 5-view V1.3 corpus")

    # v13.main installs its global margin function into the base trainer, so
    # replace that global with our curriculum/full-eval implementation first.
    v13.multiview_worst_margin_batch = curriculum_multiview_margin_batch

    # The base trainer constructs its RNG via random.Random(seed+1101).  Replace
    # only that constructor; module-level seed() and all retain sampling semantics
    # remain otherwise unchanged. AdaptiveForgetRandom delegates non-forget
    # populations to the standard implementation.
    v13.runner.random.Random = AdaptiveForgetRandom

    print(
        json.dumps(
            {
                "protocol": "mcf_private_vocab_rewiring_v1_3c_fullfit_5view",
                "target": "50_of_50_cases_pass_all_5_training_views",
                "margin_threshold": _HARD_MARGIN,
                "curriculum": [
                    {"steps": "1-150", "views": 2},
                    {"steps": "151-300", "views": 3},
                    {"steps": "301+", "views": 5},
                ],
                "hard_case_oversample_fraction": _HARD_FRACTION,
                "checkpoint_evaluation_views": 5,
                "heldout_probe_text_used": False,
            },
            indent=2,
        ),
        flush=True,
    )
    v13.main()


if __name__ == "__main__":
    main()
