#!/usr/bin/env python3
"""Deterministic balanced fact-cycle schedules for RWKU entity-fact training.

The schedule is constructed before any optimization begins.  Each complete
cycle visits every fact exactly once in a deterministically shuffled order.
Consequently, for ``S`` requested steps and ``K`` facts, every fact receives
either ``floor(S/K)`` or ``ceil(S/K)`` updates.

This module is deliberately torch-free so preparation and unit tests remain
CPU-only.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


SAMPLER_VERSION = "rwku_balanced_fact_cycle_v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stable_view_id(fact_id: str, view: Mapping[str, Any]) -> str:
    declared = view.get("view_id")
    if declared:
        return str(declared)
    return hashlib.sha256(
        fact_id.encode("utf-8") + b"\0" + _canonical_bytes(dict(view))
    ).hexdigest()


def implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build_fact_cycle_plan(
    views_by_fact: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    steps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Return a deterministic one-view-per-step balanced fact schedule."""

    if steps < 0:
        raise ValueError("steps must be non-negative")
    fact_ids = sorted(str(fact_id) for fact_id in views_by_fact)
    if not fact_ids and steps:
        raise ValueError("Balanced fact-cycle sampling requires at least one fact")

    normalized: Dict[str, List[Dict[str, Any]]] = {}
    for fact_id in fact_ids:
        views = [dict(view) for view in views_by_fact[fact_id]]
        if not views:
            raise ValueError(f"Fact {fact_id} has no optimization views")
        for view in views:
            view.setdefault("view_id", _stable_view_id(fact_id, view))
        normalized[fact_id] = sorted(views, key=lambda item: str(item["view_id"]))

    rng = random.Random(seed)
    per_fact_view_order: Dict[str, List[int]] = {}
    per_fact_view_cursor: Dict[str, int] = {}

    def next_view(fact_id: str) -> Dict[str, Any]:
        views = normalized[fact_id]
        order = per_fact_view_order.get(fact_id, [])
        cursor = per_fact_view_cursor.get(fact_id, len(order))
        if cursor >= len(order):
            order = list(range(len(views)))
            rng.shuffle(order)
            cursor = 0
            per_fact_view_order[fact_id] = order
        selected = dict(views[order[cursor]])
        per_fact_view_cursor[fact_id] = cursor + 1
        return selected

    plan: List[Dict[str, Any]] = []
    while len(plan) < steps:
        cycle_order = list(fact_ids)
        rng.shuffle(cycle_order)
        cycle_index = len(plan) // max(1, len(fact_ids))
        for fact_id in cycle_order:
            if len(plan) >= steps:
                break
            view = next_view(fact_id)
            plan.append(
                {
                    "step": len(plan),
                    "cycle": cycle_index,
                    "fact_id": fact_id,
                    "view_id": str(view["view_id"]),
                    "prompt_style": str(view.get("prompt_style", view.get("style", "unknown"))),
                    "sensitive_answer_alias": str(
                        view.get(
                            "sensitive_answer_alias",
                            view.get("answer", view.get("canonical_sensitive_answer", "")),
                        )
                    ),
                    "view": view,
                }
            )
    return plan


def exposure_report(
    plan: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    tokenizer: Any | None = None,
) -> Dict[str, Any]:
    """Summarize fact/view/style/alias and actual tokenizer-token exposure."""

    fact_counts: Counter[str] = Counter()
    view_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    alias_counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    piece_counts: Counter[str] = Counter()
    fact_view_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    for entry in plan:
        fact_id = str(entry["fact_id"])
        view_id = str(entry["view_id"])
        style = str(entry.get("prompt_style", "unknown"))
        answer = str(entry.get("sensitive_answer_alias", ""))
        fact_counts[fact_id] += 1
        view_counts[view_id] += 1
        fact_view_counts[fact_id][view_id] += 1
        style_counts[style] += 1
        alias_counts[answer] += 1
        if tokenizer is not None and answer:
            token_ids = tokenizer.encode(answer, add_special_tokens=False)
            for token_id in token_ids:
                token_counts[str(int(token_id))] += 1
                piece_counts[str(tokenizer.decode([int(token_id)]))] += 1

    minimum = min(fact_counts.values()) if fact_counts else 0
    maximum = max(fact_counts.values()) if fact_counts else 0
    report: Dict[str, Any] = {
        "sampler": SAMPLER_VERSION,
        "sampler_seed": int(seed),
        "sampler_implementation_sha256": implementation_sha256(),
        "requested_steps": len(plan),
        "fact_exposure_count": dict(sorted(fact_counts.items())),
        "view_exposure_count": dict(sorted(view_counts.items())),
        "fact_view_exposure_count": {
            fact_id: dict(sorted(counts.items()))
            for fact_id, counts in sorted(fact_view_counts.items())
        },
        "prompt_style_exposure_count": dict(sorted(style_counts.items())),
        "sensitive_answer_alias_exposure_count": dict(sorted(alias_counts.items())),
        "answer_token_id_exposure_count": dict(sorted(token_counts.items(), key=lambda pair: int(pair[0]))),
        "decoded_token_piece_exposure_count": dict(sorted(piece_counts.items())),
        "minimum_fact_exposure": minimum,
        "maximum_fact_exposure": maximum,
        "exposure_imbalance": maximum - minimum,
        "token_accounting_complete": tokenizer is not None,
    }
    if report["exposure_imbalance"] > 1:
        raise AssertionError("Balanced fact-cycle exposure imbalance exceeded one")
    return report


def plan_sha256(plan: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_bytes(list(plan))).hexdigest()
