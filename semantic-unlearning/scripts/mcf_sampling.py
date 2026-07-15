"""Dependency-free sampling helpers shared by MCF training and evaluation."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence, Tuple


Record = Dict[str, Any]


def sample_official_mcf_records(
    data: Sequence[Record],
    forget_num: int,
    retain_num: int,
    seed: int,
    *,
    strict: bool = False,
) -> Tuple[List[Record], List[Record]]:
    """Match ZeroUnlearn: forget from half two, retain from half one."""
    half = len(data) // 2
    retain_pool = list(data[:half])
    forget_pool = list(data[half:])
    if strict and (len(forget_pool) < forget_num or len(retain_pool) < retain_num):
        raise ValueError(
            "Official MCF split does not contain enough records: "
            f"forget_pool={len(forget_pool)}, retain_pool={len(retain_pool)}, "
            f"requested forget={forget_num}, retain={retain_num}."
        )

    forget_num = min(len(forget_pool), forget_num)
    retain_num = min(len(retain_pool), retain_num)
    rng = random.Random(seed)
    forget_records = rng.sample(forget_pool, k=forget_num)
    retain_records = rng.sample(retain_pool, k=retain_num)
    return forget_records, retain_records


def sample_first_mcf_records(
    data: Sequence[Record],
    forget_num: int,
    retain_num: int,
    *,
    strict: bool = False,
) -> Tuple[List[Record], List[Record]]:
    """Return the legacy/debug contiguous split."""
    need = forget_num + retain_num
    if strict and len(data) < need:
        raise ValueError(f"MCF has only {len(data)} records, need {need} for first mode.")
    return list(data[:forget_num]), list(data[forget_num:need])
