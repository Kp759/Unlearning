#!/usr/bin/env python3
"""Run semantic TOFU repair without projecting away abstention directions."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import torch

import tofu_gagd_neighborhood_confidence as tofu
import tofu_gagd_semantic_forget_repair as semantic


ORIGINAL_LOAD = semantic.load_tofu_repair_instances
ORIGINAL_BASIS = tofu.limited_hidden_row_basis
NORMAL_UTILITY_COUNT = 0
ABSTENTION_COUNT = 0


def load_instances(
    args: Any,
    tok: Any,
) -> Tuple[
    List[tofu.TOFUAnswerInstance],
    List[tofu.TOFUAnswerInstance],
    List[tofu.TOFUAnswerInstance],
]:
    global NORMAL_UTILITY_COUNT, ABSTENTION_COUNT
    forget, full_retain, utility = ORIGINAL_LOAD(args, tok)
    NORMAL_UTILITY_COUNT = sum(
        instance.split != "abstention" for instance in utility
    )
    ABSTENTION_COUNT = len(utility) - NORMAL_UTILITY_COUNT
    return forget, full_retain, utility


def limited_hidden_row_basis(
    caches: Sequence[tofu.TOFUAnswerDeltaCache],
    *,
    max_rank: int,
    max_rows: int,
) -> torch.Tensor:
    selected = list(caches)
    if (
        ABSTENTION_COUNT > 0
        and len(selected) == NORMAL_UTILITY_COUNT + ABSTENTION_COUNT
    ):
        selected = selected[:NORMAL_UTILITY_COUNT]
    return ORIGINAL_BASIS(
        selected,
        max_rank=max_rank,
        max_rows=max_rows,
    )


def main() -> None:
    semantic.load_tofu_repair_instances = load_instances
    tofu.limited_hidden_row_basis = limited_hidden_row_basis
    semantic.main()


if __name__ == "__main__":
    main()
