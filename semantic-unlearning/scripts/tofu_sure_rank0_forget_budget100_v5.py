#!/usr/bin/env python3
"""SURE-TOFU v5: exact 100-row sparse rank-0 forgetting.

This is a thin, fail-closed wrapper around the validated progressive v3
Stage1B implementation.  V5 keeps the useful v3 row policy (Stage1A values on
sensitive input/output rows, exact Full-TOFU Base on non-sensitive rows) but
changes the initial selection to an EXACT global budget of 100 unique visible
answer-token rows.  Rows are allocated round-robin across the currently
violating training-visible forget QAs using the v3 rare/content-first ranking.

V5 is intentionally a fixed-budget ablation: if those 100 rows cannot satisfy
all direct-forget constraints within the configured rank-0 optimization budget,
the run fails closed rather than promoting additional rows.

The runner for this ablation sets target_nll_buffer=0 so the declared
answer-probability threshold itself is the forgetting boundary.

No retain95, paraphrases, same-author heldout, real-authors, world-facts, PPL,
or final evaluation metric is used for selection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import tofu_sure_rank0_forget_progressive_v3 as v3


METHOD = "SURE-TOFU-exact100-sparse-rank0-v5"
DEFAULT_BUDGET = 100


def pop_budget_arg(argv: List[str]) -> int:
    budget = DEFAULT_BUDGET
    flag = "--initial-unique-row-budget"
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 >= len(argv):
            raise ValueError(f"{flag} requires an integer value")
        budget = int(argv[idx + 1])
        del argv[idx : idx + 2]
    if budget <= 0:
        raise ValueError("initial-unique-row-budget must be positive")
    return budget


def round_robin_unique_budget(
    sensitive: set[int],
    rankings: Sequence[Sequence[int]],
    positions: Sequence[int],
    budget: int,
) -> Dict[int, List[int]]:
    """Fill sensitive to exactly ``budget`` unique rows fairly across QAs."""
    if sensitive:
        raise ValueError("v5 exact-budget selector expects an empty initial sensitive set")
    ordered_positions = [int(p) for p in positions]
    if not ordered_positions:
        return {}

    promoted: Dict[int, List[int]] = {p: [] for p in ordered_positions}
    cursor = {p: 0 for p in ordered_positions}

    while len(sensitive) < budget:
        made_progress = False
        for position in ordered_positions:
            ranking = rankings[position]
            i = cursor[position]
            while i < len(ranking) and int(ranking[i]) in sensitive:
                i += 1
            cursor[position] = i
            if i >= len(ranking):
                continue

            token_id = int(ranking[i])
            cursor[position] = i + 1
            if token_id in sensitive:
                continue
            sensitive.add(token_id)
            promoted[position].append(token_id)
            made_progress = True
            if len(sensitive) == budget:
                break

        if not made_progress:
            break

    if len(sensitive) != budget:
        raise RuntimeError(
            f"v5 could select only {len(sensitive)} unique rows, requested exact budget {budget}"
        )
    return {p: rows for p, rows in promoted.items() if rows}


def find_output_dir(argv: Sequence[str]) -> Path:
    flag = "--output-dir"
    if flag not in argv:
        raise ValueError("v5 requires --output-dir")
    idx = list(argv).index(flag)
    if idx + 1 >= len(argv):
        raise ValueError("--output-dir requires a value")
    return Path(argv[idx + 1])


def patch_reports(root: Path, budget: int) -> None:
    summary_path = root / "repair_summary.json"
    config_path = root / "config_used.json"
    ranking_path = root / "progressive_row_ranking.json"
    restoration_path = root / "answer_row_restoration.json"

    if summary_path.is_file():
        x = json.loads(summary_path.read_text(encoding="utf-8"))
        x["method"] = METHOD
        x["initial_unique_row_budget"] = budget
        x["fixed_sensitive_row_budget"] = True
        x["promotion_beyond_budget_allowed"] = False
        if int(x.get("initial_sensitive_answer_row_count", -1)) != budget:
            raise RuntimeError("v5 report does not contain the exact requested initial row budget")
        if int(x.get("sensitive_answer_row_count", -1)) != budget:
            raise RuntimeError("v5 final sensitive row count changed beyond the fixed budget")
        summary_path.write_text(json.dumps(x, indent=2) + "\n", encoding="utf-8")

    if config_path.is_file():
        x = json.loads(config_path.read_text(encoding="utf-8"))
        x["schema_version"] = 5
        x["method"] = METHOD
        x["initial_unique_row_budget"] = budget
        x["fixed_sensitive_row_budget"] = True
        x["promotion_beyond_budget_allowed"] = False
        x["sensitivity_rule"] = (
            "exact global unique-row budget; round-robin across violating visible QAs "
            "using rare/content-first per-QA ranking; fail closed if budget infeasible"
        )
        config_path.write_text(json.dumps(x, indent=2) + "\n", encoding="utf-8")

    if ranking_path.is_file():
        x = json.loads(ranking_path.read_text(encoding="utf-8"))
        x["initial_unique_row_budget"] = budget
        x["fixed_sensitive_row_budget"] = True
        ranking_path.write_text(json.dumps(x, indent=2) + "\n", encoding="utf-8")

    if restoration_path.is_file():
        x = json.loads(restoration_path.read_text(encoding="utf-8"))
        x["policy"] = (
            "v5 exact-100 sparse sensitive rows; sensitive input/output rows keep Stage1A "
            "baseline before rank0; every non-sensitive visible answer row is exact Full-TOFU Base"
        )
        x["initial_unique_row_budget"] = budget
        x["promotion_beyond_budget_allowed"] = False
        restoration_path.write_text(json.dumps(x, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    argv = sys.argv
    budget = pop_budget_arg(argv)
    output_dir = find_output_dir(argv)

    original_add_next_rows = v3.add_next_rows
    call_count = 0

    def fixed_budget_add_next_rows(
        sensitive: set[int],
        rankings: Sequence[Sequence[int]],
        positions: Sequence[int],
        count_per_example: int,
    ) -> Dict[int, List[int]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return round_robin_unique_budget(sensitive, rankings, positions, budget)
        # Exact-budget experiment: never allow promotion beyond the initial 100.
        return {}

    v3.add_next_rows = fixed_budget_add_next_rows
    v3.METHOD = METHOD
    try:
        v3.main()
    finally:
        v3.add_next_rows = original_add_next_rows

    patch_reports(output_dir, budget)
    print(
        f"SURE-TOFU-v5 exact-budget PASS: sensitive_rows={budget}; "
        "promotion_beyond_budget_allowed=False"
    )


if __name__ == "__main__":
    main()
