#!/usr/bin/env python3
"""Build a fixed-order TOFU comparison from ``tofu_eval.py`` summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


METHOD_ORDER = (
    "Base",
    "Retain-only retraining oracle",
    "Full model, all answer tokens",
    "Full model, selective answer tokens",
    "Embedding + LM head, all answer tokens",
    "Embedding + LM head, selective answer tokens",
    "TOFU Setting 5e restoration",
    "Setting 5e + active forget repair",
    "Setting 5e + active + neighborhood repair",
)
METHOD_KEYS = {
    "Base": "base",
    "Retain-only retraining oracle": "retain_only_oracle",
    "Full model, all answer tokens": "full_all_tokens",
    "Full model, selective answer tokens": "full_selective_tokens",
    "Embedding + LM head, all answer tokens": "emb_lm_all_tokens",
    "Embedding + LM head, selective answer tokens": "emb_lm_selective_tokens",
    "TOFU Setting 5e restoration": "tofu_setting5e_restore",
    "Setting 5e + active forget repair": "tofu_active_forget_repair",
    "Setting 5e + active + neighborhood repair": (
        "gagd_neighborhood_confidence_tofu"
    ),
}
COLUMNS = (
    "Method",
    "Forget answer probability ↓",
    "Change from Base",
    "Meets forgetting target",
    "Forget ROUGE-L recall ↓",
    "Forget truth ratio",
    "Retain answer probability ↑",
    "Retain probability ratio vs Base ↑",
    "Meets retain target",
    "Meets joint target",
    "Retain ROUGE-L recall ↑",
    "Retain truth ratio ↑",
    "Real-authors normalized answer probability ↑",
    "World-facts normalized answer probability ↑",
    "Real-authors ROUGE-L recall ↑",
    "World-facts ROUGE-L recall ↑",
    "KS p-value",
    "Seed",
    "Forget split",
    "Retain split",
    "Source result path",
    "Same protocol verified",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        metavar="METHOD_KEY=SUMMARY_JSON",
        help="May be repeated; all nine method keys are required by default.",
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--max-forget-answer-probability",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--min-retain-probability-ratio",
        type=float,
        default=0.9998,
        help=(
            "Required candidate retain answer probability divided by the Base "
            "retain answer probability."
        ),
    )
    parser.add_argument(
        "--required-target-method",
        default="gagd_neighborhood_confidence_tofu",
        choices=sorted(set(METHOD_KEYS.values())),
    )
    parser.add_argument(
        "--require-target",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def parse_result_specs(specs: Sequence[str]) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}
    valid_keys = set(METHOD_KEYS.values())
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --result value: {spec!r}")
        key, raw_path = spec.split("=", 1)
        if key not in valid_keys:
            raise ValueError(
                f"Unknown method key {key!r}; expected one of {sorted(valid_keys)}"
            )
        if key in parsed:
            raise ValueError(f"Duplicate result for method key {key!r}")
        parsed[key] = Path(raw_path).resolve()
    return parsed


def _number(summary: Dict[str, Any], key: str) -> float:
    value = summary.get(key)
    if value is None:
        return float("nan")
    return float(value)


def row_from_summary(
    display_name: str,
    path: Path,
    summary: Dict[str, Any],
    max_forget_answer_probability: float = 2e-5,
) -> Dict[str, Any]:
    forget_answer_probability = _number(summary, "forget_answer_prob")
    return {
        "Method": display_name,
        "Forget answer probability ↓": forget_answer_probability,
        "Change from Base": float("nan"),
        "Meets forgetting target": bool(
            math.isfinite(forget_answer_probability)
            and forget_answer_probability <= max_forget_answer_probability
        ),
        "Forget ROUGE-L recall ↓": _number(summary, "forget_rougeL_recall"),
        "Forget truth ratio": _number(summary, "forget_truth_ratio"),
        "Retain answer probability ↑": _number(summary, "retain_answer_prob"),
        "Retain probability ratio vs Base ↑": float("nan"),
        "Meets retain target": False,
        "Meets joint target": False,
        "Retain ROUGE-L recall ↑": _number(summary, "retain_rougeL_recall"),
        "Retain truth ratio ↑": _number(summary, "retain_truth_ratio"),
        "Real-authors normalized answer probability ↑": _number(
            summary,
            "real_authors_normalized_answer_prob",
        ),
        "World-facts normalized answer probability ↑": _number(
            summary,
            "world_facts_normalized_answer_prob",
        ),
        "Real-authors ROUGE-L recall ↑": _number(
            summary,
            "tofu_real_authors_rougeL_recall",
        ),
        "World-facts ROUGE-L recall ↑": _number(
            summary,
            "tofu_world_facts_rougeL_recall",
        ),
        "KS p-value": _number(summary, "ks_test_p_value"),
        "Seed": int(summary["seed"]),
        "Forget split": str(summary["forget_split"]),
        "Retain split": str(summary["retain_split"]),
        "Source result path": str(path),
        "Same protocol verified": False,
    }


def verify_protocol(rows: Sequence[Dict[str, Any]]) -> Tuple[int, str, str]:
    if not rows:
        raise ValueError("No TOFU result rows were supplied")
    protocols = {
        (
            int(row["Seed"]),
            str(row["Forget split"]),
            str(row["Retain split"]),
        )
        for row in rows
    }
    if len(protocols) != 1:
        raise ValueError(
            "TOFU results do not share one seed/forget/retain protocol: "
            f"{sorted(protocols)}"
        )
    protocol = next(iter(protocols))
    for row in rows:
        row["Same protocol verified"] = True
    return protocol


def add_base_differences(
    rows: Sequence[Dict[str, Any]],
    min_retain_probability_ratio: float = 0.9998,
) -> None:
    base_rows = [row for row in rows if row["Method"] == "Base"]
    if len(base_rows) != 1:
        raise ValueError("Exactly one Base row is required")
    base_value = float(base_rows[0]["Forget answer probability ↓"])
    base_retain = float(base_rows[0]["Retain answer probability ↑"])
    if not math.isfinite(base_retain) or base_retain <= 0:
        raise ValueError("Base retain answer probability must be positive and finite")
    for row in rows:
        row["Change from Base"] = (
            float(row["Forget answer probability ↓"]) - base_value
        )
        retain_value = float(row["Retain answer probability ↑"])
        retain_ratio = (
            retain_value / base_retain
            if math.isfinite(retain_value)
            else float("nan")
        )
        retain_met = bool(
            math.isfinite(retain_ratio)
            and retain_ratio + 1e-12 >= min_retain_probability_ratio
        )
        row["Retain probability ratio vs Base ↑"] = retain_ratio
        row["Meets retain target"] = retain_met
        row["Meets joint target"] = bool(
            row["Meets forgetting target"] and retain_met
        )


def require_forgetting_target(
    rows: Sequence[Dict[str, Any]],
    *,
    required_display_name: str,
    max_forget_answer_probability: float,
) -> None:
    matches = [
        row for row in rows if row["Method"] == required_display_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Required target method {required_display_name!r} is missing"
        )
    value = float(matches[0]["Forget answer probability ↓"])
    if not math.isfinite(value) or value > max_forget_answer_probability:
        raise RuntimeError(
            f"{required_display_name} has forget answer probability {value}, "
            f"above the hard target {max_forget_answer_probability}. "
            "The comparison is diagnostic and must not be selected as the "
            "final repair candidate."
        )


def require_joint_target(
    rows: Sequence[Dict[str, Any]],
    *,
    required_display_name: str,
    max_forget_answer_probability: float,
    min_retain_probability_ratio: float,
) -> None:
    require_forgetting_target(
        rows,
        required_display_name=required_display_name,
        max_forget_answer_probability=max_forget_answer_probability,
    )
    matches = [row for row in rows if row["Method"] == required_display_name]
    if len(matches) != 1:
        raise ValueError(
            f"Required target method {required_display_name!r} is missing"
        )
    retain_ratio = float(
        matches[0]["Retain probability ratio vs Base ↑"]
    )
    if (
        not math.isfinite(retain_ratio)
        or retain_ratio + 1e-12 < min_retain_probability_ratio
    ):
        raise RuntimeError(
            f"{required_display_name} has retain/Base probability ratio "
            f"{retain_ratio}, below the hard target "
            f"{min_retain_probability_ratio}. The candidate is diagnostic and "
            "must not be selected."
        )


def _format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return "—" if math.isnan(value) else f"{value:.6f}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def write_outputs(
    output_dir: Path,
    rows: Sequence[Dict[str, Any]],
    protocol: Tuple[int, str, str],
    max_forget_answer_probability: float,
    min_retain_probability_ratio: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison_tofu.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "comparison_tofu.md"
    display_columns = COLUMNS[:17]
    lines = [
        "# TOFU GA/GD comparison",
        "",
        (
            f"Protocol: seed {protocol[0]}, `{protocol[1]}` / "
            f"`{protocol[2]}`. TOFU does not define MCF `Spe`; utility is "
            "reported separately on retain, real-authors, and world-facts. "
            "The hard forget-answer probability target is "
            f"{max_forget_answer_probability:.8g}; the hard retain/Base "
            "probability-ratio target is "
            f"{min_retain_probability_ratio:.8g}."
        ),
        "",
        "| " + " | ".join(display_columns) + " |",
        "| " + " | ".join("---" for _ in display_columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _format_markdown_value(row[column])
                for column in display_columns
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_path = output_dir / "comparison_tofu.json"
    json_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "seed": protocol[0],
                    "forget_split": protocol[1],
                    "retain_split": protocol[2],
                },
                "max_forget_answer_probability": (
                    max_forget_answer_probability
                ),
                "min_retain_probability_ratio": (
                    min_retain_probability_ratio
                ),
                "metric_note": (
                    "Forget answer probability and forget ROUGE-L are lower-is-"
                    "better. Utility columns are higher-is-better. Forget truth "
                    "ratio should be interpreted with its KS p-value rather than "
                    "as a standalone monotonic score."
                ),
                "rows": list(rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.max_forget_answer_probability < 1.0:
        raise ValueError("--max-forget-answer-probability must lie in (0,1)")
    if not 0.0 < args.min_retain_probability_ratio <= 1.0:
        raise ValueError("--min-retain-probability-ratio must lie in (0,1]")
    result_paths = parse_result_specs(args.result)
    missing = [
        key
        for key in METHOD_KEYS.values()
        if key not in result_paths
    ]
    if missing and not args.allow_partial:
        raise ValueError(f"Missing required TOFU result summaries: {missing}")

    rows: List[Dict[str, Any]] = []
    for display_name in METHOD_ORDER:
        key = METHOD_KEYS[display_name]
        if key not in result_paths:
            continue
        path = result_paths[key]
        if not path.exists():
            raise FileNotFoundError(f"TOFU result does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        rows.append(
            row_from_summary(
                display_name,
                path,
                summary,
                args.max_forget_answer_probability,
            )
        )
    protocol = verify_protocol(rows)
    add_base_differences(rows, args.min_retain_probability_ratio)
    output_dir = Path(args.output_dir).resolve()
    write_outputs(
        output_dir,
        rows,
        protocol,
        args.max_forget_answer_probability,
        args.min_retain_probability_ratio,
    )
    if args.require_target:
        display_by_key = {
            key: display_name
            for display_name, key in METHOD_KEYS.items()
        }
        require_joint_target(
            rows,
            required_display_name=display_by_key[args.required_target_method],
            max_forget_answer_probability=(
                args.max_forget_answer_probability
            ),
            min_retain_probability_ratio=(
                args.min_retain_probability_ratio
            ),
        )
    print(f"TOFU comparison written to {output_dir}")


if __name__ == "__main__":
    main()
