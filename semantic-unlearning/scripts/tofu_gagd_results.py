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
    "Full model, all answer tokens",
    "Full model, selective answer tokens",
    "Embedding + LM head, all answer tokens",
    "Embedding + LM head, selective answer tokens",
    "GA/GD + neighborhood-confidence repair",
)
METHOD_KEYS = {
    "Base": "base",
    "Full model, all answer tokens": "full_all_tokens",
    "Full model, selective answer tokens": "full_selective_tokens",
    "Embedding + LM head, all answer tokens": "emb_lm_all_tokens",
    "Embedding + LM head, selective answer tokens": "emb_lm_selective_tokens",
    "GA/GD + neighborhood-confidence repair": "gagd_neighborhood_confidence_tofu",
}
COLUMNS = (
    "Method",
    "Forget answer probability ↓",
    "Forget ROUGE-L recall ↓",
    "Forget truth ratio",
    "Retain answer probability ↑",
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
        help="May be repeated; all six method keys are required by default.",
    )
    parser.add_argument("--allow-partial", action="store_true")
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
) -> Dict[str, Any]:
    return {
        "Method": display_name,
        "Forget answer probability ↓": _number(summary, "forget_answer_prob"),
        "Forget ROUGE-L recall ↓": _number(summary, "forget_rougeL_recall"),
        "Forget truth ratio": _number(summary, "forget_truth_ratio"),
        "Retain answer probability ↑": _number(summary, "retain_answer_prob"),
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
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison_tofu.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "comparison_tofu.md"
    display_columns = COLUMNS[:12]
    lines = [
        "# TOFU GA/GD comparison",
        "",
        (
            f"Protocol: seed {protocol[0]}, `{protocol[1]}` / "
            f"`{protocol[2]}`. TOFU does not define MCF `Spe`; utility is "
            "reported separately on retain, real-authors, and world-facts."
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
        rows.append(row_from_summary(display_name, path, summary))
    protocol = verify_protocol(rows)
    output_dir = Path(args.output_dir).resolve()
    write_outputs(output_dir, rows, protocol)
    print(f"TOFU comparison written to {output_dir}")


if __name__ == "__main__":
    main()
