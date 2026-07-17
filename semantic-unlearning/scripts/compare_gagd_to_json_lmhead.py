#!/usr/bin/env python3
"""Aggregate GA/GD settings and JSON-LMHead-Zero MCF evaluations.

The script consumes official-compatible JSON results rather than model weights.
It validates that every row uses the same seed, sample mode, forget count, and
retain count before producing per-seed and mean +/- population-std tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_METHODS = [
    "full_all_tokens",
    "full_selective_tokens",
    "emb_lm_all_tokens",
    "emb_lm_selective_tokens",
    "emb_lm_all_restore_post_training_true",
]
METRICS = ["Eff", "Gen", "Spe", "PPL"]


def parse_seed_list(values: Sequence[str]) -> List[int]:
    seeds: List[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                seeds.append(int(part))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return list(dict.fromkeys(seeds))


def format_pattern(pattern: str, seed: int, method: Optional[str] = None) -> Path:
    fields = {"seed": seed, "method": method or ""}
    try:
        return Path(pattern.format(**fields))
    except KeyError as exc:
        raise ValueError(f"Unknown placeholder {exc} in path pattern: {pattern}") from exc


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Official result must be a JSON object: {path}")
    return data


def metric_value(data: Dict[str, Any], metric: str) -> Optional[float]:
    if metric in data and data[metric] is not None:
        return float(data[metric])
    forget = data.get("forget")
    if isinstance(forget, dict) and metric in forget and forget[metric] is not None:
        return float(forget[metric])
    summary = data.get("summary")
    if isinstance(summary, dict) and metric in summary and summary[metric] is not None:
        return float(summary[metric])
    if metric == "PPL" and data.get("forget_PPL") is not None:
        return float(data["forget_PPL"])
    return None


def validate_result(
    data: Dict[str, Any],
    path: Path,
    expected_seed: int,
    sample_mode: str,
    unlearn_num: int,
    retain_num: int,
) -> None:
    expected = {
        "seed": expected_seed,
        "sample_mode": sample_mode,
        "unlearn_num": unlearn_num,
        "retain_num": retain_num,
    }
    for key, expected_value in expected.items():
        actual = data.get(key)
        if actual != expected_value:
            raise ValueError(
                f"Incomparable result {path}: {key}={actual!r}, expected {expected_value!r}."
            )
    dataset = str(data.get("dataset", "MCF")).upper()
    if dataset != "MCF":
        raise ValueError(f"Incomparable result {path}: dataset={dataset!r}, expected 'MCF'.")


def result_row(
    method: str,
    family: str,
    seed: int,
    path: Path,
    sample_mode: str,
    unlearn_num: int,
    retain_num: int,
) -> Dict[str, Any]:
    data = read_json(path)
    validate_result(data, path, seed, sample_mode, unlearn_num, retain_num)
    row: Dict[str, Any] = {
        "method": method,
        "family": family,
        "seed": seed,
        "source": str(path),
    }
    for metric in METRICS:
        row[metric] = metric_value(data, metric)
    missing = [metric for metric in METRICS if row[metric] is None]
    if missing:
        raise ValueError(f"Result {path} is missing required metrics: {missing}")
    return row


def collect_rows(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for seed in args.seeds:
        specs = [
            ("base", "reference", format_pattern(args.base_pattern, seed)),
            (
                args.baseline_name,
                "json_lmhead",
                format_pattern(args.baseline_pattern, seed),
            ),
        ]
        specs.extend(
            (method, "gagd", format_pattern(args.gagd_pattern, seed, method))
            for method in args.methods
        )
        for method, family, path in specs:
            if not path.exists():
                missing.append(str(path))
                continue
            rows.append(
                result_row(
                    method=method,
                    family=family,
                    seed=seed,
                    path=path,
                    sample_mode=args.sample_mode,
                    unlearn_num=args.unlearn_num,
                    retain_num=args.retain_num,
                )
            )
    if missing and not args.allow_missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(f"Missing comparison results:\n  - {formatted}")
    return rows, missing


def aggregate_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), str(row["family"])), []).append(row)

    aggregate: List[Dict[str, Any]] = []
    for (method, family), method_rows in grouped.items():
        out: Dict[str, Any] = {
            "method": method,
            "family": family,
            "n_seeds": len(method_rows),
            "seeds": ",".join(str(row["seed"]) for row in sorted(method_rows, key=lambda x: x["seed"])),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in method_rows if row.get(metric) is not None]
            out[f"{metric}_mean"] = statistics.fmean(values) if values else None
            out[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None)
            out[f"{metric}_n"] = len(values)
        aggregate.append(out)

    family_order = {"reference": 0, "json_lmhead": 1, "gagd": 2}
    method_order = {method: i for i, method in enumerate(DEFAULT_METHODS)}
    return sorted(
        aggregate,
        key=lambda row: (
            family_order.get(str(row["family"]), 99),
            method_order.get(str(row["method"]), -1),
            str(row["method"]),
        ),
    )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def display(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_per_seed_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Per-seed GA/GD vs JSON-LMHead-Zero comparison",
        "",
        "All rows use the same official MCF split for their seed. Eff/Gen are lower-is-better; Spe is higher-is-better; PPL should remain low/stable.",
        "",
        "| Method | Family | Seed | Eff ↓ | Gen ↓ | Spe ↑ | PPL ↓ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    family_order = {"reference": 0, "json_lmhead": 1, "gagd": 2}
    method_order = {method: i for i, method in enumerate(DEFAULT_METHODS)}
    for row in sorted(
        rows,
        key=lambda x: (
            int(x["seed"]),
            family_order.get(str(x["family"]), 99),
            method_order.get(str(x["method"]), -1),
            str(x["method"]),
        ),
    ):
        lines.append(
            "| {method} | {family} | {seed} | {Eff} | {Gen} | {Spe} | {PPL} |".format(
                **{key: display(value) for key, value in row.items()}
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aggregate_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Aggregate GA/GD vs JSON-LMHead-Zero comparison",
        "",
        "Values are mean ± population standard deviation across available seeds.",
        "",
        "| Method | Family | Seeds | Eff ↓ | Gen ↓ | Spe ↑ | PPL ↓ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        cells = []
        for metric in METRICS:
            mean = row.get(f"{metric}_mean")
            std = row.get(f"{metric}_std")
            cells.append("NA" if mean is None else f"{mean:.3f} ± {std:.3f}")
        lines.append(
            f"| {row['method']} | {row['family']} | {row['n_seeds']} | "
            + " | ".join(cells)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", default=["0", "1", "2", "3", "4"])
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument(
        "--gagd-pattern",
        default="outputs/gagd_vs_json_lmhead/seed{seed}/official_eval/{method}_official_eval.json",
        help="Path pattern with {seed} and {method} placeholders.",
    )
    p.add_argument(
        "--baseline-pattern",
        default="outputs/official_eval_lmhead_zero_true_restore150_seed{seed}_spefix.json",
        help="JSON-LMHead baseline path pattern with a {seed} placeholder.",
    )
    p.add_argument(
        "--base-pattern",
        default="outputs/official_eval_base_seed{seed}_spefix.json",
        help="Base-model result path pattern with a {seed} placeholder.",
    )
    p.add_argument("--baseline-name", default="json_lmhead_zero_true_restore150")
    p.add_argument("--output-dir", default="outputs/gagd_vs_json_lmhead/comparison")
    p.add_argument("--sample-mode", default="official")
    p.add_argument("--unlearn-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--allow-missing", action="store_true")
    args = p.parse_args()
    args.seeds = parse_seed_list(args.seeds)
    return args


def main() -> None:
    args = parse_args()
    rows, missing = collect_rows(args)
    if not rows:
        raise RuntimeError("No comparison result files were found.")
    aggregate = aggregate_rows(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "per_seed.csv", rows)
    write_csv(out_dir / "aggregate.csv", aggregate)
    write_per_seed_md(out_dir / "per_seed.md", rows)
    write_aggregate_md(out_dir / "aggregate.md", aggregate)
    with (out_dir / "comparison.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "seeds": args.seeds,
                    "methods": args.methods,
                    "sample_mode": args.sample_mode,
                    "unlearn_num": args.unlearn_num,
                    "retain_num": args.retain_num,
                    "gagd_pattern": args.gagd_pattern,
                    "baseline_pattern": args.baseline_pattern,
                    "base_pattern": args.base_pattern,
                },
                "missing": missing,
                "per_seed": rows,
                "aggregate": aggregate,
            },
            f,
            indent=2,
        )
    print(f"Wrote comparison tables to {out_dir}")
    if missing:
        print(f"Skipped {len(missing)} missing result files (--allow-missing).")


if __name__ == "__main__":
    main()
