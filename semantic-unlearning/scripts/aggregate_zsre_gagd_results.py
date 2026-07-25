#!/usr/bin/env python3
"""Aggregate per-seed ZsRE Setting 5e/active-repair results."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


METHOD_KEYS: Tuple[Tuple[str, str], ...] = (
    ("base", "Base"),
    ("setting5e", "Setting 5e"),
    ("active_candidate", "Setting 5e + active candidate"),
    ("selected", "Selected"),
)
METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("forget", "Eff", "forget_Eff_down"),
    ("forget", "Gen", "forget_Gen_down"),
    ("forget", "Spe", "forget_Spe_up"),
    ("retain", "Eff", "retain_Eff_up"),
    ("retain", "Gen", "retain_Gen_up"),
    ("retain", "Spe", "retain_Spe_up"),
    ("root", "PPL", "PPL_down"),
)


def discover_results(pattern: str) -> List[Path]:
    paths = [Path(value) for value in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"No ZsRE result files match {pattern!r}")
    return paths


def load_results(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    seen_seeds = set()
    protocol = None
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        if result.get("dataset") != "ZsRE":
            raise ValueError(f"{path} is not a ZsRE result")
        seed = int(result["seed"])
        if seed in seen_seeds:
            raise ValueError(f"Duplicate ZsRE seed {seed}")
        seen_seeds.add(seed)
        current = (
            int(result["forget_num"]),
            int(result["retain_num"]),
            str(result["zsre_sha256"]),
        )
        if protocol is None:
            protocol = current
        elif current != protocol:
            raise ValueError(
                "Cannot aggregate different ZsRE counts or dataset hashes: "
                f"{current} != {protocol}"
            )
        result["_path"] = str(path)
        results.append(result)
    return sorted(results, key=lambda item: int(item["seed"]))


def metric_value(result: Mapping[str, Any], method: str, split: str, key: str) -> float:
    block = result[method]
    if split == "root":
        value = block[key]
    else:
        value = block[split][key]
    if value is None:
        return float("nan")
    return float(value)


def mean_std(
    values: Sequence[float],
) -> Tuple[Optional[float], Optional[float]]:
    array = np.array(values, dtype=np.float64)
    if not np.isfinite(array).any():
        return None, None
    return float(np.nanmean(array)), float(np.nanstd(array))


def aggregate(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method_key, display in METHOD_KEYS:
        if not all(method_key in result for result in results):
            continue
        row: Dict[str, Any] = {
            "method": display,
            "seeds": len(results),
        }
        for split, metric, output_name in METRICS:
            mean, std = mean_std(
                [
                    metric_value(result, method_key, split, metric)
                    for result in results
                ]
            )
            row[f"{output_name}_mean"] = (
                None if mean is None else round(mean, 6)
            )
            row[f"{output_name}_std"] = (
                None if std is None else round(std, 6)
            )
        rows.append(row)
    if not rows:
        raise ValueError("ZsRE results share no aggregatable method blocks")
    return rows


def require_selected_targets(
    results: Sequence[Mapping[str, Any]],
    *,
    eff_max: float,
    gen_max: float,
) -> None:
    """Refuse to publish an aggregate containing a failed ZsRE seed."""

    failures: List[str] = []
    for result in results:
        selected = result.get("selected")
        accepted = bool(result.get("repair", {}).get("candidate_accepted"))
        if selected is None:
            failures.append(f"seed {result.get('seed')}: missing selected metrics")
            continue
        eff = float(selected["forget"]["Eff"])
        gen = float(selected["forget"]["Gen"])
        if not accepted or eff > eff_max or gen > gen_max:
            failures.append(
                f"seed {result.get('seed')}: accepted={accepted}, "
                f"Eff={eff:g}, Gen={gen:g}"
            )
    if failures:
        raise ValueError(
            "Refusing to aggregate ZsRE runs that missed the selected target "
            f"(Eff <= {eff_max:g}, Gen <= {gen_max:g}): "
            + "; ".join(failures)
        )


def format_value(mean: Any, std: Any) -> str:
    if mean is None or np.isnan(float(mean)):
        return "null"
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def write_outputs(
    output_dir: Path,
    results: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "aggregate.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "dataset": "ZsRE",
        "seeds": [int(result["seed"]) for result in results],
        "forget_num": int(results[0]["forget_num"]),
        "retain_num": int(results[0]["retain_num"]),
        "zsre_sha256": results[0]["zsre_sha256"],
        "candidate_acceptance_rate": float(
            np.mean(
                [
                    bool(result["repair"]["candidate_accepted"])
                    for result in results
                ]
            )
        ),
        "source_files": [
            result["_path"] for result in results if "_path" in result
        ],
        "rows": list(rows),
    }
    (output_dir / "aggregate.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    columns = [output_name for _, _, output_name in METRICS]
    lines = [
        "# ZsRE Setting 5e + active LM-head repair aggregate",
        "",
        (
            f"Seeds: {', '.join(str(seed) for seed in payload['seeds'])}. "
            f"Forget records: {payload['forget_num']}; retain records: "
            f"{payload['retain_num']}; active-candidate acceptance rate: "
            f"{payload['candidate_acceptance_rate']:.1%}."
        ),
        "",
        "Forget Eff/Gen are lower-is-better. Forget Spe and retain Eff/Gen/Spe are higher-is-better. PPL is lower-is-better.",
        "",
        "| method | " + " | ".join(columns) + " |",
        "| --- | " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [
            format_value(
                row[f"{column}_mean"],
                row[f"{column}_std"],
            )
            for column in columns
        ]
        lines.append(f"| {row['method']} | " + " | ".join(values) + " |")
    (output_dir / "aggregate.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern",
        default="outputs/zsre_setting5e_active/seed*/zsre_results.json",
    )
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        help=(
            "Explicit per-seed result path. Repeat for each seed. When supplied, "
            "--pattern is ignored so stale seed directories cannot enter the aggregate."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/zsre_setting5e_active/aggregate",
    )
    parser.add_argument("--require-selected-eff-max", type=float, default=0.0)
    parser.add_argument("--require-selected-gen-max", type=float, default=0.0)
    args = parser.parse_args()
    paths = (
        [Path(value) for value in args.result]
        if args.result
        else discover_results(args.pattern)
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing explicit ZsRE result files: "
            + ", ".join(str(path) for path in missing)
        )
    results = load_results(paths)
    require_selected_targets(
        results,
        eff_max=args.require_selected_eff_max,
        gen_max=args.require_selected_gen_max,
    )
    rows = aggregate(results)
    write_outputs(Path(args.output_dir), results, rows)
    print(f"Aggregate table: {Path(args.output_dir) / 'aggregate.md'}")


if __name__ == "__main__":
    main()
