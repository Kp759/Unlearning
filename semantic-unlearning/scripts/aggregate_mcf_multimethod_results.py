#!/usr/bin/env python3
"""Aggregate the reviewed eight-method MCF experiment across seeds.

The script consumes official-evaluator JSON files only. It validates that every
requested seed has exactly one result per method and writes per-seed plus
mean/population-standard-deviation tables for Eff, Gen, Spe, and PPL.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

METRICS = ("Eff", "Gen", "Spe", "PPL")


@dataclass(frozen=True)
class MethodSpec:
    key: str
    display_name: str
    pattern_kind: str
    pattern: str


DEFAULT_METHODS = (
    MethodSpec("base", "Base", "base", ""),
    MethodSpec("original_zerounlearn", "Original ZeroUnlearn", "zero", ""),
    MethodSpec("full_all_tokens", "Full all tokens", "gagd", ""),
    MethodSpec("full_selective_tokens", "Full selective tokens", "gagd", ""),
    MethodSpec("emb_lm_all_tokens", "Emb/LM all tokens(s3)", "gagd", ""),
    MethodSpec("emb_lm_selective_tokens", "Emb/LM selective tokens", "gagd", ""),
    MethodSpec(
        "emb_lm_all_restore_post_training_true",
        "Setting 5e (used setting 3)",
        "gagd",
        "",
    ),
    MethodSpec(
        "protected_lm_head_repair",
        "5e + protected LM-head repair",
        "repair",
        "",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument(
        "--base-pattern",
        required=True,
        help="Path template containing {seed} for Base official JSON.",
    )
    parser.add_argument(
        "--zero-pattern",
        required=True,
        help="Path template containing {seed} for Original ZeroUnlearn JSON.",
    )
    parser.add_argument(
        "--gagd-pattern",
        required=True,
        help="Path template containing {seed} and {method} for GA/GD official JSON.",
    )
    parser.add_argument(
        "--repair-pattern",
        required=True,
        help="Path template containing {seed} for protected-repair official JSON.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def _finite_float(value: Any, *, field: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {field} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path}: {field} is not finite: {result!r}")
    return result


def extract_metrics(payload: Mapping[str, Any], path: Path) -> Dict[str, float]:
    forget = payload.get("forget", payload)
    if not isinstance(forget, Mapping):
        raise ValueError(f"{path}: missing official forget metric block")
    ppl = payload.get("forget_PPL", payload.get("PPL", forget.get("PPL")))
    return {
        "Eff": _finite_float(forget.get("Eff"), field="Eff", path=path),
        "Gen": _finite_float(forget.get("Gen"), field="Gen", path=path),
        "Spe": _finite_float(forget.get("Spe"), field="Spe", path=path),
        "PPL": _finite_float(ppl, field="PPL", path=path),
    }


def read_result(path: Path, *, expected_seed: int) -> Dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing result for seed {expected_seed}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    stored_seed = payload.get("seed")
    if stored_seed is not None and int(stored_seed) != expected_seed:
        raise ValueError(
            f"{path}: stored seed {stored_seed!r} does not match {expected_seed}"
        )
    dataset = payload.get("dataset")
    if dataset is not None and str(dataset).upper() not in {"MCF", "MULTI_COUNTERFACT"}:
        raise ValueError(f"{path}: expected MCF result, found dataset={dataset!r}")
    return extract_metrics(payload, path)


def result_path(
    spec: MethodSpec,
    seed: int,
    *,
    base_pattern: str,
    zero_pattern: str,
    gagd_pattern: str,
    repair_pattern: str,
) -> Path:
    if spec.pattern_kind == "base":
        template = base_pattern
    elif spec.pattern_kind == "zero":
        template = zero_pattern
    elif spec.pattern_kind == "gagd":
        template = gagd_pattern
    elif spec.pattern_kind == "repair":
        template = repair_pattern
    else:  # pragma: no cover - guarded by static method definitions
        raise ValueError(f"Unknown pattern kind: {spec.pattern_kind}")
    return Path(template.format(seed=seed, method=spec.key)).expanduser()


def collect_rows(
    seeds: Sequence[int],
    *,
    base_pattern: str,
    zero_pattern: str,
    gagd_pattern: str,
    repair_pattern: str,
    methods: Sequence[MethodSpec] = DEFAULT_METHODS,
) -> List[Dict[str, Any]]:
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds are not allowed: {list(seeds)}")

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for spec in methods:
            path = result_path(
                spec,
                seed,
                base_pattern=base_pattern,
                zero_pattern=zero_pattern,
                gagd_pattern=gagd_pattern,
                repair_pattern=repair_pattern,
            )
            metrics = read_result(path, expected_seed=seed)
            rows.append(
                {
                    "method_key": spec.key,
                    "method": spec.display_name,
                    "seed": seed,
                    **metrics,
                    "source": str(path),
                }
            )
    return rows


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[MethodSpec] = DEFAULT_METHODS,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method_key"]), []).append(row)

    output: List[Dict[str, Any]] = []
    for spec in methods:
        method_rows = sorted(grouped.get(spec.key, []), key=lambda row: int(row["seed"]))
        if not method_rows:
            raise ValueError(f"No rows collected for method {spec.display_name}")
        row: Dict[str, Any] = {
            "method_key": spec.key,
            "method": spec.display_name,
            "n_seeds": len(method_rows),
            "seeds": ",".join(str(item["seed"]) for item in method_rows),
        }
        for metric in METRICS:
            values = [float(item[metric]) for item in method_rows]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )
        output.append(row)
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# MCF eight-method aggregate",
        "",
        "Values are mean ± population standard deviation across validated seeds.",
        "",
        "| Method | Eff ↓ | Gen ↓ | Spe ↑ | PPL ↓ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        cells = [
            f"{float(row[f'{metric}_mean']):.3f} ± {float(row[f'{metric}_std']):.3f}"
            for metric in METRICS
        ]
        lines.append(f"| {row['method']} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    aggregate: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_seed.csv", rows)
    write_csv(output_dir / "aggregate.csv", aggregate)
    write_markdown(output_dir / "aggregate.md", aggregate)
    (output_dir / "aggregate.json").write_text(
        json.dumps(
            {
                "dataset": "MCF",
                "seeds": list(seeds),
                "standard_deviation": "population",
                "metrics": {
                    "Eff": "lower_is_better",
                    "Gen": "lower_is_better",
                    "Spe": "higher_is_better",
                    "PPL": "lower_is_better",
                },
                "rows": list(aggregate),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    rows = collect_rows(
        args.seeds,
        base_pattern=args.base_pattern,
        zero_pattern=args.zero_pattern,
        gagd_pattern=args.gagd_pattern,
        repair_pattern=args.repair_pattern,
    )
    aggregate = aggregate_rows(rows)
    output_dir = Path(args.output_dir).expanduser()
    write_outputs(output_dir, rows, aggregate, args.seeds)
    print(f"MCF aggregate: {output_dir / 'aggregate.md'}")


if __name__ == "__main__":
    main()
