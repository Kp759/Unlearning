#!/usr/bin/env python3
"""Compare locked MCF fixed-SVD repair against sparse-LoRA repair seed-by-seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


METRICS = ("Eff", "Gen", "Spe", "Spe_success")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fixed-root",
        default="outputs/mcf_zerounlearn_forget_only_locked_3b",
    )
    p.add_argument(
        "--lora-root",
        default="outputs/mcf_zerounlearn_forget_only_locked_3b_lora_r1",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument(
        "--out-prefix",
        default="outputs/mcf_locked_fixed_vs_lora_r1",
    )
    return p.parse_args()


def mean_sd(values: List[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "population_sd": float(a.std(ddof=0)),
        "sample_sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
    }


def read_eval(root: Path, seed: int) -> Dict[str, Any]:
    path = root / f"seed{seed}" / "official_eval_locked.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_repair_summary(root: Path, seed: int, lora: bool) -> Dict[str, Any]:
    dirname = "repair_forget_only_lora" if lora else "repair_forget_only"
    path = root / f"seed{seed}" / dirname / "repair_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    fixed_root = Path(args.fixed_root)
    lora_root = Path(args.lora_root)

    per_seed: List[Dict[str, Any]] = []
    aggregate_values: Dict[str, Dict[str, List[float]]] = {
        "fixed": {key: [] for key in (*METRICS, "PPL")},
        "lora": {key: [] for key in (*METRICS, "PPL")},
        "paired_delta_lora_minus_fixed": {key: [] for key in (*METRICS, "PPL")},
    }

    for seed in args.seeds:
        fixed = read_eval(fixed_root, seed)
        lora = read_eval(lora_root, seed)
        fixed_summary = read_repair_summary(fixed_root, seed, lora=False)
        lora_summary = read_repair_summary(lora_root, seed, lora=True)

        row: Dict[str, Any] = {
            "seed": seed,
            "fixed_repair": {
                "selected_rows": fixed_summary.get("selected_lm_head_rows"),
                "rank_requested": fixed_summary.get("repair_rank_requested"),
                "rank_actual": fixed_summary.get("repair_rank_actual"),
                "delta_norm": fixed_summary.get("selected_lm_head_delta_norm"),
            },
            "lora_repair": {
                "selected_rows": lora_summary.get("selected_lm_head_rows"),
                "rank_requested": lora_summary.get("lora_rank_requested"),
                "rank_effective_upper_bound": lora_summary.get("lora_rank_effective_upper_bound"),
                "stage2_trainable_parameters": lora_summary.get("stage2_trainable_parameters"),
                "delta_norm": lora_summary.get("selected_lm_head_delta_norm"),
                "shapes": lora_summary.get("lora_shapes"),
            },
            "metrics": {},
        }

        for metric in METRICS:
            fv = float(fixed["forget"][metric])
            lv = float(lora["forget"][metric])
            aggregate_values["fixed"][metric].append(fv)
            aggregate_values["lora"][metric].append(lv)
            aggregate_values["paired_delta_lora_minus_fixed"][metric].append(lv - fv)
            row["metrics"][metric] = {
                "fixed": fv,
                "lora": lv,
                "lora_minus_fixed": lv - fv,
            }

        fppl = float(fixed["forget_PPL"])
        lppl = float(lora["forget_PPL"])
        aggregate_values["fixed"]["PPL"].append(fppl)
        aggregate_values["lora"]["PPL"].append(lppl)
        aggregate_values["paired_delta_lora_minus_fixed"]["PPL"].append(lppl - fppl)
        row["metrics"]["PPL"] = {
            "fixed": fppl,
            "lora": lppl,
            "lora_minus_fixed": lppl - fppl,
        }
        per_seed.append(row)

    aggregate = {
        group: {metric: mean_sd(values) for metric, values in metrics.items()}
        for group, metrics in aggregate_values.items()
    }
    payload = {
        "schema_version": 1,
        "comparison": "locked MCF fixed-SVD active repair vs selected-row sparse LoRA",
        "fixed_root": str(fixed_root),
        "lora_root": str(lora_root),
        "seeds": args.seeds,
        "metric_directions": {
            "Eff": "lower better",
            "Gen": "lower better",
            "Spe": "higher better",
            "Spe_success": "higher better",
            "PPL": "lower/stable better",
        },
        "aggregate": aggregate,
        "per_seed": per_seed,
    }

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Locked MCF: fixed-SVD repair vs sparse-LoRA repair",
        "",
        "Population SD (`ddof=0`) is shown in the headline table; sample SD is preserved in JSON.",
        "",
        "| Architecture | Eff ↓ | Gen ↓ | Spe ↑ | Spe_success ↑ | PPL ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, group in (("Fixed SVD", "fixed"), ("Sparse LoRA", "lora")):
        vals = aggregate[group]
        cells = []
        for metric in ("Eff", "Gen", "Spe", "Spe_success", "PPL"):
            item = vals[metric]
            cells.append(f"{item['mean']:.4f} ± {item['population_sd']:.4f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Paired LoRA − fixed deltas",
        "",
        "| Metric | Mean paired delta | Population SD |",
        "|---|---:|---:|",
    ]
    for metric in ("Eff", "Gen", "Spe", "Spe_success", "PPL"):
        item = aggregate["paired_delta_lora_minus_fixed"][metric]
        lines.append(
            f"| {metric} | {item['mean']:+.4f} | {item['population_sd']:.4f} |"
        )

    lines += [
        "",
        "## Per-seed repair complexity",
        "",
        "| Seed | Fixed rows | Fixed actual rank | LoRA rows | LoRA rank | LoRA trainable params |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_seed:
        fr = row["fixed_repair"]
        lr = row["lora_repair"]
        lines.append(
            f"| {row['seed']} | {fr.get('selected_rows')} | {fr.get('rank_actual')} | "
            f"{lr.get('selected_rows')} | {lr.get('rank_requested')} | "
            f"{lr.get('stage2_trainable_parameters')} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
