#!/usr/bin/env python3
"""Validate and aggregate frozen MCF scoped-bias results against matched Base."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = (
    "forget_Eff",
    "forget_Gen",
    "forget_Spe",
    "forget_Spe_success",
    "retain_Eff",
    "retain_Gen",
    "retain_Spe",
    "retain_Spe_success",
    "PPL",
)
EXPECTED_PROTOCOL = "mcf_scoped_bias_reader_v1"
EXPECTED_PENALTY = 512.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(range(2, 11))
    )
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args(argv)


def _load(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _finite(value, label):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {result!r}")
    return result


def _metrics(payload, path):
    values = {}
    for split in ("forget", "retain"):
        block = payload.get(split)
        if not isinstance(block, dict):
            raise ValueError(f"{path}: missing {split} metrics")
        for metric in ("Eff", "Gen", "Spe", "Spe_success"):
            key = f"{split}_{metric}"
            values[key] = _finite(block.get(metric), f"{path}:{key}")
    values["PPL"] = _finite(payload.get("PPL"), f"{path}:PPL")
    return values


def _validate_scoped(payload, path, seed):
    if int(payload.get("seed", -1)) != int(seed):
        raise ValueError(f"{path}: stored seed does not match {seed}")
    scoped = payload.get("scoped_span_edit")
    if not isinstance(scoped, dict) or scoped.get("loaded") is not True:
        raise ValueError(f"{path}: scoped sidecar was not loaded")
    if int(scoped.get("record_scopes", -1)) != 50:
        raise ValueError(f"{path}: expected 50 record scopes")
    metadata = scoped.get("metadata") or {}
    if metadata.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError(f"{path}: unexpected scoped protocol")
    if float(metadata.get("penalty", float("nan"))) != EXPECTED_PENALTY:
        raise ValueError(f"{path}: penalty is not frozen at {EXPECTED_PENALTY}")
    if metadata.get("base_weights_modified") is not False:
        raise ValueError(f"{path}: base-weights audit is not false")
    acceptance = payload.get("post_reload_acceptance") or {}
    if acceptance.get("passed") is not True:
        raise ValueError(f"{path}: post-reload acceptance did not pass")
    audit = scoped.get("per_split_prompt_fire_audit") or {}
    if audit.get("used_for_training_or_checkpoint_selection") is not False:
        raise ValueError(f"{path}: routing audit provenance is invalid")
    return audit.get("groups") or {}


def collect(root, seeds):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    if 1 in seeds:
        raise ValueError("seed 1 is exploratory and excluded from confirmatory aggregation")
    rows = []
    for seed in seeds:
        seed_dir = Path(root) / f"seed{seed}"
        base_path = seed_dir / "base_official_eval.json"
        scoped_path = seed_dir / "official_eval.json"
        base = _load(base_path)
        scoped = _load(scoped_path)
        if int(base.get("seed", -1)) != int(seed):
            raise ValueError(f"{base_path}: stored seed does not match {seed}")
        routing = _validate_scoped(scoped, scoped_path, seed)
        base_metrics = _metrics(base, base_path)
        scoped_metrics = _metrics(scoped, scoped_path)
        row = {"seed": int(seed)}
        for metric in METRICS:
            row[f"base_{metric}"] = base_metrics[metric]
            row[f"scoped_{metric}"] = scoped_metrics[metric]
            row[f"delta_{metric}"] = scoped_metrics[metric] - base_metrics[metric]
        for group_name, values in sorted(routing.items()):
            row[f"route_{group_name}_matched"] = int(values["matched_prompts"])
            row[f"route_{group_name}_total"] = int(values["prompt_count"])
        rows.append(row)
    return rows


def aggregate(rows):
    result = {"n_seeds": len(rows), "seeds": [row["seed"] for row in rows]}
    for metric in METRICS:
        for prefix in ("base", "scoped", "delta"):
            values = [float(row[f"{prefix}_{metric}"]) for row in rows]
            result[f"{prefix}_{metric}_mean"] = statistics.fmean(values)
            result[f"{prefix}_{metric}_population_sd"] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )
    return result


def write_outputs(root, out_dir, rows, summary):
    destination = Path(out_dir or (Path(root) / "aggregate"))
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "CONFIRMATORY_FROZEN_NO_TUNING",
        "method_commit": "9284d91",
        "method_tag": "mcf-scoped-bias-v1",
        "penalty": EXPECTED_PENALTY,
        "seed1_excluded_as_exploratory": True,
        "claim": "exact-name-scoped conditional suppression, not weight-level unlearning",
        "per_seed": rows,
        "aggregate": summary,
    }
    (destination / "aggregate.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (destination / "per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Frozen MCF scoped-bias confirmatory aggregate",
        "",
        "Seed 1 is excluded because the method was developed after its Gen failure. ",
        "The claim is exact-name-scoped suppression/model editing, not weight-level unlearning.",
        "",
        "| Seed | Base Eff | Scoped Eff | Base Gen | Scoped Gen | ΔSpe | Δretain Spe | ΔPPL |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['base_forget_Eff']:.4f} | "
            f"{row['scoped_forget_Eff']:.4f} | {row['base_forget_Gen']:.4f} | "
            f"{row['scoped_forget_Gen']:.4f} | {row['delta_forget_Spe']:+.4f} | "
            f"{row['delta_retain_Spe']:+.4f} | {row['delta_PPL']:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"Validated seeds: {summary['n_seeds']} ({summary['seeds']})",
            f"Mean scoped Eff: {summary['scoped_forget_Eff_mean']:.6f}",
            f"Mean scoped Gen: {summary['scoped_forget_Gen_mean']:.6f}",
            f"Mean ΔSpe: {summary['delta_forget_Spe_mean']:+.6f}",
            f"Mean Δretain Spe: {summary['delta_retain_Spe_mean']:+.6f}",
            f"Mean ΔPPL: {summary['delta_PPL_mean']:+.6f}",
        ]
    )
    (destination / "aggregate.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return destination


def main(argv=None):
    args = parse_args(argv)
    rows = collect(args.root, args.seeds)
    summary = aggregate(rows)
    destination = write_outputs(args.root, args.out_dir, rows, summary)
    print(json.dumps(summary, indent=2))
    print(f"wrote: {destination}")


if __name__ == "__main__":
    main()
