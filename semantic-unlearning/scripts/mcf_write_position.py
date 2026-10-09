#!/usr/bin/env python3
"""Write-position experiment (MCF seed 1, 50 facts, layer 19): where the routed row is added.

The router is unchanged: the linear classifier reads the request boundary (last
prompt token) at layer 19 and picks at most one fact. Only the set of prompt
positions that receive that fact's row changes:

  A  last          last prompt token (shipped behaviour)
  B  last_subject  last subject token + last prompt token
  C  subject_span  every subject token + last prompt token
  D  all_prompt    every prompt token except BOS

Subcommands
  make-arm   copy a fitted (zero-row) router artifact and set its write_mode
  summarize  one table: the seed-1 references + arms A-D (official MCF metrics)

    python scripts/mcf_write_position.py make-arm --router-dir R --output-dir O --arm B
    python scripts/mcf_write_position.py summarize --root outputs/mcf_write_position_seed1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

ARMS = {"A": "last", "B": "last_subject", "C": "subject_span", "D": "all_prompt"}
DESCRIPTION = {
    "A": "last prompt token (shipped)",
    "B": "last subject token + last token",
    "C": "whole subject span + last token",
    "D": "every prompt token except BOS",
}
REFERENCES = (
    ("seed-1 reference (multiseed regular, 10800 s)",
     "outputs/mcf_multiseed_regular_v1/seed1/L19/linear_global"),
    ("seed-1 exploratory sweep (3600 s cap)",
     "outputs/mcf_layer_sweep_linear_regular_v1/L19/linear_global"),
)
COLUMNS = (
    ("forget_Eff", "forget Eff ↓"), ("forget_Gen", "forget Gen ↓"),
    ("forget_Spe", "forget Spe"), ("retain_Eff", "retain Eff"),
    ("retain_Gen", "retain Gen"), ("PPL", "PPL"),
    ("facts_passing_train", "facts <1e-6 (train)"),
    ("facts_passing_dev", "facts <1e-6 (dev)"),
    ("row_norm_median", "‖Δe‖ median"), ("training_stop_reason", "stop"),
)


def make_arm(router_dir, output_dir, arm):
    import torch

    router_dir, output_dir = Path(router_dir), Path(output_dir)
    mode = ARMS[arm]
    artifact = torch.load(router_dir / "fact_association_embeddings.pt",
                          map_location="cpu", weights_only=False)
    if float(artifact["rows"].abs().max()) != 0.0:
        raise ValueError(f"{router_dir} has trained rows; expected a fitted, untrained router")
    tmp = output_dir.with_name(output_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(router_dir, tmp)
    artifact["write_mode"] = mode
    torch.save(artifact, tmp / "fact_association_embeddings.pt")
    manifest_path = tmp / "association_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"write_mode": mode, "write_position_arm": arm,
                     "write_position_description": DESCRIPTION[arm],
                     "router_copied_from": str(router_dir.resolve())})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.rename(output_dir)
    print(json.dumps({"arm": arm, "write_mode": mode, "router_dir": str(output_dir)}))


def _load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def _row(run_dir, label):
    from summarize_mcf_layer_sweep import collect

    row = collect(run_dir, label)
    training = _load(Path(run_dir) / "training_report.json") or {}
    final = training.get("final_metrics_classifier_routing_all_views") or {}
    for split, key in (("train", "facts_passing_train"), ("development", "facts_passing_dev")):
        part = final.get(split) or {}
        row[key] = (f"{part['facts_passing']}/{part['facts_total']}"
                    if "facts_passing" in part else None)
    manifest = _load(Path(run_dir) / "association_manifest.json") or {}
    row["write_mode"] = manifest.get("write_mode", "last")
    return row


def _fmt(value):
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def summarize(root):
    root = Path(root)
    rows = []
    for label, path in REFERENCES:
        if Path(path).is_dir():
            rows.append(_row(path, label))
    for arm, mode in ARMS.items():
        rows.append(_row(root / f"arm_{arm}" / "linear_global",
                         f"{arm}: {DESCRIPTION[arm]}"))
    header = ["setting", "status"] + [title for _, title in COLUMNS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(
            [row["label"], _fmt(row.get("status"))] + [_fmt(row.get(k)) for k, _ in COLUMNS]
        ) + " |")
    table = "\n".join(lines)
    (root / "summary.md").write_text(
        "# MCF seed 1, 50 facts, layer 19: write position of the routed row\n\n"
        "Router identical in every arm (reads the last prompt token). Eff/Gen: complete "
        "sensitive-answer probability in %, lower is better.\n\n" + table + "\n"
    )
    (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    with open(root / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    print(table)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make-arm")
    make.add_argument("--router-dir", required=True)
    make.add_argument("--output-dir", required=True)
    make.add_argument("--arm", choices=sorted(ARMS), required=True)
    summ = sub.add_parser("summarize")
    summ.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    if args.command == "make-arm":
        make_arm(args.router_dir, args.output_dir, args.arm)
    else:
        summarize(args.root)


if __name__ == "__main__":
    main()
