#!/usr/bin/env python3
"""Summarize matched V2 A/B/C/D development runs without official prompts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args(argv)

    rows = []
    for value in args.run_dirs:
        run = Path(value).resolve()
        manifest = load_json(run / "association_manifest.json")
        report = load_json(run / "training_report.json")
        route = manifest["route_audit"]
        wrong = manifest["wrong_relation_route_audit"]
        constraints = report["constraint_metrics"]
        absolute = report["absolute_metrics"]
        ppl_files = sorted(run.glob("runtime_aligned_ppl_audit_*.json"))
        ppl = load_json(ppl_files[-1]) if ppl_files else None
        rows.append({
            "arm": manifest["arm"],
            "gate": manifest["gate"],
            "objective": manifest["objective"],
            "train_route_recall": route["train"]["correct_row_active_fraction"],
            "dev_route_recall": route["development"]["correct_row_active_fraction"],
            "wrong_relation_route_fraction": wrong["expected_owner_route_fraction"],
            "train_abs_pass": absolute["train"]["facts_passing"],
            "dev_abs_pass": absolute["development"]["facts_passing"],
            "train_joint_pass": constraints["train"]["facts_passing"],
            "dev_joint_pass": constraints["development"]["facts_passing"],
            "train_min_margin": constraints["train"]["minimum_margin"],
            "dev_min_margin": constraints["development"]["minimum_margin"],
            "train_max_violation": constraints["train"]["max_violation"],
            "dev_max_violation": constraints["development"]["max_violation"],
            "runtime_ppl": (
                ppl["runtime_aligned"]["edited"]["ppl"] if ppl else None
            ),
            "runtime_ppl_delta": (
                ppl["runtime_aligned"]["delta_ppl"] if ppl else None
            ),
            "runtime_ppl_route_rows": (
                ppl["runtime_aligned"][
                    "route_activity_during_runtime_aligned_scoring"
                ]["active_batch_rows"]
                if ppl else None
            ),
            "stop_reason": report["stop_reason"],
            "best_step": report["best_step"],
        })

    rows.sort(key=lambda row: row["arm"])
    print(json.dumps(rows, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
