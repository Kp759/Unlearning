#!/usr/bin/env python3
"""Re-summarize a saved MCF result with static-branch probability metrics.

Read-only by default. This does not rerun model inference and never overwrites
the source JSON. Strict conversion requires target_true NLL sums, token counts,
and teacher-forced correctness flags in the saved raw rows.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from mcf_zero_unlearn_metric_parity import summarize_probability_metrics


KEYS = (
    "Eff",
    "Gen",
    "Spe",
    "ReleasedAccuracy_Eff",
    "ReleasedAccuracy_Gen",
    "TokenGeometricMean_Eff",
    "TokenGeometricMean_Gen",
    "SensitivePref_Eff",
    "SensitivePref_Gen",
    "Legacy_Spe_ProbabilityDiff",
)


def display_zero(metrics):
    return bool(
        0.0 <= float(metrics["Eff"]) < 0.005
        and 0.0 <= float(metrics["Gen"]) < 0.005
        and float(metrics["ReleasedAccuracy_Eff"]) == 0.0
        and float(metrics["ReleasedAccuracy_Gen"]) == 0.0
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional NEW JSON path for converted payload; source is never overwritten.",
    )
    args = parser.parse_args(argv)

    source = Path(args.result_json).resolve()
    result = json.loads(source.read_text())
    converted = deepcopy(result)
    converted["legacy_counterfact"] = {}

    compact = {}
    for split in ("forget", "retain"):
        converted["legacy_counterfact"][split] = deepcopy(result[split])
        metrics = summarize_probability_metrics(
            result[split],
            result[f"{split}_raw"],
        )
        converted[split] = metrics
        compact[split] = {key: metrics.get(key) for key in KEYS}
        compact[split]["metric_version"] = metrics["metric_version"]

    converted["metric_version"] = "zerounlearn_answer_probability_v2"
    converted["static_branch_display_zero_check"] = {
        "definition": (
            "Eff < 0.005% and Gen < 0.005%, with ReleasedAccuracy_Eff == 0 "
            "and ReleasedAccuracy_Gen == 0; display-zero, not exact-zero probability"
        ),
        "passed": display_zero(converted["forget"]),
    }

    print(json.dumps({
        "source": str(source),
        "source_modified": False,
        "forget": compact["forget"],
        "retain": compact["retain"],
        "static_branch_display_zero_check": (
            converted["static_branch_display_zero_check"]["passed"]
        ),
    }, indent=2, allow_nan=False))

    if args.out:
        out = Path(args.out).resolve()
        if out == source:
            raise ValueError("Refusing to overwrite the source result JSON")
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {out}")
        out.write_text(json.dumps(converted, indent=2, allow_nan=False) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
