#!/usr/bin/env python3
"""Create a compact Base-vs-compositional-marker result and acceptance table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping


METRICS = ("Eff", "Gen", "Spe", "Spe_success")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--edited", required=True)
    p.add_argument("--method-summary", required=True)
    p.add_argument("--post-reload-acceptance", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-abs-spe-delta", type=float, default=0.2)
    p.add_argument("--max-abs-retain-eff-gen-delta", type=float, default=1.0)
    p.add_argument("--max-ppl-percent-delta", type=float, default=5.0)
    return p.parse_args(argv)


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _metric_block(payload: Mapping[str, Any], split: str) -> Dict[str, float]:
    value = payload.get(split)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"official evaluation lacks {split!r} metrics")
    block: Dict[str, float] = {}
    for metric in METRICS:
        number = float(value[metric])
        if not math.isfinite(number):
            raise RuntimeError(f"non-finite {split}.{metric}")
        block[metric] = number
    return block


def build_report(
    base: Mapping[str, Any],
    edited: Mapping[str, Any],
    method: Mapping[str, Any],
    post_reload: Mapping[str, Any],
    *,
    max_abs_spe_delta: float,
    max_abs_retain_eff_gen_delta: float = 1.0,
    max_ppl_percent_delta: float,
) -> Dict[str, Any]:
    base_forget = _metric_block(base, "forget")
    edit_forget = _metric_block(edited, "forget")
    base_retain = _metric_block(base, "retain")
    edit_retain = _metric_block(edited, "retain")
    base_ppl = float(base["PPL"])
    edit_ppl = float(edited["PPL"])
    ppl_percent = 100.0 * (edit_ppl - base_ppl) / max(abs(base_ppl), 1e-12)
    delta_forget = {
        metric: edit_forget[metric] - base_forget[metric] for metric in METRICS
    }
    delta_retain = {
        metric: edit_retain[metric] - base_retain[metric] for metric in METRICS
    }
    checks = {
        "Eff_zero": edit_forget["Eff"] == 0.0,
        "Gen_zero": edit_forget["Gen"] == 0.0,
        "abs_forget_Spe_delta_within_limit": (
            abs(delta_forget["Spe"]) <= float(max_abs_spe_delta)
        ),
        "abs_retain_Spe_delta_within_limit": (
            abs(delta_retain["Spe"]) <= float(max_abs_spe_delta)
        ),
        "abs_retain_Eff_delta_within_limit": (
            abs(delta_retain["Eff"]) <= float(max_abs_retain_eff_gen_delta)
        ),
        "abs_retain_Gen_delta_within_limit": (
            abs(delta_retain["Gen"]) <= float(max_abs_retain_eff_gen_delta)
        ),
        "abs_PPL_percent_delta_within_limit": (
            abs(ppl_percent) <= float(max_ppl_percent_delta)
        ),
        "reader_gate_passed": bool(
            method.get("reader_gate", {}).get("passed", False)
        ),
        "method_training_acceptance_passed": bool(
            method.get("acceptance", {}).get("passed", False)
        ),
        "post_reload_acceptance_passed": bool(post_reload.get("passed", False)),
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "target": {
            "Eff": 0.0,
            "Gen": 0.0,
            "max_abs_Spe_delta": float(max_abs_spe_delta),
            "max_abs_retain_Eff_Gen_delta": float(max_abs_retain_eff_gen_delta),
            "max_abs_PPL_percent_delta": float(max_ppl_percent_delta),
        },
        "base": {
            "forget": base_forget,
            "retain": base_retain,
            "PPL": base_ppl,
        },
        "edited": {
            "forget": edit_forget,
            "retain": edit_retain,
            "PPL": edit_ppl,
        },
        "delta": {
            "forget": delta_forget,
            "retain": delta_retain,
            "PPL": edit_ppl - base_ppl,
            "PPL_percent": ppl_percent,
        },
        "checks": checks,
        "reader_gate": method.get("reader_gate"),
        "post_reload_acceptance": dict(post_reload),
        "architecture": method.get("architecture"),
    }


def write_report(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    base = report["base"]
    edited = report["edited"]
    delta = report["delta"]
    lines = [
        "# MCF compositional marker result",
        "",
        f"Overall: **{report['status']}**",
        "",
        "| Split | Metric | Base | Edited | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("forget", "retain"):
        for metric in METRICS:
            lines.append(
                f"| {split} | {metric} | {base[split][metric]:.4f} | "
                f"{edited[split][metric]:.4f} | {delta[split][metric]:+.4f} |"
            )
    lines.extend(
        [
            f"| utility | PPL | {base['PPL']:.6f} | {edited['PPL']:.6f} | "
            f"{delta['PPL']:+.6f} ({delta['PPL_percent']:+.3f}%) |",
            "",
            "## Acceptance",
            "",
        ]
    )
    for key, passed in report["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{key}`")
    (output_dir / "comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(argv=None) -> None:
    a = parse_args(argv)
    report = build_report(
        _load(Path(a.base)),
        _load(Path(a.edited)),
        _load(Path(a.method_summary)),
        _load(Path(a.post_reload_acceptance)),
        max_abs_spe_delta=float(a.max_abs_spe_delta),
        max_abs_retain_eff_gen_delta=float(a.max_abs_retain_eff_gen_delta),
        max_ppl_percent_delta=float(a.max_ppl_percent_delta),
    )
    output = Path(a.output_dir)
    write_report(report, output)
    print(json.dumps(report, indent=2))
    print(f"comparison: {output / 'comparison.md'}")


if __name__ == "__main__":
    main()
