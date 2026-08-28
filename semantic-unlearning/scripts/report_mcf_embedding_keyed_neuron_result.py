#!/usr/bin/env python3
"""Build the preregistered Base-vs-neuron-erasure acceptance report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


METRICS = ("Eff", "Gen", "Spe", "Spe_success")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--edited", required=True)
    parser.add_argument("--method-summary", required=True)
    parser.add_argument("--post-reload-acceptance", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-abs-spe-delta", type=float, default=0.2)
    parser.add_argument("--max-abs-retain-eff-gen-delta", type=float, default=1.0)
    parser.add_argument("--max-ppl-percent-delta", type=float, default=5.0)
    return parser.parse_args(list(argv) if argv is not None else None)


def _load(path: str) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _metrics(payload: Mapping[str, Any], split: str) -> Dict[str, float]:
    value = payload.get(split)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"evaluation lacks {split!r} metrics")
    result = {metric: float(value[metric]) for metric in METRICS}
    if not all(math.isfinite(number) for number in result.values()):
        raise RuntimeError(f"non-finite {split} metric")
    return result


def build_report(
    base: Mapping[str, Any],
    edited: Mapping[str, Any],
    method: Mapping[str, Any],
    reload: Mapping[str, Any],
    *,
    max_abs_spe_delta: float,
    max_abs_retain_eff_gen_delta: float,
    max_ppl_percent_delta: float,
) -> Dict[str, Any]:
    base_forget = _metrics(base, "forget")
    edited_forget = _metrics(edited, "forget")
    base_retain = _metrics(base, "retain")
    edited_retain = _metrics(edited, "retain")
    forget_delta = {key: edited_forget[key] - base_forget[key] for key in METRICS}
    retain_delta = {key: edited_retain[key] - base_retain[key] for key in METRICS}
    base_ppl = float(base["PPL"])
    edited_ppl = float(edited["PPL"])
    ppl_percent = 100.0 * (edited_ppl - base_ppl) / max(abs(base_ppl), 1e-12)
    acceptance = method.get("acceptance", {})
    causal = method.get("causal_component_ablation", {})
    architecture = method.get("architecture", {})
    firewall = method.get("data_firewall", {})
    data_access = (
        firewall.get("data_access", {}) if isinstance(firewall, Mapping) else {}
    )
    checks = {
        "Eff_zero": edited_forget["Eff"] == 0.0,
        "Gen_zero": edited_forget["Gen"] == 0.0,
        "forget_Spe_local": abs(forget_delta["Spe"]) <= max_abs_spe_delta,
        "retain_Spe_local": abs(retain_delta["Spe"]) <= max_abs_spe_delta,
        "retain_Eff_local": abs(retain_delta["Eff"]) <= max_abs_retain_eff_gen_delta,
        "retain_Gen_local": abs(retain_delta["Gen"]) <= max_abs_retain_eff_gen_delta,
        "PPL_local": abs(ppl_percent) <= max_ppl_percent_delta,
        "locked_training_acceptance": bool(acceptance.get("passed")),
        "fresh_reload_acceptance": bool(reload.get("passed")),
        "embedding_writer_causally_necessary": bool(causal.get("writer_is_necessary")),
        "neuron_decoder_causally_necessary": bool(causal.get("decoder_is_necessary")),
        "detector_gate_passed": bool(acceptance.get("detector_gate_passed")),
        "LM_head_unchanged": architecture.get("lm_head_edited") is False
        and bool(acceptance.get("lm_head_bit_identical")),
        "no_router_or_sidecar": not any(
            bool(architecture.get(key))
            for key in (
                "runtime_string_matcher",
                "external_router",
                "retrieval_cache",
                "sidecar",
            )
        ),
        "no_evaluation_training_access": data_access
        == {
            "official_paraphrases_seen": 0,
            "official_neighborhoods_seen": 0,
            "benchmark_retain_seen": 0,
            "official_ppl_seen": False,
        },
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "target": {
            "Eff": 0.0,
            "Gen": 0.0,
            "max_abs_Spe_delta": max_abs_spe_delta,
            "max_abs_retain_Eff_Gen_delta": max_abs_retain_eff_gen_delta,
            "max_abs_PPL_percent_delta": max_ppl_percent_delta,
        },
        "base": {"forget": base_forget, "retain": base_retain, "PPL": base_ppl},
        "edited": {
            "forget": edited_forget,
            "retain": edited_retain,
            "PPL": edited_ppl,
        },
        "delta": {
            "forget": forget_delta,
            "retain": retain_delta,
            "PPL": edited_ppl - base_ppl,
            "PPL_percent": ppl_percent,
        },
        "checks": checks,
        "architecture": architecture,
        "causal_component_ablation": causal,
        "post_reload_acceptance": dict(reload),
    }


def write_report(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# MCF embedding-keyed sparse neuron erasure",
        "",
        f"Overall: **{report['status']}**",
        "",
        "| Split | Metric | Base | Edited | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("forget", "retain"):
        for metric in METRICS:
            lines.append(
                f"| {split} | {metric} | {report['base'][split][metric]:.4f} | "
                f"{report['edited'][split][metric]:.4f} | "
                f"{report['delta'][split][metric]:+.4f} |"
            )
    lines.append(
        f"| utility | PPL | {report['base']['PPL']:.6f} | "
        f"{report['edited']['PPL']:.6f} | {report['delta']['PPL']:+.6f} "
        f"({report['delta']['PPL_percent']:+.3f}%) |"
    )
    lines.extend(["", "## Preregistered acceptance", ""])
    for key, passed in report["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{key}`")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(
        _load(args.base),
        _load(args.edited),
        _load(args.method_summary),
        _load(args.post_reload_acceptance),
        max_abs_spe_delta=float(args.max_abs_spe_delta),
        max_abs_retain_eff_gen_delta=float(args.max_abs_retain_eff_gen_delta),
        max_ppl_percent_delta=float(args.max_ppl_percent_delta),
    )
    output_dir = Path(args.output_dir).resolve()
    write_report(report, output_dir)
    print(json.dumps(report, indent=2))
    print(f"comparison: {output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
