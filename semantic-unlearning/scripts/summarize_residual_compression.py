#!/usr/bin/env python3
"""One table for a residual-bank compression run.

Reads <compression-dir>/compression_report.json, the source run's official
eval and each variant's official eval, and prints a markdown table:
storage, reconstruction fidelity, and the benchmark's metrics, with the
uncompressed source run as the first row. Also writes
<compression-dir>/compression_summary.json.

    python -u scripts/summarize_residual_compression.py \
      --compression-dir outputs/mcf_linear_global_bankcomp_seed1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compress_residual_bank import EVAL_FILES, REPORT_NAME


def _get(payload, *path):
    for key in path:
        if not isinstance(payload, dict) or key not in payload:
            return None
        payload = payload[key]
    return payload


def _ppl(value):
    if isinstance(value, dict):
        return value.get("ppl")
    return value


def extract_metrics(benchmark, result):
    """Headline metrics of one official eval file (None where absent)."""
    if result is None:
        return None
    if benchmark in ("mcf", "zsre"):
        metrics = {
            "forget_Eff": _get(result, "forget", "Eff"),
            "forget_Gen": _get(result, "forget", "Gen"),
            "forget_Spe": _get(result, "forget", "Spe"),
            "retain_Eff": _get(result, "retain", "Eff"),
            "retain_Gen": _get(result, "retain", "Gen"),
            "retain_Spe": _get(result, "retain", "Spe"),
        }
        if benchmark == "mcf":
            metrics["display_zero"] = _get(result, "static_branch_display_zero_check", "passed")
            metrics["PPL"] = result.get("forget_PPL")
        else:
            metrics["PPL"] = _ppl(result.get("runtime_aligned_PPL"))
        return metrics
    if benchmark == "mquake":
        return {
            "forget_Eff": _get(result, "forget", "Eff"),
            "forget_AtomicGen": _get(result, "forget", "AtomicGen"),
            "retain_Eff": _get(result, "retain", "Eff"),
            "retain_AtomicGen": _get(result, "retain", "AtomicGen"),
            "PPL": _ppl(result.get("runtime_aligned_PPL")),
        }
    if benchmark == "rwku":
        return {
            "same50": _get(result, "same_50_efficacy", "recovery_accuracy"),
            "heldout_L1": _get(result, "heldout_level1", "recovery_accuracy"),
            "heldout_L2": _get(result, "heldout_level2", "recovery_accuracy"),
            "paraphrase": _get(result, "heldout_level2_paraphrase", "recovery_accuracy"),
            "level3": _get(result, "adversarial_level3", "recovery_accuracy"),
            "neighbor": _get(result, "neighbors", "recovery_accuracy"),
            "PPL": _ppl(result.get("runtime_aligned_PPL")),
        }
    return {}


def _load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def _fmt(value):
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "✓" if value else "✗"
    if isinstance(value, float):
        if value != 0 and abs(value) < 0.01:
            return f"{value:.2e}"
        return f"{value:.4g}"
    return str(value)


def summarize(compression_dir):
    compression_dir = Path(compression_dir)
    report = json.loads((compression_dir / REPORT_NAME).read_text())
    benchmark = report["benchmark"]
    eval_file = EVAL_FILES.get(benchmark)
    source_eval = _load(Path(report["source_run_dir"]) / eval_file) if eval_file else None
    rows = [{
        "variant": "source (uncompressed)",
        "bank_bytes": report["full_bank_bytes_fp32"],
        "ratio": 1.0,
        "row_cosine_mean": 1.0,
        "metrics": extract_metrics(benchmark, source_eval),
    }]
    for name, meta in report["variants"].items():
        result = _load(Path(meta["run_dir"]) / eval_file) if eval_file else None
        rows.append({
            "variant": name,
            "bank_bytes": meta["storage"]["bytes_as_saved"],
            "ratio": meta["storage"]["ratio_vs_full_fp32"],
            "row_cosine_mean": meta["reconstruction"]["row_cosine_mean"],
            "metrics": extract_metrics(benchmark, result),
        })
    keys = []
    for row in rows:
        for key in (row["metrics"] or {}):
            if key not in keys:
                keys.append(key)
    header = ["variant", "bank KB", "× smaller", "row cos"] + keys
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        metrics = row["metrics"] or {}
        cells = [
            row["variant"],
            f"{row['bank_bytes'] / 1024:.1f}",
            f"{row['ratio']:.2f}",
            f"{row['row_cosine_mean']:.4f}",
        ] + [_fmt(metrics.get(key)) for key in keys]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    missing = [row["variant"] for row in rows if row["metrics"] is None]
    summary = {
        "benchmark": benchmark,
        "n_facts": report["n_facts"],
        "router_bytes_fp32": report["router"]["bytes_fp32"],
        "rows": rows,
        "missing_evals": missing,
        "table_markdown": table,
    }
    (compression_dir / "compression_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compression-dir", required=True)
    args = parser.parse_args(argv)
    summary = summarize(args.compression_dir)
    print(f"benchmark={summary['benchmark']}  N={summary['n_facts']}  "
          f"router={summary['router_bytes_fp32'] / 1024:.1f} KB (unchanged)")
    print(summary["table_markdown"])
    if summary["missing_evals"]:
        print("missing evals:", ", ".join(summary["missing_evals"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
