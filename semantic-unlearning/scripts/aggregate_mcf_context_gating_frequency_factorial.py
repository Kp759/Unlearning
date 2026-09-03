#!/usr/bin/env python3
"""Aggregate the preregistered frequency-cap x decoder-component factorial."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


CONDITIONS = ("frequency_capped", "uniform_same_cap", "uniform_raised_cap")
COMPONENTS = ("embedding_only", "full_embedding_plus_neuron")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(list(argv) if argv is not None else None)


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _ppl_delta_percent(value: float, base: float) -> float:
    return 100.0 * (float(value) - float(base)) / max(abs(float(base)), 1e-12)


def _same_finite_float(first: Any, second: Any) -> bool:
    try:
        left = float(first)
        right = float(second)
    except (TypeError, ValueError):
        return False
    return bool(
        math.isfinite(left)
        and math.isfinite(right)
        and math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
    )


def build_aggregate(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    expected_writer_conditions: Mapping[str, Mapping[str, Any]],
    max_abs_spe_delta: float,
    max_abs_ppl_percent_delta: float,
) -> Dict[str, Any]:
    rows = []
    lookup: Dict[tuple[str, str], Dict[str, Any]] = {}
    for condition in CONDITIONS:
        payload = payloads.get(condition)
        if payload is None:
            continue
        components = payload.get("components", {})
        base = components.get("reconstructed_base", {})
        for component in COMPONENTS:
            value = components.get(component)
            if not isinstance(value, Mapping) or not isinstance(base, Mapping):
                continue
            common = value.get("forget_frequency_strata", {}).get("common", {})
            base_common = base.get("forget_frequency_strata", {}).get("common", {})
            common_metrics = common.get("metrics")
            base_common_metrics = base_common.get("metrics")
            common_spe_leakage = (
                abs(float(common_metrics["Spe"]) - float(base_common_metrics["Spe"]))
                if isinstance(common_metrics, Mapping)
                and isinstance(base_common_metrics, Mapping)
                else None
            )
            row = {
                "writer_condition": condition,
                "decoder_condition": component,
                "forget_Eff": float(value["forget"]["Eff"]),
                "forget_Gen": float(value["forget"]["Gen"]),
                "retain_Spe_delta": float(value["retain"]["Spe"])
                - float(base["retain"]["Spe"]),
                "PPL_percent_delta": _ppl_delta_percent(value["PPL"], base["PPL"]),
                "common_record_count": int(common.get("record_count") or 0),
                "common_forget_Eff": (
                    float(common_metrics["Eff"])
                    if isinstance(common_metrics, Mapping)
                    else None
                ),
                "common_forget_Gen": (
                    float(common_metrics["Gen"])
                    if isinstance(common_metrics, Mapping)
                    else None
                ),
                "common_Spe_leakage_abs": common_spe_leakage,
            }
            rows.append(row)
            lookup[(condition, component)] = row

    required = [
        (condition, component) for condition in CONDITIONS for component in COMPONENTS
    ]
    complete = all(key in lookup for key in required)
    receipt_validity = {}
    configuration_matches = {}
    for condition in CONDITIONS:
        payload = payloads.get(condition)
        expected = expected_writer_conditions.get(condition)
        observed = payload.get("writer_configuration") if payload else None
        receipt_validity[condition] = bool(
            isinstance(payload, Mapping)
            and payload.get("kind")
            == "mcf_embedding_keyed_neuron_post_freeze_component_evaluation"
            and payload.get("writer_mode") == "embedding_keyed"
            and payload.get("used_for_training_checkpoint_selection_or_retry") is False
            and payload.get("dataset") == "MCF"
            and payload.get("sample_mode") == "official"
            and isinstance(payload.get("seed"), int)
            and isinstance(payload.get("unlearn_num"), int)
            and isinstance(payload.get("retain_num"), int)
            and isinstance(payload.get("source_stage1_state_sha256"), str)
            and bool(payload.get("source_stage1_state_sha256"))
        )
        configuration_matches[condition] = bool(
            isinstance(observed, Mapping)
            and isinstance(expected, Mapping)
            and _same_finite_float(
                observed.get("row_norm_cap"), expected.get("row_norm_cap")
            )
            and _same_finite_float(
                observed.get("row_norm_cap_frequency_alpha"),
                expected.get("row_norm_cap_frequency_alpha"),
            )
            and observed.get("max_subject_token_frequency")
            == expected.get("max_subject_token_frequency")
        )
    checks: Dict[str, bool] = {
        "all_six_cells_present": complete,
        "all_component_receipts_valid": all(receipt_validity.values()),
        "writer_conditions_match_registry": all(configuration_matches.values()),
    }
    if all(condition in payloads for condition in CONDITIONS):
        bindings = {
            (
                payloads[condition].get("dataset"),
                payloads[condition].get("sample_mode"),
                payloads[condition].get("seed"),
                payloads[condition].get("unlearn_num"),
                payloads[condition].get("retain_num"),
            )
            for condition in CONDITIONS
        }
        writer_hashes = {
            payloads[condition].get("source_stage1_state_sha256")
            for condition in CONDITIONS
        }
        checks["official_evaluation_bindings_match"] = len(bindings) == 1
        checks[
            "writers_independently_materialized"
        ] = None not in writer_hashes and len(writer_hashes) == len(CONDITIONS)
    else:
        checks["official_evaluation_bindings_match"] = False
        checks["writers_independently_materialized"] = False
    if complete:
        capped_full = lookup[("frequency_capped", "full_embedding_plus_neuron")]
        raised_full = lookup[("uniform_raised_cap", "full_embedding_plus_neuron")]
        capped_embedding = lookup[("frequency_capped", "embedding_only")]
        raised_embedding = lookup[("uniform_raised_cap", "embedding_only")]
        checks.update(
            {
                "raised_full_retain_Spe_within_primary_margin": abs(
                    raised_full["retain_Spe_delta"]
                )
                <= float(max_abs_spe_delta),
                "raised_full_PPL_within_primary_margin": abs(
                    raised_full["PPL_percent_delta"]
                )
                <= float(max_abs_ppl_percent_delta),
                "common_stratum_present_in_all_cells": all(
                    row["common_record_count"] > 0 for row in rows
                ),
            }
        )
        leakage_values = [
            capped_full["common_Spe_leakage_abs"],
            raised_full["common_Spe_leakage_abs"],
            capped_embedding["common_Spe_leakage_abs"],
            raised_embedding["common_Spe_leakage_abs"],
        ]
        if all(value is not None for value in leakage_values):
            full_increase = float(raised_full["common_Spe_leakage_abs"]) - float(
                capped_full["common_Spe_leakage_abs"]
            )
            embedding_increase = float(
                raised_embedding["common_Spe_leakage_abs"]
            ) - float(capped_embedding["common_Spe_leakage_abs"])
            checks["decoder_attenuates_common_token_leakage_increase"] = (
                full_increase < embedding_increase
            )
        else:
            checks["decoder_attenuates_common_token_leakage_increase"] = False
    return {
        "schema_version": 1,
        "kind": "mcf_context_gating_frequency_factorial_aggregate",
        "rows": rows,
        "receipt_validity": receipt_validity,
        "writer_configuration_matches_registry": configuration_matches,
        "acceptance": {"checks": checks, "passed": all(checks.values())},
        "interpretation": (
            "Passing supports the narrow mechanism claim that contextual decoding "
            "attenuates the extra common-token leakage induced by removing and "
            "raising the writer frequency cap."
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    registry = _load(Path(args.registry).resolve())
    acceptance = registry["primary_acceptance"]
    writer_rows = registry["frequency_cap_factorial"]["writer_conditions"]
    expected_writer_conditions = {
        str(row["name"]): row for row in writer_rows if isinstance(row, Mapping)
    }
    payloads = {
        condition: _load(root / condition / "comparison" / "official_components.json")
        for condition in CONDITIONS
        if (root / condition / "comparison" / "official_components.json").is_file()
    }
    report = build_aggregate(
        payloads,
        expected_writer_conditions=expected_writer_conditions,
        max_abs_spe_delta=float(acceptance["max_abs_retain_Spe_delta"]),
        max_abs_ppl_percent_delta=float(acceptance["max_abs_PPL_percent_delta"]),
    )
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frequency_factorial.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if report["rows"]:
        columns = list(report["rows"][0])
        with (out_dir / "frequency_factorial.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(report["rows"])
    lines = [
        "# Context gating × writer frequency cap",
        "",
        f"Overall: **{'PASS' if report['acceptance']['passed'] else 'FAIL'}**",
        "",
        "| Writer | Decoder | Eff | Gen | Retain Spe Δ | PPL Δ% | Common n | Common Spe leakage |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        leakage = row["common_Spe_leakage_abs"]
        leakage_text = f"{float(leakage):.3f}" if leakage is not None else "-"
        lines.append(
            f"| {row['writer_condition']} | {row['decoder_condition']} | "
            f"{row['forget_Eff']:.3f} | {row['forget_Gen']:.3f} | "
            f"{row['retain_Spe_delta']:+.3f} | {row['PPL_percent_delta']:+.3f} | "
            f"{row['common_record_count']} | "
            f"{leakage_text} |"
        )
    lines.extend(["", "## Registered checks", ""])
    for key, passed in report["acceptance"]["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{key}`")
    (out_dir / "frequency_factorial.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"frequency factorial: {out_dir / 'frequency_factorial.md'}")


if __name__ == "__main__":
    main()
