#!/usr/bin/env python3
"""Measure frozen Stage-1 marker portability on official forget paraphrases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

import gagd_compare as gagd
import mcf_compositional_marker_write_read as compositional
import mcf_embedding_keyed_neuron_core as neuron_core
from mcf_zero_unlearn_official_eval import load_official_eval_records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--stage1-state", required=True)
    parser.add_argument("--stage1-report", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--unlearn-num", type=int, default=50)
    parser.add_argument(
        "--sample-mode", choices=("official", "first"), default="official"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--expected-amplitude-threshold", type=float, default=4.5)
    parser.add_argument("--minimum-global-complete-fraction", type=float, default=0.95)
    parser.add_argument(
        "--minimum-per-record-complete-fraction", type=float, default=0.8
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    value = parser.parse_args(list(argv) if argv is not None else None)
    if int(value.batch_size) <= 0:
        parser.error("--batch-size must be positive")
    if float(value.expected_amplitude_threshold) < 0:
        parser.error("--expected-amplitude-threshold must be non-negative")
    for name in (
        "minimum_global_complete_fraction",
        "minimum_per_record_complete_fraction",
    ):
        if not 0.0 <= float(getattr(value, name)) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must lie in [0, 1]")
    return value


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {"n": 0}
    return {
        "n": len(finite),
        "min": finite[0],
        "p10": finite[max(0, len(finite) // 10)],
        "median": finite[len(finite) // 2],
        "p90": finite[min(len(finite) - 1, 9 * len(finite) // 10)],
        "max": finite[-1],
        "mean": sum(finite) / len(finite),
    }


def summarize_portability(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_global_complete_fraction: float,
    minimum_per_record_complete_fraction: float,
) -> Dict[str, Any]:
    prompt_rows = [prompt for row in rows for prompt in row["prompts"]]
    if not prompt_rows:
        raise ValueError("portability summary needs prompts")
    global_fraction = sum(bool(row["complete"]) for row in prompt_rows) / len(
        prompt_rows
    )
    per_record_fractions = [
        sum(bool(prompt["complete"]) for prompt in row["prompts"]) / len(row["prompts"])
        for row in rows
    ]
    checks = {
        "global_complete_fraction": global_fraction
        >= float(minimum_global_complete_fraction),
        "minimum_per_record_complete_fraction": min(per_record_fractions)
        >= float(minimum_per_record_complete_fraction),
    }
    by_type: Dict[str, Dict[str, Any]] = {}
    for prompt_type in ("rewrite", "paraphrase"):
        selected = [row for row in prompt_rows if row["prompt_type"] == prompt_type]
        by_type[prompt_type] = {
            "prompt_count": len(selected),
            "complete_count": sum(bool(row["complete"]) for row in selected),
            "complete_fraction": (
                sum(bool(row["complete"]) for row in selected) / len(selected)
                if selected
                else None
            ),
            "own_marker_amplitude": _distribution(
                [float(row["own_marker_amplitude"]) for row in selected]
            ),
        }
    return {
        "prompt_count": len(prompt_rows),
        "record_count": len(rows),
        "global_complete_fraction": global_fraction,
        "per_record_complete_fraction": _distribution(per_record_fractions),
        "by_prompt_type": by_type,
        "acceptance": {"checks": checks, "passed": all(checks.values())},
    }


def _record_case_id(record: Mapping[str, Any]) -> int:
    if "case_id" not in record:
        raise RuntimeError("official MCF record lacks case_id")
    return int(record["case_id"])


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    stage1_path = Path(args.stage1_state).resolve()
    report_path = Path(args.stage1_report).resolve()
    for path in (
        Path(args.model_path).resolve(),
        stage1_path,
        report_path,
        Path(args.mcf_path).resolve(),
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    state = torch.load(stage1_path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise RuntimeError("Stage-1 state must be a mapping")
    writer_report = _load_json(report_path)
    criterion = writer_report.get("criterion")
    if not isinstance(criterion, Mapping):
        raise RuntimeError("Stage-1 report lacks its marker criterion")
    amplitude_threshold = float(criterion["positive_amplitude_min"])
    threshold_checks = {
        "report_matches_preregistered_threshold": math.isclose(
            amplitude_threshold,
            float(args.expected_amplitude_threshold),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "state_matches_report_when_recorded": bool(
            state.get("writer_positive_amplitude_threshold") is None
            or math.isclose(
                float(state["writer_positive_amplitude_threshold"]),
                amplitude_threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
    }
    if not all(threshold_checks.values()):
        raise RuntimeError(
            "frozen writer amplitude threshold is not bound to the preregistration"
        )

    forget_records, _retain = load_official_eval_records(
        args.mcf_path,
        int(args.unlearn_num),
        1,
        int(args.seed),
        str(args.sample_mode),
    )
    official_by_case = {_record_case_id(record): record for record in forget_records}
    case_ids = [int(value) for value in state.get("case_ids", [])]
    if set(case_ids) != set(official_by_case):
        raise RuntimeError("frozen writer case IDs do not match official forget split")
    markers = state.get("markers")
    if not isinstance(markers, Mapping):
        raise RuntimeError("Stage-1 state lacks marker directions")
    marker_rows: List[torch.Tensor] = []
    for case_id in case_ids:
        marker = markers.get(case_id, markers.get(str(case_id)))
        if not isinstance(marker, torch.Tensor) or marker.ndim != 1:
            raise RuntimeError(f"Stage-1 marker missing for case {case_id}")
        marker_rows.append(marker.detach().float().cpu())
    marker_bank = torch.stack(marker_rows)

    namespace = argparse.Namespace(
        model_path=str(Path(args.model_path).resolve()),
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    model, tokenizer = gagd.load_model_and_tokenizer(namespace, for_training=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.config.use_cache = False
    device = gagd.first_device(model)
    input_layer = model.get_input_embeddings()
    if input_layer is None:
        raise RuntimeError("model lacks input embeddings")
    selected_rows = [int(value) for value in state.get("selected_embedding_rows", [])]
    embedding_delta = state.get("embedding_delta")
    if not selected_rows or not isinstance(embedding_delta, torch.Tensor):
        raise RuntimeError("Stage-1 state lacks sparse embedding delta")
    writer = neuron_core.ToggleableEmbeddingDelta(
        input_layer, selected_rows, embedding_delta
    )

    output_rows: List[Dict[str, Any]] = []
    try:
        for record_index, case_id in enumerate(case_ids):
            record = official_by_case[case_id]
            rewrite = record["requested_rewrite"]
            prompt_entries = [
                ("rewrite", str(rewrite["prompt"]).format(str(rewrite["subject"])))
            ]
            prompt_entries.extend(
                ("paraphrase", str(prompt))
                for prompt in record.get("paraphrase_prompts", [])
            )
            prompts = [prompt for _kind, prompt in prompt_entries]
            writer.enabled = False
            base_hidden = compositional.batched_last_hidden_only(
                model,
                tokenizer,
                prompts,
                device,
                batch_size=int(args.batch_size),
            )
            writer.enabled = True
            writer_hidden = compositional.batched_last_hidden_only(
                model,
                tokenizer,
                prompts,
                device,
                batch_size=int(args.batch_size),
            )
            displacement = writer_hidden - base_hidden
            amplitudes = displacement @ marker_bank.T
            own = amplitudes[:, record_index]
            if amplitudes.shape[1] > 1:
                peer_mask = torch.ones(amplitudes.shape[1], dtype=torch.bool)
                peer_mask[record_index] = False
                peer_abs_max = amplitudes[:, peer_mask].abs().max(dim=1).values
            else:
                peer_abs_max = torch.zeros_like(own)
            prompt_rows = [
                {
                    "prompt_type": prompt_entries[index][0],
                    "own_marker_amplitude": float(own[index]),
                    "peer_marker_abs_max": float(peer_abs_max[index]),
                    "complete": float(own[index]) >= amplitude_threshold,
                }
                for index in range(len(prompt_entries))
            ]
            output_rows.append(
                {
                    "case_id": case_id,
                    "prompt_count": len(prompt_rows),
                    "complete_fraction": sum(
                        bool(row["complete"]) for row in prompt_rows
                    )
                    / len(prompt_rows),
                    "prompts": prompt_rows,
                }
            )
            print(
                f"case {case_id}: "
                f"{sum(bool(row['complete']) for row in prompt_rows)}/{len(prompt_rows)} complete"
            )
    finally:
        writer.remove()

    summary = summarize_portability(
        output_rows,
        minimum_global_complete_fraction=float(args.minimum_global_complete_fraction),
        minimum_per_record_complete_fraction=float(
            args.minimum_per_record_complete_fraction
        ),
    )
    payload = {
        "schema_version": 1,
        "kind": "mcf_frozen_stage1_writer_official_portability_audit",
        "dataset": "MCF",
        "seed": int(args.seed),
        "unlearn_num": int(args.unlearn_num),
        "sample_mode": str(args.sample_mode),
        "writer_mode": "embedding_keyed",
        "amplitude_threshold": amplitude_threshold,
        "expected_amplitude_threshold": float(args.expected_amplitude_threshold),
        "threshold_binding_checks": threshold_checks,
        "threshold_binding_passed": all(threshold_checks.values()),
        "threshold_source": str(report_path),
        "decoder_loaded": False,
        "decoder_capacity_used": False,
        "writer_parameters_updated": False,
        "used_for_training_checkpoint_selection_or_retry": False,
        "interpretation_boundary": (
            "This measures completion of the claimed Stage-1 marker pathway. It "
            "does not mathematically upper-bound every possible nonlinear decoder."
        ),
        **summary,
        "per_record": output_rows,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["acceptance"], indent=2))
    print(f"writer portability audit: {out}")


if __name__ == "__main__":
    main()
