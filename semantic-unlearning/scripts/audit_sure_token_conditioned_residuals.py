#!/usr/bin/env python3
"""Audit token-conditioned utility coverage and unresolved SURE Stage-2 cases.

This is a read-only diagnostic over an existing learner directory. It does not
train, select, or save a checkpoint, and it never reads official replacement
targets, paraphrases, neighborhood prompts, benchmark retain examples, or PPL
text. It answers two questions using artifacts already produced by the shared
MCF/ZsRE learner:

1. Did the Wikipedia candidate reservoir contain contexts where each edited
   token had meaningful Base probability?
2. Which direct token cases remain unresolved after a saved Stage-2 residual,
   and did the requested rank produce real per-row capacity?
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

import gagd_compare as gagd
import sure_canonical_core as core
import sure_minimal_two_stage as learner
import sure_shared_suppression as shared


METHOD = "sure_token_conditioning_and_residual_audit_v1"
DEFAULT_SCALES = "0,1,1.25"
DEFAULT_TOP_COUNTS = (10, 128)
DEFAULT_PROBABILITY_THRESHOLDS = (1e-4, 1e-3, 1e-2)


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_positive_ints(text: str) -> Tuple[int, ...]:
    values: List[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("top counts must be positive")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("at least one top count is required")
    return tuple(values)


def parse_positive_floats(text: str) -> Tuple[float, ...]:
    values: List[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("probability thresholds must be finite and positive")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("at least one probability threshold is required")
    return tuple(values)


def probability_row_reports(
    probabilities: torch.Tensor,
    row_ids: Sequence[int],
    *,
    token_texts: Sequence[str] | None = None,
    top_counts: Sequence[int] = DEFAULT_TOP_COUNTS,
    thresholds: Sequence[float] = DEFAULT_PROBABILITY_THRESHOLDS,
) -> List[Dict[str, Any]]:
    values = probabilities.detach().cpu().float().contiguous()
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("candidate probabilities must be non-empty [prompt,row]")
    if values.shape[1] != len(row_ids):
        raise ValueError("candidate probabilities do not align with row ids")
    if token_texts is not None and len(token_texts) != len(row_ids):
        raise ValueError("token texts do not align with row ids")
    if not torch.isfinite(values).all() or bool((values < 0).any()):
        raise ValueError("candidate probabilities are invalid")

    reports: List[Dict[str, Any]] = []
    for column, token_id in enumerate(row_ids):
        column_values = values[:, column]
        top_means: Dict[str, float] = {}
        for count in top_counts:
            take = min(int(count), int(column_values.numel()))
            if take <= 0:
                raise ValueError("top counts must be positive")
            top_means[str(int(count))] = float(
                column_values.topk(take, largest=True).values.mean().item()
            )
        counts_above = {
            format(float(threshold), ".8g"): int(
                (column_values > float(threshold)).sum().item()
            )
            for threshold in thresholds
        }
        reports.append(
            {
                "token_id": int(token_id),
                "token_text": (
                    None if token_texts is None else str(token_texts[column])
                ),
                "maximum_base_probability": float(column_values.max().item()),
                "mean_base_probability": float(column_values.mean().item()),
                "top_mean_base_probability": top_means,
                "top_actual_context_count": {
                    str(int(count)): min(int(count), int(column_values.numel()))
                    for count in top_counts
                },
                "counts_above_probability": counts_above,
            }
        )
    return reports


def coverage_summary(
    per_row: Sequence[Mapping[str, Any]], thresholds: Sequence[float]
) -> Dict[str, Any]:
    if not per_row:
        raise ValueError("per-row coverage cannot be empty")
    return {
        "edited_row_count": len(per_row),
        "rows_with_maximum_probability_above": {
            format(float(threshold), ".8g"): sum(
                float(row["maximum_base_probability"]) > float(threshold)
                for row in per_row
            )
            for threshold in thresholds
        },
        "rows_with_zero_contexts_above": {
            format(float(threshold), ".8g"): sum(
                int(row["counts_above_probability"][format(float(threshold), ".8g")])
                == 0
                for row in per_row
            )
            for threshold in thresholds
        },
    }


def constraint_snapshot(
    state: Mapping[str, torch.Tensor],
    *,
    required_margin: float,
    required_nll: float,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    margins = state["logit_margin"].detach().cpu().float()
    nll = state["sensitive_nll_increase"].detach().cpu().float()
    failures = shared.failure_mask(
        margins,
        nll,
        required_logit_margin=required_margin,
        required_nll_increase=required_nll,
    )
    failure_indices = torch.where(failures)[0].tolist()
    report = {
        "direct_failures": int(failures.sum().item()),
        "failure_case_indices": [int(index) for index in failure_indices],
        "minimum_logit_margin": float(margins.min().item()),
        "minimum_sensitive_nll_increase": float(nll.min().item()),
        "constraint_shortfall_sum": float(
            (
                torch.relu(torch.tensor(required_margin) - margins)
                + torch.relu(torch.tensor(required_nll) - nll)
            )
            .sum()
            .item()
        ),
    }
    return report, margins, nll, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learner-dir", required=True)
    parser.add_argument(
        "--protocol-dir",
        help="Defaults to the learner directory's sibling protocol directory",
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--scales", default=DEFAULT_SCALES)
    parser.add_argument(
        "--top-counts",
        default=",".join(str(value) for value in DEFAULT_TOP_COUNTS),
    )
    parser.add_argument(
        "--probability-thresholds",
        default=",".join(str(value) for value in DEFAULT_PROBABILITY_THRESHOLDS),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rank <= 0 or args.batch_size <= 0:
        raise ValueError("rank and batch size must be positive")
    scales = core.parse_scales(args.scales)
    top_counts = parse_positive_ints(args.top_counts)
    thresholds = parse_positive_floats(args.probability_thresholds)

    run = Path(args.learner_dir).resolve()
    protocol = (
        Path(args.protocol_dir).resolve()
        if args.protocol_dir
        else run.parent / "protocol"
    )
    output_path = (
        Path(args.output_json).resolve()
        if args.output_json
        else run / f"stage2_rank{args.rank}_token_conditioning_audit.json"
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Audit output exists: {output_path}; pass --overwrite explicitly"
        )

    required_files = {
        "stage1_checkpoint": run / "stage1_checkpoint" / "config.json",
        "probabilities": run / "base_wikipedia_selected_probabilities.pt",
        "basis": run / f"stage2_rank{args.rank}_basis_reports.json",
        "residual": run / f"stage2_rank{args.rank}_unscaled_residual.pt",
        "base_logits": run / "base_sensitive_case_logits.pt",
        "architecture": run / "architecture_lock.json",
        "forget": protocol / "training_visible_forget.json",
        "manifest": protocol / "split_manifest.json",
    }
    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required audit inputs: " + ", ".join(missing))

    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)
    model_args = argparse.Namespace(
        model_path=str(run / "stage1_checkpoint"),
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(model_args, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model.eval()
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tok)

    probability_payload = safe_torch_load(required_files["probabilities"])
    if not isinstance(probability_payload, Mapping):
        raise ValueError("selected-probability artifact must be a mapping")
    probabilities = probability_payload.get("candidate_probabilities")
    row_ids_raw = probability_payload.get("row_ids")
    train_indices = probability_payload.get("train_indices")
    guard_indices = probability_payload.get("guard_indices")
    if not isinstance(probabilities, torch.Tensor):
        raise ValueError("selected-probability artifact lacks candidate probabilities")
    if not isinstance(row_ids_raw, Sequence):
        raise ValueError("selected-probability artifact lacks row ids")
    if not isinstance(train_indices, torch.Tensor) or not isinstance(
        guard_indices, torch.Tensor
    ):
        raise ValueError("selected-probability artifact lacks train/guard indices")
    row_ids = [int(value) for value in row_ids_raw]
    token_texts = [tok.decode([token_id]) for token_id in row_ids]
    probability_rows = probability_row_reports(
        probabilities,
        row_ids,
        token_texts=token_texts,
        top_counts=top_counts,
        thresholds=thresholds,
    )
    stored_pool_report_path = run / "token_conditioned_utility_pool_report.json"
    stored_pool_report = (
        load_json(stored_pool_report_path)
        if stored_pool_report_path.exists()
        else None
    )
    utility_report = {
        "candidate_prompt_count": int(probabilities.shape[0]),
        "edited_row_count": int(probabilities.shape[1]),
        "selected_train_prompt_count": int(train_indices.numel()),
        "selected_guard_prompt_count": int(guard_indices.numel()),
        "top_counts": list(top_counts),
        "probability_thresholds": list(thresholds),
        "coverage_summary": coverage_summary(probability_rows, thresholds),
        "per_row_candidate_coverage": probability_rows,
        "stored_disjoint_pool_report": stored_pool_report,
    }

    manifest = load_json(required_files["manifest"])
    records = load_json(required_files["forget"])
    if not isinstance(manifest, Mapping) or not isinstance(records, list):
        raise ValueError("protocol manifest/training-visible forget data are invalid")
    adapter = learner.adapter_contract(manifest)
    sensitive_field = adapter["sensitive_answer_field"]
    cases = core.expand_sensitive_cases(
        records,
        tok,
        sensitive_field=sensitive_field,
        llama_like=llama_like,
    )
    target_ids = core.official_target_ids(
        tok, cases, llama_like=llama_like, device=device
    ).detach().cpu()
    base_logits = safe_torch_load(required_files["base_logits"])
    if not isinstance(base_logits, torch.Tensor):
        raise ValueError("Base-logit artifact must be a tensor")

    residual_payload = safe_torch_load(required_files["residual"])
    if not isinstance(residual_payload, Mapping):
        raise ValueError("Stage-2 residual artifact must be a mapping")
    active_ids_raw = residual_payload.get("row_ids")
    residual_delta = residual_payload.get("delta")
    if not isinstance(active_ids_raw, Sequence) or not isinstance(
        residual_delta, torch.Tensor
    ):
        raise ValueError("Stage-2 residual artifact lacks row ids/delta")
    active_ids = [int(value) for value in active_ids_raw]
    residual_delta = residual_delta.detach().cpu().float().contiguous()

    basis_rows = load_json(required_files["basis"])
    if not isinstance(basis_rows, list):
        raise ValueError("Stage-2 basis report must be a list")
    basis_by_token = {int(row["token_id"]): dict(row) for row in basis_rows}
    architecture = load_json(required_files["architecture"])
    shared_architecture = architecture["shared_architecture_parameters"]
    required_margin = float(shared_architecture["constraint_margin"])
    required_nll = float(shared_architecture["min_sensitive_nll_increase"])

    output_layer = model.get_output_embeddings()
    if output_layer is None:
        raise ValueError("Stage-1 checkpoint has no output embeddings")
    snapshots: Dict[float, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    scale_reports: List[Dict[str, Any]] = []
    for scale in scales:
        with learner.temporary_materialized_output_delta(
            output_layer,
            active_ids,
            residual_delta * float(scale),
        ):
            state = shared.evaluate_shared_constraints(
                model,
                tok,
                cases,
                base_logits,
                llama_like=llama_like,
                device=device,
                batch_size=args.batch_size,
            )
        scale_report, margins, nll, failures = constraint_snapshot(
            state,
            required_margin=required_margin,
            required_nll=required_nll,
        )
        scale_report.update({"scale": float(scale), "rank": int(args.rank)})
        scale_reports.append(scale_report)
        snapshots[float(scale)] = (margins, nll, failures)

    focus_scales = [float(scale) for scale in scales if float(scale) > 0]
    if not focus_scales:
        focus_scales = [float(scale) for scale in scales]
    failure_sets = [
        set(torch.where(snapshots[scale][2])[0].tolist()) for scale in focus_scales
    ]
    focused_indices = sorted(set().union(*failure_sets))
    focused_cases: List[Dict[str, Any]] = []
    for index in focused_indices:
        case = cases[index]
        token_id = int(target_ids[index].item())
        rewrite = records[case.record_position]["requested_rewrite"]
        basis = basis_by_token.get(token_id, {})
        per_scale: Dict[str, Any] = {}
        for scale in scales:
            margins, nll, failures = snapshots[float(scale)]
            margin_value = float(margins[index].item())
            nll_value = float(nll[index].item())
            per_scale[format(float(scale), ".8g")] = {
                "logit_margin": margin_value,
                "sensitive_nll_increase": nll_value,
                "margin_shortfall": max(0.0, required_margin - margin_value),
                "nll_shortfall": max(0.0, required_nll - nll_value),
                "failed": bool(failures[index].item()),
            }
        focused_cases.append(
            {
                "case_index": int(index),
                "case_id": int(case.case_id),
                "record_position": int(case.record_position),
                "token_index": int(case.token_index),
                "subject": str(rewrite.get("subject", "")),
                "sensitive_answer": str(rewrite[sensitive_field]["str"]),
                "teacher_forced_prompt": str(case.prompt),
                "token_id": token_id,
                "token_text": str(tok.decode([token_id])),
                "actual_contrastive_rank": basis.get("actual_contrastive_rank"),
                "forget_context_count_for_row": basis.get("forget_context_count"),
                "per_scale": per_scale,
            }
        )

    failure_identity_stable = all(
        failure_set == failure_sets[0] for failure_set in failure_sets[1:]
    )
    focused_token_ids = [int(row["token_id"]) for row in focused_cases]
    repeated_failure_token_ids = sorted(
        token_id
        for token_id in set(focused_token_ids)
        if focused_token_ids.count(token_id) > 1
    )
    actual_ranks = [
        int(row["actual_contrastive_rank"])
        for row in basis_rows
        if row.get("actual_contrastive_rank") is not None
    ]
    residual_report = {
        "rank_requested": int(args.rank),
        "active_row_ids": active_ids,
        "required_logit_margin": required_margin,
        "required_sensitive_nll_increase": required_nll,
        "scales": [float(scale) for scale in scales],
        "scale_reports": scale_reports,
        "failure_identity_focus_scales": focus_scales,
        "failure_identity_stable_across_focus_scales": failure_identity_stable,
        "focused_failure_case_count": len(focused_cases),
        "focused_failure_cases": focused_cases,
        "repeated_failure_token_ids": repeated_failure_token_ids,
        "basis_actual_rank_min": min(actual_ranks) if actual_ranks else None,
        "basis_actual_rank_max": max(actual_ranks) if actual_ranks else None,
        "basis_reports": basis_rows,
    }

    report = {
        "schema_version": 1,
        "method": METHOD,
        "dataset_adapter": manifest.get("dataset"),
        "learner_protocol": manifest.get("protocol"),
        "learner_dir": str(run),
        "protocol_dir": str(protocol),
        "diagnostic_is_read_only_over_checkpoint": True,
        "training_or_checkpoint_selection_performed": False,
        "official_replacement_targets_seen": False,
        "official_paraphrases_seen": False,
        "benchmark_retain_examples_seen": 0,
        "utility_coverage": utility_report,
        "stage2_residual": residual_report,
    }
    core.write_json(output_path, report)

    print("SURE token-conditioning/residual audit:", output_path)
    print(
        "candidate states / edited rows:",
        probabilities.shape[0],
        probabilities.shape[1],
    )
    print("rows with zero contexts above thresholds:")
    for key, value in utility_report["coverage_summary"][
        "rows_with_zero_contexts_above"
    ].items():
        print(f"  p > {key}: {value}/{len(probability_rows)}")
    print(
        "Stage-2 actual rank range:",
        residual_report["basis_actual_rank_min"],
        residual_report["basis_actual_rank_max"],
    )
    for scale_report in scale_reports:
        print(
            "scale",
            scale_report["scale"],
            "failures",
            scale_report["direct_failures"],
            "min margin",
            scale_report["minimum_logit_margin"],
            "min dNLL",
            scale_report["minimum_sensitive_nll_increase"],
        )
    print("failure identities stable:", failure_identity_stable)
    print("focused unresolved cases:")
    for row in focused_cases:
        print(
            " ",
            row["case_index"],
            "case_id",
            row["case_id"],
            "token",
            repr(row["token_text"]),
            "tid",
            row["token_id"],
            "actual rank",
            row["actual_contrastive_rank"],
            "contexts",
            row["forget_context_count_for_row"],
        )


if __name__ == "__main__":
    main()
