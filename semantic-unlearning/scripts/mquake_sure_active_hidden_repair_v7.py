#!/usr/bin/env python3
"""SURE-MQuAKE V7 Stage 2: residual active-row repair, Rank 0 or Rank R.

Input is the sparse sensitive-row GA/GD Stage-1 checkpoint.  This stage is
aligned to MQuAKE's official teacher-forced token-level Eff definition:

1. score every direct sensitive token from the same 50 visible forget instances;
2. mark as residual-active every token whose best competitor does not beat the
   sensitive token by ``--target-logit-margin``;
3. edit only the union of sensitive LM-head rows belonging to residual-active
   token cases;
4. enforce the target margin on ALL visible direct sensitive token cases;
5. Rank 0 learns unrestricted selected-row deltas;
6. Rank R restricts every selected-row delta to an R-dimensional basis built
   from hidden states of ALL visible direct sensitive token cases;
7. materialize BF16 and fail closed unless the all-visible exact audit passes.

No retain records, atomic questions, multi-hop questions, PPL text, or MQuAKE
counterfactual targets are used for active selection, basis construction,
optimization, stopping, or materialization selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

import gagd_active_case_repair as active
import gagd_compare as gagd
import mquake_forget_only_active_repair as locked
import mquake_sure_sparse_lm_gagd_v7 as stage1
import mquake_zero_unlearn_official_eval as mquake


METHOD = "SURE-MQuAKE-v7-active-sensitive-row-hidden-repair"
PROTOCOL = "mquake_zerounlearn_forget_only_locked_probes"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="MQuAKE V7 Stage-1 checkpoint")
    p.add_argument("--reference-model-path", required=True, help="Protected pretrained Base")
    p.add_argument("--repair-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50, help="MQuAKE instance count")
    p.add_argument(
        "--target-logit-margin",
        type=float,
        default=0.25,
        help="Exact final competitor-minus-sensitive margin required on every visible token.",
    )
    p.add_argument(
        "--bf16-buffer-margin",
        type=float,
        default=0.05,
        help="Extra cached margin optimized before BF16 materialization.",
    )
    p.add_argument(
        "--repair-rank",
        type=int,
        default=256,
        help="0=unrestricted selected active rows; >0=all-visible hidden basis rank.",
    )
    p.add_argument("--repair-steps", type=int, default=5000)
    p.add_argument("--repair-lr", type=float, default=5e-3)
    p.add_argument("--repair-optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--forget-hinge-weight", type=float, default=100.0)
    p.add_argument("--hardest-forget-hinge-weight", type=float, default=25.0)
    p.add_argument("--delta-l2-lambda", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--boundary-bisection-steps", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def make_optimizer(module: torch.nn.Module, name: str, lr: float) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(module.parameters(), lr=lr)
    if name == "adam":
        return torch.optim.Adam(module.parameters(), lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=0.0)
    raise ValueError(name)


def cached_metrics(
    stacked: Mapping[str, torch.Tensor],
    delta_rows: torch.Tensor,
    *,
    target_margin: float,
    required_margin: float,
) -> Dict[str, Any]:
    margins = stage1.competitor_minus_sensitive_margins(stacked, delta_rows)
    nll = stage1.token_nlls_from_delta(stacked, delta_rows)
    return {
        "official_active_sensitive_token_count": int((margins <= 0.0).sum().item()),
        "target_margin_unmet_token_count": int((margins < target_margin).sum().item()),
        "buffered_margin_unmet_token_count": int((margins < required_margin).sum().item()),
        "minimum_competitor_minus_sensitive_margin": float(margins.min().detach().cpu()),
        "mean_competitor_minus_sensitive_margin": float(margins.mean().detach().cpu()),
        "sensitive_token_probability_mean": float(torch.exp(-nll).mean().detach().cpu()),
        "sensitive_token_probability_max": float(torch.exp(-nll).max().detach().cpu()),
        "selected_lm_head_delta_norm": float(delta_rows.norm().detach().cpu()),
    }


def priority(metrics: Mapping[str, Any]) -> Tuple[int, int, float, float]:
    return (
        int(metrics["target_margin_unmet_token_count"]),
        int(metrics["buffered_margin_unmet_token_count"]),
        -float(metrics["minimum_competitor_minus_sensitive_margin"]),
        float(metrics["selected_lm_head_delta_norm"]),
    )


def feasible(
    stacked: Mapping[str, torch.Tensor],
    delta_rows: torch.Tensor,
    required_margin: float,
) -> bool:
    margins = stage1.competitor_minus_sensitive_margins(stacked, delta_rows)
    return bool(torch.all(margins >= required_margin).item())


def boundary_bisect(
    stacked: Mapping[str, torch.Tensor],
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    required_margin: float,
    iterations: int,
) -> torch.Tensor:
    if feasible(stacked, low, required_margin):
        return low.detach().clone()
    if not feasible(stacked, high, required_margin):
        raise ValueError("boundary high endpoint is not feasible")
    lo = low.detach().clone()
    hi = high.detach().clone()
    for _ in range(iterations):
        mid = lo + 0.5 * (hi - lo)
        if feasible(stacked, mid, required_margin):
            hi = mid.detach().clone()
        else:
            lo = mid.detach().clone()
    return hi


def validate_stage1_reference(model_path: Path, reference_path: Path) -> None:
    config_path = model_path.parent / "config_used.json"
    if not config_path.is_file():
        return
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("method") != stage1.METHOD:
        raise RuntimeError("Stage2 input checkpoint is not SURE-MQuAKE V7 Stage1")
    configured = payload.get("model_path")
    if configured:
        configured_path = Path(str(configured)).expanduser()
        try:
            if configured_path.resolve() != reference_path.resolve():
                raise RuntimeError(
                    "Stage2 --reference-model-path does not match Stage1 protected Base"
                )
        except FileNotFoundError:
            pass


def main() -> None:
    a = parse_args()
    if a.forget_num <= 0 or a.repair_steps <= 0 or a.batch_size <= 0:
        raise ValueError("forget-num, repair-steps, and batch-size must be positive")
    if a.target_logit_margin < 0 or a.bf16_buffer_margin < 0:
        raise ValueError("target/buffer margins must be non-negative")
    if a.repair_rank < 0 or a.repair_lr <= 0:
        raise ValueError("repair-rank must be >=0 and repair-lr positive")
    if a.forget_hinge_weight <= 0 or a.hardest_forget_hinge_weight < 0:
        raise ValueError("invalid Stage2 hinge weights")
    if a.delta_l2_lambda < 0 or a.grad_clip < 0 or a.boundary_bisection_steps <= 0:
        raise ValueError("invalid Stage2 regularization/bisection controls")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    model_path = Path(a.model_path).resolve()
    reference_path = Path(a.reference_model_path).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if not reference_path.is_dir():
        raise FileNotFoundError(reference_path)
    validate_stage1_reference(model_path, reference_path)

    visible_path = Path(a.repair_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, split_manifest = locked.load_locked_records(
        visible_path,
        manifest_path,
        a.forget_num,
        a.seed,
    )

    model_args = argparse.Namespace(
        model_path=str(model_path),
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(model_args, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    output_layer = active.freeze_model_for_output_repair(model)
    output_weight = output_layer.weight
    input_weight = model.get_input_embeddings().weight
    input_pointer = int(input_weight.data_ptr())
    input_version = int(input_weight._version)
    device = gagd.first_device(model)
    llama_like = mquake.is_llama_like(model, tok)

    cases = stage1.direct_rewrite_cases(records, tok, llama_like=llama_like)
    before_reports, before_summary = stage1.exact_materialized_reports(
        model,
        tok,
        cases,
        device=device,
        llama_like=llama_like,
        batch_size=a.batch_size,
        target_margin=a.target_logit_margin,
    )

    active_positions = [
        idx
        for idx, report in enumerate(before_reports)
        if float(report["competitor_minus_sensitive_margin"]) < a.target_logit_margin
    ]
    selected_ids = sorted(
        {
            int(before_reports[idx]["target_token_id"])
            for idx in active_positions
        }
    )

    root = gagd.resolve_output_path(a.output_dir)
    ckpt = root / "checkpoint"
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "all_visible_tokens_before.jsonl", before_reports)
    write_json(
        root / "selected_active_rows.json",
        {
            "active_case_definition": (
                "direct rewrite sensitive token with competitor-minus-sensitive "
                f"margin < {a.target_logit_margin:g} after Stage1"
            ),
            "active_sensitive_token_case_count": len(active_positions),
            "active_positions": active_positions,
            "selected_lm_head_row_count": len(selected_ids),
            "selected_lm_head_token_ids": selected_ids,
            "selected_lm_head_tokens": {
                str(token_id): stage1.decoded_token(tok, token_id) for token_id in selected_ids
            },
            "row_policy": "union of sensitive target-token rows from residual-active token cases only",
            "basis_source": "hidden states from ALL visible direct sensitive token cases",
            "repair_rank_requested": int(a.repair_rank),
            "retain_or_heldout_data_consulted": False,
        },
    )

    required_margin = float(a.target_logit_margin + a.bf16_buffer_margin)
    selected_candidate = torch.zeros(
        (len(selected_ids), int(output_weight.shape[1])),
        dtype=torch.float32,
        device=device,
    )
    actual_rank = 0
    optimizer_crossing_step: int | None = None
    best_metrics: Dict[str, Any] | None = None
    crossing_metrics: Dict[str, Any] | None = None
    logs: List[Dict[str, Any]] = []

    if active_positions:
        selected_tensor = torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device)
        baseline_rows = output_weight.index_select(0, selected_tensor).detach().clone()
        caches = stage1.build_token_delta_caches(
            model,
            tok,
            cases,
            selected_ids,
            device=device,
            llama_like=llama_like,
            batch_size=a.batch_size,
            desc=f"cache MQuAKE V7 Stage2 rank{a.repair_rank}",
        )
        stacked = stage1.stack_cache_fields(caches, device=device)

        direction_basis = None
        if a.repair_rank > 0:
            direction_basis = active.orthonormal_row_basis(
                stacked["hidden"],
                max_rank=a.repair_rank,
            )
            if direction_basis.numel() == 0:
                raise RuntimeError("all-visible MQuAKE hidden states produced zero-rank basis")
            actual_rank = int(direction_basis.shape[0])

        module = active.SelectedRowDelta(
            len(selected_ids),
            int(output_weight.shape[1]),
            direction_basis=direction_basis,
            retained_basis=None,
            device=device,
        )
        optimizer = make_optimizer(module, a.repair_optimizer, a.repair_lr)
        zero = torch.zeros_like(selected_candidate)
        previous = zero.detach().clone()
        best_candidate = zero.detach().clone()
        best_metrics = cached_metrics(
            stacked,
            zero,
            target_margin=a.target_logit_margin,
            required_margin=required_margin,
        )
        crossing_low: torch.Tensor | None = None
        crossing_high: torch.Tensor | None = None

        print(
            f"===== MQUAKE V7 STAGE2 rank={a.repair_rank} actual_rank={actual_rank} "
            f"active_tokens={len(active_positions)} active_rows={len(selected_ids)} ====="
        )

        for step in range(1, a.repair_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            delta = module.effective_delta()
            margins = stage1.competitor_minus_sensitive_margins(stacked, delta)
            errors = torch.relu(required_margin - margins)
            loss = (
                a.forget_hinge_weight * errors.square().mean()
                + a.hardest_forget_hinge_weight * errors.square().max()
                + a.delta_l2_lambda * delta.square().sum()
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite MQuAKE V7 Stage2 loss at step {step}")
            loss.backward()
            grad_norm_value = None
            if a.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), a.grad_clip)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite Stage2 gradient norm at step {step}")
                grad_norm_value = float(grad_norm.detach().cpu())
            optimizer.step()

            with torch.no_grad():
                candidate = module.effective_delta().detach().clone()
                candidate_metrics = cached_metrics(
                    stacked,
                    candidate,
                    target_margin=a.target_logit_margin,
                    required_margin=required_margin,
                )
                if priority(candidate_metrics) < priority(best_metrics):
                    best_candidate = candidate.detach().clone()
                    best_metrics = dict(candidate_metrics)

            if step == 1 or step % a.log_every == 0:
                logs.append(
                    {
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "mean_squared_hinge": float(errors.square().mean().detach().cpu()),
                        "hardest_squared_hinge": float(errors.square().max().detach().cpu()),
                        "delta_l2": float(delta.square().sum().detach().cpu()),
                        "gradient_norm_before_clip": grad_norm_value,
                        **candidate_metrics,
                    }
                )
                print(
                    f"stage2-step={step} target_unmet={candidate_metrics['target_margin_unmet_token_count']} "
                    f"buffered_unmet={candidate_metrics['buffered_margin_unmet_token_count']} "
                    f"min_margin={candidate_metrics['minimum_competitor_minus_sensitive_margin']:.6g} "
                    f"norm={candidate_metrics['selected_lm_head_delta_norm']:.6g}"
                )

            if feasible(stacked, candidate, required_margin):
                crossing_low = previous.detach().clone()
                crossing_high = candidate.detach().clone()
                optimizer_crossing_step = step
                crossing_metrics = dict(candidate_metrics)
                break
            previous = candidate.detach().clone()

        del optimizer
        write_jsonl(root / "repair_log.jsonl", logs)
        if crossing_low is None or crossing_high is None:
            write_json(
                root / "failure.json",
                {
                    "status": "FAILED_NO_CACHED_FEASIBLE_ACTIVE_REPAIR",
                    "repair_rank_requested": int(a.repair_rank),
                    "repair_rank_actual": actual_rank,
                    "active_sensitive_token_case_count": len(active_positions),
                    "selected_lm_head_row_count": len(selected_ids),
                    "best_metrics": best_metrics,
                    "best_candidate_norm": float(best_candidate.norm().detach().cpu()),
                },
            )
            raise RuntimeError("MQuAKE V7 active repair did not reach buffered all-visible feasibility")

        selected_candidate = boundary_bisect(
            stacked,
            crossing_low,
            crossing_high,
            required_margin=required_margin,
            iterations=a.boundary_bisection_steps,
        )
        stage1.set_selected_rows(
            output_weight,
            selected_ids,
            baseline_rows,
            selected_candidate,
        )
    else:
        baseline_rows = output_weight.new_empty((0, output_weight.shape[1]))

    if int(input_weight.data_ptr()) != input_pointer or int(input_weight._version) != input_version:
        raise RuntimeError("MQuAKE V7 Stage2 modified input embeddings")

    after_reports, after_summary = stage1.exact_materialized_reports(
        model,
        tok,
        cases,
        device=device,
        llama_like=llama_like,
        batch_size=a.batch_size,
        target_margin=a.target_logit_margin,
    )

    if after_summary["buffered_margin_unmet_token_count"] != 0:
        # The cached optimization targeted target_margin + BF16 buffer.  If even
        # that did not survive materialization, fail closed instead of silently
        # accepting a checkpoint that can miss official forgetting.
        write_json(
            root / "failure.json",
            {
                "status": "FAILED_BF16_ALL_VISIBLE_AUDIT",
                "repair_rank_requested": int(a.repair_rank),
                "repair_rank_actual": actual_rank,
                "target_logit_margin": float(a.target_logit_margin),
                "bf16_buffer_margin": float(a.bf16_buffer_margin),
                "before_summary": before_summary,
                "after_summary": after_summary,
            },
        )
        raise RuntimeError("MQuAKE V7 Stage2 BF16 all-visible target-margin audit failed")
    if after_summary["official_active_sensitive_token_count"] != 0:
        raise RuntimeError("MQuAKE V7 Stage2 official token-level forgetting audit failed")

    write_jsonl(root / "all_visible_tokens_after_bf16.jsonl", after_reports)
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    summary = {
        "status": "PASS",
        "method": METHOD,
        "protocol": PROTOCOL,
        "seed": int(a.seed),
        "forget_instances": int(a.forget_num),
        "forget_atomic_facts": len(records),
        "visible_sensitive_token_cases": len(cases),
        "repair_rank_requested": int(a.repair_rank),
        "repair_rank_actual": actual_rank,
        "repair_rank_semantics": (
            "unrestricted selected residual-active sensitive LM-head rows"
            if a.repair_rank == 0
            else "selected residual-active sensitive LM-head rows restricted to hidden basis from all visible direct sensitive token cases"
        ),
        "initially_active_sensitive_token_case_count": len(active_positions),
        "initially_active_positions": active_positions,
        "selected_active_lm_head_row_count": len(selected_ids),
        "selected_active_lm_head_token_ids": selected_ids,
        "incremental_stage2_delta_norm": float(selected_candidate.norm().detach().cpu()),
        "optimizer_crossing_step": optimizer_crossing_step,
        "cached_crossing_metrics": crossing_metrics,
        "cached_best_metrics": best_metrics,
        "stage1_exact_before_metrics": before_summary,
        "materialized_bf16_metrics": after_summary,
        "target_logit_margin": float(a.target_logit_margin),
        "cached_required_margin_with_bf16_buffer": required_margin,
        "basis_source": "all visible direct sensitive token hidden states from the same 50 sampled forget instances",
        "training_data_access": {
            "forget_instances": int(a.forget_num),
            "forget_atomic_facts": len(records),
            "prompt_types": ["requested_rewrite"],
            "benchmark_retain_instances": 0,
            "atomic_questions": 0,
            "multihop_questions": 0,
            "benchmark_counterfactual_targets": 0,
            "PPL": False,
        },
        "checkpoint_selection_uses_retain_or_heldout": False,
        "checkpoint": str(ckpt.resolve()),
    }
    write_json(root / "repair_summary.json", summary)
    write_json(
        root / "config_used.json",
        {
            "schema_version": 1,
            "method": METHOD,
            "protocol": PROTOCOL,
            **vars(a),
            "model_path_resolved": str(model_path),
            "reference_model_path_resolved": str(reference_path),
            "repair_visible_path_resolved": str(visible_path),
            "split_manifest_resolved": str(manifest_path),
            "split_sampling": split_manifest.get("sampling"),
            "parameter_scope": "Stage1 checkpoint plus residual-active sensitive LM-head rows only; transformer/input embeddings frozen",
            "data_firewall": "active selection, basis, optimization, stopping, and BF16 audit use only direct rewrite tokens from the same 50 forget instances",
            "checkpoint": str(ckpt.resolve()),
        },
    )

    print("===== SURE-MQuAKE V7 STAGE2 PASS =====")
    print(
        f"rank_requested={a.repair_rank} rank_actual={actual_rank} "
        f"active_tokens={len(active_positions)} active_rows={len(selected_ids)} "
        f"stage2_norm={float(selected_candidate.norm().detach().cpu()):.6g}"
    )
    print(
        f"final_official_active={after_summary['official_active_sensitive_token_count']} "
        f"final_target_unmet={after_summary['buffered_margin_unmet_token_count']} "
        f"min_margin={after_summary['minimum_competitor_minus_sensitive_margin']:.6g}"
    )
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
