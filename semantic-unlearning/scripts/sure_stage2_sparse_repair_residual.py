#!/usr/bin/env python3
"""Residual-only MCF SURE-LM Stage 2.

This is the locality-focused Stage-2 variant for the canonical target-true-
sensitive MCF track.  Stage 1 is unchanged.  Relative to the historical
``sure_stage2_sparse_repair.py`` MCF path, this variant makes three deliberate
changes while preserving the two-stage architecture:

1. The low-rank basis is built from predictor hidden states for the SENSITIVE
   answer only (canonical ``target_new`` in the locked adapter, i.e. ORIGINAL
   MCF ``target_true``).  Reference-answer hidden states never enter the basis.
2. The repair hinge is optimized only on the direct cases that fail the
   Stage-1 margin constraint.  Initially passing direct cases do not contribute
   repair loss.
3. Every optimization checkpoint and every scale candidate is guarded on ALL
   direct training-visible cases.  A candidate is accepted only if no direct
   case falls below the required margin.

No official paraphrase, neighborhood/locality, retain-eval, or PPL data enters
repair or selection.  The default direct margin is intentionally small (0.05):
Stage 2 is a minimum intervention intended to cross the behavioral forgetting
boundary robustly, not to maximize direct separation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

import gagd_compare as gagd
import gagd_active_case_repair as mcf_repair
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import sure_stage2_sparse_repair as canonical


METHOD = "SURE-LM-residual-sensitive-context-sparse-row-repair"
PROTOCOL = "sure_residual_sensitive_context_locked_direct_only_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument(
        "--candidate-ranks",
        default="2,8,0",
        help="Ordered rank caps; 0 means unrestricted selected-row delta.",
    )
    p.add_argument("--repair-steps", type=int, default=800)
    p.add_argument("--repair-lr", type=float, default=5e-3)
    p.add_argument("--constraint-margin", type=float, default=0.05)
    p.add_argument("--repair-l2", type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument(
        "--candidate-scales",
        default="1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0",
    )
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def _rank_basis(active_hidden: torch.Tensor, requested_rank: int):
    if requested_rank == 0:
        return None, None
    basis = core.orthonormal_row_basis(active_hidden, max_rank=requested_rank)
    actual = int(basis.shape[0])
    if actual <= 0:
        raise RuntimeError("Sensitive active hidden directions have zero numerical rank")
    return basis, actual


def _all_margins(caches, delta: torch.Tensor) -> torch.Tensor:
    return mcf_repair.margins_from_delta_caches(caches, delta).float()


def _positions_tensor(positions: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(list(positions), dtype=torch.long, device=device)


def _candidate_key(report: Dict[str, Any], order: int) -> Tuple[int, int, float]:
    return (
        int(report["direct_failures"]),
        int(order),
        float(report["delta_norm"]),
    )


def optimize_candidate(
    *,
    rank: int,
    active_hidden: torch.Tensor,
    selected_ids: Sequence[int],
    output_layer,
    all_caches,
    active_positions: Sequence[int],
    protected_positions: Sequence[int],
    required_margin: float,
    repair_steps: int,
    repair_lr: float,
    repair_l2: float,
    check_every: int,
    order: int,
):
    """Optimize only residual cases, but accept/stop using an all-direct guard."""
    basis, actual_rank = _rank_basis(active_hidden, rank)
    delta_module = core.SelectedRowDelta(
        len(selected_ids),
        output_layer.weight.shape[1],
        direction_basis=basis,
        device=output_layer.weight.device,
    )
    opt = torch.optim.AdamW(delta_module.parameters(), lr=repair_lr, weight_decay=0.0)

    active_idx = _positions_tensor(active_positions, output_layer.weight.device)
    protected_idx = _positions_tensor(protected_positions, output_layer.weight.device)

    best_failures = 10**9
    best_delta_norm = float("inf")
    best_step = 0
    best_delta = delta_module.effective_delta().detach().clone()
    logs: List[Dict[str, Any]] = []

    for step in range(1, repair_steps + 1):
        opt.zero_grad(set_to_none=True)
        delta = delta_module.effective_delta()
        margins_all = _all_margins(all_caches, delta)
        margins_active = margins_all.index_select(0, active_idx)

        # True residual repair: only initially active Stage-1 cases enter the loss.
        hinge = F.relu(float(required_margin) - margins_active).square().mean()
        l2 = delta.square().mean()
        loss = hinge + repair_l2 * l2
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite residual MCF repair loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(delta_module.parameters()), 1.0)
        opt.step()

        if step == 1 or step % check_every == 0 or step == repair_steps:
            with torch.no_grad():
                current = delta_module.effective_delta()
                margins_all = _all_margins(all_caches, current)
                margins_active = margins_all.index_select(0, active_idx)
                direct_failures = int((margins_all < required_margin).sum().item())
                active_failures = int((margins_active < required_margin).sum().item())
                if protected_idx.numel():
                    protected_margins = margins_all.index_select(0, protected_idx)
                    protected_regressions = int(
                        (protected_margins < required_margin).sum().item()
                    )
                    protected_min = float(protected_margins.min().detach().cpu())
                else:
                    protected_regressions = 0
                    protected_min = None
                delta_norm = float(current.norm().detach().cpu())
                row = {
                    "step": int(step),
                    "direct_failures": direct_failures,
                    "active_failures": active_failures,
                    "protected_regressions": protected_regressions,
                    "minimum_margin_all": float(margins_all.min().detach().cpu()),
                    "minimum_margin_active": float(margins_active.min().detach().cpu()),
                    "minimum_margin_protected": protected_min,
                    "loss": float(loss.detach().cpu()),
                    "hinge_active_only": float(hinge.detach().cpu()),
                    "delta_norm": delta_norm,
                }
                logs.append(row)

                if (
                    direct_failures < best_failures
                    or (
                        direct_failures == best_failures
                        and delta_norm < best_delta_norm
                    )
                ):
                    best_failures = direct_failures
                    best_delta_norm = delta_norm
                    best_step = step
                    best_delta = current.detach().clone()

                # Stop only when the all-direct guard passes.
                if direct_failures == 0:
                    break

    del opt

    with torch.no_grad():
        final_margins = _all_margins(all_caches, best_delta)
        final_active = final_margins.index_select(0, active_idx)
        direct_failures = int((final_margins < required_margin).sum().item())
        active_failures = int((final_active < required_margin).sum().item())
        if protected_idx.numel():
            final_protected = final_margins.index_select(0, protected_idx)
            protected_regressions = int(
                (final_protected < required_margin).sum().item()
            )
            protected_min = float(final_protected.min().detach().cpu())
        else:
            protected_regressions = 0
            protected_min = None

    report = {
        "requested_rank": int(rank),
        "actual_rank": actual_rank,
        "parameterization": (
            "unrestricted_selected_rows"
            if rank == 0
            else "fixed_sensitive_hidden_direction_basis"
        ),
        "basis_hidden_scope": "initially_active_sensitive_answer_predictor_states_only",
        "optimization_scope": "initially_active_direct_cases_only",
        "guard_scope": "all_direct_training_visible_cases",
        "trainable_parameters": int(delta_module.trainable_parameter_count),
        "best_step": int(best_step),
        "direct_failures": direct_failures,
        "active_failures": active_failures,
        "protected_regressions": protected_regressions,
        "minimum_margin": float(final_margins.min().detach().cpu()),
        "minimum_active_margin": float(final_active.min().detach().cpu()),
        "minimum_protected_margin": protected_min,
        "delta_norm": float(best_delta.norm().detach().cpu()),
        "candidate_order": int(order),
        "logs": logs,
    }
    return report, best_delta


def _select_scale(scale_reports: Sequence[Dict[str, Any]]) -> float:
    valid = [r for r in scale_reports if int(r["direct_failures"]) == 0]
    if valid:
        # Minimum intervention among candidates that preserve ALL direct cases.
        return float(min(valid, key=lambda r: float(r["scale"]))["scale"])
    best = min(
        scale_reports,
        key=lambda r: (
            int(r["direct_failures"]),
            float(r["effective_delta_norm"]),
        ),
    )
    return float(best["scale"])


def main() -> None:
    a = parse_args()
    if a.forget_num <= 0 or a.repair_steps <= 0 or a.repair_lr <= 0:
        raise ValueError("forget-num, repair-steps, and repair-lr must be positive")
    if a.batch_size <= 0 or a.check_every <= 0:
        raise ValueError("batch-size and check-every must be positive")
    if a.constraint_margin < 0 or a.repair_l2 < 0:
        raise ValueError("constraint margin and repair L2 must be non-negative")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    ranks = core.parse_rank_list(a.candidate_ranks)
    scales = core.parse_scales(a.candidate_scales)
    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, manifest = canonical.load_locked(
        "mcf", visible_path, manifest_path, a.seed, a.forget_num
    )

    ns = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    output_layer = core.untie_and_freeze_output_head(model)
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)

    instances = canonical.mcf_instances(records)
    direct_total = len(instances)
    original_margins = canonical.mcf_direct_margins(
        model, tok, instances, device, llama_like, a.batch_size
    )
    original_margin_values = [float(x) for x in original_margins.detach().cpu()]
    active_positions = [
        i for i, value in enumerate(original_margin_values)
        if value < a.constraint_margin
    ]
    protected_positions = [
        i for i, value in enumerate(original_margin_values)
        if value >= a.constraint_margin
    ]
    selected_ids = canonical.mcf_sensitive_rows(tok, instances, active_positions)

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_reports: List[Dict[str, Any]] = []
    candidate_deltas: List[torch.Tensor] = []
    scale_reports: List[Dict[str, Any]] = []
    chosen_index = None
    selected_scale = 0.0

    if selected_ids:
        all_caches = mcf_repair.build_prompt_instance_delta_caches(
            model, tok, instances, selected_ids, device, a.batch_size, llama_like
        )

        # Critical v2 change: do NOT include reference-answer hidden states.
        active_hidden = torch.cat(
            [all_caches[position].target_new.hidden for position in active_positions],
            dim=0,
        ).float()

        for order, rank in enumerate(ranks):
            report, delta = optimize_candidate(
                rank=rank,
                active_hidden=active_hidden,
                selected_ids=selected_ids,
                output_layer=output_layer,
                all_caches=all_caches,
                active_positions=active_positions,
                protected_positions=protected_positions,
                required_margin=a.constraint_margin,
                repair_steps=a.repair_steps,
                repair_lr=a.repair_lr,
                repair_l2=a.repair_l2,
                check_every=a.check_every,
                order=order,
            )
            candidate_reports.append(report)
            candidate_deltas.append(delta)
            print(
                "Residual MCF repair candidate",
                {
                    k: report[k]
                    for k in (
                        "requested_rank",
                        "actual_rank",
                        "direct_failures",
                        "active_failures",
                        "protected_regressions",
                        "delta_norm",
                    )
                },
            )
            if report["direct_failures"] == 0:
                break

        chosen_index = min(
            range(len(candidate_reports)),
            key=lambda i: _candidate_key(candidate_reports[i], i),
        )
        chosen_delta = candidate_deltas[chosen_index]

        for scale in scales:
            delta = chosen_delta * float(scale)
            margins = _all_margins(all_caches, delta)
            active_idx = _positions_tensor(active_positions, margins.device)
            protected_idx = _positions_tensor(protected_positions, margins.device)
            active_margins = margins.index_select(0, active_idx)
            if protected_idx.numel():
                protected_margins = margins.index_select(0, protected_idx)
                protected_regressions = int(
                    (protected_margins < a.constraint_margin).sum().item()
                )
                protected_min = float(protected_margins.min().detach().cpu())
            else:
                protected_regressions = 0
                protected_min = None
            scale_reports.append(
                {
                    "scale": float(scale),
                    "direct_failures": int((margins < a.constraint_margin).sum().item()),
                    "active_failures": int(
                        (active_margins < a.constraint_margin).sum().item()
                    ),
                    "protected_regressions": protected_regressions,
                    "minimum_margin_all": float(margins.min().detach().cpu()),
                    "minimum_margin_active": float(active_margins.min().detach().cpu()),
                    "minimum_margin_protected": protected_min,
                    "effective_delta_norm": float(delta.norm().detach().cpu()),
                }
            )

        selected_scale = _select_scale(scale_reports)
        final_delta = chosen_delta * selected_scale
        core.materialize_output_delta(output_layer, selected_ids, final_delta)
    else:
        chosen_delta = torch.empty(
            (0, output_layer.weight.shape[1]), device=output_layer.weight.device
        )

    final_margins = canonical.mcf_direct_margins(
        model, tok, instances, device, llama_like, a.batch_size
    )
    final_failures = int((final_margins < a.constraint_margin).sum().item())

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    chosen_report = (
        candidate_reports[chosen_index]
        if chosen_index is not None and candidate_reports
        else None
    )
    summary = {
        "schema_version": 3,
        "method": METHOD,
        "dataset": "mcf",
        "protocol": PROTOCOL,
        "source_protocol": manifest.get("protocol"),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "direct_total": int(direct_total),
        "active_before": len(active_positions),
        "protected_before": len(protected_positions),
        "active_after": int(final_failures),
        "constraint_margin": float(a.constraint_margin),
        "selected_rows_semantics": "sensitive_answer_rows_only",
        "selected_lm_head_rows": len(selected_ids),
        "selected_token_ids": selected_ids,
        "basis_hidden_scope": "active_sensitive_answer_predictor_states_only",
        "reference_hidden_states_used_in_basis": False,
        "optimization_scope": "initially_active_direct_cases_only",
        "all_direct_cases_in_repair_loss": False,
        "guard_scope": "all_direct_training_visible_cases",
        "candidate_ranks": ranks,
        "candidate_reports": candidate_reports,
        "chosen_candidate": chosen_report,
        "candidate_scales": scales,
        "scale_reports": scale_reports,
        "selected_scale": float(selected_scale),
        "effective_delta_norm": (
            float(chosen_delta.norm().detach().cpu() * selected_scale)
            if chosen_delta.numel()
            else 0.0
        ),
        "transformer_trainable": 0,
        "input_embeddings_modified": False,
        "lm_head_untied_before_repair": True,
        "repair_steps": int(a.repair_steps),
        "repair_lr": float(a.repair_lr),
        "repair_l2": float(a.repair_l2),
        "benchmark_retain_seen": 0,
        "heldout_paraphrases_or_rephrases_seen": 0,
        "locality_or_neighborhood_seen": 0,
        "PPL_seen": False,
        "selection_uses_heldout": False,
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "repair_summary.json", summary)
    core.write_json(out_dir / "rank_candidates.json", candidate_reports)
    core.write_json(out_dir / "scale_sweep_direct_only.json", scale_reports)
    core.write_json(
        out_dir / "stage1_direct_margins.json",
        {
            "constraint_margin": float(a.constraint_margin),
            "margins": original_margin_values,
            "active_positions": active_positions,
            "protected_positions": protected_positions,
        },
    )

    print(
        f"Residual MCF Stage 2: direct failures {len(active_positions)} -> {final_failures}; "
        f"selected rows={len(selected_ids)}; selected scale={selected_scale:g}"
    )
    print("basis hidden scope: sensitive answer only")
    print("optimization scope: initially active cases only; all-direct guard enabled")
    if final_failures != 0:
        print("WARNING: residual Stage 2 finished with direct failures")


if __name__ == "__main__":
    main()
