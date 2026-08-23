#!/usr/bin/env python3
"""MQuAKE SURE Stage 2 v4: robust-view exact-P-nullspace head repair.

Input is the Stage-1 v4 untied checkpoint.  The same deterministic training-only
synthetic wrappers are reconstructed from the locked direct facts; official
AtomicGen/MQuAKE questions are never used.

Let P be every direct/synthetic Stage-1 case already satisfying its required
margin and F every residual case.  With embeddings and transformer frozen:

    B_P = rowspace(H_P)
    R_F = H_F - Proj_B_P(H_F)
    B_F = rowspace(R_F)
    Delta W_A_F = C_F B_F

Only failed sensitive LM-head rows can move.  Every accepted proposal must keep
all P cases passed and stay within exact full-vocabulary Stage1||Stage2 KL.
Repair uses a bounded margin hinge with per-view margins.  After the best
feasible repair is found, a binary minimum-norm shrink finds the smallest
scalar alpha that preserves all direct+synthetic constraints, with a small
configurable safety factor for BF16 materialization.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import mquake_sure_stage1_prompt_invariant_v4 as s1v4
import mquake_sure_stage2_head_directional as v2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, required=True)
    p.add_argument("--synthetic-view-count", type=int, default=3)
    p.add_argument("--repair-steps", type=int, default=800)
    p.add_argument("--repair-lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--constraint-margin", type=float, default=0.05)
    p.add_argument("--robust-margin", type=float, default=0.25)
    p.add_argument("--max-protected-kl", type=float, default=0.05)
    p.add_argument("--l2-weight", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--backtrack-scales", default="0.5,0.25,0.125,0.0625,0.03125,0.015625")
    p.add_argument("--shrink-iters", type=int, default=24)
    p.add_argument("--shrink-safety", type=float, default=1.01)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    p.add_argument("--skip-frozen-hash", action="store_true")
    return p.parse_args()


def threshold_tensor(
    cases: Sequence[s1v4.V4PredictionCase],
    direct_margin: float,
    robust_margin: float,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [s1v4.required_margin(c, direct_margin, robust_margin) for c in cases],
        dtype=torch.float32,
        device=device,
    )


def report_logits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    cases: Sequence[s1v4.V4PredictionCase],
    direct_margin: float,
    robust_margin: float,
) -> Dict[str, Any]:
    x = logits.float()
    tids = target_ids.to(device=x.device, dtype=torch.long)
    rows = torch.arange(x.shape[0], device=x.device)
    target = x[rows, tids]
    other = x.clone()
    other[rows, tids] = -torch.inf
    margins = other.max(dim=-1).values - target
    needs = threshold_tensor(cases, direct_margin, robust_margin, x.device)
    slack = margins - needs
    failed_mask = slack < 0.0
    failed = torch.nonzero(failed_mask, as_tuple=False).flatten().detach().cpu().tolist()
    by_view: Dict[str, Dict[str, int]] = {}
    for i, case in enumerate(cases):
        bucket = by_view.setdefault(case.view, {"total": 0, "failed": 0})
        bucket["total"] += 1
        if bool(failed_mask[i].item()):
            bucket["failed"] += 1
    for bucket in by_view.values():
        bucket["passed"] = int(bucket["total"] - bucket["failed"])
    return {
        "total": len(cases),
        "passed": len(cases) - len(failed),
        "failed": len(failed),
        "residual_indices": [int(i) for i in failed],
        "minimum_margin": float(margins.min().item()) if margins.numel() else None,
        "minimum_slack": float(slack.min().item()) if slack.numel() else None,
        "mean_margin": float(margins.mean().item()) if margins.numel() else None,
        "direct_margin": float(direct_margin),
        "robust_margin": float(robust_margin),
        "by_view": by_view,
    }


def geometry_report(
    geometry: v2.SelectedRowGeometry,
    indices: Sequence[int],
    delta_rows: torch.Tensor,
    all_thresholds: torch.Tensor,
) -> Dict[str, Any]:
    if not indices:
        return {
            "count": 0,
            "failed": 0,
            "minimum_margin": None,
            "minimum_slack": None,
            "mean_margin": None,
            "kl_mean": 0.0,
            "kl_max": 0.0,
        }
    with torch.no_grad():
        margins, kl = geometry.metrics(indices, delta_rows)
        idx = torch.tensor([int(i) for i in indices], dtype=torch.long, device=geometry.device)
        needs = all_thresholds.index_select(0, idx)
        slack = margins - needs
        return {
            "count": len(indices),
            "failed": int((slack < 0.0).sum().item()),
            "minimum_margin": float(margins.min().item()),
            "minimum_slack": float(slack.min().item()),
            "mean_margin": float(margins.mean().item()),
            "kl_mean": float(kl.mean().item()),
            "kl_max": float(kl.max().item()),
        }


def restore_output_rows(output_layer, row_ids: Sequence[int], saved_rows: torch.Tensor) -> None:
    with torch.no_grad():
        ids = torch.tensor([int(x) for x in row_ids], dtype=torch.long, device=output_layer.weight.device)
        output_layer.weight.index_copy_(
            0,
            ids,
            saved_rows.to(device=output_layer.weight.device, dtype=output_layer.weight.dtype),
        )


def direct_only_report(
    full_logits: torch.Tensor,
    target_ids: torch.Tensor,
    cases: Sequence[s1v4.V4PredictionCase],
    direct_margin: float,
) -> Dict[str, Any]:
    indices = [i for i, case in enumerate(cases) if case.view == "direct"]
    sub_logits = full_logits[indices]
    sub_tids = target_ids[indices]
    sub_cases = [cases[i] for i in indices]
    return report_logits(sub_logits, sub_tids, sub_cases, direct_margin, direct_margin)


def main() -> None:
    a = parse_args()
    if min(a.repair_steps, a.batch_size, a.cache_batch_size, a.check_every, a.shrink_iters) <= 0:
        raise ValueError("steps/batches/check/shrink iterations must be positive")
    if a.repair_lr <= 0 or a.max_protected_kl < 0 or a.l2_weight < 0:
        raise ValueError("invalid optimization settings")
    if a.robust_margin < a.constraint_margin:
        raise ValueError("robust-margin must be >= constraint-margin")
    if a.shrink_safety < 1.0:
        raise ValueError("shrink-safety must be >= 1")
    backtrack_scales = v2.parse_scales(a.backtrack_scales)

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    records, manifest = v2.load_locked(a)
    ns = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)

    direct_cases = core.expand_sensitive_cases(
        records, tok, sensitive_field="target_true", llama_like=llama_like
    )
    if not direct_cases:
        raise RuntimeError("no generated MQuAKE PredictionCases")
    cases = s1v4.make_augmented_cases(direct_cases, a.synthetic_view_count)

    output_layer = core.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("Stage2 requires an untied LM head")
    model.eval()

    frozen_before = None if a.skip_frozen_hash else v2.hash_frozen(model, output_layer.weight)
    base_logits = core.cache_base_logits(model, tok, cases, device, batch_size=a.cache_batch_size)
    target_ids = core.official_target_ids(tok, cases, llama_like=llama_like, device=device)
    thresholds = threshold_tensor(cases, a.constraint_margin, a.robust_margin, device)

    level1_gate = report_logits(
        base_logits, target_ids.cpu(), cases, a.constraint_margin, a.robust_margin
    )
    direct_level1_gate = direct_only_report(
        base_logits, target_ids.cpu(), cases, a.constraint_margin
    )
    F_indices = [int(i) for i in level1_gate["residual_indices"]]
    failed_set = set(F_indices)
    P_indices = [i for i in range(len(cases)) if i not in failed_set]

    out = gagd.resolve_output_path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint"

    report: Dict[str, Any] = {
        "schema_version": 5,
        "method": "MQuAKE SURE Stage2 Robust Exact-P-Nullspace Minimum-Norm Repair",
        "source_protocol": manifest.get("protocol"),
        "level1_robust_gate": level1_gate,
        "level1_direct_gate": direct_level1_gate,
        "synthetic_view_count": int(a.synthetic_view_count),
        "constraint_margin": float(a.constraint_margin),
        "robust_margin": float(a.robust_margin),
        "embedding_frozen_in_stage2": True,
        "transformer_frozen_in_stage2": True,
        "output_head_only_stage2": True,
        "benchmark_retain_seen": 0,
        "official_atomicgen_seen": 0,
        "target_new_seen": False,
    }

    if not F_indices:
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        frozen_after = None if a.skip_frozen_hash else v2.hash_frozen(model, output_layer.weight)
        frozen_exact = frozen_before == frozen_after
        report.update({
            "level2": {"skipped": True, "reason": "Stage1 passed all direct+synthetic constraints", "F": 0, "P": len(P_indices)},
            "final_robust_gate": level1_gate,
            "final_direct_gate": direct_level1_gate,
            "stage1_successes_regressed": 0,
            "protected_kl": 0.0,
            "max_protected_kl": float(a.max_protected_kl),
            "frozen_non_head_exact": bool(frozen_exact),
            "final_gates_pass": bool(level1_gate["failed"] == 0 and direct_level1_gate["failed"] == 0 and frozen_exact),
            "checkpoint": str(ckpt.resolve()),
        })
        core.write_json(out / "two_stage_summary.json", report)
        print("Level1 robust gate: {}/{} pass; Stage2 identity".format(level1_gate["passed"], level1_gate["total"]))
        print("Final gates pass:", report["final_gates_pass"])
        return

    hidden = core.forward_last_hidden(model, tok, cases, device, batch_size=a.cache_batch_size).float()
    H_F = hidden[F_indices]
    H_P = hidden[P_indices] if P_indices else hidden.new_empty((0, hidden.shape[1]))

    protected_basis = (
        core.orthonormal_row_basis(H_P, max_rank=None).to(device=device, dtype=torch.float32)
        if H_P.numel() else hidden.new_empty((0, hidden.shape[1]), dtype=torch.float32)
    )
    residual = H_F - (H_F @ protected_basis.transpose(0, 1)) @ protected_basis if protected_basis.numel() else H_F
    repair_basis = core.orthonormal_row_basis(residual, max_rank=None).to(device=device, dtype=torch.float32)
    if repair_basis.ndim != 2 or repair_basis.shape[0] == 0:
        raise RuntimeError("No repair direction remains after exact robust-P subspace removal")

    protected_leak = 0.0
    if H_P.numel():
        protected_leak = float((H_P @ repair_basis.transpose(0, 1)).abs().max().detach().cpu())

    special = set(gagd.special_token_ids(tok))
    A_F = sorted(set(int(x) for x in target_ids[F_indices].detach().cpu().tolist()) - special)
    if not A_F:
        raise RuntimeError("A_F empty for non-empty F")

    delta = core.SelectedRowDelta(
        len(A_F), int(output_layer.weight.shape[1]), direction_basis=repair_basis, device=device
    )
    geometry = v2.SelectedRowGeometry(
        base_logits=base_logits,
        hidden=hidden,
        target_ids=target_ids,
        selected_row_ids=A_F,
        device=device,
    )
    parameters = list(delta.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=a.repair_lr, weight_decay=0.0)
    sampler = core.IndexSampler(len(F_indices), a.batch_size, a.seed + 100003)

    initial_f = geometry_report(geometry, F_indices, delta.effective_delta(), thresholds)
    initial_p = geometry_report(geometry, P_indices, delta.effective_delta(), thresholds)
    best_state = v2.capture_state(delta)
    best_key = (
        int(initial_f["failed"]),
        -float(initial_f["minimum_slack"]),
        float(delta.effective_delta().detach().float().norm().cpu()),
    )
    best_step = 0
    best_f = initial_f
    best_p = initial_p
    accepted_steps = 0
    rolled_back_steps = 0
    logs: List[Dict[str, Any]] = []

    log_path = out / "repair_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_f:
        for step in range(1, a.repair_steps + 1):
            local = sampler.next()
            ids = [F_indices[i] for i in local]
            optimizer.zero_grad(set_to_none=True)
            delta_rows = delta.effective_delta()
            margins, _ = geometry.metrics(ids, delta_rows)
            idx_tensor = torch.tensor(ids, dtype=torch.long, device=device)
            needs = thresholds.index_select(0, idx_tensor)
            hinge = F.relu(needs - margins).square().mean()
            l2 = delta_rows.square().mean()
            loss = hinge + float(a.l2_weight) * l2
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage2 loss at step {step}")

            old_state = v2.capture_state(delta)
            old_opt = copy.deepcopy(optimizer.state_dict())
            loss.backward()
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(parameters, a.grad_clip)
                if a.grad_clip > 0 else None
            )
            if grad_norm is not None and not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite Stage2 gradient at step {step}")
            optimizer.step()
            proposed_state = v2.capture_state(delta)

            accepted_scale = 1.0
            p_report = geometry_report(geometry, P_indices, delta.effective_delta(), thresholds)
            feasible = (
                int(p_report["failed"]) == 0
                and float(p_report["kl_mean"]) <= float(a.max_protected_kl)
            )
            if not feasible:
                accepted_scale = 0.0
                for scale in backtrack_scales:
                    v2.interpolate_state(delta, old_state, proposed_state, scale)
                    trial_p = geometry_report(geometry, P_indices, delta.effective_delta(), thresholds)
                    if int(trial_p["failed"]) == 0 and float(trial_p["kl_mean"]) <= float(a.max_protected_kl):
                        feasible = True
                        accepted_scale = float(scale)
                        p_report = trial_p
                        break
            if not feasible:
                v2.restore_state(delta, old_state)
                optimizer.load_state_dict(old_opt)
                accepted_scale = 0.0
                rolled_back_steps += 1
                p_report = geometry_report(geometry, P_indices, delta.effective_delta(), thresholds)
            else:
                accepted_steps += 1

            f_report = geometry_report(geometry, F_indices, delta.effective_delta(), thresholds)
            norm = float(delta.effective_delta().detach().float().norm().cpu())
            key = (int(f_report["failed"]), -float(f_report["minimum_slack"]), norm)
            if key < best_key:
                best_key = key
                best_state = v2.capture_state(delta)
                best_step = step
                best_f = f_report
                best_p = p_report

            row = {
                "step": step,
                "repair_hinge": float(hinge.detach().cpu()),
                "l2": float(l2.detach().cpu()),
                "gradient_norm_before_clip": None if grad_norm is None else float(grad_norm.detach().cpu()),
                "accepted_scale": float(accepted_scale),
                "F_failed": int(f_report["failed"]),
                "F_minimum_slack": float(f_report["minimum_slack"]),
                "P_regressions": int(p_report["failed"]),
                "P_exact_kl_mean": float(p_report["kl_mean"]),
                "P_exact_kl_max": float(p_report["kl_max"]),
                "delta_norm": norm,
                "best_step": int(best_step),
            }
            logs.append(row)
            if step == 1 or step % a.check_every == 0 or step == a.repair_steps or int(f_report["failed"]) == 0:
                print(
                    "Stage2-v4 step {s}: F_fail={ff} P_reg={pr} KL={kl:.6g} scale={sc:g} ||dW||={n:.6g}".format(
                        s=step, ff=f_report["failed"], pr=p_report["failed"],
                        kl=p_report["kl_mean"], sc=accepted_scale, n=norm
                    )
                )
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()
            if int(f_report["failed"]) == 0:
                break

    v2.restore_state(delta, best_state)
    best_delta = delta.effective_delta().detach()
    best_f = geometry_report(geometry, F_indices, best_delta, thresholds)
    best_p = geometry_report(geometry, P_indices, best_delta, thresholds)

    # Minimum-norm scalar shrink.  Zero is known to fail F; alpha=1 is feasible
    # only if the optimizer solved every residual robust constraint.
    shrink_alpha = 1.0
    shrink_possible = int(best_f["failed"]) == 0 and int(best_p["failed"]) == 0 and float(best_p["kl_mean"]) <= float(a.max_protected_kl)
    if shrink_possible:
        low, high = 0.0, 1.0
        for _ in range(a.shrink_iters):
            mid = 0.5 * (low + high)
            trial = best_delta * mid
            f_trial = geometry_report(geometry, F_indices, trial, thresholds)
            p_trial = geometry_report(geometry, P_indices, trial, thresholds)
            ok = int(f_trial["failed"]) == 0 and int(p_trial["failed"]) == 0 and float(p_trial["kl_mean"]) <= float(a.max_protected_kl)
            if ok:
                high = mid
            else:
                low = mid
        shrink_alpha = min(1.0, high * float(a.shrink_safety))

    chosen_delta = best_delta * float(shrink_alpha)
    head_ids_tensor = torch.tensor(A_F, dtype=torch.long, device=output_layer.weight.device)
    head_rows_before = output_layer.weight.index_select(0, head_ids_tensor).detach().clone()

    def materialize_and_measure(delta_rows: torch.Tensor):
        restore_output_rows(output_layer, A_F, head_rows_before)
        core.materialize_output_delta(output_layer, A_F, delta_rows)
        model.eval()
        logits = core.cache_base_logits(model, tok, cases, device, batch_size=a.cache_batch_size)
        robust_gate = report_logits(logits, target_ids.cpu(), cases, a.constraint_margin, a.robust_margin)
        direct_gate = direct_only_report(logits, target_ids.cpu(), cases, a.constraint_margin)
        final_failed = set(robust_gate["residual_indices"])
        p_reg = sum(1 for i in P_indices if i in final_failed)
        p_kl = v2.actual_full_kl_mean(base_logits, logits, P_indices, a.cache_batch_size)
        return logits, robust_gate, direct_gate, int(p_reg), float(p_kl)

    final_logits, final_robust_gate, final_direct_gate, p_regressions, protected_kl = materialize_and_measure(chosen_delta)
    used_full_delta_fallback = False
    if int(final_robust_gate["failed"]) != 0 and float(shrink_alpha) < 1.0:
        # BF16 rounding or mild non-monotonicity can invalidate the nearly
        # minimal alpha.  Preserve correctness by falling back to the full best
        # feasible repair rather than saving a failed checkpoint.
        used_full_delta_fallback = True
        shrink_alpha = 1.0
        chosen_delta = best_delta
        final_logits, final_robust_gate, final_direct_gate, p_regressions, protected_kl = materialize_and_measure(chosen_delta)

    frozen_after = None if a.skip_frozen_hash else v2.hash_frozen(model, output_layer.weight)
    frozen_exact = frozen_before == frozen_after
    final_pass = bool(
        int(final_robust_gate["failed"]) == 0
        and int(final_direct_gate["failed"]) == 0
        and int(p_regressions) == 0
        and float(protected_kl) <= float(a.max_protected_kl)
        and frozen_exact
    )

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    residual_energy = float(
        (residual.square().sum() / H_F.square().sum().clamp_min(1e-12)).detach().cpu()
    )
    report["level2"] = {
        "skipped": False,
        "F": len(F_indices),
        "P": len(P_indices),
        "A_F": A_F,
        "A_F_count": len(A_F),
        "protected_basis_rank": int(protected_basis.shape[0]),
        "repair_basis_rank": int(repair_basis.shape[0]),
        "protected_nullspace_max_abs_inner_product": float(protected_leak),
        "residual_hidden_energy_fraction": residual_energy,
        "parameterization": "Delta W_A_F=C_F B_F; B_F in orthogonal complement of full robust H_P rowspace",
        "repair_objective": "bounded squared hinge with direct/robust per-view margins",
        "hard_protection": "all robust P must remain passed; exact full-vocabulary Stage1||Stage2 KL budget",
        "repair_lr": float(a.repair_lr),
        "repair_steps_requested": int(a.repair_steps),
        "accepted_steps": int(accepted_steps),
        "rolled_back_steps": int(rolled_back_steps),
        "best_checkpoint_step": int(best_step),
        "best_selection_key": list(best_key),
        "best_pre_materialization_F": best_f,
        "best_pre_materialization_P": best_p,
        "best_delta_norm": float(best_delta.float().norm().cpu()),
        "minimum_norm_shrink_alpha": float(shrink_alpha),
        "shrink_safety": float(a.shrink_safety),
        "used_full_delta_fallback": bool(used_full_delta_fallback),
        "materialized_delta_norm": float(chosen_delta.float().norm().cpu()),
        "logs": logs,
    }
    report["final_robust_gate"] = final_robust_gate
    report["final_direct_gate"] = final_direct_gate
    report["stage1_successes_regressed"] = int(p_regressions)
    report["protected_kl"] = float(protected_kl)
    report["max_protected_kl"] = float(a.max_protected_kl)
    report["frozen_non_head_exact"] = bool(frozen_exact)
    report["final_gates_pass"] = final_pass
    report["checkpoint"] = str(ckpt.resolve())
    core.write_json(out / "two_stage_summary.json", report)
    torch.save(
        {
            "protected_basis": protected_basis.detach().cpu(),
            "repair_basis": repair_basis.detach().cpu(),
            "A_F": A_F,
            "best_delta": best_delta.float().cpu(),
            "chosen_delta": chosen_delta.float().cpu(),
            "shrink_alpha": float(shrink_alpha),
        },
        out / "stage2_robust_nullspace_state.pt",
    )

    print("Level1 direct gate: {}/{} pass".format(direct_level1_gate["passed"], direct_level1_gate["total"]))
    print("Level1 robust gate: {}/{} pass; F={}; P={}".format(level1_gate["passed"], level1_gate["total"], len(F_indices), len(P_indices)))
    print("Stage2-v4: A_F={}, P-basis-rank={}, repair-rank={}, nullspace-leak={:.6g}, best_step={}".format(
        len(A_F), protected_basis.shape[0], repair_basis.shape[0], protected_leak, best_step
    ))
    print("Minimum-norm shrink alpha: {:.8g}; fallback_full={}".format(shrink_alpha, used_full_delta_fallback))
    print("Final direct gate: {}/{} pass".format(final_direct_gate["passed"], final_direct_gate["total"]))
    print("Final robust gate: {}/{} pass; protected_KL={:.6g}; P regressions={}; frozen_non_head_exact={}".format(
        final_robust_gate["passed"], final_robust_gate["total"], protected_kl, p_regressions, frozen_exact
    ))
    print("Final gates pass:", final_pass)
    print("Checkpoint:", ckpt)


if __name__ == "__main__":
    main()
