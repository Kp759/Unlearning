#!/usr/bin/env python3
"""MQuAKE SURE v4.2: BF16-aware final cleanup inside Stage 2.

This script starts from a materialized v4.1 Stage-2 checkpoint. It does not
change Stage 1 and does not introduce any held-out evaluation prompt.

If BF16 materialization flips a small number of direct/synthetic training cases,
those residual cases are repaired once more in the exact numerical nullspace of
all materialized successes. The repair objective uses a small extra margin
buffer so the final BF16 checkpoint remains on the safe side of the original
training constraints.
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
import mquake_sure_stage2_robust_nullspace_v4 as v4
import mquake_sure_stage2_robust_nullspace_v41 as v41
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
    p.add_argument("--repair-steps", type=int, default=200)
    p.add_argument("--repair-lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--constraint-margin", type=float, default=0.05)
    p.add_argument("--robust-margin", type=float, default=0.25)
    p.add_argument("--materialization-buffer", type=float, default=0.10)
    p.add_argument("--max-protected-kl", type=float, default=0.05)
    p.add_argument("--l2-weight", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--backtrack-scales", default="0.5,0.25,0.125,0.0625,0.03125,0.015625")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    p.add_argument("--skip-frozen-hash", action="store_true")
    return p.parse_args()


def _stable_nullspace_bases(H_P: torch.Tensor, H_F: torch.Tensor):
    device = H_F.device
    if H_P.numel():
        p64 = v41._svd_row_basis64(H_P.detach().to(torch.float64), rtol=1e-10)
    else:
        p64 = torch.empty((0, H_F.shape[1]), dtype=torch.float64, device=device)

    f64 = H_F.detach().to(torch.float64)
    cleaned = f64 - (f64 @ p64.transpose(0, 1)) @ p64 if p64.numel() else f64
    r64 = v41._svd_row_basis64(cleaned, rtol=1e-8)
    if p64.numel():
        r64 = r64 - (r64 @ p64.transpose(0, 1)) @ p64
    q, _ = torch.linalg.qr(r64.transpose(0, 1), mode="reduced")
    r64 = q.transpose(0, 1).contiguous()
    if p64.numel():
        r64 = r64 - (r64 @ p64.transpose(0, 1)) @ p64
        q, _ = torch.linalg.qr(r64.transpose(0, 1), mode="reduced")
        r64 = q.transpose(0, 1).contiguous()

    leak = float((p64 @ r64.transpose(0, 1)).abs().max().item()) if p64.numel() else 0.0
    if leak > 1e-8:
        raise RuntimeError(f"v4.2 cleanup basis failed P-nullspace validation: leak={leak}")
    return p64.to(device=device, dtype=torch.float32), r64.to(device=device, dtype=torch.float32), leak


def _buffered_f_report(
    geometry: v2.SelectedRowGeometry,
    indices: Sequence[int],
    delta_rows: torch.Tensor,
    thresholds: torch.Tensor,
    buffer: float,
) -> Dict[str, Any]:
    if not indices:
        return {"count": 0, "failed": 0, "minimum_slack": None, "kl_mean": 0.0, "kl_max": 0.0}
    with torch.no_grad():
        margins, kl = geometry.metrics(indices, delta_rows)
        idx = torch.tensor([int(i) for i in indices], dtype=torch.long, device=geometry.device)
        needs = thresholds.index_select(0, idx) + float(buffer)
        slack = margins - needs
        return {
            "count": len(indices),
            "failed": int((slack < 0).sum().item()),
            "minimum_slack": float(slack.min().item()),
            "mean_slack": float(slack.mean().item()),
            "kl_mean": float(kl.mean().item()),
            "kl_max": float(kl.max().item()),
        }


def main() -> None:
    a = parse_args()
    if a.repair_steps <= 0 or a.repair_lr <= 0 or a.materialization_buffer <= 0:
        raise ValueError("repair steps/lr/materialization buffer must be positive")
    if a.robust_margin < a.constraint_margin:
        raise ValueError("robust margin must be >= direct margin")
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

    direct_cases = core.expand_sensitive_cases(records, tok, sensitive_field="target_true", llama_like=llama_like)
    cases = s1v4.make_augmented_cases(direct_cases, a.synthetic_view_count)

    output_layer = core.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("v4.2 cleanup requires untied E/W")
    model.eval()

    frozen_before = None if a.skip_frozen_hash else v2.hash_frozen(model, output_layer.weight)
    base_logits = core.cache_base_logits(model, tok, cases, device, batch_size=a.cache_batch_size)
    target_ids = core.official_target_ids(tok, cases, llama_like=llama_like, device=device)
    thresholds = v4.threshold_tensor(cases, a.constraint_margin, a.robust_margin, device)
    initial_gate = v4.report_logits(base_logits, target_ids.cpu(), cases, a.constraint_margin, a.robust_margin)
    direct_initial = v4.direct_only_report(base_logits, target_ids.cpu(), cases, a.constraint_margin)

    F_indices = [int(i) for i in initial_gate["residual_indices"]]
    failed_set = set(F_indices)
    P_indices = [i for i in range(len(cases)) if i not in failed_set]

    out = gagd.resolve_output_path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint"

    print("v4.2 materialized input gate: robust={}/{} direct={}/{} F={}".format(
        initial_gate["passed"], initial_gate["total"], direct_initial["passed"], direct_initial["total"], len(F_indices)
    ))

    if not F_indices:
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        core.write_json(out / "materialization_cleanup_summary.json", {
            "method": "SURE v4.2 BF16-aware Stage2 cleanup",
            "skipped": True,
            "reason": "materialized v4.1 checkpoint already passes all robust constraints",
            "initial_gate": initial_gate,
            "final_gate": initial_gate,
            "final_direct_gate": direct_initial,
            "final_gates_pass": True,
            "checkpoint": str(ckpt.resolve()),
        })
        return

    hidden = core.forward_last_hidden(model, tok, cases, device, batch_size=a.cache_batch_size).float()
    H_F = hidden[F_indices]
    H_P = hidden[P_indices]
    protected_basis, repair_basis, leak = _stable_nullspace_bases(H_P, H_F)
    print("v4.2 cleanup basis: P-rank={} F-rank={} P_leak64={:.3e}".format(
        protected_basis.shape[0], repair_basis.shape[0], leak
    ))

    special = set(gagd.special_token_ids(tok))
    A_F = sorted(set(int(x) for x in target_ids[F_indices].detach().cpu().tolist()) - special)
    if not A_F:
        raise RuntimeError("v4.2 cleanup has no content-bearing failed target row")

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
    params = list(delta.parameters())
    optimizer = torch.optim.AdamW(params, lr=a.repair_lr, weight_decay=0.0)
    sampler = core.IndexSampler(len(F_indices), a.batch_size, a.seed + 200003)

    best_state = v2.capture_state(delta)
    best_report = _buffered_f_report(geometry, F_indices, delta.effective_delta(), thresholds, a.materialization_buffer)
    best_key = (int(best_report["failed"]), -float(best_report["minimum_slack"]), 0.0)
    best_step = 0
    logs: List[Dict[str, Any]] = []

    for step in range(1, a.repair_steps + 1):
        local = sampler.next()
        ids = [F_indices[i] for i in local]
        optimizer.zero_grad(set_to_none=True)
        delta_rows = delta.effective_delta()
        margins, _ = geometry.metrics(ids, delta_rows)
        idx = torch.tensor(ids, dtype=torch.long, device=device)
        needs = thresholds.index_select(0, idx) + float(a.materialization_buffer)
        hinge = F.relu(needs - margins).square().mean()
        l2 = delta_rows.square().mean()
        loss = hinge + float(a.l2_weight) * l2
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite v4.2 cleanup loss at step {step}")

        old_state = v2.capture_state(delta)
        old_opt = copy.deepcopy(optimizer.state_dict())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, a.grad_clip) if a.grad_clip > 0 else None
        optimizer.step()
        proposed = v2.capture_state(delta)

        p_report = v4.geometry_report(geometry, P_indices, delta.effective_delta(), thresholds)
        accepted_scale = 1.0
        feasible = int(p_report["failed"]) == 0 and float(p_report["kl_mean"]) <= float(a.max_protected_kl)
        if not feasible:
            accepted_scale = 0.0
            for scale in backtrack_scales:
                v2.interpolate_state(delta, old_state, proposed, scale)
                trial_p = v4.geometry_report(geometry, P_indices, delta.effective_delta(), thresholds)
                if int(trial_p["failed"]) == 0 and float(trial_p["kl_mean"]) <= float(a.max_protected_kl):
                    feasible = True
                    accepted_scale = float(scale)
                    p_report = trial_p
                    break
        if not feasible:
            v2.restore_state(delta, old_state)
            optimizer.load_state_dict(old_opt)
            accepted_scale = 0.0
            p_report = v4.geometry_report(geometry, P_indices, delta.effective_delta(), thresholds)

        f_report = _buffered_f_report(geometry, F_indices, delta.effective_delta(), thresholds, a.materialization_buffer)
        norm = float(delta.effective_delta().detach().float().norm().cpu())
        key = (int(f_report["failed"]), -float(f_report["minimum_slack"]), norm)
        if key < best_key:
            best_key = key
            best_state = v2.capture_state(delta)
            best_report = f_report
            best_step = step

        row = {
            "step": step,
            "F_buffered_failed": int(f_report["failed"]),
            "F_minimum_buffered_slack": float(f_report["minimum_slack"]),
            "P_regressions": int(p_report["failed"]),
            "P_kl_mean": float(p_report["kl_mean"]),
            "accepted_scale": float(accepted_scale),
            "delta_norm": norm,
        }
        logs.append(row)
        if step == 1 or step % 10 == 0 or int(f_report["failed"]) == 0:
            print("v4.2 step {}: F_buffer_fail={} slack={:.6g} P_reg={} KL={:.3e} scale={} ||dW||={:.6g}".format(
                step, f_report["failed"], f_report["minimum_slack"], p_report["failed"], p_report["kl_mean"], accepted_scale, norm
            ))
        if int(f_report["failed"]) == 0:
            break

    v2.restore_state(delta, best_state)
    chosen_delta = delta.effective_delta().detach()
    core.materialize_output_delta(output_layer, A_F, chosen_delta)
    model.eval()

    final_logits = core.cache_base_logits(model, tok, cases, device, batch_size=a.cache_batch_size)
    final_gate = v4.report_logits(final_logits, target_ids.cpu(), cases, a.constraint_margin, a.robust_margin)
    final_direct = v4.direct_only_report(final_logits, target_ids.cpu(), cases, a.constraint_margin)
    final_failed = set(final_gate["residual_indices"])
    p_regressions = sum(1 for i in P_indices if i in final_failed)
    protected_kl = v2.actual_full_kl_mean(base_logits, final_logits, P_indices, a.cache_batch_size)
    frozen_after = None if a.skip_frozen_hash else v2.hash_frozen(model, output_layer.weight)
    frozen_exact = frozen_before == frozen_after
    final_pass = bool(
        int(final_gate["failed"]) == 0
        and int(final_direct["failed"]) == 0
        and int(p_regressions) == 0
        and float(protected_kl) <= float(a.max_protected_kl)
        and frozen_exact
    )

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    core.write_json(out / "materialization_cleanup_summary.json", {
        "schema_version": 1,
        "method": "MQuAKE SURE v4.2 BF16-aware Stage2 materialization cleanup",
        "source_protocol": manifest.get("protocol"),
        "materialization_buffer": float(a.materialization_buffer),
        "initial_robust_gate": initial_gate,
        "initial_direct_gate": direct_initial,
        "F": len(F_indices),
        "P": len(P_indices),
        "A_F": A_F,
        "protected_basis_rank": int(protected_basis.shape[0]),
        "repair_basis_rank": int(repair_basis.shape[0]),
        "protected_nullspace_leak64": float(leak),
        "best_step": int(best_step),
        "best_buffered_F": best_report,
        "delta_norm": float(chosen_delta.float().norm().cpu()),
        "final_robust_gate": final_gate,
        "final_direct_gate": final_direct,
        "P_regressions": int(p_regressions),
        "protected_kl_from_materialized_v41": float(protected_kl),
        "frozen_non_head_exact": bool(frozen_exact),
        "final_gates_pass": bool(final_pass),
        "official_atomicgen_seen": 0,
        "benchmark_retain_seen": 0,
        "target_new_seen": False,
        "logs": logs,
        "checkpoint": str(ckpt.resolve()),
    })

    print("v4.2 final direct gate: {}/{}".format(final_direct["passed"], final_direct["total"]))
    print("v4.2 final robust gate: {}/{}; P_reg={}; KL={:.6g}; frozen_non_head_exact={}".format(
        final_gate["passed"], final_gate["total"], p_regressions, protected_kl, frozen_exact
    ))
    print("Final gates pass:", final_pass)
    print("Checkpoint:", ckpt)


if __name__ == "__main__":
    main()
