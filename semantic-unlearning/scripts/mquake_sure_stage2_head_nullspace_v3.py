#!/usr/bin/env python3
"""MQuAKE Stage 2 v3: LM-head-only residual repair in the exact P-nullspace.

Input is the untied Stage-1 v3 checkpoint.

Let P be every Stage-1-success atomic token decision and F every Stage-1 failure.
Since embeddings and transformer are frozen in Stage 2, all final hidden states
are constant. We construct the full numerical row-space B_P of H_P and project
failed hidden states into its orthogonal complement:

    R_F = H_F - Proj_{B_P}(H_F)
    B_F = rowspace(R_F)

Only failed sensitive LM-head rows A_F can move and every update is constrained:

    Delta W_A_F = C_F B_F

Therefore, in exact arithmetic, H_P Delta W_A_F^T = 0: Stage-1-success logits
are unchanged by construction. A hard all-P margin and exact full-vocabulary KL
check remains as a numerical safety guard. Embeddings and transformer stay
bit-exact. Repair uses a bounded margin hinge and best-feasible checkpointing.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import mquake_sure_stage2_head_directional as v2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, required=True)
    p.add_argument("--repair-steps", type=int, default=800)
    p.add_argument("--repair-lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--constraint-margin", type=float, default=0.05)
    p.add_argument("--max-protected-kl", type=float, default=0.05)
    p.add_argument("--l2-weight", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--backtrack-scales", default="0.5,0.25,0.125,0.0625,0.03125,0.015625")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    p.add_argument("--skip-frozen-hash", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if min(a.repair_steps, a.batch_size, a.cache_batch_size, a.check_every) <= 0:
        raise ValueError("steps/batches/check interval must be positive")
    if a.repair_lr <= 0 or a.max_protected_kl < 0 or a.l2_weight < 0:
        raise ValueError("invalid optimization settings")
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
    cases = core.expand_sensitive_cases(records, tok, sensitive_field="target_true", llama_like=llama_like)
    if not cases:
        raise RuntimeError("no generated MQuAKE PredictionCases")

    # Stage1 v3 is already untied. Calling this helper is identity for an untied
    # head and freezes every model parameter before coefficient optimization.
    output_layer = core.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("Stage2 requires an untied LM head")
    model.eval()

    frozen_before = None if a.skip_frozen_hash else v2.hash_frozen(model, output_layer.weight)
    base_logits = core.cache_base_logits(model, tok, cases, device, batch_size=a.cache_batch_size)
    target_ids = core.official_target_ids(tok, cases, llama_like=llama_like, device=device)
    level1_gate = v2.gate_from_logits(base_logits, target_ids.cpu(), a.constraint_margin)
    F_indices = [int(x) for x in level1_gate["residual_indices"]]
    failed_set = set(F_indices)
    P_indices = [i for i in range(len(cases)) if i not in failed_set]

    out = gagd.resolve_output_path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint"

    report: Dict[str, Any] = {
        "schema_version": 4,
        "method": "MQuAKE Stage2 LM-Head Exact-P-Nullspace Residual Repair",
        "source_protocol": manifest.get("protocol"),
        "level1_gate": level1_gate,
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
        report.update({
            "level2": {"skipped": True, "reason": "Stage1 passed all atomic constraints", "F": 0, "P": len(P_indices)},
            "final_gate": level1_gate,
            "stage1_successes_regressed": 0,
            "protected_kl": 0.0,
            "max_protected_kl": float(a.max_protected_kl),
            "frozen_non_head_exact": frozen_before == frozen_after,
            "final_gates_pass": bool(level1_gate["failed"] == 0 and frozen_before == frozen_after),
            "checkpoint": str(ckpt.resolve()),
        })
        core.write_json(out / "two_stage_summary.json", report)
        print("Level1 gate: {}/{} pass; Stage2 identity".format(level1_gate["passed"], level1_gate["total"]))
        print("Final gates pass:", report["final_gates_pass"])
        return

    hidden = core.forward_last_hidden(model, tok, cases, device, batch_size=a.cache_batch_size).float()
    H_F = hidden[F_indices]
    H_P = hidden[P_indices] if P_indices else hidden.new_empty((0, hidden.shape[1]))

    # Full protected row-space: not rank-32, not a sweep. Any Delta W in the
    # residual span is orthogonal to every observed Stage1-success hidden vector.
    protected_basis = (
        core.orthonormal_row_basis(H_P, max_rank=None).to(device=device, dtype=torch.float32)
        if H_P.numel() else hidden.new_empty((0, hidden.shape[1]), dtype=torch.float32)
    )
    residual = H_F - (H_F @ protected_basis.transpose(0, 1)) @ protected_basis if protected_basis.numel() else H_F
    repair_basis = core.orthonormal_row_basis(residual, max_rank=None).to(device=device, dtype=torch.float32)
    if repair_basis.ndim != 2 or repair_basis.shape[0] == 0:
        raise RuntimeError("No repair direction remains after exact P-subspace removal")

    # Numerical orthogonality diagnostic.
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

    initial_f = geometry.report(F_indices, delta.effective_delta(), a.constraint_margin)
    initial_p = geometry.report(P_indices, delta.effective_delta(), a.constraint_margin)
    best_state = v2.capture_state(delta)
    best_key = (
        int(initial_f["failed"]),
        -float(initial_f["minimum_margin"]),
        float(delta.effective_delta().detach().float().norm().cpu()),
    )
    best_step = 0
    best_f = initial_f
    best_p = initial_p
    accepted_steps = 0
    rolled_back_steps = 0
    logs = []

    log_path = out / "repair_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_f:
        for step in range(1, a.repair_steps + 1):
            local = sampler.next()
            ids = [F_indices[i] for i in local]
            optimizer.zero_grad(set_to_none=True)
            delta_rows = delta.effective_delta()
            margins, _ = geometry.metrics(ids, delta_rows)
            hinge = F.relu(float(a.constraint_margin) - margins).square().mean()
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

            # Hard all-P guard remains, although nullspace geometry should make
            # every scale feasible up to floating-point noise.
            accepted_scale = 1.0
            p_report = geometry.report(P_indices, delta.effective_delta(), a.constraint_margin)
            feasible = int(p_report["failed"]) == 0 and float(p_report["kl_mean"]) <= float(a.max_protected_kl)
            if not feasible:
                accepted_scale = 0.0
                for scale in backtrack_scales:
                    v2.interpolate_state(delta, old_state, proposed_state, scale)
                    trial_p = geometry.report(P_indices, delta.effective_delta(), a.constraint_margin)
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
                p_report = geometry.report(P_indices, delta.effective_delta(), a.constraint_margin)
            else:
                accepted_steps += 1

            f_report = geometry.report(F_indices, delta.effective_delta(), a.constraint_margin)
            norm = float(delta.effective_delta().detach().float().norm().cpu())
            key = (int(f_report["failed"]), -float(f_report["minimum_margin"]), norm)
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
                "F_minimum_margin": float(f_report["minimum_margin"]),
                "P_regressions": int(p_report["failed"]),
                "P_exact_kl_mean": float(p_report["kl_mean"]),
                "P_exact_kl_max": float(p_report["kl_max"]),
                "delta_norm": norm,
                "best_step": int(best_step),
            }
            logs.append(row)
            if step == 1 or step % a.check_every == 0 or step == a.repair_steps or int(f_report["failed"]) == 0:
                print(
                    "Stage2-v3 step {s}: F_fail={ff} P_reg={pr} KL={kl:.6g} scale={sc:g} ||dW||={n:.6g}".format(
                        s=step, ff=f_report["failed"], pr=p_report["failed"], kl=p_report["kl_mean"], sc=accepted_scale, n=norm
                    )
                )
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()
            if int(f_report["failed"]) == 0:
                break

    v2.restore_state(delta, best_state)
    best_delta = delta.effective_delta().detach()
    best_f = geometry.report(F_indices, best_delta, a.constraint_margin)
    best_p = geometry.report(P_indices, best_delta, a.constraint_margin)

    core.materialize_output_delta(output_layer, A_F, best_delta)
    model.eval()
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    final_logits = core.cache_base_logits(model, tok, cases, device, batch_size=a.cache_batch_size)
    final_gate = v2.gate_from_logits(final_logits, target_ids.cpu(), a.constraint_margin)
    final_failed = set(final_gate["residual_indices"])
    p_regressions = sum(1 for i in P_indices if i in final_failed)
    protected_kl = v2.actual_full_kl_mean(base_logits, final_logits, P_indices, a.cache_batch_size)
    frozen_after = None if a.skip_frozen_hash else v2.hash_frozen(model, output_layer.weight)
    frozen_exact = frozen_before == frozen_after
    final_pass = bool(
        int(final_gate["failed"]) == 0
        and int(p_regressions) == 0
        and float(protected_kl) <= float(a.max_protected_kl)
        and frozen_exact
    )

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
        "parameterization": "Delta W_A_F=C_F B_F; B_F in orthogonal complement of full H_P rowspace",
        "repair_objective": "bounded squared hinge on required forget margin",
        "hard_protection": "all P must remain passed; exact full-vocabulary Stage1||Stage2 KL budget",
        "repair_lr": float(a.repair_lr),
        "repair_steps_requested": int(a.repair_steps),
        "accepted_steps": int(accepted_steps),
        "rolled_back_steps": int(rolled_back_steps),
        "best_checkpoint_step": int(best_step),
        "best_selection_key": list(best_key),
        "best_pre_materialization_F": best_f,
        "best_pre_materialization_P": best_p,
        "materialized_delta_norm": float(best_delta.float().norm().cpu()),
        "logs": logs,
    }
    report["final_gate"] = final_gate
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
        },
        out / "stage2_nullspace_state.pt",
    )

    print("Level1 gate: {}/{} pass; F={}; P={}".format(level1_gate["passed"], level1_gate["total"], len(F_indices), len(P_indices)))
    print("Stage2-v3: A_F={}, P-basis-rank={}, repair-rank={}, nullspace-leak={:.6g}, best_step={}".format(
        len(A_F), protected_basis.shape[0], repair_basis.shape[0], protected_leak, best_step
    ))
    print("Final gate: {}/{} pass; protected_KL={:.6g}; Stage1 regressions={}; frozen_non_head_exact={}".format(
        final_gate["passed"], final_gate["total"], protected_kl, p_regressions, frozen_exact
    ))
    print("Final gates pass:", final_pass)
    print("Checkpoint:", ckpt)


if __name__ == "__main__":
    main()
