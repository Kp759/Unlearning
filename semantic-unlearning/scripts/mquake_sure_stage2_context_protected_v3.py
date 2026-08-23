#!/usr/bin/env python3
"""MQuAKE SURE v3 Stage 2 with training-visible context protection.

This is a v3-compatible utility-preserving repair variant.  It uses exactly the
same locked direct target_true cases as v3 and never reads benchmark retain,
AtomicGen, target_new, paraphrase, neighborhood, or multihop data.

Let P be Stage-1-success token decisions, F Stage-1 failures, and H_NS the
preceding non-prediction context hidden states already used by Stage 1 v3.
Instead of protecting only rowspace(H_P), Stage 2 protects

    B_G = rowspace([H_P ; H_NS])

and repairs failures only in its orthogonal complement:

    R_F = H_F - Proj_{B_G}(H_F)
    B_F = rowspace(R_F)
    Delta W_A_F = C_F B_F.

Thus the strong training-calibrated repair margin can act on F while Stage-2
logit changes are structurally suppressed on the training-visible non-sensitive
context subspace.  The repair margin is the same p-median rule used by the
training-calibrated v3 variant.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import mquake_sure_stage1_directional as s1v2
import mquake_sure_stage2_head_directional as v2
from mquake_sure_stage2_head_nullspace_v3 import margins_from_logits


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
    p.add_argument("--protected-context-tokens", type=int, default=4)
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


def context_selected_logits(h_ns: torch.Tensor, output_layer, row_ids) -> torch.Tensor:
    if not row_ids or h_ns.numel() == 0:
        return torch.empty((int(h_ns.shape[0]), 0), dtype=torch.float32)
    ids = torch.tensor(row_ids, dtype=torch.long, device=output_layer.weight.device)
    rows = output_layer.weight.index_select(0, ids)
    vals = F.linear(h_ns.to(device=rows.device, dtype=rows.dtype), rows).float()
    return vals.detach().cpu()


def main() -> None:
    a = parse_args()
    if min(a.repair_steps, a.batch_size, a.cache_batch_size, a.protected_context_tokens, a.check_every) <= 0:
        raise ValueError("steps/batches/context/check interval must be positive")
    if a.repair_lr <= 0 or a.max_protected_kl < 0 or a.l2_weight < 0 or a.constraint_margin < 0:
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

    if not F_indices:
        raise RuntimeError("Stage1 already passes all direct constraints; context-protected Stage2 is unnecessary")

    stage1_margins = margins_from_logits(base_logits, target_ids)
    p_idx = torch.tensor(P_indices, dtype=torch.long, device=stage1_margins.device)
    p_median = float(torch.median(stage1_margins.index_select(0, p_idx)).detach().cpu()) if P_indices else float(a.constraint_margin)
    repair_margin = max(float(a.constraint_margin), p_median)

    hidden = core.forward_last_hidden(model, tok, cases, device, batch_size=a.cache_batch_size).float()
    H_F = hidden[F_indices]
    H_P = hidden[P_indices] if P_indices else hidden.new_empty((0, hidden.shape[1]))
    H_NS = s1v2.collect_preceding_context_hidden(
        model,
        tok,
        cases,
        device,
        batch_size=a.cache_batch_size,
        last_n=a.protected_context_tokens,
    ).float()

    H_guard = torch.cat([H_P, H_NS], dim=0) if H_P.numel() else H_NS
    protected_basis = core.orthonormal_row_basis(H_guard, max_rank=None).to(device=device, dtype=torch.float32)
    residual = H_F - (H_F @ protected_basis.transpose(0, 1)) @ protected_basis if protected_basis.numel() else H_F
    repair_basis = core.orthonormal_row_basis(residual, max_rank=None).to(device=device, dtype=torch.float32)
    if repair_basis.ndim != 2 or repair_basis.shape[0] == 0:
        raise RuntimeError("No repair direction remains after P+H_NS subspace removal")

    p_leak = float((H_P @ repair_basis.transpose(0, 1)).abs().max().detach().cpu()) if H_P.numel() else 0.0
    ns_leak = float((H_NS @ repair_basis.transpose(0, 1)).abs().max().detach().cpu()) if H_NS.numel() else 0.0
    residual_energy = float((residual.square().sum() / H_F.square().sum().clamp_min(1e-12)).detach().cpu())

    special = set(gagd.special_token_ids(tok))
    A_F = sorted(set(int(x) for x in target_ids[F_indices].detach().cpu().tolist()) - special)
    if not A_F:
        raise RuntimeError("A_F empty for non-empty F")

    context_logits_before = context_selected_logits(H_NS, output_layer, A_F)

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

    initial_f = geometry.report(F_indices, delta.effective_delta(), repair_margin)
    initial_p = geometry.report(P_indices, delta.effective_delta(), a.constraint_margin)
    best_state = v2.capture_state(delta)
    best_key = (int(initial_f["failed"]), -float(initial_f["minimum_margin"]), 0.0)
    best_step = 0
    accepted_steps = 0
    rolled_back_steps = 0
    logs = []

    out = gagd.resolve_output_path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint"
    log_path = out / "repair_log.jsonl"

    print(
        "Stage2-context-v3: protection_margin={:.6g} repair_margin={:.6g} P={} H_NS={} guard_rank={} repair_rank={} P_leak={:.6g} NS_leak={:.6g}".format(
            a.constraint_margin, repair_margin, len(P_indices), H_NS.shape[0], protected_basis.shape[0], repair_basis.shape[0], p_leak, ns_leak
        )
    )

    with log_path.open("w", encoding="utf-8") as log_f:
        for step in range(1, a.repair_steps + 1):
            local = sampler.next()
            ids = [F_indices[i] for i in local]
            optimizer.zero_grad(set_to_none=True)
            delta_rows = delta.effective_delta()
            margins, _ = geometry.metrics(ids, delta_rows)
            hinge = F.relu(float(repair_margin) - margins).square().mean()
            l2 = delta_rows.square().mean()
            loss = hinge + float(a.l2_weight) * l2
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage2 loss at step {step}")

            old_state = v2.capture_state(delta)
            old_opt = copy.deepcopy(optimizer.state_dict())
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, a.grad_clip) if a.grad_clip > 0 else None
            if grad_norm is not None and not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite Stage2 gradient at step {step}")
            optimizer.step()
            proposed_state = v2.capture_state(delta)

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
                rolled_back_steps += 1
                accepted_scale = 0.0
                p_report = geometry.report(P_indices, delta.effective_delta(), a.constraint_margin)
            else:
                accepted_steps += 1

            f_report = geometry.report(F_indices, delta.effective_delta(), repair_margin)
            norm = float(delta.effective_delta().detach().float().norm().cpu())
            key = (int(f_report["failed"]), -float(f_report["minimum_margin"]), norm)
            if key < best_key:
                best_key = key
                best_state = v2.capture_state(delta)
                best_step = step

            row = {
                "step": step,
                "repair_hinge": float(hinge.detach().cpu()),
                "accepted_scale": float(accepted_scale),
                "F_failed_at_repair_margin": int(f_report["failed"]),
                "F_minimum_margin": float(f_report["minimum_margin"]),
                "P_regressions": int(p_report["failed"]),
                "P_kl_mean": float(p_report["kl_mean"]),
                "delta_norm": norm,
                "best_step": int(best_step),
            }
            logs.append(row)
            if step == 1 or step % a.check_every == 0 or step == a.repair_steps or int(f_report["failed"]) == 0:
                print(
                    "Stage2-context-v3 step {s}: F_fail@{rm:.4g}={ff} P_reg={pr} KL={kl:.6g} ||dW||={n:.6g}".format(
                        s=step, rm=repair_margin, ff=f_report["failed"], pr=p_report["failed"], kl=p_report["kl_mean"], n=norm
                    )
                )
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()
            if int(f_report["failed"]) == 0:
                break

    v2.restore_state(delta, best_state)
    best_delta = delta.effective_delta().detach()
    best_f = geometry.report(F_indices, best_delta, repair_margin)
    best_p = geometry.report(P_indices, best_delta, a.constraint_margin)

    core.materialize_output_delta(output_layer, A_F, best_delta)
    model.eval()

    final_logits = core.cache_base_logits(model, tok, cases, device, batch_size=a.cache_batch_size)
    final_gate = v2.gate_from_logits(final_logits, target_ids.cpu(), a.constraint_margin)
    final_margins = margins_from_logits(final_logits, target_ids)
    fidx = torch.tensor(F_indices, dtype=torch.long, device=final_margins.device)
    final_f_margins = final_margins.index_select(0, fidx)
    bad_f = torch.nonzero(final_f_margins < float(repair_margin), as_tuple=False).flatten()
    final_repair_gate = {
        "total": len(F_indices),
        "passed": len(F_indices) - int(bad_f.numel()),
        "failed": int(bad_f.numel()),
        "minimum_margin": float(final_f_margins.min().detach().cpu()),
        "mean_margin": float(final_f_margins.mean().detach().cpu()),
        "required_margin": float(repair_margin),
    }
    final_failed = set(final_gate["residual_indices"])
    p_regressions = sum(1 for i in P_indices if i in final_failed)
    protected_kl = v2.actual_full_kl_mean(base_logits, final_logits, P_indices, a.cache_batch_size)

    context_logits_after = context_selected_logits(H_NS, output_layer, A_F)
    if context_logits_before.numel():
        context_drift = (context_logits_after - context_logits_before).abs()
        context_drift_max = float(context_drift.max().item())
        context_drift_mean = float(context_drift.mean().item())
    else:
        context_drift_max = 0.0
        context_drift_mean = 0.0

    frozen_after = None if a.skip_frozen_hash else v2.hash_frozen(model, output_layer.weight)
    frozen_exact = frozen_before == frozen_after
    final_pass = bool(
        int(final_gate["failed"]) == 0
        and int(final_repair_gate["failed"]) == 0
        and int(p_regressions) == 0
        and float(protected_kl) <= float(a.max_protected_kl)
        and frozen_exact
    )

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    report: Dict[str, Any] = {
        "schema_version": 1,
        "method": "MQuAKE SURE v3 Context-Protected Training-Calibrated Stage2",
        "protocol": "training-visible-direct-only",
        "source_protocol": manifest.get("protocol"),
        "constraint_margin": float(a.constraint_margin),
        "repair_margin_mode": "p-median",
        "stage1_P_median_margin": float(p_median),
        "repair_margin": float(repair_margin),
        "protected_context_tokens": int(a.protected_context_tokens),
        "level1_gate": level1_gate,
        "level2": {
            "F": len(F_indices),
            "P": len(P_indices),
            "A_F": A_F,
            "A_F_count": len(A_F),
            "H_NS_rows": int(H_NS.shape[0]),
            "P_basis_input_rows": int(H_P.shape[0]),
            "combined_guard_basis_rank": int(protected_basis.shape[0]),
            "repair_basis_rank": int(repair_basis.shape[0]),
            "P_nullspace_leak": float(p_leak),
            "NS_nullspace_leak": float(ns_leak),
            "residual_hidden_energy_fraction": residual_energy,
            "best_checkpoint_step": int(best_step),
            "best_pre_materialization_F": best_f,
            "best_pre_materialization_P": best_p,
            "materialized_delta_norm": float(best_delta.float().norm().cpu()),
            "accepted_steps": int(accepted_steps),
            "rolled_back_steps": int(rolled_back_steps),
            "logs": logs,
        },
        "final_gate": final_gate,
        "final_F_repair_margin_gate": final_repair_gate,
        "stage1_successes_regressed": int(p_regressions),
        "protected_kl": float(protected_kl),
        "training_context_selected_logit_drift_max": context_drift_max,
        "training_context_selected_logit_drift_mean": context_drift_mean,
        "frozen_non_head_exact": bool(frozen_exact),
        "final_gates_pass": bool(final_pass),
        "benchmark_retain_seen": 0,
        "official_atomicgen_seen": 0,
        "target_new_seen": False,
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out / "two_stage_summary.json", report)
    torch.save(
        {
            "protected_basis": protected_basis.detach().cpu(),
            "repair_basis": repair_basis.detach().cpu(),
            "A_F": A_F,
            "best_delta": best_delta.float().cpu(),
            "constraint_margin": float(a.constraint_margin),
            "repair_margin": float(repair_margin),
            "repair_margin_mode": "p-median",
            "protected_context_tokens": int(a.protected_context_tokens),
        },
        out / "stage2_nullspace_state.pt",
    )

    print("===== CONTEXT-PROTECTED v3 TRAINING-ONLY DIGEST =====")
    print("repair_margin:", repair_margin)
    print("H_NS_rows:", H_NS.shape[0])
    print("combined_guard_basis_rank:", protected_basis.shape[0])
    print("repair_basis_rank:", repair_basis.shape[0])
    print("residual_hidden_energy_fraction:", residual_energy)
    print("P_nullspace_leak:", p_leak)
    print("NS_nullspace_leak:", ns_leak)
    print("materialized_delta_norm:", float(best_delta.float().norm().cpu()))
    print("training_context_selected_logit_drift_max:", context_drift_max)
    print("final_direct_gate:", final_gate)
    print("final_F_repair_margin_gate:", final_repair_gate)
    print("stage1_successes_regressed:", p_regressions)
    print("protected_kl:", protected_kl)
    print("frozen_non_head_exact:", frozen_exact)
    print("final_gates_pass:", final_pass)
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
