#!/usr/bin/env python3
"""MQuAKE SURE v3 Stage 2 with row-conditioned training-visible utility protection.

Motivation: full H_NS protection is too restrictive because it forces every edited
LM-head row to be orthogonal to every training-visible context.  Utility damage,
however, is row-specific: an edited output row matters most where that token is a
legitimate next-token target.

This variant therefore keeps the original hard Stage1-success protection H_P for
all edited rows, but adds a row-specific structural guard.  For edited token row
j, collect hidden states h_t from token-index-0 direct training prompts whose
observed next token is j.  Then constrain only Delta w_j to the orthogonal
complement of those token-use states:

    Delta w_j \perp rowspace(H_P)
    Delta w_j \perp rowspace(H_use,j residualized against H_P).

No benchmark retain, AtomicGen, target_new, paraphrase, neighborhood, multihop,
or external utility corpus is read.  The repair margin remains the deterministic
p-median rule used by the training-calibrated v3 variant.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
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


def svd_row_basis64(rows: torch.Tensor, rtol: float = 1e-10) -> torch.Tensor:
    if rows.numel() == 0:
        hidden = rows.shape[-1] if rows.ndim == 2 else 0
        return rows.new_empty((0, hidden), dtype=torch.float64)
    x = rows.detach().to(dtype=torch.float64)
    _, s, vh = torch.linalg.svd(x, full_matrices=False)
    threshold = float(rtol) * s.max().clamp_min(1.0)
    rank = int((s > threshold).sum().item())
    return vh[:rank].contiguous() if rank > 0 else x.new_empty((0, x.shape[1]))


def project_out64(rows64: torch.Tensor, basis64: torch.Tensor) -> torch.Tensor:
    if rows64.numel() == 0 or basis64.numel() == 0:
        return rows64
    return rows64 - (rows64 @ basis64.transpose(0, 1)) @ basis64


def residual_basis64(rows: torch.Tensor, common64: torch.Tensor, rtol: float = 1e-10) -> torch.Tensor:
    if rows.numel() == 0:
        return torch.empty((0, common64.shape[1]), dtype=torch.float64, device=common64.device)
    x = project_out64(rows.detach().to(device=common64.device, dtype=torch.float64), common64)
    b = svd_row_basis64(x, rtol=rtol)
    if b.numel() == 0:
        return b
    # One explicit cleanup is enough because b is small and common64 is orthonormal.
    b = project_out64(b, common64)
    q, _ = torch.linalg.qr(b.transpose(0, 1), mode="reduced")
    return q.transpose(0, 1).contiguous()


@torch.no_grad()
def collect_source_prompt_token_uses(
    model,
    tok,
    cases: Sequence[core.SensitivePredictionCase],
    device: torch.device,
    *,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (hidden_t, observed token_{t+1}) from token-index-0 direct prompts only."""
    source_cases = [c for c in cases if int(c.token_index) == 0]
    rows: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    model.eval()
    for start in range(0, len(source_cases), batch_size):
        batch = source_cases[start : start + batch_size]
        encoded = tok([c.prompt for c in batch], padding=True, return_tensors="pt").to(device)
        out = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[-1]
        lengths = encoded["attention_mask"].sum(dim=1)
        input_ids = encoded["input_ids"]
        for i, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            if length <= 1:
                continue
            # Hidden state at p predicts the observed prompt token at p+1.
            rows.append(hidden[i, : length - 1, :].float().detach())
            labels.append(input_ids[i, 1:length].long().detach())
    if not rows:
        raise RuntimeError("could not collect source-prompt token-use states")
    return torch.cat(rows, dim=0), torch.cat(labels, dim=0)


class RowConditionedDelta(torch.nn.Module):
    def __init__(
        self,
        n_rows: int,
        hidden_size: int,
        common_basis: torch.Tensor,
        extra_bases: Sequence[torch.Tensor],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.raw_delta = torch.nn.Parameter(torch.zeros((n_rows, hidden_size), device=device, dtype=torch.float32))
        self.register_buffer("common_basis", common_basis.to(device=device, dtype=torch.float32))
        self.extra_names: List[str] = []
        for i, basis in enumerate(extra_bases):
            name = f"extra_basis_{i}"
            self.register_buffer(name, basis.to(device=device, dtype=torch.float32))
            self.extra_names.append(name)

    def effective_delta(self) -> torch.Tensor:
        x = self.raw_delta
        b = self.common_basis
        if b.numel():
            x = x - (x @ b.transpose(0, 1)) @ b
        rows = []
        for i, name in enumerate(self.extra_names):
            r = x[i : i + 1]
            e = getattr(self, name)
            if e.numel():
                r = r - (r @ e.transpose(0, 1)) @ e
            rows.append(r)
        return torch.cat(rows, dim=0)


def repair_gate_from_margins(margins: torch.Tensor, indices: Sequence[int], margin: float) -> Dict[str, Any]:
    idx = torch.tensor([int(x) for x in indices], dtype=torch.long, device=margins.device)
    vals = margins.index_select(0, idx)
    bad = torch.nonzero(vals < float(margin), as_tuple=False).flatten()
    return {
        "total": len(indices),
        "passed": len(indices) - int(bad.numel()),
        "failed": int(bad.numel()),
        "minimum_margin": float(vals.min().detach().cpu()) if vals.numel() else None,
        "mean_margin": float(vals.mean().detach().cpu()) if vals.numel() else None,
        "required_margin": float(margin),
    }


def main() -> None:
    a = parse_args()
    if min(a.repair_steps, a.batch_size, a.cache_batch_size, a.check_every) <= 0:
        raise ValueError("steps/batches/check interval must be positive")
    if a.repair_lr <= 0 or a.max_protected_kl < 0 or a.l2_weight < 0 or a.constraint_margin < 0:
        raise ValueError("invalid optimization settings")
    backtrack_scales = v2.parse_scales(a.backtrack_scales)

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    records, manifest = v2.load_locked(a)
    ns = argparse.Namespace(model_path=a.model_path, dtype=a.dtype, device_map=a.device_map, gradient_checkpointing=False)
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)
    cases = core.expand_sensitive_cases(records, tok, sensitive_field="target_true", llama_like=llama_like)
    if not cases:
        raise RuntimeError("no generated PredictionCases")

    output_layer = core.untie_and_freeze_output_head(model)
    if model.get_input_embeddings().weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("Stage2 requires untied LM head")
    model.eval()
    frozen_before = None if a.skip_frozen_hash else v2.hash_frozen(model, output_layer.weight)

    base_logits = core.cache_base_logits(model, tok, cases, device, batch_size=a.cache_batch_size)
    target_ids = core.official_target_ids(tok, cases, llama_like=llama_like, device=device)
    level1_gate = v2.gate_from_logits(base_logits, target_ids.cpu(), a.constraint_margin)
    F_indices = [int(x) for x in level1_gate["residual_indices"]]
    failed_set = set(F_indices)
    P_indices = [i for i in range(len(cases)) if i not in failed_set]
    if not F_indices:
        raise RuntimeError("Stage1 already passes all direct constraints")

    stage1_margins = margins_from_logits(base_logits, target_ids)
    p_idx = torch.tensor(P_indices, dtype=torch.long, device=stage1_margins.device)
    p_median = float(torch.median(stage1_margins.index_select(0, p_idx)).detach().cpu()) if P_indices else float(a.constraint_margin)
    repair_margin = max(float(a.constraint_margin), p_median)

    hidden = core.forward_last_hidden(model, tok, cases, device, batch_size=a.cache_batch_size).float()
    H_P = hidden[P_indices] if P_indices else hidden.new_empty((0, hidden.shape[1]))
    common64 = svd_row_basis64(H_P, rtol=1e-10)
    common_leak64 = 0.0
    if H_P.numel() and common64.numel():
        common_resid = project_out64(H_P.double(), common64)
        common_leak64 = float(common_resid.abs().max().detach().cpu())

    special = set(gagd.special_token_ids(tok))
    A_F = sorted(set(int(x) for x in target_ids[F_indices].detach().cpu().tolist()) - special)
    if not A_F:
        raise RuntimeError("A_F empty")

    use_hidden, use_labels = collect_source_prompt_token_uses(model, tok, cases, device, batch_size=a.cache_batch_size)
    extra_bases: List[torch.Tensor] = []
    use_counts: Dict[str, int] = {}
    use_ranks: Dict[str, int] = {}
    max_row_guard_leak64 = 0.0
    for token_id in A_F:
        mask = use_labels == int(token_id)
        h_use = use_hidden[mask]
        extra64 = residual_basis64(h_use, common64, rtol=1e-10)
        leak = 0.0
        if extra64.numel() and common64.numel():
            leak = float((common64 @ extra64.transpose(0, 1)).abs().max().detach().cpu())
        max_row_guard_leak64 = max(max_row_guard_leak64, leak)
        extra_bases.append(extra64.to(device=device, dtype=torch.float32))
        use_counts[str(token_id)] = int(h_use.shape[0])
        use_ranks[str(token_id)] = int(extra64.shape[0])

    rows_with_uses = sum(1 for v in use_counts.values() if v > 0)
    total_uses = sum(use_counts.values())
    print(
        "Row-conditioned utility guard: P={} P_rank={} prompt_use_states={} edited_rows={} rows_with_uses={} matched_uses={} common_resid64={:.3e} row_guard_leak64={:.3e}".format(
            len(P_indices), common64.shape[0], use_hidden.shape[0], len(A_F), rows_with_uses, total_uses, common_leak64, max_row_guard_leak64
        )
    )

    delta = RowConditionedDelta(
        len(A_F), int(output_layer.weight.shape[1]), common64.to(device=device, dtype=torch.float32), extra_bases, device
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
    sampler = core.IndexSampler(len(F_indices), a.batch_size, a.seed + 100003)

    initial_f = geometry.report(F_indices, delta.effective_delta(), repair_margin)
    initial_p = geometry.report(P_indices, delta.effective_delta(), a.constraint_margin)
    best_state = v2.capture_state(delta)
    best_key = (int(initial_f["failed"]), -float(initial_f["minimum_margin"]), 0.0)
    best_step = 0
    best_f = initial_f
    best_p = initial_p
    accepted_steps = 0
    rolled_back_steps = 0
    logs = []

    out = gagd.resolve_output_path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint"
    with (out / "repair_log.jsonl").open("w", encoding="utf-8") as log_f:
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
                raise FloatingPointError(f"non-finite loss at step {step}")

            old_state = v2.capture_state(delta)
            old_opt = copy.deepcopy(optimizer.state_dict())
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, a.grad_clip) if a.grad_clip > 0 else None
            if grad_norm is not None and not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient at step {step}")
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
                best_f = f_report
                best_p = p_report

            row = {
                "step": step,
                "F_failed_at_repair_margin": int(f_report["failed"]),
                "F_minimum_margin": float(f_report["minimum_margin"]),
                "P_regressions": int(p_report["failed"]),
                "P_kl_mean": float(p_report["kl_mean"]),
                "accepted_scale": float(accepted_scale),
                "delta_norm": norm,
                "best_step": int(best_step),
            }
            logs.append(row)
            if step == 1 or step % a.check_every == 0 or step == a.repair_steps or int(f_report["failed"]) == 0:
                print(
                    "Stage2-rowutil-v3 step {s}: F_fail@{rm:.4g}={ff} P_reg={pr} KL={kl:.6g} ||dW||={n:.6g}".format(
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

    # Structural row-use leakage before low-precision materialization.
    row_use_leak = 0.0
    for j, token_id in enumerate(A_F):
        mask = use_labels == int(token_id)
        if bool(mask.any()):
            vals = use_hidden[mask].float() @ best_delta[j].float()
            row_use_leak = max(row_use_leak, float(vals.abs().max().detach().cpu()))

    row_ids_tensor = torch.tensor(A_F, dtype=torch.long, device=output_layer.weight.device)
    stage1_rows = output_layer.weight.detach().index_select(0, row_ids_tensor).clone()
    core.materialize_output_delta(output_layer, A_F, best_delta)
    model.eval()
    final_logits = core.cache_base_logits(model, tok, cases, device, batch_size=a.cache_batch_size)
    final_gate = v2.gate_from_logits(final_logits, target_ids.cpu(), a.constraint_margin)
    final_margins = margins_from_logits(final_logits, target_ids)
    final_repair_gate = repair_gate_from_margins(final_margins, F_indices, repair_margin)
    final_failed = set(final_gate["residual_indices"])
    p_regressions = sum(1 for i in P_indices if i in final_failed)
    protected_kl = v2.actual_full_kl_mean(base_logits, final_logits, P_indices, a.cache_batch_size)

    actual_rows = output_layer.weight.detach().index_select(0, row_ids_tensor)
    actual_delta = (actual_rows - stage1_rows).float()
    materialized_row_use_leak = 0.0
    for j, token_id in enumerate(A_F):
        mask = use_labels == int(token_id)
        if bool(mask.any()):
            vals = use_hidden[mask].float() @ actual_delta[j].to(device=use_hidden.device)
            materialized_row_use_leak = max(materialized_row_use_leak, float(vals.abs().max().detach().cpu()))

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
        "method": "MQuAKE SURE v3 Row-Conditioned Utility-Protected Training-Calibrated Stage2",
        "protocol": "training-visible-direct-only",
        "source_protocol": manifest.get("protocol"),
        "constraint_margin": float(a.constraint_margin),
        "repair_margin_mode": "p-median",
        "stage1_P_median_margin": float(p_median),
        "repair_margin": float(repair_margin),
        "level1_gate": level1_gate,
        "level2": {
            "F": len(F_indices),
            "P": len(P_indices),
            "A_F": A_F,
            "A_F_count": len(A_F),
            "common_P_basis_rank": int(common64.shape[0]),
            "source_prompt_use_state_count": int(use_hidden.shape[0]),
            "edited_rows_with_observed_prompt_uses": int(rows_with_uses),
            "matched_prompt_use_count": int(total_uses),
            "row_prompt_use_counts": use_counts,
            "row_prompt_use_basis_ranks": use_ranks,
            "common_P_projection_residual_max64": float(common_leak64),
            "row_guard_common_leak_max64": float(max_row_guard_leak64),
            "best_checkpoint_step": int(best_step),
            "best_pre_materialization_F": best_f,
            "best_pre_materialization_P": best_p,
            "materialized_delta_norm": float(best_delta.float().norm().cpu()),
            "row_use_logit_drift_pre_materialization_max": float(row_use_leak),
            "row_use_logit_drift_materialized_max": float(materialized_row_use_leak),
            "accepted_steps": int(accepted_steps),
            "rolled_back_steps": int(rolled_back_steps),
            "logs": logs,
        },
        "final_gate": final_gate,
        "final_F_repair_margin_gate": final_repair_gate,
        "stage1_successes_regressed": int(p_regressions),
        "protected_kl": float(protected_kl),
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
            "A_F": A_F,
            "best_delta": best_delta.float().cpu(),
            "constraint_margin": float(a.constraint_margin),
            "repair_margin": float(repair_margin),
            "repair_margin_mode": "p-median",
            "row_prompt_use_counts": use_counts,
        },
        out / "stage2_nullspace_state.pt",
    )

    print("===== ROW-CONDITIONED UTILITY v3 TRAINING-ONLY DIGEST =====")
    print("repair_margin:", repair_margin)
    print("common_P_basis_rank:", common64.shape[0])
    print("source_prompt_use_state_count:", use_hidden.shape[0])
    print("edited_rows_with_observed_prompt_uses:", rows_with_uses, "/", len(A_F))
    print("matched_prompt_use_count:", total_uses)
    print("row_use_logit_drift_pre_materialization_max:", row_use_leak)
    print("row_use_logit_drift_materialized_max:", materialized_row_use_leak)
    print("best_pre_materialization_F:", best_f)
    print("materialized_delta_norm:", float(best_delta.float().norm().cpu()))
    print("final_direct_gate:", final_gate)
    print("final_F_repair_margin_gate:", final_repair_gate)
    print("stage1_successes_regressed:", p_regressions)
    print("protected_kl:", protected_kl)
    print("frozen_non_head_exact:", frozen_exact)
    print("final_gates_pass:", final_pass)
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
