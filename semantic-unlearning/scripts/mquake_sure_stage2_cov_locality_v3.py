#!/usr/bin/env python3
"""SURE-v3 Stage 2 with Wikipedia covariance geometry and relation locality.

The transformer and input embeddings remain frozen.  As in v3, P denotes every
Stage-1-success direct token decision and F the residual failures.  We first
construct the numerically stable P-nullspace repair space B_F.  We then load an
external Wikipedia second moment C_wiki of the final hidden states feeding the
LM head and whiten B_F under that metric:

    G = B_F C_wiki B_F^T
    B_cov = G^{-1/2} B_F.

Updates are Delta W_A_F = C_F B_cov.  Consequently Euclidean coefficient norm
corresponds (up to the reported numerical floor) to expected squared Wikipedia
logit disturbance, without introducing a hand-tuned utility-loss weight.

Relation locality is a hard control, not a loss.  Training-visible relation
prompt templates are paired deterministically with external Wikipedia titles.
The Stage-1 model's own top-1 prediction on each resulting prompt is cached and
every accepted Stage-2 step must preserve all of those predictions.

No MQuAKE retain, AtomicGen, target_new, paraphrase, neighborhood, or multihop
field is read.  Wikipedia is explicitly external unlabeled utility data.
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
    p.add_argument("--wiki-covariance", required=True)
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
    p.add_argument("--locality-prompts", type=int, default=1000)
    p.add_argument("--cov-eigen-floor-rtol", type=float, default=1e-8)
    p.add_argument("--contraction-steps", type=int, default=40)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    p.add_argument("--skip-frozen-hash", action="store_true")
    return p.parse_args()


def svd_row_basis64(rows: torch.Tensor, rtol: float) -> torch.Tensor:
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


def stable_p_nullspace(H_F: torch.Tensor, H_P: torch.Tensor):
    device = H_F.device
    hidden = H_F.shape[1]
    if H_P.numel():
        B_P = svd_row_basis64(H_P, rtol=1e-10).to(device=device)
    else:
        B_P = torch.empty((0, hidden), dtype=torch.float64, device=device)
    residual = project_out64(H_F.double(), B_P)
    B_F = svd_row_basis64(residual, rtol=1e-8).to(device=device)
    if B_F.numel() == 0:
        raise RuntimeError("No repair direction remains after P-nullspace removal")
    for _ in range(2):
        B_F = project_out64(B_F, B_P)
        q, _ = torch.linalg.qr(B_F.transpose(0, 1), mode="reduced")
        B_F = q.transpose(0, 1).contiguous()
    leak64 = float((B_P @ B_F.transpose(0, 1)).abs().max().cpu()) if B_P.numel() else 0.0
    eye = torch.eye(B_F.shape[0], dtype=torch.float64, device=device)
    gram64 = float((B_F @ B_F.transpose(0, 1) - eye).abs().max().cpu())
    if leak64 > 1e-8:
        raise RuntimeError(f"P-nullspace leak too large: {leak64}")
    return B_P, B_F, residual, leak64, gram64


def load_covariance(path: Path, hidden_size: int):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "covariance" not in payload:
        raise RuntimeError("Wikipedia covariance file lacks covariance tensor")
    cov = payload["covariance"].detach().float().cpu()
    meta = dict(payload.get("metadata", {}))
    if cov.ndim != 2 or cov.shape != (hidden_size, hidden_size):
        raise RuntimeError(f"covariance shape {tuple(cov.shape)} != {(hidden_size, hidden_size)}")
    if int(meta.get("document_count", 0)) <= 0 or int(meta.get("state_count", 0)) <= 0:
        raise RuntimeError("Wikipedia covariance metadata is incomplete")
    return cov, meta


def covariance_whitened_basis(B_F64: torch.Tensor, cov_cpu: torch.Tensor, floor_rtol: float):
    if floor_rtol <= 0:
        raise ValueError("covariance eigen floor rtol must be positive")
    device = B_F64.device
    C = cov_cpu.to(device=device, dtype=torch.float64)
    G = B_F64 @ C @ B_F64.transpose(0, 1)
    G = 0.5 * (G + G.transpose(0, 1))
    evals, evecs = torch.linalg.eigh(G)
    lambda_max = float(evals.max().cpu())
    if lambda_max <= 0:
        raise RuntimeError("restricted Wikipedia covariance is not positive")
    floor = max(lambda_max * float(floor_rtol), 1e-12)
    clamped = evals.clamp_min(floor)
    invsqrt = (evecs * clamped.rsqrt().unsqueeze(0)) @ evecs.transpose(0, 1)
    B_cov = invsqrt @ B_F64
    metric_gram = B_cov @ C @ B_cov.transpose(0, 1)
    eye = torch.eye(B_cov.shape[0], dtype=torch.float64, device=device)
    whitening_error = float((metric_gram - eye).abs().max().cpu())
    return B_cov, metric_gram, {
        "restricted_eigen_min_raw": float(evals.min().cpu()),
        "restricted_eigen_max_raw": lambda_max,
        "eigen_floor": float(floor),
        "eigen_floor_rtol": float(floor_rtol),
        "raw_condition_estimate": float(lambda_max / max(float(evals.min().cpu()), 1e-30)) if float(evals.min().cpu()) > 0 else None,
        "clamped_condition": float(clamped.max().cpu() / clamped.min().cpu()),
        "whitening_identity_max_abs_error": whitening_error,
    }


def normalize_text(x: str) -> str:
    return " ".join(str(x).casefold().split())


def relation_templates(records: Sequence[Dict[str, Any]]):
    by_key: Dict[str, Tuple[str, str]] = {}
    for record in records:
        rr = record.get("requested_rewrite", {})
        if not isinstance(rr, dict):
            continue
        template = str(rr.get("prompt", "")).strip()
        subject = str(rr.get("subject", "")).strip()
        if not template:
            continue
        relation_id = str(rr.get("relation_id", "")).strip()
        key = relation_id if relation_id else template
        by_key.setdefault(key, (template, subject))
    if not by_key:
        raise RuntimeError("No training-visible relation prompt templates found")
    return [(key, *by_key[key]) for key in sorted(by_key)]


def render_relation_prompt(template: str, exemplar_subject: str, title: str) -> Tuple[str, str]:
    if "{}" in template:
        return template.format(title), "format_placeholder"
    if exemplar_subject and exemplar_subject in template:
        return template.replace(exemplar_subject, title, 1), "replace_subject"
    return f"{template} {title}", "append_title_fallback"


def build_external_locality_prompts(
    records: Sequence[Dict[str, Any]], wiki_meta: Dict[str, Any], requested: int
):
    if requested <= 0:
        raise ValueError("locality prompt count must be positive")
    rels = relation_templates(records)
    train_subjects = {
        normalize_text(str(r.get("requested_rewrite", {}).get("subject", "")))
        for r in records
        if isinstance(r.get("requested_rewrite", {}), dict)
    }
    titles = []
    for item in wiki_meta.get("documents", []):
        title = str(item.get("title", "")).strip() if isinstance(item, dict) else ""
        if title and normalize_text(title) not in train_subjects:
            titles.append(title)
    if not titles:
        raise RuntimeError("Wikipedia covariance metadata has no usable external titles")
    n = min(int(requested), len(titles))
    prompts: List[str] = []
    audit: List[Dict[str, Any]] = []
    for i in range(n):
        relation_key, template, subject = rels[i % len(rels)]
        prompt, mode = render_relation_prompt(template, subject, titles[i])
        prompts.append(prompt)
        audit.append({"relation_key": relation_key, "title": titles[i], "render_mode": mode})
    return prompts, audit, len(rels), len(titles)


class LocalityGeometry:
    def __init__(
        self,
        hidden: torch.Tensor,
        base_selected: torch.Tensor,
        fixed_max: torch.Tensor,
        fixed_id: torch.Tensor,
        baseline_top1: torch.Tensor,
        selected_row_ids: Sequence[int],
        device: torch.device,
    ) -> None:
        self.hidden = hidden.float().to(device)
        self.base_selected = base_selected.float().to(device)
        self.fixed_max = fixed_max.float().to(device)
        self.fixed_id = fixed_id.long().to(device)
        self.baseline_top1 = baseline_top1.long().to(device)
        self.row_ids = torch.tensor([int(x) for x in selected_row_ids], dtype=torch.long, device=device)

    def report(self, delta_rows: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            selected = self.base_selected + self.hidden @ delta_rows.float().transpose(0, 1)
            sel_max, sel_pos = selected.max(dim=-1)
            sel_id = self.row_ids.index_select(0, sel_pos)
            use_selected = sel_max >= self.fixed_max
            final_id = torch.where(use_selected, sel_id, self.fixed_id)
            changed = final_id != self.baseline_top1
            return {
                "total": int(final_id.numel()),
                "regressions": int(changed.sum().item()),
                "preserved": int(final_id.numel() - changed.sum().item()),
            }


@torch.no_grad()
def cache_locality_geometry(model, tok, prompts: Sequence[str], row_ids: Sequence[int], device, batch_size: int):
    ids = torch.tensor([int(x) for x in row_ids], dtype=torch.long, device=device)
    hs: List[torch.Tensor] = []
    bs: List[torch.Tensor] = []
    fmax: List[torch.Tensor] = []
    fid: List[torch.Tensor] = []
    top: List[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        texts = prompts[start : start + batch_size]
        enc = tok(list(texts), padding=True, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        pos = enc["attention_mask"].sum(dim=1) - 1
        rr = torch.arange(len(texts), device=device)
        logits = out.logits[rr, pos, :].float()
        hidden = out.hidden_states[-1][rr, pos, :].float()
        selected = logits.index_select(1, ids)
        fixed = logits.clone()
        fixed[:, ids] = -torch.inf
        maxv, maxi = fixed.max(dim=-1)
        hs.append(hidden.detach().cpu())
        bs.append(selected.detach().cpu())
        fmax.append(maxv.detach().cpu())
        fid.append(maxi.detach().cpu())
        top.append(logits.argmax(dim=-1).detach().cpu())
    h = torch.cat(hs, dim=0)
    b = torch.cat(bs, dim=0)
    fm = torch.cat(fmax, dim=0)
    fi = torch.cat(fid, dim=0)
    tp = torch.cat(top, dim=0)
    overlap = int(torch.isin(tp, torch.tensor(list(row_ids), dtype=torch.long)).sum().item())
    return LocalityGeometry(h, b, fm, fi, tp, row_ids, device), tp, overlap


@torch.no_grad()
def exact_locality_regressions(model, tok, prompts: Sequence[str], baseline_top1: torch.Tensor, device, batch_size: int):
    changed = 0
    cursor = 0
    for start in range(0, len(prompts), batch_size):
        texts = prompts[start : start + batch_size]
        enc = tok(list(texts), padding=True, return_tensors="pt").to(device)
        logits = model(**enc, use_cache=False).logits
        pos = enc["attention_mask"].sum(dim=1) - 1
        rr = torch.arange(len(texts), device=device)
        pred = logits[rr, pos, :].argmax(dim=-1).cpu()
        ref = baseline_top1[cursor : cursor + len(texts)]
        changed += int((pred != ref).sum().item())
        cursor += len(texts)
    return {"total": len(prompts), "regressions": changed, "preserved": len(prompts) - changed}


def coefficient_covariance_cost(delta, metric_gram64: torch.Tensor) -> torch.Tensor:
    coeff = delta.coefficients
    if coeff is None:
        raise RuntimeError("covariance-whitened repair requires coefficient parameterization")
    G = metric_gram64.to(device=coeff.device, dtype=torch.float32)
    return ((coeff @ G) * coeff).sum()


def repair_gate_from_logits(logits: torch.Tensor, target_ids: torch.Tensor, indices: Sequence[int], margin: float):
    all_m = margins_from_logits(logits, target_ids)
    idx = torch.tensor([int(x) for x in indices], dtype=torch.long, device=all_m.device)
    vals = all_m.index_select(0, idx)
    bad = vals < float(margin)
    return {
        "total": len(indices),
        "passed": len(indices) - int(bad.sum().item()),
        "failed": int(bad.sum().item()),
        "minimum_margin": float(vals.min().cpu()) if vals.numel() else None,
        "mean_margin": float(vals.mean().cpu()) if vals.numel() else None,
        "required_margin": float(margin),
    }


def storage_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def main() -> None:
    a = parse_args()
    if min(a.repair_steps, a.batch_size, a.cache_batch_size, a.check_every, a.locality_prompts, a.contraction_steps) <= 0:
        raise ValueError("steps/batches/locality/contraction settings must be positive")
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
        raise RuntimeError("no generated MQuAKE PredictionCases")

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
    p_median = float(torch.median(stage1_margins.index_select(0, p_idx)).cpu()) if P_indices else float(a.constraint_margin)
    repair_margin = max(float(a.constraint_margin), p_median)

    hidden = core.forward_last_hidden(model, tok, cases, device, batch_size=a.cache_batch_size).float()
    H_F = hidden[F_indices]
    H_P = hidden[P_indices] if P_indices else hidden.new_empty((0, hidden.shape[1]))
    B_P64, B_F64, residual64, null_leak64, repair_gram64 = stable_p_nullspace(H_F, H_P)

    cov_path = Path(a.wiki_covariance).resolve()
    cov_cpu, wiki_meta = load_covariance(cov_path, int(hidden.shape[1]))
    B_cov64, metric_gram64, cov_report = covariance_whitened_basis(B_F64, cov_cpu, a.cov_eigen_floor_rtol)
    p_cov_leak64 = float((B_P64 @ B_cov64.transpose(0, 1)).abs().max().cpu()) if B_P64.numel() else 0.0

    special = set(gagd.special_token_ids(tok))
    A_F = sorted(set(int(x) for x in target_ids[F_indices].detach().cpu().tolist()) - special)
    if not A_F:
        raise RuntimeError("A_F empty")

    locality_prompts, locality_audit, relation_count, usable_title_count = build_external_locality_prompts(
        records, wiki_meta, a.locality_prompts
    )
    locality, locality_baseline_top1, locality_top1_edited = cache_locality_geometry(
        model, tok, locality_prompts, A_F, device, a.cache_batch_size
    )
    zero_delta = torch.zeros((len(A_F), hidden.shape[1]), dtype=torch.float32, device=device)
    initial_locality = locality.report(zero_delta)
    if int(initial_locality["regressions"]) != 0:
        raise RuntimeError("locality geometry does not reproduce Stage1 baseline at zero delta")

    print(
        "Cov-locality Stage2: repair_margin={:.6g} P={} F={} P_rank={} repair_rank={} null_leak64={:.3e} cov_white_err={:.3e} locality={} relations={}".format(
            repair_margin, len(P_indices), len(F_indices), B_P64.shape[0], B_F64.shape[0],
            null_leak64, cov_report["whitening_identity_max_abs_error"], len(locality_prompts), relation_count
        )
    )

    delta = core.SelectedRowDelta(
        len(A_F), int(output_layer.weight.shape[1]),
        direction_basis=B_cov64.to(device=device, dtype=torch.float32), device=device
    )
    geometry = v2.SelectedRowGeometry(
        base_logits=base_logits, hidden=hidden, target_ids=target_ids,
        selected_row_ids=A_F, device=device
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
    best_loc = initial_locality
    accepted_steps = 0
    rolled_back_steps = 0
    logs: List[Dict[str, Any]] = []

    out_dir = gagd.resolve_output_path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint"
    with (out_dir / "repair_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in range(1, a.repair_steps + 1):
            local = sampler.next()
            ids = [F_indices[i] for i in local]
            optimizer.zero_grad(set_to_none=True)
            drows = delta.effective_delta()
            margins, _ = geometry.metrics(ids, drows)
            hinge = F.relu(float(repair_margin) - margins).square().mean()
            cov_cost = coefficient_covariance_cost(delta, metric_gram64)
            # This is only a tiny tie-breaking regularizer.  Covariance whitening
            # itself supplies the utility geometry; no utility tradeoff weight is introduced.
            loss = hinge + float(a.l2_weight) * cov_cost / max(len(A_F), 1)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage2 loss at step {step}")

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
            loc_report = locality.report(delta.effective_delta())
            feasible = (
                int(p_report["failed"]) == 0
                and float(p_report["kl_mean"]) <= float(a.max_protected_kl)
                and int(loc_report["regressions"]) == 0
            )
            if not feasible:
                accepted_scale = 0.0
                for scale in backtrack_scales:
                    v2.interpolate_state(delta, old_state, proposed_state, scale)
                    trial_p = geometry.report(P_indices, delta.effective_delta(), a.constraint_margin)
                    trial_loc = locality.report(delta.effective_delta())
                    if (
                        int(trial_p["failed"]) == 0
                        and float(trial_p["kl_mean"]) <= float(a.max_protected_kl)
                        and int(trial_loc["regressions"]) == 0
                    ):
                        feasible = True
                        accepted_scale = float(scale)
                        p_report = trial_p
                        loc_report = trial_loc
                        break
            if not feasible:
                v2.restore_state(delta, old_state)
                optimizer.load_state_dict(old_opt)
                rolled_back_steps += 1
                accepted_scale = 0.0
                p_report = geometry.report(P_indices, delta.effective_delta(), a.constraint_margin)
                loc_report = locality.report(delta.effective_delta())
            else:
                accepted_steps += 1

            f_report = geometry.report(F_indices, delta.effective_delta(), repair_margin)
            cov_value = float(coefficient_covariance_cost(delta, metric_gram64).detach().cpu())
            key = (int(f_report["failed"]), -float(f_report["minimum_margin"]), cov_value)
            if key < best_key:
                best_key = key
                best_state = v2.capture_state(delta)
                best_step = step
                best_f = f_report
                best_p = p_report
                best_loc = loc_report

            row = {
                "step": step,
                "repair_hinge": float(hinge.detach().cpu()),
                "covariance_cost": cov_value,
                "accepted_scale": float(accepted_scale),
                "F_failed_at_repair_margin": int(f_report["failed"]),
                "F_minimum_margin": float(f_report["minimum_margin"]),
                "P_regressions": int(p_report["failed"]),
                "P_kl_mean": float(p_report["kl_mean"]),
                "relation_locality_regressions": int(loc_report["regressions"]),
                "delta_norm": float(delta.effective_delta().detach().float().norm().cpu()),
                "best_step": int(best_step),
            }
            logs.append(row)
            if step == 1 or step % a.check_every == 0 or int(f_report["failed"]) == 0 or step == a.repair_steps:
                print(
                    "Cov-locality step {s}: F_fail@{m:.4g}={ff} P_reg={pr} P_KL={kl:.3g} Loc_reg={lr} cov_cost={cc:.6g} ||dW||={n:.6g}".format(
                        s=step, m=repair_margin, ff=f_report["failed"], pr=p_report["failed"],
                        kl=p_report["kl_mean"], lr=loc_report["regressions"], cc=cov_value,
                        n=row["delta_norm"]
                    )
                )
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()
            if int(f_report["failed"]) == 0:
                break

    v2.restore_state(delta, best_state)
    best_delta = delta.effective_delta().detach()
    best_coeff_cost = float(coefficient_covariance_cost(delta, metric_gram64).detach().cpu())
    best_f = geometry.report(F_indices, best_delta, repair_margin)
    best_p = geometry.report(P_indices, best_delta, a.constraint_margin)
    best_loc = locality.report(best_delta)
    if int(best_f["failed"]) != 0 or int(best_p["failed"]) != 0 or int(best_loc["regressions"]) != 0:
        raise RuntimeError("Cov-locality Stage2 did not reach a float-space feasible repair")

    # Deterministically contract the first feasible ray to the smallest scalar
    # found by bisection that still satisfies every training-side hard gate.
    def float_feasible(scale: float):
        d = best_delta * float(scale)
        fr = geometry.report(F_indices, d, repair_margin)
        pr = geometry.report(P_indices, d, a.constraint_margin)
        lr = locality.report(d)
        ok = int(fr["failed"]) == 0 and int(pr["failed"]) == 0 and float(pr["kl_mean"]) <= float(a.max_protected_kl) and int(lr["regressions"]) == 0
        return ok, fr, pr, lr

    low, high = 0.0, 1.0
    for _ in range(a.contraction_steps):
        mid = 0.5 * (low + high)
        ok, _, _, _ = float_feasible(mid)
        if ok:
            high = mid
        else:
            low = mid
    contraction_scale = float(high)
    contracted_delta = best_delta * contraction_scale
    contracted_ok, contracted_f, contracted_p, contracted_loc = float_feasible(contraction_scale)
    if not contracted_ok:
        raise RuntimeError("contracted covariance repair unexpectedly infeasible")
    contracted_cov_cost = best_coeff_cost * contraction_scale * contraction_scale

    row_tensor = torch.tensor(A_F, dtype=torch.long, device=output_layer.weight.device)
    stage1_rows = output_layer.weight.detach().index_select(0, row_tensor).clone()

    def reset_rows() -> None:
        with torch.no_grad():
            output_layer.weight.index_copy_(0, row_tensor, stage1_rows)

    def materialized_direct_report(scale: float):
        reset_rows()
        core.materialize_output_delta(output_layer, A_F, contracted_delta * float(scale))
        final_logits = core.cache_base_logits(model, tok, cases, device, batch_size=a.cache_batch_size)
        dg = v2.gate_from_logits(final_logits, target_ids.cpu(), a.constraint_margin)
        rg = repair_gate_from_logits(final_logits, target_ids, F_indices, repair_margin)
        failed = set(dg["residual_indices"])
        preg = sum(1 for i in P_indices if i in failed)
        pkl = v2.actual_full_kl_mean(base_logits, final_logits, P_indices, a.cache_batch_size)
        actual = output_layer.weight.detach().index_select(0, row_tensor).float() - stage1_rows.float()
        loc_approx = locality.report(actual)
        ok = int(dg["failed"]) == 0 and int(rg["failed"]) == 0 and preg == 0 and pkl <= float(a.max_protected_kl) and int(loc_approx["regressions"]) == 0
        return ok, dg, rg, preg, pkl, loc_approx, actual

    sdtype = storage_dtype(a.dtype)
    st = torch.tensor(1.0, dtype=sdtype)
    inf = torch.tensor(float("inf"), dtype=sdtype)
    attempts: List[Dict[str, Any]] = []
    chosen = None
    while float(st) <= 2.0:
        scale = float(st)
        ok, dg, rg, preg, pkl, lrep, actual = materialized_direct_report(scale)
        attempts.append({
            "scale": scale,
            "direct_failed": int(dg["failed"]),
            "repair_failed": int(rg["failed"]),
            "P_regressions": int(preg),
            "P_KL": float(pkl),
            "locality_regressions_geometry": int(lrep["regressions"]),
            "feasible_geometry": bool(ok),
        })
        print(
            "Cov-locality materialize scale={:.9g}: direct={}/{} repair={}/{} P_reg={} KL={:.3g} Loc_reg={} pass={}".format(
                scale, dg["passed"], dg["total"], rg["passed"], rg["total"], preg, pkl, lrep["regressions"], ok
            )
        )
        if ok:
            chosen = (scale, dg, rg, preg, pkl, lrep, actual)
            break
        nxt = torch.nextafter(st, inf)
        if float(nxt) <= float(st):
            break
        st = nxt
    if chosen is None:
        reset_rows()
        raise RuntimeError("No BF16-materialized covariance/locality repair satisfies direct training gates in [1,2]")

    chosen_scale, final_gate, final_repair_gate, p_regressions, protected_kl, loc_geometry_final, actual_delta = chosen
    exact_loc = exact_locality_regressions(
        model, tok, locality_prompts, locality_baseline_top1, device, a.cache_batch_size
    )
    frozen_after = None if a.skip_frozen_hash else v2.hash_frozen(model, output_layer.weight)
    frozen_exact = frozen_before == frozen_after
    final_pass = bool(
        int(final_gate["failed"]) == 0
        and int(final_repair_gate["failed"]) == 0
        and int(p_regressions) == 0
        and float(protected_kl) <= float(a.max_protected_kl)
        and int(exact_loc["regressions"]) == 0
        and frozen_exact
    )

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    residual_energy = float((residual64.square().sum() / H_F.double().square().sum().clamp_min(1e-30)).cpu())
    report: Dict[str, Any] = {
        "schema_version": 1,
        "method": "MQuAKE SURE v3 Wikipedia-Covariance + Relation-Locality Stage2",
        "protocol": "training-visible-direct plus external-unlabeled-Wikipedia utility geometry",
        "source_protocol": manifest.get("protocol"),
        "constraint_margin": float(a.constraint_margin),
        "repair_margin_mode": "p-median",
        "stage1_P_median_margin": float(p_median),
        "repair_margin": float(repair_margin),
        "level1_gate": level1_gate,
        "wiki_covariance": {
            "path": str(cov_path),
            "document_count": int(wiki_meta.get("document_count", 0)),
            "state_count": int(wiki_meta.get("state_count", 0)),
            "states_per_document": int(wiki_meta.get("states_per_document", 0)),
            "corpus_seed": wiki_meta.get("corpus_seed"),
            "hidden_size": int(wiki_meta.get("hidden_size", hidden.shape[1])),
            **cov_report,
        },
        "relation_locality": {
            "construction": "training-visible relation templates x external Wikipedia titles; preserve Stage1 self-predicted top1",
            "relation_count": int(relation_count),
            "prompt_count": int(len(locality_prompts)),
            "usable_wikipedia_title_count": int(usable_title_count),
            "baseline_top1_in_edited_rows": int(locality_top1_edited),
            "final_geometry_report": loc_geometry_final,
            "final_exact_report": exact_loc,
            "audit": locality_audit,
        },
        "level2": {
            "F": len(F_indices),
            "P": len(P_indices),
            "A_F": A_F,
            "A_F_count": len(A_F),
            "protected_basis_rank": int(B_P64.shape[0]),
            "repair_basis_rank": int(B_F64.shape[0]),
            "P_nullspace_leak64": float(null_leak64),
            "repair_basis_gram_error64": float(repair_gram64),
            "covariance_whitened_P_leak64": float(p_cov_leak64),
            "residual_hidden_energy_fraction": residual_energy,
            "best_checkpoint_step": int(best_step),
            "best_pre_contraction_F": best_f,
            "best_pre_contraction_P": best_p,
            "best_pre_contraction_locality": best_loc,
            "pre_contraction_covariance_cost": float(best_coeff_cost),
            "contraction_scale": contraction_scale,
            "post_contraction_covariance_cost": float(contracted_cov_cost),
            "contracted_F": contracted_f,
            "contracted_P": contracted_p,
            "contracted_locality": contracted_loc,
            "materialization_recovery_scale": float(chosen_scale),
            "materialized_delta_norm": float(actual_delta.norm().cpu()),
            "materialization_attempts": attempts,
            "accepted_steps": int(accepted_steps),
            "rolled_back_steps": int(rolled_back_steps),
            "logs": logs,
        },
        "final_gate": final_gate,
        "final_F_repair_margin_gate": final_repair_gate,
        "stage1_successes_regressed": int(p_regressions),
        "protected_kl": float(protected_kl),
        "max_protected_kl": float(a.max_protected_kl),
        "frozen_non_head_exact": bool(frozen_exact),
        "final_gates_pass": bool(final_pass),
        "benchmark_retain_seen": 0,
        "official_atomicgen_seen": 0,
        "target_new_seen": False,
        "paraphrase_seen": False,
        "benchmark_neighborhood_seen": False,
        "external_wikipedia_used": True,
        "relation_locality_labels": "Stage1 self-predictions on external Wikipedia-title prompts",
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "two_stage_summary.json", report)
    torch.save(
        {
            "protected_basis": B_P64.float().cpu(),
            "repair_basis": B_F64.float().cpu(),
            "covariance_whitened_basis": B_cov64.float().cpu(),
            "A_F": A_F,
            "best_delta": (contracted_delta * float(chosen_scale)).float().cpu(),
            "constraint_margin": float(a.constraint_margin),
            "repair_margin": float(repair_margin),
            "repair_margin_mode": "p-median",
            "contraction_scale": contraction_scale,
            "materialization_scale": float(chosen_scale),
            "wiki_covariance": str(cov_path),
        },
        out_dir / "stage2_nullspace_state.pt",
    )

    print("===== COVARIANCE + RELATION-LOCALITY v3 TRAINING DIGEST =====")
    print("repair_margin:", repair_margin)
    print("P_basis_rank:", B_P64.shape[0])
    print("repair_basis_rank:", B_F64.shape[0])
    print("P_nullspace_leak64:", null_leak64)
    print("wiki_documents:", wiki_meta.get("document_count"))
    print("wiki_states:", wiki_meta.get("state_count"))
    print("covariance_whitening_error:", cov_report["whitening_identity_max_abs_error"])
    print("relation_locality_prompts:", len(locality_prompts))
    print("relation_locality_exact_regressions:", exact_loc["regressions"])
    print("pre_contraction_covariance_cost:", best_coeff_cost)
    print("contraction_scale:", contraction_scale)
    print("post_contraction_covariance_cost:", contracted_cov_cost)
    print("materialization_scale:", chosen_scale)
    print("final_direct_gate:", final_gate)
    print("final_F_repair_margin_gate:", final_repair_gate)
    print("stage1_successes_regressed:", p_regressions)
    print("protected_kl:", protected_kl)
    print("frozen_non_head_exact:", frozen_exact)
    print("final_gates_pass:", final_pass)
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
