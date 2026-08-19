#!/usr/bin/env python3
"""Shared two-stage contrastive SURE-LM v2 for MCF and ZsRE.

Training-visible data only:
  * direct forget prompts with the sensitive answer;
  * a disjoint prompt-only retain-train set.
Official paraphrase, neighborhood, retain-eval, and PPL prompts never participate
in training or checkpoint selection.

For each sensitive output row s, the editable basis is obtained from the
row-specific contrastive generalized eigenproblem, restricted to the observed
direct-forget context span for numerical efficiency:

    C_F,s v = lambda (C_R,s + eps I) v,

where C_R,s uses normalized/clipped frozen-Base p(s|retain) weights. Only the
untied sensitive LM-head rows are editable; transformer and input embeddings
remain Base.

Stage 1 optimizes GA + non-sensitive GD + row-weighted retain action + L2.
Scale selection enforces shared direct constraints, a retain-action budget, a
total Delta-W norm guard, and a training-visible non-sensitive-KL utility proxy.

Stage 2 is an explicit identity/no-op when Stage 1 passes. If Stage 1 has no
fully feasible scale but has a guard-safe handoff scale, Stage 2 repairs only
residual direct prediction cases. It uses active row-specific contrastive bases
against the SAME weighted C_R,s. Stage-2 retain action is computed on the TOTAL
Base-relative update Delta-W^(1)+delta-W^(2), whereas Stage-2 L2 regularizes only
the new residual repair delta-W^(2). Candidate repair ranks are restricted
(default 2,4); there is no unrestricted/full-rank fallback.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import sure_context_projection as context
import sure_retain_kl as retain
import sure_shared_suppression as shared


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("mcf", "zsre"), required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--training-visible-retain-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--retain-train-num", type=int, default=1000)

    p.add_argument("--stage1-steps", type=int, default=600)
    p.add_argument("--stage1-batch-size", type=int, default=1)
    p.add_argument("--retain-batch-size", type=int, default=64)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--stage1-lr", type=float, default=5e-3)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--gd-weight", type=float, default=1.0)
    p.add_argument("--retain-action-weight", type=float, default=1.0)
    p.add_argument("--delta-l2", type=float, default=0.0)
    p.add_argument("--contrastive-rank", type=int, default=2)
    p.add_argument("--contrastive-eps", type=float, default=1e-3)
    p.add_argument("--retain-weight-clip", type=float, default=10.0)

    p.add_argument("--constraint-margin", type=float, default=0.05)
    p.add_argument("--min-sensitive-nll-increase", type=float, default=4.0)
    p.add_argument("--retain-action-budget", type=float, default=0.25)
    p.add_argument("--max-total-delta-norm", type=float, default=1.5)
    p.add_argument("--max-forget-nonsensitive-kl", type=float, default=0.25)
    p.add_argument("--candidate-scales", default="1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0")
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--stage2-ranks", default="2,4")
    p.add_argument("--stage2-steps", type=int, default=500)
    p.add_argument("--stage2-lr", type=float, default=5e-3)
    p.add_argument("--stage2-batch-size", type=int, default=8)
    p.add_argument("--stage2-check-every", type=int, default=25)
    p.add_argument("--stage2-l2", type=float, default=1e-6)

    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def load_locked(a: argparse.Namespace):
    forget_path = Path(a.training_visible_path).resolve()
    retain_path = Path(a.training_visible_retain_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    forget_records = json.loads(forget_path.read_text(encoding="utf-8"))
    retain_records = json.loads(retain_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not isinstance(forget_records, list) or len(forget_records) != a.forget_num:
        raise RuntimeError(f"Expected {a.forget_num} direct forget records")
    if not isinstance(retain_records, list) or len(retain_records) != a.retain_train_num:
        raise RuntimeError(f"Expected {a.retain_train_num} prompt-only retain records")
    if int(manifest.get("seed", -1)) != a.seed:
        raise RuntimeError("split manifest seed mismatch")
    sampling = manifest.get("sampling", {})
    if int(sampling.get("retain_train_eval_overlap", -1)) != 0:
        raise RuntimeError("retain-train/eval overlap is not zero")
    if set(sampling.get("retain_train_case_ids", [])) & set(sampling.get("retain_eval_case_ids", [])):
        raise RuntimeError("retain-train and retain-eval IDs overlap")

    sensitive_field = core.sensitive_answer_field(a.dataset)
    for i, record in enumerate(forget_records):
        if (record.get("paraphrase_prompts") or record.get("neighborhood_prompts")
                or record.get("generation_prompts") or record.get("attribute_prompts")):
            raise RuntimeError(f"forget record {i} exposes held-out probes")
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping) or not rr.get(sensitive_field, {}).get("str"):
            raise RuntimeError(f"forget record {i} lacks sensitive field {sensitive_field}")
        if a.dataset == "zsre" and "target_new" in rr:
            raise RuntimeError("ZsRE contrastive path forbids neutral/replacement target_new")
        if a.dataset == "mcf" and "target_true" in rr:
            raise RuntimeError("MCF training-visible record must not expose reference target_true after canonical mapping")

    for i, record in enumerate(retain_records):
        if (record.get("paraphrase_prompts") or record.get("neighborhood_prompts")
                or record.get("generation_prompts") or record.get("attribute_prompts")):
            raise RuntimeError(f"retain record {i} exposes held-out probes")
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping) or not rr.get("prompt"):
            raise RuntimeError(f"retain record {i} lacks direct prompt")
        if "target_true" in rr or "target_new" in rr:
            raise RuntimeError("retain-train must be prompt-only; answer labels leaked")
    return forget_records, retain_records, manifest


@torch.no_grad()
def cache_retain_hidden_and_probs(model, tok, retain_cases, selected_ids, *, device, batch_size):
    hidden = core.forward_last_hidden(model, tok, retain_cases, device, batch_size).float()
    ids = torch.tensor([int(x) for x in selected_ids], dtype=torch.long, device=device)
    chunks: List[torch.Tensor] = []
    for start in range(0, len(retain_cases), batch_size):
        batch = retain_cases[start:start + batch_size]
        logits = core.forward_last_logits(model, tok, batch, device).float()
        log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
        selected = logits.index_select(1, ids)
        chunks.append(torch.exp(selected - log_z).detach().cpu())
    probs = torch.cat(chunks, dim=0).float()
    return hidden.detach(), probs, {
        "retain_hidden_count": int(hidden.shape[0]),
        "hidden_size": int(hidden.shape[1]),
        "raw_base_probability_mean": float(probs.mean()),
        "raw_base_probability_max": float(probs.max()),
    }


def normalized_clipped_weights(probs: torch.Tensor, clip: float) -> torch.Tensor:
    if probs.ndim != 2 or probs.shape[0] == 0 or probs.shape[1] == 0:
        raise ValueError("retain probability matrix must be [n_retain,n_rows]")
    if clip <= 0:
        raise ValueError("retain-weight-clip must be positive")
    p = probs.float().clamp_min(0.0)
    w = p / p.mean(dim=0, keepdim=True).clamp_min(1e-20)
    w = torch.clamp(w, max=float(clip))
    return w / w.mean(dim=0, keepdim=True).clamp_min(1e-20)


@torch.no_grad()
def build_contrastive_bases_from_hidden(
    forget_hidden, forget_tids, retain_hidden, retain_weights,
    *, all_selected_ids, requested_ids, rank_cap, relative_eps,
):
    if rank_cap <= 0:
        raise ValueError("contrastive rank must be positive; unrestricted rank is forbidden")
    if relative_eps <= 0:
        raise ValueError("contrastive epsilon must be positive")
    id_to_col = {int(t): j for j, t in enumerate(all_selected_ids)}
    bases, reports = [], []
    for token_id in [int(x) for x in requested_ids]:
        rows = forget_hidden[forget_tids.eq(token_id)].float()
        if rows.numel() == 0:
            raise RuntimeError(f"Sensitive token {token_id} has no direct forget hidden states")
        q = core.orthonormal_row_basis(rows, max_rank=None).float()
        if q.ndim != 2 or q.shape[0] <= 0:
            raise RuntimeError(f"Sensitive token {token_id} has zero forget-context rank")

        xf = rows @ q.transpose(0, 1)
        cf = (xf.transpose(0, 1) @ xf) / float(max(1, rows.shape[0]))
        xr = retain_hidden.float() @ q.transpose(0, 1)
        w = retain_weights[:, id_to_col[token_id]].to(device=xr.device, dtype=torch.float32)
        cr = xr.transpose(0, 1) @ (xr * w[:, None]) / w.sum().clamp_min(1e-12)

        dim = int(q.shape[0])
        avg_cr = (torch.trace(cr) / float(dim)).abs().clamp_min(1e-6)
        ridge = float(relative_eps) * avg_cr
        b = cr + ridge * torch.eye(dim, device=cr.device, dtype=torch.float32)
        chol = torch.linalg.cholesky(b)
        left = torch.linalg.solve_triangular(chol, cf, upper=False)
        whitened = torch.linalg.solve_triangular(chol, left.transpose(0, 1), upper=False).transpose(0, 1)
        whitened = 0.5 * (whitened + whitened.transpose(0, 1))
        evals, evecs = torch.linalg.eigh(whitened)
        order = torch.argsort(evals, descending=True)
        take = min(int(rank_cap), dim)
        y = evecs[:, order[:take]]
        coords = torch.linalg.solve_triangular(chol.transpose(0, 1), y, upper=True)
        directions = coords.transpose(0, 1) @ q
        basis = core.orthonormal_row_basis(directions, max_rank=take).float()
        if basis.ndim != 2 or basis.shape[0] <= 0:
            raise RuntimeError(f"Sensitive token {token_id} has zero contrastive rank")
        bases.append(basis.detach().float().contiguous())
        reports.append({
            "token_id": token_id,
            "forget_context_count": int(rows.shape[0]),
            "forget_context_rank": int(q.shape[0]),
            "requested_contrastive_rank": int(rank_cap),
            "actual_contrastive_rank": int(basis.shape[0]),
            "top_generalized_eigenvalues": [float(x) for x in evals[order[:take]].detach().cpu().tolist()],
            "retain_weight_mean": float(w.mean().cpu()),
            "retain_weight_max": float(w.max().cpu()),
            "relative_eps": float(relative_eps),
            "absolute_ridge": float(ridge.detach().cpu()),
            "domain": "observed_direct_forget_context_span",
        })
    return bases, reports


def retain_action(retain_hidden, retain_weights, delta_rows):
    h = retain_hidden.to(device=delta_rows.device, dtype=torch.float32)
    w = retain_weights.to(device=delta_rows.device, dtype=torch.float32)
    shift = h @ delta_rows.float().transpose(0, 1)
    if shift.shape != w.shape:
        raise ValueError("retain-action dimensions do not align")
    return (w * shift.square()).mean()


def total_delta_with_residual(stage1_delta, stage1_ids, residual, residual_ids):
    total = stage1_delta.clone()
    pos = {int(t): i for i, t in enumerate(stage1_ids)}
    for j, token_id in enumerate(residual_ids):
        total[pos[int(token_id)]] = total[pos[int(token_id)]] + residual[j]
    return total


@torch.no_grad()
def evaluate_forget_nonsensitive_kl(model, tok, cases, base_logits, *, llama_like, device, batch_size):
    total = 0.0
    count = 0
    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]
        logits = core.forward_last_logits(model, tok, batch, device)
        tids = core.official_target_ids(tok, batch, llama_like=llama_like, device=device)
        gd = core.gd_non_sensitive_kl(logits, base_logits[start:start + len(batch)], tids)
        total += float(gd.detach().cpu()) * len(batch)
        count += len(batch)
    return total / float(max(1, count))


def count_failures(state, a):
    return shared.count_failures(
        state["logit_margin"], state["sensitive_nll_increase"],
        required_logit_margin=a.constraint_margin,
        required_nll_increase=a.min_sensitive_nll_increase,
    )


@torch.no_grad()
def evaluate_stage1_scale(*, a, model, tok, output_layer, selected_ids, trained_delta, scale,
                          forget_cases, base_forget_logits, retain_hidden, retain_weights,
                          llama_like, device):
    scaled = trained_delta * float(scale)
    hook = core.register_output_delta_hook(output_layer, selected_ids, lambda: scaled)
    try:
        state = shared.evaluate_shared_constraints(
            model, tok, forget_cases, base_forget_logits,
            llama_like=llama_like, device=device, batch_size=a.cache_batch_size,
        )
        proxy = evaluate_forget_nonsensitive_kl(
            model, tok, forget_cases, base_forget_logits,
            llama_like=llama_like, device=device, batch_size=a.cache_batch_size,
        )
    finally:
        hook.remove()
    action = float(retain_action(retain_hidden, retain_weights, scaled).detach().cpu())
    norm = float(scaled.norm().cpu())
    failures = count_failures(state, a)
    retain_ok = action <= a.retain_action_budget
    norm_ok = norm <= a.max_total_delta_norm
    proxy_ok = proxy <= a.max_forget_nonsensitive_kl
    direct_ok = failures == 0
    return {
        "scale": float(scale),
        "direct_failures": int(failures),
        "minimum_logit_margin": float(state["logit_margin"].min().cpu()),
        "minimum_sensitive_nll_increase": float(state["sensitive_nll_increase"].min().cpu()),
        "retain_action": action,
        "total_delta_norm": norm,
        "forget_nonsensitive_kl_proxy": float(proxy),
        "guard_safe": bool(retain_ok and norm_ok and proxy_ok),
        "feasible": bool(direct_ok and retain_ok and norm_ok and proxy_ok),
    }


def choose_stage1_report(reports):
    feasible = [r for r in reports if r["feasible"]]
    if feasible:
        return min(feasible, key=lambda r: (r["retain_action"], r["total_delta_norm"], r["scale"])), "stage1_feasible"
    guarded = [r for r in reports if r["guard_safe"]]
    if guarded:
        return min(guarded, key=lambda r: (r["direct_failures"], r["retain_action"], r["total_delta_norm"], r["scale"])), "guarded_handoff_to_stage2"
    raise RuntimeError("No Stage-1 scale satisfies retain-action, norm, and utility-proxy guards")


def optimize_stage1(*, a, model, tok, output_layer, selected_ids, row_bases, forget_cases,
                    base_forget_logits, retain_hidden, retain_weights, llama_like, device, out_dir):
    dm = context.RowSpecificProjectedDelta(selected_ids, row_bases, device=output_layer.weight.device)
    opt = torch.optim.AdamW(dm.parameters(), lr=a.stage1_lr, weight_decay=0.0)
    fs = core.IndexSampler(len(forget_cases), a.stage1_batch_size, a.seed)
    rs = core.IndexSampler(len(retain_hidden), min(a.retain_batch_size, len(retain_hidden)), a.seed + 300007)
    hook = core.register_output_delta_hook(output_layer, selected_ids, dm.effective_delta)
    try:
        model.eval()
        with (out_dir / "stage1_train_log.jsonl").open("w", encoding="utf-8") as f:
            for step in range(1, a.stage1_steps + 1):
                fidx, ridx = fs.next(), rs.next()
                batch = [forget_cases[i] for i in fidx]
                opt.zero_grad(set_to_none=True)
                logits = core.forward_last_logits(model, tok, batch, device)
                tids = core.official_target_ids(tok, batch, llama_like=llama_like, device=device)
                ga = core.ga_sensitive_logprob(logits, tids)
                gd = core.gd_non_sensitive_kl(logits, base_forget_logits[fidx], tids)
                delta = dm.effective_delta()
                action = retain_action(retain_hidden[ridx], retain_weights[ridx], delta)
                l2 = delta.square().mean()
                loss = a.ga_weight * ga + a.gd_weight * gd + a.retain_action_weight * action + a.delta_l2 * l2
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite Stage-1 loss at step {step}")
                loss.backward()
                if a.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(list(dm.parameters()), a.grad_clip)
                opt.step()
                if step == 1 or step % 25 == 0 or step == a.stage1_steps:
                    f.write(json.dumps({
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "ga_sensitive_logprob": float(ga.detach().cpu()),
                        "forget_non_sensitive_gd_kl": float(gd.detach().cpu()),
                        "retain_action_batch": float(action.detach().cpu()),
                        "delta_norm": float(delta.detach().norm().cpu()),
                        "heldout_probes_seen": 0,
                        "retain_eval_seen": 0,
                    }) + "\n")
                    f.flush()
    finally:
        hook.remove()
    del opt
    return dm.effective_delta().detach().clone()


def optimize_stage2_candidate(*, a, model, tok, output_layer, stage1_ids, stage1_delta,
                              active_ids, row_bases, active_cases, active_base_logits,
                              all_forget_cases, base_forget_logits, retain_hidden, retain_weights,
                              llama_like, device, rank):
    dm = context.RowSpecificProjectedDelta(active_ids, row_bases, device=output_layer.weight.device)
    opt = torch.optim.AdamW(dm.parameters(), lr=a.stage2_lr, weight_decay=0.0)
    fs = core.IndexSampler(len(active_cases), min(a.stage2_batch_size, len(active_cases)), a.seed + 100003 + int(rank))
    rs = core.IndexSampler(len(retain_hidden), min(a.retain_batch_size, len(retain_hidden)), a.seed + 500009 + int(rank))
    best_key, best_delta, best_row = None, dm.effective_delta().detach().clone(), {}
    hook = core.register_output_delta_hook(output_layer, active_ids, dm.effective_delta)
    try:
        model.eval()
        for step in range(1, a.stage2_steps + 1):
            fidx, ridx = fs.next(), rs.next()
            batch = [active_cases[i] for i in fidx]
            opt.zero_grad(set_to_none=True)
            logits = core.forward_last_logits(model, tok, batch, device)
            tids = core.official_target_ids(tok, batch, llama_like=llama_like, device=device)
            ga = core.ga_sensitive_logprob(logits, tids)
            gd = core.gd_non_sensitive_kl(logits, active_base_logits[fidx], tids)
            residual = dm.effective_delta()
            total = total_delta_with_residual(stage1_delta, stage1_ids, residual, active_ids)
            action = retain_action(retain_hidden[ridx], retain_weights[ridx], total)
            residual_l2 = residual.square().mean()
            loss = a.ga_weight * ga + a.gd_weight * gd + a.retain_action_weight * action + a.stage2_l2 * residual_l2
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage-2 loss at step {step}")
            loss.backward()
            if a.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(dm.parameters()), a.grad_clip)
            opt.step()

            if step == 1 or step % a.stage2_check_every == 0 or step == a.stage2_steps:
                state = shared.evaluate_shared_constraints(
                    model, tok, all_forget_cases, base_forget_logits,
                    llama_like=llama_like, device=device, batch_size=a.cache_batch_size,
                )
                failures = count_failures(state, a)
                cur = dm.effective_delta().detach().clone()
                total_cur = total_delta_with_residual(stage1_delta, stage1_ids, cur, active_ids)
                action_cur = float(retain_action(retain_hidden, retain_weights, total_cur).detach().cpu())
                repair_norm = float(cur.norm().cpu())
                key = (int(failures), repair_norm, action_cur, float(total_cur.norm().cpu()))
                if best_key is None or key < best_key:
                    best_key, best_delta = key, cur
                    best_row = {
                        "step": int(step),
                        "direct_failures": int(failures),
                        "residual_repair_norm": repair_norm,
                        "retain_action_total": action_cur,
                        "total_delta_norm": float(total_cur.norm().cpu()),
                        "minimum_logit_margin": float(state["logit_margin"].min().cpu()),
                        "minimum_sensitive_nll_increase": float(state["sensitive_nll_increase"].min().cpu()),
                    }
    finally:
        hook.remove()
    del opt
    return best_delta, best_row


@torch.no_grad()
def evaluate_stage2_scale(*, a, model, tok, output_layer, stage1_ids, stage1_delta,
                          active_ids, residual_delta, scale, forget_cases, base_forget_logits,
                          retain_hidden, retain_weights, llama_like, device, rank):
    scaled = residual_delta * float(scale)
    total = total_delta_with_residual(stage1_delta, stage1_ids, scaled, active_ids)
    hook = core.register_output_delta_hook(output_layer, active_ids, lambda: scaled)
    try:
        state = shared.evaluate_shared_constraints(
            model, tok, forget_cases, base_forget_logits,
            llama_like=llama_like, device=device, batch_size=a.cache_batch_size,
        )
        proxy = evaluate_forget_nonsensitive_kl(
            model, tok, forget_cases, base_forget_logits,
            llama_like=llama_like, device=device, batch_size=a.cache_batch_size,
        )
    finally:
        hook.remove()
    action = float(retain_action(retain_hidden, retain_weights, total).detach().cpu())
    norm = float(total.norm().cpu())
    repair_norm = float(scaled.norm().cpu())
    failures = count_failures(state, a)
    feasible = failures == 0 and action <= a.retain_action_budget and norm <= a.max_total_delta_norm and proxy <= a.max_forget_nonsensitive_kl
    return {
        "rank": int(rank),
        "scale": float(scale),
        "direct_failures": int(failures),
        "minimum_logit_margin": float(state["logit_margin"].min().cpu()),
        "minimum_sensitive_nll_increase": float(state["sensitive_nll_increase"].min().cpu()),
        "residual_repair_norm": repair_norm,
        "retain_action_total": action,
        "total_delta_norm": norm,
        "forget_nonsensitive_kl_proxy": float(proxy),
        "feasible": bool(feasible),
    }


def main() -> None:
    a = parse_args()
    if a.contrastive_rank <= 0:
        raise ValueError("contrastive rank must be positive")
    ranks = core.parse_rank_list(a.stage2_ranks)
    if not ranks or any(int(r) <= 0 for r in ranks):
        raise ValueError("Stage-2 ranks must be positive; rank-0/full fallback is forbidden")
    scales = core.parse_scales(a.candidate_scales)
    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    forget_records, retain_records, manifest = load_locked(a)
    ns = argparse.Namespace(model_path=a.model_path, dtype=a.dtype, device_map=a.device_map, gradient_checkpointing=False)
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    output_layer = core.untie_and_freeze_output_head(model)
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)

    forget_cases = core.expand_sensitive_cases(forget_records, tok, dataset=a.dataset, llama_like=llama_like)
    retain_cases = retain.retain_prompt_cases(retain_records)
    forget_tids = core.official_target_ids(tok, forget_cases, llama_like=llama_like, device=device).detach()
    selected_ids = sorted(set(int(x) for x in forget_tids.detach().cpu().tolist()))
    base_forget_logits = core.cache_base_logits(model, tok, forget_cases, device, batch_size=a.cache_batch_size)
    forget_hidden = core.forward_last_hidden(model, tok, forget_cases, device, a.cache_batch_size).float().detach()
    retain_hidden, raw_probs, retain_stats = cache_retain_hidden_and_probs(
        model, tok, retain_cases, selected_ids, device=device, batch_size=a.cache_batch_size,
    )
    retain_weights = normalized_clipped_weights(raw_probs, a.retain_weight_clip)
    retain_hidden = retain_hidden.to(device=device, dtype=torch.float32)
    retain_weights = retain_weights.to(device=device, dtype=torch.float32)

    row_bases, row_reports = build_contrastive_bases_from_hidden(
        forget_hidden, forget_tids, retain_hidden, retain_weights,
        all_selected_ids=selected_ids, requested_ids=selected_ids,
        rank_cap=a.contrastive_rank, relative_eps=a.contrastive_eps,
    )
    out_dir = gagd.resolve_output_path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(base_forget_logits, out_dir / "base_sensitive_case_logits.pt")
    torch.save(retain_hidden.detach().cpu(), out_dir / "base_retain_hidden.pt")
    torch.save(retain_weights.detach().cpu(), out_dir / "base_retain_weights.pt")
    core.write_json(out_dir / "contrastive_row_basis_reports.json", row_reports)

    trained_delta = optimize_stage1(
        a=a, model=model, tok=tok, output_layer=output_layer, selected_ids=selected_ids,
        row_bases=row_bases, forget_cases=forget_cases, base_forget_logits=base_forget_logits,
        retain_hidden=retain_hidden, retain_weights=retain_weights,
        llama_like=llama_like, device=device, out_dir=out_dir,
    )
    stage1_reports = [
        evaluate_stage1_scale(
            a=a, model=model, tok=tok, output_layer=output_layer, selected_ids=selected_ids,
            trained_delta=trained_delta, scale=s, forget_cases=forget_cases,
            base_forget_logits=base_forget_logits, retain_hidden=retain_hidden,
            retain_weights=retain_weights, llama_like=llama_like, device=device,
        ) for s in scales
    ]
    chosen1, stage1_selection_mode = choose_stage1_report(stage1_reports)
    stage1_delta = trained_delta * float(chosen1["scale"])
    core.materialize_output_delta(output_layer, selected_ids, stage1_delta)
    torch.save({"row_ids": selected_ids, "delta": stage1_delta.detach().cpu()}, out_dir / "stage1_total_delta.pt")

    stage1_ckpt = out_dir / "stage1_checkpoint"
    stage1_ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(stage1_ckpt)
    tok.save_pretrained(stage1_ckpt)

    after1 = shared.evaluate_shared_constraints(
        model, tok, forget_cases, base_forget_logits,
        llama_like=llama_like, device=device, batch_size=a.cache_batch_size,
    )
    mask = shared.failure_mask(
        after1["logit_margin"], after1["sensitive_nll_increase"],
        required_logit_margin=a.constraint_margin,
        required_nll_increase=a.min_sensitive_nll_increase,
    )
    active_indices = [i for i, flag in enumerate(mask.detach().cpu().tolist()) if bool(flag)]

    stage2_mode = "identity_noop"
    stage2_training_reports, stage2_candidate_reports = [], []
    final_delta = stage1_delta.detach().clone()
    chosen2 = None

    if active_indices:
        stage2_mode = "residual_contrastive_repair"
        active_cases = [forget_cases[i] for i in active_indices]
        active_hidden = forget_hidden[active_indices]
        active_tids = forget_tids[active_indices]
        active_base_logits = base_forget_logits[active_indices]
        active_ids = sorted(set(int(x) for x in active_tids.detach().cpu().tolist()))
        for rank in ranks:
            bases2, reports2 = build_contrastive_bases_from_hidden(
                active_hidden, active_tids, retain_hidden, retain_weights,
                all_selected_ids=selected_ids, requested_ids=active_ids,
                rank_cap=int(rank), relative_eps=a.contrastive_eps,
            )
            residual, train_report = optimize_stage2_candidate(
                a=a, model=model, tok=tok, output_layer=output_layer,
                stage1_ids=selected_ids, stage1_delta=stage1_delta,
                active_ids=active_ids, row_bases=bases2, active_cases=active_cases,
                active_base_logits=active_base_logits, all_forget_cases=forget_cases,
                base_forget_logits=base_forget_logits, retain_hidden=retain_hidden,
                retain_weights=retain_weights, llama_like=llama_like, device=device, rank=int(rank),
            )
            train_report = dict(train_report)
            train_report["rank"] = int(rank)
            train_report["row_basis_reports"] = reports2
            stage2_training_reports.append(train_report)
            for s in scales:
                stage2_candidate_reports.append(
                    evaluate_stage2_scale(
                        a=a, model=model, tok=tok, output_layer=output_layer,
                        stage1_ids=selected_ids, stage1_delta=stage1_delta,
                        active_ids=active_ids, residual_delta=residual, scale=s,
                        forget_cases=forget_cases, base_forget_logits=base_forget_logits,
                        retain_hidden=retain_hidden, retain_weights=retain_weights,
                        llama_like=llama_like, device=device, rank=int(rank),
                    ) | {"_residual": residual}
                )
        feasible2 = [r for r in stage2_candidate_reports if r["feasible"]]
        if not feasible2:
            serializable = [{k: v for k, v in r.items() if k != "_residual"} for r in stage2_candidate_reports]
            core.write_json(out_dir / "stage2_infeasible_candidates.json", serializable)
            raise RuntimeError("Stage 2 infeasible under restricted contrastive ranks and final guards")
        chosen2 = min(feasible2, key=lambda r: (
            r["residual_repair_norm"], r["retain_action_total"],
            r["total_delta_norm"], r["rank"], r["scale"],
        ))
        chosen_residual = chosen2["_residual"] * float(chosen2["scale"])
        core.materialize_output_delta(output_layer, active_ids, chosen_residual)
        final_delta = total_delta_with_residual(stage1_delta, selected_ids, chosen_residual, active_ids)
        for r in stage2_candidate_reports:
            r.pop("_residual", None)

    final_ckpt = out_dir / "checkpoint"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_ckpt)
    tok.save_pretrained(final_ckpt)
    torch.save({"row_ids": selected_ids, "delta": final_delta.detach().cpu()}, out_dir / "final_total_delta.pt")

    final_state = shared.evaluate_shared_constraints(
        model, tok, forget_cases, base_forget_logits,
        llama_like=llama_like, device=device, batch_size=a.cache_batch_size,
    )
    final_failures = count_failures(final_state, a)
    final_action = float(retain_action(retain_hidden, retain_weights, final_delta).detach().cpu())
    final_norm = float(final_delta.norm().cpu())
    final_proxy = evaluate_forget_nonsensitive_kl(
        model, tok, forget_cases, base_forget_logits,
        llama_like=llama_like, device=device, batch_size=a.cache_batch_size,
    )
    final_guard_pass = final_failures == 0 and final_action <= a.retain_action_budget and final_norm <= a.max_total_delta_norm and final_proxy <= a.max_forget_nonsensitive_kl
    if not final_guard_pass:
        raise RuntimeError("Final checkpoint violates one or more locked direct/retain/norm/proxy guards")

    config = {
        "schema_version": 2,
        "method": "SURE-LM-contrastive-row-specific-two-stage-v2",
        "protocol": "sure_contrastive_two_stage_v2",
        "dataset": a.dataset,
        "source_protocol": manifest.get("protocol"),
        "seed": int(a.seed),
        "architecture_shared_across_mcf_zsre": True,
        "transformer_trainable": 0,
        "input_embeddings_modified": False,
        "lm_head_untied": True,
        "editable_rows": "sensitive_answer_rows_only",
        "sensitive_answer_field": core.sensitive_answer_field(a.dataset),
        "target_new_used_in_loss_or_selection": False,
        "heldout_probes_seen": 0,
        "retain_eval_seen": 0,
        "retain_answer_labels_seen": False,
        "contrastive_generalized_eigenproblem": True,
        "contrastive_domain": "observed_direct_forget_context_span",
        "contrastive_rank": int(a.contrastive_rank),
        "contrastive_eps": float(a.contrastive_eps),
        "retain_weight_rule": "Base p(s|retain), divide by row mean, clip, renormalize row mean to 1",
        "retain_weight_clip": float(a.retain_weight_clip),
        "retain_stats": retain_stats,
        "stage1_row_basis_reports": row_reports,
        "stage1_objective": "GA_sensitive + non_sensitive_GD + retain_action(Stage1 total) + Stage1_delta_L2",
        "stage2_objective": "GA_active + non_sensitive_GD_active_to_Base + retain_action(total Base-relative delta) + residual_repair_L2",
        "ga_weight": float(a.ga_weight),
        "gd_weight": float(a.gd_weight),
        "retain_action_weight": float(a.retain_action_weight),
        "delta_l2": float(a.delta_l2),
        "stage2_l2": float(a.stage2_l2),
        "constraint_margin": float(a.constraint_margin),
        "min_sensitive_nll_increase": float(a.min_sensitive_nll_increase),
        "retain_action_budget": float(a.retain_action_budget),
        "max_total_delta_norm": float(a.max_total_delta_norm),
        "max_forget_nonsensitive_kl": float(a.max_forget_nonsensitive_kl),
        "candidate_scales": scales,
        "stage1_scale_reports": stage1_reports,
        "stage1_selected_scale": float(chosen1["scale"]),
        "stage1_selection_mode": stage1_selection_mode,
        "stage1_direct_failures_after_materialization": int(len(active_indices)),
        "stage2_mode": stage2_mode,
        "stage2_candidate_ranks": [int(x) for x in ranks],
        "stage2_training_reports": stage2_training_reports,
        "stage2_candidate_reports": stage2_candidate_reports,
        "stage2_selected": None if chosen2 is None else {k: v for k, v in chosen2.items() if k != "_residual"},
        "final_direct_failures": int(final_failures),
        "final_minimum_logit_margin": float(final_state["logit_margin"].min().cpu()),
        "final_minimum_sensitive_nll_increase": float(final_state["sensitive_nll_increase"].min().cpu()),
        "final_retain_action": final_action,
        "final_total_delta_norm": final_norm,
        "final_forget_nonsensitive_kl_proxy": float(final_proxy),
        "final_guard_pass": bool(final_guard_pass),
        "stage1_checkpoint": str(stage1_ckpt.resolve()),
        "final_checkpoint": str(final_ckpt.resolve()),
    }
    core.write_json(out_dir / "config_used.json", config)

    print("Contrastive SURE-LM v2 complete:", final_ckpt)
    print("dataset:", a.dataset)
    print("sensitive field:", config["sensitive_answer_field"])
    print("Stage-1 selected scale:", config["stage1_selected_scale"])
    print("Stage-1 selection mode:", stage1_selection_mode)
    print("Stage-1 residual direct failures:", len(active_indices))
    print("Stage-2 mode:", stage2_mode)
    print("Final direct failures:", final_failures)
    print("Final retain action:", final_action)
    print("Final total delta norm:", final_norm)
    print("Final non-sensitive KL proxy:", final_proxy)


if __name__ == "__main__":
    main()
