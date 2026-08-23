#!/usr/bin/env python3
"""SURE-MCF Stage 2 v7: strong full-row repair + Wiki covariance + relation locality.

This deliberately starts from the strongest MCF SURE Stage-2 architecture:
unrestricted sparse target_true LM-head rows and exact direct MCF sequence
margins.  It does NOT project away repair directions.

Utility preservation is added in two orthogonal ways:

1. Wikipedia covariance preconditioning.
   A separately cached 1000-document / 100000-state final-hidden second moment C
   is normalized to unit average eigenvalue.  Every full-row repair gradient g
   is right-preconditioned by

       (C + ridge I)^(-1)

   via Cholesky solve.  High-occupancy Wikipedia directions therefore receive
   smaller updates without introducing a covariance-loss weight sweep.

2. Leak-free relation-locality controls.
   For every training-visible MCF record, its own relation prompt template is
   instantiated with other TRAINING-VISIBLE subjects.  Stage-1 logits on these
   synthetic controls are cached.  Training uses soft full-vocabulary KL and a
   hard guard requiring zero top-1 changes and bounded mean KL.

Official paraphrases, official neighborhood prompts, benchmark retain records,
and official PPL text are never read by this script.  The covariance cache
builder reserves the first 20 Wikipedia docs used by the official PPL evaluator.

The final checkpoint is selected by an exact bf16 line search: choose the
smallest materialized scale that passes every direct MCF record, preserves all
Stage-1 passing direct records, and satisfies the relation-locality guards.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

import gagd_compare as gagd
import gagd_active_case_repair as mcf_repair
import sure_canonical_core as core
import sure_stage2_sparse_repair as shared
import mcf_sure_fullrow_failure_repair as fullrow
import mcf_sure_protected_subspace_stage2_mcf_margin as v3
import mcf_sure_rowspecific_minimal_stage2 as v4


METHOD = "SURE-MCF-CovRel-fullrow-sequence-margin-stage2"
PROTOCOL = "mcf_target_true_covrel_fullrow_stage2_v7"
BACKTRACK_SCALES = tuple([1.0] + [2.0 ** (-k) for k in range(1, 25)])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Directional SURE Stage-1 checkpoint")
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--wiki-covariance", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--repair-steps", type=int, default=800)
    p.add_argument("--repair-lr", type=float, default=5e-3)
    p.add_argument("--train-mcf-margin", type=float, default=0.10)
    p.add_argument("--final-mcf-margin", type=float, default=0.05)
    p.add_argument("--protected-mcf-margin-floor", type=float, default=0.0)
    p.add_argument("--pass-guard-weight", type=float, default=1.0)
    p.add_argument("--relation-controls-per-record", type=int, default=4)
    p.add_argument("--relation-kl-weight", type=float, default=1.0)
    p.add_argument("--relation-kl-max", type=float, default=0.01)
    p.add_argument("--cov-ridge", type=float, default=0.10,
                   help="Ridge after covariance normalization to mean eigenvalue 1.")
    p.add_argument("--repair-l2", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--scale-bisect-steps", type=int, default=12)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    a = p.parse_args(list(argv) if argv is not None else None)
    if min(a.forget_num, a.repair_steps, a.relation_controls_per_record,
           a.batch_size, a.check_every, a.scale_bisect_steps) <= 0:
        p.error("counts/steps/batch/check/bisect settings must be positive")
    if a.repair_lr <= 0:
        p.error("repair-lr must be positive")
    if min(a.train_mcf_margin, a.final_mcf_margin,
           a.protected_mcf_margin_floor, a.pass_guard_weight,
           a.relation_kl_weight, a.relation_kl_max,
           a.cov_ridge, a.repair_l2, a.grad_clip) < 0:
        p.error("margins/weights/ridge/L2/clip must be non-negative")
    if a.final_mcf_margin > a.train_mcf_margin:
        p.error("final-mcf-margin must not exceed train-mcf-margin")
    return a


def rewrite(record: Mapping[str, Any]) -> Mapping[str, Any]:
    rr = record.get("requested_rewrite")
    if isinstance(rr, list):
        rr = rr[0] if rr else None
    if not isinstance(rr, Mapping):
        raise ValueError("MCF record lacks requested_rewrite mapping")
    return rr


def build_relation_controls(
    records: Sequence[Mapping[str, Any]],
    *,
    controls_per_record: int,
    seed: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Same-relation templates with alternate training-visible subjects only."""
    subjects = [str(rewrite(r).get("subject", "")) for r in records]
    if len(set(subjects)) < 2:
        raise RuntimeError("Need at least two distinct training-visible subjects")
    rng = random.Random(int(seed))
    prompts: List[str] = []
    metadata: List[Dict[str, Any]] = []
    seen = set()
    for position, record in enumerate(records):
        rr = rewrite(record)
        own_subject = str(rr.get("subject", ""))
        template = str(rr.get("prompt", ""))
        relation_key = rr.get("relation_id", rr.get("relation", template))
        donor_positions = [i for i, s in enumerate(subjects) if s and s != own_subject]
        rng.shuffle(donor_positions)
        if not donor_positions:
            continue
        used = 0
        cursor = 0
        while used < int(controls_per_record) and cursor < len(donor_positions):
            donor_position = int(donor_positions[cursor])
            cursor += 1
            donor = subjects[donor_position]
            if "{}" in template:
                prompt = template.format(donor)
            elif own_subject and own_subject in template:
                prompt = template.replace(own_subject, donor, 1)
            else:
                # CounterFact templates normally contain {}, but do not create
                # an unverifiable relation control if the subject cannot be substituted.
                continue
            key = (position, prompt)
            if key in seen:
                continue
            seen.add(key)
            prompts.append(prompt)
            metadata.append({
                "source_record_position": int(position),
                "donor_record_position": donor_position,
                "relation_key": str(relation_key),
                "source_subject": own_subject,
                "donor_subject": donor,
                "prompt": prompt,
            })
            used += 1
    if not prompts:
        raise RuntimeError("No relation-locality controls could be constructed")
    return prompts, metadata


@torch.no_grad()
def cache_prompt_states(model, tok, prompts: Sequence[str], device: torch.device, batch_size: int):
    hidden_chunks: List[torch.Tensor] = []
    logit_chunks: List[torch.Tensor] = []
    for start in range(0, len(prompts), int(batch_size)):
        batch = list(prompts[start:start + int(batch_size)])
        encoded = tok(batch, padding=True, return_tensors="pt").to(device)
        out = model(**encoded, output_hidden_states=True, use_cache=False)
        positions = encoded["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(batch), device=device)
        hidden_chunks.append(out.hidden_states[-1][rows, positions, :].float().detach())
        logit_chunks.append(out.logits[rows, positions, :].float().detach())
    return torch.cat(hidden_chunks, dim=0), torch.cat(logit_chunks, dim=0)


def relation_metrics(
    *,
    base_logits: torch.Tensor,
    hidden: torch.Tensor,
    selected_ids: Sequence[int],
    delta_rows: torch.Tensor,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    current = v3.logits_with_sparse_delta(
        base_logits, hidden, selected_ids, delta_rows
    )
    ref_logp = F.log_softmax(base_logits.float(), dim=-1)
    cur_logp = F.log_softmax(current.float(), dim=-1)
    kl = (ref_logp.exp() * (ref_logp - cur_logp)).sum(dim=-1).mean()
    base_top1 = base_logits.argmax(dim=-1)
    current_top1 = current.argmax(dim=-1)
    changes = int((base_top1 != current_top1).sum().detach().cpu())
    max_abs_selected_shift = (
        hidden.float() @ delta_rows.float().transpose(0, 1)
    ).abs().max()
    return kl, changes, max_abs_selected_shift


def load_covariance(path: Path, hidden_size: int, device: torch.device, ridge: float):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "covariance" not in payload:
        raise ValueError("Invalid Wikipedia covariance cache")
    cov = payload["covariance"].float()
    if cov.shape != (hidden_size, hidden_size):
        raise ValueError(
            f"Covariance shape {tuple(cov.shape)} does not match hidden size {hidden_size}"
        )
    if int(payload.get("skip_first_docs", 0)) < 20:
        raise RuntimeError(
            "Covariance cache includes one or more of the first 20 Wikipedia docs "
            "used by official PPL evaluation; rebuild with --skip-first-docs 20."
        )
    avg = float(torch.trace(cov).item() / hidden_size)
    if not torch.isfinite(cov).all() or avg <= 0:
        raise ValueError("Non-finite/degenerate covariance")
    cov = (cov / avg).to(device=device, dtype=torch.float32)
    eye = torch.eye(hidden_size, dtype=torch.float32, device=device)
    regularized = cov + float(ridge) * eye
    chol = torch.linalg.cholesky(regularized)
    return cov, chol, dict(payload), avg


def covariance_energy(delta: torch.Tensor, normalized_cov: torch.Tensor) -> torch.Tensor:
    if delta.numel() == 0:
        return delta.sum() * 0.0
    projected = delta.float() @ normalized_cov
    return (projected * delta.float()).sum(dim=1).mean()


def precondition_fullrow_gradient(delta_module, chol: torch.Tensor) -> None:
    raw = getattr(delta_module, "raw_delta", None)
    if raw is None or raw.grad is None:
        raise RuntimeError("CovRel requires unrestricted full-row raw_delta")
    g = raw.grad.detach().float()
    # Solve X C_reg = G by transposing into C_reg X^T = G^T.
    preconditioned_t = torch.cholesky_solve(g.transpose(0, 1), chol)
    raw.grad.copy_(preconditioned_t.transpose(0, 1).to(raw.grad.dtype))


def hard_metrics(
    *,
    all_caches,
    protected_positions: Sequence[int],
    delta: torch.Tensor,
    relation_base_logits: torch.Tensor,
    relation_hidden: torch.Tensor,
    selected_ids: Sequence[int],
    protected_floor: float,
):
    if protected_positions:
        pcaches = [all_caches[int(i)] for i in protected_positions]
        pmargins = fullrow.margins_from_caches(pcaches, delta)
        regressions = int((pmargins < float(protected_floor)).sum().item())
        pmin = float(pmargins.min().detach().cpu())
    else:
        regressions = 0
        pmin = None
    rkl, top1_changes, max_shift = relation_metrics(
        base_logits=relation_base_logits,
        hidden=relation_hidden,
        selected_ids=selected_ids,
        delta_rows=delta,
    )
    return {
        "protected_mcf_regressions": regressions,
        "protected_min_mcf_margin": pmin,
        "relation_kl": max(0.0, float(rkl.detach().cpu())),
        "relation_top1_changes": int(top1_changes),
        "relation_max_abs_selected_logit_shift": float(max_shift.detach().cpu()),
    }


@torch.no_grad()
def exact_materialized_evaluation(
    *,
    model,
    tok,
    output_layer,
    selected_ids: Sequence[int],
    base_rows: torch.Tensor,
    delta_rows: torch.Tensor,
    scale: float,
    instances,
    device: torch.device,
    llama_like: bool,
    batch_size: int,
    relation_base_logits: torch.Tensor,
    relation_hidden: torch.Tensor,
    protected_positions: Sequence[int],
    final_margin: float,
    protected_floor: float,
    relation_kl_max: float,
):
    margins, qdelta = v4.exact_materialized_margins(
        model=model,
        tok=tok,
        output_layer=output_layer,
        selected_ids=selected_ids,
        base_rows=base_rows,
        delta_rows=delta_rows,
        scale=float(scale),
        instances=instances,
        device=device,
        llama_like=llama_like,
        batch_size=int(batch_size),
    )
    rkl, rchanges, max_shift = relation_metrics(
        base_logits=relation_base_logits,
        hidden=relation_hidden,
        selected_ids=selected_ids,
        delta_rows=qdelta.to(device=relation_hidden.device),
    )
    if protected_positions:
        pidx = torch.tensor([int(x) for x in protected_positions], dtype=torch.long)
        pvals = margins.index_select(0, pidx)
        preg = int((pvals < float(protected_floor)).sum().item())
        pmin = float(pvals.min().item())
    else:
        preg = 0
        pmin = None
    all_pass = bool((margins >= float(final_margin)).all().item())
    feasible = (
        all_pass
        and preg == 0
        and int(rchanges) == 0
        and max(0.0, float(rkl.detach().cpu())) <= float(relation_kl_max)
    )
    metrics = {
        "scale": float(scale),
        "feasible": bool(feasible),
        "all_direct_records_pass": bool(all_pass),
        "minimum_direct_margin": float(margins.min().item()),
        "protected_mcf_regressions": int(preg),
        "protected_min_mcf_margin": pmin,
        "relation_kl": max(0.0, float(rkl.detach().cpu())),
        "relation_top1_changes": int(rchanges),
        "relation_max_abs_selected_logit_shift": float(max_shift.detach().cpu()),
        "effective_delta_norm": float(qdelta.norm().detach().cpu()),
    }
    return feasible, margins, qdelta, metrics


@torch.no_grad()
def minimum_exact_scale(**kwargs):
    steps = int(kwargs.pop("bisect_steps"))
    evaluations: List[Dict[str, Any]] = []

    def evaluate(alpha: float):
        ok, margins, qdelta, metrics = exact_materialized_evaluation(
            scale=float(alpha), **kwargs
        )
        evaluations.append(metrics)
        return ok, margins, qdelta, metrics

    ok1, _, q1, m1 = evaluate(1.0)
    if not ok1:
        raise RuntimeError("Full learned delta is not exact-bf16 CovRel feasible")
    ok0, _, _, _ = evaluate(0.0)
    if ok0:
        return 0.0, torch.zeros_like(q1), m1, evaluations

    lo, hi = 0.0, 1.0
    best_delta = q1
    best_metrics = m1
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        ok, _, qdelta, metrics = evaluate(mid)
        if ok:
            hi = mid
            best_delta = qdelta
            best_metrics = metrics
        else:
            lo = mid
    ok, _, best_delta, best_metrics = evaluate(hi)
    if not ok:
        raise RuntimeError("Internal exact-scale search inconsistency")
    return float(hi), best_delta, best_metrics, evaluations


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(int(a.seed))
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, manifest = fullrow.validate_locked(
        visible_path, manifest_path, int(a.seed), int(a.forget_num)
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
    llama_like = core.is_llama_like(model, tok)
    hidden_size = int(output_layer.weight.shape[1])

    normalized_cov, cov_chol, cov_meta, cov_avg = load_covariance(
        Path(a.wiki_covariance).resolve(), hidden_size, output_layer.weight.device,
        float(a.cov_ridge)
    )

    instances = shared.mcf_instances(records)
    baseline_margins = shared.mcf_direct_margins(
        model, tok, instances, device, llama_like, int(a.batch_size),
        sensitive_field="target_true", reference_field="target_new",
    ).detach().float()
    active_positions = [
        i for i, x in enumerate(baseline_margins.cpu().tolist())
        if float(x) < float(a.train_mcf_margin)
    ]
    protected_positions = [
        i for i in range(len(instances)) if i not in set(active_positions)
    ]
    selected_ids = fullrow.failure_sensitive_rows(tok, instances, active_positions)
    if not selected_ids:
        raise RuntimeError("No residual target_true LM-head rows selected")

    relation_prompts, relation_metadata = build_relation_controls(
        records,
        controls_per_record=int(a.relation_controls_per_record),
        seed=int(a.seed),
    )
    relation_hidden, relation_base_logits = cache_prompt_states(
        model, tok, relation_prompts, device, int(a.batch_size)
    )

    all_caches = mcf_repair.build_prompt_instance_delta_caches(
        model, tok, instances, selected_ids, device, int(a.batch_size), llama_like
    )
    active_caches = [all_caches[int(i)] for i in active_positions]
    protected_caches = [all_caches[int(i)] for i in protected_positions]

    delta_module = core.SelectedRowDelta(
        len(selected_ids), hidden_size, direction_basis=None,
        device=output_layer.weight.device,
    )
    opt = torch.optim.AdamW(
        delta_module.parameters(), lr=float(a.repair_lr), weight_decay=0.0
    )
    ids_tensor = torch.tensor(
        selected_ids, dtype=torch.long, device=output_layer.weight.device
    )
    base_rows = output_layer.weight.index_select(0, ids_tensor).detach().clone()
    frozen_hash_before = v3.hash_frozen_parameters(model, output_layer)

    logs: List[Dict[str, Any]] = []
    accepted_hist: Dict[str, int] = {}
    rollback_count = 0
    feasible_delta = None

    for step in range(1, int(a.repair_steps) + 1):
        opt.zero_grad(set_to_none=True)
        delta = delta_module.effective_delta()
        active_margins = fullrow.margins_from_caches(active_caches, delta)
        hinge = F.relu(float(a.train_mcf_margin) - active_margins).square().mean()
        if protected_caches and float(a.pass_guard_weight) > 0:
            pass_margins = fullrow.margins_from_caches(protected_caches, delta)
            pass_guard = F.relu(
                float(a.protected_mcf_margin_floor) - pass_margins
            ).square().mean()
        else:
            pass_guard = delta.sum() * 0.0
        relation_kl, relation_changes, _ = relation_metrics(
            base_logits=relation_base_logits,
            hidden=relation_hidden,
            selected_ids=selected_ids,
            delta_rows=delta,
        )
        l2 = delta.square().mean()
        loss = (
            hinge
            + float(a.pass_guard_weight) * pass_guard
            + float(a.relation_kl_weight) * relation_kl
            + float(a.repair_l2) * l2
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite CovRel loss at step {step}")

        old_params = v3.parameter_snapshot(delta_module)
        opt_state_before = copy.deepcopy(opt.state_dict())
        loss.backward()
        precondition_fullrow_gradient(delta_module, cov_chol)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(delta_module.parameters()), float(a.grad_clip)
        ) if float(a.grad_clip) > 0 else None
        if grad_norm is not None and not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite CovRel gradient at step {step}")
        opt.step()
        proposed_params = v3.parameter_snapshot(delta_module)

        accepted_alpha = None
        accepted_metrics = None
        for alpha in BACKTRACK_SCALES:
            v3.set_interpolated_parameters(
                delta_module, old_params, proposed_params, float(alpha)
            )
            proposal = delta_module.effective_delta()
            hm = hard_metrics(
                all_caches=all_caches,
                protected_positions=protected_positions,
                delta=proposal,
                relation_base_logits=relation_base_logits,
                relation_hidden=relation_hidden,
                selected_ids=selected_ids,
                protected_floor=float(a.protected_mcf_margin_floor),
            )
            if (
                int(hm["protected_mcf_regressions"]) == 0
                and int(hm["relation_top1_changes"]) == 0
                and float(hm["relation_kl"]) <= float(a.relation_kl_max)
            ):
                accepted_alpha = float(alpha)
                accepted_metrics = hm
                break
        if accepted_alpha is None:
            v3.set_interpolated_parameters(delta_module, old_params, old_params, 1.0)
            opt.load_state_dict(opt_state_before)
            rollback_count += 1
            accepted_alpha = 0.0
            accepted_metrics = hard_metrics(
                all_caches=all_caches,
                protected_positions=protected_positions,
                delta=delta_module.effective_delta(),
                relation_base_logits=relation_base_logits,
                relation_hidden=relation_hidden,
                selected_ids=selected_ids,
                protected_floor=float(a.protected_mcf_margin_floor),
            )
        else:
            key = f"{accepted_alpha:g}"
            accepted_hist[key] = int(accepted_hist.get(key, 0)) + 1

        with torch.no_grad():
            current = delta_module.effective_delta().detach()
            all_margins = fullrow.margins_from_caches(all_caches, current)
            failures = int((all_margins < float(a.train_mcf_margin)).sum().item())
            wiki_energy = covariance_energy(current, normalized_cov)
            exact_pass = False
            exact_min = None
            if failures == 0:
                exact_pass, exact_margins, _, exact_metrics = exact_materialized_evaluation(
                    model=model,
                    tok=tok,
                    output_layer=output_layer,
                    selected_ids=selected_ids,
                    base_rows=base_rows,
                    delta_rows=current,
                    scale=1.0,
                    instances=instances,
                    device=device,
                    llama_like=llama_like,
                    batch_size=int(a.batch_size),
                    relation_base_logits=relation_base_logits,
                    relation_hidden=relation_hidden,
                    protected_positions=protected_positions,
                    final_margin=float(a.final_mcf_margin),
                    protected_floor=float(a.protected_mcf_margin_floor),
                    relation_kl_max=float(a.relation_kl_max),
                )
                exact_min = float(exact_margins.min().item())
                if exact_pass:
                    feasible_delta = current.clone()
            row = {
                "step": int(step),
                "cached_mcf_failures": failures,
                "cached_min_margin": float(all_margins.min().detach().cpu()),
                "accepted_backtrack_alpha": float(accepted_alpha),
                "protected_mcf_regressions": int(accepted_metrics["protected_mcf_regressions"]),
                "relation_kl": float(accepted_metrics["relation_kl"]),
                "relation_top1_changes": int(accepted_metrics["relation_top1_changes"]),
                "wiki_covariance_energy": float(wiki_energy.detach().cpu()),
                "delta_norm": float(current.norm().cpu()),
                "exact_bf16_feasible": bool(exact_pass),
                "exact_bf16_min_margin": exact_min,
            }
            if grad_norm is not None:
                row["preconditioned_grad_norm"] = float(grad_norm.detach().cpu())
            if (step == 1 or step % int(a.check_every) == 0 or
                    failures == 0 or step == int(a.repair_steps)):
                logs.append(row)
                print(json.dumps(row))
            if feasible_delta is not None:
                break

    del opt
    if feasible_delta is None:
        raise RuntimeError(
            "No exact-bf16 feasible CovRel repair found; refusing to save checkpoint"
        )

    selected_scale, quantized_delta, final_guard, scale_evals = minimum_exact_scale(
        model=model,
        tok=tok,
        output_layer=output_layer,
        selected_ids=selected_ids,
        base_rows=base_rows,
        delta_rows=feasible_delta,
        instances=instances,
        device=device,
        llama_like=llama_like,
        batch_size=int(a.batch_size),
        relation_base_logits=relation_base_logits,
        relation_hidden=relation_hidden,
        protected_positions=protected_positions,
        final_margin=float(a.final_mcf_margin),
        protected_floor=float(a.protected_mcf_margin_floor),
        relation_kl_max=float(a.relation_kl_max),
        bisect_steps=int(a.scale_bisect_steps),
    )

    final_rows = base_rows.float() + quantized_delta.to(base_rows.device).float()
    output_layer.weight.index_copy_(
        0, ids_tensor, final_rows.to(output_layer.weight.dtype)
    )
    frozen_hash_after = v3.hash_frozen_parameters(model, output_layer)
    frozen_bit_exact = frozen_hash_before == frozen_hash_after
    if not frozen_bit_exact:
        raise RuntimeError("CovRel Stage 2 changed embeddings and/or transformer parameters")

    final_margins = shared.mcf_direct_margins(
        model, tok, instances, device, llama_like, int(a.batch_size),
        sensitive_field="target_true", reference_field="target_new",
    ).detach().float().cpu()
    final_failures = [
        i for i, x in enumerate(final_margins.tolist())
        if float(x) < float(a.final_mcf_margin)
    ]
    final_wiki_energy = covariance_energy(
        quantized_delta.to(normalized_cov.device), normalized_cov
    )

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    core.write_json(out_dir / "relation_controls.json", relation_metadata)

    final_gate = {
        "all_mcf_direct_records_pass": len(final_failures) == 0,
        "minimum_final_mcf_margin": float(final_margins.min().item()),
        "protected_mcf_regressions_zero": int(final_guard["protected_mcf_regressions"]) == 0,
        "relation_top1_changes_zero": int(final_guard["relation_top1_changes"]) == 0,
        "relation_kl_within_limit": float(final_guard["relation_kl"]) <= float(a.relation_kl_max),
        "embeddings_and_transformer_bit_exact": bool(frozen_bit_exact),
    }
    final_gate["passed"] = all(bool(x) for x in final_gate.values())

    summary: Dict[str, Any] = {
        "schema_version": 7,
        "method": METHOD,
        "protocol": PROTOCOL,
        "source_protocol": manifest.get("protocol"),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "stage1_failure_count": len(active_positions),
        "stage1_passing_count": len(protected_positions),
        "selected_lm_head_rows": len(selected_ids),
        "selected_token_ids": selected_ids,
        "parameterization": "unrestricted sparse full LM-head rows",
        "primary_gate": "NLL(target_true)-NLL(target_new)",
        "train_mcf_margin": float(a.train_mcf_margin),
        "final_mcf_margin": float(a.final_mcf_margin),
        "wikipedia_covariance": {
            "path": str(Path(a.wiki_covariance).resolve()),
            "hidden_state_count": int(cov_meta.get("hidden_state_count", -1)),
            "documents_used": int(cov_meta.get("documents_used", -1)),
            "prompts_per_document": int(cov_meta.get("prompts_per_document", -1)),
            "skip_first_docs": int(cov_meta.get("skip_first_docs", -1)),
            "official_ppl_first_20_reserved": bool(cov_meta.get("official_ppl_first_20_reserved", False)),
            "raw_average_second_moment_eigenvalue": float(cov_avg),
            "normalization": "divide by mean eigenvalue",
            "ridge_after_normalization": float(a.cov_ridge),
            "optimization_role": "right-precondition full-row gradient by inverse regularized covariance",
        },
        "relation_locality": {
            "controls_per_record_requested": int(a.relation_controls_per_record),
            "control_prompt_count": len(relation_prompts),
            "source": "training-visible relation templates x alternate training-visible subjects",
            "soft_kl_weight": float(a.relation_kl_weight),
            "hard_kl_max": float(a.relation_kl_max),
            "hard_top1_changes_allowed": 0,
            "official_neighborhood_prompts_used": 0,
        },
        "accepted_backtrack_histogram": accepted_hist,
        "rollback_count": int(rollback_count),
        "optimizer_logs": logs,
        "unscaled_feasible_delta_norm": float(feasible_delta.norm().cpu()),
        "minimum_exact_bf16_scale": float(selected_scale),
        "final_effective_delta_norm": float(quantized_delta.norm().cpu()),
        "final_wikipedia_covariance_energy": float(final_wiki_energy.detach().cpu()),
        "scale_line_search": scale_evals,
        "final_mcf_record_failure_count": len(final_failures),
        "final_mcf_record_failure_positions": final_failures,
        "final_mcf_record_min_margin": float(final_margins.min().item()),
        "final_relation_metrics": final_guard,
        "final_gate": final_gate,
        "official_paraphrases_seen": 0,
        "official_neighborhood_seen": 0,
        "benchmark_retain_seen": 0,
        "ppl_eval_text_seen": 0,
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "repair_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(
        f"CovRel full-row Stage 2: failures {len(active_positions)} -> {len(final_failures)}; "
        f"scale={selected_scale:.6f}; norm={float(quantized_delta.norm().cpu()):.4f}; "
        f"relation_top1_changes={final_guard['relation_top1_changes']}; "
        f"gate={final_gate['passed']}"
    )
    print(f"Final checkpoint: {ckpt}")
    if not final_gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
