#!/usr/bin/env python3
"""MCF directional SURE Stage-1: GA in forget span, GD/utility in its nullspace.

Target contract:
  * requested_rewrite.target_true = sensitive / unwanted answer
  * requested_rewrite.target_new  = non-sensitive benchmark reference (diagnostic only)

This ablation makes the gradient-separation hypothesis explicit.  The Base model
is loaded, its tied LM head is cloned/untied, and the transformer plus input
embeddings remain frozen.  Only sparse FP32 deltas for target_true-sensitive
output rows are optimized.

For each sensitive output token y, Base final hidden states from every
training-visible teacher-forced target_true prediction of y define an orthonormal
forget basis B_F,y.  Two independent sparse row deltas are maintained:

    Delta W = Delta W_GA + Delta W_GD

with hard parameter constraints after every optimizer step:

    Delta w_GA,y <- Proj_span(B_F,y)(Delta w_GA,y)
    Delta w_GD,y <- Proj_null(B_F,y)(Delta w_GD,y)

The GA objective receives gradients only through Delta W_GA.  Same-prompt
non-sensitive KL, relation-donor locality KL, protected sensitive-logit MSE, and
external Wikipedia KL receive gradients only through Delta W_GD.  Therefore GA
and preservation updates cannot cancel each other in the same hidden direction.
A small row-delta penalty is applied within each component's own constrained
subspace.

No official MCF neighborhood, paraphrase, retain, generation, or PPL evaluation
example is visible to training/checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import nn
from tqdm import tqdm

import gagd_compare as gagd
import mcf_frozen_head_representation_repair as contract_helpers
import mcf_sensitive_rows_projected_gagd as projected
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import sure_stage1_gagd_w1k as wikipedia_utility
import sure_stage2_sparse_repair as stage2


METHOD = "SURE-LM-MCF-sensitive-output-rows-directional-GAGD"
PROTOCOL = "mcf_target_true_directional_gagd_forget_span_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Base model checkpoint")
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)

    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--lm-row-lr", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--gd-weight", type=float, default=1.0)
    p.add_argument(
        "--forget-basis-rank",
        type=int,
        default=0,
        help="0 keeps each token's full effective Base forget-hidden rank; positive caps it",
    )

    p.add_argument("--subject-control-count", type=int, default=4)
    p.add_argument("--locality-batch-size", type=int, default=4)
    p.add_argument("--locality-cache-batch-size", type=int, default=8)
    p.add_argument("--locality-kl-weight", type=float, default=2.0)
    p.add_argument("--locality-sensitive-logit-weight", type=float, default=5.0)
    p.add_argument("--row-delta-weight", type=float, default=0.01)

    p.add_argument("--utility-wikipedia-dir", required=True)
    p.add_argument("--utility-sample-size", type=int, default=200)
    p.add_argument("--utility-batch-size", type=int, default=4)
    p.add_argument("--utility-cache-batch-size", type=int, default=8)
    p.add_argument("--utility-max-length", type=int, default=128)
    p.add_argument("--utility-seed", type=int, default=1)
    p.add_argument("--utility-exclude-first", type=int, default=20)
    p.add_argument("--utility-kl-weight", type=float, default=2.0)

    p.add_argument("--benchmark-pair-margin", type=float, default=0.05)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    a = p.parse_args(list(argv) if argv is not None else None)

    positive = (
        a.forget_num, a.steps, a.batch_size, a.cache_batch_size, a.lm_row_lr,
        a.check_every, a.ga_weight, a.subject_control_count, a.locality_batch_size,
        a.locality_cache_batch_size, a.locality_kl_weight,
        a.locality_sensitive_logit_weight, a.utility_sample_size,
        a.utility_batch_size, a.utility_cache_batch_size, a.utility_max_length,
        a.utility_kl_weight,
    )
    if any(float(v) <= 0 for v in positive):
        p.error("counts, LR, GA/locality/utility weights must be positive")
    nonnegative = (
        a.gd_weight, a.grad_clip, a.forget_basis_rank, a.row_delta_weight,
        a.utility_exclude_first, a.benchmark_pair_margin,
    )
    if any(float(v) < 0 for v in nonnegative):
        p.error("GD/clip/rank/delta/exclusion/margin values must be non-negative")
    if a.utility_exclude_first < 20:
        p.error("utility-exclude-first must be at least 20 to protect the fixed PPL prefix")
    return a


def _optimizer(params: Iterable[nn.Parameter], kind: str, lr: float):
    ps = list(params)
    if kind == "sgd":
        return torch.optim.SGD(ps, lr=lr)
    if kind == "adam":
        return torch.optim.Adam(ps, lr=lr)
    return torch.optim.AdamW(ps, lr=lr, weight_decay=0.0)


def build_token_forget_bases(
    base_hidden: torch.Tensor,
    target_ids: torch.Tensor,
    selected_ids: Sequence[int],
    rank_cap: int = 0,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, Any]]:
    """Build token-specific Base forget-hidden bases from training-visible cases."""
    h = base_hidden.detach().float().cpu()
    tids = target_ids.detach().long().cpu()
    if h.ndim != 2 or tids.ndim != 1 or h.shape[0] != tids.shape[0]:
        raise ValueError("forget hidden/target-id shape mismatch")
    bases: Dict[int, torch.Tensor] = {}
    receipt: Dict[str, Any] = {}
    for token_id in sorted(set(int(x) for x in selected_ids)):
        rows = torch.nonzero(tids == int(token_id), as_tuple=False).flatten()
        if not rows.numel():
            raise RuntimeError(f"selected token {token_id} has no forget hidden states")
        values = h.index_select(0, rows).contiguous()
        _u, singular, vh = torch.linalg.svd(values, full_matrices=False)
        tol = max(values.shape) * torch.finfo(values.dtype).eps * singular.max().clamp_min(1.0)
        effective = int((singular > tol).sum().item())
        keep = effective if int(rank_cap) <= 0 else min(effective, int(rank_cap))
        if keep <= 0:
            raise RuntimeError(f"selected token {token_id} has zero effective forget rank")
        basis = vh[:keep].float().contiguous()
        identity = basis @ basis.T
        if not torch.allclose(identity, torch.eye(keep), atol=1e-4, rtol=1e-4):
            raise RuntimeError(f"forget basis for token {token_id} is not orthonormal")
        bases[token_id] = basis
        receipt[str(token_id)] = {
            "case_count": int(rows.numel()),
            "effective_rank": int(effective),
            "kept_rank": int(keep),
            "singular_values": [float(v) for v in singular[:keep].tolist()],
        }
    return bases, receipt


def project_rows_tokenwise(
    rows: torch.Tensor,
    row_ids: Sequence[int],
    token_bases: Mapping[int, torch.Tensor],
    *,
    mode: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Project each row into its token forget span or its orthogonal complement."""
    if rows.ndim != 2 or rows.shape[0] != len(row_ids):
        raise ValueError("row tensor/row-id mismatch")
    if mode not in {"span", "null"}:
        raise ValueError("mode must be 'span' or 'null'")
    out = torch.empty_like(rows, dtype=torch.float32)
    aligned_sq = 0.0
    null_sq = 0.0
    for i, token_id in enumerate(row_ids):
        basis = token_bases.get(int(token_id))
        if basis is None or not basis.numel():
            raise RuntimeError(f"missing forget basis for selected token {token_id}")
        b = basis.to(device=rows.device, dtype=torch.float32)
        value = rows[i].float()
        aligned = (value @ b.T) @ b
        null = value - aligned
        out[i] = aligned if mode == "span" else null
        aligned_sq += float(aligned.square().sum().detach().cpu())
        null_sq += float(null.square().sum().detach().cpu())
    return out, {
        "aligned_norm": float(aligned_sq ** 0.5),
        "null_norm": float(null_sq ** 0.5),
    }


@torch.no_grad()
def enforce_directional_parameters_(
    ga_delta: torch.Tensor,
    gd_delta: torch.Tensor,
    row_ids: Sequence[int],
    token_bases: Mapping[int, torch.Tensor],
) -> Dict[str, float]:
    ga_projected, ga_diag = project_rows_tokenwise(
        ga_delta.detach(), row_ids, token_bases, mode="span"
    )
    gd_projected, gd_diag = project_rows_tokenwise(
        gd_delta.detach(), row_ids, token_bases, mode="null"
    )
    ga_removed = ga_delta.detach().float() - ga_projected
    gd_removed = gd_delta.detach().float() - gd_projected
    ga_delta.copy_(ga_projected.to(dtype=ga_delta.dtype))
    gd_delta.copy_(gd_projected.to(dtype=gd_delta.dtype))
    return {
        "ga_removed_null_component_norm": float(ga_removed.norm().cpu()),
        "gd_removed_forget_component_norm": float(gd_removed.norm().cpu()),
        "ga_span_norm": float(ga_diag["aligned_norm"]),
        "gd_null_norm": float(gd_diag["null_norm"]),
    }


@torch.no_grad()
def directional_residual_diagnostics(
    ga_delta: torch.Tensor,
    gd_delta: torch.Tensor,
    row_ids: Sequence[int],
    token_bases: Mapping[int, torch.Tensor],
) -> Dict[str, float]:
    ga_null_vals: List[float] = []
    gd_span_vals: List[float] = []
    cross_vals: List[float] = []
    for i, token_id in enumerate(row_ids):
        b = token_bases[int(token_id)].to(device=ga_delta.device, dtype=torch.float32)
        ga = ga_delta[i].float()
        gd = gd_delta[i].float()
        ga_null = ga - (ga @ b.T) @ b
        gd_span = (gd @ b.T) @ b
        ga_null_vals.append(float(ga_null.abs().max().cpu()))
        gd_span_vals.append(float(gd_span.abs().max().cpu()))
        denom = ga.norm() * gd.norm()
        cross_vals.append(float((ga @ gd).abs().div(denom.clamp_min(1e-12)).cpu()))
    return {
        "max_abs_ga_null_residual": max(ga_null_vals) if ga_null_vals else 0.0,
        "max_abs_gd_forget_span_residual": max(gd_span_vals) if gd_span_vals else 0.0,
        "max_abs_cosine_ga_vs_gd": max(cross_vals) if cross_vals else 0.0,
        "checked_rows": int(len(row_ids)),
    }


def _benchmark_pair_diagnostics(
    model: nn.Module,
    tok: Any,
    instances: Sequence[Any],
    device: torch.device,
    llama_like: bool,
    batch_size: int,
    margin: float,
) -> Dict[str, float]:
    margins = stage2.mcf_direct_margins(
        model, tok, instances, device, llama_like, int(batch_size),
        "target_true", "target_new"
    ).detach().float().cpu()
    return {
        "failures": int((margins < float(margin)).sum().item()),
        "minimum_margin": float(margins.min()),
        "mean_margin": float(margins.mean()),
        "maximum_margin": float(margins.max()),
    }


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(int(a.seed))
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, manifest = stage2.load_locked(
        "mcf", visible_path, manifest_path, int(a.seed), int(a.forget_num)
    )
    contract_helpers.assert_target_contract(manifest)
    contract_helpers.validate_direct_only_records(records)

    ns = argparse.Namespace(
        model_path=a.model_path, dtype=a.dtype, device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)
    model.eval()

    cases = core.expand_sensitive_cases(
        records, tok, sensitive_field="target_true", llama_like=llama_like
    )
    if not cases:
        raise RuntimeError("no target_true sensitive PredictionCases were created")
    same_prompt_base_logits = core.cache_base_logits(
        model, tok, cases, device, batch_size=int(a.cache_batch_size)
    )
    all_tids = core.official_target_ids(tok, cases, llama_like=llama_like, device=device)
    sensitive_ids = sorted(set(int(x) for x in all_tids.detach().cpu().tolist()))
    sensitive_ids = [i for i in sensitive_ids if i not in gagd.special_token_ids(tok)]
    if not sensitive_ids:
        raise RuntimeError("no non-special sensitive target_true token rows")

    print("Caching Base forget hidden states...", flush=True)
    base_forget_hidden = core.forward_last_hidden(
        model, tok, cases, device, int(a.cache_batch_size)
    ).detach().cpu()
    token_bases_cpu, forget_basis_receipt = build_token_forget_bases(
        base_forget_hidden, all_tids.detach().cpu(), sensitive_ids, int(a.forget_basis_rank)
    )
    token_bases = {k: v.to(device=device) for k, v in token_bases_cpu.items()}

    benchmark_instances = stage2.mcf_instances(records)
    benchmark_before = _benchmark_pair_diagnostics(
        model, tok, benchmark_instances, device, llama_like,
        int(a.batch_size), float(a.benchmark_pair_margin)
    )

    locality_prompts, locality_protected, locality_receipt = projected.build_relation_locality_controls(
        records, tok, benchmark_instances, int(a.subject_control_count)
    )
    print(f"Caching Base locality references for {len(locality_prompts)} prompts...", flush=True)
    _base_local_hidden, base_local_logits = projected.cache_relation_locality_reference(
        model, tok, locality_prompts, device, int(a.locality_cache_batch_size)
    )

    utility_prompts, utility_receipt = wikipedia_utility.build_utility_prompts(
        tok, Path(a.utility_wikipedia_dir).resolve(),
        sample_size=int(a.utility_sample_size), seed=int(a.utility_seed),
        exclude_first=int(a.utility_exclude_first), max_length=int(a.utility_max_length)
    )
    print(f"Caching Base Wikipedia logits for {len(utility_prompts)} prompts...", flush=True)
    utility_base_logits = wikipedia_utility.cache_utility_base_logits(
        model, tok, utility_prompts, device, int(a.utility_cache_batch_size)
    )

    # Clean hidden-space experiment: untie output head, freeze transformer + input embeddings.
    output_layer = core.untie_and_freeze_output_head(model)
    if model.get_input_embeddings().weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("directional experiment requires an untied LM head")
    selected_base_rows = output_layer.weight.detach().index_select(
        0, torch.tensor(sensitive_ids, dtype=torch.long, device=output_layer.weight.device)
    ).float().cpu()

    hidden_size = int(output_layer.weight.shape[1])
    ga_module = core.SelectedRowDelta(
        len(sensitive_ids), hidden_size, direction_basis=None, device=device
    )
    gd_module = core.SelectedRowDelta(
        len(sensitive_ids), hidden_size, direction_basis=None, device=device
    )
    if ga_module.raw_delta is None or gd_module.raw_delta is None:
        raise RuntimeError("directional experiment requires unrestricted sparse row deltas")
    hook = core.register_output_delta_hook(
        output_layer, sensitive_ids,
        lambda: ga_module.effective_delta() + gd_module.effective_delta(),
    )
    params = [ga_module.raw_delta, gd_module.raw_delta]
    opt = _optimizer(params, a.optimizer, float(a.lm_row_lr))

    forget_sampler = core.IndexSampler(len(cases), int(a.batch_size), int(a.seed) + 71001)
    locality_sampler = core.IndexSampler(
        len(locality_prompts), int(a.locality_batch_size), int(a.seed) + 71003
    )
    utility_sampler = core.IndexSampler(
        len(utility_prompts), int(a.utility_batch_size), int(a.utility_seed) + 71007
    )

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    core.write_json(out_dir / "forget_hidden_basis_receipt.json", forget_basis_receipt)
    core.write_json(out_dir / "relation_locality_receipt.json", locality_receipt)
    core.write_json(out_dir / "wikipedia_utility_receipt.json", utility_receipt)

    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(1, int(a.steps) + 1), desc="MCF directional GA/GD"):
            fidx = forget_sampler.next()
            forget_batch = [cases[i] for i in fidx]
            lidx = locality_sampler.next()
            locality_batch = [locality_prompts[i] for i in lidx]
            locality_ids = [locality_protected[i] for i in lidx]
            uidx = utility_sampler.next()
            utility_batch = [utility_prompts[i] for i in uidx]

            opt.zero_grad(set_to_none=True)
            logits = core.forward_last_logits(model, tok, forget_batch, device)
            tids = core.official_target_ids(
                tok, forget_batch, llama_like=llama_like, device=device
            )
            ga = core.ga_sensitive_logprob(logits, tids)
            gd = core.gd_non_sensitive_kl(logits, same_prompt_base_logits[fidx], tids)

            _lh, local_logits = projected._prompt_hidden_and_logits(
                model, tok, locality_batch, device
            )
            local_base = base_local_logits[lidx]
            lkl = projected.locality_kl(local_logits, local_base)
            lrow = projected.protected_sensitive_logit_mse(
                local_logits, local_base, locality_ids
            )
            utility_logits = wikipedia_utility._forward_prompt_logits(
                model, tok, utility_batch, device
            )
            ukl = wikipedia_utility.utility_kl(
                utility_logits, utility_base_logits[uidx]
            )

            ga_reg = ga_module.raw_delta.square().mean()
            gd_reg = gd_module.raw_delta.square().mean()
            ga_objective = float(a.ga_weight) * ga + 0.5 * float(a.row_delta_weight) * ga_reg
            preserve_objective = (
                float(a.gd_weight) * gd
                + float(a.locality_kl_weight) * lkl
                + float(a.locality_sensitive_logit_weight) * lrow
                + float(a.utility_kl_weight) * ukl
                + 0.5 * float(a.row_delta_weight) * gd_reg
            )
            if not torch.isfinite(ga_objective) or not torch.isfinite(preserve_objective):
                raise FloatingPointError(f"non-finite directional objective at step {step}")

            ga_grad = torch.autograd.grad(
                ga_objective, ga_module.raw_delta, retain_graph=True
            )[0]
            gd_grad = torch.autograd.grad(
                preserve_objective, gd_module.raw_delta, retain_graph=False
            )[0]
            ga_grad_projected, ga_grad_diag = project_rows_tokenwise(
                ga_grad, sensitive_ids, token_bases, mode="span"
            )
            gd_grad_projected, gd_grad_diag = project_rows_tokenwise(
                gd_grad, sensitive_ids, token_bases, mode="null"
            )
            ga_module.raw_delta.grad = ga_grad_projected.to(ga_module.raw_delta.dtype)
            gd_module.raw_delta.grad = gd_grad_projected.to(gd_module.raw_delta.dtype)

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(params, float(a.grad_clip))
                if a.grad_clip > 0 else None
            )
            if grad_norm is not None and not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at step {step}")
            opt.step()
            direction_projection = enforce_directional_parameters_(
                ga_module.raw_delta, gd_module.raw_delta, sensitive_ids, token_bases
            )

            if step == 1 or step % int(a.check_every) == 0 or step == int(a.steps):
                total_delta = ga_module.effective_delta() + gd_module.effective_delta()
                row = {
                    "step": int(step),
                    "ga_sensitive_logprob": float(ga.detach().cpu()),
                    "gd_same_prompt_non_sensitive_kl": float(gd.detach().cpu()),
                    "relation_locality_kl": float(lkl.detach().cpu()),
                    "relation_sensitive_logit_mse": float(lrow.detach().cpu()),
                    "wikipedia_utility_kl": float(ukl.detach().cpu()),
                    "ga_delta_mse": float(ga_module.raw_delta.square().mean().detach().cpu()),
                    "gd_delta_mse": float(gd_module.raw_delta.square().mean().detach().cpu()),
                    "total_delta_mse": float(total_delta.square().mean().detach().cpu()),
                    "raw_ga_gradient_span_norm": float(ga_grad_diag["aligned_norm"]),
                    "raw_ga_gradient_null_norm": float(ga_grad_diag["null_norm"]),
                    "raw_preserve_gradient_span_norm": float(gd_grad_diag["aligned_norm"]),
                    "raw_preserve_gradient_null_norm": float(gd_grad_diag["null_norm"]),
                    **direction_projection,
                    "selected_sensitive_row_count": int(len(sensitive_ids)),
                    "benchmark_retain_seen": 0,
                    "official_neighborhood_seen": 0,
                    "official_paraphrase_seen": 0,
                    "PPL_seen": False,
                }
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()

    del opt
    enforce_directional_parameters_(
        ga_module.raw_delta, gd_module.raw_delta, sensitive_ids, token_bases
    )
    model.eval()

    benchmark_pre_materialize = _benchmark_pair_diagnostics(
        model, tok, benchmark_instances, device, llama_like,
        int(a.batch_size), float(a.benchmark_pair_margin)
    )
    directional_after = directional_residual_diagnostics(
        ga_module.raw_delta.detach(), gd_module.raw_delta.detach(),
        sensitive_ids, token_bases
    )

    total_delta = (
        ga_module.effective_delta() + gd_module.effective_delta()
    ).detach().float()
    ga_delta_final = ga_module.effective_delta().detach().float().cpu()
    gd_delta_final = gd_module.effective_delta().detach().float().cpu()
    hook.remove()
    core.materialize_output_delta(output_layer, sensitive_ids, total_delta)
    model.eval()

    ids = torch.tensor(sensitive_ids, dtype=torch.long, device=output_layer.weight.device)
    materialized_delta = output_layer.weight.index_select(0, ids).float().cpu() - selected_base_rows
    materialization_error = float(
        (materialized_delta - total_delta.cpu()).abs().max()
    )

    benchmark_after = _benchmark_pair_diagnostics(
        model, tok, benchmark_instances, device, llama_like,
        int(a.batch_size), float(a.benchmark_pair_margin)
    )
    locality_after = projected.evaluate_locality_guards(
        model, tok, locality_prompts, locality_protected, base_local_logits,
        device, int(a.locality_cache_batch_size)
    )
    utility_after = wikipedia_utility.evaluate_utility_kl(
        model, tok, utility_prompts, utility_base_logits,
        device, int(a.utility_cache_batch_size)
    )

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    summary = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "source_model_path": str(Path(a.model_path).resolve()),
        "training_visible_path": str(visible_path),
        "split_manifest": str(manifest_path),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "target_contract": {
            "sensitive_unwanted": "requested_rewrite.target_true",
            "benchmark_reference": "requested_rewrite.target_new",
            "field_swapping": False,
        },
        "training_objective": "GA routed to token forget span; same-prompt GD + donor locality + Wikipedia utility routed to token forget-nullspace",
        "target_new_used_as_training_target": False,
        "transformer_trainable": False,
        "input_embeddings_trainable": False,
        "lm_head_untied": True,
        "sensitive_output_rows_trainable_via_sparse_fp32_delta": True,
        "selected_sensitive_token_ids": [int(x) for x in sensitive_ids],
        "selected_sensitive_row_count": int(len(sensitive_ids)),
        "all_non_sensitive_output_rows_equal_base": True,
        "official_neighborhood_seen": 0,
        "official_paraphrase_seen": 0,
        "benchmark_retain_seen": 0,
        "PPL_seen": False,
        "forget_basis_rank_cap": int(a.forget_basis_rank),
        "directional_parameterization": {
            "ga": "token-specific span(B_F,y)",
            "gd_locality_utility": "token-specific span(B_F,y)^perp",
            "hard_parameter_projection_after_every_step": True,
            "optimizer_parameters": "two FP32 sparse selected-row deltas only",
        },
        "weights": {
            "ga": float(a.ga_weight),
            "gd": float(a.gd_weight),
            "relation_locality_kl": float(a.locality_kl_weight),
            "relation_sensitive_logit": float(a.locality_sensitive_logit_weight),
            "wikipedia_utility_kl": float(a.utility_kl_weight),
            "row_delta": float(a.row_delta_weight),
        },
        "steps": int(a.steps),
        "lm_row_lr": float(a.lm_row_lr),
        "optimizer": a.optimizer,
        "trainable_parameter_count": int(
            ga_module.trainable_parameter_count + gd_module.trainable_parameter_count
        ),
        "benchmark_pair_before": benchmark_before,
        "benchmark_pair_pre_materialize": benchmark_pre_materialize,
        "benchmark_pair_after": benchmark_after,
        "relation_locality_after": locality_after,
        "wikipedia_utility_after": utility_after,
        "directional_residual_after": directional_after,
        "ga_delta_mse": float(ga_delta_final.square().mean()),
        "gd_delta_mse": float(gd_delta_final.square().mean()),
        "materialization_max_abs_error": materialization_error,
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "sensitive_rows_directional_gagd_summary.json", summary)

    print("Directional GA/GD checkpoint:", ckpt)
    print("Transformer trainable: False")
    print("Input embeddings trainable: False")
    print("LM head untied: True")
    print("Sensitive output row count:", len(sensitive_ids))
    print("Benchmark pair before:", benchmark_before)
    print("Benchmark pair pre-materialize:", benchmark_pre_materialize)
    print("Benchmark pair after:", benchmark_after)
    print("Relation-locality after:", locality_after)
    print("Wikipedia utility after:", utility_after)
    print("Directional residual after:", directional_after)
    print("Materialization max abs error:", materialization_error)
    print("Official MCF neighborhood/paraphrase probes were NOT used in training.")
    print("Run official evaluation only after this checkpoint is finalized.")


if __name__ == "__main__":
    main()
