#!/usr/bin/env python3
"""MCF Stage-1 GA/GD on sensitive embedding/LM-head rows with locality projection.

Target contract:
  * requested_rewrite.target_true = sensitive / unwanted answer
  * requested_rewrite.target_new  = non-sensitive benchmark reference
  * no field swapping

Only vocabulary rows belonging to target_true tokens may change.  The transformer
is frozen.  On tied models such as the current Llama checkpoint, the same selected
row is simultaneously the input-embedding row and the LM-head row.  On untied
models both input and output selected rows are trainable.  Every non-sensitive row
is gradient-masked throughout training and restored exactly to Base before save.

Forget objective is the canonical SURE Stage-1 GA/GD objective:

    L_forget = lambda_ga * GA(target_true)
             + lambda_gd * KL_same_prompt_non_sensitive(Base || current)

To preserve MCF specificity without reading official neighborhood/paraphrase
probes, each direct training-visible relation template q_i is paired with donor
subjects s_j from other direct training-visible records.  For each sensitive token
y of record i, Base hidden states h0(q_i(s_j)) define a token-specific locality
subspace B_y.  After every optimizer step the selected row update is projected:

    Delta w_y <- Delta w_y - (Delta w_y B_y^T) B_y.

Thus the learned row change is forced to be orthogonal to Base directions used by
relation-matched locality controls.  Two soft guards are used as well:

    L_local_KL = KL(Base || current) on donor prompts
    L_local_row = MSE of protected sensitive-row logits vs Base on donor prompts.

External Wikipedia KL (with the fixed PPL prefix excluded) remains a generic
utility guard.  No official MCF neighborhood, paraphrase, retain, generation, or
PPL evaluation example is visible to training/checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

import gagd_compare as gagd
import mcf_frozen_head_representation_repair as contract_helpers
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import sure_stage1_gagd as stage1_shared
import sure_stage1_gagd_w1k as wikipedia_utility
import sure_stage2_sparse_repair as stage2
import sure_stage2_sparse_repair_subject_contrast_materialized as subject_contrast


METHOD = "SURE-LM-MCF-sensitive-rows-projected-GAGD"
PROTOCOL = "mcf_target_true_sensitive_rows_relation_locality_projection_v1"


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
    p.add_argument("--emb-lm-lr", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--gd-weight", type=float, default=1.0)

    p.add_argument("--subject-control-count", type=int, default=4)
    p.add_argument("--locality-batch-size", type=int, default=4)
    p.add_argument("--locality-cache-batch-size", type=int, default=8)
    p.add_argument("--locality-kl-weight", type=float, default=2.0)
    p.add_argument("--locality-sensitive-logit-weight", type=float, default=5.0)
    p.add_argument(
        "--per-token-locality-rank",
        type=int,
        default=0,
        help="0 keeps the full effective per-token donor-hidden rank; positive caps it",
    )
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
        a.forget_num,
        a.steps,
        a.batch_size,
        a.cache_batch_size,
        a.emb_lm_lr,
        a.check_every,
        a.ga_weight,
        a.subject_control_count,
        a.locality_batch_size,
        a.locality_cache_batch_size,
        a.locality_kl_weight,
        a.locality_sensitive_logit_weight,
        a.utility_sample_size,
        a.utility_batch_size,
        a.utility_cache_batch_size,
        a.utility_max_length,
        a.utility_kl_weight,
    )
    if any(float(v) <= 0 for v in positive):
        p.error("counts, LR, GA/locality/utility weights must be positive")
    nonnegative = (
        a.gd_weight,
        a.grad_clip,
        a.per_token_locality_rank,
        a.row_delta_weight,
        a.utility_exclude_first,
        a.benchmark_pair_margin,
    )
    if any(float(v) < 0 for v in nonnegative):
        p.error("GD/clip/rank/delta/exclusion/margin values must be non-negative")
    if a.utility_exclude_first < 20:
        p.error("utility-exclude-first must be at least 20 to protect the fixed PPL prefix")
    return a


def _prompt_hidden_and_logits(
    model: nn.Module,
    tok: Any,
    prompts: Sequence[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = tok(list(prompts), padding=True, return_tensors="pt").to(device)
    out = model(**encoded, output_hidden_states=True, use_cache=False)
    positions = encoded["attention_mask"].sum(dim=1) - 1
    rows = torch.arange(len(prompts), device=device)
    return out.hidden_states[-1][rows, positions, :], out.logits[rows, positions, :]


def build_relation_locality_controls(
    records: Sequence[Mapping[str, Any]],
    tok: Any,
    benchmark_instances: Sequence[Any],
    control_count: int,
) -> Tuple[List[str], List[List[int]], List[Dict[str, Any]]]:
    """Create donor-subject controls and the sensitive rows each control protects."""
    subjects = subject_contrast._subjects(records)
    prompts: List[str] = []
    protected_ids: List[List[int]] = []
    receipt: List[Dict[str, Any]] = []
    specials = gagd.special_token_ids(tok)

    for position, record in enumerate(records):
        rr = record["requested_rewrite"]
        template = str(rr["prompt"])
        token_ids = stage2.mcf_sensitive_rows(
            tok, benchmark_instances, [position], "target_true"
        )
        token_ids = [int(t) for t in token_ids if int(t) not in specials]
        if not token_ids:
            raise RuntimeError(f"record {position} produced no sensitive target_true rows")
        donors = subject_contrast._donor_indices(position, subjects, int(control_count))
        for donor in donors:
            prompt = template.format(subjects[donor])
            prompts.append(prompt)
            protected_ids.append(list(token_ids))
            receipt.append(
                {
                    "source_position": int(position),
                    "donor_position": int(donor),
                    "original_subject": subjects[position],
                    "donor_subject": subjects[donor],
                    "protected_token_ids": list(token_ids),
                    "prompt": prompt,
                }
            )
    if not prompts:
        raise RuntimeError("no relation-locality controls were constructed")
    return prompts, protected_ids, receipt


@torch.no_grad()
def cache_relation_locality_reference(
    model: nn.Module,
    tok: Any,
    prompts: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden_chunks: List[torch.Tensor] = []
    logit_chunks: List[torch.Tensor] = []
    model.eval()
    for start in range(0, len(prompts), int(batch_size)):
        h, z = _prompt_hidden_and_logits(
            model, tok, prompts[start : start + int(batch_size)], device
        )
        hidden_chunks.append(h.detach().float().cpu())
        logit_chunks.append(z.detach().to(dtype=torch.float16, device="cpu"))
    return torch.cat(hidden_chunks, dim=0), torch.cat(logit_chunks, dim=0)


def build_token_locality_bases(
    base_hidden: torch.Tensor,
    protected_ids: Sequence[Sequence[int]],
    rank_cap: int = 0,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, Any]]:
    """Build orthonormal row-space bases B_y from controls protecting token y."""
    if base_hidden.ndim != 2 or base_hidden.shape[0] != len(protected_ids):
        raise ValueError("base_hidden/control metadata shape mismatch")
    groups: Dict[int, List[int]] = {}
    for row, ids in enumerate(protected_ids):
        for token_id in set(int(x) for x in ids):
            groups.setdefault(token_id, []).append(row)

    bases: Dict[int, torch.Tensor] = {}
    receipt: Dict[str, Any] = {}
    for token_id, rows in groups.items():
        h = base_hidden[rows].float().contiguous()
        _u, singular, vh = torch.linalg.svd(h, full_matrices=False)
        tol = max(h.shape) * torch.finfo(h.dtype).eps * singular.max().clamp_min(1.0)
        effective = int((singular > tol).sum().item())
        keep = effective if int(rank_cap) <= 0 else min(effective, int(rank_cap))
        if keep <= 0:
            continue
        basis = vh[:keep].float().contiguous()
        bases[int(token_id)] = basis
        receipt[str(token_id)] = {
            "control_count": int(len(rows)),
            "effective_rank": int(effective),
            "kept_rank": int(keep),
            "singular_values": [float(v) for v in singular[:keep].tolist()],
        }
    return bases, receipt


def locality_kl(current_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    ref = base_logits.to(device=current_logits.device, dtype=torch.float32)
    ref_logp = F.log_softmax(ref, dim=-1)
    cur_logp = F.log_softmax(current_logits.float(), dim=-1)
    return (ref_logp.exp() * (ref_logp - cur_logp)).sum(dim=-1).mean()


def protected_sensitive_logit_mse(
    current_logits: torch.Tensor,
    base_logits: torch.Tensor,
    protected_ids: Sequence[Sequence[int]],
) -> torch.Tensor:
    if current_logits.shape != base_logits.shape:
        raise ValueError("current/base locality logits must have identical shapes")
    if current_logits.shape[0] != len(protected_ids):
        raise ValueError("logit rows and protected-id metadata differ")
    terms: List[torch.Tensor] = []
    base = base_logits.to(device=current_logits.device, dtype=torch.float32)
    for row, ids in enumerate(protected_ids):
        valid = sorted({int(i) for i in ids if 0 <= int(i) < current_logits.shape[1]})
        if not valid:
            continue
        idx = torch.tensor(valid, dtype=torch.long, device=current_logits.device)
        terms.append(
            (current_logits[row].float().index_select(0, idx) - base[row].index_select(0, idx))
            .square()
            .mean()
        )
    if not terms:
        return current_logits.sum() * 0.0
    return torch.stack(terms).mean()


def selected_row_delta_mse(
    tied_info: Mapping[str, Any],
    base_rows: Mapping[str, torch.Tensor],
    selected_ids: Sequence[int],
) -> torch.Tensor:
    in_w: torch.Tensor = tied_info["input_weight"]
    out_w: torch.Tensor = tied_info["output_weight"]
    tied = bool(tied_info.get("tied"))
    ids = gagd.valid_row_ids(in_w, selected_ids)
    if not ids.numel():
        raise RuntimeError("no valid selected vocabulary rows")
    base_in = base_rows["input"].index_select(0, ids.cpu()).to(
        device=in_w.device, dtype=torch.float32
    )
    terms = [(in_w.index_select(0, ids).float() - base_in).square().mean()]
    if not tied:
        out_ids = gagd.valid_row_ids(out_w, selected_ids)
        base_out = base_rows["output"].index_select(0, out_ids.cpu()).to(
            device=out_w.device, dtype=torch.float32
        )
        terms.append((out_w.index_select(0, out_ids).float() - base_out).square().mean())
    return torch.stack(terms).mean()


@torch.no_grad()
def project_selected_rows_to_locality_nullspace(
    tied_info: Mapping[str, Any],
    base_rows: Mapping[str, torch.Tensor],
    token_bases: Mapping[int, torch.Tensor],
    selected_ids: Sequence[int],
) -> Dict[str, float]:
    """Project selected row deltas away from each token's Base locality subspace."""
    in_w: torch.Tensor = tied_info["input_weight"]
    out_w: torch.Tensor = tied_info["output_weight"]
    tied = bool(tied_info.get("tied"))
    max_removed = 0.0

    def project_weight(weight: torch.Tensor, base: torch.Tensor) -> None:
        nonlocal max_removed
        for token_id in selected_ids:
            tid = int(token_id)
            if tid < 0 or tid >= weight.shape[0]:
                continue
            base_row = base[tid].to(device=weight.device, dtype=weight.dtype)
            delta = (weight[tid] - base_row).float()
            basis = token_bases.get(tid)
            if basis is not None and basis.numel():
                b = basis.to(device=weight.device, dtype=torch.float32)
                component = (delta @ b.T) @ b
                max_removed = max(max_removed, float(component.norm().detach().cpu()))
                delta = delta - component
            weight[tid].copy_(base_row + delta.to(dtype=weight.dtype))

    project_weight(in_w, base_rows["input"])
    if not tied:
        project_weight(out_w, base_rows["output"])
    return {"max_removed_component_norm": float(max_removed)}


@torch.no_grad()
def projection_residual_diagnostics(
    tied_info: Mapping[str, Any],
    base_rows: Mapping[str, torch.Tensor],
    token_bases: Mapping[int, torch.Tensor],
    selected_ids: Sequence[int],
) -> Dict[str, float]:
    in_w: torch.Tensor = tied_info["input_weight"]
    out_w: torch.Tensor = tied_info["output_weight"]
    tied = bool(tied_info.get("tied"))
    vals: List[float] = []

    def inspect(weight: torch.Tensor, base: torch.Tensor) -> None:
        for token_id in selected_ids:
            tid = int(token_id)
            basis = token_bases.get(tid)
            if basis is None or not basis.numel() or tid >= weight.shape[0]:
                continue
            delta = (weight[tid].float() - base[tid].to(weight.device).float())
            b = basis.to(weight.device).float()
            vals.append(float((delta @ b.T).abs().max().detach().cpu()))

    inspect(in_w, base_rows["input"])
    if not tied:
        inspect(out_w, base_rows["output"])
    return {
        "max_abs_delta_dot_locality_basis": max(vals) if vals else 0.0,
        "checked_row_matrix_pairs": int(len(vals)),
    }


@torch.no_grad()
def evaluate_locality_guards(
    model: nn.Module,
    tok: Any,
    prompts: Sequence[str],
    protected_ids: Sequence[Sequence[int]],
    base_logits: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    kl_values: List[torch.Tensor] = []
    protected_errors: List[torch.Tensor] = []
    model.eval()
    for start in range(0, len(prompts), int(batch_size)):
        stop = min(len(prompts), start + int(batch_size))
        _h, z = _prompt_hidden_and_logits(model, tok, prompts[start:stop], device)
        ref = base_logits[start:stop].to(device=device, dtype=torch.float32)
        ref_logp = F.log_softmax(ref, dim=-1)
        cur_logp = F.log_softmax(z.float(), dim=-1)
        kl_values.append(
            (ref_logp.exp() * (ref_logp - cur_logp)).sum(dim=-1).detach().cpu()
        )
        for row, ids in enumerate(protected_ids[start:stop]):
            valid = sorted({int(i) for i in ids if 0 <= int(i) < z.shape[1]})
            if not valid:
                continue
            idx = torch.tensor(valid, dtype=torch.long, device=device)
            err = (z[row].float().index_select(0, idx) - ref[row].index_select(0, idx)).abs()
            protected_errors.append(err.detach().cpu())
    kl = torch.cat(kl_values).float()
    pe = torch.cat(protected_errors).float() if protected_errors else torch.zeros(1)
    return {
        "prompt_count": int(len(prompts)),
        "kl_mean": float(kl.mean()),
        "kl_p95": float(torch.quantile(kl, 0.95)),
        "kl_max": float(kl.max()),
        "protected_sensitive_logit_abs_mean": float(pe.mean()),
        "protected_sensitive_logit_abs_p95": float(torch.quantile(pe, 0.95)),
        "protected_sensitive_logit_abs_max": float(pe.max()),
    }


def _optimizer(params: Iterable[nn.Parameter], kind: str, lr: float):
    params = list(params)
    if kind == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if kind == "adam":
        return torch.optim.Adam(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)


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
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=True)
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

    benchmark_instances = stage2.mcf_instances(records)
    benchmark_before = _benchmark_pair_diagnostics(
        model, tok, benchmark_instances, device, llama_like,
        int(a.batch_size), float(a.benchmark_pair_margin)
    )

    locality_prompts, locality_protected, locality_receipt = build_relation_locality_controls(
        records, tok, benchmark_instances, int(a.subject_control_count)
    )
    print(
        f"Caching Base hidden/logit references for {len(locality_prompts)} relation-locality controls...",
        flush=True,
    )
    base_local_hidden, base_local_logits = cache_relation_locality_reference(
        model, tok, locality_prompts, device, int(a.locality_cache_batch_size)
    )
    token_bases_cpu, basis_receipt = build_token_locality_bases(
        base_local_hidden, locality_protected, int(a.per_token_locality_rank)
    )
    token_bases = {k: v.to(device=device) for k, v in token_bases_cpu.items()}

    utility_prompts, utility_receipt = wikipedia_utility.build_utility_prompts(
        tok,
        Path(a.utility_wikipedia_dir).resolve(),
        sample_size=int(a.utility_sample_size),
        seed=int(a.utility_seed),
        exclude_first=int(a.utility_exclude_first),
        max_length=int(a.utility_max_length),
    )
    print(f"Caching Base logits for {len(utility_prompts)} Wikipedia controls...", flush=True)
    utility_base_logits = wikipedia_utility.cache_utility_base_logits(
        model, tok, utility_prompts, device, int(a.utility_cache_batch_size)
    )

    trainable_summary, tied_info = gagd.configure_trainable(
        model, gagd.POST_TRAINING_RESTORE_MODE
    )
    params = gagd.unique_trainable_params(model)
    base_rows = gagd.snapshot_embedding_output_weights(tied_info)
    # Full transformer must remain frozen; only embedding/output matrices are trainable.
    allowed_ids = {id(tied_info["input_weight"]), id(tied_info["output_weight"])}
    bad = [name for name, p in model.named_parameters() if p.requires_grad and id(p) not in allowed_ids]
    if bad:
        raise RuntimeError(f"non-vocabulary parameters unexpectedly trainable: {bad[:5]}")

    opt = _optimizer(params, a.optimizer, float(a.emb_lm_lr))
    forget_sampler = core.IndexSampler(len(cases), int(a.batch_size), int(a.seed) + 51001)
    locality_sampler = core.IndexSampler(
        len(locality_prompts), int(a.locality_batch_size), int(a.seed) + 51003
    )
    utility_sampler = core.IndexSampler(
        len(utility_prompts), int(a.utility_batch_size), int(a.utility_seed) + 51007
    )

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    core.write_json(out_dir / "relation_locality_receipt.json", locality_receipt)
    core.write_json(out_dir / "token_locality_basis_receipt.json", basis_receipt)
    core.write_json(out_dir / "wikipedia_utility_receipt.json", utility_receipt)

    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(1, int(a.steps) + 1), desc="MCF sensitive-row projected GA/GD"):
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

            _lh, local_logits = _prompt_hidden_and_logits(
                model, tok, locality_batch, device
            )
            local_base = base_local_logits[lidx]
            lkl = locality_kl(local_logits, local_base)
            lrow = protected_sensitive_logit_mse(local_logits, local_base, locality_ids)

            utility_logits = wikipedia_utility._forward_prompt_logits(
                model, tok, utility_batch, device
            )
            ukl = wikipedia_utility.utility_kl(
                utility_logits, utility_base_logits[uidx]
            )
            drow = selected_row_delta_mse(tied_info, base_rows, sensitive_ids)

            total = (
                float(a.ga_weight) * ga
                + float(a.gd_weight) * gd
                + float(a.locality_kl_weight) * lkl
                + float(a.locality_sensitive_logit_weight) * lrow
                + float(a.utility_kl_weight) * ukl
                + float(a.row_delta_weight) * drow
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}")
            total.backward()

            # Hard row mask: only target_true-sensitive vocabulary rows may update.
            gagd.apply_row_mask_and_restore(tied_info, sensitive_ids, base_rows)
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(params, float(a.grad_clip))
                if a.grad_clip > 0 else None
            )
            if grad_norm is not None and not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at step {step}")
            opt.step()

            # Hard locality geometry: strip components that alter the selected row
            # along relation-matched Base hidden directions.
            projection = project_selected_rows_to_locality_nullspace(
                tied_info, base_rows, token_bases, sensitive_ids
            )

            if step == 1 or step % int(a.check_every) == 0 or step == int(a.steps):
                row = {
                    "step": int(step),
                    "total_loss": float(total.detach().cpu()),
                    "ga_sensitive_logprob": float(ga.detach().cpu()),
                    "gd_same_prompt_non_sensitive_kl": float(gd.detach().cpu()),
                    "relation_locality_kl": float(lkl.detach().cpu()),
                    "relation_sensitive_logit_mse": float(lrow.detach().cpu()),
                    "wikipedia_utility_kl": float(ukl.detach().cpu()),
                    "selected_row_delta_mse": float(drow.detach().cpu()),
                    "projection_max_removed_component_norm": projection["max_removed_component_norm"],
                    "selected_sensitive_row_count": int(len(sensitive_ids)),
                    "benchmark_retain_seen": 0,
                    "official_neighborhood_seen": 0,
                    "official_paraphrase_seen": 0,
                    "PPL_seen": False,
                }
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()

    del opt
    # Exact Base restoration for every non-sensitive row before all post-training diagnostics/save.
    gagd.restore_non_selected_rows(tied_info, sensitive_ids, base_rows)
    project_selected_rows_to_locality_nullspace(
        tied_info, base_rows, token_bases, sensitive_ids
    )
    model.eval()

    benchmark_after = _benchmark_pair_diagnostics(
        model, tok, benchmark_instances, device, llama_like,
        int(a.batch_size), float(a.benchmark_pair_margin)
    )
    locality_after = evaluate_locality_guards(
        model, tok, locality_prompts, locality_protected, base_local_logits,
        device, int(a.locality_cache_batch_size)
    )
    utility_after = wikipedia_utility.evaluate_utility_kl(
        model, tok, utility_prompts, utility_base_logits,
        device, int(a.utility_cache_batch_size)
    )
    projection_after = projection_residual_diagnostics(
        tied_info, base_rows, token_bases, sensitive_ids
    )
    final_row_delta = selected_row_delta_mse(tied_info, base_rows, sensitive_ids)

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
        "training_objective": "GA(target_true)+same-prompt non-sensitive KL+relation locality KL+protected sensitive-logit MSE+Wikipedia KL+row delta penalty",
        "target_new_used_as_training_target": False,
        "transformer_trainable": False,
        "input_embedding_sensitive_rows_trainable": True,
        "lm_head_sensitive_rows_trainable": True,
        "input_output_tied": bool(tied_info.get("tied")),
        "selected_sensitive_token_ids": [int(x) for x in sensitive_ids],
        "selected_sensitive_row_count": int(len(sensitive_ids)),
        "all_non_sensitive_rows_restored_to_base": True,
        "relation_locality_prompt_count": int(len(locality_prompts)),
        "official_neighborhood_seen": 0,
        "official_paraphrase_seen": 0,
        "benchmark_retain_seen": 0,
        "PPL_seen": False,
        "per_token_locality_rank_cap": int(a.per_token_locality_rank),
        "weights": {
            "ga": float(a.ga_weight),
            "gd": float(a.gd_weight),
            "relation_locality_kl": float(a.locality_kl_weight),
            "relation_sensitive_logit": float(a.locality_sensitive_logit_weight),
            "wikipedia_utility_kl": float(a.utility_kl_weight),
            "row_delta": float(a.row_delta_weight),
        },
        "steps": int(a.steps),
        "emb_lm_lr": float(a.emb_lm_lr),
        "optimizer": a.optimizer,
        "trainable_summary": {
            "n_trainable_params": int(trainable_summary.n_trainable_params),
            "n_total_params": int(trainable_summary.n_total_params),
            "trainable_param_percent": float(trainable_summary.trainable_param_percent),
            "trainable_names": list(trainable_summary.trainable_names),
        },
        "benchmark_pair_before": benchmark_before,
        "benchmark_pair_after": benchmark_after,
        "relation_locality_after": locality_after,
        "wikipedia_utility_after": utility_after,
        "projection_after": projection_after,
        "selected_row_delta_mse": float(final_row_delta.detach().cpu()),
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "sensitive_rows_projected_gagd_summary.json", summary)

    print("Sensitive-row projected GA/GD checkpoint:", ckpt)
    print("Transformer trainable: False")
    print("Input embedding sensitive rows trainable: True")
    print("LM-head sensitive rows trainable: True")
    print("Input/output tied:", bool(tied_info.get("tied")))
    print("Sensitive row count:", len(sensitive_ids))
    print("Benchmark pair before:", benchmark_before)
    print("Benchmark pair after:", benchmark_after)
    print("Relation-locality after:", locality_after)
    print("Wikipedia utility after:", utility_after)
    print("Projection residual after:", projection_after)
    print("Official MCF neighborhood/paraphrase probes were NOT used in training.")
    print("Run official evaluation only after this checkpoint is finalized.")


if __name__ == "__main__":
    main()
