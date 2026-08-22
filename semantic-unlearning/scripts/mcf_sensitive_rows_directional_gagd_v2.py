#!/usr/bin/env python3
"""MCF Directional SURE v2: sparse input rows + protected/exclusive output geometry.

Target contract:
  * requested_rewrite.target_true = sensitive / unwanted answer
  * requested_rewrite.target_new  = non-sensitive benchmark reference (diagnostic only)

This ablation keeps the transformer frozen, unties the LM head, and trains only
three sparse FP32 components on target_true-sensitive vocabulary rows:

  1. input-embedding delta Delta E_y (original SURE-style mixed GA/GD signal),
  2. output GA delta Delta W_GA,y,
  3. output preservation delta Delta W_GD,y.

The output geometry is built from training-visible / external protection data.
At each basis refresh, relation-donor and external Wikipedia hidden states form
an observed protected span B_P.  For each sensitive token y, current forget
hidden states H_F,y are residualized against B_P:

    H_S,y = H_F,y - (H_F,y B_P^T) B_P
    B_S,y = row_basis(H_S,y)

The output deltas are then hard constrained after every optimizer step:

    Delta W_GA,y in span(B_S,y)
    Delta W_GD,y in span(B_P)

The GA objective routes gradients to Delta E and Delta W_GA only.  Same-prompt
non-sensitive KL, relation-donor locality losses, and external Wikipedia KL
route gradients to Delta E and Delta W_GD only.  Thus input embeddings retain
the original sparse SURE mixed objective, while LM-head updates separate
sensitive-exclusive and observed-protected directions explicitly.

Because Delta E changes hidden states, B_P and B_S are rebuilt periodically.
On refresh, existing output deltas are reprojected into the new spaces and only
the output-delta optimizer moments are reset; input-delta optimizer state is
preserved.

No official MCF neighborhood, paraphrase, retain, generation, or PPL evaluation
example is visible to training/checkpoint selection.
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
import mcf_sensitive_rows_projected_gagd as projected
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import sure_stage1_gagd_w1k as wikipedia_utility
import sure_stage2_sparse_repair as stage2


METHOD = "SURE-LM-MCF-directional-GAGD-v2-protected-exclusive"
PROTOCOL = "mcf_target_true_directional_gagd_protected_exclusive_v2"


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
    p.add_argument("--input-row-lr", type=float, default=5e-5)
    p.add_argument("--lm-row-lr", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--gd-weight", type=float, default=1.0)

    p.add_argument("--basis-refresh-every", type=int, default=25)
    p.add_argument(
        "--protected-basis-rank",
        type=int,
        default=0,
        help="0 keeps full effective rank of donor+Wikipedia protected hidden bank",
    )
    p.add_argument(
        "--sensitive-exclusive-rank",
        type=int,
        default=0,
        help="0 keeps full effective rank of each token's protected-residualized forget states",
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
        a.forget_num, a.steps, a.batch_size, a.cache_batch_size,
        a.input_row_lr, a.lm_row_lr, a.check_every, a.ga_weight,
        a.basis_refresh_every, a.subject_control_count, a.locality_batch_size,
        a.locality_cache_batch_size, a.locality_kl_weight,
        a.locality_sensitive_logit_weight, a.utility_sample_size,
        a.utility_batch_size, a.utility_cache_batch_size, a.utility_max_length,
        a.utility_kl_weight,
    )
    if any(float(v) <= 0 for v in positive):
        p.error("counts, LRs, refresh cadence, and positive loss weights must be positive")
    nonnegative = (
        a.gd_weight, a.grad_clip, a.protected_basis_rank,
        a.sensitive_exclusive_rank, a.row_delta_weight,
        a.utility_exclude_first, a.benchmark_pair_margin,
    )
    if any(float(v) < 0 for v in nonnegative):
        p.error("GD/clip/ranks/delta/exclusion/margin values must be non-negative")
    if a.utility_exclude_first < 20:
        p.error("utility-exclude-first must be at least 20 to protect the fixed PPL prefix")
    return a


def _optimizer(
    input_param: nn.Parameter,
    output_params: Iterable[nn.Parameter],
    kind: str,
    input_lr: float,
    output_lr: float,
):
    groups = [
        {"params": [input_param], "lr": float(input_lr)},
        {"params": list(output_params), "lr": float(output_lr)},
    ]
    if kind == "sgd":
        return torch.optim.SGD(groups)
    if kind == "adam":
        return torch.optim.Adam(groups)
    return torch.optim.AdamW(groups, weight_decay=0.0)


class SparseInputRowDelta(nn.Module):
    """Sparse FP32 input-embedding deltas over a frozen embedding matrix."""

    def __init__(self, embedding: nn.Module, row_ids: Sequence[int]) -> None:
        super().__init__()
        if embedding is None or not hasattr(embedding, "weight"):
            raise ValueError("input embedding layer must expose weight")
        if embedding.weight.requires_grad:
            raise ValueError("base input embedding matrix must be frozen")
        ids = sorted({int(x) for x in row_ids})
        if not ids:
            raise ValueError("SparseInputRowDelta requires selected rows")
        device = embedding.weight.device
        self.embedding = embedding
        self.row_ids = tuple(ids)
        indices = torch.tensor(ids, dtype=torch.long, device=device)
        self.register_buffer("indices", indices)
        lookup = torch.full(
            (embedding.weight.shape[0],), -1, dtype=torch.long, device=device
        )
        lookup[indices] = torch.arange(indices.numel(), dtype=torch.long, device=device)
        self.register_buffer("lookup", lookup)
        self.delta = nn.Parameter(
            torch.zeros(
                (len(ids), embedding.weight.shape[1]),
                dtype=torch.float32,
                device=device,
            )
        )
        self._hook = embedding.register_forward_hook(self._forward_hook)

    def _forward_hook(
        self, _module: nn.Module, inputs: Tuple[Any, ...], output: torch.Tensor
    ) -> torch.Tensor:
        if not inputs:
            raise RuntimeError("embedding hook did not receive input IDs")
        input_ids = inputs[0]
        positions = self.lookup[input_ids]
        valid = positions >= 0
        safe = positions.clamp_min(0)
        correction = F.embedding(safe, self.delta)
        correction = correction * valid.unsqueeze(-1)
        return output + correction.to(dtype=output.dtype)

    def remove(self) -> None:
        self._hook.remove()


@torch.no_grad()
def _materialize_selected_delta(
    weight: torch.Tensor, row_ids: Sequence[int], delta: torch.Tensor
) -> None:
    ids = torch.tensor([int(x) for x in row_ids], dtype=torch.long, device=weight.device)
    if ids.numel() != delta.shape[0]:
        raise ValueError("selected row/delta count mismatch")
    current = weight.index_select(0, ids)
    weight.index_copy_(
        0, ids, current + delta.to(device=weight.device, dtype=weight.dtype)
    )


@torch.no_grad()
def _prompt_hidden(
    model: nn.Module,
    tok: Any,
    prompts: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    model.eval()
    for start in range(0, len(prompts), int(batch_size)):
        batch = list(prompts[start : start + int(batch_size)])
        encoded = tok(batch, padding=True, return_tensors="pt").to(device)
        out = model(**encoded, output_hidden_states=True, use_cache=False)
        positions = encoded["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(batch), device=device)
        chunks.append(out.hidden_states[-1][rows, positions, :].detach().float())
    if not chunks:
        raise ValueError("prompt hidden bank cannot be empty")
    return torch.cat(chunks, dim=0)


def _orthonormal_basis(
    rows: torch.Tensor, rank_cap: int
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    values = rows.detach().float()
    if values.ndim != 2:
        raise ValueError("basis rows must be a matrix")
    hidden = int(values.shape[1])
    if values.shape[0] == 0 or values.numel() == 0:
        return values.new_empty((0, hidden)), {
            "source_row_count": 0,
            "effective_rank": 0,
            "kept_rank": 0,
            "energy_fraction_retained": 0.0,
            "singular_values": [],
        }
    _u, singular, vh = torch.linalg.svd(values, full_matrices=False)
    tol = max(values.shape) * torch.finfo(values.dtype).eps * singular.max().clamp_min(1.0)
    effective = int((singular > tol).sum().item())
    keep = effective if int(rank_cap) <= 0 else min(effective, int(rank_cap))
    basis = vh[:keep].float().contiguous()
    if keep:
        identity = basis @ basis.T
        if not torch.allclose(
            identity,
            torch.eye(keep, device=basis.device, dtype=basis.dtype),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise RuntimeError("constructed basis is not orthonormal")
    total_energy = float(singular.square().sum().detach().cpu())
    kept_energy = float(singular[:keep].square().sum().detach().cpu())
    return basis, {
        "source_row_count": int(values.shape[0]),
        "effective_rank": int(effective),
        "kept_rank": int(keep),
        "energy_fraction_retained": (
            kept_energy / total_energy if total_energy > 0.0 else 0.0
        ),
        "singular_values": [float(x) for x in singular[:keep].detach().cpu().tolist()],
    }


def build_sensitive_exclusive_bases(
    forget_hidden: torch.Tensor,
    target_ids: torch.Tensor,
    selected_ids: Sequence[int],
    protected_basis: torch.Tensor,
    rank_cap: int = 0,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, Any]]:
    h = forget_hidden.detach().float()
    tids = target_ids.detach().long().to(h.device)
    bp = protected_basis.to(device=h.device, dtype=torch.float32)
    if h.ndim != 2 or tids.ndim != 1 or h.shape[0] != tids.shape[0]:
        raise ValueError("forget hidden/target-id shape mismatch")
    result: Dict[int, torch.Tensor] = {}
    receipt: Dict[str, Any] = {}
    for token_id in sorted({int(x) for x in selected_ids}):
        rows = torch.nonzero(tids == token_id, as_tuple=False).flatten()
        if not rows.numel():
            raise RuntimeError(f"selected token {token_id} has no forget hidden states")
        hf = h.index_select(0, rows).contiguous()
        protected_component = (hf @ bp.T) @ bp if bp.numel() else torch.zeros_like(hf)
        hs = hf - protected_component
        basis, diag = _orthonormal_basis(hs, int(rank_cap))
        forget_energy = float(hf.square().sum().detach().cpu())
        exclusive_energy = float(hs.square().sum().detach().cpu())
        overlap = (
            float((basis @ bp.T).abs().max().detach().cpu())
            if basis.numel() and bp.numel()
            else 0.0
        )
        result[token_id] = basis.detach()
        receipt[str(token_id)] = {
            "case_count": int(rows.numel()),
            "forget_energy": forget_energy,
            "sensitive_exclusive_energy": exclusive_energy,
            "sensitive_exclusive_energy_fraction": (
                exclusive_energy / forget_energy if forget_energy > 0.0 else 0.0
            ),
            "max_abs_BS_BP_overlap": overlap,
            **diag,
        }
    return result, receipt


def project_rows_tokenwise_span(
    rows: torch.Tensor,
    row_ids: Sequence[int],
    token_bases: Mapping[int, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if rows.ndim != 2 or rows.shape[0] != len(row_ids):
        raise ValueError("row tensor/row-id mismatch")
    out = torch.zeros_like(rows, dtype=torch.float32)
    kept_sq = 0.0
    removed_sq = 0.0
    for index, token_id in enumerate(row_ids):
        basis = token_bases.get(int(token_id))
        value = rows[index].float()
        if basis is None or not basis.numel():
            projected_value = torch.zeros_like(value)
        else:
            b = basis.to(device=rows.device, dtype=torch.float32)
            projected_value = (value @ b.T) @ b
        out[index] = projected_value
        kept_sq += float(projected_value.square().sum().detach().cpu())
        removed_sq += float((value - projected_value).square().sum().detach().cpu())
    return out, {
        "kept_norm": float(kept_sq ** 0.5),
        "removed_norm": float(removed_sq ** 0.5),
    }


def project_rows_global_span(
    rows: torch.Tensor, basis: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, float]]:
    values = rows.float()
    b = basis.to(device=values.device, dtype=torch.float32)
    projected_value = (
        (values @ b.T) @ b if b.numel() else torch.zeros_like(values)
    )
    removed = values - projected_value
    return projected_value, {
        "kept_norm": float(projected_value.norm().detach().cpu()),
        "removed_norm": float(removed.norm().detach().cpu()),
    }


@torch.no_grad()
def enforce_output_geometry_(
    ga_delta: torch.Tensor,
    gd_delta: torch.Tensor,
    row_ids: Sequence[int],
    sensitive_bases: Mapping[int, torch.Tensor],
    protected_basis: torch.Tensor,
) -> Dict[str, float]:
    ga_projected, ga_diag = project_rows_tokenwise_span(
        ga_delta.detach(), row_ids, sensitive_bases
    )
    gd_projected, gd_diag = project_rows_global_span(
        gd_delta.detach(), protected_basis
    )
    ga_removed = ga_delta.detach().float() - ga_projected
    gd_removed = gd_delta.detach().float() - gd_projected
    ga_delta.copy_(ga_projected.to(dtype=ga_delta.dtype))
    gd_delta.copy_(gd_projected.to(dtype=gd_delta.dtype))
    return {
        "ga_removed_outside_sensitive_exclusive_norm": float(ga_removed.norm().cpu()),
        "gd_removed_outside_protected_norm": float(gd_removed.norm().cpu()),
        "ga_sensitive_exclusive_norm": float(ga_diag["kept_norm"]),
        "gd_protected_norm": float(gd_diag["kept_norm"]),
    }


@torch.no_grad()
def directional_residual_diagnostics(
    ga_delta: torch.Tensor,
    gd_delta: torch.Tensor,
    row_ids: Sequence[int],
    sensitive_bases: Mapping[int, torch.Tensor],
    protected_basis: torch.Tensor,
) -> Dict[str, float]:
    bp = protected_basis.to(device=ga_delta.device, dtype=torch.float32)
    ga_protected: List[float] = []
    gd_exclusive: List[float] = []
    cosine: List[float] = []
    for i, token_id in enumerate(row_ids):
        ga = ga_delta[i].float()
        gd = gd_delta[i].float()
        ga_protected.append(
            float((ga @ bp.T).abs().max().cpu()) if bp.numel() else 0.0
        )
        bs = sensitive_bases.get(int(token_id))
        if bs is not None and bs.numel():
            b = bs.to(device=ga_delta.device, dtype=torch.float32)
            gd_exclusive.append(float((gd @ b.T).abs().max().cpu()))
        else:
            gd_exclusive.append(0.0)
        denom = ga.norm() * gd.norm()
        cosine.append(float((ga @ gd).abs().div(denom.clamp_min(1e-12)).cpu()))
    return {
        "max_abs_ga_dot_protected_basis": max(ga_protected) if ga_protected else 0.0,
        "max_abs_gd_dot_sensitive_exclusive_basis": max(gd_exclusive) if gd_exclusive else 0.0,
        "max_abs_cosine_ga_vs_gd": max(cosine) if cosine else 0.0,
        "checked_rows": int(len(row_ids)),
    }


def _geometry_summary(
    protected_diag: Mapping[str, Any],
    sensitive_receipt: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    fractions = [
        float(value["sensitive_exclusive_energy_fraction"])
        for value in sensitive_receipt.values()
    ]
    ranks = [int(value["kept_rank"]) for value in sensitive_receipt.values()]
    overlaps = [float(value["max_abs_BS_BP_overlap"]) for value in sensitive_receipt.values()]
    return {
        "protected_source_row_count": int(protected_diag["source_row_count"]),
        "protected_effective_rank": int(protected_diag["effective_rank"]),
        "protected_kept_rank": int(protected_diag["kept_rank"]),
        "sensitive_token_count": int(len(sensitive_receipt)),
        "sensitive_exclusive_rank_min": min(ranks) if ranks else 0,
        "sensitive_exclusive_rank_max": max(ranks) if ranks else 0,
        "sensitive_exclusive_energy_fraction_min": min(fractions) if fractions else 0.0,
        "sensitive_exclusive_energy_fraction_mean": (
            sum(fractions) / len(fractions) if fractions else 0.0
        ),
        "sensitive_exclusive_energy_fraction_max": max(fractions) if fractions else 0.0,
        "max_abs_BS_BP_overlap": max(overlaps) if overlaps else 0.0,
    }


@torch.no_grad()
def refresh_geometry(
    model: nn.Module,
    tok: Any,
    cases: Sequence[Any],
    target_ids: torch.Tensor,
    selected_ids: Sequence[int],
    locality_prompts: Sequence[str],
    utility_prompts: Sequence[str],
    device: torch.device,
    *,
    forget_batch_size: int,
    locality_batch_size: int,
    utility_batch_size: int,
    protected_rank: int,
    sensitive_rank: int,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[str, Any]]:
    forget_hidden = core.forward_last_hidden(
        model, tok, cases, device, int(forget_batch_size)
    ).detach().float()
    locality_hidden = _prompt_hidden(
        model, tok, locality_prompts, device, int(locality_batch_size)
    )
    utility_hidden = _prompt_hidden(
        model, tok, utility_prompts, device, int(utility_batch_size)
    )
    protected_rows = torch.cat([locality_hidden, utility_hidden], dim=0)
    protected_basis, protected_diag = _orthonormal_basis(
        protected_rows, int(protected_rank)
    )
    sensitive_bases, sensitive_receipt = build_sensitive_exclusive_bases(
        forget_hidden,
        target_ids.to(device=forget_hidden.device),
        selected_ids,
        protected_basis,
        int(sensitive_rank),
    )
    receipt = {
        "protected": protected_diag,
        "sensitive_tokens": sensitive_receipt,
        "summary": _geometry_summary(protected_diag, sensitive_receipt),
    }
    return protected_basis.detach(), sensitive_bases, receipt


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


def _zero_if_none(value: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(reference) if value is None else value


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
    all_tids = core.official_target_ids(
        tok, cases, llama_like=llama_like, device=device
    )
    sensitive_ids = sorted(set(int(x) for x in all_tids.detach().cpu().tolist()))
    sensitive_ids = [i for i in sensitive_ids if i not in gagd.special_token_ids(tok)]
    if not sensitive_ids:
        raise RuntimeError("no non-special sensitive target_true token rows")

    benchmark_instances = stage2.mcf_instances(records)
    benchmark_before = _benchmark_pair_diagnostics(
        model, tok, benchmark_instances, device, llama_like,
        int(a.batch_size), float(a.benchmark_pair_margin)
    )

    locality_prompts, locality_protected, locality_receipt = (
        projected.build_relation_locality_controls(
            records, tok, benchmark_instances, int(a.subject_control_count)
        )
    )
    print(
        f"Caching Base locality references for {len(locality_prompts)} prompts...",
        flush=True,
    )
    _base_local_hidden, base_local_logits = projected.cache_relation_locality_reference(
        model, tok, locality_prompts, device, int(a.locality_cache_batch_size)
    )

    utility_prompts, utility_receipt = wikipedia_utility.build_utility_prompts(
        tok,
        Path(a.utility_wikipedia_dir).resolve(),
        sample_size=int(a.utility_sample_size),
        seed=int(a.utility_seed),
        exclude_first=int(a.utility_exclude_first),
        max_length=int(a.utility_max_length),
    )
    print(f"Caching Base Wikipedia logits for {len(utility_prompts)} prompts...", flush=True)
    utility_base_logits = wikipedia_utility.cache_utility_base_logits(
        model, tok, utility_prompts, device, int(a.utility_cache_batch_size)
    )

    # Untie LM head and freeze every Base parameter. Sparse hooks carry all learning.
    output_layer = core.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("Directional SURE v2 requires an untied LM head")
    if input_layer.weight.requires_grad or output_layer.weight.requires_grad:
        raise RuntimeError("Base input/output matrices must remain frozen")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("all Base model parameters must be frozen")

    ids_device = torch.tensor(
        sensitive_ids, dtype=torch.long, device=input_layer.weight.device
    )
    selected_base_input = input_layer.weight.detach().index_select(0, ids_device).float().cpu()
    selected_base_output = output_layer.weight.detach().index_select(
        0, ids_device.to(output_layer.weight.device)
    ).float().cpu()

    input_module = SparseInputRowDelta(input_layer, sensitive_ids)
    hidden_size = int(output_layer.weight.shape[1])
    ga_module = core.SelectedRowDelta(
        len(sensitive_ids), hidden_size, direction_basis=None, device=device
    )
    gd_module = core.SelectedRowDelta(
        len(sensitive_ids), hidden_size, direction_basis=None, device=device
    )
    if ga_module.raw_delta is None or gd_module.raw_delta is None:
        raise RuntimeError("Directional SURE v2 requires unrestricted sparse output deltas")
    output_hook = core.register_output_delta_hook(
        output_layer,
        sensitive_ids,
        lambda: ga_module.effective_delta() + gd_module.effective_delta(),
    )

    opt = _optimizer(
        input_module.delta,
        [ga_module.raw_delta, gd_module.raw_delta],
        a.optimizer,
        float(a.input_row_lr),
        float(a.lm_row_lr),
    )

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    core.write_json(out_dir / "relation_locality_receipt.json", locality_receipt)
    core.write_json(out_dir / "wikipedia_utility_receipt.json", utility_receipt)

    print("Building initial protected / sensitive-exclusive geometry...", flush=True)
    protected_basis, sensitive_bases, geometry_receipt = refresh_geometry(
        model,
        tok,
        cases,
        all_tids,
        sensitive_ids,
        locality_prompts,
        utility_prompts,
        device,
        forget_batch_size=int(a.cache_batch_size),
        locality_batch_size=int(a.locality_cache_batch_size),
        utility_batch_size=int(a.utility_cache_batch_size),
        protected_rank=int(a.protected_basis_rank),
        sensitive_rank=int(a.sensitive_exclusive_rank),
    )
    geometry_receipt["step"] = 0
    core.write_json(out_dir / "initial_geometry_receipt.json", geometry_receipt)
    print("Initial geometry:", geometry_receipt["summary"], flush=True)

    forget_sampler = core.IndexSampler(len(cases), int(a.batch_size), int(a.seed) + 81001)
    locality_sampler = core.IndexSampler(
        len(locality_prompts), int(a.locality_batch_size), int(a.seed) + 81003
    )
    utility_sampler = core.IndexSampler(
        len(utility_prompts), int(a.utility_batch_size), int(a.utility_seed) + 81007
    )

    geometry_log_path = out_dir / "basis_refresh_log.jsonl"
    with geometry_log_path.open("w", encoding="utf-8") as geometry_log:
        geometry_log.write(json.dumps(geometry_receipt) + "\n")
        geometry_log.flush()

        with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
            for step in tqdm(range(1, int(a.steps) + 1), desc="MCF Directional SURE v2"):
                if step > 1 and (step - 1) % int(a.basis_refresh_every) == 0:
                    protected_basis, sensitive_bases, refresh = refresh_geometry(
                        model,
                        tok,
                        cases,
                        all_tids,
                        sensitive_ids,
                        locality_prompts,
                        utility_prompts,
                        device,
                        forget_batch_size=int(a.cache_batch_size),
                        locality_batch_size=int(a.locality_cache_batch_size),
                        utility_batch_size=int(a.utility_cache_batch_size),
                        protected_rank=int(a.protected_basis_rank),
                        sensitive_rank=int(a.sensitive_exclusive_rank),
                    )
                    refresh["step"] = int(step - 1)
                    reprojection = enforce_output_geometry_(
                        ga_module.raw_delta,
                        gd_module.raw_delta,
                        sensitive_ids,
                        sensitive_bases,
                        protected_basis,
                    )
                    # Preserve input optimizer history; discard stale output geometry moments.
                    opt.state[ga_module.raw_delta].clear()
                    opt.state[gd_module.raw_delta].clear()
                    refresh["output_reprojection"] = reprojection
                    refresh["output_optimizer_moments_reset"] = True
                    refresh["input_optimizer_moments_preserved"] = True
                    geometry_log.write(json.dumps(refresh) + "\n")
                    geometry_log.flush()

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
                gd = core.gd_non_sensitive_kl(
                    logits, same_prompt_base_logits[fidx], tids
                )

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

                input_reg = input_module.delta.square().mean()
                ga_reg = ga_module.raw_delta.square().mean()
                gd_reg = gd_module.raw_delta.square().mean()
                ga_objective = (
                    float(a.ga_weight) * ga
                    + 0.5 * float(a.row_delta_weight) * (input_reg + ga_reg)
                )
                preserve_objective = (
                    float(a.gd_weight) * gd
                    + float(a.locality_kl_weight) * lkl
                    + float(a.locality_sensitive_logit_weight) * lrow
                    + float(a.utility_kl_weight) * ukl
                    + 0.5 * float(a.row_delta_weight) * (input_reg + gd_reg)
                )
                if not torch.isfinite(ga_objective) or not torch.isfinite(preserve_objective):
                    raise FloatingPointError(f"non-finite Directional SURE v2 objective at step {step}")

                ga_input_grad, ga_output_grad = torch.autograd.grad(
                    ga_objective,
                    [input_module.delta, ga_module.raw_delta],
                    retain_graph=True,
                    allow_unused=True,
                )
                preserve_input_grad, preserve_output_grad = torch.autograd.grad(
                    preserve_objective,
                    [input_module.delta, gd_module.raw_delta],
                    retain_graph=False,
                    allow_unused=True,
                )
                ga_input_grad = _zero_if_none(ga_input_grad, input_module.delta)
                preserve_input_grad = _zero_if_none(
                    preserve_input_grad, input_module.delta
                )
                ga_output_grad = _zero_if_none(
                    ga_output_grad, ga_module.raw_delta
                )
                preserve_output_grad = _zero_if_none(
                    preserve_output_grad, gd_module.raw_delta
                )

                ga_output_projected, ga_grad_diag = project_rows_tokenwise_span(
                    ga_output_grad,
                    sensitive_ids,
                    sensitive_bases,
                )
                preserve_output_projected, preserve_grad_diag = project_rows_global_span(
                    preserve_output_grad,
                    protected_basis,
                )

                input_module.delta.grad = (
                    ga_input_grad + preserve_input_grad
                ).to(input_module.delta.dtype)
                ga_module.raw_delta.grad = ga_output_projected.to(
                    ga_module.raw_delta.dtype
                )
                gd_module.raw_delta.grad = preserve_output_projected.to(
                    gd_module.raw_delta.dtype
                )

                params = [input_module.delta, ga_module.raw_delta, gd_module.raw_delta]
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(params, float(a.grad_clip))
                    if a.grad_clip > 0
                    else None
                )
                if grad_norm is not None and not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite grad norm at step {step}")
                opt.step()
                geometry_projection = enforce_output_geometry_(
                    ga_module.raw_delta,
                    gd_module.raw_delta,
                    sensitive_ids,
                    sensitive_bases,
                    protected_basis,
                )

                if step == 1 or step % int(a.check_every) == 0 or step == int(a.steps):
                    total_output_delta = (
                        ga_module.effective_delta() + gd_module.effective_delta()
                    )
                    row = {
                        "step": int(step),
                        "ga_sensitive_logprob": float(ga.detach().cpu()),
                        "gd_same_prompt_non_sensitive_kl": float(gd.detach().cpu()),
                        "relation_locality_kl": float(lkl.detach().cpu()),
                        "relation_sensitive_logit_mse": float(lrow.detach().cpu()),
                        "wikipedia_utility_kl": float(ukl.detach().cpu()),
                        "input_delta_mse": float(input_module.delta.square().mean().detach().cpu()),
                        "ga_output_delta_mse": float(ga_module.raw_delta.square().mean().detach().cpu()),
                        "gd_output_delta_mse": float(gd_module.raw_delta.square().mean().detach().cpu()),
                        "total_output_delta_mse": float(total_output_delta.square().mean().detach().cpu()),
                        "ga_input_gradient_norm": float(ga_input_grad.norm().detach().cpu()),
                        "preserve_input_gradient_norm": float(preserve_input_grad.norm().detach().cpu()),
                        "raw_ga_output_gradient_kept_sensitive_exclusive_norm": float(ga_grad_diag["kept_norm"]),
                        "raw_ga_output_gradient_removed_norm": float(ga_grad_diag["removed_norm"]),
                        "raw_preserve_output_gradient_kept_protected_norm": float(preserve_grad_diag["kept_norm"]),
                        "raw_preserve_output_gradient_removed_norm": float(preserve_grad_diag["removed_norm"]),
                        **geometry_projection,
                        "selected_sensitive_row_count": int(len(sensitive_ids)),
                        "benchmark_retain_seen": 0,
                        "official_neighborhood_seen": 0,
                        "official_paraphrase_seen": 0,
                        "PPL_seen": False,
                    }
                    log_f.write(json.dumps(row) + "\n")
                    log_f.flush()

    del opt

    # Final current-model geometry is authoritative for final output-row projection.
    protected_basis, sensitive_bases, final_geometry = refresh_geometry(
        model,
        tok,
        cases,
        all_tids,
        sensitive_ids,
        locality_prompts,
        utility_prompts,
        device,
        forget_batch_size=int(a.cache_batch_size),
        locality_batch_size=int(a.locality_cache_batch_size),
        utility_batch_size=int(a.utility_cache_batch_size),
        protected_rank=int(a.protected_basis_rank),
        sensitive_rank=int(a.sensitive_exclusive_rank),
    )
    final_geometry["step"] = int(a.steps)
    final_projection = enforce_output_geometry_(
        ga_module.raw_delta,
        gd_module.raw_delta,
        sensitive_ids,
        sensitive_bases,
        protected_basis,
    )
    final_geometry["output_reprojection"] = final_projection
    core.write_json(out_dir / "final_geometry_receipt.json", final_geometry)

    model.eval()
    benchmark_pre_materialize = _benchmark_pair_diagnostics(
        model, tok, benchmark_instances, device, llama_like,
        int(a.batch_size), float(a.benchmark_pair_margin)
    )
    locality_pre_materialize = projected.evaluate_locality_guards(
        model,
        tok,
        locality_prompts,
        locality_protected,
        base_local_logits,
        device,
        int(a.locality_cache_batch_size),
    )
    utility_pre_materialize = wikipedia_utility.evaluate_utility_kl(
        model,
        tok,
        utility_prompts,
        utility_base_logits,
        device,
        int(a.utility_cache_batch_size),
    )
    directional_after = directional_residual_diagnostics(
        ga_module.raw_delta.detach(),
        gd_module.raw_delta.detach(),
        sensitive_ids,
        sensitive_bases,
        protected_basis,
    )

    input_delta_final = input_module.delta.detach().float().cpu()
    ga_delta_final = ga_module.effective_delta().detach().float().cpu()
    gd_delta_final = gd_module.effective_delta().detach().float().cpu()
    total_output_delta = (ga_delta_final + gd_delta_final).float()

    input_module.remove()
    output_hook.remove()
    _materialize_selected_delta(
        input_layer.weight, sensitive_ids, input_delta_final
    )
    core.materialize_output_delta(
        output_layer, sensitive_ids, total_output_delta
    )
    model.eval()

    materialized_input = (
        input_layer.weight.index_select(0, ids_device).float().cpu()
        - selected_base_input
    )
    output_ids = ids_device.to(output_layer.weight.device)
    materialized_output = (
        output_layer.weight.index_select(0, output_ids).float().cpu()
        - selected_base_output
    )
    input_materialization_error = float(
        (materialized_input - input_delta_final).abs().max()
    )
    output_materialization_error = float(
        (materialized_output - total_output_delta).abs().max()
    )

    benchmark_after = _benchmark_pair_diagnostics(
        model, tok, benchmark_instances, device, llama_like,
        int(a.batch_size), float(a.benchmark_pair_margin)
    )
    locality_after = projected.evaluate_locality_guards(
        model,
        tok,
        locality_prompts,
        locality_protected,
        base_local_logits,
        device,
        int(a.locality_cache_batch_size),
    )
    utility_after = wikipedia_utility.evaluate_utility_kl(
        model,
        tok,
        utility_prompts,
        utility_base_logits,
        device,
        int(a.utility_cache_batch_size),
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
        "training_objective": (
            "sparse input rows receive GA+preservation gradients; output GA is constrained "
            "to protected-residualized sensitive span; output GD/locality/utility is constrained "
            "to observed donor+Wikipedia protected span"
        ),
        "target_new_used_as_training_target": False,
        "transformer_trainable": False,
        "input_embedding_sensitive_rows_trainable": True,
        "input_embedding_non_sensitive_rows_trainable": False,
        "lm_head_untied": True,
        "lm_head_sensitive_rows_trainable": True,
        "lm_head_non_sensitive_rows_trainable": False,
        "sparse_fp32_components": [
            "input_sensitive_delta",
            "output_sensitive_ga_delta",
            "output_sensitive_preservation_delta",
        ],
        "selected_sensitive_token_ids": [int(x) for x in sensitive_ids],
        "selected_sensitive_row_count": int(len(sensitive_ids)),
        "official_neighborhood_seen": 0,
        "official_paraphrase_seen": 0,
        "benchmark_retain_seen": 0,
        "PPL_seen": False,
        "basis_refresh_every": int(a.basis_refresh_every),
        "protected_basis_rank_cap": int(a.protected_basis_rank),
        "sensitive_exclusive_rank_cap": int(a.sensitive_exclusive_rank),
        "initial_geometry": geometry_receipt,
        "final_geometry": final_geometry,
        "directional_residual_after": directional_after,
        "weights": {
            "ga": float(a.ga_weight),
            "gd": float(a.gd_weight),
            "relation_locality_kl": float(a.locality_kl_weight),
            "relation_sensitive_logit": float(a.locality_sensitive_logit_weight),
            "wikipedia_utility_kl": float(a.utility_kl_weight),
            "row_delta": float(a.row_delta_weight),
        },
        "steps": int(a.steps),
        "input_row_lr": float(a.input_row_lr),
        "lm_row_lr": float(a.lm_row_lr),
        "optimizer": a.optimizer,
        "trainable_parameter_count": int(
            input_module.delta.numel()
            + ga_module.trainable_parameter_count
            + gd_module.trainable_parameter_count
        ),
        "benchmark_pair_before": benchmark_before,
        "benchmark_pair_pre_materialize": benchmark_pre_materialize,
        "benchmark_pair_after": benchmark_after,
        "relation_locality_pre_materialize": locality_pre_materialize,
        "relation_locality_after": locality_after,
        "wikipedia_utility_pre_materialize": utility_pre_materialize,
        "wikipedia_utility_after": utility_after,
        "input_delta_mse": float(input_delta_final.square().mean()),
        "ga_output_delta_mse": float(ga_delta_final.square().mean()),
        "gd_output_delta_mse": float(gd_delta_final.square().mean()),
        "input_materialization_max_abs_error": input_materialization_error,
        "output_materialization_max_abs_error": output_materialization_error,
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(
        out_dir / "sensitive_rows_directional_gagd_v2_summary.json", summary
    )

    print("Directional SURE v2 checkpoint:", ckpt)
    print("Transformer trainable: False")
    print("Input embedding sensitive rows trainable: True")
    print("LM head untied: True")
    print("LM-head sensitive rows trainable: True")
    print("Sensitive row count:", len(sensitive_ids))
    print("Initial geometry:", geometry_receipt["summary"])
    print("Final geometry:", final_geometry["summary"])
    print("Benchmark pair before:", benchmark_before)
    print("Benchmark pair pre-materialize:", benchmark_pre_materialize)
    print("Benchmark pair after:", benchmark_after)
    print("Relation-locality after:", locality_after)
    print("Wikipedia utility after:", utility_after)
    print("Directional residual after:", directional_after)
    print("Input materialization max abs error:", input_materialization_error)
    print("Output materialization max abs error:", output_materialization_error)
    print("Official MCF neighborhood/paraphrase probes were NOT used in training.")
    print("Run official evaluation only after this checkpoint is finalized.")


if __name__ == "__main__":
    main()
