#!/usr/bin/env python3
"""MCF SURE-H -> residual representation-LoRA cascade.

Stage 1 is an already-finalized SURE-H checkpoint: a model whose readout/LM head
contains the head-side unlearning intervention.  This script never trains or
modifies that head.  It detects the remaining training-visible direct MCF facts
whose target_true-vs-target_new margin is below the solver threshold and adds a
small LoRA intervention to transformer representation space only.

Default representation intervention for Llama-like models:

    final decoder layer .mlp.down_proj

so the cascade is

    h_0 --W_H--> logits                  (SURE-H parent)
      |
      +-- residual LoRA --> h_R --W_H--> logits  (H + R fallback)

Only residual direct failures receive GA.  Same-prompt GD is computed on all
training-visible sensitive cases, while relation-donor controls and an external
Wikipedia utility sample preserve Stage-1 behavior.  target_new is never a
replacement training target; it is used only for the direct margin gate,
stopping criterion, and final LoRA scale selection.

No official MCF paraphrase, neighborhood, retain-1000, generation, or PPL
example is visible to training/checkpoint selection.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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


METHOD = "SURE-H-then-residual-representation-LoRA"
PROTOCOL = "mcf_target_true_sure_h_then_residual_lora_v1"


def _parse_int_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("at least one LoRA layer must be specified")
    if len(set(values)) != len(values):
        raise ValueError("LoRA layer list contains duplicates")
    return values


def _parse_scales(text: str) -> List[float]:
    values = sorted({float(x.strip()) for x in str(text).split(",") if x.strip()})
    if not values:
        raise ValueError("candidate LoRA scales cannot be empty")
    if any((not math.isfinite(x)) or x < 0.0 or x > 1.0 for x in values):
        raise ValueError("candidate LoRA scales must be finite values in [0,1]")
    if 1.0 not in values:
        raise ValueError("candidate LoRA scales must include 1")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sure-h-model-path", required=True,
                   help="Finalized head-side SURE-H checkpoint")
    p.add_argument("--base-model-path", required=True,
                   help="Original Base checkpoint; recorded for restore-head diagnostic")
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)

    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--optimizer", choices=("adam", "adamw", "sgd"), default="adamw")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--gd-weight", type=float, default=1.0)
    p.add_argument("--lora-l2-weight", type=float, default=1e-4)

    p.add_argument("--lora-layers", default="-1",
                   help="Comma-separated decoder layer indices; negative indices allowed")
    p.add_argument("--lora-target", choices=("mlp.down_proj", "mlp.up_proj", "self_attn.o_proj"),
                   default="mlp.down_proj")
    p.add_argument("--lora-rank", type=int, default=1)
    p.add_argument("--lora-alpha", type=float, default=1.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)

    p.add_argument("--solver-margin", type=float, default=0.25)
    p.add_argument("--acceptance-margin", type=float, default=0.05)
    p.add_argument(
        "--candidate-lora-scales",
        default="0,0.03125,0.0625,0.09375,0.125,0.1875,0.25,0.375,0.5,0.625,0.75,0.875,1",
    )

    p.add_argument("--subject-control-count", type=int, default=4)
    p.add_argument("--locality-batch-size", type=int, default=4)
    p.add_argument("--locality-cache-batch-size", type=int, default=8)
    p.add_argument("--locality-kl-weight", type=float, default=2.0)
    p.add_argument("--locality-sensitive-logit-weight", type=float, default=5.0)

    p.add_argument("--utility-wikipedia-dir", required=True)
    p.add_argument("--utility-sample-size", type=int, default=200)
    p.add_argument("--utility-batch-size", type=int, default=4)
    p.add_argument("--utility-cache-batch-size", type=int, default=8)
    p.add_argument("--utility-max-length", type=int, default=128)
    p.add_argument("--utility-seed", type=int, default=1)
    p.add_argument("--utility-exclude-first", type=int, default=20)
    p.add_argument("--utility-kl-weight", type=float, default=2.0)

    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    a = p.parse_args(list(argv) if argv is not None else None)

    if a.forget_num <= 0 or a.steps <= 0 or a.batch_size <= 0 or a.cache_batch_size <= 0:
        p.error("forget-num, steps, and batch sizes must be positive")
    if a.check_every <= 0 or a.lr <= 0 or a.lora_rank <= 0 or a.lora_alpha <= 0:
        p.error("check cadence, LR, rank, and alpha must be positive")
    if not 0.0 <= a.lora_dropout < 1.0:
        p.error("LoRA dropout must be in [0,1)")
    nonnegative = (
        a.grad_clip, a.gd_weight, a.lora_l2_weight, a.acceptance_margin,
        a.solver_margin, a.locality_kl_weight, a.locality_sensitive_logit_weight,
        a.utility_kl_weight, a.utility_exclude_first,
    )
    if any(float(x) < 0 for x in nonnegative):
        p.error("clip/weights/margins/exclusion must be non-negative")
    if a.ga_weight <= 0:
        p.error("ga-weight must be positive")
    if a.solver_margin < a.acceptance_margin:
        p.error("solver-margin must be >= acceptance-margin")
    if a.utility_exclude_first < 20:
        p.error("utility-exclude-first must be >=20 to protect the fixed PPL prefix")
    try:
        a.lora_layers = _parse_int_list(a.lora_layers)
        a.candidate_lora_scales = _parse_scales(a.candidate_lora_scales)
    except ValueError as exc:
        p.error(str(exc))
    return a


class ResidualLoRALinear(nn.Module):
    """Frozen Linear + trainable low-rank residual with runtime scale control."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("ResidualLoRALinear requires nn.Linear base")
        if rank <= 0 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        device = base.weight.device
        self.lora_A = nn.Parameter(
            torch.empty((self.rank, base.in_features), dtype=torch.float32, device=device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros((base.out_features, self.rank), dtype=torch.float32, device=device)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.multiplier = 1.0

    def effective_delta(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling * float(self.multiplier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        z = F.linear(self.dropout(x).float(), self.lora_A)
        z = F.linear(z, self.lora_B) * self.scaling * float(self.multiplier)
        return base_out + z.to(dtype=base_out.dtype)

    @property
    def trainable_parameter_count(self) -> int:
        return int(self.lora_A.numel() + self.lora_B.numel())



def _decoder_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("SURE-H residual LoRA currently expects a Llama-like model.model.layers stack")


def _resolve_layer_index(index: int, count: int) -> int:
    resolved = int(index) if int(index) >= 0 else count + int(index)
    if resolved < 0 or resolved >= count:
        raise IndexError(f"LoRA layer index {index} resolves outside 0..{count-1}")
    return resolved


def _get_target(layer: nn.Module, target: str) -> Tuple[nn.Module, str, nn.Linear]:
    if target == "mlp.down_proj":
        parent, attr = layer.mlp, "down_proj"
    elif target == "mlp.up_proj":
        parent, attr = layer.mlp, "up_proj"
    elif target == "self_attn.o_proj":
        parent, attr = layer.self_attn, "o_proj"
    else:
        raise ValueError(f"unsupported LoRA target {target}")
    base = getattr(parent, attr)
    if not isinstance(base, nn.Linear):
        raise TypeError(f"target {target} is not nn.Linear: {type(base)!r}")
    return parent, attr, base


def install_lora(
    model: nn.Module,
    requested_layers: Sequence[int],
    target: str,
    rank: int,
    alpha: float,
    dropout: float,
) -> List[Tuple[int, nn.Module, str, ResidualLoRALinear]]:
    layers = _decoder_layers(model)
    result: List[Tuple[int, nn.Module, str, ResidualLoRALinear]] = []
    seen = set()
    for requested in requested_layers:
        idx = _resolve_layer_index(int(requested), len(layers))
        if idx in seen:
            raise ValueError(f"duplicate resolved LoRA layer {idx}")
        seen.add(idx)
        parent, attr, base = _get_target(layers[idx], target)
        wrapped = ResidualLoRALinear(base, rank, alpha, dropout)
        setattr(parent, attr, wrapped)
        result.append((idx, parent, attr, wrapped))
    return result


def set_lora_scale(installed, scale: float) -> None:
    for _idx, _parent, _attr, module in installed:
        module.multiplier = float(scale)


@torch.no_grad()
def merge_lora_(installed) -> List[Dict[str, Any]]:
    receipts: List[Dict[str, Any]] = []
    for idx, parent, attr, module in installed:
        delta = module.effective_delta().to(
            device=module.base.weight.device, dtype=module.base.weight.dtype
        )
        module.base.weight.add_(delta)
        receipts.append({
            "layer": int(idx),
            "target": str(attr),
            "rank": int(module.rank),
            "alpha": float(module.alpha),
            "multiplier": float(module.multiplier),
            "delta_norm": float(delta.float().norm().cpu()),
        })
        setattr(parent, attr, module.base)
    return receipts


def _optimizer(params: Iterable[nn.Parameter], kind: str, lr: float):
    params = list(params)
    if kind == "sgd":
        return torch.optim.SGD(params, lr=float(lr))
    if kind == "adam":
        return torch.optim.Adam(params, lr=float(lr))
    return torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)


def _direct_margins(model, tok, instances, device, llama_like, batch_size) -> torch.Tensor:
    return stage2.mcf_direct_margins(
        model, tok, instances, device, llama_like, int(batch_size),
        "target_true", "target_new"
    ).detach().float().cpu()


def _margin_report(margins: torch.Tensor, threshold: float) -> Dict[str, Any]:
    m = margins.detach().float().cpu()
    return {
        "failures": int((m < float(threshold)).sum().item()),
        "minimum_margin": float(m.min()),
        "mean_margin": float(m.mean()),
        "maximum_margin": float(m.max()),
        "threshold": float(threshold),
    }


def _active_case_indices(cases, margins: torch.Tensor, solver_margin: float):
    active_records = [
        int(i) for i, value in enumerate(margins.tolist())
        if float(value) < float(solver_margin)
    ]
    active = set(active_records)
    active_cases = [
        i for i, case in enumerate(cases)
        if int(case.record_position) in active
    ]
    return active_records, active_cases


def _choose_scale(
    model,
    tok,
    instances,
    installed,
    scales: Sequence[float],
    acceptance_margin: float,
    device,
    llama_like: bool,
    batch_size: int,
):
    reports: List[Dict[str, Any]] = []
    for scale in scales:
        set_lora_scale(installed, float(scale))
        margins = _direct_margins(model, tok, instances, device, llama_like, batch_size)
        report = _margin_report(margins, acceptance_margin)
        report["scale"] = float(scale)
        reports.append(report)
    zero = [r for r in reports if int(r["failures"]) == 0]
    if zero:
        chosen = min(zero, key=lambda r: float(r["scale"]))
    else:
        # Prefer fewer failures; among ties prefer the stronger worst-case and mean
        # margins.  This deliberately avoids selecting scale 0 merely because it
        # ties a non-zero scale on failure count.
        chosen = min(
            reports,
            key=lambda r: (
                int(r["failures"]),
                -float(r["minimum_margin"]),
                -float(r["mean_margin"]),
                float(r["scale"]),
            ),
        )
    set_lora_scale(installed, float(chosen["scale"]))
    return float(chosen["scale"]), reports


def _lora_delta_l2(installed) -> torch.Tensor:
    values = [module.effective_delta().square().mean() for *_x, module in installed]
    return torch.stack(values).mean() if values else torch.tensor(0.0)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
        model_path=a.sure_h_model_path,
        dtype=a.dtype,
        device_map=a.device_map,
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
    benchmark_instances = stage2.mcf_instances(records)
    parent_margins = _direct_margins(
        model, tok, benchmark_instances, device, llama_like, int(a.cache_batch_size)
    )
    parent_solver = _margin_report(parent_margins, float(a.solver_margin))
    parent_acceptance = _margin_report(parent_margins, float(a.acceptance_margin))

    # Cache Stage-1 behavior BEFORE inserting LoRA.  These are the only utility
    # references used by the residual representation repair.
    same_prompt_parent_logits = core.cache_base_logits(
        model, tok, cases, device, batch_size=int(a.cache_batch_size)
    )
    locality_prompts, locality_protected, locality_receipt = (
        projected.build_relation_locality_controls(
            records, tok, benchmark_instances, int(a.subject_control_count)
        )
    )
    print(f"Caching SURE-H locality references for {len(locality_prompts)} prompts...", flush=True)
    _lh, locality_parent_logits = projected.cache_relation_locality_reference(
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
    print(f"Caching SURE-H Wikipedia references for {len(utility_prompts)} prompts...", flush=True)
    utility_parent_logits = wikipedia_utility.cache_utility_base_logits(
        model, tok, utility_prompts, device, int(a.utility_cache_batch_size)
    )

    # Freeze SURE-H completely, including the modified head.
    for p in model.parameters():
        p.requires_grad_(False)
    output_head = model.get_output_embeddings()
    if output_head is None or not hasattr(output_head, "weight"):
        raise RuntimeError("SURE-H model has no output embedding/LM-head weight")
    sensitive_ids = sorted({
        int(x) for x in core.official_target_ids(
            tok, cases, llama_like=llama_like, device=device
        ).detach().cpu().tolist()
        if int(x) not in gagd.special_token_ids(tok)
    })
    selected_head_before = output_head.weight.detach().index_select(
        0, torch.tensor(sensitive_ids, dtype=torch.long, device=output_head.weight.device)
    ).float().cpu().clone()

    installed = install_lora(
        model, a.lora_layers, a.lora_target,
        int(a.lora_rank), float(a.lora_alpha), float(a.lora_dropout)
    )
    lora_params: List[nn.Parameter] = []
    for _idx, _parent, _attr, module in installed:
        lora_params.extend([module.lora_A, module.lora_B])
    trainable = [p for p in model.parameters() if p.requires_grad]
    if set(map(id, trainable)) != set(map(id, lora_params)):
        raise RuntimeError("non-LoRA model parameters unexpectedly trainable")
    opt = _optimizer(lora_params, a.optimizer, float(a.lr))

    active_records, active_case_ids = _active_case_indices(
        cases, parent_margins, float(a.solver_margin)
    )
    gd_sampler = core.IndexSampler(len(cases), int(a.batch_size), int(a.seed) + 13001)
    locality_sampler = core.IndexSampler(
        len(locality_prompts), int(a.locality_batch_size), int(a.seed) + 13003
    )
    utility_sampler = core.IndexSampler(
        len(utility_prompts), int(a.utility_batch_size), int(a.utility_seed) + 13005
    )

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "relation_locality_receipt.json", locality_receipt)
    _write_json(out_dir / "wikipedia_utility_receipt.json", utility_receipt)
    direct_log = (out_dir / "direct_gate_log.jsonl").open("w", encoding="utf-8")
    train_log = (out_dir / "train_log.jsonl").open("w", encoding="utf-8")

    direct_log.write(json.dumps({
        "step": 0,
        "active_record_count": len(active_records),
        "active_records": active_records,
        "solver": parent_solver,
        "acceptance": parent_acceptance,
    }) + "\n")
    direct_log.flush()

    stopped_step = 0
    margins = parent_margins.clone()
    if not active_case_ids:
        print("SURE-H already satisfies every solver margin; residual LoRA is a no-op.", flush=True)
    else:
        active_sampler = core.IndexSampler(
            len(active_case_ids), int(a.batch_size), int(a.seed) + 13007
        )
        for step in tqdm(range(1, int(a.steps) + 1), desc="SURE-H residual LoRA"):
            stopped_step = int(step)
            # Rebuild the active sampler after each full-margin check so solved
            # records leave GA and regressed records can re-enter.
            f_local = active_sampler.next()
            fidx = [active_case_ids[i] for i in f_local]
            forget_batch = [cases[i] for i in fidx]
            gidx = gd_sampler.next()
            gd_batch = [cases[i] for i in gidx]
            lidx = locality_sampler.next()
            locality_batch = [locality_prompts[i] for i in lidx]
            locality_ids = [locality_protected[i] for i in lidx]
            uidx = utility_sampler.next()
            utility_batch = [utility_prompts[i] for i in uidx]

            opt.zero_grad(set_to_none=True)
            ga_logits = core.forward_last_logits(model, tok, forget_batch, device)
            ga_tids = core.official_target_ids(
                tok, forget_batch, llama_like=llama_like, device=device
            )
            ga = core.ga_sensitive_logprob(ga_logits, ga_tids)

            gd_logits = core.forward_last_logits(model, tok, gd_batch, device)
            gd_tids = core.official_target_ids(
                tok, gd_batch, llama_like=llama_like, device=device
            )
            gd = core.gd_non_sensitive_kl(
                gd_logits, same_prompt_parent_logits[gidx], gd_tids
            )

            _cur_h, local_logits = projected._prompt_hidden_and_logits(
                model, tok, locality_batch, device
            )
            local_base = locality_parent_logits[lidx]
            lkl = projected.locality_kl(local_logits, local_base)
            lrow = projected.protected_sensitive_logit_mse(
                local_logits, local_base, locality_ids
            )
            utility_logits = wikipedia_utility._forward_prompt_logits(
                model, tok, utility_batch, device
            )
            ukl = wikipedia_utility.utility_kl(
                utility_logits, utility_parent_logits[uidx]
            )
            lora_l2 = _lora_delta_l2(installed).to(device)
            loss = (
                float(a.ga_weight) * ga
                + float(a.gd_weight) * gd
                + float(a.locality_kl_weight) * lkl
                + float(a.locality_sensitive_logit_weight) * lrow
                + float(a.utility_kl_weight) * ukl
                + float(a.lora_l2_weight) * lora_l2
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite residual LoRA loss at step {step}")
            loss.backward()
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(lora_params, float(a.grad_clip))
                if a.grad_clip > 0 else torch.tensor(0.0, device=device)
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite residual LoRA grad norm at step {step}")
            opt.step()

            train_log.write(json.dumps({
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "ga": float(ga.detach().cpu()),
                "gd": float(gd.detach().cpu()),
                "locality_kl": float(lkl.detach().cpu()),
                "locality_sensitive_logit": float(lrow.detach().cpu()),
                "wikipedia_kl": float(ukl.detach().cpu()),
                "lora_l2": float(lora_l2.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "active_record_count": int(len(active_records)),
            }) + "\n")
            train_log.flush()

            if step % int(a.check_every) == 0 or step == int(a.steps):
                margins = _direct_margins(
                    model, tok, benchmark_instances, device, llama_like,
                    int(a.cache_batch_size)
                )
                active_records, active_case_ids = _active_case_indices(
                    cases, margins, float(a.solver_margin)
                )
                direct_log.write(json.dumps({
                    "step": int(step),
                    "active_record_count": len(active_records),
                    "active_records": active_records,
                    "solver": _margin_report(margins, float(a.solver_margin)),
                    "acceptance": _margin_report(margins, float(a.acceptance_margin)),
                }) + "\n")
                direct_log.flush()
                if not active_case_ids:
                    break
                active_sampler = core.IndexSampler(
                    len(active_case_ids), int(a.batch_size), int(a.seed) + 13007 + int(step)
                )

    direct_log.close()
    train_log.close()

    chosen_scale, scale_reports = _choose_scale(
        model, tok, benchmark_instances, installed,
        a.candidate_lora_scales, float(a.acceptance_margin),
        device, llama_like, int(a.cache_batch_size)
    )
    _write_json(out_dir / "lora_scale_reports.json", scale_reports)

    premerge_margins = _direct_margins(
        model, tok, benchmark_instances, device, llama_like, int(a.cache_batch_size)
    )
    locality_premerge = projected.evaluate_locality_guards(
        model, tok, locality_prompts, locality_protected, locality_parent_logits,
        device, int(a.locality_cache_batch_size)
    )
    utility_premerge = wikipedia_utility.evaluate_utility_kl(
        model, tok, utility_prompts, utility_parent_logits,
        device, int(a.utility_cache_batch_size)
    )

    # Verify the SURE-H head was not touched by Stage 2 before merging LoRA.
    selected_head_premerge = output_head.weight.detach().index_select(
        0, torch.tensor(sensitive_ids, dtype=torch.long, device=output_head.weight.device)
    ).float().cpu()
    head_selected_max_abs_change = float(
        (selected_head_premerge - selected_head_before).abs().max()
    )
    if head_selected_max_abs_change != 0.0:
        raise RuntimeError("SURE-H sensitive LM-head rows changed during residual LoRA training")

    # Save factors before merging for mechanistic analysis.
    factor_payload = {
        f"layer_{idx}_{a.lora_target.replace('.', '_')}": {
            "lora_A": module.lora_A.detach().float().cpu(),
            "lora_B": module.lora_B.detach().float().cpu(),
            "rank": module.rank,
            "alpha": module.alpha,
            "chosen_scale": chosen_scale,
        }
        for idx, _parent, _attr, module in installed
    }
    torch.save(factor_payload, out_dir / "lora_factors.pt")

    merge_receipt = merge_lora_(installed)
    model.eval()
    postmerge_margins = _direct_margins(
        model, tok, benchmark_instances, device, llama_like, int(a.cache_batch_size)
    )
    locality_after = projected.evaluate_locality_guards(
        model, tok, locality_prompts, locality_protected, locality_parent_logits,
        device, int(a.locality_cache_batch_size)
    )
    utility_after = wikipedia_utility.evaluate_utility_kl(
        model, tok, utility_prompts, utility_parent_logits,
        device, int(a.utility_cache_batch_size)
    )

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    summary = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "sure_h_model_path": str(Path(a.sure_h_model_path).resolve()),
        "base_model_path": str(Path(a.base_model_path).resolve()),
        "training_visible_path": str(visible_path),
        "split_manifest": str(manifest_path),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "target_contract": {
            "sensitive_unwanted": "requested_rewrite.target_true",
            "benchmark_reference": "requested_rewrite.target_new",
            "target_new_is_replacement_training_target": False,
        },
        "stage1": "SURE-H finalized parent",
        "stage2": "residual representation LoRA on direct failures only",
        "sure_h_head_frozen_during_stage2": True,
        "selected_sensitive_head_rows_checked": len(sensitive_ids),
        "selected_sensitive_head_max_abs_change_during_stage2": head_selected_max_abs_change,
        "transformer_base_parameters_trainable": 0,
        "lora_target": a.lora_target,
        "lora_layers_requested": [int(x) for x in a.lora_layers],
        "lora_layers_resolved": [int(x[0]) for x in installed],
        "lora_rank": int(a.lora_rank),
        "lora_alpha": float(a.lora_alpha),
        "lora_dropout": float(a.lora_dropout),
        "lora_trainable_parameters": int(sum(p.numel() for p in lora_params)),
        "chosen_lora_scale": float(chosen_scale),
        "scale_reports": scale_reports,
        "solver_margin": float(a.solver_margin),
        "acceptance_margin": float(a.acceptance_margin),
        "parent_solver": parent_solver,
        "parent_acceptance": parent_acceptance,
        "premerge_acceptance": _margin_report(premerge_margins, float(a.acceptance_margin)),
        "postmerge_acceptance": _margin_report(postmerge_margins, float(a.acceptance_margin)),
        "stopped_step": int(stopped_step),
        "weights": {
            "ga": float(a.ga_weight),
            "gd": float(a.gd_weight),
            "relation_locality_kl": float(a.locality_kl_weight),
            "relation_sensitive_logit": float(a.locality_sensitive_logit_weight),
            "wikipedia_utility_kl": float(a.utility_kl_weight),
            "lora_l2": float(a.lora_l2_weight),
        },
        "relation_locality_premerge": locality_premerge,
        "relation_locality_after": locality_after,
        "wikipedia_utility_premerge": utility_premerge,
        "wikipedia_utility_after": utility_after,
        "merge_receipt": merge_receipt,
        "official_paraphrase_seen": 0,
        "official_neighborhood_seen": 0,
        "benchmark_retain_seen": 0,
        "PPL_seen": False,
        "checkpoint": str(ckpt.resolve()),
        "restore_head_diagnostic_command": (
            "python scripts/mcf_restore_base_head_diagnostic.py "
            f"--model-path {ckpt.resolve()} --base-model-path {Path(a.base_model_path).resolve()} "
            f"--output-dir {(out_dir / 'checkpoint_restore_base_head').resolve()}"
        ),
    }
    _write_json(out_dir / "sure_h_then_residual_lora_summary.json", summary)

    print("SURE-H -> residual LoRA checkpoint:", ckpt)
    print("SURE-H head frozen during Stage 2: True")
    print("LoRA target:", a.lora_target)
    print("LoRA resolved layers:", [int(x[0]) for x in installed])
    print("LoRA rank / alpha:", a.lora_rank, "/", a.lora_alpha)
    print("Stage-1 acceptance:", parent_acceptance)
    print("Chosen LoRA scale:", chosen_scale)
    print("Final acceptance:", summary["postmerge_acceptance"])
    print("Relation-locality after:", locality_after)
    print("Wikipedia utility after:", utility_after)
    print("SURE-H selected-head max abs change during Stage 2:", head_selected_max_abs_change)
    print("Official MCF neighborhood/paraphrase/retain/PPL data were NOT used.")


if __name__ == "__main__":
    main()
