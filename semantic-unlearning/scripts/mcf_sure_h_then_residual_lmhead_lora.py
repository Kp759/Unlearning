#!/usr/bin/env python3
"""MCF SURE-H -> residual sparse LM-head LoRA cascade.

Stage 1 is an already-finalized SURE-H checkpoint.  All Stage-1 parameters are
frozen.  If training-visible direct MCF facts still fail the target_true-vs-
target_new margin, Stage 2 adds a low-rank residual adapter ONLY to selected
sensitive LM-head vocabulary rows:

    W_final[selected] = W_H[selected] + (B @ A) * (alpha / rank)

The underlying SURE-H head W_H remains frozen while the adapter is trained.
Only residual direct failures receive GA; same-prompt GD is computed across all
training-visible sensitive cases, with relation-donor controls and an external
Wikipedia sample preserving Stage-1 behavior.  target_new is never a supervised
replacement target; it is used only for direct margin gating, early stopping,
and final adapter-scale selection.

No official MCF paraphrase, neighborhood, retain-1000, generation, or PPL
example is visible to training/checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

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


METHOD = "SURE-H-then-residual-sparse-LM-head-LoRA"
PROTOCOL = "mcf_target_true_sure_h_then_residual_lmhead_lora_v1"


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
                   help="Finalized SURE-H checkpoint; all of it remains frozen")
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)

    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--optimizer", choices=("adam", "adamw", "sgd"), default="adamw")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--gd-weight", type=float, default=1.0)
    p.add_argument("--lora-l2-weight", type=float, default=1e-4)

    p.add_argument("--lora-rank", type=int, default=1)
    p.add_argument("--lora-alpha", type=float, default=1.0)
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

    positive = (
        a.forget_num, a.steps, a.batch_size, a.cache_batch_size, a.check_every,
        a.lr, a.ga_weight, a.lora_rank, a.lora_alpha,
        a.locality_batch_size, a.locality_cache_batch_size,
        a.utility_sample_size, a.utility_batch_size, a.utility_cache_batch_size,
        a.utility_max_length,
    )
    if any(float(x) <= 0 for x in positive):
        p.error("counts, cadence, LR, GA, LoRA rank/alpha, and batch sizes must be positive")
    nonnegative = (
        a.grad_clip, a.gd_weight, a.lora_l2_weight,
        a.solver_margin, a.acceptance_margin,
        a.locality_kl_weight, a.locality_sensitive_logit_weight,
        a.utility_kl_weight, a.utility_exclude_first,
    )
    if any(float(x) < 0 for x in nonnegative):
        p.error("clip/weights/margins/exclusion must be non-negative")
    if a.solver_margin < a.acceptance_margin:
        p.error("solver-margin must be >= acceptance-margin")
    if a.utility_exclude_first < 20:
        p.error("utility-exclude-first must be >=20 to protect the fixed PPL prefix")
    try:
        a.candidate_lora_scales = _parse_scales(a.candidate_lora_scales)
    except ValueError as exc:
        p.error(str(exc))
    return a


class SparseLMHeadLoRA(nn.Module):
    """Low-rank FP32 delta restricted to selected LM-head rows."""

    def __init__(
        self,
        row_ids: Sequence[int],
        hidden_size: int,
        rank: int,
        alpha: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        ids = sorted({int(x) for x in row_ids})
        if not ids:
            raise ValueError("SparseLMHeadLoRA requires selected rows")
        if hidden_size <= 0 or rank <= 0 or alpha <= 0:
            raise ValueError("hidden size, rank, and alpha must be positive")
        self.row_ids = tuple(ids)
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.multiplier = 1.0
        self.lora_A = nn.Parameter(
            torch.empty((self.rank, self.hidden_size), dtype=torch.float32, device=device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros((len(ids), self.rank), dtype=torch.float32, device=device)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def effective_delta(self) -> torch.Tensor:
        return (
            (self.lora_B @ self.lora_A)
            * self.scaling
            * float(self.multiplier)
        )

    @property
    def trainable_parameter_count(self) -> int:
        return int(self.lora_A.numel() + self.lora_B.numel())



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
    active_set = set(active_records)
    active_cases = [
        i for i, case in enumerate(cases)
        if int(case.record_position) in active_set
    ]
    return active_records, active_cases


def selected_rows_for_active_cases(
    cases,
    active_case_ids: Sequence[int],
    all_target_ids: torch.Tensor,
    special_ids: set[int],
) -> List[int]:
    if len(cases) != int(all_target_ids.numel()):
        raise ValueError("case/target-id count mismatch")
    result = sorted({
        int(all_target_ids[i].item())
        for i in active_case_ids
        if int(all_target_ids[i].item()) not in special_ids
    })
    if active_case_ids and not result:
        raise RuntimeError("active failures produced no non-special sensitive LM-head rows")
    return result


@torch.no_grad()
def _materialize_selected_delta(
    weight: torch.Tensor,
    row_ids: Sequence[int],
    delta: torch.Tensor,
) -> None:
    ids = torch.tensor(list(row_ids), dtype=torch.long, device=weight.device)
    current = weight.index_select(0, ids)
    weight.index_copy_(
        0, ids, current + delta.to(device=weight.device, dtype=weight.dtype)
    )


def _choose_scale(
    model,
    tok,
    instances,
    delta_module: SparseLMHeadLoRA,
    scales: Sequence[float],
    acceptance_margin: float,
    device,
    llama_like: bool,
    batch_size: int,
):
    reports: List[Dict[str, Any]] = []
    for scale in scales:
        delta_module.multiplier = float(scale)
        margins = _direct_margins(model, tok, instances, device, llama_like, batch_size)
        report = _margin_report(margins, acceptance_margin)
        report["scale"] = float(scale)
        reports.append(report)
    zero = [r for r in reports if int(r["failures"]) == 0]
    if zero:
        chosen = min(zero, key=lambda r: float(r["scale"]))
    else:
        # If perfect acceptance is impossible, choose the actual best direct
        # state, not scale zero just because it ties on failure count.
        chosen = min(
            reports,
            key=lambda r: (
                int(r["failures"]),
                -float(r["minimum_margin"]),
                -float(r["mean_margin"]),
                float(r["scale"]),
            ),
        )
    delta_module.multiplier = float(chosen["scale"])
    return float(chosen["scale"]), reports


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
    if not cases:
        raise RuntimeError("no target_true PredictionCases")
    all_tids = core.official_target_ids(
        tok, cases, llama_like=llama_like, device=device
    ).detach().cpu()
    benchmark_instances = stage2.mcf_instances(records)
    parent_margins = _direct_margins(
        model, tok, benchmark_instances, device, llama_like, int(a.cache_batch_size)
    )
    parent_solver = _margin_report(parent_margins, float(a.solver_margin))
    parent_acceptance = _margin_report(parent_margins, float(a.acceptance_margin))
    active_records, active_case_ids = _active_case_indices(
        cases, parent_margins, float(a.solver_margin)
    )
    selected_ids = selected_rows_for_active_cases(
        cases, active_case_ids, all_tids, gagd.special_token_ids(tok)
    )

    # Cache SURE-H behavior before installing the adapter.
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

    # Freeze the entire SURE-H parent.  LoRA is carried only by an output hook.
    for p in model.parameters():
        p.requires_grad_(False)
    output_layer = model.get_output_embeddings()
    if output_layer is None or not hasattr(output_layer, "weight"):
        raise RuntimeError("model has no LM-head weight")
    parent_head = output_layer.weight.detach().clone()

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "relation_locality_receipt.json", locality_receipt)
    _write_json(out_dir / "wikipedia_utility_receipt.json", utility_receipt)

    # If SURE-H already satisfies the solver, save an exact no-op copy.
    if not active_case_ids:
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        summary = {
            "schema_version": 1,
            "method": METHOD,
            "protocol": PROTOCOL,
            "sure_h_model_path": str(Path(a.sure_h_model_path).resolve()),
            "stage2_noop": True,
            "parent_solver": parent_solver,
            "parent_acceptance": parent_acceptance,
            "selected_lm_head_rows": 0,
            "chosen_lora_scale": 0.0,
            "official_paraphrase_seen": 0,
            "official_neighborhood_seen": 0,
            "benchmark_retain_seen": 0,
            "PPL_seen": False,
            "checkpoint": str(ckpt.resolve()),
        }
        _write_json(out_dir / "sure_h_then_residual_lmhead_lora_summary.json", summary)
        print("SURE-H already satisfies all solver margins; Stage 2 is a no-op.")
        return

    delta_module = SparseLMHeadLoRA(
        selected_ids,
        int(output_layer.weight.shape[1]),
        int(a.lora_rank),
        float(a.lora_alpha),
        output_layer.weight.device,
    )
    output_hook = core.register_output_delta_hook(
        output_layer, selected_ids, delta_module.effective_delta
    )
    lora_params = [delta_module.lora_A, delta_module.lora_B]
    opt = _optimizer(lora_params, a.optimizer, float(a.lr))

    gd_sampler = core.IndexSampler(len(cases), int(a.batch_size), int(a.seed) + 14001)
    locality_sampler = core.IndexSampler(
        len(locality_prompts), int(a.locality_batch_size), int(a.seed) + 14003
    )
    utility_sampler = core.IndexSampler(
        len(utility_prompts), int(a.utility_batch_size), int(a.utility_seed) + 14005
    )
    active_sampler = core.IndexSampler(
        len(active_case_ids), int(a.batch_size), int(a.seed) + 14007
    )

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
    for step in tqdm(range(1, int(a.steps) + 1), desc="SURE-H residual LM-head LoRA"):
        stopped_step = int(step)
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
        lora_l2 = delta_module.effective_delta().square().mean()
        loss = (
            float(a.ga_weight) * ga
            + float(a.gd_weight) * gd
            + float(a.locality_kl_weight) * lkl
            + float(a.locality_sensitive_logit_weight) * lrow
            + float(a.utility_kl_weight) * ukl
            + float(a.lora_l2_weight) * lora_l2
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite Stage-2 loss at step {step}")
        loss.backward()
        grad_norm = (
            torch.nn.utils.clip_grad_norm_(lora_params, float(a.grad_clip))
            if a.grad_clip > 0 else torch.tensor(0.0, device=device)
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite Stage-2 grad norm at step {step}")
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
            "active_record_count": len(active_records),
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
                len(active_case_ids), int(a.batch_size), int(a.seed) + 14007 + int(step)
            )

    direct_log.close()
    train_log.close()

    chosen_scale, scale_reports = _choose_scale(
        model, tok, benchmark_instances, delta_module,
        a.candidate_lora_scales, float(a.acceptance_margin),
        device, llama_like, int(a.cache_batch_size)
    )
    _write_json(out_dir / "lora_scale_reports.json", scale_reports)

    pre_materialize_margins = _direct_margins(
        model, tok, benchmark_instances, device, llama_like, int(a.cache_batch_size)
    )
    locality_pre_materialize = projected.evaluate_locality_guards(
        model, tok, locality_prompts, locality_protected, locality_parent_logits,
        device, int(a.locality_cache_batch_size)
    )
    utility_pre_materialize = wikipedia_utility.evaluate_utility_kl(
        model, tok, utility_prompts, utility_parent_logits,
        device, int(a.utility_cache_batch_size)
    )

    # The SURE-H parent head itself must still be bit-identical before merge.
    parent_head_max_abs_change = float(
        (output_layer.weight.detach() - parent_head).abs().max().cpu()
    )
    if parent_head_max_abs_change != 0.0:
        raise RuntimeError("underlying SURE-H LM head changed before LoRA materialization")

    lora_delta_final = delta_module.effective_delta().detach().float().cpu()
    torch.save({
        "selected_lm_head_token_ids": selected_ids,
        "lora_A": delta_module.lora_A.detach().float().cpu(),
        "lora_B": delta_module.lora_B.detach().float().cpu(),
        "rank": int(delta_module.rank),
        "alpha": float(delta_module.alpha),
        "chosen_scale": float(chosen_scale),
        "effective_delta": lora_delta_final,
    }, out_dir / "lmhead_lora_factors.pt")

    output_hook.remove()
    _materialize_selected_delta(output_layer.weight, selected_ids, lora_delta_final)
    model.eval()

    post_materialize_margins = _direct_margins(
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

    # Check materialization only on adapter-selected rows.
    ids_tensor = torch.tensor(selected_ids, dtype=torch.long, device=output_layer.weight.device)
    materialized = (
        output_layer.weight.detach().index_select(0, ids_tensor).float().cpu()
        - parent_head.index_select(0, ids_tensor.to(parent_head.device)).float().cpu()
    )
    materialization_error = float((materialized - lora_delta_final).abs().max())

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    summary = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "sure_h_model_path": str(Path(a.sure_h_model_path).resolve()),
        "training_visible_path": str(visible_path),
        "split_manifest": str(manifest_path),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "target_contract": {
            "sensitive_unwanted": "requested_rewrite.target_true",
            "benchmark_reference": "requested_rewrite.target_new",
            "target_new_is_replacement_training_target": False,
        },
        "stage1": "frozen SURE-H parent",
        "stage2": "residual sparse LM-head LoRA on residual direct failures",
        "sure_h_parent_parameters_trainable": 0,
        "underlying_sure_h_head_frozen_during_stage2": True,
        "underlying_sure_h_head_max_abs_change_before_materialization": parent_head_max_abs_change,
        "lora_rank": int(a.lora_rank),
        "lora_alpha": float(a.lora_alpha),
        "lora_scaling": float(a.lora_alpha) / int(a.lora_rank),
        "lora_trainable_parameters": int(delta_module.trainable_parameter_count),
        "selected_lm_head_rows": len(selected_ids),
        "selected_lm_head_token_ids": selected_ids,
        "chosen_lora_scale": float(chosen_scale),
        "lora_scale_reports": scale_reports,
        "solver_margin": float(a.solver_margin),
        "acceptance_margin": float(a.acceptance_margin),
        "parent_solver": parent_solver,
        "parent_acceptance": parent_acceptance,
        "pre_materialize_acceptance": _margin_report(
            pre_materialize_margins, float(a.acceptance_margin)
        ),
        "post_materialize_acceptance": _margin_report(
            post_materialize_margins, float(a.acceptance_margin)
        ),
        "stopped_step": int(stopped_step),
        "weights": {
            "ga": float(a.ga_weight),
            "gd": float(a.gd_weight),
            "relation_locality_kl": float(a.locality_kl_weight),
            "relation_sensitive_logit": float(a.locality_sensitive_logit_weight),
            "wikipedia_utility_kl": float(a.utility_kl_weight),
            "lora_l2": float(a.lora_l2_weight),
        },
        "relation_locality_pre_materialize": locality_pre_materialize,
        "relation_locality_after": locality_after,
        "wikipedia_utility_pre_materialize": utility_pre_materialize,
        "wikipedia_utility_after": utility_after,
        "lora_delta_norm": float(lora_delta_final.norm()),
        "lora_delta_mse": float(lora_delta_final.square().mean()),
        "materialization_max_abs_error": materialization_error,
        "official_paraphrase_seen": 0,
        "official_neighborhood_seen": 0,
        "benchmark_retain_seen": 0,
        "PPL_seen": False,
        "checkpoint": str(ckpt.resolve()),
    }
    _write_json(out_dir / "sure_h_then_residual_lmhead_lora_summary.json", summary)

    print("SURE-H -> residual LM-head LoRA checkpoint:", ckpt)
    print("Stage-1 SURE-H acceptance:", parent_acceptance)
    print("Selected residual LM-head rows:", len(selected_ids))
    print("LoRA rank / alpha:", a.lora_rank, "/", a.lora_alpha)
    print("Chosen LoRA scale:", chosen_scale)
    print("Final acceptance:", summary["post_materialize_acceptance"])
    print("Relation-locality after:", locality_after)
    print("Wikipedia utility after:", utility_after)
    print("Underlying SURE-H head max abs change before materialization:", parent_head_max_abs_change)
    print("Materialization max abs error:", materialization_error)
    print("Official MCF neighborhood/paraphrase/retain/PPL data were NOT used.")


if __name__ == "__main__":
    main()
