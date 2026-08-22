#!/usr/bin/env python3
"""Minimal Eff=0 repair anchored to the Spe-preserving projected-GA checkpoint.

This experiment is intentionally narrower than ``mcf_sensitive_rows_projected_eff0_repair``.
It starts from the finalized Spe-preserving sensitive-row Stage1 checkpoint and solves only
remaining direct target_true-vs-target_new failures while minimizing *additional* movement
from that checkpoint.

Parameter contract:
  * target_true = sensitive / unwanted answer;
  * target_new = non-sensitive benchmark reference;
  * transformer frozen;
  * only target_true-sensitive embedding/LM-head rows may change;
  * all non-sensitive rows are literal Base before save;
  * no LoRA/adapters/full-head repair;
  * official MCF paraphrase/neighborhood/retain/PPL examples are never read.

Direct constraint:

    m_i = NLL(target_true_i) - NLL(target_new_i)
    L_eff = mean ReLU(solver_margin - m_i)^2

Minimal-repair changes relative to the previous Eff0 repair:
  1. solver margin defaults to 0.10 (final/reload acceptance remains 0.05);
  2. only currently failing direct cases receive the direct hinge gradient;
  3. selected-row movement is anchored to the *input Stage1 checkpoint*, not Base;
  4. the hard locality projection acts only on the additional repair delta
     (W - W_stage1), preserving the parent edit while preventing the new correction
     from entering Base relation-locality directions;
  5. after convergence, the additional repair delta is scale-shrunk and the smallest
     materialized zero-failure candidate is selected.

The optimization therefore approximates

    min ||W - W_stage1||
    s.t. m_i >= solver_margin for every direct training-visible fact,
         DeltaW_repair is orthogonal to the learned locality subspace.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
from tqdm import tqdm

import gagd_compare as gagd
import mcf_frozen_head_representation_repair as contract_helpers
import mcf_sensitive_rows_projected_eff0_repair as eff0
import mcf_sensitive_rows_projected_gagd as projected
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import sure_stage1_gagd_w1k as wikipedia_utility
import sure_stage2_sparse_repair as stage2


METHOD = "SURE-LM-MCF-sensitive-rows-projected-minimal-Eff0-repair"
PROTOCOL = "mcf_target_true_sensitive_rows_relation_locality_minimal_eff0_v1"
DEFAULT_SCALES = eff0.DEFAULT_SCALES


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Spe-preserving projected-GA Stage1 checkpoint")
    p.add_argument("--base-model-path", required=True, help="Original Base model")
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)

    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--check-every", type=int, default=10)
    p.add_argument("--solver-margin", type=float, default=0.10)
    p.add_argument("--acceptance-margin", type=float, default=0.05)
    p.add_argument("--eff-weight", type=float, default=1.0)
    p.add_argument(
        "--stage1-anchor-weight",
        type=float,
        default=50.0,
        help="MSE penalty on additional selected-row movement from the Stage1 checkpoint",
    )

    p.add_argument("--subject-control-count", type=int, default=4)
    p.add_argument("--per-token-locality-rank", type=int, default=0)
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

    p.add_argument("--candidate-scales", default=DEFAULT_SCALES)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    a = p.parse_args(list(argv) if argv is not None else None)

    positive = (
        a.forget_num,
        a.steps,
        a.batch_size,
        a.lr,
        a.check_every,
        a.eff_weight,
        a.stage1_anchor_weight,
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
        p.error("counts, LR, and loss weights must be positive")
    nonnegative = (
        a.grad_clip,
        a.solver_margin,
        a.acceptance_margin,
        a.per_token_locality_rank,
        a.utility_exclude_first,
    )
    if any(float(v) < 0 for v in nonnegative):
        p.error("margins/rank/clipping/exclusion must be non-negative")
    if a.solver_margin < a.acceptance_margin:
        p.error("solver-margin must be >= acceptance-margin")
    if a.utility_exclude_first < 20:
        p.error("utility-exclude-first must be at least 20")
    return a


def _snapshot_selected_rows(
    tied_info: Dict[str, Any], selected_ids: Sequence[int]
) -> Dict[str, torch.Tensor]:
    in_w: nn.Parameter = tied_info["input_weight"]
    out_w: nn.Parameter = tied_info["output_weight"]
    tied = bool(tied_info.get("tied"))
    in_ids = gagd.valid_row_ids(in_w, selected_ids)
    inp = in_w.detach().index_select(0, in_ids).float().cpu().clone()
    if tied:
        out = inp
    else:
        out_ids = gagd.valid_row_ids(out_w, selected_ids)
        out = out_w.detach().index_select(0, out_ids).float().cpu().clone()
    return {"input": inp, "output": out}


def stage1_anchor_mse(
    tied_info: Dict[str, Any],
    stage1_selected: Dict[str, torch.Tensor],
    selected_ids: Sequence[int],
) -> torch.Tensor:
    in_w: nn.Parameter = tied_info["input_weight"]
    out_w: nn.Parameter = tied_info["output_weight"]
    tied = bool(tied_info.get("tied"))
    in_ids = gagd.valid_row_ids(in_w, selected_ids)
    cur_in = in_w.index_select(0, in_ids).float()
    ref_in = stage1_selected["input"].to(device=in_w.device, dtype=torch.float32)
    losses = [(cur_in - ref_in).square().mean()]
    if not tied:
        out_ids = gagd.valid_row_ids(out_w, selected_ids)
        cur_out = out_w.index_select(0, out_ids).float()
        ref_out = stage1_selected["output"].to(device=out_w.device, dtype=torch.float32)
        losses.append((cur_out - ref_out).square().mean())
    return torch.stack(losses).mean()


@torch.no_grad()
def project_repair_delta_to_locality_nullspace(
    tied_info: Dict[str, Any],
    stage1_selected: Dict[str, torch.Tensor],
    token_bases: Dict[int, torch.Tensor],
    selected_ids: Sequence[int],
) -> Dict[str, float]:
    """Project only W-current minus W-stage1, not the parent Stage1 edit itself."""
    in_w: nn.Parameter = tied_info["input_weight"]
    out_w: nn.Parameter = tied_info["output_weight"]
    tied = bool(tied_info.get("tied"))
    id_to_pos = {int(tid): pos for pos, tid in enumerate(selected_ids)}
    removed: List[float] = []

    def project(weight: nn.Parameter, reference: torch.Tensor) -> None:
        for tid in selected_ids:
            token_id = int(tid)
            if token_id < 0 or token_id >= weight.shape[0]:
                continue
            basis = token_bases.get(token_id)
            if basis is None or not basis.numel():
                continue
            pos = id_to_pos[token_id]
            ref = reference[pos].to(device=weight.device, dtype=torch.float32)
            delta = weight[token_id].float() - ref
            b = basis.to(device=weight.device, dtype=torch.float32)
            coeff = delta @ b.T
            component = coeff @ b
            weight[token_id].sub_(component.to(dtype=weight.dtype))
            removed.append(float(component.norm().detach().cpu()))

    project(in_w, stage1_selected["input"])
    if not tied:
        project(out_w, stage1_selected["output"])
    return {
        "max_removed_component_norm": max(removed) if removed else 0.0,
        "projected_row_matrix_pairs": int(len(removed)),
    }


@torch.no_grad()
def repair_projection_residual(
    tied_info: Dict[str, Any],
    stage1_selected: Dict[str, torch.Tensor],
    token_bases: Dict[int, torch.Tensor],
    selected_ids: Sequence[int],
) -> Dict[str, float]:
    in_w: nn.Parameter = tied_info["input_weight"]
    out_w: nn.Parameter = tied_info["output_weight"]
    tied = bool(tied_info.get("tied"))
    id_to_pos = {int(tid): pos for pos, tid in enumerate(selected_ids)}
    values: List[float] = []

    def inspect(weight: nn.Parameter, reference: torch.Tensor) -> None:
        for tid in selected_ids:
            token_id = int(tid)
            basis = token_bases.get(token_id)
            if basis is None or not basis.numel() or not 0 <= token_id < weight.shape[0]:
                continue
            pos = id_to_pos[token_id]
            ref = reference[pos].to(device=weight.device, dtype=torch.float32)
            delta = weight[token_id].float() - ref
            b = basis.to(device=weight.device, dtype=torch.float32)
            values.append(float((delta @ b.T).abs().max().detach().cpu()))

    inspect(in_w, stage1_selected["input"])
    if not tied:
        inspect(out_w, stage1_selected["output"])
    return {
        "max_abs_repair_delta_dot_locality_basis": max(values) if values else 0.0,
        "checked_row_matrix_pairs": int(len(values)),
    }


@torch.no_grad()
def _apply_selected_blend(
    tied_info: Dict[str, Any],
    stage1_selected: Dict[str, torch.Tensor],
    repaired_selected: Dict[str, torch.Tensor],
    selected_ids: Sequence[int],
    scale: float,
) -> None:
    in_w: nn.Parameter = tied_info["input_weight"]
    out_w: nn.Parameter = tied_info["output_weight"]
    tied = bool(tied_info.get("tied"))
    in_ids = gagd.valid_row_ids(in_w, selected_ids)
    if in_ids.numel():
        s0 = stage1_selected["input"].to(device=in_w.device, dtype=torch.float32)
        s1 = repaired_selected["input"].to(device=in_w.device, dtype=torch.float32)
        blended = s0 + float(scale) * (s1 - s0)
        in_w.index_copy_(0, in_ids, blended.to(dtype=in_w.dtype))
    if not tied:
        out_ids = gagd.valid_row_ids(out_w, selected_ids)
        if out_ids.numel():
            s0 = stage1_selected["output"].to(device=out_w.device, dtype=torch.float32)
            s1 = repaired_selected["output"].to(device=out_w.device, dtype=torch.float32)
            blended = s0 + float(scale) * (s1 - s0)
            out_w.index_copy_(0, out_ids, blended.to(dtype=out_w.dtype))


def _active_batch(
    active_positions: Sequence[int], batch_size: int, rng: random.Random
) -> List[int]:
    if not active_positions:
        return []
    if len(active_positions) <= int(batch_size):
        return list(active_positions)
    return rng.sample(list(active_positions), k=int(batch_size))


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    scales = eff0.parse_scales(a.candidate_scales)
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

    # Base is used only for exact non-sensitive restoration and locality/utility teachers.
    base_ns = argparse.Namespace(
        model_path=a.base_model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    base_model, tok = gagd.load_model_and_tokenizer(base_ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = gagd.first_device(base_model)
    llama_like = is_llama_like(base_model, tok)
    benchmark_instances = stage2.mcf_instances(records)

    cases = core.expand_sensitive_cases(
        records, tok, sensitive_field="target_true", llama_like=llama_like
    )
    tids = core.official_target_ids(tok, cases, llama_like=llama_like, device=device)
    sensitive_ids = sorted(set(int(x) for x in tids.detach().cpu().tolist()))
    sensitive_ids = [i for i in sensitive_ids if i not in gagd.special_token_ids(tok)]
    if not sensitive_ids:
        raise RuntimeError("no target_true sensitive rows")

    locality_prompts, locality_protected, locality_receipt = projected.build_relation_locality_controls(
        records, tok, benchmark_instances, int(a.subject_control_count)
    )
    print(f"Caching Base locality references for {len(locality_prompts)} prompts...", flush=True)
    base_local_hidden, base_local_logits = projected.cache_relation_locality_reference(
        base_model, tok, locality_prompts, device, int(a.locality_cache_batch_size)
    )
    token_bases_cpu, basis_receipt = projected.build_token_locality_bases(
        base_local_hidden, locality_protected, int(a.per_token_locality_rank)
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
        base_model, tok, utility_prompts, device, int(a.utility_cache_batch_size)
    )
    base_rows, base_tied = eff0._snapshot_base_rows(base_model)
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    src_ns = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok2 = gagd.load_model_and_tokenizer(src_ns, for_training=True)
    if tok2.pad_token is None:
        tok2.pad_token = tok2.eos_token
    tok = tok2
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)

    before_margins = eff0._all_margins(
        model, tok, benchmark_instances, device, llama_like, a.batch_size
    )
    before_accept = eff0._margin_summary(before_margins, a.acceptance_margin)
    before_solver = eff0._margin_summary(before_margins, a.solver_margin)
    active_positions = [
        i for i, value in enumerate(before_margins.tolist())
        if float(value) < float(a.solver_margin)
    ]
    print(
        f"Direct failures before minimal Eff0 repair: acceptance={before_accept['failures']} "
        f"solver={before_solver['failures']}",
        flush=True,
    )
    if not active_positions:
        raise RuntimeError("source checkpoint already satisfies the solver margin on all direct cases")

    trainable_summary, tied_info = gagd.configure_trainable(
        model, gagd.POST_TRAINING_RESTORE_MODE
    )
    if bool(tied_info.get("tied")) != base_tied:
        raise RuntimeError("Base/source tied-embedding contract mismatch")
    allowed = {id(tied_info["input_weight"]), id(tied_info["output_weight"])}
    bad = [
        name for name, p in model.named_parameters()
        if p.requires_grad and id(p) not in allowed
    ]
    if bad:
        raise RuntimeError(f"non-vocabulary parameters trainable: {bad[:5]}")

    # This is the optimization anchor. It is intentionally the Spe-preserving parent,
    # not Base, and stores only the selected rows.
    stage1_selected = _snapshot_selected_rows(tied_info, sensitive_ids)
    gagd.restore_non_selected_rows(tied_info, sensitive_ids, base_rows)
    token_bases = {k: v.to(device=device) for k, v in token_bases_cpu.items()}

    params = gagd.unique_trainable_params(model)
    opt = eff0._optimizer(params, a.optimizer, float(a.lr))
    locality_sampler = core.IndexSampler(
        len(locality_prompts), int(a.locality_batch_size), int(a.seed) + 71003
    )
    utility_sampler = core.IndexSampler(
        len(utility_prompts), int(a.utility_batch_size), int(a.utility_seed) + 71007
    )
    direct_rng = random.Random(int(a.seed) + 71001)

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    core.write_json(out_dir / "relation_locality_receipt.json", locality_receipt)
    core.write_json(out_dir / "token_locality_basis_receipt.json", basis_receipt)
    core.write_json(out_dir / "wikipedia_utility_receipt.json", utility_receipt)
    core.write_json(out_dir / "direct_before_acceptance.json", before_accept)
    core.write_json(out_dir / "direct_before_solver.json", before_solver)

    stopped_step = int(a.steps)
    active_history: List[Dict[str, Any]] = []
    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(1, int(a.steps) + 1), desc="MCF minimal projected sensitive-row Eff0"):
            didx = _active_batch(active_positions, int(a.batch_size), direct_rng)
            if not didx:
                stopped_step = max(0, step - 1)
                break
            direct_batch = [benchmark_instances[i] for i in didx]
            lidx = locality_sampler.next()
            locality_batch = [locality_prompts[i] for i in lidx]
            locality_ids = [locality_protected[i] for i in lidx]
            uidx = utility_sampler.next()
            utility_batch = [utility_prompts[i] for i in uidx]

            opt.zero_grad(set_to_none=True)
            eff_loss, batch_margins = eff0._pair_hinge(
                model, tok, direct_batch, device, llama_like, a.solver_margin
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
            anchor = stage1_anchor_mse(tied_info, stage1_selected, sensitive_ids)

            total = (
                float(a.eff_weight) * eff_loss
                + float(a.locality_kl_weight) * lkl
                + float(a.locality_sensitive_logit_weight) * lrow
                + float(a.utility_kl_weight) * ukl
                + float(a.stage1_anchor_weight) * anchor
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite minimal Eff0 loss at step {step}")
            total.backward()

            # Only sensitive target_true rows can update. Non-sensitive rows have zero
            # gradients and AdamW uses zero weight decay in this experiment.
            gagd.apply_row_mask_and_restore(tied_info, sensitive_ids, base_rows)
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(params, float(a.grad_clip))
                if a.grad_clip > 0 else None
            )
            if grad_norm is not None and not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite grad norm at step {step}")
            opt.step()

            projection = project_repair_delta_to_locality_nullspace(
                tied_info, stage1_selected, token_bases, sensitive_ids
            )

            if step == 1 or step % int(a.check_every) == 0 or step == int(a.steps):
                allm = eff0._all_margins(
                    model, tok, benchmark_instances, device, llama_like, a.batch_size
                )
                active_positions = [
                    i for i, value in enumerate(allm.tolist())
                    if float(value) < float(a.solver_margin)
                ]
                accept_failures = int((allm < float(a.acceptance_margin)).sum().item())
                anchor_now = stage1_anchor_mse(
                    tied_info, stage1_selected, sensitive_ids
                ).detach()
                row = {
                    "step": int(step),
                    "total_loss": float(total.detach().cpu()),
                    "eff_pair_hinge": float(eff_loss.detach().cpu()),
                    "batch_margin_mean": float(batch_margins.detach().mean().cpu()),
                    "active_solver_failures": int(len(active_positions)),
                    "acceptance_failures": int(accept_failures),
                    "minimum_margin": float(allm.min()),
                    "mean_margin": float(allm.mean()),
                    "stage1_anchor_mse": float(anchor_now.cpu()),
                    "relation_locality_kl": float(lkl.detach().cpu()),
                    "relation_sensitive_logit_mse": float(lrow.detach().cpu()),
                    "wikipedia_utility_kl": float(ukl.detach().cpu()),
                    "repair_projection_max_removed_component_norm": projection["max_removed_component_norm"],
                }
                active_history.append(
                    {
                        "step": int(step),
                        "solver_failures": int(len(active_positions)),
                        "acceptance_failures": int(accept_failures),
                    }
                )
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()
                if not active_positions:
                    stopped_step = int(step)
                    print(
                        f"All 50 direct cases satisfy minimal solver margin at step {step}.",
                        flush=True,
                    )
                    break

    del opt
    gagd.restore_non_selected_rows(tied_info, sensitive_ids, base_rows)
    project_repair_delta_to_locality_nullspace(
        tied_info, stage1_selected, token_bases, sensitive_ids
    )
    repaired_selected = _snapshot_selected_rows(tied_info, sensitive_ids)

    # Shrink only the additional Stage1->Eff0 correction.
    scale_receipt: List[Dict[str, Any]] = []
    accepted: List[Tuple[float, torch.Tensor]] = []
    for scale in scales:
        _apply_selected_blend(
            tied_info, stage1_selected, repaired_selected, sensitive_ids, float(scale)
        )
        gagd.restore_non_selected_rows(tied_info, sensitive_ids, base_rows)
        project_repair_delta_to_locality_nullspace(
            tied_info, stage1_selected, token_bases, sensitive_ids
        )
        margins = eff0._all_margins(
            model, tok, benchmark_instances, device, llama_like, a.batch_size
        )
        failures = int((margins < float(a.acceptance_margin)).sum().item())
        entry = {
            "scale": float(scale),
            "failures": int(failures),
            "minimum_margin": float(margins.min()),
            "mean_margin": float(margins.mean()),
        }
        scale_receipt.append(entry)
        if failures == 0:
            accepted.append((float(scale), margins.clone()))

    if not accepted:
        _apply_selected_blend(
            tied_info, stage1_selected, repaired_selected, sensitive_ids, 1.0
        )
        raise RuntimeError("minimal Eff0 repair produced no zero-failure candidate scale")

    selected_scale, _selected_margins = min(accepted, key=lambda item: item[0])
    _apply_selected_blend(
        tied_info, stage1_selected, repaired_selected, sensitive_ids, selected_scale
    )
    gagd.restore_non_selected_rows(tied_info, sensitive_ids, base_rows)
    project_repair_delta_to_locality_nullspace(
        tied_info, stage1_selected, token_bases, sensitive_ids
    )

    final_margins = eff0._all_margins(
        model, tok, benchmark_instances, device, llama_like, a.batch_size
    )
    final_pre_save = eff0._margin_summary(final_margins, a.acceptance_margin)
    if final_pre_save["failures"] != 0:
        raise RuntimeError("selected minimal scale lost Eff=0 after final projection")

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
    projection_after = repair_projection_residual(
        tied_info, stage1_selected, token_bases, sensitive_ids
    )
    final_anchor_mse = float(
        stage1_anchor_mse(tied_info, stage1_selected, sensitive_ids).detach().cpu()
    )

    model.eval()
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    # Materialization-safe acceptance guard.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reload_ns = argparse.Namespace(
        model_path=str(ckpt),
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    reloaded, rtok = gagd.load_model_and_tokenizer(reload_ns, for_training=False)
    if rtok.pad_token is None:
        rtok.pad_token = rtok.eos_token
    rdevice = gagd.first_device(reloaded)
    rllama = is_llama_like(reloaded, rtok)
    reload_margins = eff0._all_margins(
        reloaded, rtok, benchmark_instances, rdevice, rllama, a.batch_size
    )
    reload_summary = eff0._margin_summary(reload_margins, a.acceptance_margin)
    if reload_summary["failures"] != 0:
        raise RuntimeError(
            f"materialized minimal checkpoint has {reload_summary['failures']} direct failures"
        )

    summary = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "source_stage1_checkpoint": str(Path(a.model_path).resolve()),
        "base_model_path": str(Path(a.base_model_path).resolve()),
        "training_visible_path": str(visible_path),
        "split_manifest": str(manifest_path),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "target_contract": {
            "sensitive_unwanted": "requested_rewrite.target_true",
            "benchmark_reference": "requested_rewrite.target_new",
            "field_swapping": False,
        },
        "optimization": {
            "active_only_direct_training": True,
            "solver_margin": float(a.solver_margin),
            "acceptance_margin": float(a.acceptance_margin),
            "stage1_anchor_weight": float(a.stage1_anchor_weight),
            "anchor_reference": "input Spe-preserving Stage1 selected rows",
            "repair_projection_reference": "Stage1 selected rows",
            "lr": float(a.lr),
            "steps": int(a.steps),
            "stopped_step": int(stopped_step),
        },
        "transformer_trainable": False,
        "input_embedding_sensitive_rows_trainable": True,
        "lm_head_sensitive_rows_trainable": True,
        "input_output_tied": bool(tied_info.get("tied")),
        "selected_sensitive_token_ids": [int(x) for x in sensitive_ids],
        "selected_sensitive_row_count": int(len(sensitive_ids)),
        "all_non_sensitive_rows_restored_to_base": True,
        "official_neighborhood_seen": 0,
        "official_paraphrase_seen": 0,
        "benchmark_retain_seen": 0,
        "PPL_seen": False,
        "before_acceptance": before_accept,
        "before_solver": before_solver,
        "active_history": active_history,
        "candidate_scales": scale_receipt,
        "selected_scale": float(selected_scale),
        "final_stage1_anchor_mse": float(final_anchor_mse),
        "final_pre_save": final_pre_save,
        "reload": reload_summary,
        "relation_locality_after": locality_after,
        "wikipedia_utility_after": utility_after,
        "repair_projection_after": projection_after,
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "sensitive_rows_projected_eff0_minimal_summary.json", summary)

    print("Minimal Eff0 repair checkpoint:", ckpt)
    print("Direct before acceptance:", before_accept)
    print("Direct before solver:", before_solver)
    print("Stopped step:", stopped_step)
    print("Selected repair scale:", selected_scale)
    print("Final Stage1-anchor MSE:", final_anchor_mse)
    print("Final direct pre-save:", final_pre_save)
    print("Reload direct:", reload_summary)
    print("Relation-locality after:", locality_after)
    print("Wikipedia utility after:", utility_after)
    print("Repair projection residual after:", projection_after)
    print("Official neighborhood/paraphrase probes were NOT used in training or selection.")


if __name__ == "__main__":
    main()
