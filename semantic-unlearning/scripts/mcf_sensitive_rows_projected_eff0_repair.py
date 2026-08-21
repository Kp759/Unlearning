#!/usr/bin/env python3
"""Residual Eff=0 repair for locality-projected sensitive-row MCF Stage 1.

This stage starts from a finalized sensitive-row GA/GD checkpoint and repairs only
the remaining direct target_true-vs-target_new failures.  It preserves the same
parameter contract and locality geometry as ``mcf_sensitive_rows_projected_gagd``:

* target_true is sensitive/unwanted; target_new is the non-sensitive reference;
* transformer frozen;
* only target_true-sensitive embedding/LM-head rows may change;
* all non-sensitive vocabulary rows are restored exactly to Base;
* on tied Llama checkpoints the selected embedding and LM-head row is one tensor;
* official MCF paraphrase/neighborhood/retain/PPL evaluation data are never read.

Direct constraint (same sign convention as materialized Stage 2):

    m_i = NLL(target_true_i) - NLL(target_new_i)
    L_eff = mean ReLU(solver_margin - m_i)^2

The row update is projected after every optimizer step into the orthogonal
complement of token-specific Base locality directions derived only from relation-
matched donor subjects from the locked 50 training-visible direct records.
Soft relation-local KL, protected-sensitive-logit, Wikipedia KL, and Base-row
regularization guards remain active.

After optimization, the residual repair delta relative to the input Stage-1
checkpoint is scaled over a deterministic candidate list.  The *smallest* scale
that still has zero direct failures at ``acceptance_margin`` is selected, so the
additional movement is minimized.  The checkpoint is then saved, reloaded, and
zero direct failures are required again after materialization.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

import gagd_compare as gagd
import mcf_frozen_head_representation_repair as contract_helpers
import mcf_frozen_head_unknown_representation_repair as pair_helpers
import mcf_sensitive_rows_projected_gagd as projected
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import sure_stage1_gagd_w1k as wikipedia_utility
import sure_stage2_sparse_repair as stage2


METHOD = "SURE-LM-MCF-sensitive-rows-projected-Eff0-repair"
PROTOCOL = "mcf_target_true_sensitive_rows_relation_locality_eff0_v1"
DEFAULT_SCALES = "1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Input projected-GA Stage1 checkpoint")
    p.add_argument("--base-model-path", required=True, help="Original Base checkpoint for locality/reference rows")
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)

    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--solver-margin", type=float, default=0.25)
    p.add_argument("--acceptance-margin", type=float, default=0.05)
    p.add_argument("--eff-weight", type=float, default=1.0)

    p.add_argument("--subject-control-count", type=int, default=4)
    p.add_argument("--per-token-locality-rank", type=int, default=0)
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

    p.add_argument("--candidate-scales", default=DEFAULT_SCALES)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    a = p.parse_args(list(argv) if argv is not None else None)

    positive = (
        a.forget_num, a.steps, a.batch_size, a.lr, a.check_every, a.eff_weight,
        a.subject_control_count, a.locality_batch_size, a.locality_cache_batch_size,
        a.locality_kl_weight, a.locality_sensitive_logit_weight,
        a.utility_sample_size, a.utility_batch_size, a.utility_cache_batch_size,
        a.utility_max_length, a.utility_kl_weight,
    )
    if any(float(v) <= 0 for v in positive):
        p.error("counts, LR, and non-delta loss weights must be positive")
    nonnegative = (
        a.grad_clip, a.solver_margin, a.acceptance_margin,
        a.per_token_locality_rank, a.row_delta_weight, a.utility_exclude_first,
    )
    if any(float(v) < 0 for v in nonnegative):
        p.error("margins/rank/clipping/delta/exclusion must be non-negative")
    if a.solver_margin < a.acceptance_margin:
        p.error("solver-margin must be >= acceptance-margin")
    if a.utility_exclude_first < 20:
        p.error("utility-exclude-first must be at least 20")
    return a


def parse_scales(text: str) -> List[float]:
    vals: List[float] = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = float(raw)
        if not 0.0 <= value <= 1.0:
            raise ValueError("candidate scales must be in [0,1]")
        if value not in vals:
            vals.append(value)
    if not vals:
        raise ValueError("no candidate scales")
    if 1.0 not in vals:
        vals.insert(0, 1.0)
    if 0.0 not in vals:
        vals.append(0.0)
    return vals


def _optimizer(params: Iterable[nn.Parameter], kind: str, lr: float):
    ps = list(params)
    if kind == "sgd":
        return torch.optim.SGD(ps, lr=lr)
    if kind == "adam":
        return torch.optim.Adam(ps, lr=lr)
    return torch.optim.AdamW(ps, lr=lr, weight_decay=0.0)


def _snapshot_base_rows(model: nn.Module) -> Tuple[Dict[str, torch.Tensor], bool]:
    inp = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if inp is None or out is None:
        raise RuntimeError("model must expose input/output embeddings")
    tied = inp.weight.data_ptr() == out.weight.data_ptr()
    in_cpu = inp.weight.detach().cpu().clone()
    out_cpu = in_cpu if tied else out.weight.detach().cpu().clone()
    return {"input": in_cpu, "output": out_cpu}, bool(tied)


def _pair_hinge(
    model: nn.Module,
    tok: Any,
    instances: Sequence[Any],
    device: torch.device,
    llama_like: bool,
    margin: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    new_nll, true_nll = pair_helpers._unknown_forward(
        model, tok, instances, device, llama_like
    )
    margins = true_nll.float() - new_nll.float()
    return F.relu(float(margin) - margins).square().mean(), margins


@torch.no_grad()
def _all_margins(
    model: nn.Module,
    tok: Any,
    instances: Sequence[Any],
    device: torch.device,
    llama_like: bool,
    batch_size: int,
) -> torch.Tensor:
    return stage2.mcf_direct_margins(
        model, tok, instances, device, llama_like, int(batch_size),
        "target_true", "target_new"
    ).detach().float().cpu()


def _margin_summary(margins: torch.Tensor, required: float) -> Dict[str, Any]:
    return {
        "required_margin": float(required),
        "failures": int((margins < float(required)).sum().item()),
        "successes": int((margins >= float(required)).sum().item()),
        "minimum_margin": float(margins.min()),
        "mean_margin": float(margins.mean()),
        "maximum_margin": float(margins.max()),
        "margins": [float(x) for x in margins.tolist()],
    }


def _apply_blend(
    tied_info: Dict[str, Any],
    stage1_rows: Dict[str, torch.Tensor],
    repaired_rows: Dict[str, torch.Tensor],
    sensitive_ids: Sequence[int],
    scale: float,
) -> None:
    in_w: nn.Parameter = tied_info["input_weight"]
    out_w: nn.Parameter = tied_info["output_weight"]
    tied = bool(tied_info.get("tied"))
    ids = gagd.valid_row_ids(in_w, sensitive_ids)
    with torch.no_grad():
        if ids.numel():
            s0 = stage1_rows["input"].index_select(0, ids.cpu()).to(in_w.device, dtype=in_w.dtype)
            s1 = repaired_rows["input"].index_select(0, ids.cpu()).to(in_w.device, dtype=in_w.dtype)
            in_w.index_copy_(0, ids, s0 + float(scale) * (s1 - s0))
        if not tied:
            oids = gagd.valid_row_ids(out_w, sensitive_ids)
            if oids.numel():
                s0 = stage1_rows["output"].index_select(0, oids.cpu()).to(out_w.device, dtype=out_w.dtype)
                s1 = repaired_rows["output"].index_select(0, oids.cpu()).to(out_w.device, dtype=out_w.dtype)
                out_w.index_copy_(0, oids, s0 + float(scale) * (s1 - s0))


def _current_selected_rows(tied_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    in_w: nn.Parameter = tied_info["input_weight"]
    out_w: nn.Parameter = tied_info["output_weight"]
    tied = bool(tied_info.get("tied"))
    inc = in_w.detach().cpu().clone()
    return {
        "input": inc,
        "output": inc if tied else out_w.detach().cpu().clone(),
    }


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    scales = parse_scales(a.candidate_scales)
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

    # ----- Base-only references: locality geometry, utility logits, exact Base rows.
    base_ns = argparse.Namespace(
        model_path=a.base_model_path, dtype=a.dtype, device_map=a.device_map,
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
        tok, Path(a.utility_wikipedia_dir).resolve(),
        sample_size=int(a.utility_sample_size), seed=int(a.utility_seed),
        exclude_first=int(a.utility_exclude_first), max_length=int(a.utility_max_length)
    )
    print(f"Caching Base Wikipedia logits for {len(utility_prompts)} prompts...", flush=True)
    utility_base_logits = wikipedia_utility.cache_utility_base_logits(
        base_model, tok, utility_prompts, device, int(a.utility_cache_batch_size)
    )
    base_rows, base_tied = _snapshot_base_rows(base_model)
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----- Load the Spe-preserving Stage1 checkpoint to repair.
    src_ns = argparse.Namespace(
        model_path=a.model_path, dtype=a.dtype, device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok2 = gagd.load_model_and_tokenizer(src_ns, for_training=True)
    if tok2.pad_token is None:
        tok2.pad_token = tok2.eos_token
    tok = tok2
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)
    before_margins = _all_margins(
        model, tok, benchmark_instances, device, llama_like, a.batch_size
    )
    before = _margin_summary(before_margins, a.acceptance_margin)
    print("Direct failures before Eff0 repair:", before["failures"], flush=True)

    trainable_summary, tied_info = gagd.configure_trainable(
        model, gagd.POST_TRAINING_RESTORE_MODE
    )
    if bool(tied_info.get("tied")) != base_tied:
        raise RuntimeError("Base/source tied-embedding contract mismatch")
    bad = [
        name for name, p in model.named_parameters()
        if p.requires_grad and id(p) not in {id(tied_info["input_weight"]), id(tied_info["output_weight"])}
    ]
    if bad:
        raise RuntimeError(f"non-vocabulary parameters trainable: {bad[:5]}")

    # Stage1 row snapshot is the anchor for post-repair scale selection.
    stage1_rows = _current_selected_rows(tied_info)
    # Restore non-sensitive rows to literal Base immediately, then throughout repair.
    gagd.restore_non_selected_rows(tied_info, sensitive_ids, base_rows)
    token_bases = {k: v.to(device=device) for k, v in token_bases_cpu.items()}

    params = gagd.unique_trainable_params(model)
    opt = _optimizer(params, a.optimizer, float(a.lr))
    direct_sampler = core.IndexSampler(len(benchmark_instances), int(a.batch_size), int(a.seed) + 61001)
    locality_sampler = core.IndexSampler(len(locality_prompts), int(a.locality_batch_size), int(a.seed) + 61003)
    utility_sampler = core.IndexSampler(len(utility_prompts), int(a.utility_batch_size), int(a.utility_seed) + 61007)

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    core.write_json(out_dir / "relation_locality_receipt.json", locality_receipt)
    core.write_json(out_dir / "token_locality_basis_receipt.json", basis_receipt)
    core.write_json(out_dir / "wikipedia_utility_receipt.json", utility_receipt)
    core.write_json(out_dir / "direct_before.json", before)

    stopped_step = int(a.steps)
    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(1, int(a.steps) + 1), desc="MCF projected sensitive-row Eff0 repair"):
            didx = direct_sampler.next()
            direct_batch = [benchmark_instances[i] for i in didx]
            lidx = locality_sampler.next()
            locality_batch = [locality_prompts[i] for i in lidx]
            locality_ids = [locality_protected[i] for i in lidx]
            uidx = utility_sampler.next()
            utility_batch = [utility_prompts[i] for i in uidx]

            opt.zero_grad(set_to_none=True)
            eff_loss, batch_margins = _pair_hinge(
                model, tok, direct_batch, device, llama_like, a.solver_margin
            )
            _lh, local_logits = projected._prompt_hidden_and_logits(
                model, tok, locality_batch, device
            )
            local_base = base_local_logits[lidx]
            lkl = projected.locality_kl(local_logits, local_base)
            lrow = projected.protected_sensitive_logit_mse(local_logits, local_base, locality_ids)
            utility_logits = wikipedia_utility._forward_prompt_logits(
                model, tok, utility_batch, device
            )
            ukl = wikipedia_utility.utility_kl(utility_logits, utility_base_logits[uidx])
            drow = projected.selected_row_delta_mse(tied_info, base_rows, sensitive_ids)

            total = (
                float(a.eff_weight) * eff_loss
                + float(a.locality_kl_weight) * lkl
                + float(a.locality_sensitive_logit_weight) * lrow
                + float(a.utility_kl_weight) * ukl
                + float(a.row_delta_weight) * drow
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite Eff0 loss at step {step}")
            total.backward()
            gagd.apply_row_mask_and_restore(tied_info, sensitive_ids, base_rows)
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(params, float(a.grad_clip))
                if a.grad_clip > 0 else None
            )
            if grad_norm is not None and not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite grad norm at step {step}")
            opt.step()
            gagd.restore_non_selected_rows(tied_info, sensitive_ids, base_rows)
            projection = projected.project_selected_rows_to_locality_nullspace(
                tied_info, base_rows, token_bases, sensitive_ids
            )

            if step == 1 or step % int(a.check_every) == 0 or step == int(a.steps):
                allm = _all_margins(model, tok, benchmark_instances, device, llama_like, a.batch_size)
                solver_failures = int((allm < float(a.solver_margin)).sum().item())
                accept_failures = int((allm < float(a.acceptance_margin)).sum().item())
                row = {
                    "step": int(step),
                    "total_loss": float(total.detach().cpu()),
                    "eff_pair_hinge": float(eff_loss.detach().cpu()),
                    "batch_margin_mean": float(batch_margins.detach().mean().cpu()),
                    "solver_failures": solver_failures,
                    "acceptance_failures": accept_failures,
                    "minimum_margin": float(allm.min()),
                    "mean_margin": float(allm.mean()),
                    "relation_locality_kl": float(lkl.detach().cpu()),
                    "relation_sensitive_logit_mse": float(lrow.detach().cpu()),
                    "wikipedia_utility_kl": float(ukl.detach().cpu()),
                    "selected_row_delta_mse_from_base": float(drow.detach().cpu()),
                    "projection_max_removed_component_norm": projection["max_removed_component_norm"],
                }
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()
                if solver_failures == 0:
                    stopped_step = int(step)
                    print(f"All 50 direct cases satisfy solver margin at step {step}.", flush=True)
                    break

    del opt
    gagd.restore_non_selected_rows(tied_info, sensitive_ids, base_rows)
    projected.project_selected_rows_to_locality_nullspace(
        tied_info, base_rows, token_bases, sensitive_ids
    )
    repaired_rows = _current_selected_rows(tied_info)

    # ----- Shrink only the new repair delta; choose the smallest scale retaining Eff=0.
    scale_receipt: List[Dict[str, Any]] = []
    accepted: List[Tuple[float, torch.Tensor]] = []
    for scale in scales:
        _apply_blend(tied_info, stage1_rows, repaired_rows, sensitive_ids, float(scale))
        gagd.restore_non_selected_rows(tied_info, sensitive_ids, base_rows)
        margins = _all_margins(model, tok, benchmark_instances, device, llama_like, a.batch_size)
        failures = int((margins < float(a.acceptance_margin)).sum().item())
        entry = {
            "scale": float(scale),
            "failures": failures,
            "minimum_margin": float(margins.min()),
            "mean_margin": float(margins.mean()),
        }
        scale_receipt.append(entry)
        if failures == 0:
            accepted.append((float(scale), margins.clone()))

    if not accepted:
        _apply_blend(tied_info, stage1_rows, repaired_rows, sensitive_ids, 1.0)
        raise RuntimeError("Eff0 repair did not produce any zero-failure candidate scale")

    # Minimum added movement = smallest accepted repair scale.
    selected_scale, selected_margins = min(accepted, key=lambda item: item[0])
    _apply_blend(tied_info, stage1_rows, repaired_rows, sensitive_ids, selected_scale)
    gagd.restore_non_selected_rows(tied_info, sensitive_ids, base_rows)
    projected.project_selected_rows_to_locality_nullspace(
        tied_info, base_rows, token_bases, sensitive_ids
    )
    final_margins = _all_margins(model, tok, benchmark_instances, device, llama_like, a.batch_size)
    final_pre_save = _margin_summary(final_margins, a.acceptance_margin)
    if final_pre_save["failures"] != 0:
        raise RuntimeError("selected scale lost Eff=0 after final locality projection")

    locality_after = projected.evaluate_locality_guards(
        model, tok, locality_prompts, locality_protected, base_local_logits,
        device, int(a.locality_cache_batch_size)
    )
    utility_after = wikipedia_utility.evaluate_utility_kl(
        model, tok, utility_prompts, utility_base_logits,
        device, int(a.utility_cache_batch_size)
    )
    projection_after = projected.projection_residual_diagnostics(
        tied_info, base_rows, token_bases, sensitive_ids
    )

    model.eval()
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    # ----- Reload/materialization guard.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reload_ns = argparse.Namespace(
        model_path=str(ckpt), dtype=a.dtype, device_map=a.device_map,
        gradient_checkpointing=False,
    )
    reloaded, rtok = gagd.load_model_and_tokenizer(reload_ns, for_training=False)
    if rtok.pad_token is None:
        rtok.pad_token = rtok.eos_token
    rdevice = gagd.first_device(reloaded)
    rllama = is_llama_like(reloaded, rtok)
    reload_margins = _all_margins(
        reloaded, rtok, benchmark_instances, rdevice, rllama, a.batch_size
    )
    reload_summary = _margin_summary(reload_margins, a.acceptance_margin)
    if reload_summary["failures"] != 0:
        raise RuntimeError(
            f"materialized checkpoint has {reload_summary['failures']} direct failures"
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
        "solver_margin": float(a.solver_margin),
        "acceptance_margin": float(a.acceptance_margin),
        "stopped_step": int(stopped_step),
        "before": before,
        "candidate_scales": scale_receipt,
        "selected_scale": float(selected_scale),
        "final_pre_save": final_pre_save,
        "reload": reload_summary,
        "relation_locality_after": locality_after,
        "wikipedia_utility_after": utility_after,
        "projection_after": projection_after,
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "sensitive_rows_projected_eff0_summary.json", summary)

    print("Eff0 repair checkpoint:", ckpt)
    print("Direct before:", before)
    print("Stopped step:", stopped_step)
    print("Selected repair scale:", selected_scale)
    print("Final direct pre-save:", final_pre_save)
    print("Reload direct:", reload_summary)
    print("Relation-locality after:", locality_after)
    print("Wikipedia utility after:", utility_after)
    print("Projection residual after:", projection_after)
    print("Official neighborhood/paraphrase probes were NOT used in training or selection.")


if __name__ == "__main__":
    main()
