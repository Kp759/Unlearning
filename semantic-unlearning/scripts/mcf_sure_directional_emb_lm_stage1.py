#!/usr/bin/env python3
"""MCF SURE Stage 1: untied sensitive Emb+LM directional GA.

Target contract (hard enforced):
  * requested_rewrite.target_true = sensitive / unwanted answer
  * requested_rewrite.target_new  = non-sensitive reference answer
  * fields are not swapped

Architecture:
  1. Load Base, clone/untie the LM head from the input embedding matrix.
  2. Freeze the transformer and every Base parameter.
  3. Select only vocabulary rows occurring in target_true.
  4. For every selected sensitive row, construct a row-specific direction basis:
       d = h_sensitive - h_reference
     using matched teacher-forced direct contexts from target_true/target_new.
     At the first answer token both prefixes are identical, so this hidden
     contrast can be exactly zero.  In that case use the decoder discriminant
       d = w_sensitive - w_reference,
     which is the hidden-space gradient of the corresponding logit gap.
  5. Parameterize BOTH the selected input-embedding row deltas and selected
     LM-head row deltas inside the same fixed sensitive direction basis.
  6. Minimize sensitive log-probability (GA convention used in this repo), with
     an optional frozen-Base KL over the remaining vocabulary and delta L2.
  7. Select one common scale for Emb+LM using direct-only MCF margins and
     materialize only those selected rows.

No LoRA is used. No official paraphrase, neighborhood, benchmark-retain, or PPL
content is opened. target_new is used only to define the sensitive-vs-reference
direction and the direct margin gate; it is not a GD/CE training target here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from torch import nn

import gagd_compare as gagd
import mcf_synthetic_paraphrase_templates as synth
import sure_canonical_core as core
import sure_context_projection as context
import sure_stage2_sparse_repair as stage2


METHOD = "SURE-MCF-directional-EmbLM-GA-stage1"
PROTOCOL = "mcf_target_true_sensitive_directional_emb_lm_ga_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--distribution-kl-weight", type=float, default=1.0)
    p.add_argument("--delta-l2", type=float, default=1e-6)
    p.add_argument(
        "--direction-rank",
        type=int,
        default=1,
        help="Per-sensitive-row contrast basis rank cap; 0 uses full numerical rank.",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--stage1-constraint-margin", type=float, default=0.05)
    p.add_argument(
        "--synthetic-paraphrases-per-record",
        type=int,
        default=3,
        help=(
            "Hand-authored synthetic paraphrase templates per record used to "
            "fit the sensitive direction, GA-train, and gate scale selection. "
            "Never derived from the record's real (held-out) paraphrase_prompts. "
            "Set 0 to disable and match the original direct-only behavior."
        ),
    )
    p.add_argument(
        "--candidate-scales",
        default="1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0",
    )
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    a = p.parse_args(list(argv) if argv is not None else None)
    if min(a.forget_num, a.steps, a.batch_size, a.cache_batch_size) <= 0:
        p.error("forget-num, steps and batch sizes must be positive")
    if a.lr <= 0 or a.ga_weight <= 0:
        p.error("lr and ga-weight must be positive")
    if min(a.distribution_kl_weight, a.delta_l2, a.direction_rank, a.grad_clip, a.stage1_constraint_margin) < 0:
        p.error("KL/L2/rank/clip/margin must be non-negative")
    if a.synthetic_paraphrases_per_record < 0:
        p.error("synthetic-paraphrases-per-record must be non-negative")
    return a


def validate_locked(
    visible_path: Path,
    manifest_path: Path,
    seed: int,
    forget_num: int,
) -> Tuple[List[Mapping[str, Any]], Mapping[str, Any]]:
    records, manifest = stage2.load_locked(
        "mcf", visible_path, manifest_path, seed, forget_num
    )
    contract = manifest.get("target_contract", {})
    if isinstance(contract, Mapping) and contract:
        sensitive = contract.get("sensitive_answer")
        reference = contract.get("non_sensitive_reference")
        swapping = contract.get("field_swapping")
        if sensitive not in (None, "requested_rewrite.target_true"):
            raise RuntimeError(f"Expected target_true sensitive, got {sensitive!r}")
        if reference not in (None, "requested_rewrite.target_new"):
            raise RuntimeError(f"Expected target_new reference, got {reference!r}")
        if swapping not in (None, False):
            raise RuntimeError("Directional Stage 1 requires unswapped target fields")
    for i, record in enumerate(records):
        if (
            record.get("paraphrase_prompts")
            or record.get("neighborhood_prompts")
            or record.get("generation_prompts")
        ):
            raise RuntimeError(f"record {i} exposes held-out probes")
    return records, manifest


def _reference_index_by_sensitive_case(
    sensitive_cases: Sequence[core.SensitivePredictionCase],
    reference_cases: Sequence[core.SensitivePredictionCase],
) -> List[int]:
    """Match each sensitive teacher-forced state to the closest reference state."""
    by_record: Dict[int, List[int]] = {}
    for j, case in enumerate(reference_cases):
        by_record.setdefault(int(case.record_position), []).append(j)
    for positions in by_record.values():
        positions.sort(key=lambda j: int(reference_cases[j].token_index))

    matched: List[int] = []
    for case in sensitive_cases:
        choices = by_record.get(int(case.record_position), [])
        if not choices:
            raise RuntimeError(
                f"record position {case.record_position} has no reference token cases"
            )
        desired = int(case.token_index)
        exact = [j for j in choices if int(reference_cases[j].token_index) == desired]
        matched.append(exact[0] if exact else choices[min(desired, len(choices) - 1)])
    return matched


def contrast_direction(
    sensitive_hidden: torch.Tensor,
    reference_hidden: torch.Tensor,
    sensitive_decoder_row: torch.Tensor,
    reference_decoder_row: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, str]:
    """Return normalized hidden contrast, with decoder-discriminant fallback."""
    direction = sensitive_hidden.float() - reference_hidden.float()
    norm = direction.norm()
    source = "hidden_sensitive_minus_reference"
    if not torch.isfinite(norm) or float(norm) <= eps:
        direction = sensitive_decoder_row.float() - reference_decoder_row.float()
        norm = direction.norm()
        source = "decoder_row_sensitive_minus_reference_fallback"
    if not torch.isfinite(norm) or float(norm) <= eps:
        direction = sensitive_hidden.float()
        norm = direction.norm()
        source = "sensitive_hidden_fallback"
    if not torch.isfinite(norm) or float(norm) <= eps:
        raise RuntimeError("Unable to construct a non-zero sensitive direction")
    return direction / norm, source


@torch.no_grad()
def build_row_specific_contrast_bases(
    model: nn.Module,
    tok: Any,
    output_layer: nn.Module,
    sensitive_cases: Sequence[core.SensitivePredictionCase],
    reference_cases: Sequence[core.SensitivePredictionCase],
    selected_ids: Sequence[int],
    *,
    llama_like: bool,
    device: torch.device,
    batch_size: int,
    max_rank: int | None,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    sensitive_hidden = core.forward_last_hidden(
        model, tok, sensitive_cases, device, batch_size
    )
    reference_hidden = core.forward_last_hidden(
        model, tok, reference_cases, device, batch_size
    )
    sensitive_tids = core.official_target_ids(
        tok, sensitive_cases, llama_like=llama_like, device=device
    ).detach()
    reference_tids = core.official_target_ids(
        tok, reference_cases, llama_like=llama_like, device=device
    ).detach()
    ref_match = _reference_index_by_sensitive_case(sensitive_cases, reference_cases)

    directions_by_tid: Dict[int, List[torch.Tensor]] = {
        int(tid): [] for tid in selected_ids
    }
    sources_by_tid: Dict[int, Dict[str, int]] = {
        int(tid): {} for tid in selected_ids
    }

    for i, case in enumerate(sensitive_cases):
        sensitive_tid = int(sensitive_tids[i].item())
        if sensitive_tid not in directions_by_tid:
            continue
        j = int(ref_match[i])
        reference_tid = int(reference_tids[j].item())
        direction, source = contrast_direction(
            sensitive_hidden[i],
            reference_hidden[j],
            output_layer.weight[sensitive_tid],
            output_layer.weight[reference_tid],
        )
        directions_by_tid[sensitive_tid].append(direction.detach().cpu())
        sources = sources_by_tid[sensitive_tid]
        sources[source] = int(sources.get(source, 0)) + 1

    bases: List[torch.Tensor] = []
    reports: List[Dict[str, Any]] = []
    for tid in [int(x) for x in selected_ids]:
        rows = directions_by_tid[tid]
        if not rows:
            raise RuntimeError(f"sensitive token {tid} has no contrast directions")
        matrix = torch.stack(rows, dim=0).float()
        basis = core.orthonormal_row_basis(matrix, max_rank=max_rank)
        if basis.ndim != 2 or basis.shape[0] <= 0:
            raise RuntimeError(f"sensitive token {tid} has zero contrast rank")
        bases.append(basis.contiguous())
        reports.append(
            {
                "token_id": tid,
                "direction_count": int(matrix.shape[0]),
                "direction_rank": int(basis.shape[0]),
                "hidden_size": int(basis.shape[1]),
                "direction_sources": sources_by_tid[tid],
            }
        )
    return bases, reports


def register_input_embedding_delta_hook(
    input_layer: nn.Module,
    row_ids: Sequence[int],
    delta_getter,
):
    """Add differentiable deltas only when selected vocabulary rows are inputs."""
    if not hasattr(input_layer, "weight"):
        raise ValueError("input embedding module must expose weight")
    vocab = int(input_layer.weight.shape[0])
    device = input_layer.weight.device
    lookup = torch.full((vocab,), -1, dtype=torch.long, device=device)
    ids = torch.tensor([int(x) for x in row_ids], dtype=torch.long, device=device)
    if ids.numel() > 0:
        lookup[ids] = torch.arange(ids.numel(), dtype=torch.long, device=device)

    def hook(_module: nn.Module, inputs: Any, output: torch.Tensor) -> torch.Tensor:
        if ids.numel() == 0:
            return output
        token_ids = inputs[0].to(device=lookup.device)
        local = lookup[token_ids]
        mask = local.ge(0)
        if not bool(mask.any()):
            return output
        safe = local.clamp_min(0)
        delta = delta_getter().to(device=output.device, dtype=torch.float32)
        correction = delta.index_select(0, safe.reshape(-1)).reshape(
            *safe.shape, delta.shape[-1]
        )
        correction = correction * mask.to(output.device).unsqueeze(-1)
        return output + correction.to(dtype=output.dtype)

    return input_layer.register_forward_hook(hook)


@torch.no_grad()
def materialize_input_delta(
    input_layer: nn.Module,
    row_ids: Sequence[int],
    delta: torch.Tensor,
) -> None:
    if len(row_ids) != int(delta.shape[0]):
        raise ValueError("embedding row count does not match delta")
    if not row_ids:
        return
    ids = torch.tensor(
        [int(x) for x in row_ids], dtype=torch.long, device=input_layer.weight.device
    )
    current = input_layer.weight.index_select(0, ids)
    input_layer.weight.index_copy_(
        0,
        ids,
        current + delta.to(device=current.device, dtype=current.dtype),
    )


def _direct_margins(model, tok, records, device, llama_like, batch_size):
    instances = stage2.mcf_instances(records)
    return stage2.mcf_direct_margins(
        model,
        tok,
        instances,
        device,
        llama_like,
        batch_size,
        sensitive_field="target_true",
        reference_field="target_new",
    )


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, manifest = validate_locked(
        visible_path, manifest_path, int(a.seed), int(a.forget_num)
    )

    synthetic_records = synth.build_synthetic_records(
        records, count=int(a.synthetic_paraphrases_per_record)
    )
    all_records = list(records) + synthetic_records
    synthetic_coverage = synth.coverage_report(records)
    if int(a.synthetic_paraphrases_per_record) > 0 and synthetic_coverage["generic_fallback_records"]:
        print(
            "WARNING: "
            f"{synthetic_coverage['generic_fallback_records']}/{len(records)} records "
            "fell back to the generic synthetic-paraphrase templates (relation_id "
            "missing or unrecognized): "
            f"{synthetic_coverage['generic_fallback_relation_ids']}. The 34-relation "
            "hand-authored bank is not being exercised for these records."
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
    input_layer = model.get_input_embeddings()
    if input_layer is None:
        raise RuntimeError("model has no input embedding layer")
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("embedding and LM head must be untied before Stage 1")
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tok)

    # Direction fitting, GA training, and base-logit caching all use
    # all_records (locked direct prompts + hand-authored synthetic
    # paraphrase templates), so the fitted per-row direction is not tied to
    # one single prompt template.
    sensitive_cases = context.expand_answer_field_cases(
        all_records, tok, field="target_true", llama_like=llama_like
    )
    reference_cases = context.expand_answer_field_cases(
        all_records, tok, field="target_new", llama_like=llama_like
    )
    sensitive_tids_all = core.official_target_ids(
        tok, sensitive_cases, llama_like=llama_like, device=device
    )
    selected_ids = sorted(
        set(int(x) for x in sensitive_tids_all.detach().cpu().tolist())
        - set(int(x) for x in gagd.special_token_ids(tok))
    )
    if not selected_ids:
        raise RuntimeError("No sensitive target_true token rows selected")

    max_rank = None if int(a.direction_rank) == 0 else int(a.direction_rank)
    bases, direction_reports = build_row_specific_contrast_bases(
        model,
        tok,
        output_layer,
        sensitive_cases,
        reference_cases,
        selected_ids,
        llama_like=llama_like,
        device=device,
        batch_size=int(a.cache_batch_size),
        max_rank=max_rank,
    )

    emb_delta = context.RowSpecificProjectedDelta(
        selected_ids, bases, device=input_layer.weight.device
    )
    head_delta = context.RowSpecificProjectedDelta(
        selected_ids, bases, device=output_layer.weight.device
    )

    # Base teacher distribution is cached before either virtual delta is active.
    base_sensitive_logits = core.cache_base_logits(
        model,
        tok,
        sensitive_cases,
        device,
        batch_size=int(a.cache_batch_size),
    )

    parameters = list(emb_delta.parameters()) + list(head_delta.parameters())
    opt = torch.optim.AdamW(parameters, lr=float(a.lr), weight_decay=0.0)
    sampler = core.IndexSampler(len(sensitive_cases), int(a.batch_size), int(a.seed))

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_hook = register_input_embedding_delta_hook(
        input_layer, selected_ids, emb_delta.effective_delta
    )
    head_hook = core.register_output_delta_hook(
        output_layer, selected_ids, head_delta.effective_delta
    )
    try:
        model.eval()
        with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
            for step in range(1, int(a.steps) + 1):
                idx = sampler.next()
                batch = [sensitive_cases[i] for i in idx]
                opt.zero_grad(set_to_none=True)

                logits = core.forward_last_logits(model, tok, batch, device)
                tids = core.official_target_ids(
                    tok, batch, llama_like=llama_like, device=device
                )
                ga = core.ga_sensitive_logprob(logits, tids)
                dist_kl = core.gd_non_sensitive_kl(
                    logits,
                    base_sensitive_logits[idx],
                    tids,
                )
                emb_now = emb_delta.effective_delta()
                head_now = head_delta.effective_delta()
                l2 = emb_now.square().mean() + head_now.square().mean()
                total = (
                    float(a.ga_weight) * ga
                    + float(a.distribution_kl_weight) * dist_kl
                    + float(a.delta_l2) * l2
                )
                if not torch.isfinite(total):
                    raise FloatingPointError(
                        f"Non-finite directional Emb+LM Stage-1 loss at step {step}"
                    )
                total.backward()
                grad_norm = None
                if float(a.grad_clip) > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters, float(a.grad_clip)
                    )
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError(
                            f"Non-finite gradient norm at step {step}"
                        )
                opt.step()

                if step == 1 or step % 25 == 0 or step == int(a.steps):
                    emb_grad_rows = 0
                    for coeff in emb_delta.coefficients:
                        if coeff.grad is not None and bool(coeff.grad.detach().abs().sum() > 0):
                            emb_grad_rows += 1
                    row = {
                        "step": int(step),
                        "total_loss": float(total.detach().cpu()),
                        "ga_sensitive_logprob": float(ga.detach().cpu()),
                        "gd_non_sensitive_distribution_kl": float(dist_kl.detach().cpu()),
                        "delta_l2": float(l2.detach().cpu()),
                        "embedding_delta_norm": float(
                            emb_delta.effective_delta().detach().norm().cpu()
                        ),
                        "lm_head_delta_norm": float(
                            head_delta.effective_delta().detach().norm().cpu()
                        ),
                        "embedding_rows_with_nonzero_current_grad": int(emb_grad_rows),
                        "benchmark_retain_seen": 0,
                        "heldout_probes_seen": 0,
                        "lora_used": False,
                    }
                    if grad_norm is not None:
                        row["grad_norm"] = float(grad_norm.detach().cpu())
                    log_f.write(json.dumps(row) + "\n")
                    log_f.flush()
    finally:
        head_hook.remove()
        emb_hook.remove()
    del opt

    trained_emb = emb_delta.effective_delta().detach().clone()
    trained_head = head_delta.effective_delta().detach().clone()

    direct_count = len(records)
    scales = core.parse_scales(a.candidate_scales)
    scale_reports: List[Dict[str, Any]] = []
    for scale in scales:
        emb_handle = register_input_embedding_delta_hook(
            input_layer,
            selected_ids,
            lambda scale=scale: trained_emb * float(scale),
        )
        head_handle = core.register_output_delta_hook(
            output_layer,
            selected_ids,
            lambda scale=scale: trained_head * float(scale),
        )
        try:
            # Combined direct+synthetic margins drive scale selection so the
            # chosen scale is required to work on both, not only the single
            # literal direct prompt.
            margins = _direct_margins(
                model,
                tok,
                all_records,
                device,
                llama_like,
                int(a.cache_batch_size),
            )
        finally:
            head_handle.remove()
            emb_handle.remove()
        direct_margins = margins[:direct_count]
        synthetic_margins = margins[direct_count:]
        scale_reports.append(
            {
                "scale": float(scale),
                "direct_failures": int(
                    (margins < float(a.stage1_constraint_margin)).sum().item()
                ),
                "minimum_margin": float(margins.min().detach().cpu()),
                "direct_only_failures": int(
                    (direct_margins < float(a.stage1_constraint_margin)).sum().item()
                ),
                "direct_only_minimum_margin": float(
                    direct_margins.min().detach().cpu()
                ),
                "synthetic_failures": (
                    int((synthetic_margins < float(a.stage1_constraint_margin)).sum().item())
                    if synthetic_margins.numel()
                    else 0
                ),
                "synthetic_minimum_margin": (
                    float(synthetic_margins.min().detach().cpu())
                    if synthetic_margins.numel()
                    else None
                ),
                "embedding_delta_norm": float(trained_emb.norm().cpu() * scale),
                "lm_head_delta_norm": float(trained_head.norm().cpu() * scale),
            }
        )

    selected_scale = core.choose_scale(scale_reports)
    final_emb = trained_emb * float(selected_scale)
    final_head = trained_head * float(selected_scale)
    materialize_input_delta(input_layer, selected_ids, final_emb)
    core.materialize_output_delta(output_layer, selected_ids, final_head)

    final_all_margins = _direct_margins(
        model,
        tok,
        all_records,
        device,
        llama_like,
        int(a.cache_batch_size),
    )
    final_margins = final_all_margins[:direct_count]
    final_synthetic_margins = final_all_margins[direct_count:]
    final_failures = [
        int(i)
        for i, value in enumerate(final_margins.detach().cpu().tolist())
        if float(value) < float(a.stage1_constraint_margin)
    ]
    final_synthetic_failures = [
        int(i)
        for i, value in enumerate(final_synthetic_margins.detach().cpu().tolist())
        if float(value) < float(a.stage1_constraint_margin)
    ]

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    config: Dict[str, Any] = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "dataset": "mcf",
        "source_protocol": manifest.get("protocol"),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "target_contract": {
            "sensitive_answer": "requested_rewrite.target_true",
            "non_sensitive_reference": "requested_rewrite.target_new",
            "field_swapping": False,
        },
        "lm_head_untied_before_training": True,
        "input_output_tied_after_training": False,
        "transformer_trainable_parameters": 0,
        "editable_embedding_rows": "target_true_sensitive_token_rows_only",
        "editable_lm_head_rows": "target_true_sensitive_token_rows_only",
        "selected_token_ids": selected_ids,
        "selected_row_count": len(selected_ids),
        "direction_definition": (
            "matched h_target_true - h_target_new; decoder row "
            "w_target_true - w_target_new fallback when hidden contrast is zero"
        ),
        "direction_rank_cap": int(a.direction_rank),
        "direction_reports": direction_reports,
        "synthetic_paraphrases_per_record": int(a.synthetic_paraphrases_per_record),
        "synthetic_record_count": len(synthetic_records),
        "synthetic_paraphrase_source": (
            "hand-authored per-relation-id alternate templates + generic "
            "context prefixes; never derived from real paraphrase_prompts"
        ),
        "synthetic_paraphrase_coverage": synthetic_coverage,
        "embedding_trainable_parameters": int(emb_delta.trainable_parameter_count),
        "lm_head_trainable_parameters": int(head_delta.trainable_parameter_count),
        "ga_loss": "mean(log p(target_true sensitive token)); minimized",
        "target_new_training_role": "direction construction and direct margin only; no GD/CE",
        "distribution_kl_loss": (
            "KL(base_non_sensitive || current_non_sensitive) with active sensitive token removed"
        ),
        "ga_weight": float(a.ga_weight),
        "distribution_kl_weight": float(a.distribution_kl_weight),
        "delta_l2": float(a.delta_l2),
        "steps": int(a.steps),
        "batch_size": int(a.batch_size),
        "lr": float(a.lr),
        "selected_scale": float(selected_scale),
        "scale_reports": scale_reports,
        "stage1_constraint_margin": float(a.stage1_constraint_margin),
        "stage1_direct_failures": len(final_failures),
        "stage1_failing_positions": final_failures,
        "stage1_minimum_margin": float(final_margins.min().detach().cpu()),
        "stage1_synthetic_failures": len(final_synthetic_failures),
        "stage1_synthetic_failing_positions": final_synthetic_failures,
        "stage1_synthetic_minimum_margin": (
            float(final_synthetic_margins.min().detach().cpu())
            if final_synthetic_margins.numel()
            else None
        ),
        "stage1_combined_failures": len(final_failures) + len(final_synthetic_failures),
        "stage1_combined_minimum_margin": float(
            final_all_margins.min().detach().cpu()
        ),
        "final_embedding_delta_norm": float(final_emb.norm().cpu()),
        "final_lm_head_delta_norm": float(final_head.norm().cpu()),
        "lora_used": False,
        "official_paraphrases_seen": 0,
        "official_neighborhood_seen": 0,
        "benchmark_retain_seen": 0,
        "ppl_eval_text_seen": 0,
    }
    core.write_json(out_dir / "stage1_config.json", config)
    print(json.dumps(config, indent=2))
    print(f"Stage-1 checkpoint: {ckpt}")
    print(
        f"Stage-1 direct failures: {len(final_failures)}/{len(records)}; "
        f"min margin={config['stage1_minimum_margin']:.6f}"
    )
    print(
        f"Stage-1 synthetic-paraphrase failures: {len(final_synthetic_failures)}/"
        f"{len(synthetic_records)}; "
        f"min margin={config['stage1_synthetic_minimum_margin']}"
    )


if __name__ == "__main__":
    main()
