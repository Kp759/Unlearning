#!/usr/bin/env python3
"""Canonical SURE-LM Stage 2 shared by MCF and ZsRE.

Common mechanics:
  * reload the restored Stage-1 checkpoint;
  * detect residual failures using direct training-visible requests only;
  * clone/untie the LM head and freeze the model;
  * select only sensitive-answer LM-head rows;
  * try the same ordered rank candidates (default: 2,8,unrestricted);
  * choose the first candidate with zero direct failures, otherwise the best
    direct-only candidate;
  * run the same direct-only scale sweep and choose the smallest valid scale;
  * materialize only selected output rows and freeze the checkpoint.

Only the benchmark direct constraint differs:
  * MCF: NLL(sensitive)-NLL(reference) >= constraint_margin.
  * ZsRE: sensitive token must lose top-1 by constraint_margin.

Historical MCF runs default to ``target_new`` sensitive and ``target_true``
reference. Explicit field arguments allow target-true-sensitive MCF runs to use
the original unswapped benchmark records.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

import gagd_compare as gagd
import gagd_active_case_repair as mcf_repair
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("mcf", "zsre"), required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument(
        "--mcf-sensitive-field",
        choices=("target_true", "target_new"),
        default="target_new",
        help="MCF answer field whose NLL must be increased",
    )
    p.add_argument(
        "--mcf-reference-field",
        choices=("target_true", "target_new"),
        default="target_true",
        help="MCF non-sensitive answer field that must outrank the sensitive field",
    )
    p.add_argument("--candidate-ranks", default="2,8,0", help="0 means unrestricted full selected-row delta")
    p.add_argument("--repair-steps", type=int, default=800)
    p.add_argument("--repair-lr", type=float, default=5e-3)
    p.add_argument("--constraint-margin", type=float, required=True)
    p.add_argument("--repair-l2", type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--candidate-scales", default="1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def load_locked(
    dataset: str,
    visible_path: Path,
    manifest_path: Path,
    seed: int,
    forget_num: int,
):
    records = json.loads(visible_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != forget_num:
        raise RuntimeError(f"Expected {forget_num} training-visible forget records")
    if int(manifest.get("seed", -1)) != seed:
        raise RuntimeError("Split manifest seed mismatch")
    sampling = manifest.get("sampling", {})
    if int(sampling.get("forget_num", -1)) != forget_num:
        raise RuntimeError("Split manifest forget count mismatch")
    expected = [int(x) for x in sampling.get("forget_case_ids", [])]
    actual = [int(r.get("case_id", -1)) for r in records]
    if expected and expected != actual:
        raise RuntimeError("Training-visible IDs do not match split manifest")
    for index, record in enumerate(records):
        if record.get("paraphrase_prompts") or record.get("neighborhood_prompts"):
            raise RuntimeError(f"Record {index} exposes held-out probes")
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping):
            raise RuntimeError(f"Record {index} lacks requested_rewrite")
        if dataset == "mcf":
            if not rr.get("target_new", {}).get("str") or not rr.get("target_true", {}).get("str"):
                raise RuntimeError("MCF Stage 2 requires direct target_new and target_true")
        else:
            if not rr.get("target_true", {}).get("str"):
                raise RuntimeError("ZsRE Stage 2 requires target_true")
            if "target_new" in rr:
                raise RuntimeError("Canonical ZsRE Stage 2 forbids target_new/neutral targets")
    return records, manifest


def _optimizer(module: torch.nn.Module, lr: float):
    return torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=0.0)


def _rank_basis(active_hidden: torch.Tensor, requested_rank: int):
    if requested_rank == 0:
        return None, 0
    basis = core.orthonormal_row_basis(active_hidden, max_rank=requested_rank)
    actual = int(basis.shape[0])
    if actual <= 0:
        raise RuntimeError("Active hidden directions have zero numerical rank")
    return basis, actual


def _candidate_key(report: Dict[str, Any], order: int) -> Tuple[int, int, float]:
    return (int(report["direct_failures"]), int(order), float(report["delta_norm"]))


def resolve_mcf_fields(sensitive_field: str, reference_field: str) -> Tuple[str, str]:
    allowed = {"target_true", "target_new"}
    if sensitive_field not in allowed or reference_field not in allowed:
        raise ValueError("MCF fields must be target_true or target_new")
    if sensitive_field == reference_field:
        raise ValueError("MCF sensitive and reference fields must differ")
    return sensitive_field, reference_field


def mcf_instances(records: Sequence[Mapping[str, Any]]) -> List[mcf_repair.MCFPromptInstance]:
    instances: List[mcf_repair.MCFPromptInstance] = []
    for position, record in enumerate(records):
        rr = record["requested_rewrite"]
        subject = str(rr["subject"])
        instances.append(
            mcf_repair.MCFPromptInstance(
                record_index=int(record.get("case_id", position)),
                sampled_position=position,
                prompt_type="rewrite",
                prompt_index=0,
                prompt=str(rr["prompt"]).format(subject),
                target_new=str(rr["target_new"]["str"]),
                target_true=str(rr["target_true"]["str"]),
            )
        )
    return instances


@torch.no_grad()
def mcf_direct_margins(
    model,
    tok,
    instances,
    device,
    llama_like,
    batch_size,
    sensitive_field: str = "target_new",
    reference_field: str = "target_true",
) -> torch.Tensor:
    sensitive_field, reference_field = resolve_mcf_fields(
        sensitive_field, reference_field
    )
    values: List[torch.Tensor] = []
    for start in range(0, len(instances), batch_size):
        new_nll, true_nll = mcf_repair.official_prompt_instance_nll_tensors(
            model,
            tok,
            instances[start : start + batch_size],
            device,
            llama_like,
        )
        nll = {"target_new": new_nll, "target_true": true_nll}
        values.append((nll[sensitive_field] - nll[reference_field]).float())
    return torch.cat(values, dim=0) if values else torch.empty(0, device=device)


def mcf_sensitive_rows(
    tok,
    instances,
    active_positions,
    sensitive_field: str = "target_new",
) -> List[int]:
    if sensitive_field not in {"target_true", "target_new"}:
        raise ValueError("MCF sensitive field must be target_true or target_new")
    selected: set[int] = set()
    for position in active_positions:
        text = gagd.normalize_answer(getattr(instances[position], sensitive_field))
        selected.update(gagd.token_ids_for_text(tok, text))
    selected -= gagd.special_token_ids(tok)
    return sorted(int(x) for x in selected)


def optimize_mcf_candidate(
    *,
    rank: int,
    active_hidden: torch.Tensor,
    selected_ids: Sequence[int],
    output_layer,
    caches,
    required_margin: float,
    repair_steps: int,
    repair_lr: float,
    repair_l2: float,
    check_every: int,
    order: int,
    sensitive_field: str = "target_new",
    reference_field: str = "target_true",
):
    basis, actual_rank = _rank_basis(active_hidden, rank)
    delta_module = core.SelectedRowDelta(
        len(selected_ids),
        output_layer.weight.shape[1],
        direction_basis=basis,
        device=output_layer.weight.device,
    )
    opt = _optimizer(delta_module, repair_lr)
    best_failures = 10**9
    best_step = 0
    best_delta = delta_module.effective_delta().detach().clone()
    logs: List[Dict[str, Any]] = []

    for step in range(1, repair_steps + 1):
        opt.zero_grad(set_to_none=True)
        delta = delta_module.effective_delta()
        margins = mcf_margins_from_delta_caches(
            caches,
            delta,
            sensitive_field=sensitive_field,
            reference_field=reference_field,
        )
        hinge = F.relu(float(required_margin) - margins).square().mean()
        l2 = delta.square().mean()
        loss = hinge + repair_l2 * l2
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite MCF repair loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(delta_module.parameters()), 1.0)
        opt.step()

        if step == 1 or step % check_every == 0 or step == repair_steps:
            with torch.no_grad():
                current = delta_module.effective_delta()
                current_margins = mcf_margins_from_delta_caches(
                    caches,
                    current,
                    sensitive_field=sensitive_field,
                    reference_field=reference_field,
                )
                failures = int((current_margins < required_margin).sum().item())
                row = {
                    "step": step,
                    "direct_failures": failures,
                    "minimum_margin": float(current_margins.min().detach().cpu()),
                    "loss": float(loss.detach().cpu()),
                    "delta_norm": float(current.norm().detach().cpu()),
                }
                logs.append(row)
                if failures < best_failures:
                    best_failures = failures
                    best_step = step
                    best_delta = current.detach().clone()
                if failures == 0:
                    best_failures = 0
                    best_step = step
                    best_delta = current.detach().clone()
                    break
    del opt
    with torch.no_grad():
        final_margins = mcf_margins_from_delta_caches(
            caches,
            best_delta,
            sensitive_field=sensitive_field,
            reference_field=reference_field,
        )
    report = {
        "requested_rank": int(rank),
        "actual_rank": int(actual_rank) if rank > 0 else None,
        "parameterization": "unrestricted_selected_rows" if rank == 0 else "fixed_hidden_direction_basis",
        "trainable_parameters": int(delta_module.trainable_parameter_count),
        "best_step": int(best_step),
        "direct_failures": int((final_margins < required_margin).sum().item()),
        "minimum_margin": float(final_margins.min().detach().cpu()),
        "delta_norm": float(best_delta.norm().detach().cpu()),
        "candidate_order": int(order),
        "logs": logs,
    }
    return report, best_delta


def mcf_margins_from_delta_caches(
    caches,
    delta_rows: torch.Tensor,
    *,
    sensitive_field: str = "target_new",
    reference_field: str = "target_true",
) -> torch.Tensor:
    """Return positive margins when the MCF sensitive answer is disfavored."""

    sensitive_field, reference_field = resolve_mcf_fields(
        sensitive_field, reference_field
    )
    return torch.stack(
        [
            mcf_repair.answer_nll_from_delta_cache(
                getattr(cache, sensitive_field), delta_rows
            )
            - mcf_repair.answer_nll_from_delta_cache(
                getattr(cache, reference_field), delta_rows
            )
            for cache in caches
        ]
    )


def zsre_flags(model, tok, cases, llama_like, device, batch_size) -> List[bool]:
    flags: List[bool] = []
    with torch.no_grad():
        for start in range(0, len(cases), batch_size):
            batch = cases[start : start + batch_size]
            logits = core.forward_last_logits(model, tok, batch, device)
            tids = core.official_target_ids(tok, batch, llama_like=llama_like, device=device)
            flags.extend(bool(x) for x in (logits.argmax(dim=-1) == tids).detach().cpu().tolist())
    return flags


def zsre_margin_loss(logits: torch.Tensor, tids: torch.Tensor, margin: float) -> torch.Tensor:
    rows = torch.arange(logits.shape[0], device=logits.device)
    sensitive = logits[rows, tids].float()
    detached = logits.detach().float().clone()
    detached[rows, tids] = -torch.inf
    best_other = detached.max(dim=-1).values
    return F.relu(sensitive - best_other + float(margin)).mean()


def optimize_zsre_candidate(
    *,
    rank: int,
    active_hidden: torch.Tensor,
    selected_ids: Sequence[int],
    output_layer,
    model,
    tok,
    all_cases,
    active_cases,
    llama_like,
    device,
    constraint_margin: float,
    repair_steps: int,
    repair_lr: float,
    repair_l2: float,
    batch_size: int,
    check_every: int,
    seed: int,
    order: int,
):
    basis, actual_rank = _rank_basis(active_hidden, rank)
    delta_module = core.SelectedRowDelta(
        len(selected_ids),
        output_layer.weight.shape[1],
        direction_basis=basis,
        device=output_layer.weight.device,
    )
    opt = _optimizer(delta_module, repair_lr)
    sampler = core.IndexSampler(len(active_cases), batch_size, seed + 100003)
    best_failures = 10**9
    best_step = 0
    best_delta = delta_module.effective_delta().detach().clone()
    logs: List[Dict[str, Any]] = []
    handle = core.register_output_delta_hook(
        output_layer, selected_ids, delta_module.effective_delta
    )
    try:
        for step in range(1, repair_steps + 1):
            idx = sampler.next()
            batch = [active_cases[i] for i in idx]
            opt.zero_grad(set_to_none=True)
            logits = core.forward_last_logits(model, tok, batch, device)
            tids = core.official_target_ids(tok, batch, llama_like=llama_like, device=device)
            margin_loss = zsre_margin_loss(logits, tids, constraint_margin)
            delta = delta_module.effective_delta()
            loss = margin_loss + repair_l2 * delta.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite ZsRE repair loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(delta_module.parameters()), 1.0)
            opt.step()

            if step == 1 or step % check_every == 0 or step == repair_steps:
                failures = sum(zsre_flags(model, tok, all_cases, llama_like, device, batch_size))
                current = delta_module.effective_delta().detach()
                row = {
                    "step": step,
                    "direct_failures": int(failures),
                    "loss": float(loss.detach().cpu()),
                    "delta_norm": float(current.norm().cpu()),
                }
                logs.append(row)
                if failures < best_failures:
                    best_failures = int(failures)
                    best_step = step
                    best_delta = current.clone()
                if failures == 0:
                    best_failures = 0
                    best_step = step
                    best_delta = current.clone()
                    break
    finally:
        handle.remove()
    del opt
    report = {
        "requested_rank": int(rank),
        "actual_rank": int(actual_rank) if rank > 0 else None,
        "parameterization": "unrestricted_selected_rows" if rank == 0 else "fixed_hidden_direction_basis",
        "trainable_parameters": int(delta_module.trainable_parameter_count),
        "best_step": int(best_step),
        "direct_failures": int(best_failures),
        "delta_norm": float(best_delta.norm().cpu()),
        "candidate_order": int(order),
        "logs": logs,
    }
    return report, best_delta


def main() -> None:
    a = parse_args()
    if a.forget_num <= 0 or a.repair_steps <= 0 or a.repair_lr <= 0:
        raise ValueError("forget-num, repair-steps, and repair-lr must be positive")
    if a.batch_size <= 0 or a.check_every <= 0:
        raise ValueError("batch-size and check-every must be positive")
    if a.constraint_margin < 0 or a.repair_l2 < 0:
        raise ValueError("constraint margin and repair L2 must be non-negative")
    if a.dataset == "mcf":
        mcf_sensitive_field, mcf_reference_field = resolve_mcf_fields(
            a.mcf_sensitive_field, a.mcf_reference_field
        )
    else:
        mcf_sensitive_field, mcf_reference_field = "target_true", "target_new"

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    ranks = core.parse_rank_list(a.candidate_ranks)
    scales = core.parse_scales(a.candidate_scales)
    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, manifest = load_locked(
        a.dataset, visible_path, manifest_path, a.seed, a.forget_num
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
    llama_like = is_llama_like(model, tok)

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_reports: List[Dict[str, Any]] = []
    candidate_deltas: List[torch.Tensor] = []
    selected_ids: List[int] = []
    active_before = 0
    direct_total = 0
    final_failures = 0
    chosen_delta: torch.Tensor

    if a.dataset == "mcf":
        instances = mcf_instances(records)
        direct_total = len(instances)
        original_margins = mcf_direct_margins(
            model,
            tok,
            instances,
            device,
            llama_like,
            a.batch_size,
            mcf_sensitive_field,
            mcf_reference_field,
        )
        active_positions = [
            i for i, value in enumerate(original_margins.detach().cpu().tolist())
            if float(value) < a.constraint_margin
        ]
        active_before = len(active_positions)
        selected_ids = mcf_sensitive_rows(
            tok, instances, active_positions, mcf_sensitive_field
        )

        if selected_ids:
            caches = mcf_repair.build_prompt_instance_delta_caches(
                model, tok, instances, selected_ids, device, a.batch_size, llama_like
            )
            active_hidden = torch.cat(
                [
                    answer.hidden
                    for position in active_positions
                    for answer in (caches[position].target_new, caches[position].target_true)
                ],
                dim=0,
            )
            for order, rank in enumerate(ranks):
                report, delta = optimize_mcf_candidate(
                    rank=rank,
                    active_hidden=active_hidden,
                    selected_ids=selected_ids,
                    output_layer=output_layer,
                    caches=caches,
                    required_margin=a.constraint_margin,
                    repair_steps=a.repair_steps,
                    repair_lr=a.repair_lr,
                    repair_l2=a.repair_l2,
                    check_every=a.check_every,
                    order=order,
                    sensitive_field=mcf_sensitive_field,
                    reference_field=mcf_reference_field,
                )
                candidate_reports.append(report)
                candidate_deltas.append(delta)
                print("MCF repair candidate", {k: report[k] for k in ("requested_rank", "actual_rank", "direct_failures", "delta_norm")})
                if report["direct_failures"] == 0:
                    break

            chosen_index = min(
                range(len(candidate_reports)),
                key=lambda i: _candidate_key(candidate_reports[i], i),
            )
            chosen_delta = candidate_deltas[chosen_index]
            margin_fn = lambda scale: mcf_margins_from_delta_caches(
                caches,
                chosen_delta * float(scale),
                sensitive_field=mcf_sensitive_field,
                reference_field=mcf_reference_field,
            )
            scale_reports: List[Dict[str, Any]] = []
            for scale in scales:
                margins = margin_fn(scale)
                scale_reports.append(
                    {
                        "scale": float(scale),
                        "direct_failures": int((margins < a.constraint_margin).sum().item()),
                        "minimum_margin": float(margins.min().detach().cpu()),
                        "effective_delta_norm": float(chosen_delta.norm().cpu() * scale),
                    }
                )
            selected_scale = core.choose_scale(scale_reports)
            final_delta = chosen_delta * selected_scale
            core.materialize_output_delta(output_layer, selected_ids, final_delta)
            final_margins = mcf_direct_margins(
                model,
                tok,
                instances,
                device,
                llama_like,
                a.batch_size,
                mcf_sensitive_field,
                mcf_reference_field,
            )
            final_failures = int((final_margins < a.constraint_margin).sum().item())
        else:
            chosen_index = None
            chosen_delta = torch.empty((0, output_layer.weight.shape[1]), device=output_layer.weight.device)
            selected_scale = 0.0
            scale_reports = []
            final_failures = active_before

    else:
        all_cases = core.expand_sensitive_cases(
            records, tok, dataset="zsre", llama_like=llama_like
        )
        direct_total = len(all_cases)
        before_flags = zsre_flags(
            model, tok, all_cases, llama_like, device, a.batch_size
        )
        active_cases = [c for c, failed in zip(all_cases, before_flags) if failed]
        active_before = len(active_cases)
        if active_cases:
            tids = core.official_target_ids(
                tok, active_cases, llama_like=llama_like, device=device
            )
            selected_ids = sorted(set(int(x) for x in tids.detach().cpu().tolist()))
            active_hidden = core.forward_last_hidden(
                model, tok, active_cases, device, a.batch_size
            )
            for order, rank in enumerate(ranks):
                report, delta = optimize_zsre_candidate(
                    rank=rank,
                    active_hidden=active_hidden,
                    selected_ids=selected_ids,
                    output_layer=output_layer,
                    model=model,
                    tok=tok,
                    all_cases=all_cases,
                    active_cases=active_cases,
                    llama_like=llama_like,
                    device=device,
                    constraint_margin=a.constraint_margin,
                    repair_steps=a.repair_steps,
                    repair_lr=a.repair_lr,
                    repair_l2=a.repair_l2,
                    batch_size=a.batch_size,
                    check_every=a.check_every,
                    seed=a.seed,
                    order=order,
                )
                candidate_reports.append(report)
                candidate_deltas.append(delta)
                print("ZsRE repair candidate", {k: report[k] for k in ("requested_rank", "actual_rank", "direct_failures", "delta_norm")})
                if report["direct_failures"] == 0:
                    break

            chosen_index = min(
                range(len(candidate_reports)),
                key=lambda i: _candidate_key(candidate_reports[i], i),
            )
            chosen_delta = candidate_deltas[chosen_index]
            scale_reports = []
            for scale in scales:
                handle = core.register_output_delta_hook(
                    output_layer,
                    selected_ids,
                    lambda scale=scale: chosen_delta * float(scale),
                )
                try:
                    failures = sum(
                        zsre_flags(model, tok, all_cases, llama_like, device, a.batch_size)
                    )
                finally:
                    handle.remove()
                scale_reports.append(
                    {
                        "scale": float(scale),
                        "direct_failures": int(failures),
                        "effective_delta_norm": float(chosen_delta.norm().cpu() * scale),
                    }
                )
            selected_scale = core.choose_scale(scale_reports)
            final_delta = chosen_delta * selected_scale
            core.materialize_output_delta(output_layer, selected_ids, final_delta)
            final_failures = sum(
                zsre_flags(model, tok, all_cases, llama_like, device, a.batch_size)
            )
        else:
            chosen_index = None
            chosen_delta = torch.empty((0, output_layer.weight.shape[1]), device=output_layer.weight.device)
            selected_scale = 0.0
            scale_reports = []
            final_failures = 0

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    chosen_report = (
        candidate_reports[chosen_index]
        if chosen_index is not None and candidate_reports
        else None
    )
    summary = {
        "schema_version": 2,
        "method": "SURE-LM-canonical-shared-sparse-row-repair",
        "dataset": a.dataset,
        "protocol": "sure_canonical_locked_direct_only",
        "source_protocol": manifest.get("protocol"),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "direct_unit": "record_margin" if a.dataset == "mcf" else "sensitive_prediction_case",
        "direct_total": int(direct_total),
        "active_before": int(active_before),
        "active_after": int(final_failures),
        "constraint_margin": float(a.constraint_margin),
        "selected_rows_semantics": "sensitive_answer_rows_only",
        "sensitive_answer_field": (
            mcf_sensitive_field
            if a.dataset == "mcf"
            else core.sensitive_answer_field(a.dataset)
        ),
        "reference_answer_field": (
            mcf_reference_field if a.dataset == "mcf" else None
        ),
        "selected_lm_head_rows": len(selected_ids),
        "selected_token_ids": selected_ids,
        "transformer_trainable": 0,
        "input_embeddings_modified": False,
        "lm_head_untied_before_repair": True,
        "candidate_ranks": ranks,
        "rank_selection_rule": "ordered lowest-complexity candidate with zero direct failures; otherwise minimize (direct_failures, candidate_order, delta_norm)",
        "candidate_reports": candidate_reports,
        "chosen_candidate": chosen_report,
        "candidate_scales": scales,
        "scale_selection_rule": "smallest scale with zero direct failures; otherwise minimize (direct_failures, scale)",
        "scale_reports": scale_reports,
        "selected_scale": float(selected_scale),
        "effective_delta_norm": float(chosen_delta.norm().cpu() * selected_scale) if chosen_delta.numel() else 0.0,
        "repair_steps": int(a.repair_steps),
        "repair_lr": float(a.repair_lr),
        "repair_l2": float(a.repair_l2),
        "batch_size": int(a.batch_size),
        "benchmark_retain_seen": 0,
        "heldout_paraphrases_or_rephrases_seen": 0,
        "locality_or_neighborhood_seen": 0,
        "PPL_seen": False,
        "selection_uses_heldout": False,
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "repair_summary.json", summary)
    core.write_json(out_dir / "rank_candidates.json", candidate_reports)
    core.write_json(out_dir / "scale_sweep_direct_only.json", scale_reports)
    print(
        f"Canonical Stage 2 {a.dataset}: direct failures {active_before} -> {final_failures}; "
        f"selected rows={len(selected_ids)}; selected scale={selected_scale:g}"
    )
    if final_failures != 0:
        print("WARNING: canonical repair finished with residual direct failures")


if __name__ == "__main__":
    main()
