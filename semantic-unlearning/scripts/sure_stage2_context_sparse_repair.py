#!/usr/bin/env python3
"""Context-conditioned canonical SURE Stage 2 for target-true-sensitive MCF.

The Stage-1 checkpoint already has Base input embeddings/frozen transformer and
only sensitive output-row changes.  This repair:
  * detects failures using direct training-visible requests only;
  * edits only canonical target_new rows (ORIGINAL target_true sensitive rows);
  * constrains each row to its own direct forget-context hidden subspace;
  * tries per-row context-rank caps 2 -> 8 -> full-context by default;
  * combines margin repair with explicit GD on the non-sensitive/reference
    canonical target_true answer;
  * chooses the smallest direct-only scale satisfying the forget constraint.

No held-out paraphrases, neighborhoods, retain examples, or PPL text are used.
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
import sure_context_projection as context


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument(
        "--candidate-ranks",
        default="2,8,0",
        help="Per-row context-rank caps; 0 means full observed forget-context rank.",
    )
    p.add_argument("--repair-steps", type=int, default=800)
    p.add_argument("--repair-lr", type=float, default=5e-3)
    p.add_argument("--constraint-margin", type=float, default=0.25)
    p.add_argument("--reference-gd-weight", type=float, default=1.0)
    p.add_argument("--repair-l2", type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument(
        "--candidate-scales",
        default="1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0",
    )
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def load_locked(visible_path: Path, manifest_path: Path, seed: int, forget_num: int):
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
    semantics = manifest.get("target_semantics", {})
    if semantics.get("original_sensitive_field") != "target_true":
        raise RuntimeError("Stage 2 requires target-true-sensitive adapter")
    for index, record in enumerate(records):
        if (
            record.get("paraphrase_prompts")
            or record.get("neighborhood_prompts")
            or record.get("generation_prompts")
        ):
            raise RuntimeError(f"Record {index} exposes held-out probes")
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping):
            raise RuntimeError(f"Record {index} lacks requested_rewrite")
        if not rr.get("target_new", {}).get("str") or not rr.get("target_true", {}).get("str"):
            raise RuntimeError(f"Record {index} lacks direct target fields")
    return records, manifest


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
def direct_margins(model, tok, instances, device, llama_like, batch_size):
    values: List[torch.Tensor] = []
    for start in range(0, len(instances), batch_size):
        new_nll, true_nll = mcf_repair.official_prompt_instance_nll_tensors(
            model, tok, instances[start : start + batch_size], device, llama_like
        )
        values.append((new_nll - true_nll).float())
    return torch.cat(values, dim=0) if values else torch.empty(0, device=device)


def reference_nll_from_caches(caches, delta_rows: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            mcf_repair.answer_nll_from_delta_cache(cache.target_true, delta_rows)
            for cache in caches
        ]
    ).mean()


def candidate_key(report: Dict[str, Any], order: int) -> Tuple[int, int, float]:
    return (
        int(report["direct_failures"]),
        int(order),
        float(report["delta_norm"]),
    )


def optimize_candidate(
    *,
    requested_rank: int,
    bases,
    basis_reports,
    selected_ids,
    caches,
    required_margin: float,
    reference_gd_weight: float,
    repair_steps: int,
    repair_lr: float,
    repair_l2: float,
    check_every: int,
    order: int,
    device: torch.device,
):
    delta_module = context.RowSpecificProjectedDelta(
        selected_ids, bases, device=device
    )
    opt = torch.optim.AdamW(delta_module.parameters(), lr=repair_lr, weight_decay=0.0)
    best_failures = 10**9
    best_step = 0
    best_loss = float("inf")
    best_delta = delta_module.effective_delta().detach().clone()
    logs: List[Dict[str, Any]] = []

    for step in range(1, repair_steps + 1):
        opt.zero_grad(set_to_none=True)
        delta = delta_module.effective_delta()
        margins = mcf_repair.margins_from_delta_caches(caches, delta)
        hinge = F.relu(float(required_margin) - margins).square().mean()
        reference_gd = reference_nll_from_caches(caches, delta)
        l2 = delta.square().mean()
        loss = hinge + float(reference_gd_weight) * reference_gd + repair_l2 * l2
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite context repair loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(delta_module.parameters()), 1.0)
        opt.step()

        if step == 1 or step % check_every == 0 or step == repair_steps:
            with torch.no_grad():
                current = delta_module.effective_delta()
                current_margins = mcf_repair.margins_from_delta_caches(caches, current)
                failures = int((current_margins < required_margin).sum().item())
                current_ref_nll = float(
                    reference_nll_from_caches(caches, current).detach().cpu()
                )
                row = {
                    "step": int(step),
                    "direct_failures": failures,
                    "minimum_margin": float(current_margins.min().detach().cpu()),
                    "reference_gd_nll": current_ref_nll,
                    "loss": float(loss.detach().cpu()),
                    "delta_norm": float(current.norm().detach().cpu()),
                }
                logs.append(row)
                current_loss = float(loss.detach().cpu())
                if failures < best_failures or (
                    failures == best_failures and current_loss < best_loss
                ):
                    best_failures = failures
                    best_step = step
                    best_loss = current_loss
                    best_delta = current.detach().clone()
                if failures == 0:
                    break
    del opt

    with torch.no_grad():
        final_margins = mcf_repair.margins_from_delta_caches(caches, best_delta)
        final_ref_nll = reference_nll_from_caches(caches, best_delta)
    report = {
        "requested_context_rank": int(requested_rank),
        "rank_semantics": "0=full_observed_row_specific_forget_context_rank",
        "row_basis_reports": basis_reports,
        "row_context_ranks": [int(x["context_rank"]) for x in basis_reports],
        "trainable_parameters": int(delta_module.trainable_parameter_count),
        "best_step": int(best_step),
        "direct_failures": int((final_margins < required_margin).sum().item()),
        "minimum_margin": float(final_margins.min().detach().cpu()),
        "reference_gd_nll": float(final_ref_nll.detach().cpu()),
        "delta_norm": float(best_delta.norm().detach().cpu()),
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
    if a.constraint_margin < 0 or a.reference_gd_weight < 0 or a.repair_l2 < 0:
        raise ValueError("margin/GD/L2 settings must be non-negative")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    ranks = core.parse_rank_list(a.candidate_ranks)
    scales = core.parse_scales(a.candidate_scales)
    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, manifest = load_locked(
        visible_path, manifest_path, a.seed, a.forget_num
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

    instances = mcf_instances(records)
    original_margins = direct_margins(
        model, tok, instances, device, llama_like, a.batch_size
    )
    active_positions = [
        i
        for i, value in enumerate(original_margins.detach().cpu().tolist())
        if float(value) < a.constraint_margin
    ]
    active_before = len(active_positions)

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_reports: List[Dict[str, Any]] = []
    candidate_deltas: List[torch.Tensor] = []
    selected_ids: List[int] = []
    selected_scale = 0.0
    scale_reports: List[Dict[str, Any]] = []
    chosen_index = None

    if active_positions:
        all_sensitive_cases = context.expand_answer_field_cases(
            records, tok, field="target_new", llama_like=llama_like
        )
        active_set = set(active_positions)
        active_sensitive_cases = [
            case
            for case in all_sensitive_cases
            if int(case.record_position) in active_set
        ]
        active_tids = core.official_target_ids(
            tok, active_sensitive_cases, llama_like=llama_like, device=device
        )
        selected_ids = sorted(set(int(x) for x in active_tids.detach().cpu().tolist()))
        caches = mcf_repair.build_prompt_instance_delta_caches(
            model, tok, instances, selected_ids, device, a.batch_size, llama_like
        )

        for order, rank in enumerate(ranks):
            max_rank = None if rank == 0 else int(rank)
            bases, basis_reports = context.build_row_specific_bases(
                model,
                tok,
                active_sensitive_cases,
                selected_ids=selected_ids,
                llama_like=llama_like,
                device=device,
                batch_size=a.batch_size,
                max_rank=max_rank,
            )
            report, delta = optimize_candidate(
                requested_rank=rank,
                bases=bases,
                basis_reports=basis_reports,
                selected_ids=selected_ids,
                caches=caches,
                required_margin=a.constraint_margin,
                reference_gd_weight=a.reference_gd_weight,
                repair_steps=a.repair_steps,
                repair_lr=a.repair_lr,
                repair_l2=a.repair_l2,
                check_every=a.check_every,
                order=order,
                device=output_layer.weight.device,
            )
            candidate_reports.append(report)
            candidate_deltas.append(delta)
            print(
                "Context repair candidate",
                {
                    "rank": rank,
                    "row_ranks": report["row_context_ranks"],
                    "direct_failures": report["direct_failures"],
                    "reference_gd_nll": report["reference_gd_nll"],
                    "delta_norm": report["delta_norm"],
                },
            )
            if report["direct_failures"] == 0:
                break

        chosen_index = min(
            range(len(candidate_reports)),
            key=lambda i: candidate_key(candidate_reports[i], i),
        )
        chosen_delta = candidate_deltas[chosen_index]
        for scale in scales:
            scaled = chosen_delta * float(scale)
            margins = mcf_repair.margins_from_delta_caches(caches, scaled)
            ref_nll = reference_nll_from_caches(caches, scaled)
            scale_reports.append(
                {
                    "scale": float(scale),
                    "direct_failures": int(
                        (margins < a.constraint_margin).sum().item()
                    ),
                    "minimum_margin": float(margins.min().detach().cpu()),
                    "reference_gd_nll": float(ref_nll.detach().cpu()),
                    "effective_delta_norm": float(chosen_delta.norm().cpu() * scale),
                }
            )
        selected_scale = core.choose_scale(scale_reports)
        final_delta = chosen_delta * float(selected_scale)
        core.materialize_output_delta(output_layer, selected_ids, final_delta)

    final_margins = direct_margins(
        model, tok, instances, device, llama_like, a.batch_size
    )
    final_failures = int((final_margins < a.constraint_margin).sum().item())

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    chosen_report = (
        candidate_reports[chosen_index]
        if chosen_index is not None and candidate_reports
        else None
    )
    summary = {
        "schema_version": 1,
        "method": "SURE-LM-context-conditioned-sensitive-row-repair",
        "dataset": "mcf",
        "protocol": "sure_context_target_true_locked_direct_only",
        "source_protocol": manifest.get("protocol"),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "target_semantics": {
            "canonical_sensitive_slot": "target_new",
            "canonical_reference_slot": "target_true",
            "original_sensitive_field": "target_true",
            "original_reference_field": "target_new",
        },
        "direct_total": len(instances),
        "active_before": int(active_before),
        "active_after": int(final_failures),
        "constraint_margin": float(a.constraint_margin),
        "selected_rows_semantics": "sensitive_answer_rows_only",
        "selected_lm_head_rows": len(selected_ids),
        "selected_token_ids": selected_ids,
        "row_specific_context_projection": True,
        "candidate_ranks": ranks,
        "candidate_rank_semantics": "per-row cap; 0=full observed forget-context rank",
        "candidate_reports": candidate_reports,
        "chosen_candidate": chosen_report,
        "candidate_scales": scales,
        "scale_reports": scale_reports,
        "selected_scale": float(selected_scale),
        "reference_gd_weight": float(a.reference_gd_weight),
        "reference_gd_loss": "teacher-forced canonical target_true NLL over direct requests",
        "reference_answer_rows_editable": False,
        "repair_steps": int(a.repair_steps),
        "repair_lr": float(a.repair_lr),
        "repair_l2": float(a.repair_l2),
        "transformer_trainable": 0,
        "input_embeddings_modified": False,
        "lm_head_untied": True,
        "benchmark_retain_seen": 0,
        "heldout_paraphrases_seen": 0,
        "neighborhood_prompts_seen": 0,
        "PPL_seen": False,
        "selection_uses_heldout": False,
        "minimum_direct_margin_after": float(final_margins.min().detach().cpu()),
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "repair_summary.json", summary)
    core.write_json(out_dir / "rank_candidates.json", candidate_reports)
    core.write_json(out_dir / "scale_sweep_direct_only.json", scale_reports)

    print(
        f"Context Stage 2: direct failures {active_before} -> {final_failures}; "
        f"selected rows={len(selected_ids)}; selected scale={selected_scale:g}"
    )
    if final_failures != 0:
        print("WARNING: context-conditioned repair finished with residual direct failures")


if __name__ == "__main__":
    main()
