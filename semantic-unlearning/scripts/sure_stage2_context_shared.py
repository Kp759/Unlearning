#!/usr/bin/env python3
"""Fixed shared context-conditioned SURE Stage 2 for MCF and ZsRE.

Architecture and objective are identical across benchmarks.  Stage 2:
  * reloads the shared Stage-1 checkpoint;
  * detects residual direct sensitive-token failures using the same universal
    best-non-sensitive minus sensitive logit margin;
  * edits only sensitive LM-head rows;
  * constrains each row to its own direct forget-context hidden subspace;
  * optimizes the SAME GA + non-sensitive-distribution GD + L2 objective used
    in Stage 1, restricted to residual direct cases;
  * tries row-specific context-rank caps 2 -> 8 -> full-context by default;
  * uses the same direct-only scale selection.

No benchmark-specific hinge/ReLU loss is used.  The dataset adapter only tells
us which answer field is sensitive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import sure_context_projection as context
import sure_shared_suppression as shared


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("mcf", "zsre"), required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--base-logits-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--candidate-ranks", default="2,8,0",
                   help="Per-row context-rank caps; 0 means full observed numerical rank.")
    p.add_argument("--repair-steps", type=int, default=800)
    p.add_argument("--repair-lr", type=float, default=5e-3)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--gd-weight", type=float, default=1.0)
    p.add_argument("--repair-l2", type=float, default=1e-6)
    p.add_argument("--constraint-margin", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--candidate-scales", default="1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def load_locked(dataset: str, visible_path: Path, manifest_path: Path, seed: int, forget_num: int):
    records = json.loads(visible_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != forget_num:
        raise RuntimeError(f"Expected {forget_num} direct training-visible forget records")
    if int(manifest.get("seed", -1)) != seed:
        raise RuntimeError("Split manifest seed mismatch")
    expected = [int(x) for x in manifest.get("sampling", {}).get("forget_case_ids", [])]
    actual = [int(r.get("case_id", -1)) for r in records]
    if expected and expected != actual:
        raise RuntimeError("Training-visible IDs do not match split manifest")
    sensitive_field = core.sensitive_answer_field(dataset)
    for i, record in enumerate(records):
        if (record.get("paraphrase_prompts") or record.get("neighborhood_prompts")
                or record.get("generation_prompts")):
            raise RuntimeError(f"Record {i} exposes held-out probes")
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping) or not rr.get(sensitive_field, {}).get("str"):
            raise RuntimeError(f"Record {i} lacks sensitive {sensitive_field}.str")
        if dataset == "zsre" and "target_new" in rr:
            raise RuntimeError("Fixed shared ZsRE architecture forbids target_new/neutral targets")
    return records, manifest


def candidate_key(report: Dict[str, Any], order: int) -> Tuple[int, int, float]:
    return (int(report["direct_failures"]), int(order), float(report["delta_norm"]))


def optimize_candidate(
    *,
    cases,
    active_indices: Sequence[int],
    selected_ids,
    bases,
    basis_reports,
    base_logits: torch.Tensor,
    model,
    tok,
    output_layer,
    llama_like: bool,
    device: torch.device,
    required_margin: float,
    steps: int,
    lr: float,
    ga_weight: float,
    gd_weight: float,
    l2_weight: float,
    batch_size: int,
    check_every: int,
    seed: int,
    requested_rank: int,
    order: int,
):
    active_cases = [cases[i] for i in active_indices]
    active_base_logits = base_logits[list(active_indices)]
    delta_module = context.RowSpecificProjectedDelta(
        selected_ids, bases, device=output_layer.weight.device
    )
    opt = torch.optim.AdamW(delta_module.parameters(), lr=lr, weight_decay=0.0)
    sampler = core.IndexSampler(len(active_cases), batch_size, seed + 100003)
    best_failures = 10**9
    best_loss = float("inf")
    best_step = 0
    best_delta = delta_module.effective_delta().detach().clone()
    logs: List[Dict[str, Any]] = []

    hook = core.register_output_delta_hook(output_layer, selected_ids, delta_module.effective_delta)
    try:
        model.eval()
        for step in range(1, steps + 1):
            local_idx = sampler.next()
            batch = [active_cases[i] for i in local_idx]
            opt.zero_grad(set_to_none=True)
            logits = core.forward_last_logits(model, tok, batch, device)
            tids = core.official_target_ids(tok, batch, llama_like=llama_like, device=device)
            ga = core.ga_sensitive_logprob(logits, tids)
            gd = core.gd_non_sensitive_kl(logits, active_base_logits[local_idx], tids)
            delta = delta_module.effective_delta()
            l2 = delta.square().mean()
            loss = ga_weight * ga + gd_weight * gd + l2_weight * l2
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite shared Stage-2 loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(delta_module.parameters()), 1.0)
            opt.step()

            if step == 1 or step % check_every == 0 or step == steps:
                margins = shared.evaluate_suppression_margins(
                    model, tok, cases, llama_like=llama_like, device=device,
                    batch_size=batch_size,
                )
                failures = shared.count_failures(margins, required_margin)
                cur = delta_module.effective_delta().detach()
                cur_loss = float(loss.detach().cpu())
                row = {
                    "step": int(step),
                    "direct_failures": int(failures),
                    "minimum_suppression_margin": float(margins.min().detach().cpu()),
                    "ga_sensitive_logprob": float(ga.detach().cpu()),
                    "gd_non_sensitive_kl": float(gd.detach().cpu()),
                    "loss": cur_loss,
                    "delta_norm": float(cur.norm().cpu()),
                }
                logs.append(row)
                if failures < best_failures or (failures == best_failures and cur_loss < best_loss):
                    best_failures = int(failures)
                    best_loss = cur_loss
                    best_step = step
                    best_delta = cur.clone()
                if failures == 0:
                    break
    finally:
        hook.remove()
    del opt

    # Evaluate the selected candidate delta by hook without materializing it.
    h = core.register_output_delta_hook(output_layer, selected_ids, lambda: best_delta)
    try:
        margins = shared.evaluate_suppression_margins(
            model, tok, cases, llama_like=llama_like, device=device,
            batch_size=batch_size,
        )
    finally:
        h.remove()

    report = {
        "requested_context_rank": int(requested_rank),
        "rank_semantics": "0=full_observed_row_specific_forget_context_rank",
        "row_basis_reports": basis_reports,
        "row_context_ranks": [int(x["context_rank"]) for x in basis_reports],
        "trainable_parameters": int(delta_module.trainable_parameter_count),
        "best_step": int(best_step),
        "direct_failures": shared.count_failures(margins, required_margin),
        "minimum_suppression_margin": float(margins.min().detach().cpu()),
        "delta_norm": float(best_delta.norm().cpu()),
        "candidate_order": int(order),
        "logs": logs,
    }
    return report, best_delta


def main() -> None:
    a = parse_args()
    if a.forget_num <= 0 or a.repair_steps <= 0 or a.repair_lr <= 0:
        raise ValueError("forget-num, repair-steps, repair-lr must be positive")
    if a.batch_size <= 0 or a.check_every <= 0:
        raise ValueError("batch-size/check-every must be positive")
    if a.ga_weight <= 0 or a.gd_weight < 0 or a.repair_l2 < 0 or a.constraint_margin < 0:
        raise ValueError("invalid shared Stage-2 weights")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    ranks = core.parse_rank_list(a.candidate_ranks)
    scales = core.parse_scales(a.candidate_scales)
    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, manifest = load_locked(a.dataset, visible_path, manifest_path, a.seed, a.forget_num)

    ns = argparse.Namespace(model_path=a.model_path, dtype=a.dtype,
                            device_map=a.device_map, gradient_checkpointing=False)
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    output_layer = core.untie_and_freeze_output_head(model)
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)

    cases = core.expand_sensitive_cases(records, tok, dataset=a.dataset, llama_like=llama_like)
    if not cases:
        raise RuntimeError("No sensitive PredictionCases")
    base_logits = torch.load(Path(a.base_logits_cache).resolve(), map_location="cpu")
    if not isinstance(base_logits, torch.Tensor) or base_logits.shape[0] != len(cases):
        raise RuntimeError("Base-logit cache does not align with direct sensitive PredictionCases")

    before = shared.evaluate_suppression_margins(
        model, tok, cases, llama_like=llama_like, device=device, batch_size=a.batch_size
    )
    active_indices = [i for i, x in enumerate(before.detach().cpu().tolist())
                      if float(x) < a.constraint_margin]
    active_before = len(active_indices)

    out_dir = gagd.resolve_output_path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint"
    candidate_reports: List[Dict[str, Any]] = []
    candidate_deltas: List[torch.Tensor] = []
    selected_ids: List[int] = []
    selected_scale = 0.0
    scale_reports: List[Dict[str, Any]] = []
    chosen_index = None

    if active_indices:
        active_cases = [cases[i] for i in active_indices]
        active_tids = core.official_target_ids(
            tok, active_cases, llama_like=llama_like, device=device
        )
        selected_ids = sorted(set(int(x) for x in active_tids.detach().cpu().tolist()))

        for order, rank in enumerate(ranks):
            max_rank = None if rank == 0 else int(rank)
            bases, basis_reports = context.build_row_specific_bases(
                model, tok, active_cases, selected_ids=selected_ids,
                llama_like=llama_like, device=device, batch_size=a.batch_size,
                max_rank=max_rank,
            )
            report, delta = optimize_candidate(
                cases=cases,
                active_indices=active_indices,
                selected_ids=selected_ids,
                bases=bases,
                basis_reports=basis_reports,
                base_logits=base_logits,
                model=model,
                tok=tok,
                output_layer=output_layer,
                llama_like=llama_like,
                device=device,
                required_margin=a.constraint_margin,
                steps=a.repair_steps,
                lr=a.repair_lr,
                ga_weight=a.ga_weight,
                gd_weight=a.gd_weight,
                l2_weight=a.repair_l2,
                batch_size=a.batch_size,
                check_every=a.check_every,
                seed=a.seed,
                requested_rank=rank,
                order=order,
            )
            candidate_reports.append(report)
            candidate_deltas.append(delta)
            print("Shared context repair candidate", {
                "rank": rank,
                "row_ranks": report["row_context_ranks"],
                "direct_failures": report["direct_failures"],
                "delta_norm": report["delta_norm"],
            })
            if report["direct_failures"] == 0:
                break

        chosen_index = min(
            range(len(candidate_reports)),
            key=lambda i: candidate_key(candidate_reports[i], i),
        )
        chosen_delta = candidate_deltas[chosen_index]
        for scale in scales:
            h = core.register_output_delta_hook(
                output_layer, selected_ids,
                lambda scale=scale: chosen_delta * float(scale),
            )
            try:
                margins = shared.evaluate_suppression_margins(
                    model, tok, cases, llama_like=llama_like, device=device,
                    batch_size=a.batch_size,
                )
            finally:
                h.remove()
            scale_reports.append({
                "scale": float(scale),
                "direct_failures": shared.count_failures(margins, a.constraint_margin),
                "minimum_suppression_margin": float(margins.min().detach().cpu()),
                "effective_delta_norm": float(chosen_delta.norm().cpu() * scale),
            })
        selected_scale = core.choose_scale(scale_reports)
        final_delta = chosen_delta * float(selected_scale)
        core.materialize_output_delta(output_layer, selected_ids, final_delta)

    final_margins = shared.evaluate_suppression_margins(
        model, tok, cases, llama_like=llama_like, device=device, batch_size=a.batch_size
    )
    final_failures = shared.count_failures(final_margins, a.constraint_margin)

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    chosen_report = candidate_reports[chosen_index] if chosen_index is not None else None
    summary: Dict[str, Any] = {
        "schema_version": 1,
        "method": "SURE-LM-fixed-shared-context-stage2",
        "dataset": a.dataset,
        "protocol": "sure_fixed_shared_context_v1",
        "source_protocol": manifest.get("protocol"),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "architecture_shared_across_mcf_zsre": True,
        "objective": "ga_sensitive_logprob + gd_non_sensitive_distribution_kl + delta_l2",
        "no_hinge_or_relu_training_loss": True,
        "no_reference_answer_ce": True,
        "constraint_semantics": "best_non_sensitive_logit_minus_sensitive_logit",
        "constraint_margin": float(a.constraint_margin),
        "direct_unit": "teacher_forced_sensitive_prediction_case",
        "direct_total": len(cases),
        "active_before": int(active_before),
        "active_after": int(final_failures),
        "selected_lm_head_rows": len(selected_ids),
        "selected_token_ids": selected_ids,
        "editable_rows": "sensitive_answer_rows_only",
        "input_embeddings_modified": False,
        "transformer_trainable": 0,
        "row_specific_context_projection": True,
        "candidate_ranks": ranks,
        "candidate_reports": candidate_reports,
        "chosen_candidate": chosen_report,
        "candidate_scales": scales,
        "scale_reports": scale_reports,
        "selected_scale": float(selected_scale),
        "ga_weight": float(a.ga_weight),
        "gd_weight": float(a.gd_weight),
        "repair_l2": float(a.repair_l2),
        "repair_steps": int(a.repair_steps),
        "repair_lr": float(a.repair_lr),
        "benchmark_retain_seen": 0,
        "heldout_probes_seen": 0,
        "selection_uses_heldout": False,
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "repair_summary.json", summary)
    core.write_json(out_dir / "rank_candidates.json", candidate_reports)
    core.write_json(out_dir / "scale_sweep_direct_only.json", scale_reports)
    print(f"Fixed shared Stage 2 {a.dataset}: failures {active_before} -> {final_failures}; selected rows={len(selected_ids)}; scale={selected_scale:g}")


if __name__ == "__main__":
    main()
