#!/usr/bin/env python3
"""MCF SURE Stage 2 v5: exact protected-sequence nullspace + rowwise MCF gradients.

Concrete locality fix:
  * P/F is defined by direct MCF sequence margin
        NLL(target_true) - NLL(target_new).
  * The protected basis contains EVERY teacher-forced prediction hidden state
    used by the official-compatible target_true AND target_new NLLs of P.
    Therefore any LM-head delta in its orthogonal complement leaves every
    protected sequence logit unchanged in exact arithmetic.
  * A residual low-rank pre-answer context basis is added for broader locality.
  * Each editable target_true LM-head row receives its OWN repair basis built
    from per-record gradients of the actual differentiable MCF sequence margin,
    projected into the exact protected nullspace.
  * Optimization uses the sequence-margin hinge, KL(P), L2, hard protection,
    geometric backtracking, then the minimum exact bf16-feasible save scale.

No official paraphrases, neighborhoods, benchmark-retain rows, or PPL text are
read before the final checkpoint is frozen.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

import gagd_compare as gagd
import gagd_active_case_repair as mcf_repair
import sure_canonical_core as core
import sure_context_projection as context
import sure_stage2_sparse_repair as shared_stage2
import mcf_sure_protected_subspace_stage1 as stage1v2
import mcf_sure_protected_subspace_stage2_mcf_margin as v3
import mcf_sure_rowspecific_minimal_stage2 as v4

METHOD = "SURE-MCF-exact-sequence-null-gradient-stage2"
PROTOCOL = "mcf_target_true_exact_sequence_null_gradient_stage2_v5"
BACKTRACK_SCALES = tuple([1.0] + [2.0 ** (-k) for k in range(1, 25)])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--stage1-config-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--repair-steps", type=int, default=800)
    p.add_argument("--repair-lr", type=float, default=5e-3)
    p.add_argument("--train-mcf-margin", type=float, default=0.10)
    p.add_argument("--final-mcf-margin", type=float, default=0.05)
    p.add_argument("--protected-mcf-margin-floor", type=float, default=0.0)
    p.add_argument("--context-rank", type=int, default=32,
                   help="Extra pre-answer context rank after exact P projection.")
    p.add_argument("--repair-rank", type=int, default=4)
    p.add_argument("--protected-kl-weight", type=float, default=1.0)
    p.add_argument("--protected-kl-max", type=float, default=0.05)
    p.add_argument("--delta-l2", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--scale-bisect-steps", type=int, default=12)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    a = p.parse_args(list(argv) if argv is not None else None)
    if min(a.forget_num, a.repair_steps, a.context_rank, a.repair_rank,
           a.check_every, a.cache_batch_size, a.scale_bisect_steps) <= 0:
        p.error("counts/ranks/check/bisect settings must be positive")
    if a.repair_lr <= 0:
        p.error("repair-lr must be positive")
    if min(a.train_mcf_margin, a.final_mcf_margin,
           a.protected_mcf_margin_floor, a.protected_kl_weight,
           a.protected_kl_max, a.delta_l2, a.grad_clip) < 0:
        p.error("margins/KL/L2/clip must be non-negative")
    if a.final_mcf_margin > a.train_mcf_margin:
        p.error("final-mcf-margin must not exceed train-mcf-margin")
    return a


@torch.no_grad()
def collect_official_sequence_hidden(
    model,
    tok,
    instances: Sequence[Any],
    record_positions: Sequence[int],
    *,
    device: torch.device,
    llama_like: bool,
    batch_size: int,
) -> torch.Tensor:
    """Hidden rows at every target prediction position for both MCF answers."""
    wanted = {int(x) for x in record_positions}
    selected_instances = [
        instance for i, instance in enumerate(instances) if i in wanted
    ]
    rows: List[torch.Tensor] = []
    for start in range(0, len(selected_instances), int(batch_size)):
        chunk = selected_instances[start:start + int(batch_size)]
        encoded, target_token_ids, prefix_lens = mcf_repair.official_batch_components(
            tok, chunk, device, llama_like
        )
        output = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden = output.hidden_states[-1]
        if llama_like:
            hidden = hidden[:, 1:, :]
        for row, (target_ids, prefix_len) in enumerate(
            zip(target_token_ids, prefix_lens)
        ):
            positions = torch.arange(
                prefix_len - 1,
                prefix_len + len(target_ids) - 1,
                dtype=torch.long,
                device=hidden.device,
            )
            rows.append(hidden[row].index_select(0, positions).float().detach())
    if not rows:
        # Caller will know hidden size from model.
        hidden_size = int(model.get_output_embeddings().weight.shape[1])
        return torch.empty((0, hidden_size), dtype=torch.float32, device=device)
    return torch.cat(rows, dim=0)


def build_exact_safe_basis(
    *,
    model,
    tok,
    instances,
    protected_records: Sequence[int],
    sensitive_cases,
    device: torch.device,
    llama_like: bool,
    batch_size: int,
    context_rank: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    exact_p = collect_official_sequence_hidden(
        model, tok, instances, protected_records,
        device=device, llama_like=llama_like, batch_size=batch_size,
    )
    if exact_p.numel():
        b_exact = core.orthonormal_row_basis(exact_p.float(), max_rank=None)
    else:
        hidden_size = int(model.get_output_embeddings().weight.shape[1])
        b_exact = torch.empty((0, hidden_size), dtype=torch.float32, device=device)

    # Broader direct-context protection, but only after exact P directions have
    # been removed so it cannot dilute the exact sequence protection.
    _, h_context = stage1v2.collect_prediction_and_context_hidden(
        model, tok, sensitive_cases, device, int(batch_size)
    )
    if h_context.numel():
        context_residual = stage1v2.project_away(h_context.float(), b_exact)
        b_context = core.orthonormal_row_basis(
            context_residual, max_rank=int(context_rank)
        )
    else:
        b_context = b_exact.new_empty((0, b_exact.shape[1]))

    if b_exact.numel() and b_context.numel():
        safe = core.orthonormal_row_basis(
            torch.cat([b_exact, b_context], dim=0), max_rank=None
        )
    elif b_exact.numel():
        safe = b_exact
    else:
        safe = b_context

    return safe.contiguous(), {
        "protected_sequence_hidden_rows": int(exact_p.shape[0]),
        "exact_protected_rank": int(b_exact.shape[0]),
        "pre_answer_context_rows": int(h_context.shape[0]),
        "extra_context_rank_requested": int(context_rank),
        "extra_context_rank_actual": int(b_context.shape[0]),
        "safe_rank_total": int(safe.shape[0]),
        "definition": "full rowspace(all P target_true+target_new teacher-forced scoring states) + residual rank-capped pre-answer context",
    }


def selected_sensitive_rows(tok, instances, failure_records: Sequence[int]) -> List[int]:
    return shared_stage2.mcf_sensitive_rows(
        tok,
        instances,
        failure_records,
        sensitive_field="target_true",
    )


def build_rowwise_margin_gradient_bases(
    *,
    caches,
    selected_ids: Sequence[int],
    hidden_size: int,
    safe_basis: torch.Tensor,
    repair_rank: int,
    device: torch.device,
) -> Tuple[List[int], List[torch.Tensor], List[Dict[str, Any]]]:
    """Per-row bases from gradients of each failure record's exact MCF margin."""
    gradients: Dict[int, List[torch.Tensor]] = {int(t): [] for t in selected_ids}
    for cache in caches:
        delta = torch.zeros(
            (len(selected_ids), int(hidden_size)),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        margin = v3.record_margins_from_caches([cache], delta)[0]
        grad = torch.autograd.grad(margin, delta, retain_graph=False)[0].detach()
        for row, tid in enumerate(selected_ids):
            g = grad[row].float()
            if torch.isfinite(g).all() and float(g.norm().cpu()) > 1e-9:
                gradients[int(tid)].append(g)

    kept: List[int] = []
    bases: List[torch.Tensor] = []
    reports: List[Dict[str, Any]] = []
    for tid in selected_ids:
        gs = gradients[int(tid)]
        if not gs:
            reports.append({
                "token_id": int(tid),
                "gradient_count": 0,
                "repair_rank_actual": 0,
                "skipped": True,
            })
            continue
        matrix = torch.stack(gs, dim=0).float()
        residual = stage1v2.project_away(matrix, safe_basis)
        residual_norm = float(residual.norm().detach().cpu())
        if residual_norm <= 1e-9:
            reports.append({
                "token_id": int(tid),
                "gradient_count": int(matrix.shape[0]),
                "residual_gradient_norm": residual_norm,
                "repair_rank_actual": 0,
                "skipped": True,
            })
            continue
        basis = core.orthonormal_row_basis(
            residual, max_rank=int(repair_rank)
        )
        overlap = (
            float((basis @ safe_basis.transpose(0, 1)).abs().max().detach().cpu())
            if safe_basis.numel() else 0.0
        )
        kept.append(int(tid))
        bases.append(basis.contiguous())
        reports.append({
            "token_id": int(tid),
            "gradient_count": int(matrix.shape[0]),
            "raw_gradient_norm": float(matrix.norm().detach().cpu()),
            "residual_gradient_norm": residual_norm,
            "repair_rank_requested": int(repair_rank),
            "repair_rank_actual": int(basis.shape[0]),
            "max_abs_overlap_with_safe_basis": overlap,
            "skipped": False,
        })
    if not kept:
        raise RuntimeError(
            "All MCF-margin gradients vanish in the exact protected nullspace; "
            "the requested direct repair is incompatible with exact P protection."
        )
    return kept, bases, reports


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(int(a.seed))
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    stage1_config_path = Path(a.stage1_config_path).resolve()
    records, manifest, stage1_config = v3.load_and_validate_protocol(
        visible_path, manifest_path, stage1_config_path,
        int(a.seed), int(a.forget_num),
    )

    ns = argparse.Namespace(
        model_path=a.model_path, dtype=a.dtype, device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    output_layer = core.untie_and_freeze_output_head(model)
    if model.get_input_embeddings() is None:
        raise RuntimeError("model lacks input embeddings")
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tok)

    sensitive_cases = context.expand_answer_field_cases(
        records, tok, field="target_true", llama_like=llama_like
    )
    hidden, atomic_base_logits, target_ids = v3.cache_atomic_state(
        model, tok, sensitive_cases, llama_like=llama_like,
        device=device, batch_size=int(a.cache_batch_size),
    )
    instances = shared_stage2.mcf_instances(records)
    baseline_margins = shared_stage2.mcf_direct_margins(
        model, tok, instances, device, llama_like, int(a.cache_batch_size),
        sensitive_field="target_true", reference_field="target_new",
    ).detach().float()
    protected_records, failure_records = v3.record_partition(
        baseline_margins, float(a.train_mcf_margin)
    )
    protected_atomic_positions = v3.atomic_positions_for_records(
        sensitive_cases, protected_records
    )

    safe_basis, safe_report = build_exact_safe_basis(
        model=model, tok=tok, instances=instances,
        protected_records=protected_records, sensitive_cases=sensitive_cases,
        device=device, llama_like=llama_like,
        batch_size=int(a.cache_batch_size), context_rank=int(a.context_rank),
    )

    candidate_ids = selected_sensitive_rows(tok, instances, failure_records)
    if not candidate_ids:
        raise RuntimeError("no target_true LM-head rows selected for residual failures")
    candidate_caches = mcf_repair.build_prompt_instance_delta_caches(
        model, tok, instances, candidate_ids, device,
        int(a.cache_batch_size), llama_like,
    )
    failure_candidate_caches = [candidate_caches[int(i)] for i in failure_records]

    selected_ids, row_bases, row_reports = build_rowwise_margin_gradient_bases(
        caches=failure_candidate_caches,
        selected_ids=candidate_ids,
        hidden_size=int(output_layer.weight.shape[1]),
        safe_basis=safe_basis,
        repair_rank=int(a.repair_rank),
        device=output_layer.weight.device,
    )

    # Rebuild caches against the final kept row set so columns align exactly.
    all_caches = mcf_repair.build_prompt_instance_delta_caches(
        model, tok, instances, selected_ids, device,
        int(a.cache_batch_size), llama_like,
    )
    failure_caches = [all_caches[int(i)] for i in failure_records]

    delta_module = context.RowSpecificProjectedDelta(
        selected_ids, row_bases, device=output_layer.weight.device
    )
    opt = torch.optim.AdamW(
        delta_module.parameters(), lr=float(a.repair_lr), weight_decay=0.0
    )
    params = list(delta_module.parameters())
    ids_tensor = torch.tensor(
        selected_ids, dtype=torch.long, device=output_layer.weight.device
    )
    base_rows = output_layer.weight.index_select(0, ids_tensor).detach().clone()
    frozen_hash_before = v3.hash_frozen_parameters(model, output_layer)

    logs: List[Dict[str, Any]] = []
    accepted_hist: Dict[str, int] = {}
    rollback_count = 0
    feasible_delta = None

    for step in range(1, int(a.repair_steps) + 1):
        opt.zero_grad(set_to_none=True)
        delta = delta_module.effective_delta()
        margins_f = v3.record_margins_from_caches(failure_caches, delta)
        hinge = F.relu(float(a.train_mcf_margin) - margins_f).square().mean()
        kl_p = v3.protected_kl(
            base_logits=atomic_base_logits, hidden=hidden,
            protected_atomic_positions=protected_atomic_positions,
            selected_ids=selected_ids, delta_rows=delta,
        )
        l2 = delta.square().mean()
        loss = hinge + float(a.protected_kl_weight) * kl_p + float(a.delta_l2) * l2
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite Stage-2 loss at step {step}")

        old_params = v3.parameter_snapshot(delta_module)
        opt_state_before = copy.deepcopy(opt.state_dict())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(a.grad_clip)) \
            if float(a.grad_clip) > 0 else None
        if grad_norm is not None and not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        opt.step()
        proposed_params = v3.parameter_snapshot(delta_module)

        accepted_alpha = None
        accepted_protection = None
        for alpha in BACKTRACK_SCALES:
            v3.set_interpolated_parameters(
                delta_module, old_params, proposed_params, float(alpha)
            )
            protection = v3.hard_protection_metrics(
                all_record_caches=all_caches,
                protected_record_positions=protected_records,
                protected_atomic_positions=protected_atomic_positions,
                base_logits=atomic_base_logits, hidden=hidden,
                selected_ids=selected_ids,
                delta_rows=delta_module.effective_delta(),
                mcf_margin_floor=float(a.protected_mcf_margin_floor),
            )
            if (int(protection["protected_mcf_regressions"]) == 0 and
                    max(0.0, float(protection["protected_kl"])) <= float(a.protected_kl_max)):
                accepted_alpha = float(alpha)
                accepted_protection = protection
                break
        if accepted_alpha is None:
            v3.set_interpolated_parameters(delta_module, old_params, old_params, 1.0)
            opt.load_state_dict(opt_state_before)
            rollback_count += 1
            accepted_alpha = 0.0
            accepted_protection = v3.hard_protection_metrics(
                all_record_caches=all_caches,
                protected_record_positions=protected_records,
                protected_atomic_positions=protected_atomic_positions,
                base_logits=atomic_base_logits, hidden=hidden,
                selected_ids=selected_ids,
                delta_rows=delta_module.effective_delta(),
                mcf_margin_floor=float(a.protected_mcf_margin_floor),
            )
        else:
            key = f"{accepted_alpha:g}"
            accepted_hist[key] = int(accepted_hist.get(key, 0)) + 1

        with torch.no_grad():
            cur = delta_module.effective_delta().detach()
            cached_all = v3.record_margins_from_caches(all_caches, cur)
            cached_fail = int((cached_all < float(a.train_mcf_margin)).sum().item())
            exact_pass = False
            exact_min = None
            if cached_fail == 0:
                exact_margins, _ = v4.exact_materialized_margins(
                    model=model, tok=tok, output_layer=output_layer,
                    selected_ids=selected_ids, base_rows=base_rows,
                    delta_rows=cur, scale=1.0, instances=instances,
                    device=device, llama_like=llama_like,
                    batch_size=int(a.cache_batch_size),
                )
                exact_min = float(exact_margins.min().item())
                exact_pass = bool(
                    (exact_margins >= float(a.final_mcf_margin)).all().item()
                )
                if exact_pass:
                    feasible_delta = cur.clone()
            row = {
                "step": int(step),
                "cached_mcf_failures": int(cached_fail),
                "cached_min_margin": float(cached_all.min().detach().cpu()),
                "exact_bf16_feasible": bool(exact_pass),
                "exact_bf16_min_margin": exact_min,
                "protected_mcf_regressions": int(
                    accepted_protection["protected_mcf_regressions"]
                ),
                "protected_kl": max(
                    0.0, float(accepted_protection["protected_kl"])
                ),
                "accepted_backtrack_alpha": float(accepted_alpha),
                "delta_norm": float(cur.norm().cpu()),
            }
            if grad_norm is not None:
                row["grad_norm"] = float(grad_norm.detach().cpu())
            if (step == 1 or step % int(a.check_every) == 0 or
                    cached_fail == 0 or step == int(a.repair_steps)):
                logs.append(row)
                print(json.dumps(row))
            if feasible_delta is not None:
                break

    del opt
    if feasible_delta is None:
        raise RuntimeError(
            "No exact-bf16 feasible exact-null repair found; refusing to save."
        )

    selected_scale, quantized_delta, scale_metrics, scale_evals = \
        v4.minimum_exact_scale(
            model=model, tok=tok, output_layer=output_layer,
            selected_ids=selected_ids, base_rows=base_rows,
            delta_rows=feasible_delta, instances=instances, device=device,
            llama_like=llama_like, batch_size=int(a.cache_batch_size),
            final_margin=float(a.final_mcf_margin),
            protected_records=protected_records,
            protected_floor=float(a.protected_mcf_margin_floor),
            protected_atomic_positions=protected_atomic_positions,
            atomic_base_logits=atomic_base_logits, hidden=hidden,
            protected_kl_max=float(a.protected_kl_max),
            bisect_steps=int(a.scale_bisect_steps),
        )

    final_rows = base_rows.float() + quantized_delta.to(base_rows.device).float()
    output_layer.weight.index_copy_(
        0, ids_tensor, final_rows.to(output_layer.weight.dtype)
    )
    frozen_hash_after = v3.hash_frozen_parameters(model, output_layer)
    frozen_bit_exact = frozen_hash_before == frozen_hash_after
    if not frozen_bit_exact:
        raise RuntimeError("Stage 2 changed embeddings and/or transformer parameters")

    final_margins = shared_stage2.mcf_direct_margins(
        model, tok, instances, device, llama_like, int(a.cache_batch_size),
        sensitive_field="target_true", reference_field="target_new",
    ).detach().float().cpu()
    final_failures = [
        i for i, x in enumerate(final_margins.tolist())
        if float(x) < float(a.final_mcf_margin)
    ]

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    final_gate = {
        "all_mcf_direct_records_pass": len(final_failures) == 0,
        "minimum_final_mcf_margin": float(final_margins.min().item()),
        "protected_regressions_zero": int(scale_metrics["protected_regressions"]) == 0,
        "protected_kl_within_limit": float(scale_metrics["protected_kl"]) <= float(a.protected_kl_max),
        "embeddings_and_transformer_bit_exact": bool(frozen_bit_exact),
    }
    final_gate["passed"] = bool(
        final_gate["all_mcf_direct_records_pass"]
        and final_gate["protected_regressions_zero"]
        and final_gate["protected_kl_within_limit"]
        and final_gate["embeddings_and_transformer_bit_exact"]
    )

    summary: Dict[str, Any] = {
        "schema_version": 5,
        "method": METHOD,
        "protocol": PROTOCOL,
        "source_stage1_protocol": stage1_config.get("protocol"),
        "source_protocol": manifest.get("protocol"),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "primary_gate": "NLL(target_true)-NLL(target_new)",
        "train_mcf_margin": float(a.train_mcf_margin),
        "final_mcf_margin": float(a.final_mcf_margin),
        "stage1_mcf_success_count_P": len(protected_records),
        "stage1_mcf_failure_count_F": len(failure_records),
        "safe_basis": safe_report,
        "row_basis_source": "per-record gradient of differentiable MCF sequence margin",
        "row_specific_repair": True,
        "row_basis_reports": row_reports,
        "selected_lm_head_rows": len(selected_ids),
        "selected_token_ids": selected_ids,
        "parameterization": "Delta w_s = c_s B_gradient,s in exact P nullspace",
        "trainable_parameters": int(delta_module.trainable_parameter_count),
        "optimizer_logs": logs,
        "accepted_backtrack_histogram": accepted_hist,
        "rollback_count": int(rollback_count),
        "unscaled_feasible_delta_norm": float(feasible_delta.norm().cpu()),
        "minimum_exact_bf16_scale": float(selected_scale),
        "final_effective_delta_norm": float(quantized_delta.norm().cpu()),
        "scale_line_search": scale_evals,
        "final_mcf_record_failure_count": len(final_failures),
        "final_mcf_record_failure_positions": final_failures,
        "final_mcf_record_min_margin": float(final_margins.min().item()),
        "final_gate": final_gate,
        "official_paraphrases_seen": 0,
        "official_neighborhood_seen": 0,
        "benchmark_retain_seen": 0,
        "ppl_eval_text_seen": 0,
        "training_visible_sha256": stage1v2.sha256_file(visible_path),
        "split_manifest_sha256": stage1v2.sha256_file(manifest_path),
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "repair_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(
        f"Exact-null gradient Stage 2: failures {len(failure_records)} -> "
        f"{len(final_failures)}; scale={selected_scale:.6f}; "
        f"norm={float(quantized_delta.norm().cpu()):.4f}; "
        f"gate={final_gate['passed']}"
    )
    print(f"Final checkpoint: {ckpt}")
    if not final_gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
