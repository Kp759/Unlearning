#!/usr/bin/env python3
"""SURE-MQuAKE V9 Stage 1: per-instance rare/easy sparse LM-head forgetting.

Stage 1 uses ONLY the locked direct requested_rewrite facts from the 50 forget
instances.  For each original MQuAKE instance it considers sensitive token
positions that are active in the untouched protected Base, ranks them by:

  (1) lower instance-level document frequency of the target token (rarer first),
  (2) larger Base competitor-minus-sensitive margin (closer to zero first),

and selects up to K DISTINCT target-token ids per instance (default K=3).
Thus at most 50*K token positions are explicitly targeted, while duplicate token
ids across instances collapse to the same editable LM-head row.

Only those selected LM-head rows are editable.  The forget loss is a squared
hinge on selected positions only.  Same-prompt non-target KL, an all-visible
Base-safe hinge, and L2 preserve behavior.  Transformer blocks, input embeddings
and every non-selected LM-head row remain exact Base.

Stage 1 is NOT required to forget every visible token.  It stops when all
selected rare/easy positions are BF16-safe while all positions that were safe in
Base remain safe.  It then bisects the update toward Base with exact BF16 audits.
Residual active positions are intentionally left for V9 Stage 2.

No benchmark retain instances, AtomicGen, multihop questions, target_new, or PPL
data are loaded for optimization or selection.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

import gagd_active_case_repair as active
import gagd_compare as gagd
import mquake_forget_only_active_repair as locked
import mquake_sure_active_rows_v72 as v72
import mquake_sure_sparse_lm_gagd_v7 as v7
import mquake_v7_locked_case_compat as compat
import mquake_zero_unlearn_official_eval as mquake


METHOD = "SURE-MQuAKE-v9.0-rare-easy-token-sparse-stage1"
PROTOCOL = "mquake_zerounlearn_forget_only_locked_probes"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--repair-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--top-k-per-instance", type=int, default=3)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--forget-hinge-weight", type=float, default=5.0)
    p.add_argument("--hardest-forget-hinge-weight", type=float, default=1.0)
    p.add_argument("--safe-hinge-weight", type=float, default=10.0)
    p.add_argument("--gd-weight", type=float, default=10.0)
    p.add_argument("--delta-l2-lambda", type=float, default=1e-3)
    p.add_argument("--target-logit-margin", type=float, default=0.0)
    p.add_argument("--bf16-buffer-margin", type=float, default=0.04)
    p.add_argument("--safe-margin-floor", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--bf16-bisection-steps", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def source_index_by_case_id(records: Sequence[Mapping[str, Any]]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for record in records:
        cid = int(record["case_id"])
        src = int(record["source_index"])
        if cid in out and out[cid] != src:
            raise RuntimeError(f"case_id {cid} maps to multiple source instances")
        out[cid] = src
    return out


def choose_positions(
    cases: Sequence[mquake.PredictionCase],
    target_ids: Sequence[int],
    base_margins: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    top_k: int,
) -> Tuple[List[int], Dict[int, int], List[Dict[str, Any]]]:
    case_to_source = source_index_by_case_id(records)
    source_by_position = [case_to_source[int(case.case_id)] for case in cases]

    # Instance-level DF: one token counts at most once per original MQuAKE instance.
    token_sources: Dict[int, set[int]] = collections.defaultdict(set)
    for pos, token_id in enumerate(target_ids):
        token_sources[int(token_id)].add(int(source_by_position[pos]))
    token_df = {token_id: len(sources) for token_id, sources in token_sources.items()}

    active_by_source: Dict[int, List[int]] = collections.defaultdict(list)
    margins_cpu = base_margins.detach().cpu().tolist()
    for pos, margin in enumerate(margins_cpu):
        if float(margin) <= 0.0:
            active_by_source[int(source_by_position[pos])].append(pos)

    selected: List[int] = []
    rows: List[Dict[str, Any]] = []
    all_sources = sorted(set(source_by_position))
    for source_index in all_sources:
        candidates = active_by_source.get(source_index, [])
        candidates = sorted(
            candidates,
            key=lambda pos: (
                int(token_df[int(target_ids[pos])]),
                -float(margins_cpu[pos]),  # closest to zero first among equal rarity
                int(cases[pos].case_id),
                int(cases[pos].token_index),
            ),
        )
        chosen: List[int] = []
        seen_token_ids: set[int] = set()
        for pos in candidates:
            token_id = int(target_ids[pos])
            if token_id in seen_token_ids:
                continue
            chosen.append(pos)
            seen_token_ids.add(token_id)
            if len(chosen) >= top_k:
                break
        selected.extend(chosen)
        for local_rank, pos in enumerate(chosen, start=1):
            rows.append({
                "source_index": int(source_index),
                "selection_rank_within_instance": int(local_rank),
                "case_position": int(pos),
                "case_id": int(cases[pos].case_id),
                "token_index": int(cases[pos].token_index),
                "target_token_id": int(target_ids[pos]),
                "instance_document_frequency": int(token_df[int(target_ids[pos])]),
                "base_competitor_minus_sensitive_margin": float(margins_cpu[pos]),
            })

    return sorted(selected), token_df, rows


def cached_stage1_pass(
    margins: torch.Tensor,
    selected_mask: torch.Tensor,
    base_safe_mask: torch.Tensor,
    *,
    selected_required_margin: float,
    safe_floor: float,
) -> bool:
    selected_ok = torch.all(margins[selected_mask] >= float(selected_required_margin))
    safe_ok = torch.all(margins[base_safe_mask] > float(safe_floor))
    return bool((selected_ok & safe_ok).item())


def exact_stage1_pass(
    reports: Sequence[Mapping[str, Any]],
    selected_positions: Sequence[int],
    base_safe_positions: Sequence[int],
) -> bool:
    for pos in selected_positions:
        if bool(reports[pos]["official_sensitive_token_still_argmax"]):
            return False
    for pos in base_safe_positions:
        if bool(reports[pos]["official_sensitive_token_still_argmax"]):
            return False
    return True


def exact_counts(
    reports: Sequence[Mapping[str, Any]],
    selected_positions: Sequence[int],
    base_safe_positions: Sequence[int],
) -> Dict[str, int]:
    selected_active = sum(bool(reports[p]["official_sensitive_token_still_argmax"]) for p in selected_positions)
    safe_failed = sum(bool(reports[p]["official_sensitive_token_still_argmax"]) for p in base_safe_positions)
    total_active = sum(bool(row["official_sensitive_token_still_argmax"]) for row in reports)
    return {
        "selected_position_active_count": int(selected_active),
        "base_safe_reactivated_count": int(safe_failed),
        "residual_active_sensitive_token_count": int(total_active),
    }


def main() -> None:
    a = parse_args()
    if a.forget_num <= 0 or a.top_k_per_instance <= 0 or a.steps <= 0 or a.batch_size <= 0:
        raise ValueError("forget-num/top-k/steps/batch-size must be positive")
    if a.lr <= 0 or a.forget_hinge_weight <= 0 or a.gd_weight < 0:
        raise ValueError("invalid V9 Stage1 optimization controls")
    if min(a.hardest_forget_hinge_weight, a.safe_hinge_weight, a.delta_l2_lambda, a.grad_clip) < 0:
        raise ValueError("invalid V9 Stage1 regularization controls")
    if min(a.target_logit_margin, a.bf16_buffer_margin, a.safe_margin_floor) < 0:
        raise ValueError("margins/floors must be non-negative")
    if a.bf16_bisection_steps <= 0:
        raise ValueError("bf16-bisection-steps must be positive")

    compat.install()
    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.repair_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, split_manifest = locked.load_locked_records(visible_path, manifest_path, a.forget_num, a.seed)

    model_args = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(model_args, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    output_layer = active.freeze_model_for_output_repair(model)
    output_weight = output_layer.weight
    input_weight = model.get_input_embeddings().weight
    input_pointer = int(input_weight.data_ptr())
    input_version = int(input_weight._version)
    device = gagd.first_device(model)
    llama_like = mquake.is_llama_like(model, tok)

    cases = v7.direct_rewrite_cases(records, tok, llama_like=llama_like)
    target_ids = v7.resolve_case_target_ids(tok, cases, llama_like=llama_like, device=device)
    special_ids = gagd.special_token_ids(tok)
    all_sensitive_ids = sorted(set(target_ids) - special_ids)
    if not all_sensitive_ids:
        raise RuntimeError("V9 Stage1 found no sensitive target rows")
    if set(target_ids) - set(all_sensitive_ids):
        raise RuntimeError("V9 Stage1 sensitive targets unexpectedly include special ids")

    # Untouched Base geometry over all visible direct sensitive-token decisions.
    all_caches = v7.build_token_delta_caches(
        model, tok, cases, all_sensitive_ids,
        device=device, llama_like=llama_like, batch_size=a.batch_size,
        desc="cache MQuAKE V9 Base geometry",
    )
    all_stacked = v7.stack_cache_fields(all_caches, device=device)
    all_zero = torch.zeros(
        (len(all_sensitive_ids), int(output_weight.shape[1])),
        dtype=torch.float32, device=device,
    )
    base_margins = v7.competitor_minus_sensitive_margins(all_stacked, all_zero)
    base_active_mask_cpu = (base_margins <= 0.0).detach().cpu()
    base_safe_positions = (~base_active_mask_cpu).nonzero(as_tuple=False).flatten().tolist()

    selected_positions, token_df, selection_rows = choose_positions(
        cases, target_ids, base_margins, records, a.top_k_per_instance
    )
    if not selected_positions:
        raise RuntimeError("V9 Stage1 selected zero rare active positions")
    selected_ids = sorted({int(target_ids[pos]) for pos in selected_positions})
    selected_position_mask = torch.zeros(len(cases), dtype=torch.bool, device=device)
    selected_position_mask[torch.tensor(selected_positions, dtype=torch.long, device=device)] = True
    base_safe_mask = (~base_active_mask_cpu).to(device=device, dtype=torch.bool)

    selected_tensor = torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device)
    base_selected_rows = output_weight.index_select(0, selected_tensor).detach().clone()

    # Re-cache every visible case against only the rows Stage 1 may edit.
    caches = v7.build_token_delta_caches(
        model, tok, cases, selected_ids,
        device=device, llama_like=llama_like, batch_size=a.batch_size,
        desc="cache MQuAKE V9 sparse selected rows",
    )
    stacked = v7.stack_cache_fields(caches, device=device)
    zero = torch.zeros(
        (len(selected_ids), int(output_weight.shape[1])),
        dtype=torch.float32, device=device,
    )
    restricted_base = v7.competitor_minus_sensitive_margins(stacked, zero)
    if not torch.allclose(base_margins, restricted_base, atol=1e-5, rtol=0.0):
        diff = float((base_margins - restricted_base).abs().max().detach().cpu())
        raise RuntimeError(f"V9 Stage1 restricted Base margin reconstruction mismatch: {diff}")

    required_margin = float(a.target_logit_margin + a.bf16_buffer_margin)
    module = active.SelectedRowDelta(
        len(selected_ids), int(output_weight.shape[1]),
        direction_basis=None, retained_basis=None, device=device,
    )
    optimizer = torch.optim.AdamW(module.parameters(), lr=a.lr, weight_decay=0.0)

    feasible_delta: torch.Tensor | None = None
    feasible_step: int | None = None
    feasible_reports: List[Dict[str, Any]] | None = None
    feasible_summary: Dict[str, Any] | None = None
    logs: List[Dict[str, Any]] = []

    for step in range(1, a.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = module.effective_delta()
        margins = v7.competitor_minus_sensitive_margins(stacked, delta)
        selected_error = torch.relu(required_margin - margins[selected_position_mask])
        safe_error = torch.relu(float(a.safe_margin_floor) - margins[base_safe_mask])
        gd_kl = v7.same_prompt_non_target_kl(stacked, delta)
        delta_l2 = delta.square().sum()
        selected_loss = selected_error.square().mean()
        selected_hard = selected_error.square().max()
        safe_loss = safe_error.square().mean() if safe_error.numel() else delta.new_zeros(())
        loss = (
            a.forget_hinge_weight * selected_loss
            + a.hardest_forget_hinge_weight * selected_hard
            + a.safe_hinge_weight * safe_loss
            + a.gd_weight * gd_kl
            + a.delta_l2_lambda * delta_l2
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite V9 Stage1 loss at step {step}")
        loss.backward()
        grad_norm_value = None
        if a.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), a.grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite V9 Stage1 gradient at step {step}")
            grad_norm_value = float(grad_norm.detach().cpu())
        optimizer.step()

        with torch.no_grad():
            candidate = module.effective_delta().detach().clone()
            candidate_margins = v7.competitor_minus_sensitive_margins(stacked, candidate)
            candidate_kl = float(v7.same_prompt_non_target_kl(stacked, candidate).detach().cpu())
            selected_active = int((candidate_margins[selected_position_mask] <= 0.0).sum().item())
            selected_unmet = int((candidate_margins[selected_position_mask] < required_margin).sum().item())
            safe_failed = int((candidate_margins[base_safe_mask] <= a.safe_margin_floor).sum().item())
            total_active = int((candidate_margins <= 0.0).sum().item())

        if step == 1 or step % a.log_every == 0 or step == a.steps:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "selected_active": selected_active,
                "selected_buffer_unmet": selected_unmet,
                "base_safe_failed": safe_failed,
                "all_visible_active": total_active,
                "minimum_margin": float(candidate_margins.min().detach().cpu()),
                "gd_same_prompt_non_target_kl": candidate_kl,
                "selected_lm_head_delta_norm": float(candidate.norm().detach().cpu()),
                "gradient_norm_before_clip": grad_norm_value,
            }
            logs.append(row)
            print(
                f"v90-s1-step={step} selected_active={selected_active} selected_unmet={selected_unmet} "
                f"safe_failed={safe_failed} all_active={total_active} "
                f"min_margin={row['minimum_margin']:.6g} KL={candidate_kl:.6g} "
                f"norm={row['selected_lm_head_delta_norm']:.6g}"
            )

        if cached_stage1_pass(
            candidate_margins,
            selected_position_mask,
            base_safe_mask,
            selected_required_margin=required_margin,
            safe_floor=a.safe_margin_floor,
        ):
            reports, summary = v72.exact_materialized_audit(
                model, tok, cases,
                output_weight=output_weight,
                selected_ids=selected_ids,
                base_selected_rows=base_selected_rows,
                delta=candidate,
                device=device, llama_like=llama_like,
                batch_size=a.batch_size, target_margin=a.target_logit_margin,
            )
            if exact_stage1_pass(reports, selected_positions, base_safe_positions):
                feasible_delta = candidate.detach().clone()
                feasible_step = step
                feasible_reports = reports
                feasible_summary = summary
                break
            v7.set_selected_rows(output_weight, selected_ids, base_selected_rows, zero)

    del optimizer
    root = gagd.resolve_output_path(a.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    ckpt = root / "checkpoint"
    v7.write_jsonl(root / "train_log.jsonl", logs)

    if feasible_delta is None or feasible_step is None or feasible_reports is None or feasible_summary is None:
        v7.set_selected_rows(output_weight, selected_ids, base_selected_rows, zero)
        write_json(root / "failure.json", {
            "status": "FAILED_NO_BF16_FEASIBLE_RARE_TOKEN_STAGE1",
            "method": METHOD,
            "seed": int(a.seed),
            "selected_position_count": len(selected_positions),
            "editable_unique_row_count": len(selected_ids),
            "last_logged_metrics": logs[-1] if logs else None,
        })
        raise RuntimeError("V9 Stage1 did not make all selected rare tokens BF16-safe")

    # Shrink Stage1 toward Base while preserving selected forgetting and every
    # originally Base-safe position.  Residual Base-active positions are allowed.
    low, high = 0.0, 1.0
    bisection_log: List[Dict[str, Any]] = []
    for iteration in range(1, a.bf16_bisection_steps + 1):
        mid = 0.5 * (low + high)
        mid_delta = feasible_delta * float(mid)
        reports, _ = v72.exact_materialized_audit(
            model, tok, cases,
            output_weight=output_weight,
            selected_ids=selected_ids,
            base_selected_rows=base_selected_rows,
            delta=mid_delta,
            device=device, llama_like=llama_like,
            batch_size=a.batch_size, target_margin=a.target_logit_margin,
        )
        ok = exact_stage1_pass(reports, selected_positions, base_safe_positions)
        counts = exact_counts(reports, selected_positions, base_safe_positions)
        bisection_log.append({
            "iteration": iteration,
            "alpha": mid,
            "passed": ok,
            "scaled_delta_norm": float(mid_delta.norm().detach().cpu()),
            **counts,
        })
        if ok:
            high = mid
        else:
            low = mid

    final_alpha = float(high)
    final_delta = feasible_delta * final_alpha
    final_reports, final_summary = v72.exact_materialized_audit(
        model, tok, cases,
        output_weight=output_weight,
        selected_ids=selected_ids,
        base_selected_rows=base_selected_rows,
        delta=final_delta,
        device=device, llama_like=llama_like,
        batch_size=a.batch_size, target_margin=a.target_logit_margin,
    )
    if not exact_stage1_pass(final_reports, selected_positions, base_safe_positions):
        raise RuntimeError("V9 Stage1 final BF16 shrink-back failed selected/safe audit")
    final_counts = exact_counts(final_reports, selected_positions, base_safe_positions)

    if int(input_weight.data_ptr()) != input_pointer or int(input_weight._version) != input_version:
        raise RuntimeError("V9 Stage1 modified input embeddings")

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    v7.write_jsonl(root / "all_visible_tokens_after_bf16.jsonl", final_reports)
    v7.write_jsonl(root / "bf16_bisection_log.jsonl", bisection_log)

    for row in selection_rows:
        token_id = int(row["target_token_id"])
        row["token"] = v7.decoded_token(tok, token_id)
    write_json(root / "rare_token_selection.json", {
        "definition": "up to K distinct Base-active target-token ids per original MQuAKE source instance; lower instance-level DF first, then Base margin closest to zero",
        "top_k_per_instance": int(a.top_k_per_instance),
        "selected_position_count": len(selected_positions),
        "editable_unique_lm_head_row_count": len(selected_ids),
        "selected_positions": selected_positions,
        "selected_token_ids": selected_ids,
        "token_instance_document_frequency": {str(k): int(v) for k, v in sorted(token_df.items())},
        "selection_rows": selection_rows,
    })

    summary = {
        "status": "PASS_V90_RARE_TOKEN_STAGE1",
        "method": METHOD,
        "protocol": PROTOCOL,
        "seed": int(a.seed),
        "forget_instances": int(a.forget_num),
        "forget_atomic_facts": len(records),
        "visible_sensitive_token_cases": len(cases),
        "base_active_sensitive_token_count": int(base_active_mask_cpu.sum().item()),
        "selected_position_count": len(selected_positions),
        "selected_unique_lm_head_row_count": len(selected_ids),
        "optimizer_feasible_step": int(feasible_step),
        "bf16_minimal_ray_alpha": final_alpha,
        "selected_lm_head_delta_norm": float(final_delta.norm().detach().cpu()),
        "materialized_bf16_metrics": final_summary,
        **final_counts,
        "stage2_required": bool(final_counts["residual_active_sensitive_token_count"] > 0),
        "input_embeddings_exact_base": True,
        "transformer_exact_base": True,
        "non_selected_lm_rows_exact_base": True,
        "training_data_access": {
            "forget_instances": int(a.forget_num),
            "forget_atomic_facts": len(records),
            "prompt_types": ["requested_rewrite"],
            "benchmark_retain_instances": 0,
            "atomic_questions": 0,
            "multihop_questions": 0,
            "benchmark_counterfactual_targets": 0,
            "PPL": False,
        },
        "checkpoint_selection_uses_retain_or_heldout": False,
        "checkpoint": str(ckpt.resolve()),
    }
    write_json(root / "repair_summary.json", summary)
    write_json(root / "config_used.json", {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        **vars(a),
        "repair_visible_path_resolved": str(visible_path),
        "split_manifest_resolved": str(manifest_path),
        "split_sampling": split_manifest.get("sampling"),
        "selection_definition": "per source instance: Base-active distinct token ids sorted by (instance DF ascending, Base margin descending), take top K",
        "parameter_scope": "selected rare/easy LM-head rows only; transformer/input embeddings/non-selected LM rows exact Base",
        "forget_definition": "squared hinge only on selected rare/easy positions",
        "gd_definition": "same-prompt non-target KL over all visible direct token decisions plus Base-safe hinge",
    })

    print("===== SURE-MQuAKE V9.0 RARE-TOKEN STAGE1 =====")
    print(
        f"instances={a.forget_num} atomic_facts={len(records)} token_cases={len(cases)} "
        f"base_active={int(base_active_mask_cpu.sum().item())} selected_positions={len(selected_positions)} "
        f"editable_rows={len(selected_ids)}"
    )
    print(
        f"optimizer_feasible_step={feasible_step} final_alpha={final_alpha:.8f} "
        f"final_norm={float(final_delta.norm().detach().cpu()):.6g}"
    )
    print(
        f"selected_active={final_counts['selected_position_active_count']} "
        f"base_safe_reactivated={final_counts['base_safe_reactivated_count']} "
        f"residual_active_all_visible={final_counts['residual_active_sensitive_token_count']}"
    )
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
