#!/usr/bin/env python3
"""MCF SURE Stage 2: unrestricted sparse LM-head repair for Stage-1 failures.

This stage is deliberately simpler than the canonical rank-candidate repair:
  * target_true is sensitive; target_new is the non-sensitive reference;
  * detect failed direct records at the input checkpoint;
  * select LM-head rows only from target_true tokens of those failed records;
  * optimize one unrestricted full-row delta (no LoRA, no low-rank basis);
  * the primary hinge loss is computed only on the failed records;
  * previously passing direct records contribute a regression guard AND an
    exact same-prompt non-target KL penalty, so the unrestricted delta cannot
    freely drift in directions that only happen not to flip the 50 in-sample
    margins while still damaging the row's behavior everywhere else it is
    used (this is what specificity/PPL evaluation actually probes);
  * select the smallest direct-only scale yielding zero failures when possible;
  * materialize only the selected LM-head rows.

Input embeddings and all transformer parameters are frozen during Stage 2.
No official paraphrases, neighborhoods, benchmark-retain records, or PPL text
are opened for optimization or checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F

import gagd_compare as gagd
import gagd_active_case_repair as mcf_repair
import mcf_synthetic_paraphrase_templates as synth
import sure_canonical_core as core
import sure_stage2_sparse_repair as shared


METHOD = "SURE-MCF-failure-only-unrestricted-LM-head-repair"
PROTOCOL = "mcf_target_true_sensitive_failure_fullrow_repair_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--repair-steps", type=int, default=800)
    p.add_argument("--repair-lr", type=float, default=5e-3)
    p.add_argument("--constraint-margin", type=float, default=0.05)
    p.add_argument(
        "--repair-l2",
        type=float,
        default=1e-3,
        help=(
            "Raised from 1e-6: with the wider direct+synthetic objective, "
            "effective_delta_norm reached ~10.6 (||delta||^2 ~ 112), making "
            "the old 1e-6 * ||delta||^2 ~ 0.0001 term negligible next to a "
            "failure hinge that starts around 100+ per case (margins near "
            "-13, squared). At 1e-3 the L2 term is ~0.1 -- still small "
            "relative to an unsatisfied hinge, but large enough to actually "
            "discourage unnecessary delta magnitude once cases are passing."
        ),
    )
    p.add_argument(
        "--protected-kl-max",
        type=float,
        default=0.05,
        help=(
            "Hard gate (not a loss weight): every optimizer step is "
            "backtracked/rolled back unless it keeps (a) every currently-"
            "passing direct-only record's margin >= constraint-margin, "
            "unconditionally, and (b) same-prompt non-target KL on all "
            "currently-passing records <= this value. Replaces the earlier "
            "soft pass-guard-weight/distribution-kl-weight loss terms: those "
            "were a competing pressure that could still be outweighed by the "
            "failure hinge (weight 1.0 left Spe collapsed at 0.16) or, when "
            "raised enough to matter, could itself outweigh the hinge and "
            "regress already-passing direct records (weight 10.0 dropped "
            "Eff from 0.0 to 12.0). A hard gate makes 'never regress a "
            "passing record' a guarantee instead of a trial-and-error "
            "weight search. Mirrors mcf_sure_protected_subspace_stage2.py's "
            "--protected-kl-max (same default). Set 0 to disable the KL half "
            "of the gate and keep only the margin-regression guard."
        ),
    )
    p.add_argument(
        "--backtrack-scales",
        default="1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0.00390625,0.001953125,0.0009765625,0.00048828125,0.0",
        help=(
            "Fractions of each raw optimizer step tried, largest first, "
            "until the hard gate is satisfied. The list always ends in 0.0 "
            "(full rollback to the pre-step delta), which trivially "
            "satisfies the gate -- so a step can never be silently dropped "
            "without a fallback, unlike a plain rollback-only implementation."
        ),
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument(
        "--candidate-scales",
        default="1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0",
    )
    p.add_argument(
        "--synthetic-paraphrases-per-record",
        type=int,
        default=3,
        help=(
            "Hand-authored synthetic paraphrase templates per record used to "
            "detect active/failing cases, train the repair delta, and gate "
            "scale selection. Unlike Stage 1's direction-constrained delta, "
            "Stage 2's unrestricted delta is fit directly against hidden "
            "states (no sensitive-minus-reference contrast direction), so it "
            "is not subject to the decoder-row fallback that made Stage 1's "
            "synthetic-prompt augmentation a no-op for single-token answers. "
            "Set 0 to disable and match the original direct-only behavior."
        ),
    )
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    a = p.parse_args(list(argv) if argv is not None else None)
    if min(a.forget_num, a.repair_steps, a.batch_size, a.check_every) <= 0:
        p.error("counts, repair steps, batch size and check interval must be positive")
    if a.repair_lr <= 0:
        p.error("repair-lr must be positive")
    if min(a.constraint_margin, a.repair_l2, a.protected_kl_max) < 0:
        p.error("margin, L2 and protected-kl-max must be non-negative")
    if a.synthetic_paraphrases_per_record < 0:
        p.error("synthetic-paraphrases-per-record must be non-negative")
    backtrack_scales = parse_backtrack_scales(a.backtrack_scales)
    if backtrack_scales[-1] != 0.0:
        p.error("backtrack-scales must end in 0.0 (guaranteed-safe full rollback)")
    return a


def parse_backtrack_scales(text: str) -> List[float]:
    scales = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not scales:
        raise ValueError("No backtrack scales provided")
    for value in scales:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Backtrack scales must be finite and within [0, 1]")
    return scales


def select_repair_scale(reports: List[Dict[str, Any]]) -> float:
    """Same three-pass selection as Stage 1's select_stage1_scale: never let
    the harder combined (direct+synthetic) objective discard a scale that
    already achieves the best available direct-only result."""
    best_direct_only = min(int(r["direct_only_failures"]) for r in reports)
    candidates = [r for r in reports if int(r["direct_only_failures"]) == best_direct_only]
    best_combined = min(int(r["direct_failures"]) for r in candidates)
    candidates = [r for r in candidates if int(r["direct_failures"]) == best_combined]
    return float(max(float(r["scale"]) for r in candidates))


def validate_locked(
    visible_path: Path,
    manifest_path: Path,
    seed: int,
    forget_num: int,
):
    records, manifest = shared.load_locked(
        "mcf", visible_path, manifest_path, seed, forget_num
    )
    contract = manifest.get("target_contract", {})
    if isinstance(contract, Mapping) and contract:
        if contract.get("sensitive_answer") not in (
            None,
            "requested_rewrite.target_true",
        ):
            raise RuntimeError("Stage 2 requires target_true-sensitive MCF")
        if contract.get("non_sensitive_reference") not in (
            None,
            "requested_rewrite.target_new",
        ):
            raise RuntimeError("Stage 2 requires target_new as reference")
        if contract.get("field_swapping") not in (None, False):
            raise RuntimeError("Stage 2 requires unswapped MCF fields")
    return records, manifest


def margins_from_caches(caches, delta: torch.Tensor) -> torch.Tensor:
    return shared.mcf_margins_from_delta_caches(
        caches,
        delta,
        sensitive_field="target_true",
        reference_field="target_new",
    )


def failure_sensitive_rows(tok, instances, active_positions: Sequence[int]) -> List[int]:
    return shared.mcf_sensitive_rows(
        tok,
        instances,
        active_positions,
        sensitive_field="target_true",
    )


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(int(a.seed))
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
            f"{synthetic_coverage['generic_fallback_relation_ids']}."
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
        raise RuntimeError("model lacks input embeddings")
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tok)

    direct_count = len(records)
    all_instances = shared.mcf_instances(all_records)
    original_margins = shared.mcf_direct_margins(
        model,
        tok,
        all_instances,
        device,
        llama_like,
        int(a.batch_size),
        sensitive_field="target_true",
        reference_field="target_new",
    )
    original_cpu = original_margins.detach().float().cpu()
    # Active/failing now spans direct + synthetic-paraphrase instances, so
    # Stage 2 repairs paraphrase-margin residuals too, not only the literal
    # direct prompt.
    active_positions = [
        i
        for i, value in enumerate(original_cpu.tolist())
        if float(value) < float(a.constraint_margin)
    ]
    passing_positions = [
        i for i in range(len(all_instances)) if i not in set(active_positions)
    ]
    direct_active_positions = [i for i in active_positions if i < direct_count]
    synthetic_active_positions = [i for i in active_positions if i >= direct_count]
    selected_ids = failure_sensitive_rows(tok, all_instances, active_positions)

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    logs: List[Dict[str, Any]] = []
    scale_reports: List[Dict[str, Any]] = []
    best_step = 0
    best_failures = len(active_positions)
    selected_scale = 0.0
    gate_rejected_steps = 0
    gate_backtracked_steps = 0
    backtrack_scales = parse_backtrack_scales(a.backtrack_scales)
    final_delta = torch.empty(
        (0, int(output_layer.weight.shape[1])),
        dtype=torch.float32,
        device=output_layer.weight.device,
    )

    if selected_ids:
        caches = mcf_repair.build_prompt_instance_delta_caches(
            model,
            tok,
            all_instances,
            selected_ids,
            device,
            int(a.batch_size),
            llama_like,
        )
        active_caches = [caches[i] for i in active_positions]

        delta_module = core.SelectedRowDelta(
            len(selected_ids),
            int(output_layer.weight.shape[1]),
            direction_basis=None,
            device=output_layer.weight.device,
        )
        opt = torch.optim.AdamW(
            delta_module.parameters(), lr=float(a.repair_lr), weight_decay=0.0
        )
        best_delta = delta_module.effective_delta().detach().clone()
        best_key = (10**9, 10**9, float("inf"))

        for step in range(1, int(a.repair_steps) + 1):
            # Hard gate, computed fresh each step from the model's current
            # state: a record becomes "protected" the moment it passes, and
            # can never be allowed to regress afterwards. This replaces the
            # earlier soft pass-guard/KL-weight loss terms, which were a
            # competing pressure a large-enough failure hinge could still
            # outweigh (weight 1.0: Spe collapsed) or that, once raised
            # enough to matter, could itself outweigh the hinge and regress
            # already-passing direct records (weight 10.0: Eff 0.0 -> 12.0).
            with torch.no_grad():
                pre_delta = delta_module.raw_delta.detach().clone()
                pre_all_margins = margins_from_caches(caches, pre_delta)
                protected_direct_positions = [
                    i
                    for i in range(direct_count)
                    if float(pre_all_margins[i]) >= float(a.constraint_margin)
                ]
                protected_positions = [
                    i
                    for i, value in enumerate(pre_all_margins.tolist())
                    if float(value) >= float(a.constraint_margin)
                ]
                protected_direct_caches = [caches[i] for i in protected_direct_positions]
                protected_caches = [caches[i] for i in protected_positions]

            opt.zero_grad(set_to_none=True)
            delta = delta_module.effective_delta()
            active_margins = margins_from_caches(active_caches, delta)
            failure_hinge = F.relu(
                float(a.constraint_margin) - active_margins
            ).square().mean()
            l2 = delta.square().mean()
            loss = failure_hinge + float(a.repair_l2) * l2
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage-2 loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(delta_module.parameters()), 1.0)
            opt.step()

            with torch.no_grad():
                raw_update = delta_module.raw_delta.detach() - pre_delta
                accepted_scale = 0.0
                accepted_delta = pre_delta
                for bscale in backtrack_scales:
                    candidate = pre_delta + raw_update * float(bscale)
                    ok = True
                    if protected_direct_caches:
                        candidate_direct_margins = margins_from_caches(
                            protected_direct_caches, candidate
                        )
                        if bool(
                            (candidate_direct_margins < float(a.constraint_margin)).any()
                        ):
                            ok = False
                    if ok and protected_caches and float(a.protected_kl_max) > 0:
                        candidate_kl = float(
                            mcf_repair.mcf_same_prompt_non_target_kl(
                                protected_caches, candidate
                            )
                        )
                        if candidate_kl > float(a.protected_kl_max):
                            ok = False
                    if ok:
                        accepted_scale = float(bscale)
                        accepted_delta = candidate
                        break
                if accepted_scale == 0.0:
                    gate_rejected_steps += 1
                elif accepted_scale < 1.0:
                    gate_backtracked_steps += 1
                delta_module.raw_delta.data.copy_(accepted_delta)

            if step == 1 or step % int(a.check_every) == 0 or step == int(a.repair_steps):
                with torch.no_grad():
                    current = delta_module.effective_delta()
                    all_margins = margins_from_caches(caches, current)
                    direct_only_margins = all_margins[:direct_count]
                    failures = int(
                        (all_margins < float(a.constraint_margin)).sum().item()
                    )
                    direct_only_failures = int(
                        (direct_only_margins < float(a.constraint_margin)).sum().item()
                    )
                    norm = float(current.norm().detach().cpu())
                    row = {
                        "step": int(step),
                        "all_direct_failures": failures,
                        "direct_only_failures": direct_only_failures,
                        "active_failure_hinge": float(failure_hinge.detach().cpu()),
                        "accepted_backtrack_scale": accepted_scale,
                        "gate_rejected_steps_so_far": gate_rejected_steps,
                        "gate_backtracked_steps_so_far": gate_backtracked_steps,
                        "minimum_margin": float(all_margins.min().detach().cpu()),
                        "delta_norm": norm,
                        "lora_used": False,
                        "rank_constraint": False,
                    }
                    logs.append(row)
                    # direct_only_failures first: never let a lower combined
                    # failure count or smaller norm elsewhere in training be
                    # preferred over an already-achieved perfect direct-only
                    # result -- same principle as select_repair_scale's
                    # scale-sweep priority. The hard gate above should make
                    # this ordering redundant in practice (protected direct
                    # records cannot regress), but it costs nothing to keep
                    # as a second line of defense.
                    key = (direct_only_failures, failures, norm)
                    if key < best_key:
                        best_key = key
                        best_step = int(step)
                        best_failures = int(failures)
                        best_delta = current.detach().clone()
                    if failures == 0:
                        break
        del opt

        scales = core.parse_scales(a.candidate_scales)
        for scale in scales:
            margins = margins_from_caches(caches, best_delta * float(scale))
            direct_margins = margins[:direct_count]
            scale_reports.append(
                {
                    "scale": float(scale),
                    "direct_failures": int(
                        (margins < float(a.constraint_margin)).sum().item()
                    ),
                    "direct_only_failures": int(
                        (direct_margins < float(a.constraint_margin)).sum().item()
                    ),
                    "minimum_margin": float(margins.min().detach().cpu()),
                    "direct_only_minimum_margin": float(
                        direct_margins.min().detach().cpu()
                    ),
                    "effective_delta_norm": float(
                        best_delta.norm().detach().cpu() * float(scale)
                    ),
                }
            )
        # select_repair_scale (not core.choose_scale): never let the harder
        # combined direct+synthetic objective collapse to scale=0.0 when it
        # cannot reach zero failures -- that would silently discard an edit
        # that already achieves the best available direct-only result (see
        # mcf_sure_directional_emb_lm_stage1.py's identical fix).
        selected_scale = select_repair_scale(scale_reports)
        final_delta = best_delta * float(selected_scale)
        final_distribution_kl = float(
            mcf_repair.mcf_same_prompt_non_target_kl(caches, final_delta)
            .detach()
            .cpu()
        )
        core.materialize_output_delta(output_layer, selected_ids, final_delta)
    else:
        final_distribution_kl = 0.0

    final_all_margins = shared.mcf_direct_margins(
        model,
        tok,
        all_instances,
        device,
        llama_like,
        int(a.batch_size),
        sensitive_field="target_true",
        reference_field="target_new",
    )
    final_margins = final_all_margins[:direct_count]
    final_synthetic_margins = final_all_margins[direct_count:]
    final_cpu = final_margins.detach().float().cpu()
    final_failure_positions = [
        i
        for i, value in enumerate(final_cpu.tolist())
        if float(value) < float(a.constraint_margin)
    ]
    final_synthetic_failure_positions = [
        i
        for i, value in enumerate(final_synthetic_margins.detach().cpu().tolist())
        if float(value) < float(a.constraint_margin)
    ]

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    summary: Dict[str, Any] = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "source_protocol": manifest.get("protocol"),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "target_contract": {
            "sensitive_answer": "requested_rewrite.target_true",
            "non_sensitive_reference": "requested_rewrite.target_new",
            "field_swapping": False,
        },
        "stage1_failure_count": len(active_positions),
        "stage1_failure_positions": active_positions,
        "stage1_passing_positions": passing_positions,
        "stage1_direct_failure_count": len(direct_active_positions),
        "stage1_synthetic_failure_count": len(synthetic_active_positions),
        "synthetic_paraphrases_per_record": int(a.synthetic_paraphrases_per_record),
        "synthetic_record_count": len(synthetic_records),
        "synthetic_paraphrase_coverage": synthetic_coverage,
        "selected_lm_head_rows": len(selected_ids),
        "selected_token_ids": selected_ids,
        "parameterization": "unrestricted_sparse_full_lm_head_rows",
        "lora_used": False,
        "rank_constraint": False,
        "repair_primary_training_records": (
            "Stage-1 failed direct records + synthetic-paraphrase templates"
        ),
        "passing_records_role": (
            "hard gate only: any record whose margin is >= constraint-margin "
            "at the start of a step is protected for that step -- its margin "
            "may not drop below constraint-margin, and same-prompt "
            "non-target KL over all protected records may not exceed "
            "protected-kl-max. A step violating either is backtracked "
            "(geometric scale-down of the raw update) or fully rolled back."
        ),
        "protected_kl_max": float(a.protected_kl_max),
        "backtrack_scales": backtrack_scales,
        "gate_rejected_steps": gate_rejected_steps,
        "gate_backtracked_steps": gate_backtracked_steps,
        "final_distribution_kl": final_distribution_kl,
        "final_distribution_kl_definition": (
            "post-hoc diagnostic only (not a gate input): exact KL(input-"
            "checkpoint non-target || current non-target) at every visible "
            "direct+synthetic teacher-forced position for the final "
            "materialized delta"
        ),
        "constraint_margin": float(a.constraint_margin),
        "repair_steps": int(a.repair_steps),
        "repair_lr": float(a.repair_lr),
        "repair_l2": float(a.repair_l2),
        "best_step": int(best_step),
        "best_unscaled_direct_failures": int(best_failures),
        "logs": logs,
        "scale_reports": scale_reports,
        "selected_scale": float(selected_scale),
        "effective_delta_norm": float(final_delta.norm().detach().cpu())
        if final_delta.numel()
        else 0.0,
        "final_direct_failures": len(final_failure_positions),
        "final_failing_positions": final_failure_positions,
        "final_minimum_margin": float(final_cpu.min().item()),
        "final_synthetic_failures": len(final_synthetic_failure_positions),
        "final_synthetic_failing_positions": final_synthetic_failure_positions,
        "final_synthetic_minimum_margin": (
            float(final_synthetic_margins.min().detach().cpu())
            if final_synthetic_margins.numel()
            else None
        ),
        "final_combined_failures": (
            len(final_failure_positions) + len(final_synthetic_failure_positions)
        ),
        "input_embeddings_modified_in_stage2": False,
        "transformer_trainable_parameters": 0,
        "lm_head_untied": True,
        "official_paraphrases_seen": 0,
        "official_neighborhood_seen": 0,
        "benchmark_retain_seen": 0,
        "ppl_eval_text_seen": 0,
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "repair_summary.json", summary)
    core.write_json(out_dir / "scale_sweep_direct_only.json", scale_reports)
    print(json.dumps(summary, indent=2))
    print(
        f"Full-row Stage 2: direct failures {len(direct_active_positions)} -> "
        f"{len(final_failure_positions)}; synthetic failures "
        f"{len(synthetic_active_positions)} -> {len(final_synthetic_failure_positions)}; "
        f"selected rows={len(selected_ids)}; scale={selected_scale:g}"
    )


if __name__ == "__main__":
    main()
