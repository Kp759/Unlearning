#!/usr/bin/env python3
"""Train MCF V2.3: frozen-LM-head projected row-partition embedding rewiring.

V2.3 is the frozen-head sibling of V2.2.  The Transformer is frozen, the LM head
is untied and must stay bit-identical, and only selected input-embedding rows
may change.  The head is still used -- as the closed-form answer-contrast
reference ``q_i`` and as the reader whose retain-context outputs define the
protected subspace -- but it is never an edit site.

Why the head is frozen here
---------------------------
``logits = W h`` has no context input of its own, so one ``Delta W[a]`` moves
that answer's logit in *every* context.  On CounterFact the neighborhood prompts
that define specificity have different subjects but the same correct answer, so
efficacy and specificity are coupled through a single shared parameter.  V2,
V2.1 and V2.2 all attempted to defeat that coupling with geometry; V2.1 ended
with a protected correction of 1.741288 against a 0.02 ceiling.  V2.3 removes
the coupling by construction and spends its whole budget on the input side,
where the frozen Transformer supplies context dependence for free and a prompt
containing none of the edited rows is bit-identical to Base.

The three mechanisms
--------------------
1. ``q_i = normalize(W[target_new_i] - W[target_true_i])`` is the exact axis of
   the trained margin under a frozen head, replacing ``W[target_true]``.
2. Every candidate row is scored by how much of its forget gradient survives
   projection out of its retain-readout subspace (``efficacy``) and by that
   surviving magnitude scaled by its own norm cap (``potential``).  Rows are
   partitioned into free / projected / excluded from those two numbers.
3. Projection is a warm start, not a certificate.  Every proposed step is
   projected, capped, then *forward evaluated*; it is accepted only if the
   retain constraints actually hold on the active bank.

Claim boundary: a passing run supports behavioural, context-selective
forgetting.  Because the Transformer stays frozen, it does not establish latent
erasure, and a prompt containing none of the edited token ids is Base by
construction -- token-disjoint aliases are out of scope unless a training-safe
alias lexicon is registered before the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F

import build_mcf_biendpoint_nullspace_rewiring_v2_split as split_builder
import gagd_compare as gagd
import mcf_biendpoint_nullspace_rewiring_v2_core as geometry
import mcf_folded_sensitivity_rewiring_v2_1_core as folded
import mcf_joint_forget_retain_endpoint_v2_2_core as joint
import mcf_projected_row_partition_embedding_v2_3_core as partition
import mcf_sure_directional_emb_lm_stage1 as endpoint_hooks
import mcf_sure_subject_directional_emb_stage1 as subject_stage1
import mcf_synthetic_paraphrase_templates as synthetic
import run_mcf_biendpoint_nullspace_rewiring_v2 as v2
import run_mcf_folded_sensitivity_rewiring_v2_1 as v21
import run_mcf_joint_forget_retain_endpoint_v2_2 as v22
import sure_canonical_core as canonical


BACKTRACKING_FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625)
REACHABILITY_FACTORS = (0.0625, 0.125, 0.25, 0.5, 1.0)
PROBE_FAMILIES = ("base_top1_logprob", "centered_topk_logit", "retain_answer_logprob")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--experiment-registry", required=True)
    parser.add_argument("--wikidata-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--frequency-doc-start", type=int, default=20)
    parser.add_argument("--frequency-docs", type=int, default=12000)
    parser.add_argument("--corpus-fit-prompts", type=int, default=1000)
    parser.add_argument("--corpus-development-prompts", type=int, default=250)
    parser.add_argument("--corpus-certification-prompts", type=int, default=1000)
    parser.add_argument("--synthetic-paraphrases", type=int, default=3)
    parser.add_argument("--same-subject-prompts", type=int, default=4)
    parser.add_argument("--targeted-fit-per-row", type=int, default=2)
    parser.add_argument("--targeted-development-per-row", type=int, default=1)
    parser.add_argument("--targeted-certification-per-row", type=int, default=1)
    parser.add_argument("--retain-jacobian-sketches", type=int, default=192)
    parser.add_argument("--retain-rank-cap", type=int, default=64)
    parser.add_argument("--partition-efficacy-min", type=float, default=0.05)
    parser.add_argument("--partition-potential-floor", type=float, default=0.01)
    parser.add_argument("--partition-frequency-max", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--check-every", type=int, default=50)
    parser.add_argument("--hard-tail-refresh-every", type=int, default=50)
    parser.add_argument("--hard-tail-add", type=int, default=32)
    parser.add_argument("--hard-tail-capacity", type=int, default=256)
    parser.add_argument("--hard-tail-active", type=int, default=16)
    parser.add_argument("--forget-batch-size", type=int, default=16)
    parser.add_argument("--random-retain-batch-size", type=int, default=16)
    parser.add_argument("--overlap-retain-batch-size", type=int, default=16)
    parser.add_argument("--active-retain-maximum", type=int, default=48)
    parser.add_argument("--capture-batch-size", type=int, default=8)
    parser.add_argument("--forget-margin-target", type=float, default=6.0)
    parser.add_argument("--minimum-forget-margin", type=float, default=0.1)
    parser.add_argument("--forget-weight", type=float, default=1.0)
    parser.add_argument("--retain-kl-weight", type=float, default=100.0)
    parser.add_argument("--retain-top1-weight", type=float, default=100.0)
    parser.add_argument("--retain-target-weight", type=float, default=100.0)
    parser.add_argument("--surgical-weight", type=float, default=0.0)
    parser.add_argument("--surgical-sign-margin", type=float, default=1.0)
    parser.add_argument("--contrast-epsilon", type=float, default=1e-3)
    parser.add_argument("--delta-l2-weight", type=float, default=1e-4)
    parser.add_argument("--input-relative-cap", type=float, default=0.5)
    parser.add_argument("--input-frequency-alpha", type=float, default=0.25)
    parser.add_argument("--input-step-cap-fraction", type=float, default=0.004)
    parser.add_argument("--require-per-prompt-reachability", action="store_true")
    parser.add_argument("--require-per-example-sensitivity", action="store_true")
    parser.add_argument("--require-hybrid-role-policy", action="store_true")
    parser.add_argument("--require-iterative-reachability", action="store_true")
    parser.add_argument("--iterative-reachability-steps", type=int, default=64)
    parser.add_argument(
        "--iterative-reachability-step-fraction", type=float, default=0.02
    )
    parser.add_argument(
        "--iterative-reachability-minimum-improvement",
        type=float,
        default=1e-6,
    )
    parser.add_argument("--sensitivity-forget-records", type=int, default=50)
    parser.add_argument("--sensitivity-retain-records", type=int, default=2000)
    parser.add_argument(
        "--sensitivity-coverage-relative-epsilon", type=float, default=1e-4
    )
    parser.add_argument(
        "--forget-importance-floor-relative", type=float, default=0.01
    )
    parser.add_argument("--importance-ratio-min", type=float, default=1.0)
    parser.add_argument("--forget-specific-ratio-min", type=float, default=4.0)
    parser.add_argument(
        "--forget-specific-retain-coverage-max", type=float, default=0.01
    )
    parser.add_argument("--retain-tail-ratio-min", type=float, default=0.25)
    parser.add_argument(
        "--minimum-forget-loss-improvement", type=float, default=1e-5
    )
    parser.add_argument("--full-fit-rollback", action="store_true")
    parser.add_argument("--protection-topk", type=int, default=64)
    parser.add_argument("--protected-kl-mean-max", type=float, default=1e-4)
    parser.add_argument("--protected-kl-max", type=float, default=1e-2)
    parser.add_argument("--protected-top1-drift-max", type=float, default=5e-2)
    parser.add_argument("--protected-target-drift-max", type=float, default=5e-2)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single",), default="single")
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.gradient_checkpointing = False
    if args.seed != 1 or args.forget_num != 50:
        parser.error("V2.3 is locked to consumed seed 1 / 50 facts")
    if args.frequency_doc_start < 20:
        parser.error("Wikipedia documents 0:20 remain reserved for official PPL")
    if args.steps % args.check_every or args.steps % args.hard_tail_refresh_every:
        parser.error("steps must divide check and hard-tail refresh intervals")
    if args.hard_tail_active >= args.active_retain_maximum:
        parser.error("hard tail must leave room for overlap and random retain strata")
    if (
        args.hard_tail_active
        + args.overlap_retain_batch_size
        + args.random_retain_batch_size
        < args.active_retain_maximum
    ):
        parser.error("active retain maximum exceeds the registered strata budget")
    if not 0.0 < args.partition_efficacy_min < 1.0:
        parser.error("partition efficacy floor must lie strictly inside (0, 1)")
    if not 0.0 <= args.partition_potential_floor < 1.0:
        parser.error("partition potential floor is relative to the strongest row")
    if args.minimum_forget_loss_improvement <= 0.0:
        parser.error("minimum forget-loss improvement must be positive")
    if args.sensitivity_forget_records != args.forget_num:
        parser.error("sensitivity must use all 50 direct forget records")
    if args.sensitivity_retain_records != 2000:
        parser.error("sensitivity is locked to all 2000 protection-fit records")
    if not 0.0 < args.sensitivity_coverage_relative_epsilon < 1.0:
        parser.error("sensitivity coverage epsilon must lie inside (0, 1)")
    if not 0.0 <= args.forget_importance_floor_relative < 1.0:
        parser.error("forget importance floor must lie inside [0, 1)")
    if args.importance_ratio_min <= 0.0:
        parser.error("importance ratio floor must be positive")
    if args.forget_specific_ratio_min < args.importance_ratio_min:
        parser.error("forget-specific ratio must not be below the keep ratio")
    if not 0.0 <= args.forget_specific_retain_coverage_max <= 1.0:
        parser.error("forget-specific retain coverage must lie inside [0, 1]")
    if args.retain_tail_ratio_min <= 0.0:
        parser.error("retain-tail ratio floor must be positive")
    if args.iterative_reachability_steps <= 0:
        parser.error("iterative reachability steps must be positive")
    if not 0.0 < args.iterative_reachability_step_fraction <= 1.0:
        parser.error("iterative reachability step fraction must lie inside (0, 1]")
    if args.iterative_reachability_minimum_improvement <= 0.0:
        parser.error("iterative reachability improvement must be positive")
    return args


def validate_registry(registry: Mapping[str, Any], args: argparse.Namespace) -> None:
    architecture = registry.get("architecture", {})
    diagnostic = registry.get("row_partition_diagnostic", {})
    optimization = registry.get("optimization", {})
    retain_strata = registry.get("retain_strata", {})
    registered_protocol = registry.get("protocol")
    if (
        registered_protocol
        not in (
            partition.PROTOCOL,
            partition.REACHABILITY_PROTOCOL,
            partition.SENSITIVITY_PROTOCOL,
            partition.HYBRID_PROTOCOL,
        )
        or registry.get("status")
        != "training_only_implementation_available_not_executed"
        or architecture.get("trainable_parameter_families")
        != ["selected_input_embedding_rows"]
        or architecture.get("transformer_frozen") is not True
        or architecture.get("lm_head_frozen_bit_identical") is not True
        or architecture.get("lm_head_trainable") is not False
        or architecture.get("external_classifier") is not False
        or architecture.get("runtime_gate") is not False
        or architecture.get("sidecar") is not False
        or diagnostic.get("empty_basis_implies_forget_exclusive") is not False
        or diagnostic.get("projection_is_first_order_only") is not True
        or optimization.get("adam_forbidden") is not True
        or optimization.get("nonlinear_forward_acceptance") is not True
        or retain_strata.get("present_on_every_embedding_update") is not True
    ):
        raise RuntimeError("V2.3 registry architecture/status mismatch")
    reachability_revision = registered_protocol in (
        partition.REACHABILITY_PROTOCOL,
        partition.SENSITIVITY_PROTOCOL,
        partition.HYBRID_PROTOCOL,
    )
    sensitivity_revision = registered_protocol in (
        partition.SENSITIVITY_PROTOCOL,
        partition.HYBRID_PROTOCOL,
    )
    hybrid_revision = registered_protocol == partition.HYBRID_PROTOCOL
    reachability = registry.get("per_prompt_reachability", {})
    if reachability_revision:
        if (
            args.require_per_prompt_reachability is not True
            or args.full_fit_rollback is not True
            or reachability.get("required_before_optimization") is not True
            or reachability.get("linear_cap_bound") is not True
            or reachability.get("all_direct_and_synthetic_prompts_must_pass")
            is not True
            or optimization.get("strict_forget_improvement") is not True
            or optimization.get("equality_is_rejected") is not True
            or optimization.get("full_fit_rollback") is not True
            or optimization.get("development_controls_updates") is not False
        ):
            raise RuntimeError("V2.3.1 reachability/optimization contract mismatch")
        if hybrid_revision:
            if (
                args.require_iterative_reachability is not True
                or reachability.get("nonlinear_directional_sweep") is not False
                or reachability.get("iterative_gradient_recomputation") is not True
                or reachability.get("retain_guard_checked_during_path") is not True
            ):
                raise RuntimeError("V2.3.3 iterative reachability contract mismatch")
        elif (
            args.require_iterative_reachability
            or reachability.get("nonlinear_directional_sweep") is not True
        ):
            raise RuntimeError("V2.3.1 single-ray reachability contract mismatch")
    elif (
        args.require_per_prompt_reachability
        or args.require_iterative_reachability
        or args.full_fit_rollback
    ):
        raise RuntimeError("V2.3 registry cannot enable V2.3.1-only controls")
    sensitivity = registry.get("per_example_sensitivity", {})
    if sensitivity_revision:
        if (
            args.require_per_example_sensitivity is not True
            or sensitivity.get("required_before_row_partition") is not True
            or sensitivity.get("prompt_classifier") is not False
            or sensitivity.get("ratio_is_sole_safety_criterion") is not False
            or sensitivity.get("retain_hard_tail_guard") is not True
        ):
            raise RuntimeError("V2.3.2 per-example sensitivity contract mismatch")
    elif args.require_per_example_sensitivity:
        raise RuntimeError("only V2.3.2+ may enable per-example sensitivity")
    hybrid_policy = registry.get("hybrid_role_policy", {})
    if hybrid_revision:
        if (
            args.require_hybrid_role_policy is not True
            or hybrid_policy.get("forget_specific") != partition.GUARDED
            or hybrid_policy.get("shared") != partition.PROJECTED
            or hybrid_policy.get("retain_dominant") != partition.EXCLUDED
            or hybrid_policy.get("low_forget") != partition.EXCLUDED
            or hybrid_policy.get("frequency_caps_unchanged") is not True
            or hybrid_policy.get("guarded_forward_retain_checks") is not True
        ):
            raise RuntimeError("V2.3.3 hybrid role contract mismatch")
    elif args.require_hybrid_role_policy:
        raise RuntimeError("only V2.3.3 may enable the hybrid role policy")
    expected = {
        "data": {
            "seed": args.seed,
            "forget_records": args.forget_num,
            "wikipedia_doc_start": args.frequency_doc_start,
            "wikipedia_documents": args.frequency_docs,
            "corpus_fit_prompts": args.corpus_fit_prompts,
            "corpus_development_prompts": args.corpus_development_prompts,
            "corpus_certification_prompts": args.corpus_certification_prompts,
            "synthetic_paraphrases_per_forget_record": args.synthetic_paraphrases,
            "same_subject_different_relation_prompts_per_record": args.same_subject_prompts,
            "targeted_row_fit_prompts": args.targeted_fit_per_row,
            "targeted_row_development_prompts": args.targeted_development_per_row,
            "targeted_row_certification_prompts": args.targeted_certification_per_row,
        },
        "row_partition_diagnostic": {
            "retain_jacobian_sketches": args.retain_jacobian_sketches,
            "retain_rank_cap": args.retain_rank_cap,
            "efficacy_min": args.partition_efficacy_min,
            "relative_potential_floor": args.partition_potential_floor,
            "frequency_max": args.partition_frequency_max,
            "probe_families": list(PROBE_FAMILIES),
        },
        "optimization": {
            "steps": args.steps,
            "check_every": args.check_every,
            "hard_tail_refresh_every": args.hard_tail_refresh_every,
            "hard_tail_add": args.hard_tail_add,
            "hard_tail_capacity": args.hard_tail_capacity,
            "hard_tail_active": args.hard_tail_active,
            "forget_batch_size": args.forget_batch_size,
            "random_retain_batch_size": args.random_retain_batch_size,
            "overlap_retain_batch_size": args.overlap_retain_batch_size,
            "active_retain_maximum": args.active_retain_maximum,
            "forget_weight": args.forget_weight,
            "retain_kl_weight": args.retain_kl_weight,
            "retain_top1_weight": args.retain_top1_weight,
            "retain_target_weight": args.retain_target_weight,
            "surgical_weight": args.surgical_weight,
            "delta_l2_weight": args.delta_l2_weight,
            "input_relative_row_cap": args.input_relative_cap,
            "input_frequency_alpha": args.input_frequency_alpha,
            "input_step_cap_fraction": args.input_step_cap_fraction,
            "backtracking_factors": list(BACKTRACKING_FACTORS),
        },
        "acceptance": {
            "minimum_forget_margin": args.minimum_forget_margin,
            "protected_topk_kl_mean_max": args.protected_kl_mean_max,
            "protected_topk_kl_absolute_max": args.protected_kl_max,
            "protected_top1_logprob_abs_max": args.protected_top1_drift_max,
            "protected_target_logprob_abs_max": args.protected_target_drift_max,
        },
    }
    if reachability_revision:
        expected["per_prompt_reachability"] = {
            "required_margin": args.minimum_forget_margin,
        }
        if hybrid_revision:
            expected["per_prompt_reachability"].update(
                {
                    "iterative_steps": args.iterative_reachability_steps,
                    "step_cap_fraction": (
                        args.iterative_reachability_step_fraction
                    ),
                    "minimum_step_improvement": (
                        args.iterative_reachability_minimum_improvement
                    ),
                    "backtracking_factors": list(BACKTRACKING_FACTORS),
                }
            )
        else:
            expected["per_prompt_reachability"]["directional_factors"] = list(
                REACHABILITY_FACTORS
            )
        expected["optimization"].update(
            {
                "minimum_forget_loss_improvement": (
                    args.minimum_forget_loss_improvement
                ),
                "full_fit_rollback": args.full_fit_rollback,
            }
        )
    if sensitivity_revision:
        expected["per_example_sensitivity"] = {
            "forget_examples": args.sensitivity_forget_records,
            "retain_examples": args.sensitivity_retain_records,
            "coverage_relative_epsilon": (
                args.sensitivity_coverage_relative_epsilon
            ),
            "forget_importance_floor_relative": (
                args.forget_importance_floor_relative
            ),
            "importance_ratio_min": args.importance_ratio_min,
            "forget_specific_ratio_min": args.forget_specific_ratio_min,
            "forget_specific_retain_coverage_max": (
                args.forget_specific_retain_coverage_max
            ),
            "retain_tail_ratio_min": args.retain_tail_ratio_min,
        }
    for section, values in expected.items():
        registered = registry.get(section)
        if not isinstance(registered, Mapping):
            raise RuntimeError(f"V2.3 registry lacks {section}")
        for key, value in values.items():
            if registered.get(key) != value:
                raise RuntimeError(
                    f"V2.3 argument differs from registry: {section}.{key}"
                )


def build_retain_readout_bases(
    model: torch.nn.Module,
    tok: Any,
    cache: v2.ProtectionCache,
    target_ids: torch.Tensor,
    device: torch.device,
    *,
    input_delta: canonical.SelectedRowDelta,
    sketches: int,
    batch_size: int,
    topk: int,
    max_rank: int,
) -> tuple[List[torch.Tensor], Dict[str, Any]]:
    """Per-row retain-readout subspace from three probe families.

    V2 sketched a single top-k *column logit*.  Top-k KL is invariant to a
    constant shift of those logits, and the acceptance gates also constrain the
    Base top-1 log-probability and each labelled retain answer's
    log-probability, so a one-family sketch under-represents what must be
    preserved.  V2.3 cycles the three quantities the gates actually measure:
    the Base top-1 log-probability, the *centered* top-k logit (the shape
    top-k KL responds to), and the labelled retain answer log-probability.
    """
    if input_delta.raw_delta is None:
        raise RuntimeError("V2.3 requires an unrestricted input delta")
    per_row: List[List[torch.Tensor]] = [[] for _ in range(input_delta.n_rows)]
    family_counts = {name: 0 for name in PROBE_FAMILIES}
    total = len(cache.cases)
    if total == 0:
        raise RuntimeError("V2.3 retain basis construction needs a non-empty bank")
    for sketch_index in range(int(sketches)):
        input_delta.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        start = (sketch_index * batch_size) % total
        indices = [(start + offset) % total for offset in range(batch_size)]
        batch = v2.cache_batch(cache, indices, device)
        logits = canonical.forward_last_logits(model, tok, batch["cases"], device)
        log_probs = F.log_softmax(logits.float(), dim=1)
        family = PROBE_FAMILIES[sketch_index % len(PROBE_FAMILIES)]
        local_targets = target_ids.index_select(
            0, torch.tensor(indices, dtype=torch.long)
        ).to(device)
        if family == "retain_answer_logprob" and not bool((local_targets >= 0).any()):
            family = "base_top1_logprob"
        if family == "base_top1_logprob":
            probe = log_probs.gather(1, batch["base_top1_ids"][:, None]).squeeze(1)
        elif family == "centered_topk_logit":
            column = sketch_index % int(topk)
            selected = logits.float().gather(1, batch["topk_ids"])
            probe = selected[:, column] - selected.mean(dim=1)
        else:
            valid = local_targets >= 0
            gathered = log_probs.gather(
                1, local_targets.clamp_min(0)[:, None]
            ).squeeze(1)
            probe = torch.where(valid, gathered, torch.zeros_like(gathered))
        family_counts[family] += 1
        signs = torch.tensor(
            [1.0 if ((sketch_index + value) % 2 == 0) else -1.0 for value in indices],
            dtype=torch.float32,
            device=device,
        )
        (probe * signs).mean().backward()
        gradient = input_delta.raw_delta.grad
        if gradient is None:
            raise RuntimeError("V2.3 retain probe produced no input gradient")
        for row_index, vector in enumerate(gradient.detach().float().cpu()):
            if float(vector.norm().item()) > 1e-12:
                per_row[row_index].append(vector.clone())
    input_delta.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)

    bases: List[torch.Tensor] = []
    ranks: List[int] = []
    observations: List[int] = []
    hidden = int(input_delta.hidden_size)
    for vectors in per_row:
        observations.append(len(vectors))
        if vectors:
            basis = canonical.orthonormal_row_basis(
                torch.stack(vectors), max_rank=int(max_rank)
            )
        else:
            basis = torch.empty((0, hidden), dtype=torch.float32)
        bases.append(basis.to(device))
        ranks.append(int(basis.shape[0]))
    report = {
        "sketches": int(sketches),
        "probe_families": dict(family_counts),
        "rows": len(bases),
        "rows_observed_in_retain": sum(value > 0 for value in observations),
        "rows_with_empty_basis": sum(value == 0 for value in observations),
        "observation_min": min(observations),
        "observation_max": max(observations),
        "rank_min": min(ranks),
        "rank_median": sorted(ranks)[len(ranks) // 2],
        "rank_max": max(ranks),
        "empty_basis_means_unobserved_not_unused": True,
    }
    return bases, report


def build_contrast_cells(
    records: Sequence[Mapping[str, Any]],
    tok: Any,
    *,
    llama_like: bool,
) -> tuple[List[canonical.SensitivePredictionCase], torch.Tensor, torch.Tensor]:
    """Teacher-forced target_true cells paired with the target_new token id.

    At token index ``i`` the forward state is the ``target_true`` prefix, and the
    quantity the hinge moves there is ``logit(new_i) - logit(true_i)``.  Answers
    of unequal length leave the surplus positions unpaired, marked ``-1``.
    """
    cases = canonical.expand_sensitive_cases(
        records, tok, sensitive_field="target_true", llama_like=llama_like
    )
    true_ids: List[int] = []
    new_ids: List[int] = []
    for case in cases:
        rewrite = records[case.record_position]["requested_rewrite"]
        true_tokens = canonical.answer_token_ids(
            tok, str(rewrite["target_true"]["str"]), llama_like=llama_like
        )
        new_tokens = canonical.answer_token_ids(
            tok, str(rewrite["target_new"]["str"]), llama_like=llama_like
        )
        index = int(case.token_index)
        true_ids.append(int(true_tokens[index]) if index < len(true_tokens) else -1)
        new_ids.append(int(new_tokens[index]) if index < len(new_tokens) else -1)
    return (
        cases,
        torch.tensor(true_ids, dtype=torch.long),
        torch.tensor(new_ids, dtype=torch.long),
    )


def direct_live_rows(
    records: Sequence[Mapping[str, Any]], tok: Any, selected: Sequence[int]
) -> Dict[int, List[int]]:
    """Selected token ids occurring in each record's own direct prompt."""
    allowed = set(int(value) for value in selected)
    live: Dict[int, List[int]] = {}
    for position, record in enumerate(records):
        rewrite = record["requested_rewrite"]
        case_id = int(record.get("case_id", position))
        prompt = str(rewrite["prompt"]).format(str(rewrite["subject"]))
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        live[case_id] = sorted(
            {int(value) for value in ids if int(value) in allowed}
        )
    return live


def collect_per_example_sensitivity(
    model: torch.nn.Module,
    tok: Any,
    forget_records: Sequence[Mapping[str, Any]],
    retain_records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    input_delta: canonical.SelectedRowDelta,
    llama_like: bool,
    coverage_relative_epsilon: float,
    forget_importance_floor_relative: float,
    importance_ratio_min: float,
    forget_specific_ratio_min: float,
    forget_specific_retain_coverage_max: float,
    retain_tail_ratio_min: float,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Measure one input-row gradient norm vector per forget/retain record.

    Forget importance uses the direct answer-preference margin. Retain
    importance uses the ground-truth answer NLL. Both are measured in
    log-probability units from exact zero. Only norm summaries and coverage are
    retained; no prompt text or per-example gradient tensor is serialized.
    """
    if input_delta.raw_delta is None:
        raise RuntimeError("per-example sensitivity requires a full row delta")
    original = input_delta.raw_delta.detach().clone()
    if float(original.norm().cpu()) > 1e-12:
        raise RuntimeError("per-example sensitivity must run from exact zero")
    row_count = int(input_delta.raw_delta.shape[0])
    forget_norms: List[torch.Tensor] = []
    retain_norms: List[torch.Tensor] = []
    forget_coverage = torch.zeros(row_count, dtype=torch.long)
    retain_coverage = torch.zeros(row_count, dtype=torch.long)
    forget_gradient_sum = torch.zeros_like(input_delta.raw_delta)
    zero_graph_examples = {"forget": 0, "retain": 0}

    def record_norms(
        scalar: torch.Tensor, *, group: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_delta.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        # The Transformer and LM head are frozen.  When a prompt contains none
        # of the selected input rows, the embedding hook returns the untouched
        # Base tensor, so the scalar correctly has no autograd graph.  That is
        # an exact zero sensitivity observation, not an error.  Calling
        # backward() on such a scalar raises "does not require grad".
        if not scalar.requires_grad:
            zero_graph_examples[group] += 1
            local = torch.zeros_like(input_delta.raw_delta)
        else:
            scalar.backward()
            gradient = input_delta.raw_delta.grad
            if gradient is None:
                local = torch.zeros_like(input_delta.raw_delta)
            else:
                local = gradient.detach().clone()
        norms = local.float().norm(dim=1).cpu()
        maximum = float(norms.max().item()) if norms.numel() else 0.0
        threshold = max(
            1e-12, float(coverage_relative_epsilon) * maximum
        )
        return local, norms >= threshold

    try:
        with torch.no_grad():
            input_delta.raw_delta.zero_()
        for index, record in enumerate(forget_records):
            margin = v2.records_margin_tensor(
                model, tok, [record], device, llama_like=llama_like
            )[0]
            gradient, covered = record_norms(margin, group="forget")
            norms = gradient.float().norm(dim=1).cpu()
            forget_norms.append(norms)
            forget_coverage.add_(covered.long())
            forget_gradient_sum.add_(gradient)
            if (index + 1) % 10 == 0 or index + 1 == len(forget_records):
                print(
                    f"    forget sensitivity {index + 1}/{len(forget_records)}"
                )

        for index, record in enumerate(retain_records):
            rewrite = record["requested_rewrite"]
            prompt = str(rewrite["prompt"]).format(str(rewrite["subject"]))
            answer = str(rewrite["target_true"]["str"])
            target_nll = v2.answer_nlls(
                model,
                tok,
                [prompt],
                [answer],
                device,
                llama_like=llama_like,
            )[0]
            gradient, covered = record_norms(target_nll, group="retain")
            retain_norms.append(gradient.float().norm(dim=1).cpu())
            retain_coverage.add_(covered.long())
            if (index + 1) % 250 == 0 or index + 1 == len(retain_records):
                print(
                    f"    retain sensitivity {index + 1}/{len(retain_records)}"
                )
    finally:
        with torch.no_grad():
            input_delta.raw_delta.copy_(original)
        input_delta.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

    if not forget_norms or not retain_norms:
        raise RuntimeError("per-example sensitivity produced an empty bank")
    report = partition.summarize_per_example_sensitivity(
        torch.stack(forget_norms),
        torch.stack(retain_norms),
        forget_coverage=forget_coverage,
        retain_coverage=retain_coverage,
        forget_importance_floor_relative=forget_importance_floor_relative,
        importance_ratio_min=importance_ratio_min,
        forget_specific_ratio_min=forget_specific_ratio_min,
        forget_specific_retain_coverage_max=(
            forget_specific_retain_coverage_max
        ),
        retain_tail_ratio_min=retain_tail_ratio_min,
    )
    report.update(
        {
            "schema_version": 1,
            "kind": "mcf_v2_3_2_per_example_forget_retain_sensitivity",
            "forget_scalar": "direct_target_new_minus_target_true_logprob_margin",
            "retain_scalar": "ground_truth_answer_nll",
            "coverage_relative_epsilon": float(coverage_relative_epsilon),
            "prompt_classifier": False,
            "prompt_text_serialized": 0,
            "ratio_is_sole_safety_criterion": False,
            "retain_hard_tail_guard": True,
            "zero_graph_examples": zero_graph_examples,
            "zero_graph_semantics": (
                "exact_zero_selected_embedding_row_sensitivity"
            ),
        }
    )
    return forget_gradient_sum / float(len(forget_records)), report


def retain_case_indices_by_selected_row(
    cases: Sequence[canonical.SensitivePredictionCase],
    tok: Any,
    row_ids: Sequence[int],
) -> tuple[List[List[int]], Dict[str, Any]]:
    """Index training-visible retain cases that exercise each selected row."""
    position = {int(token_id): index for index, token_id in enumerate(row_ids)}
    per_row: List[List[int]] = [[] for _ in row_ids]
    for case_index, case in enumerate(cases):
        present = set(canonical.flat_ids(tok, case.prompt)).intersection(position)
        for token_id in present:
            per_row[position[int(token_id)]].append(int(case_index))
    counts = [len(values) for values in per_row]
    return per_row, {
        "cases": len(cases),
        "rows": len(row_ids),
        "rows_with_cases": sum(value > 0 for value in counts),
        "rows_without_cases": sum(value == 0 for value in counts),
        "case_count_min": min(counts) if counts else 0,
        "case_count_max": max(counts) if counts else 0,
        "prompt_text_serialized": 0,
    }


def iterative_per_prompt_reachability_report(
    model: torch.nn.Module,
    tok: Any,
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    device: torch.device,
    *,
    input_delta: canonical.SelectedRowDelta,
    input_caps: torch.Tensor,
    bases: Sequence[torch.Tensor],
    roles: Sequence[str],
    retain_indices_by_row: Sequence[Sequence[int]],
    protection_cache: v2.ProtectionCache,
    protection_target_ids: torch.Tensor,
    protection_base_target_log_probs: torch.Tensor,
    llama_like: bool,
    required_margin: float,
    max_steps: int,
    step_fraction: float,
    minimum_improvement: float,
    protection_args: argparse.Namespace,
) -> Dict[str, Any]:
    """Find a guarded nonlinear path by relinearizing after every small step.

    V2.3.2 tested one ray derived at Base.  That rejects a curved feasible path
    whenever the frozen Transformer changes its local gradient.  V2.3.3 starts
    from exact zero for each prompt, recomputes the margin gradient after every
    accepted step, and backtracks the frequency-capped proposal.  Guarded rows
    are never projected; candidates using them must remain within all locked
    retain limits on every training-visible fit case containing those rows.
    """
    if input_delta.raw_delta is None:
        raise RuntimeError("iterative reachability requires a full row delta")
    if len(retain_indices_by_row) != len(roles):
        raise ValueError("retain guard index map is not aligned with row roles")
    original = input_delta.raw_delta.detach().clone()
    if float(original.norm().cpu()) > 1e-12:
        raise RuntimeError("iterative reachability must run from exact zero")

    def margin_gradient(
        record: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_delta.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        margin = v2.records_margin_tensor(
            model, tok, [record], device, llama_like=llama_like
        )[0]
        if margin.requires_grad:
            margin.backward()
            gradient = input_delta.raw_delta.grad
        else:
            gradient = None
        if gradient is None:
            local = torch.zeros_like(input_delta.raw_delta)
        else:
            local = gradient.detach().clone()
        return margin.detach(), local

    def guarded_indices(gradient: torch.Tensor) -> List[int]:
        usable = partition.role_constrained_gradient(gradient, bases, roles)
        ordered: List[int] = []
        for row_index, role in enumerate(roles):
            if role != partition.GUARDED:
                continue
            gradient_active = float(usable[row_index].norm().item()) > 1e-7
            delta_active = (
                float(input_delta.raw_delta[row_index].detach().norm().item())
                > 1e-7
            )
            if not gradient_active and not delta_active:
                continue
            for case_index in retain_indices_by_row[row_index]:
                value = int(case_index)
                if value not in ordered:
                    ordered.append(value)
        return ordered

    def constraint_score(indices: Sequence[int]) -> float:
        if not indices:
            return 0.0
        with torch.no_grad():
            _, kl_max, drift_max, target_drift_max = v22.active_retain_loss(
                model,
                tok,
                protection_cache,
                protection_target_ids,
                protection_base_target_log_probs,
                indices,
                device,
                batch_size=protection_args.capture_batch_size,
            )
        return joint.constraint_score(
            kl_max=float(kl_max.cpu()),
            drift_max=float(drift_max.cpu()),
            target_drift_max=float(target_drift_max.cpu()),
            kl_limit=protection_args.protected_kl_max,
            drift_limit=protection_args.protected_top1_drift_max,
            target_drift_limit=protection_args.protected_target_drift_max,
        )

    rows: List[Dict[str, Any]] = []
    try:
        for group, records in groups.items():
            for prompt_index, record in enumerate(records):
                with torch.no_grad():
                    input_delta.raw_delta.zero_()
                initial_margin_tensor, initial_gradient = margin_gradient(record)
                base_margin = float(initial_margin_tensor.cpu())
                linear = partition.cap_aware_prompt_reachability(
                    initial_gradient,
                    bases,
                    roles,
                    input_caps,
                    base_margin=base_margin,
                    required_margin=required_margin,
                )
                current_margin = base_margin
                best_margin = base_margin
                termination = "required_margin_already_met"
                trajectory: List[Dict[str, Any]] = [
                    {
                        "step": 0,
                        "margin": base_margin,
                        "accepted_factor": 0.0,
                        "guard_cases": 0,
                        "guard_constraint_score": 0.0,
                    }
                ]
                guard_rejections = 0
                accepted_steps = 0
                if current_margin < float(required_margin):
                    termination = "step_budget_exhausted"
                    for path_step in range(1, int(max_steps) + 1):
                        _, gradient = margin_gradient(record)
                        usable = partition.role_constrained_gradient(
                            gradient, bases, roles
                        )
                        norms = usable.norm(dim=1)
                        if not bool((norms > 1e-7).any().item()):
                            termination = "no_usable_gradient"
                            break
                        budgets = (
                            float(step_fraction)
                            * input_caps.to(device=usable.device, dtype=torch.float32)
                        )
                        relative = (
                            norms / norms.max().clamp_min(1e-20)
                        ).clamp_min(0.0).sqrt()
                        scales = torch.where(
                            norms > 0,
                            budgets * relative / norms.clamp_min(1e-20),
                            torch.zeros_like(norms),
                        )
                        proposed = usable * scales[:, None]
                        guard_indices = guarded_indices(gradient)
                        old = input_delta.raw_delta.detach().clone()
                        accepted_factor = 0.0
                        accepted_score = 0.0
                        accepted_margin = current_margin
                        for factor in BACKTRACKING_FACTORS:
                            with torch.no_grad():
                                input_delta.raw_delta.copy_(
                                    old + float(factor) * proposed
                                )
                                partition.apply_role_constraints_(
                                    input_delta.raw_delta, bases, roles
                                )
                                geometry.apply_row_caps_(
                                    input_delta.raw_delta, input_caps
                                )
                                candidate_margin = float(
                                    v2.records_margin_tensor(
                                        model,
                                        tok,
                                        [record],
                                        device,
                                        llama_like=llama_like,
                                    )[0]
                                    .detach()
                                    .cpu()
                                )
                            candidate_score = constraint_score(guard_indices)
                            if candidate_score > 1.0 + 1e-7:
                                guard_rejections += 1
                                continue
                            if (
                                candidate_margin - current_margin
                                < float(minimum_improvement)
                            ):
                                continue
                            accepted_factor = float(factor)
                            accepted_score = candidate_score
                            accepted_margin = candidate_margin
                            break
                        if accepted_factor == 0.0:
                            with torch.no_grad():
                                input_delta.raw_delta.copy_(old)
                            termination = "no_safe_improving_step"
                            break
                        accepted_steps += 1
                        current_margin = accepted_margin
                        best_margin = max(best_margin, current_margin)
                        trajectory.append(
                            {
                                "step": int(path_step),
                                "margin": current_margin,
                                "accepted_factor": accepted_factor,
                                "guard_cases": len(guard_indices),
                                "guard_constraint_score": accepted_score,
                            }
                        )
                        if current_margin >= float(required_margin):
                            termination = "required_margin_reached"
                            break
                path_passed = best_margin >= float(required_margin)
                rewrite = record["requested_rewrite"]
                rendered = str(rewrite["prompt"]).format(str(rewrite["subject"]))
                rows.append(
                    {
                        "group": str(group),
                        "prompt_index": int(prompt_index),
                        "case_id": int(record.get("case_id", prompt_index)),
                        "prompt_sha256": hashlib.sha256(
                            rendered.encode("utf-8")
                        ).hexdigest(),
                        "linear": linear,
                        "accepted_steps": accepted_steps,
                        "best_iterative_margin": best_margin,
                        "termination": termination,
                        "guard_rejections": guard_rejections,
                        "trajectory": trajectory,
                        "iterative_passed": bool(path_passed),
                        "passed": bool(path_passed),
                    }
                )
                if (prompt_index + 1) % 10 == 0 or prompt_index + 1 == len(records):
                    local_failures = sum(
                        not row["passed"]
                        for row in rows
                        if row["group"] == str(group)
                    )
                    print(
                        f"    {group} iterative reachability "
                        f"{prompt_index + 1}/{len(records)}, "
                        f"failures={local_failures}"
                    )
    finally:
        with torch.no_grad():
            input_delta.raw_delta.copy_(original)
        input_delta.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

    group_reports: Dict[str, Any] = {}
    for group in groups:
        local = [row for row in rows if row["group"] == group]
        group_reports[str(group)] = {
            "prompts": len(local),
            "initial_linear_failures": sum(
                not row["linear"]["passed"] for row in local
            ),
            "iterative_failures": sum(
                not row["iterative_passed"] for row in local
            ),
            "guard_rejections": sum(row["guard_rejections"] for row in local),
            "failures": sum(not row["passed"] for row in local),
            "passed": all(row["passed"] for row in local),
        }
    failed = [row for row in rows if not row["passed"]]
    return {
        "schema_version": 1,
        "kind": "mcf_v2_3_3_iterative_guarded_per_prompt_reachability",
        "required_margin": float(required_margin),
        "iterative_steps": int(max_steps),
        "step_cap_fraction": float(step_fraction),
        "minimum_step_improvement": float(minimum_improvement),
        "backtracking_factors": list(BACKTRACKING_FACTORS),
        "recompute_gradient_after_every_accepted_step": True,
        "guarded_rows_unprojected": True,
        "guarded_retain_cases_forward_checked_per_candidate": True,
        "initial_linear_bound_is_diagnostic_not_acceptance_gate": True,
        "groups": group_reports,
        "prompts": len(rows),
        "failures": len(failed),
        "failed_prompts": [
            {
                "group": row["group"],
                "prompt_index": row["prompt_index"],
                "case_id": row["case_id"],
                "prompt_sha256": row["prompt_sha256"],
                "linear": row["linear"],
                "accepted_steps": row["accepted_steps"],
                "best_iterative_margin": row["best_iterative_margin"],
                "termination": row["termination"],
                "guard_rejections": row["guard_rejections"],
            }
            for row in failed
        ],
        "per_prompt": rows,
        "all_direct_and_synthetic_prompts_must_pass": True,
        "passed": not failed,
    }


def per_prompt_reachability_report(
    model: torch.nn.Module,
    tok: Any,
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    device: torch.device,
    *,
    input_delta: canonical.SelectedRowDelta,
    input_caps: torch.Tensor,
    bases: Sequence[torch.Tensor],
    roles: Sequence[str],
    llama_like: bool,
    required_margin: float,
) -> Dict[str, Any]:
    """Certify local and forward reachability for every training-visible prompt.

    V2.3 used one aggregate forget gradient to choose rows and then checked only
    that every direct prompt contained a non-excluded token.  This audit keeps
    prompts separate, applies the final row roles before measuring capacity,
    and probes the frozen nonlinear model along the cap-saturating local
    direction.  It runs from exact zero and restores the incoming state.
    """
    if input_delta.raw_delta is None:
        raise RuntimeError("per-prompt reachability requires a full row delta")
    original = input_delta.raw_delta.detach().clone()
    if float(original.norm().cpu()) > 1e-12:
        raise RuntimeError("per-prompt reachability must run from exact zero")
    rows: List[Dict[str, Any]] = []
    try:
        for group, records in groups.items():
            for prompt_index, record in enumerate(records):
                with torch.no_grad():
                    input_delta.raw_delta.zero_()
                input_delta.zero_grad(set_to_none=True)
                model.zero_grad(set_to_none=True)
                margin = v2.records_margin_tensor(
                    model, tok, [record], device, llama_like=llama_like
                )[0]
                margin.backward()
                gradient = input_delta.raw_delta.grad
                if gradient is None:
                    gradient = torch.zeros_like(input_delta.raw_delta)
                else:
                    gradient = gradient.detach().clone()
                base_margin = float(margin.detach().cpu())
                linear = partition.cap_aware_prompt_reachability(
                    gradient,
                    bases,
                    roles,
                    input_caps,
                    base_margin=base_margin,
                    required_margin=required_margin,
                )
                direction = partition.cap_saturating_margin_direction(
                    gradient, bases, roles, input_caps
                )
                sweep: List[Dict[str, float]] = []
                best_margin = base_margin
                best_factor = 0.0
                for factor in REACHABILITY_FACTORS:
                    with torch.no_grad():
                        input_delta.raw_delta.copy_(float(factor) * direction)
                        partition.apply_role_constraints_(
                            input_delta.raw_delta, bases, roles
                        )
                        geometry.apply_row_caps_(input_delta.raw_delta, input_caps)
                        candidate_margin = float(
                            v2.records_margin_tensor(
                                model,
                                tok,
                                [record],
                                device,
                                llama_like=llama_like,
                            )[0]
                            .detach()
                            .cpu()
                        )
                    sweep.append(
                        {"factor": float(factor), "margin": candidate_margin}
                    )
                    if candidate_margin > best_margin:
                        best_margin = candidate_margin
                        best_factor = float(factor)
                rewrite = record["requested_rewrite"]
                rendered = str(rewrite["prompt"]).format(str(rewrite["subject"]))
                forward_passed = best_margin >= float(required_margin)
                row = {
                    "group": str(group),
                    "prompt_index": int(prompt_index),
                    "case_id": int(record.get("case_id", prompt_index)),
                    "prompt_sha256": hashlib.sha256(
                        rendered.encode("utf-8")
                    ).hexdigest(),
                    "linear": linear,
                    "directional_sweep": sweep,
                    "best_directional_factor": best_factor,
                    "best_directional_margin": best_margin,
                    "directional_passed": bool(forward_passed),
                    "passed": bool(linear["passed"] and forward_passed),
                }
                rows.append(row)
    finally:
        with torch.no_grad():
            input_delta.raw_delta.copy_(original)
        input_delta.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

    group_reports: Dict[str, Any] = {}
    for group in groups:
        local = [row for row in rows if row["group"] == group]
        group_reports[str(group)] = {
            "prompts": len(local),
            "linear_failures": sum(not row["linear"]["passed"] for row in local),
            "directional_failures": sum(
                not row["directional_passed"] for row in local
            ),
            "failures": sum(not row["passed"] for row in local),
            "passed": all(row["passed"] for row in local),
        }
    failed = [row for row in rows if not row["passed"]]
    return {
        "schema_version": 1,
        "kind": "mcf_v2_3_1_per_prompt_cap_aware_reachability",
        "required_margin": float(required_margin),
        "directional_factors": list(REACHABILITY_FACTORS),
        "linear_bound": (
            "sum_t cap_t * ||P_t grad_t margin|| from exact zero"
        ),
        "nonlinear_probe": (
            "frozen-model forward sweep along each prompt's cap-saturating "
            "projected margin gradient"
        ),
        "groups": group_reports,
        "prompts": len(rows),
        "failures": len(failed),
        "failed_prompts": [
            {
                "group": row["group"],
                "prompt_index": row["prompt_index"],
                "case_id": row["case_id"],
                "prompt_sha256": row["prompt_sha256"],
                "linear": row["linear"],
                "best_directional_factor": row["best_directional_factor"],
                "best_directional_margin": row["best_directional_margin"],
                "directional_passed": row["directional_passed"],
            }
            for row in failed
        ],
        "per_prompt": rows,
        "all_direct_and_synthetic_prompts_must_pass": True,
        "passed": not failed,
    }


def frozen_head_certificate(
    model: torch.nn.Module,
    tok: Any,
    forget: Sequence[Mapping[str, Any]],
    synthetic_records: Sequence[Mapping[str, Any]],
    protection_cache: v2.ProtectionCache,
    target_ids: torch.Tensor,
    base_target_log_probs: torch.Tensor,
    input_delta: canonical.SelectedRowDelta,
    input_caps: torch.Tensor,
    bases: Sequence[torch.Tensor],
    roles: Sequence[str],
    device: torch.device,
    *,
    llama_like: bool,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    direct = v2.margin_report(
        model,
        tok,
        forget,
        device,
        llama_like=llama_like,
        batch_size=args.capture_batch_size,
        minimum=args.minimum_forget_margin,
    )
    synth = v2.margin_report(
        model,
        tok,
        synthetic_records,
        device,
        llama_like=llama_like,
        batch_size=args.capture_batch_size,
        minimum=args.minimum_forget_margin,
    )
    protection = v22.evaluate_endpoint_protection(
        model,
        tok,
        protection_cache,
        target_ids,
        base_target_log_probs,
        device,
        batch_size=args.capture_batch_size,
        args=args,
    )
    caps = geometry.cap_report(input_delta.raw_delta, input_caps)
    compliance = partition.role_compliance_report(
        input_delta.raw_delta, bases, roles
    )
    result = {
        "direct": direct,
        "synthetic": synth,
        "protection": protection,
        "input_caps": caps,
        "role_compliance": compliance,
    }
    result["passed"] = bool(
        direct["passed"]
        and synth["passed"]
        and protection["passed"]
        and caps["passed"]
        and compliance["passed"]
    )
    return result


def completion_failure(
    output: Path,
    *,
    phase: str,
    detail: Mapping[str, Any],
    protocol: str = partition.PROTOCOL,
) -> None:
    v2.write_json(
        output / "method" / "completion.json",
        {
            "schema_version": 1,
            "protocol": str(protocol),
            "passed": False,
            "phase": phase,
            "detail": dict(detail),
            "candidate_saved": False,
            "official_evaluation_prompts_seen": 0,
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    v2.validate_environment_firewall()
    gagd.set_seed(args.seed)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry_path = Path(args.experiment_registry).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_registry(registry, args)
    active_protocol = str(registry["protocol"])
    sensitivity_revision = active_protocol in (
        partition.SENSITIVITY_PROTOCOL,
        partition.HYBRID_PROTOCOL,
    )
    hybrid_revision = active_protocol == partition.HYBRID_PROTOCOL
    protocol_dir = Path(args.protocol_dir).resolve()
    manifest_path = protocol_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != split_builder.PROTOCOL:
        raise RuntimeError("V2.3 split manifest protocol mismatch")
    if manifest.get("serialized_prompt_counts", {}).get("official_retain") != 0:
        raise RuntimeError("V2.3 split exposes official retain prompt text")
    forget = v2.load_partition(protocol_dir, manifest, "forget", permitted=True)
    protection_fit = v2.load_partition(
        protocol_dir, manifest, "protection_fit", permitted=True
    )
    protection_development = v2.load_partition(
        protocol_dir, manifest, "protection_development", permitted=True
    )

    model, tok = gagd.load_model_and_tokenizer(args, for_training=True)
    device = gagd.first_device(model)
    output_layer = canonical.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    llama_like = canonical.is_llama_like(model, tok)
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("V2.3 requires an untied LM head")
    output_layer.weight.requires_grad_(False)
    transformer_before = v2.transformer_sha256(
        model, excluded_parameters=[input_layer.weight, output_layer.weight]
    )
    lm_head_before = v2.tensor_collection_sha256([("lm_head", output_layer.weight)])

    endpoint_rows, expansion_report = joint.expanded_endpoint_rows(
        forget, tok, llama_like=llama_like
    )
    overlap_report = geometry.endpoint_overlap_report(endpoint_rows)
    v2.write_json(
        output / "method" / "input_row_manifest.json",
        {
            "expansion": expansion_report,
            "overlap": overlap_report,
            "output_rows_selected": 0,
            "lm_head_edited": False,
        },
    )
    if (
        not expansion_report["one_delta_per_physical_input_row"]
        or not overlap_report["one_delta_per_physical_input_row"]
    ):
        raise RuntimeError("V2.3 input row coherence failed")
    input_ids_tensor = torch.tensor(
        endpoint_rows.input_ids, dtype=torch.long, device=device
    )
    base_input_rows = (
        input_layer.weight.index_select(0, input_ids_tensor).detach().float()
    )
    hidden_size = int(base_input_rows.shape[1])
    input_delta = canonical.SelectedRowDelta(
        len(endpoint_rows.input_ids), hidden_size, direction_basis=None, device=device
    )
    if input_delta.raw_delta is None:
        raise RuntimeError("V2.3 requires an unrestricted input row delta")
    input_handle = endpoint_hooks.register_input_embedding_delta_hook(
        input_layer, endpoint_rows.input_ids, input_delta.effective_delta
    )

    documents = subject_stage1.load_frequency_documents(
        args.wikidata_dir, args.frequency_doc_start, args.frequency_docs
    )
    if len(documents) < args.frequency_docs:
        raise RuntimeError("V2.3 did not load the registered Wikipedia slice")
    sentence_pool = v2.corpus_sentence_pool(documents)
    targeted, targeted_report = folded.targeted_token_prompt_partitions(
        tok,
        sentence_pool,
        endpoint_rows.input_ids,
        fit_per_row=args.targeted_fit_per_row,
        development_per_row=args.targeted_development_per_row,
        certification_per_row=args.targeted_certification_per_row,
    )
    v2.write_json(output / "method" / "targeted_row_coverage.json", targeted_report)
    all_targeted = targeted["fit"] + targeted["development"] + targeted["certification"]
    corpus = v21.generic_corpus_partitions(
        sentence_pool,
        excluded=all_targeted,
        fit=args.corpus_fit_prompts,
        development=args.corpus_development_prompts,
        certification=args.corpus_certification_prompts,
        seed=args.seed,
    )
    frequency_counts = subject_stage1.token_frequency_counts(
        tok, documents, int(input_layer.weight.shape[0])
    )
    selected_counts = frequency_counts.index_select(
        0, torch.tensor(endpoint_rows.input_ids, dtype=torch.long)
    )
    input_caps = geometry.frequency_adjusted_caps(
        base_input_rows.cpu(),
        selected_counts,
        relative_cap=args.input_relative_cap,
        alpha=args.input_frequency_alpha,
    ).to(device)

    synthetic_prefixes = synthetic.corpus_context_prefixes(
        documents, count=256, seed=args.seed + 71
    )
    synthetic_records = synthetic.build_synthetic_records(
        forget,
        count=args.synthetic_paraphrases,
        context_prefixes=synthetic_prefixes,
    )
    training_records = list(forget) + list(synthetic_records)
    forget_teacher_forced_contexts: set[str] = set()
    for sensitive_field in ("target_true", "target_new"):
        forget_teacher_forced_contexts.update(
            case.prompt
            for case in canonical.expand_sensitive_cases(
                training_records,
                tok,
                sensitive_field=sensitive_field,
                llama_like=llama_like,
            )
        )

    raw_fit_cases = v21.make_protection_cases(
        protection_fit,
        targeted=targeted["fit"],
        generic=corpus["fit"],
        forget=forget,
        role="fit",
        tok=tok,
        llama_like=llama_like,
        same_subject_count=args.same_subject_prompts,
    )
    raw_development_cases = v21.make_protection_cases(
        protection_development,
        targeted=targeted["development"],
        generic=corpus["development"],
        forget=forget,
        role="development",
        tok=tok,
        llama_like=llama_like,
        same_subject_count=args.same_subject_prompts,
    )
    fit_cases = [
        case for case in raw_fit_cases if case.prompt not in forget_teacher_forced_contexts
    ]
    development_cases = [
        case
        for case in raw_development_cases
        if case.prompt not in forget_teacher_forced_contexts
    ]
    collision_report = {
        "precedence_rule": "forget_positive_precedes_exact_retain_collision",
        "forget_teacher_forced_contexts": len(forget_teacher_forced_contexts),
        "fit_raw": len(raw_fit_cases),
        "fit_retained": len(fit_cases),
        "fit_excluded": len(raw_fit_cases) - len(fit_cases),
        "development_raw": len(raw_development_cases),
        "development_retained": len(development_cases),
        "development_excluded": len(raw_development_cases) - len(development_cases),
        "official_metrics_modified": False,
    }
    v2.write_json(
        output / "method" / "retain_collision_precedence.json", collision_report
    )
    print(
        f"Stage 1: cache {len(fit_cases)} fit and {len(development_cases)} "
        "development retain prompts"
    )
    fit_cache = v2.cache_protection(
        model,
        tok,
        fit_cases,
        device,
        batch_size=args.capture_batch_size,
        topk=args.protection_topk,
    )
    fit_target_ids, fit_base_target_log_probs = v22.cache_retain_targets(
        model,
        tok,
        fit_cases,
        device,
        llama_like=llama_like,
        batch_size=args.capture_batch_size,
    )
    development_cache = v2.cache_protection(
        model,
        tok,
        development_cases,
        device,
        batch_size=args.capture_batch_size,
        topk=args.protection_topk,
    )
    (
        development_target_ids,
        development_base_target_log_probs,
    ) = v22.cache_retain_targets(
        model,
        tok,
        development_cases,
        device,
        llama_like=llama_like,
        batch_size=args.capture_batch_size,
    )

    print("Stage 2: retain-readout subspaces and the row-partition diagnostic")
    bases, basis_report = build_retain_readout_bases(
        model,
        tok,
        fit_cache,
        fit_target_ids,
        device,
        input_delta=input_delta,
        sketches=args.retain_jacobian_sketches,
        batch_size=args.capture_batch_size,
        topk=args.protection_topk,
        max_rank=args.retain_rank_cap,
    )
    targeted_rows = {
        int(row["token_id"]): row for row in targeted_report.get("per_row", [])
    }
    basis_report["corpus_absent_rows"] = sum(
        bool(targeted_rows.get(int(token_id), {}).get("corpus_absent"))
        for token_id in endpoint_rows.input_ids
    )
    basis_report["unobserved_and_corpus_absent_rows"] = sum(
        bool(targeted_rows.get(int(token_id), {}).get("corpus_absent"))
        and int(bases[index].shape[0]) == 0
        for index, token_id in enumerate(endpoint_rows.input_ids)
    )
    v2.write_json(output / "method" / "retain_readout_bases.json", basis_report)

    if sensitivity_revision:
        if len(forget) != args.sensitivity_forget_records:
            raise RuntimeError("V2.3.2 forget sensitivity bank size changed")
        if len(protection_fit) != args.sensitivity_retain_records:
            raise RuntimeError("V2.3.2 retain sensitivity bank size changed")
        print(
            "Stage 2a: per-example forget/retain embedding-row sensitivity"
        )
        forget_gradient, sensitivity_report = collect_per_example_sensitivity(
            model,
            tok,
            forget,
            protection_fit,
            device,
            input_delta=input_delta,
            llama_like=llama_like,
            coverage_relative_epsilon=(
                args.sensitivity_coverage_relative_epsilon
            ),
            forget_importance_floor_relative=(
                args.forget_importance_floor_relative
            ),
            importance_ratio_min=args.importance_ratio_min,
            forget_specific_ratio_min=args.forget_specific_ratio_min,
            forget_specific_retain_coverage_max=(
                args.forget_specific_retain_coverage_max
            ),
            retain_tail_ratio_min=args.retain_tail_ratio_min,
        )
        for index, row in enumerate(sensitivity_report["per_row"]):
            row["token_id"] = int(endpoint_rows.input_ids[index])
        v2.write_json(
            output / "method" / "per_example_row_sensitivity.json",
            sensitivity_report,
        )
        print(
            "  sensitivity: "
            + ", ".join(
                f"{name}={count}"
                for name, count in sensitivity_report["class_counts"].items()
            )
        )
    else:
        input_delta.zero_grad(set_to_none=True)
        forget_batcher_probe = folded.BalancedBatcher(
            len(training_records), args.forget_batch_size, args.seed + 1501
        )
        probe_batches = max(
            1, len(training_records) // max(args.forget_batch_size, 1)
        )
        for _ in range(probe_batches):
            indices = forget_batcher_probe.next()
            batch = [training_records[index] for index in indices]
            margins = v2.records_margin_tensor(
                model, tok, batch, device, llama_like=llama_like
            )
            (
                F.relu(args.forget_margin_target - margins).square().mean()
                / float(probe_batches)
            ).backward()
        forget_gradient = input_delta.raw_delta.grad
        if forget_gradient is None:
            raise RuntimeError("V2.3 diagnostic produced no forget gradient")
    efficacy = partition.residual_efficacy(forget_gradient, bases, input_caps)
    input_delta.zero_grad(set_to_none=True)

    potential = efficacy["potential"]
    potential_floor = float(args.partition_potential_floor) * float(
        potential.max().item()
    )
    geometry_roles, geometry_partition_report = partition.partition_rows(
        row_ids=endpoint_rows.input_ids,
        retain_observed=[int(basis.shape[0]) > 0 for basis in bases],
        efficacy=efficacy["efficacy"],
        potential=potential,
        frequency=selected_counts.float(),
        direct_live_rows=(
            {}
            if sensitivity_revision
            else direct_live_rows(forget, tok, endpoint_rows.input_ids)
        ),
        efficacy_min=args.partition_efficacy_min,
        potential_min=potential_floor,
        frequency_max=args.partition_frequency_max,
    )
    if sensitivity_revision:
        geometry_partition_report["absolute_potential_floor"] = potential_floor
        geometry_partition_report["relative_potential_floor"] = float(
            args.partition_potential_floor
        )
        v2.write_json(
            output / "method" / "geometry_row_partition.json",
            geometry_partition_report,
        )
        if hybrid_revision:
            roles, partition_report = (
                partition.combine_hybrid_sensitivity_roles(
                    row_ids=endpoint_rows.input_ids,
                    geometry_roles=geometry_roles,
                    geometry_report=geometry_partition_report,
                    sensitivity_report=sensitivity_report,
                )
            )
        else:
            roles, partition_report = (
                partition.combine_geometry_and_sensitivity_roles(
                    row_ids=endpoint_rows.input_ids,
                    geometry_roles=geometry_roles,
                    geometry_report=geometry_partition_report,
                    sensitivity_report=sensitivity_report,
                )
            )
        partition_report["liveness_forced_records"] = 0
        partition_report["liveness_forced"] = []
        partition_report["sensitivity_criterion"] = sensitivity_report["criterion"]
    else:
        roles = geometry_roles
        partition_report = geometry_partition_report
    partition_report["absolute_potential_floor"] = potential_floor
    partition_report["relative_potential_floor"] = float(
        args.partition_potential_floor
    )
    partition_report["prediction"] = partition.diagnostic_prediction(partition_report)
    v2.write_json(output / "method" / "row_partition.json", partition_report)
    print(
        "  partition: "
        + ", ".join(
            f"{role}={partition_report['role_counts'][role]}"
            for role in (
                partition.HYBRID_ROLES if hybrid_revision else partition.ROLES
            )
        )
        + f", liveness-forced={partition_report['liveness_forced_records']}"
    )
    if not partition_report["passed"]:
        completion_failure(
            output,
            phase="row_partition_diagnostic",
            detail=partition_report,
            protocol=active_protocol,
        )
        raise RuntimeError("V2.3 excluded every selected row")

    retain_indices_by_row: List[List[int]] = [
        [] for _ in endpoint_rows.input_ids
    ]
    guarded_overlap_indices: List[int] = []
    if hybrid_revision:
        retain_indices_by_row, retain_index_report = (
            retain_case_indices_by_selected_row(
                fit_cases, tok, endpoint_rows.input_ids
            )
        )
        guarded_row_indices = [
            index
            for index, role in enumerate(roles)
            if role == partition.GUARDED
        ]
        guarded_without_fit_cases = [
            int(endpoint_rows.input_ids[index])
            for index in guarded_row_indices
            if not retain_indices_by_row[index]
        ]
        for row_index in guarded_row_indices:
            for case_index in retain_indices_by_row[row_index]:
                if int(case_index) not in guarded_overlap_indices:
                    guarded_overlap_indices.append(int(case_index))
        guard_report = {
            **retain_index_report,
            "guarded_rows": len(guarded_row_indices),
            "guarded_rows_without_fit_cases": guarded_without_fit_cases,
            "guarded_overlap_fit_cases": len(guarded_overlap_indices),
            "all_guarded_rows_forward_auditable": not guarded_without_fit_cases,
            "used_by_iterative_reachability": True,
            "used_as_training_overlap_stratum": True,
        }
        v2.write_json(
            output / "method" / "guarded_retain_bank.json", guard_report
        )
        if guarded_without_fit_cases:
            completion_failure(
                output,
                phase="guarded_retain_bank",
                detail=guard_report,
                protocol=active_protocol,
            )
            raise RuntimeError(
                "V2.3.3 has guarded rows without a forward-auditable retain case"
            )

    if args.require_per_prompt_reachability:
        print(
            "Stage 2b: per-prompt "
            + (
                "iterative guarded reachability from exact zero"
                if hybrid_revision
                else "cap-aware reachability from exact zero"
            )
        )
        if hybrid_revision:
            reachability_report = iterative_per_prompt_reachability_report(
                model,
                tok,
                {"direct": forget, "synthetic": synthetic_records},
                device,
                input_delta=input_delta,
                input_caps=input_caps,
                bases=bases,
                roles=roles,
                retain_indices_by_row=retain_indices_by_row,
                protection_cache=fit_cache,
                protection_target_ids=fit_target_ids,
                protection_base_target_log_probs=(
                    fit_base_target_log_probs
                ),
                llama_like=llama_like,
                required_margin=args.minimum_forget_margin,
                max_steps=args.iterative_reachability_steps,
                step_fraction=args.iterative_reachability_step_fraction,
                minimum_improvement=(
                    args.iterative_reachability_minimum_improvement
                ),
                protection_args=args,
            )
        else:
            reachability_report = per_prompt_reachability_report(
                model,
                tok,
                {"direct": forget, "synthetic": synthetic_records},
                device,
                input_delta=input_delta,
                input_caps=input_caps,
                bases=bases,
                roles=roles,
                llama_like=llama_like,
                required_margin=args.minimum_forget_margin,
            )
        v2.write_json(
            output / "method" / "per_prompt_reachability.json",
            reachability_report,
        )
        print(
            "  reachability: "
            f"direct failures={reachability_report['groups']['direct']['failures']}, "
            "synthetic failures="
            f"{reachability_report['groups']['synthetic']['failures']}"
        )
        if not reachability_report["passed"]:
            completion_failure(
                output,
                phase="per_prompt_cap_aware_reachability",
                detail={
                    "groups": reachability_report["groups"],
                    "failures": reachability_report["failures"],
                    "failed_prompts": reachability_report["failed_prompts"],
                },
                protocol=active_protocol,
            )
            raise RuntimeError(
                "per-prompt reachability failed; optimization refused"
            )

    contrast_cases, contrast_true_ids, contrast_new_ids = build_contrast_cells(
        training_records, tok, llama_like=llama_like
    )
    contrast_directions, contrast_valid = partition.answer_contrast_directions(
        output_layer.weight,
        contrast_true_ids,
        contrast_new_ids,
        epsilon=args.contrast_epsilon,
    )
    contrast_base_hidden = canonical.forward_last_hidden(
        model, tok, contrast_cases, device, batch_size=args.capture_batch_size
    ).detach()
    v2.write_json(
        output / "method" / "answer_contrast_directions.json",
        {
            "definition": "q_i = normalize(W[target_new_i] - W[target_true_i])",
            "frozen_lm_head": True,
            "cells": int(contrast_valid.numel()),
            "valid_cells": int(contrast_valid.sum().item()),
            "degenerate_cells": int((~contrast_valid).sum().item()),
            "epsilon": float(args.contrast_epsilon),
            "surgical_weight": float(args.surgical_weight),
            "registered_as_ablation_when_zero": True,
            "multi_token_note": (
                "position i is evaluated at the teacher-forced target_true "
                "prefix, so for i > 0 the paired target_new token is read at "
                "the target_true state; unequal answer lengths and shared "
                "leading tokens leave a position degenerate and unused"
            ),
        },
    )

    prompt_to_fit = {case.prompt: index for index, case in enumerate(fit_cases)}
    explicit_overlap_prompts = list(targeted["fit"]) + v2.same_subject_prompts(
        forget, role="fit", count=args.same_subject_prompts
    )
    overlap_indices = [
        prompt_to_fit[prompt]
        for prompt in explicit_overlap_prompts
        if prompt in prompt_to_fit
    ]
    overlap_indices = list(
        dict.fromkeys(overlap_indices + guarded_overlap_indices)
    )
    if not overlap_indices:
        raise RuntimeError("V2.3 constructed no explicit overlap retain stratum")
    v2.write_json(
        output / "method" / "retain_strata.json",
        {
            "explicit_input_overlap_prompts": len(explicit_overlap_prompts),
            "labeled_retain_target_cases": int((fit_target_ids >= 0).sum().item()),
            "generic_distribution_only_cases": int((fit_target_ids < 0).sum().item()),
            "unique_overlap_fit_cases": len(overlap_indices),
            "guarded_overlap_fit_cases": len(guarded_overlap_indices),
            "guarded_rows": sum(
                role == partition.GUARDED for role in roles
            ),
            "guarded_overlap_rotates_every_update": bool(hybrid_revision),
            "complete_fit_bank_checked_at_every_refresh": bool(
                hybrid_revision
            ),
            "hard_tail_active": args.hard_tail_active,
            "overlap_active": args.overlap_retain_batch_size,
            "random_active": args.random_retain_batch_size,
            "active_maximum": args.active_retain_maximum,
            "present_on_every_embedding_update": True,
        },
    )

    forget_batcher = folded.BalancedBatcher(
        len(training_records), args.forget_batch_size, args.seed + 1201
    )
    random_retain_batcher = folded.BalancedBatcher(
        len(fit_cache.cases), args.random_retain_batch_size, args.seed + 1301
    )
    overlap_batcher = folded.BalancedBatcher(
        len(overlap_indices), args.overlap_retain_batch_size, args.seed + 1401
    )
    contrast_batcher = folded.BalancedBatcher(
        len(contrast_cases), args.forget_batch_size, args.seed + 1601
    )
    hard_tail = joint.PersistentHardTail(args.hard_tail_capacity)
    tail_reports: List[Dict[str, Any]] = []
    training_log: List[Dict[str, Any]] = []
    candidate_states: Dict[int, torch.Tensor] = {}
    last_fit_safe_state = input_delta.raw_delta.detach().clone()
    last_fit_safe_step = 0

    def surgical_term() -> tuple[torch.Tensor, Dict[str, Any]]:
        if float(args.surgical_weight) <= 0.0:
            return torch.zeros((), device=device), {"active": False}
        local = contrast_batcher.next()
        index = torch.tensor(local, dtype=torch.long)
        batch = [contrast_cases[value] for value in local]
        # canonical.forward_last_hidden is no_grad and is only correct for the
        # cached Base states; the trained term needs the differentiable twin.
        _, hidden = subject_stage1.forward_last_logits_and_hidden(
            model, tok, batch, device
        )
        penalty, detail = partition.contrast_surgical_penalty(
            hidden,
            contrast_base_hidden.index_select(0, index.to(contrast_base_hidden.device)),
            contrast_directions.index_select(0, index.to(contrast_directions.device)),
            contrast_valid.index_select(0, index.to(contrast_valid.device)),
            sign_margin=args.surgical_sign_margin,
        )
        detail["active"] = True
        return penalty, detail

    def embedding_step(
        *,
        forget_batch: Sequence[Mapping[str, Any]],
        retain_indices: Sequence[int],
    ) -> Dict[str, Any]:
        input_delta.zero_grad(set_to_none=True)
        margins = v2.records_margin_tensor(
            model, tok, forget_batch, device, llama_like=llama_like
        )
        forget_loss = F.relu(args.forget_margin_target - margins).square().mean()
        (
            retain_kl_mean,
            retain_kl_max,
            retain_drift_max,
            retain_target_drift_max,
        ) = v22.active_retain_loss(
            model,
            tok,
            fit_cache,
            fit_target_ids,
            fit_base_target_log_probs,
            retain_indices,
            device,
            batch_size=args.capture_batch_size,
        )
        surgical, surgical_detail = surgical_term()
        loss = (
            args.forget_weight * forget_loss
            + args.retain_kl_weight * retain_kl_mean
            + args.retain_top1_weight * retain_drift_max.square()
            + args.retain_target_weight * retain_target_drift_max.square()
            + args.surgical_weight * surgical
            + args.delta_l2_weight * input_delta.raw_delta.square().mean()
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite V2.3 embedding loss")
        loss.backward()
        if input_delta.raw_delta.grad is None:
            raise RuntimeError("V2.3 embedding update produced no gradient")
        proposed = joint.normalized_row_step(
            input_delta.raw_delta.grad.detach(),
            input_caps,
            fraction=args.input_step_cap_fraction,
        )
        old = input_delta.raw_delta.detach().clone()
        before_forget = float(forget_loss.detach().cpu())
        before_score = joint.constraint_score(
            kl_max=float(retain_kl_max.detach().cpu()),
            drift_max=float(retain_drift_max.detach().cpu()),
            target_drift_max=float(retain_target_drift_max.detach().cpu()),
            kl_limit=args.protected_kl_max,
            drift_limit=args.protected_top1_drift_max,
            target_drift_limit=args.protected_target_drift_max,
        )
        accepted = 0.0
        candidate_forget_value = before_forget
        candidate_score = before_score
        for factor in BACKTRACKING_FACTORS:
            with torch.no_grad():
                input_delta.raw_delta.copy_(old + float(factor) * proposed)
                # Projection is the warm start; the accept test below is the
                # only thing that certifies the step through the nonlinearity.
                partition.apply_role_constraints_(
                    input_delta.raw_delta, bases, roles
                )
                geometry.apply_row_caps_(input_delta.raw_delta, input_caps)
                candidate_margins = v2.records_margin_tensor(
                    model, tok, forget_batch, device, llama_like=llama_like
                )
                candidate_forget = (
                    F.relu(args.forget_margin_target - candidate_margins)
                    .square()
                    .mean()
                )
                (
                    _,
                    candidate_kl_max,
                    candidate_drift_max,
                    candidate_target_drift_max,
                ) = v22.active_retain_loss(
                    model,
                    tok,
                    fit_cache,
                    fit_target_ids,
                    fit_base_target_log_probs,
                    retain_indices,
                    device,
                    batch_size=args.capture_batch_size,
                )
            local_score = joint.constraint_score(
                kl_max=float(candidate_kl_max.cpu()),
                drift_max=float(candidate_drift_max.cpu()),
                target_drift_max=float(candidate_target_drift_max.cpu()),
                kl_limit=args.protected_kl_max,
                drift_limit=args.protected_top1_drift_max,
                target_drift_limit=args.protected_target_drift_max,
            )
            if args.require_per_prompt_reachability:
                accept = partition.accept_strict_trust_region_candidate(
                    before_forget=before_forget,
                    candidate_forget=float(candidate_forget.cpu()),
                    before_constraint_score=before_score,
                    candidate_constraint_score=local_score,
                    minimum_forget_improvement=(
                        args.minimum_forget_loss_improvement
                    ),
                )
            else:
                accept = joint.accept_trust_region_candidate(
                    before_forget=before_forget,
                    candidate_forget=float(candidate_forget.cpu()),
                    before_constraint_score=before_score,
                    candidate_constraint_score=local_score,
                )
            if accept:
                accepted = float(factor)
                candidate_forget_value = float(candidate_forget.cpu())
                candidate_score = local_score
                break
        if accepted == 0.0:
            with torch.no_grad():
                input_delta.raw_delta.copy_(old)
        input_delta.zero_grad(set_to_none=True)
        return {
            "endpoint": "embedding",
            "accepted_factor": accepted,
            "before_forget_loss": before_forget,
            "after_forget_loss": candidate_forget_value,
            "before_constraint_score": before_score,
            "after_constraint_score": candidate_score,
            "strict_forget_improvement_required": bool(
                args.require_per_prompt_reachability
            ),
            "minimum_forget_loss_improvement": (
                float(args.minimum_forget_loss_improvement)
                if args.require_per_prompt_reachability
                else 0.0
            ),
            "surgical": surgical_detail,
        }

    print(
        "Stage 3: "
        + (
            "hybrid guarded/projected joint forget/retain embedding optimization"
            if hybrid_revision
            else "projected joint forget/retain embedding optimization"
        )
    )
    for step in range(1, args.steps + 1):
        if step == 1 or step % args.hard_tail_refresh_every == 0:
            fit_kl, fit_drift, fit_target_drift = v22.protection_vectors(
                model,
                tok,
                fit_cache,
                fit_target_ids,
                fit_base_target_log_probs,
                device,
                batch_size=args.capture_batch_size,
            )
            tail = hard_tail.refresh(
                fit_kl,
                fit_drift,
                fit_target_drift,
                kl_limit=args.protected_kl_max,
                drift_limit=args.protected_top1_drift_max,
                target_drift_limit=args.protected_target_drift_max,
                add=args.hard_tail_add,
            )
            tail["step"] = step
            tail_reports.append(tail)
        forget_indices = forget_batcher.next()
        forget_batch = [training_records[index] for index in forget_indices]
        random_indices = random_retain_batcher.next()
        overlap_local = overlap_batcher.next()
        overlap_active = [overlap_indices[index] for index in overlap_local]
        retain_indices = joint.compose_active_retain_indices(
            random_indices=random_indices,
            overlap_indices=overlap_active,
            hard_indices=hard_tail.indices[: args.hard_tail_active],
            maximum=args.active_retain_maximum,
        )
        update = embedding_step(
            forget_batch=forget_batch, retain_indices=retain_indices
        )
        if step % args.check_every == 0:
            direct = v2.margin_report(
                model,
                tok,
                forget,
                device,
                llama_like=llama_like,
                batch_size=args.capture_batch_size,
                minimum=args.minimum_forget_margin,
            )
            synth_report = v2.margin_report(
                model,
                tok,
                synthetic_records,
                device,
                llama_like=llama_like,
                batch_size=args.capture_batch_size,
                minimum=args.minimum_forget_margin,
            )
            fit_report = v22.evaluate_endpoint_protection(
                model,
                tok,
                fit_cache,
                fit_target_ids,
                fit_base_target_log_probs,
                device,
                batch_size=args.capture_batch_size,
                args=args,
            )
            development = v22.evaluate_endpoint_protection(
                model,
                tok,
                development_cache,
                development_target_ids,
                development_base_target_log_probs,
                device,
                batch_size=args.capture_batch_size,
                args=args,
            )
            cap = geometry.cap_report(input_delta.raw_delta, input_caps)
            compliance = partition.role_compliance_report(
                input_delta.raw_delta, bases, roles
            )
            total_norm = float(
                input_delta.raw_delta.detach().norm().cpu()
            )
            rolled_back = False
            rollback_fit_report: Dict[str, Any] | None = None
            rollback_to_step: int | None = None
            if args.full_fit_rollback and not fit_report["passed"]:
                rollback_to_step = int(last_fit_safe_step)
                with torch.no_grad():
                    input_delta.raw_delta.copy_(last_fit_safe_state)
                    partition.apply_role_constraints_(
                        input_delta.raw_delta, bases, roles
                    )
                    geometry.apply_row_caps_(input_delta.raw_delta, input_caps)
                rollback_fit_report = v22.evaluate_endpoint_protection(
                    model,
                    tok,
                    fit_cache,
                    fit_target_ids,
                    fit_base_target_log_probs,
                    device,
                    batch_size=args.capture_batch_size,
                    args=args,
                )
                if not rollback_fit_report["passed"]:
                    completion_failure(
                        output,
                        phase="full_fit_rollback_integrity",
                        detail={
                            "failed_state": fit_report,
                            "restored_state": rollback_fit_report,
                            "rollback_to_step": rollback_to_step,
                        },
                        protocol=active_protocol,
                    )
                    raise RuntimeError(
                        "V2.3.1 could not restore the last full-fit-safe state"
                    )
                rolled_back = True
            elif fit_report["passed"]:
                last_fit_safe_state = input_delta.raw_delta.detach().clone()
                last_fit_safe_step = int(step)
            passed = bool(
                direct["passed"]
                and synth_report["passed"]
                and fit_report["passed"]
                and development["passed"]
                and cap["passed"]
                and compliance["passed"]
            )
            row = {
                "step": step,
                "update": update,
                "hard_tail_size": len(hard_tail.indices),
                "direct": direct,
                "synthetic": synth_report,
                "protection_fit": fit_report,
                "protection_development": development,
                "input_caps": cap,
                "role_compliance": compliance,
                "total_delta_norm": total_norm,
                "full_fit_rollback": {
                    "enabled": bool(args.full_fit_rollback),
                    "performed": rolled_back,
                    "rollback_to_step": rollback_to_step,
                    "restored_fit": rollback_fit_report,
                },
                "passed": passed,
            }
            training_log.append(row)
            if passed:
                candidate_states[step] = (
                input_delta.raw_delta.detach().cpu().clone()
                )
            v2.write_json(
                output / "method" / "projected_training.json",
                {"hard_tail_refreshes": tail_reports, "checks": training_log},
            )
            print(
                f"  step {step:4d}: direct fail {direct['failures']}, "
                f"synth fail {synth_report['failures']}, "
                f"accept {update['accepted_factor']:.4f}, "
                f"fit KL {fit_report['topk_kl_max']:.6f}, "
                f"dev KL {development['topk_kl_max']:.6f}, "
                f"dev top1 {development['top1_logprob_abs_max']:.6f}, "
                f"dev target {development['target_logprob_abs_max']:.6f}, "
                f"rollback={rolled_back}, pass={passed}"
            )
    v2.write_json(
        output / "method" / "projected_training.json",
        {"hard_tail_refreshes": tail_reports, "checks": training_log},
    )
    if not candidate_states:
        completion_failure(
            output,
            phase="projected_row_partition_development",
            detail={
                "checks": training_log,
                "last_hard_tail": tail_reports[-1] if tail_reports else None,
                "partition": partition_report["role_counts"],
            },
            protocol=active_protocol,
        )
        raise RuntimeError("V2.3 produced no development-passing candidate")
    selected_step = min(
        candidate_states,
        key=lambda value: next(
            row["total_delta_norm"] for row in training_log if row["step"] == value
        ),
    )
    with torch.no_grad():
        input_delta.raw_delta.copy_(candidate_states[selected_step].to(device))

    print(f"Stage 4: open certification once for selected step {selected_step}")
    protection_certification = v2.load_partition(
        protocol_dir, manifest, "protection_certification", permitted=True
    )
    raw_certification_cases = v21.make_protection_cases(
        protection_certification,
        targeted=targeted["certification"],
        generic=corpus["certification"],
        forget=forget,
        role="certification",
        tok=tok,
        llama_like=llama_like,
        same_subject_count=args.same_subject_prompts,
    )
    certification_cases = [
        case
        for case in raw_certification_cases
        if case.prompt not in forget_teacher_forced_contexts
    ]
    collision_report.update(
        {
            "certification_raw": len(raw_certification_cases),
            "certification_retained": len(certification_cases),
            "certification_excluded": len(raw_certification_cases)
            - len(certification_cases),
        }
    )
    v2.write_json(
        output / "method" / "retain_collision_precedence.json", collision_report
    )
    held = input_delta.raw_delta.detach().clone()
    with torch.no_grad():
        input_delta.raw_delta.zero_()
    certification_cache = v2.cache_protection(
        model,
        tok,
        certification_cases,
        device,
        batch_size=args.capture_batch_size,
        topk=args.protection_topk,
    )
    (
        certification_target_ids,
        certification_base_target_log_probs,
    ) = v22.cache_retain_targets(
        model,
        tok,
        certification_cases,
        device,
        llama_like=llama_like,
        batch_size=args.capture_batch_size,
    )
    with torch.no_grad():
        input_delta.raw_delta.copy_(held)
    pre = frozen_head_certificate(
        model,
        tok,
        forget,
        synthetic_records,
        certification_cache,
        certification_target_ids,
        certification_base_target_log_probs,
        input_delta,
        input_caps,
        bases,
        roles,
        device,
        llama_like=llama_like,
        args=args,
    )
    v2.write_json(output / "method" / "pre_materialization_certificate.json", pre)
    if not pre["passed"]:
        completion_failure(
            output,
            phase="certification",
            detail=pre,
            protocol=active_protocol,
        )
        raise RuntimeError("V2.3 selected candidate failed certification")

    input_handle.remove()
    endpoint_hooks.materialize_input_delta(
        input_layer, endpoint_rows.input_ids, input_delta.raw_delta.detach().cpu()
    )
    post = frozen_head_certificate(
        model,
        tok,
        forget,
        synthetic_records,
        certification_cache,
        certification_target_ids,
        certification_base_target_log_probs,
        input_delta,
        input_caps,
        bases,
        roles,
        device,
        llama_like=llama_like,
        args=args,
    )
    transformer_after = v2.transformer_sha256(
        model, excluded_parameters=[input_layer.weight, output_layer.weight]
    )
    lm_head_after = v2.tensor_collection_sha256([("lm_head", output_layer.weight)])
    all_model_parameters_require_grad_false = all(
        not parameter.requires_grad for parameter in model.parameters()
    )
    post["pre_post_acceptance_match"] = bool(
        pre["direct"]["failures"] == post["direct"]["failures"]
        and pre["synthetic"]["failures"] == post["synthetic"]["failures"]
        and pre["protection"]["passed"] == post["protection"]["passed"]
    )
    post["transformer_bit_identical"] = transformer_before == transformer_after
    post["lm_head_bit_identical"] = lm_head_before == lm_head_after
    post[
        "all_model_parameters_require_grad_false"
    ] = all_model_parameters_require_grad_false
    post["passed"] = bool(
        post["passed"]
        and post["pre_post_acceptance_match"]
        and post["transformer_bit_identical"]
        and post["lm_head_bit_identical"]
        and post["all_model_parameters_require_grad_false"]
    )
    v2.write_json(output / "method" / "post_materialization_certificate.json", post)
    if not post["passed"]:
        completion_failure(
            output,
            phase="materialization",
            detail=post,
            protocol=active_protocol,
        )
        raise RuntimeError("V2.3 materialization certificate failed")

    candidate_path = output / "method" / "v2_3_candidate_sparse_rows.pt"
    torch.save(
        {
            "schema_version": 1,
            "protocol": active_protocol,
            "base_model_path": str(args.model_path),
            "input_row_ids": endpoint_rows.input_ids,
            "input_rows": input_layer.weight.index_select(0, input_ids_tensor)
            .detach()
            .cpu(),
            "row_roles": list(roles),
            "output_row_ids": [],
            "selected_step": selected_step,
            "split_manifest_sha256": v2.sha256_file(manifest_path),
            "registry_sha256": v2.sha256_file(registry_path),
            "transformer_sha256": transformer_after,
            "lm_head_sha256": lm_head_after,
        },
        candidate_path,
    )
    completion = {
        "schema_version": 1,
        "protocol": active_protocol,
        "passed": True,
        "selected_step": selected_step,
        "row_partition": partition_report["role_counts"],
        "liveness_forced_records": partition_report["liveness_forced_records"],
        "registered_prediction": partition_report["prediction"],
        "per_example_sensitivity_partition": bool(sensitivity_revision),
        "hybrid_guarded_role_policy": bool(hybrid_revision),
        "per_prompt_cap_aware_reachability": bool(
            args.require_per_prompt_reachability
        ),
        "iterative_gradient_recomputed_reachability": bool(
            hybrid_revision and args.require_iterative_reachability
        ),
        "guarded_overlap_fit_cases": len(guarded_overlap_indices),
        "strict_forget_improvement": bool(
            args.require_per_prompt_reachability
        ),
        "full_fit_rollback": bool(args.full_fit_rollback),
        "joint_forget_retain_every_update": True,
        "persistent_hard_tail": True,
        "embedding_rows_changed": int(
            (input_delta.raw_delta.norm(dim=1) > 0).sum().item()
        ),
        "lm_head_rows_changed": 0,
        "lm_head_bit_identical": True,
        "transformer_bit_identical": True,
        "all_model_parameters_require_grad_false": (
            all_model_parameters_require_grad_false
        ),
        "external_classifier": False,
        "runtime_gate": False,
        "candidate_saved": True,
        "candidate_path": str(candidate_path),
        "candidate_sha256": v2.sha256_file(candidate_path),
        "eligible_for_separate_official_evaluation": True,
        "official_evaluation_allowed_in_this_process": False,
        "official_evaluation_prompts_seen": 0,
        "token_disjoint_alias_limitation": (
            "a prompt containing none of the edited token ids produces exactly "
            "Base behaviour, so token-disjoint aliases are out of scope by "
            "construction rather than by optimization failure"
        ),
        "claim_scope": (
            "internal_embedding_behavioral_unlearning_not_latent_erasure"
        ),
    }
    v2.write_json(output / "method" / "completion.json", completion)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
