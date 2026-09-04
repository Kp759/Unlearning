#!/usr/bin/env python3
"""RSNR-V1B-PreHead: benchmark-aligned factual unlearning with exact routing locality.

This development experiment keeps the Llama base model completely frozen and
trains only the same rank-16 3072->16->3072 residual adapter immediately before
the frozen LM head.  The adapter is activated only for an oracle-known
(subject, relation) forget query; gate OFF is the exact Base computation.

The training objective is deliberately different from the earlier IDK-only
PreHead experiment.  It targets four simultaneous properties on the locked,
training-safe five-view corpus:

  1. CounterFact / ZeroUnlearn-compatible ordering:
       NLL(true) - NLL(new) >= minimum_cf_margin
     equivalently logP(new) - logP(true) >= minimum_cf_margin.
     target_new is a DETACHED reference only: no gradient is propagated through
     its likelihood and the method never positively trains target_new.

  2. Nondisclosure:
       logP(IDK) - logP(true) >= minimum_idk_margin.

  3. True-answer suppression relative to frozen Base:
       logP_Base(true) - logP_Edit(true) >= minimum_true_logprob_drop.

  4. Exact off-route locality:
       max |logits_gate_off - logits_Base| <= gate_off_logit_drift_max.

In addition, ordinary gradient ascent on the forget-answer CE is implemented as
minimization of mean logP(true).  A natural-abstention CE term is retained so
that satisfying the IDK margin does not rely only on crushing the true answer.

The official CounterFact paraphrases/neighborhood prompts are never used for
training or checkpoint selection.  Seed 1 remains DEVELOPMENT ONLY.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

import run_mcf_private_vocab_rewiring_v1 as base_runner
import run_mcf_rsnr_v1a_oracle as rsnr
import run_mcf_rsnr_v1a_prehead as prehead


PROTOCOL = "mcf_rsnr_v1b_prehead_standard_unlearn"
VARIANT = "RSNR-V1B-PreHead-StandardUnlearn"
ABSTENTION = rsnr.ABSTENTION
INTERVENTION_SITE = prehead.INTERVENTION_SITE


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--protocol-dir", required=True)
    p.add_argument("--view-corpus", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--case-batch-size", type=int, default=4)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--adapter-rank", type=int, default=16)
    p.add_argument("--adapter-alpha", type=float, default=16.0)

    # Optimization terms.  GA is deliberately modest because the hard margin
    # losses already specify how far forgetting must move.
    p.add_argument("--ga-weight", type=float, default=0.25)
    p.add_argument("--abstain-weight", type=float, default=1.0)
    p.add_argument("--cf-margin-weight", type=float, default=1.0)
    p.add_argument("--idk-margin-weight", type=float, default=1.0)
    p.add_argument("--drop-weight", type=float, default=1.0)
    p.add_argument("--anchor-weight", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Hard training-safe certification thresholds.
    p.add_argument("--minimum-cf-margin", type=float, default=0.1)
    p.add_argument("--minimum-idk-margin", type=float, default=0.1)
    p.add_argument("--minimum-true-logprob-drop", type=float, default=2.0)
    p.add_argument("--gate-off-logit-drift-max", type=float, default=0.0)

    args = p.parse_args(list(argv) if argv is not None else None)
    if args.seed != 1 or args.forget_num != 50:
        p.error("RSNR-V1B development run is locked to consumed seed1/forget50")
    if args.steps <= 0 or args.case_batch_size <= 0 or args.check_every <= 0:
        p.error("steps/batch/check must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        p.error("invalid optimizer configuration")
    if args.adapter_rank <= 0 or args.adapter_alpha <= 0:
        p.error("adapter rank/alpha must be positive")
    for name in (
        "ga_weight", "abstain_weight", "cf_margin_weight", "idk_margin_weight",
        "drop_weight", "anchor_weight", "minimum_cf_margin", "minimum_idk_margin",
        "minimum_true_logprob_drop", "gate_off_logit_drift_max",
    ):
        if float(getattr(args, name)) < 0:
            p.error(f"{name} must be non-negative")
    if not any(
        float(getattr(args, name)) > 0
        for name in ("ga_weight", "abstain_weight", "cf_margin_weight", "idk_margin_weight", "drop_weight")
    ):
        p.error("at least one sensitive objective must be active")
    return args


def _membership(forget: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": int(row["case_id"]),
            "subject": rsnr.fact_key(row)[0],
            "relation_id": rsnr.fact_key(row)[1],
            "target_true": str(row["requested_rewrite"]["target_true"]["str"]),
        }
        for row in forget
    ]


def prompts_true_new_for_cases(
    cases: Sequence[Mapping[str, Any]],
    view_map: Mapping[int, Sequence[str]],
) -> tuple[list[str], list[str], list[str], list[int]]:
    prompts: list[str] = []
    true_answers: list[str] = []
    new_answers: list[str] = []
    owners: list[int] = []
    for local_index, row in enumerate(cases):
        rr = row["requested_rewrite"]
        subject = str(rr["subject"])
        true = str(rr["target_true"]["str"])
        new = str(rr["target_new"]["str"])
        for template in view_map[int(row["case_id"])]:
            prompts.append(str(template).format(subject))
            true_answers.append(true)
            new_answers.append(new)
            owners.append(local_index)
    return prompts, true_answers, new_answers, owners


def base_true_tensor_for_batch(
    cases: Sequence[Mapping[str, Any]],
    owners: Sequence[int],
    base_true: Mapping[int, Sequence[float]],
    *,
    device: torch.device,
) -> torch.Tensor:
    offsets = [0 for _ in cases]
    values: list[float] = []
    for owner in owners:
        local = int(owner)
        case_id = int(cases[local]["case_id"])
        k = offsets[local]
        values.append(float(base_true[case_id][k]))
        offsets[local] += 1
    return torch.tensor(values, device=device, dtype=torch.float32)


def compute_objective_terms(
    true_lp: torch.Tensor,
    new_lp: torch.Tensor,
    idk_lp: torch.Tensor,
    base_true_lp: torch.Tensor,
    owners: Sequence[int],
    case_count: int,
    *,
    minimum_cf_margin: float,
    minimum_idk_margin: float,
    minimum_true_drop: float,
) -> dict[str, torch.Tensor]:
    """Return differentiable losses and raw margins for one training minibatch.

    `new_lp` is detached in the CounterFact reference margin.  Consequently the
    target_new sequence can define the benchmark boundary but receives no
    gradient encouraging it to become more likely.
    """
    if not (true_lp.shape == new_lp.shape == idk_lp.shape == base_true_lp.shape):
        raise ValueError("all log-probability vectors must have identical shape")

    cf_margin = new_lp.detach() - true_lp
    idk_margin = idk_lp - true_lp
    true_drop = base_true_lp.detach() - true_lp

    cf_hinge = F.relu(float(minimum_cf_margin) - cf_margin)
    idk_hinge = F.relu(float(minimum_idk_margin) - idk_margin)
    drop_hinge = F.relu(float(minimum_true_drop) - true_drop)

    # GA on forget CE: maximize CE(true) == minimize logP(true).
    ga_loss = rsnr.worst_by_owner(true_lp, owners, case_count, maximum=True).mean()
    abstain_loss = rsnr.worst_by_owner(-idk_lp, owners, case_count, maximum=True).mean()
    cf_margin_loss = rsnr.worst_by_owner(cf_hinge, owners, case_count, maximum=True).mean()
    idk_margin_loss = rsnr.worst_by_owner(idk_hinge, owners, case_count, maximum=True).mean()
    drop_loss = rsnr.worst_by_owner(drop_hinge, owners, case_count, maximum=True).mean()

    return {
        "ga_loss": ga_loss,
        "abstain_loss": abstain_loss,
        "cf_margin_loss": cf_margin_loss,
        "idk_margin_loss": idk_margin_loss,
        "drop_loss": drop_loss,
        "cf_margin": cf_margin,
        "idk_margin": idk_margin,
        "true_drop": true_drop,
    }


def summarize_four_conditions(
    *,
    case_rows: Sequence[Mapping[str, Any]],
    minimum_cf_margin: float,
    minimum_idk_margin: float,
    minimum_true_drop: float,
) -> dict[str, Any]:
    rows = [dict(row) for row in case_rows]
    for row in rows:
        row["cf_pass"] = float(row["worst_cf_margin"]) >= float(minimum_cf_margin)
        row["idk_pass"] = float(row["worst_idk_margin"]) >= float(minimum_idk_margin)
        row["drop_pass"] = float(row["worst_true_drop"]) >= float(minimum_true_drop)
        row["joint_pass"] = bool(row["cf_pass"] and row["idk_pass"] and row["drop_pass"])
    return {
        "count": len(rows),
        "joint_passed": sum(bool(r["joint_pass"]) for r in rows),
        "joint_failures": sum(not bool(r["joint_pass"]) for r in rows),
        "cf_passed": sum(bool(r["cf_pass"]) for r in rows),
        "idk_passed": sum(bool(r["idk_pass"]) for r in rows),
        "drop_passed": sum(bool(r["drop_pass"]) for r in rows),
        "minimum_worst_cf_margin": min(float(r["worst_cf_margin"]) for r in rows) if rows else None,
        "minimum_worst_idk_margin": min(float(r["worst_idk_margin"]) for r in rows) if rows else None,
        "minimum_worst_true_drop": min(float(r["worst_true_drop"]) for r in rows) if rows else None,
        "thresholds": {
            "minimum_cf_margin": float(minimum_cf_margin),
            "minimum_idk_margin": float(minimum_idk_margin),
            "minimum_true_drop": float(minimum_true_drop),
        },
        "per_case": rows,
    }


@torch.no_grad()
def evaluate_training_safe_conditions(
    model: Any,
    hook: prehead.PreHeadNullHook,
    tokenizer: Any,
    forget: Sequence[Mapping[str, Any]],
    view_map: Mapping[int, Sequence[str]],
    base_true: Mapping[int, Sequence[float]],
    *,
    device: torch.device,
    batch_cases: int,
    minimum_cf_margin: float,
    minimum_idk_margin: float,
    minimum_true_drop: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(forget), int(batch_cases)):
        cases = forget[start : start + int(batch_cases)]
        prompts, true_answers, new_answers, owners = prompts_true_new_for_cases(cases, view_map)
        idk_answers = [ABSTENTION] * len(prompts)
        true_lp = rsnr.sequence_logprobs(
            model, hook, tokenizer, prompts, true_answers, device=device, gated=True
        )
        new_lp = rsnr.sequence_logprobs(
            model, hook, tokenizer, prompts, new_answers, device=device, gated=True
        )
        idk_lp = rsnr.sequence_logprobs(
            model, hook, tokenizer, prompts, idk_answers, device=device, gated=True
        )
        base_tensor = base_true_tensor_for_batch(cases, owners, base_true, device=device)
        cf = new_lp - true_lp
        idk = idk_lp - true_lp
        drop = base_tensor - true_lp
        for local, row in enumerate(cases):
            idxs = [j for j, owner in enumerate(owners) if owner == local]
            rows.append({
                "case_id": int(row["case_id"]),
                "subject": str(row["requested_rewrite"]["subject"]),
                "relation_id": str(row["requested_rewrite"]["relation_id"]),
                "worst_cf_margin": min(float(cf[j].item()) for j in idxs),
                "worst_idk_margin": min(float(idk[j].item()) for j in idxs),
                "worst_true_drop": min(float(drop[j].item()) for j in idxs),
            })
    return summarize_four_conditions(
        case_rows=rows,
        minimum_cf_margin=minimum_cf_margin,
        minimum_idk_margin=minimum_idk_margin,
        minimum_true_drop=minimum_true_drop,
    )


def _build_gate_off_contexts(
    forget: Sequence[Mapping[str, Any]],
    protection_fit: Sequence[Mapping[str, Any]],
) -> list[str]:
    forget_pairs = {rsnr.fact_key(row) for row in forget}
    forget_subjects = {s for s, _ in forget_pairs}
    forget_relations = {r for _, r in forget_pairs}
    contexts: list[str] = []
    for row in forget:
        subject = str(row["requested_rewrite"]["subject"])
        contexts.extend([subject, f"Tell me about {subject}."])
    for row in protection_fit:
        if rsnr.fact_key(row) in forget_pairs:
            continue
        subject, relation = rsnr.fact_key(row)
        if subject in forget_subjects or relation in forget_relations:
            contexts.append(base_runner.render_prompt(row))
        if len(contexts) >= 192:
            break
    return list(dict.fromkeys(contexts))[:192]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    method_dir = output / "method"
    method_dir.mkdir()

    protocol = rsnr.load_protocol(Path(args.protocol_dir), int(args.forget_num))
    forget = protocol["forget"]
    protection_fit = protocol["protection_fit"]
    view_path = Path(args.view_corpus).resolve()
    view_map, view_meta = rsnr.load_training_views(view_path)
    rsnr.validate_case_alignment(forget, view_map)
    oracle_audit = rsnr.build_oracle_negative_audit(forget, protection_fit)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    dtype = base_runner.dtype_from_name(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for RSNR-V1B PreHead training")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    hidden_size = int(getattr(model.config, "hidden_size"))
    adapter = rsnr.NullResidualAdapter(
        hidden_size, int(args.adapter_rank), float(args.adapter_alpha), device
    ).to(device)
    hook = prehead.PreHeadNullHook.install(prehead.get_lm_head(model), adapter)

    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    frozen_base = sum(p.numel() for p in model.parameters())
    expected_trainable = 2 * hidden_size * int(args.adapter_rank)
    if trainable != expected_trainable:
        raise RuntimeError(f"unexpected adapter parameter count {trainable} != {expected_trainable}")

    print(json.dumps({
        "protocol": PROTOCOL,
        "variant": VARIANT,
        "intervention_site": INTERVENTION_SITE,
        "adapter_shape": [hidden_size, int(args.adapter_rank), hidden_size],
        "adapter_rank": int(args.adapter_rank),
        "adapter_alpha": float(args.adapter_alpha),
        "trainable_adapter_parameters": trainable,
        "frozen_base_parameters": frozen_base,
        "transformer_weights_modified": False,
        "lm_head_weights_modified": False,
        "input_embeddings_modified": False,
        "target_new_role": "detached benchmark reference only",
        "target_new_positive_likelihood_training": False,
        "target_new_gradient": False,
        "ga_on_target_true": True,
        "registered_abstention": ABSTENTION,
        "views_per_case": int(view_meta["views_per_case"]),
        "heldout_probe_text_used": False,
    }, indent=2), flush=True)

    contexts = _build_gate_off_contexts(forget, protection_fit)
    equivalence = rsnr.gate_off_equivalence(model, hook, tokenizer, contexts, device=device)
    if equivalence["max_abs_logit_drift"] > float(args.gate_off_logit_drift_max):
        raise RuntimeError("initial PreHead gate-off path is not Base-identical")

    base_true = rsnr.base_true_logprobs_for_all_views(
        model, hook, tokenizer, forget, view_map,
        device=device, batch_cases=int(args.case_batch_size),
    )

    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    rng = random.Random(int(args.seed) + 44017)
    best_state = copy.deepcopy(adapter.state_dict())
    best_key = (10**9, 10**9, 10**9, 10**9, float("inf"), float("inf"), float("inf"))
    training_log: list[dict[str, Any]] = []

    for step in range(1, int(args.steps) + 1):
        cases = rng.sample(forget, min(int(args.case_batch_size), len(forget)))
        prompts, true_answers, new_answers, owners = prompts_true_new_for_cases(cases, view_map)
        idk_answers = [ABSTENTION] * len(prompts)
        optimizer.zero_grad(set_to_none=True)

        true_lp = rsnr.sequence_logprobs(
            model, hook, tokenizer, prompts, true_answers, device=device, gated=True
        )
        new_lp = rsnr.sequence_logprobs(
            model, hook, tokenizer, prompts, new_answers, device=device, gated=True
        )
        idk_lp = rsnr.sequence_logprobs(
            model, hook, tokenizer, prompts, idk_answers, device=device, gated=True
        )
        base_tensor = base_true_tensor_for_batch(cases, owners, base_true, device=device)
        terms = compute_objective_terms(
            true_lp, new_lp, idk_lp, base_tensor, owners, len(cases),
            minimum_cf_margin=float(args.minimum_cf_margin),
            minimum_idk_margin=float(args.minimum_idk_margin),
            minimum_true_drop=float(args.minimum_true_logprob_drop),
        )
        anchor = adapter.up.weight.float().pow(2).mean()
        loss = (
            float(args.ga_weight) * terms["ga_loss"]
            + float(args.abstain_weight) * terms["abstain_loss"]
            + float(args.cf_margin_weight) * terms["cf_margin_loss"]
            + float(args.idk_margin_weight) * terms["idk_margin_loss"]
            + float(args.drop_weight) * terms["drop_loss"]
            + float(args.anchor_weight) * anchor
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(adapter.parameters()), float(args.grad_clip))
        optimizer.step()

        if step == 1 or step % int(args.check_every) == 0 or step == int(args.steps):
            metrics = evaluate_training_safe_conditions(
                model, hook, tokenizer, forget, view_map, base_true,
                device=device,
                batch_cases=int(args.case_batch_size),
                minimum_cf_margin=float(args.minimum_cf_margin),
                minimum_idk_margin=float(args.minimum_idk_margin),
                minimum_true_drop=float(args.minimum_true_logprob_drop),
            )
            eq = rsnr.gate_off_equivalence(model, hook, tokenizer, contexts[:32], device=device)
            per_case = metrics["per_case"]
            key = (
                int(metrics["joint_failures"]),
                len(per_case) - int(metrics["cf_passed"]),
                len(per_case) - int(metrics["idk_passed"]),
                len(per_case) - int(metrics["drop_passed"]),
                -float(metrics["minimum_worst_cf_margin"]),
                -float(metrics["minimum_worst_idk_margin"]),
                -float(metrics["minimum_worst_true_drop"]),
            )
            if key < best_key:
                best_key = key
                best_state = copy.deepcopy(adapter.state_dict())
            row = {
                "step": int(step),
                "loss": float(loss.detach().item()),
                "ga_loss": float(terms["ga_loss"].detach().item()),
                "abstain_loss": float(terms["abstain_loss"].detach().item()),
                "cf_margin_loss": float(terms["cf_margin_loss"].detach().item()),
                "idk_margin_loss": float(terms["idk_margin_loss"].detach().item()),
                "drop_loss": float(terms["drop_loss"].detach().item()),
                "anchor": float(anchor.detach().item()),
                "joint_passed": int(metrics["joint_passed"]),
                "joint_failures": int(metrics["joint_failures"]),
                "cf_passed": int(metrics["cf_passed"]),
                "idk_passed": int(metrics["idk_passed"]),
                "drop_passed": int(metrics["drop_passed"]),
                "minimum_worst_cf_margin": float(metrics["minimum_worst_cf_margin"]),
                "minimum_worst_idk_margin": float(metrics["minimum_worst_idk_margin"]),
                "minimum_worst_true_drop": float(metrics["minimum_worst_true_drop"]),
                "gate_off_max_abs_logit_drift": float(eq["max_abs_logit_drift"]),
                **rsnr.adapter_norm(adapter),
            }
            training_log.append(row)
            print(
                f"step {step:4d}: all4={metrics['joint_passed']}/50, "
                f"CF={metrics['cf_passed']}/50 IDK={metrics['idk_passed']}/50 "
                f"DROP={metrics['drop_passed']}/50; "
                f"worst CF={metrics['minimum_worst_cf_margin']:.4f}, "
                f"IDK={metrics['minimum_worst_idk_margin']:.4f}, "
                f"drop={metrics['minimum_worst_true_drop']:.4f}, "
                f"off-drift={eq['max_abs_logit_drift']:.3g}",
                flush=True,
            )
            if (
                int(metrics["joint_failures"]) == 0
                and float(eq["max_abs_logit_drift"]) <= float(args.gate_off_logit_drift_max)
            ):
                print("all 50 cases pass all four conditions on all five training-safe views", flush=True)
                break

    adapter.load_state_dict(best_state)
    final_metrics = evaluate_training_safe_conditions(
        model, hook, tokenizer, forget, view_map, base_true,
        device=device,
        batch_cases=int(args.case_batch_size),
        minimum_cf_margin=float(args.minimum_cf_margin),
        minimum_idk_margin=float(args.minimum_idk_margin),
        minimum_true_drop=float(args.minimum_true_logprob_drop),
    )
    final_equivalence = rsnr.gate_off_equivalence(model, hook, tokenizer, contexts, device=device)
    exact_locality_pass = (
        float(final_equivalence["max_abs_logit_drift"])
        <= float(args.gate_off_logit_drift_max)
    )
    if not exact_locality_pass:
        raise RuntimeError("final PreHead gate-off Base equivalence failed")

    training_gate_passed = int(final_metrics["joint_failures"]) == 0 and exact_locality_pass
    membership = _membership(forget)
    artifact_path = method_dir / "rsnr_prehead_standard_unlearn_adapter.pt"
    torch.save({
        "protocol": PROTOCOL,
        "variant": VARIANT,
        "base_model": str(args.model_path),
        "intervention_site": INTERVENTION_SITE,
        "hidden_size": hidden_size,
        "adapter_rank": int(args.adapter_rank),
        "adapter_alpha": float(args.adapter_alpha),
        "adapter_state_dict": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
        "abstention": ABSTENTION,
        "forget_membership": membership,
        "transformer_weights_modified": False,
        "lm_head_weights_modified": False,
        "input_embeddings_modified": False,
        "target_new_role": "detached_reference_only",
        "target_new_gradient": False,
    }, artifact_path)

    sidecar = {
        "protocol": PROTOCOL,
        "variant": VARIANT,
        "routing": "oracle_exact_subject_relation_membership",
        "intervention_site": INTERVENTION_SITE,
        "atomic_query_scope": True,
        "non_target_behavior": "adapter_off_exact_base_path",
        "sensitive_behavior": "activate_pre_lm_head_standard_unlearn_adapter",
        "abstention_text": ABSTENTION,
        "target_new_used": True,
        "target_new_role": "detached CounterFact/ZeroUnlearn reference only",
        "target_new_positive_likelihood_training": False,
        "target_new_gradient": False,
        "forget_membership": membership,
    }
    (method_dir / "relation_scoped_prehead_standard_unlearn.json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report = {
        "protocol": PROTOCOL,
        "variant": VARIANT,
        "seed": int(args.seed),
        "development_only": True,
        "architecture": {
            "intervention_site": INTERVENTION_SITE,
            "adapter_shape": [hidden_size, int(args.adapter_rank), hidden_size],
            "base_model_frozen": True,
            "all_transformer_blocks_frozen": True,
            "final_norm_frozen": True,
            "lm_head_frozen": True,
            "input_embeddings_frozen": True,
            "trainable_parameters": trainable,
            "adapter_rank": int(args.adapter_rank),
            "adapter_alpha": float(args.adapter_alpha),
            "non_target_path": "exact Base path; adapter disabled",
        },
        "objective": {
            "ga_true_weight": float(args.ga_weight),
            "natural_abstention_weight": float(args.abstain_weight),
            "cf_reference_margin_weight": float(args.cf_margin_weight),
            "idk_margin_weight": float(args.idk_margin_weight),
            "base_true_drop_weight": float(args.drop_weight),
            "adapter_anchor_weight": float(args.anchor_weight),
            "minimum_cf_margin": float(args.minimum_cf_margin),
            "minimum_idk_margin": float(args.minimum_idk_margin),
            "minimum_true_logprob_drop": float(args.minimum_true_logprob_drop),
            "target_new_role": "detached reference; not a positive training target",
            "target_new_gradient": False,
            "worst_of_5_training_views": True,
        },
        "training_view_corpus": {
            **view_meta,
            "path": str(view_path),
            "official_paraphrase_text_used": False,
            "official_neighborhood_text_used": False,
            "heldout_probe_text_used": False,
        },
        "oracle_gate_audit": oracle_audit,
        "gate_off_equivalence": final_equivalence,
        "final_training_view_metrics": final_metrics,
        "adapter_norm": rsnr.adapter_norm(adapter),
        "training_log": training_log,
        "claim_boundary": {
            "benchmark_compatible_factual_unlearning_targeted": True,
            "conditional_unlearning": True,
            "latent_knowledge_erasure_claimed": False,
            "oracle_gate_is_not_learned": True,
            "disabling_intervention_recovers_base": True,
            "target_new_learned_as_replacement": False,
            "lm_head_weights_changed": False,
            "transformer_weights_changed": False,
        },
    }
    (method_dir / "rsnr_v1b_prehead_standard_unlearn.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    completion = {
        "protocol": PROTOCOL,
        "variant": VARIANT,
        "training_gate_passed": bool(training_gate_passed),
        "all_four_conditions_passed": bool(training_gate_passed),
        "joint_passed": int(final_metrics["joint_passed"]),
        "joint_failures": int(final_metrics["joint_failures"]),
        "cf_passed": int(final_metrics["cf_passed"]),
        "idk_passed": int(final_metrics["idk_passed"]),
        "drop_passed": int(final_metrics["drop_passed"]),
        "minimum_worst_cf_margin": float(final_metrics["minimum_worst_cf_margin"]),
        "minimum_worst_idk_margin": float(final_metrics["minimum_worst_idk_margin"]),
        "minimum_worst_true_drop": float(final_metrics["minimum_worst_true_drop"]),
        "gate_off_max_abs_logit_drift": float(final_equivalence["max_abs_logit_drift"]),
        "adapter_saved": True,
        "adapter_path": str(artifact_path),
        "trainable_parameters": trainable,
        "base_weights_modified": False,
        "transformer_weights_modified": False,
        "lm_head_weights_modified": False,
        "target_new_positive_likelihood_training": False,
        "target_new_gradient": False,
        "heldout_probe_text_used": False,
    }
    (method_dir / "completion.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(completion, indent=2), flush=True)
    hook.remove()
    if not training_gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
