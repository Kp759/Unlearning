#!/usr/bin/env python3
"""RSNR-V1B-PreHead: benchmark-aligned fact-scoped factual unlearning.

This experiment keeps the successful RSNR PreHead architecture fixed at rank 16
(3072 -> 16 -> 3072 for Llama-3.2-3B-Instruct) while changing the training
objective.  The Base model, all Transformer blocks, final norm, embeddings and
LM head remain frozen.  Only the gated residual adapter immediately before the
LM head is trainable.

For a forgotten (subject, relation) pair, every *training-visible* five-view
prompt is optimized to satisfy all of:

  1) CounterFact/ZeroUnlearn target ordering:
       NLL(true) >= NLL(target_new) + m_cf
     equivalently logP(target_new) - logP(true) >= m_cf.
  2) Nondisclosure:
       logP(IDK) - logP(true) >= m_idk.
  3) True-answer suppression relative to frozen Base:
       logP_Base(true) - logP_Edit(true) >= d.
  4) Exact locality:
       gate OFF => exact Base logits (max drift == 0 by default).

No official paraphrase, neighborhood, fresh-retain, alias, or held-out probe text
is used for training/checkpoint selection.  target_new is used only because this
variant intentionally tests whether the tiny PreHead adapter can satisfy the
same factual target ordering used by the ZeroUnlearn-compatible benchmark.

This is still not a claim of irreversible latent knowledge deletion: disabling
the router recovers the frozen Base model exactly.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

import run_mcf_private_vocab_rewiring_v1 as base_runner
import run_mcf_rsnr_v1a_oracle as rsnr
import run_mcf_rsnr_v1a_prehead as prehead


PROTOCOL = "mcf_rsnr_v1b_prehead_standard_unlearn"
VARIANT = "RSNR-V1B-PreHead-StandardUnlearn"
INTERVENTION_SITE = prehead.INTERVENTION_SITE
ABSTENTION = rsnr.ABSTENTION


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--protocol-dir", required=True)
    p.add_argument("--view-corpus", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--case-batch-size", type=int, default=4)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--adapter-rank", type=int, default=16)
    p.add_argument("--adapter-alpha", type=float, default=16.0)
    p.add_argument("--cf-margin-weight", type=float, default=1.0)
    p.add_argument("--idk-margin-weight", type=float, default=1.0)
    p.add_argument("--true-suppression-weight", type=float, default=1.0)
    p.add_argument("--anchor-weight", type=float, default=1e-4)
    p.add_argument("--minimum-cf-margin", type=float, default=0.1)
    p.add_argument("--minimum-idk-vs-true-margin", type=float, default=0.1)
    p.add_argument("--minimum-true-logprob-drop", type=float, default=2.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--gate-off-logit-drift-max", type=float, default=0.0)
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.seed != 1 or args.forget_num != 50:
        p.error("V1B development experiment is locked to consumed seed1/forget50")
    if args.adapter_rank != 16:
        p.error("V1B capacity test is intentionally locked to rank 16")
    if args.adapter_alpha <= 0:
        p.error("adapter-alpha must be positive")
    if args.steps <= 0 or args.case_batch_size <= 0 or args.check_every <= 0:
        p.error("steps/case-batch-size/check-every must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        p.error("invalid optimizer configuration")
    for name in ("cf_margin_weight", "idk_margin_weight", "true_suppression_weight", "anchor_weight"):
        if float(getattr(args, name)) < 0:
            p.error(f"{name.replace('_', '-')} must be non-negative")
    if args.cf_margin_weight == 0:
        p.error("cf-margin-weight must be > 0 for the standard-unlearning experiment")
    if min(args.minimum_cf_margin, args.minimum_idk_vs_true_margin, args.minimum_true_logprob_drop) < 0:
        p.error("constraint thresholds must be non-negative")
    return args


def prompts_for_cases_with_targets(
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
        if not true or not new:
            raise RuntimeError(f"case {row.get('case_id')} has empty target_true/target_new")
        for template in view_map[int(row["case_id"])]:
            prompts.append(str(template).format(subject))
            true_answers.append(true)
            new_answers.append(new)
            owners.append(local_index)
    return prompts, true_answers, new_answers, owners


def aligned_base_true_tensor(
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
        series = base_true[case_id]
        if k >= len(series):
            raise RuntimeError(f"base true-logprob alignment overflow for case {case_id}")
        values.append(float(series[k]))
        offsets[local] += 1
    return torch.tensor(values, device=device, dtype=torch.float32)


def four_constraint_losses(
    *,
    true_lp: torch.Tensor,
    new_lp: torch.Tensor,
    idk_lp: torch.Tensor,
    base_true_lp: torch.Tensor,
    owners: Sequence[int],
    case_count: int,
    minimum_cf_margin: float,
    minimum_idk_margin: float,
    minimum_true_drop: float,
) -> Dict[str, torch.Tensor]:
    """Return worst-view hinge losses and raw per-prompt margins.

    Because NLL(x) = -logP(x), the benchmark condition
      NLL(true) >= NLL(new) + m
    is exactly
      logP(new) - logP(true) >= m.
    """
    cf_margin = new_lp - true_lp
    idk_margin = idk_lp - true_lp
    true_drop = base_true_lp - true_lp

    cf_hinge = F.relu(float(minimum_cf_margin) - cf_margin)
    idk_hinge = F.relu(float(minimum_idk_margin) - idk_margin)
    suppression_hinge = F.relu(float(minimum_true_drop) - true_drop)

    return {
        "cf_margin": cf_margin,
        "idk_margin": idk_margin,
        "true_drop": true_drop,
        "cf_loss": rsnr.worst_by_owner(cf_hinge, owners, case_count, maximum=True).mean(),
        "idk_loss": rsnr.worst_by_owner(idk_hinge, owners, case_count, maximum=True).mean(),
        "suppression_loss": rsnr.worst_by_owner(
            suppression_hinge, owners, case_count, maximum=True
        ).mean(),
    }


def evaluate_four_constraints(
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
) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(forget), int(batch_cases)):
            cases = forget[start : start + int(batch_cases)]
            prompts, true_answers, new_answers, owners = prompts_for_cases_with_targets(cases, view_map)
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
            base_lp = aligned_base_true_tensor(cases, owners, base_true, device=device)

            for local, row in enumerate(cases):
                idxs = [j for j, owner in enumerate(owners) if owner == local]
                cf_values = [float((new_lp[j] - true_lp[j]).item()) for j in idxs]
                idk_values = [float((idk_lp[j] - true_lp[j]).item()) for j in idxs]
                drop_values = [float((base_lp[j] - true_lp[j]).item()) for j in idxs]
                worst_cf = min(cf_values)
                worst_idk = min(idk_values)
                worst_drop = min(drop_values)
                cf_pass = worst_cf >= float(minimum_cf_margin)
                idk_pass = worst_idk >= float(minimum_idk_margin)
                suppression_pass = worst_drop >= float(minimum_true_drop)
                rows.append({
                    "case_id": int(row["case_id"]),
                    "subject": str(row["requested_rewrite"]["subject"]),
                    "relation_id": str(row["requested_rewrite"]["relation_id"]),
                    "worst_cf_new_vs_true_logprob_margin": worst_cf,
                    "worst_idk_vs_true_logprob_margin": worst_idk,
                    "worst_true_logprob_drop": worst_drop,
                    "cf_pass": bool(cf_pass),
                    "idk_pass": bool(idk_pass),
                    "suppression_pass": bool(suppression_pass),
                    "joint_pass": bool(cf_pass and idk_pass and suppression_pass),
                })

    if not rows:
        raise RuntimeError("no forget rows evaluated")
    return {
        "count": len(rows),
        "joint_passed": sum(bool(r["joint_pass"]) for r in rows),
        "joint_failures": sum(not bool(r["joint_pass"]) for r in rows),
        "cf_passed": sum(bool(r["cf_pass"]) for r in rows),
        "cf_failures": sum(not bool(r["cf_pass"]) for r in rows),
        "idk_passed": sum(bool(r["idk_pass"]) for r in rows),
        "idk_failures": sum(not bool(r["idk_pass"]) for r in rows),
        "suppression_passed": sum(bool(r["suppression_pass"]) for r in rows),
        "suppression_failures": sum(not bool(r["suppression_pass"]) for r in rows),
        "minimum_worst_cf_margin": min(float(r["worst_cf_new_vs_true_logprob_margin"]) for r in rows),
        "minimum_worst_idk_margin": min(float(r["worst_idk_vs_true_logprob_margin"]) for r in rows),
        "minimum_worst_true_logprob_drop": min(float(r["worst_true_logprob_drop"]) for r in rows),
        "cf_margin_threshold": float(minimum_cf_margin),
        "idk_margin_threshold": float(minimum_idk_margin),
        "true_drop_threshold": float(minimum_true_drop),
        "per_case": rows,
    }


def _equivalence_contexts(
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


def _membership(forget: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    return [
        {
            "case_id": int(row["case_id"]),
            "subject": rsnr.fact_key(row)[0],
            "relation_id": rsnr.fact_key(row)[1],
        }
        for row in forget
    ]


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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = base_runner.dtype_from_name(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for V1B PreHead training")
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
    expected_trainable = 2 * hidden_size * int(args.adapter_rank)
    if trainable != expected_trainable:
        raise RuntimeError(f"unexpected adapter parameter count {trainable} != {expected_trainable}")

    print(json.dumps({
        "protocol": PROTOCOL,
        "variant": VARIANT,
        "intervention_site": INTERVENTION_SITE,
        "adapter_shape": [hidden_size, int(args.adapter_rank), hidden_size],
        "trainable_adapter_parameters": trainable,
        "base_model_frozen": True,
        "transformer_weights_modified": False,
        "lm_head_weights_modified": False,
        "input_embeddings_modified": False,
        "target_new_used": True,
        "target_new_scope": "training-visible five-view forget corpus only",
        "official_probe_text_used": False,
        "views_per_case": int(view_meta["views_per_case"]),
        "success_conditions": {
            "cf_new_vs_true_logprob_margin": float(args.minimum_cf_margin),
            "idk_vs_true_logprob_margin": float(args.minimum_idk_vs_true_margin),
            "true_logprob_drop": float(args.minimum_true_logprob_drop),
            "gate_off_max_abs_logit_drift": float(args.gate_off_logit_drift_max),
        },
    }, indent=2), flush=True)

    contexts = _equivalence_contexts(forget, protection_fit)
    initial_eq = rsnr.gate_off_equivalence(model, hook, tokenizer, contexts, device=device)
    if initial_eq["max_abs_logit_drift"] > float(args.gate_off_logit_drift_max):
        raise RuntimeError("initial gate-off Base equivalence failed")

    base_true = rsnr.base_true_logprobs_for_all_views(
        model, hook, tokenizer, forget, view_map,
        device=device, batch_cases=int(args.case_batch_size),
    )

    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    rng = random.Random(int(args.seed) + 44017)
    best_state = copy.deepcopy(adapter.state_dict())
    best_key: tuple[Any, ...] = (10**9, 10**9, 10**9, 10**9, float("inf"), float("inf"), float("inf"))
    training_log: list[Dict[str, Any]] = []

    for step in range(1, int(args.steps) + 1):
        cases = rng.sample(forget, min(int(args.case_batch_size), len(forget)))
        prompts, true_answers, new_answers, owners = prompts_for_cases_with_targets(cases, view_map)
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
        base_lp = aligned_base_true_tensor(cases, owners, base_true, device=device)
        terms = four_constraint_losses(
            true_lp=true_lp,
            new_lp=new_lp,
            idk_lp=idk_lp,
            base_true_lp=base_lp,
            owners=owners,
            case_count=len(cases),
            minimum_cf_margin=float(args.minimum_cf_margin),
            minimum_idk_margin=float(args.minimum_idk_vs_true_margin),
            minimum_true_drop=float(args.minimum_true_logprob_drop),
        )
        anchor = adapter.up.weight.float().pow(2).mean()
        loss = (
            float(args.cf_margin_weight) * terms["cf_loss"]
            + float(args.idk_margin_weight) * terms["idk_loss"]
            + float(args.true_suppression_weight) * terms["suppression_loss"]
            + float(args.anchor_weight) * anchor
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(adapter.parameters()), float(args.grad_clip))
        optimizer.step()

        if step == 1 or step % int(args.check_every) == 0 or step == int(args.steps):
            metrics = evaluate_four_constraints(
                model, hook, tokenizer, forget, view_map, base_true,
                device=device,
                batch_cases=int(args.case_batch_size),
                minimum_cf_margin=float(args.minimum_cf_margin),
                minimum_idk_margin=float(args.minimum_idk_vs_true_margin),
                minimum_true_drop=float(args.minimum_true_logprob_drop),
            )
            eq = rsnr.gate_off_equivalence(
                model, hook, tokenizer, contexts[:32], device=device
            )
            key = (
                int(metrics["joint_failures"]),
                int(metrics["cf_failures"]),
                int(metrics["idk_failures"]),
                int(metrics["suppression_failures"]),
                -float(metrics["minimum_worst_cf_margin"]),
                -float(metrics["minimum_worst_idk_margin"]),
                -float(metrics["minimum_worst_true_logprob_drop"]),
            )
            if key < best_key:
                best_key = key
                best_state = copy.deepcopy(adapter.state_dict())
            row = {
                "step": int(step),
                "loss": float(loss.detach().item()),
                "cf_margin_loss": float(terms["cf_loss"].detach().item()),
                "idk_margin_loss": float(terms["idk_loss"].detach().item()),
                "true_suppression_loss": float(terms["suppression_loss"].detach().item()),
                "anchor": float(anchor.detach().item()),
                "joint_passed": int(metrics["joint_passed"]),
                "joint_failures": int(metrics["joint_failures"]),
                "cf_passed": int(metrics["cf_passed"]),
                "idk_passed": int(metrics["idk_passed"]),
                "suppression_passed": int(metrics["suppression_passed"]),
                "minimum_worst_cf_margin": float(metrics["minimum_worst_cf_margin"]),
                "minimum_worst_idk_margin": float(metrics["minimum_worst_idk_margin"]),
                "minimum_worst_true_logprob_drop": float(metrics["minimum_worst_true_logprob_drop"]),
                "gate_off_max_abs_logit_drift": float(eq["max_abs_logit_drift"]),
                **rsnr.adapter_norm(adapter),
            }
            training_log.append(row)
            print(
                f"step {step:4d}: joint={metrics['joint_passed']}/50 "
                f"CF={metrics['cf_passed']}/50 IDK={metrics['idk_passed']}/50 "
                f"DROP={metrics['suppression_passed']}/50 | "
                f"worst CF={metrics['minimum_worst_cf_margin']:.4f} "
                f"IDK={metrics['minimum_worst_idk_margin']:.4f} "
                f"drop={metrics['minimum_worst_true_logprob_drop']:.4f} "
                f"off-drift={eq['max_abs_logit_drift']:.3g}",
                flush=True,
            )
            if metrics["joint_failures"] == 0 and eq["max_abs_logit_drift"] <= float(args.gate_off_logit_drift_max):
                print("all 50 cases pass all four conditions on all five training views; stopping early", flush=True)
                break

    adapter.load_state_dict(best_state)
    final_metrics = evaluate_four_constraints(
        model, hook, tokenizer, forget, view_map, base_true,
        device=device,
        batch_cases=int(args.case_batch_size),
        minimum_cf_margin=float(args.minimum_cf_margin),
        minimum_idk_margin=float(args.minimum_idk_vs_true_margin),
        minimum_true_drop=float(args.minimum_true_logprob_drop),
    )
    final_eq = rsnr.gate_off_equivalence(model, hook, tokenizer, contexts, device=device)
    gate_passed = (
        int(final_metrics["joint_failures"]) == 0
        and float(final_eq["max_abs_logit_drift"]) <= float(args.gate_off_logit_drift_max)
    )

    membership = _membership(forget)
    adapter_path = method_dir / "rsnr_prehead_oracle_null_adapter.pt"
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
        "target_new_used": True,
        "transformer_weights_modified": False,
        "lm_head_weights_modified": False,
        "input_embeddings_modified": False,
        "training_gate_passed": bool(gate_passed),
    }, adapter_path)

    sidecar = {
        "protocol": PROTOCOL,
        "variant": VARIANT,
        "routing": "oracle_exact_subject_relation_membership",
        "intervention_site": INTERVENTION_SITE,
        "non_target_behavior": "adapter_off_exact_base_path",
        "sensitive_behavior": "activate_rank16_prehead_standard_unlearn_adapter",
        "abstention_text": ABSTENTION,
        "target_new_used": True,
        "target_new_scope": "training-visible five-view forget corpus only",
        "transformer_weights_modified": False,
        "lm_head_weights_modified": False,
        "forget_membership": membership,
    }
    (method_dir / "relation_scoped_null_routing_prehead.json").write_text(
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
            "adapter_rank": int(args.adapter_rank),
            "adapter_alpha": float(args.adapter_alpha),
            "trainable_parameters": trainable,
            "base_model_frozen": True,
            "transformer_frozen": True,
            "lm_head_frozen": True,
            "input_embeddings_frozen": True,
            "off_route": "exact Base bypass",
        },
        "objective": {
            "definition": "worst-of-five hinge constraints on training-visible forgotten facts",
            "cf_condition": "logP(target_new)-logP(target_true) >= minimum_cf_margin",
            "cf_equivalent": "NLL(target_true) >= NLL(target_new)+minimum_cf_margin",
            "idk_condition": "logP(IDK)-logP(target_true) >= minimum_idk_vs_true_margin",
            "suppression_condition": "logP_Base(target_true)-logP_Edit(target_true) >= minimum_true_logprob_drop",
            "cf_margin_weight": float(args.cf_margin_weight),
            "idk_margin_weight": float(args.idk_margin_weight),
            "true_suppression_weight": float(args.true_suppression_weight),
            "anchor_weight": float(args.anchor_weight),
            "minimum_cf_margin": float(args.minimum_cf_margin),
            "minimum_idk_vs_true_margin": float(args.minimum_idk_vs_true_margin),
            "minimum_true_logprob_drop": float(args.minimum_true_logprob_drop),
            "target_new_used": True,
            "retain_GD_used": False,
            "retain_GD_note": "off-route retain examples have no adapter gradient because the adapter is structurally bypassed",
        },
        "training_view_corpus": {
            **view_meta,
            "path": str(view_path),
            "official_paraphrase_text_used": False,
            "official_neighborhood_text_used": False,
            "fresh_retain_text_used": False,
            "heldout_aliases_used": False,
            "heldout_probe_text_used": False,
        },
        "oracle_gate_audit": oracle_audit,
        "gate_off_equivalence": final_eq,
        "final_training_view_metrics": final_metrics,
        "adapter_norm": rsnr.adapter_norm(adapter),
        "training_log": training_log,
        "claim_boundary": {
            "benchmark_compatible_factual_unlearning_targeted": True,
            "conditional_nondisclosure_targeted": True,
            "exact_off_route_identity": True,
            "irreversible_latent_knowledge_deletion_claimed": False,
            "disabling_intervention_recovers_base": True,
            "oracle_gate_is_not_learned": True,
        },
    }
    (method_dir / "rsnr_v1b_prehead_standard_unlearn.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    completion = {
        "protocol": PROTOCOL,
        "variant": VARIANT,
        "joint_passed": int(final_metrics["joint_passed"]),
        "joint_failures": int(final_metrics["joint_failures"]),
        "cf_passed": int(final_metrics["cf_passed"]),
        "cf_failures": int(final_metrics["cf_failures"]),
        "idk_passed": int(final_metrics["idk_passed"]),
        "idk_failures": int(final_metrics["idk_failures"]),
        "suppression_passed": int(final_metrics["suppression_passed"]),
        "suppression_failures": int(final_metrics["suppression_failures"]),
        "minimum_worst_cf_margin": float(final_metrics["minimum_worst_cf_margin"]),
        "minimum_worst_idk_margin": float(final_metrics["minimum_worst_idk_margin"]),
        "minimum_worst_true_logprob_drop": float(final_metrics["minimum_worst_true_logprob_drop"]),
        "gate_off_max_abs_logit_drift": float(final_eq["max_abs_logit_drift"]),
        "training_gate_passed": bool(gate_passed),
        "adapter_saved": True,
        "base_weights_modified": False,
        "transformer_weights_modified": False,
        "lm_head_weights_modified": False,
        "heldout_probe_text_used": False,
        "target_new_used": True,
    }
    (method_dir / "completion.json").write_text(
        json.dumps(completion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "training_gate_passed": bool(gate_passed),
        "final_training_view_metrics": {k: v for k, v in final_metrics.items() if k != "per_case"},
        "gate_off_equivalence": final_eq,
        "adapter_path": str(adapter_path),
    }, indent=2), flush=True)
    hook.remove()
    if not gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
