#!/usr/bin/env python3
"""RSNR-V1C-PreHead: bounded benchmark-compatible factual unlearning.

V1B showed that an unbounded GA + abstention objective can crush the sensitive
answer and drive IDK arbitrarily high while simultaneously pushing the
CounterFact target_new down even faster on hard facts.  V1C keeps the exact
same rank-16 PreHead architecture but makes every behavioral objective bounded.

Training-safe requirements:
  1) logP(new) - logP(true) >= minimum_cf_margin
  2) logP(IDK) - logP(true) >= minimum_idk_margin
  3) logP_Base(true) - logP_Edit(true) >= minimum_true_logprob_drop
  4) gate-OFF logits are exactly Base

Additional anti-gaming constraint:
  |logP_Edit(new) - logP_Base(new)| <= max_target_new_logprob_drift

The CounterFact target_new is NOT trained as a replacement answer.  In the CF
margin its likelihood is detached.  A separate symmetric preservation hinge
only keeps target_new near its frozen-Base likelihood, preventing the adapter
from satisfying/failing the benchmark by moving the fake answer itself.

GA is bounded: it is active only on prompt instances that still fail the CF or
true-drop constraints.  Once a prompt satisfies those constraints its GA
contribution is exactly zero.  There is no unbounded abstention CE; IDK is
optimized only until the requested IDK-vs-true margin is met.

Official CounterFact paraphrases/neighborhood prompts are never used for
training or checkpoint selection.  Seed 1 is development-only.
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
import run_mcf_rsnr_v1b_prehead_standard_unlearn as v1b


PROTOCOL = "mcf_rsnr_v1c_prehead_bounded_unlearn"
VARIANT = "RSNR-V1C-PreHead-BoundedUnlearn"
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
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--case-batch-size", type=int, default=4)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--adapter-rank", type=int, default=16)
    p.add_argument("--adapter-alpha", type=float, default=16.0)

    p.add_argument("--bounded-ga-weight", type=float, default=0.10)
    p.add_argument("--cf-margin-weight", type=float, default=4.0)
    p.add_argument("--idk-margin-weight", type=float, default=1.0)
    p.add_argument("--drop-weight", type=float, default=1.0)
    p.add_argument("--target-new-preserve-weight", type=float, default=4.0)
    p.add_argument("--anchor-weight", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--hard-case-fraction", type=float, default=0.75)

    p.add_argument("--minimum-cf-margin", type=float, default=0.1)
    p.add_argument("--minimum-idk-margin", type=float, default=0.1)
    p.add_argument("--minimum-true-logprob-drop", type=float, default=2.0)
    p.add_argument("--max-target-new-logprob-drift", type=float, default=0.25)
    p.add_argument("--gate-off-logit-drift-max", type=float, default=0.0)

    args = p.parse_args(list(argv) if argv is not None else None)
    if args.seed != 1 or args.forget_num != 50:
        p.error("RSNR-V1C development run is locked to consumed seed1/forget50")
    if args.steps <= 0 or args.case_batch_size <= 0 or args.check_every <= 0:
        p.error("steps/batch/check must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        p.error("invalid optimizer configuration")
    if args.adapter_rank <= 0 or args.adapter_alpha <= 0:
        p.error("adapter rank/alpha must be positive")
    if not 0.0 <= float(args.hard_case_fraction) <= 1.0:
        p.error("hard-case-fraction must be in [0, 1]")
    for name in (
        "bounded_ga_weight", "cf_margin_weight", "idk_margin_weight", "drop_weight",
        "target_new_preserve_weight", "anchor_weight", "minimum_cf_margin",
        "minimum_idk_margin", "minimum_true_logprob_drop",
        "max_target_new_logprob_drift", "gate_off_logit_drift_max",
    ):
        if float(getattr(args, name)) < 0:
            p.error(f"{name} must be non-negative")
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


def _base_answer_logprobs_for_all_views(
    model: Any,
    hook: prehead.PreHeadNullHook,
    tokenizer: Any,
    forget: Sequence[Mapping[str, Any]],
    view_map: Mapping[int, Sequence[str]],
    *,
    answer_kind: str,
    device: torch.device,
    batch_cases: int,
) -> dict[int, list[float]]:
    if answer_kind not in {"true", "new"}:
        raise ValueError("answer_kind must be 'true' or 'new'")
    out: dict[int, list[float]] = {}
    with torch.no_grad():
        for start in range(0, len(forget), int(batch_cases)):
            cases = forget[start : start + int(batch_cases)]
            prompts, true_answers, new_answers, owners = v1b.prompts_true_new_for_cases(cases, view_map)
            answers = true_answers if answer_kind == "true" else new_answers
            values = rsnr.sequence_logprobs(
                model, hook, tokenizer, prompts, answers, device=device, gated=False
            )
            for local, row in enumerate(cases):
                out[int(row["case_id"])] = [
                    float(values[j].item())
                    for j, owner in enumerate(owners)
                    if owner == local
                ]
    return out


def _base_tensor_for_batch(
    cases: Sequence[Mapping[str, Any]],
    owners: Sequence[int],
    base_values: Mapping[int, Sequence[float]],
    *,
    device: torch.device,
) -> torch.Tensor:
    return v1b.base_true_tensor_for_batch(cases, owners, base_values, device=device)


def compute_bounded_objective_terms(
    true_lp: torch.Tensor,
    new_lp: torch.Tensor,
    idk_lp: torch.Tensor,
    base_true_lp: torch.Tensor,
    base_new_lp: torch.Tensor,
    owners: Sequence[int],
    case_count: int,
    *,
    minimum_cf_margin: float,
    minimum_idk_margin: float,
    minimum_true_drop: float,
    max_target_new_drift: float,
) -> dict[str, torch.Tensor]:
    if not (
        true_lp.shape == new_lp.shape == idk_lp.shape
        == base_true_lp.shape == base_new_lp.shape
    ):
        raise ValueError("all log-probability vectors must have identical shape")

    # Benchmark boundary: do not backpropagate through target_new here.
    cf_margin = new_lp.detach() - true_lp
    idk_margin = idk_lp - true_lp
    true_drop = base_true_lp.detach() - true_lp

    # Anti-gaming: target_new is allowed to move only inside a small symmetric
    # band around frozen Base.  This term can move it back toward Base, but never
    # rewards increasing it beyond Base.
    target_new_drift = new_lp - base_new_lp.detach()
    target_new_abs_drift = target_new_drift.abs()

    cf_hinge = F.relu(float(minimum_cf_margin) - cf_margin)
    idk_hinge = F.relu(float(minimum_idk_margin) - idk_margin)
    drop_hinge = F.relu(float(minimum_true_drop) - true_drop)
    new_preserve_hinge = F.relu(target_new_abs_drift - float(max_target_new_drift))

    cf_margin_loss = rsnr.worst_by_owner(cf_hinge, owners, case_count, maximum=True).mean()
    idk_margin_loss = rsnr.worst_by_owner(idk_hinge, owners, case_count, maximum=True).mean()
    drop_loss = rsnr.worst_by_owner(drop_hinge, owners, case_count, maximum=True).mean()
    target_new_preserve_loss = rsnr.worst_by_owner(
        new_preserve_hinge, owners, case_count, maximum=True
    ).mean()

    # Explicit GA, but bounded to instances still failing CF or true-drop.
    ga_active = ((cf_hinge > 0) | (drop_hinge > 0)).detach().to(true_lp.dtype)
    ga_count = ga_active.sum()
    if float(ga_count.detach().item()) > 0.0:
        bounded_ga_loss = (true_lp * ga_active).sum() / ga_count
    else:
        bounded_ga_loss = true_lp.sum() * 0.0

    return {
        "bounded_ga_loss": bounded_ga_loss,
        "cf_margin_loss": cf_margin_loss,
        "idk_margin_loss": idk_margin_loss,
        "drop_loss": drop_loss,
        "target_new_preserve_loss": target_new_preserve_loss,
        "cf_margin": cf_margin,
        "idk_margin": idk_margin,
        "true_drop": true_drop,
        "target_new_drift": target_new_drift,
        "target_new_abs_drift": target_new_abs_drift,
        "ga_active_count": ga_count,
    }


@torch.no_grad()
def evaluate_training_safe_conditions(
    model: Any,
    hook: prehead.PreHeadNullHook,
    tokenizer: Any,
    forget: Sequence[Mapping[str, Any]],
    view_map: Mapping[int, Sequence[str]],
    base_true: Mapping[int, Sequence[float]],
    base_new: Mapping[int, Sequence[float]],
    *,
    device: torch.device,
    batch_cases: int,
    minimum_cf_margin: float,
    minimum_idk_margin: float,
    minimum_true_drop: float,
    max_target_new_drift: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(forget), int(batch_cases)):
        cases = forget[start : start + int(batch_cases)]
        prompts, true_answers, new_answers, owners = v1b.prompts_true_new_for_cases(cases, view_map)
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
        base_true_tensor = _base_tensor_for_batch(cases, owners, base_true, device=device)
        base_new_tensor = _base_tensor_for_batch(cases, owners, base_new, device=device)
        cf = new_lp - true_lp
        idk = idk_lp - true_lp
        drop = base_true_tensor - true_lp
        new_abs_drift = (new_lp - base_new_tensor).abs()
        for local, row in enumerate(cases):
            idxs = [j for j, owner in enumerate(owners) if owner == local]
            worst_cf = min(float(cf[j].item()) for j in idxs)
            worst_idk = min(float(idk[j].item()) for j in idxs)
            worst_drop = min(float(drop[j].item()) for j in idxs)
            worst_new_drift = max(float(new_abs_drift[j].item()) for j in idxs)
            cf_pass = worst_cf >= float(minimum_cf_margin)
            idk_pass = worst_idk >= float(minimum_idk_margin)
            drop_pass = worst_drop >= float(minimum_true_drop)
            new_preserve_pass = worst_new_drift <= float(max_target_new_drift)
            rows.append({
                "case_id": int(row["case_id"]),
                "subject": str(row["requested_rewrite"]["subject"]),
                "relation_id": str(row["requested_rewrite"]["relation_id"]),
                "worst_cf_margin": worst_cf,
                "worst_idk_margin": worst_idk,
                "worst_true_drop": worst_drop,
                "worst_abs_target_new_logprob_drift": worst_new_drift,
                "cf_pass": cf_pass,
                "idk_pass": idk_pass,
                "drop_pass": drop_pass,
                "new_preserve_pass": new_preserve_pass,
                "all_four_pass": bool(cf_pass and idk_pass and drop_pass),
                "safe_five_pass": bool(cf_pass and idk_pass and drop_pass and new_preserve_pass),
            })
    return {
        "count": len(rows),
        "all_four_passed": sum(bool(r["all_four_pass"]) for r in rows),
        "all_four_failures": sum(not bool(r["all_four_pass"]) for r in rows),
        "safe_five_passed": sum(bool(r["safe_five_pass"]) for r in rows),
        "safe_five_failures": sum(not bool(r["safe_five_pass"]) for r in rows),
        "cf_passed": sum(bool(r["cf_pass"]) for r in rows),
        "idk_passed": sum(bool(r["idk_pass"]) for r in rows),
        "drop_passed": sum(bool(r["drop_pass"]) for r in rows),
        "new_preserve_passed": sum(bool(r["new_preserve_pass"]) for r in rows),
        "minimum_worst_cf_margin": min(float(r["worst_cf_margin"]) for r in rows),
        "minimum_worst_idk_margin": min(float(r["worst_idk_margin"]) for r in rows),
        "minimum_worst_true_drop": min(float(r["worst_true_drop"]) for r in rows),
        "maximum_worst_abs_target_new_logprob_drift": max(
            float(r["worst_abs_target_new_logprob_drift"]) for r in rows
        ),
        "thresholds": {
            "minimum_cf_margin": float(minimum_cf_margin),
            "minimum_idk_margin": float(minimum_idk_margin),
            "minimum_true_drop": float(minimum_true_drop),
            "max_target_new_logprob_drift": float(max_target_new_drift),
        },
        "per_case": rows,
    }


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


def _sample_cases(
    rng: random.Random,
    forget: Sequence[Mapping[str, Any]],
    hard_case_ids: set[int],
    batch_size: int,
    hard_fraction: float,
) -> list[Mapping[str, Any]]:
    n = min(int(batch_size), len(forget))
    hard = [row for row in forget if int(row["case_id"]) in hard_case_ids]
    take_hard = min(len(hard), int(round(n * float(hard_fraction))))
    chosen = rng.sample(hard, take_hard) if take_hard else []
    chosen_ids = {int(row["case_id"]) for row in chosen}
    pool = [row for row in forget if int(row["case_id"]) not in chosen_ids]
    need = n - len(chosen)
    if need:
        chosen.extend(rng.sample(pool, need))
    rng.shuffle(chosen)
    return chosen


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
        raise RuntimeError("CUDA is required for RSNR-V1C PreHead training")
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
        "bounded_ga": True,
        "unbounded_abstention_ce": False,
        "target_new_role": "detached CF reference plus symmetric Base-preservation constraint",
        "target_new_positive_likelihood_training": False,
        "views_per_case": int(view_meta["views_per_case"]),
        "heldout_probe_text_used": False,
    }, indent=2), flush=True)

    contexts = _build_gate_off_contexts(forget, protection_fit)
    equivalence = rsnr.gate_off_equivalence(model, hook, tokenizer, contexts, device=device)
    if equivalence["max_abs_logit_drift"] > float(args.gate_off_logit_drift_max):
        raise RuntimeError("initial PreHead gate-off path is not Base-identical")

    base_true = _base_answer_logprobs_for_all_views(
        model, hook, tokenizer, forget, view_map,
        answer_kind="true", device=device, batch_cases=int(args.case_batch_size),
    )
    base_new = _base_answer_logprobs_for_all_views(
        model, hook, tokenizer, forget, view_map,
        answer_kind="new", device=device, batch_cases=int(args.case_batch_size),
    )

    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    rng = random.Random(int(args.seed) + 44017)
    best_state = copy.deepcopy(adapter.state_dict())
    best_key = (10**9, 10**9, float("inf"), float("inf"), float("inf"), float("inf"))
    training_log: list[dict[str, Any]] = []
    hard_case_ids = {int(row["case_id"]) for row in forget}

    for step in range(1, int(args.steps) + 1):
        cases = _sample_cases(
            rng, forget, hard_case_ids, int(args.case_batch_size), float(args.hard_case_fraction)
        )
        prompts, true_answers, new_answers, owners = v1b.prompts_true_new_for_cases(cases, view_map)
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
        base_true_tensor = _base_tensor_for_batch(cases, owners, base_true, device=device)
        base_new_tensor = _base_tensor_for_batch(cases, owners, base_new, device=device)

        terms = compute_bounded_objective_terms(
            true_lp, new_lp, idk_lp, base_true_tensor, base_new_tensor,
            owners, len(cases),
            minimum_cf_margin=float(args.minimum_cf_margin),
            minimum_idk_margin=float(args.minimum_idk_margin),
            minimum_true_drop=float(args.minimum_true_logprob_drop),
            max_target_new_drift=float(args.max_target_new_logprob_drift),
        )
        anchor = adapter.up.weight.float().pow(2).mean()
        loss = (
            float(args.bounded_ga_weight) * terms["bounded_ga_loss"]
            + float(args.cf_margin_weight) * terms["cf_margin_loss"]
            + float(args.idk_margin_weight) * terms["idk_margin_loss"]
            + float(args.drop_weight) * terms["drop_loss"]
            + float(args.target_new_preserve_weight) * terms["target_new_preserve_loss"]
            + float(args.anchor_weight) * anchor
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(adapter.parameters()), float(args.grad_clip))
        optimizer.step()

        if step == 1 or step % int(args.check_every) == 0 or step == int(args.steps):
            metrics = evaluate_training_safe_conditions(
                model, hook, tokenizer, forget, view_map, base_true, base_new,
                device=device,
                batch_cases=int(args.case_batch_size),
                minimum_cf_margin=float(args.minimum_cf_margin),
                minimum_idk_margin=float(args.minimum_idk_margin),
                minimum_true_drop=float(args.minimum_true_logprob_drop),
                max_target_new_drift=float(args.max_target_new_logprob_drift),
            )
            eq = rsnr.gate_off_equivalence(model, hook, tokenizer, contexts[:32], device=device)
            hard_case_ids = {
                int(row["case_id"])
                for row in metrics["per_case"]
                if not bool(row["cf_pass"])
            }
            key = (
                int(metrics["safe_five_failures"]),
                len(forget) - int(metrics["cf_passed"]),
                -float(metrics["minimum_worst_cf_margin"]),
                float(metrics["maximum_worst_abs_target_new_logprob_drift"]),
                -float(metrics["minimum_worst_idk_margin"]),
                -float(metrics["minimum_worst_true_drop"]),
            )
            if key < best_key:
                best_key = key
                best_state = copy.deepcopy(adapter.state_dict())
            row = {
                "step": int(step),
                "loss": float(loss.detach().item()),
                "bounded_ga_loss": float(terms["bounded_ga_loss"].detach().item()),
                "cf_margin_loss": float(terms["cf_margin_loss"].detach().item()),
                "idk_margin_loss": float(terms["idk_margin_loss"].detach().item()),
                "drop_loss": float(terms["drop_loss"].detach().item()),
                "target_new_preserve_loss": float(terms["target_new_preserve_loss"].detach().item()),
                "ga_active_count": int(terms["ga_active_count"].detach().item()),
                "all_four_passed": int(metrics["all_four_passed"]),
                "safe_five_passed": int(metrics["safe_five_passed"]),
                "cf_passed": int(metrics["cf_passed"]),
                "idk_passed": int(metrics["idk_passed"]),
                "drop_passed": int(metrics["drop_passed"]),
                "new_preserve_passed": int(metrics["new_preserve_passed"]),
                "minimum_worst_cf_margin": float(metrics["minimum_worst_cf_margin"]),
                "minimum_worst_idk_margin": float(metrics["minimum_worst_idk_margin"]),
                "minimum_worst_true_drop": float(metrics["minimum_worst_true_drop"]),
                "maximum_worst_abs_target_new_logprob_drift": float(
                    metrics["maximum_worst_abs_target_new_logprob_drift"]
                ),
                "hard_cf_cases": len(hard_case_ids),
                "gate_off_max_abs_logit_drift": float(eq["max_abs_logit_drift"]),
                **rsnr.adapter_norm(adapter),
            }
            training_log.append(row)
            print(
                f"step {step:4d}: ALL4={metrics['all_four_passed']}/50 "
                f"SAFE5={metrics['safe_five_passed']}/50; "
                f"CF={metrics['cf_passed']}/50 IDK={metrics['idk_passed']}/50 "
                f"DROP={metrics['drop_passed']}/50 NEWP={metrics['new_preserve_passed']}/50; "
                f"worst CF={metrics['minimum_worst_cf_margin']:.4f}, "
                f"IDK={metrics['minimum_worst_idk_margin']:.4f}, "
                f"drop={metrics['minimum_worst_true_drop']:.4f}, "
                f"max|new-base|={metrics['maximum_worst_abs_target_new_logprob_drift']:.4f}, "
                f"hardCF={len(hard_case_ids)}, off-drift={eq['max_abs_logit_drift']:.3g}",
                flush=True,
            )
            if (
                int(metrics["safe_five_failures"]) == 0
                and float(eq["max_abs_logit_drift"]) <= float(args.gate_off_logit_drift_max)
            ):
                print("all 50 cases pass the four requested conditions plus target_new preservation", flush=True)
                break

    adapter.load_state_dict(best_state)
    final_metrics = evaluate_training_safe_conditions(
        model, hook, tokenizer, forget, view_map, base_true, base_new,
        device=device,
        batch_cases=int(args.case_batch_size),
        minimum_cf_margin=float(args.minimum_cf_margin),
        minimum_idk_margin=float(args.minimum_idk_margin),
        minimum_true_drop=float(args.minimum_true_logprob_drop),
        max_target_new_drift=float(args.max_target_new_logprob_drift),
    )
    final_equivalence = rsnr.gate_off_equivalence(model, hook, tokenizer, contexts, device=device)
    exact_locality_pass = (
        float(final_equivalence["max_abs_logit_drift"])
        <= float(args.gate_off_logit_drift_max)
    )
    if not exact_locality_pass:
        raise RuntimeError("final PreHead gate-off Base equivalence failed")

    training_gate_passed = int(final_metrics["safe_five_failures"]) == 0 and exact_locality_pass
    membership = _membership(forget)
    artifact_path = method_dir / "rsnr_prehead_bounded_unlearn_adapter.pt"
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
        "target_new_role": "detached_cf_reference_plus_base_preservation",
        "target_new_positive_likelihood_training": False,
    }, artifact_path)

    sidecar = {
        "protocol": PROTOCOL,
        "variant": VARIANT,
        "routing": "oracle_exact_subject_relation_membership",
        "intervention_site": INTERVENTION_SITE,
        "atomic_query_scope": True,
        "non_target_behavior": "adapter_off_exact_base_path",
        "sensitive_behavior": "activate_pre_lm_head_bounded_unlearn_adapter",
        "abstention_text": ABSTENTION,
        "target_new_used": True,
        "target_new_role": "detached CF reference plus symmetric Base-preservation constraint",
        "target_new_positive_likelihood_training": False,
        "forget_membership": membership,
    }
    (method_dir / "relation_scoped_prehead_bounded_unlearn.json").write_text(
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
            "bounded_ga_weight": float(args.bounded_ga_weight),
            "cf_reference_margin_weight": float(args.cf_margin_weight),
            "idk_margin_weight": float(args.idk_margin_weight),
            "base_true_drop_weight": float(args.drop_weight),
            "target_new_preserve_weight": float(args.target_new_preserve_weight),
            "adapter_anchor_weight": float(args.anchor_weight),
            "minimum_cf_margin": float(args.minimum_cf_margin),
            "minimum_idk_margin": float(args.minimum_idk_margin),
            "minimum_true_logprob_drop": float(args.minimum_true_logprob_drop),
            "max_target_new_logprob_drift": float(args.max_target_new_logprob_drift),
            "target_new_positive_likelihood_training": False,
            "all_losses_bounded_after_constraint_satisfaction": True,
            "worst_of_5_training_views": True,
            "hard_cf_case_sampling_fraction": float(args.hard_case_fraction),
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
            "target_new_preserved_near_base": True,
            "lm_head_weights_changed": False,
            "transformer_weights_changed": False,
        },
    }
    (method_dir / "rsnr_v1c_prehead_bounded_unlearn.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    completion = {
        "protocol": PROTOCOL,
        "variant": VARIANT,
        "training_gate_passed": bool(training_gate_passed),
        "all_four_conditions_passed": int(final_metrics["all_four_failures"]) == 0,
        "safe_five_conditions_passed": int(final_metrics["safe_five_failures"]) == 0,
        "all_four_passed": int(final_metrics["all_four_passed"]),
        "safe_five_passed": int(final_metrics["safe_five_passed"]),
        "cf_passed": int(final_metrics["cf_passed"]),
        "idk_passed": int(final_metrics["idk_passed"]),
        "drop_passed": int(final_metrics["drop_passed"]),
        "new_preserve_passed": int(final_metrics["new_preserve_passed"]),
        "minimum_worst_cf_margin": float(final_metrics["minimum_worst_cf_margin"]),
        "minimum_worst_idk_margin": float(final_metrics["minimum_worst_idk_margin"]),
        "minimum_worst_true_drop": float(final_metrics["minimum_worst_true_drop"]),
        "maximum_worst_abs_target_new_logprob_drift": float(
            final_metrics["maximum_worst_abs_target_new_logprob_drift"]
        ),
        "gate_off_max_abs_logit_drift": float(final_equivalence["max_abs_logit_drift"]),
        "adapter_saved": True,
        "adapter_path": str(artifact_path),
        "trainable_parameters": trainable,
        "base_weights_modified": False,
        "transformer_weights_modified": False,
        "lm_head_weights_modified": False,
        "target_new_positive_likelihood_training": False,
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
