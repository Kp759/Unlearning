#!/usr/bin/env python3
"""Unknown-neutral frozen-head representation repair for target_true-sensitive MCF.

This stage implements the answer-level objective

    m_W0(q) = NLL_W0(q, a_sensitive) - NLL_W0(q, Unknown)

and minimizes

    L_R = lambda_f * ReLU(margin - m_W0(q))^2
        + lambda_u * KL(pre-repair || current) on external Wikipedia
        + lambda_delta * ||Delta Phi_R||_F^2.

The decoder head W0 is frozen.  Only the final transformer decoder block is
trainable, so any change in m_W0 comes from the representation h'(q), not from
editing the decoder.  requested_rewrite.target_true is the sensitive answer.
requested_rewrite.target_new is NEVER used as a training target in this stage;
it is consulted only for direct benchmark diagnostics before/after training.

No official paraphrases, neighborhoods, benchmark-retain records, or PPL text
are visible to training or checkpoint selection.  External Wikipedia utility
contexts exclude the first 20 rows used by the fixed PPL evaluation prefix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

import gagd_compare as gagd
import gagd_active_case_repair as mcf_repair
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import sure_stage1_gagd_w1k as wikipedia_utility
import sure_stage2_sparse_repair as stage2
import mcf_frozen_head_representation_repair as legacy_rep


METHOD = "SURE-LM-MCF-frozen-head-Unknown-representation-repair"
PROTOCOL = "mcf_target_true_sensitive_frozen_head_unknown_last_block_w200_v1"
DEFAULT_NEUTRAL_ANSWER = "Unknown"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--neutral-answer", default=DEFAULT_NEUTRAL_ANSWER)
    p.add_argument(
        "--repair-scope",
        choices=("active", "all"),
        default="active",
        help=(
            "active trains only records whose sensitive-vs-neutral frozen-head "
            "margin is below --forget-margin; all trains all direct forget records"
        ),
    )

    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--check-every", type=int, default=25)

    p.add_argument("--forget-margin", type=float, default=0.05)
    p.add_argument("--forget-weight", type=float, default=1.0)
    p.add_argument("--utility-kl-weight", type=float, default=2.0)
    p.add_argument(
        "--delta-weight",
        type=float,
        default=1e-8,
        help="Weight on literal ||Delta Phi_R||_F^2 for the final decoder block",
    )

    p.add_argument("--utility-wikipedia-dir", required=True)
    p.add_argument("--utility-sample-size", type=int, default=200)
    p.add_argument("--utility-batch-size", type=int, default=4)
    p.add_argument("--utility-cache-batch-size", type=int, default=8)
    p.add_argument("--utility-max-length", type=int, default=128)
    p.add_argument("--utility-seed", type=int, default=1)
    p.add_argument("--utility-exclude-first", type=int, default=20)

    p.add_argument("--benchmark-pair-margin", type=float, default=0.05)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")

    a = p.parse_args(list(argv) if argv is not None else None)
    positive = (
        a.forget_num,
        a.steps,
        a.batch_size,
        a.lr,
        a.check_every,
        a.forget_weight,
        a.utility_kl_weight,
        a.utility_sample_size,
        a.utility_batch_size,
        a.utility_cache_batch_size,
        a.utility_max_length,
    )
    if any(float(v) <= 0 for v in positive):
        p.error("counts, LR, forget weight, and utility weight must be positive")
    nonnegative = (
        a.grad_clip,
        a.forget_margin,
        a.delta_weight,
        a.utility_exclude_first,
        a.benchmark_pair_margin,
    )
    if any(float(v) < 0 for v in nonnegative):
        p.error("margins, clipping, delta weight, and exclusion must be non-negative")
    if not str(a.neutral_answer).strip():
        p.error("--neutral-answer must be non-empty")
    if a.utility_exclude_first < 20:
        p.error("utility-exclude-first must be at least 20 to protect the fixed PPL prefix")
    if a.utility_max_length < 8:
        p.error("utility-max-length must be at least 8")
    return a


def build_unknown_instances(
    records: Sequence[Mapping[str, Any]], neutral_answer: str
) -> List[mcf_repair.MCFPromptInstance]:
    """Build direct instances with Unknown in the reference slot.

    target_new from the raw MCF record is deliberately not copied here.  The
    first answer slot is the explicit neutral answer and the second is the
    sensitive target_true answer, matching official_batch_components ordering.
    """
    neutral = str(neutral_answer).strip()
    if not neutral:
        raise ValueError("neutral answer must be non-empty")
    instances: List[mcf_repair.MCFPromptInstance] = []
    for position, record in enumerate(records):
        rr = record["requested_rewrite"]
        subject = str(rr["subject"])
        sensitive = str(rr["target_true"]["str"])
        instances.append(
            mcf_repair.MCFPromptInstance(
                record_index=int(record.get("case_id", position)),
                sampled_position=int(position),
                prompt_type="rewrite",
                prompt_index=0,
                prompt=str(rr["prompt"]).format(subject),
                target_new=neutral,
                target_true=sensitive,
            )
        )
    return instances


def unknown_margin_loss(
    sensitive_nll: torch.Tensor,
    neutral_nll: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared hinge; positive margin means sensitive is worse than Unknown."""
    if sensitive_nll.shape != neutral_nll.shape:
        raise ValueError("sensitive and neutral NLL tensors must match")
    margins = sensitive_nll.float() - neutral_nll.float()
    loss = F.relu(float(margin) - margins).square().mean()
    return loss, margins


def _unknown_forward(
    model: nn.Module,
    tok: Any,
    instances: Sequence[mcf_repair.MCFPromptInstance],
    device: torch.device,
    llama_like: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable official-compatible Unknown/sensitive sequence NLLs."""
    encoded, target_token_ids, prefix_lens = mcf_repair.official_batch_components(
        tok, instances, device, llama_like
    )
    logits = model(**encoded, use_cache=False).logits
    if llama_like:
        logits = logits[:, 1:, :]

    losses: List[torch.Tensor] = []
    for row, (target_ids, prefix_len) in enumerate(zip(target_token_ids, prefix_lens)):
        token_losses: List[torch.Tensor] = []
        for offset, target_id in enumerate(target_ids):
            position = int(prefix_len) + int(offset) - 1
            token_losses.append(
                -F.log_softmax(logits[row, position, :].float(), dim=0)[int(target_id)]
            )
        losses.append(torch.stack(token_losses).mean())
    paired = torch.stack(losses).reshape(len(instances), 2)
    # target_new slot == neutral answer; target_true slot == sensitive answer.
    return paired[:, 0], paired[:, 1]


@torch.no_grad()
def evaluate_unknown_diagnostics(
    model: nn.Module,
    tok: Any,
    instances: Sequence[mcf_repair.MCFPromptInstance],
    device: torch.device,
    llama_like: bool,
    batch_size: int,
    required_margin: float,
) -> Dict[str, Any]:
    model.eval()
    chunks: List[torch.Tensor] = []
    for start in range(0, len(instances), int(batch_size)):
        neutral_nll, sensitive_nll = _unknown_forward(
            model, tok, instances[start : start + int(batch_size)], device, llama_like
        )
        chunks.append((sensitive_nll - neutral_nll).detach().float().cpu())
    margins = torch.cat(chunks) if chunks else torch.empty(0)
    return {
        "direct_record_count": int(len(instances)),
        "neutral_answer": instances[0].target_new if instances else None,
        "required_margin": float(required_margin),
        "failures": int((margins < float(required_margin)).sum().item()),
        "successes": int((margins >= float(required_margin)).sum().item()),
        "minimum_margin": float(margins.min()) if margins.numel() else None,
        "mean_margin": float(margins.mean()) if margins.numel() else None,
        "maximum_margin": float(margins.max()) if margins.numel() else None,
        "margins": [float(x) for x in margins.tolist()],
    }


@torch.no_grad()
def evaluate_benchmark_pair_diagnostics(
    model: nn.Module,
    tok: Any,
    benchmark_instances: Sequence[mcf_repair.MCFPromptInstance],
    device: torch.device,
    llama_like: bool,
    batch_size: int,
    required_margin: float,
) -> Dict[str, Any]:
    """Diagnostic only: target_true-vs-target_new; never used by the loss."""
    margins = stage2.mcf_direct_margins(
        model,
        tok,
        benchmark_instances,
        device,
        llama_like,
        int(batch_size),
        "target_true",
        "target_new",
    ).detach().float().cpu()
    return {
        "direct_record_count": int(len(benchmark_instances)),
        "required_margin": float(required_margin),
        "failures": int((margins < float(required_margin)).sum().item()),
        "minimum_margin": float(margins.min()) if margins.numel() else None,
        "mean_margin": float(margins.mean()) if margins.numel() else None,
    }


def _optimizer(parameters: Iterable[nn.Parameter], kind: str, lr: float):
    params = list(parameters)
    if kind == "adam":
        return torch.optim.Adam(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(int(a.seed))
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, manifest = stage2.load_locked(
        "mcf", visible_path, manifest_path, int(a.seed), int(a.forget_num)
    )
    legacy_rep.assert_target_contract(manifest)
    legacy_rep.validate_direct_only_records(records)

    ns = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)

    unknown_instances = build_unknown_instances(records, a.neutral_answer)
    benchmark_instances = stage2.mcf_instances(records)
    model.eval()

    unknown_before = evaluate_unknown_diagnostics(
        model, tok, unknown_instances, device, llama_like, a.batch_size, a.forget_margin
    )
    benchmark_before = evaluate_benchmark_pair_diagnostics(
        model,
        tok,
        benchmark_instances,
        device,
        llama_like,
        a.batch_size,
        a.benchmark_pair_margin,
    )

    initial_unknown_margins = torch.tensor(
        unknown_before["margins"], dtype=torch.float32
    )
    active_positions = [
        i
        for i, value in enumerate(initial_unknown_margins.tolist())
        if float(value) < float(a.forget_margin)
    ]
    train_positions = (
        active_positions if a.repair_scope == "active" else list(range(len(records)))
    )
    if not train_positions:
        raise RuntimeError(
            "No records selected: the source checkpoint already satisfies the "
            "configured sensitive-vs-Unknown margin"
        )
    train_instances = [unknown_instances[i] for i in train_positions]

    utility_prompts, utility_receipt = wikipedia_utility.build_utility_prompts(
        tok,
        Path(a.utility_wikipedia_dir).resolve(),
        sample_size=int(a.utility_sample_size),
        seed=int(a.utility_seed),
        exclude_first=int(a.utility_exclude_first),
        max_length=int(a.utility_max_length),
    )
    print(
        f"Caching pre-repair logits for {len(utility_prompts)} external Wikipedia utility contexts...",
        flush=True,
    )
    utility_base_logits = wikipedia_utility.cache_utility_base_logits(
        model,
        tok,
        utility_prompts,
        device,
        int(a.utility_cache_batch_size),
    )

    last_block, trainable_summary = legacy_rep.configure_last_block_only(model)
    trainable_params = [p for p in last_block.parameters() if p.requires_grad]
    initial_params = [p.detach().clone() for p in trainable_params]
    opt = _optimizer(trainable_params, a.optimizer, a.lr)
    forget_sampler = core.IndexSampler(
        len(train_instances), int(a.batch_size), int(a.seed) + 41001
    )
    utility_sampler = core.IndexSampler(
        len(utility_prompts), int(a.utility_batch_size), int(a.utility_seed) + 41003
    )

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    core.write_json(out_dir / "utility_receipt.json", utility_receipt)
    core.write_json(out_dir / "unknown_before.json", unknown_before)
    core.write_json(out_dir / "benchmark_pair_before.json", benchmark_before)

    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in tqdm(
            range(1, int(a.steps) + 1),
            desc="MCF frozen-head Unknown representation repair",
        ):
            forget_idx = forget_sampler.next()
            batch = [train_instances[i] for i in forget_idx]
            utility_idx = utility_sampler.next()
            utility_batch = [utility_prompts[i] for i in utility_idx]

            opt.zero_grad(set_to_none=True)
            neutral_nll, sensitive_nll = _unknown_forward(
                model, tok, batch, device, llama_like
            )
            forget_loss, forget_margins = unknown_margin_loss(
                sensitive_nll, neutral_nll, a.forget_margin
            )
            utility_logits = wikipedia_utility._forward_prompt_logits(
                model, tok, utility_batch, device
            )
            utility_loss = wikipedia_utility.utility_kl(
                utility_logits, utility_base_logits[utility_idx]
            )
            delta_f2 = legacy_rep.parameter_delta_f2(trainable_params, initial_params)
            total = (
                float(a.forget_weight) * forget_loss
                + float(a.utility_kl_weight) * utility_loss
                + float(a.delta_weight) * delta_f2
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite Unknown representation loss at step {step}")
            total.backward()
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(trainable_params, float(a.grad_clip))
                if a.grad_clip > 0
                else None
            )
            if grad_norm is not None and not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at step {step}")
            opt.step()

            if step == 1 or step % int(a.check_every) == 0 or step == int(a.steps):
                row = {
                    "step": int(step),
                    "total_loss": float(total.detach().cpu()),
                    "unknown_forget_loss": float(forget_loss.detach().cpu()),
                    "utility_kl_loss": float(utility_loss.detach().cpu()),
                    "delta_phi_frobenius_squared": float(delta_f2.detach().cpu()),
                    "delta_phi_frobenius_norm": float(delta_f2.detach().sqrt().cpu()),
                    "batch_unknown_min_margin": float(forget_margins.min().detach().cpu()),
                    "batch_unknown_mean_margin": float(forget_margins.mean().detach().cpu()),
                    "neutral_answer": str(a.neutral_answer),
                    "forget_weight": float(a.forget_weight),
                    "utility_kl_weight": float(a.utility_kl_weight),
                    "delta_weight": float(a.delta_weight),
                    "target_new_used_as_training_target": False,
                    "benchmark_retain_seen": 0,
                    "heldout_paraphrases_seen": 0,
                    "locality_or_neighborhood_seen": 0,
                    "PPL_seen": False,
                }
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()

    del opt
    model.eval()
    unknown_after = evaluate_unknown_diagnostics(
        model, tok, unknown_instances, device, llama_like, a.batch_size, a.forget_margin
    )
    benchmark_after = evaluate_benchmark_pair_diagnostics(
        model,
        tok,
        benchmark_instances,
        device,
        llama_like,
        a.batch_size,
        a.benchmark_pair_margin,
    )
    utility_post = wikipedia_utility.evaluate_utility_kl(
        model,
        tok,
        utility_prompts,
        utility_base_logits,
        device,
        int(a.utility_cache_batch_size),
    )
    with torch.no_grad():
        final_delta_f2 = legacy_rep.parameter_delta_f2(trainable_params, initial_params)

    if model.get_output_embeddings().weight.requires_grad:
        raise RuntimeError("frozen decoder W0 became trainable")
    if model.get_input_embeddings().weight.requires_grad:
        raise RuntimeError("input embeddings became trainable")

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    receipt = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "source_model_path": str(Path(a.model_path).resolve()),
        "training_visible_path": str(visible_path),
        "split_manifest": str(manifest_path),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "target_contract": {
            "sensitive_unwanted": "requested_rewrite.target_true",
            "neutral_training_answer": str(a.neutral_answer),
            "requested_rewrite.target_new_used_as_training_target": False,
            "field_swapping": False,
        },
        "objective": (
            "lambda_f*ReLU(forget_margin-(NLL_W0(target_true)-NLL_W0(Unknown)))^2 "
            "+ lambda_u*KL(source||current)_Wikipedia + lambda_delta*||DeltaPhi_R||_F^2"
        ),
        "repair_scope": str(a.repair_scope),
        "initial_active_positions": active_positions,
        "initial_active_count": int(len(active_positions)),
        "training_positions": train_positions,
        "training_record_count": int(len(train_positions)),
        "neutral_answer": str(a.neutral_answer),
        "forget_margin": float(a.forget_margin),
        "forget_weight": float(a.forget_weight),
        "utility_kl_weight": float(a.utility_kl_weight),
        "delta_weight": float(a.delta_weight),
        "benchmark_pair_margin_diagnostic_only": float(a.benchmark_pair_margin),
        "target_new_used_in_loss": False,
        "teacher_forcing": True,
        "frozen_head_definition": "unchanged lm_head W0 from --model-path",
        "trainable_architecture": trainable_summary,
        "steps": int(a.steps),
        "batch_size": int(a.batch_size),
        "lr": float(a.lr),
        "optimizer": str(a.optimizer),
        "gradient_clip": float(a.grad_clip),
        "utility_sample_size": int(a.utility_sample_size),
        "utility_exclude_first": int(a.utility_exclude_first),
        "unknown_before": unknown_before,
        "unknown_after": unknown_after,
        "benchmark_pair_before": benchmark_before,
        "benchmark_pair_after": benchmark_after,
        "utility_post_kl": utility_post,
        "delta_phi_frobenius_squared": float(final_delta_f2.detach().cpu()),
        "delta_phi_frobenius_norm": float(final_delta_f2.detach().sqrt().cpu()),
        "benchmark_retain_seen": 0,
        "heldout_paraphrases_seen": 0,
        "locality_or_neighborhood_seen": 0,
        "PPL_seen": False,
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "unknown_representation_summary.json", receipt)
    core.write_json(out_dir / "unknown_after.json", unknown_after)
    core.write_json(out_dir / "benchmark_pair_after.json", benchmark_after)
    core.write_json(out_dir / "utility_post_kl.json", utility_post)

    print("Unknown-neutral representation checkpoint:", ckpt)
    print("Objective margin: NLL(target_true sensitive) - NLL(Unknown)")
    print("Neutral answer:", str(a.neutral_answer))
    print("target_new used as training target: False")
    print("Trainable architecture:", trainable_summary)
    print("Unknown margin before:", unknown_before)
    print("Unknown margin after:", unknown_after)
    print("Benchmark target_true-vs-target_new diagnostic before:", benchmark_before)
    print("Benchmark target_true-vs-target_new diagnostic after:", benchmark_after)
    print("Wikipedia utility post KL:", utility_post)
    print("Delta Phi_R Frobenius norm:", float(final_delta_f2.detach().sqrt().cpu()))
    print("Stage 2 was NOT run; official evaluation is held out from this stage.")


if __name__ == "__main__":
    main()
