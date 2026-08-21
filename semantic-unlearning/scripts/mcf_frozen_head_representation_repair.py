#!/usr/bin/env python3
"""Frozen-head representation repair for target_true-sensitive MCF.

This is an isolated SURE-LM representation-level ablation that starts from an
already-trained Stage-1 checkpoint and asks whether the sensitive fact remains
linearly recoverable by the *unchanged* decoder head.

Target contract (hard enforced):

* requested_rewrite.target_true = sensitive / unwanted;
* requested_rewrite.target_new  = non-sensitive reference;
* fields are never swapped.

Only the final transformer decoder block is trainable.  Input embeddings, the
LM head W0, every earlier transformer block, and the final normalization remain
frozen.  Therefore all forget logits during this stage are exactly W0 h'(x):
changes in sensitive-token recoverability must come from the representation
h'(x), not from an edited decoder.

The pilot objective intentionally omits a separate subspace regularizer so the
contribution of the frozen-head criterion is identifiable:

    L_R = lambda_f L_frozen_head
        + lambda_p L_pair
        + lambda_u L_utility
        + lambda_delta ||Delta Phi_R||_F^2.

For each teacher-forced sensitive target_true token y_t,

    L_frozen_head = mean ReLU(z_y - max_{j != y} z_j + m_f)^2,
    z = W0 h'(x, y_<t).

The MCF direct pairwise contract is also enforced through the same frozen head:

    margin = NLL(target_true) - NLL(target_new),
    L_pair = mean ReLU(m_pair - margin)^2.

External Wikipedia utility contexts are drawn deterministically using the same
helper as the W1K/W200 Stage-1 runs.  Their pre-repair next-token distributions
from the input checkpoint are cached and preserved by KL(reference || current).
The first 20 Wikipedia rows remain excluded from training visibility.

No official paraphrases, neighborhoods, benchmark-retain records, or PPL text
are used for training or checkpoint selection.  The script stops after this
representation stage; ordinary Stage 2, if desired, must be run separately.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

import gagd_compare as gagd
import gagd_active_case_repair as mcf_repair
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_stage1_gagd_w1k as wikipedia_utility
import sure_stage2_sparse_repair as stage2
import sure_canonical_core as core


METHOD = "SURE-LM-MCF-frozen-head-representation-repair"
PROTOCOL = "mcf_target_true_sensitive_frozen_head_last_block_w200_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument(
        "--repair-scope",
        choices=("active", "all"),
        default="active",
        help=(
            "active trains only direct records whose target_true-vs-target_new "
            "margin is below --pair-margin at the input checkpoint; all trains "
            "all direct training-visible forget records"
        ),
    )

    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--check-every", type=int, default=25)

    p.add_argument("--frozen-head-margin", type=float, default=0.25)
    p.add_argument("--pair-margin", type=float, default=0.05)
    p.add_argument("--frozen-head-weight", type=float, default=1.0)
    p.add_argument("--pair-weight", type=float, default=1.0)
    p.add_argument("--utility-kl-weight", type=float, default=2.0)
    p.add_argument(
        "--delta-weight",
        type=float,
        default=1e-8,
        help="Weight on the literal Frobenius-squared last-block parameter delta",
    )

    p.add_argument("--utility-wikipedia-dir", required=True)
    p.add_argument("--utility-sample-size", type=int, default=200)
    p.add_argument("--utility-batch-size", type=int, default=4)
    p.add_argument("--utility-cache-batch-size", type=int, default=8)
    p.add_argument("--utility-max-length", type=int, default=128)
    p.add_argument("--utility-seed", type=int, default=1)
    p.add_argument("--utility-exclude-first", type=int, default=20)

    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")

    a = p.parse_args(list(argv) if argv is not None else None)
    positive = (
        a.forget_num,
        a.steps,
        a.batch_size,
        a.lr,
        a.check_every,
        a.frozen_head_weight,
        a.pair_weight,
        a.utility_kl_weight,
        a.utility_sample_size,
        a.utility_batch_size,
        a.utility_cache_batch_size,
        a.utility_max_length,
    )
    if any(float(v) <= 0 for v in positive):
        p.error("counts, LR, and non-delta loss weights must be positive")
    nonnegative = (
        a.grad_clip,
        a.frozen_head_margin,
        a.pair_margin,
        a.delta_weight,
        a.utility_exclude_first,
    )
    if any(float(v) < 0 for v in nonnegative):
        p.error("margins, clipping, delta weight, and exclusion must be non-negative")
    if a.utility_exclude_first < 20:
        p.error("utility-exclude-first must be at least 20 to protect the fixed PPL prefix")
    if a.utility_max_length < 8:
        p.error("utility-max-length must be at least 8")
    return a


def assert_target_contract(manifest: Mapping[str, Any]) -> None:
    contract = manifest.get("target_contract", {})
    if isinstance(contract, Mapping) and contract:
        sensitive = contract.get("sensitive_answer")
        reference = contract.get("non_sensitive_reference")
        swapping = contract.get("field_swapping")
        if sensitive not in (None, "requested_rewrite.target_true"):
            raise RuntimeError(
                f"split manifest sensitive contract mismatch: {sensitive!r}"
            )
        if reference not in (None, "requested_rewrite.target_new"):
            raise RuntimeError(
                f"split manifest reference contract mismatch: {reference!r}"
            )
        if swapping not in (None, False):
            raise RuntimeError("split manifest unexpectedly enables field swapping")


def validate_direct_only_records(records: Sequence[Mapping[str, Any]]) -> None:
    for i, record in enumerate(records):
        if record.get("paraphrase_prompts") or record.get("neighborhood_prompts"):
            raise RuntimeError(f"record {i} exposes held-out MCF probes")
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping):
            raise RuntimeError(f"record {i} lacks requested_rewrite")
        if not rr.get("target_true", {}).get("str"):
            raise RuntimeError(f"record {i} lacks sensitive target_true.str")
        if not rr.get("target_new", {}).get("str"):
            raise RuntimeError(f"record {i} lacks reference target_new.str")


def find_decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    """Return decoder blocks for common HF causal-LM layouts."""
    candidates: List[Any] = []
    if hasattr(model, "model"):
        inner = getattr(model, "model")
        candidates.extend(
            [
                getattr(inner, "layers", None),
                getattr(inner, "h", None),
            ]
        )
        if hasattr(inner, "decoder"):
            candidates.append(getattr(inner.decoder, "layers", None))
    if hasattr(model, "transformer"):
        candidates.append(getattr(model.transformer, "h", None))
    for layers in candidates:
        if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
            return layers
    raise RuntimeError(
        "Could not locate transformer decoder blocks; expected model.model.layers "
        "or a compatible HF layout"
    )


def configure_last_block_only(model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = find_decoder_layers(model)
    last = layers[-1]
    for parameter in last.parameters():
        parameter.requires_grad_(True)

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise RuntimeError("model must expose input and output embeddings")
    if input_embeddings.weight.requires_grad:
        raise RuntimeError("input embeddings are unexpectedly trainable")
    if output_embeddings.weight.requires_grad:
        raise RuntimeError("frozen decoder head W0 is unexpectedly trainable")

    trainable = [p for p in model.parameters() if p.requires_grad]
    last_ids = {id(p) for p in last.parameters()}
    if not trainable or any(id(p) not in last_ids for p in trainable):
        raise RuntimeError("trainable parameters are not restricted to final decoder block")
    summary = {
        "decoder_block_count": int(len(layers)),
        "trainable_decoder_block_index": int(len(layers) - 1),
        "trainable_parameter_count": int(sum(p.numel() for p in trainable)),
        "lm_head_frozen": True,
        "input_embeddings_frozen": True,
        "earlier_decoder_blocks_frozen": True,
        "final_normalization_frozen": True,
        "output_input_tied": bool(
            input_embeddings.weight.data_ptr() == output_embeddings.weight.data_ptr()
        ),
    }
    return last, summary


def sensitive_relu_loss_from_token_logits(
    token_logits: torch.Tensor,
    target_ids: torch.Tensor,
    margin: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Squared ReLU margin against the strongest non-sensitive vocabulary token."""
    if token_logits.ndim != 2 or target_ids.ndim != 1:
        raise ValueError("expected token_logits [N,V] and target_ids [N]")
    if token_logits.shape[0] != target_ids.shape[0]:
        raise ValueError("token-logit and target-id counts differ")
    if token_logits.shape[0] == 0:
        zero = token_logits.sum() * 0.0
        return zero, target_ids.new_empty((0,), dtype=torch.bool), zero.new_empty((0,))
    rows = torch.arange(token_logits.shape[0], device=token_logits.device)
    logits32 = token_logits.float()
    sensitive = logits32[rows, target_ids]
    mask = torch.zeros_like(logits32, dtype=torch.bool)
    mask[rows, target_ids] = True
    best_other = logits32.masked_fill(mask, -torch.inf).max(dim=-1).values
    gaps = sensitive - best_other
    loss = F.relu(gaps + float(margin)).square().mean()
    sensitive_top1 = gaps >= 0
    return loss, sensitive_top1, gaps


def pair_margin_loss(
    target_true_nll: torch.Tensor,
    target_new_nll: torch.Tensor,
    margin: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if target_true_nll.shape != target_new_nll.shape:
        raise ValueError("target_true and target_new NLL tensors must match")
    pair_margins = target_true_nll.float() - target_new_nll.float()
    return F.relu(float(margin) - pair_margins).square().mean(), pair_margins


def parameter_delta_f2(
    parameters: Sequence[torch.Tensor],
    reference: Sequence[torch.Tensor],
) -> torch.Tensor:
    if len(parameters) != len(reference):
        raise ValueError("parameter/reference lengths differ")
    total: torch.Tensor | None = None
    for parameter, initial in zip(parameters, reference):
        term = (
            parameter.float()
            - initial.to(device=parameter.device, dtype=torch.float32)
        ).square().sum()
        total = term if total is None else total + term
    if total is None:
        raise ValueError("no parameters supplied for delta penalty")
    return total


def _paired_forward(
    model: nn.Module,
    tok: Any,
    instances: Sequence[mcf_repair.MCFPromptInstance],
    device: torch.device,
    llama_like: bool,
    frozen_head_margin: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Official-compatible MCF NLLs plus per-token frozen-head suppression."""
    encoded, target_token_ids, prefix_lens = mcf_repair.official_batch_components(
        tok, instances, device, llama_like
    )
    logits = model(**encoded, use_cache=False).logits
    if llama_like:
        logits = logits[:, 1:, :]

    losses: List[torch.Tensor] = []
    sensitive_logits: List[torch.Tensor] = []
    sensitive_ids: List[int] = []
    for row, (target_ids, prefix_len) in enumerate(zip(target_token_ids, prefix_lens)):
        token_nlls: List[torch.Tensor] = []
        for offset, target_id in enumerate(target_ids):
            position = int(prefix_len) + int(offset) - 1
            row_logits = logits[row, position, :]
            token_nlls.append(-F.log_softmax(row_logits.float(), dim=0)[int(target_id)])
            # official_batch_components stores [target_new, target_true] rows.
            if row % 2 == 1:
                sensitive_logits.append(row_logits)
                sensitive_ids.append(int(target_id))
        losses.append(torch.stack(token_nlls).mean())

    paired = torch.stack(losses).reshape(len(instances), 2)
    true_token_logits = torch.stack(sensitive_logits, dim=0)
    true_target_ids = torch.tensor(
        sensitive_ids, dtype=torch.long, device=true_token_logits.device
    )
    frozen_loss, sensitive_top1, gaps = sensitive_relu_loss_from_token_logits(
        true_token_logits, true_target_ids, frozen_head_margin
    )
    return paired[:, 0], paired[:, 1], frozen_loss, {
        "sensitive_top1": sensitive_top1,
        "sensitive_logit_gap": gaps,
    }


@torch.no_grad()
def evaluate_direct_diagnostics(
    model: nn.Module,
    tok: Any,
    instances: Sequence[mcf_repair.MCFPromptInstance],
    device: torch.device,
    llama_like: bool,
    batch_size: int,
    pair_margin: float,
    frozen_head_margin: float,
) -> Dict[str, Any]:
    model.eval()
    all_pair_margins: List[torch.Tensor] = []
    all_top1: List[torch.Tensor] = []
    all_gaps: List[torch.Tensor] = []
    for start in range(0, len(instances), int(batch_size)):
        chunk = instances[start : start + int(batch_size)]
        new_nll, true_nll, _loss, extra = _paired_forward(
            model,
            tok,
            chunk,
            device,
            llama_like,
            frozen_head_margin,
        )
        all_pair_margins.append((true_nll - new_nll).detach().float().cpu())
        all_top1.append(extra["sensitive_top1"].detach().cpu())
        all_gaps.append(extra["sensitive_logit_gap"].detach().float().cpu())
    margins = torch.cat(all_pair_margins) if all_pair_margins else torch.empty(0)
    top1 = torch.cat(all_top1) if all_top1 else torch.empty(0, dtype=torch.bool)
    gaps = torch.cat(all_gaps) if all_gaps else torch.empty(0)
    return {
        "direct_record_count": int(len(instances)),
        "pair_margin_required": float(pair_margin),
        "pair_failures": int((margins < float(pair_margin)).sum().item()),
        "pair_minimum_margin": float(margins.min()) if margins.numel() else None,
        "pair_mean_margin": float(margins.mean()) if margins.numel() else None,
        "sensitive_teacher_forced_token_count": int(top1.numel()),
        "sensitive_token_top1_count": int(top1.sum().item()),
        "sensitive_token_top1_rate": float(top1.float().mean()) if top1.numel() else None,
        "frozen_head_margin_required": float(frozen_head_margin),
        "sensitive_logit_gap_max": float(gaps.max()) if gaps.numel() else None,
        "sensitive_logit_gap_mean": float(gaps.mean()) if gaps.numel() else None,
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
    assert_target_contract(manifest)
    validate_direct_only_records(records)

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
    instances = stage2.mcf_instances(records)

    # Keep the model in eval mode during optimization: gradients remain enabled,
    # while dropout/stochastic training behavior cannot contaminate the frozen
    # pre-repair utility teacher comparison.
    model.eval()
    before = evaluate_direct_diagnostics(
        model,
        tok,
        instances,
        device,
        llama_like,
        a.batch_size,
        a.pair_margin,
        a.frozen_head_margin,
    )
    initial_pair_margins = stage2.mcf_direct_margins(
        model,
        tok,
        instances,
        device,
        llama_like,
        a.batch_size,
        "target_true",
        "target_new",
    ).detach().cpu()
    active_positions = [
        i
        for i, value in enumerate(initial_pair_margins.tolist())
        if float(value) < float(a.pair_margin)
    ]
    if a.repair_scope == "active":
        train_positions = active_positions
    else:
        train_positions = list(range(len(instances)))
    if not train_positions:
        raise RuntimeError(
            "No direct MCF records selected for representation repair; input "
            "checkpoint already satisfies the configured scope"
        )
    train_instances = [instances[i] for i in train_positions]

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

    last_block, trainable_summary = configure_last_block_only(model)
    trainable_params = [p for p in last_block.parameters() if p.requires_grad]
    # Same-device frozen copies make the literal Frobenius penalty differentiable
    # without repeatedly transferring a full decoder block from CPU.
    initial_params = [p.detach().clone() for p in trainable_params]
    opt = _optimizer(trainable_params, a.optimizer, a.lr)
    forget_sampler = core.IndexSampler(len(train_instances), a.batch_size, a.seed + 31001)
    utility_sampler = core.IndexSampler(
        len(utility_prompts), a.utility_batch_size, a.utility_seed + 31003
    )

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    core.write_json(out_dir / "utility_receipt.json", utility_receipt)
    core.write_json(out_dir / "direct_before.json", before)

    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(1, int(a.steps) + 1), desc="MCF frozen-head representation repair"):
            forget_idx = forget_sampler.next()
            batch = [train_instances[i] for i in forget_idx]
            utility_idx = utility_sampler.next()
            utility_batch = [utility_prompts[i] for i in utility_idx]

            opt.zero_grad(set_to_none=True)
            new_nll, true_nll, frozen_head_loss, extra = _paired_forward(
                model,
                tok,
                batch,
                device,
                llama_like,
                a.frozen_head_margin,
            )
            pair_loss, pair_margins = pair_margin_loss(
                true_nll, new_nll, a.pair_margin
            )
            utility_logits = wikipedia_utility._forward_prompt_logits(
                model, tok, utility_batch, device
            )
            utility_loss = wikipedia_utility.utility_kl(
                utility_logits, utility_base_logits[utility_idx]
            )
            delta_f2 = parameter_delta_f2(trainable_params, initial_params)
            total = (
                float(a.frozen_head_weight) * frozen_head_loss
                + float(a.pair_weight) * pair_loss
                + float(a.utility_kl_weight) * utility_loss
                + float(a.delta_weight) * delta_f2
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite representation loss at step {step}")
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, float(a.grad_clip)
            ) if a.grad_clip > 0 else None
            if grad_norm is not None and not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at step {step}")
            opt.step()

            if step == 1 or step % int(a.check_every) == 0 or step == int(a.steps):
                row = {
                    "step": int(step),
                    "total_loss": float(total.detach().cpu()),
                    "frozen_head_relu_loss": float(frozen_head_loss.detach().cpu()),
                    "pair_margin_loss": float(pair_loss.detach().cpu()),
                    "utility_kl_loss": float(utility_loss.detach().cpu()),
                    "delta_phi_frobenius_squared": float(delta_f2.detach().cpu()),
                    "delta_phi_frobenius_norm": float(delta_f2.detach().sqrt().cpu()),
                    "batch_pair_min_margin": float(pair_margins.min().detach().cpu()),
                    "batch_sensitive_token_top1_count": int(
                        extra["sensitive_top1"].sum().detach().cpu()
                    ),
                    "frozen_head_weight": float(a.frozen_head_weight),
                    "pair_weight": float(a.pair_weight),
                    "utility_kl_weight": float(a.utility_kl_weight),
                    "delta_weight": float(a.delta_weight),
                    "benchmark_retain_seen": 0,
                    "heldout_paraphrases_seen": 0,
                    "locality_or_neighborhood_seen": 0,
                    "PPL_seen": False,
                }
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()

    del opt
    model.eval()
    after = evaluate_direct_diagnostics(
        model,
        tok,
        instances,
        device,
        llama_like,
        a.batch_size,
        a.pair_margin,
        a.frozen_head_margin,
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
        final_delta_f2 = parameter_delta_f2(trainable_params, initial_params)

    # Hard architectural assertions immediately before save.
    if model.get_output_embeddings().weight.requires_grad:
        raise RuntimeError("W0 became trainable during representation repair")
    if model.get_input_embeddings().weight.requires_grad:
        raise RuntimeError("input embeddings became trainable during representation repair")
    non_last_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in {id(p) for p in last_block.parameters()}
    ]
    if non_last_trainable:
        raise RuntimeError(f"parameters outside final block became trainable: {non_last_trainable[:5]}")

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
            "non_sensitive_reference": "requested_rewrite.target_new",
            "field_swapping": False,
        },
        "repair_scope": a.repair_scope,
        "initial_active_positions": active_positions,
        "initial_active_count": int(len(active_positions)),
        "training_positions": train_positions,
        "training_record_count": int(len(train_positions)),
        "teacher_forcing": True,
        "frozen_head_definition": "the unchanged lm_head W0 from --model-path",
        "frozen_head_loss": "mean ReLU(z_target_true - max_{j!=target_true} z_j + margin)^2 over teacher-forced target_true tokens",
        "pair_loss": "mean ReLU(pair_margin - (NLL(target_true)-NLL(target_new)))^2",
        "subspace_loss_included": False,
        "utility_loss": "KL(pre-representation checkpoint || current) on external Wikipedia next-token distributions",
        "parameter_delta_loss": "literal ||Delta Phi_R||_F^2 over the trainable final decoder block",
        "weights": {
            "frozen_head": float(a.frozen_head_weight),
            "pair": float(a.pair_weight),
            "utility": float(a.utility_kl_weight),
            "delta": float(a.delta_weight),
        },
        "margins": {
            "frozen_head": float(a.frozen_head_margin),
            "pair": float(a.pair_margin),
        },
        "steps": int(a.steps),
        "batch_size": int(a.batch_size),
        "lr": float(a.lr),
        "optimizer": a.optimizer,
        "gradient_clip": float(a.grad_clip),
        "trainable_architecture": trainable_summary,
        "utility_sample_size": int(a.utility_sample_size),
        "utility_exclude_first": int(a.utility_exclude_first),
        "utility_post_kl": utility_post,
        "direct_before": before,
        "direct_after": after,
        "delta_phi_frobenius_squared": float(final_delta_f2.detach().cpu()),
        "delta_phi_frobenius_norm": float(final_delta_f2.detach().sqrt().cpu()),
        "benchmark_retain_seen": 0,
        "heldout_paraphrases_seen": 0,
        "locality_or_neighborhood_seen": 0,
        "PPL_seen": False,
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "representation_repair_summary.json", receipt)
    core.write_json(out_dir / "direct_after.json", after)
    core.write_json(out_dir / "utility_post_kl.json", utility_post)

    print("Frozen-head representation-repair checkpoint:", ckpt)
    print("Target contract: target_true=sensitive, target_new=reference, swapping=false")
    print("Trainable architecture:", trainable_summary)
    print("Direct before:", before)
    print("Direct after:", after)
    print("Wikipedia utility post KL:", utility_post)
    print("Delta Phi_R Frobenius norm:", float(final_delta_f2.detach().sqrt().cpu()))
    print("Stage 2 was NOT run; evaluate this representation checkpoint before any decoder repair.")


if __name__ == "__main__":
    main()
