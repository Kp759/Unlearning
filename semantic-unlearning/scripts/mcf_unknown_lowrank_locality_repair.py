#!/usr/bin/env python3
"""Low-rank Unknown-neutral MCF repair with Base hidden-locality preservation.

This experiment starts from the Base causal LM, not a previously edited SURE
checkpoint.  It hard-enforces the target-true-sensitive MCF contract and uses
only the locked direct forget records plus external Wikipedia utility text.
Official paraphrases, official neighborhood prompts, benchmark retain examples,
and the fixed PPL prefix are unavailable to training/checkpoint selection.

Forgetting objective (frozen original decoder W0):

    m_U(q) = NLL_W0(q, target_true_sensitive) - NLL_W0(q, Unknown)
    L_forget = mean ReLU(forget_margin - m_U(q))^2

Locality proxy: for every direct training-visible relation template q_i, build
subject-swapped donor prompts q_i(s_j) using distinct subjects from the same
50-record training-visible set.  Cache their Base final hidden states h0.  While
training, preserve those hidden states directly:

    L_local = mean || h_theta(q_i(s_j)) - h0(q_i(s_j)) ||_2^2

The official MCF neighborhood prompts are never read.

Architecture: every original model parameter is frozen.  A rank-r residual
adapter is inserted only around the final decoder block's MLP down projection:

    down'(x) = W_down x + B A x

Only A and B are trained.  B is initialized to zero, so the initial model is
exactly Base.  After optimization, BA is merged into W_down and the wrapper is
removed, yielding a standard Hugging Face checkpoint that can be evaluated by
existing MCF evaluators and passed to ordinary SURE Stage 2.

Full pilot objective:

    L = lambda_f L_forget
      + lambda_h L_local
      + lambda_u KL(Base || current) on external Wikipedia
      + lambda_delta ||BA||_F^2.

requested_rewrite.target_new is never a training target.  It is used only for a
post-hoc direct diagnostic so we can measure eventual Stage-2 burden.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

import gagd_compare as gagd
import gagd_active_case_repair as mcf_repair
import mcf_frozen_head_representation_repair as legacy_rep
import mcf_frozen_head_unknown_representation_repair as unknown_rep
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import sure_stage1_gagd_w1k as wikipedia_utility
import sure_stage2_sparse_repair as stage2
import sure_stage2_sparse_repair_subject_contrast_materialized as subject_contrast


METHOD = "SURE-LM-MCF-Base-Unknown-lowrank-hidden-locality"
PROTOCOL = "mcf_target_true_sensitive_base_unknown_lowrank_locality_v1"
DEFAULT_NEUTRAL_ANSWER = "Unknown"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Base model checkpoint")
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
        help="active uses only records failing the Base sensitive-vs-Unknown margin",
    )

    p.add_argument("--adapter-rank", type=int, required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--check-every", type=int, default=25)

    p.add_argument("--forget-margin", type=float, default=0.05)
    p.add_argument("--forget-weight", type=float, default=1.0)
    p.add_argument("--locality-hidden-weight", type=float, default=10.0)
    p.add_argument("--delta-weight", type=float, default=1e-4)

    p.add_argument("--subject-control-count", type=int, default=4)
    p.add_argument("--locality-batch-size", type=int, default=4)
    p.add_argument("--locality-cache-batch-size", type=int, default=8)

    p.add_argument("--utility-wikipedia-dir", required=True)
    p.add_argument("--utility-sample-size", type=int, default=200)
    p.add_argument("--utility-batch-size", type=int, default=4)
    p.add_argument("--utility-cache-batch-size", type=int, default=8)
    p.add_argument("--utility-max-length", type=int, default=128)
    p.add_argument("--utility-seed", type=int, default=1)
    p.add_argument("--utility-exclude-first", type=int, default=20)
    p.add_argument("--utility-kl-weight", type=float, default=2.0)

    p.add_argument("--benchmark-pair-margin", type=float, default=0.05)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")

    a = p.parse_args(list(argv) if argv is not None else None)
    positive = (
        a.forget_num,
        a.adapter_rank,
        a.steps,
        a.batch_size,
        a.lr,
        a.check_every,
        a.forget_weight,
        a.locality_hidden_weight,
        a.subject_control_count,
        a.locality_batch_size,
        a.locality_cache_batch_size,
        a.utility_sample_size,
        a.utility_batch_size,
        a.utility_cache_batch_size,
        a.utility_max_length,
        a.utility_kl_weight,
    )
    if any(float(v) <= 0 for v in positive):
        p.error("counts, rank, LR, and non-delta loss weights must be positive")
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
    if a.subject_control_count >= a.forget_num:
        p.error("subject-control-count must be smaller than forget-num")
    return a


class LowRankResidualLinear(nn.Module):
    """Frozen linear layer plus a trainable rank-r BA residual."""

    def __init__(self, base: nn.Linear, rank: int) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LowRankResidualLinear requires nn.Linear")
        if int(rank) <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.A = nn.Linear(
            base.in_features,
            self.rank,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.B = nn.Linear(
            self.rank,
            base.out_features,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        # LoRA-style zero initial residual: exact Base function at step zero.
        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.B(self.A(x))

    def effective_delta(self) -> torch.Tensor:
        return self.B.weight.float() @ self.A.weight.float()

    @property
    def trainable_parameter_count(self) -> int:
        return int(self.A.weight.numel() + self.B.weight.numel())

    @torch.no_grad()
    def merge_into_base(self) -> nn.Linear:
        delta = self.effective_delta().to(
            device=self.base.weight.device, dtype=self.base.weight.dtype
        )
        self.base.weight.add_(delta)
        return self.base


def install_final_mlp_adapter(
    model: nn.Module, rank: int
) -> Tuple[nn.Module, LowRankResidualLinear, Dict[str, Any]]:
    """Freeze Base and wrap only final decoder block mlp.down_proj."""
    for p in model.parameters():
        p.requires_grad_(False)
    layers = legacy_rep.find_decoder_layers(model)
    last = layers[-1]
    mlp = getattr(last, "mlp", None)
    down = getattr(mlp, "down_proj", None) if mlp is not None else None
    if not isinstance(down, nn.Linear):
        raise RuntimeError(
            "Expected final decoder block .mlp.down_proj to be nn.Linear; "
            "this pilot currently targets Llama-like decoder blocks"
        )
    adapter = LowRankResidualLinear(down, int(rank))
    mlp.down_proj = adapter

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    allowed_ids = {id(adapter.A.weight), id(adapter.B.weight)}
    if not trainable or any(id(p) not in allowed_ids for _, p in trainable):
        raise RuntimeError(
            "Original Base parameters became trainable; expected adapter A/B only"
        )
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise RuntimeError("model must expose input/output embeddings")
    summary = {
        "decoder_block_count": int(len(layers)),
        "adapter_decoder_block_index": int(len(layers) - 1),
        "adapter_location": "final_decoder_block.mlp.down_proj",
        "adapter_rank": int(rank),
        "adapter_parameter_count": int(adapter.trainable_parameter_count),
        "base_parameter_count_trainable": 0,
        "lm_head_frozen": not output_embeddings.weight.requires_grad,
        "input_embeddings_frozen": not input_embeddings.weight.requires_grad,
        "all_original_model_weights_frozen": True,
        "initial_residual_exactly_zero": True,
    }
    return last, adapter, summary


@torch.no_grad()
def merge_and_remove_adapter(model: nn.Module, adapter: LowRankResidualLinear) -> Dict[str, Any]:
    layers = legacy_rep.find_decoder_layers(model)
    last = layers[-1]
    mlp = getattr(last, "mlp", None)
    if mlp is None or getattr(mlp, "down_proj", None) is not adapter:
        raise RuntimeError("installed adapter is no longer at final mlp.down_proj")
    delta = adapter.effective_delta().detach().float().cpu()
    base = adapter.merge_into_base()
    mlp.down_proj = base
    for p in model.parameters():
        p.requires_grad_(False)
    return {
        "rank": int(adapter.rank),
        "effective_delta_frobenius_norm": float(delta.norm()),
        "effective_delta_frobenius_squared": float(delta.square().sum()),
        "materialized_into": "final_decoder_block.mlp.down_proj.weight",
        "wrapper_removed_before_save": True,
    }


def build_locality_prompts(
    records: Sequence[Mapping[str, Any]], control_count: int
) -> tuple[List[str], List[Dict[str, Any]]]:
    """Build same-template subject-swapped locality calibration prompts."""
    subjects = subject_contrast._subjects(records)
    prompts: List[str] = []
    receipt: List[Dict[str, Any]] = []
    for position, record in enumerate(records):
        rr = record["requested_rewrite"]
        template = str(rr["prompt"])
        own_subject = subjects[position]
        donor_ids = subject_contrast._donor_indices(
            int(position), subjects, int(control_count)
        )
        for donor_position in donor_ids:
            donor_subject = subjects[donor_position]
            prompt = template.format(donor_subject)
            prompts.append(prompt)
            receipt.append(
                {
                    "source_record_position": int(position),
                    "source_case_id": int(record.get("case_id", position)),
                    "relation_template": template,
                    "original_subject": own_subject,
                    "donor_record_position": int(donor_position),
                    "donor_subject": donor_subject,
                    "prompt": prompt,
                }
            )
    if not prompts:
        raise RuntimeError("No locality calibration prompts were built")
    return prompts, receipt


def _final_hidden_for_prompts(
    model: nn.Module,
    tok: Any,
    prompts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    encoded = tok(list(prompts), padding=True, truncation=True, return_tensors="pt").to(device)
    out = model(
        **encoded,
        output_hidden_states=True,
        use_cache=False,
    )
    positions = encoded["attention_mask"].sum(dim=1) - 1
    rows = torch.arange(len(prompts), device=device)
    return out.hidden_states[-1][rows, positions, :]


@torch.no_grad()
def cache_base_locality_hidden(
    model: nn.Module,
    tok: Any,
    prompts: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    chunks: List[torch.Tensor] = []
    for start in range(0, len(prompts), int(batch_size)):
        hidden = _final_hidden_for_prompts(
            model, tok, prompts[start : start + int(batch_size)], device
        )
        chunks.append(hidden.detach().to(dtype=torch.float16, device="cpu"))
    return torch.cat(chunks, dim=0).contiguous()


def locality_hidden_loss(
    current: torch.Tensor, base_hidden: torch.Tensor
) -> torch.Tensor:
    ref = base_hidden.to(device=current.device, dtype=torch.float32)
    return F.mse_loss(current.float(), ref, reduction="mean")


@torch.no_grad()
def evaluate_locality_hidden_drift(
    model: nn.Module,
    tok: Any,
    prompts: Sequence[str],
    base_hidden: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    per_prompt: List[torch.Tensor] = []
    for start in range(0, len(prompts), int(batch_size)):
        current = _final_hidden_for_prompts(
            model, tok, prompts[start : start + int(batch_size)], device
        ).float()
        ref = base_hidden[start : start + len(current)].to(
            device=current.device, dtype=torch.float32
        )
        # Root mean squared hidden drift per prompt, invariant to hidden width.
        rms = (current - ref).square().mean(dim=-1).sqrt()
        per_prompt.append(rms.detach().cpu())
    values = torch.cat(per_prompt).float()
    return {
        "mean_rms": float(values.mean()),
        "p95_rms": float(torch.quantile(values, 0.95)),
        "max_rms": float(values.max()),
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
    model.eval()

    unknown_instances = unknown_rep.build_unknown_instances(records, a.neutral_answer)
    benchmark_instances = stage2.mcf_instances(records)
    unknown_before = unknown_rep.evaluate_unknown_diagnostics(
        model,
        tok,
        unknown_instances,
        device,
        llama_like,
        int(a.batch_size),
        float(a.forget_margin),
    )
    benchmark_before = unknown_rep.evaluate_benchmark_pair_diagnostics(
        model,
        tok,
        benchmark_instances,
        device,
        llama_like,
        int(a.batch_size),
        float(a.benchmark_pair_margin),
    )
    active_positions = [
        i
        for i, value in enumerate(unknown_before["margins"])
        if float(value) < float(a.forget_margin)
    ]
    train_positions = (
        active_positions if a.repair_scope == "active" else list(range(len(records)))
    )
    if not train_positions:
        raise RuntimeError("Base already satisfies the configured sensitive-vs-Unknown margin")
    train_instances = [unknown_instances[i] for i in train_positions]

    locality_prompts, locality_receipt = build_locality_prompts(
        records, int(a.subject_control_count)
    )
    print(
        f"Caching Base final hidden states for {len(locality_prompts)} same-template donor prompts...",
        flush=True,
    )
    locality_base_hidden = cache_base_locality_hidden(
        model,
        tok,
        locality_prompts,
        device,
        int(a.locality_cache_batch_size),
    )

    utility_prompts, utility_receipt = wikipedia_utility.build_utility_prompts(
        tok,
        Path(a.utility_wikipedia_dir).resolve(),
        sample_size=int(a.utility_sample_size),
        seed=int(a.utility_seed),
        exclude_first=int(a.utility_exclude_first),
        max_length=int(a.utility_max_length),
    )
    print(
        f"Caching Base logits for {len(utility_prompts)} external Wikipedia utility contexts...",
        flush=True,
    )
    utility_base_logits = wikipedia_utility.cache_utility_base_logits(
        model,
        tok,
        utility_prompts,
        device,
        int(a.utility_cache_batch_size),
    )

    _last_block, adapter, architecture = install_final_mlp_adapter(
        model, int(a.adapter_rank)
    )
    adapter_params = [adapter.A.weight, adapter.B.weight]
    opt = _optimizer(adapter_params, a.optimizer, float(a.lr))
    forget_sampler = core.IndexSampler(
        len(train_instances), int(a.batch_size), int(a.seed) + 51001
    )
    locality_sampler = core.IndexSampler(
        len(locality_prompts), int(a.locality_batch_size), int(a.seed) + 51003
    )
    utility_sampler = core.IndexSampler(
        len(utility_prompts), int(a.utility_batch_size), int(a.utility_seed) + 51005
    )

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    core.write_json(out_dir / "locality_calibration_receipt.json", {
        "kind": "same-template subject-swapped direct-training-only locality calibration",
        "prompt_count": int(len(locality_prompts)),
        "subject_control_count": int(a.subject_control_count),
        "official_neighborhood_prompts_seen": 0,
        "official_paraphrases_seen": 0,
        "records": locality_receipt,
    })
    core.write_json(out_dir / "utility_receipt.json", utility_receipt)
    core.write_json(out_dir / "unknown_before.json", unknown_before)
    core.write_json(out_dir / "benchmark_pair_before.json", benchmark_before)

    model.eval()
    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in tqdm(
            range(1, int(a.steps) + 1),
            desc=f"MCF Base Unknown low-rank locality r{a.adapter_rank}",
        ):
            forget_idx = forget_sampler.next()
            forget_batch = [train_instances[i] for i in forget_idx]
            loc_idx = locality_sampler.next()
            loc_prompts = [locality_prompts[i] for i in loc_idx]
            util_idx = utility_sampler.next()
            util_prompts = [utility_prompts[i] for i in util_idx]

            opt.zero_grad(set_to_none=True)

            neutral_nll, sensitive_nll = unknown_rep._unknown_forward(
                model, tok, forget_batch, device, llama_like
            )
            forget_loss, forget_margins = unknown_rep.unknown_margin_loss(
                sensitive_nll, neutral_nll, float(a.forget_margin)
            )

            loc_hidden = _final_hidden_for_prompts(
                model, tok, loc_prompts, device
            )
            loc_loss = locality_hidden_loss(
                loc_hidden, locality_base_hidden[loc_idx]
            )

            util_logits = wikipedia_utility._forward_prompt_logits(
                model, tok, util_prompts, device
            )
            util_loss = wikipedia_utility.utility_kl(
                util_logits, utility_base_logits[util_idx]
            )

            delta = adapter.effective_delta()
            delta_f2 = delta.square().sum()
            total = (
                float(a.forget_weight) * forget_loss
                + float(a.locality_hidden_weight) * loc_loss
                + float(a.utility_kl_weight) * util_loss
                + float(a.delta_weight) * delta_f2
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite low-rank locality loss at step {step}")
            total.backward()
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(adapter_params, float(a.grad_clip))
                if a.grad_clip > 0
                else None
            )
            if grad_norm is not None and not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite adapter gradient at step {step}")
            opt.step()

            if step == 1 or step % int(a.check_every) == 0 or step == int(a.steps):
                row = {
                    "step": int(step),
                    "total_loss": float(total.detach().cpu()),
                    "unknown_forget_loss": float(forget_loss.detach().cpu()),
                    "locality_hidden_mse": float(loc_loss.detach().cpu()),
                    "wikipedia_utility_kl": float(util_loss.detach().cpu()),
                    "effective_delta_frobenius_norm": float(delta.detach().norm().cpu()),
                    "batch_unknown_min_margin": float(forget_margins.min().detach().cpu()),
                    "adapter_rank": int(a.adapter_rank),
                    "forget_weight": float(a.forget_weight),
                    "locality_hidden_weight": float(a.locality_hidden_weight),
                    "utility_kl_weight": float(a.utility_kl_weight),
                    "delta_weight": float(a.delta_weight),
                    "benchmark_retain_seen": 0,
                    "heldout_paraphrases_seen": 0,
                    "official_neighborhood_seen": 0,
                    "PPL_seen": False,
                }
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()

    del opt
    model.eval()
    unknown_after_adapter = unknown_rep.evaluate_unknown_diagnostics(
        model,
        tok,
        unknown_instances,
        device,
        llama_like,
        int(a.batch_size),
        float(a.forget_margin),
    )
    benchmark_after_adapter = unknown_rep.evaluate_benchmark_pair_diagnostics(
        model,
        tok,
        benchmark_instances,
        device,
        llama_like,
        int(a.batch_size),
        float(a.benchmark_pair_margin),
    )
    locality_drift = evaluate_locality_hidden_drift(
        model,
        tok,
        locality_prompts,
        locality_base_hidden,
        device,
        int(a.locality_cache_batch_size),
    )
    utility_post = wikipedia_utility.evaluate_utility_kl(
        model,
        tok,
        utility_prompts,
        utility_base_logits,
        device,
        int(a.utility_cache_batch_size),
    )

    materialization = merge_and_remove_adapter(model, adapter)
    model.eval()
    # Verify the merged checkpoint function preserves the trained diagnostics.
    unknown_after = unknown_rep.evaluate_unknown_diagnostics(
        model,
        tok,
        unknown_instances,
        device,
        llama_like,
        int(a.batch_size),
        float(a.forget_margin),
    )
    benchmark_after = unknown_rep.evaluate_benchmark_pair_diagnostics(
        model,
        tok,
        benchmark_instances,
        device,
        llama_like,
        int(a.batch_size),
        float(a.benchmark_pair_margin),
    )
    if unknown_after["failures"] != unknown_after_adapter["failures"]:
        raise RuntimeError("adapter merge changed Unknown direct failure count")
    if benchmark_after["failures"] != benchmark_after_adapter["failures"]:
        raise RuntimeError("adapter merge changed benchmark direct failure count")

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    receipt = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "source_model_path": str(Path(a.model_path).resolve()),
        "source_model_role": "Base model; no previous SURE/MCF edited checkpoint",
        "training_visible_path": str(visible_path),
        "split_manifest": str(manifest_path),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "target_contract": {
            "sensitive_unwanted": "requested_rewrite.target_true",
            "neutral_training_answer": str(a.neutral_answer),
            "requested_rewrite.target_new_used_in_training_loss": False,
            "field_swapping": False,
        },
        "forget_objective": "ReLU(m - [NLL_W0(target_true)-NLL_W0(Unknown)])^2",
        "locality_objective": "MSE(current final hidden, Base final hidden) on same-template subject-swapped donor prompts",
        "locality_calibration_source": "direct training-visible 50 only; no official neighborhoods",
        "utility_objective": "KL(Base || current) on external Wikipedia next-token distributions",
        "architecture": architecture,
        "materialization": materialization,
        "repair_scope": a.repair_scope,
        "initial_unknown_active_positions": active_positions,
        "initial_unknown_active_count": int(len(active_positions)),
        "training_positions": train_positions,
        "training_record_count": int(len(train_positions)),
        "weights": {
            "forget": float(a.forget_weight),
            "locality_hidden": float(a.locality_hidden_weight),
            "wikipedia_utility": float(a.utility_kl_weight),
            "effective_delta_f2": float(a.delta_weight),
        },
        "forget_margin": float(a.forget_margin),
        "steps": int(a.steps),
        "batch_size": int(a.batch_size),
        "lr": float(a.lr),
        "optimizer": a.optimizer,
        "subject_control_count": int(a.subject_control_count),
        "locality_prompt_count": int(len(locality_prompts)),
        "utility_sample_size": int(a.utility_sample_size),
        "utility_exclude_first": int(a.utility_exclude_first),
        "unknown_before": unknown_before,
        "unknown_after": unknown_after,
        "benchmark_pair_before": benchmark_before,
        "benchmark_pair_after": benchmark_after,
        "locality_hidden_drift": locality_drift,
        "utility_post_kl": utility_post,
        "benchmark_retain_seen": 0,
        "heldout_paraphrases_seen": 0,
        "official_neighborhood_seen": 0,
        "PPL_seen": False,
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "lowrank_locality_summary.json", receipt)
    core.write_json(out_dir / "unknown_after.json", unknown_after)
    core.write_json(out_dir / "benchmark_pair_after.json", benchmark_after)
    core.write_json(out_dir / "locality_hidden_drift.json", locality_drift)
    core.write_json(out_dir / "utility_post_kl.json", utility_post)

    print("Low-rank locality-preserving checkpoint:", ckpt)
    print("Source role: Base model (not W200/KL2 Stage1)")
    print("Target contract: target_true=sensitive, Unknown=neutral, target_new training=false")
    print("Architecture:", architecture)
    print("Unknown before:", {k: v for k, v in unknown_before.items() if k != "margins"})
    print("Unknown after:", {k: v for k, v in unknown_after.items() if k != "margins"})
    print("Benchmark true-vs-new before (diagnostic only):", benchmark_before)
    print("Benchmark true-vs-new after (diagnostic only):", benchmark_after)
    print("Locality calibration hidden drift:", locality_drift)
    print("Wikipedia utility post KL:", utility_post)
    print("Materialized adapter:", materialization)
    print("Official MCF paraphrase/neighborhood evaluation was NOT used in training.")


if __name__ == "__main__":
    main()
