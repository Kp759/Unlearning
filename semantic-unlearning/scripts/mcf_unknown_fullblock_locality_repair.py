#!/usr/bin/env python3
"""Full-block Unknown-neutral MCF repair with Base hidden-locality preservation.

No LoRA, no low-rank adapter, and no trainable embedding/LM-head parameters.
This experiment starts from the Base causal LM. Every parameter is frozen except
for the final transformer decoder block.

Forget objective under the frozen Base decoder W0:

    m_U(q) = NLL_W0(q, target_true_sensitive) - NLL_W0(q, Unknown)
    L_forget = mean ReLU(forget_margin - m_U(q))^2

Locality calibration uses only the stripped, training-visible direct MCF records.
For each relation template q_i and subject s_i, deterministic donor subjects s_j
from other direct records create prompts q_i(s_j). Official MCF neighborhood and
paraphrase prompts are never read. Their Base final hidden states and Base
next-token distributions are cached before training.

    L_hidden = mean ||h_theta(q_i(s_j)) - h_0(q_i(s_j))||_2^2
    L_localKL = KL(p_0(.|q_i(s_j)) || p_theta(.|q_i(s_j)))

External Wikipedia KL is retained as a generic utility guard. The final loss is

    L = lambda_f L_forget
      + lambda_h L_hidden
      + lambda_l L_localKL
      + lambda_u L_wikiKL
      + lambda_delta ||Delta Phi_last||_F^2.

requested_rewrite.target_new is never used as a training target. It is used only
for a post-hoc direct diagnostic to estimate ordinary Stage-2 burden.
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
import mcf_frozen_head_representation_repair as legacy_rep
import mcf_frozen_head_unknown_representation_repair as unknown_rep
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import sure_stage1_gagd_w1k as wikipedia_utility
import sure_stage2_sparse_repair as stage2
import sure_stage2_sparse_repair_subject_contrast_materialized as subject_contrast


METHOD = "SURE-LM-MCF-Base-Unknown-fullblock-hidden-locality"
PROTOCOL = "mcf_target_true_sensitive_base_unknown_fullblock_locality_v1"
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
    p.add_argument("--repair-scope", choices=("active", "all"), default="active")

    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--check-every", type=int, default=25)

    p.add_argument("--forget-margin", type=float, default=0.05)
    p.add_argument("--forget-weight", type=float, default=1.0)
    p.add_argument("--locality-hidden-weight", type=float, default=10.0)
    p.add_argument("--locality-kl-weight", type=float, default=2.0)
    p.add_argument("--delta-weight", type=float, default=1e-8)

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
        a.steps,
        a.batch_size,
        a.lr,
        a.check_every,
        a.forget_weight,
        a.locality_hidden_weight,
        a.locality_kl_weight,
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
        p.error("counts, LR, and non-delta loss weights must be positive")
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
        p.error("utility-exclude-first must be at least 20")
    return a


def build_locality_prompts(
    records: Sequence[Mapping[str, Any]], control_count: int
) -> Tuple[List[str], List[Dict[str, Any]]]:
    subjects = subject_contrast._subjects(records)
    prompts: List[str] = []
    receipt: List[Dict[str, Any]] = []
    for position, record in enumerate(records):
        rr = record["requested_rewrite"]
        template = str(rr["prompt"])
        donors = subject_contrast._donor_indices(position, subjects, int(control_count))
        for donor in donors:
            prompt = template.format(subjects[donor])
            prompts.append(prompt)
            receipt.append(
                {
                    "source_position": int(position),
                    "donor_position": int(donor),
                    "original_subject": subjects[position],
                    "donor_subject": subjects[donor],
                    "prompt": prompt,
                }
            )
    if not prompts:
        raise RuntimeError("no locality calibration prompts were created")
    return prompts, receipt


def _encode_prompts(tok: Any, prompts: Sequence[str], device: torch.device):
    encoded = tok(list(prompts), padding=True, return_tensors="pt")
    return encoded.to(device)


def _final_hidden_and_logits(
    model: nn.Module,
    tok: Any,
    prompts: Sequence[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = _encode_prompts(tok, prompts, device)
    out = model(**encoded, output_hidden_states=True, use_cache=False)
    positions = encoded["attention_mask"].sum(dim=1) - 1
    rows = torch.arange(len(prompts), device=device)
    hidden = out.hidden_states[-1][rows, positions, :]
    logits = out.logits[rows, positions, :]
    return hidden, logits


@torch.no_grad()
def cache_locality_reference(
    model: nn.Module,
    tok: Any,
    prompts: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    hidden_chunks: List[torch.Tensor] = []
    logit_chunks: List[torch.Tensor] = []
    for start in range(0, len(prompts), int(batch_size)):
        batch = prompts[start : start + int(batch_size)]
        h, z = _final_hidden_and_logits(model, tok, batch, device)
        hidden_chunks.append(h.detach().float().cpu())
        logit_chunks.append(z.detach().to(dtype=torch.float16).cpu())
    return torch.cat(hidden_chunks, dim=0), torch.cat(logit_chunks, dim=0)


def locality_hidden_loss(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if current.shape != reference.shape:
        raise ValueError("current/reference hidden-state shapes differ")
    return (current.float() - reference.float()).square().mean()


def locality_kl(current_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    if current_logits.shape != reference_logits.shape:
        raise ValueError("current/reference locality logit shapes differ")
    ref_logp = F.log_softmax(reference_logits.float(), dim=-1)
    cur_logp = F.log_softmax(current_logits.float(), dim=-1)
    ref_p = ref_logp.exp()
    return (ref_p * (ref_logp - cur_logp)).sum(dim=-1).mean()


def _optimizer(parameters: Iterable[nn.Parameter], kind: str, lr: float):
    params = list(parameters)
    if kind == "adam":
        return torch.optim.Adam(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)


@torch.no_grad()
def evaluate_locality_drift(
    model: nn.Module,
    tok: Any,
    prompts: Sequence[str],
    base_hidden: torch.Tensor,
    base_logits: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    rms_values: List[torch.Tensor] = []
    kl_values: List[torch.Tensor] = []
    for start in range(0, len(prompts), int(batch_size)):
        stop = min(len(prompts), start + int(batch_size))
        h, z = _final_hidden_and_logits(model, tok, prompts[start:stop], device)
        ref_h = base_hidden[start:stop].to(device=device, dtype=torch.float32)
        ref_z = base_logits[start:stop].to(device=device, dtype=torch.float32)
        rms = (h.float() - ref_h).square().mean(dim=-1).sqrt().cpu()
        ref_logp = F.log_softmax(ref_z, dim=-1)
        cur_logp = F.log_softmax(z.float(), dim=-1)
        kl = (ref_logp.exp() * (ref_logp - cur_logp)).sum(dim=-1).cpu()
        rms_values.append(rms)
        kl_values.append(kl)
    rms = torch.cat(rms_values)
    kl = torch.cat(kl_values)
    return {
        "prompt_count": int(len(prompts)),
        "hidden_rms_mean": float(rms.mean()),
        "hidden_rms_p95": float(torch.quantile(rms, 0.95)),
        "hidden_rms_max": float(rms.max()),
        "locality_kl_mean": float(kl.mean()),
        "locality_kl_p95": float(torch.quantile(kl, 0.95)),
        "locality_kl_max": float(kl.max()),
    }


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
        model, tok, unknown_instances, device, llama_like, a.batch_size, a.forget_margin
    )
    benchmark_before = unknown_rep.evaluate_benchmark_pair_diagnostics(
        model, tok, benchmark_instances, device, llama_like,
        a.batch_size, a.benchmark_pair_margin
    )
    active_positions = [
        i for i, value in enumerate(unknown_before["margins"])
        if float(value) < float(a.forget_margin)
    ]
    train_positions = active_positions if a.repair_scope == "active" else list(range(len(records)))
    if not train_positions:
        raise RuntimeError("no sensitive-vs-Unknown failures selected for repair")
    train_instances = [unknown_instances[i] for i in train_positions]

    locality_prompts, locality_receipt = build_locality_prompts(
        records, int(a.subject_control_count)
    )
    print(f"Caching Base hidden/logit references for {len(locality_prompts)} locality prompts...", flush=True)
    base_local_h, base_local_z = cache_locality_reference(
        model, tok, locality_prompts, device, int(a.locality_cache_batch_size)
    )

    utility_prompts, utility_receipt = wikipedia_utility.build_utility_prompts(
        tok, Path(a.utility_wikipedia_dir).resolve(),
        sample_size=int(a.utility_sample_size), seed=int(a.utility_seed),
        exclude_first=int(a.utility_exclude_first), max_length=int(a.utility_max_length)
    )
    print(f"Caching Base logits for {len(utility_prompts)} Wikipedia utility contexts...", flush=True)
    utility_base_logits = wikipedia_utility.cache_utility_base_logits(
        model, tok, utility_prompts, device, int(a.utility_cache_batch_size)
    )

    last_block, trainable_summary = legacy_rep.configure_last_block_only(model)
    trainable_params = [p for p in last_block.parameters() if p.requires_grad]
    initial_params = [p.detach().clone() for p in trainable_params]
    opt = _optimizer(trainable_params, a.optimizer, a.lr)

    forget_sampler = core.IndexSampler(len(train_instances), int(a.batch_size), int(a.seed) + 51001)
    local_sampler = core.IndexSampler(len(locality_prompts), int(a.locality_batch_size), int(a.seed) + 51003)
    utility_sampler = core.IndexSampler(len(utility_prompts), int(a.utility_batch_size), int(a.utility_seed) + 51005)

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    core.write_json(out_dir / "locality_receipt.json", locality_receipt)
    core.write_json(out_dir / "utility_receipt.json", utility_receipt)
    core.write_json(out_dir / "unknown_before.json", unknown_before)
    core.write_json(out_dir / "benchmark_pair_before.json", benchmark_before)

    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(1, int(a.steps) + 1), desc="MCF full-block Unknown + locality"):
            forget_idx = forget_sampler.next()
            local_idx = local_sampler.next()
            utility_idx = utility_sampler.next()

            batch = [train_instances[i] for i in forget_idx]
            local_batch = [locality_prompts[i] for i in local_idx]
            utility_batch = [utility_prompts[i] for i in utility_idx]

            opt.zero_grad(set_to_none=True)
            neutral_nll, sensitive_nll = unknown_rep._unknown_forward(
                model, tok, batch, device, llama_like
            )
            forget_loss, forget_margins = unknown_rep.unknown_margin_loss(
                sensitive_nll, neutral_nll, a.forget_margin
            )

            current_h, current_local_z = _final_hidden_and_logits(
                model, tok, local_batch, device
            )
            ref_h = base_local_h[local_idx].to(device=device, dtype=torch.float32)
            ref_local_z = base_local_z[local_idx].to(device=device, dtype=torch.float32)
            hidden_loss = locality_hidden_loss(current_h, ref_h)
            local_kl_loss = locality_kl(current_local_z, ref_local_z)

            utility_logits = wikipedia_utility._forward_prompt_logits(
                model, tok, utility_batch, device
            )
            wiki_loss = wikipedia_utility.utility_kl(
                utility_logits, utility_base_logits[utility_idx]
            )
            delta_f2 = legacy_rep.parameter_delta_f2(trainable_params, initial_params)

            total = (
                float(a.forget_weight) * forget_loss
                + float(a.locality_hidden_weight) * hidden_loss
                + float(a.locality_kl_weight) * local_kl_loss
                + float(a.utility_kl_weight) * wiki_loss
                + float(a.delta_weight) * delta_f2
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite total loss at step {step}")
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, float(a.grad_clip)) if a.grad_clip > 0 else None
            if grad_norm is not None and not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at step {step}")
            opt.step()

            if step == 1 or step % int(a.check_every) == 0 or step == int(a.steps):
                row = {
                    "step": int(step),
                    "total_loss": float(total.detach().cpu()),
                    "forget_loss": float(forget_loss.detach().cpu()),
                    "locality_hidden_loss": float(hidden_loss.detach().cpu()),
                    "locality_kl_loss": float(local_kl_loss.detach().cpu()),
                    "wikipedia_kl_loss": float(wiki_loss.detach().cpu()),
                    "delta_phi_frobenius_norm": float(delta_f2.detach().sqrt().cpu()),
                    "batch_forget_min_margin": float(forget_margins.min().detach().cpu()),
                    "official_neighborhoods_seen": 0,
                    "official_paraphrases_seen": 0,
                    "benchmark_retain_seen": 0,
                    "target_new_used_in_loss": False,
                    "lora_used": False,
                }
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()

    del opt
    model.eval()
    unknown_after = unknown_rep.evaluate_unknown_diagnostics(
        model, tok, unknown_instances, device, llama_like, a.batch_size, a.forget_margin
    )
    benchmark_after = unknown_rep.evaluate_benchmark_pair_diagnostics(
        model, tok, benchmark_instances, device, llama_like,
        a.batch_size, a.benchmark_pair_margin
    )
    locality_after = evaluate_locality_drift(
        model, tok, locality_prompts, base_local_h, base_local_z,
        device, int(a.locality_cache_batch_size)
    )
    utility_post = wikipedia_utility.evaluate_utility_kl(
        model, tok, utility_prompts, utility_base_logits,
        device, int(a.utility_cache_batch_size)
    )
    with torch.no_grad():
        final_delta_f2 = legacy_rep.parameter_delta_f2(trainable_params, initial_params)

    if model.get_input_embeddings().weight.requires_grad:
        raise RuntimeError("input embeddings became trainable")
    if model.get_output_embeddings().weight.requires_grad:
        raise RuntimeError("LM head became trainable")

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    summary = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "source_model_path": str(Path(a.model_path).resolve()),
        "source_is_intended_base_model": True,
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "neutral_answer": str(a.neutral_answer),
        "repair_scope": a.repair_scope,
        "active_positions": active_positions,
        "active_count": int(len(active_positions)),
        "training_positions": train_positions,
        "architecture": trainable_summary,
        "lora_used": False,
        "low_rank_adapter_used": False,
        "input_embeddings_trainable": False,
        "lm_head_trainable": False,
        "trainable_component": "full final transformer decoder block only",
        "forget_definition": "ReLU(m-(NLL(target_true_sensitive)-NLL(Unknown)))^2",
        "target_new_used_in_loss": False,
        "locality_definition": "Base hidden-state MSE plus Base next-token KL on training-visible relation-template donor-subject prompts",
        "official_neighborhoods_seen": 0,
        "official_paraphrases_seen": 0,
        "benchmark_retain_seen": 0,
        "weights": {
            "forget": float(a.forget_weight),
            "locality_hidden": float(a.locality_hidden_weight),
            "locality_kl": float(a.locality_kl_weight),
            "wikipedia_kl": float(a.utility_kl_weight),
            "delta": float(a.delta_weight),
        },
        "steps": int(a.steps),
        "lr": float(a.lr),
        "unknown_before": unknown_before,
        "unknown_after": unknown_after,
        "benchmark_pair_before": benchmark_before,
        "benchmark_pair_after": benchmark_after,
        "locality_after": locality_after,
        "utility_post_kl": utility_post,
        "delta_phi_frobenius_norm": float(final_delta_f2.detach().sqrt().cpu()),
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "fullblock_locality_summary.json", summary)
    core.write_json(out_dir / "unknown_after.json", unknown_after)
    core.write_json(out_dir / "benchmark_pair_after.json", benchmark_after)
    core.write_json(out_dir / "locality_after.json", locality_after)
    core.write_json(out_dir / "utility_post_kl.json", utility_post)

    print("Full-block Unknown+locality checkpoint:", ckpt)
    print("LoRA used: False")
    print("Input embeddings trainable: False")
    print("LM head trainable: False")
    print("Trainable component: full final decoder block only")
    print("Unknown before:", unknown_before)
    print("Unknown after:", unknown_after)
    print("Benchmark pair before:", benchmark_before)
    print("Benchmark pair after:", benchmark_after)
    print("Locality after:", locality_after)
    print("Wikipedia utility post KL:", utility_post)
    print("Delta Phi_last Frobenius norm:", float(final_delta_f2.detach().sqrt().cpu()))
    print("Stage 2 was NOT run.")


if __name__ == "__main__":
    main()
