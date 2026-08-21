#!/usr/bin/env python3
"""Canonical SURE Stage 1 with W1K external-Wikipedia utility preservation.

This is the shared MCF/ZsRE GA/KL Stage-1 architecture with one additional,
dataset-independent preservation term.  Exactly ``--utility-sample-size``
external Wikipedia predictor contexts (1,000 by default) are selected before
training.  Their Base next-token distributions are cached and the edited model
is penalized with KL(Base || current) on a deterministic utility minibatch at
every update.

The benchmark forget set remains unchanged.  No benchmark retain examples,
paraphrases/rephrases, neighborhoods/locality probes, generation probes, or PPL
texts are visible to this script.  After training, the normal vocabulary
restoration is applied: Base everywhere plus the trained sensitive rows only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

import build_sure_wikipedia_stats as wikipedia
import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import sure_stage1_gagd as shared


METHOD = "SURE-LM-canonical-stage1-gagd-W1K"
UTILITY_PROTOCOL = "sure_external_wikipedia_w1k_next_token_kl_v1"
DEFAULT_LR = 4e-5
DEFAULT_UTILITY_SAMPLE_SIZE = 1_000
DEFAULT_UTILITY_BATCH_SIZE = 4
DEFAULT_UTILITY_MAX_LENGTH = 128
DEFAULT_UTILITY_KL_WEIGHT = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("mcf", "zsre"), required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument(
        "--sensitive-field",
        choices=("target_true", "target_new"),
        default=None,
    )
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--emb-lm-lr", type=float, default=DEFAULT_LR)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--gd-weight", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="adamw")

    p.add_argument("--utility-wikipedia-dir", required=True)
    p.add_argument(
        "--utility-sample-size", type=int, default=DEFAULT_UTILITY_SAMPLE_SIZE
    )
    p.add_argument("--utility-batch-size", type=int, default=DEFAULT_UTILITY_BATCH_SIZE)
    p.add_argument("--utility-cache-batch-size", type=int, default=8)
    p.add_argument("--utility-max-length", type=int, default=DEFAULT_UTILITY_MAX_LENGTH)
    p.add_argument("--utility-seed", type=int, default=1)
    p.add_argument("--utility-exclude-first", type=int, default=20)
    p.add_argument("--utility-kl-weight", type=float, default=DEFAULT_UTILITY_KL_WEIGHT)

    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    a = p.parse_args()

    positive = (
        a.forget_num,
        a.steps,
        a.batch_size,
        a.cache_batch_size,
        a.emb_lm_lr,
        a.ga_weight,
        a.utility_sample_size,
        a.utility_batch_size,
        a.utility_cache_batch_size,
        a.utility_max_length,
        a.utility_kl_weight,
    )
    if any(float(value) <= 0 for value in positive):
        p.error("counts, LR, GA weight, and utility KL weight must be positive")
    if a.gd_weight < 0 or a.grad_clip < 0 or a.utility_exclude_first < 0:
        p.error("GD/clip/exclusion values must be non-negative")
    if a.utility_max_length < 8:
        p.error("utility-max-length must be at least 8 tokens")
    return a


def _first_sentence(text: str) -> str | None:
    """Return a deterministic non-trivial first sentence from an article."""
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    if not normalized:
        return None
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    for part in parts:
        candidate = part.strip()
        if len(candidate.split()) >= 8:
            return candidate
    return normalized if len(normalized.split()) >= 8 else None


def build_utility_prompts(
    tok: Any,
    wikipedia_dir: Path,
    *,
    sample_size: int,
    seed: int,
    exclude_first: int,
    max_length: int,
) -> tuple[List[str], Dict[str, Any]]:
    texts, metadata = wikipedia.load_wikipedia_train(wikipedia_dir)
    eligible, _target_rejected = wikipedia.eligible_document_indices(
        texts,
        exclude_first=exclude_first,
        excluded_casefold_substrings=(),
    )
    rng = random.Random(int(seed))
    order = list(eligible)
    rng.shuffle(order)

    prompts: List[str] = []
    selected_indices: List[int] = []
    for index in order:
        sentence = _first_sentence(str(texts[index]))
        if sentence is None:
            continue
        ids = tok.encode(sentence, add_special_tokens=False)
        if len(ids) < 4:
            continue
        ids = ids[: int(max_length)]
        prompt = tok.decode(ids, skip_special_tokens=True).strip()
        if not prompt:
            continue
        prompts.append(prompt)
        selected_indices.append(int(index))
        if len(prompts) >= int(sample_size):
            break
    if len(prompts) != int(sample_size):
        raise RuntimeError(
            f"requested {sample_size} Wikipedia utility sentences but built {len(prompts)}"
        )
    receipt = {
        "protocol": UTILITY_PROTOCOL,
        "dataset_source": metadata.get("dataset_source"),
        "dataset_row_count": metadata.get("dataset_row_count"),
        "sample_size": int(sample_size),
        "utility_seed": int(seed),
        "exclude_first": int(exclude_first),
        "max_length": int(max_length),
        "selected_document_indices": selected_indices,
        "benchmark_examples_used": 0,
        "ppl_prefix_rows_excluded": int(exclude_first),
    }
    return prompts, receipt


def _forward_prompt_logits(
    model: torch.nn.Module,
    tok: Any,
    prompts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    encoded = tok(
        list(prompts), padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    output = model(**encoded, use_cache=False)
    positions = encoded["attention_mask"].sum(dim=1) - 1
    rows = torch.arange(len(prompts), device=device)
    return output.logits[rows, positions, :]


@torch.no_grad()
def cache_utility_base_logits(
    model: torch.nn.Module,
    tok: Any,
    prompts: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    model.eval()
    for start in range(0, len(prompts), batch_size):
        logits = _forward_prompt_logits(
            model, tok, prompts[start : start + batch_size], device
        )
        # FP16 CPU storage halves the W1K cache footprint.  KL is evaluated in
        # FP32 after each minibatch is copied back to the device.
        chunks.append(logits.detach().to(dtype=torch.float16, device="cpu"))
    return torch.cat(chunks, dim=0).contiguous()


def utility_kl(current_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    cur_logp = F.log_softmax(current_logits.float(), dim=-1)
    ref_logp = F.log_softmax(
        base_logits.to(device=current_logits.device, dtype=torch.float32), dim=-1
    )
    ref_p = ref_logp.exp()
    return (ref_p * (ref_logp - cur_logp)).sum(dim=-1).mean()


@torch.no_grad()
def evaluate_utility_kl(
    model: torch.nn.Module,
    tok: Any,
    prompts: Sequence[str],
    base_logits: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    values: List[torch.Tensor] = []
    model.eval()
    for start in range(0, len(prompts), batch_size):
        current = _forward_prompt_logits(
            model, tok, prompts[start : start + batch_size], device
        )
        cur_logp = F.log_softmax(current.float(), dim=-1)
        ref_logp = F.log_softmax(
            base_logits[start : start + batch_size].to(
                device=current.device, dtype=torch.float32
            ),
            dim=-1,
        )
        per = (ref_logp.exp() * (ref_logp - cur_logp)).sum(dim=-1)
        values.append(per.detach().cpu())
    joined = torch.cat(values).float()
    return {
        "mean": float(joined.mean()),
        "p95": float(torch.quantile(joined, 0.95)),
        "max": float(joined.max()),
    }


def main() -> None:
    a = parse_args()
    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    sensitive_field = a.sensitive_field or core.sensitive_answer_field(a.dataset)
    records, manifest = shared._validate_locked_records(
        a.dataset,
        visible_path,
        manifest_path,
        a.seed,
        a.forget_num,
        sensitive_field,
    )

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

    cases = core.expand_sensitive_cases(
        records, tok, sensitive_field=sensitive_field, llama_like=llama_like
    )
    if not cases:
        raise RuntimeError("No sensitive PredictionCases were created")
    base_logits = core.cache_base_logits(
        model, tok, cases, device, batch_size=a.cache_batch_size
    )

    utility_prompts, utility_receipt = build_utility_prompts(
        tok,
        Path(a.utility_wikipedia_dir).resolve(),
        sample_size=a.utility_sample_size,
        seed=a.utility_seed,
        exclude_first=a.utility_exclude_first,
        max_length=a.utility_max_length,
    )
    print(
        f"Caching Base logits for {len(utility_prompts)} external Wikipedia utility sentences...",
        flush=True,
    )
    utility_base_logits = cache_utility_base_logits(
        model,
        tok,
        utility_prompts,
        device,
        a.utility_cache_batch_size,
    )

    summary, tied_info = gagd.configure_trainable(
        model, gagd.POST_TRAINING_RESTORE_MODE
    )
    params = gagd.unique_trainable_params(model)
    base_rows = gagd.snapshot_embedding_output_weights(tied_info)
    all_tids = core.official_target_ids(
        tok, cases, llama_like=llama_like, device=device
    )
    sensitive_ids = sorted(set(int(x) for x in all_tids.detach().cpu().tolist()))

    if a.optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=a.emb_lm_lr)
    elif a.optimizer == "adam":
        opt = torch.optim.Adam(params, lr=a.emb_lm_lr)
    else:
        opt = torch.optim.AdamW(params, lr=a.emb_lm_lr, weight_decay=0.0)

    forget_sampler = core.IndexSampler(len(cases), a.batch_size, a.seed)
    utility_sampler = core.IndexSampler(
        len(utility_prompts), a.utility_batch_size, a.utility_seed
    )
    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    core.write_json(out_dir / "utility_w1k_receipt.json", utility_receipt)

    model.train()
    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(1, a.steps + 1), desc=f"SURE {a.dataset} GA/KL + W1K"):
            idx = forget_sampler.next()
            batch = [cases[i] for i in idx]
            utility_idx = utility_sampler.next()
            utility_batch = [utility_prompts[i] for i in utility_idx]

            opt.zero_grad(set_to_none=True)
            logits = core.forward_last_logits(model, tok, batch, device)
            tids = core.official_target_ids(
                tok, batch, llama_like=llama_like, device=device
            )
            ga = core.ga_sensitive_logprob(logits, tids)
            gd = core.gd_non_sensitive_kl(logits, base_logits[idx], tids)
            utility_logits = _forward_prompt_logits(model, tok, utility_batch, device)
            ukl = utility_kl(utility_logits, utility_base_logits[utility_idx])
            total = (
                a.ga_weight * ga
                + a.gd_weight * gd
                + a.utility_kl_weight * ukl
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite Stage-1 loss at step {step}")
            total.backward()
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
                if a.grad_clip > 0
                else None
            )
            if grad_norm is not None and not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm at step {step}")
            opt.step()

            if step == 1 or step % 25 == 0 or step == a.steps:
                log_f.write(
                    json.dumps(
                        {
                            "step": step,
                            "total_loss": float(total.detach().cpu()),
                            "ga_sensitive_logprob": float(ga.detach().cpu()),
                            "gd_non_sensitive_kl": float(gd.detach().cpu()),
                            "wikipedia_utility_kl": float(ukl.detach().cpu()),
                            "ga_weight": float(a.ga_weight),
                            "gd_weight": float(a.gd_weight),
                            "utility_kl_weight": float(a.utility_kl_weight),
                            "emb_lm_lr": float(a.emb_lm_lr),
                            "benchmark_retain_seen": 0,
                            "heldout_probes_seen": 0,
                        }
                    )
                    + "\n"
                )
                log_f.flush()

    del opt
    restoration = core.restore_sensitive_rows_only(tied_info, base_rows, sensitive_ids)
    utility_post = evaluate_utility_kl(
        model,
        tok,
        utility_prompts,
        utility_base_logits,
        device,
        a.utility_cache_batch_size,
    )

    model.eval()
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    config: Dict[str, Any] = {
        "schema_version": 3,
        "method": METHOD,
        "dataset": a.dataset,
        "protocol": "sure_canonical_locked_direct_only_plus_external_w1k",
        "source_protocol": manifest.get("protocol"),
        "model_path": a.model_path,
        "training_visible_path": str(visible_path),
        "split_manifest": str(manifest_path),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "sensitive_answer_field": sensitive_field,
        "prediction_case_count": len(cases),
        "teacher_forcing": True,
        "benchmark_retain_seen": 0,
        "heldout_paraphrases_or_rephrases_seen": 0,
        "locality_or_neighborhood_seen": 0,
        "PPL_seen": False,
        "ga_loss": "mean(log p_theta(sensitive_token | direct_prompt, sensitive_prefix)); minimized",
        "gd_loss": "KL(base_non_sensitive || current_non_sensitive), sensitive token removed and renormalized",
        "gd_teacher_scope": "same direct training-visible forget PredictionCases only",
        "external_utility_loss": "KL(Base || current) on W1K external Wikipedia next-token distributions",
        "external_utility_protocol": UTILITY_PROTOCOL,
        "external_utility_sample_size": int(a.utility_sample_size),
        "external_utility_batch_size": int(a.utility_batch_size),
        "external_utility_kl_weight": float(a.utility_kl_weight),
        "external_utility_post_restoration_kl": utility_post,
        "external_utility_receipt": str((out_dir / "utility_w1k_receipt.json").resolve()),
        "steps": int(a.steps),
        "batch_size": int(a.batch_size),
        "cache_batch_size": int(a.cache_batch_size),
        "emb_lm_lr": float(a.emb_lm_lr),
        "ga_weight": float(a.ga_weight),
        "gd_weight": float(a.gd_weight),
        "optimizer": a.optimizer,
        "gradient_clip": float(a.grad_clip),
        "dtype": a.dtype,
        "device_map": a.device_map,
        "trainable_parameter_summary": asdict(summary),
        "vocabulary_restoration": restoration,
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "config_used.json", config)
    core.write_json(out_dir / "vocabulary_restoration.json", restoration)
    core.write_json(out_dir / "utility_post_restoration_kl.json", utility_post)
    print("W1K Stage-1 checkpoint:", ckpt)
    print("sensitive field:", sensitive_field)
    print("forget records:", a.forget_num)
    print("PredictionCases:", len(cases))
    print("Stage-1 LR:", a.emb_lm_lr)
    print("Wikipedia utility sentences:", len(utility_prompts))
    print("Wikipedia utility post-restoration KL:", utility_post)
    print("sensitive rows retained after restoration:", restoration["sensitive_row_count"])


if __name__ == "__main__":
    main()
