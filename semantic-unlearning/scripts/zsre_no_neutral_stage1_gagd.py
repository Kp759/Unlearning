#!/usr/bin/env python3
"""No-neutral ZsRE SURE Stage 1: Emb+LM GA/GD on the locked forget split.

GA is applied to each teacher-forced original sensitive answer token y:

    L_GA = log p_theta(y | x, y_<t)

and the total objective is minimized, so p_theta(y) is driven downward.

GD preserves the base model's distribution over every *non-sensitive* vocabulary
item at that same direct-rewrite decision.  The sensitive token is removed from
both distributions and the remainder is renormalized:

    L_GD = KL(p_base(-y) || p_theta(-y)).

This gives a true no-neutral GA/GD objective: there is no target_new, Unknown,
IDK, refusal, replacement answer, benchmark retain example, paraphrase,
locality probe, or PPL sample in Stage 1.  Transformer blocks remain frozen;
only input embeddings and lm_head train.  After training, vocabulary
restoration returns every non-sensitive embedding/output row to the base
snapshot and retains only sensitive-answer row displacement.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

import gagd_compare as gagd
import zsre_gagd_setting5e_active_repair as zsre_sure
import zsre_no_neutral_stage1_emb_lm as locked
import zsre_zero_unlearn_official_eval as zsre

PROTOCOL = "zsre_zerounlearn_forget_only_locked_no_neutral"
METHOD = "SURE-ZsRE-no-neutral-EmbLM-GAGD-Stage1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--steps", type=int, default=450)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--emb-lm-lr", type=float, default=7.5e-5)
    p.add_argument("--ga-weight", type=float, default=1.75)
    p.add_argument("--gd-weight", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ga_sensitive_logprob(logits: torch.Tensor, tids: torch.Tensor) -> torch.Tensor:
    """Minimizing this is gradient ascent on target-token NLL."""
    rows = torch.arange(logits.shape[0], device=logits.device)
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp[rows, tids].mean()


def gd_non_sensitive_kl(
    current_logits: torch.Tensor,
    base_logits: torch.Tensor,
    tids: torch.Tensor,
) -> torch.Tensor:
    """KL(base || current) after removing each row's sensitive token."""
    cur = current_logits.float()
    ref = base_logits.to(device=cur.device, dtype=torch.float32)
    if cur.shape != ref.shape:
        raise ValueError("current/base logit shapes differ")
    if cur.ndim != 2:
        raise ValueError("expected [batch,vocab] logits")
    bsz, vocab = cur.shape
    rows = torch.arange(bsz, device=cur.device)
    mask = torch.ones((bsz, vocab), dtype=torch.bool, device=cur.device)
    mask[rows, tids] = False
    # Exactly one vocabulary item is excluded per row, so reshape is valid.
    cur_rest = cur[mask].view(bsz, vocab - 1)
    ref_rest = ref[mask].view(bsz, vocab - 1)
    cur_logp = F.log_softmax(cur_rest, dim=-1)
    ref_logp = F.log_softmax(ref_rest, dim=-1)
    ref_p = ref_logp.exp()
    return (ref_p * (ref_logp - cur_logp)).sum(dim=-1).mean()


class IndexSampler:
    def __init__(self, total: int, batch_size: int, seed: int):
        if total <= 0 or batch_size <= 0:
            raise ValueError("sampler requires positive total and batch size")
        self.total = int(total)
        self.batch_size = min(int(batch_size), self.total)
        self.rng = random.Random(seed)
        self.order: List[int] = []
        self.cursor = 0

    def next(self) -> List[int]:
        result: List[int] = []
        while len(result) < self.batch_size:
            if self.cursor >= len(self.order):
                self.order = list(range(self.total))
                self.rng.shuffle(self.order)
                self.cursor = 0
            take = min(self.batch_size - len(result), len(self.order) - self.cursor)
            result.extend(self.order[self.cursor:self.cursor + take])
            self.cursor += take
        return result


@torch.no_grad()
def cache_base_logits(
    model: torch.nn.Module,
    tok: Any,
    cases: Sequence[zsre.PredictionCase],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Cache immutable pre-training logits on CPU in FP32."""
    cached: List[torch.Tensor] = []
    model.eval()
    for start in tqdm(range(0, len(cases), batch_size), desc="cache ZsRE base logits"):
        batch = cases[start:start + batch_size]
        z = locked.forward_logits(model, tok, batch, device)
        cached.append(z.detach().float().cpu())
    return torch.cat(cached, dim=0)


def main() -> None:
    a = parse_args()
    if a.steps <= 0 or a.batch_size <= 0 or a.cache_batch_size <= 0:
        raise ValueError("steps and batch sizes must be positive")
    if a.emb_lm_lr <= 0 or a.ga_weight <= 0 or a.gd_weight < 0:
        raise ValueError("lr/ga-weight must be positive and gd-weight non-negative")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records = locked.load_locked(visible_path, manifest_path, a.seed, a.forget_num)

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
    llama_like = zsre.is_llama_like(model, tok)
    cases = locked.rewrite_cases(records, tok, llama_like)

    # Cache the frozen base teacher before any parameter update.  This uses only
    # the same training-visible direct forget prompts.
    base_logits = cache_base_logits(
        model, tok, cases, device, batch_size=a.cache_batch_size
    )

    summary, tied_info = gagd.configure_trainable(model, gagd.POST_TRAINING_RESTORE_MODE)
    params = gagd.unique_trainable_params(model)
    base_rows = gagd.snapshot_embedding_output_weights(tied_info)
    all_tids = locked.target_ids(tok, cases, llama_like, device)
    sensitive_ids = sorted(set(int(x) for x in all_tids.detach().cpu().tolist()))
    before_correct = locked.count_correct(
        model, tok, cases, llama_like, device, max(1, a.cache_batch_size)
    )

    if a.optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=a.emb_lm_lr)
    elif a.optimizer == "adam":
        opt = torch.optim.Adam(params, lr=a.emb_lm_lr)
    else:
        opt = torch.optim.AdamW(params, lr=a.emb_lm_lr, weight_decay=0.0)

    sampler = IndexSampler(len(cases), a.batch_size, a.seed)
    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(1, a.steps + 1), desc="ZsRE no-neutral Emb+LM GA/GD"):
            idx = sampler.next()
            batch = [cases[i] for i in idx]
            opt.zero_grad(set_to_none=True)
            z = locked.forward_logits(model, tok, batch, device)
            tids = locked.target_ids(tok, batch, llama_like, device)
            ref = base_logits[idx]
            ga = ga_sensitive_logprob(z, tids)
            gd = gd_non_sensitive_kl(z, ref, tids)
            total = a.ga_weight * ga + a.gd_weight * gd
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite GA/GD loss at step {step}")
            total.backward()
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
                if a.grad_clip > 0
                else None
            )
            opt.step()

            if step == 1 or step % 25 == 0 or step == a.steps:
                row = {
                    "step": step,
                    "total_loss": float(total.detach().cpu()),
                    "ga_sensitive_logprob": float(ga.detach().cpu()),
                    "ga_sensitive_nll": float((-ga).detach().cpu()),
                    "gd_non_sensitive_kl": float(gd.detach().cpu()),
                    "ga_weight": float(a.ga_weight),
                    "gd_weight": float(a.gd_weight),
                    "benchmark_retain_seen": 0,
                    "rephrases_seen": 0,
                    "locality_seen": 0,
                    "target_new_seen": False,
                    "neutral_target_seen": False,
                    "gradient_norm_before_clip": (
                        None if grad_norm is None else float(grad_norm.detach().cpu())
                    ),
                }
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()

    del opt
    restore_report = locked.restore_sensitive_rows_only(
        tied_info, base_rows, sensitive_ids
    )
    model.eval()
    after_correct = locked.count_correct(
        model, tok, cases, llama_like, device, max(1, a.cache_batch_size)
    )

    zsre_sure.save_checkpoint(model, tok, ckpt)
    config = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "model_path": a.model_path,
        "training_visible_path": str(visible_path),
        "split_manifest": str(manifest_path),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "benchmark_retain_seen": 0,
        "rephrases_seen": 0,
        "locality_seen": 0,
        "PPL_seen": False,
        "target_new_seen": False,
        "Unknown_used": False,
        "IDK_used": False,
        "replacement_target_used": False,
        "teacher_forcing": True,
        "ga_loss": "mean(log p_theta(sensitive_token)); minimized",
        "gd_loss": "KL(base_non_sensitive || current_non_sensitive), sensitive token removed and renormalized",
        "gd_teacher_scope": "same 50 direct training-visible forget rewrites only",
        "steps": int(a.steps),
        "batch_size": int(a.batch_size),
        "cache_batch_size": int(a.cache_batch_size),
        "emb_lm_lr": float(a.emb_lm_lr),
        "ga_weight": float(a.ga_weight),
        "gd_weight": float(a.gd_weight),
        "optimizer": a.optimizer,
        "trainable_parameter_summary": asdict(summary),
        "visible_rewrite_correct_tokens_before": int(before_correct),
        "visible_rewrite_total_tokens": len(cases),
        "visible_rewrite_correct_tokens_after_restore": int(after_correct),
        "vocabulary_restoration": restore_report,
        "checkpoint": str(ckpt.resolve()),
    }
    write_json(out_dir / "config_used.json", config)
    write_json(out_dir / "vocabulary_restoration.json", restore_report)

    print("ZsRE no-neutral GA/GD Stage1 checkpoint:", ckpt)
    print(f"visible rewrite correct tokens: {before_correct} -> {after_correct} / {len(cases)}")
    print("Stage 1 trainables: input embeddings + LM head; transformer frozen")
    print("GA: original sensitive answer probability DOWN")
    print("GD: base non-sensitive distribution PRESERVED on same direct forget prompts")
    print("Stage 1 data access: 50 direct forget; 0 retain/rephrase/locality/PPL; Unknown/IDK/target_new=NO")


if __name__ == "__main__":
    main()
