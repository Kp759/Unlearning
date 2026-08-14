#!/usr/bin/env python3
"""MQuAKE collision-safe SURE Stage 1.

Training visibility remains the locked direct-forget artifact only.
Architecture:
  * transformer frozen;
  * input embeddings trainable during Stage 1, then fully restored to base;
  * LM head untied before training; base head frozen;
  * only a sparse FP32 delta for sensitive LM-head rows is trainable;
  * active-only GA suppresses a sensitive token only while it is still within
    ``active_margin`` of the best non-sensitive competitor;
  * non-sensitive GD matches the frozen base distribution at factual decisions;
  * context preservation makes the sensitive-row delta nearly null on ordinary
    non-answer positions from the SAME direct forget prompts;
  * no retain, PPL, atomic questions, multihop questions, target_new, Unknown,
    IDK, paraphrases, or locality data are visible.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, List

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

import gagd_compare as gagd
import mquake_forget_only_no_neutral as locked
import mquake_no_neutral_stage1_gagd as old
import mquake_zero_unlearn_official_eval as mq
from zsre_no_neutral_stage1_gagd import gd_non_sensitive_kl

PROTOCOL = "mquake_zerounlearn_forget_only_locked_no_neutral"
METHOD = "SURE-MQuAKE-collision-safe-Emb-plus-sparse-LM-GAGD"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--emb-lr", type=float, default=1e-4)
    p.add_argument("--row-lr", type=float, default=5e-3)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--gd-weight", type=float, default=1.0)
    p.add_argument("--active-margin", type=float, default=0.05)
    p.add_argument("--context-weight", type=float, default=1.0)
    p.add_argument("--context-batch-size", type=int, default=64)
    p.add_argument("--row-l2", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def direct_prompt(record: Any) -> str:
    rr = record["requested_rewrite"]
    return str(rr["prompt"]).format(str(rr["subject"]))


@torch.no_grad()
def collect_ordinary_base_hidden(model, tok, records, device) -> torch.Tensor:
    """Collect base final-layer states excluding each prompt's factual decision."""
    rows: List[torch.Tensor] = []
    model.eval()
    for r in tqdm(records, desc="cache ordinary direct-prompt hidden states"):
        enc = tok([direct_prompt(r)], return_tensors="pt").to(device)
        out = model(**enc, use_cache=False, output_hidden_states=True)
        h = out.hidden_states[-1][0].detach().float()
        length = int(enc["attention_mask"][0].sum().item())
        # Position length-1 predicts the first sensitive answer token and must
        # remain free to change. Earlier positions are ordinary prompt contexts.
        if length > 1:
            rows.append(h[: length - 1].cpu())
    if not rows:
        raise RuntimeError("no ordinary prompt hidden states were collected")
    return torch.cat(rows, dim=0)


def active_ga(logits: torch.Tensor, tids: torch.Tensor, margin: float):
    idx = torch.arange(logits.size(0), device=logits.device)
    zt = logits[idx, tids]
    with torch.no_grad():
        detached = logits.detach().clone()
        detached[idx, tids] = -torch.inf
        other = detached.max(dim=-1).values
        active = zt.detach() >= (other - float(margin))
    logp = F.log_softmax(logits.float(), dim=-1)[idx, tids]
    if bool(active.any()):
        loss = logp[active].mean()
    else:
        loss = logp.sum() * 0.0
    return loss, active, zt.detach(), other.detach()


def sample_context(h: torch.Tensor, rng: random.Random, n: int, device: torch.device):
    k = min(int(n), int(h.size(0)))
    ids = rng.sample(range(int(h.size(0))), k=k)
    return h[ids].to(device=device, dtype=torch.float32)


def main() -> None:
    a = args()
    gagd.set_seed(a.seed)
    vp, mp = Path(a.training_visible_path).resolve(), Path(a.split_manifest).resolve()
    records, man = locked.load_locked(vp, mp, a.seed, a.forget_num)

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
    llama_like = mq.is_llama_like(model, tok)
    cases = locked.all_cases(records, tok, llama_like)

    # Untie before any optimization: input-embedding and output-row gradients
    # can no longer contaminate the same physical weight row.
    in_w = model.get_input_embeddings().weight
    tied_before = bool(in_w.data_ptr() == model.get_output_embeddings().weight.data_ptr())
    out = locked.untie(model)
    out_w = out.weight
    if in_w.data_ptr() == out_w.data_ptr():
        raise RuntimeError("LM head remained tied")

    for p in model.parameters():
        p.requires_grad_(False)
    in_w.requires_grad_(True)

    # Base snapshots/caches are taken before training and use only locked direct
    # forget material.
    base_input = in_w.detach().cpu().clone()
    base_logits = old.cache(model, tok, cases, device, a.cache_batch_size)
    ordinary_h = collect_ordinary_base_hidden(model, tok, records, device)
    tids_all = locked.target_ids(tok, cases, llama_like, device)
    sensitive_ids = sorted(set(int(x) for x in tids_all.cpu().tolist()))
    row_index = torch.tensor(sensitive_ids, dtype=torch.long, device=device)
    base_sensitive_rows = out_w.index_select(0, row_index).detach().float().clone()

    delta = nn.Parameter(
        torch.zeros((len(sensitive_ids), int(out_w.shape[1])), device=device, dtype=torch.float32)
    )
    hook = locked.output_hook(out, sensitive_ids, delta)

    before = old.correct(model, tok, cases, llama_like, device, a.cache_batch_size)
    emb_opt = torch.optim.AdamW([in_w], lr=a.emb_lr, weight_decay=0.0)
    row_opt = torch.optim.AdamW([delta], lr=a.row_lr, weight_decay=0.0)
    sampler = old.Sampler(len(cases), a.batch_size, a.seed)
    context_rng = random.Random(a.seed + 700001)

    root = gagd.resolve_output_path(a.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    model.train()
    try:
        with (root / "train_log.jsonl").open("w", encoding="utf-8") as f:
            for step in tqdm(range(1, a.steps + 1), desc="MQuAKE collision-safe Stage1"):
                ix = sampler.next()
                batch = [cases[i] for i in ix]
                emb_opt.zero_grad(set_to_none=True)
                row_opt.zero_grad(set_to_none=True)

                z = locked.forward_logits(model, tok, batch, device).float()
                tids = locked.target_ids(tok, batch, llama_like, device)
                ga, active, zt, other = active_ga(z, tids, a.active_margin)
                gd = gd_non_sensitive_kl(z, base_logits[ix], tids)

                # The final checkpoint restores input embeddings to base, so the
                # relevant utility geometry is the BASE hidden-state geometry.
                hc = sample_context(ordinary_h, context_rng, a.context_batch_size, device)
                context_delta_logits = hc @ delta.T
                context_loss = context_delta_logits.pow(2).mean()
                row_reg = delta.pow(2).mean()
                loss = (
                    a.ga_weight * ga
                    + a.gd_weight * gd
                    + a.context_weight * context_loss
                    + a.row_l2 * row_reg
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_([in_w, delta], a.grad_clip)
                emb_opt.step()
                row_opt.step()

                if step == 1 or step % 25 == 0 or step == a.steps:
                    row = {
                        "step": step,
                        "loss": float(loss.detach()),
                        "active_ga_logprob": float(ga.detach()),
                        "active_cases_in_batch": int(active.sum().item()),
                        "gd_non_sensitive_kl": float(gd.detach()),
                        "context_sensitive_logit_mse": float(context_loss.detach()),
                        "row_l2": float(row_reg.detach()),
                        "delta_norm": float(delta.detach().norm()),
                        "grad_norm": float(gn.detach()),
                        "retain_seen": 0,
                        "atomic_questions_seen": 0,
                        "multihop_questions_seen": 0,
                        "PPL_seen": False,
                        "target_new_seen": False,
                        "Unknown_used": False,
                        "IDK_used": False,
                    }
                    f.write(json.dumps(row) + "\n")
                    f.flush()
    finally:
        hook.remove()

    # Final restoration/materialization: base transformer, base input embeddings,
    # base non-sensitive head rows, and only the learned sparse sensitive delta.
    with torch.no_grad():
        in_w.copy_(base_input.to(device=in_w.device, dtype=in_w.dtype))
        materialized = base_sensitive_rows + delta.detach()
        out_w.index_copy_(
            0,
            row_index.to(out_w.device),
            materialized.to(device=out_w.device, dtype=out_w.dtype),
        )
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    after = old.correct(model, tok, cases, llama_like, device, a.cache_batch_size)

    ckpt = root / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    config = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "seed": a.seed,
        "forget_instances": a.forget_num,
        "forget_atomic_facts": len(records),
        "direct_sensitive_token_cases": len(cases),
        "sensitive_rows": len(sensitive_ids),
        "ordinary_context_hidden_states": int(ordinary_h.size(0)),
        "hidden_size": int(ordinary_h.size(1)),
        "tied_before_training": tied_before,
        "tied_during_training": False,
        "transformer_trainable": False,
        "input_embeddings_trainable_during_stage1": True,
        "input_embeddings_fully_restored_after_stage1": True,
        "base_lm_head_frozen": True,
        "trainable_lm_component": "FP32 sparse sensitive-row delta",
        "non_sensitive_lm_rows_modified_final": 0,
        "loss": {
            "active_ga": "mean(log p sensitive) only while z_sensitive >= best_other-active_margin",
            "gd": "KL(base_non_sensitive || current_non_sensitive) at direct factual decisions",
            "context_preservation": "mean((H_base_ordinary @ delta_sensitive.T)^2) on non-answer positions from same direct forget prompts",
            "row_anchor": "mean(delta_sensitive^2)",
        },
        "hyperparameters": {
            "steps": a.steps,
            "batch_size": a.batch_size,
            "cache_batch_size": a.cache_batch_size,
            "emb_lr": a.emb_lr,
            "row_lr": a.row_lr,
            "ga_weight": a.ga_weight,
            "gd_weight": a.gd_weight,
            "active_margin": a.active_margin,
            "context_weight": a.context_weight,
            "context_batch_size": a.context_batch_size,
            "row_l2": a.row_l2,
            "grad_clip": a.grad_clip,
        },
        "correct_before": before,
        "correct_after_restoration": after,
        "sensitive_delta_fro_norm": float(delta.detach().norm().cpu()),
        "visibility_audit": {
            "retain_seen": 0,
            "atomic_questions_seen": 0,
            "multihop_questions_seen": 0,
            "PPL_seen": False,
            "target_new_seen": False,
            "Unknown_used": False,
            "IDK_used": False,
        },
        "split_sampling": man.get("sampling"),
        "checkpoint": str(ckpt.resolve()),
    }
    old.write(root / "config_used.json", config)
    print("===== MQuAKE COLLISION-SAFE STAGE1 =====")
    print("direct correct:", before, "->", after, "/", len(cases))
    print("sensitive rows:", len(sensitive_ids))
    print("ordinary context states:", tuple(ordinary_h.shape))
    print("sensitive delta norm:", float(delta.detach().norm().cpu()))
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
