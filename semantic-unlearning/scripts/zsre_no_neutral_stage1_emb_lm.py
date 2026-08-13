#!/usr/bin/env python3
"""SURE ZsRE Stage 1 with Emb+LM training and no replacement target.

Input data is the exact ZeroUnlearn-matched no-neutral locked split. Stage 1 sees
only the 50 direct requested_rewrite prompts and original sensitive target_true.
No target_new, Unknown, IDK, retain records, rephrases, locality, or PPL data are
loaded.

For every teacher-forced original-answer token y, minimize

    relu(z_y - stopgrad(max_{j != y} z_j) + margin)

so the sensitive answer loses its top-1 decision without choosing what replaces
it. Transformer blocks stay frozen; only input embeddings and LM head train.
After training, vocabulary restoration resets every embedding/output row to the
base snapshot except rows belonging to sensitive answer tokens.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

import gagd_compare as gagd
import zsre_gagd_setting5e_active_repair as zsre_sure
import zsre_zero_unlearn_official_eval as zsre

PROTOCOL = "zsre_zerounlearn_forget_only_locked_no_neutral"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--emb-lm-lr", type=float, default=1e-4)
    p.add_argument("--forget-weight", type=float, default=2.0)
    p.add_argument("--forget-margin", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_locked(visible_path: Path, manifest_path: Path, seed: int, forget_num: int) -> List[Dict[str, Any]]:
    records = json.loads(visible_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError("wrong no-neutral ZsRE protocol")
    if int(manifest.get("seed", -1)) != seed:
        raise RuntimeError("split manifest seed mismatch")
    if int(manifest.get("sampling", {}).get("forget_num", -1)) != forget_num:
        raise RuntimeError("split manifest forget count mismatch")
    if not isinstance(records, list) or len(records) != forget_num:
        raise RuntimeError(f"expected {forget_num} training-visible forget records")
    expected_ids = [int(i) for i in manifest["sampling"]["forget_case_ids"]]
    actual_ids = [int(r["case_id"]) for r in records]
    if actual_ids != expected_ids:
        raise RuntimeError("training-visible IDs do not match locked manifest")
    for r in records:
        if r.get("paraphrase_prompts") or r.get("neighborhood_prompts"):
            raise RuntimeError("Stage 1 received held-out ZsRE probes")
        rr = r.get("requested_rewrite")
        if not isinstance(rr, dict) or not rr.get("target_true", {}).get("str"):
            raise RuntimeError("Stage 1 record lacks sensitive target_true")
        if "target_new" in rr:
            raise RuntimeError("target_new/neutral target leaked into Stage 1")
    return records


def rewrite_cases(records: Sequence[Dict[str, Any]], tok: Any, llama_like: bool) -> List[zsre.PredictionCase]:
    return [
        case
        for record in records
        for case in zsre.expand_prediction_cases(
            record, tok, llama_like=llama_like, prompt_types=("rewrite",)
        )
    ]


def forward_logits(model, tok, cases: Sequence[zsre.PredictionCase], device: torch.device) -> torch.Tensor:
    enc = tok([c.prompt for c in cases], padding=True, return_tensors="pt").to(device)
    out = model(**enc, use_cache=False)
    pos = enc["attention_mask"].sum(dim=1) - 1
    rows = torch.arange(len(cases), device=device)
    return out.logits[rows, pos, :]


def target_ids(tok, cases: Sequence[zsre.PredictionCase], llama_like: bool, device: torch.device) -> torch.Tensor:
    return zsre.official_target_ids(
        tok,
        [c.target_text for c in cases],
        llama_like=llama_like,
        device=device,
    )


def suppression_loss(logits: torch.Tensor, tids: torch.Tensor, margin: float) -> torch.Tensor:
    rows = torch.arange(logits.shape[0], device=logits.device)
    sensitive = logits[rows, tids].float()
    with torch.no_grad():
        detached = logits.detach().float().clone()
        detached[rows, tids] = -torch.inf
        best_other = detached.max(dim=-1).values
    return F.relu(sensitive - best_other + margin).mean()


class CaseSampler:
    def __init__(self, cases: Sequence[zsre.PredictionCase], batch_size: int, seed: int):
        self.cases = list(cases)
        self.batch_size = min(int(batch_size), len(self.cases))
        self.rng = random.Random(seed)
        self.order: List[int] = []
        self.cursor = 0

    def next(self) -> List[zsre.PredictionCase]:
        batch: List[zsre.PredictionCase] = []
        while len(batch) < self.batch_size:
            if self.cursor >= len(self.order):
                self.order = list(range(len(self.cases)))
                self.rng.shuffle(self.order)
                self.cursor = 0
            take = min(self.batch_size - len(batch), len(self.order) - self.cursor)
            batch.extend(self.cases[i] for i in self.order[self.cursor:self.cursor + take])
            self.cursor += take
        return batch


@torch.no_grad()
def count_correct(model, tok, cases, llama_like, device, batch_size: int) -> int:
    total = 0
    for start in range(0, len(cases), batch_size):
        b = cases[start:start + batch_size]
        z = forward_logits(model, tok, b, device)
        tids = target_ids(tok, b, llama_like, device)
        total += int((z.argmax(dim=-1) == tids).sum().item())
    return total


@torch.no_grad()
def restore_sensitive_rows_only(tied_info: Dict[str, Any], base_rows: Dict[str, torch.Tensor], sensitive_ids: Sequence[int]) -> Dict[str, Any]:
    in_w = tied_info["input_weight"]
    out_w = tied_info["output_weight"]
    tied = bool(tied_info.get("tied"))
    ids = torch.tensor(sorted(set(int(i) for i in sensitive_ids)), dtype=torch.long, device=in_w.device)
    if not ids.numel():
        raise RuntimeError("no sensitive answer rows found for vocabulary restoration")

    trained_in = in_w.index_select(0, ids).detach().clone()
    trained_out = trained_in if tied else out_w.index_select(0, ids.to(out_w.device)).detach().clone()

    in_base = base_rows["input"].to(device=in_w.device, dtype=in_w.dtype)
    in_w.copy_(in_base)
    in_w.index_copy_(0, ids, trained_in)

    if not tied:
        out_ids = ids.to(out_w.device)
        out_base = base_rows["output"].to(device=out_w.device, dtype=out_w.dtype)
        out_w.copy_(out_base)
        out_w.index_copy_(0, out_ids, trained_out)

    return {
        "sensitive_row_count": int(ids.numel()),
        "sensitive_token_ids": [int(i) for i in ids.detach().cpu().tolist()],
        "all_non_sensitive_rows_restored_to_base": True,
        "tied_input_output": tied,
    }


def main() -> None:
    a = parse_args()
    if a.steps <= 0 or a.batch_size <= 0 or a.emb_lm_lr <= 0:
        raise ValueError("steps, batch-size, and emb-lm-lr must be positive")
    if a.forget_margin < 0 or a.forget_weight <= 0:
        raise ValueError("forget margin must be non-negative and weight positive")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records = load_locked(visible_path, manifest_path, a.seed, a.forget_num)

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

    cases = rewrite_cases(records, tok, llama_like)
    summary, tied_info = gagd.configure_trainable(model, gagd.POST_TRAINING_RESTORE_MODE)
    params = gagd.unique_trainable_params(model)
    base_rows = gagd.snapshot_embedding_output_weights(tied_info)

    sensitive_ids = sorted(set(
        int(x)
        for x in target_ids(tok, cases, llama_like, device).detach().cpu().tolist()
    ))
    before_correct = count_correct(model, tok, cases, llama_like, device, max(1, a.batch_size))

    if a.optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=a.emb_lm_lr)
    elif a.optimizer == "adam":
        opt = torch.optim.Adam(params, lr=a.emb_lm_lr)
    else:
        opt = torch.optim.AdamW(params, lr=a.emb_lm_lr, weight_decay=0.0)

    sampler = CaseSampler(cases, a.batch_size, a.seed)
    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(1, a.steps + 1), desc="ZsRE no-neutral Emb+LM Stage1"):
            batch = sampler.next()
            opt.zero_grad(set_to_none=True)
            z = forward_logits(model, tok, batch, device)
            tids = target_ids(tok, batch, llama_like, device)
            raw_loss = suppression_loss(z, tids, a.forget_margin)
            total = a.forget_weight * raw_loss
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite Stage1 loss at step {step}")
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, a.grad_clip) if a.grad_clip > 0 else None
            opt.step()
            if step == 1 or step % 25 == 0 or step == a.steps:
                row = {
                    "step": step,
                    "total_loss": float(total.detach().cpu()),
                    "sensitive_suppression_loss": float(raw_loss.detach().cpu()),
                    "benchmark_retain_seen": 0,
                    "rephrases_seen": 0,
                    "locality_seen": 0,
                    "target_new_seen": False,
                    "neutral_target_seen": False,
                    "gradient_norm_before_clip": None if grad_norm is None else float(grad_norm.detach().cpu()),
                }
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()

    del opt
    restore_report = restore_sensitive_rows_only(tied_info, base_rows, sensitive_ids)
    model.eval()
    after_correct = count_correct(model, tok, cases, llama_like, device, max(1, a.batch_size))

    zsre_sure.save_checkpoint(model, tok, ckpt)
    config = {
        "schema_version": 1,
        "method": "SURE-ZsRE-no-neutral-EmbLM-Stage1",
        "protocol": PROTOCOL,
        "model_path": a.model_path,
        "training_visible_path": str(visible_path),
        "split_manifest": str(manifest_path),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "benchmark_retain_seen": 0,
        "rephrases_seen": 0,
        "locality_seen": 0,
        "target_new_seen": False,
        "Unknown_used": False,
        "IDK_used": False,
        "replacement_target_used": False,
        "loss": "relu(sensitive_logit-stopgrad(best_other_logit)+margin)",
        "teacher_forcing": True,
        "steps": int(a.steps),
        "batch_size": int(a.batch_size),
        "emb_lm_lr": float(a.emb_lm_lr),
        "forget_weight": float(a.forget_weight),
        "forget_margin": float(a.forget_margin),
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

    print("ZsRE no-neutral Stage1 checkpoint:", ckpt)
    print(f"visible rewrite correct tokens: {before_correct} -> {after_correct} / {len(cases)}")
    print("Stage 1 trainables: input embeddings + LM head; transformer frozen")
    print("Stage 1 data access: 50 direct forget; 0 retain/rephrase/locality; Unknown/IDK/target_new=NO")


if __name__ == "__main__":
    main()
