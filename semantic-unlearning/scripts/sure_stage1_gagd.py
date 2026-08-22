#!/usr/bin/env python3
"""Canonical SURE-LM Stage 1 for MCF, ZsRE, and MQuAKE.

All benchmarks use exactly the same mechanics:
  * direct forget prompts only;
  * teacher-forced sensitive-token GA;
  * same-prompt non-sensitive KL preservation to the frozen base distribution;
  * frozen transformer blocks;
  * tied input/output vocabulary matrix trainable;
  * full base restoration followed by reapplication of sensitive rows only.

The benchmark adapter only defines which answer is sensitive:
  * MCF    -> target_new
  * ZsRE   -> target_true
  * MQuAKE -> target_true
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch
from tqdm import tqdm

import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("mcf", "zsre", "mquake"), required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--emb-lm-lr", type=float, default=1e-4)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--gd-weight", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def _sensitive_answer_field(dataset: str) -> str:
    return "target_true" if dataset == "mquake" else core.sensitive_answer_field(dataset)


def _validate_locked_records(
    dataset: str,
    visible_path: Path,
    manifest_path: Path,
    seed: int,
    forget_num: int,
):
    records = json.loads(visible_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != forget_num:
        raise RuntimeError(f"Expected {forget_num} training-visible forget records")
    if int(manifest.get("seed", -1)) != seed:
        raise RuntimeError("Split manifest seed mismatch")
    sampling = manifest.get("sampling", {})
    if int(sampling.get("forget_num", -1)) != forget_num:
        raise RuntimeError("Split manifest forget count mismatch")
    expected_ids = [int(x) for x in sampling.get("forget_case_ids", [])]
    actual_ids = [int(r.get("case_id", -1)) for r in records]
    if expected_ids and actual_ids != expected_ids:
        raise RuntimeError("Training-visible IDs do not match split manifest")

    sensitive_field = _sensitive_answer_field(dataset)
    for index, record in enumerate(records):
        if record.get("paraphrase_prompts") or record.get("neighborhood_prompts"):
            raise RuntimeError(f"Record {index} exposes held-out probes")
        rr = record.get("requested_rewrite")
        if not isinstance(rr, dict):
            raise RuntimeError(f"Record {index} lacks requested_rewrite")
        block = rr.get(sensitive_field)
        if not isinstance(block, dict) or not block.get("str"):
            raise RuntimeError(f"Record {index} lacks sensitive {sensitive_field}")
        if dataset in ("zsre", "mquake") and "target_new" in rr:
            raise RuntimeError(
                f"Canonical {dataset} Stage 1 forbids target_new/neutral targets"
            )
    return records, manifest


def main() -> None:
    a = parse_args()
    if a.steps <= 0 or a.batch_size <= 0 or a.cache_batch_size <= 0:
        raise ValueError("steps and batch sizes must be positive")
    if a.emb_lm_lr <= 0 or a.ga_weight <= 0 or a.gd_weight < 0:
        raise ValueError("lr/GA must be positive and GD non-negative")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, manifest = _validate_locked_records(
        a.dataset, visible_path, manifest_path, a.seed, a.forget_num
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
    sensitive_field = _sensitive_answer_field(a.dataset)
    cases = core.expand_sensitive_cases(
        records, tok, sensitive_field=sensitive_field, llama_like=llama_like
    )
    if not cases:
        raise RuntimeError("No sensitive PredictionCases were created")

    # Immutable base teacher on the same direct forget token decisions only.
    base_logits = core.cache_base_logits(
        model, tok, cases, device, batch_size=a.cache_batch_size
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

    sampler = core.IndexSampler(len(cases), a.batch_size, a.seed)
    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(1, a.steps + 1), desc=f"SURE {a.dataset} canonical GA/GD"):
            idx = sampler.next()
            batch = [cases[i] for i in idx]
            opt.zero_grad(set_to_none=True)
            logits = core.forward_last_logits(model, tok, batch, device)
            tids = core.official_target_ids(
                tok, batch, llama_like=llama_like, device=device
            )
            ga = core.ga_sensitive_logprob(logits, tids)
            gd = core.gd_non_sensitive_kl(logits, base_logits[idx], tids)
            total = a.ga_weight * ga + a.gd_weight * gd
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
                            "ga_weight": float(a.ga_weight),
                            "gd_weight": float(a.gd_weight),
                            "gradient_norm_before_clip": (
                                None
                                if grad_norm is None
                                else float(grad_norm.detach().cpu())
                            ),
                            "benchmark_retain_seen": 0,
                            "heldout_probes_seen": 0,
                        }
                    )
                    + "\n"
                )
                log_f.flush()

    del opt
    restoration = core.restore_sensitive_rows_only(
        tied_info, base_rows, sensitive_ids
    )
    model.eval()
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    config: Dict[str, Any] = {
        "schema_version": 2,
        "method": "SURE-LM-canonical-stage1-gagd",
        "dataset": a.dataset,
        "protocol": "sure_canonical_locked_direct_only",
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
    print("Canonical Stage-1 checkpoint:", ckpt)
    print("dataset:", a.dataset)
    print("sensitive field:", config["sensitive_answer_field"])
    print("PredictionCases:", len(cases))
    print("sensitive rows retained after restoration:", restoration["sensitive_row_count"])


if __name__ == "__main__":
    main()
