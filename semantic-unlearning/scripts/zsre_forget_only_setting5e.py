#!/usr/bin/env python3
"""Train locked ZeroUnlearn-style ZsRE Setting 5e using forget requests only.

Data access:
- exactly 50 sampled forget records from the ZeroUnlearn second-half pool;
- requested_rewrite only;
- zero benchmark-retain records;
- zero rephrase/locality probes.

ZsRE semantics are mapped into the existing Setting-5e margin code as:
- internal target_new (unwanted) = original sensitive answer;
- internal target_true (desired) = neutral target ``Unknown``.

The transformer stays frozen. Input embeddings and the LM head are optimized,
then the standard overlap-aware post-training row restoration is applied. The
neutral ``Unknown`` token row is excluded from Stage-1 row groups so it returns
to its base value before the Stage-2 output-only repair, matching the existing
ZsRE SURE architecture while removing evaluation-conditioned data access.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

import gagd_compare as gagd
import zsre_gagd_setting5e_active_repair as zsre_sure
import zsre_zero_unlearn_official_eval as zsre


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--repair-visible-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--emb-lm-lr", type=float, default=1e-4)
    p.add_argument("--forget-weight", type=float, default=2.0)
    p.add_argument("--forget-margin", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="adamw")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device-map", choices=["single", "auto"], default="single")
    p.add_argument("--post-training-new-true-alpha", type=float, default=0.75)
    p.add_argument("--post-training-new-retain-alpha", type=float, default=0.50)
    p.add_argument("--post-training-new-true-retain-alpha", type=float, default=0.25)
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_locked_records(path: Path, forget_num: int) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("repair-visible ZsRE must be a JSON list of objects")
    if len(data) != forget_num:
        raise ValueError(f"expected {forget_num} forget records, got {len(data)}")
    for row in data:
        if row.get("paraphrase_prompts"):
            raise RuntimeError("Stage 1 received locked ZsRE paraphrases")
        if row.get("neighborhood_prompts"):
            raise RuntimeError("Stage 1 received locked ZsRE locality probes")
        rewrite = row.get("requested_rewrite")
        if not isinstance(rewrite, dict):
            raise RuntimeError("Stage 1 record lacks requested_rewrite")
        if rewrite.get("target_new", {}).get("str") != zsre.NEUTRAL_TARGET:
            raise RuntimeError("ZsRE neutral target is not Unknown")
    return data


def main() -> None:
    args = parse_args()
    if args.forget_num <= 0 or args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("forget-num, steps and batch-size must be positive")
    if args.emb_lm_lr <= 0 or args.forget_weight <= 0:
        raise ValueError("learning rate and forget weight must be positive")
    if args.forget_margin < 0:
        raise ValueError("forget margin must be non-negative")

    gagd.set_seed(args.seed)
    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)

    visible_path = Path(args.repair_visible_path).resolve()
    records = load_locked_records(visible_path, args.forget_num)

    model_args = argparse.Namespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(model_args, for_training=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    neutral_token_id = zsre.resolve_neutral_target_token_id(tok)
    forget_examples = zsre_sure.canonical_examples(records, tok)
    if any(example.paraphrase_prompts for example in forget_examples):
        raise RuntimeError("canonical Stage-1 examples unexpectedly expose paraphrases")

    summary, tied_info = gagd.configure_trainable(
        model, gagd.POST_TRAINING_RESTORE_MODE
    )
    params = gagd.unique_trainable_params(model)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.emb_lm_lr)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=args.emb_lm_lr)
    else:
        optimizer = torch.optim.AdamW(params, lr=args.emb_lm_lr, weight_decay=0.0)

    base_rows = gagd.snapshot_embedding_output_weights(tied_info)
    row_groups = gagd.collect_post_training_token_groups(
        tok,
        forget_examples,
        [],
        excluded_token_ids=[neutral_token_id],
    )
    sampler = gagd.EpochBatchSampler(forget_examples, args.batch_size, args.seed)
    device = gagd.first_device(model)

    output_dir = gagd.resolve_output_path(args.output_dir)
    mode_dir = output_dir / gagd.POST_TRAINING_RESTORE_MODE
    checkpoint_dir = mode_dir / "checkpoint"
    mode_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    with (mode_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(1, args.steps + 1), desc="ZsRE forget-only Setting5e"):
            batch = sampler.next_batch()
            optimizer.zero_grad(set_to_none=True)
            margin_res = gagd.mcf_margin_forget_loss(
                model,
                tok,
                batch,
                selected_token_ids=None,
                device=device,
                forget_margin=args.forget_margin,
                # ZsRE's adapter puts the sensitive answer in target_new and
                # the neutral "Unknown" in target_true (canonical_examples).
                sensitive_field="target_new",
            )
            total = args.forget_weight * margin_res.loss
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"non-finite ZsRE forget-only loss at step {step}: "
                    f"{float(total.detach().cpu())}"
                )
            total.backward()
            grad_norm = None
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"non-finite ZsRE gradient norm at step {step}"
                    )
            optimizer.step()
            log_f.write(
                json.dumps(
                    {
                        "step": step,
                        "total_loss": float(total.detach().cpu()),
                        "forget_margin_loss": float(margin_res.loss.detach().cpu()),
                        "sensitive_target_nll": float(margin_res.target_new_nll.detach().cpu()),
                        "neutral_target_nll": float(margin_res.target_true_nll.detach().cpu()),
                        "benchmark_retain_examples_seen": 0,
                        "paraphrases_seen": 0,
                        "locality_prompts_seen": 0,
                        "gradient_norm_before_clip": (
                            None if grad_norm is None else float(grad_norm.detach().cpu())
                        ),
                    }
                )
                + "\n"
            )

    del optimizer
    applied_counts = gagd.apply_post_training_row_restore(
        tied_info,
        base_rows,
        row_groups,
        new_true_alpha=args.post_training_new_true_alpha,
        new_retain_alpha=args.post_training_new_retain_alpha,
        new_true_retain_alpha=args.post_training_new_true_retain_alpha,
    )
    row_policy = gagd.post_training_policy_report(
        tok,
        row_groups,
        applied_counts,
        new_true_alpha=args.post_training_new_true_alpha,
        new_retain_alpha=args.post_training_new_retain_alpha,
        new_true_retain_alpha=args.post_training_new_true_retain_alpha,
    )
    row_policy.update(
        {
            "benchmark_retain_examples_used": 0,
            "policy_basis": "forget_requested_rewrite_rows_only",
            "neutral_token_id_excluded_and_restored_to_base": neutral_token_id,
        }
    )
    write_json(mode_dir / "post_training_row_policy.json", row_policy)

    zsre_sure.save_checkpoint(model, tok, checkpoint_dir)

    config: Dict[str, Any] = {
        "schema_version": 1,
        "dataset": "zsre",
        "mode": gagd.POST_TRAINING_RESTORE_MODE,
        "protocol": "zsre_zerounlearn_forget_only_locked_probes",
        "model_path": args.model_path,
        "repair_visible_path": str(visible_path),
        "seed": int(args.seed),
        "forget_num": int(args.forget_num),
        "retain_num": 0,
        "benchmark_retain_examples_used_during_training": 0,
        "paraphrases_used_during_training": False,
        "locality_used_during_training": False,
        "zsre_semantic_mapping": {
            "internal_target_new_unwanted": "ZsRE target_true original answer",
            "internal_target_true_desired": zsre.NEUTRAL_TARGET,
        },
        "neutral_target": {
            "text": zsre.NEUTRAL_TARGET,
            "token_id": neutral_token_id,
            "restored_to_base_after_stage1": True,
        },
        "forget_loss_type": "mcf_margin",
        "forget_margin": args.forget_margin,
        "forget_weight": args.forget_weight,
        "retain_weight": 0.0,
        "kl_retain_weight": 0.0,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "emb_lm_lr": args.emb_lm_lr,
        "optimizer": args.optimizer,
        "post_training_new_true_alpha": args.post_training_new_true_alpha,
        "post_training_new_retain_alpha": args.post_training_new_retain_alpha,
        "post_training_new_true_retain_alpha": args.post_training_new_true_retain_alpha,
        "trainable_parameter_summary": asdict(summary),
        "checkpoint": str(checkpoint_dir),
    }
    write_json(output_dir / "config_used.json", config)
    write_json(mode_dir / "config_used.json", config)

    print(f"ZsRE forget-only Setting 5e checkpoint: {checkpoint_dir}")
    print("Stage 1 data access: 50 forget requested_rewrite only; 0 retain/probes")


if __name__ == "__main__":
    main()
