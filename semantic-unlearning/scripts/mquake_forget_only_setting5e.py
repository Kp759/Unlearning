#!/usr/bin/env python3
"""Train locked ZeroUnlearn-style MQuAKE Setting 5e using forget requests only.

Data access:
- exactly the atomic requested_rewrite facts belonging to 50 sampled MQuAKE
  forget instances from the ZeroUnlearn second-half pool;
- zero benchmark-retain records;
- zero atomic natural-language questions;
- zero multi-hop questions;
- zero benchmark counterfactual target_new values.

MQuAKE semantics are mapped into the existing Setting-5e margin code as:
- internal target_new (unwanted) = original sensitive answer;
- internal target_true (desired) = neutral target ``Unknown``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch
from tqdm import tqdm

import gagd_compare as gagd
import mquake_gagd_setting5e_active_repair as mquake_sure
import mquake_zero_unlearn_official_eval as mquake


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--repair-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50, help="MQuAKE instance count")
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


def load_locked_records(
    visible_path: Path,
    manifest_path: Path,
    forget_num: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    data = json.loads(Path(visible_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("repair-visible MQuAKE must be a JSON list of atomic facts")
    if not isinstance(manifest, dict):
        raise ValueError("MQuAKE split manifest must be an object")
    if manifest.get("protocol") != "mquake_zerounlearn_forget_only_locked_probes":
        raise ValueError("unexpected MQuAKE split protocol")
    if int(manifest.get("seed", -1)) != int(seed):
        raise ValueError("MQuAKE split seed does not match training seed")
    sampling = manifest.get("sampling", {})
    if int(sampling.get("forget_num_instances", -1)) != int(forget_num):
        raise ValueError("MQuAKE manifest forget instance count does not match")
    expected_atomic = int(sampling.get("forget_atomic_fact_count", -1))
    if len(data) != expected_atomic:
        raise ValueError(
            f"expected {expected_atomic} flattened forget facts, got {len(data)}"
        )
    source_indices = {int(row["source_index"]) for row in data}
    if len(source_indices) != forget_num:
        raise ValueError(
            f"expected facts from {forget_num} forget instances, got {len(source_indices)}"
        )

    for row in data:
        forbidden = {
            "atomic_gen_prompt",
            "multihop_questions",
            "multihop_answer",
            "multihop_new_answer",
            "question",
            "questions",
        }
        leaked = forbidden.intersection(row)
        if leaked:
            raise RuntimeError(f"Stage 1 received held-out MQuAKE fields: {sorted(leaked)}")
        rewrite = row.get("requested_rewrite")
        if not isinstance(rewrite, Mapping):
            raise RuntimeError("Stage 1 MQuAKE record lacks requested_rewrite")
        if "question" in rewrite or "mquake_target_new" in rewrite:
            raise RuntimeError("Stage 1 received held-out MQuAKE rewrite fields")
        if rewrite.get("target_new", {}).get("str") != mquake.NEUTRAL_TARGET:
            raise RuntimeError("MQuAKE neutral target is not Unknown")
    return data, manifest


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
    manifest_path = Path(args.split_manifest).resolve()
    records, split_manifest = load_locked_records(
        visible_path, manifest_path, args.forget_num, args.seed
    )

    model_args = argparse.Namespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(model_args, for_training=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    neutral_token_id = mquake.resolve_neutral_target_token_id(tok)
    forget_examples = mquake_sure.canonical_examples(records, tok)
    if any(example.paraphrase_prompts for example in forget_examples):
        raise RuntimeError("canonical MQuAKE Stage-1 examples unexpectedly expose probes")

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
        for step in tqdm(range(1, args.steps + 1), desc="MQuAKE forget-only Setting5e"):
            batch = sampler.next_batch()
            optimizer.zero_grad(set_to_none=True)
            margin_res = gagd.mcf_margin_forget_loss(
                model,
                tok,
                batch,
                selected_token_ids=None,
                device=device,
                forget_margin=args.forget_margin,
            )
            total = args.forget_weight * margin_res.loss
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"non-finite MQuAKE forget-only loss at step {step}: "
                    f"{float(total.detach().cpu())}"
                )
            total.backward()
            grad_norm = None
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"non-finite MQuAKE gradient norm at step {step}"
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
                        "benchmark_retain_instances_seen": 0,
                        "atomic_questions_seen": 0,
                        "multihop_questions_seen": 0,
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
            "benchmark_retain_instances_used": 0,
            "policy_basis": "forget_requested_rewrite_rows_only",
            "neutral_token_id_excluded_and_restored_to_base": neutral_token_id,
        }
    )
    write_json(mode_dir / "post_training_row_policy.json", row_policy)

    mquake_sure.save_checkpoint(model, tok, checkpoint_dir)

    config: Dict[str, Any] = {
        "schema_version": 1,
        "dataset": mquake.MQUAKE_FILENAME,
        "dataset_revision": mquake.MQUAKE_REV,
        "mode": gagd.POST_TRAINING_RESTORE_MODE,
        "protocol": "mquake_zerounlearn_forget_only_locked_probes",
        "model_path": args.model_path,
        "repair_visible_path": str(visible_path),
        "split_manifest": str(manifest_path),
        "seed": int(args.seed),
        "forget_num_instances": int(args.forget_num),
        "forget_num_atomic_facts": len(records),
        "retain_num_instances": 0,
        "benchmark_retain_used_during_training": False,
        "atomic_questions_used_during_training": False,
        "multihop_questions_used_during_training": False,
        "benchmark_counterfactual_targets_used_during_training": False,
        "mquake_semantic_mapping": {
            "internal_target_new_unwanted": "MQuAKE target_true original answer",
            "internal_target_true_desired": mquake.NEUTRAL_TARGET,
        },
        "neutral_target": {
            "text": mquake.NEUTRAL_TARGET,
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
        "split_sampling": split_manifest.get("sampling"),
    }
    write_json(output_dir / "config_used.json", config)
    write_json(mode_dir / "config_used.json", config)

    print(f"MQuAKE forget-only Setting 5e checkpoint: {checkpoint_dir}")
    print(
        f"Stage 1 data access: {args.forget_num} forget instances / "
        f"{len(records)} atomic rewrites; 0 retain/questions"
    )


if __name__ == "__main__":
    main()
