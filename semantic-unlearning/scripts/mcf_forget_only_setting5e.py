#!/usr/bin/env python3
"""Train the MCF Setting-5e stage using forget requests only.

This is the ZeroUnlearn-access variant used for a fair data-access comparison:
- sample forget records from the official second-half pool;
- expose only requested_rewrite prompts from the repair-visible MCF copy;
- use no MCF retain records, retain loss, or retain KL during training;
- compute the post-training vocabulary-row restoration policy from forget rows
  only; and
- save a normal Setting-5e checkpoint/config that can feed the sparse LM-head
  repair stage.

The 1000 MCF retain records are intentionally reserved for final evaluation.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--mcf-cache-path", required=True)
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
    p.add_argument("--post-training-true-alpha", type=float, default=0.75)
    p.add_argument("--post-training-new-true-alpha", type=float, default=0.75)
    p.add_argument("--post-training-new-retain-alpha", type=float, default=0.50)
    p.add_argument("--post-training-new-true-retain-alpha", type=float, default=0.25)
    p.add_argument(
        "--untie-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Break weight sharing between input embeddings and the LM head "
            "before training, so editing a token's answer-confidence role "
            "does not also perturb its input-embedding role in unrelated "
            "prompts. No-op if the model is already untied."
        ),
    )
    p.add_argument(
        "--train-scope",
        choices=("lm_head", "emb", "emb_lm"),
        default="lm_head",
        help=(
            "Which vocabulary role Stage 1 may update. lm_head (default) "
            "freezes input embeddings and trains only the LM head, matching "
            "the answer-token forget loss and Stage 2's own LM-head-only "
            "design. emb does the reverse. emb_lm trains both (previous "
            "behavior). Anything but emb_lm requires --untie-embeddings."
        ),
    )
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument(
        "--frequency-docs",
        type=int,
        default=5000,
        help=(
            "Wikipedia documents used to count subject/answer-token corpus "
            "frequency. Common tokens have their unique_target_new/"
            "unique_target_true post-training edit shrunk toward base, since "
            "they are more likely to be the correct answer elsewhere in "
            "ordinary text. Set 0 to disable."
        ),
    )
    p.add_argument(
        "--frequency-doc-start",
        type=int,
        default=20,
        help=(
            "Must stay >= 20: mcf_zero_unlearn_official_eval's official PPL "
            "is hardcoded to documents [:20]; counting frequencies on that "
            "same text would contaminate the frequency-cap protection."
        ),
    )
    p.add_argument("--frequency-cap-alpha", type=float, default=0.5)
    p.add_argument("--mcf-url", default=gagd.MCF_URL)
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.forget_num <= 0 or args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("forget-num, steps, and batch-size must be positive")
    if args.emb_lm_lr <= 0 or args.forget_weight <= 0:
        raise ValueError("learning rate and forget weight must be positive")
    if args.forget_margin < 0:
        raise ValueError("forget margin must be non-negative")
    for name in (
        "post_training_true_alpha",
        "post_training_new_true_alpha",
        "post_training_new_retain_alpha",
        "post_training_new_true_retain_alpha",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if args.frequency_docs < 0:
        raise ValueError("frequency-docs must be non-negative")
    if args.frequency_doc_start < 20:
        raise ValueError(
            "frequency-doc-start must be >= 20 to stay disjoint from the "
            "official PPL evaluation's first 20 documents"
        )
    if args.frequency_cap_alpha < 0:
        raise ValueError("frequency-cap-alpha must be non-negative")

    gagd.set_seed(args.seed)
    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)

    output_dir = gagd.resolve_output_path(args.output_dir)
    mode_dir = output_dir / gagd.POST_TRAINING_RESTORE_MODE
    checkpoint_dir = mode_dir / "checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build exactly the official forget sample while requesting zero MCF retain
    # records. The source file is the sanitized repair-visible MCF copy, so the
    # forget examples contain requested_rewrite only and no paraphrases.
    data_args = argparse.Namespace(
        mcf_cache_path=args.mcf_cache_path,
        mcf_url=args.mcf_url,
        forget_num=args.forget_num,
        retain_num=0,
        seed=args.seed,
        mcf_sample_mode="official",
        mcf_answer_field="target_new",
    )
    forget, retain = gagd.load_mcf(data_args)
    if len(forget) != args.forget_num:
        raise RuntimeError(
            f"Expected {args.forget_num} forget records, got {len(forget)}"
        )
    if retain:
        raise RuntimeError("Forget-only Setting 5e unexpectedly loaded MCF retain records")
    if any(example.paraphrase_prompts for example in forget):
        raise RuntimeError("Repair-visible MCF still exposes paraphrases during Stage 1")

    model_args = argparse.Namespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(model_args, for_training=True)
    was_tied = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    untied_now = gagd.untie_embeddings(model) if args.untie_embeddings else False
    print(
        f"Embedding/lm_head tied before this run: {was_tied}; "
        f"untied now: {untied_now}"
    )

    vocab_size = model.get_input_embeddings().weight.shape[0]
    frequency_docs = gagd.load_wiki_frequency_documents(
        args.wikidata_dir, args.frequency_doc_start, args.frequency_docs
    )
    if args.frequency_docs > 0 and not frequency_docs:
        print(
            f"WARNING: --wikidata-dir {args.wikidata_dir!r} has no documents "
            f"at/after index {args.frequency_doc_start}; post-training edits "
            "will not be frequency-capped."
        )
    token_frequencies = (
        gagd.token_frequency_counts(tok, frequency_docs, vocab_size)
        if frequency_docs
        else None
    )
    print(f"Frequency-cap corpus: {len(frequency_docs)} documents")

    summary, tied_info = gagd.configure_trainable(
        model, gagd.POST_TRAINING_RESTORE_MODE
    )
    # The forget loss is computed purely on answer tokens, so the LM head --
    # "how confident to be in this token as an answer" -- is the role it
    # actually needs. The input embedding's gradient path to that loss is
    # indirect, and moving it also shifts what the token means as
    # subject/context in unrelated neighborhood prompts. Freezing it is a
    # stronger guarantee than training it and restoring rows afterward.
    if args.train_scope != "emb_lm":
        if not untied_now and was_tied:
            raise ValueError(
                f"--train-scope {args.train_scope} requires untied embeddings; "
                "pass --untie-embeddings (a tied model shares one weight, so "
                "freezing one role would freeze both)"
            )
        frozen = (
            model.get_input_embeddings()
            if args.train_scope == "lm_head"
            else model.get_output_embeddings()
        )
        frozen.weight.requires_grad_(False)
    params = gagd.unique_trainable_params(model)
    print(
        f"Stage 1 train scope: {args.train_scope} "
        f"({len(params)} trainable tensor(s))"
    )
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.emb_lm_lr)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=args.emb_lm_lr)
    else:
        optimizer = torch.optim.AdamW(params, lr=args.emb_lm_lr, weight_decay=0.0)

    base_rows = gagd.snapshot_embedding_output_weights(tied_info)
    row_groups = gagd.collect_post_training_token_groups(tok, forget, [])
    sampler = gagd.EpochBatchSampler(forget, args.batch_size, args.seed)
    device = gagd.first_device(model)

    mode_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    with (mode_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(1, args.steps + 1), desc="forget-only Setting5e"):
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
                    f"Non-finite forget-only loss at step {step}: {float(total.detach().cpu())}"
                )
            total.backward()
            grad_norm = None
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient norm at step {step}: {float(grad_norm.detach().cpu())}"
                    )
            optimizer.step()
            log_f.write(
                json.dumps(
                    {
                        "step": step,
                        "total_loss": float(total.detach().cpu()),
                        "forget_margin_loss": float(margin_res.loss.detach().cpu()),
                        "forget_target_new_nll": float(margin_res.target_new_nll.detach().cpu()),
                        "forget_target_true_nll": float(margin_res.target_true_nll.detach().cpu()),
                        "retain_loss": None,
                        "kl_retain_loss": None,
                        "benchmark_retain_examples_seen": 0,
                        "gradient_norm_before_clip": (
                            float(grad_norm.detach().cpu()) if grad_norm is not None else None
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
        true_alpha=args.post_training_true_alpha,
        new_true_alpha=args.post_training_new_true_alpha,
        new_retain_alpha=args.post_training_new_retain_alpha,
        new_true_retain_alpha=args.post_training_new_true_retain_alpha,
        token_frequencies=token_frequencies,
        frequency_cap_alpha=args.frequency_cap_alpha,
    )
    row_policy = gagd.post_training_policy_report(
        tok,
        row_groups,
        applied_counts,
        true_alpha=args.post_training_true_alpha,
        new_true_alpha=args.post_training_new_true_alpha,
        new_retain_alpha=args.post_training_new_retain_alpha,
        new_true_retain_alpha=args.post_training_new_true_retain_alpha,
    )
    row_policy["benchmark_retain_examples_used"] = 0
    row_policy["policy_basis"] = "forget_rows_only"
    write_json(mode_dir / "post_training_row_policy.json", row_policy)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tok.save_pretrained(checkpoint_dir)

    config: Dict[str, Any] = {
        "schema_version": 1,
        "dataset": "mcf",
        "mode": gagd.POST_TRAINING_RESTORE_MODE,
        "protocol": "zerounlearn_data_access_forget_only",
        "model_path": args.model_path,
        "mcf_cache_path": args.mcf_cache_path,
        "mcf_sample_mode": "official",
        "seed": args.seed,
        "forget_num": args.forget_num,
        "retain_num": 0,
        "benchmark_retain_examples_used_during_training": 0,
        "benchmark_retain_used_for_gradient": False,
        "benchmark_retain_used_for_row_policy": False,
        "paraphrases_used_during_training": False,
        "forget_loss_type": "mcf_margin",
        "forget_margin": args.forget_margin,
        "forget_weight": args.forget_weight,
        "retain_weight": 0.0,
        "kl_retain_weight": 0.0,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "emb_lm_lr": args.emb_lm_lr,
        "optimizer": args.optimizer,
        "sampling_strategy": "epoch",
        "dtype": args.dtype,
        "device_map": args.device_map,
        "post_training_true_alpha": args.post_training_true_alpha,
        "post_training_new_true_alpha": args.post_training_new_true_alpha,
        "post_training_new_retain_alpha": args.post_training_new_retain_alpha,
        "post_training_new_true_retain_alpha": args.post_training_new_true_retain_alpha,
        "embeddings_tied_before_run": was_tied,
        "embeddings_untied_this_run": untied_now,
        "train_scope": args.train_scope,
        "wikidata_dir": args.wikidata_dir,
        "frequency_docs_requested": args.frequency_docs,
        "frequency_docs_used": len(frequency_docs),
        "frequency_doc_start": args.frequency_doc_start,
        "frequency_cap_alpha": args.frequency_cap_alpha,
        "trainable_parameter_summary": asdict(summary),
        "checkpoint": str(checkpoint_dir),
    }
    write_json(output_dir / "config_used.json", config)
    write_json(mode_dir / "config_used.json", config)

    print(f"Forget-only Setting 5e checkpoint: {checkpoint_dir}")
    print("MCF retain examples used during training: 0")


if __name__ == "__main__":
    main()
