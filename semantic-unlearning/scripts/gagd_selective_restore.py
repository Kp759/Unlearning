#!/usr/bin/env python3
"""Sixth GA/GD setting: selective embedding/lm_head training plus post-training restore.

The transformer backbone stays frozen. During training, only the selected MCF
subject/target-new/target-true vocabulary rows may update and only selected
answer-token positions contribute to the GA/GD losses. After training, the
same overlap-aware row projection used by the all-token restore setting is
applied:

* unique target-new rows keep the full learned update;
* target-new overlap rows keep configurable fractions of the learned update;
* unique target-true rows are set to 1.25x their base rows;
* subject-only, retain-only, target-true/retain, special, and unrelated rows
  return to the base model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

import gagd_compare as gagd


MODE = "emb_lm_selective_restore_post_training_true"
TRAINING_MODE = "emb_lm_selective_tokens"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=gagd.DEFAULT_MODEL_PATH)
    p.add_argument("--output-dir", default="outputs/gagd_selective_restore")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--max-eval-examples", type=int, default=None)

    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--retain-batch-size", type=int, default=2)
    p.add_argument("--emb-lm-lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--forget-weight", type=float, default=6.0)
    p.add_argument("--retain-weight", type=float, default=0.25)
    p.add_argument(
        "--forget-loss-type",
        choices=["answer_nll", "mcf_margin", "zerounlearn_ga"],
        default="mcf_margin",
    )
    p.add_argument("--forget-margin", type=float, default=3.0)
    p.add_argument("--kl-retain-weight", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=2.0)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument(
        "--emb-lm-optimizer",
        choices=["sgd", "adam", "adamw", "adamw8bit"],
        default="adamw",
    )
    p.add_argument(
        "--sampling-strategy",
        choices=["epoch", "with_replacement"],
        default="epoch",
    )
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device-map", choices=["single", "auto"], default="single")
    p.add_argument("--save-model", action="store_true")

    p.add_argument("--mcf-url", default=gagd.MCF_URL)
    p.add_argument("--mcf-cache-path", default="data/multi_counterfact.json")
    p.add_argument(
        "--mcf-sample-mode",
        choices=["official", "first", "shuffled"],
        default="official",
    )
    p.add_argument(
        "--mcf-answer-field",
        choices=["target_new", "target_true"],
        default="target_new",
    )
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument(
        "--official-sample-mode", choices=["official", "first"], default="official"
    )
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--run-official-mcf-eval", action="store_true")

    p.add_argument(
        "--post-training-new-true-alpha",
        type=float,
        default=1.0,
        help="Fraction of the trained delta retained on target-new/target-true rows.",
    )
    p.add_argument(
        "--post-training-new-retain-alpha",
        type=float,
        default=1.0,
        help="Fraction of the trained delta retained on target-new/retain rows.",
    )
    p.add_argument(
        "--post-training-new-true-retain-alpha",
        type=float,
        default=1.0,
        help="Fraction retained on target-new/target-true/retain rows.",
    )

    # Compatibility fields consumed by imported gagd_compare helpers.
    p.set_defaults(
        dataset="mcf",
        lr=1e-5,
        full_lr=None,
        optimizer=None,
        full_optimizer=None,
        official_device_map="auto",
        forget_split="forget05",
        retain_split="retain95",
        semantic_token_json=None,
        selective_top_k=1000,
    )
    return p


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0 or args.batch_size <= 0 or args.retain_batch_size <= 0:
        raise ValueError("--steps and batch sizes must be positive")
    if args.emb_lm_lr <= 0:
        raise ValueError("--emb-lm-lr must be positive")
    if args.forget_weight <= 0 or args.retain_weight < 0:
        raise ValueError("forget weight must be positive and retain weight non-negative")
    if args.forget_margin < 0:
        raise ValueError("--forget-margin must be non-negative")
    for name in (
        "post_training_new_true_alpha",
        "post_training_new_retain_alpha",
        "post_training_new_true_retain_alpha",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")


def embedding_output_info(model: torch.nn.Module) -> Dict[str, Any]:
    inp = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if inp is None or out is None:
        raise ValueError("Model must expose input embeddings and output embeddings.")
    tied = inp.weight.data_ptr() == out.weight.data_ptr()
    return {
        "input_weight": inp.weight,
        "output_weight": out.weight,
        "tied": tied,
        "selected_mask": None,
    }


def write_config(path: Path, args: argparse.Namespace) -> None:
    config = vars(args).copy()
    config.update(
        {
            "mode": MODE,
            "training_mode": TRAINING_MODE,
            "post_training_true_alpha": gagd.POST_TRAINING_TRUE_ALPHA,
        }
    )
    gagd.write_json(path, config)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    gagd.set_seed(args.seed)
    gagd.require_cuda_if_needed(args.device_map)

    out_dir = gagd.resolve_output_path(args.output_dir)
    mode_dir = out_dir / MODE
    official_dir = out_dir / "official_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_dir.mkdir(parents=True, exist_ok=True)
    write_config(out_dir / "config_used.json", args)

    print("Loading MCF forget/retain data")
    forget, retain = gagd.load_data(args)
    if not forget or not retain:
        raise ValueError("Both forget and retain splits must be non-empty.")

    print("Loading base model for token selection and base metrics")
    base_model, tok = gagd.load_model_and_tokenizer(args, for_training=False)
    selected_ids = gagd.select_tokens(tok, forget, retain, args)
    if not selected_ids:
        raise ValueError("Selective token selection returned no rows.")
    gagd.write_json(
        out_dir / "selected_token_ids.json",
        {"token_ids": selected_ids, "n_selected_tokens": len(selected_ids)},
    )
    print(f"Selected embedding/lm_head rows: {len(selected_ids)}")
    base_metrics = gagd.evaluate(base_model, tok, forget, retain, args)
    gagd.write_json(out_dir / "base_metrics.json", base_metrics)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n=== Running sixth mode: {MODE} ===")
    gagd.set_seed(args.seed)
    model, tok = gagd.load_model_and_tokenizer(args, for_training=True)
    tied_info = embedding_output_info(model)
    originals = gagd.snapshot_embedding_output_weights(tied_info)
    groups = gagd.collect_post_training_token_groups(tok, forget, retain)

    # train_mode must not save yet: the final checkpoint belongs after projection.
    requested_save_model = bool(args.save_model)
    args.save_model = False
    summary = gagd.train_mode(
        model,
        tok,
        forget,
        retain,
        selected_ids,
        TRAINING_MODE,
        args,
        mode_dir,
    )
    args.save_model = requested_save_model

    applied_counts = gagd.apply_post_training_row_restore(
        tied_info,
        originals,
        groups,
        new_true_alpha=args.post_training_new_true_alpha,
        new_retain_alpha=args.post_training_new_retain_alpha,
        new_true_retain_alpha=args.post_training_new_true_retain_alpha,
    )
    policy = gagd.post_training_policy_report(
        tok,
        groups,
        applied_counts,
        new_true_alpha=args.post_training_new_true_alpha,
        new_retain_alpha=args.post_training_new_retain_alpha,
        new_true_retain_alpha=args.post_training_new_true_retain_alpha,
    )
    policy["mode"] = MODE
    policy["training_mode"] = TRAINING_MODE
    policy["selective_training_rows"] = len(selected_ids)
    policy["selective_training_rule"] = (
        "Only selected MCF subject/target-new/target-true rows could update during GA/GD; "
        "the overlap-aware projection was then applied inside that selected parameter space."
    )
    gagd.write_json(mode_dir / "post_training_row_policy.json", policy)
    print("Applied selective-row post-training restoration")

    metrics = gagd.evaluate(model, tok, forget, retain, args)
    metrics.update(
        {
            "mode": MODE,
            "training_mode": TRAINING_MODE,
            "n_selected_tokens": len(selected_ids),
            "forget_loss_type": args.forget_loss_type,
            "forget_margin": args.forget_margin,
            "sampling_strategy": args.sampling_strategy,
            "learning_rate": args.effective_lr,
            "optimizer": args.effective_optimizer,
            "post_training_new_true_alpha": args.post_training_new_true_alpha,
            "post_training_new_retain_alpha": args.post_training_new_retain_alpha,
            "post_training_new_true_retain_alpha": args.post_training_new_true_retain_alpha,
            **summary.__dict__,
        }
    )
    gagd.write_json(mode_dir / "metrics.json", metrics)

    if requested_save_model:
        checkpoint = mode_dir / "checkpoint"
        model.save_pretrained(checkpoint)
        tok.save_pretrained(checkpoint)

    if args.run_official_mcf_eval:
        official_dir.mkdir(parents=True, exist_ok=True)
        official_path = official_dir / f"{MODE}_official_eval.json"
        official_result = gagd.evaluate_loaded_model_official(
            method=MODE,
            model=model,
            tok=tok,
            model_dir=(mode_dir / "checkpoint") if requested_save_model else f"in-memory:{MODE}",
            mcf_path=args.mcf_cache_path,
            wikidata_dir=args.wikidata_dir,
            out_path=official_path,
            unlearn_num=args.forget_num,
            retain_num=args.retain_num,
            seed=args.seed,
            sample_mode=args.official_sample_mode,
            skip_ppl=args.skip_ppl,
        )
        gagd.write_json(out_dir / "official_eval_result.json", official_result)
        print(
            "Official result: "
            f"Eff={official_result['forget']['Eff']}, "
            f"Gen={official_result['forget']['Gen']}, "
            f"Spe={official_result['forget']['Spe']}, "
            f"PPL={official_result.get('forget_PPL')}"
        )

    print(f"Done. Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
