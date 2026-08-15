#!/usr/bin/env python3
"""TOFU Setting-5e Stage 1 with 50 direct forget QAs and zero retain access.

This is the locked-data analogue of the existing TOFU ``emb_lm_all_tokens``
GA stage plus post-training restoration.  It never loads Hugging Face data:
the only data input is the builder-produced ``train_visible/forget.json``.

During GA the input embedding / LM-head matrix is trainable (one shared tensor
when the Full-TOFU model ties them).  After training, the output head is safely
untied, the complete input embedding matrix is restored to the pre-unlearning
Full-TOFU checkpoint, and every output row except answer-token rows from the
50 visible forget QAs is restored.  No retain/paraphrase/utility row can affect
training or restoration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import gagd_active_case_repair as active
import gagd_compare as gagd
import tofu_gagd_neighborhood_confidence as tofu
from controlled_unlearning_protocol import load_json_or_jsonl


MODE = "emb_lm_all_tokens_forget_only_restore"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--forget-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--emb-lm-lr", type=float, default=2e-4)
    parser.add_argument("--forget-weight", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="adamw")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["single", "auto"], default="single")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_forget_examples(
    path: Path,
    tok: Any,
    expected_count: int,
) -> tuple[List[gagd.Example], List[Dict[str, Any]]]:
    rows = load_json_or_jsonl(path)
    if len(rows) != expected_count:
        raise ValueError(
            f"locked forget file has {len(rows)} rows, expected {expected_count}"
        )
    examples: List[gagd.Example] = []
    normalized_rows: List[Dict[str, Any]] = []
    for position, row in enumerate(rows):
        allowed = {"question", "answer", "_source_index"}
        extras = set(row) - allowed
        if extras:
            raise ValueError(
                f"training-visible row {position} exposes forbidden fields: {sorted(extras)}"
            )
        if not row.get("question") or not row.get("answer"):
            raise ValueError(f"training-visible row {position} lacks question/answer")
        question = str(row["question"])
        answer = str(row["answer"])
        examples.append(
            gagd.Example(
                prompt=tofu.format_question_prompt(tok, question),
                answer=gagd.normalize_answer(answer.strip()),
                source=f"tofu_locked:{row.get('_source_index', position)}",
            )
        )
        normalized_rows.append(dict(row))
    return examples, normalized_rows


def forget_answer_row_ids(tok: Any, examples: Sequence[gagd.Example]) -> List[int]:
    selected: set[int] = set()
    for example in examples:
        selected.update(gagd.token_ids_for_text(tok, example.answer))
    selected -= gagd.special_token_ids(tok)
    if not selected:
        raise RuntimeError("forget answers produced no editable vocabulary rows")
    return sorted(selected)


@torch.no_grad()
def restore_matrix_in_chunks(
    weight: torch.Tensor,
    base: torch.Tensor,
    *,
    chunk_rows: int,
) -> None:
    if weight.shape != base.shape:
        raise ValueError("trained/base matrix shapes differ")
    for start in range(0, weight.shape[0], chunk_rows):
        stop = min(start + chunk_rows, weight.shape[0])
        weight[start:stop].copy_(
            base[start:stop].to(device=weight.device, dtype=weight.dtype)
        )


def main() -> None:
    args = parse_args()
    if args.forget_num <= 0 or args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("forget-num, steps and batch-size must be positive")
    if args.emb_lm_lr <= 0 or args.forget_weight <= 0:
        raise ValueError("learning rate and forget weight must be positive")
    if args.weight_decay < 0 or args.grad_clip < 0 or args.chunk_rows <= 0:
        raise ValueError("invalid optimizer/restoration controls")

    forget_path = Path(args.forget_json).resolve()
    if not forget_path.is_file():
        raise FileNotFoundError(forget_path)

    gagd.set_seed(args.seed)
    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)

    output_dir = gagd.resolve_output_path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_tok = AutoTokenizer.from_pretrained(args.model_path)
    if data_tok.pad_token is None:
        data_tok.pad_token = data_tok.eos_token
    forget, raw_rows = load_forget_examples(
        forget_path, data_tok, args.forget_num
    )

    model_args = argparse.Namespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(model_args, for_training=True)
    trainable_summary, tied_info = gagd.configure_trainable(model, "emb_lm_all_tokens")
    params = gagd.unique_trainable_params(model)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params, lr=args.emb_lm_lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            params, lr=args.emb_lm_lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            params, lr=args.emb_lm_lr, weight_decay=args.weight_decay
        )

    # Snapshot the exact Full-TOFU vocabulary matrix before any forgetting.
    base_rows = gagd.snapshot_embedding_output_weights(tied_info)
    answer_rows = forget_answer_row_ids(tok, forget)
    sampler = gagd.EpochBatchSampler(forget, args.batch_size, args.seed)
    device = gagd.first_device(model)

    model.train()
    with (output_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(1, args.steps + 1), desc="TOFU forget-only Stage1"):
            batch = sampler.next_batch()
            optimizer.zero_grad(set_to_none=True)
            forget_res = gagd.answer_ce_loss(
                model,
                tok,
                batch,
                selected_token_ids=None,
                device=device,
                append_eos=False,
            )
            # Gradient ascent on answer NLL == minimize negative CE.
            total = -args.forget_weight * forget_res.loss
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite Stage1 loss at step {step}")
            total.backward()
            grad_norm = None
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient norm at step {step}")
            optimizer.step()
            log_f.write(
                json.dumps(
                    {
                        "step": step,
                        "objective": "gradient_ascent_answer_nll",
                        "answer_ce": float(forget_res.loss.detach().cpu()),
                        "total_loss": float(total.detach().cpu()),
                        "answer_tokens": forget_res.contributing_tokens,
                        "fallback_examples": forget_res.fallback_examples,
                        "benchmark_retain_examples_seen": 0,
                        "paraphrases_seen": 0,
                        "gradient_norm_before_clip": (
                            float(grad_norm.detach().cpu())
                            if grad_norm is not None
                            else None
                        ),
                    }
                )
                + "\n"
            )

    del optimizer

    # Untie the trained head before restoring input embeddings.  Preserve only
    # forget-answer output rows; all other rows return to the exact Full-TOFU
    # starting checkpoint.
    output_embeddings = active.freeze_model_for_output_repair(model)
    input_weight = model.get_input_embeddings().weight
    output_weight = output_embeddings.weight
    selected_tensor = torch.tensor(
        answer_rows, dtype=torch.long, device=output_weight.device
    )
    trained_answer_rows = output_weight.index_select(0, selected_tensor).detach().clone()

    restore_matrix_in_chunks(
        input_weight, base_rows["input"], chunk_rows=args.chunk_rows
    )
    restore_matrix_in_chunks(
        output_weight, base_rows["output"], chunk_rows=args.chunk_rows
    )
    with torch.no_grad():
        output_weight.index_copy_(0, selected_tensor, trained_answer_rows)

    if input_weight.data_ptr() == output_weight.data_ptr():
        raise RuntimeError("TOFU Stage1 output head remained tied after restoration")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tok.save_pretrained(checkpoint_dir)

    row_report = {
        "policy": "forget_answer_output_rows_only",
        "selected_lm_head_row_count": len(answer_rows),
        "selected_lm_head_token_ids": answer_rows,
        "selected_lm_head_tokens": {
            str(token_id): tok.decode([token_id]) for token_id in answer_rows
        },
        "input_embedding_policy": "all rows restored to exact Full-TOFU base",
        "output_other_rows_policy": "restored to exact Full-TOFU base",
        "retain_or_utility_rows_consulted": False,
    }
    write_json(output_dir / "post_training_row_policy.json", row_report)

    config: Dict[str, Any] = {
        "schema_version": 1,
        "dataset": "tofu",
        "forget_split": "forget05",
        "mode": MODE,
        "protocol": "tofu_zerounlearn_data_access_forget_only_locked",
        "model_path": args.model_path,
        "forget_json": str(forget_path),
        "seed": args.seed,
        "forget_num": args.forget_num,
        "retain_num": 0,
        "paraphrases_used_during_training": False,
        "perturbed_answers_used_during_training": False,
        "real_authors_used_during_training": False,
        "world_facts_used_during_training": False,
        "forget_loss_type": "answer_nll_gradient_ascent",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "equivalent_passes_over_visible_forget": (
            args.steps * args.batch_size / args.forget_num
        ),
        "emb_lm_lr": args.emb_lm_lr,
        "forget_weight": args.forget_weight,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "trainable_parameter_summary": asdict(trainable_summary),
        "stage1_visible_source_indices": [
            int(row.get("_source_index", -1)) for row in raw_rows
        ],
        "post_training_restoration": row_report,
        "checkpoint": str(checkpoint_dir),
    }
    write_json(output_dir / "config_used.json", config)
    print(f"TOFU locked Stage1 checkpoint: {checkpoint_dir}")
    print(f"Visible data: {args.forget_num} direct forget QAs; retain/paraphrases=0")


if __name__ == "__main__":
    main()
