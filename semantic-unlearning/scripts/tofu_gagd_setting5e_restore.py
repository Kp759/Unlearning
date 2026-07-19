#!/usr/bin/env python3
"""TOFU analogue of Setting 5e post-training row restoration.

The input must be a saved ``emb_lm_all_tokens`` TOFU checkpoint and
``--base-model-path`` must be the exact model from which that checkpoint was
trained.  No GA/GD training is rerun.

TOFU has no CounterFact target-new/target-true pair, so the vocabulary policy
is defined from answer tokens:

* unique forget-answer rows keep the learned GA/GD update;
* forget/retain overlap rows keep a configurable fraction of the update;
* retain-only and unrelated rows are restored to the base model.

The policy is applied to both input embeddings and the LM head.  Tied weights
are edited once and remain tied.
"""

from __future__ import annotations

import argparse
import gc
import math
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import gagd_active_case_repair as active
import gagd_compare as gagd
import tofu_gagd_neighborhood_confidence as tofu


METHOD = "tofu_setting5e_restore"


@dataclass(frozen=True)
class TOFURowGroups:
    forget: Tuple[int, ...]
    retain: Tuple[int, ...]
    unique_forget: Tuple[int, ...]
    forget_retain_overlap: Tuple[int, ...]
    retain_only: Tuple[int, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source-mode",
        default="emb_lm_all_tokens",
        help="Recorded for provenance; normally emb_lm_all_tokens.",
    )
    parser.add_argument(
        "--forget-split",
        choices=sorted(tofu.PAIRED_RETAIN_SPLITS),
        default="forget05",
    )
    parser.add_argument("--retain-split", default="retain95")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forget-num", type=int, default=200)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument("--unique-forget-alpha", type=float, default=1.0)
    parser.add_argument("--overlap-alpha", type=float, default=0.25)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["single", "auto"], default="single")
    parser.add_argument(
        "--save-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    expected = tofu.PAIRED_RETAIN_SPLITS[args.forget_split]
    if args.retain_split != expected:
        raise ValueError(
            f"{args.forget_split} must be paired with {expected}, "
            f"not {args.retain_split}"
        )
    for name in ("forget_num", "retain_num", "chunk_rows"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("unique_forget_alpha", "overlap_alpha"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must lie in [0,1]")


def _sample_rows(split: str, count: int, seed: int) -> List[Dict[str, Any]]:
    rows = list(load_dataset("locuslab/TOFU", name=split, split="train"))
    indices = tofu.deterministic_sample_indices(len(rows), count, seed)
    return [rows[index] for index in indices]


def _answer_token_ids(
    tok: Any,
    rows: Sequence[Dict[str, Any]],
) -> set[int]:
    selected: set[int] = set()
    specials = gagd.special_token_ids(tok)
    for row in rows:
        answer = gagd.normalize_answer(str(row["answer"]).strip())
        selected.update(gagd.token_ids_for_text(tok, answer))
    return selected - specials


def build_tofu_row_groups(
    tok: Any,
    forget_rows: Sequence[Dict[str, Any]],
    retain_rows: Sequence[Dict[str, Any]],
) -> TOFURowGroups:
    forget = _answer_token_ids(tok, forget_rows)
    retain = _answer_token_ids(tok, retain_rows)
    overlap = forget & retain
    return TOFURowGroups(
        forget=tuple(sorted(forget)),
        retain=tuple(sorted(retain)),
        unique_forget=tuple(sorted(forget - retain)),
        forget_retain_overlap=tuple(sorted(overlap)),
        retain_only=tuple(sorted(retain - forget)),
    )


def build_row_alphas(
    vocab_size: int,
    groups: TOFURowGroups,
    *,
    unique_forget_alpha: float,
    overlap_alpha: float,
    device: torch.device,
) -> torch.Tensor:
    alphas = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    if groups.unique_forget:
        ids = torch.tensor(
            groups.unique_forget,
            dtype=torch.long,
            device=device,
        )
        alphas.index_fill_(0, ids, unique_forget_alpha)
    if groups.forget_retain_overlap:
        ids = torch.tensor(
            groups.forget_retain_overlap,
            dtype=torch.long,
            device=device,
        )
        alphas.index_fill_(0, ids, overlap_alpha)
    return alphas


@torch.no_grad()
def apply_row_policy(
    trained_weight: torch.Tensor,
    base_weight: torch.Tensor,
    row_alphas: torch.Tensor,
    *,
    chunk_rows: int,
) -> Dict[str, float]:
    if trained_weight.shape != base_weight.shape:
        raise ValueError("Trained and base vocabulary matrices have different shapes")
    if row_alphas.shape != (trained_weight.shape[0],):
        raise ValueError("Row alpha vector does not match vocabulary size")
    delta_norm_before_sq = 0.0
    delta_norm_after_sq = 0.0
    for start in range(0, trained_weight.shape[0], chunk_rows):
        stop = min(start + chunk_rows, trained_weight.shape[0])
        trained_chunk = trained_weight[start:stop]
        base_chunk = base_weight[start:stop].to(
            device=trained_chunk.device,
            dtype=trained_chunk.dtype,
        )
        delta = trained_chunk - base_chunk
        delta_norm_before_sq += float(delta.float().square().sum().cpu())
        alpha = row_alphas[start:stop].to(
            device=trained_chunk.device,
            dtype=trained_chunk.dtype,
        ).unsqueeze(-1)
        final_delta = alpha * delta
        trained_chunk.copy_(base_chunk + final_delta)
        delta_norm_after_sq += float(final_delta.float().square().sum().cpu())
    return {
        "delta_norm_before": math.sqrt(delta_norm_before_sq),
        "delta_norm_after": math.sqrt(delta_norm_after_sq),
    }


def _model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )


def _group_report(tok: Any, groups: TOFURowGroups) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for name, token_ids in asdict(groups).items():
        report[name] = {
            "count": len(token_ids),
            "token_ids": list(token_ids),
            "tokens": {
                str(token_id): tok.decode([token_id])
                for token_id in token_ids
            },
        }
    return report


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    gagd.set_seed(args.seed)
    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)
    output_dir = gagd.resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    forget_rows = _sample_rows(args.forget_split, args.forget_num, args.seed)
    retain_rows = _sample_rows(args.retain_split, args.retain_num, args.seed)
    groups = build_tofu_row_groups(tok, forget_rows, retain_rows)
    gagd.write_json(output_dir / "token_group_report.json", _group_report(tok, groups))

    print(f"Loading trained checkpoint: {args.model_path}")
    model, tok = gagd.load_model_and_tokenizer(
        _model_args(args),
        for_training=False,
    )
    print(f"Loading exact pre-GA/GD base on CPU: {args.base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=gagd.torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
    )

    trained_input = model.get_input_embeddings().weight
    trained_output = model.get_output_embeddings().weight
    base_input = base_model.get_input_embeddings().weight
    base_output = base_model.get_output_embeddings().weight
    if trained_input.shape[0] != len(tok):
        raise ValueError("Tokenizer vocabulary and model embeddings disagree")
    row_alphas = build_row_alphas(
        trained_input.shape[0],
        groups,
        unique_forget_alpha=args.unique_forget_alpha,
        overlap_alpha=args.overlap_alpha,
        device=trained_input.device,
    )

    input_output_tied = (
        trained_input.data_ptr() == trained_output.data_ptr()
    )
    input_stats = apply_row_policy(
        trained_input,
        base_input,
        row_alphas,
        chunk_rows=args.chunk_rows,
    )
    if input_output_tied:
        output_stats = dict(input_stats)
    else:
        output_stats = apply_row_policy(
            trained_output,
            base_output,
            row_alphas.to(trained_output.device),
            chunk_rows=args.chunk_rows,
        )

    source_config_path, source_config = tofu.discover_source_config(
        args.model_path
    )
    config_used = {
        **vars(args),
        "method": METHOD,
        "source_experiment_config_path": (
            str(source_config_path) if source_config_path is not None else None
        ),
        "source_experiment_config": source_config,
        "policy": {
            "unique_forget": (
                "base + unique_forget_alpha * (trained - base)"
            ),
            "forget_retain_overlap": (
                "base + overlap_alpha * (trained - base)"
            ),
            "retain_only": "base row",
            "unrelated": "base row",
        },
    }
    gagd.write_json(output_dir / "config_used.json", config_used)
    summary = {
        "method": METHOD,
        "input_checkpoint": args.model_path,
        "base_model": args.base_model_path,
        "source_mode": args.source_mode,
        "input_output_tied": input_output_tied,
        "unique_forget_alpha": args.unique_forget_alpha,
        "overlap_alpha": args.overlap_alpha,
        "row_counts": {
            "forget": len(groups.forget),
            "retain": len(groups.retain),
            "unique_forget": len(groups.unique_forget),
            "forget_retain_overlap": len(groups.forget_retain_overlap),
            "retain_only": len(groups.retain_only),
            "unrelated": (
                trained_input.shape[0]
                - len(set(groups.forget) | set(groups.retain))
            ),
        },
        "input_embedding_delta": input_stats,
        "lm_head_delta": output_stats,
        "checkpoint_saved": bool(args.save_model),
    }
    gagd.write_json(output_dir / "restore_summary.json", summary)
    if args.save_model:
        active.save_repair_checkpoint(
            model,
            tok,
            output_dir / "checkpoint",
            repair_config=config_used,
        )
    del base_model, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"TOFU Setting 5e checkpoint: {output_dir / 'checkpoint'}")


if __name__ == "__main__":
    main()
