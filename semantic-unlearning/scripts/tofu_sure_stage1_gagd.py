#!/usr/bin/env python3
"""SURE-TOFU Stage 1A: same-prompt GA/GD with no held-out data access.

The only training data are the author-balanced training-visible direct forget
QAs.  For every teacher-forced answer-token context:

* GA minimizes log p(current true answer token), suppressing the sensitive
  target token.
* GD minimizes KL(Base_non_sensitive || Current_non_sensitive) after removing
  that position's sensitive target token from both vocabulary distributions
  and renormalizing.

The Full-TOFU starting model is used only as a teacher on these same visible
contexts.  No retain95, paraphrase, same-author holdout, real-authors,
world-facts, PPL, or final-evaluation signal is loaded.

As in the MQuAKE SURE path, post-training vocabulary restoration keeps only
sensitive answer-token rows.  ``sensitive_both`` preserves those rows in both
input embeddings and the detached LM head; ``output_only`` restores all input
rows and preserves sensitive LM-head rows only.  All non-sensitive vocabulary
rows are restored exactly to the Full-TOFU starting checkpoint.
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
from transformers import AutoTokenizer

import gagd_active_case_repair as active
import gagd_compare as gagd
import tofu_gagd_neighborhood_confidence as tofu
from controlled_unlearning_protocol import load_json_or_jsonl


METHOD = "SURE-TOFU-same-prompt-GAGD-Stage1A"
PROTOCOL = "tofu_author_balanced_forget_only_locked_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--forget-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--emb-lm-lr", type=float, default=1e-4)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--gd-weight", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument(
        "--restoration-mode",
        choices=("sensitive_both", "output_only"),
        default="sensitive_both",
    )
    p.add_argument("--cache-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    p.add_argument("--chunk-rows", type=int, default=2048)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cache_dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def load_forget(
    path: Path,
    tok: Any,
    expected_count: int,
) -> tuple[List[gagd.Example], List[Dict[str, Any]]]:
    rows = load_json_or_jsonl(path)
    if len(rows) != expected_count:
        raise ValueError(f"locked forget file has {len(rows)} rows, expected {expected_count}")
    examples: List[gagd.Example] = []
    normalized: List[Dict[str, Any]] = []
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
                source=f"tofu_sure:{row.get('_source_index', position)}",
            )
        )
        normalized.append(dict(row))
    return examples, normalized


def sensitive_row_ids(tok: Any, examples: Sequence[gagd.Example]) -> List[int]:
    ids: set[int] = set()
    for ex in examples:
        ids.update(gagd.token_ids_for_text(tok, ex.answer))
    ids -= gagd.special_token_ids(tok)
    if not ids:
        raise RuntimeError("visible forget answers produced no sensitive vocabulary rows")
    return sorted(ids)


class IndexSampler:
    def __init__(self, n: int, batch_size: int, seed: int):
        if n <= 0 or batch_size <= 0:
            raise ValueError("sampler requires positive n and batch size")
        self.n = n
        self.batch_size = min(batch_size, n)
        self.rng = random.Random(seed)
        self.order: List[int] = []
        self.cursor = 0

    def next(self) -> List[int]:
        result: List[int] = []
        while len(result) < self.batch_size:
            if self.cursor >= len(self.order):
                self.order = list(range(self.n))
                self.rng.shuffle(self.order)
                self.cursor = 0
            take = min(self.batch_size - len(result), len(self.order) - self.cursor)
            result.extend(self.order[self.cursor : self.cursor + take])
            self.cursor += take
        return result


@torch.no_grad()
def cache_base_answer_logits(
    model: torch.nn.Module,
    tok: Any,
    examples: Sequence[gagd.Example],
    device: torch.device,
    storage_dtype: torch.dtype,
) -> tuple[List[torch.Tensor], List[torch.Tensor], int]:
    """Cache Base full-vocabulary logits only at scored answer positions."""
    model.eval()
    logits_cache: List[torch.Tensor] = []
    labels_cache: List[torch.Tensor] = []
    total_tokens = 0
    for ex in tqdm(examples, desc="cache Full-TOFU same-prompt logits"):
        batch = gagd.build_batch(tok, [ex], device, append_eos=False)
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = output.logits[0, :-1, :]
        labels = batch["labels"][0, 1:]
        mask = labels.ne(-100)
        if not mask.any():
            raise RuntimeError("training-visible answer produced no scored tokens")
        selected_logits = logits[mask].detach().to(device="cpu", dtype=storage_dtype).contiguous()
        selected_labels = labels[mask].detach().to(device="cpu", dtype=torch.long).contiguous()
        logits_cache.append(selected_logits)
        labels_cache.append(selected_labels)
        total_tokens += int(selected_labels.numel())
        del output
    return logits_cache, labels_cache, total_tokens


def same_prompt_gagd_loss(
    model: torch.nn.Module,
    tok: Any,
    examples: Sequence[gagd.Example],
    example_indices: Sequence[int],
    reference_logits: Sequence[torch.Tensor],
    reference_labels: Sequence[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    batch_examples = [examples[index] for index in example_indices]
    batch = gagd.build_batch(tok, batch_examples, device, append_eos=False)
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    shifted_logits = output.logits[:, :-1, :]
    shifted_labels = batch["labels"][:, 1:]

    ga_terms: List[torch.Tensor] = []
    gd_terms: List[torch.Tensor] = []
    token_count = 0
    for row, example_index in enumerate(example_indices):
        mask = shifted_labels[row].ne(-100)
        current = shifted_logits[row][mask].float()
        target = shifted_labels[row][mask]
        base = reference_logits[example_index].to(device=device, dtype=torch.float32)
        base_target = reference_labels[example_index].to(device=device)
        if current.shape != base.shape or not torch.equal(target, base_target):
            raise RuntimeError("cached Base answer contexts no longer align with training batch")

        current_logp = F.log_softmax(current, dim=-1)
        ga_terms.append(current_logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean())

        # Exact non-sensitive KL on the same answer contexts.  Use a very
        # negative *finite* value rather than -inf so the masked coordinate
        # cannot create 0 * NaN in KL arithmetic.  The masked coordinate is
        # explicitly zeroed from the KL sum after renormalization.
        base_masked = base.clone()
        current_masked = current.clone()
        row_ids = torch.arange(target.numel(), device=device)
        finite_mask = -1.0e9
        base_masked[row_ids, target] = finite_mask
        current_masked[row_ids, target] = finite_mask
        base_non_sensitive_logp = F.log_softmax(base_masked, dim=-1)
        current_non_sensitive_logp = F.log_softmax(current_masked, dim=-1)
        per_vocab_kl = base_non_sensitive_logp.exp() * (
            base_non_sensitive_logp - current_non_sensitive_logp
        )
        per_vocab_kl[row_ids, target] = 0.0
        gd_terms.append(per_vocab_kl.sum(dim=-1).mean())
        token_count += int(target.numel())

    del output
    return torch.stack(ga_terms).mean(), torch.stack(gd_terms).mean(), token_count


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
        weight[start:stop].copy_(base[start:stop].to(device=weight.device, dtype=weight.dtype))


def main() -> None:
    a = parse_args()
    if a.forget_num <= 0 or a.steps <= 0 or a.batch_size <= 0:
        raise ValueError("forget-num, steps and batch-size must be positive")
    if a.emb_lm_lr <= 0 or a.ga_weight <= 0 or a.gd_weight < 0:
        raise ValueError("invalid GA/GD optimization weights")
    if a.grad_clip < 0 or a.weight_decay < 0 or a.chunk_rows <= 0:
        raise ValueError("invalid optimizer/restoration controls")

    forget_path = Path(a.forget_json).resolve()
    if not forget_path.is_file():
        raise FileNotFoundError(forget_path)
    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    root = gagd.resolve_output_path(a.output_dir)
    ckpt = root / "checkpoint"
    root.mkdir(parents=True, exist_ok=True)

    data_tok = AutoTokenizer.from_pretrained(a.model_path)
    if data_tok.pad_token is None:
        data_tok.pad_token = data_tok.eos_token
    examples, raw_rows = load_forget(forget_path, data_tok, a.forget_num)

    ns = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=True)
    summary, tied = gagd.configure_trainable(model, "emb_lm_all_tokens")
    params = gagd.unique_trainable_params(model)
    base_rows = gagd.snapshot_embedding_output_weights(tied)
    sensitive_ids = sensitive_row_ids(tok, examples)
    device = gagd.first_device(model)

    ref_logits, ref_labels, cached_tokens = cache_base_answer_logits(
        model, tok, examples, device, cache_dtype(a.cache_dtype)
    )
    approx_cache_bytes = sum(t.numel() * t.element_size() for t in ref_logits)

    optimizer = torch.optim.AdamW(params, lr=a.emb_lm_lr, weight_decay=a.weight_decay)
    sampler = IndexSampler(len(examples), a.batch_size, a.seed)
    model.train()
    with (root / "train_log.jsonl").open("w", encoding="utf-8") as handle:
        for step in tqdm(range(1, a.steps + 1), desc="SURE-TOFU Stage1A GA/GD"):
            indices = sampler.next()
            optimizer.zero_grad(set_to_none=True)
            ga, gd, tokens = same_prompt_gagd_loss(
                model, tok, examples, indices, ref_logits, ref_labels, device
            )
            loss = a.ga_weight * ga + a.gd_weight * gd
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage1A loss at step {step}")
            loss.backward()
            grad_norm = None
            if a.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient norm at step {step}")
            optimizer.step()
            if step == 1 or step % 25 == 0 or step == a.steps:
                handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "loss": float(loss.detach().cpu()),
                            "ga_sensitive_logprob": float(ga.detach().cpu()),
                            "gd_base_non_sensitive_kl": float(gd.detach().cpu()),
                            "answer_tokens": tokens,
                            "gradient_norm_before_clip": (
                                float(grad_norm.detach().cpu()) if grad_norm is not None else None
                            ),
                            "retain95_seen": 0,
                            "paraphrases_seen": 0,
                            "same_author_holdout_seen": 0,
                            "real_authors_seen": 0,
                            "world_facts_seen": 0,
                            "PPL_seen": False,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                handle.flush()
    del optimizer

    output_embeddings = active.freeze_model_for_output_repair(model)
    input_weight = model.get_input_embeddings().weight
    output_weight = output_embeddings.weight
    input_ids = torch.tensor(sensitive_ids, dtype=torch.long, device=input_weight.device)
    output_ids = input_ids.to(output_weight.device)
    trained_input = input_weight.index_select(0, input_ids).detach().clone()
    trained_output = output_weight.index_select(0, output_ids).detach().clone()

    restore_matrix_in_chunks(input_weight, base_rows["input"], chunk_rows=a.chunk_rows)
    restore_matrix_in_chunks(output_weight, base_rows["output"], chunk_rows=a.chunk_rows)
    with torch.no_grad():
        if a.restoration_mode == "sensitive_both":
            input_weight.index_copy_(0, input_ids, trained_input)
        output_weight.index_copy_(0, output_ids, trained_output)

    if input_weight.data_ptr() == output_weight.data_ptr():
        raise RuntimeError("SURE-TOFU Stage1A LM head remained tied after restoration")

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    restoration = {
        "mode": a.restoration_mode,
        "sensitive_row_count": len(sensitive_ids),
        "sensitive_token_ids": sensitive_ids,
        "sensitive_tokens": {str(i): tok.decode([i]) for i in sensitive_ids},
        "input_embeddings": (
            "sensitive rows keep Stage1A displacement; all others Base"
            if a.restoration_mode == "sensitive_both"
            else "all rows restored to Base"
        ),
        "lm_head": "sensitive rows keep Stage1A displacement; all others Base",
        "retain_or_heldout_rows_consulted": False,
    }
    write_json(root / "vocabulary_restoration.json", restoration)
    write_json(
        root / "config_used.json",
        {
            "schema_version": 1,
            "method": METHOD,
            "protocol": PROTOCOL,
            "model_path": a.model_path,
            "forget_json": str(forget_path),
            "seed": a.seed,
            "forget_num": a.forget_num,
            "steps": a.steps,
            "batch_size": a.batch_size,
            "emb_lm_lr": a.emb_lm_lr,
            "ga_weight": a.ga_weight,
            "gd_weight": a.gd_weight,
            "grad_clip": a.grad_clip,
            "weight_decay": a.weight_decay,
            "cache_dtype": a.cache_dtype,
            "cached_answer_tokens": cached_tokens,
            "approx_reference_logit_cache_bytes": approx_cache_bytes,
            "gd_definition": "KL(Base_non_sensitive || Current_non_sensitive) with per-position true answer token removed and renormalized",
            "ga_definition": "mean log probability of teacher-forced true answer token, minimized",
            "trainable_parameter_summary": asdict(summary),
            "training_visible_source_indices": [int(r.get("_source_index", -1)) for r in raw_rows],
            "data_access": {
                "direct_forget_qas": a.forget_num,
                "retain95": 0,
                "paraphrases": 0,
                "same_author_holdout": 0,
                "real_authors": 0,
                "world_facts": 0,
                "PPL": False,
            },
            "post_training_restoration": restoration,
            "checkpoint": str(ckpt.resolve()),
        },
    )
    print(f"SURE-TOFU Stage1A checkpoint: {ckpt}")
    print(
        f"Visible data: {a.forget_num} direct forget QAs only; "
        f"cached answer tokens={cached_tokens}; teacher-cache≈{approx_cache_bytes / (1024**3):.3f} GiB"
    )


if __name__ == "__main__":
    main()
