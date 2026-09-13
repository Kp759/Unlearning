#!/usr/bin/env python3
"""Frozen-base evaluation for the exact RWKU Batch-50 seed-1 comparison.

Uses the same split construction, prompt formatting, uncached greedy generation,
teacher-forced sensitive-token scoring, Level-3/neighbor rows, and PPL text as
evaluate_rwku_fact_association_embeddings_seed1.py. No residual bank is loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from rwku_batch50 import build_batch_split
from evaluate_rwku_fact_association_embeddings_seed1 import (
    evaluate_rows,
    _native_rows,
)
from mcf_zero_unlearn_official_eval import (
    dtype_from_str,
    load_official_ppl_text,
    official_perplexity,
    runtime_aligned_perplexity,
)


class NoRouteBank:
    """Minimal route recorder matching the edited evaluator interface."""

    def __init__(self):
        self.last_active_fact_indices = [[]]

    def counters(self):
        return {
            "hook_calls": 0,
            "active_batch_rows": 0,
            "active_token_positions": 0,
            "active_fact_counts": [],
        }


class FrozenBaseCausalLM(nn.Module):
    """Frozen base model with the edited evaluator's uncached generation API."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.base_model.requires_grad_(False)
        self.base_model.eval()

    @property
    def config(self):
        return self.base_model.config

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def set_association_prefix_lengths(self, lengths):
        # Intentional no-op: base model has no association intervention.
        return None

    def forward(self, input_ids=None, **kwargs):
        return self.base_model(input_ids=input_ids, **kwargs)

    @torch.no_grad()
    def generate_uncached_fixed_boundary(
        self,
        input_ids,
        attention_mask=None,
        *,
        max_new_tokens=32,
        eos_token_id=None,
    ):
        """Mirror AssociationCausalLM's reference greedy decoding exactly."""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                "Fixed-boundary base generation currently supports batch size 1"
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        prompt_length = int(attention_mask[0].sum().item())
        if prompt_length <= 0:
            raise ValueError("Generation prompt is empty")
        if not bool(attention_mask[0, :prompt_length].all()):
            raise ValueError(
                "Base reference generation requires right-padded prompt input"
            )
        ids = input_ids[:, :prompt_length].clone()
        mask = torch.ones_like(ids)
        eos = (
            int(eos_token_id)
            if eos_token_id is not None
            else getattr(self.config, "eos_token_id", None)
        )
        for _ in range(int(max_new_tokens)):
            logits = self.forward(
                input_ids=ids,
                attention_mask=mask,
                use_cache=False,
            ).logits
            next_token = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)
            mask = torch.ones_like(ids)
            if eos is not None and int(next_token.item()) == int(eos):
                break
        return ids


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-root", default="data/rwku")
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--skip-level3", action="store_true")
    p.add_argument("--skip-neighbors", action="store_true")
    p.add_argument(
        "--out",
        default="outputs/rwku_fact_assoc_seed1_base_eval.json",
    )
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(args.model_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    split = build_batch_split(
        data_root=Path(args.data_root).resolve(),
        batch_seed=1,
        allow_download=not args.no_download,
    )
    forget_rows = list(split["efficacy_forget"])
    if len(forget_rows) != 50:
        raise RuntimeError("Frozen RWKU seed-1 Batch-50 split must contain 50 rows")

    dtype = dtype_from_str(args.dtype)
    raw_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    raw_model.requires_grad_(False)
    raw_model.config.use_cache = False
    model = FrozenBaseCausalLM(raw_model).to(args.device).eval()
    bank = NoRouteBank()

    same50, same50_detail = evaluate_rows(
        model, bank, tokenizer, forget_rows, score_answers=True
    )
    heldout_l1, heldout_l1_detail = evaluate_rows(
        model, bank, tokenizer, split["heldout_level1"], score_answers=True
    )
    heldout_l2, heldout_l2_detail = evaluate_rows(
        model, bank, tokenizer, split["heldout_level2"], score_answers=True
    )
    paraphrase, paraphrase_detail = evaluate_rows(
        model, bank, tokenizer, split["heldout_paraphrase"], score_answers=True
    )

    level3 = level3_detail = None
    if not args.skip_level3:
        level3_rows = _native_rows(split, "forget_level3.json", 3)
        level3, level3_detail = evaluate_rows(
            model, bank, tokenizer, level3_rows, score_answers=False
        )

    neighbors = neighbor_detail = None
    if not args.skip_neighbors:
        neighbor_rows = [
            *_native_rows(split, "neighbor_level1.json", 1),
            *_native_rows(split, "neighbor_level2.json", 2),
        ]
        neighbors, neighbor_detail = evaluate_rows(
            model, bank, tokenizer, neighbor_rows, score_answers=False
        )

    legacy_ppl = runtime_ppl = None
    if not args.skip_ppl:
        ppl_text = load_official_ppl_text(args.wikidata_dir)
        if ppl_text is not None:
            device = next(model.parameters()).device
            legacy_ppl = official_perplexity(
                model,
                tokenizer,
                ppl_text,
                device,
                max_input_length=100,
            )
            runtime = runtime_aligned_perplexity(
                model,
                tokenizer,
                ppl_text,
                device,
                max_input_length=100,
            )
            runtime_ppl = runtime["ppl"]

    result = {
        "method": "FrozenBase",
        "dataset": "RWKU",
        "protocol_id": split["manifest"]["protocol_id"],
        "protocol_status": split["manifest"]["protocol_status"],
        "seed": 1,
        "target_seeds": split["manifest"]["target_seeds"],
        "subjects": [item["subject"] for item in split["manifest"]["targets"]],
        "forget_train_count": len(forget_rows),
        "comparison_contract": {
            "same_split_as_edited": True,
            "same_prompt_formatter_as_edited": True,
            "same_uncached_greedy_generation_as_edited": True,
            "same_teacher_forced_scoring_as_edited": True,
            "same_level3_neighbor_rows_as_edited": True,
            "same_ppl_text_as_edited": True,
            "residual_bank_loaded": False,
            "base_weights_edited": False,
        },
        "same_50_efficacy": same50,
        "heldout_level1": heldout_l1,
        "heldout_level2": heldout_l2,
        "heldout_level2_paraphrase": paraphrase,
        "adversarial_level3": level3,
        "neighbors": neighbors,
        "legacy_PPL": legacy_ppl,
        "runtime_aligned_PPL": runtime_ppl,
        "details": {
            "same_50_efficacy": same50_detail,
            "heldout_level1": heldout_l1_detail,
            "heldout_level2": heldout_l2_detail,
            "heldout_level2_paraphrase": paraphrase_detail,
            "adversarial_level3": level3_detail,
            "neighbors": neighbor_detail,
        },
    }

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite frozen-base evaluation: {out}")
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print(
        json.dumps(
            {
                "same50_recovery": same50["recovery_accuracy"],
                "same50_sensitive_token_top1_accuracy": same50.get(
                    "sensitive_token_top1_accuracy"
                ),
                "heldout_l1_recovery": heldout_l1["recovery_accuracy"],
                "heldout_l2_recovery": heldout_l2["recovery_accuracy"],
                "heldout_paraphrase_recovery": paraphrase["recovery_accuracy"],
                "level3_recovery": (
                    None if level3 is None else level3["recovery_accuracy"]
                ),
                "neighbor_recovery": (
                    None if neighbors is None else neighbors["recovery_accuracy"]
                ),
                "runtime_aligned_PPL": runtime_ppl,
                "legacy_PPL": legacy_ppl,
                "out": str(out),
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
