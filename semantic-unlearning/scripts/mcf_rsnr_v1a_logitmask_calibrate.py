#!/usr/bin/env python3
"""Calibrate the RSNR direct-logit baseline on training-safe five-view prompts only.

Two modes are supported:

* suppress_only: choose the smallest canonical-sensitive-token penalty whose
  worst-of-five true-answer log-probability drop is >= 2.0 for every forget fact.
* suppress_plus_idk: additionally choose an IDK-token boost so every fact also
  has worst-of-five logP(IDK)-logP(true) >= 0.1.

No official paraphrase, neighborhood, fresh-retain, alias, target_new objective,
or held-out probe text is read. The selected values are frozen before official
evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_mcf_private_vocab_rewiring_v1 as base_runner
import run_mcf_rsnr_v1a_oracle as rsnr
import mcf_rsnr_v1a_logitmask_common as lm


def parse_grid(text: str, *, allow_zero: bool) -> list[float]:
    values = sorted(set(float(v.strip()) for v in str(text).split(",") if v.strip()))
    if not values:
        raise ValueError("grid must be non-empty")
    if any(v < 0 if allow_zero else v <= 0 for v in values):
        raise ValueError("invalid grid value")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--protocol-dir", required=True)
    p.add_argument("--view-corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("suppress_only", "suppress_plus_idk"), required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--batch-cases", type=int, default=4)
    p.add_argument("--true-penalties", default="2,3,4,5,6,8,10,12")
    p.add_argument("--idk-boosts", default="0,1,2,3,4,5,6,8")
    p.add_argument("--minimum-idk-vs-true-margin", type=float, default=0.1)
    p.add_argument("--minimum-true-logprob-drop", type=float, default=2.0)
    args = p.parse_args()
    if args.seed != 1 or args.forget_num != 50:
        p.error("development calibration is locked to consumed seed1/forget50")
    if args.batch_cases <= 0:
        p.error("batch-cases must be positive")
    args.penalty_grid = parse_grid(args.true_penalties, allow_zero=False)
    args.boost_grid = parse_grid(args.idk_boosts, allow_zero=True)
    if args.mode == "suppress_only":
        args.boost_grid = [0.0]
    return args


def evaluate_training_views(
    model: Any,
    hook: lm.DirectLogitMaskHook,
    tokenizer: Any,
    forget: Sequence[Mapping[str, Any]],
    view_map: Mapping[int, Sequence[str]],
    canonical_answers: Mapping[tuple[str, str], str],
    token_sets: Mapping[tuple[str, str], Sequence[int]],
    base_true: Mapping[int, Sequence[float]],
    *,
    device: torch.device,
    batch_cases: int,
    margin_threshold: float,
    drop_threshold: float,
):
    rows = []
    for start in range(0, len(forget), batch_cases):
        cases = forget[start : start + batch_cases]
        prompts, true_answers, owners = rsnr.prompts_for_cases(cases, view_map)
        pairs: list[tuple[str, str]] = []
        for local, row in enumerate(cases):
            pair = rsnr.fact_key(row)
            pairs.extend([pair] * sum(owner == local for owner in owners))
        # prompts_for_cases groups views by case, so this construction aligns.
        if len(pairs) != len(prompts):
            raise RuntimeError("training-view pair alignment failed")
        idk_answers = [lm.ABSTENTION] * len(prompts)
        masked_true = lm.sequence_logprobs_masked(
            model, hook, tokenizer, prompts, true_answers, pairs, token_sets,
            device=device, gated=True,
        )
        masked_idk = lm.sequence_logprobs_masked(
            model, hook, tokenizer, prompts, idk_answers, pairs, token_sets,
            device=device, gated=True,
        )
        for local, row in enumerate(cases):
            idxs = [j for j, owner in enumerate(owners) if owner == local]
            margins = [float((masked_idk[j] - masked_true[j]).item()) for j in idxs]
            drops = [
                float(base_true[int(row["case_id"])][k] - masked_true[j].item())
                for k, j in enumerate(idxs)
            ]
            rows.append({
                "case_id": int(row["case_id"]),
                "subject": rsnr.fact_key(row)[0],
                "relation_id": rsnr.fact_key(row)[1],
                "worst_idk_vs_true_margin": min(margins),
                "worst_true_logprob_drop": min(drops),
                "margin_pass": min(margins) >= float(margin_threshold),
                "drop_pass": min(drops) >= float(drop_threshold),
            })
    return rows


def main() -> None:
    args = parse_args()
    protocol_dir = Path(args.protocol_dir).resolve()
    protocol = rsnr.load_protocol(protocol_dir, int(args.forget_num))
    forget = protocol["forget"]
    view_map, view_meta = rsnr.load_training_views(Path(args.view_corpus).resolve())
    rsnr.validate_case_alignment(forget, view_map)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for direct-logit calibration")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=base_runner.dtype_from_name(args.dtype),
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)

    canonical_answers = lm.canonical_answer_map(forget)
    token_sets = lm.canonical_token_sets(tokenizer, canonical_answers)
    idk_ids = lm.abstention_token_set(tokenizer)
    hook = lm.DirectLogitMaskHook.install(
        lm.get_lm_head(model), true_penalty=0.0, idk_boost=0.0, idk_token_ids=idk_ids
    )

    base_true = rsnr.base_true_logprobs_for_all_views(
        model, hook, tokenizer, forget, view_map,
        device=device, batch_cases=int(args.batch_cases),
    )

    candidates = []
    selected = None
    try:
        for penalty in args.penalty_grid:
            for boost in args.boost_grid:
                hook.true_penalty = float(penalty)
                hook.idk_boost = float(boost)
                rows = evaluate_training_views(
                    model, hook, tokenizer, forget, view_map, canonical_answers, token_sets, base_true,
                    device=device,
                    batch_cases=int(args.batch_cases),
                    margin_threshold=float(args.minimum_idk_vs_true_margin),
                    drop_threshold=float(args.minimum_true_logprob_drop),
                )
                drop_passed = sum(bool(r["drop_pass"]) for r in rows)
                margin_passed = sum(bool(r["margin_pass"]) for r in rows)
                if args.mode == "suppress_only":
                    passed = drop_passed == len(rows)
                else:
                    passed = drop_passed == len(rows) and margin_passed == len(rows)
                summary = {
                    "true_penalty": float(penalty),
                    "idk_boost": float(boost),
                    "drop_passed": drop_passed,
                    "margin_passed": margin_passed,
                    "minimum_worst_true_logprob_drop": min(r["worst_true_logprob_drop"] for r in rows),
                    "minimum_worst_idk_vs_true_margin": min(r["worst_idk_vs_true_margin"] for r in rows),
                    "calibration_passed": bool(passed),
                }
                candidates.append(summary)
                print(json.dumps(summary), flush=True)
                if passed:
                    key = (float(penalty) + float(boost), float(penalty), float(boost))
                    if selected is None or key < selected[0]:
                        selected = (key, summary, rows)
        if selected is None:
            raise RuntimeError("no direct-logit calibration candidate satisfied the locked training gate")

        _key, best, rows = selected
        membership = [
            {
                "case_id": int(row["case_id"]),
                "subject": rsnr.fact_key(row)[0],
                "relation_id": rsnr.fact_key(row)[1],
                "target_true": canonical_answers[rsnr.fact_key(row)],
            }
            for row in forget
        ]
        result = {
            "protocol": lm.PROTOCOL,
            "variant": args.mode,
            "intervention_site": lm.INTERVENTION_SITE,
            "base_model": str(args.model_path),
            "seed": int(args.seed),
            "development_only": True,
            "calibration_source": "locked training-safe V1.3 five-view corpus only",
            "views_per_case": int(view_meta["views_per_case"]),
            "true_penalty": float(best["true_penalty"]),
            "idk_boost": float(best["idk_boost"]),
            "minimum_idk_vs_true_margin": float(args.minimum_idk_vs_true_margin),
            "minimum_true_logprob_drop": float(args.minimum_true_logprob_drop),
            "calibration_passed": True,
            "calibration_summary": best,
            "calibration_candidates": candidates,
            "per_case": rows,
            "forget_membership": membership,
            "mask_definition": "all tokenizer ids appearing in canonical training-visible target_true are penalized at each gated answer position",
            "idk_definition": "all tokenizer ids appearing in the registered abstention are boosted at each gated answer position",
            "aliases_used_for_mask": False,
            "target_new_used": False,
            "official_paraphrase_text_used": False,
            "official_neighborhood_text_used": False,
            "heldout_probe_text_used": False,
            "base_weights_modified": False,
            "transformer_weights_modified": False,
            "lm_head_weights_modified": False,
            "trainable_parameters": 0,
            "claim_boundary": {
                "training_free_output_intervention": True,
                "oracle_gate": True,
                "canonical_surface_token_mask_only": True,
                "latent_knowledge_erasure_claimed": False,
            },
        }
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "selected": best,
            "output": str(out),
        }, indent=2), flush=True)
    finally:
        hook.remove()


if __name__ == "__main__":
    main()
