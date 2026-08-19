#!/usr/bin/env python3
"""Exact Base->edited retain-KL audit for sparse SURE LM-head row updates.

This audit uses ONLY the prompt-only retain-training set. No official paraphrase,
neighborhood, retain-eval, or PPL prompts are touched.

For a retain prompt r and selected sensitive row s:

    p_rs = p_Base(s | r)
    d_rs = h_r^T Delta w_s
    A_r  = 1 - sum_s p_rs + sum_s p_rs exp(d_rs)

Because only the selected output rows change, the exact full-vocabulary KL is

    KL(Base || Edited)_r = log A_r - sum_s p_rs d_rs.

The script reports mean/median/p95/p99/max KL, counts above user-specified
thresholds, worst retain prompts, worst prompt/token contributors, the original
row-normalized SURE retain action for comparison, and a globally probability-
weighted action:

    sum p(1-p)d^2 / sum p(1-p).

If --retain-hidden-cache and --retain-prob-cache are supplied, no Base model
forward pass is needed. Otherwise they are recomputed once from the Base model.
No edited-model forward pass is ever performed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

import gagd_compare as gagd
import sure_canonical_core as core
import sure_contrastive_two_stage_v2 as v2
import sure_retain_kl as retain


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-retain-path", required=True)
    p.add_argument("--delta-path", required=True,
                   help="torch file with {'row_ids': [...], 'delta': [S,d]}; unscaled is fine")
    p.add_argument("--scale", type=float, required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--retain-hidden-cache", default=None)
    p.add_argument("--retain-prob-cache", default=None,
                   help="Raw Base probabilities [R,S], NOT normalized SURE weights")
    p.add_argument("--cache-output-dir", default=None,
                   help="Optional directory to save recomputed raw retain caches")
    p.add_argument("--thresholds", default="1e-6,1e-5,1e-4,1e-3,1e-2,1e-1")
    p.add_argument("--top-prompts", type=int, default=20)
    p.add_argument("--top-contributors", type=int, default=50)
    p.add_argument("--retain-weight-clip", type=float, default=10.0,
                   help="Only for reproducing the old normalized D_R diagnostic")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def load_prompt_only_retain(path: Path) -> Tuple[List[Dict[str, Any]], List[int], List[str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("retain-training JSON must be a non-empty list")
    ids: List[int] = []
    prompts: List[str] = []
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"retain record {i} is not an object")
        if (row.get("paraphrase_prompts") or row.get("neighborhood_prompts")
                or row.get("generation_prompts") or row.get("attribute_prompts")):
            raise RuntimeError(f"retain record {i} exposes held-out probes")
        rr = row.get("requested_rewrite")
        if not isinstance(rr, Mapping) or not rr.get("prompt"):
            raise RuntimeError(f"retain record {i} lacks prompt")
        if "target_true" in rr or "target_new" in rr:
            raise RuntimeError("retain answer labels leaked into audit input")
        case_id = int(row.get("case_id", i))
        subject = str(rr.get("subject", ""))
        template = str(rr["prompt"])
        try:
            prompt = template.format(subject)
        except Exception:
            prompt = template
        ids.append(case_id)
        prompts.append(prompt)
    return rows, ids, prompts


def load_delta(path: Path, scale: float) -> Tuple[List[int], torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, Mapping) or "row_ids" not in obj or "delta" not in obj:
        raise ValueError("delta file must contain row_ids and delta")
    ids = [int(x) for x in obj["row_ids"]]
    delta = obj["delta"].float()
    if delta.ndim != 2 or delta.shape[0] != len(ids):
        raise ValueError("delta shape does not match row_ids")
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and non-negative")
    return ids, delta * float(scale)


def load_or_build_caches(a: argparse.Namespace, retain_records, selected_ids):
    hidden = probs = None
    source = "recomputed_from_base_once"
    if a.retain_hidden_cache and a.retain_prob_cache:
        hidden = torch.load(Path(a.retain_hidden_cache).resolve(), map_location="cpu").float()
        probs = torch.load(Path(a.retain_prob_cache).resolve(), map_location="cpu").float()
        source = "loaded_raw_caches"
    elif bool(a.retain_hidden_cache) != bool(a.retain_prob_cache):
        raise ValueError("provide both retain caches or neither")

    if hidden is None:
        ns = argparse.Namespace(model_path=a.model_path, dtype=a.dtype,
                                device_map=a.device_map, gradient_checkpointing=False)
        model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        core.untie_and_freeze_output_head(model)
        device = gagd.first_device(model)
        retain_cases = retain.retain_prompt_cases(retain_records)
        hidden, probs, _ = v2.cache_retain_hidden_and_probs(
            model, tok, retain_cases, selected_ids,
            device=device, batch_size=a.batch_size,
        )
        hidden = hidden.detach().cpu().float()
        probs = probs.detach().cpu().float()
        if a.cache_output_dir:
            out = Path(a.cache_output_dir).resolve()
            out.mkdir(parents=True, exist_ok=True)
            torch.save(hidden, out / "base_retain_hidden.pt")
            torch.save(probs, out / "base_retain_raw_selected_probs.pt")

    if hidden.ndim != 2 or probs.ndim != 2:
        raise ValueError("retain caches must be rank-2")
    if hidden.shape[0] != probs.shape[0] or probs.shape[1] != len(selected_ids):
        raise ValueError("retain caches do not align with selected rows")
    return hidden.float(), probs.float(), source


def quantile(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.float(), q).item())


def main() -> None:
    a = parse_args()
    if a.batch_size <= 0 or a.top_prompts <= 0 or a.top_contributors <= 0:
        raise ValueError("batch/top counts must be positive")

    retain_path = Path(a.training_visible_retain_path).resolve()
    retain_records, case_ids, prompt_texts = load_prompt_only_retain(retain_path)
    selected_ids, delta = load_delta(Path(a.delta_path).resolve(), a.scale)
    hidden, probs, cache_source = load_or_build_caches(a, retain_records, selected_ids)
    if hidden.shape[1] != delta.shape[1]:
        raise ValueError("hidden size and delta hidden dimension differ")

    # All audit math is CPU FP64 for stable exp/log at large sparse logit shifts.
    h = hidden.double()
    p = probs.double().clamp(min=0.0, max=1.0)
    d = h @ delta.double().transpose(0, 1)

    # Exact partition ratio. log1p(sum p(expm1(d))) is algebraically log(A).
    partition_increment = (p * torch.expm1(d)).sum(dim=1)
    A = 1.0 + partition_increment
    if torch.any(A <= 0) or not torch.isfinite(A).all():
        raise FloatingPointError("non-positive/non-finite exact partition ratio A")
    exact_kl = torch.log(A) - (p * d).sum(dim=1)
    # Roundoff may produce tiny negative values around zero only.
    min_kl = float(exact_kl.min().item())
    if min_kl < -1e-9:
        raise FloatingPointError(f"exact KL became materially negative: {min_kl}")
    exact_kl = exact_kl.clamp_min(0.0)

    thresholds = [float(x.strip()) for x in a.thresholds.split(",") if x.strip()]
    if not thresholds or any((not math.isfinite(x) or x < 0) for x in thresholds):
        raise ValueError("thresholds must be finite non-negative numbers")

    # Old normalized D_R, reproduced exactly for comparison.
    normalized_w = v2.normalized_clipped_weights(probs.float(), a.retain_weight_clip).double()
    old_normalized_action = float((normalized_w * d.square()).mean().item())

    # Probability-weighted action requested in the protocol note.
    bern_var = p * (1.0 - p)
    denom = bern_var.sum().clamp_min(1e-300)
    global_prob_action = float((bern_var * d.square()).sum().div(denom).item())

    top_k = min(a.top_prompts, exact_kl.numel())
    top_vals, top_idx = torch.topk(exact_kl, k=top_k, largest=True)
    worst_prompts: List[Dict[str, Any]] = []
    for kval, ridx_t in zip(top_vals.tolist(), top_idx.tolist()):
        ridx = int(ridx_t)
        worst_prompts.append({
            "retain_index": ridx,
            "case_id": int(case_ids[ridx]),
            "prompt": prompt_texts[ridx],
            "exact_kl_base_to_edited": float(kval),
            "partition_ratio_A": float(A[ridx].item()),
            "sum_p_d": float((p[ridx] * d[ridx]).sum().item()),
            "max_abs_selected_logit_shift": float(d[ridx].abs().max().item()),
            "selected_probability_mass": float(p[ridx].sum().item()),
        })

    # Token-level diagnostics. Exact KL is not additive because of log(A), so we
    # explicitly report additive pre-log partition and quadratic contributors.
    partition_abs = (p * torch.expm1(d)).abs()
    quadratic = bern_var * d.square()
    flat_score = torch.maximum(partition_abs / partition_abs.max().clamp_min(1e-300),
                               quadratic / quadratic.max().clamp_min(1e-300))
    flat = flat_score.reshape(-1)
    ck = min(a.top_contributors, flat.numel())
    _, cidx = torch.topk(flat, k=ck, largest=True)
    contributors: List[Dict[str, Any]] = []
    S = len(selected_ids)
    for flat_i in cidx.tolist():
        ridx = int(flat_i // S)
        sidx = int(flat_i % S)
        contributors.append({
            "retain_index": ridx,
            "case_id": int(case_ids[ridx]),
            "prompt": prompt_texts[ridx],
            "token_id": int(selected_ids[sidx]),
            "base_probability": float(p[ridx, sidx].item()),
            "logit_shift_d": float(d[ridx, sidx].item()),
            "abs_partition_contribution_p_expm1_d": float(partition_abs[ridx, sidx].item()),
            "quadratic_p1mp_d2": float(quadratic[ridx, sidx].item()),
            "prompt_exact_kl": float(exact_kl[ridx].item()),
        })

    result = {
        "schema_version": 1,
        "audit": "exact_sparse_output_row_retain_kl",
        "retain_role": "training_visible_prompt_only",
        "heldout_probes_seen": 0,
        "retain_eval_seen": 0,
        "edited_model_forward_passes": 0,
        "cache_source": cache_source,
        "retain_prompt_count": int(len(retain_records)),
        "selected_row_count": int(len(selected_ids)),
        "selected_row_ids": selected_ids,
        "delta_path": str(Path(a.delta_path).resolve()),
        "scale": float(a.scale),
        "exact_kl_base_to_edited": {
            "mean": float(exact_kl.mean().item()),
            "median": quantile(exact_kl, 0.5),
            "p95": quantile(exact_kl, 0.95),
            "p99": quantile(exact_kl, 0.99),
            "max": float(exact_kl.max().item()),
            "counts_above_threshold": {
                format(t, ".12g"): int((exact_kl > t).sum().item()) for t in thresholds
            },
        },
        "actions": {
            "old_row_normalized_D_R": old_normalized_action,
            "global_probability_weighted_p1mp_action": global_prob_action,
            "global_probability_weight_denominator": float(denom.item()),
        },
        "selected_probability": {
            "mean": float(p.mean().item()),
            "max": float(p.max().item()),
            "mean_selected_mass_per_prompt": float(p.sum(dim=1).mean().item()),
            "max_selected_mass_per_prompt": float(p.sum(dim=1).max().item()),
        },
        "logit_shift": {
            "mean_abs": float(d.abs().mean().item()),
            "p99_abs": quantile(d.abs().reshape(-1), 0.99),
            "max_abs": float(d.abs().max().item()),
        },
        "worst_prompts": worst_prompts,
        "worst_prompt_token_contributors": contributors,
        "contributor_note": (
            "Exact prompt KL is not token-additive because of log(A); token diagnostics report "
            "absolute p*expm1(d) partition contribution and p(1-p)d^2 quadratic contribution."
        ),
    }

    out = Path(a.output_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    k = result["exact_kl_base_to_edited"]
    print("Exact retain-KL audit complete:", out)
    print("retain prompts:", len(retain_records))
    print("selected rows:", len(selected_ids))
    print("scale:", a.scale)
    print("mean KL:", k["mean"])
    print("median KL:", k["median"])
    print("p95 KL:", k["p95"])
    print("p99 KL:", k["p99"])
    print("max KL:", k["max"])
    print("counts above thresholds:", json.dumps(k["counts_above_threshold"], sort_keys=True))
    print("old normalized D_R:", old_normalized_action)
    print("global p(1-p)-weighted action:", global_prob_action)
    print("edited-model forward passes: 0")


if __name__ == "__main__":
    main()
