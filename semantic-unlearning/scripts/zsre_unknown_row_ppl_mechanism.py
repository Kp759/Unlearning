#!/usr/bin/env python3
"""Mechanistically attribute ZsRE Stage-2 PPL changes to the Unknown LM-head row.

This diagnostic uses only:
  * an existing Stage-1 checkpoint,
  * its corresponding Stage-2 checkpoint, and
  * the fixed Wikidata text used by the ZeroUnlearn-compatible PPL evaluator.

It never loads ZsRE benchmark records, rephrases, locality probes, or retain
records.  It is diagnostic only and must not be used for checkpoint selection.

For the exact PPL input positions it measures:
  z2(Unknown) - z1(Unknown),
  p2(Unknown) versus p1(Unknown),
  the correct-next-token log-probability drop,
  whether non-Unknown logits changed, and
  a counterfactual Stage-2 PPL after replacing only the Stage-2 Unknown logit
  with its Stage-1 value.  If that counterfactual PPL returns to Stage-1 PPL,
  the single Unknown output row is sufficient to explain the utility collapse.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mcf_zero_unlearn_official_eval import dtype_from_str, load_official_ppl_text
from zsre_zero_unlearn_official_eval import resolve_neutral_target_token_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1-checkpoint", required=True)
    p.add_argument("--stage2-checkpoint", required=True)
    p.add_argument("--wikidata-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--max-input-length", type=int, default=100)
    p.add_argument("--top-k-positions", type=int, default=20)
    return p.parse_args()


def stats(x: torch.Tensor) -> Dict[str, float]:
    a = x.detach().float().cpu().numpy()
    return {
        "min": float(np.min(a)),
        "p05": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.percentile(a, 50)),
        "mean": float(np.mean(a)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def official_ppl_from_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    """Match ZeroUnlearn util/perplexity.py denominator convention exactly."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    targets = input_ids[:, 1:].to(logp.device)
    gathered = torch.gather(logp[:, :-1, :], 2, targets[..., None])[0]
    return float(torch.exp(-gathered.sum() / input_ids.size(1)).item())


def load_logits_and_row(
    checkpoint: Path,
    input_ids_cpu: torch.Tensor,
    dtype: str,
    unknown_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    kwargs = {"torch_dtype": dtype_from_str(dtype)}
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), **kwargs)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")
    model = model.to("cuda")
    model.eval()
    model.config.use_cache = False
    ids = input_ids_cpu.to("cuda")
    with torch.no_grad():
        logits = model(input_ids=ids).logits.detach().float().cpu()
        out = model.get_output_embeddings()
        if out is None:
            raise RuntimeError("Model has no output embedding / LM head")
        row = out.weight[unknown_id].detach().float().cpu().clone()
    del model
    torch.cuda.empty_cache()
    return logits, row


def main() -> None:
    args = parse_args()
    s1_path = Path(args.stage1_checkpoint)
    s2_path = Path(args.stage2_checkpoint)
    wikidata = Path(args.wikidata_dir)
    if not wikidata.exists():
        raise FileNotFoundError(wikidata)

    tok = AutoTokenizer.from_pretrained(str(s1_path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    unknown_id = resolve_neutral_target_token_id(tok)

    ppl_text = load_official_ppl_text(wikidata)
    if ppl_text is None:
        raise RuntimeError(f"Could not load fixed PPL text from {wikidata}")
    encoded = tok(
        [ppl_text],
        return_tensors="pt",
        max_length=args.max_input_length,
        truncation=True,
    )
    input_ids = encoded["input_ids"].cpu()

    print(f"seed={args.seed} unknown_token_id={unknown_id} token={tok.decode([unknown_id])!r}")
    print(f"PPL tokenized length={input_ids.size(1)}")
    print("Loading Stage 1...")
    logits1, row1 = load_logits_and_row(s1_path, input_ids, args.dtype, unknown_id)
    print("Loading Stage 2...")
    logits2, row2 = load_logits_and_row(s2_path, input_ids, args.dtype, unknown_id)

    if logits1.shape != logits2.shape:
        raise RuntimeError(f"logit shape mismatch: {logits1.shape} vs {logits2.shape}")

    ppl1 = official_ppl_from_logits(logits1, input_ids)
    ppl2 = official_ppl_from_logits(logits2, input_ids)

    # Counterfactual: keep every Stage-2 logit exactly as observed, but restore
    # only the Unknown logit at each position to its Stage-1 value.
    cf_logits = logits2.clone()
    cf_logits[:, :, unknown_id] = logits1[:, :, unknown_id]
    ppl2_unknown_restored = official_ppl_from_logits(cf_logits, input_ids)

    # Conversely, inject only the Stage-2 Unknown logits into otherwise Stage-1
    # logits. This should reproduce Stage-2 PPL if the Unknown row is sufficient.
    inject_logits = logits1.clone()
    inject_logits[:, :, unknown_id] = logits2[:, :, unknown_id]
    ppl1_unknown_injected = official_ppl_from_logits(inject_logits, input_ids)

    positions = input_ids.size(1) - 1
    z1u = logits1[0, :positions, unknown_id]
    z2u = logits2[0, :positions, unknown_id]
    dz = z2u - z1u

    lp1 = torch.log_softmax(logits1[0, :positions, :], dim=-1)
    lp2 = torch.log_softmax(logits2[0, :positions, :], dim=-1)
    targets = input_ids[0, 1:]
    idx = torch.arange(positions)
    correct_lp1 = lp1[idx, targets]
    correct_lp2 = lp2[idx, targets]
    correct_nll_increase = -(correct_lp2 - correct_lp1)
    p1u = lp1[:, unknown_id].exp()
    p2u = lp2[:, unknown_id].exp()

    # Validate the architectural claim that Stage 2 changed only Unknown output
    # logits. Ignore that vocabulary column and report residual differences.
    residual = (logits2[0, :positions, :] - logits1[0, :positions, :]).abs()
    residual[:, unknown_id] = 0.0
    max_non_unknown_logit_change = float(residual.max().item())
    mean_non_unknown_logit_change = float(residual.mean().item())

    row_delta = row2 - row1
    row_delta_norm = float(row_delta.norm().item())
    row_cosine = float(torch.nn.functional.cosine_similarity(row1[None], row2[None]).item())

    unknown_top1_stage1 = float((logits1[0, :positions, :].argmax(dim=-1) == unknown_id).float().mean().item())
    unknown_top1_stage2 = float((logits2[0, :positions, :].argmax(dim=-1) == unknown_id).float().mean().item())

    k = min(args.top_k_positions, positions)
    top = torch.topk(dz, k=k).indices.tolist()
    per_position = []
    for pos in top:
        target_id = int(targets[pos].item())
        per_position.append({
            "position": int(pos),
            "context_token_id": int(input_ids[0, pos].item()),
            "context_token": tok.decode([int(input_ids[0, pos].item())]),
            "target_token_id": target_id,
            "target_token": tok.decode([target_id]),
            "stage1_unknown_logit": float(z1u[pos].item()),
            "stage2_unknown_logit": float(z2u[pos].item()),
            "delta_unknown_logit": float(dz[pos].item()),
            "stage1_unknown_probability": float(p1u[pos].item()),
            "stage2_unknown_probability": float(p2u[pos].item()),
            "stage1_correct_logprob": float(correct_lp1[pos].item()),
            "stage2_correct_logprob": float(correct_lp2[pos].item()),
            "correct_nll_increase": float(correct_nll_increase[pos].item()),
        })

    # Fraction of Stage-2 NLL increase removed by restoring the Unknown logit,
    # expressed in log-PPL space so multiplicative PPL changes are additive.
    log_gap = math.log(ppl2) - math.log(ppl1)
    residual_log_gap = math.log(ppl2_unknown_restored) - math.log(ppl1)
    if abs(log_gap) < 1e-12:
        explained_fraction = float("nan")
    else:
        explained_fraction = float(1.0 - residual_log_gap / log_gap)

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "zsre_unknown_row_ppl_mechanism",
        "seed": int(args.seed),
        "stage1_checkpoint": str(s1_path.resolve()),
        "stage2_checkpoint": str(s2_path.resolve()),
        "wikidata_dir": str(wikidata.resolve()),
        "unknown_token_id": int(unknown_id),
        "unknown_token": tok.decode([unknown_id]),
        "ppl": {
            "stage1": ppl1,
            "stage2": ppl2,
            "stage2_with_unknown_logit_restored_to_stage1": ppl2_unknown_restored,
            "stage1_with_stage2_unknown_logit_injected": ppl1_unknown_injected,
            "stage2_over_stage1_ratio": float(ppl2 / ppl1),
            "unknown_row_explained_fraction_of_log_ppl_increase": explained_fraction,
        },
        "unknown_row": {
            "delta_norm": row_delta_norm,
            "stage1_stage2_cosine": row_cosine,
        },
        "unknown_logit_delta": stats(dz),
        "unknown_probability_stage1": stats(p1u),
        "unknown_probability_stage2": stats(p2u),
        "correct_token_nll_increase_stage2_minus_stage1": stats(correct_nll_increase),
        "unknown_is_top1_fraction": {
            "stage1": unknown_top1_stage1,
            "stage2": unknown_top1_stage2,
        },
        "non_unknown_logit_change_validation": {
            "max_abs": max_non_unknown_logit_change,
            "mean_abs": mean_non_unknown_logit_change,
        },
        "largest_positive_unknown_logit_shifts": per_position,
        "data_access": {
            "fixed_wikidata_ppl_text_only": True,
            "zsre_benchmark_records_loaded": 0,
            "zsre_rephrases_loaded": 0,
            "zsre_locality_loaded": 0,
            "zsre_retain_loaded": 0,
            "selection_or_tuning_use": False,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n===== UNKNOWN-ROW PPL MECHANISM =====")
    print(f"Stage-1 PPL: {ppl1:.6f}")
    print(f"Stage-2 PPL: {ppl2:.6f}")
    print(f"Stage-2 with Unknown restored: {ppl2_unknown_restored:.6f}")
    print(f"Stage-1 with Stage-2 Unknown injected: {ppl1_unknown_injected:.6f}")
    print(f"Unknown-row explained fraction (log-PPL): {explained_fraction:.6f}")
    print(f"Unknown row delta norm: {row_delta_norm:.6f}")
    print(f"Unknown delta-logit mean/p95/max: {stats(dz)['mean']:.4f} / {stats(dz)['p95']:.4f} / {stats(dz)['max']:.4f}")
    print(f"Unknown top-1 fraction Stage1 -> Stage2: {unknown_top1_stage1:.4f} -> {unknown_top1_stage2:.4f}")
    print(f"Max non-Unknown logit change: {max_non_unknown_logit_change:.6g}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
