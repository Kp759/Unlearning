#!/usr/bin/env python3
"""Decompose MQuAKE PPL change into sensitive-row softmax effects.

This is a post-hoc diagnostic only.  It is never used for training, repair,
checkpoint selection, or scale selection.

For the output-only-restored MQuAKE variants the transformer and all input
embeddings are base-identical, while only a sparse set of LM-head rows differ.
For each of the first <=100 ZeroUnlearn PPL tokens, this script computes in
FP32:

    delta_NLL = delta_logZ - delta_target_logit

and separates positions whose target token is / is not one of the sensitive
LM-head rows.  It also ranks sensitive rows by their added softmax mass relative
to the base normalizer.  This identifies whether PPL degradation is caused by
(1) target-token collisions or (2) global softmax-normalizer inflation.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mcf_zero_unlearn_official_eval import dtype_from_str, load_official_ppl_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", required=True)
    p.add_argument("--edited-model", required=True)
    p.add_argument("--restoration-json", required=True)
    p.add_argument("--wikidata-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--max-input-length", type=int, default=100)
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def load_logits(model_path: str, input_ids: torch.Tensor, dtype: str) -> torch.Tensor:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype_from_str(dtype)
    ).to("cuda")
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids.to("cuda"), use_cache=False).logits
        logits = logits[0, :-1, :].float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return logits


def stats(xs: torch.Tensor) -> Dict[str, float | int | None]:
    if xs.numel() == 0:
        return {"count": 0, "mean": None, "sum": None, "max": None, "min": None}
    return {
        "count": int(xs.numel()),
        "mean": float(xs.mean()),
        "sum": float(xs.sum()),
        "max": float(xs.max()),
        "min": float(xs.min()),
    }


def main() -> None:
    a = parse_args()
    restore = json.loads(Path(a.restoration_json).read_text())
    sensitive_ids = sorted(set(int(x) for x in restore["sensitive_token_ids"]))
    sensitive_set = set(sensitive_ids)

    text = load_official_ppl_text(a.wikidata_dir)
    if text is None:
        raise FileNotFoundError(f"PPL dataset not found: {a.wikidata_dir}")

    tok = AutoTokenizer.from_pretrained(a.base_model)
    enc = tok(
        [text], return_tensors="pt", max_length=a.max_input_length, truncation=True
    )
    ids = enc["input_ids"]
    targets = ids[0, 1:].long().cpu()

    base = load_logits(a.base_model, ids, a.dtype)
    edited = load_logits(a.edited_model, ids, a.dtype)
    if base.shape != edited.shape or base.shape[0] != targets.numel():
        raise RuntimeError(
            f"shape mismatch base={tuple(base.shape)} edited={tuple(edited.shape)} "
            f"targets={targets.numel()}"
        )

    rows = torch.arange(targets.numel())
    base_logz = torch.logsumexp(base, dim=-1)
    edit_logz = torch.logsumexp(edited, dim=-1)
    base_target = base[rows, targets]
    edit_target = edited[rows, targets]
    base_nll = base_logz - base_target
    edit_nll = edit_logz - edit_target
    delta_nll = edit_nll - base_nll
    delta_logz = edit_logz - base_logz
    delta_target = edit_target - base_target
    identity_error = (delta_nll - (delta_logz - delta_target)).abs()

    sensitive_mask = torch.tensor(
        [int(t) in sensitive_set for t in targets.tolist()], dtype=torch.bool
    )
    nonsensitive_mask = ~sensitive_mask

    # Match the project/ZeroUnlearn implementation: it sums N-1 next-token NLLs
    # but divides by the full input sequence length N.
    seq_len = int(ids.shape[1])
    base_ppl_official_fp32 = math.exp(float(base_nll.sum()) / seq_len)
    edit_ppl_official_fp32 = math.exp(float(edit_nll.sum()) / seq_len)
    # Also report conventional next-token PPL with denominator N-1.
    base_ppl_standard_fp32 = math.exp(float(base_nll.mean()))
    edit_ppl_standard_fp32 = math.exp(float(edit_nll.mean()))

    sid = torch.tensor(sensitive_ids, dtype=torch.long)
    base_s = base.index_select(1, sid)
    edit_s = edited.index_select(1, sid)
    # Added probability mass of each changed row relative to the *base* Z.
    # Summing across sensitive rows gives (Z'_s-Z_s)/Z_base at each position.
    mass_delta = torch.exp(edit_s - base_logz[:, None]) - torch.exp(
        base_s - base_logz[:, None]
    )
    row_mass_delta_mean = mass_delta.mean(dim=0)
    row_logit_delta_abs_mean = (edit_s - base_s).abs().mean(dim=0)
    order = torch.argsort(row_mass_delta_mean, descending=True)
    top_rows: List[Dict[str, Any]] = []
    for j in order[: min(a.top_k, len(sensitive_ids))].tolist():
        tid = sensitive_ids[j]
        top_rows.append(
            {
                "token_id": tid,
                "token": tok.decode([tid]),
                "mean_added_mass_relative_to_base_Z": float(row_mass_delta_mean[j]),
                "mean_abs_logit_delta_on_ppl_contexts": float(row_logit_delta_abs_mean[j]),
                "max_logit_delta_on_ppl_contexts": float((edit_s[:, j] - base_s[:, j]).max()),
                "min_logit_delta_on_ppl_contexts": float((edit_s[:, j] - base_s[:, j]).min()),
            }
        )

    pos_order = torch.argsort(delta_nll, descending=True)
    top_positions: List[Dict[str, Any]] = []
    for i in pos_order[: min(a.top_k, targets.numel())].tolist():
        tid = int(targets[i])
        top_positions.append(
            {
                "position": int(i + 1),
                "target_token_id": tid,
                "target_token": tok.decode([tid]),
                "target_is_sensitive": bool(sensitive_mask[i]),
                "base_nll": float(base_nll[i]),
                "edited_nll": float(edit_nll[i]),
                "delta_nll": float(delta_nll[i]),
                "delta_logZ": float(delta_logz[i]),
                "delta_target_logit": float(delta_target[i]),
            }
        )

    payload = {
        "schema_version": 1,
        "kind": "posthoc_mquake_ppl_sensitive_row_decomposition",
        "selection_uses_this_diagnostic": False,
        "base_model": a.base_model,
        "edited_model": a.edited_model,
        "restoration_json": str(Path(a.restoration_json).resolve()),
        "sequence_length_tokens": seq_len,
        "next_token_positions": int(targets.numel()),
        "sensitive_row_count": len(sensitive_ids),
        "sensitive_target_positions": int(sensitive_mask.sum()),
        "non_sensitive_target_positions": int(nonsensitive_mask.sum()),
        "ppl_fp32": {
            "project_official_denominator_N": {
                "base": base_ppl_official_fp32,
                "edited": edit_ppl_official_fp32,
                "delta": edit_ppl_official_fp32 - base_ppl_official_fp32,
            },
            "standard_denominator_N_minus_1": {
                "base": base_ppl_standard_fp32,
                "edited": edit_ppl_standard_fp32,
                "delta": edit_ppl_standard_fp32 - base_ppl_standard_fp32,
            },
        },
        "delta_nll_all": stats(delta_nll),
        "delta_nll_sensitive_targets": stats(delta_nll[sensitive_mask]),
        "delta_nll_non_sensitive_targets": stats(delta_nll[nonsensitive_mask]),
        "delta_logZ_all": stats(delta_logz),
        "delta_target_logit_sensitive_targets": stats(delta_target[sensitive_mask]),
        "delta_target_logit_non_sensitive_targets": stats(delta_target[nonsensitive_mask]),
        "decomposition_identity_max_abs_error": float(identity_error.max()),
        "mean_total_added_sensitive_mass_relative_to_base_Z": float(mass_delta.sum(dim=1).mean()),
        "max_total_added_sensitive_mass_relative_to_base_Z": float(mass_delta.sum(dim=1).max()),
        "top_sensitive_rows_by_added_softmax_mass": top_rows,
        "top_positions_by_nll_increase": top_positions,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    print("===== MQuAKE PPL SENSITIVE-ROW DECOMPOSITION (FP32) =====")
    print("tokens:", seq_len, "predictions:", targets.numel())
    print("sensitive target positions:", int(sensitive_mask.sum()), "/", targets.numel())
    print("official-style FP32 PPL:", base_ppl_official_fp32, "->", edit_ppl_official_fp32)
    print("standard FP32 PPL:", base_ppl_standard_fp32, "->", edit_ppl_standard_fp32)
    print("mean delta NLL all:", float(delta_nll.mean()))
    print("mean delta logZ all:", float(delta_logz.mean()))
    if sensitive_mask.any():
        print("mean delta NLL sensitive targets:", float(delta_nll[sensitive_mask].mean()))
        print("mean target-logit delta sensitive targets:", float(delta_target[sensitive_mask].mean()))
    if nonsensitive_mask.any():
        print("mean delta NLL non-sensitive targets:", float(delta_nll[nonsensitive_mask].mean()))
    print("mean added sensitive mass / base Z:", float(mass_delta.sum(dim=1).mean()))
    print("decomposition max error:", float(identity_error.max()))
    print("wrote:", out)


if __name__ == "__main__":
    main()
