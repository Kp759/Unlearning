#!/usr/bin/env python3
"""Minimum-norm projection of MQuAKE sensitive LM-head row deltas.

Input checkpoint must be the output-only-restored Stage-1 checkpoint:
  * transformer frozen/base,
  * all input embeddings restored to base,
  * LM head untied,
  * non-sensitive LM-head rows restored to base,
  * sensitive LM-head rows retain Stage-1 displacement.

For the locked direct forget token decisions, collect final hidden states H.  Since
input embeddings and transformer are base, these are base hidden states.  Let D
be the sensitive output-row displacement relative to the base input-embedding
rows (valid because the original Llama head is tied to embeddings).  Replace

    D -> D P_H

where P_H projects onto row-span(H).  This is the minimum-Frobenius-norm
component of D that preserves D's action on every training-visible forget hidden
state.  It removes components that cannot affect any locked forget decision,
including tied input-gradient contamination carried into the output rows.

No retain, AtomicGen, multi-hop question, PPL text, target_new, Unknown, or IDK
is loaded or used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List

import torch
from tqdm import tqdm

import gagd_compare as gagd
import mquake_forget_only_no_neutral as locked
import mquake_zero_unlearn_official_eval as mq

PROTOCOL = "mquake_zerounlearn_forget_only_locked_no_neutral"
METHOD = "SURE-MQuAKE-minimum-norm-forget-subspace-projection"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--svd-relative-tol", type=float, default=1e-6)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@torch.no_grad()
def hidden_states(model, tok, cases, device, batch_size: int) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    model.eval()
    for start in tqdm(range(0, len(cases), batch_size), desc="collect MQuAKE forget hidden states"):
        batch = cases[start:start + batch_size]
        enc = tok([c.prompt for c in batch], padding=True, return_tensors="pt").to(device)
        result = model(**enc, use_cache=False, output_hidden_states=True)
        h = result.hidden_states[-1]
        pos = enc["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(batch), device=device)
        chunks.append(h[rows, pos, :].detach().float().cpu())
    return torch.cat(chunks, dim=0)


def main() -> None:
    a = parse_args()
    if a.batch_size <= 0 or a.svd_relative_tol <= 0:
        raise ValueError("batch-size and svd-relative-tol must be positive")
    gagd.set_seed(a.seed)

    vp = Path(a.training_visible_path).resolve()
    mp = Path(a.split_manifest).resolve()
    records, manifest = locked.load_locked(vp, mp, a.seed, a.forget_num)

    ns = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = gagd.first_device(model)
    llama_like = mq.is_llama_like(model, tok)

    inp = model.get_input_embeddings()
    out = model.get_output_embeddings()
    tied_before = inp.weight.data_ptr() == out.weight.data_ptr()
    if tied_before:
        raise RuntimeError(
            "projection expects output-only Stage1 checkpoint with untied LM head"
        )

    cases = locked.all_cases(records, tok, llama_like)
    tids = locked.target_ids(tok, cases, llama_like, device)
    sensitive_ids = sorted(set(int(x) for x in tids.detach().cpu().tolist()))
    ids_in = torch.tensor(sensitive_ids, dtype=torch.long, device=inp.weight.device)
    ids_out = ids_in.to(out.weight.device)

    # The output-only Stage1 checkpoint restored the entire input matrix to base.
    # Original Llama is tied, therefore the base output row equals this base input row.
    base_sensitive = inp.weight.index_select(0, ids_in).detach().float().cpu()
    trained_sensitive = out.weight.index_select(0, ids_out).detach().float().cpu()
    delta = trained_sensitive - base_sensitive

    H = hidden_states(model, tok, cases, device, a.batch_size)
    if H.ndim != 2 or H.shape[1] != delta.shape[1]:
        raise RuntimeError(f"hidden/delta dimension mismatch: H={tuple(H.shape)} D={tuple(delta.shape)}")

    # Row-space basis of H.  Vh rows are orthonormal directions in hidden space.
    _u, s, vh = torch.linalg.svd(H.float(), full_matrices=False)
    if s.numel() == 0 or float(s.max()) <= 0:
        raise RuntimeError("forget hidden-state matrix has no nonzero singular values")
    threshold = float(s.max()) * float(a.svd_relative_tol)
    rank = int((s > threshold).sum().item())
    if rank <= 0:
        raise RuntimeError("numerical forget hidden-state rank is zero")
    basis = vh[:rank, :].contiguous()  # [rank, hidden]

    projected = (delta @ basis.T) @ basis
    removed = delta - projected

    # Projection must preserve each row delta's logit action on every direct
    # training hidden state (up to floating-point SVD tolerance).
    action_before = H @ delta.T
    action_after = H @ projected.T
    max_action_error = float((action_before - action_after).abs().max().item())
    rms_action_error = float((action_before - action_after).pow(2).mean().sqrt().item())

    with torch.no_grad():
        new_rows = base_sensitive.to(out.weight.device, dtype=out.weight.dtype) + projected.to(
            out.weight.device, dtype=out.weight.dtype
        )
        out.weight.index_copy_(0, ids_out, new_rows)

    # Verify input embeddings remain untouched by output materialization.
    input_after = inp.weight.index_select(0, ids_in).detach().float().cpu()
    input_base_error = float((input_after - base_sensitive).abs().max().item())

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    dnorm = float(delta.norm().item())
    pnorm = float(projected.norm().item())
    rnorm = float(removed.norm().item())
    report = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "seed": int(a.seed),
        "forget_instances": int(a.forget_num),
        "forget_atomic_facts": len(records),
        "direct_sensitive_token_cases": len(cases),
        "sensitive_lm_head_rows": len(sensitive_ids),
        "hidden_matrix_shape": [int(H.shape[0]), int(H.shape[1])],
        "hidden_subspace_rank": rank,
        "svd_relative_tol": float(a.svd_relative_tol),
        "singular_value_max": float(s.max().item()),
        "singular_value_min_kept": float(s[rank - 1].item()),
        "sensitive_delta_shape": [int(delta.shape[0]), int(delta.shape[1])],
        "delta_fro_norm_before": dnorm,
        "delta_fro_norm_after_projection": pnorm,
        "delta_fro_norm_removed": rnorm,
        "norm_retained_fraction": None if dnorm == 0 else pnorm / dnorm,
        "norm_removed_fraction": None if dnorm == 0 else rnorm / dnorm,
        "max_direct_forget_logit_action_error_fp32": max_action_error,
        "rms_direct_forget_logit_action_error_fp32": rms_action_error,
        "input_embeddings_all_rows_already_base": True,
        "input_sensitive_row_max_change_from_base": input_base_error,
        "lm_head_non_sensitive_rows_already_base": True,
        "lm_head_sensitive_rows_projected_only": True,
        "tied_input_output": False,
        "selection_uses_heldout": False,
        "retain_seen": 0,
        "atomic_questions_seen": 0,
        "multihop_questions_seen": 0,
        "PPL_seen": False,
        "target_new_seen": False,
        "Unknown_used": False,
        "IDK_used": False,
        "split_sampling": manifest.get("sampling"),
        "checkpoint": str(ckpt.resolve()),
    }
    write_json(out_dir / "projection_report.json", report)
    print("===== MQuAKE MINIMUM-NORM FORGET-SUBSPACE PROJECTION =====")
    print("hidden matrix:", tuple(H.shape), "rank:", rank)
    print("sensitive rows:", len(sensitive_ids))
    print("delta norm:", dnorm, "->", pnorm, "removed", rnorm)
    print("direct-logit action max error:", max_action_error)
    print("input base max error:", input_base_error)
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
