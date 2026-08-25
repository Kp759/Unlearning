#!/usr/bin/env python3
"""Is the Stage-1 hidden-state change separable enough to read back from the LM head?

Context. probe_marker_reachability showed that an input-embedding edit cannot
place h into a CHOSEN low-variance direction: to move h.v at all, the optimizer
moved h with norm ~300 while only 0.5-2.4% of that change lay along v, and
confining the write simply stopped it. Choosing the readout direction first is
therefore not viable.

This asks the reachable question instead. Let Stage 1's edit produce whatever
dh = h_edited - h_base it naturally produces, and find the BEST readout for it.
Over all directions v, the separation between the edited state and the base
distribution is maximized by the whitened direction

    v* = C^-1 dh / ||C^-1 dh||,        k_max = sqrt( dh^T C^-1 dh )

with C the base hidden-state covariance. k_max is the Mahalanobis norm of dh:
the largest achievable marker strength for ANY readout, i.e. an upper bound the
low-variance-eigenvector construction could never beat.

If that holds up, the architecture is:

    Stage 1  E[subject]          trained -- already gives Eff 0.00 / Gen 1.70
    Stage 2  W[target_true] -= beta v*   closed form, beta = delta / (dh . v*)

and the spurious shift on a prompt without the subject is delta / k_max. Because
Stage 2 touches only the sensitive token's row, other facts about the subject
stay untouched -- the fact-level property an embedding-only edit cannot provide.

Requires no training: it reads an existing Stage-1 checkpoint against base.

    python scripts/measure_whitened_separability.py \\
      --base-model /path/to/Llama-3.2-3B-Instruct \\
      --edited-model outputs/.../stage1_subject_emb/checkpoint \\
      --training-visible-path outputs/.../protocol/training_visible_mcf_target_true.json \\
      --wikidata-dir data/wikipedia_sure_100020 \\
      --out outputs/whitened_separability.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

import gagd_compare as gagd
import mcf_sure_subject_directional_emb_stage1 as subj
import sure_canonical_core as core


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", required=True)
    p.add_argument("--edited-model", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--out", required=True)
    p.add_argument("--hidden-samples", type=int, default=2500,
                   help="documents for the base covariance; every token position is used")
    p.add_argument("--doc-start", type=int, default=20)
    p.add_argument("--shrinkage", type=float, default=1e-3,
                   help="ridge on C before inversion, as a fraction of trace/d")
    p.add_argument("--deltas", default="5,10,15")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device-map", default="single")
    a = p.parse_args(argv)
    if a.doc_start < 20:
        p.error("--doc-start must be >= 20")
    return a


@torch.no_grad()
def hidden_states_over_text(model, tok, texts, device, batch_size, min_states):
    chunks: List[torch.Tensor] = []
    total = 0
    for start in range(0, len(texts), batch_size):
        if total >= min_states:
            break
        batch = [t[:2000] for t in texts[start : start + batch_size]]
        enc = tok(batch, padding=True, truncation=True, max_length=256,
                  return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        sel = out.hidden_states[-1].float()[enc["attention_mask"].bool()].cpu()
        chunks.append(sel)
        total += int(sel.shape[0])
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0)


@torch.no_grad()
def forget_prompt_hidden(model, tok, records, device, batch_size):
    prompts = [
        str(r["requested_rewrite"]["prompt"]).format(
            str(r["requested_rewrite"]["subject"])
        )
        for r in records
    ]
    chunks: List[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tok(batch, padding=True, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        pos = enc["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(batch), device=out.logits.device)
        chunks.append(out.hidden_states[-1][rows, pos, :].float().cpu())
    return torch.cat(chunks, dim=0), prompts


def load(path: str, a: argparse.Namespace):
    ns = argparse.Namespace(model_path=path, dtype=a.dtype,
                            device_map=a.device_map, gradient_checkpointing=False)
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok, gagd.first_device(model)


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(0)
    records = json.loads(Path(a.training_visible_path).read_text(encoding="utf-8"))
    deltas = [float(x) for x in str(a.deltas).split(",") if x.strip()]

    model, tok, device = load(a.base_model, a)
    d = int(model.get_input_embeddings().weight.shape[1])
    docs = subj.load_frequency_documents(a.wikidata_dir, a.doc_start, a.hidden_samples)
    H = hidden_states_over_text(model, tok, docs, device, a.batch_size, 4 * d)
    if H.shape[0] < 2 * d:
        raise SystemExit(
            f"only {H.shape[0]} hidden states for d={d}; the covariance would be "
            "rank-deficient. Point --wikidata-dir at a larger corpus."
        )
    h_base, prompts = forget_prompt_hidden(model, tok, records, device, a.batch_size)
    del model
    torch.cuda.empty_cache()

    model, tok, device = load(a.edited_model, a)
    h_edit, _ = forget_prompt_hidden(model, tok, records, device, a.batch_size)
    del model
    torch.cuda.empty_cache()

    mu = H.mean(dim=0, keepdim=True)
    Hc = (H - mu).double()
    C = (Hc.T @ Hc) / max(1, Hc.shape[0] - 1)
    ridge = float(a.shrinkage) * float(torch.diagonal(C).mean())
    C = C + ridge * torch.eye(d, dtype=torch.float64)
    L = torch.linalg.cholesky(C)

    rows: List[Dict[str, Any]] = []
    for i in range(min(h_base.shape[0], h_edit.shape[0])):
        dh = (h_edit[i] - h_base[i]).double()
        # solve C x = dh without forming the inverse
        x = torch.cholesky_solve(dh.unsqueeze(1), L).squeeze(1)
        k_max = float(torch.sqrt(torch.clamp(dh @ x, min=0.0)))
        v = x / x.norm().clamp_min(1e-12)
        # sanity: base spread along v, and the achieved projection
        sigma_v = float(torch.sqrt(torch.clamp(v @ (C @ v), min=0.0)))
        proj = float(dh @ v)
        rows.append({
            "record_index": i,
            "prompt": prompts[i],
            "dh_norm": float(dh.norm()),
            "k_max_mahalanobis": k_max,
            "sigma_along_v_star": sigma_v,
            "dh_projection_on_v_star": proj,
            "leak_per_delta": {str(dv): (dv / k_max if k_max > 0 else None)
                               for dv in deltas},
        })

    ks = sorted(r["k_max_mahalanobis"] for r in rows)
    med = ks[len(ks) // 2] if ks else 0.0
    payload = {
        "base_model": a.base_model,
        "edited_model": a.edited_model,
        "hidden_dim": d,
        "covariance_states": int(H.shape[0]),
        "shrinkage": float(a.shrinkage),
        "records": len(rows),
        "k_max_median": med,
        "k_max_min": ks[0] if ks else None,
        "k_max_max": ks[-1] if ks else None,
        "median_leak_per_delta": {str(dv): (dv / med if med > 0 else None)
                                  for dv in deltas},
        "definition": (
            "k_max = sqrt(dh^T C^-1 dh), the Mahalanobis norm of the Stage-1 "
            "hidden-state change. It is the maximum separation achievable by ANY "
            "readout direction, attained at v* = C^-1 dh / ||C^-1 dh||. The "
            "spurious logit shift on a prompt without the subject is delta/k_max."
        ),
        "interpretation": (
            "k_max >= 50 means a rank-one LM-head read of the existing Stage-1 "
            "edit leaks <= 0.2 logits at delta=10, and the paired architecture is "
            "viable with NO change to Stage 1. k_max ~ 5 means the leak is ~2 "
            "logits and the read reintroduces the Eff/Spe coupling. Unlike "
            "probe_marker_reachability this asks nothing of the network: v* is "
            "derived from the dh the edit already produces."
        ),
        "per_record": rows,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\ncovariance states {H.shape[0]} (d={d}), ridge={ridge:.3e}")
    print(f"k_max  median={med:.1f}  min={ks[0]:.1f}  max={ks[-1]:.1f}")
    for dv in deltas:
        print(f"   leak at delta={dv:g}: {dv/med if med>0 else float('nan'):.3f} logits")
    print("\nviable if k_max is large (>=50 -> <=0.2 logits at delta=10);"
          "\nreintroduces the coupling if k_max is small.")
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
