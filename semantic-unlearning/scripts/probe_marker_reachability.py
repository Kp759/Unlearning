#!/usr/bin/env python3
"""Feasibility probe for paired write-read conjunctive editing.

The proposed architecture writes a marker into a rarely-used hidden direction
via the subject's embedding rows, and reads it back with a rank-one edit to the
sensitive token's LM-head row. Suppression is the product of the two:

    logit_true  -=  beta * (h . v),      beta = delta / alpha

and on a prompt WITHOUT the subject the marker is absent, so the spurious shift
is beta * sigma = delta / k, where alpha = k * sigma is the achieved marker
strength in units of the base standard deviation along v.

Everything therefore hinges on two numbers this script measures, before any of
the architecture is built:

  1. REACHABILITY -- what k can the embedding edit actually achieve, steering h
     along a *chosen* low-variance direction through 28 frozen layers? If k is
     small, beta must grow and the read end stops being clean.

  2. WRITE-SIDE COLLATERAL -- the write perturbs h by alpha*v, shifting every
     token's logit by alpha*(W_t . v). That is a rank-one bias on
     subject-containing prompts, and it is the part of the design that is NOT
     zero by construction. We report how much it reorders the top of the
     distribution on prompts about the same subject.

Reported: achieved k, the resulting leak delta/k at several delta, the marker's
LM-head coupling max_t |W_t . v|, and the top-1 agreement on same-subject
probes after the write alone.

This trains only a tiny delta on one record's subject rows for a few hundred
steps. It is a go/no-go measurement, not a method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

import gagd_compare as gagd
import mcf_sure_directional_emb_lm_stage1 as base1
import mcf_sure_subject_directional_emb_stage1 as subj
import sure_canonical_core as core


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--multi-counterfact", default="data/multi_counterfact.json")
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--out", required=True)
    p.add_argument("--records", type=int, default=3,
                   help="how many MCF records to probe")
    p.add_argument("--hidden-samples", type=int, default=512,
                   help="documents used to estimate the base hidden covariance")
    p.add_argument("--doc-start", type=int, default=20,
                   help="must stay >= 20, clear of official PPL documents")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target-k", type=float, default=50.0,
                   help="marker strength target, in base standard deviations")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device-map", default="single")
    a = p.parse_args(argv)
    if a.doc_start < 20:
        p.error("--doc-start must be >= 20")
    return a


@torch.no_grad()
def base_hidden_matrix(model, tok, texts, device, batch_size) -> torch.Tensor:
    """Last-position hidden states over ordinary text."""
    chunks: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = [t[:600] for t in texts[start : start + batch_size]]
        enc = tok(batch, padding=True, truncation=True, max_length=128,
                  return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        pos = enc["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(batch), device=out.logits.device)
        chunks.append(out.hidden_states[-1][rows, pos, :].float().cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0)


def pick_markers(H: torch.Tensor, W: torch.Tensor, n: int) -> Dict[str, Any]:
    """Bottom eigenvectors of the hidden covariance, ranked by LM-head coupling.

    Two criteria, not one. Low hidden variance keeps the READ end clean (leak
    scales with sigma). Low coupling to the LM-head rows keeps the WRITE end
    clean, since the write shifts every token's logit by alpha*(W_t . v).
    """
    Hc = H - H.mean(dim=0, keepdim=True)
    cov = (Hc.T @ Hc) / max(1, Hc.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov.double())
    evals = evals.float()
    evecs = evecs.float()
    # eigh returns ascending; the lowest-variance directions come first
    pool = min(4 * n + 32, evecs.shape[1])
    cand = evecs[:, :pool].T.contiguous()          # [pool, d]
    sigmas = evals[:pool].clamp_min(0).sqrt()
    coupling = (cand @ W.T).abs().max(dim=1).values  # max_t |W_t . v|
    # prefer small sigma AND small coupling; rank by their product
    score = sigmas * coupling
    order = torch.argsort(score)[:n]
    return {
        "vectors": cand[order],
        "sigmas": sigmas[order],
        "couplings": coupling[order],
        "median_sigma_all": float(evals.clamp_min(0).sqrt().median()),
    }


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(0)
    ns = argparse.Namespace(model_path=a.model_path, dtype=a.dtype,
                            device_map=a.device_map, gradient_checkpointing=False)
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    output_layer = core.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    output_layer.weight.requires_grad_(False)
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tok)
    model.eval()

    docs = subj.load_frequency_documents(a.wikidata_dir, a.doc_start, a.hidden_samples)
    if len(docs) < 32:
        raise SystemExit(f"only {len(docs)} documents loaded; need a real sample")
    H = base_hidden_matrix(model, tok, docs, device, a.batch_size)
    W = output_layer.weight.detach().float().cpu()
    markers = pick_markers(H, W, a.records)
    print(f"hidden states {tuple(H.shape)}; median sigma over all directions "
          f"{markers['median_sigma_all']:.4f}")

    mcf = json.loads(Path(a.multi_counterfact).read_text(encoding="utf-8"))
    results: List[Dict[str, Any]] = []

    for idx in range(a.records):
        record = mcf[idx]
        rw = record["requested_rewrite"]
        subject = str(rw["subject"])
        prompt = str(rw["prompt"]).format(subject)
        probes = [p for p in (record.get("generation_prompts") or [])
                  if subject in str(p)][:8] or [prompt]
        v = markers["vectors"][idx].to(device)
        sigma = float(markers["sigmas"][idx])
        coupling = float(markers["couplings"][idx])
        target_alpha = float(a.target_k) * max(sigma, 1e-6)

        rows = subj.subject_token_ids(tok, subject, llama_like=llama_like)
        live = subj.live_prompt_token_ids([record], tok).get(int(record["case_id"]), set())
        rows = [r for r in rows if r in live]
        if not rows:
            continue

        # base marker projection and base top-1 on the same-subject probes
        with torch.no_grad():
            enc = tok(probes, padding=True, return_tensors="pt").to(device)
            out = model(**enc, use_cache=False, output_hidden_states=True)
            pos = enc["attention_mask"].sum(dim=1) - 1
            r = torch.arange(len(probes), device=device)
            h_base = out.hidden_states[-1][r, pos, :].float()
            base_top = out.logits[r, pos, :].argmax(dim=-1).cpu()
            base_alpha = float((h_base @ v).mean())

        delta = core.SelectedRowDelta(len(rows), int(input_layer.weight.shape[1]),
                                      direction_basis=None,
                                      device=input_layer.weight.device)
        opt = torch.optim.AdamW(delta.parameters(), lr=a.lr, weight_decay=0.0)
        hook = base1.register_input_embedding_delta_hook(
            input_layer, rows, delta.effective_delta)
        try:
            for _ in range(a.steps):
                opt.zero_grad(set_to_none=True)
                enc = tok(probes, padding=True, return_tensors="pt").to(device)
                out = model(**enc, use_cache=False, output_hidden_states=True)
                pos = enc["attention_mask"].sum(dim=1) - 1
                r = torch.arange(len(probes), device=device)
                h = out.hidden_states[-1][r, pos, :].float()
                alpha = (h @ v).mean()
                loss = F.relu(target_alpha - alpha) + 1e-4 * delta.effective_delta().square().mean()
                if not torch.isfinite(loss):
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(delta.parameters()), 1.0)
                opt.step()
            with torch.no_grad():
                enc = tok(probes, padding=True, return_tensors="pt").to(device)
                out = model(**enc, use_cache=False, output_hidden_states=True)
                pos = enc["attention_mask"].sum(dim=1) - 1
                r = torch.arange(len(probes), device=device)
                h_new = out.hidden_states[-1][r, pos, :].float()
                achieved = float((h_new @ v).mean())
                new_top = out.logits[r, pos, :].argmax(dim=-1).cpu()
                delta_norm = float(delta.effective_delta().detach().norm())
        finally:
            hook.remove()

        k = (achieved - base_alpha) / max(sigma, 1e-6)
        agree = float((base_top == new_top).float().mean())
        results.append({
            "case_id": int(record["case_id"]), "subject": subject,
            "rows_edited": len(rows), "sigma": sigma,
            "marker_lm_head_coupling": coupling,
            "base_alpha": base_alpha, "achieved_alpha": achieved,
            "achieved_k": k, "target_k": float(a.target_k),
            "embedding_delta_norm": delta_norm,
            "same_subject_top1_agreement_after_write": agree,
            "leak_at_delta_10": 10.0 / k if k > 0 else None,
        })
        print(f"[{subject[:28]:30s}] k={k:8.1f} (target {a.target_k})  "
              f"leak@delta10={10.0/k if k>0 else float('nan'):.3f}  "
              f"top1 agreement after write={agree:.2f}  coupling={coupling:.4f}")

    ks = [r["achieved_k"] for r in results if r["achieved_k"] > 0]
    payload = {
        "target_k": float(a.target_k),
        "median_sigma_all_directions": markers["median_sigma_all"],
        "results": results,
        "verdict": (
            "GO if achieved_k is large (leak = delta/k small, e.g. k>=50 gives "
            "<=0.2 logits at delta=10) AND same-subject top-1 agreement stays "
            "high after the write alone. Low k means the embedding cannot steer "
            "h into a chosen low-variance direction through the frozen stack, so "
            "beta must grow and the read end leaks. Low agreement means the "
            "write itself is the collateral, and the conjunction does not help."
        ),
    }
    if ks:
        payload["median_achieved_k"] = sorted(ks)[len(ks) // 2]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nmedian achieved k: {payload.get('median_achieved_k')}")
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
