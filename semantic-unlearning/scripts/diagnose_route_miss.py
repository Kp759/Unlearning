#!/usr/bin/env python3
"""Why does the linear classifier miss a prompt? Per layer, for one fact's prompts.

    python scripts/diagnose_route_miss.py \
        --sweep-dir outputs/mcf_multiseed_regular_v1/seed4 --fact-index 31

Reads the fact's prompts from each layer's decomposition (rows_v2.json =
the artifact's own router, i.e. the linear classifier), reruns them through the
saved bank and prints, per prompt:
  eligible   the subject gate lets fact i's head compete (subject tokens found)
  z_i        fact i's calibrated logit (fires when >= threshold, default 0)
  best       the best subject-eligible head and its logit
  qual/amb   heads over threshold / rejected as ambiguous
A miss with eligible=False is a gate failure; eligible=True with z_i < thr is a
classifier score failure; amb=True is a collision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from linear_router import eligibility_matrix, load_router_artifact


def _rows(path):
    data = json.loads(path.read_text())
    return data.get("rows", data) if isinstance(data, dict) else data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--fact-index", type=int, required=True)
    parser.add_argument("--layers", nargs="+", default=["01", "03", "07", "13", "19", "23", "27"])
    parser.add_argument("--groups", nargs="+", default=["paraphrase"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    sweep = Path(args.sweep_dir)
    base_model = tokenizer = None
    fi = args.fact_index
    for layer in args.layers:
        run = sweep / f"L{layer}" / "linear_global"
        decomp = run / "decomposition" / "rows_v2.json"
        if not decomp.exists():
            print(f"L{layer}: no decomposition, skip")
            continue
        prompts = [r["prompt"] for r in _rows(decomp)
                   if r["fact_index"] == fi and r["group"] in args.groups]
        manifest = json.loads((run / "association_manifest.json").read_text())
        if base_model is None:
            tokenizer = AutoTokenizer.from_pretrained(
                manifest["model_path"], use_fast=True, local_files_only=args.local_files_only)
            base_model = AutoModelForCausalLM.from_pretrained(
                manifest["model_path"], dtype=getattr(torch, args.dtype),
                local_files_only=args.local_files_only, attn_implementation="eager",
            ).to(args.device).eval()
            base_model.requires_grad_(False)
        artifact = torch.load(run / "fact_association_embeddings.pt",
                              map_location="cpu", weights_only=False)
        model, bank = load_router_artifact(base_model, artifact)
        captured = {}
        original = bank.router_logits

        def spy(query, _orig=original):
            logits = _orig(query)
            captured["logits"] = logits.detach().float().cpu()
            return logits

        bank.router_logits = spy
        eligible = eligibility_matrix(tokenizer, prompts, bank.subject_patterns)
        thr = bank.threshold
        try:
            for p_idx, prompt in enumerate(prompts):
                ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(args.device)
                model.set_association_prefix_lengths([ids.shape[1]])
                with torch.no_grad():
                    model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                z = captured["logits"][0]
                s = bank.last_route_scores[0]
                rank = int((z > z[fi]).sum()) + 1
                print(f"L{layer} eligible={bool(eligible[p_idx, fi])!s:5} "
                      f"z_{fi}={float(z[fi]):+.2f} thr={thr:+.2f} rank={rank:>2}/{z.numel()} | "
                      f"best_eligible={s['best_eligible_fact']} ({s['best_eligible_logit']}) "
                      f"fired={s['fact_index']} qual={s['qualifying_candidates']} "
                      f"amb={s['rejected_as_ambiguous']} | {prompt[-45:]!r}")
                n_elig = int(eligible[p_idx].sum())
                if n_elig != int(eligible[p_idx, fi]):
                    others = [i for i in range(eligible.shape[1]) if eligible[p_idx, i] and i != fi]
                    print(f"      other eligible facts: {others}")
        finally:
            bank._hook_handle.remove()
    if base_model is None:
        print("nothing to diagnose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
