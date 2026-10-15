#!/usr/bin/env python3
"""Where is each forget fact recalled? (base model, MCF seed 1, subject tokens)

Picks the write layer for early-write rows (scripts/early_write.py). On the canonical +
development forget prompts, at the subject's tokens:

  readable   logit lens at the LAST subject token, every layer: is the answer among the
             top-10 tokens? (when the fact has been written into the subject's state)
  recall     zero the MLP output at ALL subject tokens for a window of W layers starting
             at l, and measure the base P(first answer token) at the prompt boundary,
             relative to the unablated model (ROME-style localisation: where knocking out
             the subject's MLPs removes the answer). Only prompts the model answers with
             P >= --min-p count.

Write the rows just AFTER the main recall window and before the answer becomes readable
downstream: the row can then cancel what was recalled before later positions copy it.

    python -u scripts/mcf_recall_site.py --router-dir R --out-dir outputs/mcf_recall_site_seed1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from early_write import EarlyWriteCausalLM
from prepare_mcf_association_source import load_mcf_forget_data
from train_mcf_jlens_feedback_rows import answer_token_sets


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--router-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--min-p", type=float, default=0.05)
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    router_dir = Path(args.router_dir)
    artifact = torch.load(router_dir / "fact_association_embeddings.pt", map_location="cpu",
                          weights_only=False)
    manifest = json.loads((router_dir / "association_manifest.json").read_text())
    seed = int((manifest.get("sampling") or {}).get("seed", 1))
    tok = AutoTokenizer.from_pretrained(manifest["model_path"], use_fast=True,
                                        local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_path"], dtype=torch.float32, local_files_only=args.local_files_only,
        attn_implementation="eager").to(args.device).eval()
    model.requires_grad_(False)
    n_layers = len(model.model.layers)
    _, facts, examples = load_mcf_forget_data(tok, manifest["mcf_path"], seed=seed)
    answers = answer_token_sets(tok, examples)
    fact_index = {f["id"]: i for i, f in enumerate(artifact["facts"])}
    finder = EarlyWriteCausalLM.__new__(EarlyWriteCausalLM)          # reuse the subject search
    finder.subject_patterns, finder.write_mode = artifact["subject_patterns"], "subject_span"

    views = []
    for e in examples:
        if not ((e.split == "train" and e.group == "canonical_rewrite") or e.split == "development"):
            continue
        first = next(k for k, lab in enumerate(e.labels) if lab != -100)
        ids = list(e.input_ids[:first])
        span = finder._subject_positions(list(enumerate(ids)), fact_index[e.fact_id])
        if span:
            views.append({"id": e.id, "ids": ids, "span": span, "answer": answers[e.fact_id]})

    norm, head = model.model.norm, model.get_output_embeddings()
    store, ablate = {}, {"layers": set(), "span": None}

    def grab(layer):
        def hook(_m, _a, out):
            store[layer] = (out[0] if isinstance(out, tuple) else out)
        return hook

    def zero_mlp(layer):
        def hook(_m, _a, out):
            if layer in ablate["layers"]:
                out = out.clone()
                out[:, ablate["span"]] = 0
            return out
        return hook

    handles = [model.model.layers[l].register_forward_hook(grab(l)) for l in range(n_layers)]
    handles += [model.model.layers[l].mlp.register_forward_hook(zero_mlp(l)) for l in range(n_layers)]
    rows = []
    try:
        with torch.no_grad():
            for v in views:
                ids = torch.tensor([v["ids"]], device=args.device)
                answer = torch.tensor(v["answer"], device=args.device)
                ablate["layers"], ablate["span"] = set(), v["span"]
                p0 = float(model(input_ids=ids).logits[0, -1].float().softmax(-1)[answer].max())
                last = v["span"][-1]
                readable = []
                for l in range(n_layers):
                    lg = head(norm(store[l][0, last])).float()
                    readable.append(int((lg > lg[answer].max()).sum()) < 10)
                recall = []
                if p0 >= args.min_p:
                    for l in range(n_layers):
                        ablate["layers"] = set(range(l, min(n_layers, l + args.window)))
                        pa = float(model(input_ids=ids).logits[0, -1].float().softmax(-1)[answer].max())
                        recall.append(pa / p0)
                    ablate["layers"] = set()
                rows.append({"id": v["id"], "p_base": p0, "readable_top10": readable,
                             "ablated_over_base": recall})
    finally:
        for h in handles:
            h.remove()

    known = [r for r in rows if r["ablated_over_base"]]
    table = []
    for l in range(n_layers):
        table.append({
            "layer": l,
            "readable_top10_at_last_subject_token": sum(r["readable_top10"][l] for r in rows) / len(rows),
            "p_after_mlp_ablation_over_base_median": (
                statistics.median(r["ablated_over_base"][l] for r in known) if known else None),
        })
    drops = [(t["p_after_mlp_ablation_over_base_median"], t["layer"]) for t in table
             if t["p_after_mlp_ablation_over_base_median"] is not None]
    best = min(drops)[1] if drops else None
    summary = {"prompts": len(rows), "known_prompts": len(known), "window": args.window,
               "strongest_recall_window": [best, best + args.window - 1] if best is not None else None,
               "suggested_write_layer": (min(best + args.window - 1, int(artifact["layer"]))
                                         if best is not None else None)}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "recall_site.json").write_text(json.dumps({"summary": summary, "layers": table,
                                                      "rows": rows}, indent=2) + "\n")
    lines = [f"# Recall site of the MCF seed-{seed} forget facts (base model)\n",
             f"{len(rows)} prompts with the subject found; {len(known)} answered with "
             f"P >= {args.min_p}. MLP window = {args.window} layers starting at that layer.\n",
             "| layer | answer readable at last subject token (top-10) | P(answer) after zeroing subject MLPs, ÷ base (median) |",
             "|---|---|---|"]
    for t in table:
        r = t["p_after_mlp_ablation_over_base_median"]
        lines.append(f"| {t['layer']} | {100 * t['readable_top10_at_last_subject_token']:.0f}% | "
                     f"{'–' if r is None else f'{r:.3f}'} |")
    lines.append(f"\nStrongest recall window: layers {summary['strongest_recall_window']}; "
                 f"suggested write layer: {summary['suggested_write_layer']}.")
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
