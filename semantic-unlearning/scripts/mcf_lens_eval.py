#!/usr/bin/env python3
"""Is the forgotten answer still readable INSIDE the edited forward pass? (MCF, one seed)

For each run (any router artifact; a zero-row router = exact base) and each prompt:
greedy-generate 16 tokens with the run's edit, then re-run the full sequence and read
the answer at every generation position t (boundary onwards) from

  logit lens  softmax(lm_head(final_norm(h_l(t))))  at the requested layers
              (the last layer is the real output)
  J-lens      z = (u_w . h_l(t) - mu_w) / sd_w, u_w = mean d logit_w / d h_l over generic
              MCF prompts (forget subjects excluded); z <= 0 = no more answer evidence than
              an unrelated prompt. Layers other than the one a run was TRAINED against are
              held-out lenses (guards against hiding the answer from one readout only).

Prompts: our training views (canonical + development) and the held-out MCF paraphrases.
A fixed-reply guardrail ("router + fixed IDK") never touches the forward pass, so its
internal readouts equal base by construction.

    python -u scripts/mcf_lens_eval.py --runs base=R0 delta_e=R1 J_C=R2 \\
        --out-dir outputs/mcf_lens_eval_seed1 --local-files-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from mcf_heldout_generation import _fact_case, heldout_prompts
from mcf_projection_edit import generic_prompts
from prepare_mcf_association_source import load_mcf_forget_data
from train_mcf_jlens_feedback_rows import _grab, answer_token_sets, jlens


def build_prompts(tok, mcf_path, seed, facts, examples, answers):
    case_to_fact = {_fact_case(f): f["id"] for f in facts}
    items = []
    for e in examples:
        if (e.split == "train" and e.group == "canonical_rewrite") or e.split == "development":
            first = next(k for k, lab in enumerate(e.labels) if lab != -100)
            items.append({"id": e.id, "set": "training_view", "fact_id": e.fact_id,
                          "prompt_ids": list(e.input_ids[:first]), "answer": e.completion.strip()})
    _, held = heldout_prompts(mcf_path, seed)
    for h in held:
        if h["set"] == "paraphrase" and h["case_id"] in case_to_fact:
            items.append({"id": h["id"], "set": "heldout_paraphrase",
                          "fact_id": case_to_fact[h["case_id"]],
                          "prompt_ids": tok(h["prompt"])["input_ids"], "answer": h["answer"]})
    for it in items:
        it["answer_tokens"] = answers[it["fact_id"]]
    return items


@torch.no_grad()
def evaluate_run(model, tok, artifact, items, logit_layers, lenses, args):
    from linear_router import load_router_artifact

    lm, bank = load_router_artifact(model, artifact)
    n_layers = len(model.model.layers)
    layers = sorted(set(logit_layers) | set(lenses))
    store, handles = {}, []
    for layer in layers:
        handles.append(model.model.layers[layer].register_forward_hook(_grab(store, layer)))
    norm, head = model.model.norm, model.get_output_embeddings()
    rows = []
    try:
        for it in items:
            ids = torch.tensor([it["prompt_ids"]], device=args.device)
            out = lm.generate_uncached_fixed_boundary(ids, max_new_tokens=args.max_new_tokens)
            routed = bool(bank.last_active_fact_indices and bank.last_active_fact_indices[0])
            P, n = ids.shape[1], out.shape[1]
            lm.set_association_prefix_lengths([P])
            logits = lm(input_ids=out, use_cache=False).logits[0].float()
            answer_ids = torch.tensor(it["answer_tokens"], device=args.device)
            span = slice(P - 1, n)
            rec = {"id": it["id"], "set": it["set"], "routed": routed,
                   "continuation": tok.decode(out[0, P:], skip_special_tokens=True),
                   "logit_lens": {}, "jlens_max_z": {}}
            rec["leak"] = it["answer"].casefold() in rec["continuation"].casefold()
            for layer in logit_layers:
                if layer == n_layers - 1:
                    lg = logits[span]
                else:
                    lg = head(norm(store[layer][0, span])).float()
                probs = lg.softmax(-1)
                p_ans = probs[:, answer_ids].amax(-1)                                  # [T]
                rank = (lg > lg[:, answer_ids].amax(-1, keepdim=True)).sum(-1)          # [T]
                rec["logit_lens"][str(layer)] = {"p_boundary": float(p_ans[0]),
                                                 "p_max": float(p_ans.max()),
                                                 "best_rank": int(rank.min())}
            for layer, (U, mu, sd, index) in lenses.items():
                cols = torch.tensor([index[int(t)] for t in it["answer_tokens"]], device=args.device)
                h = store[layer][0, span].float()
                z = (h @ U[cols].T - mu[cols]) / sd[cols]
                rec["jlens_max_z"][str(layer)] = float(z.max())
            rows.append(rec)
    finally:
        for h in handles:
            h.remove()
        bank.close()
    return rows


def summarize(rows, logit_layers, lens_layers, topk):
    out = {}
    for name in ("training_view", "heldout_paraphrase"):
        for scope in ("all", "routed"):
            sub = [r for r in rows if r["set"] == name and (scope == "all" or r["routed"])]
            if not sub:
                continue
            n = len(sub)
            out[f"{name}/{scope}"] = {
                "prompts": n,
                "leak": sum(r["leak"] for r in sub) / n,
                "logit_lens_topk_any_position": {
                    str(l): sum(r["logit_lens"][str(l)]["best_rank"] < topk for r in sub) / n
                    for l in logit_layers},
                "logit_lens_p_max_median": {
                    str(l): statistics.median(r["logit_lens"][str(l)]["p_max"] for r in sub)
                    for l in logit_layers},
                "jlens_max_z_median": {
                    str(l): statistics.median(r["jlens_max_z"][str(l)] for r in sub)
                    for l in lens_layers},
            }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="NAME=RUN_DIR (artifact dir)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--logit-layers", type=int, nargs="+", default=[16, 19, 20, 22, 24, 26, 27])
    p.add_argument("--jlens-layers", type=int, nargs="+", default=[20, 22, 24, 26])
    p.add_argument("--jlens-prompts", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--limit", type=int, default=0, help="first N prompts only (smoke tests)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    runs = [spec.split("=", 1) for spec in args.runs]
    first = Path(runs[0][1])
    manifest = json.loads((first / "association_manifest.json").read_text())
    seed = int((manifest.get("sampling") or {}).get("seed", 1))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(manifest["model_path"], use_fast=True,
                                        local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_path"], dtype=torch.float32, local_files_only=args.local_files_only,
        attn_implementation="eager").to(args.device).eval()
    model.requires_grad_(False)
    n_layers = len(model.model.layers)
    logit_layers = [l for l in args.logit_layers if 0 <= l < n_layers]
    lens_layers = [l for l in args.jlens_layers if 0 <= l < n_layers]

    _, facts, examples = load_mcf_forget_data(tok, manifest["mcf_path"], seed=seed)
    answers = answer_token_sets(tok, examples)
    items = build_prompts(tok, manifest["mcf_path"], seed, facts, examples, answers)
    if args.limit:
        items = items[:args.limit]
    tokens = sorted({t for ts in answers.values() for t in ts})
    index = {t: i for i, t in enumerate(tokens)}
    generic = generic_prompts(tok, manifest["mcf_path"], facts, args.jlens_prompts)
    lenses = {}
    for layer in lens_layers:
        cache = out_dir / f"jlens_L{layer}.pt"
        if cache.is_file():
            saved = torch.load(cache, weights_only=False)
        else:
            U, mu, sd = jlens(model, layer, generic, tokens, args.device)
            saved = {"layer": layer, "tokens": tokens, "U": U, "mu": mu, "sd": sd}
            torch.save(saved, cache)
        if saved["tokens"] != tokens:
            raise RuntimeError(f"{cache} was built for other answer tokens")
        lenses[layer] = (saved["U"].to(args.device), saved["mu"].to(args.device),
                         saved["sd"].to(args.device), index)
    print(json.dumps({"prompts": len(items), "logit_layers": logit_layers,
                      "jlens_layers": lens_layers}), flush=True)

    results = {}
    for name, run_dir in runs:
        path = out_dir / f"{name}.json"
        if path.is_file():
            results[name] = json.loads(path.read_text())
            print(f"skip {name} (exists)")
            continue
        t0 = time.perf_counter()
        artifact = torch.load(Path(run_dir) / "fact_association_embeddings.pt", map_location="cpu",
                              weights_only=False)
        rows = evaluate_run(model, tok, artifact, items, logit_layers, lenses, args)
        results[name] = {"run_dir": str(run_dir), "summary": summarize(rows, logit_layers, lens_layers,
                                                                       args.topk), "rows": rows}
        path.write_text(json.dumps(results[name], indent=2) + "\n")
        print(f"{name}: {time.perf_counter() - t0:.0f}s", flush=True)

    lines = [f"# Is the answer still inside the forward pass? MCF seed {seed}\n",
             f"Greedy {args.max_new_tokens}-token continuation, read at every generation position. "
             f"**lens top-{args.topk}**: answer among the top {args.topk} logit-lens tokens at some "
             f"position (layer {n_layers - 1} = actual output). **J-lens z**: median over prompts of "
             "the max answer z-score (0 = unrelated-prompt level). Lower = less answer inside.\n"]
    for key in ("training_view/routed", "heldout_paraphrase/routed", "heldout_paraphrase/all"):
        lines += [f"\n## {key}\n",
                  "| run | n | leak | " + " | ".join(f"lens top-{args.topk} L{l}" for l in logit_layers)
                  + " | " + " | ".join(f"J-lens z L{l}" for l in lens_layers) + " |",
                  "|" + "---|" * (3 + len(logit_layers) + len(lens_layers))]
        for name, res in results.items():
            s = res["summary"].get(key)
            if not s:
                continue
            lines.append("| " + " | ".join(
                [name, str(s["prompts"]), f"{100 * s['leak']:.1f}%"]
                + [f"{100 * s['logit_lens_topk_any_position'][str(l)]:.0f}%" for l in logit_layers]
                + [f"{s['jlens_max_z_median'][str(l)]:.2f}" for l in lens_layers]) + " |")
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
