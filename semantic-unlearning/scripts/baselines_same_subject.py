#!/usr/bin/env python3
"""Relation specificity of every method: does the edited model still answer the forgotten
subject's OTHER questions as the base model does? (Analysis only; MCF seed 1.)

Prompts: the no-association set's "protected subject, other relation" group
(outputs/router_diag_ours_control_v1/mcf/seed1/probes_null.json), e.g. "Belgium is located in"
when the forgotten fact is Belgium -> NATO. For the base model and each edited model:
  changed      the greedy continuation (24 tokens) differs from the base model's
  top1_changed the base model's first answer token is no longer the greedy next token
Baselines are re-applied with the recipe their official evaluation recorded (paper_baselines.apply,
same seed, same views, same hparams), so each edited model is the evaluated one. Ours is the
paper's router + edit vectors (outputs/l16_ours_v1), the deployed decision (fires or not).

    python -u scripts/baselines_same_subject.py --dataset mcf --seed 1
    # -> outputs/relation_specificity_v1/mcf_seed1.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_baselines
from linear_router import load_router_artifact


@torch.no_grad()
def first_token_logprob(lm, ids_list, first_tokens):
    """log p(base model's first answer token | prompt) under lm, one prompt at a time."""
    device = next(lm.parameters()).device
    out = []
    for ids, t in zip(ids_list, first_tokens):
        if t < 0:
            out.append(float("nan"))
            continue
        x = torch.tensor([ids], device=device)
        lp = lm(input_ids=x, use_cache=False).logits[0, -1].float().log_softmax(-1)
        out.append(float(lp[t]))
    return out


@torch.no_grad()
def greedy(lm, tok, ids_list, max_new=24, batch=16):
    """Greedy continuations and the first generated token, one prompt at a time (routers fix
    the prompt boundary per sequence)."""
    outs, firsts = [], []
    device = next(lm.parameters()).device
    eos = lm.config.eos_token_id if hasattr(lm, "config") else None
    eos = set(eos) if isinstance(eos, (list, tuple)) else {eos}
    for ids in ids_list:
        x = torch.tensor([ids], device=device)
        if hasattr(lm, "generate_uncached_fixed_boundary"):
            out = lm.generate_uncached_fixed_boundary(x, torch.ones_like(x), max_new_tokens=max_new)
        else:
            out = x
            for _ in range(max_new):
                nxt = lm(input_ids=out, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
                out = torch.cat([out, nxt], 1)
                if int(nxt) in eos:
                    break
        gen = out[0, len(ids):].tolist()
        outs.append(tok.decode(gen, skip_special_tokens=True))
        firsts.append(gen[0] if gen else -1)
    return outs, firsts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="mcf")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--root", default="outputs/router_diag_ours_control_v1")
    ap.add_argument("--run-root", default="outputs/l16_ours_v1")
    ap.add_argument("--baselines", default="ga,gd,npo,rmu,zerounlearn,rome,memit")
    ap.add_argument("--max-prompts", type=int, default=0)
    ap.add_argument("--groups", default="null_protected_other_relation",
                    help="no-association groups to test, comma-separated (all four for the full set)")
    ap.add_argument("--tag", default="", help="output name suffix")
    ap.add_argument("--out", default="outputs/relation_specificity_v1")
    args, rest = ap.parse_known_args(argv)
    cache = Path(args.root) / args.dataset / f"seed{args.seed}"
    man = json.loads((cache / "manifest.json").read_text())
    probes = json.loads((cache / "probes_null.json").read_text())
    groups = args.groups.split(",")
    same = [p for p in probes if p["group"] in groups]
    if args.max_prompts:
        same = same[:args.max_prompts]
    group_of = [p["group"] for p in same]
    run = Path(args.run_root) / args.dataset / f"seed{args.seed}"
    mcf_path = "data/multi_counterfact.json"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(man["model_path"], local_files_only=True)
    if tok.pad_token is None:                       # the vendored editing baselines pad batches
        tok.pad_token = tok.eos_token
    ids = [p["input_ids"] if p.get("input_ids") is not None else tok(p["prompt"])["input_ids"] for p in same]

    def fresh():
        m = AutoModelForCausalLM.from_pretrained(man["model_path"], dtype=torch.bfloat16, local_files_only=True,
                                                 attn_implementation="eager").cuda().eval()
        m.requires_grad_(False)
        return m

    base = fresh()
    base_out, base_first = greedy(base, tok, ids)
    base_lp = first_token_logprob(base, ids, base_first)
    results = {"prompts": len(same)}
    rows = {"base": base_out}
    prev_path = Path(args.out) / f"{args.dataset}_seed{args.seed}{args.tag}.partial.json"

    def record(name, outs, firsts, lps=None):
        changed = [a.strip() != b.strip() for a, b in zip(outs, base_out)]
        top1 = [a != b for a, b in zip(firsts, base_first)]
        # answer probability kept: mean over prompts of p_edited / p_base for the base model's first answer token
        import math
        ratio = [math.exp(a - b) for a, b in zip(lps or base_lp, base_lp) if not (math.isnan(a) or math.isnan(b))]
        kept = sum(min(1.0, r) for r in ratio) / max(len(ratio), 1)
        results[name] = {"changed": sum(changed), "top1_changed": sum(top1), "of": len(same), "answer_prob_kept": kept,
                         "by_group": {g: {"changed": sum(c for c, gg in zip(changed, group_of) if gg == g),
                                          "top1_changed": sum(t for t, gg in zip(top1, group_of) if gg == g),
                                          "of": sum(1 for gg in group_of if gg == g)} for g in groups}}
        rows[name] = outs
        print(json.dumps({name: results[name]}), flush=True)
        prev_path.parent.mkdir(parents=True, exist_ok=True)
        prev_path.write_text(json.dumps({"results": results, "outputs": rows}, indent=1) + "\n")

    if prev_path.is_file():                        # resume: keep finished methods
        prev = json.loads(prev_path.read_text())
        results.update({k: v for k, v in prev["results"].items() if k != "prompts"})
        rows.update(prev["outputs"])
    # ours: the paper's deployed router + edit vectors
    if "ours" not in results:
        art = torch.load(run / "ours" / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
        lm, bank = load_router_artifact(base, art)
        if hasattr(bank, "tokenizer"):
            bank.tokenizer = tok
        o, f = greedy(lm, tok, ids)
        record("ours", o, f, first_token_logprob(lm, ids, base_first))
        if hasattr(bank, "close"):
            bank.close()
        del lm
    del base
    torch.cuda.empty_cache()

    # baselines: re-apply each recipe to a fresh base model
    bp = argparse.ArgumentParser()
    paper_baselines.add_args(bp)
    base_art = torch.load(run / "base" / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    for b in args.baselines.split(","):
        if b in results:
            continue
        rec = json.loads((run / b / f"official_{args.dataset}_eval.json").read_text())["baseline"]
        flags = ["--baseline", b, "--baseline-views-from", str(run / "ours")] + rest
        if rec.get("mom2_dataset"):
            flags += ["--mom2-dataset", rec["mom2_dataset"]]
            if rec.get("mom2_text"):
                flags += ["--mom2-text", rec["mom2_text"]]
        bargs = bp.parse_args(flags)
        model = fresh()
        paper_baselines.apply(bargs, model, tok, base_art, run / "base", args.seed, mcf_path=mcf_path)
        o, f = greedy(model, tok, ids)
        record(b, o, f, first_token_logprob(model, ids, base_first))
        del model
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.dataset}_seed{args.seed}{args.tag}.json").write_text(json.dumps(
        {"results": results, "prompts": [p.get("prompt") for p in same], "outputs": rows}, indent=1) + "\n")
    short = {"null_generic": "generic text", "null_ppl_prefix": "perplexity text", "null_unseen_subject": "unseen subject",
             "null_protected_other_relation": "protected subject, other relation"}
    lines = [f"### {args.dataset} seed {args.seed}: no-association prompts, answer changed vs the base model "
             f"(first answer token changed / 24-token continuation changed)", "",
             "| method | " + " | ".join(short.get(g, g) for g in groups) + " | base answer prob. kept (all prompts) |",
             "|---" * (len(groups) + 2) + "|"]
    for name, r in results.items():
        if name == "prompts":
            continue
        cells = []
        for g in groups:
            bg = (r.get("by_group") or {}).get(g)
            cells.append("–" if bg is None else f"{bg['top1_changed']} / {bg['changed']} of {bg['of']}")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {r.get('answer_prob_kept', float('nan')):.2f} |")
    (out / f"{args.dataset}_seed{args.seed}{args.tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
