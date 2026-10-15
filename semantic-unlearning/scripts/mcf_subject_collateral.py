#!/usr/bin/env python3
"""Collateral damage to the SAME subject's other facts (MCF, one seed).

For each forget fact (e.g. Karl Krumbacher -> worked in Munich) we ask about the same
subject with OTHER relations, using MCF's own templates for relations that fit the
subject's type ("Karl Krumbacher, a citizen of", "Karl Krumbacher works as", ...).

Two conditions per run (any router artifact; early-write or layer-19 bank):
  forced   the fact's own edit is switched on for these prompts, as it would be inside a
           reply where the router fired -> does erasing/blocking "work location" also
           change what the model says about nationality, job, ...?
  natural  the router decides (it should NOT fire on other relations) -> fire rate, and
           whether the output equals base

Metrics vs the base model's greedy continuation (12 tokens):
  same_first5   first 5 generated tokens identical to base
  same_answer   base's first content word still appears in the edited continuation
  dNLL          mean NLL of base's continuation under the edited model minus under base
                (teacher forced; 0 = no change, larger = the base answer got less likely)

Relations are split (deterministically) into a PRESERVE half, usable as a training
signal, and an EVAL half that training never sees.

    python -u scripts/mcf_subject_collateral.py --runs E_L16_logit=R1 Q_L16_logit=R2 \\
        --out-dir outputs/mcf_subject_collateral_seed1 --local-files-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics

import torch
import torch.nn.functional as F

# Coarse subject types by the relation MCF asks about; other relations of the same type
# are questions that make sense for that subject.
TYPE_RELATIONS = {
    "person": ["P27", "P19", "P106", "P20", "P1412", "P101", "P108", "P140", "P937", "P103", "P39"],
    "place": ["P30", "P17", "P37", "P36", "P131", "P463", "P190"],
    "work": ["P495", "P176", "P449", "P364", "P178", "P407", "P127"],
    "org": ["P159", "P740", "P17", "P127", "P463"],
}
RELATION_TYPE = {
    **{r: "person" for r in ("P27", "P19", "P106", "P20", "P1412", "P101", "P108", "P140",
                             "P937", "P103", "P39", "P413", "P1303", "P641", "P136")},
    **{r: "place" for r in ("P30", "P17", "P37", "P36", "P131", "P463", "P190", "P276", "P138", "P47")},
    **{r: "work" for r in ("P495", "P176", "P449", "P364", "P178", "P407", "P264")},
    **{r: "org" for r in ("P159", "P740", "P127")},
}


def relation_templates(mcf_path):
    """relation_id -> its most common MCF prompt template."""
    from collections import Counter, defaultdict

    counts = defaultdict(Counter)
    for record in json.loads(Path(mcf_path).read_text()):
        rw = record["requested_rewrite"]
        rw = rw[0] if isinstance(rw, list) else rw
        counts[rw["relation_id"]][rw["prompt"]] += 1
    return {r: c.most_common(1)[0][0] for r, c in counts.items()}


def subject_probes(forget_records, templates, split):
    """Per forget record (same order): [(relation_id, prompt)] for the given split."""
    if split not in ("preserve", "eval", "all"):
        raise ValueError(split)
    out = []
    for record in forget_records:
        rw = record["requested_rewrite"]
        rw = rw[0] if isinstance(rw, list) else rw
        kind = RELATION_TYPE.get(rw["relation_id"], "person")
        others = sorted(r for r in TYPE_RELATIONS[kind] if r != rw["relation_id"] and r in templates)
        if split == "preserve":
            others = others[0::2]
        elif split == "eval":
            others = others[1::2]
        out.append([(r, templates[r].format(rw["subject"])) for r in others])
    return out


def first_content_word(text):
    for w in re.findall(r"[A-Za-z][A-Za-z\-']+", text):
        if w.casefold() not in {"the", "a", "an", "of", "in", "is", "was", "and", "to", "as", "he",
                                "she", "it", "they", "his", "her", "its", "at", "on", "by"}:
            return w
    return None


@torch.no_grad()
def greedy(model, ids, n, eos):
    for _ in range(n):
        nxt = model(input_ids=ids, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], 1)
        if int(nxt) in eos:
            break
    return ids


@torch.no_grad()
def cont_nll(model, prompt_len, full):
    if hasattr(model, "set_association_prefix_lengths"):
        model.set_association_prefix_lengths([prompt_len])
    logits = model(input_ids=full, use_cache=False).logits[0].float()
    return float(F.cross_entropy(logits[prompt_len - 1:-1], full[0, prompt_len:]))


def evaluate(model, tok, artifact, items, args):
    from linear_router import load_router_artifact
    from zerounlearn_generation_leak import eos_id_set

    lm, bank = load_router_artifact(model, artifact)
    eos = eos_id_set(model, tok)
    rows = []
    try:
        for it in items:
            ids = torch.tensor([it["prompt_ids"]], device=args.device)
            P = ids.shape[1]
            base_full = torch.tensor([it["base_ids"]], device=args.device)
            rec = {"fact": it["fact"], "relation": it["relation"], "prompt": it["prompt"],
                   "base": it["base_text"]}
            for cond in ("natural", "forced"):
                if cond == "forced":
                    if hasattr(lm, "force_route"):
                        lm.force_route(it["prompt_ids"], it["fact_index"])
                    else:
                        bank.set_oracle_routes({tuple(it["prompt_ids"]): it["fact_index"]})
                out = lm.generate_uncached_fixed_boundary(ids, max_new_tokens=args.max_new_tokens)
                fired = bool(bank.last_active_fact_indices and bank.last_active_fact_indices[0])
                text = tok.decode(out[0, P:], skip_special_tokens=True)
                gen = out[0, P:].tolist()
                base_gen = it["base_ids"][P:]
                word = first_content_word(it["base_text"])
                rec[cond] = {
                    "text": text, "fired": fired,
                    "same_first5": gen[:5] == base_gen[:5],
                    "same_answer": (word is None) or (word.casefold() in text.casefold()),
                    "dNLL": cont_nll(lm, P, base_full) - it["base_nll"],
                }
                if cond == "forced":
                    if hasattr(lm, "force_route"):
                        lm.route_cache.pop(tuple(it["prompt_ids"]), None)
                    else:
                        bank.set_oracle_routes(None)
            rows.append(rec)
    finally:
        bank.close()
    return rows


def summarize(rows):
    out = {}
    for cond in ("natural", "forced"):
        n = max(1, len(rows))
        out[cond] = {
            "prompts": len(rows),
            "router_fired": sum(r[cond]["fired"] for r in rows) / n,
            "same_first5": sum(r[cond]["same_first5"] for r in rows) / n,
            "same_answer": sum(r[cond]["same_answer"] for r in rows) / n,
            "dNLL_mean": statistics.mean(r[cond]["dNLL"] for r in rows) if rows else 0.0,
        }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="NAME=RUN_DIR")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="eval", choices=("eval", "preserve", "all"))
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--show", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prepare_mcf_association_source import load_mcf_forget_data
    from zerounlearn_generation_leak import eos_id_set

    runs = [s.split("=", 1) for s in args.runs]
    manifest = json.loads((Path(runs[0][1]) / "association_manifest.json").read_text())
    seed = int((manifest.get("sampling") or {}).get("seed", 1))
    tok = AutoTokenizer.from_pretrained(manifest["model_path"], use_fast=True,
                                        local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_path"], dtype=torch.bfloat16, local_files_only=args.local_files_only,
        attn_implementation="eager").to(args.device).eval()
    model.requires_grad_(False)
    forget_records, facts, _ = load_mcf_forget_data(tok, manifest["mcf_path"], seed=seed)
    probes = subject_probes(forget_records, relation_templates(manifest["mcf_path"]), args.split)

    eos = eos_id_set(model, tok)
    items = []
    with torch.no_grad():
        for i, (fact, plist) in enumerate(zip(facts, probes)):
            for rel, prompt in plist:
                ids = tok(prompt)["input_ids"]
                t = torch.tensor([ids], device=args.device)
                full = greedy(model, t, args.max_new_tokens, eos)
                items.append({"fact": fact["id"], "fact_index": i, "relation": rel, "prompt": prompt,
                              "prompt_ids": ids, "base_ids": full[0].tolist(),
                              "base_text": tok.decode(full[0, len(ids):], skip_special_tokens=True),
                              "base_nll": cont_nll(model, len(ids), full) if full.shape[1] > len(ids) else 0.0})
    items = [it for it in items if len(it["base_ids"]) > len(it["prompt_ids"])]
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} same-subject other-relation prompts ({args.split} relations)", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, run_dir in runs:
        path = out_dir / f"{name}.json"
        if path.is_file():
            results[name] = json.loads(path.read_text())
            continue
        artifact = torch.load(Path(run_dir) / "fact_association_embeddings.pt", map_location="cpu",
                              weights_only=False)
        rows = evaluate(model, tok, artifact, items, args)
        results[name] = {"run_dir": str(run_dir), "summary": summarize(rows), "rows": rows}
        path.write_text(json.dumps(results[name], indent=2) + "\n")
        print(name, json.dumps(results[name]["summary"]), flush=True)

    lines = [f"# Same subject, other relations: collateral damage (MCF seed {seed}, {args.split} relations)\n",
             "**forced** = the fact's edit switched on (as inside a routed reply); **natural** = router "
             "decides. vs base greedy continuation: same first 5 tokens ↑, base answer word kept ↑, "
             "ΔNLL of base continuation ↓ (0 = unchanged).\n",
             "| run | n | forced: same first 5 | forced: answer kept | forced: ΔNLL | natural: router fired ↓ "
             "| natural: same first 5 |", "|---|---|---|---|---|---|---|"]
    for name, res in results.items():
        f, n = res["summary"]["forced"], res["summary"]["natural"]
        lines.append(f"| {name} | {f['prompts']} | {100 * f['same_first5']:.0f}% | "
                     f"{100 * f['same_answer']:.0f}% | {f['dNLL_mean']:.2f} | "
                     f"{100 * n['router_fired']:.1f}% | {100 * n['same_first5']:.0f}% |")
    first = next(iter(results.values()))["rows"]
    lines.append("\n## Examples (forced edit)\n")
    for k, row in enumerate(first[:args.show]):
        lines.append(f"**{row['prompt']}**\n\n- base: `{row['base'].strip()}`")
        for name, res in results.items():
            lines.append(f"- {name}: `{res['rows'][k]['forced']['text'].strip()}`")
        lines.append("")
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
