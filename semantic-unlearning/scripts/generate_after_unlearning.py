#!/usr/bin/env python3
"""What does the model actually generate after unlearning? MCF and ZsRE.

Greedy generations for the official evaluation prompts, from the base model and
from one or more unlearned banks of the same dataset and seed, side by side:

    python -u scripts/generate_after_unlearning.py --local-files-only --run-dirs \
        outputs/mcf_multiseed_regular_v1/seed1/L19/linear_global \
        outputs/compressed_multiseed_v1/mcf/seed1/L19/tied_answer \
        --labels shipped tied_answer

Prompt groups (the official samples for the run's seed):
  rewrite, paraphrase   forget prompts: the true answer should be GONE
  neighborhood          other subjects (MCF: same answer; ZsRE: unrelated NQ
                        questions): output should be UNCHANGED
  retain                retained facts: the answer should STAY
Generation uses the bank's own contract: uncached greedy decoding with the edit
bound to the original request boundary (AssociationCausalLM.
generate_uncached_fixed_boundary). The base model is decoded the same way.

Per prompt it records the continuation, whether the true answer appears
(case-insensitive substring), and which bank row fired (and that row's answer).
Writes <out>.jsonl (every prompt) and <out>.md (summary + examples).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

import torch

from linear_router import load_router_artifact

FORGET = ("rewrite", "paraphrase")


def norm(text):
    return " ".join(str(text).casefold().split())


def contains_answer(text, answer):
    a = norm(answer)
    return bool(a) and a in norm(text)


def first_line(text, limit=120):
    line = str(text).strip().split("\n")[0].strip()
    return line if len(line) <= limit else line[: limit - 1] + "…"


def manifest_seed(manifest):
    for v in (manifest.get("seed"), (manifest.get("sampling") or {}).get("seed"),
              (manifest.get("plan") or {}).get("seed")):
        if v is not None:
            return int(v)
    return 1


def dataset_of(manifest, artifact, run_dir):
    for v in (artifact.get("dataset"), manifest.get("dataset"), str(run_dir)):
        v = str(v or "").lower()
        if "zsre" in v:
            return "zsre"
        if "mcf" in v or "counterfact" in v:
            return "mcf"
    raise ValueError(f"cannot tell whether {run_dir} is MCF or ZsRE")


# ---------------------------------------------------------------------------
# Prompts: dicts with group, prompt, answer, fact_id (forget) or None
# ---------------------------------------------------------------------------

def prompts_mcf(args, seed):
    from evaluate_router_decomposition import build_records_from_mcf

    _, rows = build_records_from_mcf(args.mcf_path, 50, 1000, seed)
    out, per_fact = [], Counter()
    for r in rows:
        g = r["group"]
        if g == "neighborhood":
            if per_fact[r["fact_id"]] >= args.neighborhood_per_fact:
                continue
            per_fact[r["fact_id"]] += 1
        out.append({"group": g, "prompt": r["prompt"], "answer": r["answer"],
                    "fact_id": r["fact_id"] if g in FORGET else None,
                    "about_fact": r["fact_id"] if g == "neighborhood" else None})
    return out


def prompts_zsre(args, seed, tok):
    import zsre_zero_unlearn_official_eval as zsre

    forget, retain = zsre.load_official_eval_records(
        Path(args.zsre_path), tok, forget_num=50, retain_num=1000, seed=seed)
    out = []
    for rec in forget:
        rr = rec["requested_rewrite"]
        fid = f"zsre_forget_{int(rec['case_id'])}"
        ans = rr["target_true"]["str"]
        out.append({"group": "rewrite", "prompt": str(rr["prompt"]).format(rr["subject"]),
                    "answer": ans, "fact_id": fid, "about_fact": None})
        out += [{"group": "paraphrase", "prompt": str(p), "answer": ans, "fact_id": fid,
                 "about_fact": None} for p in rec["paraphrase_prompts"]]
        nb = rec.get("neighborhood_prompts") or []
        if nb and args.neighborhood_per_fact > 0:
            out.append({"group": "neighborhood", "prompt": nb[0]["prompt"],
                        "answer": "".join(t["target"] for t in nb).strip(),
                        "fact_id": None, "about_fact": fid})
    for rec in retain:
        rr = rec["requested_rewrite"]
        out.append({"group": "retain", "prompt": str(rr["prompt"]).format(rr["subject"]),
                    "answer": rr["target_true"]["str"], "fact_id": None, "about_fact": None})
    return out


def limit_retain(prompts, n, seed):
    keep = [p for p in prompts if p["group"] != "retain"]
    retain = [p for p in prompts if p["group"] == "retain"]
    if n and len(retain) > n:
        retain = random.Random(seed).sample(retain, n)
    return keep + retain


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_base(model, ids, max_new_tokens, eos):
    """Uncached greedy decoding of the plain model (same loop as the bank's)."""
    for _ in range(max_new_tokens):
        nxt = model(input_ids=ids, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if eos is not None and int(nxt) == int(eos):
            break
    return ids


def encode(tok, prompt, device):
    return tok(prompt, return_tensors="pt")["input_ids"].to(device)


def decode_new(tok, ids, prompt_len):
    return tok.decode(ids[0, prompt_len:], skip_special_tokens=True)


def summarize(rows, labels):
    """Per run and group: answer kept / removed, outputs changed vs base."""
    table = {}
    for label in labels:
        by = defaultdict(Counter)
        for r in rows:
            g, base_has = r["group"], r["base_has_answer"]
            run = r["runs"][label]
            c = by[g]
            c["prompts"] += 1
            c["base_has_answer"] += base_has
            c["unlearned_has_answer"] += run["has_answer"]
            c["removed"] += base_has and not run["has_answer"]
            c["output_changed"] += run["output"].strip() != r["base_output"].strip()
            c["row_fired"] += run["routed_row"] is not None
            if g in FORGET:
                c["fired_own_row"] += run["routed_fact_id"] == r["fact_id"]
        table[label] = {g: dict(c) for g, c in by.items()}
    return table


def write_markdown(path, meta, summary, rows, labels, examples):
    L = [f"# Generations after unlearning ({meta['dataset'].upper()}, seed {meta['seed']})", "",
         f"Model: `{meta['model_path']}`, greedy, {meta['max_new_tokens']} new tokens, "
         "edit bound to the request boundary.", "",
         "Answer present = the true answer appears in the continuation (case-insensitive).", ""]
    for label in labels:
        L += [f"## {label}", f"`{meta['runs'][label]}`", "",
              "| group | prompts | answer in base | answer after | removed | output changed | row fired | fired own row |",
              "|---|---|---|---|---|---|---|---|"]
        for g in ("rewrite", "paraphrase", "neighborhood", "retain"):
            c = summary[label].get(g)
            if not c:
                continue
            own = c.get("fired_own_row", "–")
            L.append(f"| {g} | {c['prompts']} | {c['base_has_answer']} | {c['unlearned_has_answer']} | "
                     f"{c['removed']} | {c['output_changed']} | {c['row_fired']} | {own} |")
        L.append("")
    L += ["## Examples", ""]
    shown = Counter()
    for r in rows:
        if shown[r["group"]] >= examples:
            continue
        shown[r["group"]] += 1
        L += [f"**[{r['group']}]** {r['prompt']}  ", f"*true answer:* {r['answer']}  ",
              f"- base: {first_line(r['base_output'])}" + (" ✅" if r["base_has_answer"] else "")]
        for label in labels:
            run = r["runs"][label]
            fired = (f" _(row {run['routed_row']} → {run['routed_answer']})_"
                     if run["routed_row"] is not None else " _(no row)_")
            L.append(f"- {label}: {first_line(run['output'])}"
                     + (" ⚠️ answer" if run["has_answer"] else "") + fired)
        L.append("")
    Path(path).write_text("\n".join(L) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dirs", nargs="+", required=True,
                   help="unlearned run dirs (same dataset and seed)")
    p.add_argument("--labels", nargs="+", default=None, help="one label per run dir")
    p.add_argument("--mcf-path", default="data/multi_counterfact.json")
    p.add_argument("--zsre-path", default="data/zsre_mend_eval.json")
    p.add_argument("--groups", nargs="+", default=["rewrite", "paraphrase", "neighborhood", "retain"])
    p.add_argument("--neighborhood-per-fact", type=int, default=1)
    p.add_argument("--retain", type=int, default=50, help="retain prompts sampled (0 = all 1000)")
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--examples", type=int, default=10, help="examples per group in the .md")
    p.add_argument("--out", default=None, help="output prefix (default: <first run>/generations)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--local-files-only", action="store_true")
    a = p.parse_args(argv)

    runs = [Path(d).resolve() for d in a.run_dirs]
    labels = a.labels or [d.name if d.name != "linear_global" else "shipped" for d in runs]
    if len(labels) != len(runs) or len(set(labels)) != len(labels):
        raise SystemExit("--labels must give one distinct label per run dir")
    manifests = [json.loads((d / "association_manifest.json").read_text()) for d in runs]
    artifacts = [torch.load(d / "fact_association_embeddings.pt", map_location="cpu",
                            weights_only=False) for d in runs]
    datasets = {dataset_of(m, art, d) for m, art, d in zip(manifests, artifacts, runs)}
    seeds = {manifest_seed(m) for m in manifests}
    models = {m["model_path"] for m in manifests}
    if len(datasets) != 1 or len(seeds) != 1 or len(models) != 1:
        raise SystemExit(f"run dirs must share dataset/seed/model: {datasets} {seeds} {models}")
    ds, seed, model_path = datasets.pop(), seeds.pop(), models.pop()
    fact_ids = [[str(f["id"]) for f in art["facts"]] for art in artifacts]
    if any(f != fact_ids[0] for f in fact_ids):
        raise SystemExit("run dirs forget different fact sets")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=a.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    prompts = prompts_mcf(a, seed) if ds == "mcf" else prompts_zsre(a, seed, tok)
    prompts = [q for q in limit_retain(prompts, a.retain, seed) if q["group"] in a.groups]
    known = set(fact_ids[0])
    missing = {q["fact_id"] for q in prompts if q["fact_id"] and q["fact_id"] not in known}
    if missing:
        raise SystemExit(f"forget prompts reference facts not in the bank: {sorted(missing)[:3]}")
    print(f"{ds} seed {seed}: {len(prompts)} prompts "
          f"({dict(Counter(q['group'] for q in prompts))}) x {len(runs)} runs", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=getattr(torch, a.dtype), local_files_only=a.local_files_only,
        attn_implementation="eager").to(a.device).eval()
    base.requires_grad_(False)
    eos = tok.eos_token_id

    # Base model first, before any hook is attached.
    rows = []
    for k, q in enumerate(prompts):
        ids = encode(tok, q["prompt"], a.device)
        text = decode_new(tok, greedy_base(base, ids, a.max_new_tokens, eos), ids.shape[1])
        rows.append({**q, "base_output": text, "base_has_answer": contains_answer(text, q["answer"]),
                     "runs": {}})
        if (k + 1) % 100 == 0:
            print(f"  base {k + 1}/{len(prompts)}", flush=True)

    for label, run, art in zip(labels, runs, artifacts):
        wrapped, bank = load_router_artifact(base, art)
        facts = list(getattr(bank, "facts", art["facts"]))
        try:
            for k, r in enumerate(rows):
                ids = encode(tok, r["prompt"], a.device)
                out = wrapped.generate_uncached_fixed_boundary(
                    ids, torch.ones_like(ids), max_new_tokens=a.max_new_tokens, eos_token_id=eos)
                text = decode_new(tok, out, ids.shape[1])
                active = list(getattr(bank, "last_active_fact_indices", [[]])[0] or [])
                row = int(active[0]) if active else None
                r["runs"][label] = {
                    "output": text, "has_answer": contains_answer(text, r["answer"]),
                    "routed_row": row,
                    "routed_fact_id": str(facts[row]["id"]) if row is not None else None,
                    "routed_answer": str(facts[row].get("object", "")) if row is not None else None,
                }
                if (k + 1) % 100 == 0:
                    print(f"  {label} {k + 1}/{len(rows)}", flush=True)
        finally:
            handle = getattr(bank, "_hook_handle", None)
            if handle is not None:
                handle.remove()

    summary = summarize(rows, labels)
    out = Path(a.out) if a.out else runs[0] / "generations"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"dataset": ds, "seed": seed, "model_path": model_path,
            "max_new_tokens": a.max_new_tokens, "runs": dict(zip(labels, map(str, runs)))}
    with open(f"{out}.jsonl", "w") as f:
        f.write(json.dumps({"meta": meta, "summary": summary}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    write_markdown(f"{out}.md", meta, summary, rows, labels, a.examples)
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}.jsonl and {out}.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
