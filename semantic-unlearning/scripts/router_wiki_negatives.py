#!/usr/bin/env python3
"""Up to 100 Wikipedia negatives per fact, covering ~100 relations (analysis only; MCF seed 1).

For every fact, T-REx sentences (outputs/relation_gate_analysis_v1/trex_prompts.json: a Wikipedia
sentence cut right before its object, labelled with its Wikidata relation) of relations OTHER
than the fact's, with the sentence's subject replaced by the fact's subject:
    "Neverwinter Nights 2, developed by"  ->  "Belgium, developed by"      (P178, not Belgium's P463)
Relations whose answers overlap the fact's (RELATION_ANSWER_GROUPS, + the fact's own) are skipped;
sentences containing the fact's answer are skipped; at most --per-relation per relation so the
100 cover as many relations as possible. Then the heads are refit on the C positives
(--positives-json, fit split) against these negatives (+ the C negatives when --keep-c-negatives),
per-head bias on the C calibration split, scored on TEST prompts only (router_test_false_fires
style). Heads are the paper's: class-balanced BCE + L2 on the unit, centred block-19 state.

    python -u scripts/router_wiki_negatives.py --dataset mcf --seed 1 \
        --positives-json outputs/router_data_grid_v1/mcf/seed1/p40_n40/router/linear_router_dataset.json
    # -> outputs/router_wiki_negatives_v1/<ds>_seed<S>.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re

import torch
from torch.nn import functional as F

from cache_router_states import capture_last_token_states
from linear_router import RELATION_ANSWER_GROUPS, decide_routes
from router_negative_clean_gate import contains, overlap
from router_rewrites import RewriteTarget, answer_transferred, transfer_prompt


def norm(t):
    return " ".join(str(t or "").casefold().split())


SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
CUTS = (" is a ", " is an ", " was a ", " was an ", " is the ", " was the ", " is located in ", " is in ",
        " was born in ", " is a member of ", " belongs to ", " works for ", " plays for ", " was founded ",
        " is known for ", " lies in ", " is part of ", " was created by ", " was developed by ", " was written by ")


def wikipedia_pool(path, max_articles, seed):
    """ZeroUnlearn's Wikipedia dump as a pool of (prompt, subject, pseudo-relation): the first
    sentence of an article that starts with its title, cut right after a relation phrase, so the
    object is left out. No labels: the relation key is the phrase itself (it only spreads the
    sample over phrasings)."""
    from datasets import load_from_disk
    ds = load_from_disk(path)["train"]
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx[:max_articles]:
        title, text = str(ds[i]["title"]), str(ds[i]["text"])
        first = SENT.split(text.strip(), 1)[0].strip()
        if not first.startswith(title) or len(first) > 240 or len(title) < 3:
            continue
        rest = first[len(title):]
        m = re.search(r"\s(?=[A-Z0-9])", rest[1:])          # the first capitalised word / number after the title
        if m is None or m.start() < 6:
            continue
        prompt = (title + rest[:1 + m.start()]).rstrip(" ,;:(")
        if len(prompt.split()) < 3 or len(prompt.split()) > 40:
            continue
        words = prompt[len(title):].split()
        key = " ".join(words[-2:]) if len(words) >= 2 else "x"       # pseudo-relation: the two words before the cut
        out.append({"prompt": prompt, "subject": title, "relation": key})
    return out


def fit_heads(x, y, mask, gamma, iters=300):
    yb, m = y.float(), mask.float()
    n_pos = (yb * m).sum(0).clamp_min(1)
    n_neg = ((1 - yb) * m).sum(0).clamp_min(1)
    weight = m * (yb / (2 * n_pos) + (1 - yb) / (2 * n_neg))
    w = torch.zeros(x.shape[1], y.shape[1], requires_grad=True)
    b = torch.zeros(y.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(x @ w + b, yb, reduction="none") * weight).sum() + gamma * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="mcf")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--root", default="outputs/router_diag_ours_control_v1")
    ap.add_argument("--trex", default="outputs/relation_gate_analysis_v1/trex_prompts.json")
    ap.add_argument("--source", choices=("trex", "wikipedia"), default="trex",
                    help="trex: relation-labelled T-REx sentences, ~100 relations per fact; wikipedia: ZeroUnlearn's "
                         "unlabelled 100k-article dump (--wikipedia), first sentences cut before their object")
    ap.add_argument("--wikipedia", default="/project/yl258/kp759/datasets/wikipedia_20231101_100k")
    ap.add_argument("--positives-json", required=True)
    ap.add_argument("--per-fact", type=int, default=100)
    ap.add_argument("--per-relation", type=int, default=2)
    ap.add_argument("--recall-floors", default="0.90,0.95", help="per-head bias: strictest threshold keeping this share of calibration positives")
    ap.add_argument("--mix", action="store_true", help="also a 1:1 mix (as many wiki negatives per fact as C negatives)")
    ap.add_argument("--wiki-positives", type=int, default=0,
                    help="also up to this many Wikipedia POSITIVES per fact: pool sentences whose relation key occurs in one "
                         "of the fact's own rewrites, the subject swapped in, kept only if the base model (told the fact) "
                         "answers them with the fact's answer")
    ap.add_argument("--gammas", default="1e-4,1e-3")
    ap.add_argument("--out", default="outputs/router_wiki_negatives_v1")
    args = ap.parse_args(argv)
    cache = Path(args.root) / args.dataset / f"seed{args.seed}"
    man = json.loads((cache / "manifest.json").read_text())
    art = torch.load(Path(man["router_dir"]) / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    facts, layer = art["facts"], int(man["router_layer"])
    n = len(facts)
    fid = {f["id"]: i for i, f in enumerate(facts)}
    subjects = [str(f["subject"]) for f in facts]

    # ---- Wikipedia negatives per fact
    if args.source == "trex":
        trex = json.loads(Path(args.trex).read_text())["prompts"]
    else:
        trex = wikipedia_pool(args.wikipedia, max_articles=40000, seed=args.seed)
    by_rel = {}
    for p in trex:
        if p["subject"] and p["subject"] in p["prompt"]:
            by_rel.setdefault(p["relation"], []).append(p)
    rels = sorted(by_rel)
    wiki = []                                                    # (prompt, fact index, relation)
    rng = random.Random(args.seed)
    stats = {"facts": n, "relations_in_pool": len(rels), "skipped_overlap_relation": 0, "skipped_answer_in_text": 0}
    for i, f in enumerate(facts):
        rel_f, ans = str(f["relation"]), str(f["object"])
        order = rels[:]
        rng.shuffle(order)
        taken = 0
        for r in order:
            if r == rel_f or (r.startswith("P") and rel_f.startswith("P") and overlap(rel_f, r)) or (
                    args.source == "wikipedia" and any(w in r for w in ("affiliated", "member", "language", "located", "born", "citizen")) and False):
                stats["skipped_overlap_relation"] += 1
                continue
            k = 0
            for p in rng.sample(by_rel[r], min(len(by_rel[r]), args.per_relation * 3)):
                text = p["prompt"].replace(p["subject"], subjects[i], 1)
                if contains(text, ans) or not re.search(r"[A-Za-z]$", text):
                    stats["skipped_answer_in_text"] += 1
                    continue
                wiki.append((text, i, r))
                taken += 1
                k += 1
                if k >= args.per_relation or taken >= args.per_fact:
                    break
            if taken >= args.per_fact:
                break
    stats["wiki_negatives"] = len(wiki)
    stats["per_fact_min"] = min(sum(1 for _, i, _ in wiki if i == j) for j in range(n))
    stats["relations_per_fact_median"] = sorted(len({r for _, i, r in wiki if i == j}) for j in range(n))[n // 2]
    print(json.dumps({"phase": "wiki_negatives", **stats, "examples": [w[0] for w in wiki[:4]]}), flush=True)

    # ---- C positives and negatives
    data = json.loads(Path(args.positives_json).read_text())
    rows = [e for e in data if e["split"] in ("fit", "calibration")]
    lab = torch.zeros(len(rows) + len(wiki), n, dtype=torch.long)
    for k, e in enumerate(rows):
        if e["kind"] == "positive":
            lab[k, fid[e["owner_fact_id"]]] = 1
            continue
        for fact_id in e.get("negative_for") or []:
            i = fid.get(fact_id)
            if i is None:
                continue
            f, donor = facts[i], str(e.get("donor_relation") or "")
            amb = contains(e["prompt"], f["object"]) or (e["kind"] == "subject_transplant" and donor.startswith("P")
                                                         and str(f["relation"]).startswith("P") and overlap(f["relation"], donor))
            lab[k, i] = -2 if amb else -1
    for k, (_, i, _) in enumerate(wiki):
        lab[len(rows) + k, i] = -3                               # wiki negative (fit split)
    texts = [e["prompt"] for e in rows] + [w[0] for w in wiki]
    is_fit = torch.tensor([e["split"] == "fit" for e in rows] + [True] * len(wiki))
    is_cal = torch.tensor([e["split"] == "calibration" for e in rows] + [False] * len(wiki))
    elig = torch.zeros(len(texts), n, dtype=torch.bool)
    for k, t in enumerate(texts):
        tl = t.casefold()
        for i in range(n):
            elig[k, i] = subjects[i].casefold() in tl
    trained = {norm(t) for t in texts}

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(man["model_path"], local_files_only=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(man["model_path"], dtype=torch.float32, local_files_only=True,
                                                 attn_implementation="eager").cuda().eval()
    wiki_pos = []                                                # (prompt, fact index)
    if args.wiki_positives and args.source == "wikipedia":
        pos_texts = {i: [e["prompt"] for e in rows if e["kind"] == "positive" and fid[e["owner_fact_id"]] == i] for i in range(n)}
        pstats = {"candidates": 0, "kept": 0}
        cand = []
        for i, f in enumerate(facts):
            keys = {w for t in pos_texts[i] for w in norm(t).split() if len(w) > 3 and w not in norm(subjects[i]).split()}
            hits = [p for r, ps in by_rel.items() for p in ps if set(norm(r).split()) & keys][:args.wiki_positives * 4]
            for p in hits:
                text = p["prompt"].replace(p["subject"], subjects[i], 1)
                if contains(text, str(f["object"])) or not re.search(r"[A-Za-z]$", text):
                    continue
                cand.append((text, i))
        pstats["candidates"] = len(cand)
        targets = {i: RewriteTarget(str(f["id"]), subjects[i], str(f["object"]), str(f["canonical_prompt"]), "cloze") for i, f in enumerate(facts)}
        model_bf = AutoModelForCausalLM.from_pretrained(man["model_path"], dtype=torch.bfloat16, local_files_only=True,
                                                        attn_implementation="eager").cuda().eval()
        items = [(transfer_prompt(tok, targets[i], text, "cloze"), str(facts[i]["object"]), True) for text, i in cand]
        ok = answer_transferred(model_bf, tok, items, batch_size=16) if items else []
        del model_bf
        torch.cuda.empty_cache()
        per = {i: 0 for i in range(n)}
        for (text, i), good in zip(cand, ok):
            if good and per[i] < args.wiki_positives:
                wiki_pos.append((text, i))
                per[i] += 1
        pstats["kept"] = len(wiki_pos)
        pstats["facts_with_any"] = sum(1 for v in per.values() if v)
        pstats["per_fact_median"] = sorted(per.values())[n // 2]
        print(json.dumps({"phase": "wiki_positives", **pstats, "examples": [w[0] for w in wiki_pos[:4]]}), flush=True)
        # append as fit-split positives
        extra = torch.zeros(len(wiki_pos), n, dtype=torch.long)
        for k, (_, i) in enumerate(wiki_pos):
            extra[k, i] = 1
        lab = torch.cat([lab, extra])
        texts = texts + [w[0] for w in wiki_pos]
        is_fit = torch.cat([is_fit, torch.ones(len(wiki_pos), dtype=torch.bool)])
        is_cal = torch.cat([is_cal, torch.zeros(len(wiki_pos), dtype=torch.bool)])
        el2 = torch.zeros(len(wiki_pos), n, dtype=torch.bool)
        for k, (t, _) in enumerate(wiki_pos):
            tl = t.casefold()
            for i in range(n):
                el2[k, i] = subjects[i].casefold() in tl
        elig = torch.cat([elig, el2])
        trained |= {norm(t) for t, _ in wiki_pos}
    h_tr = capture_last_token_states(model, [tok(t)["input_ids"] for t in texts], [layer], batch_size=32, pad_id=pad)[0].float()
    del model
    torch.cuda.empty_cache()
    T = {}
    for name in ("eval", "null"):
        st = torch.load(cache / f"states_{name}.pt", weights_only=False, mmap=True)
        T[name] = {"probes": json.loads((cache / f"probes_{name}.json").read_text()),
                   "h": st["states"][list(st["layers"]).index(layer)].float(),
                   "eligible": torch.load(cache / f"router_{name}.pt", weights_only=False)["eligible"]}
    should = torch.tensor([p["should_fire"] for p in T["eval"]["probes"]])
    null_group = [p["group"] for p in T["null"]["probes"]]
    null_text = [norm(p.get("prompt")) for p in T["null"]["probes"]]
    groups = ["null_generic", "null_ppl_prefix", "null_unseen_subject", "null_protected_other_relation"]

    u_all = F.normalize(h_tr, dim=-1)
    results = {}
    # a 1:1 mix: per fact, as many wiki negatives as the head has C negatives in the fit split
    mix_ok = torch.zeros_like(lab, dtype=torch.bool)
    if args.mix:
        rng2 = random.Random(args.seed + 1)
        for i in range(n):
            n_c = int(((lab[:, i] == -1) & is_fit).sum())
            wk = [k for k in range(len(rows), len(rows) + len(wiki)) if lab[k, i] == -3]
            for k in rng2.sample(wk, min(n_c, len(wk))):
                mix_ok[k, i] = True
    ptag = f" + {len(wiki_pos)} wiki positives" if wiki_pos else ""
    variants = [("C negatives only" + ptag, True, False, False), ("wiki negatives only" + ptag, False, True, False),
                ("C + wiki negatives" + ptag, True, True, False)]
    if args.mix:
        variants.append(("C + wiki 1:1 mix" + ptag, True, False, True))
    floors = [None] + [float(f) for f in args.recall_floors.split(",")]
    for variant, use_c_neg, use_wiki, use_mix in variants:
        neg_ok = torch.zeros_like(lab, dtype=torch.bool)
        if use_c_neg:
            neg_ok |= lab == -1
        if use_wiki:
            neg_ok |= lab == -3
        if use_mix:
            neg_ok |= mix_ok
        mask = is_fit[:, None] & ((lab == 1) | neg_ok) & elig
        for gamma in [float(g) for g in args.gammas.split(",")]:
            mu = u_all[is_fit & mask.any(1)].mean(0)
            x = u_all - mu
            w, b = fit_heads(x[is_fit], (lab == 1)[is_fit], mask[is_fit], gamma)
            z_cal = x[is_cal] @ w + b
            cand = (lab[is_cal] == -1) & elig[is_cal]
            pos_c = (lab[is_cal] == 1) & elig[is_cal]
            for floor in floors:
                bias = b.clone()
                for i in range(n):
                    s = z_cal[cand[:, i], i]
                    if floor is None:
                        if s.numel() and float(s.max()) >= 0:
                            bias[i] = b[i] - (float(s.max()) + 1e-3)
                    else:
                        sp = z_cal[pos_c[:, i], i]
                        if not s.numel() or not sp.numel():
                            continue
                        # strictest threshold keeping >= floor of the head's calibration positives
                        kth = max(0, int((1 - floor) * sp.numel()))
                        t_pos = float(sp.sort().values[kth])
                        t = min(float(s.max()) + 1e-3, t_pos)
                        if t > 0:
                            bias[i] = b[i] - t
                res = {}
                for name in ("eval", "null"):
                    z = (F.normalize(T[name]["h"], dim=-1) - mu) @ w + bias
                    dec = decide_routes(z, T[name]["eligible"], 0.0, 0.5)
                    fire, fact = dec["active"].cpu(), dec["fact"].cpu()
                    if name == "eval":
                        right = torch.tensor([bool(fire[k]) and p["should_fire"] and int(fact[k]) in p["true_facts"]
                                              for k, p in enumerate(T[name]["probes"])])
                        res["forget_recall"] = float(right[should].float().mean())
                        res["retain_fires"] = f"{int((fire & ~should).sum())}/{int((~should).sum())}"
                    else:
                        for g in groups:
                            idx = [k for k in range(len(fire)) if null_group[k] == g and null_text[k] not in trained]
                            res[g] = f"{sum(int(fire[k]) for k in idx)}/{len(idx)}"
                tagv = f"{variant} (gamma={gamma:g}, bias={'max-neg' if floor is None else f'recall>={floor:g}'})"
                results[tagv] = res
                print(json.dumps({tagv: res}), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_seed{args.seed}_{args.source}"
    (out / f"{tag}.json").write_text(json.dumps({"stats": stats, "results": results}, indent=2) + "\n")
    lines = [f"### {args.dataset} seed {args.seed}: C positives (40/fact) with {args.source} negatives, test prompts only",
             f"Wikipedia negatives: {stats['wiki_negatives']} ({stats['per_fact_min']}+ per fact, "
             f"{stats['relations_per_fact_median']} relations per fact, {stats['relations_in_pool']} in the pool)", "",
             "| heads trained on | forget recall | retain false fires | generic | perplexity | unseen subject | **protected subject, other relation** |",
             "|---|---|---|---|---|---|---|"]
    for v, r in results.items():
        lines.append(f"| {v} | {r['forget_recall']:.3f} | {r['retain_fires']} | {r['null_generic']} | {r['null_ppl_prefix']} | "
                     f"{r['null_unseen_subject']} | **{r['null_protected_other_relation']}** |")
    (out / f"{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
