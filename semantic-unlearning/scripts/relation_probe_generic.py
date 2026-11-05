#!/usr/bin/env python3
"""A relation probe that uses no benchmark data (analysis only).

relation_gate_universal_check.py trains its probe on MultiCounterFact prompts, the distribution
of one benchmark. This one starts from a hand-written list of common relations and the kind of
subject each takes; the frozen base model writes the phrasings (incomplete sentences, questions,
requests) and lists example subjects. Prompts naming a protected subject of any benchmark, and
test prompts of any benchmark, are dropped. A share is wrapped in the chat template (RWKU's
format). The probe is a softmax over relations on the unit-normalised last-token state at the
router layer, held out by subject.

    python -u scripts/relation_probe_generic.py            # GPU, ~10 min
    # -> outputs/relation_gate_analysis_v1/generic_probe.pt (+ generic_probe_data.json)
    python -u scripts/router_negative_gate_all.py --probe outputs/relation_gate_analysis_v1/generic_probe.pt \
        --datasets mcf,zsre,mquake,rwku --tag generic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import zlib

import torch
from torch.nn import functional as F

from cache_router_states import capture_last_token_states
from fit_relation_gate import fit_softmax_probe

# (relation, subject type): written from general knowledge, independent of every benchmark
RELATIONS = [
    ("place of birth", "person"), ("date of birth", "person"), ("place of death", "person"),
    ("occupation", "person"), ("employer", "person"), ("university attended", "person"),
    ("country of citizenship", "person"), ("native language", "person"), ("languages spoken", "person"),
    ("spouse", "person"), ("father", "person"), ("religion", "person"), ("cause of death", "person"),
    ("sibling", "person"), ("award received", "person"), ("political party", "politician"),
    ("political office held", "politician"), ("field of work", "scientist"), ("sport", "athlete"),
    ("sports team", "athlete"), ("playing position", "athlete"), ("instrument played", "musician"),
    ("record label", "musician"), ("music genre", "musician"), ("notable work", "writer"),
    ("artistic movement", "painter"), ("country", "city"), ("administrative region", "city"),
    ("continent", "country"), ("capital city", "country"), ("official language", "country"),
    ("currency", "country"), ("time zone", "city"), ("sister city", "city"),
    ("body of water it lies on", "city"), ("river mouth", "river"), ("headquarters location", "company"),
    ("founder", "company"), ("parent company", "company"), ("industry", "company"),
    ("chief executive officer", "company"), ("manufacturer", "product"), ("developer", "video game"),
    ("country of origin", "product"), ("original language", "film"), ("director", "film"),
    ("release year", "film"), ("author", "book"), ("publication year", "book"), ("composer", "film"),
    ("performer", "song"), ("original broadcaster", "television series"), ("named after", "landmark"),
    ("location", "landmark"), ("architect", "landmark"), ("member of", "country"),
    ("year founded", "organization"), ("home stadium", "sports team"), ("league", "sports team"),
    ("owner", "sports team"),
]
PLURAL = {"person": "famous people", "politician": "politicians", "scientist": "scientists", "athlete": "athletes",
          "musician": "musicians", "writer": "writers", "painter": "painters", "city": "cities", "country": "countries",
          "river": "rivers", "company": "companies", "product": "consumer products", "video game": "video games",
          "film": "films", "book": "books", "song": "songs", "television series": "television series",
          "landmark": "landmarks and buildings", "organization": "organizations", "sports team": "sports teams"}


def chat_ids(tok, text):
    """Token ids of a one-turn chat prompt, ready for the answer (works across transformers versions)."""
    rendered = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
    return tok(rendered, add_special_tokens=False)["input_ids"]


def chat(tok, model, text, max_new_tokens=400, temperature=0.8, seed=0):
    ids = torch.tensor([chat_ids(tok, text)], device=model.device)
    torch.manual_seed(seed)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=0.95,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def lines(text):
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').strip()
        if line:
            out.append(line)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/router_diag_ours_control_v1", help="caches: protected subjects and test prompts")
    ap.add_argument("--out", default="outputs/relation_gate_analysis_v1")
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per-relation", type=int, default=600)
    ap.add_argument("--chat-share", type=float, default=0.25)
    args = ap.parse_args(argv)
    rng = random.Random(0)
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    norm = lambda t: " ".join(str(t).casefold().split())

    protected, test_text, model_path = set(), set(), None
    for ds in ("mcf", "zsre", "mquake", "rwku"):
        d = root / ds / f"seed{args.seed}"
        man = json.loads((d / "manifest.json").read_text())
        model_path = model_path or man["model_path"]
        protected |= {norm(f["subject"]) for f in man["facts"] if len(str(f.get("subject", "")).strip()) >= 3}
        for s in ("eval", "null", "perturbed"):
            test_text |= {norm(p["prompt"]) for p in json.loads((d / f"probes_{s}.json").read_text()) if p["prompt"]}
    ok_subject = lambda s: not any(p in norm(s) or norm(s) in p for p in protected)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, local_files_only=True,
                                                 attn_implementation="eager").cuda().eval()

    subjects = {}
    for kind in sorted({k for _, k in RELATIONS}):
        names = []
        for s in range(2):
            names += lines(chat(tok, model, f"List 40 well-known {PLURAL[kind]}, one per line, names only, "
                                            f"no descriptions.", seed=s))
        names = [n for n in dict.fromkeys(names) if 2 <= len(n) <= 60 and ok_subject(n)]
        subjects[kind] = names
    templates = {}
    for rel, kind in RELATIONS:
        req = (f"Write 15 different ways to ask for the {rel} of a {kind}. Use the letter X in place of the {kind}'s "
               f"name. Mix incomplete sentences that stop right before the answer (for example 'X was born in' for "
               f"place of birth), direct questions, and polite requests. One per line, no numbering, no answers.")
        ts = []
        for s in range(2):
            ts += [t for t in lines(chat(tok, model, req, seed=s)) if re.search(r"\bX\b", t) and len(t) <= 160]
        templates[rel] = list(dict.fromkeys(ts))

    data = []                                           # (text or token ids, relation index, subject)
    for r, (rel, kind) in enumerate(RELATIONS):
        pairs = [(t, s) for t in templates[rel] for s in subjects[kind]]
        rng.shuffle(pairs)
        kept = 0
        for t, s in pairs:
            text = re.sub(r"\bX\b", s, t)
            if norm(text) in test_text:
                continue
            if rng.random() < args.chat_share and text.endswith("?"):
                data.append((chat_ids(tok, text), r, s))
            else:
                data.append((tok(text)["input_ids"], r, s))
            kept += 1
            if kept >= args.per_relation:
                break
    (out / "generic_probe_data.json").write_text(json.dumps(
        {"relations": RELATIONS, "subjects": subjects, "templates": templates,
         "counts": {rel: sum(1 for _, r, _ in data if r == i) for i, (rel, _) in enumerate(RELATIONS)}}, indent=1))
    print(json.dumps({"phase": "data", "prompts": len(data), "relations": len(RELATIONS),
                      "templates_median": sorted(len(v) for v in templates.values())[len(templates) // 2],
                      "subjects_total": sum(len(v) for v in subjects.values())}), flush=True)

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    x = F.normalize(capture_last_token_states(model, [d for d, _, _ in data], [args.layer], batch_size=128,
                                              pad_id=pad)[0].float(), dim=-1)
    y = torch.tensor([r for _, r, _ in data])
    held = torch.tensor([zlib.crc32(s.encode()) % 10 == 0 for _, _, s in data])          # held out by subject
    acc = {}
    for l2 in (1e-5, 1e-4, 1e-3):
        w, b = fit_softmax_probe(x[~held], y[~held], len(RELATIONS), l2, "cuda")
        acc[l2] = float(((x[held] @ w.T + b).argmax(1) == y[held]).float().mean())
    l2 = max(acc, key=acc.get)
    w, b = fit_softmax_probe(x, y, len(RELATIONS), l2, "cuda")
    torch.save({"weight": w.float().cpu(), "bias": b.float().cpu(), "relations": [r for r, _ in RELATIONS],
                "layer": args.layer, "l2": l2, "heldout_accuracy_by_subject": acc, "prompts": len(data),
                "source": "scripts/relation_probe_generic.py"}, out / "generic_probe.pt")
    print(json.dumps({"phase": "probe", "heldout_accuracy_by_subject": acc, "l2": l2}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
