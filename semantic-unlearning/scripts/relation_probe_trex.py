#!/usr/bin/env python3
"""A relation probe trained on Wikipedia sentences (T-REx), no benchmark data (analysis only).

T-REx (Elsahar et al., LREC 2018; Hugging Face relbert/t_rex) aligns Wikipedia abstracts with
Wikidata triples. Each row gives (relation, subject, object, text). A probe prompt is the
sentence that names the subject before the object, cut right before the object:
  "... the role-playing video game Neverwinter Nights 2, developed by"   -> P178 (developer)
The most frequent relations are kept, capped per relation. A prompt is dropped if it names a
protected subject of any benchmark and any seed (every *_multiseed_ours_v1 router), or is a test
prompt of the seed-1 caches. The probe is a softmax over relations on the unit-normalised
last-token state at the router layer, held out by subject.

    # 1. data (CPU, needs the internet: login node)
    python -u scripts/relation_probe_trex.py --stage data
    # 2. probe (GPU)
    python -u scripts/relation_probe_trex.py --stage probe
    # -> outputs/relation_gate_analysis_v1/trex_prompts.json, trex_probe.pt
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import zlib

import torch
from torch.nn import functional as F

DS = ("mcf", "zsre", "mquake", "rwku")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def norm(t):
    return " ".join(str(t).casefold().split())


def exclusions(root, outputs, seed):
    protected, test_text = set(), set()
    for ds in DS:
        for d in sorted((outputs / f"{ds}_multiseed_ours_v1").glob("seed*/L19/router/fact_association_embeddings.pt")):
            art = torch.load(d, map_location="cpu", weights_only=False)
            protected |= {norm(f["subject"]) for f in art["facts"] if len(str(f.get("subject", "")).strip()) >= 3}
        c = root / ds / f"seed{seed}"
        for s in ("eval", "null", "perturbed"):
            test_text |= {norm(p["prompt"]) for p in json.loads((c / f"probes_{s}.json").read_text()) if p["prompt"]}
    return protected, test_text


def prompt_of(text, head, tail):
    """The sentence naming head before tail, cut right before tail; None if there is none."""
    for sent in SENT.split(text):
        h, t = sent.find(head), sent.find(tail)
        if h < 0 or t < 0 or h + len(head) > t:
            continue
        prefix = sent[:t].rstrip()
        words = prefix.split()
        if 4 <= len(words) <= 40 and re.search(r"[A-Za-z]$", prefix):
            return prefix
    return None


def stage_data(args, out):
    from datasets import load_dataset
    protected, test_text = exclusions(Path(args.root), Path("outputs"), args.seed)
    prot_re = re.compile("|".join(re.escape(p) for p in sorted(protected, key=len, reverse=True)))
    by_rel = defaultdict(list)
    seen = set()
    # the hub's script loader is gone in datasets>=4: read the auto-converted Parquet files
    # (downloaded first: curl -L <hub>/api/datasets/relbert/t_rex/parquet/t_rex/train/{0,1,2}.parquet)
    files = sorted(str(f) for f in Path(args.trex_dir).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no T-REx parquet files in {args.trex_dir}")
    stream = load_dataset("parquet", data_files=files, split="train", streaming=True)
    rows = 0
    for row in stream:
        rows += 1
        if rows > args.max_rows:
            break
        head, tail, rel = str(row["head"]), str(row["tail"]), str(row["relation"])
        if len(head) < 2 or len(tail) < 2 or len(by_rel[rel]) >= 4 * args.cap:
            continue
        p = prompt_of(str(row["text"]), head, tail)
        if p is None:
            continue
        n = norm(p)
        if n in seen or n in test_text or prot_re.search(n):
            continue
        seen.add(n)
        by_rel[rel].append((p, head))
    counts = Counter({r: len(v) for r, v in by_rel.items()})
    keep = [r for r, c in counts.most_common(args.relations) if c >= args.min_per_relation]
    rng = random.Random(0)
    data = []
    for r in keep:
        items = by_rel[r]
        rng.shuffle(items)
        data += [{"prompt": p, "relation": r, "subject": h} for p, h in items[:args.cap]]
    (out / "trex_prompts.json").write_text(json.dumps(
        {"rows_read": rows, "relations": keep, "counts": {r: min(counts[r], args.cap) for r in keep},
         "protected_subjects": len(protected), "test_prompts": len(test_text), "prompts": data}, indent=1))
    print(json.dumps({"phase": "data", "rows_read": rows, "relations": len(keep), "prompts": len(data),
                      "example": data[:3]}), flush=True)


def stage_clean(args, out):
    """Cleaner T-REx prompts: start each prompt at its subject, keep it only if its last
    --ending-words words point to one relation across the corpus (purity >= --purity over >= --min-ending
    prompts), and drop relations left with fewer than --min-per-relation prompts."""
    blob = json.loads((out / "trex_prompts.json").read_text())
    items = []
    for d in blob["prompts"]:
        k = d["prompt"].find(d["subject"])
        if k < 0:
            continue
        anchored = d["prompt"][k:].strip()
        if len(anchored.split()) < 3:
            continue
        items.append((anchored, d["relation"], d["subject"], " ".join(anchored.casefold().split()[-args.ending_words:])))
    by_end = defaultdict(Counter)
    for _, r, _, e in items:
        by_end[e][r] += 1
    kept = []
    for a, r, h, e in items:
        c = by_end[e]
        total = sum(c.values())
        if total >= args.min_ending and c[r] / total >= args.purity:
            kept.append({"prompt": a, "relation": r, "subject": h})
    counts = Counter(d["relation"] for d in kept)
    rels = [r for r, c in counts.most_common() if c >= args.min_per_relation]
    kept = [d for d in kept if d["relation"] in rels]
    (out / "trex_clean_prompts.json").write_text(json.dumps(
        {"relations": rels, "counts": {r: counts[r] for r in rels}, "purity": args.purity,
         "min_ending": args.min_ending, "source_prompts": len(blob["prompts"]), "prompts": kept}, indent=1))
    print(json.dumps({"phase": "clean", "kept": len(kept), "of": len(blob["prompts"]), "relations": len(rels),
                      "example": kept[:4]}), flush=True)


def stage_probe(args, out):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cache_router_states import capture_last_token_states
    from fit_relation_gate import fit_softmax_probe
    blob = json.loads((out / args.prompts).read_text())
    rels, data = blob["relations"], blob["prompts"]
    model_path = args.model_path or json.loads((Path(args.root) / "mcf" / f"seed{args.seed}" / "manifest.json").read_text())["model_path"]
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, local_files_only=True,
                                                 attn_implementation="eager").cuda().eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = [tok(d["prompt"])["input_ids"] for d in data]
    x = F.normalize(capture_last_token_states(model, ids, [args.layer], batch_size=128, pad_id=pad)[0].float(), dim=-1)
    y = torch.tensor([rels.index(d["relation"]) for d in data])
    held = torch.tensor([zlib.crc32(d["subject"].encode()) % 10 == 0 for d in data])
    acc = {}
    for l2 in (1e-5, 1e-4, 1e-3):
        w, b = fit_softmax_probe(x[~held], y[~held], len(rels), l2, "cuda")
        acc[l2] = float(((x[held] @ w.T + b).argmax(1) == y[held]).float().mean())
    l2 = max(acc, key=acc.get)
    w, b = fit_softmax_probe(x, y, len(rels), l2, "cuda")
    torch.save({"weight": w.float().cpu(), "bias": b.float().cpu(), "relations": rels, "layer": args.layer,
                "l2": l2, "heldout_accuracy_by_subject": acc, "prompts": len(data),
                "model_path": model_path, "source": "scripts/relation_probe_trex.py (relbert/t_rex)"},
               out / f"{args.name}.pt")
    print(json.dumps({"phase": "probe", "model": model_path, "layer": args.layer, "relations": len(rels), "prompts": len(data),
                      "heldout_accuracy_by_subject": acc, "l2": l2}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("data", "clean", "probe"), required=True)
    ap.add_argument("--prompts", default="trex_prompts.json", help="probe stage: trex_prompts.json or trex_clean_prompts.json")
    ap.add_argument("--purity", type=float, default=0.7)
    ap.add_argument("--min-ending", type=int, default=5)
    ap.add_argument("--ending-words", type=int, default=2, help="clean: words at the end of a prompt that must point to one relation")
    ap.add_argument("--root", default="outputs/router_diag_ours_control_v1")
    ap.add_argument("--out", default="outputs/relation_gate_analysis_v1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--trex-dir", default="/scratch/yl258/kp759/datasets/t_rex")
    ap.add_argument("--model-path", default=None, help="probe stage: default the 3B caches' model")
    ap.add_argument("--name", default="trex_probe", help="probe stage: output file stem")
    ap.add_argument("--max-rows", type=int, default=600000)
    ap.add_argument("--relations", type=int, default=100)
    ap.add_argument("--min-per-relation", type=int, default=100, help="data: before the cap; clean: after filtering (use 50)")
    ap.add_argument("--cap", type=int, default=800)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    {"data": stage_data, "clean": stage_clean, "probe": stage_probe}[args.stage](args, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
