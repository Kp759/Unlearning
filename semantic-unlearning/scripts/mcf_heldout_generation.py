#!/usr/bin/env python3
"""Held-out free generation on the official MCF prompts (forget leak + over-refusal).

The earlier leak table used our own training views (canonical + development), which the
Δe / refusal rows were trained on. This scores greedy 24-token continuations on the MCF
record's OWN prompts for the seed's 50 forget facts:

  rewrite       requested_rewrite prompt (the canonical prompt; seen in training by us
                AND by ZeroUnlearn)                              -> leak ↓, refusal
  paraphrase    paraphrase_prompts (held out; = the Gen prompts)  -> leak ↓, refusal
  generation    generation_prompts: same subject, other topics    -> leak, refusal ↓ (over-refusal)
                (held out; e.g. "X's work office is surrounded by")
  neighborhood  neighborhood_prompts: same relation + true answer,
                OTHER subjects (held out; = the Spe prompts)     -> answer kept ↑, refusal ↓

Methods (one per call, so a Slurm array can run them in parallel):
  base         plain model
  router       any router artifact (Δe, refusal rows, projection, ...): --run-dir
  zerounlearn  original ZeroUnlearn, target_true parity, applied in-process (~11 min)

    python scripts/mcf_heldout_generation.py run --method router --run-dir R --label "Δe→EOS" \\
        --out outputs/mcf_heldout_generation_seed1/R_eos.json --local-files-only
    python scripts/mcf_heldout_generation.py summarize --root outputs/mcf_heldout_generation_seed1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

SEMANTIC_ROOT = Path(__file__).resolve().parents[1]
SETS = ("rewrite", "paraphrase", "generation", "neighborhood")


def heldout_prompts(mcf_path, seed, forget_num=50):
    from mcf_sampling import sample_official_mcf_records

    data = json.loads(Path(mcf_path).read_text())
    records, _ = sample_official_mcf_records(data, forget_num, 0, seed, strict=True)
    prompts = []
    for record in records:
        rw = record["requested_rewrite"]
        rw = rw[0] if isinstance(rw, list) else rw
        cid, subject, answer = int(record["case_id"]), rw["subject"], rw["target_true"]["str"]
        groups = {
            "rewrite": [rw["prompt"].format(subject)],
            "paraphrase": list(record.get("paraphrase_prompts", [])),
            "generation": list(dict.fromkeys(record.get("generation_prompts", []))),
            "neighborhood": list(record.get("neighborhood_prompts", [])),
        }
        for name, texts in groups.items():
            for k, text in enumerate(texts):
                prompts.append({"id": f"{cid}:{name}:{k}", "case_id": cid, "set": name,
                                "subject": subject, "prompt": text, "answer": answer})
    return records, prompts


def is_refusal(text):
    t = text.strip().casefold().replace("’", "'")
    return not t or t.startswith("i don't know") or t.startswith("i do not know")


def _fact_case(fact):
    for key in ("case_id", "mcf_case_id"):
        if fact.get(key) is not None:
            return int(fact[key])
    digits = "".join(ch for ch in str(fact.get("id", "")).split("_")[-1] if ch.isdigit())
    return int(digits) if digits else None


def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from zerounlearn_generation_leak import eos_id_set, greedy

    artifact = manifest = None
    model_path = args.model_path
    if args.method == "router":
        run_dir = Path(args.run_dir)
        artifact = torch.load(run_dir / "fact_association_embeddings.pt", map_location="cpu",
                              weights_only=False)
        manifest = json.loads((run_dir / "association_manifest.json").read_text())
        model_path = manifest["model_path"]
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True,
                                        local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=args.local_files_only,
        attn_implementation="eager", low_cpu_mem_usage=True).to(device).eval()
    model.config.use_cache = False
    model.requires_grad_(False)

    _, prompts = heldout_prompts(args.mcf_path, args.seed)
    if args.limit:
        prompts = [p for p in prompts if p["case_id"] in
                   sorted({q["case_id"] for q in prompts})[:args.limit]]
    print(f"{args.label}: {len(prompts)} prompts "
          + str({s: sum(p['set'] == s for p in prompts) for s in SETS}), flush=True)

    bank = None
    if args.method == "router":
        from linear_router import load_router_artifact

        gen_model, bank = load_router_artifact(model, artifact)
        facts = artifact["facts"]
    elif args.method == "zerounlearn":
        from zerounlearn_generation_leak import apply_zerounlearn

        model.requires_grad_(True)
        model, _, _ = apply_zerounlearn(model, tok, args)
        model.requires_grad_(False)
    eos_ids = eos_id_set(model, tok)

    started = time.perf_counter()
    rows = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p["prompt"], return_tensors="pt")["input_ids"].to(device)
            if bank is not None:
                out = gen_model.generate_uncached_fixed_boundary(ids, max_new_tokens=args.max_new_tokens)
                active = list(bank.last_active_fact_indices[0]) if bank.last_active_fact_indices else []
                routed_case = _fact_case(facts[active[0]]) if active else None
            else:
                out = greedy(model, ids, args.max_new_tokens, eos_ids)
                active, routed_case = [], None
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            answer = p["answer"].casefold()
            rows.append({**p, "continuation": text, "routed": bool(active),
                         "routed_to_own_fact": routed_case == p["case_id"] if active else False,
                         "answer_in_continuation": answer in text.casefold(),
                         "answer_in_prompt": answer in p["prompt"].casefold(),
                         "refusal": is_refusal(text)})
    seconds = time.perf_counter() - started
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "label": args.label, "method": args.method, "run_dir": args.run_dir, "seed": args.seed,
        "max_new_tokens": args.max_new_tokens, "generation_seconds": seconds,
        "summary": summarize_rows(rows), "examples": rows}, indent=2) + "\n")
    print(json.dumps(summarize_rows(rows), indent=2))


def summarize_rows(rows):
    out = {}
    for name in SETS:
        sub = [r for r in rows if r["set"] == name]
        clean = [r for r in sub if not r["answer_in_prompt"]]
        n = max(1, len(sub))
        out[name] = {
            "prompts": len(sub),
            "answer_in_continuation": sum(r["answer_in_continuation"] for r in sub) / n,
            "answer_in_continuation_excl": (sum(r["answer_in_continuation"] for r in clean)
                                            / max(1, len(clean))),
            "refusal": sum(r["refusal"] for r in sub) / n,
            "routed": sum(r["routed"] for r in sub) / n,
            "routed_to_own_fact": sum(r["routed_to_own_fact"] for r in sub) / n,
        }
    return out


# ------------------------------------------------------------------------ summarize

def _route(data, stats, key="routed"):
    if data["method"] == "router":
        return _pct(stats[key])
    return "all (weights)" if data["method"] == "zerounlearn" else "–"


def _pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def _train_view_split(path):
    """leak/refusal on our training views, split canonical vs development."""
    data = json.loads(Path(path).read_text())["examples"]
    out = {}
    for name, keep in (("canonical", lambda r: r["split"] == "train"),
                       ("development", lambda r: r["split"] == "development")):
        sub = [r for r in data if keep(r)]
        n = max(1, len(sub))
        out[name] = (sum(r["answer"].casefold() in r["continuation"].casefold() for r in sub) / n,
                     sum(is_refusal(r["continuation"]) for r in sub) / n, len(sub))
    return out


def summarize(args):
    root = Path(args.root)
    order = {name: k for k, name in enumerate(args.order)}
    results = []
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text())
        if "summary" in data and "label" in data:
            results.append((path.stem, data))
    results.sort(key=lambda x: order.get(x[0], len(order)))
    train_views = dict(spec.split("=", 1) for spec in args.train_views)

    lines = [
        f"# Held-out generation, MCF seed {args.seed} (greedy, 24 tokens)\n",
        "Forget prompts: **leak** = true answer anywhere in the continuation (lower is better; "
        "*excl.* drops prompts that already contain the answer), **refusal** = empty "
        "(stopped at EOS) or \"I don't know\". Neighbourhood prompts (other subjects, same "
        "relation and true answer): **kept** = model still says the answer (higher is better), "
        "**refusal** = over-refusal (lower is better). **routed** = router fired.\n",
        "## Forget facts\n",
        "| model | train canonical leak / refusal | train development leak / refusal "
        "| MCF rewrite leak / refusal | **MCF paraphrase (held out) leak / excl. / refusal / routed to own fact** |",
        "|---|---|---|---|---|",
    ]
    for stem, data in results:
        s = data["summary"]
        tv = _train_view_split(train_views[stem]) if stem in train_views else {}
        canon = tv.get("canonical")
        dev = tv.get("development")
        para = s["paraphrase"]
        lines.append("| " + " | ".join([
            data["label"],
            f"{_pct(canon[0])} / {_pct(canon[1])}" if canon else "–",
            f"{_pct(dev[0])} / {_pct(dev[1])}" if dev else "–",
            f"{_pct(s['rewrite']['answer_in_continuation'])} / {_pct(s['rewrite']['refusal'])}",
            f"{_pct(para['answer_in_continuation'])} / {_pct(para['answer_in_continuation_excl'])}"
            f" / {_pct(para['refusal'])} / {_route(data, para, 'routed_to_own_fact')}",
        ]) + " |")
    lines += ["", "## Over-refusal (held out)\n",
              "| model | neighbourhood: answer kept ↑ | neighbourhood: refusal ↓ | neighbourhood: routed ↓ "
              "| same subject, other topics: refusal ↓ | same subject: routed | same subject: answer appears |",
              "|---|---|---|---|---|---|---|"]
    for stem, data in results:
        s = data["summary"]
        nb, gen = s["neighborhood"], s["generation"]
        lines.append("| " + " | ".join([
            data["label"], _pct(nb["answer_in_continuation"]), _pct(nb["refusal"]),
            _route(data, nb), _pct(gen["refusal"]), _route(data, gen),
            _pct(gen["answer_in_continuation"])]) + " |")

    by_label = {d["label"]: {r["id"]: r for r in d["examples"]} for _, d in results}
    base = next((v for k, v in by_label.items() if k.startswith("base")), None)
    if base:
        def block(title, ids):
            text = [f"\n## {title}\n"]
            for i in ids:
                b = base[i]
                text.append(f"**{b['prompt'].strip()}** → *{b['answer']}*\n")
                for label, rows in by_label.items():
                    r = rows[i]
                    tag = ("LEAK" if r["answer_in_continuation"] else "") if b["set"] != "neighborhood" \
                        else ("kept" if r["answer_in_continuation"] else "")
                    tag = "refuse" if r["refusal"] else (tag or "–")
                    route = " (routed)" if r["routed"] else ""
                    shown = r["continuation"].strip().replace("\n", " ") or "∅ (stopped: EOS)"
                    text.append(f"- {label} [{tag}{route}]: `{shown}`")
                text.append("")
            return text

        para = [i for i, r in base.items() if r["set"] == "paraphrase"
                and r["answer_in_continuation"] and not r["answer_in_prompt"]]
        seen, picked = set(), []
        for i in para:
            if base[i]["case_id"] not in seen:
                picked.append(i)
                seen.add(base[i]["case_id"])
        lines += block("Held-out paraphrases where base states the answer", picked[:args.show])
        over = [i for i, r in base.items() if r["set"] in ("neighborhood", "generation")
                and any(rows[i]["refusal"] for lbl, rows in by_label.items()
                        if not lbl.startswith("base"))]
        lines += block("Over-refusal cases (held-out neighbourhood / same-subject prompts "
                       "that some method refused)", over[:args.show])
    text = "\n".join(lines) + "\n"
    (root / "summary.md").write_text(text)
    print(text)


def main(argv=None):
    from zerounlearn_generation_leak import _defaults

    d = _defaults()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--method", choices=("base", "router", "zerounlearn"), required=True)
    r.add_argument("--run-dir", default=None)
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--model-path", default=d["model"])
    r.add_argument("--mcf-path", default=d["mcf"])
    r.add_argument("--zero-unlearn-root", default=d["zero_root"])
    r.add_argument("--hparams-path", default=d["hparams"])
    r.add_argument("--max-new-tokens", type=int, default=24)
    r.add_argument("--limit", type=int, default=0, help="only the first N facts (smoke tests)")
    r.add_argument("--device", default="cuda")
    r.add_argument("--local-files-only", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--root", required=True)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--train-views", nargs="*", default=[],
                   help="STEM=routed_generation.json (training-view leak split canonical/dev)")
    s.add_argument("--order", nargs="*", default=["base", "delta_e", "R_eos", "R_idk", "zerounlearn"])
    s.add_argument("--show", type=int, default=6)
    args = p.parse_args(argv)
    if args.command == "run":
        if args.method == "router" and not args.run_dir:
            p.error("--method router needs --run-dir")
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
