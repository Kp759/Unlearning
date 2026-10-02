#!/usr/bin/env python3
"""Is a forgotten MQuAKE answer still recoverable through an alias?

    python -u scripts/evaluate_mquake_alias_leak.py \
        --run-dir outputs/mquake_multiseed_regular_v1/seed1/L19/linear_global \
        --mquake-path data/MQuAKE-CF-3k-v2.json --local-files-only

The official MQuAKE metrics score the exact original answer only. This scores,
on the same forget facts and prompts, the answer AND its Wikidata aliases
(mquake_answer_aliases.py), with the base model and with SURE (router + rows,
boundary fixed at the request as in the official evaluator):

    rewrite     the direct cloze (training-visible)
    atomic_gen  the held-out atomic question (AtomicGen)

Per target, teacher-forced: first-token probability, full score (exp mean
log-prob over the target's tokens) and greedy recovery (every token is the
argmax). Targets: answer | alias_same_first (same first token as the answer,
so suppressed whenever the answer's first token is) | alias_diff_first (the
real leak channel).

Headline, per prompt type: among (fact, prompt) pairs the base model answers
greedily and SURE does not, the fraction where SURE still greedily produces
some alias ("alias recovery"), plus the same restricted to prompts SURE routes
to the fact's own row (where the row, not the router, is responsible).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mquake_answer_aliases as aliases_mod  # noqa: E402

PROMPT_TYPES = ("rewrite", "atomic_gen")
KINDS = ("answer", "alias_same_first", "alias_diff_first")


def build_items(records, tok, table, *, llama_like, row_of_key=None):
    """One item per (fact, prompt type, target); duplicate records collapsed."""
    from mquake_fact_association_embeddings import association_key_from_record
    from mquake_zero_unlearn_official_eval import original_answer_token_ids

    items, seen, coverage = [], set(), {"facts": 0, "facts_with_aliases": 0}
    for record in records:
        key = association_key_from_record(record)
        if key in seen:
            continue
        seen.add(key)
        coverage["facts"] += 1
        rr = record["requested_rewrite"]
        answer = str(rr["target_true"]["str"])
        alias_list = aliases_mod.record_aliases(record, table)
        if not alias_list:
            continue
        coverage["facts_with_aliases"] += 1
        targets = [(answer, original_answer_token_ids(tok, answer, llama_like=llama_like), "answer")]
        targets += aliases_mod.classify_aliases(tok, answer, alias_list, llama_like=llama_like)
        prompts = {"rewrite": str(rr["prompt"]).format(str(rr["subject"])),
                   "atomic_gen": str(record["atomic_gen_prompt"])}
        for prompt_type in PROMPT_TYPES:
            for text, ids, kind in targets:
                items.append({"fact": key, "prompt_type": prompt_type, "prompt": prompts[prompt_type],
                              "target": text, "target_ids": [int(t) for t in ids], "kind": kind,
                              "own_row": None if row_of_key is None else row_of_key.get(key)})
    return items, coverage


@torch.no_grad()
def score_items(model, tok, items, device, *, bank=None, batch_size=16):
    from torch.nn import functional as F
    from mquake_zero_unlearn_official_eval import _flat_ids

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = []
    for start in range(0, len(items), int(batch_size)):
        batch = items[start:start + int(batch_size)]
        prompt_ids = [_flat_ids(tok, it["prompt"]) for it in batch]
        seqs = [p + it["target_ids"] for p, it in zip(prompt_ids, batch)]
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(batch), width), int(pad), dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for r, s in enumerate(seqs):
            input_ids[r, :len(s)] = torch.tensor(s)
            attention[r, :len(s)] = 1
        if bank is not None:
            model.set_association_prefix_lengths([len(p) for p in prompt_ids])
        logits = model(input_ids=input_ids.to(device), attention_mask=attention.to(device),
                       use_cache=False).logits.float()
        routes = list(bank.last_active_fact_indices) if bank is not None else [None] * len(batch)
        for r, (it, p) in enumerate(zip(batch, prompt_ids)):
            pos, n = len(p), len(it["target_ids"])
            step = logits[r, pos - 1:pos - 1 + n]
            target = torch.tensor(it["target_ids"], device=step.device)
            logp = F.log_softmax(step, dim=-1).gather(-1, target[:, None])[:, 0]
            route = routes[r]
            rows.append({
                **{k: it[k] for k in ("fact", "prompt_type", "target", "kind")},
                "first_token_probability": float(logp[0].exp()),
                "score": float(logp.mean().exp()),
                "greedy": bool((step.argmax(-1) == target).all()),
                "route_active": None if route is None else bool(route),
                "routed_to_own_row": (None if route is None or it["own_row"] is None
                                      else list(route) == [int(it["own_row"])]),
            })
    return rows


def _mean(values):
    values = [v for v in values if v is not None]
    return None if not values else sum(values) / len(values)


def summarize(base_rows, sure_rows):
    """Per prompt type: target-kind means (base vs SURE) and alias recovery."""
    out = {}
    for prompt_type in PROMPT_TYPES:
        pairs = [(b, s) for b, s in zip(base_rows, sure_rows) if s["prompt_type"] == prompt_type]
        kinds = {}
        for kind in KINDS:
            cell = [(b, s) for b, s in pairs if s["kind"] == kind]
            if not cell:
                continue
            kinds[kind] = {
                "targets": len(cell),
                "first_token_probability": {"base": _mean([b["first_token_probability"] for b, _ in cell]),
                                            "sure": _mean([s["first_token_probability"] for _, s in cell])},
                "score": {"base": _mean([b["score"] for b, _ in cell]),
                          "sure": _mean([s["score"] for _, s in cell])},
                "greedy": {"base": _mean([float(b["greedy"]) for b, _ in cell]),
                           "sure": _mean([float(s["greedy"]) for _, s in cell])},
            }
        by_fact = defaultdict(list)
        for b, s in pairs:
            by_fact[s["fact"]].append((b, s))
        forgotten, forgotten_own = [], []
        for fact, cell in by_fact.items():
            answer = [(b, s) for b, s in cell if s["kind"] == "answer"]
            if not answer:
                continue
            b_ans, s_ans = answer[0]
            if not (b_ans["greedy"] and not s_ans["greedy"]):
                continue
            leaked = any(s["greedy"] for _, s in cell if s["kind"] != "answer")
            leaked_diff = any(s["greedy"] for _, s in cell if s["kind"] == "alias_diff_first")
            entry = {"fact": fact, "alias_recovered": leaked, "alias_diff_first_recovered": leaked_diff}
            forgotten.append(entry)
            if s_ans["routed_to_own_row"]:
                forgotten_own.append(entry)
        out[prompt_type] = {
            "facts": len(by_fact),
            "targets_by_kind": kinds,
            "answer_forgotten_facts": len(forgotten),
            "alias_recovery_rate": _mean([float(e["alias_recovered"]) for e in forgotten]),
            "alias_diff_first_recovery_rate": _mean([float(e["alias_diff_first_recovered"]) for e in forgotten]),
            "answer_forgotten_facts_own_row": len(forgotten_own),
            "alias_recovery_rate_own_row": _mean([float(e["alias_recovered"]) for e in forgotten_own]),
            "recovered_facts": [e["fact"] for e in forgotten if e["alias_recovered"]],
        }
    return out


def _fmt(value, digits=3):
    return "–" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def markdown(result):
    lines = [f"# MQuAKE alias leak — seed {result['seed']}, L{result['layer']}", "",
             f"Forget facts: {result['coverage']['facts']}, with aliases: "
             f"{result['coverage']['facts_with_aliases']}. Base = model without SURE.", ""]
    names = {"rewrite": "Direct cloze (training-visible)", "atomic_gen": "Held-out atomic question"}
    for prompt_type in PROMPT_TYPES:
        block = result["summary"].get(prompt_type)
        if not block:
            continue
        lines += [f"## {names[prompt_type]}", "",
                  "| target | n | first-token p base | first-token p SURE | score base | score SURE | greedy base | greedy SURE |",
                  "|---|---|---|---|---|---|---|---|"]
        for kind, c in block["targets_by_kind"].items():
            lines.append(f"| {kind} | {c['targets']} | {_fmt(c['first_token_probability']['base'])} | "
                         f"{_fmt(c['first_token_probability']['sure'])} | {_fmt(c['score']['base'])} | "
                         f"{_fmt(c['score']['sure'])} | {_fmt(c['greedy']['base'], 2)} | {_fmt(c['greedy']['sure'], 2)} |")
        lines += ["",
                  f"Answer forgotten (greedy under base, not under SURE): {block['answer_forgotten_facts']} facts; "
                  f"still recoverable through an alias: **{_fmt(block['alias_recovery_rate'], 3)}** "
                  f"(different-first-token alias: {_fmt(block['alias_diff_first_recovery_rate'], 3)}); "
                  f"routed to own row: {block['answer_forgotten_facts_own_row']} facts, "
                  f"alias recovery {_fmt(block['alias_recovery_rate_own_row'], 3)}.", ""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--mquake-path", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    import mquake_zero_unlearn_official_eval as mquake
    from mcf_zero_unlearn_official_eval import dtype_from_str
    from linear_router import load_router_artifact

    run_dir = Path(a.run_dir).resolve()
    manifest = json.loads((run_dir / "association_manifest.json").read_text())
    seed = int(a.seed if a.seed is not None else manifest.get("seed", 1))
    artifact = torch.load(run_dir / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    model_path = Path(manifest["model_path"]).resolve()
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=a.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    mquake_path = Path(a.mquake_path).resolve()
    forget_records, _ = mquake.load_official_eval_records(
        mquake_path, tok, forget_num=int(manifest.get("forget_num_instances", 50)),
        retain_num=int(manifest.get("retain_num_instances_final_evaluation", 1000)), seed=seed)
    table = aliases_mod.alias_table(aliases_mod.load_raw(mquake_path))
    row_of_key = {str(f["association_key"]): i for i, f in enumerate(artifact["facts"])}

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype_from_str(a.dtype), local_files_only=a.local_files_only,
        attn_implementation="eager").to(a.device).eval()
    base_model.requires_grad_(False)
    base_model.config.use_cache = False
    device = next(base_model.parameters()).device
    llama_like = mquake.is_llama_like(base_model, tok)
    items, coverage = build_items(forget_records, tok, table, llama_like=llama_like, row_of_key=row_of_key)
    base_rows = score_items(base_model, tok, items, device, batch_size=a.batch_size)
    model, bank = load_router_artifact(base_model, artifact)
    model.eval()
    sure_rows = score_items(model, tok, items, device, bank=bank, batch_size=a.batch_size)

    result = {
        "dataset": "MQuAKE-CF-3k-v2", "seed": seed, "layer": int(artifact["layer"]),
        "run_dir": str(run_dir), "model_path": str(model_path), "coverage": coverage,
        "row_training_alias_targets": bool(manifest.get("alias_targets", False)),
        "metric_definition": {
            "first_token_probability": "teacher-forced p(first target token | prompt)",
            "score": "exp(mean log-prob) over the target's tokens",
            "greedy": "every target token is the argmax (greedy decoding produces it)",
            "alias_recovery_rate": "facts the base model answers greedily and SURE does not, "
                                   "where SURE greedily produces some alias",
        },
        "summary": summarize(base_rows, sure_rows),
        "rows": {"base": base_rows, "sure": sure_rows},
    }
    out = Path(a.out).resolve() if a.out else run_dir / "alias_leak_eval.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    text = markdown(result)
    out.with_suffix(".md").write_text(text)
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
