#!/usr/bin/env python3
"""How often do two or more router heads qualify on one prompt, and what happens then?

    python scripts/count_route_collisions.py --local-files-only --run-dirs \
        'outputs/mcf_multiseed_regular_v1/seed*/L??/linear_global' \
        'outputs/zsre_multiseed_reworded_v2/seed*/L??/linear_global' \
        'outputs/mquake_multiseed_regular_v1/seed*/L??/linear_global'
    python scripts/count_route_collisions.py --summarize --run-dirs ...   # table only

Runtime rule (linear_router.decide_routes): a head qualifies when its fact's
subject tokens are in the prompt and its logit >= the cutoff. With two or more
qualifying heads the highest logit wins, unless the top two are within
`ambiguity_margin` (0.5 logits by default): then NO row is applied.

Every official evaluation request (MCF rewrite/paraphrase/neighborhood/retain,
ZsRE rewrite/paraphrase/neighborhood/retain, MQuAKE rewrite/atomic_gen/retain)
is run through the saved bank at its request boundary. The hook's own subject
mask and logits are captured, the decision is recomputed with decide_routes and
checked against the hook's routes (parity must be 0 mismatches).

Per run, <run>/route_collisions.json. Forget prompts with >= 2 qualifying heads:
  resolved_correct   top-1 fired and it is the prompt's own fact
  resolved_wrong     top-1 fired but it is ANOTHER fact's row
  ambiguous_leak     rejected as ambiguous, own fact in the top two:
                     the margin itself made this fact leak
  ambiguous_other    rejected as ambiguous, own fact not in the top two
Must-not-route prompts (neighborhood, retain) with >= 2 qualifying heads:
  negative_ambiguous_blocked   the margin prevented a false fire
  negative_fired               a row fired anyway
resolved_wrong is split by whether the applied row forgets the SAME answer as the
prompt's own fact (wrong_same_answer: duplicate records, harmless) or not
(wrong_diff_answer: a real misroute).
Counterfactual "margin 0" (always take top-1): forget prompts recovered and
must-not-route prompts that would newly fire.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

from linear_router import decide_routes, load_router_artifact

OUT_NAME = "route_collisions.json"
FORGET_CATS = ("resolved_correct", "resolved_wrong", "ambiguous_leak", "ambiguous_other")


def _run_dirs(patterns):
    dirs = []
    for pattern in patterns:
        dirs.extend(Path(m).resolve() for m in (sorted(glob.glob(pattern)) or [pattern]))
    return [d for d in dirs if (d / "fact_association_embeddings.pt").exists()]


def manifest_seed(manifest):
    """Sample seed: top-level `seed` (ZsRE/MQuAKE/RWKU) or sampling/plan seed (MCF sweep)."""
    for value in (manifest.get("seed"), (manifest.get("sampling") or {}).get("seed"),
                  (manifest.get("plan") or {}).get("seed")):
        if value is not None:
            return int(value)
    raise ValueError("run manifest records no sample seed")


def dataset_of(artifact, manifest, run_dir):
    for v in (artifact.get("dataset"), manifest.get("dataset"), manifest.get("benchmark")):
        if v:
            v = str(v).lower()
            for name in ("mquake", "zsre", "rwku", "mcf", "counterfact"):
                if name in v:
                    return "mcf" if name == "counterfact" else name
    path = str(run_dir).lower()
    for name in ("mquake", "zsre", "rwku", "mcf"):
        if name in path:
            return name
    raise ValueError(f"cannot tell the dataset of {run_dir}")


# ---------------------------------------------------------------------------
# Official request sets: (group, prompt, owner row or None, should_route)
# ---------------------------------------------------------------------------

def requests_mcf(args, tok, bank, seed):
    from evaluate_router_decomposition import build_records_from_mcf

    facts, rows = build_records_from_mcf(args.mcf_path, 50, 1000, seed)
    row_of = {str(f["id"]): i for i, f in enumerate(bank.facts)}
    if [str(f["id"]) for f in facts] != [str(f["id"]) for f in bank.facts]:
        raise ValueError("MCF forget sample differs from the bank's facts (seed?)")
    return [{"group": r["group"], "prompt": r["prompt"],
             "owner": row_of[str(r["fact_id"])] if r["should_route"] else None,
             "should_route": bool(r["should_route"])} for r in rows]


def requests_zsre(args, tok, bank, seed, model):
    import zsre_zero_unlearn_official_eval as zsre

    forget, retain = zsre.load_official_eval_records(
        Path(args.zsre_path), tok, forget_num=50, retain_num=1000, seed=seed)
    row_of = {int(f["case_id"]): i for i, f in enumerate(bank.facts)}
    if sorted(row_of) != sorted(int(r["case_id"]) for r in forget):
        raise ValueError("ZsRE forget sample differs from the bank's facts (seed?)")
    llama_like = zsre.is_llama_like(model, tok)
    out = []
    for rec in forget:
        own = row_of[int(rec["case_id"])]
        rr = rec["requested_rewrite"]
        out.append({"group": "rewrite", "prompt": str(rr["prompt"]).format(str(rr["subject"])),
                    "owner": own, "should_route": True})
        out += [{"group": "paraphrase", "prompt": str(p), "owner": own, "should_route": True}
                for p in rec["paraphrase_prompts"]]
    neigh = sorted({c.prompt for rec in forget for c in zsre.expand_prediction_cases(
        rec, tok, llama_like=llama_like, prompt_types=("neighborhood",))})
    out += [{"group": "neighborhood", "prompt": p, "owner": None, "should_route": False}
            for p in neigh]
    for rec in retain:
        rr = rec["requested_rewrite"]
        out += [{"group": "retain", "prompt": p, "owner": None, "should_route": False}
                for p in [str(rr["prompt"]).format(str(rr["subject"]))]
                + [str(x) for x in rec["paraphrase_prompts"]]]
    return out


def requests_mquake(args, tok, bank, seed, artifact):
    import mquake_zero_unlearn_official_eval as mquake

    forget, retain = mquake.load_official_eval_records(
        Path(args.mquake_path), tok, forget_num=50, retain_num=1000, seed=seed)
    case_map = {str(k): str(v) for k, v in artifact.get("atomic_case_to_association_id", {}).items()}
    if not case_map:
        raise ValueError("artifact has no atomic_case_to_association_id")
    row_of = {str(f["id"]): i for i, f in enumerate(bank.facts)}
    out = []
    for split, records in (("forget", forget), ("retain", retain)):
        for rec in records:
            rr = rec["requested_rewrite"]
            own = row_of[case_map[str(rec["case_id"])]] if split == "forget" else None
            for group, prompt in (("rewrite", str(rr["prompt"]).format(str(rr["subject"]))),
                                  ("atomic_gen", str(rec["atomic_gen_prompt"]))):
                out.append({"group": group if split == "forget" else f"retain_{group}",
                            "prompt": prompt, "owner": own, "should_route": split == "forget"})
    return out


# ---------------------------------------------------------------------------
# Classification (pure; unit-tested)
# ---------------------------------------------------------------------------

def classify(logits, eligible, owners, should_route, threshold, margin):
    """Per prompt: qualifying count, decision, collision category. logits/eligible [P, N]."""
    eligible = eligible.bool()
    d = decide_routes(logits, eligible, threshold, margin)
    d0 = decide_routes(logits, eligible, threshold, 0.0)
    thr = torch.as_tensor(threshold, dtype=logits.dtype).expand_as(logits) \
        if torch.as_tensor(threshold).ndim else torch.full_like(logits, float(threshold))
    qualifies = eligible & (logits >= thr)
    ranked = logits.masked_fill(~qualifies, float("-inf"))
    k = min(2, logits.shape[1])
    top_idx = ranked.topk(k=k, dim=-1).indices
    rows = []
    for i in range(logits.shape[0]):
        n_q = int(qualifies[i].sum())
        own = owners[i]
        active, fact = bool(d["active"][i]), int(d["fact"][i])
        top2 = [int(j) for j in top_idx[i][:min(n_q, 2)]]
        r = {"n_eligible": int(eligible[i].sum()), "n_qualifying": n_q,
             "active": active, "chosen": fact if active else None,
             "ambiguous": bool(d["ambiguous"][i]), "top2": top2,
             "separation": float(d["separation"][i]) if n_q >= 2 else None,
             "active_margin0": bool(d0["active"][i]),
             "chosen_margin0": int(d0["fact"][i]) if bool(d0["active"][i]) else None,
             "category": None}
        if n_q >= 2:
            if should_route[i]:
                if active:
                    r["category"] = "resolved_correct" if fact == own else "resolved_wrong"
                else:
                    r["category"] = "ambiguous_leak" if own in top2 else "ambiguous_other"
            else:
                r["category"] = "negative_fired" if active else "negative_ambiguous_blocked"
        rows.append(r)
    return rows


def summarize_rows(rows):
    forget = [r for r in rows if r["should_route"]]
    neg = [r for r in rows if not r["should_route"]]
    multi_f = [r for r in forget if r["n_qualifying"] >= 2]
    multi_n = [r for r in neg if r["n_qualifying"] >= 2]
    cats = Counter(r["category"] for r in rows if r["category"])
    by_group = defaultdict(Counter)
    for r in rows:
        by_group[r["group"]]["prompts"] += 1
        by_group[r["group"]]["multi_qualifying"] += r["n_qualifying"] >= 2
        if r["category"]:
            by_group[r["group"]][r["category"]] += 1
    recovered = sum(1 for r in forget if not r["active"] and r["active_margin0"]
                    and r["chosen_margin0"] == r["owner"])
    newly_wrong = sum(1 for r in forget if not r["active"] and r["active_margin0"]
                      and r["chosen_margin0"] != r["owner"])
    newly_fire = sum(1 for r in neg if not r["active"] and r["active_margin0"])
    return {
        "forget_prompts": len(forget), "must_not_route_prompts": len(neg),
        "forget_multi_qualifying": len(multi_f),
        **{c: cats.get(c, 0) for c in FORGET_CATS},
        "negative_multi_qualifying": len(multi_n),
        "negative_ambiguous_blocked": cats.get("negative_ambiguous_blocked", 0),
        "negative_fired_multi": cats.get("negative_fired", 0),
        "margin0_forget_recovered": recovered, "margin0_forget_wrong_row": newly_wrong,
        "margin0_negative_newly_fire": newly_fire,
        "forget_max_eligible": max((r["n_eligible"] for r in forget), default=0),
        "by_group": {g: dict(c) for g, c in sorted(by_group.items())},
    }


# ---------------------------------------------------------------------------
# Running the bank
# ---------------------------------------------------------------------------

class Spy:
    def __init__(self, bank):
        self._mask_fn, self._logit_fn = bank._subject_mask, bank.router_logits
        self.mask = self.logits = None
        bank._subject_mask, bank.router_logits = self._mask, self._logits

    def _mask(self, *a, **kw):
        out = self._mask_fn(*a, **kw)
        self.mask = out.detach().cpu()
        return out

    def _logits(self, query):
        out = self._logit_fn(query)
        self.logits = out.detach().float().cpu()
        return out


@torch.no_grad()
def route_all(model, bank, tok, prompts, device, batch_size):
    spy = Spy(bank)
    logits, masks, hook = [], [], []
    for s in range(0, len(prompts), batch_size):
        enc = tok(prompts[s:s + batch_size], padding=True, return_tensors="pt").to(device)
        model.set_association_prefix_lengths(enc["attention_mask"].sum(dim=1).tolist())
        model(**enc, use_cache=False)
        logits.append(spy.logits)
        masks.append(spy.mask)
        hook.extend(list(bank.last_active_fact_indices))
    return torch.cat(logits), torch.cat(masks).bool(), hook


def run_one(run_dir, base_model, tok, args):
    manifest = json.loads((run_dir / "association_manifest.json").read_text())
    artifact = torch.load(run_dir / "fact_association_embeddings.pt", map_location="cpu",
                          weights_only=False)
    ds = dataset_of(artifact, manifest, run_dir)
    seed = manifest_seed(manifest)
    model, bank = load_router_artifact(base_model, artifact)
    try:
        device = next(model.parameters()).device
        if ds == "mcf":
            reqs = requests_mcf(args, tok, bank, seed)
        elif ds == "zsre":
            reqs = requests_zsre(args, tok, bank, seed, model)
        elif ds == "mquake":
            reqs = requests_mquake(args, tok, bank, seed, artifact)
        else:
            raise NotImplementedError(f"{ds}: not supported yet (RWKU needs its own probe loader)")
        if args.max_retain_requests:
            keep, n = [], 0
            for r in reqs:
                if r["group"].startswith("retain"):
                    n += 1
                    if n > args.max_retain_requests:
                        continue
                keep.append(r)
            reqs = keep
        z, elig, hook = route_all(model, bank, tok, [r["prompt"] for r in reqs], device,
                                  args.batch_size)
        threshold = bank.active_threshold(torch.device("cpu"))
        margin = float(bank.ambiguity_margin)
        rows = classify(z, elig, [r["owner"] for r in reqs], [r["should_route"] for r in reqs],
                        threshold, margin)
        parity = sum(([r["chosen"]] if r["active"] else []) != h for r, h in zip(rows, hook))
        if parity:
            raise RuntimeError(f"{parity} prompts: recomputed decision != runtime hook")
        rows = [{**q, **r} for q, r in zip(reqs, rows)]
        result = {"run_dir": str(run_dir), "dataset": ds, "seed": seed,
                  "layer": int(artifact["layer"]), "ambiguity_margin": margin,
                  "threshold": (threshold.tolist() if isinstance(threshold, torch.Tensor)
                                else float(threshold)),
                  "gate_mode": getattr(bank, "gate_mode", None), "facts": len(bank.facts),
                  "hook_parity_mismatches": parity, "summary": summarize_rows(rows),
                  "fact_answers": fact_answers(bank.facts),
                  "collisions": [r for r in rows if r["n_qualifying"] >= 2]}
        (run_dir / OUT_NAME).write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        bank._hook_handle.remove()


def _norm(text):
    return " ".join(str(text).casefold().split())


def fact_answers(facts):
    """The answer each bank row forgets (normalised), in row order."""
    return [_norm(f.get("object", f.get("answer", ""))) for f in facts]


def split_wrong(result):
    """resolved_wrong -> same answer as the own fact (a duplicate: the applied row
    forgets the same answer) or a different answer (a real misroute)."""
    answers = result.get("fact_answers")
    if answers is None:  # results written before fact_answers was recorded
        art = torch.load(Path(result["run_dir"]) / "fact_association_embeddings.pt",
                         map_location="cpu", weights_only=False)
        answers = fact_answers(art["facts"])
    same = diff = 0
    for c in result["collisions"]:
        if c["category"] == "resolved_wrong":
            if answers[c["chosen"]] and answers[c["chosen"]] == answers[c["owner"]]:
                same += 1
            else:
                diff += 1
    return {"wrong_same_answer": same, "wrong_diff_answer": diff}


COLS = ("forget_prompts", "forget_multi_qualifying", "resolved_correct", "resolved_wrong",
        "wrong_same_answer", "wrong_diff_answer", "ambiguous_leak", "ambiguous_other", "negative_multi_qualifying",
        "negative_ambiguous_blocked", "margin0_forget_recovered", "margin0_negative_newly_fire")


def print_table(run_dirs):
    results = [json.loads((d / OUT_NAME).read_text()) for d in run_dirs if (d / OUT_NAME).exists()]
    if not results:
        print("no route_collisions.json yet")
        return
    print("| dataset | seed | layer | margin | " + " | ".join(COLS) + " |")
    print("|" + "---|" * (len(COLS) + 4))
    totals = defaultdict(Counter)
    for r in sorted(results, key=lambda x: (x["dataset"], x["layer"], x["seed"])):
        s = {**r["summary"], **split_wrong(r)}
        print(f"| {r['dataset']} | {r['seed']} | {r['layer']} | {r['ambiguity_margin']:g} | "
              + " | ".join(str(s[c]) for c in COLS) + " |")
        for c in COLS:
            totals[r["dataset"]][c] += s[c]
        totals[r["dataset"]]["runs"] += 1
    print("\nTotals over runs:")
    for ds, t in totals.items():
        print(f"  {ds} ({t['runs']} runs): " + ", ".join(f"{c}={t[c]}" for c in COLS))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dirs", nargs="+", required=True, help="dirs or glob patterns")
    p.add_argument("--mcf-path", default="data/multi_counterfact.json")
    p.add_argument("--zsre-path", default="data/zsre_mend_eval.json")
    p.add_argument("--mquake-path", default="data/MQuAKE-CF-3k-v2.json")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-retain-requests", type=int, default=0)
    p.add_argument("--local-files-only", action="store_true")
    a = p.parse_args(argv)

    run_dirs = _run_dirs(a.run_dirs)
    if a.summarize:
        print_table(run_dirs)
        return 0
    todo = [d for d in run_dirs if a.overwrite or not (d / OUT_NAME).exists()]
    print(f"{len(run_dirs)} run dirs, {len(todo)} to evaluate", flush=True)
    if todo:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = json.loads((todo[0] / "association_manifest.json").read_text())["model_path"]
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True,
                                            local_files_only=a.local_files_only)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        base = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=getattr(torch, a.dtype), local_files_only=a.local_files_only,
            attn_implementation="eager").to(a.device).eval()
        base.requires_grad_(False)
        failed = []
        for d in todo:
            try:
                s = run_one(d, base, tok, a)["summary"]
                print(f"{d}: multi={s['forget_multi_qualifying']} wrong={s['resolved_wrong']} "
                      f"leak={s['ambiguous_leak']} neg_blocked={s['negative_ambiguous_blocked']}",
                      flush=True)
            except Exception as exc:  # keep going; report at the end
                print(f"FAILED {d}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                failed.append(str(d))
        if failed:
            print(f"{len(failed)} failed: {failed}", file=sys.stderr)
    print_table(run_dirs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
