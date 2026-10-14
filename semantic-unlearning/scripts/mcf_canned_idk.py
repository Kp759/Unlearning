#!/usr/bin/env python3
"""C_idk: the router alone + a fixed " I don't know." reply. No vectors, no model edit.

If the seed-1 linear router fires on a prompt, the reply is the fixed string; otherwise
the base model answers. Nothing here needs a GPU, because every input already exists:

  * routing does not depend on the rows (the classifier reads the raw layer-19 state
    before any edit), so the per-prompt route of ANY artifact built on this router is
    the route of the canned system; we take it from the held-out Δe run
    (outputs/mcf_heldout_generation_seed1/delta_e.json) and from the zero-row router
    run on our training views (α-sweep base/routed_generation.json)
  * unrouted prompts are exactly base, so their continuations / probabilities are
    the base ones (zero-row router = base, checked earlier: 250/250 identical)

Metrics (same definitions as the official evaluator, computed exactly for a deterministic
override; route = router decision on the prompt itself):
  Eff = 100 * mean_case P_base(answer | rewrite) * [not routed]          (P = 0 if routed)
  Gen = 100 * mean_case mean_para P_base(answer | paraphrase) * [not routed]
  Spe = 100 * mean_case mean_neigh correct_base * [not routed]    (refusing a neighbour = wrong)
  PPL = base PPL (the router never fires on the PPL text: every router arm has base PPL)

Outputs
  OUT/routed_generation.json                 training views (canonical + development)
  OUT/linear_global/official_mcf_eval.json   Eff/Gen/Spe/PPL (derived, see 'derivation')
  HELDOUT/C_idk.json                         held-out prompts, same schema as mcf_heldout_generation
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mcf_heldout_generation import is_refusal, summarize_rows
from zerounlearn_generation_leak import leak_summary

REPLY = "I don't know."


def canned_rows(rows, routes):
    out = []
    for r in rows:
        routed, own = routes(r)
        text = REPLY if routed else r["continuation"]
        answer = r["answer"].casefold()
        new = {k: v for k, v in r.items() if k not in ("base_nll_per_token", "repeated_trigram")}
        new.update({"continuation": text, "routed": routed,
                    "answer_in_continuation": answer in text.casefold(),
                    "answer_at_start": text.strip().casefold().startswith(answer),
                    "answer_in_prompt": answer in r["prompt"].casefold(),
                    "refusal": is_refusal(text), "repeated_trigram": False})
        if own is not None:
            new["routed_to_own_fact"] = own
        if not routed:
            for key in ("base_nll_per_token", "repeated_trigram"):
                if key in r:
                    new[key] = r[key]
        out.append(new)
    return out


def derived_metrics(base_eval, route_of):
    """Eff/Gen/Spe of the canned system from base per-prompt results + routes."""
    raw = base_eval["forget_raw"]
    out, check = {}, {}
    for label, group, kind in (("Eff", "rewrite", "prob"), ("Gen", "paraphrase", "prob"),
                               ("Spe", "neighborhood", "correct")):
        per_case, per_case_base, fired, total = [], [], 0, 0
        for index, record in enumerate(raw):
            post = record["post"]
            if kind == "prob":
                values = [math.exp(-float(row["target_true_nll_sum"]))
                          for row in post[f"{group}_prompts_probs"]]
            else:
                values = [float(bool(flag)) for flag in post[f"{group}_prompts_correct"]]
            keep = []
            for k, value in enumerate(values):
                routed = route_of(index, group, k)
                fired += routed
                total += 1
                keep.append(0.0 if routed else value)
            per_case.append(sum(keep) / len(keep))
            per_case_base.append(sum(values) / len(values))
        out[label] = 100.0 * sum(per_case) / len(per_case)
        check[label] = 100.0 * sum(per_case_base) / len(per_case_base)
        out[f"router_fire_rate_{group}"] = fired / max(1, total)
    return out, check


def main(argv=None):
    root = Path(__file__).resolve().parents[1] / "outputs"
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-views", default=str(root / "mcf_projection_alpha_sweep_seed1/base/routed_generation.json"))
    p.add_argument("--base-eval", default=str(root / "mcf_projection_alpha_sweep_seed1/base/linear_global/official_mcf_eval.json"))
    p.add_argument("--heldout-dir", default=str(root / "mcf_heldout_generation_seed1"))
    p.add_argument("--routes-from", default="delta_e.json",
                   help="held-out run of any artifact on the same router (routes are row-independent)")
    p.add_argument("--out-dir", default=str(root / "mcf_canned_idk_seed1"))
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    (out / "linear_global").mkdir(parents=True, exist_ok=True)
    heldout = Path(args.heldout_dir)

    # 1. Training views: zero-row router run carries base continuations + route flags.
    views = json.loads(Path(args.base_views).read_text())["examples"]
    view_rows = canned_rows(views, lambda r: (bool(r["routed"]), None))
    summary = {**leak_summary(view_rows), "model": "router + fixed IDK (no vectors)",
               "route_active_fraction": sum(r["routed"] for r in view_rows) / len(view_rows)}
    (out / "routed_generation.json").write_text(json.dumps(
        {"summary": summary, "examples": view_rows}, indent=2) + "\n")

    # 2. Held-out prompts: base continuations, routes from the Δe held-out run.
    base = json.loads((heldout / "base.json").read_text())
    routed_run = json.loads((heldout / args.routes_from).read_text())
    route = {r["id"]: (bool(r["routed"]), bool(r.get("routed_to_own_fact", False)))
             for r in routed_run["examples"]}
    if set(route) != {r["id"] for r in base["examples"]}:
        raise RuntimeError("held-out prompt sets differ between base and the routed run")
    held_rows = canned_rows(base["examples"], lambda r: route[r["id"]])
    (heldout / "C_idk.json").write_text(json.dumps({
        "label": "router + fixed IDK (no vectors)", "method": "canned",
        "run_dir": None, "seed": base.get("seed"), "routes_from": args.routes_from,
        "max_new_tokens": base.get("max_new_tokens"),
        "summary": summarize_rows(held_rows), "examples": held_rows}, indent=2) + "\n")

    # 3. Official-style metrics, derived exactly for the deterministic override.
    base_eval = json.loads(Path(args.base_eval).read_text())
    # forget_raw follows the sampled record order, as does the held-out prompt list.
    cases, subjects, by_key = [], {}, {}
    for r in routed_run["examples"]:
        if r["case_id"] not in subjects:
            cases.append(r["case_id"])
            subjects[r["case_id"]] = r["subject"]
        by_key[(r["case_id"], r["set"], int(r["id"].rsplit(":", 1)[1]))] = bool(r["routed"])
    raw_subjects = [(rec["requested_rewrite"][0] if isinstance(rec["requested_rewrite"], list)
                     else rec["requested_rewrite"])["subject"] for rec in base_eval["forget_raw"]]
    if raw_subjects != [subjects[c] for c in cases]:
        raise RuntimeError("official forget_raw order does not match the held-out records")
    metrics, check = derived_metrics(base_eval, lambda i, g, k: by_key[(cases[i], g, k)])
    reported = base_eval["forget"]
    for label in ("Eff", "Gen", "Spe"):
        if not math.isclose(check[label], float(reported[label]), rel_tol=1e-6, abs_tol=1e-9):
            raise RuntimeError(f"base {label} recomputation {check[label]} != reported {reported[label]}")
    result = {
        "method": "C_idk: router + fixed reply",
        "metric_version": "zerounlearn_answer_probability_v2",
        "forget": {"Eff": metrics["Eff"], "Gen": metrics["Gen"], "Spe": metrics["Spe"]},
        "forget_PPL": base_eval.get("forget_PPL"),
        "router_fire_rates": {k: v for k, v in metrics.items() if k.startswith("router_fire")},
        "base_recomputed": check,
        "derivation": ("routed prompts contribute P(answer)=0 / neighbour correct=0; unrouted "
                       "prompts contribute the base per-prompt value; routes are the router's "
                       "decision on the prompt alone (from " + args.routes_from + "); PPL = base"),
    }
    (out / "linear_global" / "official_mcf_eval.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"training_views": summary, "official": result,
                      "heldout": summarize_rows(held_rows)}, indent=2))


if __name__ == "__main__":
    main()
