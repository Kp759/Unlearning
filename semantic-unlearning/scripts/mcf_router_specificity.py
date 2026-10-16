#!/usr/bin/env python3
"""Router-only recall / specificity on MCF held-out prompts (no rows, no generation).

The routing decision reads the base layer-19 state at the prompt boundary, so it does
not depend on the trained rows: any router (or trained run) can be compared here in
minutes. For each NAME=DIR (router dir or trained run; rows are zeroed, early write
ignored):

  paraphrase / rewrite   official held-out prompts of the forgotten fact
                         -> recall = routed to the fact's own row
  generation             relation-related generation prompts -> fire rate (info)
  neighborhood           other subjects, same relation -> false fire
  subject: other rel.    the forgotten subject asked with OTHER relations
                         (mcf_subject_collateral 'eval' half: never a router negative)
                         -> false fire, split by answer group (born in vs works in)
  subject: preserve rel. the 'preserve' half (router negatives with --relation-negatives)

    python -u scripts/mcf_router_specificity.py \\
        --runs old=outputs/mcf_multiseed_regular_v1/seed1/L19/router \\
               relneg=outputs/mcf_router_relneg_seed1/router \\
        --out outputs/mcf_router_relneg_seed1/specificity.md --local-files-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


@torch.no_grad()
def route(lm, bank, tok, prompt, device):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    lm.set_association_prefix_lengths([ids.shape[1]])
    lm(input_ids=ids, use_cache=False)
    active = bank.last_active_fact_indices[0] if bank.last_active_fact_indices else []
    return int(active[0]) if active else -1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="NAME=DIR")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from linear_router import load_linear_classifier_artifact, same_answer_group
    from mcf_heldout_generation import _fact_case, heldout_prompts
    from mcf_subject_collateral import relation_templates, subject_probes

    runs = [s.split("=", 1) for s in args.runs]
    manifest = json.loads((Path(runs[0][1]) / "association_manifest.json").read_text())
    seed = int((manifest.get("sampling") or {}).get("seed", 1))
    mcf_path = manifest["mcf_path"]
    tok = AutoTokenizer.from_pretrained(manifest["model_path"], use_fast=True,
                                        local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_path"], dtype=torch.bfloat16, local_files_only=args.local_files_only,
        attn_implementation="eager").to(args.device).eval()
    model.requires_grad_(False)

    forget_records, prompts = heldout_prompts(mcf_path, seed)
    from factorized_router import robustness_probes

    robust = robustness_probes(json.loads(Path(mcf_path).read_text()), forget_records, prompts)
    templates = relation_templates(mcf_path)
    probes = []
    for split in ("eval", "preserve"):
        for record, plist in zip(forget_records, subject_probes(forget_records, templates, split)):
            rw = record["requested_rewrite"]
            rw = rw[0] if isinstance(rw, list) else rw
            for rel, text in plist:
                probes.append({"case_id": int(record["case_id"]), "prompt": text, "relation": rel,
                               "set": f"subject_{split}",
                               "same_group": same_answer_group(rw["relation_id"], rel)})

    results = {}
    for name, run_dir in runs:
        artifact = torch.load(Path(run_dir) / "fact_association_embeddings.pt", map_location="cpu",
                              weights_only=False)
        artifact = dict(artifact)
        artifact.pop("early_write", None)
        artifact["rows"] = torch.zeros_like(artifact["rows"])
        artifact["write_mode"], artifact["edit_mode"] = "last", "add"
        if artifact.get("factorized_router") is not None:
            from linear_router import load_router_artifact

            lm, bank = load_router_artifact(model, artifact)
        else:
            lm, bank = load_linear_classifier_artifact(model, artifact)
        cases = [_fact_case(f) for f in artifact["facts"]]
        rows = []
        try:
            for item in prompts + probes + robust:
                f = route(lm, bank, tok, item["prompt"], args.device)
                rows.append({**{k: item[k] for k in ("case_id", "set", "prompt")},
                             "same_group": item.get("same_group"), "relation": item.get("relation"),
                             "fired": f >= 0, "own": f >= 0 and cases[f] == item["case_id"]})
        finally:
            bank.close()

        def rate(sel, key):
            part = [r for r in rows if sel(r)]
            return (sum(r[key] for r in part) / len(part) if part else None), len(part)

        s = {
            "paraphrase_recall": rate(lambda r: r["set"] == "paraphrase", "own"),
            "rewrite_recall": rate(lambda r: r["set"] == "rewrite", "own"),
            "generation_fire": rate(lambda r: r["set"] == "generation", "fired"),
            "neighborhood_fire": rate(lambda r: r["set"] == "neighborhood", "fired"),
            "subject_eval_fire": rate(lambda r: r["set"] == "subject_eval", "fired"),
            "subject_eval_fire_same_group": rate(
                lambda r: r["set"] == "subject_eval" and r["same_group"], "fired"),
            "subject_eval_fire_other": rate(
                lambda r: r["set"] == "subject_eval" and not r["same_group"], "fired"),
            "subject_preserve_fire": rate(lambda r: r["set"] == "subject_preserve", "fired"),
            "typo_recall": rate(lambda r: r["set"] == "paraphrase_typo", "own"),
            "lowercase_recall": rate(lambda r: r["set"] == "paraphrase_lower", "own"),
            "near_miss_fire": rate(lambda r: r["set"] == "near_miss", "fired"),
        }
        results[name] = {"run_dir": str(run_dir), "summary": s, "rows": rows}
        print(name, json.dumps({k: v[0] for k, v in s.items()}), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n")
    write_table(results, out, seed)


HEAD = {"paraphrase_recall": "paraphrase recall ↑", "rewrite_recall": "rewrite recall ↑",
        "generation_fire": "generation fire", "neighborhood_fire": "neighbourhood fire ↓",
        "subject_eval_fire": "same subj, other rel. (held out) fire ↓",
        "subject_eval_fire_same_group": "… same answer group",
        "subject_eval_fire_other": "… other relations",
        "subject_preserve_fire": "same subj, preserve rel. fire (router negs) ↓",
        "typo_recall": "paraphrase w/ subject typo: recall ↑",
        "lowercase_recall": "paraphrase w/ lower-case subject: recall ↑",
        "near_miss_fire": "near-miss names (held out) fire ↓"}


def write_table(results, out, seed=1, examples=True):
    cols = [c for c in HEAD if any(c in r["summary"] for r in results.values())]
    lines = [f"# Router recall / specificity (MCF seed {seed}, routing only)\n",
             "| router | " + " | ".join(HEAD[c] for c in cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for name, res in results.items():
        cells = [f"{100 * v:.1f}% (n={n})" if v is not None else "–"
                 for v, n in (res["summary"].get(c, (None, 0)) for c in cols)]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    names = list(results)
    if examples and len(names) > 1:
        lines.append("\n## Held-out same-subject prompts where routers disagree\n")
        a, b = results[names[0]]["rows"], results[names[-1]]["rows"]
        shown = 0
        for ra, rb in zip(a, b):
            if ra["set"] in ("subject_eval", "paraphrase") and ra["fired"] != rb["fired"] and shown < 20:
                lines.append(f"- [{ra['set']}{', same group' if ra['same_group'] else ''}] "
                             f"`{ra['prompt']}`: {names[0]} {'fires' if ra['fired'] else 'silent'}, "
                             f"{names[-1]} {'fires' if rb['fired'] else 'silent'}")
                shown += 1
    text = "\n".join(lines) + "\n"
    Path(out).write_text(text)
    print(text)


def merge(argv=None):
    """python scripts/mcf_router_specificity.py merge OUT.md A.json B.json ...
    One table from several runs' specificity.json files (keys must be unique)."""
    argv = list(argv)
    out, files = argv[0], argv[1:]
    results = {}
    for f in files:
        for name, res in json.loads(Path(f).read_text()).items():
            results[name] = res
    write_table(results, out, examples=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        merge(sys.argv[2:])
    else:
        main()
