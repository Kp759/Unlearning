#!/usr/bin/env python3
"""The paper's router against each router fix (router_fix.slurm), seed 1, from the diagnostics.

    python scripts/router_fix_compare.py        # every variant that has finished

Per dataset and variant: missed forget prompts, false fires on retain prompts and on prompts
with no association (the deployment false-positive rate), recall and F1, how many decisions the
subject-sentence crop made, and the hook-parity mismatches (must be 0). Then misses and false
fires per probe group. Writes outputs/ROUTER_FIX_COMPARISON.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router_diag_explain import NAMES  # noqa: E402

VARIANTS = {
    "paper router (diagnosed earlier)": "outputs/router_diag_ours_v1",
    "control: router today, crop off": "outputs/router_diag_ours_control_v1",
    "crop: sentence": "outputs/router_diag_ours_crop_sentence_v1",
    "crop: sentence + subject": "outputs/router_diag_ours_crop_subject_v1",
    "floor (no judge)": "outputs/router_diag_ours_floor_v1",
    "floor + crop: sentence": "outputs/router_diag_ours_floor_crop_v1",
    "relation negatives (20)": "outputs/router_diag_ours_relneg_v1",
    "relation negatives (40)": "outputs/router_diag_ours_relneg40_v1",
    "rejected: distractors + judged floor": "outputs/router_diag_ours_fix_v1",
}


def _load(root, ds, seed):
    base = Path(root) / ds / f"seed{seed}"
    p = base / "report" / "metrics.json"
    if not p.is_file():
        return None, None
    manifest = base / "manifest.json"
    return json.loads(p.read_text()), (json.loads(manifest.read_text()) if manifest.is_file() else {})


def _f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def _forget_misses(m):
    return sum(v.get("fn", 0) for g, v in (m["eval"].get("by_group") or {}).items() if g.startswith("forget"))


def _retain_fires(m):
    return sum(v.get("fp", 0) for g, v in (m["eval"].get("by_group") or {}).items() if not g.startswith("forget"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="outputs/ROUTER_FIX_COMPARISON.md")
    a = ap.parse_args(argv)
    md = ["# Router fixes against the paper's router (seed 1, Llama-3.2-3B, block 19)", "",
          "Missed: forget prompts (request prompts and held-out phrasings) on which the right head did "
          "not fire. Retain fires: retain / neighbourhood prompts that fired. No-association rate: share "
          "of must-not-fire prompts that fired (deployment false-positive rate). Crop decisions: "
          "prompts decided by the subject-sentence view. Parity: mismatches between the offline "
          "decision and the deployed routing pass (must be 0).", "",
          "Router: first 8 hex digits of the router file's sha256 and its modification time; rows "
          "compared with each other must share the router, except the floor variants (a refit).", "",
          "Same subject, other relation: the protected subject asked about a relation outside the "
          "fact's answer group (must not fire). Same answer type: a relation in the fact's answer group "
          "(the router treats it as the same fact; not counted as a false fire).", "",
          "| dataset | variant | router | missed | retain fires | no-association rate | "
          "same subject, other relation | same answer type | recall | F1 | crop decisions (eval / null) | parity |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    groups = []
    for ds in ("mcf", "zsre", "mquake"):
        for label, root in VARIANTS.items():
            m, manifest = _load(root, ds, a.seed)
            if m is None:
                continue
            al = m["eval"]["by_split"]["all"]
            null = ((m.get("null") or {}).get("deployment_fpr") or {}).get("rate")
            same = ((m.get("null") or {}).get("by_group") or {}).get("null_protected_same_answer_type")
            other = ((m.get("null") or {}).get("by_group") or {}).get("null_protected_other_relation")
            sec = manifest.get("sections") or {}
            crops = [sec.get(s, {}).get("crop_fired") for s in ("eval", "null")]
            crop = "–" if crops[0] is None else " / ".join("–" if c is None else str(c) for c in crops)
            parity = sum(v.get("route_mismatches", 0) for v in (manifest.get("hook_parity") or {}).values())
            router = (f"{manifest['router_sha256'][:8]} {manifest.get('router_mtime', '')}"
                      if manifest.get("router_sha256") else "not recorded")
            rel = "–" if other is None else f"{other['fired']}/{other['n']}"
            md.append(f"| {NAMES[ds]} | {label} | {router} | {_forget_misses(m)} | {_retain_fires(m)} | "
                      f"{_f(null)} | {rel} | {'–' if same is None else str(same['fired']) + '/' + str(same['n'])} | "
                      f"{_f(al['recall'])} | {_f(al['f1'])} | {crop} | {parity} |")
            for g, v in (m["eval"].get("by_group") or {}).items():
                if v.get("fn") or v.get("fp"):
                    groups.append((ds, g, label, v["n"], v["fn"], v["fp"]))
    md += ["", "## Misses and false fires per probe group", "",
           "| dataset | group | variant | n | missed | false fires |", "|---|---|---|---|---|---|"]
    order = list(VARIANTS)
    for ds, g, label, n, fn, fp in sorted(groups, key=lambda x: (x[0], x[1], order.index(x[2]))):
        md.append(f"| {NAMES[ds]} | {g} | {label} | {n} | {fn} | {fp} |")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
