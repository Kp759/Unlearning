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
    "paper router": "outputs/router_diag_ours_v1",
    "crop: sentence": "outputs/router_diag_ours_crop_sentence_v1",
    "crop: sentence + subject": "outputs/router_diag_ours_crop_subject_v1",
    "floor (no judge)": "outputs/router_diag_ours_floor_v1",
    "floor + crop: sentence": "outputs/router_diag_ours_floor_crop_v1",
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
          "| dataset | variant | missed | retain fires | no-association rate | recall | F1 | crop decisions | parity |",
          "|---|---|---|---|---|---|---|---|---|"]
    groups = []
    for ds in ("mcf", "zsre", "mquake"):
        for label, root in VARIANTS.items():
            m, manifest = _load(root, ds, a.seed)
            if m is None:
                continue
            al = m["eval"]["by_split"]["all"]
            null = ((m.get("null") or {}).get("deployment_fpr") or {}).get("rate")
            crop = (manifest.get("sections") or {}).get("eval", {}).get("crop_fired")
            parity = sum(v.get("route_mismatches", 0) for v in (manifest.get("hook_parity") or {}).values())
            md.append(f"| {NAMES[ds]} | {label} | {_forget_misses(m)} | {_retain_fires(m)} | {_f(null)} | "
                      f"{_f(al['recall'])} | {_f(al['f1'])} | {'–' if crop is None else crop} | {parity} |")
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
