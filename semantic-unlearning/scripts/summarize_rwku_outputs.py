#!/usr/bin/env python3
"""RWKU: what does the unlearned model say? Answers vs "I don't know", from the official eval.

    python scripts/summarize_rwku_outputs.py [--seeds 1 2 3 4 5] [--layer 19] [--examples 5]

The RWKU evaluator already generates (greedy, fixed request boundary) and saves
every output in official_rwku_eval.json under details.<split>[i].prediction, so
nothing is regenerated here. Per arm and split, summed over seeds:
  answer      the true answer appears in the output (RWKU's recovery; forget: lower is better)
  abstains    the output says "I don't know" / "unknown" / ...
  row fired   a bank row was injected on that prompt
Arms (included when the eval file exists):
  base        outputs/rwku_multiseed_base_v1/seed<S>/official_rwku_eval.json (unedited model)
  shipped     outputs/rwku_<REF>/seed<S>/L<LL>/linear_global
  joint       outputs/compressed_multiseed_v1/rwku/seed<S>/L<LL>/full
  joint_idk   outputs/compressed_multiseed_idk_v1/rwku/seed<S>/L<LL>/full
  joint_idk_eos          outputs/compressed_multiseed_idk_eos_v1/rwku/...  (+ end token)
  shipped_subject        outputs/rwku_multiseed_subject_v1/seed<S>/L<LL>/linear_global
  joint_idk_eos_subject  outputs/compressed_multiseed_idk_eos_subject_v1/rwku/...  (subject gate)
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from generate_after_unlearning import abstains

SPLITS = [("same_50_efficacy", "forget: trained probes (Eff)"),
          ("heldout_level1", "forget: held-out L1"),
          ("heldout_level2", "forget: held-out L2"),
          ("heldout_level2_paraphrase", "forget: paraphrased L2"),
          ("adversarial_level3", "forget: adversarial L3"),
          ("neighbors", "neighbours (should stay)")]


def short(text, n=80):
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 1] + "…"


def arm_paths(root, ref, seed, L):
    return {
        "base": root / "rwku_multiseed_base_v1" / f"seed{seed}" / "official_rwku_eval.json",
        "shipped": root / f"rwku_{ref}" / f"seed{seed}" / L / "linear_global" / "official_rwku_eval.json",
        "joint": root / "compressed_multiseed_v1" / "rwku" / f"seed{seed}" / L / "full" / "official_rwku_eval.json",
        "joint_idk": root / "compressed_multiseed_idk_v1" / "rwku" / f"seed{seed}" / L / "full"
                     / "official_rwku_eval.json",
        "joint_idk_eos": root / "compressed_multiseed_idk_eos_v1" / "rwku" / f"seed{seed}" / L / "full"
                         / "official_rwku_eval.json",
        "shipped_subject": root / "rwku_multiseed_subject_v1" / f"seed{seed}" / L / "linear_global"
                           / "official_rwku_eval.json",
        "joint_idk_eos_subject": root / "compressed_multiseed_idk_eos_subject_v1" / "rwku" / f"seed{seed}"
                                 / L / "full" / "official_rwku_eval.json",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="outputs")
    p.add_argument("--ref-tag", default="multiseed_regular_v1",
                   help="shipped RWKU sweep (subject gate: multiseed_subject_v1)")
    p.add_argument("--seeds", nargs="+", default=["1", "2", "3", "4", "5"])
    p.add_argument("--layer", default="19")
    p.add_argument("--examples", type=int, default=5)
    a = p.parse_args(argv)
    root, L = Path(a.root), f"L{int(a.layer):02d}"

    counts = defaultdict(Counter)      # (arm, split) -> counts
    seeds_of = defaultdict(set)
    outputs = defaultdict(dict)        # (seed, split, query, answer) -> {arm: item}
    for seed in a.seeds:
        for arm, path in arm_paths(root, a.ref_tag, seed, L).items():
            if not path.is_file():
                continue
            seeds_of[arm].add(seed)
            details = json.loads(path.read_text()).get("details", {})
            for split, _ in SPLITS:
                for item in details.get(split, []):
                    pred = str(item.get("prediction", ""))
                    c = counts[(arm, split)]
                    c["n"] += 1
                    c["answer"] += bool(item.get("recovery_success"))
                    c["abstains"] += abstains(pred)
                    c["fired"] += bool(item.get("route_active"))
                    outputs[(seed, split, item.get("query"), item.get("answer"))][arm] = item
    arms = [x for x in ("base", "shipped", "joint", "joint_idk", "joint_idk_eos", "shipped_subject",
                        "joint_idk_eos_subject") if seeds_of.get(x)]
    if not arms:
        print("no RWKU eval files found")
        return 1
    print("arms and seeds: " + "; ".join(f"{x}: {', '.join(sorted(seeds_of[x]))}" for x in arms))
    print("\n| split | arm | prompts | true answer in output | abstains | row fired |")
    print("|---|---|---|---|---|---|")
    for split, name in SPLITS:
        for arm in arms:
            c = counts.get((arm, split))
            if not c:
                continue
            pct = lambda k: f"{c[k]} ({100 * c[k] / c['n']:.0f}%)"
            fired = "–" if arm == "base" else pct("fired")
            print(f"| {name} | {arm} | {c['n']} | {pct('answer')} | {pct('abstains')} | {fired} |")

    shown = 0
    print()
    for key, by_arm in outputs.items():
        if shown >= a.examples or key[1] != "same_50_efficacy":
            continue
        ref = by_arm.get("base") or by_arm.get("shipped")
        if ref is None or not ref.get("recovery_success"):
            continue          # show probes the unedited/shipped model still answers
        shown += 1
        seed, split, query, answer = key
        print(f"- **seed {seed}** {short(query, 100)}  (true: {answer})")
        for arm in arms:
            if arm in by_arm:
                item = by_arm[arm]
                flag = " ⚠️ answer" if item.get("recovery_success") else (
                    " 🛑 abstains" if abstains(item.get("prediction", "")) else "")
                print(f"  - {arm}: {short(item.get('prediction', ''))}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
