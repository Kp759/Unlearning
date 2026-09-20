#!/usr/bin/env python3
"""Step 2: a held-out operating point, against the per-association in-sample tau.

The shipped rule fits N independent thresholds, each from a handful of points,
using the same prompts that built the prototypes. In the separable case tau
lies above every fitting negative and below every fitting positive *by
construction*, so the separation the paper reports is a property of the fitting
rule rather than a measurement. There is no held-out false-activation estimate
anywhere in the pipeline.

Three regimes are compared on the same held-out scores:

  per_assoc_in_sample  the shipped rule. Reproduced here so the optimism has a
                       number attached rather than being asserted.
  per_assoc_held_out   same per-association form, fitted on held-out data.
                       Isolates in-sample optimism from the choice of having N
                       thresholds at all.
  global               one threshold for every association, chosen to hit a
                       target false-activation rate on held-out negatives.

The global regime is what to prefer, for a reason beyond its numbers: it
answers "how do you set tau for association N+1?" without retuning. N
in-sample thresholds have no answer to that question.

Every rate is reported with a Wilson interval. At these sample sizes a normal
approximation puts the lower bound of a low false-activation rate below zero,
which is exactly the regime where a confident-looking point estimate is worth
least.

Inputs are score records: {fact_index, d, polarity} rows, as produced by
sweep_router_read_layer.py or probe_router_surface_robustness.py, or any
scorer. Scores are not recomputed here -- this script only chooses and
evaluates thresholds, so it runs in seconds on CPU.

Usage
-----
python -u scripts/calibrate_router_operating_point.py \
  --scores outputs/<run>/probes/scores.json \
  --output-dir outputs/<run>/calibration \
  --target-fpr 0.001 --holdout-fraction 0.5 --seed 1
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random


def wilson(successes, total, z=1.96):
    """Wilson score interval. Correct near 0 and 1, unlike the normal approx."""
    if total == 0:
        return {"rate": None, "low": None, "high": None, "n": 0}
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    ) / denominator
    return {
        "rate": p,
        "low": max(0.0, centre - spread),
        "high": min(1.0, centre + spread),
        "n": int(total),
    }


def split_rows(rows, holdout_fraction, seed):
    """Per-association stratified split so every fact appears on both sides."""
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["fact_index"], row["polarity"])].append(row)
    rng = random.Random(int(seed))
    fit, held = [], []
    for key in sorted(by_key):
        items = list(by_key[key])
        rng.shuffle(items)
        cut = int(round(len(items) * (1.0 - float(holdout_fraction))))
        cut = max(0, min(len(items), cut))
        # With a single sample the fitting side wins it; a held-out set that
        # silently drops associations is worse than a thin fitting set.
        if len(items) == 1:
            fit.extend(items)
            continue
        cut = max(1, min(len(items) - 1, cut))
        fit.extend(items[:cut])
        held.extend(items[cut:])
    return fit, held


def per_association_threshold(positives, negatives, lam=0.10, slack=0.02):
    """The shipped rule, reproduced exactly (Equation 7 in the writeup)."""
    if not positives:
        return None
    positive_floor = min(positives)
    if not negatives:
        return max(-2.0, min(2.0, positive_floor - slack))
    negative_ceiling = max(negatives)
    if negative_ceiling < positive_floor:
        tau = negative_ceiling + float(lam) * (positive_floor - negative_ceiling)
    else:
        tau = positive_floor - float(slack)
    return max(-2.0, min(2.0, tau))


def global_threshold_at_fpr(negatives, target_fpr):
    """Smallest threshold whose held-out false-activation rate is <= target.

    Firing requires d >= tau, so the false-activation rate at tau is the
    fraction of negatives at or above it. Walking the sorted negatives from the
    top gives the tightest admissible threshold.
    """
    if not negatives:
        return None
    ordered = sorted(negatives, reverse=True)
    allowed = int(math.floor(float(target_fpr) * len(ordered)))
    if allowed >= len(ordered):
        return min(ordered) - 1e-6
    # Place tau just above the (allowed+1)-th highest negative so that exactly
    # `allowed` negatives remain at or above it.
    return ordered[allowed] + 1e-9


def evaluate(rows, threshold_for):
    """Apply a threshold rule and count positives kept and negatives fired."""
    kept = fired = positives = negatives = 0
    for row in rows:
        tau = threshold_for(row["fact_index"])
        if tau is None:
            continue
        active = float(row["d"]) >= float(tau)
        if row["polarity"] == "positive":
            positives += 1
            kept += int(active)
        else:
            negatives += 1
            fired += int(active)
    return {
        "recall": wilson(kept, positives),
        "false_activation": wilson(fired, negatives),
        "positive_count": positives,
        "negative_count": negatives,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-fpr", type=float, default=0.001)
    parser.add_argument("--holdout-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lam", type=float, default=0.10)
    parser.add_argument("--slack", type=float, default=0.02)
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.scores).read_text())
    rows = payload["rows"] if isinstance(payload, dict) else payload
    for row in rows:
        if row.get("polarity") not in ("positive", "negative"):
            raise SystemExit("Every score row needs polarity positive|negative")
        if "d" not in row or "fact_index" not in row:
            raise SystemExit("Every score row needs fact_index and d")

    fit_rows, held_rows = split_rows(rows, args.holdout_fraction, args.seed)
    if not held_rows:
        raise SystemExit("Held-out split is empty; lower --holdout-fraction")

    def collect(source):
        pos, neg = defaultdict(list), defaultdict(list)
        for row in source:
            target = pos if row["polarity"] == "positive" else neg
            target[row["fact_index"]].append(float(row["d"]))
        return pos, neg

    all_pos, all_neg = collect(rows)
    fit_pos, fit_neg = collect(fit_rows)

    in_sample = {
        index: per_association_threshold(
            all_pos.get(index, []), all_neg.get(index, []), args.lam, args.slack
        )
        for index in set(all_pos) | set(all_neg)
    }
    held_out_fit = {
        index: per_association_threshold(
            fit_pos.get(index, []), fit_neg.get(index, []), args.lam, args.slack
        )
        for index in set(fit_pos) | set(fit_neg)
    }
    global_tau = global_threshold_at_fpr(
        [float(r["d"]) for r in fit_rows if r["polarity"] == "negative"],
        args.target_fpr,
    )

    regimes = {
        "per_assoc_in_sample": {
            "description": (
                "the shipped rule: tau fitted on the same prompts that built "
                "the prototypes, then evaluated on held-out data"
            ),
            "thresholds": {str(k): v for k, v in sorted(in_sample.items())},
            # What the shipped rule effectively reports about itself: fitted on
            # every prompt, then scored on those same prompts. This is not a
            # result, it is the optimistic baseline the honest regime is
            # measured against.
            "self_reported": evaluate(rows, in_sample.get),
            # Scoring these same thresholds on the held-out split is NOT an
            # honest estimate either, because the thresholds saw those rows
            # while being fitted. Kept only to show that the split alone
            # changes nothing without refitting.
            "held_out_same_thresholds": evaluate(held_rows, in_sample.get),
        },
        "per_assoc_held_out": {
            "description": "same per-association form, fitted on the fitting split only",
            "thresholds": {str(k): v for k, v in sorted(held_out_fit.items())},
            "held_out": evaluate(held_rows, held_out_fit.get),
        },
        "global": {
            "description": (
                f"one threshold for all associations at a target held-out "
                f"false-activation rate of {args.target_fpr}"
            ),
            "threshold": global_tau,
            "held_out": evaluate(held_rows, lambda _index: global_tau),
            "generalizes_to_new_associations": True,
        },
    }

    # In-sample optimism is the gap between what the shipped rule reports about
    # itself (fitted and scored on the same prompts) and what the SAME rule
    # form achieves when honestly refitted on a fitting split and scored on
    # held-out data. Comparing two in-sample quantities would measure nothing.
    self_reported = regimes["per_assoc_in_sample"]["self_reported"]
    honest = regimes["per_assoc_held_out"]["held_out"]
    in_sample_gap = None
    if (
        self_reported["false_activation"]["rate"] is not None
        and honest["false_activation"]["rate"] is not None
    ):
        in_sample_gap = (
            honest["false_activation"]["rate"]
            - self_reported["false_activation"]["rate"]
        )
    recall_gap = None
    if (
        self_reported["recall"]["rate"] is not None
        and honest["recall"]["rate"] is not None
    ):
        recall_gap = honest["recall"]["rate"] - self_reported["recall"]["rate"]

    report = {
        "schema_version": "router_operating_point_v1",
        "scores": str(Path(args.scores).resolve()),
        "target_fpr": args.target_fpr,
        "holdout_fraction": args.holdout_fraction,
        "seed": args.seed,
        "row_count": len(rows),
        "fitting_rows": len(fit_rows),
        "held_out_rows": len(held_rows),
        "regimes": regimes,
        "in_sample_optimism_false_activation": in_sample_gap,
        "in_sample_optimism_recall": recall_gap,
        "interpretation": (
            "in_sample_optimism_false_activation = honest held-out false "
            "activation (per_assoc_held_out) minus the rate the shipped rule "
            "reports about itself (per_assoc_in_sample.self_reported). A large "
            "positive value means the reported separation was a property of "
            "the fitting rule rather than of the representation. Then compare "
            "the global regime at matched false activation: if its recall is "
            "close, prefer it, because one threshold also answers how to "
            "handle association N+1 without retuning."
        ),
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "router_operating_point.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "status": "calibration_complete",
        "global_threshold": global_tau,
        "held_out": {
            name: {
                "recall": block["recall"]["rate"],
                "false_activation": block["false_activation"]["rate"],
                "false_activation_ci": [
                    block["false_activation"]["low"],
                    block["false_activation"]["high"],
                ],
            }
            for name, block in (
                # The shipped rule has no honest held-out number of its own, so
                # it is summarized by what it reports about itself.
                ("per_assoc_in_sample_self_reported",
                 regimes["per_assoc_in_sample"]["self_reported"]),
                ("per_assoc_held_out", regimes["per_assoc_held_out"]["held_out"]),
                ("global", regimes["global"]["held_out"]),
            )
        },
        "in_sample_optimism_false_activation": in_sample_gap,
        "output": str(output / "router_operating_point.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
