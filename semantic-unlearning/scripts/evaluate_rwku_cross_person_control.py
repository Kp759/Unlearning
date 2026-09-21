#!/usr/bin/env python3
"""Cross-person best-of-K control for the RWKU subject genie.

The subject genie tries each of a person's K trained rows on a held-out probe
and keeps the most suppressive one. With generation-based selection it drives
held-out recovery far below V2 (54 -> 10 on Level-2, 62 -> 22 on paraphrase).
That gap is only evidence of subject-conditioned transfer if an equal number of
attempts with residuals trained on OTHER people's facts does not do the same.
A residual can derail greedy generation without carrying any knowledge about
the probe's fact, and best-of-K over a binary recovery outcome will find such
rows more often than chance. The single-random-row arm is not a fair control:
it gets one attempt against the genie's K.

Design: one suppression matrix, every arm derived from it
---------------------------------------------------------
Every held-out probe is generated once under every candidate row -- its own
person's K rows and the other-person pool -- plus once with no intervention.
All of it goes through the same functions as the paper's RWKU evaluator:
identical chat-templated prompt, tokenization, greedy decoding, max_new_tokens,
intervention boundary and recovery criterion. Both arms are then read off that
matrix, so they cannot differ in anything except which rows they may use.

  any-suppresses selection (the exact bound; matches --genie-select generation)
    A best-of-k subset stops recovery if ANY of its k rows does. With R of M
    pool rows still recovering, the probability that a uniformly random
    k-subset fails is C(R, k) / C(M, k). That is the exact expectation over
    every possible subset -- no sampling, no seed dependence. Curves are given
    for k = 1 .. K for both arms, so they can be compared at every attempt
    budget rather than only at K. At k = 1 the same-person value is the exact
    expectation of the old single-random-row arm.

  teacher-forced selection (matches --genie-select teacher_forced)
    The row with the lowest sensitive-answer log-probability is chosen and its
    generation is scored. The same-person arm uses all K rows; the cross-person
    arm is estimated from --tf-samples seeded K-subsets of the pool. Samples
    are drawn from the cached matrix, so they cost no model calls.

  per-row suppression rate
    The fraction of a pool's rows that stop recovery on a probe. It does not
    depend on the attempt budget at all, which makes it the most direct
    statistic: does a same-person residual stop a held-out probe more often
    than an other-person residual does?

Every comparison is paired by probe and reported with a seeded bootstrap
interval over probes.

Reading
-------
  cross-person best-of-K ~= same-person best-of-K
      The subject genie's advantage over V2 is attempt count plus generic
      disruption, not transfer. Router work cannot recover it.
  cross-person best-of-K  >> same-person best-of-K  (interval excludes 0)
      Same-person residuals carry probe-relevant signal beyond disruption:
      subject-conditioned transfer, and a better router could capture more of
      it.

Cost
----
probes x (pool rows + 1) generations. With 100 probes and a 50-row bank that is
about 5,100 uncached 30-token generations. The matrix is appended to a JSONL
file as it is computed and a restarted run skips finished (probe, row) pairs.
--cross-pool-cap limits the other-person pool per person to a fixed seeded
subset if the full pool is too expensive; the cap must be at least K.

Usage
-----
python -u scripts/evaluate_rwku_cross_person_control.py \
  --run-dir outputs/rwku_fact_assoc_router_v2_seed1_direct \
  --data-root data/rwku --no-download --local-files-only \
  --output-dir outputs/rwku_fact_assoc_router_v2_seed1_direct/cross_person_control \
  --groups heldout_level2,heldout_paraphrase \
  --compare-genie-rows outputs/rwku_fact_assoc_router_v2_seed1_direct/decomposition_generation/rwku_router_decomposition_rows.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random


HELDOUT_GROUPS = ("heldout_level1", "heldout_level2", "heldout_paraphrase")


# ---------------------------------------------------------------- analysis
# Pure functions over the suppression matrix. No model, no torch.

def probe_tables(entries):
    """Group matrix entries into per-probe tables.

    Returns {probe_key: {"group", "person", "base_recovered",
                         "same": [(row, sum_logprob, recovered)],
                         "other": [(row, sum_logprob, recovered)]}}
    """
    probes = {}
    for entry in entries:
        key = entry["probe_key"]
        table = probes.setdefault(key, {
            "group": entry["group"],
            "person": entry["person"],
            "base_recovered": None,
            "same": [],
            "other": [],
            "k_expected": entry.get("k_expected"),
            "pool_expected": entry.get("pool_expected"),
        })
        if entry["row"] is None:
            table["base_recovered"] = bool(entry["recovered"])
            continue
        record = (int(entry["row"]), float(entry["sum_logprob"]), bool(entry["recovered"]))
        table["same" if entry["same_person"] else "other"].append(record)
    for table in probes.values():
        table["same"].sort()
        table["other"].sort()
    return probes


def p_all_recover(recovering, pool, k):
    """Probability that a uniformly random k-subset of the pool fails to suppress.

    A best-of-k subset suppresses if any member suppresses, so it fails only
    when all k members are drawn from the `recovering` rows. Hypergeometric:
    C(recovering, k) / C(pool, k). Exact over every subset.
    """
    if k <= 0 or k > pool:
        raise ValueError(f"k={k} outside 1..{pool}")
    if recovering < k:
        return 0.0
    return math.comb(recovering, k) / math.comb(pool, k)


def any_curve(table, k):
    same, other = table["same"], table["other"]
    r_same = sum(rec for _, _, rec in same)
    r_other = sum(rec for _, _, rec in other)
    return (
        p_all_recover(r_same, len(same), k),
        p_all_recover(r_other, len(other), k),
    )


def tf_same(table):
    """Teacher-forced selection over all same-person rows: argmin log-prob."""
    best = min(table["same"], key=lambda item: item[1])
    return float(best[2])


def tf_cross(table, k, samples, rng):
    """Seeded Monte Carlo of teacher-forced selection over k-subsets of the pool."""
    pool = table["other"]
    if k > len(pool):
        raise ValueError("Cross-person pool smaller than the attempt budget")
    hits = 0
    for _ in range(int(samples)):
        subset = rng.sample(pool, k)
        best = min(subset, key=lambda item: item[1])
        hits += int(best[2])
    return hits / float(samples)


def suppression_rates(table):
    same, other = table["same"], table["other"]
    return (
        sum(not rec for _, _, rec in same) / len(same),
        sum(not rec for _, _, rec in other) / len(other),
    )


def bootstrap_mean(values, iterations, seed):
    """Seeded percentile bootstrap of the mean over probes."""
    values = list(values)
    if not values:
        return {"mean": None, "low": None, "high": None, "n": 0}
    rng = random.Random(int(seed))
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(int(iterations))
    )
    return {
        "mean": sum(values) / n,
        "low": means[int(0.025 * (len(means) - 1))],
        "high": means[int(0.975 * (len(means) - 1))],
        "n": n,
    }


def reading_for(interval, label):
    if interval["mean"] is None:
        return "no probes"
    if interval["low"] > 0:
        return (
            f"subject-conditioned: {label} interval excludes 0 on the positive "
            "side; same-person residuals stop these probes more than an equal "
            "number of other-person residuals, so the genie's advantage is not "
            "attempt count or generic disruption alone"
        )
    if interval["high"] < 0:
        return (
            f"reversed: {label} interval excludes 0 on the negative side; "
            "other-person residuals suppress MORE than same-person ones, which "
            "is disruption, not transfer"
        )
    return (
        f"not separable: {label} interval includes 0; at this sample size the "
        "subject genie cannot be distinguished from an equal number of "
        "other-person attempts, so treat its advantage over V2 as attempt "
        "count plus generic disruption"
    )


def analyze(entries, tf_samples, bootstrap_iterations, seed):
    probes = probe_tables(entries)
    by_group = defaultdict(list)
    skipped = defaultdict(int)
    for key, table in sorted(probes.items()):
        # An interrupted run leaves probes with partly filled pools. Analyzing
        # those would compare a short same-person pool against a short
        # cross-person pool of a different length, so they are skipped.
        complete = (
            table["same"] and table["other"]
            and table["base_recovered"] is not None
            and (table["k_expected"] is None or len(table["same"]) == table["k_expected"])
            and (table["pool_expected"] is None or len(table["other"]) == table["pool_expected"])
        )
        if not complete:
            skipped[table["group"]] += 1
            continue
        by_group[table["group"]].append((key, table))

    rng = random.Random(int(seed))
    report = {}
    for group, items in sorted(by_group.items()):
        k_values = sorted({len(t["same"]) for _, t in items})
        k_min = k_values[0]

        curve = []
        for k in range(1, k_min + 1):
            pairs = [any_curve(t, k) for _, t in items]
            curve.append({
                "k": k,
                "same_person_recovery_percent": 100.0 * sum(p[0] for p in pairs) / len(pairs),
                "cross_person_recovery_percent": 100.0 * sum(p[1] for p in pairs) / len(pairs),
            })

        # Matched budget: each probe uses k = its own person's row count, so
        # both arms get exactly the attempts the subject genie had.
        matched_any, matched_tf, rates = [], [], []
        per_probe = []
        for key, table in items:
            k = len(table["same"])
            same_any, cross_any = any_curve(table, k)
            same_tf = tf_same(table)
            cross_tf = tf_cross(table, k, tf_samples, rng)
            rate_same, rate_other = suppression_rates(table)
            matched_any.append((same_any, cross_any))
            matched_tf.append((same_tf, cross_tf))
            rates.append((rate_same, rate_other))
            per_probe.append({
                "probe_key": key,
                "person": table["person"],
                "k": k,
                "pool_size": len(table["other"]),
                "base_recovered": table["base_recovered"],
                "same_any_recovers": same_any,
                "cross_any_recovery_probability": cross_any,
                "same_tf_recovers": same_tf,
                "cross_tf_recovery_probability": cross_tf,
                "same_rows_suppressing": sum(not r for _, _, r in table["same"]),
                "other_rows_suppressing": sum(not r for _, _, r in table["other"]),
            })

        n = len(items)
        base_values = [t["base_recovered"] for _, t in items if t["base_recovered"] is not None]
        diff_any = bootstrap_mean(
            [100.0 * (c - s) for s, c in matched_any], bootstrap_iterations, seed
        )
        diff_tf = bootstrap_mean(
            [100.0 * (c - s) for s, c in matched_tf], bootstrap_iterations, seed + 1
        )
        diff_rate = bootstrap_mean(
            [100.0 * (s - o) for s, o in rates], bootstrap_iterations, seed + 2
        )
        report[group] = {
            "probes": n,
            "incomplete_probes_skipped": skipped.get(group, 0),
            "attempt_budget_k": k_values,
            "cross_pool_sizes": sorted({len(t["other"]) for _, t in items}),
            "base_recovery_percent": (
                100.0 * sum(base_values) / len(base_values) if base_values else None
            ),
            "matched_budget": {
                "any_suppresses": {
                    "same_person_recovery_percent": 100.0 * sum(s for s, _ in matched_any) / n,
                    "cross_person_recovery_percent": 100.0 * sum(c for _, c in matched_any) / n,
                    "cross_minus_same_points": diff_any,
                    "reading": reading_for(diff_any, "cross-minus-same"),
                },
                "teacher_forced_selection": {
                    "same_person_recovery_percent": 100.0 * sum(s for s, _ in matched_tf) / n,
                    "cross_person_recovery_percent": 100.0 * sum(c for _, c in matched_tf) / n,
                    "cross_minus_same_points": diff_tf,
                    "tf_samples_per_probe": int(tf_samples),
                    "reading": reading_for(diff_tf, "cross-minus-same"),
                },
            },
            "per_row_suppression_rate": {
                "same_person_percent": 100.0 * sum(s for s, _ in rates) / n,
                "other_person_percent": 100.0 * sum(o for _, o in rates) / n,
                "same_minus_other_points": diff_rate,
                "reading": reading_for(diff_rate, "same-minus-other rate"),
            },
            "best_of_k_curve": curve,
            "per_probe": per_probe,
        }
    return report


# ------------------------------------------------------------ model side

def load_heldout_groups(split, wanted, source_to_row, max_rows):
    """Mirror evaluate_rwku_router_decomposition's held-out group construction."""
    raw = {
        "heldout_level1": list(split["heldout_level1"]),
        "heldout_level2": list(split["heldout_level2"]),
        "heldout_paraphrase": list(split["heldout_paraphrase"]),
    }
    groups = {}
    for name in wanted:
        if name not in raw:
            raise SystemExit(f"{name} is not a held-out group; choose from {HELDOUT_GROUPS}")
        rows = []
        for index, source in enumerate(raw[name]):
            row = dict(source)
            if str(row.get("source_record_sha256", "")) in source_to_row:
                raise SystemExit(f"{name} probe {index} is a training record")
            row["_group"] = name
            row["_person"] = int(row["rwku_target_seed"])
            row["_probe_key"] = f"{name}::{row.get('source_record_sha256', '')}::{index}"
            rows.append(row)
        if max_rows:
            rows = rows[: int(max_rows)]
        groups[name] = rows
    return groups


def build_pools(person_rows, all_rows, cap, seed):
    """Fixed other-person pool per person, seeded when capped.

    The pool is fixed per person rather than per probe so every probe of a
    person faces the same comparison set.
    """
    pools = {}
    rng = random.Random(int(seed))
    for person, own in sorted(person_rows.items()):
        others = [r for r in all_rows if r not in set(own)]
        if cap:
            if int(cap) < len(own):
                raise SystemExit(
                    f"--cross-pool-cap {cap} is below person {person}'s K={len(own)}; "
                    "the cross-person arm needs at least K rows to match attempts"
                )
            others = sorted(rng.sample(others, min(int(cap), len(others))))
        pools[person] = others
    return pools


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", default="data/rwku")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--groups", default=",".join(HELDOUT_GROUPS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--max-rows-per-group", type=int, default=0)
    parser.add_argument("--cross-pool-cap", type=int, default=0)
    parser.add_argument("--tf-samples", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--compare-genie-rows",
        default="",
        help="decomposition rows JSON from a --genie-select generation run; its "
        "genie_subject recovery must equal this script's same-person matched value",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="recompute the report from an existing matrix without loading a model",
    )
    args = parser.parse_args(argv)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    matrix_path = output / "suppression_matrix.jsonl"

    entries = []
    if matrix_path.exists():
        entries = [json.loads(line) for line in matrix_path.read_text().splitlines() if line.strip()]

    if not args.analyze_only:
        entries = compute_matrix(args, matrix_path, entries)

    report = {
        "schema_version": "rwku_cross_person_control_v1",
        "run_dir": str(Path(args.run_dir).resolve()),
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "cross_pool_cap": args.cross_pool_cap,
        "seed": args.seed,
        "matrix": str(matrix_path),
        "matrix_entries": len(entries),
        "groups": analyze(entries, args.tf_samples, args.bootstrap, args.seed),
    }

    if args.compare_genie_rows:
        previous = json.loads(Path(args.compare_genie_rows).read_text())
        checks = {}
        for group, block in report["groups"].items():
            items = previous.get("genie_subject", {}).get(group)
            if not items:
                continue
            earlier = 100.0 * sum(bool(x["recovered"]) for x in items) / len(items)
            now = block["matched_budget"]["any_suppresses"]["same_person_recovery_percent"]
            checks[group] = {
                "decomposition_genie_subject_recovery_percent": earlier,
                "control_same_person_any_recovery_percent": now,
                "match": abs(earlier - now) < 1e-9,
            }
        report["consistency_with_decomposition_genie"] = checks

    (output / "cross_person_control.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    headline = {
        group: {
            "base": block["base_recovery_percent"],
            "same_person_bestofK_any": block["matched_budget"]["any_suppresses"]["same_person_recovery_percent"],
            "cross_person_bestofK_any": block["matched_budget"]["any_suppresses"]["cross_person_recovery_percent"],
            "cross_minus_same_CI": [
                block["matched_budget"]["any_suppresses"]["cross_minus_same_points"]["low"],
                block["matched_budget"]["any_suppresses"]["cross_minus_same_points"]["high"],
            ],
            "per_row_rate_same_vs_other": [
                block["per_row_suppression_rate"]["same_person_percent"],
                block["per_row_suppression_rate"]["other_person_percent"],
            ],
            "reading": block["matched_budget"]["any_suppresses"]["reading"],
        }
        for group, block in report["groups"].items()
    }
    print(json.dumps({
        "status": "cross_person_control_complete",
        "headline": headline,
        "consistency_with_decomposition_genie": report.get("consistency_with_decomposition_genie"),
        "output": str(output / "cross_person_control.json"),
    }, indent=2))
    return 0


def compute_matrix(args, matrix_path, entries):
    """Fill the probe x row matrix, resuming from any entries already on disk."""
    import torch

    import rwku_eval as rwku
    from evaluate_rwku_fact_association_embeddings_seed1 import (
        generate_fixed_boundary,
        score_answer_fixed_boundary,
    )
    from mcf_zero_unlearn_official_eval import dtype_from_str
    from oracle_router_gate import ForcedRowBank
    from rwku_batch50 import build_batch_split
    from rwku_fact_association_embeddings import build_association_facts
    from static_overlap_fact_association_embeddings import AssociationCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = Path(args.run_dir).resolve()
    manifest = json.loads((run_dir / "association_manifest.json").read_text())
    artifact = torch.load(
        run_dir / "fact_association_embeddings.pt", map_location="cpu", weights_only=False
    )
    model_path = Path(manifest["model_path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    split = build_batch_split(
        data_root=Path(args.data_root).resolve(),
        batch_seed=1,
        allow_download=not args.no_download,
    )
    expected_facts, record_to_fact_id, _ = build_association_facts(
        list(split["efficacy_forget"]), tokenizer
    )
    if [str(f["association_key"]) for f in expected_facts] != [
        str(f.get("association_key")) for f in artifact["facts"]
    ]:
        raise SystemExit("Saved RWKU bank no longer matches the frozen seed-1 split")
    fact_to_row = {fact["id"]: i for i, fact in enumerate(expected_facts)}
    source_to_row = {s: fact_to_row[f] for s, f in record_to_fact_id.items()}

    person_rows = defaultdict(list)
    for index, fact in enumerate(artifact["facts"]):
        person_rows[int(fact["rwku_target_seed"])].append(index)
    all_rows = list(range(len(artifact["facts"])))
    pools = build_pools(person_rows, all_rows, args.cross_pool_cap, args.seed)

    wanted = [g.strip() for g in args.groups.split(",") if g.strip()]
    groups = load_heldout_groups(split, wanted, source_to_row, args.max_rows_per_group)

    done = {(e["probe_key"], e["row"]) for e in entries}
    planned = sum(
        len(person_rows[row["_person"]]) + len(pools[row["_person"]]) + 1
        for rows in groups.values() for row in rows
    )
    remaining = planned - len(done)
    print(json.dumps({
        "planned_generations": planned,
        "already_on_disk": len(done),
        "remaining": remaining,
        "persons": {str(p): len(r) for p, r in sorted(person_rows.items())},
        "cross_pool_sizes": {str(p): len(r) for p, r in sorted(pools.items())},
    }, indent=2), flush=True)
    if remaining <= 0:
        return entries

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype_from_str(args.dtype),
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    base_model.requires_grad_(False)
    base_model.config.use_cache = False
    bank = ForcedRowBank(
        base_model,
        int(artifact["layer"]),
        artifact["rows"],
        artifact["subject_patterns"],
        artifact["facts"],
    )
    model = AssociationCausalLM(base_model, bank).eval()

    produced = 0
    with matrix_path.open("a") as sink:
        for name, rows in groups.items():
            for position, row in enumerate(rows):
                prompt = rwku.format_qa_prompt(tokenizer, row)
                answer = str(row["answer"])
                own = person_rows[row["_person"]]
                candidates = [None, *own, *pools[row["_person"]]]
                for candidate in candidates:
                    if (row["_probe_key"], candidate) in done:
                        continue
                    bank.forced_row = candidate
                    scored = score_answer_fixed_boundary(model, tokenizer, prompt, answer)
                    prediction, _ = generate_fixed_boundary(
                        model, bank, tokenizer, prompt, max_new_tokens=args.max_new_tokens
                    )
                    entry = {
                        "probe_key": row["_probe_key"],
                        "group": name,
                        "person": row["_person"],
                        "row": candidate,
                        "row_person": (
                            None if candidate is None
                            else int(artifact["facts"][candidate]["rwku_target_seed"])
                        ),
                        "same_person": candidate is not None and candidate in own,
                        "k_expected": len(own),
                        "pool_expected": len(pools[row["_person"]]),
                        "sum_logprob": scored["sum_logprob"],
                        "recovered": bool(rwku.recovery_success(prediction, answer)),
                        "prediction": prediction or "NOANSWER",
                    }
                    sink.write(json.dumps(entry, allow_nan=False) + "\n")
                    sink.flush()
                    entries.append(entry)
                    produced += 1
                if (position + 1) % 10 == 0:
                    print(
                        f"  [{name}] {position + 1}/{len(rows)} probes, "
                        f"{produced}/{remaining} new generations",
                        flush=True,
                    )
    bank.close()
    return entries


if __name__ == "__main__":
    raise SystemExit(main())
