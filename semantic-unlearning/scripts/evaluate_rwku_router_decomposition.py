#!/usr/bin/env python3
"""Step 0 on RWKU: separate routing failures from actuation and coverage limits.

The MCF decomposition put Router V2 at the genie ceiling, but MCF is the easy
case for routing: every official paraphrase contains the subject string word
for word, so lexical eligibility cannot miss. RWKU is where the paper's weak
numbers are (held-out Level-2 paraphrase recovery only drops 62% -> 44%), so it
is where the routing question has to be answered.

RWKU differs from MCF in a way that changes what a genie can measure. The
held-out Level-1, Level-2 and paraphrase probes are content-disjoint from the 50
training probes: they ask *different facts* about the same five people, and no
residual row exists for any of them. An exact genie therefore abstains on every
held-out probe by construction and bounds nothing there. Two genies are used:

  genie_exact    same-50 probes only: the ground-truth row is supplied. As on
                 MCF, the gap to V2 is the cost of routing on trained
                 associations, and what remains is actuation.

  genie_subject  held-out probes: for each probe, every trained row belonging
                 to the same person is tried and the most suppressive one is
                 kept. This is the best that ANY router could do with this bank
                 on a fact the bank has no row for. It separates three readings
                 of the held-out number:

                   genie_subject ~= base   no trained residual transfers to
                                           unlisted facts about the person.
                                           The held-out recovery is a coverage
                                           limit of one-row-per-association,
                                           and no router change can move it.
                   genie_subject << V2     residuals do transfer, but V2 does
                                           not fire, or picks the wrong row.
                                           Router work has headroom here.
                   V2 ~= genie_subject     V2 already realizes whatever
                          << base          transfer the bank supports.

  genie_subject_random   a random same-person row. If it matches
                 genie_subject, WHICH row is used does not matter and the
                 effect is person-level rather than fact-level.

Neighbors are other people and must not be routed, so their genie is abstention,
i.e. the base model. Any V2 recovery drop on neighbors is the router's locality
cost: the paper's RWKU neighbor recovery falls 69.18 -> 65.59, unlike MCF where
neighborhood was untouched.

Parity. Scoring, generation and the recovery criterion are imported from
evaluate_rwku_fact_association_embeddings_seed1.py rather than reimplemented,
so tokenization matches training exactly (chat-templated prompt,
add_special_tokens=True, normalized completion) and recovery matches the paper's
headline RWKU metric. Teacher-forced probability is reported alongside it
because the paper's RWKU gap -- zero sensitive-token accuracy but nonzero
generated recovery -- sits exactly between the two.

Genie row selection defaults to teacher-forced: the candidate row with the
lowest sensitive-answer log-probability is chosen, then generation runs with
that row. It is cheap (one forward per candidate) but approximates the true
best-of-recovery bound. `--genie-select generation` generates with every
candidate and keeps one that suppresses recovery if any does -- the exact bound,
at roughly ten times the cost.

Usage
-----
python -u scripts/evaluate_rwku_router_decomposition.py \
  --run-dir outputs/rwku_fact_assoc_router_v2_seed1 \
  --data-root data/rwku --output-dir outputs/rwku_fact_assoc_router_v2_seed1/decomposition \
  --device cuda --local-files-only --no-download
"""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re

import torch

import rwku_eval as rwku
from evaluate_rwku_fact_association_embeddings_seed1 import (
    _native_rows,
    generate_fixed_boundary,
    score_answer_fixed_boundary,
)
from mcf_zero_unlearn_official_eval import dtype_from_str
from oracle_router_gate import ForcedRowBank
from rwku_batch50 import build_batch_split
from rwku_fact_association_embeddings import build_association_facts
from static_overlap_fact_association_embeddings import AssociationCausalLM
from static_overlap_fact_association_v2_gate import (
    load_relation_prototype_artifact,
)


HELDOUT_GROUPS = ("heldout_level1", "heldout_level2", "heldout_paraphrase")
ALL_GROUPS = ("same50", *HELDOUT_GROUPS, "neighbors")
# Groups whose probes are trained associations and so have a ground-truth row.
# same50_paraphrase is the reworded trained probes (--same50-paraphrases): the
# RWKU analogue of MCF/zsRE Gen and MQuAKE AtomicGen.
GOLD_GROUPS = ("same50", "same50_paraphrase")


def wilson(successes, total, z=1.96):
    if not total:
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


# ---------------------------------------------------------------- per-row eval

def evaluate_row(model, bank, tokenizer, row, max_new_tokens, score=True):
    """One probe, mirroring evaluate_rows() in the paper's RWKU evaluator."""
    prompt = rwku.format_qa_prompt(tokenizer, row)
    prediction, route = generate_fixed_boundary(
        model, bank, tokenizer, prompt, max_new_tokens=max_new_tokens
    )
    item = {
        "group": row["_group"],
        "source_record_sha256": str(row.get("source_record_sha256", "")),
        "person": row["_person"],
        "subject": str(row.get("subject", "")),
        "query": str(row["query"]),
        "answer": str(row["answer"]),
        "gold_row": row["_gold_row"],
        "prediction": prediction or "NOANSWER",
        "recovered": bool(rwku.recovery_success(prediction, str(row["answer"]))),
        "rouge_l_recall": float(rwku.rouge_l_recall(prediction, str(row["answer"]))),
        "routed_row": int(route[0]) if route else None,
        "fired": bool(route),
    }
    if "paraphrase_index" in row:
        item["paraphrase_index"] = row["paraphrase_index"]
        item["paraphrase_of_source_record_sha256"] = str(
            row["paraphrase_of_source_record_sha256"]
        )
        item["paraphrase_method"] = row.get("paraphrase_method")
        item["original_query"] = row.get("original_query")
    if score:
        scored = score_answer_fixed_boundary(
            model, tokenizer, prompt, str(row["answer"])
        )
        item.update({
            "answer_sum_logprob": scored["sum_logprob"],
            "answer_prob": float(math.exp(scored["sum_logprob"])),
            "answer_geometric_prob": scored["geometric_probability"],
            "sensitive_token_top1_accuracy": scored["sensitive_token_top1_accuracy"],
        })
    return item


def run_bank(model, bank, tokenizer, rows, max_new_tokens, label, choose=None):
    """Evaluate rows under one bank. `choose` sets forced routing per row."""
    results = []
    for position, row in enumerate(rows):
        if choose is not None:
            bank.forced_row = choose(row)
        results.append(evaluate_row(model, bank, tokenizer, row, max_new_tokens))
        if (position + 1) % 50 == 0:
            print(f"  [{label}] {position + 1}/{len(rows)}", flush=True)
    return results


@torch.no_grad()
def select_genie_rows(model, bank, tokenizer, rows, person_rows, mode,
                      max_new_tokens):
    """For each probe, pick the same-person row that suppresses it best.

    teacher_forced: lowest sensitive-answer log-probability among candidates.
    generation:     generate with every candidate; prefer rows that stop
                    recovery, break ties by log-probability. Exact best-of.
    Returns {source_hash_or_index: (chosen_row, candidate_table)}.
    """
    choices = {}
    for index, row in enumerate(rows):
        candidates = person_rows.get(row["_person"], [])
        if not candidates:
            choices[_row_key(row, index)] = (None, [])
            continue
        prompt = rwku.format_qa_prompt(tokenizer, row)
        table = []
        for candidate in candidates:
            bank.forced_row = candidate
            scored = score_answer_fixed_boundary(
                model, tokenizer, prompt, str(row["answer"])
            )
            entry = {"row": candidate, "sum_logprob": scored["sum_logprob"]}
            if mode == "generation":
                prediction, _ = generate_fixed_boundary(
                    model, bank, tokenizer, prompt, max_new_tokens=max_new_tokens
                )
                entry["recovered"] = bool(
                    rwku.recovery_success(prediction, str(row["answer"]))
                )
            table.append(entry)
        if mode == "generation":
            best = min(
                table, key=lambda e: (e["recovered"], e["sum_logprob"])
            )
        else:
            best = min(table, key=lambda e: e["sum_logprob"])
        choices[_row_key(row, index)] = (best["row"], table)
        if (index + 1) % 25 == 0:
            print(f"  [genie select] {index + 1}/{len(rows)}", flush=True)
    bank.forced_row = None
    return choices


def _row_key(row, index):
    return f"{row['_group']}::{row.get('source_record_sha256') or index}"


# ------------------------------------------------------------------ summaries

def summarize(items):
    count = len(items)
    if not count:
        return {"count": 0}
    scored = [x for x in items if "answer_prob" in x]
    fired = sum(x["fired"] for x in items)
    with_gold = [x for x in items if x["gold_row"] is not None]
    return {
        "count": count,
        "recovery_percent": 100.0 * sum(x["recovered"] for x in items) / count,
        "recovery_ci": wilson(sum(x["recovered"] for x in items), count),
        "rouge_l_recall_percent": (
            100.0 * sum(x["rouge_l_recall"] for x in items) / count
        ),
        "route_active_fraction": fired / count,
        "route_active_ci": wilson(fired, count),
        "correct_route_fraction": (
            sum(x["fired"] and x["routed_row"] == x["gold_row"] for x in with_gold)
            / len(with_gold)
            if with_gold else None
        ),
        "mean_answer_prob": (
            sum(x["answer_prob"] for x in scored) / len(scored) if scored else None
        ),
        "mean_answer_geometric_prob": (
            sum(x["answer_geometric_prob"] for x in scored) / len(scored)
            if scored else None
        ),
        "mean_sensitive_token_top1_accuracy": (
            sum(x["sensitive_token_top1_accuracy"] for x in scored) / len(scored)
            if scored else None
        ),
    }


def attribute_same50(items):
    """Recovered = the failure the paper counts. Split it by routing."""
    table = defaultdict(int)
    for x in items:
        routed = x["fired"] and x["routed_row"] == x["gold_row"]
        key = ("routed" if routed else "not_routed") + "_" + (
            "recovered" if x["recovered"] else "suppressed"
        )
        table[key] += 1
    failures = table["routed_recovered"] + table["not_routed_recovered"]
    return {
        "counts": dict(table),
        "failure_count": failures,
        "actuation_failure_share": (
            table["routed_recovered"] / failures if failures else None
        ),
        "routing_failure_share": (
            table["not_routed_recovered"] / failures if failures else None
        ),
    }


def attribute_heldout(items):
    """Held-out probes have no gold row, so split by whether V2 fired at all."""
    table = defaultdict(int)
    for x in items:
        key = ("fired" if x["fired"] else "abstained") + "_" + (
            "recovered" if x["recovered"] else "suppressed"
        )
        table[key] += 1
    return {"counts": dict(table)}


def answers_overlap(a, b):
    """Same answer, or one contains the other as whole words (>= 3 chars).

    RWKU's split enforces exact-CONTENT disjointness between trained and
    held-out probes, not fact disjointness. A held-out Level-2 question can ask
    for the same fact as a trained Level-1 blank ("Where was X born?" vs
    "X was born in ___"). Its answer then matches a trained answer, and its
    suppression reflects a reworded trained fact rather than transfer to a new
    one.
    """
    a, b = rwku.normalize_text(a), rwku.normalize_text(b)
    if not a or not b:
        return None
    if a == b:
        return "exact"
    short, long = sorted((a, b), key=len)
    if len(short) >= 3 and re.search(r"\b" + re.escape(short) + r"\b", long):
        return "contained"
    return None


def trained_answers_by_person(rows):
    answers = defaultdict(list)
    for row in rows:
        person = row.get("person", row.get("_person", row.get("rwku_target_seed")))
        answers[int(person)].append(str(row["answer"]))
    return answers


def overlap_split(results, trained):
    """Split every held-out group by whether its answer matches a trained answer.

    `results` is arm -> group -> items; items carry `person` and `answer`.
    """
    report = {}
    for group in HELDOUT_GROUPS:
        reference = None
        for arm in results:
            if results[arm].get(group):
                reference = results[arm][group]
                break
        if not reference:
            continue
        flags = {}
        examples = []
        for index, item in enumerate(reference):
            match = None
            for answer in trained.get(int(item["person"]), []):
                kind = answers_overlap(item["answer"], answer)
                if kind:
                    match = (kind, answer)
                    if kind == "exact":
                        break
            flags[index] = match
            if match and len(examples) < 15:
                examples.append({
                    "query": item["query"], "answer": item["answer"],
                    "matching_trained_answer": match[1], "match": match[0],
                })
        overlapping = [i for i, m in flags.items() if m]
        block = {
            "probes": len(reference),
            "answer_overlaps_trained": len(overlapping),
            "exact": sum(1 for m in flags.values() if m and m[0] == "exact"),
            "contained": sum(1 for m in flags.values() if m and m[0] == "contained"),
            "examples": examples,
            "by_arm": {},
        }
        for arm in results:
            items = results[arm].get(group)
            if not items or len(items) != len(reference):
                continue
            block["by_arm"][arm] = {
                "answer_overlaps_trained": summarize([items[i] for i in overlapping]),
                "new_answer": summarize(
                    [items[i] for i in range(len(items)) if i not in set(overlapping)]
                ),
            }
        report[group] = block
    return report


def decompose(summaries, attributions):
    """Read the arms against each other; this is the section to look at first."""
    report = {}

    def rec(arm, group):
        block = summaries.get(arm, {}).get(group)
        return None if not block or not block.get("count") else block["recovery_percent"]

    def count(arm, group):
        block = summaries.get(arm, {}).get(group) or {}
        return int(block.get("count") or 0)

    def tolerance(n):
        # One probe's worth of recovery points. With 50 probes a single probe is
        # 2 points, so differences at or below that are not interpreted.
        return 100.0 / n if n else float("inf")

    same = {
        "base": rec("base", "same50"),
        "v2": rec("v2", "same50"),
        "genie_exact": rec("genie_exact", "same50"),
    }
    if None not in same.values():
        tol = tolerance(count("v2", "same50"))
        same["v2_minus_genie"] = same["v2"] - same["genie_exact"]
        same["tolerance_points"] = tol
        same["reading"] = (
            "routing costs nothing measurable on trained associations; any "
            "remaining recovery is actuation -- the residual suppresses "
            "teacher-forced tokens but generation still reaches the answer"
            if abs(same["v2_minus_genie"]) <= tol
            else "routing misses on trained associations cost recovery; see "
            "attribution"
        )
    same["attribution_v2"] = attributions.get("v2", {}).get("same50")
    report["same50"] = same

    reworded = {
        "base": rec("base", "same50_paraphrase"),
        "v2": rec("v2", "same50_paraphrase"),
        "genie_exact": rec("genie_exact", "same50_paraphrase"),
    }
    v2_reworded = summaries.get("v2", {}).get("same50_paraphrase") or {}
    reworded["v2_correct_route_fraction"] = v2_reworded.get("correct_route_fraction")
    reworded["v2_route_active_fraction"] = v2_reworded.get("route_active_fraction")
    if None not in (reworded["base"], reworded["v2"], reworded["genie_exact"]):
        tol = tolerance(count("v2", "same50_paraphrase"))
        reworded["v2_minus_genie"] = reworded["v2"] - reworded["genie_exact"]
        reworded["tolerance_points"] = tol
        reworded["reading"] = (
            "routing survives rewording of trained probes; remaining recovery "
            "is actuation"
            if abs(reworded["v2_minus_genie"]) <= tol
            else "rewording costs routing: V2 misses trained associations under "
            "new wording; see attribution"
        )
    if same.get("v2") is not None and reworded["v2"] is not None:
        reworded["rewording_gap_v2"] = reworded["v2"] - same["v2"]
    reworded["attribution_v2"] = attributions.get("v2", {}).get("same50_paraphrase")
    reworded["axis"] = (
        "unseen wording of trained associations -- comparable to MCF/zsRE Gen "
        "and MQuAKE AtomicGen"
    )
    report["same50_paraphrase"] = reworded

    for group in HELDOUT_GROUPS:
        block = {
            "base": rec("base", group),
            "v2": rec("v2", group),
            "genie_subject": rec("genie_subject", group),
            "genie_subject_random": rec("genie_subject_random", group),
        }
        if None not in (block["base"], block["v2"], block["genie_subject"]):
            tol = tolerance(count("genie_subject", group))
            block["tolerance_points"] = tol
            block["transfer_available"] = block["base"] - block["genie_subject"]
            block["router_headroom"] = block["v2"] - block["genie_subject"]
            block["v2_realized"] = block["base"] - block["v2"]
            # Order matters. V2 can only apply rows the subject genie also
            # tries, so under an exact best-of genie V2 can never beat it. If
            # it does, the genie's row choice was wrong, and the other readings
            # would be built on a broken bound.
            if block["router_headroom"] < -tol:
                block["reading"] = (
                    "genie proxy failure: V2 recovers LESS than the subject "
                    "genie, which is impossible for an exact best-of bound. The "
                    "row with the lowest teacher-forced answer probability is "
                    "not the row that best stops generation, so teacher-forced "
                    "suppression does not predict recovery here -- the same gap "
                    "as zero token accuracy alongside nonzero recovery. Rerun "
                    "with --genie-select generation before reading this group."
                )
            elif block["transfer_available"] <= tol:
                block["reading"] = (
                    "coverage limit: no trained residual suppresses unlisted "
                    "facts about the person, so no router change can move this "
                    "group"
                )
            elif block["router_headroom"] > tol:
                block["reading"] = (
                    "router headroom: some same-person residual suppresses "
                    "these probes, but V2 does not fire or picks another row"
                )
            else:
                block["reading"] = (
                    "V2 already realizes the transfer this bank supports"
                )
        block["attribution_v2"] = attributions.get("v2", {}).get(group)
        report[group] = block

    neighbors = {
        "base": rec("base", "neighbors"),
        "v2": rec("v2", "neighbors"),
    }
    v2_n = summaries.get("v2", {}).get("neighbors")
    if v2_n and v2_n.get("count"):
        neighbors["v2_false_activation"] = v2_n["route_active_ci"]
    if None not in (neighbors["base"], neighbors["v2"]):
        neighbors["locality_cost_recovery_points"] = neighbors["base"] - neighbors["v2"]
        neighbors["note"] = (
            "The genie abstains on other people, so its neighbor recovery is "
            "the base value. The whole base-minus-V2 drop is router cost."
        )
    report["neighbors"] = neighbors
    return report


def analyze_existing_rows(args):
    """CPU-only: held-out answer-overlap split from a saved rows file."""
    results = json.loads(Path(args.analyze_rows).read_text())
    trained_items = next(
        (results[arm]["same50"] for arm in results if results[arm].get("same50")), None
    )
    if not trained_items:
        raise SystemExit(
            "The rows file has no same50 group; use the full decomposition run's rows"
        )
    split = overlap_split(results, trained_answers_by_person(trained_items))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "rwku_heldout_answer_overlap.json").write_text(
        json.dumps(split, indent=2, allow_nan=False) + "\n"
    )
    compact = {}
    for group, block in split.items():
        compact[group] = {
            "probes": block["probes"],
            "answer_overlaps_trained": block["answer_overlaps_trained"],
            "exact": block["exact"],
            "contained": block["contained"],
            "recovery_percent": {
                arm: {
                    part: (sub.get("recovery_percent") if sub.get("count") else None)
                    for part, sub in parts.items()
                }
                for arm, parts in block["by_arm"].items()
            },
        }
    print(json.dumps({
        "status": "answer_overlap_split_complete",
        "split": compact,
        "output": str(output / "rwku_heldout_answer_overlap.json"),
    }, indent=2))
    return 0


# ----------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--data-root", default="data/rwku")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--same50-paraphrases",
        default="",
        help="frozen JSON from build_rwku_same50_paraphrases.py; adds the "
        "same50_paraphrase group (reworded trained probes)",
    )
    parser.add_argument(
        "--analyze-rows",
        default="",
        help="existing rwku_router_decomposition_rows.json: compute the held-out "
        "answer-overlap split on CPU without loading a model",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="match the paper's RWKU evaluator so recovery numbers are comparable",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--arms", default="base,v2,genie_exact,genie_subject,genie_subject_random"
    )
    parser.add_argument("--groups", default=",".join(ALL_GROUPS))
    parser.add_argument(
        "--genie-select", choices=("teacher_forced", "generation"),
        default="teacher_forced",
    )
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--max-rows-per-group", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    if args.analyze_rows:
        return analyze_existing_rows(args)
    if not args.run_dir:
        raise SystemExit("--run-dir is required unless --analyze-rows is given")

    run_dir = Path(args.run_dir).resolve()
    manifest = json.loads((run_dir / "association_manifest.json").read_text())
    if int(manifest.get("seed", -1)) != 1:
        raise SystemExit("This registered RWKU run is seed 1 only")
    artifact = torch.load(
        run_dir / "fact_association_embeddings.pt",
        map_location="cpu",
        weights_only=False,
    )
    if str(artifact.get("architecture", "")) != "relation_prototype_fact_association_bank_v2":
        raise SystemExit("Expected a Router V2 artifact")

    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    forget_rows = list(split["efficacy_forget"])
    expected_facts, record_to_fact_id, _ = build_association_facts(
        forget_rows, tokenizer
    )
    if [str(f["association_key"]) for f in expected_facts] != [
        str(f.get("association_key")) for f in artifact["facts"]
    ]:
        raise SystemExit(
            "Saved RWKU bank no longer matches the frozen seed-1 Batch-50 split"
        )
    fact_to_row = {fact["id"]: i for i, fact in enumerate(expected_facts)}
    source_to_row = {
        source: fact_to_row[fact_id] for source, fact_id in record_to_fact_id.items()
    }
    person_rows = defaultdict(list)
    for index, fact in enumerate(artifact["facts"]):
        person_rows[int(fact["rwku_target_seed"])].append(index)

    raw_groups = {
        "same50": forget_rows,
        "heldout_level1": list(split["heldout_level1"]),
        "heldout_level2": list(split["heldout_level2"]),
        "heldout_paraphrase": list(split["heldout_paraphrase"]),
        "neighbors": [
            *_native_rows(split, "neighbor_level1.json", 1),
            *_native_rows(split, "neighbor_level2.json", 2),
        ],
    }
    wanted = [g.strip() for g in args.groups.split(",") if g.strip()]
    groups = {}
    for name in wanted:
        if name not in raw_groups:
            raise SystemExit(f"Unknown group {name}; choose from {ALL_GROUPS}")
        rows = []
        for source in raw_groups[name]:
            row = dict(source)
            row["_group"] = name
            row["_person"] = int(row["rwku_target_seed"])
            row["_gold_row"] = (
                source_to_row.get(str(row.get("source_record_sha256", "")))
                if name == "same50" else None
            )
            rows.append(row)
        if args.max_rows_per_group:
            rows = rows[: int(args.max_rows_per_group)]
        groups[name] = rows

    if args.same50_paraphrases:
        payload = json.loads(Path(args.same50_paraphrases).read_text())
        rows = []
        for source in payload["rows"]:
            origin = str(source["paraphrase_of_source_record_sha256"])
            if origin not in source_to_row:
                raise SystemExit(
                    f"Paraphrase of {origin[:12]} does not map to a trained row"
                )
            row = dict(source)
            row["_group"] = "same50_paraphrase"
            row["_person"] = int(row["rwku_target_seed"])
            row["_gold_row"] = source_to_row[origin]
            rows.append(row)
        if args.max_rows_per_group:
            rows = rows[: int(args.max_rows_per_group)]
        groups["same50_paraphrase"] = rows

    # Held-out probes must not share a row with training; if one does, the
    # subject genie would be measuring the trained association, not transfer.
    for name in HELDOUT_GROUPS:
        leaked = [
            r for r in groups.get(name, [])
            if str(r.get("source_record_sha256", "")) in source_to_row
        ]
        if leaked:
            raise SystemExit(f"{len(leaked)} {name} probes are training records")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype_from_str(args.dtype),
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    base_model.requires_grad_(False)
    base_model.config.use_cache = False

    def forced_bank():
        bank = ForcedRowBank(
            base_model,
            int(artifact["layer"]),
            artifact["rows"],
            artifact["subject_patterns"],
            artifact["facts"],
        )
        return AssociationCausalLM(base_model, bank), bank

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    rng = random.Random(int(args.seed))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = defaultdict(dict)
    genie_tables = {}

    for arm in arms:
        print(f"=== arm {arm} ===", flush=True)
        if arm == "v2":
            model, bank = load_relation_prototype_artifact(base_model, artifact)
            model.eval()
            for name, rows in groups.items():
                results[arm][name] = run_bank(
                    model, bank, tokenizer, rows, args.max_new_tokens, f"v2/{name}"
                )
            bank.close()
            continue

        model, bank = forced_bank()
        model.eval()
        if arm == "base":
            for name, rows in groups.items():
                results[arm][name] = run_bank(
                    model, bank, tokenizer, rows, args.max_new_tokens,
                    f"base/{name}", choose=lambda row: None,
                )
        elif arm == "genie_exact":
            # Only trained probes (and their rewordings) have a ground-truth
            # row. Everywhere else the exact genie abstains, which is the base
            # model, so it is not recomputed.
            for name in GOLD_GROUPS:
                if name in groups:
                    results[arm][name] = run_bank(
                        model, bank, tokenizer, groups[name], args.max_new_tokens,
                        f"genie_exact/{name}", choose=lambda row: row["_gold_row"],
                    )
        elif arm == "genie_subject":
            for name in HELDOUT_GROUPS:
                if name not in groups:
                    continue
                choices = select_genie_rows(
                    model, bank, tokenizer, groups[name], person_rows,
                    args.genie_select, args.max_new_tokens,
                )
                genie_tables[name] = {
                    key: {"chosen_row": chosen, "candidates": table}
                    for key, (chosen, table) in choices.items()
                }
                keyed = {
                    _row_key(row, i): choices[_row_key(row, i)][0]
                    for i, row in enumerate(groups[name])
                }
                index_of = {id(row): i for i, row in enumerate(groups[name])}
                results[arm][name] = run_bank(
                    model, bank, tokenizer, groups[name], args.max_new_tokens,
                    f"genie_subject/{name}",
                    choose=lambda row, keyed=keyed, index_of=index_of: keyed[
                        _row_key(row, index_of[id(row)])
                    ],
                )
        elif arm == "genie_subject_random":
            for name in HELDOUT_GROUPS:
                if name not in groups:
                    continue
                draws = {
                    id(row): (
                        rng.choice(person_rows[row["_person"]])
                        if person_rows.get(row["_person"]) else None
                    )
                    for row in groups[name]
                }
                results[arm][name] = run_bank(
                    model, bank, tokenizer, groups[name], args.max_new_tokens,
                    f"genie_subject_random/{name}",
                    choose=lambda row, draws=draws: draws[id(row)],
                )
        else:
            bank.close()
            raise SystemExit(f"Unknown arm {arm}")
        bank.close()

    summaries = {
        arm: {name: summarize(items) for name, items in by_group.items()}
        for arm, by_group in results.items()
    }
    attributions = {"v2": {}}
    if "v2" in results:
        for name, items in results["v2"].items():
            attributions["v2"][name] = (
                attribute_same50(items) if name in GOLD_GROUPS
                else attribute_heldout(items) if name in HELDOUT_GROUPS
                else None
            )

    genie_row_use = {}
    for name, table in genie_tables.items():
        chosen = [entry["chosen_row"] for entry in table.values()]
        genie_row_use[name] = {
            "distinct_rows_chosen": len({c for c in chosen if c is not None}),
            "probes": len(chosen),
        }

    report = {
        "schema_version": "rwku_router_decomposition_v1",
        "run_dir": str(run_dir),
        "dtype": args.dtype,
        "genie_select": args.genie_select,
        "arms": arms,
        "group_sizes": {name: len(rows) for name, rows in groups.items()},
        "persons": {str(k): len(v) for k, v in sorted(person_rows.items())},
        "summaries": summaries,
        "decomposition": decompose(summaries, attributions),
        "answer_overlap_split": overlap_split(
            results, trained_answers_by_person(forget_rows)
        ),
        "same50_paraphrases": (
            {
                "file": str(Path(args.same50_paraphrases).resolve()),
                "sha256": hashlib.sha256(
                    Path(args.same50_paraphrases).read_bytes()
                ).hexdigest(),
            }
            if args.same50_paraphrases else None
        ),
        "genie_row_use": genie_row_use,
        "notes": {
            "recovery": "generated-answer recovery, the paper's RWKU headline metric",
            "genie_exact": "trained probes and their rewordings only; elsewhere it equals the base model",
            "generalization_axes": (
                "same50_paraphrase = unseen wording of trained associations "
                "(like MCF/zsRE Gen, MQuAKE AtomicGen); held-out groups = other "
                "facts about the protected people (same-entity scope)"
            ),
            "genie_subject": (
                "held-out only; best of the same person's trained rows, chosen by "
                + args.genie_select
            ),
            "neighbors_genie": "abstention, i.e. the base arm",
        },
    }
    (output / "rwku_router_decomposition.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (output / "rwku_router_decomposition_rows.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    if genie_tables:
        (output / "rwku_genie_subject_candidates.json").write_text(
            json.dumps(genie_tables, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps(
        {"status": "rwku_decomposition_complete",
         "decomposition": report["decomposition"],
         "output": str(output / "rwku_router_decomposition.json")},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
