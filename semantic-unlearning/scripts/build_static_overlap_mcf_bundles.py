#!/usr/bin/env python3
"""Build real Seed-N MCF train/validation and held-out evaluation bundles.

The builder uses the repository's official MCF sampling contract:
  * forget records from the second half
  * retain records from the first half
  * Python random.sample with the declared seed

Training never uses official forget paraphrase prompts or neighborhood prompts.
Those remain held out for evaluation. MCF may contain multiple sampled case IDs
for the same underlying factual association, so training/evaluation bundles are
deduplicated by (subject, relation, target_true). Official MCF scoring still uses
the unchanged sampled cases; the evaluator compares association sets.

MCF does not provide same-subject overlap controls at useful coverage, so
same-relation/different-subject is mandatory; same-answer controls are included
when naturally available in the sampled retain pool and reported separately.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mcf_sampling import sample_official_mcf_records
from static_overlap_data import validate_bundle


def normalize_record(record):
    rr = record["requested_rewrite"]
    if isinstance(rr, list):
        rr = rr[0]
    return record, rr


def target(rr):
    value = rr["target_true"]
    return str(value["str"] if isinstance(value, dict) else value).strip()


def subject(rr):
    return str(rr["subject"]).strip()


def relation(rr):
    return str(rr["relation_id"]).strip()


def association_key(rr):
    return (
        subject(rr).casefold(),
        relation(rr).casefold(),
        target(rr).casefold(),
    )


def dedupe_associations(pairs):
    """Keep the first sampled case for each factual association."""
    out, seen = [], set()
    for rec, rr in pairs:
        key = association_key(rr)
        if key in seen:
            continue
        seen.add(key)
        out.append((rec, rr))
    return out


def direct_prompt(rr):
    template = str(rr["prompt"])
    s = subject(rr)
    if "{}" in template:
        return template.format(s).strip()
    try:
        return template.format(s).strip()
    except Exception:
        return template.strip()


def unique_texts(values):
    out, seen = [], set()
    for value in values or []:
        text = str(value).strip()
        key = " ".join(text.casefold().split())
        if text and key not in seen:
            out.append(text)
            seen.add(key)
    return out


def heldout_prompt(record, rr, forbidden):
    candidates = unique_texts(record.get("paraphrase_prompts", []))
    for prompt in candidates:
        if " ".join(prompt.casefold().split()) not in forbidden:
            return prompt
    prompt = "Held-out restatement: " + direct_prompt(rr)
    if " ".join(prompt.casefold().split()) in forbidden:
        raise ValueError(f"Could not create held-out prompt for case {record.get('case_id')}")
    return prompt


def fact_id(role, record):
    return f"{role}_{record.get('case_id')}"


def fact_row(role, record, rr):
    return {
        "id": fact_id(role, record),
        "subject": subject(rr),
        "relation": relation(rr),
        "object": target(rr),
        "role": role,
        "aliases": [],
        "answer_aliases": [],
    }


def answer_example(example_id, split, prompt, answer, fid):
    completion = " " + answer
    return {
        "id": example_id,
        "split": split,
        "prompt": prompt,
        "completion": completion,
        "spans": [{"start": 1, "end": 1 + len(answer), "fact_id": fid}],
    }


def mixed_example(example_id, split, prompt_a, answer_a, fid_a, prompt_b, answer_b, fid_b):
    prompt = f"Complete both statements. First: {prompt_a} Second: {prompt_b}"
    completion = f" {answer_a}; {answer_b}"
    second_start = len(answer_a) + 3
    return {
        "id": example_id,
        "split": split,
        "prompt": prompt,
        "completion": completion,
        "spans": [
            {"start": 1, "end": 1 + len(answer_a), "fact_id": fid_a},
            {"start": second_start, "end": second_start + len(answer_b), "fact_id": fid_b},
        ],
    }


def language_split_counts(available, requested):
    """Proportional deterministic allocation, with at least one row per split."""
    if len(requested) != 3 or any(type(n) is not int or n <= 0 for n in requested):
        raise ValueError("Language train/validation/test row counts must be positive integers")
    if available < 3:
        raise ValueError(
            f"Language corpus has {available} unique rows; at least 3 are required "
            "for disjoint train/validation/test splits. Supply a larger corpus."
        )
    need = sum(requested)
    if available >= need:
        return list(requested)
    counts = [1, 1, 1]
    for _ in range(available - 3):
        # Integer quota deficits avoid rounding differences. Ties favor the
        # earlier split; no split receives duplicated text or zero examples.
        index = max(range(3), key=lambda i: available * requested[i] - counts[i] * need)
        counts[index] += 1
    return counts


def language_rows(path, train_n=12, validation_n=6, test_n=12, *,
                  strict=False, return_summary=False):
    ds = load_from_disk(str(path))["train"]
    if "text" not in ds.column_names:
        raise ValueError("Language corpus train split must contain a text column")
    nonempty = [row["text"].strip() for row in ds
                if isinstance(row["text"], str) and row["text"].strip()]
    texts = unique_texts(nonempty)
    requested = [train_n, validation_n, test_n]
    counts = language_split_counts(len(texts), requested)
    need = sum(requested)
    if strict and len(texts) < need:
        raise ValueError(f"Language corpus has {len(texts)} unique rows, need {need}")
    train_end, validation_end = counts[0], counts[0] + counts[1]
    train = texts[:train_end]
    validation = texts[train_end:validation_end]
    test = texts[validation_end:sum(counts)]
    names = ("train", "validation", "test")
    groups = (train, validation, test)
    # The existing official PPL evaluator uses the first twenty raw rows from
    # this path. A deduplicated bundle test split does not make THAT test held out.
    official = set(unique_texts(t for t in ds["text"][:20] if isinstance(t, str)))
    official_keys = {" ".join(t.casefold().split()) for t in official}
    overlap = {name: sum(" ".join(t.casefold().split()) in official_keys for t in group)
               for name, group in zip(names, groups)}
    summary = {
        "source": str(path),
        "raw_rows": len(ds),
        "nonempty_text_rows": len(nonempty),
        "unique_rows": len(texts),
        "duplicate_rows_removed": len(nonempty) - len(texts),
        "requested_rows": dict(zip(names, requested)),
        "actual_rows": dict(zip(names, counts)),
        "reduced_to_available_rows": counts != requested,
        "split_policy": "deduplicate_then_proportional_nonempty_disjoint_splits",
        "official_ppl_first_20_overlap_rows": overlap,
        "official_ppl_held_out_from_fitting_and_validation": not (overlap["train"] or overlap["validation"]),
    }
    if return_summary:
        return train, validation, test, summary
    return train, validation, test


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--language-dir", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--eval-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--unlearn-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument("--general-retain", type=int, default=24)
    parser.add_argument("--language-train-rows", type=int, default=12)
    parser.add_argument("--language-validation-rows", type=int, default=6)
    parser.add_argument("--language-test-rows", type=int, default=12)
    parser.add_argument("--require-language-counts", action="store_true",
                        help="Fail instead of reducing row counts for a small language corpus")
    args = parser.parse_args(argv)

    lang_train, lang_val, lang_test, language_summary = language_rows(
        args.language_dir, args.language_train_rows, args.language_validation_rows,
        args.language_test_rows, strict=args.require_language_counts, return_summary=True,
    )
    if language_summary["reduced_to_available_rows"]:
        print(
            f"Language corpus has {language_summary['unique_rows']} unique texts; "
            f"using train/validation/test counts {len(lang_train)}/{len(lang_val)}/{len(lang_test)}. "
            "Counts were reduced for this small corpus; use --require-language-counts for strict runs.",
            file=sys.stderr,
        )
    if not language_summary["official_ppl_held_out_from_fitting_and_validation"]:
        print(
            "Language anchors overlap the first 20 rows used by official PPL. "
            "Use --skip-official-ppl for this corpus, or evaluate official PPL on a separate corpus.",
            file=sys.stderr,
        )

    data = json.loads(Path(args.mcf_path).read_text())
    forget_raw, retain_raw = sample_official_mcf_records(
        data, args.unlearn_num, args.retain_num, args.seed, strict=True
    )
    forget_sampled = [normalize_record(r) for r in forget_raw]
    retain_sampled = [normalize_record(r) for r in retain_raw]
    forget = dedupe_associations(forget_sampled)
    retain = dedupe_associations(retain_sampled)

    retain_by_relation = defaultdict(list)
    for rec, rr in retain:
        retain_by_relation[relation(rr)].append((rec, rr))

    relation_controls = {}
    for frec, frr in forget:
        rel = relation(frr)
        candidates = [pair for pair in retain_by_relation[rel]
                      if subject(pair[1]).casefold() != subject(frr).casefold()]
        if not candidates:
            raise ValueError(
                f"No same-relation/different-subject retain control for "
                f"case {frec.get('case_id')} relation {rel}"
            )
        relation_controls.setdefault(rel, candidates[0])

    selected = {}
    selected_keys = set()

    def add_selected(rec, rr):
        key = association_key(rr)
        if key in selected_keys:
            return
        selected_keys.add(key)
        selected[rec.get("case_id")] = (rec, rr)

    for rec, rr in relation_controls.values():
        add_selected(rec, rr)

    optional_answer_control = {}
    for frec, frr in forget:
        fa = target(frr).casefold()
        candidates = [
            (rec, rr) for rec, rr in retain
            if target(rr).casefold() == fa
            and subject(rr).casefold() != subject(frr).casefold()
            and relation(rr).casefold() != relation(frr).casefold()
        ]
        if candidates:
            optional_answer_control[frec.get("case_id")] = candidates[0]
            add_selected(*candidates[0])

    base_selected = len(selected)
    for rec, rr in retain:
        if len(selected) >= base_selected + args.general_retain:
            break
        add_selected(rec, rr)

    selected_retain = list(selected.values())

    facts = [fact_row("forget", rec, rr) for rec, rr in forget]
    facts += [fact_row("retain", rec, rr) for rec, rr in selected_retain]

    training_examples = []
    training_prompt_fingerprints = set()

    def add_training(row):
        training_examples.append(row)
        if row.get("role") != "language":
            training_prompt_fingerprints.add(" ".join(row["prompt"].casefold().split()))

    for rec, rr in forget:
        cid = rec.get("case_id")
        fid = fact_id("forget", rec)
        prompt = direct_prompt(rr)
        add_training(answer_example(f"train_forget_{cid}", "train", prompt, target(rr), fid))
        add_training(answer_example(
            f"validation_forget_{cid}", "validation",
            "Training-side validation restatement: " + prompt,
            target(rr), fid,
        ))

    for rec, rr in selected_retain:
        cid = rec.get("case_id")
        fid = fact_id("retain", rec)
        prompt = direct_prompt(rr)
        add_training(answer_example(f"train_retain_{cid}", "train", prompt, target(rr), fid))
        add_training(answer_example(
            f"validation_retain_{cid}", "validation",
            "Training-side retain validation: " + prompt,
            target(rr), fid,
        ))

    for rec, rr in forget:
        cid = rec.get("case_id")
        rrec, rrr = relation_controls[relation(rr)]
        add_training(mixed_example(
            f"train_mixed_{cid}", "train",
            direct_prompt(rr), target(rr), fact_id("forget", rec),
            direct_prompt(rrr), target(rrr), fact_id("retain", rrec),
        ))
        add_training(mixed_example(
            f"validation_mixed_{cid}", "validation",
            "Restated: " + direct_prompt(rr), target(rr), fact_id("forget", rec),
            "Restated: " + direct_prompt(rrr), target(rrr), fact_id("retain", rrec),
        ))

    for i, text in enumerate(lang_train):
        training_examples.append({"id": f"train_language_{i}", "split": "train", "role": "language", "text": text})
    for i, text in enumerate(lang_val):
        training_examples.append({"id": f"validation_language_{i}", "split": "validation", "role": "language", "text": text})

    training_bundle = {
        "schema_version": 1,
        "purpose": "training",
        "facts": facts,
        "examples": training_examples,
    }
    validate_bundle(training_bundle, "training")

    evaluation_examples = []
    fallback_count = 0
    eval_forget_prompt = {}
    eval_retain_prompt = {}

    for rec, rr in forget:
        prompt = heldout_prompt(rec, rr, training_prompt_fingerprints)
        if prompt.startswith("Held-out restatement:"):
            fallback_count += 1
        eval_forget_prompt[rec.get("case_id")] = prompt
        evaluation_examples.append(answer_example(
            f"test_forget_{rec.get('case_id')}", "test", prompt,
            target(rr), fact_id("forget", rec),
        ))

    for rec, rr in selected_retain:
        prompt = heldout_prompt(rec, rr, training_prompt_fingerprints)
        if prompt.startswith("Held-out restatement:"):
            fallback_count += 1
        eval_retain_prompt[rec.get("case_id")] = prompt
        evaluation_examples.append(answer_example(
            f"test_retain_{rec.get('case_id')}", "test", prompt,
            target(rr), fact_id("retain", rec),
        ))

    for rec, rr in forget:
        cid = rec.get("case_id")
        rrec, rrr = relation_controls[relation(rr)]
        evaluation_examples.append(mixed_example(
            f"test_mixed_{cid}", "test",
            eval_forget_prompt[cid], target(rr), fact_id("forget", rec),
            eval_retain_prompt[rrec.get("case_id")], target(rrr), fact_id("retain", rrec),
        ))

    for i, text in enumerate(lang_test):
        evaluation_examples.append({"id": f"test_language_{i}", "split": "test", "role": "language", "text": text})

    evaluation_bundle = {
        "schema_version": 1,
        "purpose": "evaluation",
        "facts": facts,
        "examples": evaluation_examples,
    }
    validate_bundle(evaluation_bundle, "evaluation")

    train_path, eval_path, summary_path = map(Path, (args.train_out, args.eval_out, args.summary_out))
    for path in (train_path, eval_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text(json.dumps(training_bundle, indent=2) + "\n")
    eval_path.write_text(json.dumps(evaluation_bundle, indent=2) + "\n")

    relation_counts = Counter(relation(rr) for _, rr in forget)
    summary = {
        "seed": args.seed,
        "official_forget_cases": len(forget_sampled),
        "unique_forget_associations": len(forget),
        "official_retain_cases": len(retain_sampled),
        "unique_retain_associations": len(retain),
        "selected_retain_facts": len(selected_retain),
        "forget_relations": dict(sorted(relation_counts.items())),
        "same_relation_control_coverage": f"{len(forget)}/{len(forget)}",
        "same_answer_optional_coverage": f"{len(optional_answer_control)}/{len(forget)}",
        "same_subject_controls": "not required; full-MCF audit found 1/50",
        "same_subject_same_answer_controls": "not required; full-MCF audit found 0/50",
        "heldout_prompt_fallbacks": fallback_count,
        "training_examples": len(training_examples),
        "evaluation_examples": len(evaluation_examples),
        "language_corpus": language_summary,
        "train_out": str(train_path),
        "eval_out": str(eval_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
