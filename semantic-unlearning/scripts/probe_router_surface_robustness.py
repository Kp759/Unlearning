#!/usr/bin/env python3
"""Surface-perturbation probe for Router V2: typos, casing, punctuation, aliases.

The point of this harness is not "does the router still fire" -- it is *which
stage failed*. Router V2 has two serial gates and the paper currently reports
only their conjunction, so a paraphrase failure is unattributable. Every
perturbed prompt here is scored at both stages independently:

  stage_a_candidate   a registered subject token sequence occurs in the prompt
                      prefix. If this is False the router provably cannot fire,
                      whatever the hidden state looks like. This is the lexical
                      ceiling.
  stage_b_qualifies   the gold association passes u >= alpha and d >= tau.
  selected_gold       the gold association is the one actually routed, after
                      the top-1/top-2 ambiguity margin.

A typo inside the subject string kills stage A and tells you nothing about the
scorer. A typo elsewhere in the prompt leaves stage A intact and isolates the
scorer's sensitivity, which is the number worth reporting. The two families
are therefore generated and tabulated separately -- `subject_*` perturbations
versus `context_*` perturbations. Reporting them pooled would be the same
confound the paper already has.

Perturbation families are deterministic given --seed, so the probe set is
reproducible and can be frozen alongside a run.

Usage
-----
python -u scripts/probe_router_surface_robustness.py \
  --model-path <llama> --artifact outputs/<run>/fact_association_embeddings.pt \
  --examples outputs/<run>/association_examples.json \
  --output-dir outputs/<run>/robustness --split development --device cuda
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
import string
import unicodedata

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from static_overlap_fact_association_embeddings import (
    _contains_subsequence,
    extract_prompt_queries,
    subject_token_patterns,
)


# Rough QWERTY adjacency: a typo model that produces plausible near-misses
# rather than uniformly random characters, which no real user types.
_ADJACENT = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}


def _alpha_positions(text):
    return [i for i, ch in enumerate(text) if ch.isalpha()]


def typo_substitute(text, rng):
    positions = _alpha_positions(text)
    if not positions:
        return text
    index = rng.choice(positions)
    character = text[index]
    options = _ADJACENT.get(character.lower())
    if not options:
        return text
    replacement = rng.choice(options)
    if character.isupper():
        replacement = replacement.upper()
    return text[:index] + replacement + text[index + 1:]


def typo_transpose(text, rng):
    positions = [i for i in _alpha_positions(text) if i + 1 < len(text)]
    positions = [i for i in positions if text[i + 1].isalpha()]
    if not positions:
        return text
    index = rng.choice(positions)
    return text[:index] + text[index + 1] + text[index] + text[index + 2:]


def typo_delete(text, rng):
    positions = _alpha_positions(text)
    if len(positions) < 3:
        return text
    index = rng.choice(positions)
    return text[:index] + text[index + 1:]


def typo_double(text, rng):
    positions = _alpha_positions(text)
    if not positions:
        return text
    index = rng.choice(positions)
    return text[:index + 1] + text[index] + text[index + 1:]


def strip_accents(text):
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def strip_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


TYPO_OPERATIONS = {
    "typo_substitute": typo_substitute,
    "typo_transpose": typo_transpose,
    "typo_delete": typo_delete,
    "typo_double": typo_double,
}


def _replace_first(prompt, subject, replacement):
    pattern = re.compile(re.escape(subject), flags=re.IGNORECASE)
    new_text, count = pattern.subn(replacement, prompt, count=1)
    return new_text if count else None


def _context_only(prompt, subject):
    """Return the prompt with the subject masked out, for context perturbation."""
    pattern = re.compile(re.escape(subject), flags=re.IGNORECASE)
    match = pattern.search(prompt)
    if not match:
        return None
    return prompt[:match.start()], match.group(0), prompt[match.end():]


def build_variants(prompt, subject, rng, repeats=2):
    """Yield (family, variant_prompt) pairs.

    `subject_*` families perturb the subject surface -- they attack Stage A.
    `context_*` families leave the subject string byte-identical and perturb
    only the surrounding request -- they attack Stage B alone.
    """
    variants = []
    variants.append(("identity", prompt))

    for name, operation in TYPO_OPERATIONS.items():
        for _ in range(int(repeats)):
            mutated = operation(subject, rng)
            if mutated != subject:
                candidate = _replace_first(prompt, subject, mutated)
                if candidate and candidate != prompt:
                    variants.append((f"subject_{name}", candidate))

    for variation, label in (
        (subject.lower(), "subject_lowercase"),
        (subject.upper(), "subject_uppercase"),
        (strip_accents(subject), "subject_unaccented"),
        (subject.split()[-1] if len(subject.split()) > 1 else subject,
         "subject_surname_only"),
        (subject.split()[0], "subject_given_only"),
        (f"{subject}'s", "subject_possessive"),
    ):
        if variation and variation != subject:
            candidate = _replace_first(prompt, subject, variation)
            if candidate and candidate != prompt:
                variants.append((label, candidate))

    split = _context_only(prompt, subject)
    if split is not None:
        head, surface, tail = split
        for name, operation in TYPO_OPERATIONS.items():
            for _ in range(int(repeats)):
                mutated_head = operation(head, rng) if head.strip() else head
                mutated_tail = operation(tail, rng) if tail.strip() else tail
                candidate = mutated_head + surface + mutated_tail
                if candidate != prompt:
                    variants.append((f"context_{name}", candidate))
        for candidate, label in (
            (head + surface + strip_punctuation(tail), "context_no_punctuation"),
            (head.lower() + surface + tail.lower(), "context_lowercase"),
            ("  ".join((head + surface + tail).split()), "context_double_space"),
            (f"Question: {head}{surface}{tail}", "context_prefixed"),
            (f"{head}{surface}{tail} Please answer briefly.",
             "context_suffixed"),
        ):
            if candidate and candidate != prompt:
                variants.append((label, candidate))

    seen = set()
    unique = []
    for family, text in variants:
        key = (family, text)
        if key in seen:
            continue
        seen.add(key)
        unique.append((family, text))
    return unique


@torch.no_grad()
def stage_a(tokenizer, prompts, subject_patterns):
    """Which associations are lexically eligible for each prompt."""
    result = []
    for prompt in prompts:
        tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        eligible = [
            index
            for index, patterns in enumerate(subject_patterns)
            if any(_contains_subsequence(tokens, pattern) for pattern in patterns)
        ]
        result.append(eligible)
    return result


@torch.no_grad()
def stage_b(model, tokenizer, prompts, layer, positive_prototypes,
            negative_prototypes, batch_size=16):
    """Return the full [P, N] u and d score matrices."""
    queries = extract_prompt_queries(
        model, tokenizer, prompts, int(layer), batch_size=int(batch_size)
    )
    queries = F.normalize(queries.float(), dim=-1)
    u_columns, d_columns = [], []
    for positive, negative in zip(positive_prototypes, negative_prototypes):
        p = F.normalize(positive.float(), dim=-1)
        n = F.normalize(negative.float(), dim=-1)
        u = (queries @ p.T).max(dim=-1).values
        v = (queries @ n.T).max(dim=-1).values
        u_columns.append(u)
        d_columns.append(u - v)
    return torch.stack(u_columns, dim=-1), torch.stack(d_columns, dim=-1)


def decide(eligible, u_row, d_row, alpha, tau, ambiguity_margin):
    """Replicate the frozen Router V2 decision exactly, on CPU tensors."""
    mask = torch.zeros(len(alpha), dtype=torch.bool)
    if eligible:
        mask[torch.tensor(eligible, dtype=torch.long)] = True
    qualifies = mask & (u_row >= alpha) & (d_row >= tau)
    ranked = d_row.masked_fill(~qualifies, float("-inf"))
    if not bool(torch.isfinite(ranked).any()):
        return None, qualifies, None
    best = int(ranked.argmax())
    count = int(qualifies.sum())
    separation = None
    if len(alpha) > 1 and count > 1:
        top2 = ranked.topk(k=2).values
        if bool(torch.isfinite(top2[1])):
            separation = float(top2[0] - top2[1])
            if separation < float(ambiguity_margin):
                return None, qualifies, separation
    return best, qualifies, separation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-prompts", type=int, default=0)
    args = parser.parse_args(argv)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.seed))

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    facts = artifact["facts"]
    layer = int(artifact["layer"])
    alpha = artifact["alpha"].float()
    tau = artifact["tau"].float()
    ambiguity_margin = float(artifact.get("ambiguity_margin", 0.02))
    positive_prototypes = artifact["positive_prototypes"]
    negative_prototypes = artifact["negative_prototypes"]
    fact_index = {str(fact["id"]): index for index, fact in enumerate(facts)}

    examples = json.loads(Path(args.examples).read_text())
    selected = [
        example for example in examples
        if str(example.get("split")) == str(args.split)
        and str(example.get("fact_id", "")) in fact_index
    ]
    if not selected:
        raise SystemExit(f"No {args.split} examples matched the artifact facts")
    if args.max_prompts:
        selected = selected[: int(args.max_prompts)]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=args.local_files_only,
        torch_dtype=torch.float32,
    ).to(args.device)
    model.eval()
    model.requires_grad_(False)

    subject_patterns = [
        subject_token_patterns(tokenizer, fact["subject"]) for fact in facts
    ]

    probes = []
    for example in selected:
        gold = fact_index[str(example["fact_id"])]
        subject = str(facts[gold]["subject"])
        for family, text in build_variants(
            str(example["prompt"]), subject, rng, repeats=args.repeats
        ):
            probes.append({
                "source_id": example.get("id"),
                "fact_id": example["fact_id"],
                "gold_index": gold,
                "family": family,
                "prompt": text,
            })

    prompts = [probe["prompt"] for probe in probes]
    eligibility = stage_a(tokenizer, prompts, subject_patterns)
    u, d = stage_b(
        model,
        tokenizer,
        prompts,
        layer,
        positive_prototypes,
        negative_prototypes,
        batch_size=args.batch_size,
    )

    for index, probe in enumerate(probes):
        gold = probe["gold_index"]
        eligible = eligibility[index]
        selected_index, qualifies, separation = decide(
            eligible, u[index], d[index], alpha, tau, ambiguity_margin
        )
        probe.update({
            "stage_a_candidate": gold in eligible,
            "stage_a_candidate_count": len(eligible),
            "stage_b_u": float(u[index, gold]),
            "stage_b_d": float(d[index, gold]),
            "stage_b_margin_over_tau": float(d[index, gold] - tau[gold]),
            "stage_b_qualifies": bool(qualifies[gold]),
            "selected_index": selected_index,
            "selected_gold": selected_index == gold,
            "fired": selected_index is not None,
            "false_route": selected_index is not None and selected_index != gold,
            "top1_top2_separation": separation,
        })

    by_family = defaultdict(list)
    for probe in probes:
        by_family[probe["family"]].append(probe)

    def summarize(rows):
        count = len(rows)
        eligible = [row for row in rows if row["stage_a_candidate"]]
        return {
            "count": count,
            "stage_a_recall": sum(r["stage_a_candidate"] for r in rows) / count,
            "stage_b_qualify_rate": sum(r["stage_b_qualifies"] for r in rows) / count,
            # Conditional rate: of the prompts that survived the lexical gate,
            # how many did the scorer confirm. This is the number that is
            # actually about the classifier.
            "stage_b_qualify_rate_given_candidate": (
                sum(r["stage_b_qualifies"] for r in eligible) / len(eligible)
                if eligible else None
            ),
            "end_to_end_gold_route_rate": (
                sum(r["selected_gold"] for r in rows) / count
            ),
            "false_route_rate": sum(r["false_route"] for r in rows) / count,
            "abstain_rate": sum(not r["fired"] for r in rows) / count,
            "mean_margin_over_tau": (
                sum(r["stage_b_margin_over_tau"] for r in rows) / count
            ),
            "routing_miss_share_of_failures": _failure_share(rows),
        }

    summary = {family: summarize(rows) for family, rows in sorted(by_family.items())}
    baseline = summary.get("identity", {}).get("end_to_end_gold_route_rate")
    for family, row in summary.items():
        row["delta_vs_identity"] = (
            row["end_to_end_gold_route_rate"] - baseline
            if baseline is not None else None
        )

    report = {
        "schema_version": "router_surface_robustness_v1",
        "artifact": str(Path(args.artifact).resolve()),
        "split": args.split,
        "seed": int(args.seed),
        "layer": layer,
        "ambiguity_margin": ambiguity_margin,
        "probe_count": len(probes),
        "families": summary,
        "subject_family_mean_gold_route": _family_mean(summary, "subject_"),
        "context_family_mean_gold_route": _family_mean(summary, "context_"),
    }
    (output / "router_surface_robustness.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (output / "router_surface_robustness_probes.json").write_text(
        json.dumps(probes, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report["families"], indent=2))
    return 0


def _failure_share(rows):
    """Of the prompts that failed to route to gold, what fraction failed at Stage A."""
    failures = [row for row in rows if not row["selected_gold"]]
    if not failures:
        return None
    return sum(not row["stage_a_candidate"] for row in failures) / len(failures)


def _family_mean(summary, prefix):
    values = [
        row["end_to_end_gold_route_rate"]
        for family, row in summary.items()
        if family.startswith(prefix)
    ]
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    raise SystemExit(main())
