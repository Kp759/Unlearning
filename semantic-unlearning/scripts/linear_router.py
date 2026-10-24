"""Learned linear router: N binary logistic classifiers as one Linear(d, N).

Drop-in replacement for the Router V2 *scorer* on MCF, ZsRE, MQuAKE and RWKU.
Everything around the scorer is kept, so a comparison against V2 changes one
thing only:

    unchanged   frozen base model, read/write layer, request-boundary position,
                prompt-prefix subject eligibility (the artifact's own subject
                patterns, so RWKU name surfaces carry over), top-1 selection,
                one residual row per association, exact base path when nothing
                fires, deterministic route
    replaced    d_i = max cos(q, P_i) - max cos(q, N_i)  and per-fact tau_i
    by          z_i = w_i . phi(q) + b_i                 and one global gate

    phi(q) = normalize(q) - mu   (no PCA by default; --pca-dims 0,64,256 restores the PCA
                                  search as an ablation; saved PCA routers still load)

Exactly N independent binary classifiers
----------------------------------------
Loss = sum over heads of a masked, per-head-normalized, class-balanced BCE;
penalty = L2 on each row of W. The objective separates by head, so the joint
Linear(d, N) fit has the same optimum as N separate logistic regressions
(tests/test_linear_router.py::test_joint_fit_equals_independent_heads).

Subject-masked loss
-------------------
At runtime head i only scores prompts containing subject i, so it is trained
only on those (prompt, i) pairs, with the runtime token-subsequence rule.
Its negatives are therefore same-subject / different-relation prompts: real
competing associations of the same subject (MQuAKE, RWKU) and subject
transplants into other relations' prompts (all benchmarks).

Two-stage fit (the decision rule is the standard p >= 0.5)
----------------------------------------------------------
    stage 1    W and b by BCE on the training templates
    stage 2    b re-fit on held-out calibration prompts with W frozen:
               b' = b - t, t the calibrated cutoff (one value, or one per
               association). The router then fires the best eligible head when
               sigmoid(w.phi + b') >= 0.5. There is no separate threshold at
               runtime; `explicit_threshold` artifacts (b and t kept apart) still
               load and give the same routes under a global cutoff.

Two gates, one set of heads
---------------------------
    threshold  (MCF, ZsRE, MQuAKE; association-level forgetting)
               fire the best eligible head if its calibrated probability is
               >= 0.5; reject ambiguous top-1/top-2 pairs.
    subject    (RWKU; entity-level forgetting)
               any prompt containing a protected subject fires; the heads only
               choose WHICH of that subject's residual rows to inject. A
               threshold gate would teach the router to abstain on unseen
               questions about the same person, which are RWKU forget probes.

Splits (training-visible only; official paraphrase/neighbor/retain never read)
-----------------------------------------------------------------------------
    fit          train positives + negatives from donor group 0
                 -> weights; L2 (and PCA, if --pca-dims lists any) by grouped CV over prompt families
    calibration  one half of the development families + donor group 1
                 -> the global threshold
    audit        other half of the development families + donor group 2
                 -> reported recall / false activation, never used to choose
Benchmarks without development prompts get context-prefix variants of their
direct prompts as extra families (prefix 0,1 -> fit; 2 -> calibration;
3 -> audit). That audit measures robustness to unseen lead-in context, not to
paraphrase; the official evaluation remains the generalization test.
"""
from __future__ import annotations

from collections import defaultdict
import math
import re

import torch
from torch import nn
from torch.nn import functional as F

from fact_association_router_v2 import _norm, _replace_subject
from mcf_synthetic_paraphrase_templates import GENERIC_CONTEXT_PREFIXES
from relation_subject_types import compatible
from static_overlap_fact_association_embeddings import (
    AssociationCausalLM,
    _contains_subsequence,
)
from static_overlap_fact_association_v2_gate import (
    RelationPrototypeAssociationBank,
)


ARCHITECTURE = "linear_classifier_fact_association_bank_v1"
ROUTER_VERSION = "subject_masked_bce_linear_router_v1"
METHOD = "static_overlap_fact_association_embeddings_linear_router"
SPLITS = ("fit", "calibration", "audit")
GATE_MODES = ("threshold", "subject")
# Which prompt positions receive the routed row at the hooked layer.
#   last          the request boundary only (the original, shipped behaviour)
#   last_subject  last token of the routed fact's subject + the boundary
#   subject_span  every token of the routed fact's subject + the boundary
#   all_prompt    every attended prompt token except BOS
#   query         the boundary and every later position (each generated / answer token), in
#                 the same forward pass: the route is decided on this layer's unedited
#                 boundary state, then the row is added here. One-pass equivalent of
#                 early_write's "query" with the write layer = the router layer.
# Routing is unchanged: the classifier always reads the boundary.
WRITE_MODES = ("last", "last_subject", "subject_span", "all_prompt", "query")
# What the routed fact does to h at the write positions.
#   add      h + Δe_i                       (the trained residual row; shipped)
#   project  h - alpha * U_i U_i^T h        (remove fact i's answer directions;
#                                            U_i orthonormal, [k, d], no training)
EDIT_MODES = ("add", "project")
# Which prompts make a head eligible (artifact["learned_subject"]["gate"]; no
# learned_subject = the exact token match, as always).
#   learned           a learned subject head (v2.2's MIL head at an early layer,
#                     scripts/fit_hybrid_subject_router.py) replaces the exact match
#   learned_or_exact  eligible if the learned head OR the exact match says so
SUBJECT_GATES = ("learned", "learned_or_exact")
# How a prompt becomes eligible for a fact's head. "exact": a complete subject-token pattern
# (shipped). "fuzzy": also the subject's words in the decoded prompt after normalisation
# (case, accents, punctuation, spacing) with up to one edit per word of >= 4 characters, so a
# typo or a case change of the name still routes. Same rule at fit time and at runtime.
SUBJECT_MATCHES = ("exact", "fuzzy")
DEFAULT_LAMBDAS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
DEFAULT_PCA_DIMS = (0,)          # no PCA: redundant for the router (was (0, 64, 256))
_SYMBOLIC_RELATION = re.compile(r"^P\d+$")
# Tiny fixed penalty on the bias only. It keeps a head whose CV fold happens to
# contain a single class from diverging; at 1e-6 it does not move a normal fit.
_BIAS_PENALTY = 1e-6


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson(successes, total, z=1.96):
    """Wilson score interval; stays inside [0, 1] near the boundaries."""
    if total == 0:
        return {"rate": None, "low": None, "high": None, "n": 0, "k": 0}
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
        "k": int(successes),
    }


def pairwise_auc(positive, negative):
    """Mann-Whitney AUC with half credit for ties."""
    if positive.numel() == 0 or negative.numel() == 0:
        return None
    comparison = positive.double()[:, None] - negative.double()[None, :]
    wins = (comparison > 0).sum() + 0.5 * (comparison == 0).sum()
    return float(wins / (positive.numel() * negative.numel()))


def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


# ---------------------------------------------------------------------------
# Eligibility (identical rule to the runtime hook)
# ---------------------------------------------------------------------------

def subject_words(text):
    """Normalised words: accents stripped, casefolded, punctuation as spaces."""
    import unicodedata

    t = unicodedata.normalize("NFKD", str(text))
    t = "".join(c for c in t if not unicodedata.combining(c)).casefold()
    return "".join(c if c.isalnum() else " " for c in t).split()


def within_one_edit(a, b):
    """Optimal-string-alignment distance <= 1: one substitution, insertion, deletion or
    adjacent transposition."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        return (len(diff) == 2 and diff[1] == diff[0] + 1
                and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def fuzzy_subject_in_text(text, words_of_subject, min_len=4):
    """The subject's words appear consecutively in `text`, each equal or (>= min_len
    characters) one edit away; or the subject is written as one word ("DanielleDarrieux")."""
    if not words_of_subject:
        return False
    words = subject_words(text)
    n = len(words_of_subject)
    for start in range(len(words) - n + 1):
        if all(w == sw or (len(sw) >= min_len and within_one_edit(w, sw))
               for w, sw in zip(words[start:start + n], words_of_subject)):
            return True
    if n > 1:
        joined = "".join(words_of_subject)
        if len(joined) >= 8 and any(within_one_edit(w, joined) for w in words):
            return True
    return False


def eligibility_matrix(tokenizer, prompts, subject_patterns, *, subjects=None, subject_match="exact"):
    """[P, N] bool: prompt p contains a complete subject-token pattern of fact i.

    Tokenizes each prompt exactly as the batched runtime call does (special
    tokens on), so the mask equals RelationPrototypeAssociationBank's
    _subject_mask on the unpadded prompt tokens. With subject_match="fuzzy" a
    prompt is also eligible when the fact's subject matches the DECODED prompt
    (special tokens skipped) under fuzzy_subject_in_text, exactly as the
    runtime hook decodes it.
    """
    if subject_match not in SUBJECT_MATCHES:
        raise ValueError(f"subject_match must be one of {SUBJECT_MATCHES}")
    mask = torch.zeros((len(prompts), len(subject_patterns)), dtype=torch.bool)
    fuzzy_words = None
    if subject_match == "fuzzy":
        if subjects is None or len(subjects) != len(subject_patterns):
            raise ValueError("fuzzy subject matching needs one subject string per fact")
        fuzzy_words = [subject_words(s) for s in subjects]
    for row, prompt in enumerate(prompts):
        tokens = list(tokenizer(prompt)["input_ids"])
        for fact_index, patterns in enumerate(subject_patterns):
            if any(_contains_subsequence(tokens, pattern) for pattern in patterns):
                mask[row, fact_index] = True
        if fuzzy_words is not None:
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            for fact_index, words in enumerate(fuzzy_words):
                if not mask[row, fact_index] and fuzzy_subject_in_text(text, words):
                    mask[row, fact_index] = True
    return mask


# ---------------------------------------------------------------------------
# Training-visible examples
# ---------------------------------------------------------------------------

def _value(example, key, default=None):
    if isinstance(example, dict):
        return example.get(key, default)
    return getattr(example, key, default)


def prompt_family(example):
    """Prompt-template family of an example.

    MCF's association_examples.json stores the family in `group`
    (canonical_rewrite, authored_k, relation_alternate_k, ...); its `role` is
    the fact role ("forget"). Rows built by examples_from_facts carry both.
    """
    return str(_value(example, "group") or _value(example, "role"))


def with_context_prefix(prompt, prefix):
    """Content-free lead-in added to a direct prompt, or None if unsafe.

    QA/chat prompts (RWKU) get the lead-in inside the question after
    "Question: "; other chat-formatted prompts are left alone; plain direct
    prompts (MCF, ZsRE, MQuAKE) are prefixed. Casing is preserved so the
    subject's token pattern survives.
    """
    marker = "Question: "
    if marker in prompt:
        head, tail = prompt.split(marker, 1)
        return f"{head}{marker}{prefix} {tail}"
    if prompt.lstrip().startswith("<|"):
        return None
    return f"{prefix} {prompt}"


def examples_from_facts(facts, *, augment=True, prefixes=None):
    """Direct-prompt examples for benchmarks without an examples file.

    Each fact's canonical_prompts become train family canonical_k. With
    augment, context prefixes 0 and 1 add train families and prefixes 2 and 3
    add two development families (calibration / audit).
    """
    prefixes = list(prefixes or GENERIC_CONTEXT_PREFIXES)
    if augment and len(prefixes) < 4:
        raise ValueError("Augmentation needs four context prefixes")
    rows = []
    for fact in facts:
        prompts = list(fact.get("canonical_prompts") or [fact["canonical_prompt"]])
        prompts = [str(p).strip() for p in prompts if str(p).strip()]
        for index, prompt in enumerate(dict.fromkeys(prompts)):
            rows.append({
                "fact_id": fact["id"], "prompt": prompt, "split": "train",
                "role": f"canonical_{index}", "group": f"canonical_{index}",
            })
            if not augment:
                continue
            for position, prefix in enumerate(prefixes[:4]):
                variant = with_context_prefix(prompt, prefix)
                if variant is None:
                    continue
                rows.append({
                    "fact_id": fact["id"],
                    "prompt": variant,
                    "split": "train" if position < 2 else "development",
                    "role": f"context_prefix_{position}",
                    "group": f"context_prefix_{position}",
                    "augmented": True,
                })
    return rows


# More content-free lead-ins, used only to top a fact's positives up to a fixed count
# (balance_positive_examples). Same style as GENERIC_CONTEXT_PREFIXES; none of them is a
# SCALPEL probe prefix (snorkel_proposal/scalpel/relations.py CONTEXT_PREFIXES).
EXTRA_CONTEXT_PREFIXES = (
    "As the records show,", "According to the archives,", "As one almanac puts it,", "Reportedly,",
    "As is generally known,", "In reference works,", "From the historical record,",
    "As documented in several sources,", "Going by standard references,", "Per the available documentation,",
    "As noted in the literature,", "In the words of one chronicle,", "As several accounts confirm,",
    "By most accounts,", "As one biography states,", "In the public record,",
)


def balance_positive_examples(examples, facts, target, *, eligible=None, prefixes=None):
    """Exactly `target` positive examples per fact (fit_linear_router --positives-per-fact).

    Kept first: the fact's own examples (canonical families, then the rest in file order).
    Topped up with context-prefix variants of its canonical prompts: prefix p of
    EXTRA_CONTEXT_PREFIXES is family context_prefix_<4+p>, a development family when
    (4+p) % 4 == 3, so about a quarter of the added positives stay held out for calibration /
    audit. Longer lists are cut keeping canonical first and a quarter development.
    eligible(prompt, fact_index) (the runtime subject rule) vets every added prompt, so none
    is dropped later. Returns (examples as dicts, report)."""
    target = int(target)
    prefixes = list(prefixes or EXTRA_CONTEXT_PREFIXES)
    by_fact = defaultdict(list)
    for example in examples:
        by_fact[str(_value(example, "fact_id"))].append(example)

    def as_dict(example):
        return {"fact_id": str(_value(example, "fact_id")), "prompt": str(_value(example, "prompt")).strip(),
                "split": str(_value(example, "split")), "group": prompt_family(example),
                "role": prompt_family(example), "augmented": bool(_value(example, "augmented", False))}

    out, report = [], {"target": target, "topped_up": 0, "cut": 0, "short": {}}
    for index, fact in enumerate(facts):
        fact_id = str(fact["id"])
        own = [as_dict(e) for e in by_fact.get(fact_id, [])]
        canonical = [e for e in own if e["group"].startswith("canonical")]
        ordered = canonical + [e for e in own if not e["group"].startswith("canonical")]
        if len(ordered) > target:
            dev = [e for e in ordered if e["split"] != "train"]
            train = [e for e in ordered if e["split"] == "train"]
            n_dev = min(len(dev), max(1, target // 4)) if dev else 0
            keep = train[:target - n_dev]
            keep += dev[:target - len(keep)]
            report["cut"] += len(ordered) - len(keep)
            out += keep
            continue
        out += ordered
        need = target - len(ordered)
        seen = {e["prompt"] for e in ordered}
        bases = [e["prompt"] for e in canonical] or [str(fact.get("canonical_prompt") or "").strip()]
        for position, prefix in enumerate(prefixes):
            family = 4 + position
            for base in bases:
                if need <= 0:
                    break
                variant = with_context_prefix(base, prefix) if base else None
                if not variant or variant.strip() in seen:
                    continue
                variant = variant.strip()
                if eligible is not None and not eligible(variant, index):
                    continue
                seen.add(variant)
                out.append({"fact_id": fact_id, "prompt": variant,
                            "split": "development" if family % 4 == 3 else "train",
                            "group": f"context_prefix_{family}", "role": f"context_prefix_{family}",
                            "augmented": True})
                need -= 1
                report["topped_up"] += 1
        if need > 0:
            report["short"][fact_id] = need
    return out, report


def _is_symbolic(relation):
    return bool(_SYMBOLIC_RELATION.match(str(relation)))


# Wikidata relations whose questions about the SAME subject can reveal the same
# or an overlapping answer. A transplant from a relation in the fact's own group
# is neither a clean positive nor a clean negative (e.g. "native language" vs
# "language used for writing" both answer "French"), so it is not used as a
# negative. The grouping is a judgement call; ablate with --no-answer-groups.
RELATION_ANSWER_GROUPS = {
    "language": ("P103", "P1412", "P37", "P364", "P407"),
    "country": ("P27", "P17", "P495"),
    "place": ("P19", "P20", "P937", "P740", "P159", "P131", "P276", "P36", "P190"),
    "work_role": ("P106", "P101", "P39"),
    "maker_owner": ("P176", "P178", "P127"),
    "affiliation": ("P108", "P463"),
}
_GROUP_OF = {
    relation: group
    for group, relations in RELATION_ANSWER_GROUPS.items()
    for relation in relations
}


def same_answer_group(relation_a, relation_b):
    a, b = str(relation_a), str(relation_b)
    if a == b:
        return True
    group = _GROUP_OF.get(a)
    return group is not None and group == _GROUP_OF.get(b)


def _negative_controls_for_fact(
    fact_index,
    facts,
    prompts_by_fact_split,
    *,
    count,
    per_donor,
    type_check,
    answer_groups,
    subject_relation_pairs,
    total=None,
):
    """Same-subject real competitors, then split-matched subject transplants.

    total: exactly this many controls when enough exist: up to half same-subject real
    competitors, the rest subject transplants (calibration needs owner-less controls), more
    real competitors if transplants run short; both taken round robin over fit / calibration /
    audit. `count` then no longer applies (fit_linear_router --negatives-per-fact).

    Relative to Router V2's builder:
      * same-subject competitors contribute their prompts from EVERY split, so
        calibration and audit also contain real same-subject negatives;
      * a transplant for split s is built from the donor's split-s prompts
        (train families for fit, the calibration / audit development families
        otherwise). Each held-out negative is then a minimal pair with a
        held-out positive: same unseen template, same subject, different
        relation. Building every negative from train templates would set the
        threshold on familiar templates and apply it to unfamiliar ones;
      * donors whose relation is protected for this subject, or (Wikidata
        relations, answer_groups=True) shares this fact's answer group, are
        skipped. V2 can transplant a same-relation donor, i.e. label a
        paraphrase of the positive as negative (fact 27 of MCF seed 1);
      * donors are subject-type-checked unless type_check=False;
      * at most `per_donor` prompts per donor, rotated across template
        families so fit negatives span all families.
    """
    fact = facts[fact_index]
    subject = str(fact["subject"])
    subject_key = _norm(subject)
    relation = str(fact.get("relation", ""))
    own = {
        prompt
        for rows in prompts_by_fact_split.get(fact["id"], {}).values()
        for prompt, _ in rows
    }
    controls, seen = [], set()
    skipped = defaultdict(int)

    for other_index, other in enumerate(facts):
        if other_index == fact_index or _norm(other["subject"]) != subject_key:
            continue
        for split, rows in prompts_by_fact_split.get(other["id"], {}).items():
            for prompt, family in rows:
                if prompt in own or prompt in seen:
                    continue
                seen.add(prompt)
                controls.append({
                    "prompt": prompt,
                    "kind": "same_subject_real",
                    "split": split,
                    "donor_index": other_index,
                    "donor_relation": str(other.get("relation", "")),
                    "group": family,
                })

    real_order = []
    if total is not None:
        # round robin over fit / calibration / audit: a plain cut keeps only the first split's
        # prompts (most of a fact's 20 positives are fit), leaving calibration without negatives
        by_split = {split: [c for c in controls if c["split"] == split] for split in SPLITS}
        while any(by_split.values()):
            for split in SPLITS:
                if by_split[split]:
                    real_order.append(by_split[split].pop(0))
        controls = real_order[: min(len(real_order), int(total) // 2)]
        count = max(0, int(total) - len(controls))
    # a fixed total: at most a third of the transplant budget per donor, so the donor rotation
    # reaches every split even when the budget is small
    take_per_donor = (int(per_donor) if total is None
                      else min(int(per_donor), max(1, -(-int(count) // len(SPLITS)))))
    donor_rank = 0
    transplants = 0
    for offset in range(1, len(facts)):
        if transplants >= int(count):
            break
        other_index = (fact_index + offset) % len(facts)
        other = facts[other_index]
        if _norm(other["subject"]) == subject_key:
            continue
        donor_relation = str(other.get("relation", ""))
        if _is_symbolic(donor_relation):
            if (subject_key, donor_relation) in subject_relation_pairs:
                skipped["donor_relation_protected_for_this_subject"] += 1
                continue
            if answer_groups and same_answer_group(relation, donor_relation):
                skipped["donor_relation_in_same_answer_group"] += 1
                continue
        if type_check and not compatible(relation, donor_relation):
            skipped["subject_type_incompatible"] += 1
            continue
        split = SPLITS[donor_rank % len(SPLITS)]
        rows = list(prompts_by_fact_split.get(other["id"], {}).get(split, []))
        if not rows:
            skipped[f"donor_without_{split}_prompts"] += 1
            continue
        start = ((fact_index + donor_rank) * int(per_donor)) % len(rows)
        rotated = rows[start:] + rows[:start]
        taken = 0
        for prompt, family in rotated:
            if taken >= take_per_donor or transplants >= int(count):
                break
            transplanted = _replace_subject(prompt, other["subject"], subject)
            if not transplanted:
                continue
            transplanted = transplanted.strip()
            if transplanted in own or transplanted in seen:
                continue
            seen.add(transplanted)
            controls.append({
                "prompt": transplanted,
                "kind": "subject_transplant",
                "split": split,
                "donor_index": other_index,
                "donor_rank": donor_rank,
                "donor_relation": donor_relation,
                "group": family,
            })
            taken += 1
            transplants += 1
        if taken:
            donor_rank += 1
        else:
            skipped["donor_without_transplantable_prompt"] += 1
    if total is not None and len(controls) < int(total):      # transplants ran short
        used = {c["prompt"] for c in controls}
        controls += [c for c in real_order if c["prompt"] not in used][: int(total) - len(controls)]
    return controls, dict(skipped)


def per_fact_counts(records, n_facts):
    """Positives (owner) and negatives (negative_for) per fact of assembled router records."""
    pos, neg = [0] * n_facts, [0] * n_facts
    for record in records:
        if record["owner"] >= 0:
            pos[record["owner"]] += 1
        for index in record["negative_for"]:
            neg[index] += 1

    def summary(xs):
        return {"min": min(xs), "max": max(xs), "mean": sum(xs) / len(xs)} if xs else {}
    return {"positives": summary(pos), "negatives": summary(neg), "positives_by_fact": pos, "negatives_by_fact": neg}


def assemble_router_dataset(
    facts,
    examples,
    tokenizer,
    subject_patterns,
    *,
    negative_count=36,
    per_donor=3,
    type_check=True,
    answer_groups=True,
    extra_controls=None,
    subject_match="exact",
    negatives_per_fact=None,
):
    """Prompts, labels, runtime eligibility and fit/calibration/audit splits.

    negatives_per_fact: every fact gets exactly this many negatives when enough exist:
    same-subject real competitors, then extra_controls, then subject transplants.

    subject_match: "exact" (shipped) or "fuzzy" (see SUBJECT_MATCHES); it must be
    the rule the runtime bank uses.

    extra_controls: optional {fact index: [control dicts]} appended to that
    fact's negatives (same keys as _negative_controls_for_fact's controls:
    prompt, kind, split, donor_relation, group), e.g. the fact's subject asked
    with other relations (fit_linear_router.py --relation-negatives).

    examples: dicts (or objects) with fact_id, prompt, split in
    {train, development}, and a prompt family in `group` (or `role`).
    A prompt lives in exactly one split. A positive keeps its example split
    (train -> fit; development families alternate calibration/audit). A
    transplant negative takes the split of the donor prompts it was built
    from, so every split's negatives share that split's templates, and donor
    ranks rotate across splits so audit negatives come from donor relations
    the head never saw while fitting. Augmented positives that lose subject
    eligibility are dropped.
    """
    if len(facts) != len(subject_patterns):
        raise ValueError("subject_patterns must align with facts")
    fact_index = {str(fact["id"]): index for index, fact in enumerate(facts)}
    development_roles = sorted({
        prompt_family(example)
        for example in examples
        if str(_value(example, "split")) != "train"
    })
    if len(development_roles) < 2:
        raise ValueError(
            "Need at least two development prompt families to separate "
            "calibration from audit (use examples_from_facts(augment=True))"
        )
    development_split = {
        role: ("calibration" if position % 2 == 0 else "audit")
        for position, role in enumerate(development_roles)
    }

    records = {}
    prompts_by_fact_split = defaultdict(lambda: defaultdict(list))
    for example in examples:
        fact_id = str(_value(example, "fact_id"))
        if fact_id not in fact_index:
            continue
        prompt = str(_value(example, "prompt")).strip()
        if not prompt:
            continue
        role = prompt_family(example)
        owner = fact_index[fact_id]
        split = (
            "fit" if str(_value(example, "split")) == "train"
            else development_split[role]
        )
        existing = records.get(prompt)
        if existing is not None:
            if existing["owner"] != owner:
                raise ValueError(f"Prompt is a positive for two facts: {prompt!r}")
            continue
        records[prompt] = {
            "prompt": prompt,
            "split": split,
            "owner": owner,
            "kind": "positive",
            "group": role,
            "augmented": bool(_value(example, "augmented", False)),
            "negative_for": [],
        }
        prompts_by_fact_split[fact_id][split].append((prompt, role))

    subject_relation_pairs = {
        (_norm(fact["subject"]), str(fact.get("relation", ""))) for fact in facts
    }
    per_fact = []
    for index, fact in enumerate(facts):
        extra = list((extra_controls or {}).get(index, []))
        total = None
        if negatives_per_fact:
            extra = extra[: int(negatives_per_fact)]
            total = int(negatives_per_fact) - len(extra)
        controls, skipped = _negative_controls_for_fact(
            index,
            facts,
            prompts_by_fact_split,
            count=negative_count,
            per_donor=per_donor,
            type_check=type_check,
            answer_groups=answer_groups,
            subject_relation_pairs=subject_relation_pairs,
            total=total,
        )
        if total is not None:          # real competitors first, then the extra controls, then transplants
            real = [c for c in controls if c["kind"] == "same_subject_real"]
            controls = real + extra + [c for c in controls if c["kind"] != "same_subject_real"]
        elif extra:
            controls = controls + extra
        added = defaultdict(int)
        for control in controls:
            prompt = control["prompt"]
            existing = records.get(prompt)
            if existing is not None:
                # A real positive of another fact, or a control already built
                # for another head: keep its split; never negate its owner.
                if existing["owner"] != index and index not in existing["negative_for"]:
                    existing["negative_for"].append(index)
                    added[f"{existing['split']}_shared"] += 1
                continue
            records[prompt] = {
                "prompt": prompt,
                "split": control["split"],
                "owner": -1,
                "kind": control["kind"],
                "group": control["group"],
                "donor_relation": control["donor_relation"],
                "augmented": False,
                "negative_for": [index],
            }
            added[control["split"]] += 1
        per_fact.append({
            "fact_id": fact["id"],
            "relation": fact.get("relation"),
            "controls": len(controls),
            "new_negatives_by_split": dict(added),
            "skipped_donors": skipped,
        })

    ordered = list(records.values())
    eligible = eligibility_matrix(
        tokenizer, [record["prompt"] for record in ordered], subject_patterns,
        subjects=[str(fact.get("subject", "")) for fact in facts], subject_match=subject_match,
    )
    keep = [
        not (
            record["owner"] >= 0
            and record["augmented"]
            and not bool(eligible[row, record["owner"]])
        )
        for row, record in enumerate(ordered)
    ]
    dropped_augmented = sum(1 for flag in keep if not flag)
    ordered = [record for record, flag in zip(ordered, keep) if flag]
    eligible = eligible[torch.tensor(keep, dtype=torch.bool)]

    owner = torch.tensor([record["owner"] for record in ordered], dtype=torch.long)
    labels = torch.zeros_like(eligible)
    has_owner = owner >= 0
    labels[has_owner.nonzero(as_tuple=True)[0], owner[has_owner]] = True

    unrouteable = [
        {"prompt": ordered[row]["prompt"], "owner": int(owner[row]),
         "split": ordered[row]["split"]}
        for row in has_owner.nonzero(as_tuple=True)[0].tolist()
        if not bool(eligible[row, owner[row]])
    ]
    ineligible_controls = sum(
        1 for row, record in enumerate(ordered)
        for index in record["negative_for"]
        if not bool(eligible[row, index])
    )
    split_mask = {
        split: torch.tensor([r["split"] == split for r in ordered], dtype=torch.bool)
        for split in SPLITS
    }
    diagnostics = {
        "prompts": len(ordered),
        "by_split": {
            split: {
                "positives": int((split_mask[split] & has_owner).sum()),
                "negative_controls": int((split_mask[split] & ~has_owner).sum()),
                "eligible_pairs": int(eligible[split_mask[split]].sum()),
                "negative_families": sorted({
                    r["group"] for r in ordered
                    if r["split"] == split and r["owner"] < 0
                }),
                "positive_families": sorted({
                    r["group"] for r in ordered
                    if r["split"] == split and r["owner"] >= 0
                }),
            }
            for split in SPLITS
        },
        "development_family_split": development_split,
        "positives_not_subject_eligible_for_owner": unrouteable,
        "augmented_positives_dropped_ineligible": dropped_augmented,
        "negative_controls_not_eligible_for_their_head": ineligible_controls,
        "negative_count_requested": int(negative_count),
        "per_donor": int(per_donor),
        "type_check": bool(type_check),
        "answer_groups": bool(answer_groups),
        "per_fact": per_fact,
        "negatives_per_fact_requested": None if not negatives_per_fact else int(negatives_per_fact),
        "per_fact_counts": per_fact_counts(ordered, len(facts)),
    }
    return {
        "records": ordered,
        "prompts": [record["prompt"] for record in ordered],
        "labels": labels,
        "eligible": eligible,
        "owner": owner,
        "split": [record["split"] for record in ordered],
        "groups": [record["group"] for record in ordered],
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Features and scoring (shared by fitting, calibration and the runtime hook)
# ---------------------------------------------------------------------------

def fit_feature_map(queries, pca_dim=0):
    """Mean (and optional PCA basis) of L2-normalized queries."""
    q = F.normalize(queries.double(), dim=-1)
    mean = q.mean(dim=0)
    components = None
    if pca_dim and int(pca_dim) < q.shape[1]:
        centred = q - mean
        rank = min(int(pca_dim), centred.shape[0])
        _, _, vh = torch.linalg.svd(centred, full_matrices=False)
        components = vh[:rank].contiguous()
    return mean, components


def apply_feature_map(queries, mean, components):
    phi = F.normalize(queries.to(mean.dtype), dim=-1) - mean
    if components is not None:
        phi = phi @ components.T
    return phi


def score_queries(queries, weight, bias, mean, components):
    """Router logits [B, N], computed in float32 exactly as the hook does."""
    phi = apply_feature_map(
        queries.float(),
        mean.float(),
        None if components is None else components.float(),
    )
    return phi @ weight.float().T + bias.float()


def _threshold_tensor(threshold, logits):
    if isinstance(threshold, torch.Tensor):
        return threshold.to(device=logits.device, dtype=logits.dtype)
    if isinstance(threshold, (list, tuple)):
        return torch.tensor(threshold, device=logits.device, dtype=logits.dtype)
    return torch.tensor(float(threshold), device=logits.device, dtype=logits.dtype)


DECISION_RULES = ("calibrated_bias", "explicit_threshold")


def fold_threshold_into_bias(bias, threshold):
    """Stage 2 of the two-stage fit: absorb the calibrated cutoff into the bias.

    z_i >= t_i  <=>  w_i . phi + (b_i - t_i) >= 0, so with b' = b - t the
    standard logistic rule sigma(z') >= 0.5 (z' >= 0) fires exactly when the
    calibrated cutoff did. `threshold` is one value (global) or [N] values
    (per association). Returns (b' as float32, the per-head shift as float32).

    Global: every bias moves by the same amount, so the top-1 ranking and the
    ambiguity margin are unchanged and routing is identical. Per association:
    each head still qualifies on exactly the same prompts, but the best head
    is now ranked by its calibrated logit z'_i = z_i - t_i.
    """
    stage1 = torch.as_tensor(bias).detach().to(torch.float64).cpu()
    shift = torch.as_tensor(threshold, dtype=torch.float64).detach().cpu()
    if shift.ndim == 0:
        shift = shift.expand_as(stage1).clone()
    if tuple(shift.shape) != tuple(stage1.shape):
        raise ValueError("per-head thresholds must match the bias shape")
    if not bool(torch.isfinite(shift).all()):
        raise ValueError("Only a finite calibrated cutoff can be folded into the bias "
                         "(the subject gate has no cutoff)")
    return (stage1 - shift).float(), shift.float()


def routing_policy_name(gate_mode, decision_rule, policy):
    prefix = "subject_eligibility_mask_plus_linear_bce_heads_top1_"
    if gate_mode == "subject":
        return prefix + "subject_gate"
    if decision_rule == "calibrated_bias":
        return prefix + f"calibrated_bias_{policy}_p_ge_0.5"
    return prefix + ("per_head_thresholds" if policy == "per_head" else "global_threshold")


def bias_calibration_record(policy, stage1_bias, shift):
    """Provenance of a calibrated bias, stored in the artifact."""
    shift = torch.as_tensor(shift).detach().float().cpu()
    uniform = bool(shift.numel()) and bool(torch.all(shift == shift[0]))
    return {
        "policy": str(policy),
        "stage1": "weights and bias fit on training templates (masked, class-balanced BCE)",
        "stage2": ("bias re-fit on held-out calibration prompts with the weights frozen: "
                   "b' = b - t, t the calibrated cutoff"),
        "runtime_rule": ("fire the best subject-eligible head if sigmoid(z') >= 0.5, "
                         "i.e. z' = w.phi + b' >= 0"),
        "shift": shift.clone(),
        "global_shift": float(shift[0]) if uniform else None,
        "stage1_bias": torch.as_tensor(stage1_bias).detach().float().cpu().clone(),
    }


def decide_routes(logits, eligible, threshold, ambiguity_margin):
    """The runtime decision rule. Returns a dict of [B] tensors.

    threshold is one value (global policy; -inf for the subject gate) or a
    [N] vector (per-head policy). A head qualifies when it is
    subject-eligible and its logit is >= its threshold. The best qualifying
    head by raw score fires (V2's ranking) unless a second qualifying head is
    within `ambiguity_margin`.
    """
    eligible = eligible.to(logits.device).bool()
    qualifies = eligible & (logits >= _threshold_tensor(threshold, logits))
    ranked = logits.masked_fill(~qualifies, float("-inf"))
    best_logit, best_fact = ranked.max(dim=-1)
    active = qualifies.any(dim=-1)
    counts = qualifies.sum(dim=-1)
    if logits.shape[-1] > 1:
        top2 = ranked.topk(k=2, dim=-1).values
        separation = top2[:, 0] - top2[:, 1]
        ambiguous = (
            (counts > 1)
            & (separation < float(ambiguity_margin))
        )
        active = active & ~ambiguous
    else:
        separation = torch.full_like(best_logit, float("inf"))
        ambiguous = torch.zeros_like(active)
    eligible_best = logits.masked_fill(~eligible, float("-inf")).max(dim=-1)
    return {
        "active": active,
        "fact": best_fact,
        "logit": best_logit,
        "qualifying": counts,
        "separation": separation,
        "ambiguous": ambiguous,
        "best_eligible_logit": eligible_best.values,
        "best_eligible_fact": eligible_best.indices,
    }


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _pair_weights(labels, eligible, balance):
    """Per-(prompt, head) weights; each head's weights sum to 1."""
    elig = eligible.double()
    positives = labels.double() * elig
    if balance:
        negatives = (1.0 - labels.double()) * elig
        n_pos = positives.sum(dim=0, keepdim=True)
        n_neg = negatives.sum(dim=0, keepdim=True)
        weight = (
            torch.where(n_pos > 0, 0.5 * positives / n_pos.clamp_min(1), 0.0)
            + torch.where(n_neg > 0, 0.5 * negatives / n_neg.clamp_min(1), 0.0)
        )
    else:
        weight = elig
    total = weight.sum(dim=0, keepdim=True)
    return torch.where(total > 0, weight / total.clamp_min(1e-12), 0.0)


def _fit_heads(phi, labels, eligible, l2, balance, max_iter, tolerance):
    """Masked BCE + L2 for all heads at once with full-batch L-BFGS.

    Strictly convex (L2 on W, tiny penalty on b): unique optimum,
    deterministic on CPU in float64.
    """
    phi = phi.double()
    n_heads = labels.shape[1]
    weight = torch.zeros(
        (n_heads, phi.shape[1]), dtype=torch.float64, device=phi.device, requires_grad=True
    )
    bias = torch.zeros(n_heads, dtype=torch.float64, device=phi.device, requires_grad=True)
    pair_weight = _pair_weights(labels, eligible, balance)
    target = labels.double()
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=int(max_iter),
        max_eval=int(max_iter) * 2,
        tolerance_grad=float(tolerance),
        tolerance_change=1e-15,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def objective():
        logits = phi @ weight.T + bias
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        data = (loss * pair_weight).sum()
        penalty = 0.5 * float(l2) * (weight * weight).sum()
        return data + penalty + 0.5 * _BIAS_PENALTY * (bias * bias).sum()

    def closure():
        optimizer.zero_grad()
        value = objective()
        value.backward()
        return value

    optimizer.step(closure)
    optimizer.zero_grad()
    final = objective()
    final.backward()
    gradient = float(torch.cat([weight.grad.flatten(), bias.grad.flatten()]).abs().max())
    state = optimizer.state[optimizer._params[0]]
    return weight.detach(), bias.detach(), {
        "objective": float(final.detach()),
        "max_abs_gradient": gradient,
        "lbfgs_iterations": int(state.get("n_iter", 0)),
        "lbfgs_function_evaluations": int(state.get("func_evals", 0)),
        "converged": gradient <= max(float(tolerance) * 100, 1e-6),
    }


def fit_linear_router(
    queries,
    labels,
    eligible,
    *,
    l2,
    pca_dim=0,
    balance=True,
    max_iter=1000,
    tolerance=1e-9,
    device=None,
):
    """Fit Linear(d', N) on the rows given. Returns float32 CPU tensors + info.

    device: where the float64 L-BFGS runs (e.g. "cuda"); None keeps the
    inputs' device. The optimum does not depend on it.
    """
    if queries.shape[0] != labels.shape[0] or labels.shape != eligible.shape:
        raise ValueError("queries, labels and eligible must align")
    if device is not None:
        queries = queries.to(device)
        labels = labels.to(device)
        eligible = eligible.to(device)
    eligible = eligible.bool()
    labels = labels.bool() & eligible
    mean, components = fit_feature_map(queries, pca_dim)
    phi = apply_feature_map(queries.double(), mean, components)
    weight, bias, info = _fit_heads(
        phi, labels, eligible, l2, balance, max_iter, tolerance
    )
    logits = phi @ weight.T + bias
    errors = int(((logits >= 0) != labels)[eligible].sum())
    parameters = int(weight.numel() + bias.numel())
    info.update({
        "l2": float(l2),
        "pca_dim": int(0 if components is None else components.shape[0]),
        "feature_dim": int(phi.shape[1]),
        "balance": bool(balance),
        "heads": int(labels.shape[1]),
        "eligible_pairs": int(eligible.sum()),
        "positive_pairs": int(labels.sum()),
        "training_errors_at_logit_zero": errors,
        "linearly_separated_on_fit": errors == 0,
        "heads_without_positive": (~labels.any(dim=0)).nonzero(as_tuple=True)[0].tolist(),
        "heads_without_negative": (
            ~(eligible & ~labels).any(dim=0)
        ).nonzero(as_tuple=True)[0].tolist(),
        "router_parameters": parameters,
        "fit_pairs_per_parameter": float(eligible.sum()) / float(parameters),
    })
    info["fit_device"] = str(phi.device)
    return {
        "weight": weight.float().cpu(),
        "bias": bias.float().cpu(),
        "feature_mean": mean.float().cpu(),
        "feature_components": None if components is None else components.float().cpu(),
        "info": info,
    }


def _balanced_log_loss(logits, labels, eligible):
    positives = logits[eligible & labels]
    negatives = logits[eligible & ~labels]
    parts = []
    if positives.numel():
        parts.append(float(F.softplus(-positives.double()).mean()))
    if negatives.numel():
        parts.append(float(F.softplus(negatives.double()).mean()))
    return sum(parts) / len(parts) if parts else None


def select_hyperparameters(
    queries,
    labels,
    eligible,
    groups,
    *,
    lambdas=DEFAULT_LAMBDAS,
    pca_dims=DEFAULT_PCA_DIMS,
    folds=5,
    balance=True,
    max_iter=1000,
    device=None,
    progress=None,
):
    """Pick one (L2, PCA) pair shared by all heads by grouped CV.

    Folds hold out whole prompt families (round-robin over sorted family
    names), so each score is transfer to an unseen phrasing. Criterion:
    class-balanced held-out log-loss over eligible pairs (a proper scoring
    rule; AUC saturates at 1.0 on separable data). Ties prefer stronger L2,
    then smaller PCA (0 = none counts as largest).
    """
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("Grouped CV needs at least two prompt families in fit")
    folds = max(2, min(int(folds), len(unique)))
    fold_of = {group: position % folds for position, group in enumerate(unique)}
    assignment = torch.tensor([fold_of[g] for g in groups])
    hidden = queries.shape[1]
    table = []
    for pca_dim in pca_dims:
        for l2 in lambdas:
            losses, positives, negatives = [], [], []
            for fold in range(folds):
                held = assignment == fold
                if bool(held.all()) or not bool(held.any()):
                    continue
                model = fit_linear_router(
                    queries[~held], labels[~held], eligible[~held],
                    l2=l2, pca_dim=pca_dim, balance=balance, max_iter=max_iter,
                    device=device,
                )
                logits = score_queries(
                    queries[held], model["weight"], model["bias"],
                    model["feature_mean"], model["feature_components"],
                )
                held_labels = labels[held].bool()
                held_eligible = eligible[held].bool()
                loss = _balanced_log_loss(logits, held_labels, held_eligible)
                if loss is not None:
                    losses.append(loss)
                positives.append(logits[held_eligible & held_labels])
                negatives.append(logits[held_eligible & ~held_labels])
            table.append({
                "l2": float(l2),
                "pca_dim": int(pca_dim),
                "held_out_balanced_log_loss": (
                    sum(losses) / len(losses) if losses else None
                ),
                "held_out_pair_auc": pairwise_auc(
                    torch.cat(positives) if positives else torch.empty(0),
                    torch.cat(negatives) if negatives else torch.empty(0),
                ),
                "folds_used": len(losses),
            })
            if progress is not None:
                progress(table[-1], len(table), len(pca_dims) * len(lambdas))
    scored = [r for r in table if r["held_out_balanced_log_loss"] is not None]
    if not scored:
        raise ValueError("Grouped CV produced no scorable folds")

    def key(row):
        effective_dim = row["pca_dim"] if row["pca_dim"] else hidden
        return (round(row["held_out_balanced_log_loss"], 10), -row["l2"], effective_dim)

    best = min(scored, key=key)
    return float(best["l2"]), int(best["pca_dim"]), {
        "criterion": "held_out_class_balanced_log_loss",
        "folds": folds,
        "fold_of_group": fold_of,
        "table": table,
        "selected_l2": float(best["l2"]),
        "selected_pca_dim": int(best["pca_dim"]),
        "l2_at_grid_edge": float(best["l2"]) in (float(min(lambdas)), float(max(lambdas))),
    }


# ---------------------------------------------------------------------------
# Calibration and route evaluation
# ---------------------------------------------------------------------------

def route_outcomes(logits, eligible, owner, threshold, ambiguity_margin, facts=None):
    """Route-level outcomes for positives (owner >= 0) and negatives (-1)."""
    decision = decide_routes(logits, eligible, threshold, ambiguity_margin)
    active = decision["active"].cpu()
    chosen = decision["fact"].cpu()
    owner = owner.cpu()
    positive = owner >= 0
    negative = ~positive
    correct = positive & active & (chosen == owner)
    wrong = positive & active & (chosen != owner)
    result = {
        "correct_route": wilson(int(correct.sum()), int(positive.sum())),
        "wrong_row_on_positive": wilson(int(wrong.sum()), int(positive.sum())),
        "abstain_on_positive": wilson(int((positive & ~active).sum()), int(positive.sum())),
        "false_activation_on_negative_control": wilson(
            int((negative & active).sum()), int(negative.sum())
        ),
        "ambiguity_rejections": int(decision["ambiguous"].sum()),
    }
    if facts is not None:
        subjects = [_norm(fact["subject"]) for fact in facts]
        same_subject = torch.tensor([
            bool(p) and bool(a) and subjects[int(c)] == subjects[int(o)]
            for p, a, c, o in zip(
                positive.tolist(), active.tolist(), chosen.tolist(), owner.tolist()
            )
        ], dtype=torch.bool)
        result["correct_subject_row_on_positive"] = wilson(
            int(same_subject.sum()), int(positive.sum())
        )
    return result


def _threshold_candidates(logits, eligible, max_candidates):
    """Unique eligible logits (capped by quantiles) plus one that fires nothing.

    Candidates stay in the logits' own dtype: the runtime compares float32
    logits with the threshold, so a float64 "just above the max" would round
    back onto the max and still fire it.
    """
    values = logits[eligible.bool()].unique()
    if values.numel() == 0:
        raise ValueError("Split has no eligible pairs")
    if max_candidates and values.numel() > int(max_candidates):
        positions = torch.linspace(0, values.numel() - 1, int(max_candidates)).round().long()
        values = values[positions].unique()
    ceiling = torch.nextafter(values.max(), torch.tensor(float("inf"), dtype=values.dtype))
    return torch.cat([values, ceiling.reshape(1)]).tolist()


def _sweep(decide, candidates, owner):
    """(threshold, false_activation, correct_routes, wrong_rows) per candidate."""
    negative = owner < 0
    positive = ~negative
    rows = []
    for candidate in candidates:
        active, chosen = decide(candidate)
        active, chosen = active.cpu(), chosen.cpu()
        fpr = float((active & negative).sum()) / max(float(negative.sum()), 1.0)
        correct = int((positive & active & (chosen == owner)).sum())
        wrong = int((positive & active & (chosen != owner)).sum())
        rows.append((candidate, fpr, correct, wrong))
    return rows


def calibrate_threshold(
    logits,
    eligible,
    owner,
    *,
    target_fpr=0.0,
    min_recall=None,
    ambiguity_margin=0.5,
    placement="midpoint",
    placement_fraction=0.5,
    max_candidates=2000,
):
    """One global logit threshold chosen on the calibration split.

    The whole rule (eligibility, threshold, ambiguity) is evaluated at every
    candidate, because ambiguity rejection makes the curves non-monotone.
      target_fpr  (default) most correct routes with false activation on
                  negative controls <= target_fpr
      min_recall  (recall-first; overrides target_fpr) lowest false activation
                  with correct-route recall >= min_recall, then most correct
    Within the chosen set the threshold is placed by `placement`:
      midpoint  middle of the equivalent-threshold interval (max-margin);
      high / low  its top / bottom end.
    Any threshold in (previous candidate, low] makes the same decisions as
    `low`, so the midpoint is taken over (previous, high].
    """
    if placement not in ("midpoint", "high", "low"):
        raise ValueError("placement must be midpoint, high or low")
    if not 0.0 <= float(placement_fraction) <= 1.0:
        raise ValueError("placement_fraction must be in [0,1]")
    eligible = eligible.bool()
    owner = owner.cpu()
    negative = owner < 0
    positive = ~negative
    if not bool(negative.any()):
        raise ValueError("Calibration split has no negative controls")
    n_pos = max(int(positive.sum()), 1)
    candidates = _threshold_candidates(logits, eligible, max_candidates)

    def decide(candidate):
        decision = decide_routes(logits, eligible, candidate, ambiguity_margin)
        return decision["active"], decision["fact"]

    sweep = _sweep(decide, candidates, owner)

    def evaluate(candidate):
        return _sweep(decide, [candidate], owner)[0]

    def place(pool, fpr_limit, correct_needed):
        interval = sorted(row[0] for row in pool)
        low, high = interval[0], interval[-1]
        if placement == "low":
            return evaluate(low)
        if placement == "high":
            return evaluate(high)
        position = candidates.index(low)
        previous = candidates[position - 1] if position > 0 else low - 1.0
        middle = evaluate(float(torch.tensor((1.0 - float(placement_fraction)) * previous + float(placement_fraction) * high, dtype=logits.dtype)))
        if middle[1] <= fpr_limit + 1e-12 and middle[2] == correct_needed:
            return middle
        return evaluate(high)

    def by_fpr(target):
        admissible = [row for row in sweep if row[1] <= float(target) + 1e-12]
        best = max(row[2] for row in admissible)
        pool = [row for row in admissible if row[2] == best]
        return place(pool, float(target), best), True

    def by_recall(target):
        need = math.ceil(float(target) * n_pos - 1e-9)
        admissible = [row for row in sweep if row[2] >= need]
        met = bool(admissible)
        if not admissible:
            best = max(row[2] for row in sweep)
            admissible = [row for row in sweep if row[2] == best]
        lowest = min(row[1] for row in admissible)
        pool = [row for row in admissible if row[1] <= lowest + 1e-12]
        best = max(row[2] for row in pool)
        pool = [row for row in pool if row[2] == best]
        return place(pool, lowest, best), met

    if min_recall is not None:
        (threshold, fpr, correct, _), met = by_recall(min_recall)
        rule = "min_false_activation_subject_to_calibration_recall"
    else:
        (threshold, fpr, correct, _), met = by_fpr(target_fpr)
        rule = "max_correct_routes_subject_to_calibration_fpr"

    curve = []
    for target in sorted({0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}):
        (t, f, c, _), _ = by_fpr(target)
        curve.append({"target_fpr": target, "threshold_logit": t,
                      "calibration_fpr": f, "calibration_recall": c / n_pos})
    for target in (0.9, 0.95, 0.98, 1.0):
        (t, f, c, _), ok = by_recall(target)
        curve.append({"min_recall": target, "threshold_logit": t,
                      "calibration_fpr": f, "calibration_recall": c / n_pos,
                      "recall_target_met": ok})
    outcome = route_outcomes(logits, eligible, owner, threshold, ambiguity_margin)
    note = None
    if min_recall is None and 0.0 < float(target_fpr) < 1.0 / float(negative.sum()):
        note = (f"target_fpr {target_fpr} is below the split's resolution "
                f"1/{int(negative.sum())}; it acts as zero")
    return float(threshold), {
        "rule": rule,
        "placement": placement,
        "placement_fraction": float(placement_fraction),
        "target_fpr": None if min_recall is not None else float(target_fpr),
        "min_recall": None if min_recall is None else float(min_recall),
        "recall_target_met": met if min_recall is not None else None,
        "ambiguity_margin": float(ambiguity_margin),
        "threshold_logit": float(threshold),
        "threshold_probability": float(torch.sigmoid(torch.tensor(threshold))),
        "calibration_fpr": fpr,
        "calibration_recall": correct / n_pos,
        "candidates": len(candidates),
        "resolution_note": note,
        "operating_curve": curve,
        "calibration_outcomes_optimistic": outcome,
        "note": ("Calibration outcomes are measured on the split that chose "
                 "the threshold; report the audit split."),
    }


def _frontier_summary(sweep, n_pos, reference=None):
    """Best recall at FPR budgets, best FPR at recall floors, route AUC."""
    points = sorted({(row[1], row[2] / n_pos) for row in sweep})
    envelope, best = [], -1.0
    for fpr, recall in points:
        if recall > best:
            envelope.append((fpr, recall))
            best = recall
    area, last_fpr, last_recall = 0.0, 0.0, 0.0
    for fpr, recall in envelope:
        area += (fpr - last_fpr) * last_recall
        last_fpr, last_recall = fpr, recall
    area += (1.0 - last_fpr) * last_recall

    def recall_at(budget):
        values = [row[2] / n_pos for row in sweep if row[1] <= budget + 1e-12]
        return max(values) if values else 0.0

    def fpr_at(floor):
        values = [row[1] for row in sweep if row[2] / n_pos >= floor - 1e-12]
        return min(values) if values else None

    summary = {
        "route_auc": area,
        "recall_at_fpr": {str(b): recall_at(b) for b in (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)},
        "fpr_at_recall": {str(r): fpr_at(r) for r in (0.8, 0.9, 0.95, 0.98, 1.0)},
    }
    if reference is not None:
        ref_fpr, ref_recall = reference
        summary["at_reference_fpr"] = {"fpr": ref_fpr, "recall": recall_at(ref_fpr)}
        summary["at_reference_recall"] = {"recall": ref_recall, "fpr": fpr_at(ref_recall)}
    return summary


def linear_route_frontier(logits, eligible, owner, ambiguity_margin, *, reference=None,
                          max_candidates=2000):
    """Threshold sweep of the linear router on one split (e.g. audit).

    Choosing a threshold on the audit split is an oracle; this is a curve for
    comparing routers at matched operating points, not an operating point.
    """
    owner = owner.cpu()
    candidates = _threshold_candidates(logits, eligible, max_candidates)
    sweep = _sweep(
        lambda c: (lambda d: (d["active"], d["fact"]))(
            decide_routes(logits, eligible, c, ambiguity_margin)
        ),
        candidates,
        owner,
    )
    return _frontier_summary(sweep, max(int((owner >= 0).sum()), 1), reference)


def _v2_scores(queries, artifact):
    q = F.normalize(queries.float(), dim=-1)
    u_columns, d_columns = [], []
    for positive, negative in zip(
        artifact["positive_prototypes"], artifact["negative_prototypes"]
    ):
        p = F.normalize(positive.float(), dim=-1)
        n = F.normalize(negative.float(), dim=-1)
        u = (q @ p.T).max(dim=-1).values
        u_columns.append(u)
        d_columns.append(u - (q @ n.T).max(dim=-1).values)
    return torch.stack(u_columns, dim=-1), torch.stack(d_columns, dim=-1)


def _v2_decide(u, d, eligible, artifact, shift=0.0):
    """Router V2's rule; shift moves every tau_i by the same amount."""
    qualifies = (
        eligible.bool()
        & (u >= artifact["alpha"].float()[None, :])
        & (d - artifact["tau"].float()[None, :] >= float(shift))
    )
    ranked = d.masked_fill(~qualifies, float("-inf"))
    best, fact = ranked.max(dim=-1)
    active = torch.isfinite(best)
    if d.shape[-1] > 1:
        top2 = ranked.topk(k=2, dim=-1).values
        margin = float(artifact.get("ambiguity_margin", 0.02))
        ambiguous = (
            (qualifies.sum(dim=-1) > 1)
            & torch.isfinite(top2[:, 1])
            & ((top2[:, 0] - top2[:, 1]) < margin)
        )
        active = active & ~ambiguous
    return active, fact


def prototype_router_routes(queries, eligible, artifact):
    """Router V2's shipped decision on the same queries."""
    u, d = _v2_scores(queries, artifact)
    return _v2_decide(u, d, eligible, artifact)


def v2_route_outcomes(queries, eligible, owner, artifact):
    active, chosen = prototype_router_routes(queries, eligible, artifact)
    positive = owner >= 0
    negative = ~positive
    return {
        "correct_route": wilson(
            int((positive & active & (chosen == owner)).sum()), int(positive.sum())
        ),
        "wrong_row_on_positive": wilson(
            int((positive & active & (chosen != owner)).sum()), int(positive.sum())
        ),
        "abstain_on_positive": wilson(int((positive & ~active).sum()), int(positive.sum())),
        "false_activation_on_negative_control": wilson(
            int((negative & active).sum()), int(negative.sum())
        ),
    }


def v2_route_frontier(queries, eligible, owner, artifact, *, max_candidates=2000):
    """V2 swept by shifting every tau_i together (shift 0 = shipped V2)."""
    owner = owner.cpu()
    u, d = _v2_scores(queries, artifact)
    margins = (d - artifact["tau"].float()[None, :])
    candidates = _threshold_candidates(margins, eligible, max_candidates)
    candidates = sorted(set(candidates) | {0.0})
    sweep = _sweep(lambda s: _v2_decide(u, d, eligible, artifact, s), candidates, owner)
    return _frontier_summary(sweep, max(int((owner >= 0).sum()), 1))


def calibrate_per_head(
    scores,
    eligible,
    owner,
    *,
    fraction=0.1,
    slack=0.5,
    shrink=0.0,
    fallback,
    ceiling=None,
):
    """Association-specific thresholds from held-out calibration prompts.

    For head i with calibration positives P_i (its own prompts) and negatives
    N_i (eligible prompts it does not own, including other same-subject
    heads' prompts):
      separable       t_i = max(N_i) + fraction * (min(P_i) - max(N_i))
                      (fraction 0.1 is Router V2's tau rule, here on held-out
                      data rather than on the prototypes' own prompts)
      non-separable   t_i = min(P_i) - slack      (V2: recall-first)
      no negatives    t_i = min(P_i) - slack
      no positives    t_i = fallback              (the global threshold)
    shrink > 0 pulls t_i toward the fallback with weight m_i / (m_i + shrink),
    m_i = number of calibration prompts for head i (a regularised variant).
    ceiling ([N], optional): t_i is capped at ceiling_i. Pass the lowest score
    of each head's own TRAINING positives minus a small epsilon so a fact's
    own direct prompt always routes (Router V2 enforces the same invariant,
    and the MQuAKE evaluator requires it).
    Works for any score: linear logits or V2's cosine margin d.
    """
    scores = scores.float()
    eligible = eligible.bool().to(scores.device)
    owner = owner.to(scores.device)
    n_heads = scores.shape[1]
    thresholds = torch.full((n_heads,), float(fallback), dtype=scores.dtype)
    per_head, counts = [], defaultdict(int)
    for head in range(n_heads):
        column = eligible[:, head]
        pos = scores[column & (owner == head), head]
        neg = scores[column & (owner != head), head]
        if pos.numel() == 0:
            rule, value = "fallback_no_calibration_positive", float(fallback)
        elif neg.numel() == 0:
            rule, value = "positive_floor_minus_slack_no_negative", float(pos.min()) - float(slack)
        else:
            p, n = pos.min(), neg.max()
            if bool(p > n):
                rule = "negative_ceiling_plus_fraction_of_gap"
                value = float(n + float(fraction) * (p - n))
                # stay strictly above the hardest negative after float32 rounding
                value = max(value, float(torch.nextafter(n, torch.tensor(float("inf")))))
            else:
                rule = "positive_floor_minus_slack_nonseparable"
                value = float(p) - float(slack)
        m = int(pos.numel() + neg.numel())
        if float(shrink) > 0 and rule != "fallback_no_calibration_positive":
            value = (m * value + float(shrink) * float(fallback)) / (m + float(shrink))
        capped = False
        if ceiling is not None and bool(torch.isfinite(ceiling[head])):
            if value > float(ceiling[head]):
                value, capped = float(ceiling[head]), True
                counts["capped_at_training_positive"] += 1
        thresholds[head] = value
        counts[rule] += 1
        per_head.append({
            "head": head, "rule": rule, "threshold": value,
            "capped_at_training_positive": capped,
            "calibration_positives": int(pos.numel()),
            "calibration_negatives": int(neg.numel()),
            "positive_floor": float(pos.min()) if pos.numel() else None,
            "negative_ceiling": float(neg.max()) if neg.numel() else None,
        })
    return thresholds, {
        "policy": "per_head",
        "fraction": float(fraction),
        "slack": float(slack),
        "shrink": float(shrink),
        "fallback_threshold": float(fallback),
        "rule_counts": dict(counts),
        "ceiling_applied": ceiling is not None,
        "per_head": per_head,
    }


def training_positive_floor(scores, eligible, owner, epsilon=1e-3):
    """[N] lowest score of each head's own (eligible) positives, minus epsilon."""
    scores = scores.float()
    n_heads = scores.shape[1]
    floor = torch.full((n_heads,), float("inf"))
    for head in range(n_heads):
        own = scores[eligible[:, head].bool() & (owner == head), head]
        own = own[torch.isfinite(own)]
        if own.numel():
            floor[head] = float(own.min()) - float(epsilon)
    return floor


def v2_effective_scores(queries, artifact):
    """V2's margin d_i with its alpha condition folded in (-inf where u < alpha)."""
    u, d = _v2_scores(queries, artifact)
    alpha = artifact["alpha"].float()[None, :]
    return d.masked_fill(u < alpha, float("-inf"))


def v2_route_outcomes_with_tau(queries, eligible, owner, artifact, tau):
    """Outcomes of V2's runtime rule after replacing tau (global or per head)."""
    patched = dict(artifact)
    tau = torch.as_tensor(tau, dtype=torch.float32)
    patched["tau"] = tau.expand(len(artifact["facts"])).clone() if tau.ndim == 0 else tau
    active, chosen = prototype_router_routes(queries, eligible, patched)
    positive = owner >= 0
    negative = ~positive
    return {
        "correct_route": wilson(int((positive & active & (chosen == owner)).sum()), int(positive.sum())),
        "wrong_row_on_positive": wilson(int((positive & active & (chosen != owner)).sum()), int(positive.sum())),
        "abstain_on_positive": wilson(int((positive & ~active).sum()), int(positive.sum())),
        "false_activation_on_negative_control": wilson(int((negative & active).sum()), int(negative.sum())),
    }


def cosine_arm_artifact(source, tau, *, arm, calibration):
    """A Router V2 artifact whose tau is replaced by a held-out calibration.

    Same prototypes, rows, subject patterns and runtime bank as the source;
    loads with the unchanged V2 evaluators.
    """
    artifact = dict(source)
    tau = torch.as_tensor(tau, dtype=torch.float32)
    artifact["tau"] = tau.expand(len(source["facts"])).clone() if tau.ndim == 0 else tau.clone()
    artifact["router_arm"] = arm
    artifact["tau_source"] = "held_out_calibration"
    artifact["tau_calibration"] = calibration
    return artifact


# ---------------------------------------------------------------------------
# Runtime bank
# ---------------------------------------------------------------------------

class LinearClassifierAssociationBank(nn.Module):
    """One residual row per fact, routed by N subject-masked logistic heads.

    Same runtime contract as RelationPrototypeAssociationBank: forward hook on
    base_model.model.layers[layer], bind()/unbind(), one edited position (the
    request boundary), exact base path when nothing fires, same telemetry.
    The subject-eligibility and boundary helpers are V2's own functions.
    """

    _prefix_lengths_for = RelationPrototypeAssociationBank._prefix_lengths_for
    _subject_mask = RelationPrototypeAssociationBank._subject_mask
    bind = RelationPrototypeAssociationBank.bind
    unbind = RelationPrototypeAssociationBank.unbind
    extra = RelationPrototypeAssociationBank.extra

    def counters(self):
        out = RelationPrototypeAssociationBank.counters(self)
        if self.learned_subject is not None:
            out["learned_subject_gate"] = dict(self.gate_stats)
        return out

    def close(self):
        self._hook_handle.remove()
        if self._subject_hook is not None:
            self._subject_hook.remove()
            self._subject_hook = None

    def __init__(
        self,
        base_model,
        layer,
        weight,
        bias,
        feature_mean,
        feature_components,
        threshold,
        subject_patterns,
        facts,
        rows=None,
        ambiguity_margin=0.5,
        gate_mode="threshold",
        router_fit=None,
        per_head_thresholds=None,
        bias_calibration=None,
        head_index=None,
        write_mode="last",
        edit_mode="add",
        project_dirs=None,
        project_alpha=1.0,
        learned_subject=None,
        subject_match="exact",
        free_fire_logit=None,
    ):
        super().__init__()
        if str(subject_match) not in SUBJECT_MATCHES:
            raise ValueError(f"subject_match must be one of {SUBJECT_MATCHES}")
        self.subject_match = str(subject_match)
        # fuzzy matching decodes the prompt: set bank.tokenizer after loading
        self.tokenizer = None
        self._fuzzy_words = [subject_words(str(f.get("subject", ""))) for f in facts]
        # experimental: with no eligible subject, a head may still fire at logit >= this
        self.free_fire_logit = None if free_fire_logit is None else float(free_fire_logit)
        # True skips the hook entirely (cached decoding of tokens after the request boundary)
        self.passthrough = False
        self.learned_subject = None
        self._subject_hook = None
        self._subject_states = None
        if str(write_mode) not in WRITE_MODES:
            raise ValueError(f"write_mode must be one of {WRITE_MODES}")
        self.write_mode = str(write_mode)
        if str(edit_mode) not in EDIT_MODES:
            raise ValueError(f"edit_mode must be one of {EDIT_MODES}")
        self.edit_mode = str(edit_mode)
        self.project_alpha = float(project_alpha)
        if self.edit_mode == "project":
            if project_dirs is None:
                raise ValueError("edit_mode='project' needs project_dirs [n_facts, k, d]")
            project_dirs = torch.as_tensor(project_dirs, dtype=torch.float32)
            if project_dirs.ndim != 3 or project_dirs.shape[0] != len(facts):
                raise ValueError("project_dirs must be [n_facts, k, hidden]")
            gram = project_dirs @ project_dirs.transpose(1, 2)
            live = project_dirs.norm(dim=-1) > 0
            eye = torch.diag_embed(live.float())
            if not torch.allclose(gram, eye, atol=1e-4):
                raise ValueError("project_dirs must be orthonormal per fact (zero rows allowed)")
        elif project_dirs is not None:
            raise ValueError("project_dirs given but edit_mode is 'add'")
        config = getattr(base_model, "config", None)
        bos = getattr(config, "bos_token_id", None)
        self.bos_token_ids = (
            set() if bos is None
            else {int(b) for b in bos} if isinstance(bos, (list, tuple)) else {int(bos)}
        )
        self.subject_fallbacks = 0
        n_facts = len(facts)
        # head_index: shared heads (e.g. one per relation). weight/bias hold
        # one row per HEAD and fact i uses head head_index[i]. None keeps the
        # original one-head-per-fact layout, unchanged.
        if head_index is not None:
            head_index = torch.as_tensor(head_index, dtype=torch.long)
            if tuple(head_index.shape) != (n_facts,):
                raise ValueError("head_index must have one entry per fact")
            if int(head_index.min()) < 0 or int(head_index.max()) >= weight.shape[0]:
                raise ValueError("head_index points outside the shared heads")
            if per_head_thresholds is not None:
                raise ValueError("shared heads use the global threshold only")
        n_heads = n_facts if head_index is None else int(weight.shape[0])
        if bias_calibration is not None:
            # The calibrated cutoff already lives in the bias: standard rule only.
            if gate_mode != "threshold":
                raise ValueError("a calibrated bias needs the threshold gate")
            if per_head_thresholds is not None or float(threshold) != 0.0:
                raise ValueError(
                    "a calibrated-bias router fires at logit >= 0 (p >= 0.5); "
                    "pass threshold=0.0 and no per-head thresholds"
                )
        if len(subject_patterns) != n_facts:
            raise ValueError("subject_patterns must align with facts")
        if weight.ndim != 2 or weight.shape[0] != n_heads:
            raise ValueError("weight must be [num_heads, feature_dim]")
        if tuple(bias.shape) != (n_heads,):
            raise ValueError("bias must be [num_heads]")
        hidden_size = int(feature_mean.shape[-1])
        if feature_components is not None and int(feature_components.shape[1]) != hidden_size:
            raise ValueError("feature_components must be [k, hidden_size]")
        feature_dim = hidden_size if feature_components is None else int(feature_components.shape[0])
        if int(weight.shape[1]) != feature_dim:
            raise ValueError("weight feature dimension does not match the feature map")
        if str(gate_mode) not in GATE_MODES:
            raise ValueError(f"gate_mode must be one of {GATE_MODES}")
        if gate_mode == "threshold" and not math.isfinite(float(threshold)):
            raise ValueError("threshold gate needs a finite threshold")
        if per_head_thresholds is not None:
            per_head_thresholds = torch.as_tensor(per_head_thresholds, dtype=torch.float32)
            if tuple(per_head_thresholds.shape) != (n_facts,):
                raise ValueError("per_head_thresholds must be [num_facts]")
            if not bool(torch.isfinite(per_head_thresholds).all()):
                raise ValueError("per_head_thresholds must be finite")
            if gate_mode != "threshold":
                raise ValueError("per-head thresholds need the threshold gate")
        if float(ambiguity_margin) < 0:
            raise ValueError("ambiguity_margin must be non-negative")

        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        if rows is None:
            initial = torch.zeros((n_facts, hidden_size), device=device, dtype=dtype)
        else:
            initial = rows.to(device=device, dtype=dtype)
            if tuple(initial.shape) != (n_facts, hidden_size):
                raise ValueError("Saved rows have wrong shape")
        self.rows = nn.ParameterList(nn.Parameter(row.clone()) for row in initial)
        self.register_buffer(
            "project_dirs",
            None if project_dirs is None else project_dirs.clone().to(device),
        )
        self.register_buffer("router_weight", weight.detach().float().clone().to(device))
        self.register_buffer("router_bias", bias.detach().float().clone().to(device))
        self.register_buffer(
            "head_index", None if head_index is None else head_index.clone().to(device)
        )
        self.register_buffer("feature_mean", feature_mean.detach().float().clone().to(device))
        self.register_buffer(
            "feature_components",
            None if feature_components is None
            else feature_components.detach().float().clone().to(device),
        )
        self.gate_mode = str(gate_mode)
        self.threshold = float("-inf") if self.gate_mode == "subject" else float(threshold)
        self.ambiguity_margin = 0.0 if self.gate_mode == "subject" else float(ambiguity_margin)
        self.bias_calibration = None if bias_calibration is None else dict(bias_calibration)
        if self.bias_calibration is not None:
            self.threshold_policy = str(self.bias_calibration.get("policy", "global"))
        else:
            self.threshold_policy = "per_head" if per_head_thresholds is not None else "global"
        self.register_buffer(
            "per_head_thresholds",
            None if per_head_thresholds is None else per_head_thresholds.clone().to(device),
        )
        self.layer = int(layer)
        self.subject_patterns = subject_patterns
        self.facts = list(facts)
        self.router_fit = dict(router_fit or {})
        self._input_ids = None
        self._attention_mask = None
        self._prefix_lengths = None
        self.calls = 0
        self.active_batch_rows = 0
        self.active_token_positions = 0
        self.active_fact_counts = [0 for _ in facts]
        self.last_active_fact_indices = []
        self.last_route_scores = []
        # Genie (oracle) routing for training only; None = the linear heads
        # decide. Never serialized: a saved bank always routes by its heads.
        self.route_override = None
        # Per-prompt forced routes {prompt token tuple: fact or -1} on top of the heads
        # (training only, never serialized), as early_write.EarlyWriteCausalLM.force_route.
        self.forced_routes = {}
        layer_module = base_model.model.layers[self.layer]
        self._hook_handle = layer_module.register_forward_hook(self._hook)
        if learned_subject is not None:
            self._init_learned_subject(base_model, learned_subject, n_facts, device)

    # ------------------------------------------------- learned subject gate (hybrid)
    def _init_learned_subject(self, base_model, spec, n_facts, device):
        """v2.2's subject head as the eligibility gate: per-subject linear score of every
        prompt token paired with the next one at an early layer, max over the prompt
        (first token / BOS excluded, exactly as at fit time), sigmoid >= threshold."""
        spec = dict(spec)
        layer = int(spec["layer"])
        if not 0 <= layer < self.layer:
            raise ValueError(f"learned subject layer must be below the router layer {self.layer}")
        gate = str(spec.get("gate", "learned"))
        if gate not in SUBJECT_GATES:
            raise ValueError(f"learned_subject gate must be one of {SUBJECT_GATES}")
        fact_subject = torch.as_tensor(spec["fact_subject"], dtype=torch.long)
        weight = torch.as_tensor(spec["weight"]).float()
        if tuple(fact_subject.shape) != (n_facts,):
            raise ValueError("learned_subject fact_subject must have one entry per fact")
        if int(fact_subject.min()) < 0 or int(fact_subject.max()) >= weight.shape[0]:
            raise ValueError("learned_subject fact_subject points outside the subject heads")
        self.register_buffer("ls_weight", weight.to(device))
        self.register_buffer("ls_bias", torch.as_tensor(spec["bias"]).float().to(device))
        self.register_buffer("ls_mean", torch.as_tensor(spec["mean"]).float().to(device))
        self.register_buffer("ls_fact_subject", fact_subject.to(device))
        self.learned_subject = {k: v for k, v in spec.items()
                                if k not in ("weight", "bias", "mean", "fact_subject")}
        self.learned_subject.update({"layer": layer, "gate": gate,
                                     "threshold": float(spec["threshold"])})
        # (prompt, fact) pairs per hook call: eligible by each rule, and by the one used
        self.gate_stats = {"exact": 0, "learned": 0, "used": 0, "learned_only": 0, "exact_only": 0}
        self._subject_hook = base_model.model.layers[layer].register_forward_hook(
            self._capture_subject_states)

    def _capture_subject_states(self, module, args, output):
        if self._input_ids is not None:
            self._subject_states = output[0] if isinstance(output, tuple) else output
        return None

    def _learned_subject_mask(self, prefix_lengths):
        from factorized_router import mil_pool

        states = self._subject_states
        self._subject_states = None
        if states is None:
            raise RuntimeError("learned subject gate: no early-layer states were captured")
        batch, width, _ = states.shape
        device = states.device
        mask = torch.arange(width, device=device)[None, :] < prefix_lengths.to(device)[:, None]
        if self._attention_mask is not None:
            mask &= self._attention_mask.to(device).bool()
        first = mask.float().argmax(dim=1)
        drop = mask.sum(dim=1) > 1                  # a one-token prompt keeps its only token
        mask[torch.arange(batch, device=device)[drop], first[drop]] = False
        probs = torch.sigmoid(mil_pool(states.float(), mask, self.ls_weight, self.ls_bias,
                                       self.ls_mean))                       # [B, S]
        return probs[:, self.ls_fact_subject] >= self.learned_subject["threshold"]

    def _fuzzy_subject_mask(self, prefix_lengths):
        if self.tokenizer is None:
            raise RuntimeError("subject_match='fuzzy' needs bank.tokenizer (set it after loading)")
        rows = self._input_ids.detach().cpu().tolist()
        prefixes = prefix_lengths.detach().cpu().tolist()
        attention = (self._attention_mask.detach().cpu().bool().tolist()
                     if self._attention_mask is not None else None)
        mask = torch.zeros((len(rows), len(self.facts)), dtype=torch.bool)
        for b, (tokens, boundary) in enumerate(zip(rows, prefixes)):
            kept = [t for i, t in enumerate(tokens[:int(boundary)]) if attention is None or attention[b][i]]
            text = self.tokenizer.decode(kept, skip_special_tokens=True)
            for fact_index, words in enumerate(self._fuzzy_words):
                if fuzzy_subject_in_text(text, words):
                    mask[b, fact_index] = True
        return mask.to(self._input_ids.device)

    def _eligibility(self, prefix_lengths):
        exact = self._subject_mask(
            self._input_ids, prefix_lengths, attention_mask=self._attention_mask,
        )
        if self.subject_match == "fuzzy":
            exact = exact | self._fuzzy_subject_mask(prefix_lengths).to(exact.device)
        if self.learned_subject is None:
            return exact
        learned = self._learned_subject_mask(prefix_lengths).to(exact.device)
        used = learned | exact if self.learned_subject["gate"] == "learned_or_exact" else learned
        with torch.no_grad():
            self.gate_stats["exact"] += int(exact.sum())
            self.gate_stats["learned"] += int(learned.sum())
            self.gate_stats["used"] += int(used.sum())
            self.gate_stats["learned_only"] += int((learned & ~exact).sum())
            self.gate_stats["exact_only"] += int((exact & ~learned).sum())
        return used

    def set_oracle_routes(self, prompt_to_fact):
        """Route by ground truth: {prompt-prefix token tuple: fact index}.

        A prompt missing from the map receives no row. Pass None to restore
        classifier routing before saving or evaluating.
        """
        if prompt_to_fact is None:
            self.route_override = None
            return
        mapping = {tuple(int(t) for t in key): int(value)
                   for key, value in prompt_to_fact.items()}
        if any(not 0 <= value < len(self.facts) for value in mapping.values()):
            raise ValueError("Oracle route points outside the association bank")
        self.route_override = mapping

    def force_route(self, prompt_ids, fact):
        """Route this exact prompt to `fact` (or -1 for none) whatever the heads say; every
        other prompt is still routed by the heads. Training only; never saved."""
        self.forced_routes[tuple(int(t) for t in prompt_ids)] = int(fact)

    def _apply_forced_routes(self, prefix_lengths, best_fact, active):
        rows = self._input_ids.detach().cpu().tolist()
        attention = (self._attention_mask.detach().cpu().bool().tolist()
                     if self._attention_mask is not None else None)
        best, act = best_fact.clone(), active.clone()
        for index, (tokens, boundary) in enumerate(zip(rows, prefix_lengths.detach().cpu().tolist())):
            prompt = tuple(token for position, token in enumerate(tokens[: int(boundary)])
                           if attention is None or attention[index][position])
            fact = self.forced_routes.get(prompt)
            if fact is not None:
                act[index] = fact >= 0
                best[index] = max(0, fact)
        return best, act

    def _oracle_routes(self, prefix_lengths, device):
        rows = self._input_ids.detach().cpu().tolist()
        prefixes = prefix_lengths.detach().cpu().tolist()
        attention = (
            self._attention_mask.detach().cpu().bool().tolist()
            if self._attention_mask is not None else None
        )
        best, active = [], []
        for index, (tokens, boundary) in enumerate(zip(rows, prefixes)):
            prompt = tuple(
                token for position, token in enumerate(tokens[: int(boundary)])
                if attention is None or attention[index][position]
            )
            fact = self.route_override.get(prompt)
            best.append(0 if fact is None else fact)
            active.append(fact is not None)
        return (
            torch.tensor(best, dtype=torch.long, device=device),
            torch.tensor(active, dtype=torch.bool, device=device),
        )

    def active_threshold(self, device):
        if self.per_head_thresholds is not None:
            return self.per_head_thresholds.to(device)
        return self.threshold

    def router_logits(self, query):
        device = query.device
        logits = score_queries(
            query,
            self.router_weight.to(device),
            self.router_bias.to(device),
            self.feature_mean.to(device),
            None if self.feature_components is None else self.feature_components.to(device),
        )
        if self.head_index is not None:
            logits = logits[:, self.head_index.to(device)]
        return logits

    def _hook(self, module, args, output):
        if self.passthrough:
            return None
        if self._input_ids is None:
            raise RuntimeError("Linear-router hook fired without input_ids")
        hidden = output[0] if isinstance(output, tuple) else output
        batch, width, _ = hidden.shape
        prefix_lengths = self._prefix_lengths_for(hidden)
        subject_mask = self._eligibility(prefix_lengths)
        positions = prefix_lengths - 1
        query = hidden[torch.arange(batch, device=hidden.device), positions].float()
        logits = self.router_logits(query)
        if self.free_fire_logit is not None:
            # no eligible subject at all: a head may still fire if it is confident enough
            none = ~subject_mask.any(dim=1)
            subject_mask = subject_mask | (none[:, None] & (logits >= self.free_fire_logit))
        decision = decide_routes(
            logits, subject_mask, self.active_threshold(logits.device), self.ambiguity_margin
        )
        active = decision["active"]
        best_fact = decision["fact"]
        if self.route_override is not None:
            best_fact, active = self._oracle_routes(prefix_lengths, hidden.device)
        elif self.forced_routes:
            best_fact, active = self._apply_forced_routes(prefix_lengths, best_fact, active)

        position_mask = self._write_mask(
            positions, prefix_lengths, best_fact, active, width
        ).to(hidden.dtype)
        if self.edit_mode == "project":
            # Remove the routed fact's answer directions: h - alpha * U U^T h.
            basis = self.project_dirs.to(hidden.device)[best_fact]          # [B, k, d]
            coeff = torch.einsum("bwd,bkd->bwk", hidden.float(), basis)       # [B, W, k]
            removed = torch.einsum("bwk,bkd->bwd", coeff, basis)              # [B, W, d]
            gate = position_mask.float() * active[:, None].float()            # [B, W]
            delta = (-self.project_alpha * removed * gate.unsqueeze(-1)).to(hidden.dtype)
        else:
            rows = self.extra.to(device=hidden.device, dtype=hidden.dtype)
            selected = F.embedding(best_fact, rows)
            delta = (
                position_mask.unsqueeze(-1)
                * selected.unsqueeze(1)
                * active[:, None, None].to(hidden.dtype)
            )
        edited = hidden + delta

        self.calls += 1
        with torch.no_grad():
            self.active_batch_rows += int(active.sum())
            self.active_token_positions += int(position_mask[active].sum())
            self.last_active_fact_indices = [
                [int(best_fact[i])] if bool(active[i]) else [] for i in range(batch)
            ]
            self.last_route_scores = []
            for i in range(batch):
                fired = bool(active[i])
                best_eligible = float(decision["best_eligible_logit"][i])
                has_candidate = math.isfinite(best_eligible)
                self.last_route_scores.append({
                    "fact_index": int(best_fact[i]) if fired else None,
                    "logit": float(decision["logit"][i]) if fired else None,
                    "probability": float(torch.sigmoid(decision["logit"][i])) if fired else None,
                    "best_eligible_logit": best_eligible if has_candidate else None,
                    "best_eligible_fact": (
                        int(decision["best_eligible_fact"][i]) if has_candidate else None
                    ),
                    "threshold": (
                        float(self.per_head_thresholds[int(decision["best_eligible_fact"][i])])
                        if self.per_head_thresholds is not None and has_candidate
                        else _finite_or_none(self.threshold)
                    ),
                    "threshold_policy": self.threshold_policy,
                    "gate_mode": self.gate_mode,
                    "qualifying_candidates": int(decision["qualifying"][i]),
                    "top1_top2_logit_separation": _finite_or_none(decision["separation"][i]),
                    "rejected_as_ambiguous": bool(decision["ambiguous"][i]),
                })
            for fact_index in best_fact[active].detach().cpu().tolist():
                self.active_fact_counts[int(fact_index)] += 1

        if isinstance(output, tuple):
            return (edited, *output[1:])
        return edited

    def _subject_positions(self, row, boundary, fact):
        """Absolute positions of the LAST occurrence of the fact's subject in the prompt."""
        tokens = self._input_ids[row].detach().cpu().tolist()
        attention = (
            None if self._attention_mask is None
            else self._attention_mask[row].detach().cpu().bool().tolist()
        )
        prompt = [
            (position, token) for position, token in enumerate(tokens[: int(boundary)])
            if attention is None or attention[position]
        ]
        ids = [token for _, token in prompt]
        best = None
        for pattern in self.subject_patterns[int(fact)]:
            width = len(pattern)
            for start in range(len(ids) - width, -1, -1):
                if tuple(ids[start:start + width]) == tuple(pattern):
                    if best is None or start + width > best[1]:
                        best = (start, start + width)
                    break
        if best is None:
            return []
        return [prompt[k][0] for k in range(best[0], best[1])]

    def _write_mask(self, positions, prefix_lengths, best_fact, active, width):
        """[B, width] 0/1 mask of the positions that receive the routed row."""
        mask = F.one_hot(positions, num_classes=width).float()
        if self.write_mode == "last":
            return mask
        if self.write_mode == "query":
            later = torch.arange(width, device=positions.device)[None, :] >= positions[:, None]
            if self._attention_mask is not None and self._attention_mask.shape[1] == width:
                later = later & self._attention_mask.to(positions.device).bool()
            return later.float()
        for row in active.nonzero(as_tuple=True)[0].tolist():
            boundary = int(prefix_lengths[row])
            if self.write_mode == "all_prompt":
                tokens = self._input_ids[row].detach().cpu().tolist()
                attention = (
                    None if self._attention_mask is None
                    else self._attention_mask[row].detach().cpu().bool().tolist()
                )
                extra = [
                    position for position in range(boundary)
                    if (attention is None or attention[position])
                    and tokens[position] not in self.bos_token_ids
                ]
            else:
                span = self._subject_positions(row, boundary, best_fact[row])
                if not span:
                    self.subject_fallbacks += 1
                extra = span[-1:] if self.write_mode == "last_subject" else span
            if extra:
                mask[row, torch.tensor(extra, device=mask.device)] = 1.0
        return mask

    def decision_rule(self):
        if self.gate_mode == "subject":
            return "subject_gate"
        if self.bias_calibration is not None:
            return "calibrated_bias"
        return "explicit_threshold"

    def routing_policy(self):
        return routing_policy_name(self.gate_mode, self.decision_rule(), self.threshold_policy)

    def artifact(self):
        return {
            "architecture": ARCHITECTURE,
            "router_version": ROUTER_VERSION,
            "method": METHOD,
            "layer": self.layer,
            "router_weight": self.router_weight.detach().cpu(),
            "router_bias": self.router_bias.detach().cpu(),
            "head_index": (
                None if self.head_index is None else self.head_index.detach().cpu()
            ),
            "feature_mean": self.feature_mean.detach().cpu(),
            "feature_components": (
                None if self.feature_components is None
                else self.feature_components.detach().cpu()
            ),
            "gate_mode": self.gate_mode,
            "write_mode": self.write_mode,
            "edit_mode": self.edit_mode,
            "project_dirs": (
                None if self.project_dirs is None else self.project_dirs.detach().cpu()
            ),
            "project_alpha": self.project_alpha,
            "threshold": self.threshold,
            "threshold_policy": self.threshold_policy,
            "per_head_thresholds": (
                None if self.per_head_thresholds is None
                else self.per_head_thresholds.detach().cpu()
            ),
            "ambiguity_margin": self.ambiguity_margin,
            "rows": self.extra.detach().cpu(),
            "subject_patterns": self.subject_patterns,
            "facts": self.facts,
            "router_fit": self.router_fit,
            "bias_calibration": self.bias_calibration,
            "decision_rule": self.decision_rule(),
            "routing_policy": self.routing_policy(),
            "unique_subject_bypass": self.gate_mode == "subject",
            "subject_scan_scope": "prompt_prefix_only",
            "subject_match": self.subject_match,
            "free_fire_logit": self.free_fire_logit,
            "teacher_forced_suffix_can_affect_routing": False,
            "deterministic_route": True,
            "inactive_path_identity_exact": True,
            "generation_contract": "uncached recomputation with fixed original request boundary",
            "trainable_parameters": sum(row.numel() for row in self.rows),
            "router_parameters": int(self.router_weight.numel() + self.router_bias.numel()),
            "router_heads": int(self.router_weight.shape[0]),
            "base_parameters_trainable": 0,
            "tokenizer_extended": False,
            "lm_head_edited": False,
            "object_required_in_runtime_input": False,
            "learned_subject": self._learned_subject_state(),
        }

    def _learned_subject_state(self):
        if self.learned_subject is None:
            return None
        return {**self.learned_subject,
                "weight": self.ls_weight.detach().cpu(), "bias": self.ls_bias.detach().cpu(),
                "mean": self.ls_mean.detach().cpu(),
                "fact_subject": self.ls_fact_subject.detach().cpu()}


def load_linear_classifier_artifact(base_model, artifact):
    bank = LinearClassifierAssociationBank(
        base_model=base_model,
        layer=int(artifact["layer"]),
        weight=artifact["router_weight"],
        bias=artifact["router_bias"],
        feature_mean=artifact["feature_mean"],
        feature_components=artifact.get("feature_components"),
        threshold=float(artifact["threshold"]),
        subject_patterns=artifact["subject_patterns"],
        facts=artifact["facts"],
        rows=artifact["rows"],
        ambiguity_margin=float(artifact.get("ambiguity_margin", 0.5)),
        gate_mode=str(artifact.get("gate_mode", "threshold")),
        write_mode=str(artifact.get("write_mode", "last")),
        edit_mode=str(artifact.get("edit_mode", "add")),
        project_dirs=artifact.get("project_dirs"),
        project_alpha=float(artifact.get("project_alpha", 1.0)),
        router_fit=artifact.get("router_fit"),
        per_head_thresholds=artifact.get("per_head_thresholds"),
        bias_calibration=artifact.get("bias_calibration"),
        head_index=artifact.get("head_index"),
        learned_subject=artifact.get("learned_subject"),
        subject_match=str(artifact.get("subject_match", "exact")),
        free_fire_logit=artifact.get("free_fire_logit"),
    )
    for row in bank.rows:
        row.requires_grad_(False)
    return AssociationCausalLM(base_model, bank), bank


def load_router_artifact(base_model, artifact):
    """Load any fact-association artifact by its declared architecture."""
    architecture = str(artifact.get("architecture", ""))
    if architecture == "factorized_subject_relation_router_v1":
        # Learned subject x relation router (scripts/factorized_router.py).
        from early_write import load_early_write_artifact
        return load_early_write_artifact(base_model, artifact)
    if architecture == ARCHITECTURE and artifact.get("early_write"):
        # Route at the router layer, write at an earlier layer (scripts/early_write.py).
        from early_write import load_early_write_artifact
        return load_early_write_artifact(base_model, artifact)
    if architecture == ARCHITECTURE:
        return load_linear_classifier_artifact(base_model, artifact)
    if architecture == "relation_prototype_fact_association_bank_v2":
        from static_overlap_fact_association_v2_gate import load_relation_prototype_artifact
        return load_relation_prototype_artifact(base_model, artifact)
    if architecture == "stochastic_relation_prototype_fact_association_bank_v2":
        from stochastic_router_gate import load_stochastic_artifact
        return load_stochastic_artifact(base_model, artifact)
    from static_overlap_fact_association_embeddings import load_artifact_into_model
    return load_artifact_into_model(base_model, artifact)


def is_linear_router_artifact(artifact):
    return str(artifact.get("architecture", "")) == ARCHITECTURE
