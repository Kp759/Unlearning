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

    phi(q) = (normalize(q) - mu) [@ C^T when PCA is selected]

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

Two gates, one set of heads
---------------------------
    threshold  (MCF, ZsRE, MQuAKE; association-level forgetting)
               fire the best eligible head if its logit clears one global
               threshold, calibrated on held-out controls; reject ambiguous
               top-1/top-2 pairs.
    subject    (RWKU; entity-level forgetting)
               any prompt containing a protected subject fires; the heads only
               choose WHICH of that subject's residual rows to inject. A
               threshold gate would teach the router to abstain on unseen
               questions about the same person, which are RWKU forget probes.

Splits (training-visible only; official paraphrase/neighbor/retain never read)
-----------------------------------------------------------------------------
    fit          train positives + negatives from donor group 0
                 -> weights; (L2, PCA) by grouped CV over prompt families
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
DEFAULT_LAMBDAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
DEFAULT_PCA_DIMS = (0, 64, 256)
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

def eligibility_matrix(tokenizer, prompts, subject_patterns):
    """[P, N] bool: prompt p contains a complete subject-token pattern of fact i.

    Tokenizes each prompt exactly as the batched runtime call does (special
    tokens on), so the mask equals RelationPrototypeAssociationBank's
    _subject_mask on the unpadded prompt tokens.
    """
    mask = torch.zeros((len(prompts), len(subject_patterns)), dtype=torch.bool)
    for row, prompt in enumerate(prompts):
        tokens = list(tokenizer(prompt)["input_ids"])
        for fact_index, patterns in enumerate(subject_patterns):
            if any(_contains_subsequence(tokens, pattern) for pattern in patterns):
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


def _is_symbolic(relation):
    return bool(_SYMBOLIC_RELATION.match(str(relation)))


def _negative_controls_for_fact(
    fact_index,
    facts,
    fit_positives_by_fact,
    role_of_prompt,
    *,
    count,
    per_donor,
    type_check,
    subject_relation_pairs,
):
    """Same-subject real competitors first, then subject transplants.

    Relative to Router V2's builder:
      * a donor whose (this subject, donor relation) is itself a protected pair
        is skipped -- including this fact's OWN relation. V2 can transplant a
        same-relation donor, i.e. build a paraphrase of the positive and label
        it negative (fact 27 of the MCF seed-1 sample). Applied only to
        symbolic Wikidata relations; ZsRE/RWKU carry a placeholder relation;
      * donors are subject-type-checked unless type_check=False (unknown
        relations degrade open);
      * at most `per_donor` prompts per donor, so negatives span relations.
    """
    fact = facts[fact_index]
    subject = str(fact["subject"])
    subject_key = _norm(subject)
    relation = str(fact.get("relation", ""))
    own = set(fit_positives_by_fact.get(fact["id"], []))
    controls, seen = [], set()
    skipped = defaultdict(int)

    for other_index, other in enumerate(facts):
        if other_index == fact_index or _norm(other["subject"]) != subject_key:
            continue
        for prompt in fit_positives_by_fact.get(other["id"], []):
            if prompt in own or prompt in seen:
                continue
            seen.add(prompt)
            controls.append({
                "prompt": prompt,
                "kind": "same_subject_real",
                "donor_index": other_index,
                "donor_relation": str(other.get("relation", "")),
                "group": role_of_prompt.get(prompt, "unknown"),
            })

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
        if _is_symbolic(donor_relation) and (
            (subject_key, donor_relation) in subject_relation_pairs
        ):
            skipped["donor_relation_protected_for_this_subject"] += 1
            continue
        if type_check and not compatible(relation, donor_relation):
            skipped["subject_type_incompatible"] += 1
            continue
        taken = 0
        for prompt in fit_positives_by_fact.get(other["id"], []):
            if taken >= int(per_donor) or transplants >= int(count):
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
                "donor_index": other_index,
                "donor_rank": donor_rank,
                "donor_relation": donor_relation,
                "group": role_of_prompt.get(prompt, "unknown"),
            })
            taken += 1
            transplants += 1
        if taken:
            donor_rank += 1
        else:
            skipped["donor_without_transplantable_prompt"] += 1
    return controls, dict(skipped)


def assemble_router_dataset(
    facts,
    examples,
    tokenizer,
    subject_patterns,
    *,
    negative_count=36,
    per_donor=3,
    type_check=True,
):
    """Prompts, labels, runtime eligibility and fit/calibration/audit splits.

    examples: dicts (or objects) with fact_id, prompt, split in
    {train, development}, and a prompt family in `group` (or `role`). A prompt lives in exactly one split: a positive
    keeps its example split (train -> fit; development families alternate
    calibration/audit); a negative-only prompt is split by donor so audit
    negatives come from donor relations the head never saw while fitting.
    Augmented positives that lose subject eligibility are dropped.
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
    fit_positives_by_fact = defaultdict(list)
    role_of_prompt = {}
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
        if split == "fit":
            fit_positives_by_fact[fact_id].append(prompt)
            role_of_prompt[prompt] = role

    subject_relation_pairs = {
        (_norm(fact["subject"]), str(fact.get("relation", ""))) for fact in facts
    }
    per_fact = []
    for index, fact in enumerate(facts):
        controls, skipped = _negative_controls_for_fact(
            index,
            facts,
            fit_positives_by_fact,
            role_of_prompt,
            count=negative_count,
            per_donor=per_donor,
            type_check=type_check,
            subject_relation_pairs=subject_relation_pairs,
        )
        added = defaultdict(int)
        for control in controls:
            prompt = control["prompt"]
            existing = records.get(prompt)
            if existing is not None:
                # A real positive of another fact, or a control already built
                # for another head: keep its split; never negate its owner.
                if existing["owner"] != index and index not in existing["negative_for"]:
                    existing["negative_for"].append(index)
                continue
            split = (
                "fit" if control["kind"] == "same_subject_real"
                else SPLITS[int(control["donor_rank"]) % len(SPLITS)]
            )
            records[prompt] = {
                "prompt": prompt,
                "split": split,
                "owner": -1,
                "kind": control["kind"],
                "group": control["group"],
                "donor_relation": control["donor_relation"],
                "augmented": False,
                "negative_for": [index],
            }
            added[split] += 1
        per_fact.append({
            "fact_id": fact["id"],
            "relation": fact.get("relation"),
            "controls": len(controls),
            "new_negatives_by_split": dict(added),
            "skipped_donors": skipped,
        })

    ordered = list(records.values())
    eligible = eligibility_matrix(
        tokenizer, [record["prompt"] for record in ordered], subject_patterns
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
        "per_fact": per_fact,
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


def decide_routes(logits, eligible, threshold, ambiguity_margin):
    """The runtime decision rule. Returns a dict of [B] tensors.

    A head qualifies when it is subject-eligible and its logit is >= the
    threshold (-inf for the subject gate). The best qualifying head fires
    unless a second qualifying head is within `ambiguity_margin` logits.
    """
    eligible = eligible.to(logits.device).bool()
    qualifies = eligible & (logits >= float(threshold))
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
    weight = torch.zeros((n_heads, phi.shape[1]), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(n_heads, dtype=torch.float64, requires_grad=True)
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
):
    """Fit Linear(d', N) on the rows given. Returns float32 tensors + info."""
    if queries.shape[0] != labels.shape[0] or labels.shape != eligible.shape:
        raise ValueError("queries, labels and eligible must align")
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
    return {
        "weight": weight.float(),
        "bias": bias.float(),
        "feature_mean": mean.float(),
        "feature_components": None if components is None else components.float(),
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


def calibrate_threshold(
    logits,
    eligible,
    owner,
    *,
    target_fpr=0.0,
    ambiguity_margin=0.5,
    placement="midpoint",
):
    """One global logit threshold chosen on the calibration split.

    Candidates: every eligible logit on the split plus one value strictly
    above the largest (fires nothing). The whole rule is evaluated for each
    candidate because ambiguity rejection makes the false-activation curve
    non-monotone. Admissible = false activation <= target_fpr on negative
    controls; among admissible candidates with the most correct routes, the
    threshold is placed by `placement`:
      midpoint  middle of that interval (max-margin; default). Taking its top
                end hugs the weakest calibration positive and rejects unseen
                positives just below it; its bottom end hugs the hardest
                calibration negative.
      high      top end (most conservative)
      low       bottom end (most permissive)
    The midpoint is re-checked and falls back to the top end if it is not
    itself admissible with the same correct count.
    """
    if placement not in ("midpoint", "high", "low"):
        raise ValueError("placement must be midpoint, high or low")
    eligible = eligible.bool()
    # Candidates live in the logits' own dtype: the runtime compares float32
    # logits with the threshold, so a float64 "just above the max" would round
    # back onto the max and still fire it.
    values = logits[eligible].unique()
    if values.numel() == 0:
        raise ValueError("Calibration split has no eligible pairs")
    ceiling = torch.nextafter(values.max(), torch.tensor(float("inf"), dtype=values.dtype))
    candidates = torch.cat([values, ceiling.reshape(1)]).tolist()
    negative = owner < 0
    positive = ~negative
    if not bool(negative.any()):
        raise ValueError("Calibration split has no negative controls")
    def evaluate(candidate):
        decision = decide_routes(logits, eligible, candidate, ambiguity_margin)
        active = decision["active"].cpu()
        fpr = float((active & negative).sum()) / float(negative.sum())
        correct = int((positive & active & (decision["fact"].cpu() == owner)).sum())
        return candidate, fpr, correct

    sweep = [evaluate(candidate) for candidate in candidates]

    def choose(target):
        admissible = [row for row in sweep if row[1] <= float(target) + 1e-12]
        best_correct = max(row[2] for row in admissible)
        interval = sorted(row[0] for row in admissible if row[2] == best_correct)
        low, high = interval[0], interval[-1]
        if placement == "low":
            return evaluate(low)
        if placement == "high":
            return evaluate(high)
        # Any threshold in (previous candidate, low] makes the same decisions
        # as `low`, so the admissible interval really starts just above the
        # previous observed logit. Take the middle of (previous, high].
        position = candidates.index(low)
        previous = candidates[position - 1] if position > 0 else low - 1.0
        middle_value = float(
            torch.tensor(0.5 * (previous + high), dtype=logits.dtype)
        )
        middle = evaluate(middle_value)
        if middle[1] <= float(target) + 1e-12 and middle[2] == best_correct:
            return middle
        return evaluate(high)

    threshold, fpr, _ = choose(target_fpr)
    curve = []
    for target in sorted({0.0, 0.01, 0.02, 0.05, float(target_fpr)}):
        t, f, c = choose(target)
        curve.append({
            "target_fpr": target, "threshold_logit": t,
            "calibration_fpr": f,
            "calibration_recall": c / float(positive.sum()) if bool(positive.any()) else None,
        })
    outcome = route_outcomes(logits, eligible, owner, threshold, ambiguity_margin)
    note = None
    if 0.0 < float(target_fpr) < 1.0 / float(negative.sum()):
        note = (
            f"target_fpr {target_fpr} is below the split's resolution "
            f"1/{int(negative.sum())}; it acts as zero"
        )
    return float(threshold), {
        "rule": "max_correct_routes_subject_to_calibration_fpr",
        "placement": placement,
        "target_fpr": float(target_fpr),
        "ambiguity_margin": float(ambiguity_margin),
        "threshold_logit": float(threshold),
        "threshold_probability": float(torch.sigmoid(torch.tensor(threshold))),
        "calibration_fpr": fpr,
        "resolution_note": note,
        "operating_curve": curve,
        "calibration_outcomes_optimistic": outcome,
        "note": (
            "Calibration outcomes are measured on the split that chose the "
            "threshold; report the audit split."
        ),
    }


def prototype_router_routes(queries, eligible, artifact):
    """Router V2's decision on the same queries, for a like-for-like audit."""
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
    u = torch.stack(u_columns, dim=-1)
    d = torch.stack(d_columns, dim=-1)
    qualifies = (
        eligible.bool()
        & (u >= artifact["alpha"].float()[None, :])
        & (d >= artifact["tau"].float()[None, :])
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
    counters = RelationPrototypeAssociationBank.counters
    close = RelationPrototypeAssociationBank.close
    extra = RelationPrototypeAssociationBank.extra

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
    ):
        super().__init__()
        n_facts = len(facts)
        if len(subject_patterns) != n_facts:
            raise ValueError("subject_patterns must align with facts")
        if weight.ndim != 2 or weight.shape[0] != n_facts:
            raise ValueError("weight must be [num_facts, feature_dim]")
        if tuple(bias.shape) != (n_facts,):
            raise ValueError("bias must be [num_facts]")
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
        self.register_buffer("router_weight", weight.detach().float().clone().to(device))
        self.register_buffer("router_bias", bias.detach().float().clone().to(device))
        self.register_buffer("feature_mean", feature_mean.detach().float().clone().to(device))
        self.register_buffer(
            "feature_components",
            None if feature_components is None
            else feature_components.detach().float().clone().to(device),
        )
        self.gate_mode = str(gate_mode)
        self.threshold = float("-inf") if self.gate_mode == "subject" else float(threshold)
        self.ambiguity_margin = 0.0 if self.gate_mode == "subject" else float(ambiguity_margin)
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
        layer_module = base_model.model.layers[self.layer]
        self._hook_handle = layer_module.register_forward_hook(self._hook)

    def router_logits(self, query):
        device = query.device
        return score_queries(
            query,
            self.router_weight.to(device),
            self.router_bias.to(device),
            self.feature_mean.to(device),
            None if self.feature_components is None else self.feature_components.to(device),
        )

    def _hook(self, module, args, output):
        if self._input_ids is None:
            raise RuntimeError("Linear-router hook fired without input_ids")
        hidden = output[0] if isinstance(output, tuple) else output
        batch, width, _ = hidden.shape
        prefix_lengths = self._prefix_lengths_for(hidden)
        subject_mask = self._subject_mask(
            self._input_ids, prefix_lengths, attention_mask=self._attention_mask,
        )
        positions = prefix_lengths - 1
        query = hidden[torch.arange(batch, device=hidden.device), positions].float()
        logits = self.router_logits(query)
        decision = decide_routes(logits, subject_mask, self.threshold, self.ambiguity_margin)
        active = decision["active"]
        best_fact = decision["fact"]

        rows = self.extra.to(device=hidden.device, dtype=hidden.dtype)
        selected = F.embedding(best_fact, rows)
        position_mask = F.one_hot(positions, num_classes=width).to(hidden.dtype)
        delta = (
            position_mask.unsqueeze(-1)
            * selected.unsqueeze(1)
            * active[:, None, None].to(hidden.dtype)
        )
        edited = hidden + delta

        self.calls += 1
        with torch.no_grad():
            self.active_batch_rows += int(active.sum())
            self.active_token_positions += int(active.sum())
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
                    "threshold": _finite_or_none(self.threshold),
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

    def artifact(self):
        return {
            "architecture": ARCHITECTURE,
            "router_version": ROUTER_VERSION,
            "method": METHOD,
            "layer": self.layer,
            "router_weight": self.router_weight.detach().cpu(),
            "router_bias": self.router_bias.detach().cpu(),
            "feature_mean": self.feature_mean.detach().cpu(),
            "feature_components": (
                None if self.feature_components is None
                else self.feature_components.detach().cpu()
            ),
            "gate_mode": self.gate_mode,
            "threshold": self.threshold,
            "ambiguity_margin": self.ambiguity_margin,
            "rows": self.extra.detach().cpu(),
            "subject_patterns": self.subject_patterns,
            "facts": self.facts,
            "router_fit": self.router_fit,
            "routing_policy": (
                "subject_eligibility_mask_plus_linear_bce_heads_top1_"
                + ("global_threshold" if self.gate_mode == "threshold" else "subject_gate")
            ),
            "unique_subject_bypass": self.gate_mode == "subject",
            "subject_scan_scope": "prompt_prefix_only",
            "teacher_forced_suffix_can_affect_routing": False,
            "deterministic_route": True,
            "inactive_path_identity_exact": True,
            "generation_contract": "uncached recomputation with fixed original request boundary",
            "trainable_parameters": sum(row.numel() for row in self.rows),
            "router_parameters": int(self.router_weight.numel() + self.router_bias.numel()),
            "base_parameters_trainable": 0,
            "tokenizer_extended": False,
            "lm_head_edited": False,
            "object_required_in_runtime_input": False,
        }


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
        router_fit=artifact.get("router_fit"),
    )
    for row in bank.rows:
        row.requires_grad_(False)
    return AssociationCausalLM(base_model, bank), bank


def load_router_artifact(base_model, artifact):
    """Load any fact-association artifact by its declared architecture."""
    architecture = str(artifact.get("architecture", ""))
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
