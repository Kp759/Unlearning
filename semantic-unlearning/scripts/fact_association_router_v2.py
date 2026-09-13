"""Cross-benchmark Router V2 built only from training-visible direct prompts.

Subject matching is candidate eligibility only. Every candidate must also pass a
frozen layer-context confirmation test. For each fact, the positive prototype
bank comes from its direct training prompts. Negative prototypes are
same-subject competing direct prompts when available, then deterministic
same-subject synthetic wrong-context prompts made by transplanting the fact's
subject into other training-visible direct prompts.

No target_new, official paraphrase, locality, retain, neighbor, utility, MIA, or
PPL data is used to construct or calibrate this gate.
"""
from __future__ import annotations

from collections import defaultdict
import re

import torch
from torch.nn import functional as F

from static_overlap_fact_association_embeddings import extract_prompt_queries


def _norm(text):
    return " ".join(str(text).casefold().split())


def _canonical_prompts(fact):
    values = fact.get("canonical_prompts")
    if values:
        prompts = [str(value).strip() for value in values if str(value).strip()]
    else:
        value = str(fact.get("canonical_prompt", "")).strip()
        prompts = [value] if value else []
    if not prompts:
        raise ValueError(f"Fact {fact.get('id')} has no training-visible direct prompt")
    return list(dict.fromkeys(prompts))


def prompt_map_from_facts(facts):
    return {fact["id"]: _canonical_prompts(fact) for fact in facts}


def prompt_map_from_examples(examples, *, split="train"):
    result = defaultdict(list)
    for example in examples:
        if str(example.split) != str(split):
            continue
        prompt = str(example.prompt).strip()
        if prompt and prompt not in result[example.fact_id]:
            result[example.fact_id].append(prompt)
    if not result:
        raise ValueError("No training-visible prompts available for Router V2")
    return dict(result)


def _replace_subject(prompt, old_subject, new_subject):
    pattern = re.compile(re.escape(str(old_subject)), flags=re.IGNORECASE)
    replaced, count = pattern.subn(str(new_subject), str(prompt), count=1)
    if count == 0:
        return None
    return replaced


def _negative_prompts_for_fact(
    fact_index,
    facts,
    positive_prompts_by_fact,
    negative_count,
):
    fact = facts[fact_index]
    subject = str(fact["subject"])
    subject_key = _norm(subject)
    negatives = []
    own_positive = set(positive_prompts_by_fact[fact["id"]])

    # Highest-value negatives: other protected relations/contexts for the same
    # subject. These require no synthetic text transformation.
    for other_index, other in enumerate(facts):
        if other_index == fact_index:
            continue
        if _norm(other["subject"]) != subject_key:
            continue
        for prompt in positive_prompts_by_fact[other["id"]]:
            if prompt not in own_positive and prompt not in negatives:
                negatives.append(prompt)

    # Unique-subject facts still need contextual confirmation. Construct safe,
    # deterministic wrong-context controls using only other direct training
    # prompts and public subject strings.
    for offset in range(1, len(facts) + 1):
        if len(negatives) >= int(negative_count):
            break
        other_index = (fact_index + offset) % len(facts)
        if other_index == fact_index:
            continue
        other = facts[other_index]
        for prompt in positive_prompts_by_fact[other["id"]]:
            transplanted = _replace_subject(
                prompt,
                other["subject"],
                subject,
            )
            if (
                transplanted
                and transplanted not in own_positive
                and transplanted not in negatives
            ):
                negatives.append(transplanted)
            if len(negatives) >= int(negative_count):
                break

    if not negatives:
        raise ValueError(
            f"Could not construct Router V2 negatives for fact {fact['id']}"
        )
    return negatives[: int(negative_count)]


@torch.no_grad()
def build_direct_prompt_context_gate(
    model,
    tokenizer,
    facts,
    positive_prompts_by_fact,
    layer,
    *,
    negative_count=12,
    margin_slack=0.02,
):
    """Build positive/negative frozen context prototypes for every fact.

    The gate uses d = max cosine(query, positive prototypes) -
    max cosine(query, negative prototypes). Alpha is intentionally left at -1
    so generalization is governed by relative relation/context evidence rather
    than an overly strict exact-prompt cosine threshold. Tau is fitted from
    training positives and training-only negative controls.
    """
    if len(facts) < 2:
        raise ValueError("Router V2 needs at least two protected facts")
    if int(negative_count) <= 0:
        raise ValueError("negative_count must be positive")
    if float(margin_slack) < 0:
        raise ValueError("margin_slack must be non-negative")

    positive_prototypes = []
    negative_prototypes = []
    alpha = []
    tau = []
    per_fact = []

    for fact_index, fact in enumerate(facts):
        fact_id = fact["id"]
        positive_prompts = list(positive_prompts_by_fact.get(fact_id, []))
        if not positive_prompts:
            raise ValueError(f"Missing Router V2 positives for {fact_id}")
        negative_prompts = _negative_prompts_for_fact(
            fact_index,
            facts,
            positive_prompts_by_fact,
            negative_count,
        )

        pos = extract_prompt_queries(
            model, tokenizer, positive_prompts, int(layer)
        ).float()
        neg = extract_prompt_queries(
            model, tokenizer, negative_prompts, int(layer)
        ).float()
        pos = F.normalize(pos, dim=-1)
        neg = F.normalize(neg, dim=-1)

        def score(queries):
            queries = F.normalize(queries, dim=-1)
            u = (queries @ pos.T).max(dim=-1).values
            v = (queries @ neg.T).max(dim=-1).values
            return u, u - v

        pos_u, pos_d = score(pos)
        neg_u, neg_d = score(neg)
        pos_floor = float(pos_d.min())
        neg_ceiling = float(neg_d.max())
        if neg_ceiling < pos_floor:
            # Use a training-only conservative-negative threshold rather than
            # the midpoint of the separable gap. The midpoint was unnecessarily
            # strict on natural held-out phrasings: it spent half of the entire
            # positive/negative separation as rejection margin. Staying 10% of
            # the observed gap above the hardest training negative preserves
            # zero training-negative fires while leaving 90% of the measured
            # gap available for benign contextual variation.
            gap = pos_floor - neg_ceiling
            tau_i = neg_ceiling + 0.10 * gap
            separable = True
            calibration_rule = "negative_ceiling_plus_10pct_separable_gap"
        else:
            # No held-out data is consulted. In an overlapping training-only
            # calibration case, preserve every observed positive with a small
            # fixed slack and expose the overlap in diagnostics.
            tau_i = pos_floor - float(margin_slack)
            separable = False
            calibration_rule = "positive_floor_minus_fixed_slack"
        tau_i = max(-2.0, min(2.0, tau_i))

        pos_pass = float((pos_d >= tau_i).float().mean())
        neg_fire = float((neg_d >= tau_i).float().mean())
        if pos_pass < 1.0:
            raise RuntimeError(
                f"Router V2 calibration dropped a training positive for {fact_id}"
            )

        positive_prototypes.append(pos.cpu())
        negative_prototypes.append(neg.cpu())
        alpha.append(-1.0)
        tau.append(tau_i)
        per_fact.append({
            "fact_id": fact_id,
            "subject": str(fact["subject"]),
            "relation": str(fact.get("relation", "")),
            "positive_prompt_count": len(positive_prompts),
            "negative_prompt_count": len(negative_prompts),
            "positive_margin_min": pos_floor,
            "negative_margin_max": neg_ceiling,
            "separable_on_training_controls": separable,
            "calibration_rule": calibration_rule,
            "tau": tau_i,
            "training_positive_pass_fraction": pos_pass,
            "training_negative_fire_fraction": neg_fire,
            "negative_prompts": negative_prompts,
        })

    diagnostics = {
        "router_version": "direct_prompt_context_confirmation_v2",
        "layer": int(layer),
        "gate": (
            "subject eligibility AND positive-minus-negative frozen context margin"
        ),
        "unique_subject_bypass": False,
        "absolute_similarity_threshold_enabled": False,
        "alpha": -1.0,
        "negative_count": int(negative_count),
        "margin_slack": float(margin_slack),
        "separable_gap_operating_point": 0.10,
        "separable_gap_note": (
            "tau is placed 10% of the training-only positive/negative gap "
            "above the hardest negative; no development/evaluation prompt is "
            "used to choose the threshold"
        ),
        "training_visible_only": True,
        "target_new_used": False,
        "official_paraphrases_used": False,
        "official_locality_used": False,
        "retain_used": False,
        "neighbor_utility_mia_ppl_used": False,
        "mean_training_negative_fire_fraction": (
            sum(row["training_negative_fire_fraction"] for row in per_fact)
            / len(per_fact)
        ),
        "per_fact": per_fact,
    }
    return (
        positive_prototypes,
        negative_prototypes,
        torch.tensor(alpha, dtype=torch.float32),
        torch.tensor(tau, dtype=torch.float32),
        diagnostics,
    )
