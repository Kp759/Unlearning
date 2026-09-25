"""Invariants of the learned linear router.

The claims these protect:
  * Linear(d, N) + masked BCE + row-wise L2 IS N independent logistic
    regressions (joint fit == per-head fits);
  * ineligible (prompt, head) pairs cannot influence a head;
  * only the request-boundary position is edited; a prompt that does not
    route follows the exact base path;
  * the runtime hook makes the same decision as the offline rule used for
    calibration, and an artifact round-trips to the same routes;
  * the dataset never puts one prompt in two splits and never labels a
    same-relation transplant (a paraphrase of the positive) as a negative.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from linear_router import (  # noqa: E402
    ARCHITECTURE,
    training_positive_floor,
    calibrate_per_head,
    cosine_arm_artifact,
    linear_route_frontier,
    prototype_router_routes,
    same_answer_group,
    v2_route_frontier,
    LinearClassifierAssociationBank,
    assemble_router_dataset,
    calibrate_threshold,
    decide_routes,
    examples_from_facts,
    fit_linear_router,
    load_linear_classifier_artifact,
    load_router_artifact,
    route_outcomes,
    score_queries,
    select_hyperparameters,
    with_context_prefix,
)
from static_overlap_fact_association_embeddings import (  # noqa: E402
    make_subject_patterns,
)


HIDDEN = 16
FACTS = 4
WIDTH = 6


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _Block(nn.Module):
    def forward(self, hidden):
        return (hidden,)


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block()])


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()
        self.placeholder = nn.Parameter(torch.zeros(1))


class _WordTokenizer:
    """Whitespace tokenizer with a BOS id, enough for subject matching."""

    def __init__(self):
        self.vocab = {}

    def _id(self, word):
        return self.vocab.setdefault(word, len(self.vocab) + 10)

    def __call__(self, text, add_special_tokens=True, **_):
        ids = [self._id(word) for word in str(text).split()]
        return {"input_ids": ([1] + ids) if add_special_tokens else ids}


# ---------------------------------------------------------------------------
# Synthetic features
# ---------------------------------------------------------------------------

def _synthetic(per_fact=10, noise=0.35, seed=0):
    """Facts 0 and 1 share a subject; 2 and 3 are unique.

    Each fact has positives near its own centre and same-subject negatives
    near a distinct wrong-relation centre. Returns queries, labels, eligible,
    owner, split.
    """
    g = torch.Generator().manual_seed(seed)
    centres = F.normalize(torch.randn(FACTS, HIDDEN, generator=g), dim=-1)
    wrong = F.normalize(torch.randn(FACTS, HIDDEN, generator=g), dim=-1)
    subject_of = [0, 0, 1, 2]
    queries, owners, eligible_rows, splits = [], [], [], []
    split_cycle = ["fit", "fit", "fit", "calibration", "audit"]
    for fact in range(FACTS):
        same = [i for i in range(FACTS) if subject_of[i] == subject_of[fact]]
        for k in range(per_fact):
            for kind, centre in (("pos", centres[fact]), ("neg", wrong[fact])):
                queries.append(centre + noise * torch.randn(HIDDEN, generator=g))
                owners.append(fact if kind == "pos" else -1)
                row = torch.zeros(FACTS, dtype=torch.bool)
                row[same] = True
                eligible_rows.append(row)
                splits.append(split_cycle[k % len(split_cycle)])
    queries = torch.stack(queries)
    owner = torch.tensor(owners)
    eligible = torch.stack(eligible_rows)
    labels = torch.zeros_like(eligible)
    positive = owner >= 0
    labels[positive.nonzero(as_tuple=True)[0], owner[positive]] = True
    return queries, labels, eligible, owner, splits


def _mask(splits, name):
    return torch.tensor([s == name for s in splits])


def _bank(weight=None, bias=None, threshold=0.0, gate_mode="threshold", margin=0.5, seed=0):
    torch.manual_seed(seed)
    weight = torch.randn(FACTS, HIDDEN) if weight is None else weight
    bias = torch.zeros(FACTS) if bias is None else bias
    facts = [{"id": f"f{i}", "subject": f"S{i}", "relation": "r"} for i in range(FACTS)]
    return LinearClassifierAssociationBank(
        _FakeModel(), 0, weight, bias,
        feature_mean=torch.zeros(HIDDEN), feature_components=None,
        threshold=threshold, subject_patterns=[[(10 + i,)] for i in range(FACTS)],
        facts=facts, rows=torch.randn(FACTS, HIDDEN),
        ambiguity_margin=margin, gate_mode=gate_mode,
    )


def _ids():
    # Row 2 carries no registered subject token and must never route.
    return torch.tensor([
        [10, 1, 2, 3, 4, 5],
        [11, 1, 2, 3, 4, 5],
        [99, 1, 2, 3, 4, 5],
        [10, 11, 2, 3, 4, 5],
    ])


def _run(bank, hidden=None):
    ids = _ids()
    bank.bind(ids, attention_mask=torch.ones_like(ids))
    if hidden is None:
        torch.manual_seed(7)
        hidden = torch.randn(ids.shape[0], WIDTH, HIDDEN)
    edited = bank._hook(None, None, (hidden,))[0]
    bank.unbind()
    return hidden, edited


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def test_joint_fit_equals_independent_heads():
    queries, labels, eligible, _, _ = _synthetic()
    joint = fit_linear_router(queries, labels, eligible, l2=1e-2)
    for head in range(FACTS):
        alone = fit_linear_router(
            queries, labels[:, head:head + 1], eligible[:, head:head + 1], l2=1e-2
        )
        assert torch.allclose(joint["weight"][head], alone["weight"][0], atol=1e-4)
        assert torch.allclose(joint["bias"][head], alone["bias"][0], atol=1e-4)


def test_ineligible_labels_cannot_move_a_head():
    queries, labels, eligible, _, _ = _synthetic()
    flipped = labels.clone()
    flipped[~eligible] = ~flipped[~eligible]
    first = fit_linear_router(queries, labels, eligible, l2=1e-2)
    second = fit_linear_router(queries, flipped, eligible, l2=1e-2)
    assert torch.equal(first["weight"], second["weight"])
    assert torch.equal(first["bias"], second["bias"])


def test_fit_is_deterministic():
    queries, labels, eligible, _, _ = _synthetic()
    a = fit_linear_router(queries, labels, eligible, l2=1e-3, pca_dim=8)
    b = fit_linear_router(queries, labels, eligible, l2=1e-3, pca_dim=8)
    assert torch.equal(a["weight"], b["weight"])
    assert a["info"]["converged"]


def test_same_subject_heads_are_separated_and_threshold_meets_target():
    queries, labels, eligible, owner, splits = _synthetic()
    fit, cal, audit = (_mask(splits, s) for s in ("fit", "calibration", "audit"))
    model = fit_linear_router(queries[fit], labels[fit], eligible[fit], l2=1e-2)
    logits = score_queries(
        queries, model["weight"], model["bias"],
        model["feature_mean"], model["feature_components"],
    )
    threshold, report = calibrate_threshold(
        logits[cal], eligible[cal], owner[cal], target_fpr=0.0, ambiguity_margin=0.5
    )
    assert report["calibration_fpr"] == 0.0
    outcome = route_outcomes(logits[audit], eligible[audit], owner[audit], threshold, 0.5)
    assert outcome["correct_route"]["rate"] >= 0.75
    assert outcome["wrong_row_on_positive"]["rate"] == 0.0


def test_calibration_can_always_fall_back_to_firing_nothing():
    # The hardest calibration negative holds the single largest logit.
    logits = torch.tensor([[3.0000002, -5.0], [1.0, -5.0], [2.5, -5.0]], dtype=torch.float32)
    eligible = torch.tensor([[True, False], [True, False], [True, False]])
    owner = torch.tensor([-1, 0, 0])
    threshold, report = calibrate_threshold(logits, eligible, owner, target_fpr=0.0)
    assert report["calibration_fpr"] == 0.0
    assert threshold > 2.5
    decision = decide_routes(logits, eligible, threshold, 0.5)
    assert not bool(decision["active"][0])


def test_answer_groups():
    assert same_answer_group("P103", "P1412")
    assert same_answer_group("P19", "P20")
    assert not same_answer_group("P19", "P27")
    assert not same_answer_group("P69", "P106")


def test_recall_first_calibration_meets_the_floor():
    queries, labels, eligible, owner, splits = _synthetic()
    fit, cal = _mask(splits, "fit"), _mask(splits, "calibration")
    model = fit_linear_router(queries[fit], labels[fit], eligible[fit], l2=1e-2)
    logits = score_queries(queries, model["weight"], model["bias"],
                           model["feature_mean"], model["feature_components"])
    reachable = route_outcomes(logits[cal], eligible[cal], owner[cal], -1e9, 0.5)
    ceiling = reachable["correct_route"]["rate"]
    _, strict = calibrate_threshold(logits[cal], eligible[cal], owner[cal], target_fpr=0.0)
    _, met = calibrate_threshold(logits[cal], eligible[cal], owner[cal], min_recall=ceiling)
    assert met["recall_target_met"] and met["calibration_recall"] >= ceiling
    assert met["calibration_recall"] >= strict["calibration_recall"]
    if ceiling < 1.0:
        # Unreachable floor: flagged, and falls back to the best reachable recall.
        _, unmet = calibrate_threshold(logits[cal], eligible[cal], owner[cal], min_recall=1.0)
        assert unmet["recall_target_met"] is False
        assert unmet["calibration_recall"] == ceiling


def test_frontiers_are_well_formed_and_v2_shift_zero_is_shipped_v2():
    queries, labels, eligible, owner, splits = _synthetic()
    fit, audit = _mask(splits, "fit"), _mask(splits, "audit")
    model = fit_linear_router(queries[fit], labels[fit], eligible[fit], l2=1e-2)
    logits = score_queries(queries, model["weight"], model["bias"],
                           model["feature_mean"], model["feature_components"])
    linear = linear_route_frontier(logits[audit], eligible[audit], owner[audit], 0.5,
                                   reference=(0.5, 0.9))
    assert 0.0 <= linear["route_auc"] <= 1.0
    recalls = [linear["recall_at_fpr"][k] for k in ("0.0", "0.01", "0.05", "0.1", "0.2", "0.5", "1.0")]
    assert recalls == sorted(recalls)
    assert "at_reference_fpr" in linear and "at_reference_recall" in linear
    # A tiny V2 artifact: positives/negatives are the fit queries per head.
    positives = [F.normalize(queries[fit & (owner == i)], dim=-1) for i in range(FACTS)]
    negatives = [F.normalize(queries[fit & (owner < 0) & eligible[:, i]], dim=-1) for i in range(FACTS)]
    artifact = {"positive_prototypes": positives, "negative_prototypes": negatives,
                "alpha": torch.full((FACTS,), -1.0), "tau": torch.full((FACTS,), -0.1),
                "ambiguity_margin": 0.02}
    v2 = v2_route_frontier(queries[audit], eligible[audit], owner[audit], artifact)
    assert 0.0 <= v2["route_auc"] <= 1.0
    active, _ = prototype_router_routes(queries[audit], eligible[audit], artifact)
    positive = owner[audit] >= 0
    shipped_fpr = float((active & ~positive).sum()) / float((~positive).sum())
    assert v2["recall_at_fpr"]["1.0"] >= 0.0 and shipped_fpr <= 1.0


def test_grouped_cv_returns_grid_values():
    queries, labels, eligible, _, splits = _synthetic()
    fit = _mask(splits, "fit")
    groups = [f"family_{i % 3}" for i in range(int(fit.sum()))]
    l2, pca, table = select_hyperparameters(
        queries[fit], labels[fit], eligible[fit], groups,
        lambdas=(1e-3, 1e-1), pca_dims=(0, 8), folds=3,
    )
    assert l2 in (1e-3, 1e-1) and pca in (0, 8)
    assert len(table["table"]) == 4


# ---------------------------------------------------------------------------
# Runtime hook
# ---------------------------------------------------------------------------

def test_only_the_boundary_position_is_edited():
    hidden, edited = _run(_bank(threshold=-100.0))
    delta = (edited - hidden).norm(dim=-1)
    assert torch.equal(delta[:, :-1], torch.zeros_like(delta[:, :-1]))
    assert bool((delta[:, -1] > 0).any())


def test_ineligible_subject_never_routes():
    for gate in ("threshold", "subject"):
        hidden, edited = _run(_bank(threshold=-100.0, gate_mode=gate))
        assert torch.equal(edited[2], hidden[2])


def test_prompt_that_does_not_route_is_exact_base():
    hidden, edited = _run(_bank(threshold=1e9))
    assert torch.equal(edited, hidden)


def test_ambiguous_top_two_abstains():
    weight = torch.randn(FACTS, HIDDEN)
    weight[1] = weight[0]  # heads 0 and 1 give identical logits
    bank = _bank(weight=weight, bias=torch.full((FACTS,), 50.0), threshold=0.0)
    hidden, edited = _run(bank)
    assert bank.last_route_scores[3]["rejected_as_ambiguous"]
    assert torch.equal(edited[3], hidden[3])


def test_subject_gate_fires_on_every_eligible_prompt():
    bank = _bank(bias=torch.full((FACTS,), -1e6), gate_mode="subject", threshold=123.0)
    _run(bank)
    fired = [bool(ids) for ids in bank.last_active_fact_indices]
    assert fired == [True, True, False, True]
    assert bank.threshold == float("-inf")


def test_hook_matches_offline_decision_rule():
    bank = _bank(threshold=0.3)
    hidden, _ = _run(bank)
    query = hidden[:, -1].float()
    logits = score_queries(query, bank.router_weight, bank.router_bias, bank.feature_mean, None)
    eligible = torch.tensor([
        [True, False, False, False],
        [False, True, False, False],
        [False, False, False, False],
        [True, True, False, False],
    ])
    offline = decide_routes(logits, eligible, 0.3, 0.5)
    expected = [[int(offline["fact"][i])] if bool(offline["active"][i]) else [] for i in range(4)]
    assert bank.last_active_fact_indices == expected


def test_artifact_round_trip_gives_identical_routes():
    bank = _bank(threshold=0.1)
    hidden, first = _run(bank)
    artifact = bank.artifact()
    assert artifact["architecture"] == ARCHITECTURE
    _, restored = load_linear_classifier_artifact(_FakeModel(), artifact)
    _, second = _run(restored, hidden=hidden)
    assert torch.equal(first, second)
    _, dispatched = load_router_artifact(_FakeModel(), artifact)
    assert isinstance(dispatched, LinearClassifierAssociationBank)


def test_route_is_deterministic_across_calls():
    bank = _bank(threshold=0.0)
    hidden, first = _run(bank)
    _, second = _run(bank, hidden=hidden)
    assert torch.equal(first, second)


def test_threshold_gate_rejects_infinite_threshold():
    with pytest.raises(ValueError):
        _bank(threshold=float("inf"))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _facts():
    rows = [
        ("Alice Smith", "P19", "Alice Smith was born in"),
        ("Bob Jones", "P19", "Bob Jones was born in"),
        ("Carol King", "P106", "Carol King works as a"),
        ("Alice Smith", "P27", "Alice Smith is a citizen of"),
        ("Dan Brown", "P27", "Dan Brown is a citizen of"),
        ("Eve Adams", "P106", "Eve Adams works as a"),
        ("Frank Moore", "P69", "Frank Moore was educated at"),
        ("Gina Lopez", "P20", "Gina Lopez died in"),
        ("Hugo Weber", "P106", "Hugo Weber is employed as a"),
    ]
    return [
        {"id": f"mcf_forget_{i}", "subject": s, "relation": r, "canonical_prompt": p}
        for i, (s, r, p) in enumerate(rows)
    ]


def test_dataset_splits_are_disjoint_and_negatives_are_clean():
    facts = _facts()
    tokenizer = _WordTokenizer()
    patterns = make_subject_patterns(tokenizer, facts)
    examples = examples_from_facts(facts, augment=True)
    data = assemble_router_dataset(
        facts, examples, tokenizer, patterns, negative_count=9, per_donor=1
    )
    prompts = data["prompts"]
    assert len(prompts) == len(set(prompts))
    assert set(data["split"]) <= {"fit", "calibration", "audit"}
    eligible = data["eligible"]
    for row, record in enumerate(data["records"]):
        if record["owner"] >= 0:
            assert bool(eligible[row, record["owner"]])
        if record["kind"] == "subject_transplant":
            for head in record["negative_for"]:
                subject = facts[head]["subject"]
                protected = {f["relation"] for f in facts if f["subject"] == subject}
                assert record["donor_relation"] not in protected
    # Alice P19 gets Alice P27's real prompts as same-subject negatives, in
    # every split (so held-out splits also contain same-subject negatives).
    alice_p27 = [
        r for r in data["records"]
        if r["owner"] == 3 and 0 in r["negative_for"]
    ]
    assert {r["split"] for r in alice_p27} == {"fit", "calibration", "audit"}
    # Answer groups: P20 (place of death) is never a negative for P19.
    assert not any(
        r["kind"] == "subject_transplant" and 0 in r["negative_for"]
        and r["donor_relation"] == "P20"
        for r in data["records"]
    )
    # Held-out negatives use the same unseen templates as held-out positives.
    for split in ("calibration", "audit"):
        stats = data["diagnostics"]["by_split"][split]
        assert set(stats["negative_families"]) <= set(stats["positive_families"])
    # Every head has at least one fit negative it is eligible for.
    fit = torch.tensor([s == "fit" for s in data["split"]])
    negatives = fit[:, None] & eligible & ~data["labels"]
    assert bool(negatives.any(dim=0).all())
    by_split = data["diagnostics"]["by_split"]
    assert by_split["calibration"]["positives"] > 0 and by_split["audit"]["positives"] > 0


def test_mcf_examples_take_the_family_from_group_not_role():
    # MCF association_examples.json: role is the fact role, group the family.
    facts = _facts()
    tokenizer = _WordTokenizer()
    patterns = make_subject_patterns(tokenizer, facts)
    examples = []
    for fact in facts:
        p = fact["canonical_prompt"]
        for split, family, prompt in (
            ("train", "canonical_rewrite", p),
            ("train", "authored_0", f"Recall that {p}"),
            ("development", "authored_0", f"Note that {p}"),
            ("development", "authored_1", f"Consider that {p}"),
        ):
            examples.append({"fact_id": fact["id"], "prompt": prompt, "split": split,
                             "role": "forget", "group": family})
    data = assemble_router_dataset(facts, examples, tokenizer, patterns, negative_count=6)
    assert data["diagnostics"]["development_family_split"] == {
        "authored_0": "calibration", "authored_1": "audit",
    }
    assert {r["group"] for r in data["records"] if r["owner"] >= 0} == {
        "canonical_rewrite", "authored_0", "authored_1",
    }


def test_context_prefix_handles_qa_and_chat_prompts():
    qa = "<|start|>user\nPlease briefly answer the following question.\nQuestion: Who is Ann?\nAnswer:"
    out = with_context_prefix(qa, "As has been noted elsewhere,")
    assert "Question: As has been noted elsewhere, Who is Ann?" in out
    assert with_context_prefix("<|begin_of_text|>Ann was born in", "X,") is None
    assert with_context_prefix("Ann was born in", "X,") == "X, Ann was born in"


# ---------------------------------------------------------------------------
# Per-head thresholds and the 2x2 arms
# ---------------------------------------------------------------------------

def test_decide_routes_accepts_per_head_thresholds():
    logits = torch.tensor([[1.0, 5.0], [1.0, 5.0]])
    eligible = torch.tensor([[True, False], [False, True]])
    decision = decide_routes(logits, eligible, torch.tensor([0.5, 6.0]), 0.5)
    assert decision["active"].tolist() == [True, False]   # head 1 misses its own 6.0


def test_per_head_rules():
    # head 0 separable, head 1 non-separable, head 2 no negatives, head 3 no positives
    scores = torch.tensor([
        [4.0, 0.0, 0.0, 0.0],   # positive of head 0
        [-2.0, 0.0, 0.0, 0.0],  # negative for head 0
        [0.0, 1.0, 0.0, 0.0],   # positive of head 1
        [0.0, 3.0, 0.0, 0.0],   # negative for head 1 scoring above it
        [0.0, 0.0, 2.0, 0.0],   # positive of head 2
        [0.0, 0.0, 0.0, 7.0],   # negative for head 3
    ])
    eligible = torch.tensor([
        [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
    ], dtype=torch.bool)
    owner = torch.tensor([0, -1, 1, -1, 2, -1])
    thr, report = calibrate_per_head(scores, eligible, owner, fraction=0.1, slack=0.5, fallback=9.0)
    assert thr[0].item() == pytest.approx(-2.0 + 0.1 * 6.0)
    assert thr[1].item() == pytest.approx(0.5)
    assert thr[2].item() == pytest.approx(1.5)
    assert thr[3].item() == pytest.approx(9.0)
    rules = [row["rule"] for row in report["per_head"]]
    assert rules == [
        "negative_ceiling_plus_fraction_of_gap",
        "positive_floor_minus_slack_nonseparable",
        "positive_floor_minus_slack_no_negative",
        "fallback_no_calibration_positive",
    ]
    shrunk, _ = calibrate_per_head(scores, eligible, owner, fraction=0.1, slack=0.5,
                                   shrink=2.0, fallback=9.0)
    # m_0 = 2 prompts, so t_0 moves halfway toward the fallback
    assert shrunk[0].item() == pytest.approx(0.5 * (-1.4) + 0.5 * 9.0)


def test_per_head_threshold_stays_above_hardest_negative():
    scores = torch.tensor([[1.0], [1.0 - 1e-7]])
    thr, _ = calibrate_per_head(scores, torch.ones(2, 1, dtype=torch.bool),
                                torch.tensor([0, -1]), fraction=1e-9, fallback=0.0)
    assert thr[0] > scores[1, 0]


def test_bank_with_per_head_thresholds_routes_and_round_trips():
    weight = torch.zeros(FACTS, HIDDEN)
    bias = torch.tensor([1.0, 1.0, 1.0, 1.0])
    facts = [{"id": f"f{i}", "subject": f"S{i}", "relation": "r"} for i in range(FACTS)]
    bank = LinearClassifierAssociationBank(
        _FakeModel(), 0, weight, bias, torch.zeros(HIDDEN), None, 0.0,
        [[(10 + i,)] for i in range(FACTS)], facts, rows=torch.randn(FACTS, HIDDEN),
        per_head_thresholds=torch.tensor([0.5, 2.0, 0.5, 0.5]),
    )
    hidden, edited = _run(bank)
    # every logit is 1.0: head 0 clears 0.5, head 1 misses 2.0
    assert bank.last_active_fact_indices[0] == [0]
    assert bank.last_active_fact_indices[1] == []
    assert torch.equal(edited[1], hidden[1])
    artifact = bank.artifact()
    assert artifact["threshold_policy"] == "per_head"
    assert artifact["routing_policy"].endswith("per_head_thresholds")
    _, restored = load_router_artifact(_FakeModel(), artifact)
    _, again = _run(restored, hidden=hidden)
    assert torch.equal(edited, again)


def test_cosine_arm_is_a_v2_artifact_with_replaced_tau():
    torch.manual_seed(0)
    source = {
        "architecture": "relation_prototype_fact_association_bank_v2",
        "layer": 0,
        "positive_prototypes": [F.normalize(torch.randn(2, HIDDEN), dim=-1) for _ in range(FACTS)],
        "negative_prototypes": [F.normalize(torch.randn(3, HIDDEN), dim=-1) for _ in range(FACTS)],
        "alpha": torch.full((FACTS,), -1.0),
        "tau": torch.full((FACTS,), -0.5),
        "subject_patterns": [[(10 + i,)] for i in range(FACTS)],
        "facts": [{"id": f"f{i}", "subject": f"S{i}", "relation": "r"} for i in range(FACTS)],
        "rows": torch.randn(FACTS, HIDDEN),
        "ambiguity_margin": 0.02,
        "dataset": "X",
    }
    per_head = cosine_arm_artifact(source, torch.tensor([0.1, 0.2, 0.3, 0.4]),
                                   arm="cosine_per_head", calibration={})
    glob = cosine_arm_artifact(source, 0.25, arm="cosine_global", calibration={})
    assert per_head["tau"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert glob["tau"].tolist() == pytest.approx([0.25] * FACTS)
    assert source["tau"].tolist() == pytest.approx([-0.5] * FACTS)  # source untouched
    assert per_head["dataset"] == "X"
    from static_overlap_fact_association_v2_gate import RelationPrototypeAssociationBank
    _, bank = load_router_artifact(_FakeModel(), per_head)
    assert isinstance(bank, RelationPrototypeAssociationBank)


def test_per_head_threshold_never_rejects_a_training_positive():
    # calibration says t_0 = -2 + 0.1 * 6 = -1.4, but the fact's own training
    # prompt scores -1.5, so the threshold is capped just below it.
    cal_scores = torch.tensor([[4.0], [-2.0]])
    cal_owner = torch.tensor([0, -1])
    fit_scores = torch.tensor([[-1.5], [3.0]])
    fit_owner = torch.tensor([0, 0])
    ones = torch.ones(2, 1, dtype=torch.bool)
    ceiling = training_positive_floor(fit_scores, ones, fit_owner, epsilon=1e-3)
    thr, report = calibrate_per_head(cal_scores, ones, cal_owner, fraction=0.1,
                                     fallback=0.0, ceiling=ceiling)
    assert thr[0].item() == pytest.approx(-1.501)
    assert report["per_head"][0]["capped_at_training_positive"]
    assert decide_routes(fit_scores[:1], ones[:1], thr, 0.5)["active"].item()
    # an uncapped head is unchanged
    thr2, report2 = calibrate_per_head(cal_scores, ones, cal_owner, fraction=0.1,
                                       fallback=0.0, ceiling=torch.tensor([10.0]))
    assert thr2[0].item() == pytest.approx(-1.4)
    assert not report2["per_head"][0]["capped_at_training_positive"]
