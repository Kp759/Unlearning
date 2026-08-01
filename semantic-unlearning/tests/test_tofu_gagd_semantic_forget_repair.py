from __future__ import annotations

import torch

import tofu_gagd_neighborhood_confidence as tofu
from tofu_gagd_semantic_forget_repair import (
    abstention_instances,
    candidate_priority,
    semantic_metrics,
    semantic_preference_terms,
)


def test_semantic_preference_hinge_is_zero_when_margin_passes() -> None:
    sensitive = torch.tensor([4.0, 5.0])
    abstention = torch.tensor([2.0, 3.5])
    baseline_abstention = abstention.clone()

    terms = semantic_preference_terms(
        sensitive,
        abstention,
        baseline_abstention,
        preference_margin=1.0,
    )

    assert torch.equal(terms["preference_hinge"], torch.tensor(0.0))
    assert torch.equal(
        terms["abstention_preservation_hinge"],
        torch.tensor(0.0),
    )


def test_semantic_preference_hinge_penalizes_sensitive_preference() -> None:
    sensitive = torch.tensor([2.0])
    abstention = torch.tensor([3.0])
    baseline_abstention = torch.tensor([3.0])

    terms = semantic_preference_terms(
        sensitive,
        abstention,
        baseline_abstention,
        preference_margin=1.0,
    )

    # preference slack = 2 - 3 - 1 = -2; squared hinge = 4
    assert torch.allclose(terms["preference_hinge"], torch.tensor(4.0))


def test_semantic_metrics_reports_exact_violations() -> None:
    sensitive = torch.tensor([4.0, 2.0])
    abstention = torch.tensor([2.0, 3.0])
    baseline_abstention = torch.tensor([2.0, 3.0])

    metrics = semantic_metrics(
        sensitive,
        abstention,
        baseline_abstention,
        preference_margin=1.0,
        tolerance=0.0,
    )

    assert metrics["preference_violation_count"] == 1
    assert metrics["preference_satisfied_rate"] == 0.5
    assert metrics["sensitive_preference_rate"] == 0.5


def test_abstention_instances_preserve_prompts_and_replace_answers() -> None:
    source = [
        tofu.TOFUAnswerInstance(
            split="forget05",
            source_index=7,
            sampled_position=0,
            question="Who wrote the book?",
            answer="Sensitive Author",
            prompt="Question: Who wrote the book? Answer:",
        )
    ]

    result = abstention_instances(source, "Unknown")

    assert len(result) == 1
    assert result[0].prompt == source[0].prompt
    assert result[0].question == source[0].question
    assert result[0].answer == "Unknown"
    assert result[0].split == "abstention"


def test_candidate_priority_prefers_semantic_and_utility_safety() -> None:
    safe = {
        "utility_constraint_violation_count": 0,
        "utility_constraint_violation_count_without_preference": 0,
        "preference_violation_count": 0,
        "active_forget_instance_count": 1,
        "buffered_forget_constraint_unmet_count": 1,
        "minimum_log_probability_preference_margin": 1.0,
        "selected_lm_head_delta_norm": 1.0,
    }
    unsafe = {
        **safe,
        "utility_constraint_violation_count": 1,
        "utility_constraint_violation_count_without_preference": 1,
        "active_forget_instance_count": 0,
    }

    assert candidate_priority(safe) < candidate_priority(unsafe)
