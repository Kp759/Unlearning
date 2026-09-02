from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_joint_forget_retain_endpoint_v2_2_core as core  # noqa: E402
import run_mcf_joint_forget_retain_endpoint_v2_2 as runner  # noqa: E402


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    unk_token_id = 3

    def __init__(self) -> None:
        self.vocab = {
            "Ada": 10,
            "was": 11,
            "born": 12,
            "in": 13,
            "London": 20,
            "Paris": 21,
        }

    def encode(self, text: str, add_special_tokens: bool = True):
        ids = [self.vocab[token] for token in text.replace("?", "").split()]
        return ([self.bos_token_id] + ids) if add_special_tokens else ids

    def __call__(self, text: str):
        return {"input_ids": self.encode(text)}


def record() -> dict:
    return {
        "case_id": 7,
        "requested_rewrite": {
            "subject": "Ada",
            "prompt": "{} was born in",
            "target_true": {"str": "London"},
            "target_new": {"str": "Paris"},
        },
    }


def test_registry_locks_joint_forget_retain_internal_endpoints() -> None:
    registry = json.loads(
        (
            ROOT / "protocols" / "mcf_joint_forget_retain_endpoint_v2_2_registry.json"
        ).read_text()
    )
    assert registry["status"] == "training_only_implementation_available_not_executed"
    assert registry["architecture"]["transformer_frozen"] is True
    assert registry["architecture"]["external_classifier"] is False
    assert registry["architecture"]["runtime_gate"] is False
    assert registry["retain_strata"]["present_on_every_endpoint_update"] is True
    assert registry["optimization"]["retain_examples_on_every_embedding_update"] is True
    assert registry["optimization"]["retain_examples_on_every_lm_head_update"] is True
    assert registry["optimization"]["retain_target_weight"] == 100.0
    assert registry["acceptance"]["protected_target_logprob_abs_max"] == 0.05
    assert registry["optimization"]["adam_forbidden"] is True


def test_registry_and_cli_are_exactly_locked() -> None:
    registry = json.loads(
        (
            ROOT / "protocols" / "mcf_joint_forget_retain_endpoint_v2_2_registry.json"
        ).read_text()
    )
    args = runner.parse_args(
        [
            "--model-path",
            "model",
            "--protocol-dir",
            "protocol",
            "--experiment-registry",
            "registry",
            "--wikidata-dir",
            "wiki",
            "--output-dir",
            "out",
        ]
    )
    runner.validate_registry(registry, args)
    assert args.steps == 1000
    assert args.hard_tail_refresh_every == 50
    assert args.hard_tail_active == 16
    assert args.active_retain_maximum == 48


def test_expanded_rows_include_relation_frame_once() -> None:
    rows, report = core.expanded_endpoint_rows([record()], Tokenizer(), llama_like=True)
    assert set((10, 11, 12, 13)).issubset(rows.input_ids)
    assert rows.output_ids == [20, 21]
    assert report["relation_frame_rows_added"] == 3
    assert report["one_delta_per_physical_input_row"] is True


def test_persistent_hard_tail_keeps_worst_unique_indices() -> None:
    tail = core.PersistentHardTail(3)
    first = tail.refresh(
        torch.tensor([0.0, 0.02, 0.01, 0.03]),
        torch.tensor([0.0, 0.0, 0.2, 0.0]),
        torch.zeros(4),
        kl_limit=0.01,
        drift_limit=0.05,
        target_drift_limit=0.05,
        add=2,
    )
    assert first["indices"] == [2, 3]
    second = tail.refresh(
        torch.tensor([0.04, 0.0, 0.0, 0.0]),
        torch.zeros(4),
        torch.zeros(4),
        kl_limit=0.01,
        drift_limit=0.05,
        target_drift_limit=0.05,
        add=2,
    )
    assert second["indices"] == [0, 1, 2]


def test_hard_tail_includes_retain_target_drift() -> None:
    tail = core.PersistentHardTail(2)
    report = tail.refresh(
        torch.zeros(3),
        torch.zeros(3),
        torch.tensor([0.0, 0.2, 0.1]),
        kl_limit=0.01,
        drift_limit=0.05,
        target_drift_limit=0.05,
        add=2,
    )
    assert report["indices"] == [1, 2]
    assert report["global_target_logprob_abs_max"] == pytest.approx(0.2)


def test_unlabeled_generic_retain_prompt_has_zero_target_drift() -> None:
    logits = torch.tensor([[0.2, 0.8], [0.5, 1.0]], requires_grad=True)
    drift = runner.target_logprob_drift(
        logits,
        torch.tensor([-1, 1]),
        torch.tensor([0.0, -0.6]),
    )
    assert float(drift[0]) == 0.0
    drift[1].backward()
    assert logits.grad is not None
    assert float(logits.grad[1].abs().sum()) > 0.0


def test_active_retain_batch_prioritizes_hard_then_overlap() -> None:
    result = core.compose_active_retain_indices(
        random_indices=[5, 6],
        overlap_indices=[3, 4],
        hard_indices=[1, 2, 3],
        maximum=5,
    )
    assert result == [1, 2, 3, 4, 5]


def test_trust_region_has_distinct_feasible_and_repair_rules() -> None:
    assert core.accept_trust_region_candidate(
        before_forget=2.0,
        candidate_forget=1.9,
        before_constraint_score=0.8,
        candidate_constraint_score=0.9,
    )
    assert not core.accept_trust_region_candidate(
        before_forget=2.0,
        candidate_forget=1.9,
        before_constraint_score=0.8,
        candidate_constraint_score=1.1,
    )
    assert core.accept_trust_region_candidate(
        before_forget=2.0,
        candidate_forget=2.01,
        before_constraint_score=2.0,
        candidate_constraint_score=1.5,
    )


def test_normalized_step_downweights_tiny_noisy_rows() -> None:
    gradient = torch.tensor([[3.0, 4.0], [0.0, 0.02]])
    caps = torch.tensor([10.0, 10.0])
    step = core.normalized_row_step(gradient, caps, fraction=0.01)
    assert torch.allclose(step[0].norm(), torch.tensor(0.1), atol=1e-6)
    assert float(step[1].norm()) < 0.01
