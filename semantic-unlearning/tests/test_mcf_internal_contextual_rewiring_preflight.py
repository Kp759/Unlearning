from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_mcf_internal_contextual_rewiring_v1_split as split_builder
import mcf_internal_contextual_rewiring_v1_core as core
import run_mcf_internal_contextual_rewiring_v1_preflight as preflight


class ToyTokenizer:
    def __init__(self) -> None:
        self.encodings = {
            "Alpha Beta": [1, 2],
            " Alpha Beta": [3, 2],
            "Gamma Beta": [4, 2],
            " Gamma Beta": [5, 2],
            "Delta": [6],
            " Delta": [7],
        }

    def __call__(self, text, **_kwargs):
        if isinstance(text, list):
            return {"input_ids": [self.encodings[value] for value in text]}
        return {"input_ids": self.encodings[str(text)]}


def test_subject_incidence_has_one_coherent_shared_row() -> None:
    tokenizer = ToyTokenizer()
    rows = [
        core.subject_token_rows(tokenizer, "Alpha Beta"),
        core.subject_token_rows(tokenizer, "Gamma Beta"),
        core.subject_token_rows(tokenizer, "Delta"),
    ]
    token_ids, incidence, ownership = core.build_subject_incidence(rows)

    assert token_ids == [1, 2, 3, 4, 5, 6, 7]
    assert ownership[2] == [0, 1]
    assert incidence.shape == (3, 7)
    assert torch.allclose(incidence.sum(dim=1), torch.ones(3, dtype=torch.float64))


def test_overlap_solver_reconstructs_keys_and_caps_common_rows() -> None:
    incidence = torch.tensor(
        [
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.5],
        ],
        dtype=torch.float64,
    )
    targets = core.deterministic_subject_codes(3, 2, 1)
    basis = core.deterministic_orthonormal_basis(6, 2, 1)
    base = torch.ones((5, 6), dtype=torch.float32)
    frequencies = torch.tensor([0, 10000, 0, 0, 0], dtype=torch.long)
    delta, report = core.solve_overlap_aware_embedding_code(
        incidence,
        targets,
        basis,
        base,
        frequencies,
        ridge_lambda=1e-8,
        relative_row_cap=10.0,
        frequency_alpha=0.25,
        nearest_key_margin_floor=0.01,
    )

    relative = delta.norm(dim=1) / base.norm(dim=1)
    assert delta.shape == base.shape
    assert report["passed"] is True
    assert report["incidence_rank"] == 3
    assert relative[1] <= 10.0 / (1.0 + frequencies[1].float()).pow(0.25) + 1e-6


def test_factorized_classifier_requires_subject_and_relation_evidence() -> None:
    classifier = core.FactorizedFactClassifier(
        hidden_size=2,
        rank=2,
        fact_relation_index=[0, 1],
        relation_count=2,
        softmin_temperature=0.05,
    )
    with torch.no_grad():
        classifier.subject_projection.weight.copy_(torch.eye(2))
        classifier.relation_projection.weight.copy_(torch.eye(2))
        classifier.subject_coefficients.copy_(torch.tensor([[4.0, 0.0], [-4.0, 0.0]]))
        classifier.relation_coefficients.copy_(torch.tensor([[0.0, 4.0], [0.0, -4.0]]))
        classifier.subject_bias.zero_()
        classifier.relation_bias.zero_()

    matching = classifier(torch.tensor([[1.0, 1.0]]))["fact_scores"]
    wrong_relation = classifier(torch.tensor([[1.0, -1.0]]))["fact_scores"]

    assert matching[0, 0] > 1.0
    assert matching[0, 1] < -1.0
    assert wrong_relation[0, 0] < -1.0


def test_canonical_prompt_labels_preserve_multilabel_and_writer_off_semantics() -> None:
    specs = [
        core.SemanticPrompt("question", "Alpha", "P1", "positive_a"),
        core.SemanticPrompt("question", "Alpha", "P1", "positive_duplicate"),
        core.SemanticPrompt("question", "Alpha", "P1", "writer_off", writer_on=False),
        core.SemanticPrompt("other", None, None, "broad_negative"),
    ]
    bank = core.canonical_multilabel_prompt_bank(
        specs,
        fact_subjects=["Alpha", "Beta"],
        fact_relation_ids=["P1", "P2"],
        relation_ids=["P1", "P2"],
    )

    assert len(bank.prompts) == 3
    on = next(
        index
        for index, (prompt, writer) in enumerate(zip(bank.prompts, bank.writer_on))
        if prompt == "question" and bool(writer)
    )
    off = next(
        index
        for index, (prompt, writer) in enumerate(zip(bank.prompts, bank.writer_on))
        if prompt == "question" and not bool(writer)
    )
    assert bank.fact_labels[on].tolist() == [True, False]
    assert bank.fact_labels[off].tolist() == [False, False]
    assert bank.subject_labels[off].tolist() == [False, False]
    assert bank.relation_labels[off].tolist() == [True, False]


def test_threshold_is_frozen_before_a_large_disjoint_certificate() -> None:
    calibration_scores = torch.tensor([[2.0, -2.0], [-2.0, 2.0], [-3.0, -3.0]])
    calibration_labels = torch.tensor([[True, False], [False, True], [False, False]])
    calibration = core.calibrate_global_threshold(
        calibration_scores, calibration_labels
    )
    assert calibration["passed"] is True

    rows = 6002
    labels = torch.zeros((rows, 50), dtype=torch.bool)
    labels[0, 0] = True
    labels[1, 1] = True
    scores = torch.full((rows, 50), -3.0)
    scores[0, 0] = 3.0
    scores[1, 1] = 3.0
    certificate = core.frozen_threshold_certificate(
        scores,
        labels,
        threshold=float(calibration["threshold"]),
        distinct_prompts=6002,
        minimum_negative_cells=300000,
        minimum_distinct_prompts=6000,
    )

    assert certificate["passed"] is True
    assert certificate["negative_failures"] == 0
    assert certificate["negative_cells"] >= 300000
    assert certificate["cell_independence_not_assumed"] is True


def test_certificate_audits_each_negative_family_separately() -> None:
    scores = torch.tensor([[3.0, -3.0], [-3.0, -2.0], [-4.0, -4.0]])
    labels = torch.tensor([[True, False], [False, False], [False, False]])
    audit = core.per_kind_threshold_audit(
        scores,
        labels,
        [
            ["certification_authored_positive"],
            ["same_subject_different_relation"],
            ["shared_subject_subword_without_complete_subject"],
        ],
        threshold=0.0,
    )

    assert audit["same_subject_different_relation"]["negative_failures"] == 0
    assert audit["shared_subject_subword_without_complete_subject"]["passed"] is True
    assert audit["certification_authored_positive"]["positive_failures"] == 0


def test_direct_only_split_strips_every_probe_field() -> None:
    raw = {
        "case_id": 7,
        "requested_rewrite": {
            "prompt": "The profession of {} is",
            "subject": "Alpha Beta",
            "relation_id": "P106",
            "target_true": {"str": "writer", "id": "Q1"},
            "target_new": {"str": "painter", "id": "Q2"},
        },
        "paraphrase_prompts": ["secret paraphrase"],
        "neighborhood_prompts": ["secret neighborhood"],
        "attribute_prompts": ["secret attribute"],
        "generation_prompts": ["secret generation"],
    }
    direct = split_builder.direct_record(raw)
    split_builder.assert_direct_only([direct])
    serialized = json.dumps(direct)

    assert "secret" not in serialized
    assert set(direct) == {"case_id", "requested_rewrite", "data_role"}
    assert set(direct["requested_rewrite"]["target_true"]) == {"str"}


def test_registry_and_cli_lock_the_now_terminal_preflight() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = json.loads(
        (
            root / "protocols" / "mcf_internal_contextual_rewiring_v1_registry.json"
        ).read_text()
    )
    args = preflight.parse_args(
        [
            "--model-path",
            "model",
            "--training-visible-path",
            "visible.json",
            "--split-manifest",
            "manifest.json",
            "--experiment-registry",
            "registry.json",
            "--wikidata-dir",
            "wiki",
            "--output-dir",
            "out",
        ]
    )
    with pytest.raises(RuntimeError, match="architecture/status mismatch"):
        preflight.validate_registry(registry, args)
    assert args.candidate_layers == [8, 12, 16, 20]
    assert args.corpus_certification_prompts == 6000
    assert args.minimum_certification_negative_cells == 300000
    assert registry["status"] == "terminal_rejected_training_only_classifier_preflight"
    assert registry["classifier"]["mandatory_certificate_hard_negative_families"] == [
        "same_subject_different_relation",
        "same_relation_different_subject",
        "shared_subject_subword_without_complete_subject",
        "broad_corpus_prompt",
        "writer_off_positive_context",
    ]

    with pytest.raises(SystemExit):
        preflight.parse_args(
            [
                "--model-path",
                "model",
                "--training-visible-path",
                "visible.json",
                "--split-manifest",
                "manifest.json",
                "--experiment-registry",
                "registry.json",
                "--wikidata-dir",
                "wiki",
                "--output-dir",
                "out",
                "--candidate-layers",
                "27",
            ]
        )
