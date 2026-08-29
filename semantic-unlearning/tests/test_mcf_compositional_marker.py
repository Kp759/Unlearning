from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_compositional_marker_core as core
import mcf_compositional_marker_write_read as method
import diagnose_mcf_compositional_component_ppl as component_ppl
import diagnose_mcf_compositional_gen_failures as gen_diagnostic
import sweep_mcf_compositional_beta_frontier as beta_frontier


class WordTokenizer:
    """Tiny tokenizer where ``de`` is deliberately shared by two subjects."""

    def __init__(self):
        self.vocab = {
            "gautier": 1,
            "de": 2,
            "coincy": 3,
            "melchior": 4,
            "vogue": 5,
            "charles": 6,
            "gaulle": 7,
            "speaks": 8,
            "records": 9,
            "say": 10,
            "x": 11,
        }

    def __call__(self, text, *, add_special_tokens=False, **_kwargs):
        words = re.findall(r"[A-Za-z]+", str(text).lower())
        return {
            "input_ids": [
                self.vocab.setdefault(word, len(self.vocab) + 1) for word in words
            ]
        }


class TensorBatch(dict):
    def to(self, device):
        return TensorBatch(
            {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in self.items()
            }
        )


class CharacterBatchTokenizer:
    pad_token_id = 0
    bos_token_id = None
    eos_token_id = 127
    unk_token_id = None
    padding_side = "right"

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        padding=False,
        return_tensors=None,
        **_kwargs,
    ):
        values = text if isinstance(text, list) else [text]
        rows = [[ord(character) for character in value] for value in values]
        width = max(len(row) for row in rows)
        attention = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        if padding:
            rows = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
        if return_tensors == "pt":
            return TensorBatch(
                {
                    "input_ids": torch.tensor(rows, dtype=torch.long),
                    "attention_mask": torch.tensor(attention, dtype=torch.long),
                }
            )
        encoded = rows if isinstance(text, list) else rows[0]
        return {"input_ids": encoded}


class TinyContextLM(torch.nn.Module):
    def __init__(self, vocab_size=128, hidden_size=4):
        super().__init__()
        self.input_embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        self.output_embeddings = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.input_embeddings

    def get_output_embeddings(self):
        return self.output_embeddings

    def forward(self, input_ids, attention_mask=None, **_kwargs):
        del attention_mask
        # Cumulative context makes the final state depend on every prior row.
        hidden = self.input_embeddings(input_ids).cumsum(dim=1)
        return SimpleNamespace(
            logits=self.output_embeddings(hidden),
            hidden_states=(hidden,),
        )


def test_context_builder_makes_shared_subword_and_leave_one_out_negatives():
    tok = WordTokenizer()
    records = [
        {"case_id": 1, "subject": "Gautier de Coincy", "prompt_template": "{} speaks"},
        {"case_id": 2, "subject": "Melchior de Vogue", "prompt_template": "{} speaks"},
        {"case_id": 3, "subject": "Charles de Gaulle", "prompt_template": "{} speaks"},
    ]
    positives = {
        1: ["Gautier de Coincy speaks", "Records say Gautier de Coincy speaks"],
        2: ["Melchior de Vogue speaks"],
        3: ["Charles de Gaulle speaks"],
    }
    selected = {1: [1, 2, 3], 2: [4, 2, 5], 3: [6, 2, 7]}

    contexts, report = core.build_compositional_contexts(
        records,
        positives,
        selected,
        tok,
        seed=7,
        max_shared_subjects=4,
        max_leave_one_out=4,
        max_fragments=4,
        max_unrelated=2,
    )

    first = contexts[1]
    assert first["positive_prompts"][0] == "Gautier de Coincy speaks"
    shared = [
        x for x in first["negative_contexts"] if x["kind"] == "shared_subword_subject"
    ]
    assert {x["source_subject"] for x in shared} == {
        "Melchior de Vogue",
        "Charles de Gaulle",
    }
    assert all(2 in x["overlap_token_ids"] for x in shared)

    leave_one = [
        x for x in first["negative_contexts"] if x["kind"] == "leave_one_component_out"
    ]
    assert len(leave_one) == 3
    assert all("gautier de coincy" not in x["prompt"].casefold() for x in leave_one)
    assert report["official_paraphrases_seen"] == 0
    assert report["official_neighborhoods_seen"] == 0
    assert report["benchmark_retain_seen"] == 0


def test_context_builder_rejects_a_positive_without_the_complete_subject():
    tok = WordTokenizer()
    records = [
        {"case_id": 1, "subject": "Gautier de Coincy", "prompt_template": "{} speaks"},
        {"case_id": 2, "subject": "Melchior de Vogue", "prompt_template": "{} speaks"},
    ]
    with pytest.raises(ValueError, match="complete subject"):
        core.build_compositional_contexts(
            records,
            {
                1: ["Gautier de Coincy speaks", "Coincy speaks"],
                2: ["Melchior de Vogue speaks"],
            },
            {1: [1, 2, 3], 2: [4, 2, 5]},
            tok,
            seed=1,
        )


def test_contrastive_marker_prefers_positive_only_reachable_axis():
    positive = torch.tensor(
        [
            [3.0, 0.10, 0.0, 0.0],
            [-2.8, -0.10, 0.0, 0.0],
            [2.5, 0.05, 0.0, 0.0],
            [-2.6, -0.05, 0.0, 0.0],
        ]
    )
    negative = torch.tensor(
        [
            [0.02, 3.0, 0.0, 0.0],
            [-0.02, -2.7, 0.0, 0.0],
            [0.01, 2.5, 0.0, 0.0],
            [-0.01, -2.4, 0.0, 0.0],
        ]
    )
    forbidden = torch.tensor([[0.0, 0.0, 1.0, 0.0]])

    marker, report = core.select_contrastive_marker(
        positive,
        negative,
        forbidden_basis=forbidden,
        ridge=1e-3,
        max_rank=4,
    )

    assert abs(float(marker[0])) > 0.99
    assert abs(float(marker[1])) < 0.05
    assert abs(float(marker[2])) < 1e-6
    assert report["contrastive_ratio"] > 100.0
    assert report["forbidden_projection_abs_max"] < 1e-6


def test_distributional_reader_is_portable_and_rejects_negative_axis():
    marker = torch.tensor([1.0, 0.0, 0.0])
    positives = torch.tensor(
        [
            [5.0, 0.10, 0.0],
            [4.8, -0.10, 0.05],
            [5.2, 0.05, -0.05],
            [4.9, 0.00, 0.02],
        ]
    )
    negatives = torch.tensor(
        [
            [0.01, 5.0, 0.0],
            [-0.01, 4.5, 0.2],
            [0.02, -4.8, 0.1],
            [-0.02, -5.2, -0.1],
        ]
    )

    reader, fit = core.distributional_reader(
        marker,
        positives,
        negatives,
        ridge=0.05,
        anchor_weight=10.0,
        consistency_weight=2.0,
        negative_weight=2.0,
        refine_steps=100,
        refine_lr=0.03,
        positive_floor=0.02,
    )
    metrics = core.reader_metrics(reader, positives, negatives)

    assert fit["cos_marker_q"] > 0.99
    assert metrics["positive_sign_consistent"] is True
    assert metrics["portability_ratio"] > 0.90
    assert metrics["kappa_train"] < 0.02


def test_distributional_reader_drops_marker_when_marker_is_base_leakage_axis():
    marker = torch.tensor([1.0, 0.0, 0.0, 0.0])
    negatives = torch.tensor(
        [
            [10.0, 10.0, 0.0, 0.0],
            [10.0, -10.0, 0.0, 0.0],
            [-8.0, 8.0, 0.0, 0.0],
            [-8.0, -8.0, 0.0, 0.0],
        ]
    )
    positives = torch.tensor(
        [
            [10.0, 1.0, 5.0, 0.1],
            [9.0, -1.0, 5.2, -0.1],
            [11.0, 0.5, 4.8, 0.0],
        ]
    )

    reader, fit = core.distributional_reader(
        marker,
        positives,
        negatives,
        ridge=0.05,
        anchor_weight=0.05,
        consistency_weight=2.0,
        negative_weight=10.0,
        refine_steps=100,
        refine_lr=0.03,
        positive_floor=0.02,
    )
    metrics = core.reader_metrics(reader, positives, negatives)

    assert abs(fit["cos_marker_q"]) < 1e-4
    assert metrics["positive_sign_consistent"] is True
    assert metrics["kappa_train"] < 1e-5
    assert metrics["portability_ratio"] > 0.85


def test_directional_row_delta_sums_readers_only_on_owned_answer_rows():
    readers = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    betas = torch.tensor([2.0, 3.0, 0.5])
    answer_rows = [[10], [10, 11], [12]]

    delta = core.directional_row_deltas(
        readers, betas, answer_rows, selected_output_rows=[10, 11, 12]
    )

    assert torch.equal(delta[0], torch.tensor([-2.0, -3.0]))
    assert torch.equal(delta[1], torch.tensor([0.0, -3.0]))
    assert torch.equal(delta[2], torch.tensor([-0.5, -0.5]))


def test_sparse_output_rows_have_exact_linear_reader_factorization():
    delta = torch.tensor(
        [
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
        ]
    )

    betas, readers = core.factorize_output_rows(delta)

    assert torch.allclose(delta, -betas.unsqueeze(1) * readers)
    assert torch.allclose(betas, torch.tensor([5.0, 0.0, 2.0]))
    assert torch.allclose(readers[[0, 2]].norm(dim=1), torch.ones(2))
    assert torch.equal(readers[1], torch.zeros(3))


def test_monotone_beta_initializer_covers_joint_shared_row_constraints():
    response = torch.tensor(
        [
            [2.0, 0.2, -4.0],
            [0.1, 1.5, 0.0],
            [0.0, 0.4, 2.5],
        ]
    )
    required = torch.tensor([4.0, 3.0, 5.0])

    beta, report = core.monotone_cover_betas(response, required, safety_factor=1.25)

    retained = response.clamp_min(0.0)
    assert torch.all(retained @ beta >= required * 1.25 - 1e-4)
    assert torch.all(beta >= 0)
    assert report["residual_max"] <= 1e-4


def test_method_defaults_to_standard_weights_and_strict_reader_gate():
    args = method.parse_args(
        [
            "--model-path",
            "/model",
            "--training-visible-path",
            "/locked.json",
            "--split-manifest",
            "/manifest.json",
            "--output-dir",
            "/out",
        ]
    )
    assert args.gate_policy == "strict"
    assert args.kappa_train_max == 0.10
    assert args.portability_min == 0.50
    assert args.writer_marker_kappa_max == 0.10
    assert args.reader_anchor_weight == 0.05
    assert args.cos_marker_reader_min == 0.0
    assert args.stage2_negative_weight == 1e-2
    assert args.stage2_base_positive_weight == 1.0
    assert args.stage2_beta_l2 == 1e-3
    assert args.stage2_reference_nll_weight == 10.0
    assert args.stage2_reference_nll_tolerance == 0.05
    assert args.stage2_protection_rank == 512
    assert args.stage2_residual_rank == 4
    assert args.stage2_row_negative_rank == 32
    assert args.stage2_row_norm_cap_values == [0.05, 0.10, 0.20, 0.40]
    assert not hasattr(args, "router")
    assert not hasattr(args, "logit_bias")


def test_stage1_zero_steps_requires_a_resume_state():
    common = [
        "--model-path",
        "/model",
        "--training-visible-path",
        "/locked.json",
        "--split-manifest",
        "/manifest.json",
        "--output-dir",
        "/out",
        "--writer-steps",
        "0",
    ]
    with pytest.raises(SystemExit):
        method.parse_args(common)

    args = method.parse_args([*common, "--resume-stage1-state", "/stage1.pt"])
    assert args.writer_steps == 0
    assert args.resume_stage1_state == "/stage1.pt"


def test_zero_step_resume_requires_the_exact_context_manifest_hash():
    state = {"context_manifest_sha256": "old-context"}
    with pytest.raises(RuntimeError, match="refusing zero-step Stage-1 resume"):
        method.validate_stage1_resume_binding(
            state,
            current_context_manifest_sha256="new-context",
            writer_steps=0,
        )

    exact = method.validate_stage1_resume_binding(
        state,
        current_context_manifest_sha256="old-context",
        writer_steps=0,
    )
    assert exact["mode"] == "exact_zero_step_reuse"
    assert exact["same_context_manifest"]

    warm = method.validate_stage1_resume_binding(
        state,
        current_context_manifest_sha256="new-context",
        writer_steps=1200,
    )
    assert warm["mode"] == "cross_context_warm_start"
    assert not warm["same_context_manifest"]


def test_clean_context_policy_forbids_free_form_surrogates():
    common = [
        "--model-path",
        "/model",
        "--training-visible-path",
        "/locked.json",
        "--split-manifest",
        "/manifest.json",
        "--output-dir",
        "/out",
    ]
    args = method.parse_args(common)
    assert args.positive_context_policy == method.CLEAN_POSITIVE_CONTEXT_POLICY
    with pytest.raises(SystemExit):
        method.parse_args([*common, "--surrogate-prompts-path", "/free-form-v7.json"])
    with pytest.raises(SystemExit):
        method.parse_args([*common, "--synthetic-paraphrases-per-record", "3"])

    diagnostic = method.parse_args(
        [
            *common,
            "--positive-context-policy",
            method.SURROGATE_POSITIVE_CONTEXT_POLICY,
            "--surrogate-prompts-path",
            "/free-form-v7.json",
        ]
    )
    assert diagnostic.surrogate_prompts_path == "/free-form-v7.json"


def test_clean_relation_templates_preserve_the_two_audited_failure_relations():
    language = method.synthetic.synthetic_prompt_templates(
        relation_id="P364",
        canonical_prompt="The language of {} was",
        case_id=14801,
        count=6,
        context_prefixes=["An unrelated corpus sentence."],
    )
    instrument = method.synthetic.synthetic_prompt_templates(
        relation_id="P1303",
        canonical_prompt="{} plays",
        case_id=17256,
        count=6,
        context_prefixes=["An unrelated corpus sentence."],
    )

    assert all("{}" in prompt for prompt in language + instrument)
    assert all(
        bad not in " ".join(language).casefold()
        for bad in ("protagonist", "author", "caretaker in the novel", "spoken by")
    )
    assert all(
        bad not in " ".join(instrument).casefold()
        for bad in ("bass", "jazz bassist", "known for his")
    )
    assert "original language" in " ".join(language).casefold()
    assert "instrument" in " ".join(instrument).casefold()


def test_multi_context_reachability_moves_only_prompts_containing_selected_row():
    torch.manual_seed(3)
    model = TinyContextLM()
    tok = CharacterBatchTokenizer()
    prompts = ["ab", "cb"]
    base = method.cache_prompt_baselines(
        model,
        tok,
        prompts,
        torch.device("cpu"),
        batch_size=2,
        topk=4,
    )
    delta = method.canonical.SelectedRowDelta(
        1,
        4,
        direction_basis=None,
        device=torch.device("cpu"),
    )
    handle = method.directional.register_input_embedding_delta_hook(
        model.get_input_embeddings(), [ord("a")], delta.effective_delta
    )
    try:
        positive, negative = method.multi_context_reachability(
            model,
            tok,
            delta,
            [0],
            ["ab"],
            ["cb"],
            base,
            torch.device("cpu"),
            probes=4,
            sigma=0.2,
            generator=torch.Generator().manual_seed(8),
        )
    finally:
        handle.remove()

    assert float(positive.norm()) > 0
    assert torch.equal(negative, torch.zeros_like(negative))


def test_same_prompt_reference_first_predictor_is_not_a_valid_negative_state():
    torch.manual_seed(4)
    model = TinyContextLM()
    tok = CharacterBatchTokenizer()

    sensitive, _ = method.teacher_forced_state_groups(
        model,
        tok,
        ["ab"],
        "x",
        torch.device("cpu"),
        batch_size=1,
    )
    reference, _ = method.teacher_forced_state_groups(
        model,
        tok,
        ["ab"],
        "y",
        torch.device("cpu"),
        batch_size=1,
    )

    # Before either answer's first token, the model has seen exactly the same
    # prompt. Labeling one state positive and the other negative makes any
    # linear-reader selectivity threshold impossible.
    assert torch.equal(sensitive[0][0], reference[0][0])


def test_differentiable_nll_pair_matches_official_compatible_evaluator():
    torch.manual_seed(5)
    model = TinyContextLM()
    tok = CharacterBatchTokenizer()
    instance = method.mcf_repair.MCFPromptInstance(
        record_index=0,
        sampled_position=0,
        prompt_type="rewrite",
        prompt_index=0,
        prompt="ab",
        target_new="x",
        target_true="y",
    )

    differentiable = method.differentiable_instance_nlls(
        model,
        tok,
        [instance],
        torch.device("cpu"),
        llama_like=False,
    )
    exact = method.evaluate_instance_nlls(
        model,
        tok,
        [instance],
        torch.device("cpu"),
        llama_like=False,
        batch_size=1,
    )

    assert torch.allclose(differentiable[0].detach(), exact[0], atol=1e-6)
    assert torch.allclose(differentiable[1].detach(), exact[1], atol=1e-6)
    (differentiable[0].sum() + differentiable[1].sum()).backward()
    assert model.output_embeddings.weight.grad is not None


def test_reference_nll_constraint_penalizes_regression_but_allows_improvement():
    baseline = torch.tensor([2.0, 2.0, 2.0])
    increased = torch.tensor([2.0, 2.2, 2.0], requires_grad=True)
    decreased = torch.tensor([2.0, 1.8, 2.0], requires_grad=True)

    increase_loss = method.reference_nll_regression_penalty(increased, baseline, 0.05)
    decrease_loss = method.reference_nll_regression_penalty(decreased, baseline, 0.05)

    assert float(increase_loss) > 0.0
    assert float(decrease_loss) == 0.0
    increase_loss.backward()
    decrease_loss.backward()
    assert float(increased.grad[1]) > 0.0
    assert float(decreased.grad.abs().max()) == 0.0


def test_protected_subspace_projection_removes_only_registered_span():
    protected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    basis = core.orthonormal_row_basis(protected, max_rank=2)
    raw = torch.tensor([[3.0, 4.0, 5.0], [-2.0, 7.0, -11.0]])

    projected = core.project_out(raw, basis)

    assert torch.allclose(projected @ basis.T, torch.zeros(2, 2), atol=1e-6)
    assert torch.allclose(projected[:, 2], raw[:, 2], atol=1e-6)


def test_residual_reader_basis_stays_in_writer_span_and_rejects_protection():
    common = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    negatives = torch.tensor([[0.0, 2.0, 0.0, 0.0]])
    residuals = torch.tensor([[3.0, 4.0, 5.0, 0.0], [2.0, -3.0, 4.0, 0.0]])

    basis, report = core.residual_reader_basis(
        residuals,
        common,
        negatives,
        residual_rank=2,
        row_negative_rank=1,
    )

    assert basis.shape == (1, 4)
    assert torch.allclose(basis @ common.T, torch.zeros(1, 1), atol=1e-6)
    assert torch.allclose(basis @ negatives.T, torch.zeros(1, 1), atol=1e-6)
    assert report["residual_basis_rank"] == 1
    assert report["protected_projection_abs_max"] < 1e-6


def test_row_basis_delta_and_hard_relative_cap():
    basis = torch.zeros(2, 2, 3)
    basis[0, 0, 0] = 1.0
    basis[0, 1, 1] = 1.0
    basis[1, 0, 1] = 1.0
    basis[1, 1, 2] = 1.0
    coefficients = torch.tensor([[3.0, 4.0], [0.0, 6.0]])
    base_norms = torch.tensor([10.0, 4.0])

    report = core.clamp_basis_coefficients_(coefficients, base_norms, 0.5)
    delta = core.row_basis_deltas(coefficients, basis)

    assert torch.allclose(coefficients.norm(dim=1), torch.tensor([5.0, 2.0]))
    assert torch.allclose(delta.norm(dim=1), torch.tensor([5.0, 2.0]))
    assert report["clamped_rows"] == 1
    assert report["max_relative_norm"] <= 0.5 + 1e-6


def test_materialized_row_delta_ste_matches_bfloat16_rows_and_keeps_gradient():
    base = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    raw = torch.tensor([[0.013, -0.027]], requires_grad=True)

    effective = core.materialized_row_delta_ste(raw, base)
    expected = (base + raw.detach().to(torch.bfloat16)).float() - base.float()

    assert torch.equal(effective.detach(), expected)
    effective.sum().backward()
    assert torch.equal(raw.grad, torch.ones_like(raw))


def test_materialized_hard_cap_bounds_the_serialized_delta():
    base = torch.tensor(
        [[0.50, -0.25, 0.125], [0.30, 0.40, -0.20]],
        dtype=torch.bfloat16,
    )
    basis = torch.zeros(2, 2, 3)
    basis[0, 0, 0] = 1.0
    basis[0, 1, 1] = 1.0
    basis[1, 0, 1] = 1.0
    basis[1, 1, 2] = 1.0
    coefficients = torch.tensor([[4.0, 3.0], [5.0, -7.0]])

    report = core.clamp_materialized_basis_coefficients_(
        coefficients, basis, base, 0.10
    )
    effective = core.materialized_row_delta_ste(
        core.row_basis_deltas(coefficients, basis), base
    )
    ratios = effective.norm(dim=1) / base.float().norm(dim=1)

    assert float(ratios.max()) <= 0.10 + 1e-6
    assert report["materialized_violating_rows"] == 0


def test_component_ppl_row_replacement_toggles_only_selected_rows():
    layer = torch.nn.Embedding(5, 3)
    original = layer.weight.detach().clone()
    replacement = torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])

    component_ppl.replace_selected_rows(layer, [1, 4], replacement)

    assert torch.equal(layer.weight[1], replacement[0])
    assert torch.equal(layer.weight[4], replacement[1])
    assert torch.equal(layer.weight[0], original[0])
    assert torch.equal(layer.weight[2], original[2])
    assert torch.equal(layer.weight[3], original[3])


def test_gen_diagnostic_separates_direct_only_surrogate_coverage():
    official = {
        "forget_raw": [
            {
                "requested_rewrite": {"subject": "Ada"},
                "post": {
                    "paraphrase_prompts_probs": [
                        {"target_true": 1.0, "target_new": 2.0},
                        {"target_true": 3.0, "target_new": 2.0},
                    ]
                },
            },
            {
                "requested_rewrite": {"subject": "Grace"},
                "post": {
                    "paraphrase_prompts_probs": [
                        {"target_true": 4.0, "target_new": 2.0},
                        {"target_true": 5.0, "target_new": 2.0},
                    ]
                },
            },
        ]
    }
    surrogate = {
        "records": [
            {
                "case_id": 1,
                "subject": "Ada",
                "augmentation_status": "direct_only",
                "surrogate_prompts": [],
            },
            {
                "case_id": 2,
                "subject": "Grace",
                "augmentation_status": "robust_prompt_set",
                "surrogate_prompts": ["Records say Grace worked in"],
            },
        ]
    }

    report = gen_diagnostic.analyze(official, surrogate)

    assert report["official_gen_recomputed"] == 25.0
    assert report["groups"]["direct_only"]["sensitive_preference_percent"] == 50.0
    assert report["groups"]["robust_prompt_set"]["sensitive_preference_percent"] == 0.0
    assert [row["subject"] for row in report["failed_records"]] == ["Ada"]


def test_beta_frontier_scales_are_sorted_unique_and_bounded():
    assert beta_frontier.parse_scales("1,0.1,0,0.1,0.01") == [
        0.0,
        0.01,
        0.1,
        1.0,
    ]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        beta_frontier.parse_scales("1.1")


def test_beta_frontier_relative_norm_and_ppl_safe_selection():
    base = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    delta = torch.tensor([[0.3, 0.4], [0.0, 1.0]])
    ratios = beta_frontier.relative_row_norms(delta, base)
    rows = [
        {"scale": 0.0, "output_only_ppl_percent_delta": 0.0},
        {"scale": 0.1, "output_only_ppl_percent_delta": 4.9},
        {"scale": 0.3, "output_only_ppl_percent_delta": 5.1},
    ]

    assert torch.allclose(ratios, torch.tensor([0.1, 0.5]))
    assert beta_frontier.choose_largest_ppl_safe_scale(rows, limit_percent=5.0) == 0.1


def test_surrogate_loader_accepts_audited_robust_adapter_direct_only_rows(tmp_path):
    records = [
        {
            "case_id": 10,
            "requested_rewrite": {
                "subject": "Gautier de Coincy",
                "prompt": "{} speaks",
                "target_true": {"str": "French"},
                "target_new": {"str": "English"},
            },
        },
        {
            "case_id": 11,
            "requested_rewrite": {
                "subject": "Melchior de Vogue",
                "prompt": "{} speaks",
                "target_true": {"str": "French"},
                "target_new": {"str": "German"},
            },
        },
    ]
    artifact = {
        "schema_version": 1,
        "protocol": "mcf_direct_only_robust_prompt_adapter_v7",
        "seed": 1,
        "forget_num": 2,
        "semantic_validation": {"protocol": "semantic-validator"},
        "data_access": {
            "official_paraphrase_seen": 0,
            "official_neighborhood_seen": 0,
            "benchmark_retain_seen": 0,
            "official_PPL_seen": False,
        },
        "records": [
            {
                "case_id": 10,
                "sampled_position": 0,
                "subject": "Gautier de Coincy",
                "direct_prompt": "Gautier de Coincy speaks",
                "augmentation_status": "direct_only",
                "surrogate_prompts": [],
            },
            {
                "case_id": 11,
                "sampled_position": 1,
                "subject": "Melchior de Vogue",
                "direct_prompt": "Melchior de Vogue speaks",
                "augmentation_status": "robust_prompt_set",
                "surrogate_prompts": ["Records say Melchior de Vogue speaks"],
            },
        ],
    }
    path = tmp_path / "surrogates.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    prompts, receipt = method.load_surrogate_prompts(
        path, records, seed=1, require_semantic=True
    )

    assert prompts == [[], ["Records say Melchior de Vogue speaks"]]
    assert receipt["protocol"] == "mcf_direct_only_robust_prompt_adapter_v7"
