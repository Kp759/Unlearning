"""Regression coverage for the September pilot's metric and no-progress failures."""
from copy import deepcopy
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_static_overlap_edit import zero_forgetting_check, apply_mcf_probability_metrics
from mcf_zero_unlearn_metric_parity import summarize_probability_metrics
from mcf_zero_unlearn_official_eval import official_test_batch_prediction
from static_overlap_core import StaticEditor, constrained_step, project_update
from static_overlap_data import Example
from static_overlap_training import TrainConfig, forget_target, train, within_budgets


def tokenizer(bos, padding):
    backend = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "[BOS]": 2,
                                         "Ada": 3, "from": 4, "New": 5, "York": 6, "Rome": 7},
                                        unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    if bos:
        backend.post_processor = processors.TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 2)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                                  bos_token="[BOS]", padding_side=padding)


class DistributionLM(torch.nn.Module):
    def forward(self, input_ids, **kwargs):
        # P(New)=.1 and P(York)=.2; P(New York)=.02, geometric mean=sqrt(.02).
        probs = torch.tensor([.1, .1, .1, .1, .2, .1, .2, .1])
        return SimpleNamespace(logits=probs.log().expand(*input_ids.shape, 8))


class NextTokenLM(torch.nn.Module):
    def forward(self, input_ids, **kwargs):
        logits = torch.full((*input_ids.shape, 8), -10.0)
        logits[:, :-1].scatter_(2, input_ids[:, 1:, None], 10.0)
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("bos", [False, True])
@pytest.mark.parametrize("padding", ["left", "right"])
def test_actual_answer_tokens_with_and_without_bos_and_padding(bos, padding):
    tok = tokenizer(bos, padding)
    rows, flags = official_test_batch_prediction(DistributionLM(), tok, ["Ada", "Ada from"],
                                                "Rome", "New York", "cpu", llama_like=True,
                                                return_correct=True)
    for row in rows:
        assert row["target_true_tokens"] == 2
        assert row["target_true_nll_sum"] == pytest.approx(-math.log(.02))
        assert row["target_true"] == pytest.approx(-math.log(.02) / 2)
    assert flags == [False, False]
    _, flags = official_test_batch_prediction(NextTokenLM(), tok, ["Ada", "Ada from"],
                                              "Rome", "New York", "cpu", return_correct=True)
    assert flags == [True, True]
    one, _ = official_test_batch_prediction(DistributionLM(), tok, ["Ada"], "Rome", "New", "cpu",
                                            return_correct=True)
    assert one[0]["target_true_tokens"] == 1  # Never an empty/vacuously correct answer.
    assert math.exp(-one[0]["target_true_nll_sum"]) == pytest.approx(.1)


def raw_case(probability, count, flags=(False,)):
    item = {"target_true": -math.log(probability) / count,
            "target_true_nll_sum": -math.log(probability), "target_true_tokens": count,
            "target_new": 20.0}
    return {"post": {key: value for group in ("rewrite", "paraphrase", "neighborhood")
                     for key, value in ((f"{group}_prompts_probs", [dict(item) for _ in flags]),
                                        (f"{group}_prompts_correct", list(flags)))}}


def test_formula_uses_product_then_prompt_and_case_means():
    rows = [raw_case(.01, 2), raw_case(.04, 2, (True, False, False))]
    result = summarize_probability_metrics({"Eff": 84, "Gen": 86, "Spe": 12}, rows)
    assert result["Eff"] == pytest.approx(2.5)
    assert result["Gen"] == pytest.approx(2.5)
    assert result["TokenGeometricMean_Eff"] == pytest.approx(15.0)
    assert result["ReleasedAccuracy_Gen"] == pytest.approx(100 / 6)
    assert result["Spe"] == pytest.approx(100 / 6)
    assert result["Legacy_Spe_ProbabilityDiff"] == 12


def test_static_entry_point_uses_probability_metrics():
    rows = [raw_case(.000001, 2)]
    result = apply_mcf_probability_metrics({"forget": {"Eff": 84, "Gen": 86}, "forget_raw": rows,
                                             "retain": {}, "retain_raw": rows})
    assert result["forget"]["Eff"] == pytest.approx(.0001)
    assert zero_forgetting_check(result["forget"])["passed"]
    assert result["forget"]["Eff"] > 0  # Do not manufacture zero by rounding.


@pytest.mark.parametrize("error", ["missing", "nan", "inconsistent"])
def test_invalid_or_legacy_raw_data_cannot_be_silently_scored(error):
    rows = [raw_case(.01, 2)]
    item = rows[0]["post"]["rewrite_prompts_probs"][0]
    if error == "missing":
        del item["target_true_tokens"]
    elif error == "nan":
        item["target_true_nll_sum"] = float("nan")
    else:
        item["target_true"] = 99
    with pytest.raises(ValueError):
        summarize_probability_metrics({}, rows)


def test_zero_check_does_not_use_rounded_scores_or_only_accuracy():
    values = {"Eff": .00499, "Gen": .00499, "ReleasedAccuracy_Eff": 0., "ReleasedAccuracy_Gen": 0.}
    assert zero_forgetting_check(values)["passed"]
    for key, value in (("Eff", .005), ("Gen", float("nan")), ("ReleasedAccuracy_Gen", 1.)):
        assert not zero_forgetting_check({**values, key: value})["passed"]


def test_suppression_target_does_not_stop_at_plus_two_nll():
    config = TrainConfig()
    assert forget_target(1.0, config) > 13
    assert math.exp(-forget_target(1.0, config)) <= config.target_probability * (1 + 1e-12)
    assert forget_target(20.0, config) == 22.0


def test_forget_descent_fallback_applies_only_if_constraints_and_progress_pass():
    p = torch.nn.Parameter(torch.tensor([0.0]))
    optimizer = torch.optim.Adam([p], lr=.1)
    state = deepcopy(optimizer.state_dict())
    record = constrained_step(optimizer, [p], p.sum(), torch.empty(0, 1),
                              lambda: (0 < p.item() <= .06, {}), epsilon=0, radius=1,
                              fallback_direction=torch.ones(1))
    assert record["accepted"] and record["direction"] == "forget_descent"
    assert p.item() == pytest.approx(.05)
    assert optimizer.state_dict() == state


def test_projection_accepts_distinct_remaining_nll_and_kl_budgets():
    result = project_update(torch.tensor([1., 1.]), torch.eye(2), torch.tensor([.1, .01]), radius=2.)
    assert result.converged
    torch.testing.assert_close(result.delta, torch.tensor([.1, .01]))


class SeparableFactLM(torch.nn.Module):
    """Two distinct contexts share an answer; only one association is forgotten."""
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 8)
        self.head = torch.nn.Linear(8, 8, bias=False)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(8))
            self.head.weight.zero_()
            self.head.weight[5, 1] = self.head.weight[5, 2] = self.head.weight[6, 3] = 4

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, module):
        self.head = module

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


@pytest.mark.parametrize("protected_batch_size", [1, 8])
def test_training_can_suppress_a_fact_without_stalling_at_kl_boundary(protected_batch_size):
    torch.manual_seed(1)
    torch.set_num_threads(1)
    examples = []
    for split in ("train", "validation"):
        for role, prompt, answer, fact in (("forget", 1, 5, "f"), ("abstain", 1, 7, "f"),
                                          ("retain", 2, 5, "r"), ("language", 3, 6, None)):
            examples.append(Example(f"{split}:{role}", split, role, fact, [prompt, answer], [-100, answer],
                                    str(prompt), str(answer), f"{split}:{fact}"))
    editor = StaticEditor(SeparableFactLM(), [], [5, 7], {}, rank=8)
    config = TrainConfig(steps=100, learning_rate=.1, epsilon=.005, step_radius=.25,
                         protected_batch_size=protected_batch_size)
    result = train(editor, examples, config)
    assert result["stop_reason"] == "training_forgetting_target"
    assert result["training_forgetting"]["target_met"]
    assert result["validation_forgetting"]["target_met"]
    assert within_budgets(result["validation"], config)[0]
    assert result["training_forgetting"]["max_token_probability"] < 1e-6
    assert result["training_protection"]["retention_passed"]
    assert all(not key.startswith("validation:") for r in result["history"]
               for key in r["projected_anchor_ids"])
    if protected_batch_size == 1:
        assert any(r["discovered_anchor_ids"] for r in result["history"])
        assert any(r["active_anchor_ids"] for r in result["history"])


def test_continuation_measures_forgetting_without_rebasing_or_fitting_failed_validation():
    torch.manual_seed(1)
    examples = []
    for split in ("train", "validation"):
        for role, prompt, answer, fact in (("forget", 1, 5, "f"), ("abstain", 1, 7, "f"),
                                          ("retain", 2 if split == "train" else 1, 5, "r"),
                                          ("language", 3, 6, None)):
            examples.append(Example(f"{split}:{role}", split, role, fact, [prompt, answer], [-100, answer],
                                    str(prompt), str(answer), f"{split}:{fact}"))
    # Deliberately conflicting validation control: a real retention failure,
    # not permission to use that example's gradient or relax its budget.
    editor = StaticEditor(SeparableFactLM(), [], [5, 7], {}, rank=8)
    config = TrainConfig(steps=5, learning_rate=.1)
    before = train(editor, examples, config)
    assert not before["validation_protection"]["retention_passed"]
    assert before["training_protection"]["retention_passed"]
    with pytest.raises(ValueError, match="zero effective deltas"):
        train(editor, examples, config)
    after = train(editor, examples, config, resume=True)
    assert after["initial_training_forgetting"] == before["training_forgetting"]
    assert after["initial_training_protection"] == before["training_protection"]
    assert after["training_forget"][0]["base_nll"] == before["training_forget"][0]["base_nll"]
    assert after["training_forget_loss"] < after["initial_training_forget_loss"]
    assert after["training_protection"]["retention_passed"]
    assert not after["validation_protection"]["retention_passed"]
    assert all(not key.startswith("validation:") for r in after["history"] for key in r["projected_anchor_ids"])
    with pytest.raises(ValueError, match="outside the original training retention budgets"):
        train(editor, examples, TrainConfig(steps=1, retain_nll_budget=0., retain_kl_budget=0.), resume=True)


def test_priority_training_preserves_earlier_valid_checkpoint_and_excludes_validation_gradients(tmp_path, monkeypatch):
    torch.manual_seed(1)
    examples = []
    for split in ("train", "validation"):
        for role, prompt, answer, fact in (("forget", 1, 5, "f"), ("abstain", 1, 7, "f"),
                                          ("retain", 2 if split == "train" else 1, 5, "r"),
                                          ("language", 3, 6, None)):
            examples.append(Example(f"{split}:{role}", split, role, fact, [prompt, answer], [-100, answer],
                                    str(prompt), str(answer), f"{split}:{fact}"))
    editor = StaticEditor(SeparableFactLM(), [], [5, 7], {}, rank=8)
    import static_overlap_training as training
    original_logits = training.model_logits

    def audited_logits(model, example):
        if example.split != "train":
            assert not torch.is_grad_enabled()
        return original_logits(model, example)

    monkeypatch.setattr(training, "model_logits", audited_logits)
    config = TrainConfig(steps=40, learning_rate=.01, step_radius=.02, max_step_radius=.02,
                         radius_growth=1., hard_example_mix=1., compare_forget_candidates=True,
                         select_best_valid_checkpoint=True, fresh_start_only=True,
                         retain_nll_safety_margin=.01, retain_kl_safety_margin=.002)
    result = train(editor, examples, config, tmp_path / "training.jsonl")
    assert result["checkpoint_selection"]["selected_step"] is not None
    assert result["checkpoint_selection"]["selected_step"] < result["last_iterate"]["step"]
    assert result["validation_protection"]["retention_passed"]
    assert not result["last_iterate"]["validation_protection"]["retention_passed"]
    assert result["training_protection"]["applied_nll_budget"] == .04
    assert result["validation_protection"]["applied_nll_budget"] == .05
    assert (tmp_path / "last_training_factors.pt").is_file()
    assert len(list((tmp_path / "accepted_checkpoints").glob("*.pt"))) == result["accepted_steps"]
    snapshot = torch.load(tmp_path / "accepted_checkpoints" / f"step_{result['checkpoint_selection']['selected_step']:06d}.pt",
                          weights_only=True)
    restored = StaticEditor(SeparableFactLM(), [], [5, 7], {}, rank=8)
    restored.load_artifact(snapshot)
    from static_overlap_training import measure
    assert measure(restored, examples) == measure(editor, examples)
    assert all(not key.startswith("validation:") for row in result["history"] for key in row["projected_anchor_ids"])
    with pytest.raises(ValueError, match="fresh start"):
        train(editor, examples, config, resume=True)


@pytest.mark.parametrize("reject_all", [False, True])
def test_priority_batches_start_with_most_remembered_fact_and_keep_complete_coverage(monkeypatch, reject_all):
    torch.manual_seed(1)
    examples = []
    for split in ("train", "validation"):
        for role, prompt, answer, fact in (("forget", 1, 5, "f"), ("abstain", 1, 7, "f"),
                                          ("forget", 4, 6, "g"), ("abstain", 4, 7, "g"),
                                          ("retain", 2, 5, "r"), ("language", 3, 6, "l")):
            examples.append(Example(f"{split}:{role}:{fact}", split, role, fact, [prompt, answer], [-100, answer],
                                    str(prompt), str(answer), f"{split}:{fact}"))
    editor = StaticEditor(SeparableFactLM(), [], [5, 6, 7], {}, rank=8)
    if reject_all:
        monkeypatch.setattr("static_overlap_training.constrained_step", lambda *a, **kw:
                            {"accepted": False, "step_norm": 0., "forget_progress": 0.})
    result = train(editor, examples, TrainConfig(steps=4, batch_size=1, hard_example_mix=1.,
                                                compare_forget_candidates=True, learning_rate=.01,
                                                max_stalled_steps=1 if reject_all else 10))
    assert result["history"][0]["forget_batch_ids"] == ["train:forget:f"]
    assert result["history"][1]["forget_batch_ids"] == ["train:forget:g"]
    assert result["forget_examples_seen"] == result["forget_examples_total"] == 2
    assert all(row["global_forget_progress"] > 0 for row in result["history"] if row["accepted"])
    if reject_all:
        assert len(result["history"]) == 2
        assert result["stop_reason"] == "no_useful_feasible_step"
