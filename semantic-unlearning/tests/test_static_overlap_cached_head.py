"""Independent numerical checks for the deadline cached-head experiment."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from test_static_overlap_edit import bundle, tokenizer, tiny, deterministic
from static_overlap_cached_head import (HeadCache, RetainMetricSolver, augment_contexts,
    audit_prefix_conflicts, cache_head, cached_measure, cached_token_statistics, summarize)
from static_overlap_core import StaticEditor, answer_nll, forward_kl, model_logits
from static_overlap_data import Example, encode_bundle, endpoint_rows
from static_overlap_training import TrainConfig, measure
from run_static_overlap_cached_head import main, parse_args
from export_static_overlap_edit import main as recover
from evaluate_static_overlap_edit import verify_checkpoint


def test_cached_full_vocabulary_nll_kl_matches_model_and_merge(bundle, tokenizer):
    model = tiny(len(tokenizer))
    examples = encode_bundle(bundle, tokenizer)
    _, rows = endpoint_rows({f["id"]: f for f in bundle["facts"]}, examples, tokenizer, False)
    cache = cache_head(model, examples, rows)
    base_logits = [model_logits(model, e).detach() for e in examples]
    delta = torch.randn(len(rows), model.config.hidden_size) * 0.2
    predicted = cached_measure(cache, delta)
    editor = StaticEditor(model, [], rows, {}, len(rows))
    with torch.no_grad():
        editor.rows["head"].A.copy_(torch.eye(len(rows)))
        editor.rows["head"].B.copy_(delta.T)
    for i, e in enumerate(examples):
        edited = model_logits(model, e)
        assert predicted[i]["nll"] == pytest.approx(answer_nll(edited, e).item(), abs=2e-6)
        assert predicted[i]["kl"] == pytest.approx(forward_kl(base_logits[i], edited, e).item(), abs=2e-6)
    before = [model_logits(model, e).detach() for e in examples]
    embedding = model.get_input_embeddings().weight.clone()
    editor.merge()
    assert torch.equal(model.get_input_embeddings().weight, embedding)
    for e, logits in zip(examples, before):
        torch.testing.assert_close(model_logits(model, e), logits, atol=2e-6, rtol=1e-5)


def test_cache_handles_large_mass_shifts_without_subtraction_cancellation():
    # Almost all baseline mass is in the modified row; 1 - p would round to 0.
    full = torch.tensor([[50., 0., -3.], [-2., 0., 1.]], dtype=torch.float64)
    lp = full.log_softmax(-1)
    cache = HeadCache([], torch.eye(2), lp[:, :1], lp[:, 1:].logsumexp(-1),
                      -lp[:, 1], torch.tensor([-1, -1]), torch.arange(2), torch.tensor([0]))
    delta = torch.tensor([[-100., 80.]])
    nll, kl = cached_token_statistics(cache, delta)
    changed = full.clone()
    changed[:, 0] += delta[0]
    lq = changed.log_softmax(-1)
    torch.testing.assert_close(nll, -lq[:, 1], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(kl, (lp.exp() * (lp-lq)).sum(-1), atol=1e-12, rtol=1e-12)


def artificial_cache():
    roles = [("train", "forget"), ("train", "forget"), ("train", "retain"),
             ("train", "language"), ("validation", "retain"), ("validation", "forget")]
    examples = [Example(str(i), split, role, str(i), [0, 1], [-100, 1], "", "", str(i))
                for i, (split, role) in enumerate(roles)]
    h = torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 1.], [1., 0., 0., 0.],
                      [0., 1., 0., 0.], [1., 1., 0., 0.], [0., 0., 1., 1.]])
    lp = torch.tensor([math.log(.8)]*6, dtype=torch.float64)
    return HeadCache(examples, h, lp[:, None], torch.full((6,), math.log(.2), dtype=torch.float64),
                     -lp, torch.zeros(6, dtype=torch.long), torch.arange(6), torch.tensor([1]))


def test_nullspace_regression_achieves_near_zero_when_separation_exists():
    cache = artificial_cache()
    solver = RetainMetricSolver(cache)
    assert solver.diagnostics["retain_nullspace_dimension"] == 2
    delta = solver.solve(0, 1e-4) * 24
    report = summarize(cached_measure(cache, delta), TrainConfig())
    assert report["eligible"] and report["training_forgetting"]["target_met"]
    assert report["training_forgetting"]["max_token_probability"] < 1e-9
    assert abs(report["training_protection"]["max_retained_nll_increase"]) < 1e-12
    assert report["validation_protection"]["max_retained_kl"] < 1e-12


def test_same_hidden_forget_retain_cannot_be_erased_by_nullspace_solver():
    cache = artificial_cache()
    cache.hidden[0] = cache.hidden[2]
    delta = RetainMetricSolver(cache).solve(0, 1e-4) * 24
    report = summarize(cached_measure(cache, delta), TrainConfig())
    assert report["training_forgetting"]["max_token_probability"] == pytest.approx(.8)
    assert not report["training_forgetting"]["target_met"]


def test_solver_matches_independent_primal_ridge_solution():
    cache = artificial_cache()
    cache.hidden = torch.randn(6, 4)
    solver = RetainMetricSolver(cache)
    tau, ridge = .03, .01
    R = cache.hidden[2:4].double() / solver.scale
    M = torch.eye(4, dtype=torch.float64) + R.T @ R / tau
    expected = torch.linalg.solve(solver.F.T @ solver.F + ridge*M, solver.F.T @ solver.T)
    expected = expected.T / solver.scale
    torch.testing.assert_close(solver.solve(tau, ridge).double(), expected, atol=1e-6, rtol=1e-6)


def test_validation_features_never_enter_the_solve_or_forget_selection():
    cache = artificial_cache()
    delta = RetainMetricSolver(cache).solve(.01, .001)
    before = summarize(cached_measure(cache, delta), TrainConfig())
    cache.hidden[4:] *= -1000
    torch.testing.assert_close(RetainMetricSolver(cache).solve(.01, .001), delta, atol=0, rtol=0)
    after = summarize(cached_measure(cache, delta), TrainConfig())
    assert before["score"] == after["score"]


def test_augmentation_preserves_validation_and_audit_distinguishes_shared_tokens(bundle, tokenizer):
    augmented = augment_contexts(bundle)
    assert [r for r in augmented["examples"] if r["split"] == "validation"] == [
        r for r in bundle["examples"] if r["split"] == "validation"]
    assert len(augmented["examples"]) > len(bundle["examples"])
    examples = encode_bundle(bundle, tokenizer)
    assert not audit_prefix_conflicts(examples)
    e = next(e for e in examples if e.role == "forget" and e.split == "train")
    assert audit_prefix_conflicts(examples + [replace(e, id="conflict", role="retain")])
    with pytest.raises(ValueError, match="original bundle"):
        augment_contexts(augmented)


def test_tied_head_cache_fails_before_fitting(bundle, tokenizer):
    with pytest.raises(ValueError, match="untied"):
        cache_head(tiny(len(tokenizer), tied=True), encode_bundle(bundle, tokenizer), [4])


def test_validation_failure_cannot_be_selected_even_with_perfect_training_forgetting():
    cache = artificial_cache()
    # Validation retain shares the forgotten direction. A training-only solution
    # can be excellent while being unusable under the scientific validation limit.
    cache.hidden[4] = cache.hidden[0]
    delta = RetainMetricSolver(cache).solve(0, 1e-4) * 24
    report = summarize(cached_measure(cache, delta), TrainConfig())
    assert report["training_forgetting"]["target_met"]
    assert report["training_protection"]["retention_passed"]
    assert not report["eligible"] and not report["validation_protection"]["retention_passed"]
    assert report["validation_protection"]["nominal_retain_nll_budget"] == .05
    assert report["validation_protection"]["nominal_retain_kl_budget"] == .01


@pytest.mark.parametrize("flag,value", [("--taus", "-1"), ("--ridges", "0"), ("--strengths", "nan")])
def test_cli_rejects_invalid_grid(tmp_path, flag, value):
    with pytest.raises(SystemExit):
        parse_args(["--model-path", "base", "--training-bundle", str(ROOT / "config/static_overlap_training.example.json"),
                    "--output-dir", str(tmp_path / "new"), flag, value])


def test_cli_native_export_and_saved_factor_recovery(bundle, tokenizer, tmp_path):
    # End to end with a real tiny Llama, including its tokenizer, hashes and reload.
    from transformers import LlamaConfig, LlamaForCausalLM
    base = tmp_path / "base"
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tokenizer), hidden_size=64,
        intermediate_size=80, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, tie_word_embeddings=False, pad_token_id=0,
        bos_token_id=2, eos_token_id=3, max_position_embeddings=128)).eval()
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    source = tmp_path / "bundle.json"
    source.write_text(json.dumps(bundle))
    out = tmp_path / "run"
    result = main(["--model-path", str(base), "--training-bundle", str(source),
                   "--output-dir", str(out), "--device", "cpu", "--local-files-only",
                   "--no-context-augmentation", "--taus", "0.01", "--ridges", "0.01",
                   "--strengths", "0.01", "0.1", "1"])
    assert result == 0
    exported = verify_checkpoint(out / "checkpoint")
    assert exported["verified"]
    report = json.loads((out / "training_report.json").read_text())
    assert report["cache_model_parity"]["passed"] and report["native_checkpoint_created"]
    assert report["selected_actual"]["eligible"]
    recover(["--training-run", str(out), "--device", "cpu", "--local-files-only"])
    assert verify_checkpoint(out / "checkpoint_float32")["verified"]
