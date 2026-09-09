from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import sys

import pytest
import torch
from torch import nn
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from static_overlap_core import (StaticEditor, answer_nll, bounded_forget,
    constrained_step, flat_parameters, forward_kl, localize, model_logits,
    project_update, tied_weights)
from static_overlap_data import encode_bundle, endpoint_rows, validate_bundle
from static_overlap_training import TrainConfig, export_verified, measure, train, within_budgets
from evaluate_static_overlap_edit import evaluate_bundle, verify_checkpoint


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(4)
    torch.set_num_threads(1)


@pytest.fixture
def bundle():
    return json.loads((ROOT / "config/static_overlap_training.example.json").read_text())


@pytest.fixture
def tokenizer(bundle):
    texts = [r.get("text") or r["prompt"] + r["completion"] for r in bundle["examples"]]
    texts += ["I don't know."]
    words = sorted(set(re.findall(r"\w+|[^\w\s]+", " ".join(texts))))
    vocab = {word: i for i, word in enumerate(["[PAD]", "[UNK]", "[BOS]", "[EOS]"] + words)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 2)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                                  bos_token="[BOS]", eos_token="[EOS]")


def tiny(vocab=128, tied=False):
    return LlamaForCausalLM(LlamaConfig(vocab_size=vocab, hidden_size=16, intermediate_size=80,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        tie_word_embeddings=tied, attention_dropout=0.0, bos_token_id=2, eos_token_id=3,
        pad_token_id=0, max_position_embeddings=128)).eval()


def make_editor(model):
    return StaticEditor(model, [5, 6], [6, 8], {0: [1, 3, 5], 1: [2, 4, 6]}, rank=8)


@pytest.mark.parametrize("shared", [False, True])
def test_static_support_identity_gradients_and_native_merge(shared):
    model = tiny(tied=shared)
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    ids = torch.tensor([[2, 5, 7, 6, 8]])
    expected = model(ids).logits.detach()
    editor = make_editor(model)
    assert torch.equal(model(ids).logits, expected)
    assert len(editor.rows) == (1 if shared else 2)
    assert len({id(p) for p in editor.parameters}) == len(editor.parameters)
    model(ids).logits.sum().backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in (edit.A, edit.B)) for edit in editor.edits)
    with torch.no_grad():
        for edit in editor.edits:
            edit.A.normal_(std=0.03)
            edit.B.normal_(std=0.03)
    factorized = model(ids).logits.detach()
    with editor.base():
        assert torch.equal(model(ids).logits, expected)
    editor.merge()
    torch.testing.assert_close(model(ids).logits, factorized, atol=1e-6, rtol=1e-5)
    assert tied_weights(model) is shared
    assert not any(p.requires_grad for p in model.parameters())
    changed = set()
    for name, p in model.named_parameters():
        old = original[name]
        if torch.equal(old, p):
            continue
        changed.add(name)
        mask = torch.zeros_like(p, dtype=torch.bool)
        if name == "model.embed_tokens.weight":
            mask[[5, 6, 8] if shared else [5, 6]] = True
        elif name == "lm_head.weight":
            mask[[6, 8]] = True
        elif name.endswith("mlp.down_proj.weight"):
            mask[:, [1, 3, 5] if ".0." in name else [2, 4, 6]] = True
        else:
            pytest.fail(f"Unregistered parameter changed: {name}")
        assert torch.equal(old[~mask], p[~mask])
    assert len(changed) == (3 if shared else 4)
    with pytest.raises(RuntimeError, match="Already merged"):
        editor.merge()


def test_tie_configuration_mismatch_fails():
    model = tiny(tied=False)
    model.lm_head.weight = model.model.embed_tokens.weight
    with pytest.raises(ValueError, match="disagrees"):
        make_editor(model)


def test_hinge_stops_and_kl_uses_full_base_distribution(bundle, tokenizer):
    nll = torch.tensor(4.0, requires_grad=True)
    bounded_forget(nll, 1.0, 2.0).backward()
    assert nll.grad.item() == 0
    nll = torch.tensor(2.0, requires_grad=True)
    bounded_forget(nll, 1.0, 2.0).backward()
    assert nll.grad.item() == -1
    e = encode_bundle(bundle, tokenizer)[0]
    base = torch.randn(len(e.input_ids), len(tokenizer))
    changed = base + torch.randn_like(base)
    assert forward_kl(base, base, e).item() == 0
    assert forward_kl(base, changed, e).item() > 0
    assert not torch.allclose(forward_kl(base, changed, e), forward_kl(changed, base, e))


def test_projection_solves_halfspaces_and_ball():
    proposal = torch.tensor([2.0, 2.0])
    gradients = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    result = project_update(proposal, gradients, 0.0, 1.0)
    assert result.converged
    torch.testing.assert_close(result.delta, torch.tensor([0.0, 1.0]))
    result = project_update(proposal, torch.empty(0, 2), 0.0, 1.0)
    torch.testing.assert_close(result.delta, proposal / proposal.norm())


def test_projection_matches_independent_qp():
    from scipy.optimize import minimize
    import numpy as np
    proposal = torch.randn(5, dtype=torch.float64)
    gradients = torch.randn(4, 5, dtype=torch.float64)
    eps, radius = 0.03, 0.7
    result = project_update(proposal, gradients, eps, radius, tolerance=1e-9)
    assert result.converged
    u, g = proposal.numpy(), gradients.numpy()
    reference = minimize(lambda x: 0.5 * np.square(x - u).sum(), np.zeros(5),
        jac=lambda x: x - u, method="SLSQP", options={"ftol": 1e-12, "maxiter": 1000},
        constraints=[{"type": "ineq", "fun": lambda x: eps - g @ x},
                     {"type": "ineq", "fun": lambda x: radius ** 2 - x @ x}])
    assert reference.success
    np.testing.assert_allclose(result.delta.numpy(), reference.x, atol=2e-6)


def test_nonfinite_or_unconverged_projection_cannot_be_applied():
    failed = project_update(torch.tensor([float("nan"), 1.0]), torch.tensor([[1.0, 0.0]]), 0, 1)
    assert not failed.converged and torch.equal(failed.delta, torch.zeros(2))
    failed = project_update(torch.tensor([2.0, 2.0]), torch.tensor([[1.0, 0.0]]), 0, 1,
                            max_iterations=1)
    assert not failed.converged and torch.equal(failed.delta, torch.zeros(2))


def test_optimizer_backtracks_on_actual_loss_and_restores_rejected_state():
    p = nn.Parameter(torch.tensor([0.2]))
    optimizer = torch.optim.Adam([p], lr=1.0)
    report = constrained_step(optimizer, [p], -p.sum(), torch.tensor([[0.4]]),
        lambda: (p.square().item() <= 0.06, {}), epsilon=0.08, radius=1.0)
    assert report["accepted"] and report["backtracks"] > 0
    assert p.square().item() <= 0.06
    before, state = p.detach().clone(), deepcopy(optimizer.state_dict())
    rejected = constrained_step(optimizer, [p], -p.sum(), torch.tensor([[0.4]]),
        lambda: (False, {}), epsilon=0.08, radius=1.0, backtracks=2)
    assert not rejected["accepted"]
    assert torch.equal(p, before)
    for key, value in state["state"][0].items():
        torch.testing.assert_close(optimizer.state_dict()["state"][0][key], value)


def test_mixed_span_masks_and_alias_selection(bundle, tokenizer):
    examples = encode_bundle(bundle, tokenizer)
    original = [e for e in examples if e.group == "train_mixed" and ":0:" in e.id]
    assert {e.role for e in original} == {"forget", "retain"}
    forget, retain = original
    assert not any(a != -100 and b != -100 for a, b in zip(forget.labels, retain.labels))
    rewritten = [e for e in examples if e.group == "train_mixed" and ":1:" in e.id]
    assert {e.role for e in rewritten} == {"abstain", "retain"}
    assert all(e.completion == " I don't know. ; French" for e in rewritten)
    facts = validate_bundle(bundle)
    inputs, outputs = endpoint_rows(facts, examples, tokenizer)
    assert tokenizer.convert_tokens_to_ids("native") not in inputs
    assert tokenizer.convert_tokens_to_ids("French") in outputs
    assert not set(tokenizer.all_special_ids) & (set(inputs) | set(outputs))


def test_data_rejects_leaks_missing_overlaps_and_truncation(bundle, tokenizer):
    bad = deepcopy(bundle)
    bad["examples"][0]["paraphrase_prompts"] = ["official heldout probe"]
    with pytest.raises(ValueError, match="Unknown"):
        validate_bundle(bad)
    bad = deepcopy(bundle)
    # Same-answer restaurant controls are optional in the MCF protocol; the
    # same-relation/different-subject control remains mandatory.
    bad["examples"] = [r for r in bad["examples"] if r["id"] != "train_retain_other_person"]
    with pytest.raises(ValueError, match="overlap controls"):
        validate_bundle(bad)
    bad = deepcopy(bundle)
    bad["examples"][-1]["text"] = bad["examples"][6]["text"]
    with pytest.raises(ValueError, match="Duplicate text"):
        validate_bundle(bad)
    with pytest.raises(ValueError, match="truncate"):
        encode_bundle(bundle, tokenizer, max_length=2)
    bad = deepcopy(bundle)
    bad["examples"][7]["prompt"] = bad["examples"][0]["prompt"]
    with pytest.raises(ValueError, match="same prompt"):
        validate_bundle(bad)


def test_token_boundaries_cannot_supervise_part_of_prompt(bundle, tokenizer):
    # Force a tokenizer token to contain both non-answer prompt text and answer.
    bad = deepcopy(bundle)
    bad["examples"][0]["prompt"] = "PrefixFrench"
    bad["examples"][0]["completion"] = "French"
    bad["examples"][0]["spans"][0].update(start=0, end=6)
    with pytest.raises(ValueError, match="non-answer text"):
        encode_bundle(bad, tokenizer)


def test_localization_uses_only_training_and_removes_hooks(bundle, tokenizer):
    examples = encode_bundle(bundle, tokenizer)
    model = tiny(len(tokenizer))
    f = [e for e in examples if e.role == "forget" and e.split == "train"]
    r = [e for e in examples if e.role == "retain" and e.split == "train"]
    selected, _ = localize(model, f, r, blocks=2, channels_per_block=64)
    assert len(selected) == 2 and all(len(set(v)) == 64 for v in selected.values())
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    with pytest.raises(ValueError, match="validation"):
        localize(model, [e for e in examples if e.role == "forget"], r)


def fitted(bundle, tokenizer, steps=2, shared=True):
    examples = encode_bundle(bundle, tokenizer)
    facts = validate_bundle(bundle)
    inputs, outputs = endpoint_rows(facts, examples, tokenizer)
    editor = StaticEditor(tiny(len(tokenizer), tied=shared), inputs, outputs,
                          {0: list(range(64)), 1: list(range(64))}, rank=8)
    config = TrainConfig(steps=steps, learning_rate=0.01, step_radius=0.2,
                         epsilon=0.1, retain_nll_budget=1.0, retain_kl_budget=0.2)
    report = train(editor, examples, config)
    return editor, examples, config, report


def test_joint_training_changes_all_sites_and_respects_base_budgets(bundle, tokenizer):
    editor, examples, config, report = fitted(bundle, tokenizer, shared=False)
    assert report["accepted_steps"] > 0
    assert report["forget_examples_seen"] == report["forget_examples_total"]
    assert not report["training_forgetting"]["target_met"]
    assert all(row["forget_progress"] >= config.min_forget_progress
               for row in report["history"] if row["accepted"])
    assert all(edit.delta().norm().item() > 0 for edit in editor.edits)
    passed, _ = within_budgets(measure(editor, [e for e in examples if e.role in ("retain", "language")]), config)
    assert passed
    rows = [{"role": "retain", "nll": 2.0, "base_nll": 1.0, "nll_increase": 1.0, "kl": 0.0}]
    assert not within_budgets(rows, TrainConfig(retain_nll_budget=0.01))[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_export_native_reload_tying_and_generation(bundle, tokenizer, tmp_path, dtype):
    editor, examples, config, _ = fitted(bundle, tokenizer)
    editor.model.generation_config.suppress_tokens = [5, 6]
    editor.model.generation_config.eos_token_id = [3, 4]
    checkpoint = tmp_path / "checkpoint"
    def reload_model(path):
        return LlamaForCausalLM.from_pretrained(path, torch_dtype=dtype, local_files_only=True)
    report = export_verified(editor, tokenizer, examples, config, checkpoint, dtype,
                             reload_model, atol=0.02, rtol=0.02)
    assert report["verified"] and report["shared_endpoints"]
    assert report["forgetting_target_met"] is False
    assert report["export_retention_policy"]["float32_numeric_slack"] == 5e-6
    assert report["merged"]["protection"]["numerical_slack"] == 5e-6
    for stage in ("deployment", "reloaded"):
        protection = report[stage]["protection"]
        assert protection["numerical_slack"] == (5e-6 if dtype == torch.float32 else 0.0)
        assert protection["nominal_retain_nll_budget"] == config.retain_nll_budget
        assert protection["nominal_retain_kl_budget"] == config.retain_kl_budget
    verify_checkpoint(checkpoint)
    model = reload_model(checkpoint)
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert model.generation_config.suppress_tokens is None
    assert model.generation_config.eos_token_id == [3, 4]
    assert not any("edit" in key or ".base." in key for key in model.state_dict())
    report = evaluate_bundle(model, tokenizer, bundle, max_new_tokens=2)
    assert report["language_tokens"] > 0
    assert "same_subject_same_answer_other_relation" in report["summary"]
    assert report["runtime_router"] is False
    with (checkpoint / "config.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="changed"):
        verify_checkpoint(checkpoint)


def test_failed_export_has_no_success_marker(bundle, tokenizer, tmp_path):
    editor, examples, config, _ = fitted(bundle, tokenizer, steps=1)
    def broken_reload(path):
        model = LlamaForCausalLM.from_pretrained(path, local_files_only=True)
        with torch.no_grad():
            model.lm_head.weight.add_(50)
        return model
    with pytest.raises(RuntimeError, match="parity"):
        export_verified(editor, tokenizer, examples, config, tmp_path / "bad",
                         torch.float32, broken_reload, atol=1e-5, rtol=1e-5)
    assert not (tmp_path / "bad/static_edit_export.json").exists()
    failure = json.loads((tmp_path / "bad/static_edit_export_failure.json").read_text())
    assert failure["stage"] == "reload"
    assert failure["failing_logits"] > 0


@pytest.mark.parametrize("shared", [False, True])
def test_saved_factors_restore_exactly_and_reject_mismatches(shared):
    model = tiny(tied=shared)
    clean = deepcopy(model)
    editor = make_editor(model)
    with torch.no_grad():
        for p in editor.parameters:
            p.normal_(std=.01)
    saved = editor.artifact()
    restored = make_editor(clean)
    restored.load_artifact(saved)
    ids = torch.tensor([[2, 5, 7, 6, 8]])
    torch.testing.assert_close(clean(ids).logits, model(ids).logits, atol=0, rtol=0)
    before = flat_parameters(restored.parameters).clone()
    bad = deepcopy(saved)
    next(iter(bad["writeouts"].values()))["B"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Invalid saved factor"):
        restored.load_artifact(bad)
    torch.testing.assert_close(flat_parameters(restored.parameters), before, atol=0, rtol=0)
    bad = deepcopy(saved)
    next(iter(bad["rows"].values()))["rows"][0] = 0
    with pytest.raises(ValueError, match="indices differ"):
        restored.load_artifact(bad)


def test_export_failure_identifies_merge_stage(bundle, tokenizer, tmp_path, monkeypatch):
    editor, examples, config, _ = fitted(bundle, tokenizer, steps=1)
    merge = editor.merge
    def broken_merge():
        model = merge()
        with torch.no_grad():
            model.lm_head.weight.add_(50)
        return model
    monkeypatch.setattr(editor, "merge", broken_merge)
    with pytest.raises(RuntimeError, match="stage=merge"):
        export_verified(editor, tokenizer, examples, config, tmp_path / "bad_merge",
                        torch.float32, lambda _: pytest.fail("Should not reach reload"), atol=1e-4, rtol=1e-4)
    failure = json.loads((tmp_path / "bad_merge/static_edit_export_failure.json").read_text())
    assert failure["stage"] == "merge"
    assert not (tmp_path / "bad_merge/static_edit_export.json").exists()


def test_recovery_rejects_changed_training_statistics(bundle, tokenizer):
    from export_static_overlap_edit import verify_recovered_statistics
    editor, examples, _, report = fitted(bundle, tokenizer, steps=1)
    assert verify_recovered_statistics(editor, examples, report)["matched_examples"] > 0
    bad = deepcopy(report)
    bad["validation"][0]["base_nll"] += 1.0
    with pytest.raises(ValueError, match="Recovered base_nll differs"):
        verify_recovered_statistics(editor, examples, bad)
    bad = deepcopy(report)
    bad["validation"][0]["nll"] += 1.0
    with pytest.raises(ValueError, match="Recovered nll differs"):
        verify_recovered_statistics(editor, examples, bad)


def test_recover_float32_after_cast_failure_without_retraining(bundle, tokenizer, tmp_path, monkeypatch):
    import run_static_overlap_edit as runner
    from export_static_overlap_edit import main as recover
    from static_overlap_training import sha256_file
    base = tmp_path / "base"
    tiny(len(tokenizer), tied=True).save_pretrained(base)
    tokenizer.save_pretrained(base)
    settings = json.loads((ROOT / "config/static_overlap_edit.json").read_text())
    settings["training"].update(steps=1, retain_nll_budget=1.0, retain_kl_budget=.2)
    settings.update(export_atol=1e-4, export_rtol=1e-4)
    config_path, bundle_path = tmp_path / "config.json", tmp_path / "bundle.json"
    config_path.write_text(json.dumps(settings))
    bundle_path.write_text(json.dumps(bundle))
    run = tmp_path / "run"
    with pytest.raises(RuntimeError, match="stage=cast"):
        runner.main(["--model-path", str(base), "--training-bundle", str(bundle_path),
                     "--output-dir", str(run), "--config", str(config_path), "--device", "cpu",
                     "--dtype", "float32", "--deployment-dtype", "bfloat16", "--local-files-only"])
    failure = json.loads((run / "checkpoint/static_edit_export_failure.json").read_text())
    assert failure["stage"] == "cast"
    assert failure["max_abs_error"] > 1e-4
    assert not (run / "checkpoint/static_edit_export.json").exists()
    saved_hashes = {name: sha256_file(run / name) for name in
                    ("training_factors.pt", "training_report.json", "manifest.json")}
    monkeypatch.setattr(runner, "train", lambda *a, **kw: pytest.fail("Recovery must not train"))
    monkeypatch.setattr(runner, "localize", lambda *a, **kw: pytest.fail("Recovery must not localize"))
    args = ["--training-run", str(run), "--device", "cpu", "--local-files-only"]
    recover(args)
    checkpoint = run / "checkpoint_float32"
    report = verify_checkpoint(checkpoint)
    assert report["deployment_dtype"] == "torch.float32"
    assert report["training_dtype"] == "torch.float32"
    assert all(sha256_file(run / name) == digest for name, digest in saved_hashes.items())
    manifest = json.loads((checkpoint / "training_manifest.json").read_text())
    assert manifest["recovery"]["optimizer_steps"] == 0
    assert manifest["recovery"]["reproduction"]["matched_examples"] > 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        recover(args)
    # A different byte-level bundle cannot be paired with these factors.
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(bundle) + "\n")
    with pytest.raises(ValueError, match="bundle hash"):
        recover(args + ["--training-bundle", str(altered), "--output-dir", str(run / "other")])


def test_cli_localize_train_export(bundle, tokenizer, tmp_path):
    from run_static_overlap_edit import main
    from evaluate_static_overlap_edit import main as evaluate_main
    base = tmp_path / "base"
    tiny(len(tokenizer), tied=True).save_pretrained(base)
    tokenizer.save_pretrained(base)
    config = json.loads((ROOT / "config/static_overlap_edit.json").read_text())
    config["training"].update(steps=1, retain_nll_budget=1.0, retain_kl_budget=0.2)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))
    main(["--model-path", str(base), "--training-bundle", str(bundle_path),
          "--output-dir", str(tmp_path / "run"), "--config", str(config_path),
          "--device", "cpu", "--dtype", "float32", "--deployment-dtype", "float32", "--local-files-only"])
    assert verify_checkpoint(tmp_path / "run/checkpoint")["verified"]
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["architecture"] == "static_overlap_constrained_embedding_mlp_head_v1"
    assert len(manifest["selected_channels"]) == 2
    # An unrelated old sidecar must never be auto-attached by the new evaluator.
    (tmp_path / "run/checkpoint/scoped_span_edit.pt").write_bytes(b"not a loadable router")
    heldout = deepcopy(bundle)
    heldout["purpose"] = "evaluation"
    heldout["examples"] = [r for r in heldout["examples"] if r["split"] == "validation"]
    for row in heldout["examples"]:
        row["split"] = "test"
        key = "text" if row.get("role") == "language" else "prompt"
        row[key] = "Evaluation question: " + row[key]
    heldout_path = tmp_path / "heldout.json"
    heldout_path.write_text(json.dumps(heldout))
    eval_args = ["--checkpoint", str(tmp_path / "run/checkpoint"), "--evaluation-bundle",
                 str(heldout_path), "--out", str(tmp_path / "evaluation.json"),
                 "--device", "cpu", "--max-new-tokens", "2", "--base-model", str(base)]
    evaluate_main(eval_args)
    evaluation = json.loads((tmp_path / "evaluation.json").read_text())
    assert evaluation["runtime_guard"] is False
    assert evaluation["base"]["generation_performed"] is False
    assert "forget" in evaluation["change_vs_base"]["bundle"]
    # Exercise the actual static CLI's official scoring and failing exit gate.
    rr = {"subject": "Person A", "relation_id": "native language",
          "prompt": "The native language of {} is", "target_true": {"str": "French"},
          "target_new": {"str": "engineer"}}
    record = {"case_id": 1, "requested_rewrite": rr,
              "paraphrase_prompts": ["State Person A's native language:"],
              "neighborhood_prompts": ["The native language of Person B is"]}
    retained = deepcopy(record)
    retained["case_id"] = 0
    retained["requested_rewrite"]["subject"] = "Person B"
    mcf = tmp_path / "mcf.json"
    mcf.write_text(json.dumps([retained, record]))
    with pytest.raises(SystemExit, match="NOT met"):
        evaluate_main(eval_args + ["--mcf-path", str(mcf), "--unlearn-num", "1", "--retain-num", "1",
                                  "--skip-official-ppl", "--require-zero"])
    evaluation = json.loads((tmp_path / "evaluation.json").read_text())
    assert evaluation["official_mcf"]["forget"]["metric_version"] == "zerounlearn_answer_probability_v2"
    assert "ReleasedAccuracy_Gen" in evaluation["official_mcf"]["forget"]
    assert not evaluation["forgetting_check"]["passed"]
    assert "Gen_change" in evaluation["change_vs_base"]["official_mcf"]["forget"]
    assert evaluation["generation"] and evaluation["language_ppl"] > 0
    heldout["examples"][0]["prompt"] = bundle["examples"][0]["prompt"]
    heldout_path.write_text(json.dumps(heldout))
    with pytest.raises(ValueError, match="overlaps"):
        evaluate_main(eval_args)


def test_cli_continues_saved_factors_without_relocalizing_or_exporting(bundle, tokenizer, tmp_path, monkeypatch):
    import run_static_overlap_edit as runner
    from static_overlap_training import sha256_file

    base = tmp_path / "base"
    tiny(len(tokenizer), tied=True).save_pretrained(base)
    tokenizer.save_pretrained(base)
    config = json.loads((ROOT / "config/static_overlap_edit.json").read_text())
    config["training"].update(steps=2, retain_nll_budget=1.0, retain_kl_budget=.2)
    config_path, bundle_path = tmp_path / "config.json", tmp_path / "bundle.json"
    config_path.write_text(json.dumps(config))
    bundle_path.write_text(json.dumps(bundle))
    args = ["--model-path", str(base), "--training-bundle", str(bundle_path), "--config", str(config_path),
            "--device", "cpu", "--dtype", "float32", "--local-files-only", "--training-only"]
    parent = tmp_path / "parent"
    runner.main(args + ["--output-dir", str(parent)])
    original = json.loads((parent / "training_report.json").read_text())
    assert original["accepted_steps"] > 0
    hashes = {name: sha256_file(parent / name) for name in
              ("manifest.json", "training_factors.pt", "training_report.json")}
    monkeypatch.setattr(runner, "localize", lambda *a, **kw: pytest.fail("Must reuse saved localization"))
    monkeypatch.setattr(runner, "export_verified", lambda *a, **kw: pytest.fail("Training-only must not export"))
    output = tmp_path / "continued"
    runner.main(args + ["--resume-training-run", str(parent), "--output-dir", str(output), "--steps", "1"])
    result = json.loads((output / "training_report.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert result["resumed_from_factors"] and result["optimizer_state"] == "reset"
    assert result["initial_training_forgetting"] == original["training_forgetting"]
    assert result["initial_training_protection"] == original["training_protection"]
    assert manifest["continuation"]["retention_reference"] == "original_base_model"
    assert manifest["continuation"]["reproduction"]["matched_examples"] > 0
    assert not (output / "checkpoint").exists() and not (parent / "checkpoint").exists()
    assert all(sha256_file(parent / name) == value for name, value in hashes.items())
    changed = tmp_path / "changed_bundle.json"
    changed.write_text(json.dumps(bundle) + "\n")
    with pytest.raises(ValueError, match="original bundle"):
        runner.main(args + ["--resume-training-run", str(parent), "--output-dir", str(tmp_path / "bad"),
                            "--training-bundle", str(changed)])
    config["architecture"]["blocks"] = 1
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="original bundle, architecture"):
        runner.main(args + ["--resume-training-run", str(parent), "--output-dir", str(tmp_path / "bad_arch")])


def test_one_layer_preset_changes_only_block_count_and_selects_highest_training_contrast(bundle, tokenizer):
    from run_static_overlap_edit import load_config
    one, _ = load_config(ROOT / "config/static_overlap_one_layer.json")
    two, _ = load_config(ROOT / "config/static_overlap_edit.json")
    assert one["architecture"]["blocks"] == 1
    one["architecture"]["blocks"] = 2
    assert one == two
    examples = encode_bundle(bundle, tokenizer)
    selected, report = localize(tiny(len(tokenizer)),
        [e for e in examples if e.role == "forget" and e.split == "train"],
        [e for e in examples if e.role == "retain" and e.split == "train"], blocks=1)
    best = min(report, key=lambda layer: (-report[layer]["score"], layer))
    assert list(selected) == [best] and len(selected[best]) == 64
