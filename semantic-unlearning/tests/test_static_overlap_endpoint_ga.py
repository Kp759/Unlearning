"""Actual tied-endpoint gradients, immutable support, rollback, and export gates."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from test_static_overlap_edit import bundle, tokenizer, deterministic, tiny
from static_overlap_core import answer_nll, model_logits, tied_weights
from static_overlap_data import encode_bundle
from static_overlap_training import TrainConfig, export_verified, sha256_file
from static_overlap_endpoint_ga import EndpointEditor, endpoint_step, locality_hashes, verify_locality
from static_overlap_endpoint_protocol import PLAN, METHOD
from run_static_overlap_mlp_pilot import References, measure_pilot
from run_static_overlap_endpoint_ga import fit


def examples_for(bundle, tokenizer):
    return [replace(e, split="development" if e.split == "validation" else e.split)
            for e in encode_bundle(bundle, tokenizer, abstention="")]


def test_shared_direct_delta_matches_native_forward_and_both_gradient_paths():
    model = tiny(tied=True)
    frozen = locality_hashes(model, [5, 6, 8])
    ids = torch.tensor([[2, 5, 7, 6, 8]])
    base = model(ids).logits.detach()
    editor = EndpointEditor(model, [5, 6], [6, 8])
    assert torch.equal(model(ids).logits, base)
    assert editor.parameters == [editor.edit.delta]
    assert editor.edit.delta.shape == (3, 16)  # full width, not a rank factor
    with torch.no_grad():
        editor.edit.delta.normal_(std=.03)
    loss = model(ids).logits.square().sum()
    both = torch.autograd.grad(loss, editor.edit.delta)[0]
    # A physical shared-matrix perturbation must have exactly the same gradient
    # as the sum of its embedding path and its head path.
    wrapped_input = model.get_input_embeddings()
    model.set_input_embeddings(editor.embedding)
    head_only = torch.autograd.grad(model(ids).logits.square().sum(), editor.edit.delta)[0]
    assert both.norm() > 0 and head_only.norm() > 0
    model.set_input_embeddings(wrapped_input)
    expected = model(ids).logits.detach()
    with editor.base():
        assert torch.equal(model(ids).logits, base)
    editor.merge()
    torch.testing.assert_close(model(ids).logits, expected, atol=2e-6, rtol=1e-5)
    model.get_input_embeddings().weight.requires_grad_(True)
    native = torch.autograd.grad(model(ids).logits.square().sum(), model.get_input_embeddings().weight)[0]
    torch.testing.assert_close(native[[5, 6, 8]], both, atol=2e-5, rtol=2e-5)
    assert tied_weights(model)
    assert verify_locality(model, [5, 6, 8], frozen)["all_transformer_weights_exact"]
    with torch.no_grad():
        model.get_input_embeddings().weight[9, 0] += .1
    with pytest.raises(ValueError, match="Frozen weights"):
        verify_locality(model, [5, 6, 8], frozen)
    with pytest.raises(RuntimeError, match="Already merged"):
        editor.merge()


def test_untied_and_changed_artifact_support_are_rejected():
    with pytest.raises(ValueError, match="original unquantized tied"):
        EndpointEditor(tiny(tied=False), [5], [8])
    editor = EndpointEditor(tiny(tied=True), [5], [8])
    artifact = deepcopy(editor.artifact())
    artifact["rows"][0] = 6
    with pytest.raises(ValueError, match="support"):
        editor.load_artifact(artifact)


def test_forget_ga_has_correct_sign_and_rejected_step_rolls_back(bundle, tokenizer, tmp_path, monkeypatch):
    import static_overlap_endpoint_ga as core
    model = tiny(len(tokenizer), tied=True)
    examples = examples_for(bundle, tokenizer)
    ref = References(tmp_path / "refs")
    ref.build(model, examples)
    rows = sorted({t for e in examples if e.role == "forget" for t in e.input_ids} - set(tokenizer.all_special_ids))
    editor = EndpointEditor(model, rows, rows)
    bf = [e for e in examples if e.split == "train" and e.role == "forget"][:1]
    br = [e for e in examples if e.split == "train" and e.role == "retain"][:1]
    optimizer = torch.optim.Adam(editor.parameters, lr=.003)
    config = TrainConfig()
    before = answer_nll(model_logits(model, bf[0]), bf[0]).item()
    result = endpoint_step(editor, optimizer, bf, br, ref, config, PLAN)
    assert result["accepted"] and result["mode"] == "forget_ga"
    assert answer_nll(model_logits(model, bf[0]), bf[0]).item() > before
    assert result["forget_gradient_norm"] > 0
    delta = editor.edit.delta.detach().clone()
    state = deepcopy(optimizer.state_dict())
    monkeypatch.setattr(core, "batch_scores", lambda *a: {"gap": 1e6, "violation": 1e6, "finite": True})
    result = endpoint_step(editor, optimizer, bf, br, ref, config, PLAN)
    assert not result["accepted"] and torch.equal(editor.edit.delta, delta)
    assert optimizer.state_dict()["param_groups"] == state["param_groups"]
    for key, value in state["state"][0].items():
        assert torch.equal(optimizer.state_dict()["state"][0][key], value)
    with pytest.raises(ValueError, match="Only fitting"):
        endpoint_step(editor, optimizer, [replace(bf[0], split="development")], br, ref, config, PLAN)


def test_baseline_early_stop_and_no_export_on_failed_gate(bundle, tokenizer, tmp_path):
    model = tiny(len(tokenizer), tied=True)
    examples = examples_for(bundle, tokenizer)
    ref = References(tmp_path / "refs")
    ref.build(model, examples)
    editor = EndpointEditor(model, [5, 6], [6, 8])
    plan = {**PLAN, "steps": 5, "check_every": 1, "forget_batch": 1, "retain_batch": 1,
            "min_gate_nll_gain": 1e6, "stalled_gates": 2}
    report = fit(editor, examples, ref, TrainConfig(), plan, tmp_path)
    assert report["stop_reason"] == "insufficient_training_forgetting_progress"
    assert len(report["gates"]) == 2
    assert report["selected_step"] is None and not report["native_checkpoint_created"]
    assert report["development_used_for_gradients"] is False
    assert (tmp_path / "baseline_metrics.json").exists()
    assert (tmp_path / "last_endpoint_delta.pt").exists()
    assert not (tmp_path / "checkpoint").exists()
    with pytest.raises(RuntimeError, match="forgetting gate failed|retention budgets failed"):
        export_verified(editor, tokenizer, examples, TrainConfig(), tmp_path / "checkpoint", torch.float32,
            lambda p: pytest.fail("Failed gate must not reach reload"), numeric_slack=0., require_forgetting=True)
    assert not (tmp_path / "checkpoint/static_edit_export.json").exists()


def test_native_shared_export_strict_retention_and_mask(bundle, tokenizer, tmp_path):
    from transformers import LlamaForCausalLM
    model = tiny(len(tokenizer), tied=True)
    examples = examples_for(bundle, tokenizer)
    frozen = locality_hashes(model, [5, 6, 8])
    editor = EndpointEditor(model, [5, 6], [6, 8])
    # Identity export exercises strict success with a trivial unit-test target;
    # nonzero native parity and shared-matrix gradients are checked above.
    # Production registration always fixes the target at 1e-6 plus two NLL.
    def reload(path):
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
        assert verify_locality(model, [5, 6, 8], frozen)["original_tying_preserved"]
        return model
    result = export_verified(editor, tokenizer, examples, TrainConfig(forget_increase=0, target_probability=.999),
        tmp_path / "checkpoint", torch.float32, reload, numeric_slack=0., require_forgetting=True)
    assert result["verified"] and result["shared_endpoints"]
    assert result["export_retention_policy"]["float32_numeric_slack"] == 0


def test_endpoint_registration_uses_existing_data_and_separate_evaluation_lock(tmp_path, monkeypatch):
    import static_overlap_endpoint_protocol as p
    old = tmp_path / "mlp.json"
    old.write_text("immutable prior protocol")
    completed = tmp_path / "completed_head_results.json"
    completed.write_text("negative result")
    previous = {"base_model_path": str(tmp_path / "base"), "data": {"path": "frozen data"},
        "source_bundle": {"path": "frozen source"}, "head_protocol_path": "frozen head",
        "head_protocol_sha256": "head hash"}
    monkeypatch.setattr(p, "load_mlp", lambda path: previous)
    mask = tmp_path / "overlap.json"
    mask.write_text(json.dumps({"architecture": "static_overlap_constrained_embedding_mlp_head_v1",
        "shared_endpoints": True, "model_path": previous["base_model_path"], "input_rows": [5], "output_rows": [8]}))
    root = tmp_path / "endpoint"
    p.main(["--development-protocol", str(old), "--overlap-manifest", str(mask), "--output-dir", str(root)])
    path = root / "pilot_protocol.json"
    assert p.load_pilot(path)["data"] == previous["data"]
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "training_manifest.json").write_text(json.dumps({"method": METHOD,
        "exploratory_protocol_sha256": sha256_file(path)}))
    (checkpoint / "static_edit_export.json").write_text("model A")
    assert p.claim_evaluation(path, checkpoint) == p.claim_evaluation(path, checkpoint)
    (checkpoint / "static_edit_export.json").write_text("model B")
    with pytest.raises(ValueError, match="another checkpoint"):
        p.claim_evaluation(path, checkpoint)
    assert completed.read_text() == "negative result"
    assert old.read_text() == "immutable prior protocol"
    mask.write_text("changed mask")
    with pytest.raises(ValueError, match="mask manifest changed"):
        p.load_pilot(path)


def test_registered_cli_failed_gate_cannot_touch_final_evaluation(bundle, tokenizer, tmp_path, monkeypatch):
    import static_overlap_endpoint_protocol as p
    from run_static_overlap_endpoint_ga import main as run
    from evaluate_static_overlap_endpoint_ga import main as evaluate
    base = tmp_path / "base"
    tiny(len(tokenizer), tied=True).save_pretrained(base)
    tokenizer.save_pretrained(base)
    old = tmp_path / "old.json"
    old.write_text("previous development contract")
    data = {"authored": [
        {"id": "dev_f", "role": "forget", "split": "development", "fact_id": "forget_language",
         "prompt": "Person A's native language is", "answer": "French", "family": "development"},
        {"id": "dev_r", "role": "retain", "split": "development", "fact_id": "retain_occupation",
         "prompt": "Person A's occupation is", "answer": "engineer", "family": "development"}],
        "language": [{"id": "dev_l", "split": "development", "text": "Warm sunlight brings a quiet evening into the room."}],
        "training_text_fingerprints": []}
    previous = {"base_model_path": str(base), "head_protocol_path": "must-not-load-final", "head_protocol_sha256": "frozen"}
    for name, content in (("data", data), ("source_bundle", bundle)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(content))
        previous[name] = {"path": str(path), "sha256": sha256_file(path)}
    monkeypatch.setattr(p, "load_mlp", lambda path: previous)
    monkeypatch.setattr(p, "PLAN", {**PLAN, "steps": 1, "check_every": 1, "forget_batch": 1, "retain_batch": 1})
    mask = tmp_path / "overlap.json"
    mask.write_text(json.dumps({"architecture": "static_overlap_constrained_embedding_mlp_head_v1",
        "shared_endpoints": True, "model_path": str(base), "input_rows": [5, 6], "output_rows": [6, 8],
        "forget_associations": [f for f in bundle["facts"] if f["role"] == "forget"]}))
    root = tmp_path / "run"
    p.main(["--development-protocol", str(old), "--overlap-manifest", str(mask), "--output-dir", str(root)])
    protocol = root / "pilot_protocol.json"
    assert run(["--pilot-protocol", str(protocol), "--model-path", str(base), "--device", "cpu", "--local-files-only"]) == 2
    assert not (root / "checkpoint").exists()
    assert json.loads((root / "training_report.json").read_text())["selected_step"] is None
    with pytest.raises(ValueError, match="Development/export gate failed"):
        evaluate(["--pilot-protocol", str(protocol), "--wikidata-dir", "must-not-be-read", "--device", "cpu"])
    assert not (root / "exploratory_evaluation_started.json").exists()
    with pytest.raises(FileExistsError):
        run(["--pilot-protocol", str(protocol), "--model-path", str(base), "--device", "cpu"])
