"""Data isolation, strict gates, single-MLP locality, and separate exploratory locks."""
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
from static_overlap_core import StaticEditor, model_logits
from static_overlap_data import encode_bundle
from static_overlap_training import TrainConfig, export_verified, measure, sha256_file, within_export_budgets
from static_overlap_mlp_protocol import (PLAN, METHOD, authored_prompts, build_data,
                                         claim_evaluation, digest_json, load_pilot)
from run_static_overlap_mlp_pilot import (References, balanced_subset, development_gate,
    fitting_batch, fit, measure_pilot, select_layer, verify_locality, weight_hashes)


def split_examples(bundle, tokenizer):
    return [replace(e, split="development" if e.split == "validation" else e.split)
            for e in encode_bundle(bundle, tokenizer, abstention="")]


def test_authored_families_are_separate_and_unknown_relations_fail():
    f = {"subject": "Alice", "relation": "P17", "aliases": ["Approved Alice alias"]}
    train, dev = list(authored_prompts(f, "train")), list(authored_prompts(f, "development"))
    assert any("country" in p and "?" in p for _, p in train)
    assert any("situated" in p for _, p in train)
    assert any("Approved Alice alias" in p for _, p in train)
    assert set(p for _, p in train).isdisjoint(p for _, p in dev)
    assert all(name.startswith("development_question") for name, _ in dev)
    with pytest.raises(ValueError, match="No authored"):
        list(authored_prompts({**f, "relation": "UNKNOWN"}, "train"))


def test_data_builder_ignores_mcf_probe_fields_and_reports_overlap_availability(bundle):
    source = deepcopy(bundle)
    for f in source["facts"]:
        f["relation"] = "P17"
    records = [{"case_id": i, "requested_rewrite": {"subject": f"Fresh subject {i}",
        "relation_id": "P17", "target_true": {"str": "France"}}} for i in range(40)]
    docs = [(i, f"Paragraph {i} describes a quiet afternoon when the workers carried their tools across the courtyard and into the workshop.") for i in range(10)]
    plan = {**PLAN, "extra_retain_facts": 10, "language_documents": 5}
    expected = build_data(source, records, set(), set(), docs, plan)
    poisoned = [{**r, "paraphrase_prompts": ["DO NOT FIT THIS"], "neighborhood_prompts": ["HIDDEN FINAL"]} for r in records]
    assert expected == build_data(source, poisoned, set(), set(), docs, plan)
    assert expected["coverage_forget_facts"]["train"]["same_relation_other_subject"] == expected["forget_facts"]
    prompts = [r["prompt"] for r in expected["authored"]]
    assert len(prompts) == len(set(prompts))
    # Exclusion-only filtering removes a forbidden variant; insufficient remaining
    # development coverage must fail rather than copy an official prompt.
    forget_id = next(f["id"] for f in source["facts"] if f["role"] == "forget")
    blocked = {r["prompt"].casefold() for r in expected["authored"]
               if r["fact_id"] == forget_id and r["split"] == "development"}
    with pytest.raises(ValueError, match="Insufficient disjoint"):
        build_data(source, records, blocked, set(), docs, plan)


def test_localization_excludes_development_forget_and_uses_nll_gradients(bundle, tokenizer):
    model = tiny(len(tokenizer), tied=True)
    examples = split_examples(bundle, tokenizer)
    original = weight_hashes(model)
    selected, a = select_layer(model, examples, [0, 1], 3, 1)
    altered = [replace(e, input_ids=[2, 4, 5], labels=[-100, -100, 5])
               if e.role == "forget" and e.split == "development" else e for e in examples]
    _, b = select_layer(model, altered, [0, 1], 3, 1)
    assert a == b and selected in (0, 1)
    assert all(r["retain_gradient_norm"] > 0 for r in a["layers"])
    assert weight_hashes(model) == original
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError, match="Only training"):
        fitting_batch(examples, 1, 2)


def test_gate_requires_both_forget_splits_and_full_preservation_without_slack():
    config = TrainConfig()
    rows = [{"id": f"{split}_{role}", "split": split, "role": role,
             "base_nll": 2., "nll": 15. if role == "forget" else 2.,
             "nll_increase": 13. if role == "forget" else 0., "kl": 0.}
            for split in ("train", "development") for role in ("forget", "retain", "language")]
    assert development_gate(rows, config)["passed"]
    failed = deepcopy(rows)
    failed[3]["nll"] = 10.  # development forget
    assert not development_gate(failed, config)["passed"]
    failed = deepcopy(rows)
    failed[4].update(nll=2.050001, nll_increase=.050001)
    assert not development_gate(failed, config)["passed"]
    assert within_export_budgets(failed, config, torch.float32)[0]  # legacy allowance
    assert not within_export_budgets(failed, config, torch.float32, numeric_slack=0)[0]
    failed[4]["kl"] = float("nan")
    assert not development_gate(failed, config)["passed"]


def test_full_vocabulary_cache_real_backward_and_failed_gate_no_export(bundle, tokenizer, tmp_path):
    model = tiny(len(tokenizer), tied=True)
    original = weight_hashes(model)
    examples = split_examples(bundle, tokenizer)
    ref = References(tmp_path / "refs", max_bytes=32)  # force disk reads/eviction
    ref.build(model, examples)
    editor = StaticEditor(model, [], [], {0: list(range(80))}, rank=4)
    initial = measure_pilot(model, examples, ref)
    assert all(abs(r["nll_increase"]) < 2e-6 and r["kl"] < 1e-6 for r in initial)
    plan = {**PLAN, "steps": 2, "check_every": 1, "forget_batch": 2, "retain_batch": 2}
    result = fit(editor, examples, ref, plan, TrainConfig(), tmp_path)
    assert result["selected_step"] is None and not result["native_checkpoint_created"]
    assert not (tmp_path / "checkpoint").exists()
    assert editor.norm_sq().item() > 0  # actual gradient updates occurred
    measured = measure_pilot(model, examples, ref)
    direct = measure(editor, examples)
    for a, b in zip(measured, direct):
        assert a["nll"] == pytest.approx(b["nll"], abs=2e-6)
        assert a["kl"] == pytest.approx(b["kl"] if a["role"] != "forget" else 0., abs=2e-6)
    expected = model_logits(model, examples[0]).detach()
    editor.merge()
    torch.testing.assert_close(model_logits(model, examples[0]), expected, atol=1e-5, rtol=1e-5)
    assert verify_locality(model, original, 0)["all_other_parameters_exact"]
    with torch.no_grad():
        model.get_input_embeddings().weight[0, 0] += .1
    with pytest.raises(ValueError, match="outside"):
        verify_locality(model, original, 0)


def test_strict_export_rejects_unsuppressed_forget_even_with_perfect_retention(bundle, tokenizer, tmp_path):
    examples = split_examples(bundle, tokenizer)
    editor = StaticEditor(tiny(len(tokenizer), tied=True), [], [], {0: list(range(80))}, rank=2)
    with pytest.raises(RuntimeError, match="forgetting gate failed"):
        export_verified(editor, tokenizer, examples, TrainConfig(), tmp_path / "checkpoint", torch.float32,
            lambda p: pytest.fail("No reload before failed merge gate"), numeric_slack=0., require_forgetting=True)
    assert not (tmp_path / "checkpoint/static_edit_export.json").exists()


def test_strict_native_export_preserves_tied_endpoints(bundle, tokenizer, tmp_path):
    from transformers import LlamaForCausalLM
    model = tiny(len(tokenizer), tied=True)
    original = weight_hashes(model)
    editor = StaticEditor(model, [], [], {0: list(range(80))}, rank=2)
    examples = split_examples(bundle, tokenizer)
    # Identity model tests the success path with a deliberately trivial UNIT TEST
    # target. The actual runner's registered target remains 1e-6 plus two NLL.
    config = TrainConfig(forget_increase=0., target_probability=.999)
    def reload(path):
        loaded = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
        assert verify_locality(loaded, original, 0)["embedding_and_head_exact"]
        return loaded
    export_verified(editor, tokenizer, examples, config, tmp_path / "checkpoint", torch.float32,
                    reload, numeric_slack=0., require_forgetting=True)
    report = json.loads((tmp_path / "checkpoint/static_edit_export.json").read_text())
    assert report["verified"] and report["forgetting_target_met"]
    assert report["export_retention_policy"]["float32_numeric_slack"] == 0


def test_exploratory_binding_cannot_overwrite_completed_head_or_reselect(tmp_path, monkeypatch):
    import static_overlap_mlp_protocol as protocol_module
    old = tmp_path / "head"
    old.mkdir()
    (old / "protocol.json").write_text("original frozen head contract")
    (old / "final_retention_results.json").write_text("completed negative result")
    (old / "final_evaluation_started.json").write_text("head checkpoint binding")
    prior = {p.name: p.read_bytes() for p in old.iterdir()}
    root = tmp_path / "pilot"
    root.mkdir()
    for name in ("data", "source_bundle", "head_manifest"):
        (root / f"{name}.json").write_text("{}")
    registry = old / "exploratory_mlp_v1_registered.json"
    p = {"method": METHOD, "exploratory": True, "plan": PLAN,
         "head_protocol_path": str(old / "protocol.json"), "head_protocol_sha256": sha256_file(old / "protocol.json"),
         "registration_path": str(registry),
         **{n: {"path": str(root / f"{n}.json"), "sha256": sha256_file(root / f"{n}.json")}
            for n in ("data", "source_bundle", "head_manifest")}}
    protocol_path = root / "pilot_protocol.json"
    protocol_path.write_text(json.dumps(p))
    registry.write_text(json.dumps({"pilot_protocol_path": str(protocol_path),
                                   "pilot_protocol_sha256": sha256_file(protocol_path)}))
    monkeypatch.setattr(protocol_module, "load_protocol", lambda p: {})
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "training_manifest.json").write_text(json.dumps({"exploratory_protocol_sha256": sha256_file(protocol_path)}))
    (checkpoint / "static_edit_export.json").write_text("model A")
    assert claim_evaluation(protocol_path, checkpoint) == claim_evaluation(protocol_path, checkpoint)
    (checkpoint / "static_edit_export.json").write_text("model B")
    with pytest.raises(ValueError, match="another checkpoint"):
        claim_evaluation(protocol_path, checkpoint)
    assert {name: (old / name).read_bytes() for name in prior} == prior
    (root / "data.json").write_text("tampered")
    with pytest.raises(ValueError, match="input changed"):
        load_pilot(protocol_path)


def test_registered_runner_and_evaluator_stop_on_failed_development_gate(bundle, tokenizer, tmp_path, monkeypatch):
    import static_overlap_mlp_protocol as protocol_module
    from run_static_overlap_mlp_pilot import main as run_main
    from evaluate_static_overlap_mlp_pilot import main as eval_main
    base = tmp_path / "base"
    tiny(len(tokenizer), tied=True).save_pretrained(base)
    tokenizer.save_pretrained(base)
    root = tmp_path / "pilot"
    root.mkdir()
    plan = {**PLAN, "candidate_layers": [0, 1], "localization_examples": 2,
            "rank": 2, "steps": 1, "check_every": 1, "forget_batch": 1, "retain_batch": 1}
    monkeypatch.setattr(protocol_module, "PLAN", plan)
    monkeypatch.setattr(protocol_module, "load_protocol", lambda p: {})
    data = {"authored": [
        {"id": "dev_f", "role": "forget", "split": "development", "fact_id": "forget_language",
         "prompt": "Person A's native language is", "answer": "French", "family": "development"},
        {"id": "dev_r", "role": "retain", "split": "development", "fact_id": "retain_occupation",
         "prompt": "Person A's occupation is", "answer": "engineer", "family": "development"}],
        "language": [{"id": "new_language", "split": "development", "text": "Warm sunlight brings a quiet evening into the room."}],
        "training_text_fingerprints": []}
    for name, value in (("data", data), ("source_bundle", bundle), ("head_manifest", {"model_path": str(base)})):
        (root / f"{name}.json").write_text(json.dumps(value))
    old = tmp_path / "head.json"
    old.write_text("frozen head contract")
    registry = tmp_path / "registry.json"
    protocol = {"method": METHOD, "exploratory": True, "plan": plan, "base_model_path": str(base),
                "head_protocol_path": str(old), "head_protocol_sha256": sha256_file(old),
                "registration_path": str(registry),
                **{n: {"path": str(root / f"{n}.json"), "sha256": sha256_file(root / f"{n}.json")}
                   for n in ("data", "source_bundle", "head_manifest")}}
    path = root / "pilot_protocol.json"
    path.write_text(json.dumps(protocol))
    registry.write_text(json.dumps({"pilot_protocol_path": str(path), "pilot_protocol_sha256": sha256_file(path)}))
    assert run_main(["--pilot-protocol", str(path), "--model-path", str(base),
                     "--device", "cpu", "--local-files-only"]) == 2
    assert not (root / "checkpoint").exists()
    rows = json.loads((root / "encoded_development_examples.json").read_text())
    assert not any(r["role"] == "forget" and r["id"].startswith("validation") for r in rows)
    assert {r["id"] for r in rows if r["split"] == "development"} == {"dev_f", "dev_r", "new_language"}
    with pytest.raises(ValueError, match="Development/export gate failed"):
        eval_main(["--pilot-protocol", str(path), "--wikidata-dir", "must-not-be-read", "--device", "cpu"])
    assert not (root / "exploratory_evaluation_started.json").exists()
    with pytest.raises(FileExistsError):
        run_main(["--pilot-protocol", str(path), "--model-path", str(base), "--device", "cpu"])
