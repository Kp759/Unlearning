"""Shared writer: gradients, runtime gate/boundary, export, and comparison guards."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compressed_value_bank import CompressedValues  # noqa: E402
from linear_router import ARCHITECTURE, load_linear_classifier_artifact  # noqa: E402
from run_mcf_shared_vector_seed1 import validate_reference, validate_shared, comparison_rows  # noqa: E402
from train_mcf_compressed_bank import bank_from_artifact  # noqa: E402
from test_compressed_bank import _FakeModel, FACTS, HIDDEN  # noqa: E402


def test_shared_vector_has_one_parameter_and_combines_fact_gradients():
    values = CompressedValues("shared", None, FACTS, HIDDEN)
    assert sum(p.numel() for p in values.parameters()) == HIDDEN
    optimizer = torch.optim.SGD(values.parameters(), lr=0.1)
    loss = values.rows()[0].sum() + 2 * values.rows()[3].sum()
    loss.backward()
    assert torch.equal(values.shared_vector.grad, torch.full((HIDDEN,), 3.0))
    optimizer.step()
    assert torch.equal(values.rows(), values.rows()[:1].expand(4, -1))
    assert torch.allclose(values.rows()[2], torch.full((HIDDEN,), -0.3))
    assert values.storage()["per_fact_floats"] == 0
    assert values.storage()["extrapolated_total_floats"]["100000"] == HIDDEN


def test_shared_runtime_and_reloaded_artifact_preserve_gate_and_boundary():
    model = _FakeModel().requires_grad_(False)
    source = dict(layer=0, router_weight=torch.zeros(4, HIDDEN), router_bias=torch.ones(4),
                  feature_mean=torch.zeros(HIDDEN), feature_components=None, threshold=0.0,
                  subject_patterns=[[(10,)], [(20,)], [(30,)], [(40,)]], facts=FACTS,
                  ambiguity_margin=0.5, gate_mode="threshold")
    values = CompressedValues("shared", None, FACTS, HIDDEN)
    with torch.no_grad():
        values.shared_vector.copy_(torch.arange(HIDDEN) / 10)
    bank = bank_from_artifact(model, source, values)
    # Two different routed facts, a no-subject prompt, and an ambiguous prompt.
    ids = torch.tensor([[10, 99, 77], [40, 99, 77], [99, 99, 77], [10, 40, 77]])
    hidden = torch.randn(4, 3, HIDDEN)
    bank.bind(ids, torch.ones_like(ids), prefix_lengths=torch.tensor([2, 2, 2, 2]))
    edited = model.model.layers[0](hidden)[0]
    expected = hidden.clone()
    expected[:2, 1] += values.shared_vector
    assert torch.equal(edited, expected)
    assert bank.last_active_fact_indices == [[0], [3], [], []]
    assert sum(p.numel() for p in bank.parameters() if p.requires_grad) == HIDDEN
    artifact = bank.artifact()
    bank.close()
    _, restored = load_linear_classifier_artifact(model, artifact)
    restored.bind(ids, torch.ones_like(ids), prefix_lengths=torch.tensor([2, 2, 2, 2]))
    assert torch.equal(model.model.layers[0](hidden)[0], edited)
    restored.close()


def reference_fixture():
    facts = [{"id": str(i)} for i in range(50)]
    artifact = {"architecture": ARCHITECTURE, "layer": 19, "facts": facts,
                "rows": torch.zeros(50, HIDDEN), "router_weight": torch.zeros(50, HIDDEN)}
    manifest = {"mcf_path": "mcf.json", "sampling": {"seed": 1}}
    hp = dict(lr=.05, scale_lr=.5, batch_facts=8, epochs=300, eval_every=5,
              max_training_seconds=3600, unknown_weight=1, unknown_completion=" I don't know.",
              unknown_eos=True, clip=1, seed=1)
    report = {"value_mode": "full", "training_route": "router", "hyperparameters": hp,
              "training_coverage": {"facts_trained": 50}, "best_epoch": 5}
    return artifact, manifest, report


def test_preflight_rejects_wrong_seed_or_non_eos_baseline():
    artifact, manifest, report = reference_fixture()
    assert validate_reference(artifact, manifest, report)["batch_facts"] == 8
    manifest["sampling"]["seed"] = 2
    with pytest.raises(ValueError, match="seed-1"):
        validate_reference(artifact, manifest, report)
    manifest["sampling"]["seed"] = 1
    report["hyperparameters"]["unknown_eos"] = False
    with pytest.raises(ValueError, match="unknown_eos"):
        validate_reference(artifact, manifest, report)


def test_shared_export_guard_rejects_router_changes_and_fact_specific_edits():
    reference, _, report = reference_fixture()
    shared = deepcopy(reference)
    shared["compressed_values"] = {"mode": "shared", "compact_state": {"shared_vector": torch.zeros(HIDDEN)},
                                   "storage": {"total_floats": HIDDEN}}
    validate_shared(reference, shared, report)
    shared["rows"][2, 0] = 1
    with pytest.raises(ValueError, match="exactly the same"):
        validate_shared(reference, shared, report)
    shared["rows"].zero_()
    shared["router_weight"][0, 0] = 1
    with pytest.raises(ValueError, match="Router"):
        validate_shared(reference, shared, report)


def test_comparison_counts_leakage_even_when_output_also_says_idk():
    official = {label: {"forget": {"Eff": .1, "Gen": .2, "Spe": 20},
                        "retain": {"Eff": 11}, "forget_PPL": 12}
                for label in ("full_50", "shared_1")}
    generations = [{"group": group, "runs": {
        "full_50": {"output": "I don't know.", "has_answer": False, "abstains": True, "routed_row": 0},
        "shared_1": {"output": "I don't know. Paris?", "has_answer": True, "abstains": True, "routed_row": 0},
    }} for group in ("rewrite", "paraphrase", "neighborhood", "retain")]
    rows = comparison_rows(official, generations, HIDDEN)
    assert rows[1]["rewrite_answer_pct"] == 100
    assert rows[1]["rewrite_abstain_pct"] == 100
    assert rows[1]["rewrite_exact_idk_pct"] == 0
    assert rows[0]["rewrite_exact_idk_pct"] == 100
    generations[0]["runs"]["shared_1"]["routed_row"] = None
    with pytest.raises(ValueError, match="route mismatch"):
        comparison_rows(official, generations, HIDDEN)


def test_shared_trainer_runs_on_tiny_model_and_exports_one_learned_vector(tmp_path, monkeypatch):
    """Exercise actual loss/backprop/Adam/checkpoint/export without model downloads."""
    import transformers
    import train_mcf_compressed_bank as trainer
    from static_overlap_data import Example
    from test_layer_sweep import _tiny_llama_and_tokenizer

    model, tok = _tiny_llama_and_tokenizer()
    tok.model_input_names = ["input_ids", "attention_mask"]
    tok.eos_token = "d"  # existing vocabulary token, used as the toy end marker
    facts = [{"id": "f1", "subject": "france", "relation": "capital", "object": "paris"},
             {"id": "f2", "subject": "b", "relation": "capital", "object": "c"}]
    examples = []
    for fact in facts:
        for split, prefix in (("train", ""), ("development", "the ")):
            prompt = prefix + fact["subject"] + " is"
            ids = tok(prompt)["input_ids"]
            answer = tok(" " + fact["object"], add_special_tokens=False)["input_ids"]
            examples.append(Example(id=f"{fact['id']}_{split}", split=split, role="forget",
                                    fact_id=fact["id"], input_ids=ids + answer,
                                    labels=[-100] * len(ids) + answer, prompt=prompt,
                                    completion=" " + fact["object"], group="toy"))
    hidden_size = model.config.hidden_size
    source = dict(architecture=ARCHITECTURE, layer=0, facts=facts,
                  rows=torch.zeros(2, hidden_size), router_weight=torch.zeros(2, hidden_size),
                  router_bias=torch.ones(2), feature_mean=torch.zeros(hidden_size),
                  feature_components=None, threshold=0.0, ambiguity_margin=0.5,
                  subject_patterns=[[tuple(tok(f["subject"])["input_ids"])] for f in facts])
    reference, output = tmp_path / "reference", tmp_path / "shared"
    reference.mkdir()
    torch.save(source, reference / "fact_association_embeddings.pt")
    (reference / "association_manifest.json").write_text(json.dumps({
        "model_path": "unused", "mcf_path": "unused", "sampling": {"seed": 1}}))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tok)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: model)
    monkeypatch.setattr(trainer, "load_mcf_forget_data", lambda *a, **kw: ([], facts, examples))
    eos_labels = []
    real_objective = trainer.fact_objective

    def checked_objective(model, answers, unknowns, *args):
        eos_labels.extend(e.labels[-1] for e in unknowns)
        return real_objective(model, answers, unknowns, *args)

    monkeypatch.setattr(trainer, "fact_objective", checked_objective)
    assert trainer.main(["--router-dir", str(reference), "--output-dir", str(output),
                         "--value-mode", "shared", "--unknown-eos", "--epochs", "2",
                         "--eval-every", "1", "--batch-facts", "2", "--device", "cpu"]) == 0
    saved = torch.load(output / "fact_association_embeddings.pt", weights_only=False)
    vector = saved["compressed_values"]["compact_state"]["shared_vector"]
    assert vector.abs().sum() > 0
    assert torch.equal(saved["rows"], vector.expand(2, -1))
    assert saved["trainable_parameters"] == hidden_size
    assert eos_labels and set(eos_labels) == {tok.eos_token_id}
    assert torch.equal(saved["router_weight"], source["router_weight"])
    report = json.loads((output / "training_report.json").read_text())
    assert report["best_epoch"] in (1, 2)
    assert report["training_coverage"]["facts_trained"] == 2
