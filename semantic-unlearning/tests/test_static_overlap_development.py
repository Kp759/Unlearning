"""Protocol isolation, immutable final tests, and actual-model checks."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from test_static_overlap_edit import bundle, tokenizer, deterministic
from test_static_overlap_cached_head import artificial_cache
from static_overlap_cached_head import RetainMetricSolver, development_cache, cached_measure, summarize
from static_overlap_training import TrainConfig, sha256_file
from static_overlap_data import text_fingerprints, encode_bundle
from freeze_static_overlap_development import (PLAN, build_final_retention, check_parity,
                                              load_protocol, main as freeze_main)
from run_static_overlap_cached_head import main as train_main
from audit_static_overlap_cached_head import main as audit_main
from evaluate_static_overlap_final_retention import main as final_main, claim_final
from export_static_overlap_edit import main as recover


def evaluation_bundle(bundle):
    result = deepcopy(bundle)
    result["purpose"] = "evaluation"
    result["examples"] = [r for r in result["examples"] if r["split"] == "validation"]
    for r in result["examples"]:
        r["split"] = "test"
        r["id"] = "final:" + r["id"]
        key = "text" if r.get("role") == "language" else "prompt"
        r[key] = "Independent final test context: " + r[key]
    return result


def mcf_data(n=2200):
    return [{"case_id": i, "requested_rewrite": {"subject": f"Independent subject {i}",
        "relation_id": "P_new", "prompt": "{} lives in", "target_true": {"str": "City"}},
        "paraphrase_prompts": [f"The home of independent subject {i} is"],
        "neighborhood_prompts": []} for i in range(n)]


def test_development_constraints_change_directions_but_forget_probes_do_not():
    cache = artificial_cache()
    cache.hidden[4] = torch.tensor([1., 0., 1., 0.])
    old = RetainMetricSolver(cache).solve(0, 1e-4) * 24
    assert not summarize(cached_measure(cache, old), TrainConfig())["eligible"]
    dev = development_cache(cache)
    assert all(e.split in ("train", "development") for e in dev.examples)
    assert not any(e.id == "5" for e in dev.examples)  # old validation forget removed
    solver = RetainMetricSolver(dev, include_development=True)
    # This particular feature conflict is incompatible with exact preservation.
    # The new solve must respect it rather than claiming the old fit succeeded.
    delta = solver.solve(0, 1e-4) * 24
    report = summarize(cached_measure(dev, delta), TrainConfig())
    assert report["eligible"] and "validation_protection" not in report
    assert report["development_protection"]["max_retained_kl"] < 1e-10
    assert solver.diagnostics["development_protected_tokens"] == 1
    assert solver.diagnostics["training_protected_tokens"] == 2
    assert not torch.allclose(old, delta)
    cache.hidden[5] *= 100000
    torch.testing.assert_close(RetainMetricSolver(development_cache(cache), include_development=True).solve(0, 1e-4), delta / 24)
    with pytest.raises(ValueError, match="explicit"):
        RetainMetricSolver(dev)
    dev.examples[0] = replace(dev.examples[0], split="test")
    with pytest.raises(ValueError, match="only train/development"):
        RetainMetricSolver(dev, include_development=True)


def test_final_sample_is_deterministic_disjoint_and_not_a_new_prompt_wrapper(bundle):
    source = deepcopy(bundle)
    source["facts"].append({"id": "extra", "subject": "Independent subject 8", "relation": "P_new",
                            "object": "City", "role": "retain"})
    data = mcf_data(40)
    evaluation = evaluation_bundle(bundle)
    final, ids = build_final_retention(data, source, evaluation, count=5, unlearn_num=2, retain_num=3)
    again, again_ids = build_final_retention(data, source, evaluation, count=5, unlearn_num=2, retain_num=3)
    assert final == again and ids == again_ids
    assert len(final["examples"]) == 10 and 8 not in ids
    assert not set(text_fingerprints(final)) & set(text_fingerprints(evaluation))
    from mcf_sampling import sample_official_mcf_records
    f, r = sample_official_mcf_records(data, 2, 3, 1)
    assert not set(ids) & {x["case_id"] for x in f + r}
    assert all(e["split"] == "test" for e in final["examples"])


def write_audit_fixture(tmp_path, passed=True, error=1e-7):
    run = tmp_path / "source"
    run.mkdir()
    for name in ("training_bundle.json", "head_cache.pt", "training_report.json"):
        (run / name).write_text("fixture")
    audit = {k: sha256_file(run / f) for k, f in (("training_bundle_sha256", "training_bundle.json"),
        ("cache_sha256", "head_cache.pt"), ("source_report_sha256", "training_report.json"))}
    audit["training_only_verification"] = {"cache_model_parity_passed": passed,
        "max_abs_errors": {k: error for k in ("base_nll", "nll", "kl")},
        "actual_rows": [{"base_nll": 2., "nll": 20., "kl": .1}], "actual_summary": {}}
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    return run, path


@pytest.mark.parametrize("passed,error", [(False, 1e-7), (True, .5)])
def test_failed_parity_cannot_freeze_any_protocol(tmp_path, passed, error):
    run, path = write_audit_fixture(tmp_path, passed, error)
    out = tmp_path / "protocol"
    with pytest.raises(ValueError, match="parity"):
        freeze_main(["--source-run", str(run), "--parity-audit", str(path),
                     "--evaluation-bundle", "not-read", "--mcf-path", "not-read",
                     "--protocol-dir", str(out)])
    assert not out.exists() and not (run / "development_protocol_frozen.json").exists()


def test_parity_gate_rejects_stale_source(tmp_path):
    run, path = write_audit_fixture(tmp_path)
    check_parity(run, path)
    (run / "head_cache.pt").write_text("changed")
    with pytest.raises(ValueError, match="stale"):
        check_parity(run, path)


def test_full_development_freeze_train_export_recovery_and_final_eval(bundle, tokenizer, tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM
    base = tmp_path / "base"
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tokenizer), hidden_size=64,
        intermediate_size=80, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        tie_word_embeddings=True, pad_token_id=0, bos_token_id=2, eos_token_id=3,
        max_position_embeddings=128)).eval()
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    source = tmp_path / "bundle.json"
    source.write_text(json.dumps(bundle))
    old = tmp_path / "old"
    assert train_main(["--model-path", str(base), "--training-bundle", str(source),
        "--output-dir", str(old), "--device", "cpu", "--local-files-only", "--training-only",
        "--allow-untied-head", "--no-context-augmentation", "--taus", ".01", "--ridges", ".01",
        "--strengths", ".01", ".1", "1"]) == 0
    audit = old / "audit.json"
    audit_main(["--training-run", str(old), "--model-path", str(base), "--out", str(audit),
                "--local-files-only", "--iterations", "2", "--verify-best-training"])
    eval_path, mcf = tmp_path / "eval.json", tmp_path / "mcf.json"
    eval_path.write_text(json.dumps(evaluation_bundle(bundle)))
    mcf.write_text(json.dumps(mcf_data()))
    protocol_dir = tmp_path / "protocol"
    freeze_args = ["--source-run", str(old), "--parity-audit", str(audit), "--evaluation-bundle", str(eval_path),
        "--mcf-path", str(mcf), "--protocol-dir", str(protocol_dir), "--final-retain-count", "2"]
    assert freeze_main(freeze_args) == 0
    frozen = protocol_dir / "protocol.json"
    final_hash = load_protocol(frozen)["files"]["final_retention"]["sha256"]
    with pytest.raises(FileExistsError):
        freeze_main(freeze_args)
    freeze_args[freeze_args.index("--protocol-dir")+1] = str(tmp_path / "replacement")
    with pytest.raises(FileExistsError, match="already has"):
        freeze_main(freeze_args)
    out = tmp_path / "development"
    args = ["--model-path", str(base), "--training-bundle", str(old / "training_bundle.json"),
            "--development-protocol", str(frozen), "--output-dir", str(out), "--device", "cpu",
            "--local-files-only", "--no-context-augmentation", "--allow-untied-head",
            "--taus", "0", ".001", ".01"]
    assert train_main(args) == 0
    report = json.loads((out / "training_report.json").read_text())
    assert report["cache_model_parity"]["passed"] and report["selected_actual"]["eligible"]
    assert "development" in report and "validation" not in report
    assert report["checkpoint_selection"]["final_test_used_for_selection"] is False
    target_met = [r for r in report["history"] if r["eligible"] and r["training_forgetting"]["target_met"]]
    if target_met:
        assert report["selected"]["delta_norm"] == min(r["delta_norm"] for r in target_met)
    active = json.loads((out / "development_examples.json").read_text())
    assert not any(e["split"] == "validation" for e in active)
    assert all(e["role"] in ("retain", "language") for e in active if e["split"] == "development")
    assert not (out / "training_bundle.json").exists()  # source archive has a distinct name
    recover(["--training-run", str(out), "--device", "cpu", "--local-files-only"])
    assert final_main(["--protocol", str(frozen), "--checkpoint", str(out / "checkpoint"),
                       "--model-path", str(base), "--device", "cpu", "--local-files-only"]) == 0
    result = json.loads((protocol_dir / "final_retention_results.json").read_text())
    assert result["numeric_budget_slack"] == 0 and not result["used_for_selection"]
    assert set(result["sets"]) == {"final_retention", "evaluation_bundle"}
    assert all(x["applied_nll_budget"] == .05 and x["applied_kl_budget"] == .01 for x in result["sets"].values())
    assert sha256_file(protocol_dir / "final_retention.json") == final_hash
    with pytest.raises(ValueError, match="different checkpoint"):
        claim_final(frozen, out / "checkpoint_float32")
    retry_args = list(args)
    retry_args[retry_args.index("--output-dir")+1] = str(tmp_path / "retune")
    with pytest.raises(FileExistsError):
        train_main(retry_args)
    # Fail closed if anyone edits the frozen test, even without running a model.
    (protocol_dir / "final_retention.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        load_protocol(frozen)
