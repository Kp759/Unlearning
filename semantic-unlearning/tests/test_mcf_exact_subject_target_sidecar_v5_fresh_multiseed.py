from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_mcf_exact_subject_target_sidecar_v5_fresh_seed as candidate_build
import build_mcf_exact_subject_target_sidecar_v5_fresh_seed_split as split_build
import evaluate_mcf_exact_subject_target_sidecar_v5_fresh_seed as evaluation


ROOT = Path(__file__).resolve().parents[1]


def _record(index: int) -> dict:
    return {
        "requested_rewrite": {
            "prompt": "{} is associated with",
            "subject": f"Synthetic Subject {index}",
            "relation_id": "P999",
            "target_new": {"str": f"Reference {index}"},
            "target_true": {"str": f"Sensitive {index}"},
        },
        "paraphrase_prompts": [f"held-out paraphrase {index}"],
        "neighborhood_prompts": [f"held-out neighborhood {index}"],
    }


def test_direct_only_minimizer_never_serializes_probe_text():
    raw = [_record(index) for index in range(20)]
    records, forget_ids, retain_ids = split_build.build_split(
        raw, seed=2, forget_num=3, retain_num=5
    )
    assert len(records) == 3
    assert len(forget_ids) == 3
    assert len(retain_ids) == 5
    assert set(forget_ids).isdisjoint(retain_ids)
    encoded = json.dumps(records)
    assert "held-out paraphrase" not in encoded
    assert "held-out neighborhood" not in encoded
    assert all(
        set(record) == {"case_id", "requested_rewrite", "data_role"}
        for record in records
    )


def test_seed_one_and_out_of_registry_seeds_are_rejected():
    with pytest.raises(SystemExit):
        split_build.parse_args(
            [
                "--mcf-path",
                "x",
                "--relation-lexicon",
                "y",
                "--output-dir",
                "z",
                "--seed",
                "1",
            ]
        )
    with pytest.raises(SystemExit):
        evaluation.parse_args(
            [
                "--model-path",
                "m",
                "--candidate-run-dir",
                "c",
                "--mcf-path",
                "d",
                "--wikidata-dir",
                "w",
                "--confirmation-registry",
                "r",
                "--output-dir",
                "o",
                "--seed",
                "11",
            ]
        )


def test_fresh_registry_locks_exact_acceptance_and_no_retry():
    registry = json.loads(
        (
            ROOT
            / "protocols"
            / "mcf_exact_subject_target_sidecar_v5_fresh_multiseed_registry_v1.json"
        ).read_text()
    )
    evaluation.validate_confirmation_registry(registry, seed=2)
    candidate_build.validate_registries(
        json.loads(
            (
                ROOT
                / "protocols"
                / "mcf_exact_subject_target_sidecar_v5_0_registry.json"
            ).read_text()
        ),
        registry,
        seed=10,
    )
    assert registry["registered_seeds"] == list(range(2, 11))
    assert registry["acceptance"]["forget_eff"] == 0.0
    assert registry["acceptance"]["forget_gen"] == 0.0
    assert registry["acceptance"]["ppl_exact_base"] is True
    assert registry["one_shot_policy"]["retry_after_official_open_prohibited"] is True
    assert registry["strong_unlearning_claim_permitted"] is False


def test_candidate_binding_check_uses_dataset_identity_and_all_targets():
    record = _record(7)
    state = {
        "case_ids": [17],
        "subjects": ["Synthetic Subject 7"],
        "relation_ids": ["P999"],
        "target_new": ["Reference 7"],
        "target_true": ["Sensitive 7"],
    }
    evaluation.validate_candidate_bindings(state, [record], [17])
    state["target_true"] = ["different"]
    with pytest.raises(RuntimeError, match="bindings differ"):
        evaluation.validate_candidate_bindings(state, [record], [17])


def test_split_manifest_requires_zero_serialized_heldout_content():
    source_hash = "a" * 64
    lexicon = {"derivation": {"source_mcf_sha256": source_hash}}
    manifest = {
        "protocol": split_build.PROTOCOL,
        "seed": 2,
        "source_sha256": source_hash,
        "candidate_view": {
            "direct_forget_records": 50,
            "official_paraphrase_prompts_serialized": 0,
            "official_neighborhood_prompts_serialized": 0,
            "official_retain_prompts_serialized": 0,
            "official_ppl_documents_serialized": 0,
            "probe_fields_absent_not_masked": True,
        },
        "splitter_isolated_from_candidate_process": True,
        "candidate_process_official_evaluation_prompts_seen": 0,
    }
    candidate_build.validate_split_manifest(manifest, lexicon, seed=2)
    manifest["candidate_view"]["official_paraphrase_prompts_serialized"] = 1
    with pytest.raises(RuntimeError, match="direct-only firewall"):
        candidate_build.validate_split_manifest(manifest, lexicon, seed=2)
