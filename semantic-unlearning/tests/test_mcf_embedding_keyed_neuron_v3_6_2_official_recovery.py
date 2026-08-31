from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_mcf_embedding_keyed_neuron_v3_6_2_official as official_v1  # noqa: E402
import evaluate_mcf_embedding_keyed_neuron_v3_6_2_official_recovery as recovery  # noqa: E402


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _failed_attempt(tmp_path: Path) -> Path:
    root = tmp_path / "failed_official"
    _write_json(
        root / "pre_open_firewall_receipt.json",
        {
            "protocol": recovery.ORIGINAL_PROTOCOL,
            "official_evaluation_opened": False,
            "all_training_and_lineage_checks_passed": True,
            "candidate_checkpoint_sha256": official_v1.EXPECTED_CANDIDATE_SHA256,
            "protocol_sha256": recovery.ORIGINAL_PROTOCOL_SHA256,
            "evaluator_source_sha256": recovery.ORIGINAL_EVALUATOR_SHA256,
            "optimizer_constructed": False,
            "gradient_updates_performed": 0,
        },
    )
    _write_json(
        root / "official_evaluation_opened.json",
        {
            "protocol": recovery.ORIGINAL_PROTOCOL,
            "official_evaluation_opened": True,
            "official_mcf_sha256": official_v1.EXPECTED_MCF_SHA256,
            "expected_official_mcf_sha256": official_v1.EXPECTED_MCF_SHA256,
            "candidate_checkpoint_sha256": official_v1.EXPECTED_CANDIDATE_SHA256,
            "resume_or_retry_allowed": False,
        },
    )
    _write_json(
        root / "terminal_status.json",
        {
            "protocol": recovery.ORIGINAL_PROTOCOL,
            "status": "failed_after_official_open",
            "official_evaluation_opened": True,
            "official_evaluation_completed": False,
            "error_type": "RuntimeError",
            "error": recovery.EXPECTED_FAILURE,
            "partial_results_preserved": True,
            "retry_or_resume_permitted": False,
        },
    )
    return root


def _record(index: int):
    return {
        # Deliberately unrelated to dataset position: this reproduces the V1
        # namespace error without exposing or depending on real MCF content.
        "case_id": 100000 + index,
        "requested_rewrite": {
            "prompt": "{} has relation",
            "subject": f"subject-{index}",
            "target_true": {"str": "sensitive"},
            "target_new": {"str": "reference"},
        },
        "paraphrase_prompts": [],
        "neighborhood_prompts": [],
    }


def test_recovery_protocol_freezes_only_the_index_identity_repair():
    protocol_path = (
        ROOT
        / "protocols"
        / "mcf_embedding_keyed_neuron_v3_6_2_official_recovery_v1.json"
    )
    original_protocol_path = (
        ROOT
        / "protocols"
        / "mcf_embedding_keyed_neuron_v3_6_2_official_evaluation_v1.json"
    )
    source_path = SCRIPTS / "evaluate_mcf_embedding_keyed_neuron_v3_6_2_official_recovery.py"
    recovery.validate_recovery_protocol(
        json.loads(protocol_path.read_text(encoding="utf-8")),
        source_path=source_path,
        original_protocol_path=original_protocol_path,
        original_evaluator_path=SCRIPTS
        / "evaluate_mcf_embedding_keyed_neuron_v3_6_2_official.py",
    )


def test_failed_attempt_binding_proves_no_arm_metrics_and_preserves_hashes(tmp_path: Path):
    root = _failed_attempt(tmp_path)
    binding = recovery.validate_failed_official_attempt(root)
    assert binding["official_metric_forwards_completed"] == 0
    assert binding["official_arm_artifacts"] == 0
    assert recovery.failed_attempt_unchanged(binding) is True

    _write_json(root / "arms" / "full_candidate.json", {"metric": 1})
    assert recovery.failed_attempt_unchanged(binding) is False
    with pytest.raises(RuntimeError, match="arm metrics"):
        recovery.validate_failed_official_attempt(root)


def test_manifest_positions_not_embedded_case_ids_define_the_official_split():
    data = [_record(index) for index in range(40)]
    seed = 7
    rng = random.Random(seed)
    forget_indices = rng.sample(list(range(20, 40)), k=4)
    retain_indices = rng.sample(list(range(0, 20)), k=6)
    forget, retain = recovery.select_manifest_bound_records(
        data,
        forget_indices=forget_indices,
        retain_indices=retain_indices,
        seed=seed,
    )
    assert [record["case_id"] for record in forget] != forget_indices
    assert [record["case_id"] for record in retain] != retain_indices
    assert [record["requested_rewrite"]["subject"] for record in forget] == [
        f"subject-{index}" for index in forget_indices
    ]
    assert [record["requested_rewrite"]["subject"] for record in retain] == [
        f"subject-{index}" for index in retain_indices
    ]


def test_manifest_selection_rejects_any_positional_split_change():
    data = [_record(index) for index in range(40)]
    seed = 7
    rng = random.Random(seed)
    forget_indices = rng.sample(list(range(20, 40)), k=4)
    retain_indices = rng.sample(list(range(0, 20)), k=6)
    forget_indices[0], forget_indices[1] = forget_indices[1], forget_indices[0]
    with pytest.raises(RuntimeError, match="do not replay"):
        recovery.select_manifest_bound_records(
            data,
            forget_indices=forget_indices,
            retain_indices=retain_indices,
            seed=seed,
        )


def test_recovery_cli_exposes_paths_only_and_no_training_or_metric_knobs():
    args = recovery.parse_args(
        [
            "--model-path",
            "model",
            "--training-run-dir",
            "train",
            "--stage1-state",
            "stage1.pt",
            "--mcf-path",
            "mcf.json",
            "--wikidata-dir",
            "wiki",
            "--original-protocol",
            "original.json",
            "--recovery-protocol",
            "recovery.json",
            "--failed-official-run-dir",
            "failed",
            "--output-dir",
            "out",
        ]
    )
    assert set(vars(args)) == {
        "model_path",
        "training_run_dir",
        "stage1_state",
        "mcf_path",
        "wikidata_dir",
        "original_protocol",
        "recovery_protocol",
        "failed_official_run_dir",
        "output_dir",
    }
    source = (
        SCRIPTS
        / "evaluate_mcf_embedding_keyed_neuron_v3_6_2_official_recovery.py"
    ).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "official_v1.evaluate_fixed_arm" in source


def test_recovery_launchers_preserve_failed_output_and_disable_requeue():
    manual = (
        SCRIPTS
        / "run_mcf_embedding_keyed_neuron_v3_6_2_official_recovery_manual.sh"
    ).read_text(encoding="utf-8")
    slurm = (
        ROOT
        / "slurm"
        / "run_mcf_embedding_keyed_neuron_v3_6_2_official_recovery_seed1_3b.slurm"
    ).read_text(encoding="utf-8")
    assert "mv " not in manual
    assert "rm " not in manual
    assert "resume and retry are prohibited" in manual
    assert "#SBATCH --no-requeue" in slurm
    assert official_v1.EXPECTED_CANDIDATE_SHA256 in slurm
    assert official_v1.EXPECTED_MCF_SHA256 in slurm
    assert 'sha256sum "${MCF_PATH}"' not in slurm
    assert "evaluate_mcf_embedding_keyed_neuron_v3_6_2_official_recovery.py" in slurm


def test_multiseed_registry_freezes_independent_seeds_one_through_ten():
    path = (
        ROOT
        / "protocols"
        / "mcf_embedding_keyed_neuron_v3_6_2_multiseed_registry_v1.json"
    )
    registry = json.loads(path.read_text(encoding="utf-8"))
    assert registry["registered_seeds"] == list(range(1, 11))
    assert registry["execution_order"] == "ascending_seed_order_starting_with_seed_1"
    policy = registry["seed_independence_policy"]
    assert policy["each_seed_requires_its_own_candidate_sha256"] is True
    assert policy["official_results_cannot_select_or_modify_later_seed_hyperparameters"] is True
    assert registry["seed_1_disclosure"]["original_official_metric_forwards_completed"] == 0
