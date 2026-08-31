from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_mcf_embedding_keyed_neuron_v3_6_2_official as official  # noqa: E402


def _summary(eff: float, gen: float, spe: float, margin: float = 0.2):
    return {
        "Eff": eff,
        "Gen": gen,
        "Spe": spe,
        "minimum_rewrite_paraphrase_margin": margin,
    }


def _arm(
    *,
    forget_eff: float,
    forget_gen: float,
    forget_spe: float,
    retain_eff: float,
    retain_gen: float,
    retain_spe: float,
    ppl: float,
    margin: float = 0.2,
):
    return {
        "forget": _summary(forget_eff, forget_gen, forget_spe, margin),
        "retain": _summary(retain_eff, retain_gen, retain_spe),
        "forget_PPL": ppl,
    }


def _passing_arms():
    return {
        "full_candidate": _arm(
            forget_eff=0.0,
            forget_gen=0.0,
            forget_spe=10.1,
            retain_eff=6.5,
            retain_gen=7.0,
            retain_spe=12.1,
            ppl=10.4,
        ),
        "reconstructed_base": _arm(
            forget_eff=80.0,
            forget_gen=75.0,
            forget_spe=10.0,
            retain_eff=6.0,
            retain_gen=6.5,
            retain_spe=12.0,
            ppl=10.0,
        ),
        "writer_only": _arm(
            forget_eff=70.0,
            forget_gen=70.0,
            forget_spe=10.0,
            retain_eff=6.0,
            retain_gen=6.5,
            retain_spe=12.0,
            ppl=10.0,
        ),
        "actuator_without_writer": _arm(
            forget_eff=80.0,
            forget_gen=75.0,
            forget_spe=10.0,
            retain_eff=6.0,
            retain_gen=6.5,
            retain_spe=12.0,
            ppl=10.0,
        ),
    }


def _candidate_state():
    detector_ids = list(range(200))
    actuator_ids = list(range(200, 1000))
    return {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_v3_6_2_candidate_state",
        "protocol": official.TRAINING_PROTOCOL,
        "case_ids": list(official.EXPECTED_CASE_IDS),
        "official_evaluation_prompts_seen": 0,
        "selected_embedding_rows": [1, 2],
        "embedding_delta": torch.zeros(2, 3),
        "detector_neuron_ids": detector_ids,
        "detector_local_groups": [
            list(range(index * 4, (index + 1) * 4)) for index in range(50)
        ],
        "detector_flat_signs": torch.ones(200),
        "detector_gate_rows": torch.zeros(200, 3),
        "detector_up_rows": torch.zeros(200, 3),
        "actuator_neuron_ids": actuator_ids,
        "actuator_owner_indices": [owner for owner in range(50) for _ in range(16)],
        "actuator_width": 16,
        "actuator_relative_cap": 1.5,
        "actuator_down_delta": torch.zeros(3, 800),
        "threshold_off_boundary": 0.200001,
        "threshold_on_boundary": 0.249999,
        "training_acceptance": {
            "passed": True,
            "official_evaluation_prompts_seen": 0,
        },
    }


def test_frozen_protocol_binds_exact_candidate_dataset_source_and_arms():
    protocol_path = (
        ROOT
        / "protocols"
        / "mcf_embedding_keyed_neuron_v3_6_2_official_evaluation_v1.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    official.validate_protocol(
        protocol,
        source_path=SCRIPTS / "evaluate_mcf_embedding_keyed_neuron_v3_6_2_official.py",
    )
    assert protocol["candidate_checkpoint_sha256"] == official.EXPECTED_CANDIDATE_SHA256
    assert protocol["official_mcf_sha256"] == official.EXPECTED_MCF_SHA256
    assert protocol["forget_case_ids"] == official.EXPECTED_CASE_IDS
    assert protocol["fixed_arm_order"] == official.FIXED_ARM_ORDER
    assert protocol["one_shot_policy"]["retry_prohibited"] is True


def test_candidate_shape_requires_disjoint_four_by_sixteen_architecture():
    state = _candidate_state()
    official.validate_candidate_structure(state)
    state["actuator_neuron_ids"][0] = state["detector_neuron_ids"][0]
    with pytest.raises(RuntimeError, match="overlap"):
        official.validate_candidate_structure(state)


def test_candidate_acceptance_uses_frozen_base_relative_thresholds():
    result = official.build_candidate_acceptance(
        _passing_arms(),
        integrity_passed=True,
    )
    assert result["passed"] is True
    assert result["observed"]["forget_spe_delta"] == pytest.approx(0.1)
    assert result["observed"]["ppl_percent_delta"] == pytest.approx(4.0)


def test_candidate_acceptance_reports_failure_without_selecting_or_retrying():
    arms = _passing_arms()
    arms["full_candidate"]["forget"]["Gen"] = 2.0
    arms["full_candidate"]["forget_PPL"] = 11.0
    result = official.build_candidate_acceptance(arms, integrity_passed=True)
    assert result["passed"] is False
    assert "forget_gen_within_limit" in result["failure_reasons"]
    assert "ppl_percent_delta_within_limit" in result["failure_reasons"]
    assert result["used_for_training_checkpoint_selection_or_retry"] is False


def test_parameter_version_audit_detects_in_place_weight_changes():
    model = torch.nn.Linear(3, 2)
    before = official.parameter_versions(model)
    assert official.parameter_versions_unchanged(before, before)["passed"] is True
    with torch.no_grad():
        model.weight.add_(1.0)
    audit = official.parameter_versions_unchanged(
        before,
        official.parameter_versions(model),
    )
    assert audit["passed"] is False
    assert audit["changed_parameters"] == ["weight"]


def test_output_directory_is_single_use(tmp_path: Path):
    output = tmp_path / "official"
    official.create_fresh_output_dir(output)
    with pytest.raises(RuntimeError, match="cannot be resumed or retried"):
        official.create_fresh_output_dir(output)


def test_cli_exposes_paths_only_and_no_training_or_metric_knobs():
    args = official.parse_args(
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
            "--protocol",
            "protocol.json",
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
        "protocol",
        "output_dir",
    }
    source = (
        SCRIPTS / "evaluate_mcf_embedding_keyed_neuron_v3_6_2_official.py"
    ).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "torch.inference_mode" in source


def test_launchers_pin_one_shot_checkpoint_and_disable_requeue():
    manual = (
        SCRIPTS / "run_mcf_embedding_keyed_neuron_v3_6_2_official_manual.sh"
    ).read_text(encoding="utf-8")
    slurm = (
        ROOT / "slurm" / "run_mcf_embedding_keyed_neuron_v3_6_2_official_seed1_3b.slurm"
    ).read_text(encoding="utf-8")
    assert "resume and retry are prohibited" in manual
    assert "#SBATCH --no-requeue" in slurm
    assert official.EXPECTED_CANDIDATE_SHA256 in slurm
    assert official.EXPECTED_MCF_SHA256 in slurm
    assert 'sha256sum "${MCF_PATH}"' not in slurm
    assert "v3_6_2_candidate_state.pt" in slurm
    assert "eligible_for_separate_official_evaluation == true" in slurm
    assert "evaluate_mcf_embedding_keyed_neuron_v3_6_2_official.py" in slurm


def test_pre_open_receipt_precedes_first_official_dataset_read():
    source = (
        SCRIPTS / "evaluate_mcf_embedding_keyed_neuron_v3_6_2_official.py"
    ).read_text(encoding="utf-8")
    assert source.index(
        'write_json(output_dir / "pre_open_firewall_receipt.json"'
    ) < source.index("mcf_sha = sha256_file(mcf_path)")
