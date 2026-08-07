import argparse
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import resume_rwku_zerounlearn_evaluation as RECOVERY  # noqa: E402
import rwku_artifact_access as ACCESS  # noqa: E402
import rwku_checkpoint_receipt as RECEIPT  # noqa: E402
import rwku_zerounlearn_target_only as ADAPTER  # noqa: E402


class FakeTokenizer:
    eos_token_id = 2

    def __len__(self):
        return 128


def write_artifact(path, role, payload, *, metadata=None):
    artifact = ACCESS.make_artifact(
        role,
        payload,
        protocol_label=ACCESS.TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=ADAPTER.PROTOCOL_STATUS,
        metadata=metadata,
    )
    ACCESS.write_artifact(path, artifact)
    return artifact


def make_opened_run(root):
    experiment_id = "experiment"
    run_dir = root / experiment_id
    run_dir.mkdir()
    model = root / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"base model")
    checkpoint = run_dir / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"frozen checkpoint")
    hparams = root / "hparams.json"
    hparams.write_text('{"layers": [16, 17, 18]}\n', encoding="utf-8")
    implementation = root / "frozen_implementation.py"
    implementation.write_text("FROZEN = True\n", encoding="utf-8")

    metadata = {
        "seed": 0,
        "entity_id": "rwku:1_Stephen_King",
        "subject": "Stephen King",
    }
    bundle_path = root / "generated_training_bundle.json"
    write_artifact(
        bundle_path,
        "training_bundle",
        {"views": [{"training_allowed": True}]},
        metadata=metadata,
    )
    generator_path = root / "generator_receipt.json"
    write_artifact(
        generator_path,
        "generator_receipt",
        {
            "status": "complete",
            "official_rwku_records_accessed": False,
            "target_entity": "Stephen King",
            "entity_id": "rwku:1_Stephen_King",
        },
        metadata=metadata,
    )
    locked_path = run_dir / "official_locked_eval.json"
    write_artifact(locked_path, "official_locked_eval", {"files": {}})

    receipt_path = run_dir / "checkpoint_receipt.json"
    frozen = RECEIPT.create_checkpoint_receipt(
        destination=receipt_path,
        experiment_id=experiment_id,
        protocol_label=ACCESS.TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=ADAPTER.PROTOCOL_STATUS,
        target_entity="Stephen King",
        target_entity_id="rwku:1_Stephen_King",
        base_model_identity={
            "path": str(model.resolve()),
            "revision": "revision",
            "sha256": ACCESS.sha256_path(model),
        },
        base_model_revision="revision",
        tokenizer_identity={"source_file_sha256": {}},
        checkpoint_paths=[checkpoint],
        training_bundle_path=bundle_path,
        optimization_protection_path=None,
        mcf_retain_optimization_paths=[],
        mcf_repair_gate_paths=[],
        matched_protection_train_path=None,
        matched_protection_gate_path=None,
        method_configuration={
            "method": ADAPTER.METHOD,
            "checkpoint_dtype": "bf16",
            "add_retain": False,
        },
        implementation_files=[implementation],
        sampler_provenance={"one_request_per_fact_id": True},
        generator_receipt_path=generator_path,
        official_locked_eval_path=locked_path,
        confirmatory=True,
        additional_artifact_paths={
            "base_model_source": model,
            "zero_hparams": hparams,
        },
    )
    opened = RECEIPT.open_official_evaluation(
        receipt_path,
        experiment_id=experiment_id,
    )
    state = {
        "schema_version": ADAPTER.STATE_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "state": "OFFICIAL_EVALUATION_OPENED",
        "seed": 0,
        "target": {
            "directory": "1_Stephen_King",
            "subject": "Stephen King",
            "entity_id": "rwku:1_Stephen_King",
        },
        "method": ADAPTER.METHOD,
        "protocol_label": ACCESS.TARGET_ONLY_PROTOCOL_LABEL,
        "protocol_status": ADAPTER.PROTOCOL_STATUS,
        "confirmatory": True,
        "add_external_retain_anchors": False,
        "external_retain_artifact_paths": [],
        "prepared_training_bundle_path": str(bundle_path.resolve()),
        "prepared_training_bundle_file_sha256": ACCESS.sha256_file(bundle_path),
        "prepared_generator_receipt_path": str(generator_path.resolve()),
        "prepared_generator_receipt_file_sha256": ACCESS.sha256_file(generator_path),
        "checkpoint_receipt": str(receipt_path.resolve()),
        "checkpoint_receipt_sha256": frozen["receipt_sha256"],
        "checkpoint_path": str(checkpoint.resolve()),
        "official_evaluation_opened": True,
        "official_evaluation_opened_at_utc": opened[
            "official_evaluation_opened_at_utc"
        ],
    }
    ADAPTER._atomic_json_write(run_dir / "experiment_state.json", state)
    args = argparse.Namespace(
        run_dir=run_dir,
        data_root=root / "rwku-data",
        wikidata_dir=root / "wikidata",
        eval_batch_size=4,
        no_download=True,
    )
    return {
        "args": args,
        "run_dir": run_dir,
        "state": state,
        "receipt_path": receipt_path,
        "checkpoint": checkpoint,
        "model": model,
        "target": ADAPTER.target_for_seed(0),
    }


def evaluation_mocks(fixture):
    datasets = {
        "forget_level1.json": [
            {"query": "Stephen King was born in ___?", "answer": "Maine", "level": "1"}
        ],
        "forget_level2.json": [
            {"query": "Where was Stephen King born?", "answer": "Maine", "level": "2"}
        ],
        "forget_level3.json": [
            {"query": "Was he born in Maine?", "answer": "yes", "level": "3"}
        ],
    }
    base_result = {
        "retain_reference_mean_logprobs": {"retain": -1.0},
        "metrics": {
            "adversarial": {"answer_probability": float("nan")},
            "finite": np.float64(1.25),
        },
    }
    candidate_result = {
        "retain_reference_mean_logprobs": {"retain": -1.1},
        "metrics": {"adversarial": {"answer_probability": float("nan")}},
    }
    return (
        mock.patch.object(
            ADAPTER,
            "ensure_target_data",
            return_value=(fixture["target"], datasets, {}),
        ),
        mock.patch.object(
            ADAPTER,
            "load_tokenizer",
            return_value=FakeTokenizer(),
        ),
        mock.patch.object(
            ADAPTER,
            "_load_evaluation_model",
            side_effect=[object(), object()],
        ),
        mock.patch.object(
            ADAPTER,
            "build_frozen_head_probe",
            return_value=object(),
        ),
        mock.patch.object(
            ADAPTER,
            "evaluate_rwku",
            side_effect=[base_result, candidate_result],
        ),
    )


class JsonNormalizationTests(unittest.TestCase):
    def test_nested_non_finite_values_and_pointer_paths(self):
        value = {
            "base": {
                "metrics": {
                    "nan": float("nan"),
                    "positive": np.float64(float("inf")),
                    "negative": torch.tensor(float("-inf")),
                    "finite": np.float32(1.5),
                }
            },
            "a/b~c": [float("nan")],
        }
        normalized, replacements = RECOVERY.normalize_non_finite_for_json(value)
        self.assertIsNone(normalized["base"]["metrics"]["nan"])
        self.assertIsNone(normalized["base"]["metrics"]["positive"])
        self.assertIsNone(normalized["base"]["metrics"]["negative"])
        self.assertAlmostEqual(normalized["base"]["metrics"]["finite"], 1.5)
        self.assertEqual(
            replacements,
            [
                {"path": "/base/metrics/nan", "original": "nan"},
                {
                    "path": "/base/metrics/positive",
                    "original": "positive_infinity",
                },
                {
                    "path": "/base/metrics/negative",
                    "original": "negative_infinity",
                },
                {"path": "/a~1b~0c/0", "original": "nan"},
            ],
        )

    def test_finite_and_primitive_values_are_unchanged(self):
        value = {
            "float": 1.25,
            "numpy": np.float64(2.5),
            "torch": torch.tensor(3),
            "bool": True,
            "integer": 4,
            "string": "value",
            "null": None,
            "tuple": (1.0, 2.0),
        }
        normalized, replacements = RECOVERY.normalize_non_finite_for_json(value)
        self.assertEqual(replacements, [])
        self.assertEqual(
            normalized,
            {
                "float": 1.25,
                "numpy": 2.5,
                "torch": 3,
                "bool": True,
                "integer": 4,
                "string": "value",
                "null": None,
                "tuple": [1.0, 2.0],
            },
        )

    def test_strict_json_serialization_succeeds(self):
        normalized, _ = RECOVERY.normalize_non_finite_for_json(
            {"nan": float("nan"), "infinity": float("inf")}
        )
        encoded = json.dumps(normalized, allow_nan=False)
        self.assertEqual(json.loads(encoded), {"nan": None, "infinity": None})
        RECOVERY.assert_no_non_finite_numeric(normalized)

    def test_unevaluated_adversarial_scores_become_null_not_zero(self):
        normalized, replacements = RECOVERY.normalize_non_finite_for_json(
            {
                "base": {
                    "metrics": {
                        "adversarial": {"answer_geometric_probability": float("nan")}
                    }
                }
            }
        )
        value = normalized["base"]["metrics"]["adversarial"][
            "answer_geometric_probability"
        ]
        self.assertIsNone(value)
        self.assertIsNot(value, 0)
        self.assertEqual(
            replacements,
            [
                {
                    "path": "/base/metrics/adversarial/answer_geometric_probability",
                    "original": "nan",
                }
            ],
        )


class RecoveryPreconditionTests(unittest.TestCase):
    def test_rejects_checkpoint_frozen_state(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_opened_run(Path(directory))
            state = json.loads(
                (fixture["run_dir"] / "experiment_state.json").read_text()
            )
            state["state"] = "CHECKPOINT_FROZEN"
            ADAPTER._atomic_json_write(
                fixture["run_dir"] / "experiment_state.json", state
            )
            with self.assertRaisesRegex(ValueError, "OFFICIAL_EVALUATION_OPENED"):
                RECOVERY.validate_recovery_preconditions(fixture["args"])

    def test_rejects_evaluation_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_opened_run(Path(directory))
            completed = RECEIPT.mark_evaluation_complete(
                fixture["receipt_path"], experiment_id="experiment"
            )
            state = json.loads(
                (fixture["run_dir"] / "experiment_state.json").read_text()
            )
            state.update(
                {
                    "state": "EVALUATION_COMPLETE",
                    "evaluation_completed_at_utc": completed[
                        "evaluation_completed_at_utc"
                    ],
                }
            )
            ADAPTER._atomic_json_write(
                fixture["run_dir"] / "experiment_state.json", state
            )
            with self.assertRaisesRegex(ValueError, "OFFICIAL_EVALUATION_OPENED"):
                RECOVERY.validate_recovery_preconditions(fixture["args"])

    def test_rejects_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_opened_run(Path(directory))
            (fixture["checkpoint"] / "weights.bin").unlink()
            fixture["checkpoint"].rmdir()
            with self.assertRaises((ValueError, FileNotFoundError)):
                RECOVERY.validate_recovery_preconditions(fixture["args"])

    def test_rejects_preexisting_official_result(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_opened_run(Path(directory))
            (fixture["run_dir"] / "official_evaluation.json").write_text(
                "{}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                RECOVERY.validate_recovery_preconditions(fixture["args"])

    def test_rejects_changed_frozen_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_opened_run(Path(directory))
            (fixture["model"] / "weights.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                RECOVERY.validate_recovery_preconditions(fixture["args"])


class RecoveryExecutionTests(unittest.TestCase):
    def run_with_mocks(self, fixture, *extra_patches):
        patches = [*evaluation_mocks(fixture), *extra_patches]
        started = []
        try:
            for patcher in patches:
                started.append(patcher.start())
            return RECOVERY.run_recovery(fixture["args"])
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    def test_recovery_never_reopens_boundary_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_opened_run(Path(directory))
            before = ACCESS.sha256_path(fixture["checkpoint"])
            opener = mock.patch.object(
                ADAPTER,
                "open_official_evaluation",
                side_effect=AssertionError("official evaluation was reopened"),
            )
            result = self.run_with_mocks(fixture, opener)
            self.assertEqual(ACCESS.sha256_path(fixture["checkpoint"]), before)
            self.assertFalse(
                result["evaluation_recovery"]["official_evaluation_reopened"]
            )
            self.assertTrue(
                result["evaluation_recovery"]["checkpoint_reused_without_modification"]
            )

    def test_state_moves_forward_only_after_validated_result(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_opened_run(Path(directory))
            real_validate = RECOVERY.validate_written_result
            events = []

            def tracked_validate(path):
                current = json.loads(
                    (fixture["run_dir"] / "experiment_state.json").read_text()
                )
                self.assertEqual(current["state"], "OFFICIAL_EVALUATION_OPENED")
                events.append("validated")
                return real_validate(path)

            result = self.run_with_mocks(
                fixture,
                mock.patch.object(
                    RECOVERY,
                    "validate_written_result",
                    side_effect=tracked_validate,
                ),
            )
            self.assertEqual(events, ["validated"])
            state = json.loads(
                (fixture["run_dir"] / "experiment_state.json").read_text()
            )
            receipt = RECEIPT.load_receipt(fixture["receipt_path"])
            self.assertEqual(state["state"], "EVALUATION_COMPLETE")
            self.assertEqual(receipt["state"], "EVALUATION_COMPLETE")
            self.assertEqual(
                state["official_evaluation_opened_at_utc"],
                fixture["state"]["official_evaluation_opened_at_utc"],
            )
            self.assertGreaterEqual(result["serialization"]["replacement_count"], 2)

    def test_failed_result_write_leaves_opened_state_and_writes_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_opened_run(Path(directory))
            real_write = RECOVERY.atomic_write_strict_json

            def fail_result_only(path, value):
                if Path(path).name == "official_evaluation.json":
                    raise OSError("simulated result write failure")
                return real_write(path, value)

            with self.assertRaisesRegex(OSError, "simulated result write failure"):
                self.run_with_mocks(
                    fixture,
                    mock.patch.object(
                        RECOVERY,
                        "atomic_write_strict_json",
                        side_effect=fail_result_only,
                    ),
                )
            state = json.loads(
                (fixture["run_dir"] / "experiment_state.json").read_text()
            )
            receipt = RECEIPT.load_receipt(fixture["receipt_path"])
            self.assertEqual(state["state"], "OFFICIAL_EVALUATION_OPENED")
            self.assertEqual(receipt["state"], "OFFICIAL_EVALUATION_OPENED")
            self.assertFalse((fixture["run_dir"] / "official_evaluation.json").exists())
            diagnostic = json.loads(
                (fixture["run_dir"] / "evaluation_recovery_failure.json").read_text()
            )
            self.assertEqual(
                diagnostic["state_after_failure"], "OFFICIAL_EVALUATION_OPENED"
            )
            self.assertFalse(diagnostic["official_evaluation_reopened"])


if __name__ == "__main__":
    unittest.main()
