import argparse
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import run_zerounlearn_official_mcf as MODULE  # noqa: E402


def protocol_args(**updates):
    values = {
        "seed": 0,
        "forget_num": 50,
        "retain_num": 1000,
        "sample_mode": "official",
        "dtype": "bfloat16",
        "metric_tolerance": 0.02,
        "ppl_tolerance": 0.01,
        "model_path": f"/models/snapshots/{MODULE.MODEL_REVISION}",
    }
    values.update(updates)
    return argparse.Namespace(**values)


def official_payload(
    eff,
    gen,
    spe,
    ppl,
    *,
    method="method",
    model_path=None,
):
    payload = {
        "method": method,
        "model_dir": model_path or "in-memory:method",
        "dataset": "MCF",
        "sample_mode": "official",
        "seed": 0,
        "unlearn_num": 50,
        "retain_num": 1000,
        "forget": {
            "Eff": eff,
            "Gen": gen,
            "Spe": spe,
            "Spe_success": 80.0,
        },
        "forget_PPL": ppl,
    }
    if model_path is not None:
        payload["model_path"] = model_path
    return payload


class ZeroUnlearnOfficialRunnerTests(unittest.TestCase):
    def test_protocol_is_hard_limited_to_seed_zero(self):
        MODULE.validate_protocol_args(protocol_args())
        with self.assertRaisesRegex(ValueError, "--seed must be 0"):
            MODULE.validate_protocol_args(protocol_args(seed=1))
        with self.assertRaisesRegex(ValueError, "--forget-num must be 50"):
            MODULE.validate_protocol_args(protocol_args(forget_num=51))

    def test_official_split_matches_one_rng_for_forget_then_retain(self):
        records = [{"case_id": index} for index in range(2200)]
        forget, retain = MODULE.sample_official_mcf_records(
            records,
            50,
            1000,
            0,
            strict=True,
        )
        rng = random.Random(0)
        expected_forget = rng.sample(records[1100:], 50)
        expected_retain = rng.sample(records[:1100], 1000)
        self.assertEqual(MODULE.case_ids(forget), MODULE.case_ids(expected_forget))
        self.assertEqual(MODULE.case_ids(retain), MODULE.case_ids(expected_retain))

    def test_requests_use_original_apply_unl_structure_without_mutation(self):
        record = {
            "case_id": 17,
            "requested_rewrite": {
                "prompt": "{} works as",
                "subject": "A",
                "target_new": {"str": "new"},
                "target_true": {"str": "true"},
            },
        }
        original = json.loads(json.dumps(record))
        requests = MODULE.records_to_zero_unlearn_requests([record])
        self.assertEqual(requests[0]["case_id"], 17)
        self.assertEqual(requests[0]["prompt"], "{} works as")
        self.assertEqual(requests[0]["target_new"]["str"], "new")
        self.assertEqual(record, original)

    def test_metric_extraction_keeps_spe_separate_from_spe_success(self):
        nested = official_payload(0, 4, 14.9, 13.0625)
        metrics = MODULE.extract_result_metrics(nested)
        self.assertEqual(metrics["Spe"], 14.9)
        self.assertEqual(metrics["Spe_success"], 80.0)

        flat = {
            "Eff": 0,
            "Gen": 0,
            "Spe": 27.67,
            "PPL": 11.0625,
            "summary": {"Spe_success": 72.0},
        }
        flat_metrics = MODULE.extract_result_metrics(flat)
        self.assertEqual(flat_metrics["Spe"], 27.67)
        self.assertEqual(flat_metrics["Spe_success"], 72.0)

    def test_reference_validation_rejects_wrong_metric(self):
        spec = MODULE.EXISTING_METHOD_SPECS[0]
        matching = {
            "Eff": 6.0,
            "Gen": 6.0,
            "Spe": 10.89,
            "PPL": 11.0625,
        }
        MODULE.validate_reference_metrics(spec, matching, 0.02, 0.01)
        with self.assertRaisesRegex(ValueError, "Spe"):
            MODULE.validate_reference_metrics(
                spec,
                {**matching, "Spe": 84.4},
                0.02,
                0.01,
            )

    def test_protocol_metadata_must_be_official_seed_zero(self):
        payload = official_payload(6, 6, 10.89, 11.0625)
        metadata = MODULE.validate_result_protocol("Base", payload)
        self.assertEqual(metadata["sample_mode"], "official")
        payload["seed"] = 2
        with self.assertRaisesRegex(ValueError, "seed must be 0"):
            MODULE.validate_result_protocol("Base", payload)

    def test_model_revision_evidence_is_read_from_result_json(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result_path = root / "official_eval.json"
            model_path = root / "snapshots" / MODULE.MODEL_REVISION
            payload = official_payload(
                6,
                6,
                10.89,
                11.0625,
                model_path=str(model_path),
            )
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            verified, evidence = MODULE.find_model_evidence(
                result_path,
                payload,
                model_path,
            )
            self.assertTrue(verified)
            self.assertIn(
                result_path.resolve(),
                [Path(path).resolve() for path in evidence],
            )

    def test_result_discovery_uses_metrics_and_method_path_hints(self):
        with tempfile.TemporaryDirectory() as temp:
            semantic_root = Path(temp)
            output = (
                semantic_root
                / "outputs"
                / "my_neighborhood_confidence_repair"
                / "official_eval.json"
            )
            output.parent.mkdir(parents=True)
            output.write_text(
                json.dumps(
                    official_payload(
                        0,
                        0,
                        20.01,
                        11.0625,
                        method="gagd_neighborhood_confidence_repair",
                    )
                ),
                encoding="utf-8",
            )
            spec = MODULE.EXISTING_METHOD_SPECS[-1]
            located = MODULE.locate_existing_result(
                spec,
                semantic_root,
                {},
                0.02,
                0.01,
            )
            self.assertEqual(located, output.resolve())

    def test_hash_detects_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.json"
            path.write_text("one", encoding="utf-8")
            before = MODULE.sha256_file(path)
            path.write_text("two", encoding="utf-8")
            after = MODULE.sha256_file(path)
            self.assertNotEqual(before, after)

    def test_reviewed_mcf_hparams_and_source_hashes_match(self):
        hashes = MODULE.hash_protocol_inputs(
            MODULE.DEFAULT_MCF,
            MODULE.DEFAULT_HPARAMS,
            MODULE.DEFAULT_ZERO_ROOT,
        )
        MODULE.validate_expected_protocol_hashes(
            hashes,
            MODULE.DEFAULT_MCF,
            MODULE.DEFAULT_HPARAMS,
            MODULE.DEFAULT_ZERO_ROOT,
        )

    def test_pairwise_differences_are_zero_unlearn_minus_reference(self):
        rows = [
            {
                "Method": "Original ZeroUnlearn",
                "Eff ↓": 1.0,
                "Gen ↓": 2.0,
                "Spe ↑": 30.0,
                "PPL ↓": 12.0,
            }
        ]
        rows.extend(
            {
                "Method": name,
                "Eff ↓": 0.0,
                "Gen ↓": 0.0,
                "Spe ↑": 20.0,
                "PPL ↓": 11.0,
            }
            for name in MODULE.PAIRWISE_TARGETS
        )
        differences = MODULE.pairwise_differences(rows)
        self.assertEqual(differences[0]["Eff difference"], 1.0)
        self.assertEqual(differences[0]["Gen difference"], 2.0)
        self.assertEqual(differences[0]["Spe difference"], 10.0)
        self.assertEqual(differences[0]["PPL difference"], 1.0)

    def test_actual_mcf_seed_zero_case_ids_are_stable(self):
        mcf_path = MODULE.DEFAULT_MCF
        if not mcf_path.is_file():
            self.skipTest("MCF cache is not present")
        _, forget, retain = MODULE.load_seed0_split(mcf_path)
        self.assertEqual(
            MODULE.case_ids(forget)[:5],
            [17572, 18174, 11642, 15390, 19747],
        )
        self.assertEqual(
            MODULE.case_ids(retain)[:5],
            [5579, 1088, 3281, 9748, 3812],
        )
        self.assertEqual(len(forget), 50)
        self.assertEqual(len(retain), 1000)


if __name__ == "__main__":
    unittest.main()
