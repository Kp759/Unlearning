import argparse
import inspect
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import run_zerounlearn_multiseed_mcf as MULTI  # noqa: E402
import run_zerounlearn_official_mcf as OFFICIAL  # noqa: E402


class ZeroUnlearnMultiseedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model_path = self.root / "snapshots" / OFFICIAL.MODEL_REVISION
        self.hparams_path = self.root / "hparams.json"
        self.forget_ids = list(range(50))
        self.retain_ids = list(range(1000, 2000))
        self.source_hashes = {"reviewed-source": "sha256"}

    def multiseed_args(self, **updates):
        values = {
            "seeds": [0, 1, 9],
            "forget_num": 50,
            "retain_num": 1000,
            "sample_mode": "official",
            "dtype": "bfloat16",
            "model_path": str(self.model_path),
        }
        values.update(updates)
        return argparse.Namespace(**values)

    def completion_payloads(self, seed, paths):
        runtime = {"apply_seconds": 1.25}
        result = {
            "method": OFFICIAL.METHOD,
            "model_dir": f"in-memory:{OFFICIAL.METHOD}",
            "model_path": str(self.model_path),
            "model_revision": OFFICIAL.MODEL_REVISION,
            "dtype": OFFICIAL.DTYPE_NAME,
            "dataset": "MCF",
            "sample_mode": "official",
            "seed": seed,
            "unlearn_num": 50,
            "retain_num": 1000,
            "forget": {
                "Eff": 1.0,
                "Gen": 2.0,
                "Spe": 3.0,
                "Spe_success": 4.0,
            },
            "forget_PPL": 5.0,
            "forget_case_ids": self.forget_ids,
            "retain_case_ids": self.retain_ids,
            "case_ids_source": f"official_sampler_seed{seed}",
            "zero_unlearn_runtime": runtime,
        }
        provenance = {
            "status": "completed",
            "method": OFFICIAL.METHOD,
            "algorithm_entrypoint": ("ZeroUnlearn.ZeroUnlearn_main.apply_unl_to_model"),
            "model_path": str(self.model_path),
            "model_revision": OFFICIAL.MODEL_REVISION,
            "dtype": OFFICIAL.DTYPE_NAME,
            "zero_unlearn_compute_dtype": "float32",
            "seed": seed,
            "dataset": "MCF",
            "sample_mode": "official",
            "forget_num": 50,
            "retain_num": 1000,
            "forget_case_ids": self.forget_ids,
            "retain_case_ids": self.retain_ids,
            "case_ids_source": f"official_sampler_seed{seed}",
            "hparams_path": str(self.hparams_path),
            "edit_layer_nums": OFFICIAL.EDIT_LAYER_NUMS,
            "add_retain": OFFICIAL.ADD_RETAIN,
            "use_h": OFFICIAL.USE_H,
            "checkpoint_saved": False,
            "multi_gpu_device_map_used": False,
            "source_hashes_before": self.source_hashes,
            "source_hashes_after": self.source_hashes,
            "source_hashes_unchanged": True,
            "runtime": runtime,
            "official_evaluation_path": str(paths.zero_unlearn_result),
            "neutral_target": {
                "source": "tokenizer.eos_token",
                "token": "<eos>",
                "token_id": 2,
                "zero_unlearn_request_field": "target_new.str",
                "zero_unlearn_sensitive_request_field": "target_true",
                "zero_unlearn_sensitive_target_source": (
                    "MCF requested_rewrite.target_new"
                ),
                "benchmark_forget_target": ("MCF requested_rewrite.target_new"),
                "benchmark_correct_target": ("MCF requested_rewrite.target_true"),
                "forget_request_count": 50,
                "retain_requests_modified": False,
                "official_evaluation_records_modified": False,
                "source_mcf_modified": False,
            },
        }
        return result, provenance

    def completion_expected(self, seed):
        return {
            "seed": seed,
            "forget_num": 50,
            "retain_num": 1000,
            "sample_mode": "official",
            "model_path": self.model_path,
            "hparams_path": self.hparams_path,
            "dtype": "bfloat16",
            "forget_case_ids": self.forget_ids,
            "retain_case_ids": self.retain_ids,
            "expected_source_hashes": self.source_hashes,
        }

    def test_official_sampling_is_seed_specific(self):
        mcf_path = self.root / "mcf.json"
        records = [{"case_id": index} for index in range(2200)]
        mcf_path.write_text(json.dumps(records), encoding="utf-8")

        _, forget, retain = OFFICIAL.load_official_split(
            mcf_path,
            seed=9,
            forget_num=50,
            retain_num=1000,
            sample_mode="official",
        )
        rng = random.Random(9)
        expected_forget = rng.sample(records[1100:], 50)
        expected_retain = rng.sample(records[:1100], 1000)
        self.assertEqual(OFFICIAL.case_ids(forget), OFFICIAL.case_ids(expected_forget))
        self.assertEqual(OFFICIAL.case_ids(retain), OFFICIAL.case_ids(expected_retain))

        _, seed0_forget, _ = OFFICIAL.load_official_split(
            mcf_path,
            seed=0,
            forget_num=50,
            retain_num=1000,
            sample_mode="official",
        )
        self.assertNotEqual(
            OFFICIAL.case_ids(forget),
            OFFICIAL.case_ids(seed0_forget),
        )

    def test_seed_output_paths_cover_zero_and_nine(self):
        zero = MULTI.seed_output_paths(self.root, 0)
        nine = MULTI.seed_output_paths(self.root, 9)
        self.assertEqual(
            zero.base_result,
            self.root / "seed0" / "base_seed0_official_eval.json",
        )
        self.assertEqual(
            zero.zero_unlearn_result,
            self.root / "seed0" / "zerounlearn_seed0_official_eval.json",
        )
        self.assertEqual(
            nine.provenance,
            self.root / "seed9" / "zerounlearn_seed9_provenance.json",
        )

    def test_seed_one_is_accepted_only_by_zero_only_multiseed_mode(self):
        MULTI.validate_multiseed_args(self.multiseed_args(seeds=[1]))

        original = argparse.Namespace(
            seed=1,
            forget_num=50,
            retain_num=1000,
            sample_mode="official",
            dtype="bfloat16",
            metric_tolerance=0.02,
            ppl_tolerance=0.01,
            model_path=str(self.model_path),
        )
        with self.assertRaisesRegex(ValueError, "--seed must be 0"):
            OFFICIAL.validate_protocol_args(original)

    def test_reusable_run_boundary_has_the_required_six_arguments(self):
        parameters = list(
            inspect.signature(OFFICIAL.run_original_zerounlearn_mcf).parameters
        )
        self.assertEqual(
            parameters[:6],
            [
                "seed",
                "forget_num",
                "retain_num",
                "sample_mode",
                "model_path",
                "output_dir",
            ],
        )

    def test_target_new_eos_mapping_is_shared_unchanged(self):
        record = {
            "case_id": 17,
            "requested_rewrite": {
                "prompt": "{} works as",
                "subject": "A",
                "target_new": {"str": "counterfactual", "id": "Q-new"},
                "target_true": {"str": "original", "id": "Q-true"},
            },
        }
        requests = OFFICIAL.records_to_zero_unlearn_forget_requests(
            [record],
            neutral_target="<eos>",
        )
        self.assertEqual(requests[0]["target_new"], {"str": "<eos>"})
        self.assertEqual(
            requests[0]["target_true"],
            record["requested_rewrite"]["target_new"],
        )

    def test_aggregation_uses_mean_and_population_standard_deviation(self):
        rows = [
            {
                "method": OFFICIAL.METHOD,
                "seed": seed,
                **{metric: value for metric in MULTI.METRICS},
            }
            for seed, value in [(0, 1.0), (1, 3.0)]
        ]
        aggregate = MULTI.aggregate_rows(rows)[0]
        for metric in MULTI.METRICS:
            self.assertEqual(aggregate[f"{metric}_mean"], 2.0)
            self.assertEqual(aggregate[f"{metric}_std"], 1.0)
        self.assertEqual(aggregate["n_seeds"], 2)

    def test_skip_completed_requires_valid_zero_result_and_provenance_only(self):
        seed = 1
        paths = MULTI.seed_output_paths(self.root, seed)
        paths.seed_dir.mkdir(parents=True)
        result, provenance = self.completion_payloads(seed, paths)
        paths.zero_unlearn_result.write_text(
            json.dumps(result),
            encoding="utf-8",
        )
        paths.provenance.write_text(
            json.dumps(provenance),
            encoding="utf-8",
        )

        self.assertFalse(paths.base_result.exists())
        self.assertTrue(MULTI.is_seed_complete(paths, **self.completion_expected(seed)))

        provenance["status"] = "running"
        paths.provenance.write_text(
            json.dumps(provenance),
            encoding="utf-8",
        )
        self.assertFalse(
            MULTI.is_seed_complete(paths, **self.completion_expected(seed))
        )

    def test_skip_completed_does_not_invoke_the_seed_runner(self):
        seed = 1
        paths = MULTI.seed_output_paths(self.root.resolve(), seed)
        paths.seed_dir.mkdir(parents=True)
        result, provenance = self.completion_payloads(seed, paths)
        paths.zero_unlearn_result.write_text(
            json.dumps(result),
            encoding="utf-8",
        )
        paths.provenance.write_text(
            json.dumps(provenance),
            encoding="utf-8",
        )
        args = argparse.Namespace(
            **vars(self.multiseed_args(seeds=[seed])),
            zero_unlearn_root=str(self.root / "zero"),
            hparams_path=str(self.hparams_path),
            mcf_path=str(self.root / "mcf.json"),
            wikidata_dir=str(self.root / "wikidata"),
            output_root=str(self.root),
            skip_completed=True,
        )
        forget_records = [{"case_id": value} for value in self.forget_ids]
        retain_records = [{"case_id": value} for value in self.retain_ids]
        seed_runner = mock.Mock()
        zero_row = {
            "method": OFFICIAL.METHOD,
            "seed": seed,
            **{metric: 1.0 for metric in MULTI.METRICS},
            "source": str(paths.zero_unlearn_result),
        }

        with (
            mock.patch.object(OFFICIAL, "require_runtime_files"),
            mock.patch.object(
                OFFICIAL,
                "hash_protocol_inputs",
                return_value=self.source_hashes,
            ),
            mock.patch.object(OFFICIAL, "validate_expected_protocol_hashes"),
            mock.patch.object(
                OFFICIAL,
                "load_official_split",
                return_value=([], forget_records, retain_records),
            ),
            mock.patch.object(
                MULTI,
                "collect_rows",
                return_value=([zero_row], []),
            ),
            mock.patch.object(MULTI, "write_summary_outputs"),
        ):
            output = MULTI.run(args, seed_runner=seed_runner)

        seed_runner.assert_not_called()
        self.assertEqual(
            output["seed_status"],
            [
                {
                    "seed": seed,
                    "status": "skipped_completed",
                    **paths.as_dict(),
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
