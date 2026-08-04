import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from official_benchmarks.adapters import (  # noqa: E402
    DataRoleViolation,
    reject_evaluation_data_for_training,
    validate_pch_sequence,
)
from official_benchmarks.doctor import DoctorError, validate_official_model  # noqa: E402
from official_benchmarks.planner import command_for_track  # noqa: E402
from official_benchmarks.provenance import (  # noqa: E402
    ProvenanceError,
    manifest_template,
    validate_manifest,
)
from official_benchmarks.registry import get_track, load_registry  # noqa: E402
from official_benchmarks.runner import RunRefused, run_track  # noqa: E402


class OfficialBenchmarkFrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_registry()

    def test_registry_has_exact_unique_track_ids(self):
        ids = [track["id"] for track in self.registry["tracks"]]
        self.assertEqual(len(ids), 15)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            set(ids),
            {
                "mcf_zerounlearn_official",
                "zsre_zerounlearn_official",
                "tofu_forget05",
                "muse_news",
                "muse_books",
                "rwku",
                "wmdp_bio",
                "wmdp_cyber",
                "wmdp_chem_eval",
                "ugbench_tofu",
                "ugbench_harry_potter",
                "ugbench_zsre",
                "pch_continual",
                "hubble_yago",
                "hubble_gutenberg",
            },
        )

    def test_methods_are_not_datasets(self):
        methods = {item["id"]: item for item in self.registry["baseline_methods"]}
        for method_id in ("rmu", "permu", "rule", "fit", "shred"):
            self.assertEqual(methods[method_id]["classification"], "baseline_method")
            self.assertNotIn(method_id, {track["id"] for track in self.registry["tracks"]})

    def test_official_mode_rejects_generic_models_for_special_targets(self):
        generic = "/mnt/train/models/Llama-3.2-3B-Instruct"
        for benchmark_id in (
            "tofu_forget05",
            "muse_news",
            "ugbench_tofu",
            "pch_continual",
            "hubble_yago",
        ):
            track = get_track(self.registry, benchmark_id)
            model = {
                "id": "meta-llama/Llama-3.2-3B-Instruct",
                "path": generic,
                "role": track["required_models"][0],
            }
            with self.assertRaisesRegex(DoctorError, "generic"):
                validate_official_model(track, model, generic_model_path=generic)

    def test_rwku_probes_are_not_training_data(self):
        for role in ("forget_level1", "forget_level2", "forget_level3", "neighbor_level1", "mia_forget", "utility_general"):
            with self.assertRaises(DataRoleViolation):
                reject_evaluation_data_for_training("rwku", [role])
        reject_evaluation_data_for_training("rwku", ["forget_target"])

    def test_ugbench_generated_cases_are_evaluation_only(self):
        for benchmark_id in ("ugbench_tofu", "ugbench_harry_potter", "ugbench_zsre"):
            for role in ("paraphrase", "subject_replacement", "inverse_relation", "one_hop", "implicit", "generalization"):
                with self.assertRaises(DataRoleViolation):
                    reject_evaluation_data_for_training(benchmark_id, [role])

    def test_wmdp_chem_cannot_be_a_forget_corpus(self):
        track = get_track(self.registry, "wmdp_chem_eval")
        self.assertEqual(track["method_status"], "EVALUATION_ONLY")
        self.assertEqual(track["data_roles"]["method_visible"], [])
        with self.assertRaises(DataRoleViolation):
            reject_evaluation_data_for_training("wmdp_chem_eval", ["forget_corpus"])

    def test_muse_retain2_and_holdout_are_hidden(self):
        for benchmark_id in ("muse_news", "muse_books"):
            track = get_track(self.registry, benchmark_id)
            self.assertIn("retain2 evaluation passages", track["data_roles"]["evaluation_only"])
            self.assertIn("holdout/nonmember passages", track["data_roles"]["evaluation_only"])
            with self.assertRaises(DataRoleViolation):
                reject_evaluation_data_for_training(benchmark_id, ["retain2", "holdout"])

    def test_pch_sequence_order_is_preserved(self):
        validate_pch_sequence(
            [
                {"deletion_order": 0, "category": "Personal"},
                {"deletion_order": 1, "category": "Copyright"},
                {"deletion_order": 2, "category": "Harmful"},
            ]
        )
        with self.assertRaisesRegex(DataRoleViolation, "official order"):
            validate_pch_sequence(
                [{"deletion_order": 1}, {"deletion_order": 0}]
            )

    def test_inventory_and_plan_do_not_import_torch_or_initialize_cuda(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            code = f"""
import runpy, sys
sys.argv = [
    'official_benchmarks.py', 'plan', '--suite', 'all', '--method',
    'our_method', '--output-dir', {str(Path(temp_dir) / 'plan')!r}
]
try:
    runpy.run_path({str(PROJECT_ROOT / 'scripts' / 'official_benchmarks.py')!r}, run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0, exc.code
assert 'torch' not in sys.modules
"""
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_unsupported_contract_fails_with_clear_status(self):
        track = get_track(self.registry, "rwku")
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RunRefused, "NEEDS_METHOD_EXTENSION"):
                run_track(
                    track,
                    method="our_method",
                    output_dir=Path(temp_dir),
                    execute=False,
                )

    def test_native_metric_directions_are_retained(self):
        mcf = get_track(self.registry, "mcf_zerounlearn_official")
        self.assertEqual(
            {metric["name"]: metric["direction"] for metric in mcf["native_metrics"]},
            {"Eff": "lower", "Gen": "lower", "Spe": "higher", "PPL": "lower"},
        )
        for track in self.registry["tracks"]:
            self.assertTrue(track["native_metrics"])
            for metric in track["native_metrics"]:
                self.assertIn(metric["direction"], {"higher", "lower", "reference", "report"})

    def test_existing_three_benchmark_commands_remain_wrapper_compatible(self):
        cases = {
            "mcf_zerounlearn_official": "run_three_benchmark_experiments.sh mcf",
            "tofu_forget05": "run_three_benchmark_experiments.sh tofu",
            "zsre_zerounlearn_official": "run_three_benchmark_experiments.sh zsre",
        }
        for benchmark_id, expected in cases.items():
            command = command_for_track(
                get_track(self.registry, benchmark_id),
                Path("outputs/test") / benchmark_id,
            )
            self.assertIn(expected, command)
        self.assertIn("MCF_FORGET_NUM=50", command_for_track(get_track(self.registry, "mcf_zerounlearn_official"), Path("out")))
        self.assertIn("TOFU_SEED=42", command_for_track(get_track(self.registry, "tofu_forget05"), Path("out")))
        self.assertIn("ZSRE_SEEDS=", command_for_track(get_track(self.registry, "zsre_zerounlearn_official"), Path("out")))

    def test_official_manifest_rejects_unresolved_identities(self):
        track = get_track(self.registry, "tofu_forget05")
        manifest = manifest_template(
            track,
            "true",
            method="our_method",
            output_dir=Path("outputs/test"),
            model_entry={
                "id": "REPLACE_MODEL",
                "path": "REPLACE_PATH",
                "revision": "REPLACE_REV",
                "architecture": "LlamaForCausalLM",
                "role": "Full",
                "tokenizer": {"id": "REPLACE_TOKENIZER", "revision": "REPLACE_REV"},
            },
        )
        with self.assertRaisesRegex(ProvenanceError, "unresolved"):
            validate_manifest(manifest, official=True)


if __name__ == "__main__":
    unittest.main()
