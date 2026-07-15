import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_gagd_to_json_lmhead.py"
SPEC = importlib.util.spec_from_file_location("compare_gagd", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_result(self, name, payload):
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def metadata(self, seed=0):
        return {
            "dataset": "MCF",
            "sample_mode": "official",
            "seed": seed,
            "unlearn_num": 50,
            "retain_num": 1000,
        }

    def test_reads_compact_json_lmhead_result(self):
        path = self.write_result(
            "compact.json",
            {
                **self.metadata(),
                "Eff": 1.0,
                "Gen": 2.0,
                "Spe": 3.0,
                "PPL": 4.0,
            },
        )
        row = MODULE.result_row(
            "json_lmhead", "json_lmhead", 0, path, "official", 50, 1000
        )
        self.assertEqual([row[m] for m in MODULE.METRICS], [1.0, 2.0, 3.0, 4.0])

    def test_reads_full_in_memory_official_result(self):
        path = self.write_result(
            "full.json",
            {
                **self.metadata(),
                "forget": {"Eff": 5.0, "Gen": 6.0, "Spe": 7.0},
                "forget_PPL": 8.0,
            },
        )
        row = MODULE.result_row("gagd", "gagd", 0, path, "official", 50, 1000)
        self.assertEqual([row[m] for m in MODULE.METRICS], [5.0, 6.0, 7.0, 8.0])

    def test_rejects_mismatched_seed(self):
        path = self.write_result(
            "wrong_seed.json",
            {
                **self.metadata(seed=1),
                "Eff": 1.0,
                "Gen": 2.0,
                "Spe": 3.0,
            },
        )
        with self.assertRaisesRegex(ValueError, "seed=1, expected 0"):
            MODULE.result_row("bad", "gagd", 0, path, "official", 50, 1000)

    def test_rejects_result_without_ppl(self):
        path = self.write_result(
            "missing_ppl.json",
            {**self.metadata(), "Eff": 1.0, "Gen": 2.0, "Spe": 3.0},
        )
        with self.assertRaisesRegex(ValueError, "PPL"):
            MODULE.result_row("bad", "gagd", 0, path, "official", 50, 1000)

    def test_aggregate_uses_population_standard_deviation(self):
        rows = [
            {
                "method": "method",
                "family": "gagd",
                "seed": seed,
                "Eff": value,
                "Gen": value,
                "Spe": value,
                "PPL": value,
            }
            for seed, value in [(0, 1.0), (1, 3.0)]
        ]
        aggregate = MODULE.aggregate_rows(rows)[0]
        self.assertEqual(aggregate["Eff_mean"], 2.0)
        self.assertEqual(aggregate["Eff_std"], 1.0)
        self.assertEqual(aggregate["n_seeds"], 2)


if __name__ == "__main__":
    unittest.main()
