import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import aggregate_mcf_multimethod_results as AGG  # noqa: E402


class AggregateMCFMultimethodTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_result(self, path: Path, seed: int, value: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "dataset": "MCF",
                    "seed": seed,
                    "forget": {"Eff": value, "Gen": value + 1, "Spe": value + 2},
                    "forget_PPL": value + 3,
                }
            ),
            encoding="utf-8",
        )

    def populate(self, seeds=(0, 1)):
        for seed in seeds:
            self.write_result(self.root / f"zero/seed{seed}/base.json", seed, seed + 1)
            self.write_result(self.root / f"zero/seed{seed}/zero.json", seed, seed + 2)
            for index, spec in enumerate(AGG.DEFAULT_METHODS[2:7], start=3):
                self.write_result(
                    self.root / f"gagd/seed{seed}/{spec.key}.json",
                    seed,
                    seed + index,
                )
            self.write_result(
                self.root / f"repair/seed{seed}/repair.json", seed, seed + 8
            )

    def patterns(self):
        return {
            "base_pattern": str(self.root / "zero/seed{seed}/base.json"),
            "zero_pattern": str(self.root / "zero/seed{seed}/zero.json"),
            "gagd_pattern": str(self.root / "gagd/seed{seed}/{method}.json"),
            "repair_pattern": str(self.root / "repair/seed{seed}/repair.json"),
        }

    def test_collects_all_eight_methods_for_each_seed(self):
        self.populate()
        rows = AGG.collect_rows([0, 1], **self.patterns())
        self.assertEqual(len(rows), 16)
        self.assertEqual({row["method"] for row in rows}, {
            spec.display_name for spec in AGG.DEFAULT_METHODS
        })

    def test_uses_population_standard_deviation(self):
        self.populate()
        rows = AGG.collect_rows([0, 1], **self.patterns())
        aggregate = AGG.aggregate_rows(rows)
        base = aggregate[0]
        self.assertEqual(base["Eff_mean"], 1.5)
        self.assertEqual(base["Eff_std"], 0.5)

    def test_rejects_seed_mismatch(self):
        self.populate(seeds=(0,))
        bad = self.root / "zero/seed0/base.json"
        self.write_result(bad, seed=9, value=1.0)
        with self.assertRaisesRegex(ValueError, "stored seed"):
            AGG.collect_rows([0], **self.patterns())

    def test_writes_csv_json_and_markdown(self):
        self.populate()
        rows = AGG.collect_rows([0, 1], **self.patterns())
        aggregate = AGG.aggregate_rows(rows)
        out = self.root / "out"
        AGG.write_outputs(out, rows, aggregate, [0, 1])
        self.assertTrue((out / "per_seed.csv").is_file())
        self.assertTrue((out / "aggregate.csv").is_file())
        self.assertTrue((out / "aggregate.json").is_file())
        self.assertIn("population standard deviation", (out / "aggregate.md").read_text())


if __name__ == "__main__":
    unittest.main()
