import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tofu_sure_rank0_forget_budget100_v5.py"
spec = importlib.util.spec_from_file_location("tofu_sure_rank0_forget_budget100_v5", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
sys.path.insert(0, str(SCRIPT.parent))
spec.loader.exec_module(module)


class TofuSureBudget100V5Test(unittest.TestCase):
    def test_round_robin_reaches_exact_unique_budget_with_shared_rows(self):
        rankings = [
            [1, 2, 3, 4, 5],
            [1, 6, 7, 8, 9],
            [2, 10, 11, 12, 13],
        ]
        sensitive = set()
        selected = module.round_robin_unique_budget(
            sensitive,
            rankings,
            positions=[0, 1, 2],
            budget=8,
        )
        self.assertEqual(len(sensitive), 8)
        self.assertEqual(len(set(sensitive)), 8)
        self.assertTrue(set(selected).issubset({0, 1, 2}))

    def test_round_robin_fails_if_budget_exceeds_available_unique_rows(self):
        rankings = [[1, 2], [1, 2]]
        with self.assertRaises(RuntimeError):
            module.round_robin_unique_budget(set(), rankings, [0, 1], budget=3)

    def test_budget_arg_is_removed_before_v3_parser(self):
        argv = ["prog", "--initial-unique-row-budget", "100", "--output-dir", "x"]
        budget = module.pop_budget_arg(argv)
        self.assertEqual(budget, 100)
        self.assertNotIn("--initial-unique-row-budget", argv)
        self.assertEqual(argv, ["prog", "--output-dir", "x"])


if __name__ == "__main__":
    unittest.main()
