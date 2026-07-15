import importlib.util
import random
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mcf_sampling.py"
SPEC = importlib.util.spec_from_file_location("mcf_sampling", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MCFSamplingTests(unittest.TestCase):
    def test_official_split_matches_zerounlearn_order(self):
        records = [{"id": value} for value in range(20)]
        forget, retain = MODULE.sample_official_mcf_records(
            records, forget_num=4, retain_num=6, seed=7, strict=True
        )

        rng = random.Random(7)
        expected_forget = rng.sample(records[10:], k=4)
        expected_retain = rng.sample(records[:10], k=6)
        self.assertEqual(forget, expected_forget)
        self.assertEqual(retain, expected_retain)

    def test_official_split_rejects_undersized_pool_when_strict(self):
        with self.assertRaisesRegex(ValueError, "does not contain enough records"):
            MODULE.sample_official_mcf_records(
                [{"id": value} for value in range(8)],
                forget_num=5,
                retain_num=4,
                seed=0,
                strict=True,
            )


if __name__ == "__main__":
    unittest.main()
