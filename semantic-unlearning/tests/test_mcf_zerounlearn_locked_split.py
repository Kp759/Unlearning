import importlib.util
import random
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SCRIPT = SCRIPTS / "build_mcf_zerounlearn_locked_split.py"
SPEC = importlib.util.spec_from_file_location("mcf_locked_split", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MCFZeroUnlearnLockedSplitTests(unittest.TestCase):
    def test_repair_view_removes_eval_probes_but_preserves_rewrite(self):
        source = [
            {
                "case_id": 1,
                "requested_rewrite": {
                    "prompt": "{} was born in",
                    "subject": "Ada",
                    "target_new": {"str": "Paris"},
                    "target_true": {"str": "London"},
                    "paraphrase_prompts": ["Where was Ada born?"],
                },
                "paraphrase_prompts": ["Ada's birthplace is"],
                "neighborhood_prompts": ["Nearby fact"],
                "generation_prompts": ["Tell me about Ada"],
            }
        ]

        locked = MODULE.build_repair_visible_dataset(source)

        self.assertEqual(
            locked[0]["requested_rewrite"]["prompt"],
            source[0]["requested_rewrite"]["prompt"],
        )
        self.assertEqual(locked[0]["paraphrase_prompts"], [])
        self.assertEqual(locked[0]["neighborhood_prompts"], [])
        self.assertEqual(locked[0]["generation_prompts"], [])
        self.assertEqual(
            locked[0]["requested_rewrite"]["paraphrase_prompts"],
            [],
        )
        MODULE.assert_repair_view_locked(locked)

        # Source data must remain untouched.
        self.assertEqual(source[0]["paraphrase_prompts"], ["Ada's birthplace is"])
        self.assertEqual(
            source[0]["requested_rewrite"]["paraphrase_prompts"],
            ["Where was Ada born?"],
        )

    def test_locked_view_keeps_exact_zerounlearn_seed_selection(self):
        source = [
            {
                "case_id": index,
                "requested_rewrite": {
                    "prompt": "{} relation",
                    "subject": f"S{index}",
                    "target_new": {"str": f"N{index}"},
                    "target_true": {"str": f"T{index}"},
                },
                "paraphrase_prompts": [f"p{index}"],
                "neighborhood_prompts": [f"n{index}"],
                "generation_prompts": [f"g{index}"],
            }
            for index in range(20)
        ]
        locked = MODULE.build_repair_visible_dataset(source)

        source_forget, source_retain = MODULE.selected_indices(
            source, forget_num=4, retain_num=6, seed=7
        )
        locked_forget, locked_retain = MODULE.selected_indices(
            locked, forget_num=4, retain_num=6, seed=7
        )

        rng = random.Random(7)
        expected_forget_records = rng.sample(source[10:], k=4)
        expected_retain_records = rng.sample(source[:10], k=6)
        index_by_identity = {id(row): idx for idx, row in enumerate(source)}
        expected_forget = [index_by_identity[id(row)] for row in expected_forget_records]
        expected_retain = [index_by_identity[id(row)] for row in expected_retain_records]

        self.assertEqual(source_forget, expected_forget)
        self.assertEqual(source_retain, expected_retain)
        self.assertEqual(locked_forget, source_forget)
        self.assertEqual(locked_retain, source_retain)
        self.assertTrue(all(index >= 10 for index in source_forget))
        self.assertTrue(all(index < 10 for index in source_retain))
        self.assertFalse(set(source_forget) & set(source_retain))


if __name__ == "__main__":
    unittest.main()
