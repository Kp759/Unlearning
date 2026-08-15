import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SCRIPT = SCRIPTS / "build_tofu_zerounlearn_locked_split.py"
SPEC = importlib.util.spec_from_file_location("tofu_locked_split", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TOFUZeroUnlearnLockedSplitTests(unittest.TestCase):
    def make_data(self):
        forget = [
            {"question": f"q{i}", "answer": f"a{i}"} for i in range(8)
        ]
        perturbed = [
            {
                "question": f"q{i}",
                "answer": f"a{i}",
                "paraphrased_question": f"pq{i}",
                "paraphrased_answer": f"pa{i}",
                "perturbed_answer": [f"wrong{i}a", f"wrong{i}b"],
            }
            for i in range(8)
        ]
        retain = [
            {"question": f"rq{i}", "answer": f"ra{i}"} for i in range(12)
        ]
        return forget, perturbed, retain

    def test_training_view_contains_only_direct_qa(self):
        forget, perturbed, retain = self.make_data()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = MODULE.build_seed_split(
                forget,
                perturbed,
                retain,
                seed=7,
                forget_num=3,
                retain_num=5,
                output_dir=Path(tmp),
                dataset_revision="test-revision",
            )
            train_path = Path(manifest["paths"]["train_forget"])
            rows = json.loads(train_path.read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 3)
            MODULE.assert_training_view_locked(rows)
            for row in rows:
                self.assertEqual(
                    set(row), {"question", "answer", "_source_index"}
                )
                self.assertNotIn("paraphrased_question", row)
                self.assertNotIn("perturbed_answer", row)

    def test_sampling_matches_fresh_rng_per_primary_split(self):
        forget, perturbed, retain = self.make_data()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = MODULE.build_seed_split(
                forget,
                perturbed,
                retain,
                seed=11,
                forget_num=3,
                retain_num=5,
                output_dir=Path(tmp),
                dataset_revision=None,
            )
        expected_forget = random.Random(11).sample(range(len(forget)), 3)
        expected_retain = random.Random(11).sample(range(len(retain)), 5)
        self.assertEqual(
            manifest["sampling"]["forget_source_indices"], expected_forget
        )
        self.assertEqual(
            manifest["sampling"]["retain_source_indices"], expected_retain
        )

    def test_paraphrases_and_unseen_facts_are_eval_only_and_aligned(self):
        forget, perturbed, retain = self.make_data()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = MODULE.build_seed_split(
                forget,
                perturbed,
                retain,
                seed=3,
                forget_num=3,
                retain_num=5,
                output_dir=Path(tmp),
                dataset_revision=None,
            )
            train = json.loads(
                Path(manifest["paths"]["train_forget"]).read_text(encoding="utf-8")
            )
            direct = json.loads(
                Path(manifest["paths"]["forget_direct"]).read_text(encoding="utf-8")
            )
            para = json.loads(
                Path(manifest["paths"]["forget_paraphrase"]).read_text(encoding="utf-8")
            )
            heldout = json.loads(
                Path(manifest["paths"]["heldout_direct"]).read_text(encoding="utf-8")
            )

        train_ids = [row["_source_index"] for row in train]
        self.assertEqual(train_ids, [row["_source_index"] for row in direct])
        self.assertEqual(train_ids, [row["_source_index"] for row in para])
        self.assertTrue(
            all(row["question"].startswith("pq") for row in para)
        )
        self.assertTrue(all("perturbed_answer" in row for row in para))
        heldout_ids = {row["_source_index"] for row in heldout}
        self.assertFalse(set(train_ids) & heldout_ids)
        self.assertEqual(len(heldout_ids), len(forget) - 3)

    def test_perturbed_alignment_is_strict(self):
        forget, perturbed, _ = self.make_data()
        MODULE.assert_perturbed_alignment(forget, perturbed)
        bad = [dict(row) for row in perturbed]
        bad[2]["question"] = "different"
        with self.assertRaises(ValueError):
            MODULE.assert_perturbed_alignment(forget, bad)


if __name__ == "__main__":
    unittest.main()
