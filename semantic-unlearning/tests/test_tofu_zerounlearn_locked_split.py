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


class TOFUAuthorBalancedLockedSplitTests(unittest.TestCase):
    def make_data(self):
        # Three synthetic authors, four QAs each. Tests use two selected authors
        # and a 2-train / 2-heldout split per author.
        forget = [
            {"question": f"q{i}", "answer": f"a{i}"} for i in range(12)
        ]
        perturbed = [
            {
                "question": f"q{i}",
                "answer": f"a{i}",
                "paraphrased_question": f"pq{i}",
                "paraphrased_answer": f"pa{i}",
                "perturbed_answer": [f"wrong{i}a", f"wrong{i}b"],
            }
            for i in range(12)
        ]
        retain = [
            {"question": f"rq{i}", "answer": f"ra{i}"} for i in range(20)
        ]
        return forget, perturbed, retain

    def build(self, tmp, seed=7):
        forget, perturbed, retain = self.make_data()
        return MODULE.build_seed_split(
            forget,
            perturbed,
            retain,
            seed=seed,
            retain_num=5,
            output_dir=Path(tmp),
            dataset_revision="test-revision",
            qas_per_author=4,
            forget_authors=2,
            train_qas_per_author=2,
        )

    def test_training_view_contains_only_direct_qa(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.build(tmp)
            train_path = Path(manifest["paths"]["train_forget"])
            rows = json.loads(train_path.read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 4)
            MODULE.assert_training_view_locked(rows)
            for row in rows:
                self.assertEqual(
                    set(row), {"question", "answer", "_source_index"}
                )
                self.assertNotIn("paraphrased_question", row)
                self.assertNotIn("perturbed_answer", row)

    def test_selects_authors_then_balances_train_and_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.build(tmp, seed=11)

        sampling = manifest["sampling"]
        expected_authors = random.Random(11).sample(range(3), 2)
        self.assertEqual(sampling["selected_author_block_ids"], expected_authors)
        self.assertEqual(sampling["train_forget_num"], 4)
        self.assertEqual(sampling["same_author_heldout_num"], 4)

        train_ids = set(sampling["forget_source_indices"])
        heldout_ids = set(sampling["heldout_same_author_source_indices"])
        self.assertFalse(train_ids & heldout_ids)

        for author_id in expected_authors:
            details = sampling["per_selected_author"][str(author_id)]
            block = set(range(author_id * 4, author_id * 4 + 4))
            train = set(details["train_source_indices"])
            heldout = set(details["heldout_source_indices"])
            self.assertEqual(len(train), 2)
            self.assertEqual(len(heldout), 2)
            self.assertEqual(train | heldout, block)
            self.assertFalse(train & heldout)

    def test_retain_sampling_uses_fresh_seeded_rng(self):
        _, _, retain = self.make_data()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.build(tmp, seed=11)
        expected_retain = random.Random(11).sample(range(len(retain)), 5)
        self.assertEqual(
            manifest["sampling"]["retain_source_indices"], expected_retain
        )

    def test_seen_and_same_author_holdout_eval_are_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.build(tmp, seed=3)
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
            heldout_para = json.loads(
                Path(manifest["paths"]["heldout_paraphrase"]).read_text(encoding="utf-8")
            )

        train_ids = [row["_source_index"] for row in train]
        self.assertEqual(train_ids, [row["_source_index"] for row in direct])
        self.assertEqual(train_ids, [row["_source_index"] for row in para])
        heldout_ids = [row["_source_index"] for row in heldout]
        self.assertEqual(
            heldout_ids, [row["_source_index"] for row in heldout_para]
        )
        self.assertFalse(set(train_ids) & set(heldout_ids))
        self.assertEqual(len(train_ids), 4)
        self.assertEqual(len(heldout_ids), 4)
        self.assertTrue(all(row["question"].startswith("pq") for row in para))
        self.assertTrue(all(row["question"].startswith("pq") for row in heldout_para))

    def test_author_block_validation(self):
        self.assertEqual(
            MODULE.build_author_blocks(12, 4),
            [list(range(0, 4)), list(range(4, 8)), list(range(8, 12))],
        )
        with self.assertRaises(ValueError):
            MODULE.build_author_blocks(11, 4)

    def test_perturbed_alignment_is_strict(self):
        forget, perturbed, _ = self.make_data()
        MODULE.assert_perturbed_alignment(forget, perturbed)
        bad = [dict(row) for row in perturbed]
        bad[2]["question"] = "different"
        with self.assertRaises(ValueError):
            MODULE.assert_perturbed_alignment(forget, bad)


if __name__ == "__main__":
    unittest.main()
