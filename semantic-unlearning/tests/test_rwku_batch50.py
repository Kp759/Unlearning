import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rwku_batch50 as BATCH  # noqa: E402


EVAL_ONLY = {
    filename: [] for filename in BATCH.EVALUATION_ONLY_FILES
}


def row(level: int, index: int):
    return {
        "query": f"Level {level} question {index}?",
        "answer": f"answer-{level}-{index}",
        "level": str(level),
        "type": "",
    }


def datasets(*, duplicate_source=False):
    level1 = [row(1, index) for index in range(12)]
    level2 = [row(2, index) for index in range(4)]
    if duplicate_source:
        level1 = [*level1, dict(level1[0]), dict(level1[1])]
        level2 = [*level2, dict(level2[0])]
    return {
        "forget_level1.json": level1,
        "forget_level2.json": level2,
        **EVAL_ONLY,
    }


class BatchTargetSelectionTests(unittest.TestCase):
    def test_batch_seed_is_cyclic_five_person_window(self):
        self.assertEqual(BATCH.batch_target_seeds(0), (0, 1, 2, 3, 4))
        self.assertEqual(BATCH.batch_target_seeds(1), (1, 2, 3, 4, 5))
        self.assertEqual(BATCH.batch_target_seeds(8), (8, 9, 0, 1, 2))
        self.assertEqual(BATCH.batch_target_seeds(9), (9, 0, 1, 2, 3))

    def test_each_target_has_exact_ten_training_examples(self):
        split = BATCH.split_target_rows(datasets(), target_seed=1)
        self.assertEqual(len(split["train_level1"]), 8)
        self.assertEqual(len(split["train_level2"]), 2)
        self.assertEqual(len(split["train"]), 10)
        self.assertGreaterEqual(len(split["heldout_level1"]), 1)
        self.assertGreaterEqual(len(split["heldout_level2"]), 1)

    def test_same_ten_are_efficacy_eligible_but_heldout_is_disjoint(self):
        split = BATCH.split_target_rows(datasets(), target_seed=2)
        train_hashes = {
            value["source_record_sha256"] for value in split["train"]
        }
        heldout_hashes = {
            value["source_record_sha256"]
            for value in [*split["heldout_level1"], *split["heldout_level2"]]
        }
        self.assertEqual(len(split["train"]), BATCH.TRAIN_PER_TARGET)
        self.assertFalse(train_hashes & heldout_hashes)
        self.assertEqual(
            {value["batch50_role"] for value in split["train"]},
            {"forget_train_efficacy"},
        )

    def test_duplicate_content_never_leaks_to_heldout(self):
        split = BATCH.split_target_rows(
            datasets(duplicate_source=True), target_seed=3
        )
        train_hashes = set(split["training_source_hashes"])
        heldout_hashes = set(split["heldout_source_hashes"])
        self.assertFalse(train_hashes & heldout_hashes)

    def test_split_is_deterministic(self):
        first = BATCH.split_target_rows(datasets(), target_seed=4)
        second = BATCH.split_target_rows(datasets(), target_seed=4)
        self.assertEqual(
            first["training_source_hashes"], second["training_source_hashes"]
        )
        self.assertEqual(
            first["heldout_source_hashes"], second["heldout_source_hashes"]
        )

    def test_paraphrases_come_only_from_heldout_level2(self):
        split = BATCH.split_target_rows(datasets(), target_seed=5)
        heldout = {
            value["source_record_sha256"] for value in split["heldout_level2"]
        }
        paraphrase_sources = {
            value["source_record_sha256"]
            for value in split["heldout_paraphrase"]
        }
        self.assertEqual(paraphrase_sources, heldout)


if __name__ == "__main__":
    unittest.main()
