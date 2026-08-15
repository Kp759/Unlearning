import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import rwku_data as DATA  # noqa: E402
import rwku_split_v1 as SPLIT  # noqa: E402


class RWKUSplitV1Tests(unittest.TestCase):
    @staticmethod
    def _row(level: int, index: int):
        return {
            "query": f"Synthetic level {level} question {index}?",
            "answer": f"answer-{level}-{index}",
            "level": str(level),
            "type": "direct",
            "subject": "Synthetic Subject",
        }

    def _datasets(self):
        datasets = {
            "forget_level1.json": [self._row(1, index) for index in range(8)],
            "forget_level2.json": [self._row(2, index) for index in range(6)],
        }
        for filename in SPLIT.EVALUATION_ONLY_FILES:
            datasets[filename] = [{"source": filename}]
        return datasets

    def test_split_is_fixed_independent_of_target_seed(self):
        datasets = self._datasets()
        seed0 = SPLIT.split_target_datasets(datasets, target_seed=0)
        seed9 = SPLIT.split_target_datasets(datasets, target_seed=9)

        self.assertEqual(
            [DATA.record_sha256(row) for row in seed0["forget_train"]],
            [DATA.record_sha256(row) for row in seed9["forget_train"]],
        )
        self.assertEqual(seed0["manifest"]["split_seed"], 0)
        self.assertEqual(seed9["manifest"]["split_seed"], 0)
        self.assertEqual(seed0["manifest"]["train_fraction"], 0.5)

    def test_level1_and_level2_are_split_separately_and_disjoint(self):
        split = SPLIT.split_target_datasets(self._datasets(), target_seed=0)

        self.assertGreater(len(split["forget_train_level1"]), 0)
        self.assertGreater(len(split["forget_eval_level1"]), 0)
        self.assertGreater(len(split["forget_train_level2"]), 0)
        self.assertGreater(len(split["forget_eval_level2"]), 0)

        train_hashes = {
            DATA.record_sha256(row) for row in split["forget_train"]
        }
        eval_hashes = {
            DATA.record_sha256(row)
            for row in [
                *split["forget_eval_level1"],
                *split["forget_eval_level2"],
            ]
        }
        self.assertFalse(train_hashes & eval_hashes)
        self.assertTrue(
            split["manifest"]["disjointness"]["train_vs_held_out_record_hashes"]
        )

    def test_paraphrases_come_only_from_held_out_level2(self):
        split = SPLIT.split_target_datasets(self._datasets(), target_seed=0)
        held_out_level2_hashes = {
            DATA.record_sha256(row) for row in split["forget_eval_level2"]
        }
        paraphrase_sources = {
            row["source_record_sha256"]
            for row in split["forget_eval_paraphrase"]
        }
        self.assertEqual(paraphrase_sources, held_out_level2_hashes)
        self.assertEqual(
            len(split["forget_eval_paraphrase"]),
            len(split["forget_eval_level2"]),
        )

    def test_duplicate_content_never_crosses_train_eval_boundary(self):
        datasets = self._datasets()
        duplicate = self._row(1, 100)
        datasets["forget_level1.json"] = [
            duplicate,
            dict(duplicate),
            self._row(1, 101),
            self._row(1, 102),
        ]
        split = SPLIT.split_target_datasets(datasets, target_seed=0)
        duplicate_hash = DATA.record_sha256(duplicate)
        train_count = sum(
            DATA.record_sha256(row) == duplicate_hash
            for row in split["forget_train_level1"]
        )
        eval_count = sum(
            DATA.record_sha256(row) == duplicate_hash
            for row in split["forget_eval_level1"]
        )
        self.assertIn((train_count, eval_count), {(2, 0), (0, 2)})

    def test_native_probe_and_utility_files_are_eval_only(self):
        split = SPLIT.split_target_datasets(self._datasets(), target_seed=0)
        roles = split["manifest"]["evaluation"]["evaluation_only_sources"]
        self.assertEqual(set(roles), set(SPLIT.EVALUATION_ONLY_FILES))
        for metadata in roles.values():
            self.assertEqual(metadata["role"], "evaluation_only")
            self.assertFalse(metadata["gradient_allowed"])
            self.assertFalse(metadata["repair_selection_allowed"])
            self.assertFalse(metadata["checkpoint_selection_allowed"])


if __name__ == "__main__":
    unittest.main()
