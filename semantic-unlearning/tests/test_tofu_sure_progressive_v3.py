import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SCRIPT = SCRIPTS / "tofu_sure_rank0_forget_progressive_v3.py"
SPEC = importlib.util.spec_from_file_location("tofu_sure_progressive_v3", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TofuSureProgressiveV3Tests(unittest.TestCase):
    def test_add_next_rows_is_incremental_per_residual_example(self):
        rankings = [
            [10, 11, 12, 13],
            [20, 21, 22],
            [30, 31, 32],
        ]
        sensitive = {10, 20, 30}
        promoted = MODULE.add_next_rows(
            sensitive,
            rankings,
            positions=[0, 2],
            count_per_example=1,
        )
        self.assertEqual(promoted, {0: [11], 2: [31]})
        self.assertEqual(sensitive, {10, 11, 20, 30, 31})
        # Non-residual example 1 must not gain another row.
        self.assertNotIn(21, sensitive)

    def test_shared_global_row_is_not_promoted_twice(self):
        rankings = [[10, 11, 12], [10, 21, 22]]
        sensitive = {10}
        promoted = MODULE.add_next_rows(
            sensitive,
            rankings,
            positions=[0, 1],
            count_per_example=1,
        )
        self.assertEqual(promoted[0], [11])
        self.assertEqual(promoted[1], [21])
        self.assertEqual(sensitive, {10, 11, 21})

    def test_ranking_prefers_content_then_rare_rows(self):
        instances = [SimpleNamespace(source_index=0), SimpleNamespace(source_index=1)]
        answer_rows = {
            0: [1, 2, 3, 4],
            1: [1, 5, 6],
        }
        # token 4 is punctuation; token 1 appears in both docs; all others rare.
        def fake_answer_ids(_tok, instance, *, max_length):
            return answer_rows[int(instance.source_index)]

        def fake_content(_tok, token_id):
            return token_id != 4

        with mock.patch.object(MODULE.locked, "answer_token_ids", side_effect=fake_answer_ids), \
             mock.patch.object(MODULE.locked, "is_content_bearing_token", side_effect=fake_content), \
             mock.patch.object(MODULE.locked, "decoded_token", side_effect=lambda _t, x: str(x)):
            rankings, report = MODULE.build_progressive_rankings(
                object(), instances, max_length=32
            )

        # Rare content rows 2/3 must precede common row 1; punctuation row 4 last.
        self.assertEqual(rankings[0], [2, 3, 1, 4])
        self.assertEqual(rankings[1], [5, 6, 1])
        self.assertEqual(report["policy"], "progressive_rare_content_first")


if __name__ == "__main__":
    unittest.main()
