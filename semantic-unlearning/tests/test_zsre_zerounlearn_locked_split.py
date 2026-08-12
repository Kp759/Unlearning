import unittest

from scripts.build_zsre_zerounlearn_locked_split import direct_only_record
from scripts import zsre_zero_unlearn_official_eval as zsre


class LockedZsRESplitTests(unittest.TestCase):
    def test_direct_only_record_locks_probes_and_preserves_targets(self):
        raw = {
            "src": "Ada Lovelace was born in London",
            "subject": "Ada Lovelace",
            "answers": ["London"],
            "rephrase": "Where was Ada Lovelace born?",
            "loc": "nq question: capital of France",
            "loc_ans": "Paris",
        }
        record = direct_only_record(raw, 123)
        self.assertEqual(record["case_id"], 123)
        self.assertEqual(record["requested_rewrite"]["subject"], "Ada Lovelace")
        self.assertEqual(
            record["requested_rewrite"]["target_new"]["str"],
            zsre.NEUTRAL_TARGET,
        )
        self.assertEqual(record["requested_rewrite"]["target_true"]["str"], "London")
        self.assertEqual(record["paraphrase_prompts"], [])
        self.assertEqual(record["neighborhood_prompts"], [])
        self.assertEqual(record["generation_prompts"], [])

    def test_prompt_template_replaces_subject(self):
        raw = {
            "src": "Marie Curie worked in Paris",
            "subject": "Marie Curie",
            "answers": ["Paris"],
            "rephrase": "Where did Marie Curie work?",
            "loc": "nq question: capital of Italy",
            "loc_ans": "Rome",
        }
        record = direct_only_record(raw, 5)
        self.assertIn("{}", record["requested_rewrite"]["prompt"])
        self.assertNotIn("Marie Curie", record["requested_rewrite"]["prompt"])


if __name__ == "__main__":
    unittest.main()
