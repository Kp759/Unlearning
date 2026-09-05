"""RSNR-V2 Stage 0 anchors its erasure direction on abstention, not target_new.

RSNR-V1A's frozen spec declares ``target_new_used: False``.  The directional
Emb+LM machinery it reuses was written against ``d = h_true - h_new``, so these
tests pin the re-anchored contract: the abstention field is what gets read, the
original records are untouched, and the first-answer-token degeneracy that the
construction implies is reported rather than hidden.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import mcf_rsnr_v2_abstention_direction as anchor  # noqa: E402
import sure_context_projection as context  # noqa: E402


def _records():
    return [
        {
            "case_id": 7,
            "requested_rewrite": {
                "subject": "Ada",
                "prompt": "{} was born in",
                "target_new": {"str": "Rome"},
                "target_true": {"str": "Paris"},
            },
        },
        {
            "case_id": 9,
            "requested_rewrite": {
                "subject": "Bo",
                "prompt": "{} plays",
                "target_new": {"str": "guitar"},
                "target_true": {"str": "the cello"},
            },
        },
    ]


class WordTokenizer:
    """Whitespace tokenizer with a stable id per word; no special tokens."""

    def __init__(self):
        self._ids = {}

    def _id(self, word):
        return self._ids.setdefault(word, 100 + len(self._ids))

    def __call__(self, text, add_special_tokens=True, **kwargs):
        values = text if isinstance(text, list) else [text]
        rows = [[self._id(w) for w in v.split()] for v in values]
        return {"input_ids": rows if isinstance(text, list) else rows[0]}

    def decode(self, token_ids):
        back = {v: k for k, v in self._ids.items()}
        return " ".join(back[int(t)] for t in token_ids)


class AttachAbstentionReferenceTest(unittest.TestCase):
    def test_adds_constant_abstention_answer_to_every_record(self):
        out = anchor.attach_abstention_reference(_records())
        self.assertEqual(len(out), 2)
        for record in out:
            block = record["requested_rewrite"][anchor.ABSTENTION_FIELD]
            self.assertEqual(block["str"], anchor.ABSTENTION_TEXT)

    def test_does_not_mutate_the_input_records(self):
        records = _records()
        anchor.attach_abstention_reference(records)
        for record in records:
            self.assertNotIn(anchor.ABSTENTION_FIELD, record["requested_rewrite"])

    def test_preserves_prompt_subject_and_target_true(self):
        source = _records()
        out = anchor.attach_abstention_reference(source)
        for before, after in zip(source, out):
            rr_before, rr_after = before["requested_rewrite"], after["requested_rewrite"]
            self.assertEqual(rr_after["prompt"], rr_before["prompt"])
            self.assertEqual(rr_after["subject"], rr_before["subject"])
            self.assertEqual(rr_after["target_true"], rr_before["target_true"])
            self.assertEqual(after["case_id"], before["case_id"])

    def test_refuses_to_overwrite_an_existing_abstention_field(self):
        records = _records()
        records[0]["requested_rewrite"][anchor.ABSTENTION_FIELD] = {"str": "other"}
        with self.assertRaises(ValueError):
            anchor.attach_abstention_reference(records)

    def test_rejects_empty_abstention_text(self):
        with self.assertRaises(ValueError):
            anchor.attach_abstention_reference(_records(), abstention_text="   ")


class FirstTokenDegeneracyTest(unittest.TestCase):
    """The zero hidden contrast at token 0 is structural, not numerical."""

    def test_first_answer_token_prompts_are_identical(self):
        tok = WordTokenizer()
        records = _records()
        referenced = anchor.attach_abstention_reference(records)
        sensitive = context.expand_answer_field_cases(
            records, tok, field="target_true", llama_like=False
        )
        reference = context.expand_answer_field_cases(
            referenced, tok, field=anchor.ABSTENTION_FIELD, llama_like=False
        )
        first_sensitive = [c for c in sensitive if c.token_index == 0]
        first_reference = [c for c in reference if c.token_index == 0]
        self.assertEqual(len(first_sensitive), len(records))
        for s, r in zip(first_sensitive, first_reference):
            self.assertEqual(
                s.prompt,
                r.prompt,
                "identical prompts at token 0 force the decoder-discriminant branch",
            )

    def test_later_answer_tokens_do_differ(self):
        tok = WordTokenizer()
        records = _records()
        referenced = anchor.attach_abstention_reference(records)
        sensitive = context.expand_answer_field_cases(
            records, tok, field="target_true", llama_like=False
        )
        reference = context.expand_answer_field_cases(
            referenced, tok, field=anchor.ABSTENTION_FIELD, llama_like=False
        )
        # "the cello" is two tokens, so record position 1 has a token_index 1.
        later_sensitive = [
            c for c in sensitive if c.record_position == 1 and c.token_index == 1
        ]
        later_reference = [
            c for c in reference if c.record_position == 1 and c.token_index == 1
        ]
        self.assertTrue(later_sensitive and later_reference)
        self.assertNotEqual(later_sensitive[0].prompt, later_reference[0].prompt)


class AbstentionMarginViewTest(unittest.TestCase):
    """Re-anchoring only the direction would leave the scale gate on target_new."""

    def test_reference_slot_is_filled_with_the_abstention_text(self):
        view = anchor.abstention_margin_view(_records())
        for record in view:
            rr = record["requested_rewrite"]
            self.assertEqual(rr["target_new"]["str"], anchor.ABSTENTION_TEXT)
            self.assertEqual(rr["_reference_slot_holds"], "abstention")

    def test_sensitive_slot_and_prompt_are_untouched(self):
        source = _records()
        view = anchor.abstention_margin_view(source)
        for before, after in zip(source, view):
            self.assertEqual(
                after["requested_rewrite"]["target_true"],
                before["requested_rewrite"]["target_true"],
            )
            self.assertEqual(
                after["requested_rewrite"]["prompt"],
                before["requested_rewrite"]["prompt"],
            )

    def test_does_not_mutate_the_input_records(self):
        source = _records()
        anchor.abstention_margin_view(source)
        self.assertEqual(source[0]["requested_rewrite"]["target_new"]["str"], "Rome")
        self.assertEqual(source[1]["requested_rewrite"]["target_new"]["str"], "guitar")

    def test_margin_helper_consumes_the_view_as_the_reference(self):
        import sure_stage2_sparse_repair as stage2

        instances = stage2.mcf_instances(anchor.abstention_margin_view(_records()))
        self.assertTrue(instances)
        for instance in instances:
            self.assertEqual(instance.target_new, anchor.ABSTENTION_TEXT)
        self.assertEqual(instances[0].target_true, "Paris")

    def test_rejects_empty_abstention_text(self):
        with self.assertRaises(ValueError):
            anchor.abstention_margin_view(_records(), abstention_text="")


class DirectionSourceSummaryTest(unittest.TestCase):
    def _reports(self, **sources):
        return [{"token_id": 42, "direction_sources": dict(sources)}]

    def test_summary_records_the_anchor_and_denies_target_new(self):
        summary = anchor.summarize_direction_sources(
            self._reports(**{anchor.DISCRIMINANT_SOURCE: 3, anchor.HIDDEN_SOURCE: 1})
        )
        self.assertEqual(summary["anchor"], "abstention")
        self.assertFalse(summary["target_new_used"])
        self.assertEqual(summary["total_directions"], 4)
        self.assertAlmostEqual(summary["decoder_discriminant_fraction"], 0.75)
        self.assertAlmostEqual(summary["hidden_contrast_fraction"], 0.25)

    def test_degenerate_fallback_is_surfaced_with_its_token_ids(self):
        summary = anchor.summarize_direction_sources(
            self._reports(**{anchor.DEGENERATE_SOURCE: 2})
        )
        self.assertEqual(summary["degenerate_fallback_count"], 2)
        self.assertEqual(summary["degenerate_fallback_token_ids"], [42])

    def test_empty_reports_do_not_divide_by_zero(self):
        summary = anchor.summarize_direction_sources([])
        self.assertEqual(summary["total_directions"], 0)
        self.assertEqual(summary["hidden_contrast_fraction"], 0.0)

    def test_unrecognized_source_label_is_rejected(self):
        with self.assertRaises(RuntimeError):
            anchor.assert_target_new_unused(
                self._reports(**{"hidden_target_new_contrast": 1})
            )

    def test_known_abstention_sources_pass(self):
        anchor.assert_target_new_unused(
            self._reports(
                **{anchor.HIDDEN_SOURCE: 1, anchor.DISCRIMINANT_SOURCE: 2}
            )
        )


class SeparabilityTest(unittest.TestCase):
    def test_flags_target_true_sharing_the_abstention_first_token(self):
        tok = WordTokenizer()
        records = _records()
        records[0]["requested_rewrite"]["target_true"]["str"] = "I moved"
        report = anchor.check_abstention_separability(
            tok, records, llama_like=False, abstention_text="I don't know."
        )
        self.assertFalse(report["separable"])
        self.assertEqual(len(report["first_token_collisions"]), 1)
        self.assertEqual(report["first_token_collisions"][0]["case_id"], 7)

    def test_clean_records_are_separable(self):
        report = anchor.check_abstention_separability(
            WordTokenizer(), _records(), llama_like=False
        )
        self.assertTrue(report["separable"])
        self.assertEqual(report["first_token_collisions"], [])


if __name__ == "__main__":
    unittest.main()
