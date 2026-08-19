from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


split_mod = load_script(
    "mcf_target_true_split",
    "build_mcf_sure_target_true_canonical_split.py",
)
eval_mod = load_script(
    "mcf_target_true_eval",
    "evaluate_mcf_target_true_sensitive.py",
)


class TargetTrueSensitiveCanonicalTests(unittest.TestCase):
    def sample_record(self):
        return {
            "case_id": 17,
            "requested_rewrite": {
                "prompt": "{} works as",
                "subject": "Person X",
                "target_true": {"str": "politician"},
                "target_new": {"str": "actor"},
            },
            "paraphrase_prompts": ["Person X's profession is"],
            "neighborhood_prompts": ["Person Y works as"],
            "generation_prompts": ["Tell me about Person X"],
        }

    def test_training_adapter_maps_original_true_into_sensitive_slot(self):
        swapped = split_mod.direct_only_swapped_record(self.sample_record(), 17)
        rr = swapped["requested_rewrite"]
        self.assertEqual(rr["target_new"]["str"], "politician")
        self.assertEqual(rr["target_true"]["str"], "actor")
        self.assertEqual(
            swapped["semantic_adapter"]["original_sensitive_field"],
            "target_true",
        )

    def test_training_adapter_hides_all_evaluation_probes(self):
        swapped = split_mod.direct_only_swapped_record(self.sample_record(), 17)
        self.assertEqual(swapped["paraphrase_prompts"], [])
        self.assertEqual(swapped["neighborhood_prompts"], [])
        self.assertEqual(swapped["generation_prompts"], [])

    def test_strict_sensitive_preference_uses_original_target_true(self):
        rows = [
            {
                "post": {
                    "rewrite_prompts_probs": [
                        {"target_true": 2.0, "target_new": 3.0},  # sensitive wins
                    ]
                }
            },
            {
                "post": {
                    "rewrite_prompts_probs": [
                        {"target_true": 4.0, "target_new": 1.0},  # reference wins
                    ]
                }
            },
            {
                "post": {
                    "rewrite_prompts_probs": [
                        {"target_true": 5.0, "target_new": 5.0},  # strict tie: false
                    ]
                }
            },
        ]
        stats = eval_mod._record_stats(rows, "rewrite_prompts_probs")
        self.assertEqual(stats["sensitive_preferred_prompt_instances"], 1)
        self.assertEqual(stats["reference_preferred_prompt_instances"], 1)
        self.assertEqual(stats["exact_nll_ties"], 1)
        self.assertAlmostEqual(sum(stats["preference_per_record"]) / 3.0, 1.0 / 3.0)

    def test_positive_sensitive_nll_delta_means_suppression(self):
        base = [
            {
                "requested_rewrite": {"id": 1},
                "post": {
                    "rewrite_prompts_probs": [
                        {"target_true": 2.0, "target_new": 4.0},
                    ]
                },
            }
        ]
        post = [
            {
                "requested_rewrite": {"id": 1},
                "post": {
                    "rewrite_prompts_probs": [
                        {"target_true": 6.0, "target_new": 4.0},
                    ]
                },
            }
        ]
        summary = eval_mod._summarize_pair(base, post, "rewrite_prompts_probs")
        self.assertAlmostEqual(summary["delta_sensitive_nll"]["mean"], 4.0)
        self.assertAlmostEqual(
            summary["post_nll_separation_sensitive_minus_reference"]["mean"],
            2.0,
        )
        self.assertAlmostEqual(summary["post_sensitive_preference"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
