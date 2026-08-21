from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "MCF_Scripts" / "run_mcf_sure_two_stage.py"
SPEC = importlib.util.spec_from_file_location("mcf_sure_two_stage", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def record(
    true_nll: float,
    new_nll: float,
    *,
    paraphrases: list[tuple[float, float]] | None = None,
    neighborhoods: list[tuple[float, float]] | None = None,
    subject: str = "Person X",
) -> dict:
    paraphrases = paraphrases or [(true_nll, new_nll)]
    neighborhoods = neighborhoods or [(true_nll, new_nll)]
    rewrite = {
        "prompt": "{} works as",
        "subject": subject,
        "target_true": {"str": "politician"},
        "target_new": {"str": "actor"},
    }
    return {
        "requested_rewrite": rewrite,
        "post": {
            "rewrite_prompts_probs": [{"target_true": true_nll, "target_new": new_nll}],
            "paraphrase_prompts_probs": [
                {"target_true": true, "target_new": new} for true, new in paraphrases
            ],
            "neighborhood_prompts_probs": [
                {"target_true": true, "target_new": new} for true, new in neighborhoods
            ],
        },
    }


class MCFSURETwoStageEntrypointTests(unittest.TestCase):
    def test_target_contract_never_swaps_fields(self):
        self.assertEqual(
            MODULE.TARGET_CONTRACT["sensitive_answer"],
            "requested_rewrite.target_true",
        )
        self.assertEqual(
            MODULE.TARGET_CONTRACT["non_sensitive_replacement"],
            "requested_rewrite.target_new",
        )
        self.assertFalse(MODULE.TARGET_CONTRACT["field_swapping"])
        self.assertIn(
            "gradient_ascent",
            MODULE.TARGET_CONTRACT["stage1_target_true_operation"],
        )
        self.assertIn(
            "gradient_descent",
            MODULE.TARGET_CONTRACT["stage1_target_new_operation"],
        )

    def test_eff_gen_are_lower_residual_sensitive_preference(self):
        rows = [
            record(1.0, 2.0),  # target_true/sensitive still preferred
            record(3.0, 2.0),  # target_new/non-sensitive preferred
            record(2.0, 2.0),  # exact tie
        ]
        direct = MODULE.pairwise_prompt_metrics(rows, "rewrite_prompts_probs")
        self.assertAlmostEqual(direct["target_true_preference_percent"], 100 / 3)
        self.assertAlmostEqual(direct["target_new_preference_percent"], 100 / 3)
        self.assertAlmostEqual(direct["exact_tie_percent"], 100 / 3)

    def test_gen_is_macro_averaged_by_record(self):
        rows = [
            record(3.0, 1.0, paraphrases=[(1.0, 2.0), (3.0, 2.0)]),
            record(3.0, 1.0, paraphrases=[(1.0, 2.0)]),
        ]
        result = MODULE.pairwise_prompt_metrics(rows, "paraphrase_prompts_probs")
        # First record is 50%; second is 100%; macro mean is 75%.
        self.assertAlmostEqual(result["target_true_preference_percent"], 75.0)

    def test_spe_matches_probability_difference_definition(self):
        rows = [record(1.0, 2.0, neighborhoods=[(1.0, 2.0)])]
        result = MODULE.specificity_metrics(rows)
        expected = 100.0 * (MODULE.math.exp(-1.0) - MODULE.math.exp(-2.0))
        self.assertAlmostEqual(result["Spe"], expected)
        self.assertEqual(result["Spe_success"], 100.0)

    def test_plan_trains_before_opening_official_probes(self):
        args = MODULE.parse_args(
            [
                "--model-path",
                "/model",
                "--mcf-path",
                "/mcf.json",
                "--wikipedia-dir",
                "/wiki",
                "--output-root",
                "/out",
                "--dry-run",
            ]
        )
        paths = MODULE.seed_paths(Path("/out"), 1)
        plan = MODULE.seed_command_plan(
            args,
            paths,
            Path("/cache.pt"),
            100_000,
            100_000,
            90_000,
            1,
        )
        labels = [step.label for step in plan]
        self.assertEqual(
            labels,
            [
                "LOCKED DIRECT-ONLY SPLIT",
                "STAGE 1 + CONDITIONAL STAGE 2",
                "BASE OFFICIAL EVALUATION",
                "FROZEN SURE OFFICIAL EVALUATION",
            ],
        )
        learner = " ".join(plan[1].command)
        self.assertIn("sure_mcf_target_aware_direct_only.py", learner)
        self.assertIn("--stage1-rank 4", learner)
        self.assertIn("--stage1-true-ga-weight 10.0", learner)
        self.assertIn("--stage1-new-gd-weight 10.0", learner)
        self.assertIn("--stage2-rank-ladder 2,4,8", learner)
        self.assertNotIn("paraphrase", learner.lower())
        self.assertNotIn("--mcf-path", learner)
        self.assertTrue(all("--quiet" in step.command for step in plan[2:]))

    def test_seed_report_checks_roles_and_emits_primary_metrics(self):
        forget = [record(3.0, 1.0, neighborhoods=[(1.0, 2.0)])]
        retain = [record(1.0, 2.0, subject="Person Y")]
        payload = {
            "seed": 1,
            "unlearn_num": 1,
            "retain_num": 1,
            "sample_mode": "official",
            "forget_PPL": 12.5,
            "forget_raw": forget,
            "retain_raw": retain,
        }
        training = [
            {
                "case_id": 17,
                "data_role": "target_aware_direct_training_only",
                "requested_rewrite": forget[0]["requested_rewrite"],
            }
        ]
        manifest = {
            "dataset": "mcf",
            "seed": 1,
            "learner_adapter_contract": {
                "sensitive_answer_field": "target_true",
                "reference_answer_field": "target_new",
                "direct_only": True,
                "official_paraphrases_visible_to_learner": False,
            },
            "data_roles": {
                "GFS_checkpoint_selection": False,
                "heldout_probes_visible_during_training": False,
                "benchmark_retain_examples_visible_to_training": 0,
            },
        }
        report = MODULE.build_seed_report(
            base=payload,
            post=payload,
            manifest=manifest,
            training_rows=training,
            seed=1,
            forget_num=1,
            retain_num=1,
        )
        self.assertEqual(report["primary_metrics"]["Eff"]["value"], 0.0)
        self.assertEqual(report["primary_metrics"]["Gen"]["value"], 0.0)
        self.assertEqual(report["forget_audits"]["FS_direct_higher_is_better"], 100.0)
        self.assertEqual(report["primary_metrics"]["PPL"]["value"], 12.5)

    def test_aggregate_writes_table_rows(self):
        report = {
            "seed": 1,
            "primary_metrics": {
                name: {"value": value}
                for name, value in {
                    "Eff": 0.0,
                    "Gen": 20.0,
                    "Spe": 80.0,
                    "PPL": 12.0,
                }.items()
            },
            "base_primary_metrics": {
                name: {"value": value}
                for name, value in {
                    "Eff": 80.0,
                    "Gen": 75.0,
                    "Spe": 88.0,
                    "PPL": 11.0,
                }.items()
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregate = MODULE.aggregate_reports([report], root)
            table = (root / "table1_rows.md").read_text(encoding="utf-8")
            csv_rows = list(
                MODULE.csv.DictReader((root / "table1_rows.csv").open(encoding="utf-8"))
            )
            loaded = json.loads((root / "aggregate_metrics.json").read_text())
        self.assertFalse(aggregate["paper_ready_seed_count"])
        self.assertIn("Eff ↓", table)
        self.assertEqual(csv_rows[1]["Method"], MODULE.METHOD)
        self.assertEqual(loaded["metric_schema"], MODULE.METRIC_SCHEMA)


if __name__ == "__main__":
    unittest.main()
