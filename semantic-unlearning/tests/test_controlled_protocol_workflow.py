import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import audit_controlled_unlearning_protocol as AUDIT  # noqa: E402
import build_controlled_unlearning_protocol as BUILDER  # noqa: E402
import run_controlled_setting5e as RUNNER  # noqa: E402
import select_controlled_candidate as SELECTOR  # noqa: E402


def mcf_row(index):
    return {
        "case_id": index,
        "requested_rewrite": {
            "prompt": "{} works in",
            "subject": f"Subject {index}",
            "target_new": {"str": f"New {index}"},
            "target_true": {"str": f"True {index}"},
            "relation_id": "P1",
        },
        "paraphrase_prompts": [
            f"Validation context for Subject {index}",
            f"Final context for Subject {index}",
        ],
        "neighborhood_prompts": [f"Neighbor {index} works in"],
    }


def evaluation_summary(bundle, candidate_id, forget, utility, locality):
    judge = {
        "judge_id": "judge-a",
        "role": "judge_a_development",
        "provider": "openai_compatible",
        "model": "judge-model-a",
        "base_url": "https://judge-a.example/v1",
        "temperature": 0.0,
        "timeout_seconds": 90.0,
        "max_retries": 3,
        "prompt_version": "controlled-unlearning-judge-v1",
        "api_key_source": "environment:JUDGE_A_KEY",
        "independence_key": "judge-a-key",
    }
    return {
        "schema_version": 1,
        "kind": "controlled_unlearning_evaluation",
        "phase": "development",
        "partition": "validation",
        "protocol_id": bundle["protocol_id"],
        "dataset": bundle["dataset"],
        "fold": bundle["fold"],
        "bundle_sha256": bundle["bundle_sha256"],
        "candidate_id": candidate_id,
        "model_identity": {"candidate_id": candidate_id},
        "judge": judge,
        "metrics": {
            "by_behavior": {
                "avoid_sensitive": {
                    "count": 10,
                    "judge_pass_rate": forget,
                    "record_strict_all_prompts_pass_rate": forget,
                    "mean_max_sensitive_probability": 1.0 - forget,
                },
                "answer_correctly": {
                    "count": 10,
                    "judge_pass_rate": utility,
                    "mean_max_acceptable_probability": utility,
                },
                "preserve_locality": {
                    "count": 10,
                    "judge_pass_rate": locality,
                    "mean_max_acceptable_probability": locality,
                },
            }
        },
        "controls": {
            "token_probability_metrics_present": True,
            "test_results_used_for_repair": False,
        },
    }


class ControlledWorkflowTests(unittest.TestCase):
    def test_selector_unlocks_prompt_free_final_apply_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [mcf_row(index) for index in range(100)]
            mcf_path = root / "mcf.json"
            mcf_path.write_text(json.dumps(rows), encoding="utf-8")
            args = SimpleNamespace(
                dataset="mcf",
                output_dir=str(root / "protocol"),
                seed=0,
                n_folds=5,
                forget_num=25,
                retain_num=25,
                mcf_path=str(mcf_path),
                mcf_url="unused",
                zsre_path=str(root / "zsre.json"),
                zsre_url="unused",
                tofu_data_dir=None,
                tofu_forget_split="forget05",
                tofu_retain_split="retain95",
                tofu_rows_per_author=20,
                max_locality_records=25,
            )
            manifest_path = BUILDER.build_protocol(args)
            fold_dir = manifest_path.parent / "fold_0"
            development_path = fold_dir / "development.json"
            development = json.loads(
                development_path.read_text(encoding="utf-8")
            )
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            baseline_path.write_text(
                json.dumps(
                    evaluation_summary(
                        development,
                        "base",
                        forget=0.0,
                        utility=1.0,
                        locality=1.0,
                    )
                ),
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps(
                    evaluation_summary(
                        development,
                        "candidate",
                        forget=0.9,
                        utility=0.99,
                        locality=0.99,
                    )
                ),
                encoding="utf-8",
            )
            spec_path = root / "candidate_spec.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "candidate_id": "candidate",
                        "dataset": "mcf",
                        "base_model_path": "base-model",
                        "seed": 0,
                        "setting5e": {"steps": 1},
                    }
                ),
                encoding="utf-8",
            )
            receipt_path = root / "selection.json"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "select",
                    "--development-bundle",
                    str(development_path),
                    "--baseline-summary",
                    str(baseline_path),
                    "--candidate",
                    f"candidate={candidate_path}",
                    "--candidate-spec",
                    f"candidate={spec_path}",
                    "--output",
                    str(receipt_path),
                ],
            ):
                SELECTOR.main()
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertFalse(receipt["test_bundle_opened_by_selector"])
            self.assertFalse(
                receipt["final_apply_bundle_opened_by_selector"]
            )
            final_apply_path = Path(receipt["final_apply_bundle_path"])
            final_apply = json.loads(
                final_apply_path.read_text(encoding="utf-8")
            )
            self.assertEqual(final_apply["prompt_cases"], [])
            self.assertFalse(final_apply["contains_judge_b_prompts"])

            run_dir = root / "final_dry_run"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "run",
                    "--bundle",
                    str(final_apply_path),
                    "--phase",
                    "final_apply",
                    "--stage",
                    "final_apply",
                    "--selection-receipt",
                    str(receipt_path),
                    "--output-dir",
                    str(run_dir),
                    "--dry-run",
                ],
            ):
                RUNNER.main()
            plan = json.loads(
                (run_dir / "run_plan.json").read_text(encoding="utf-8")
            )
            self.assertFalse(plan["judge_b_prompts_loaded"])
            self.assertEqual(plan["phase"], "final_apply")

            audit_path = root / "audit.json"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "audit",
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(audit_path),
                ],
            ):
                AUDIT.main()
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertTrue(audit["passed"])


if __name__ == "__main__":
    unittest.main()
