import json
import subprocess
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
import run_controlled_judge_guided_search as SEARCH  # noqa: E402
import run_controlled_final as FINAL  # noqa: E402
import run_controlled_setting5e as RUNNER  # noqa: E402
import select_controlled_candidate as SELECTOR  # noqa: E402
from controlled_unlearning_protocol import (  # noqa: E402
    validate_mcf_post_reload_acceptance,
)


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
    def test_mcf_post_reload_contract_blocks_test_unlock_bypasses(self):
        passing = {
            "kind": "mcf_post_reload_acceptance",
            "checkpoint_was_reloaded": True,
            "thresholds": {
                "max_forget_eff": 0.0,
                "max_forget_gen": 0.0,
                "min_forget_margin": 0.1,
            },
            "observed": {
                "forget_eff": 0.0,
                "forget_gen": 0.0,
                "minimum_rewrite_paraphrase_margin": 0.12,
            },
            "checks": {
                "forget_eff_within_limit": True,
                "forget_gen_within_limit": True,
                "forget_margin_meets_floor": True,
            },
            "passed": True,
            "failure_reasons": [],
        }
        self.assertEqual(
            validate_mcf_post_reload_acceptance(passing),
            passing,
        )
        FINAL._validate_application_before_test(
            {
                "kind": "controlled_model_application_receipt",
                "status": "accepted",
                "dry_run": False,
                "post_reload_acceptance": passing,
            },
            dataset="mcf",
        )

        weak = json.loads(json.dumps(passing))
        weak["thresholds"]["min_forget_margin"] = 0.0
        with self.assertRaisesRegex(ValueError, "margin floor"):
            validate_mcf_post_reload_acceptance(weak)
        with self.assertRaisesRegex(ValueError, "not accepted"):
            FINAL._validate_application_before_test(
                {
                    "kind": "controlled_model_application_receipt",
                    "status": "rejected",
                    "dry_run": False,
                    "post_reload_acceptance": passing,
                },
                dataset="mcf",
            )

    def test_mcf_validation_enforces_reloaded_checkpoint_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mcf_path = root / "multi_counterfact.json"
            mcf_path.write_text("[]", encoding="utf-8")
            output_dir = root / "application"
            setting_checkpoint = (
                output_dir
                / "setting5e"
                / RUNNER.SETTING5E_MODE
                / "checkpoint"
            )
            setting_checkpoint.mkdir(parents=True)
            repair_dir = output_dir / "active_repair"
            selected_checkpoint = repair_dir / "candidate" / "checkpoint"
            selected_checkpoint.mkdir(parents=True)
            repair_dir.mkdir(parents=True, exist_ok=True)
            (repair_dir / "selected_candidate.json").write_text(
                json.dumps({"checkpoint": str(selected_checkpoint)}),
                encoding="utf-8",
            )
            acceptance = {
                "kind": "mcf_post_reload_acceptance",
                "passed": False,
                "failure_reasons": ["forget_gen_within_limit"],
            }
            (repair_dir / "official_eval_selected.json").write_text(
                json.dumps({"post_reload_acceptance": acceptance}),
                encoding="utf-8",
            )
            observed_environments = []

            def fake_run(command, **kwargs):
                if command[0] == "bash":
                    observed_environments.append(kwargs["env"])
                    raise subprocess.CalledProcessError(3, command)
                return SimpleNamespace(returncode=0)

            with mock.patch.object(
                RUNNER.subprocess,
                "run",
                side_effect=fake_run,
            ):
                with self.assertRaises(RUNNER.MCFPostReloadAcceptanceError):
                    RUNNER._run_mcf(
                        python=sys.executable,
                        spec={
                            "base_model_path": "base",
                            "active_repair": {
                                "min_reloaded_forget_margin": 0.1
                            },
                        },
                        materialized={"mcf_path": mcf_path},
                        counts={"forget_count": 10, "retain_count": 200},
                        seed=0,
                        stage="validation",
                        output_dir=output_dir,
                        dry_run=False,
                        commands=[],
                    )

            self.assertEqual(len(observed_environments), 1)
            self.assertEqual(
                observed_environments[0]["REQUIRE_POST_RELOAD_ZERO"],
                "1",
            )
            self.assertEqual(
                observed_environments[0]["MIN_RELOADED_FORGET_MARGIN"],
                "0.1",
            )

    def test_search_continues_only_for_recorded_post_reload_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            application_dir = Path(directory) / "application"
            application_dir.mkdir()
            (application_dir / "application_receipt.json").write_text(
                json.dumps(
                    {
                        "status": "rejected",
                        "dataset": "mcf",
                        "post_reload_acceptance": {
                            "kind": "mcf_post_reload_acceptance",
                            "passed": False,
                            "failure_reasons": [
                                "forget_gen_within_limit"
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            commands = []
            with mock.patch.object(
                SEARCH.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=3),
            ):
                accepted = SEARCH._run_candidate_application(
                    ["python", "candidate.py"],
                    application_dir=application_dir,
                    commands=commands,
                    dry_run=False,
                )
            self.assertFalse(accepted)
            self.assertEqual(commands, [["python", "candidate.py"]])

            with mock.patch.object(
                SEARCH.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=2),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    SEARCH._run_candidate_application(
                        ["python", "candidate.py"],
                        application_dir=application_dir,
                        commands=[],
                        dry_run=False,
                    )

    def test_mcf_candidate_menu_is_preregistered_margin_rank_sweep(self):
        config_dir = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "controlled_unlearning"
        )
        paths = [
            config_dir / "mcf_margin015_rank1.example.json",
            config_dir / "mcf_setting5e_active.example.json",
            config_dir / "mcf_margin025_rank2.example.json",
            config_dir / "mcf_margin040_rank2.example.json",
        ]
        candidates = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in paths
        ]

        self.assertEqual(
            [
                (
                    candidate["active_repair"]["active_margin"],
                    candidate["active_repair"]["repair_rank"],
                )
                for candidate in candidates
            ],
            [(0.15, 1), (0.25, 1), (0.25, 2), (0.4, 2)],
        )
        for candidate in candidates:
            repair = candidate["active_repair"]
            self.assertEqual(repair["repair_steps"], 100)
            self.assertEqual(repair["repair_lr"], 0.005)
            self.assertEqual(repair["retain_calibration_num"], 200)
            self.assertEqual(repair["min_reloaded_forget_margin"], 0.1)
            self.assertTrue(repair["project_away_retain_hidden"])

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
