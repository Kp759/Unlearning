from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module(
    "mcf_zsre_arch_target_true_runner",
    SCRIPTS / "MCF_Scripts" / "run_mcf_zsre_arch_target_true.py",
)
stage2 = load_module(
    "sure_stage2_sparse_repair_explicit_fields",
    SCRIPTS / "sure_stage2_sparse_repair.py",
)
repair = sys.modules["gagd_active_case_repair"]


def raw_record(index: int) -> dict:
    return {
        "case_id": index,
        "requested_rewrite": {
            "prompt": "{} works as",
            "subject": f"Person {index}",
            "target_true": {"str": f"profession {index}"},
            "target_new": {"str": f"replacement {index}"},
        },
        "paraphrase_prompts": [f"Person {index}'s profession is"],
        "neighborhood_prompts": [f"Neighbor {index} works as"],
        "attribute_prompts": ["hidden"],
        "generation_prompts": ["hidden"],
    }


def metric_record(
    true_nll: float,
    new_nll: float,
    *,
    paraphrases: list[tuple[float, float]] | None = None,
    neighborhoods: list[tuple[float, float]] | None = None,
) -> dict:
    paraphrases = paraphrases or [(true_nll, new_nll)]
    neighborhoods = neighborhoods or [(true_nll, new_nll)]
    return {
        "post": {
            "rewrite_prompts_probs": [
                {"target_true": true_nll, "target_new": new_nll}
            ],
            "paraphrase_prompts_probs": [
                {"target_true": true, "target_new": new}
                for true, new in paraphrases
            ],
            "neighborhood_prompts_probs": [
                {"target_true": true, "target_new": new}
                for true, new in neighborhoods
            ],
        }
    }


class MCFZsREArchitectureRunnerTests(unittest.TestCase):
    def test_defaults_match_canonical_zsre_configuration(self):
        args = runner.parse_args(["--model-path", "/model", "--dry-run"])
        self.assertEqual(args.stage1_steps, 600)
        self.assertEqual(args.stage1_lr, 1e-4)
        self.assertEqual(args.ga_weight, 2.0)
        self.assertEqual(args.gd_weight, 1.0)
        self.assertEqual(args.candidate_ranks, "2,8,0")
        self.assertEqual(args.repair_steps, 800)
        self.assertEqual(args.repair_lr, 0.005)
        self.assertEqual(args.repair_l2, 1e-6)
        self.assertEqual(args.constraint_margin, 0.05)
        self.assertEqual(args.retain_num, 1000)

    def test_plan_uses_original_target_roles_and_opens_probes_after_training(self):
        args = runner.parse_args(
            [
                "--model-path",
                "/model",
                "--mcf-path",
                "/mcf.json",
                "--wikidata-dir",
                "/wikidata",
                "--output-root",
                "/out",
                "--dry-run",
            ]
        )
        paths = runner.seed_paths(Path("/out"), 1)
        plan = runner.seed_command_plan(args, paths, 1)
        self.assertEqual(
            [step.label for step in plan],
            [
                "STAGE 1 — ZSRE GA/KL + TARGET_TRUE-ROW RESTORATION",
                "STAGE 2 — ZSRE SPARSE ROWS + MCF PAIRWISE CONSTRAINT",
                "BASE OFFICIAL EVALUATION",
                "FINAL OFFICIAL EVALUATION",
            ],
        )
        stage1_command = " ".join(plan[0].command)
        stage2_command = " ".join(plan[1].command)
        self.assertIn("--sensitive-field target_true", stage1_command)
        self.assertIn("--mcf-sensitive-field target_true", stage2_command)
        self.assertIn("--mcf-reference-field target_new", stage2_command)
        self.assertNotIn("--mcf-path", stage1_command)
        self.assertNotIn("--mcf-path", stage2_command)
        self.assertTrue(all("--quiet" in step.command for step in plan[2:]))

    def test_locked_split_keeps_original_fields_and_hides_probes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "mcf.json"
            source.write_text(
                json.dumps([raw_record(index) for index in range(6)]),
                encoding="utf-8",
            )
            paths = runner.seed_paths(root / "out", 7)
            paths.root.mkdir(parents=True)
            manifest = runner.build_locked_split(
                source,
                paths,
                seed=7,
                forget_num=1,
                retain_num=1,
            )
            visible = json.loads(paths.training_visible.read_text(encoding="utf-8"))
            self.assertEqual(len(visible), 1)
            row = visible[0]
            original = json.loads(source.read_text(encoding="utf-8"))[row["case_id"]]
            self.assertEqual(row["requested_rewrite"], original["requested_rewrite"])
            for field in (
                "paraphrase_prompts",
                "neighborhood_prompts",
                "attribute_prompts",
                "generation_prompts",
            ):
                self.assertNotIn(field, row)
            self.assertFalse(manifest["target_contract"]["field_swapping"])
            self.assertEqual(
                manifest["target_contract"]["sensitive_answer"],
                "requested_rewrite.target_true",
            )

    def test_eff_gen_are_residual_sensitive_preference_with_macro_averaging(self):
        rows = [
            metric_record(
                3.0,
                1.0,
                paraphrases=[(1.0, 2.0), (3.0, 2.0)],
            ),
            metric_record(1.0, 2.0, paraphrases=[(1.0, 2.0)]),
            metric_record(2.0, 2.0, paraphrases=[(2.0, 2.0)]),
        ]
        direct = runner.prompt_metrics(rows, "rewrite_prompts_probs")
        paraphrase = runner.prompt_metrics(rows, "paraphrase_prompts_probs")
        self.assertAlmostEqual(direct["target_true_preference_percent"], 100 / 3)
        self.assertAlmostEqual(direct["target_new_preference_percent"], 100 / 3)
        self.assertAlmostEqual(direct["tie_percent"], 100 / 3)
        # Record-level means are 50%, 100%, and 0%: macro mean = 50%.
        self.assertAlmostEqual(paraphrase["target_true_preference_percent"], 50.0)

    def test_stage2_margin_orientation_supports_target_true_sensitive(self):
        def answer_cache(nll: float) -> repair.AnswerDeltaCache:
            return repair.AnswerDeltaCache(
                base_token_nll=torch.tensor([nll]),
                hidden=torch.zeros((1, 2)),
                selected_probs=torch.zeros((1, 1)),
                target_selected_columns=torch.tensor([-1]),
            )

        caches = [
            repair.RewriteDeltaCache(
                target_new=answer_cache(1.0),
                target_true=answer_cache(3.0),
            )
        ]
        delta = torch.zeros((1, 2))
        historical = stage2.mcf_margins_from_delta_caches(caches, delta)
        target_true_sensitive = stage2.mcf_margins_from_delta_caches(
            caches,
            delta,
            sensitive_field="target_true",
            reference_field="target_new",
        )
        self.assertTrue(torch.allclose(historical, torch.tensor([-2.0])))
        self.assertTrue(torch.allclose(target_true_sensitive, torch.tensor([2.0])))


if __name__ == "__main__":
    unittest.main()
