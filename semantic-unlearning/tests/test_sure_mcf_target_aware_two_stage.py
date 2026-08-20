from __future__ import annotations

import hashlib
import importlib.util
import json
import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


target = load_module(
    "sure_mcf_target_aware_two_stage", "sure_mcf_target_aware_two_stage.py"
)


class TargetAwareTwoStageTests(unittest.TestCase):
    def prompt_records(self):
        return [
            {
                "case_id": 10,
                "source_record_position": 0,
                "prompt_kind": "direct",
                "prompt_index": 0,
            },
            {
                "case_id": 10,
                "source_record_position": 0,
                "prompt_kind": "paraphrase",
                "prompt_index": 0,
            },
        ]

    def test_grouped_report_requires_strict_success_and_positive_margin(self):
        report = target.grouped_pairwise_report(
            torch.tensor([0.0, 0.005]),
            self.prompt_records(),
            required_margin=0.01,
        )
        self.assertEqual(report["FS"], 0.0)
        self.assertEqual(report["GFS"], 100.0)
        self.assertEqual(report["direct_failures"], 1)
        self.assertEqual(report["paraphrase_failures"], 0)
        self.assertEqual(report["direct_margin_failures"], 1)
        self.assertEqual(report["paraphrase_margin_failures"], 1)
        self.assertEqual(report["pairwise_margin_failure_positions"], [0, 1])

    def test_true_ga_and_new_gd_delta_move_nlls_in_intended_directions(self):
        selected_ids = [0, 1]
        logits = torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        reference_logits = torch.tensor([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]])
        hidden = torch.ones(2, 1)
        true_ids = torch.tensor([0, 0])
        reference_ids = torch.tensor([1, 1])
        positions = torch.tensor([0, 1])
        true_cache = target.exact.build_sequence_cache(
            logits,
            hidden,
            true_ids,
            positions,
            selected_ids,
            record_count=2,
            device=torch.device("cpu"),
        )
        reference_cache = target.exact.build_sequence_cache(
            reference_logits,
            hidden,
            reference_ids,
            positions,
            selected_ids,
            record_count=2,
            device=torch.device("cpu"),
        )
        zero = torch.zeros(2, 1)
        base_true = target.exact.exact_sequence_record_nll(true_cache, zero)
        base_reference = target.exact.exact_sequence_record_nll(reference_cache, zero)
        delta = torch.tensor([[-1.0], [1.0]])
        losses = target.stage1_losses(
            true_cache,
            reference_cache,
            delta,
            base_true,
            base_reference,
            target.prompt_kind_masks(self.prompt_records(), device=torch.device("cpu")),
            pairwise_target=0.1,
            true_nll_increase_target=0.1,
            new_nll_decrease_target=0.1,
        )
        self.assertTrue(bool((losses["true_nll_increase"] > 0).all()))
        self.assertTrue(bool((losses["new_nll_decrease"] > 0).all()))
        self.assertTrue(bool((losses["separation"] > 0).all()))

    def test_loader_exposes_direct_and_official_paraphrases_explicitly(self):
        raw = [
            {
                "requested_rewrite": {
                    "prompt": "{} is in",
                    "subject": "X",
                    "target_true": {"str": "old"},
                    "target_new": {"str": "new"},
                },
                "paraphrase_prompts": ["Where is X?", "X can be found in"],
            }
        ]
        payload = json.dumps(raw).encode("utf-8")
        manifest = {
            "dataset": "mcf",
            "source_sha256": hashlib.sha256(payload).hexdigest(),
            "sampling": {"forget_case_ids": [0]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "mcf.json"
            split = root / "manifest.json"
            source.write_bytes(payload)
            split.write_text(json.dumps(manifest), encoding="utf-8")
            originals, prompts, _ = target.load_target_aware_records(source, split)
        self.assertEqual(len(originals), 1)
        self.assertEqual(
            [row["prompt_kind"] for row in prompts],
            [
                "direct",
                "paraphrase",
                "paraphrase",
            ],
        )
        self.assertEqual(
            prompts[0]["requested_rewrite"]["target_sensitive"]["str"], "old"
        )
        self.assertEqual(
            prompts[0]["requested_rewrite"]["target_reference"]["str"], "new"
        )

    def test_exact_residual_solver_repairs_direct_and_paraphrase_together(self):
        selected_ids = [0, 1]
        prompt_records = self.prompt_records()
        true_logits = torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        reference_logits = torch.tensor([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]])
        hidden = torch.ones(2, 1)
        positions = torch.tensor([0, 1])
        true_cache = target.exact.build_sequence_cache(
            true_logits,
            hidden,
            torch.tensor([0, 0]),
            positions,
            selected_ids,
            record_count=2,
            device=torch.device("cpu"),
        )
        reference_cache = target.exact.build_sequence_cache(
            reference_logits,
            hidden,
            torch.tensor([1, 1]),
            positions,
            selected_ids,
            record_count=2,
            device=torch.device("cpu"),
        )
        args = argparse.Namespace(
            stage2_residual_l2_weight=1e-4,
            utility_kl_mean_budget=10.0,
            utility_kl_p95_budget=10.0,
            utility_kl_max_budget=10.0,
            max_total_delta_norm=10.0,
            stage2_constraint_tolerance=1e-5,
            required_pairwise_margin=0.01,
            stage2_maxiter=100,
            stage2_ftol=1e-9,
        )
        residual, _, report = target.solve_residual(
            args,
            rank=1,
            solver_target=0.2,
            row_bases=[torch.ones(1, 1), torch.ones(1, 1)],
            active_ids=selected_ids,
            selected_ids=selected_ids,
            stage1_delta=torch.zeros(2, 1),
            true_cache=true_cache,
            reference_cache=reference_cache,
            prompt_records=prompt_records,
            utility_hidden=torch.zeros(3, 1),
            utility_probabilities=torch.full((3, 2), 0.1),
        )
        separation = target.exact.exact_pairwise_separation(
            true_cache, reference_cache, residual
        )
        self.assertTrue(report["continuous_feasible"])
        self.assertGreaterEqual(float(separation.min()), 0.2 - 1e-5)

    def test_runner_hard_gates_both_fs_and_gfs(self):
        runner = (SCRIPTS / "run_mcf_sure_target_aware.sh").read_text(encoding="utf-8")
        compatibility = (SCRIPTS / "run_mcf_sure_fs100.sh").read_text(encoding="utf-8")
        self.assertIn("sure_mcf_target_aware_two_stage.py", runner)
        self.assertIn("--require-min-fs 100", runner)
        self.assertIn("--require-min-gfs 100", runner)
        self.assertIn("run_mcf_sure_target_aware.sh", compatibility)


if __name__ == "__main__":
    unittest.main()
