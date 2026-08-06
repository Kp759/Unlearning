import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import aggregate_zsre_gagd_results as AGG  # noqa: E402
import gagd_active_case_repair as ACTIVE  # noqa: E402
import zsre_gate_aware_sensitive_row_repair as REPAIR  # noqa: E402


def metric_result(
    *,
    forget_eff=0.0,
    forget_gen=0.0,
    forget_spe=80.0,
    retain_eff=90.0,
    retain_gen=89.0,
    retain_spe=88.0,
    ppl=10.0,
):
    return {
        "forget": {
            "Eff": forget_eff,
            "Gen": forget_gen,
            "Spe": forget_spe,
        },
        "retain": {
            "Eff": retain_eff,
            "Gen": retain_gen,
            "Spe": retain_spe,
        },
        "forget_PPL": ppl,
    }


class TinyTiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(6, 3)
        self.transformer = nn.Linear(3, 3, bias=False)
        self.lm_head = nn.Linear(3, 6, bias=False)
        self.lm_head.weight = self.embed.weight
        self.config = SimpleNamespace(tie_word_embeddings=True)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value


class GateAwareSensitiveRowRepairTests(unittest.TestCase):
    def test_only_selected_sensitive_rows_receive_deltas(self):
        weight = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        baseline = weight.clone()
        selected = [1, 4]
        original_rows = baseline[selected].clone()
        delta = torch.tensor([[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]])
        REPAIR.materialize_sensitive_rows(
            weight,
            selected,
            original_rows,
            delta,
            0.5,
        )
        self.assertTrue(torch.equal(weight[0], baseline[0]))
        self.assertTrue(torch.equal(weight[2], baseline[2]))
        self.assertTrue(torch.equal(weight[3], baseline[3]))
        self.assertTrue(torch.equal(weight[1], baseline[1] + 0.5 * delta[0]))
        self.assertTrue(torch.equal(weight[4], baseline[4] + 0.5 * delta[1]))

    def test_input_embeddings_and_transformer_remain_unchanged(self):
        model = TinyTiedModel()
        embedding_before = model.embed.weight.detach().clone()
        transformer_before = model.transformer.weight.detach().clone()
        lm_head_before = model.lm_head.weight.detach().clone()
        output = ACTIVE.freeze_model_for_output_repair(model)
        selected = [2]
        REPAIR.materialize_sensitive_rows(
            output.weight,
            selected,
            output.weight[selected].detach().clone(),
            torch.tensor([[0.5, -0.25, 1.0]]),
            1.0,
        )
        self.assertTrue(torch.equal(model.embed.weight, embedding_before))
        self.assertTrue(torch.equal(model.transformer.weight, transformer_before))
        self.assertTrue(torch.equal(output.weight[0], lm_head_before[0]))
        self.assertFalse(torch.equal(output.weight[2], lm_head_before[2]))
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))

    def test_active_margin_matches_sensitive_row_contract(self):
        margin = REPAIR.active_constraint_margins(
            hidden=torch.tensor([[2.0, 1.0]]),
            sensitive_logits=torch.tensor([3.0]),
            best_other_logits=torch.tensor([4.0]),
            selected_row_columns=torch.tensor([0]),
            delta_rows=torch.tensor([[-1.0, 0.0]]),
        )
        # 4 - (3 + [2,1] @ [-1,0]) = 3.
        self.assertEqual(float(margin.item()), 3.0)

    def test_protected_constraint_detects_edited_row_overtaking_retain_target(self):
        margin = REPAIR.protected_constraint_margins(
            hidden=torch.tensor([[1.0, 0.0]]),
            target_logits=torch.tensor([2.0]),
            strongest_unchanged_logits=torch.tensor([1.0]),
            selected_logits=torch.tensor([[0.5]]),
            target_selected_columns=torch.tensor([-1]),
            delta_rows=torch.tensor([[2.0, 0.0]]),
        )
        self.assertLess(float(margin.item()), 0.0)

    def test_protected_constraint_detects_selected_target_below_unchanged(self):
        margin = REPAIR.protected_constraint_margins(
            hidden=torch.tensor([[1.0, 0.0]]),
            target_logits=torch.tensor([2.0]),
            strongest_unchanged_logits=torch.tensor([1.8]),
            selected_logits=torch.tensor([[2.0]]),
            target_selected_columns=torch.tensor([0]),
            delta_rows=torch.tensor([[-0.5, 0.0]]),
        )
        self.assertLess(float(margin.item()), 0.0)

    def test_retain_kl_zero_at_origin_and_positive_after_damage(self):
        cache = ACTIVE.RetainKLCache(
            hidden=torch.tensor([[1.0, 0.0]]),
            candidate_selected_probs=torch.tensor([[0.2]]),
            reference_selected_probs=torch.tensor([[0.2]]),
            baseline_kl=torch.tensor([0.0]),
        )
        zero = REPAIR.retain_kl_from_caches(
            [cache],
            torch.zeros((1, 2)),
        )
        damaged = REPAIR.retain_kl_from_caches(
            [cache],
            torch.tensor([[3.0, 0.0]]),
        )
        self.assertAlmostEqual(float(zero.item()), 0.0, places=7)
        self.assertGreater(float(damaged.item()), 0.0)

    def test_candidate_selection_rejects_zero_forgetting_with_utility_damage(self):
        setting = metric_result()
        damaged = metric_result(retain_eff=89.0)
        gate = REPAIR.full_gate_report(setting, damaged)
        selected = REPAIR.select_all_gates_candidate(
            [
                {
                    "scale": 0.5,
                    "materialized_delta_norm": 1.0,
                    "full_gate": gate,
                }
            ],
            setting5_target_already_met=False,
        )
        self.assertFalse(gate["passed"])
        self.assertIsNone(selected)

    def test_candidate_selection_accepts_only_all_gates_passing_candidate(self):
        setting = metric_result(forget_eff=10.0, forget_gen=12.0)
        passing = metric_result(ppl=10.1)
        failing = metric_result(retain_spe=87.0, ppl=10.1)
        candidates = [
            {
                "scale": 0.5,
                "materialized_delta_norm": 0.8,
                "full_gate": REPAIR.full_gate_report(setting, failing),
            },
            {
                "scale": 0.75,
                "materialized_delta_norm": 1.0,
                "full_gate": REPAIR.full_gate_report(setting, passing),
            },
        ]
        selected = REPAIR.select_all_gates_candidate(
            candidates,
            setting5_target_already_met=False,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["scale"], 0.75)

    def test_rejected_seed_emits_no_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            result = REPAIR.save_selected_checkpoint_if_accepted(
                accepted=False,
                model=object(),
                tok=object(),
                output_dir=output,
            )
            self.assertIsNone(result)
            self.assertFalse((output / "selected_checkpoint").exists())

    def test_zsre_aggregate_uses_sample_standard_deviation(self):
        mean, sample_sd = AGG.mean_std([0.0, 2.0])
        self.assertEqual(mean, 1.0)
        self.assertAlmostEqual(sample_sd, math.sqrt(2.0))


if __name__ == "__main__":
    unittest.main()
