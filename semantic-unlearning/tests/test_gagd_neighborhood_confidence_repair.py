import math
import sys
import unittest
from pathlib import Path

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import gagd_active_case_repair as ACTIVE  # noqa: E402
import gagd_compare as GAGD  # noqa: E402
import gagd_neighborhood_confidence_repair as MODULE  # noqa: E402


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    unk_token_id = None

    def __call__(self, text, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, token_ids):
        return "".join(chr(int(token_id)) for token_id in token_ids)


def sampled_record(
    record_index,
    sampled_position,
    target_new,
    target_true,
    neighborhoods,
):
    raw_record = {
        "requested_rewrite": {
            "prompt": "{}",
            "subject": f"subject-{record_index}",
            "target_new": {"str": target_new},
            "target_true": {"str": target_true},
        },
        "neighborhood_prompts": list(neighborhoods),
    }
    return ACTIVE.SampledMCFRecord(
        record_index=record_index,
        sampled_position=sampled_position,
        example=GAGD.Example(
            prompt=f"prompt-{record_index}",
            answer=target_new,
            target_new=target_new,
            target_true=target_true,
            source="mcf",
        ),
        raw_record=raw_record,
        rewrite_prompt=f"prompt-{record_index}",
        paraphrase_prompts=(),
        target_new=target_new,
        target_true=target_true,
    )


def report(
    *,
    record_index,
    sampled_position,
    prompt_type,
    margin,
    probability_gap=0.0,
):
    return {
        "record_index": record_index,
        "sampled_position": sampled_position,
        "prompt_type": prompt_type,
        "margin": float(margin),
        "probability_diff_true_minus_new": float(probability_gap),
    }


class NeighborhoodConfidenceRepairTests(unittest.TestCase):
    def test_schedule_is_progressive_and_ends_at_requested_target(self):
        self.assertEqual(
            MODULE.parse_spe_schedule("30,20,75,30", 50.0),
            [20.0, 30.0, 50.0],
        )

    def test_neighborhood_instances_preserve_record_and_prompt_metadata(self):
        records = [sampled_record(91, 3, "new", "true", ["neighbor A", "neighbor B"])]
        instances = MODULE.expand_neighborhood_prompt_instances(records)
        self.assertEqual(len(instances), 2)
        self.assertEqual(instances[1].record_index, 91)
        self.assertEqual(instances[1].sampled_position, 3)
        self.assertEqual(instances[1].prompt_type, "neighborhood")
        self.assertEqual(instances[1].prompt_index, 1)
        self.assertEqual(instances[1].prompt, "neighbor B")
        self.assertEqual(instances[1].target_new, "new")
        self.assertEqual(instances[1].target_true, "true")

    def test_local_metrics_use_official_record_weighting(self):
        forget = [
            report(
                record_index=0,
                sampled_position=0,
                prompt_type="rewrite",
                margin=-1,
            ),
            report(
                record_index=1,
                sampled_position=1,
                prompt_type="rewrite",
                margin=1,
            ),
            report(
                record_index=0,
                sampled_position=0,
                prompt_type="paraphrase",
                margin=-1,
            ),
            report(
                record_index=0,
                sampled_position=0,
                prompt_type="paraphrase",
                margin=1,
            ),
            report(
                record_index=1,
                sampled_position=1,
                prompt_type="paraphrase",
                margin=1,
            ),
        ]
        neighborhood = [
            report(
                record_index=0,
                sampled_position=0,
                prompt_type="neighborhood",
                margin=1,
                probability_gap=0.2,
            ),
            report(
                record_index=0,
                sampled_position=0,
                prompt_type="neighborhood",
                margin=1,
                probability_gap=0.4,
            ),
            report(
                record_index=1,
                sampled_position=1,
                prompt_type="neighborhood",
                margin=1,
                probability_gap=0.7,
            ),
        ]
        metrics = MODULE.official_metrics_from_reports(forget, neighborhood)
        self.assertAlmostEqual(metrics["Eff"], 50.0)
        self.assertAlmostEqual(metrics["Gen"], 25.0)
        self.assertAlmostEqual(metrics["Spe"], 50.0)

    def test_low_spe_row_selection_uses_both_answers_of_low_records_only(self):
        tokenizer = TinyTokenizer()
        records = [
            sampled_record(10, 0, "A", "B", ["n0"]),
            sampled_record(11, 1, "C", "D", ["n1"]),
        ]
        neighborhood = [
            report(
                record_index=10,
                sampled_position=0,
                prompt_type="neighborhood",
                margin=1,
                probability_gap=0.2,
            ),
            report(
                record_index=11,
                sampled_position=1,
                prompt_type="neighborhood",
                margin=1,
                probability_gap=0.7,
            ),
        ]
        selected, selected_records = MODULE.select_neighborhood_lm_head_rows(
            tokenizer,
            records,
            neighborhood,
            spe_target=50.0,
            row_selection="low_spe",
        )
        self.assertEqual(selected_records, [(10, 0)])
        self.assertEqual(set(selected), {ord(" "), ord("A"), ord("B")})
        self.assertNotIn(ord("C"), selected)
        self.assertNotIn(ord("D"), selected)

    def test_confidence_objective_protects_forget_margin_and_record_spe(self):
        forget_new = torch.tensor([0.5], requires_grad=True)
        forget_true = torch.tensor([1.0], requires_grad=True)
        required = torch.tensor([0.1])
        neighborhood_new = torch.tensor(
            [-math.log(0.20), -math.log(0.20), -math.log(0.10)],
            requires_grad=True,
        )
        neighborhood_true = torch.tensor(
            [-math.log(0.60), -math.log(0.40), -math.log(0.80)],
            requires_grad=True,
        )
        terms = MODULE.confidence_loss_terms(
            forget_new,
            forget_true,
            required,
            neighborhood_new,
            neighborhood_true,
            [[0, 1], [2]],
            spe_target_fraction=0.50,
            neighborhood_tail_floor=0.10,
            target_true_prob_floor=0.55,
        )
        # Record gaps are mean([.4,.2])=.3 and .7, then weighted equally.
        self.assertAlmostEqual(
            float(terms["global_spe_fraction"].detach()),
            0.5,
            places=6,
        )
        self.assertGreater(float(terms["forget_hinge"].detach()), 0.0)
        total = (
            terms["forget_hinge"]
            + terms["spe_global_hinge"]
            + terms["spe_tail_hinge"]
            + terms["true_probability_hinge"]
        )
        total.backward()
        self.assertIsNotNone(forget_new.grad)
        self.assertIsNotNone(neighborhood_true.grad)

    def test_hard_gates_require_zero_failures_target_spe_and_no_ppl_rise(self):
        passing_metrics = {
            "Eff": 0.0,
            "Gen": 0.0,
            "Spe": 50.1,
            "rewrite_failure_prompt_instances": 0,
            "paraphrase_failure_prompt_instances": 0,
        }
        accepted = MODULE.candidate_gates(
            passing_metrics,
            11.0,
            input_ppl=11.0,
            spe_target=50.0,
            max_ppl_increase=0.0,
            ppl_ceiling=None,
        )
        self.assertTrue(accepted["qualified"])

        for mutation, ppl in (
            ({"rewrite_failure_prompt_instances": 1}, 11.0),
            ({"Spe": 49.99}, 11.0),
            ({}, 11.01),
        ):
            metrics = {**passing_metrics, **mutation}
            rejected = MODULE.candidate_gates(
                metrics,
                ppl,
                input_ppl=11.0,
                spe_target=50.0,
                max_ppl_increase=0.0,
                ppl_ceiling=None,
            )
            self.assertFalse(rejected["qualified"])

    def test_candidate_selection_prefers_lower_ppl_after_hard_gates(self):
        common = {
            "Eff": 0.0,
            "Gen": 0.0,
            "selected_lm_head_delta_norm": 2.0,
        }
        gates = {"qualified": True}
        lower_ppl = MODULE.CandidateSnapshot(
            stage_target=50,
            delta_rows=torch.zeros(1, 1),
            metrics={**common, "Spe": 50.1},
            ppl=10.9,
            gates=gates,
        )
        higher_spe = MODULE.CandidateSnapshot(
            stage_target=60,
            delta_rows=torch.zeros(1, 1),
            metrics={**common, "Spe": 70.0},
            ppl=11.0,
            gates=gates,
        )
        self.assertIs(
            min([higher_spe, lower_ppl], key=MODULE.candidate_priority),
            lower_ppl,
        )

    def test_materialization_changes_only_selected_output_rows(self):
        output = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        baseline = output.index_select(0, torch.tensor([1, 3])).clone()
        unrelated_before = output.index_select(0, torch.tensor([0, 2, 4])).clone()
        input_embeddings = torch.randn(5, 4)
        input_before = input_embeddings.clone()
        MODULE.set_selected_lm_head_rows(
            output,
            [1, 3],
            baseline,
            torch.ones_like(baseline),
        )
        self.assertTrue(
            torch.equal(
                output.index_select(0, torch.tensor([0, 2, 4])),
                unrelated_before,
            )
        )
        self.assertTrue(torch.equal(output[1], baseline[0] + 1))
        self.assertTrue(torch.equal(output[3], baseline[1] + 1))
        self.assertTrue(torch.equal(input_embeddings, input_before))


if __name__ == "__main__":
    unittest.main()
