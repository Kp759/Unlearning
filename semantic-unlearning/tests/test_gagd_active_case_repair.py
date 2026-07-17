import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import gagd_active_case_repair as MODULE  # noqa: E402
import gagd_compare as GAGD  # noqa: E402


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    bos_token_id = None
    unk_token_id = None
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False, **kwargs):
        if isinstance(text, list):
            return {
                "input_ids": [[ord(character) for character in value] for value in text]
            }
        if text.endswith(self.eos_token):
            text = text[: -len(self.eos_token)]
            return {"input_ids": [ord(character) for character in text] + [99]}
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, token_ids):
        return "".join(chr(int(token_id)) for token_id in token_ids)

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "tokenizer_stub.json").write_text(
            json.dumps({"kind": "tiny"}) + "\n",
            encoding="utf-8",
        )


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=128, hidden_size=4):
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.output_embeddings = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(tie_word_embeddings=False)

    def get_input_embeddings(self):
        return self.input_embeddings

    def get_output_embeddings(self):
        return self.output_embeddings

    def set_output_embeddings(self, output_embeddings):
        self.output_embeddings = output_embeddings

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.input_embeddings(input_ids)
        return SimpleNamespace(logits=self.output_embeddings(hidden))

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "vocab_size": self.input_embeddings.num_embeddings,
                "hidden_size": self.input_embeddings.embedding_dim,
            },
            output_dir / "tiny_model.pt",
        )

    @classmethod
    def from_pretrained(cls, output_dir):
        payload = torch.load(
            Path(output_dir) / "tiny_model.pt",
            map_location="cpu",
            weights_only=True,
        )
        model = cls(payload["vocab_size"], payload["hidden_size"])
        model.load_state_dict(payload["state_dict"])
        return model


class TinyTiedCausalLM(TinyCausalLM):
    def __init__(self, vocab_size=128, hidden_size=4):
        super().__init__(vocab_size=vocab_size, hidden_size=hidden_size)
        self.output_embeddings.weight = self.input_embeddings.weight
        self.config.tie_word_embeddings = True


def sampled_record(index, target_new="A", target_true="B"):
    return MODULE.SampledMCFRecord(
        record_index=index,
        sampled_position=index,
        example=GAGD.Example(
            prompt="P",
            answer=target_new,
            target_new=target_new,
            target_true=target_true,
            source="mcf",
        ),
    )


class GAGDActiveCaseRepairTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_margin_sign_is_target_new_minus_target_true(self):
        self.assertEqual(MODULE.margin_from_nll(5.0, 3.0), 2.0)
        self.assertEqual(MODULE.margin_from_nll(2.0, 3.0), -1.0)

    def test_rewrite_batch_excludes_eos(self):
        batch = MODULE.build_target_batch(
            TinyTokenizer(),
            [sampled_record(0)],
            "target_new",
            torch.device("cpu"),
        )

        self.assertEqual(
            batch["input_ids"].tolist(),
            [[ord("P"), ord("A")]],
        )
        self.assertNotIn(99, batch["input_ids"].tolist()[0])

    def test_active_set_uses_strict_below_margin(self):
        reports = [
            {"margin": -0.2},
            {"margin": 0.099},
            {"margin": 0.1},
            {"margin": 0.5},
        ]

        self.assertEqual(
            MODULE.select_active_positions(reports, active_margin=0.1),
            [0, 1],
        )

    def test_margin_report_contains_required_case_and_token_group_fields(self):
        model = TinyCausalLM()
        tokenizer = TinyTokenizer()
        groups = GAGD.PostTrainingTokenGroups(
            target_new=[ord("A")],
            target_true=[ord("B")],
            retain=[],
            unique_target_new=[ord("A")],
            unique_target_true=[ord("B")],
            target_new_true_overlap=[],
            target_new_retain_overlap=[],
            target_new_true_retain_overlap=[],
            target_true_retain_overlap=[],
            overlap=[],
        )

        reports = MODULE.evaluate_rewrite_margin_reports(
            model,
            tokenizer,
            [sampled_record(7)],
            groups,
            active_margin=0.1,
            device=torch.device("cpu"),
            batch_size=1,
        )

        report = reports[0]
        for key in (
            "record_index",
            "prompt",
            "target_new",
            "target_true",
            "target_new_nll",
            "target_true_nll",
            "margin",
        ):
            self.assertIn(key, report)
        self.assertEqual(report["record_index"], 7)
        self.assertIn(
            "unique_target_new",
            report["target_tokens"]["target_new"][0]["groups"],
        )
        self.assertIn(
            "unique_target_true",
            report["target_tokens"]["target_true"][0]["groups"],
        )

    def test_active_only_true_scaling_preserves_input_and_other_output_rows(self):
        model = TinyCausalLM()
        output = MODULE.freeze_model_for_output_repair(model)
        input_before = model.get_input_embeddings().weight.detach().clone()
        output_before = output.weight.detach().clone()
        groups = GAGD.PostTrainingTokenGroups(
            target_new=[ord("A")],
            target_true=[ord("B"), ord("C")],
            retain=[],
            unique_target_new=[ord("A")],
            unique_target_true=[ord("B"), ord("C")],
            target_new_true_overlap=[],
            target_new_retain_overlap=[],
            target_new_true_retain_overlap=[],
            target_true_retain_overlap=[],
            overlap=[],
        )
        selected = MODULE.selected_rows_for_active_records(
            TinyTokenizer(),
            [sampled_record(0, target_true="B")],
            groups,
            "true_scale",
        )
        base_rows = torch.full((1, output.weight.shape[1]), 2.0)

        MODULE.apply_active_true_scale(
            output.weight,
            selected,
            base_rows,
            target_true_scale=1.5,
        )

        self.assertEqual(selected, [ord("B")])
        self.assertTrue(torch.equal(model.get_input_embeddings().weight, input_before))
        self.assertTrue(
            torch.equal(
                output.weight[ord("B")],
                torch.full_like(output.weight[ord("B")], 3.0),
            )
        )
        for row in (0, ord("A"), ord("C"), 127):
            self.assertTrue(torch.equal(output.weight[row], output_before[row]))

    def test_tied_lm_head_is_cloned_before_output_only_repair(self):
        model = TinyTiedCausalLM(vocab_size=12, hidden_size=3)
        input_before = model.get_input_embeddings().weight.detach().clone()

        output = MODULE.freeze_model_for_output_repair(model)
        with torch.no_grad():
            output.weight[2].add_(1.0)

        self.assertNotEqual(
            output.weight.data_ptr(),
            model.get_input_embeddings().weight.data_ptr(),
        )
        self.assertFalse(model.config.tie_word_embeddings)
        self.assertTrue(torch.equal(model.get_input_embeddings().weight, input_before))
        self.assertFalse(model.get_input_embeddings().weight.requires_grad)

    def test_gamma_extrapolation_supports_values_greater_than_one(self):
        output_weight = torch.zeros((5, 3))
        output_before = output_weight.clone()
        base_rows = torch.full((1, 3), 2.0)
        checkpoint_rows = torch.full((1, 3), 4.0)

        MODULE.apply_gamma_extrapolation(
            output_weight,
            [2],
            base_rows,
            checkpoint_rows,
            gamma=1.5,
        )

        self.assertTrue(torch.equal(output_weight[2], torch.full((3,), 5.0)))
        for row in (0, 1, 3, 4):
            self.assertTrue(torch.equal(output_weight[row], output_before[row]))

    def test_materialized_minimal_delta_changes_only_selected_output_rows(self):
        model = TinyCausalLM(vocab_size=12, hidden_size=3)
        output = MODULE.freeze_model_for_output_repair(model)
        input_before = model.get_input_embeddings().weight.detach().clone()
        output_before = output.weight.detach().clone()

        MODULE.materialize_selected_delta(
            output.weight,
            [2, 7],
            torch.ones((2, 3)),
        )

        self.assertTrue(torch.equal(model.get_input_embeddings().weight, input_before))
        for row in range(12):
            expected = output_before[row] + 1 if row in {2, 7} else output_before[row]
            self.assertTrue(torch.equal(output.weight[row], expected))

    def test_squared_hinge_has_zero_gradient_after_safety_margin(self):
        margins = torch.tensor([0.2], requires_grad=True)

        loss = MODULE.squared_hinge_loss(margins, active_margin=0.1)
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(margins.grad.item(), 0.0)

    def test_optimizer_stops_before_any_step_when_all_margins_satisfied(self):
        delta = MODULE.SelectedRowDelta(
            1,
            1,
            device=torch.device("cpu"),
        )
        margin_fn = lambda rows: rows[:, 0] + 0.2
        kl_fn = lambda rows: rows.new_zeros(())

        logs, summary = MODULE.optimize_selected_delta(
            delta,
            margin_fn,
            kl_fn,
            active_margin=0.1,
            repair_steps=10,
            repair_lr=0.1,
            repair_optimizer="sgd",
            hinge_weight=1.0,
            delta_l2_lambda=0.0,
            retain_kl_mu=0.0,
            stop_when_all_satisfied=True,
        )

        self.assertEqual(logs, [])
        self.assertEqual(summary["steps_completed"], 0)
        self.assertTrue(summary["stopped_early"])
        self.assertTrue(summary["all_satisfied"])

    def test_optimizer_stops_immediately_after_crossing_margin(self):
        class FakeOptimizer:
            def __init__(self, parameters, learning_rate):
                self.parameters = list(parameters)
                self.learning_rate = learning_rate

            def zero_grad(self, set_to_none=True):
                for parameter in self.parameters:
                    parameter.grad = None

            def step(self):
                with torch.no_grad():
                    for parameter in self.parameters:
                        parameter.add_(parameter.grad, alpha=-self.learning_rate)

        delta = MODULE.SelectedRowDelta(
            1,
            1,
            device=torch.device("cpu"),
        )
        margin_fn = lambda rows: rows[:, 0]
        kl_fn = lambda rows: rows.new_zeros(())
        fake_optimizer = FakeOptimizer(delta.parameters(), learning_rate=1.0)

        with mock.patch.object(
            MODULE,
            "make_repair_optimizer",
            return_value=fake_optimizer,
        ):
            logs, summary = MODULE.optimize_selected_delta(
                delta,
                margin_fn,
                kl_fn,
                active_margin=0.1,
                repair_steps=10,
                repair_lr=1.0,
                repair_optimizer="sgd",
                hinge_weight=1.0,
                delta_l2_lambda=0.0,
                retain_kl_mu=0.0,
                stop_when_all_satisfied=True,
            )

        self.assertEqual(summary["steps_completed"], 1)
        self.assertTrue(summary["stopped_early"])
        self.assertTrue(summary["all_satisfied"])
        self.assertEqual(len(logs), 1)
        self.assertGreaterEqual(logs[0]["minimum_margin_after_step"], 0.1)

    def test_sparse_delta_cache_matches_direct_softmax_nll(self):
        base_logits = torch.tensor([0.2, -0.1, 0.4])
        base_log_probs = torch.log_softmax(base_logits, dim=-1)
        selected_ids = torch.tensor([0, 2])
        cache = MODULE.AnswerDeltaCache(
            base_token_nll=torch.tensor([-base_log_probs[2]]),
            hidden=torch.tensor([[1.0]]),
            selected_probs=base_log_probs[selected_ids].exp().unsqueeze(0),
            target_selected_columns=torch.tensor([1]),
        )
        delta_rows = torch.tensor([[0.3], [-0.2]])

        cached_nll = MODULE.answer_nll_from_delta_cache(cache, delta_rows)
        repaired_logits = base_logits.clone()
        repaired_logits[selected_ids] += delta_rows[:, 0]
        direct_nll = -torch.log_softmax(repaired_logits, dim=-1)[2]

        self.assertTrue(torch.allclose(cached_nll, direct_nll, atol=1e-6))

    def test_retain_projection_removes_preserved_direction(self):
        retained_basis = torch.tensor([[1.0, 0.0, 0.0]])
        rows = torch.tensor([[2.0, 3.0, 4.0]])

        projected = MODULE.project_rows_away(rows, retained_basis)

        self.assertTrue(
            torch.allclose(
                projected @ retained_basis.transpose(0, 1),
                torch.zeros((1, 1)),
            )
        )
        self.assertTrue(torch.equal(projected, torch.tensor([[0.0, 3.0, 4.0]])))

    def test_retain_calibration_sampling_is_deterministic(self):
        records = [sampled_record(index) for index in range(20)]

        first = MODULE.sample_retain_calibration(records, 6, seed=11)
        second = MODULE.sample_retain_calibration(records, 6, seed=11)
        third = MODULE.sample_retain_calibration(records, 6, seed=12)

        first_indices = [record.record_index for record in first]
        self.assertEqual(
            first_indices,
            [record.record_index for record in second],
        )
        self.assertNotEqual(
            first_indices,
            [record.record_index for record in third],
        )

    def test_checkpoint_save_and_reload_preserves_repaired_weights(self):
        model = TinyCausalLM(vocab_size=16, hidden_size=4)
        tokenizer = TinyTokenizer()
        with torch.no_grad():
            model.get_output_embeddings().weight[3].add_(2.5)
        input_ids = torch.tensor([[1, 2, 3]])
        expected = model(input_ids).logits.detach()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint"
            MODULE.save_repair_checkpoint(
                model,
                tokenizer,
                checkpoint,
                repair_config={
                    "preserved_5e_overlap_alphas": {
                        "target_new_true": 1.0,
                        "target_new_retain": 1.0,
                        "target_new_true_retain": 1.0,
                    }
                },
            )
            reloaded = TinyCausalLM.from_pretrained(checkpoint)
            actual = reloaded(input_ids).logits.detach()

            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue((checkpoint / "repair_experiment_config.json").exists())

    def test_config_recovery_requires_and_preserves_alpha_tuple(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "mode" / "checkpoint"
            checkpoint.mkdir(parents=True)
            config = {
                "post_training_new_true_alpha": 1.0,
                "post_training_new_retain_alpha": 0.9,
                "post_training_new_true_retain_alpha": 0.8,
            }
            (root / "config_used.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )

            config_path, recovered, alphas = MODULE.recover_experiment_config(
                str(checkpoint),
                explicit_path=None,
            )

            self.assertEqual(config_path, (root / "config_used.json").resolve())
            self.assertEqual(recovered, config)
            self.assertEqual(
                alphas,
                {
                    "target_new_true": 1.0,
                    "target_new_retain": 0.9,
                    "target_new_true_retain": 0.8,
                },
            )

    def test_config_recovery_does_not_assume_missing_alpha_tuple(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint"
            checkpoint.mkdir()

            with self.assertRaises(FileNotFoundError):
                MODULE.recover_experiment_config(
                    str(checkpoint),
                    explicit_path=None,
                )

    def test_source_config_sampling_mismatch_is_rejected(self):
        source_config = {
            "dataset": "mcf",
            "seed": 0,
            "forget_num": 50,
            "retain_num": 1000,
            "mcf_sample_mode": "official",
        }
        args = SimpleNamespace(
            seed=1,
            forget_num=50,
            retain_num=1000,
            sample_mode="official",
        )

        with self.assertRaises(ValueError):
            MODULE.validate_source_experiment_config(source_config, args)


if __name__ == "__main__":
    unittest.main()
