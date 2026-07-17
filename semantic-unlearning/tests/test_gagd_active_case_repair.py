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
from mcf_zero_unlearn_official_eval import (  # noqa: E402
    official_test_batch_prediction,
)


class TensorBatch(dict):
    def to(self, device):
        return TensorBatch(
            {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in self.items()
            }
        )


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


class LlamaStyleTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = None
    eos_token = "<eos>"

    @staticmethod
    def _encode(text, add_special_tokens):
        token_ids = [3 + (ord(character) % 100) for character in text]
        return ([1] + token_ids) if add_special_tokens else token_ids

    def __call__(
        self,
        text,
        add_special_tokens=True,
        padding=False,
        return_tensors=None,
        **kwargs,
    ):
        values = text if isinstance(text, list) else [text]
        rows = [
            self._encode(value, add_special_tokens=add_special_tokens)
            for value in values
        ]
        if padding:
            width = max(len(row) for row in rows)
            attention = [
                [1] * len(row) + [0] * (width - len(row)) for row in rows
            ]
            rows = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
        else:
            attention = [[1] * len(row) for row in rows]
        if return_tensors == "pt":
            return TensorBatch(
                {
                    "input_ids": torch.tensor(rows, dtype=torch.long),
                    "attention_mask": torch.tensor(attention, dtype=torch.long),
                }
            )
        return {"input_ids": rows if isinstance(text, list) else rows[0]}

    def decode(self, token_ids):
        return " ".join(str(int(token_id)) for token_id in token_ids)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=128, hidden_size=4):
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.output_embeddings = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(tie_word_embeddings=False, model_type="tiny")

    def get_input_embeddings(self):
        return self.input_embeddings

    def get_output_embeddings(self):
        return self.output_embeddings

    def set_output_embeddings(self, output_embeddings):
        self.output_embeddings = output_embeddings

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.input_embeddings(input_ids)
        return SimpleNamespace(
            logits=self.output_embeddings(hidden),
            hidden_states=(hidden,),
        )

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


def sampled_record(
    index,
    target_new="A",
    target_true="B",
    *,
    sampled_position=None,
    rewrite_prompt="P",
    paraphrases=(),
):
    raw_record = {
        "requested_rewrite": {
            "prompt": "{}",
            "subject": rewrite_prompt,
            "target_new": {"str": target_new},
            "target_true": {"str": target_true},
        },
        "paraphrase_prompts": list(paraphrases),
    }
    return MODULE.SampledMCFRecord(
        record_index=index,
        sampled_position=index if sampled_position is None else sampled_position,
        example=GAGD.Example(
            prompt=rewrite_prompt,
            answer=target_new,
            target_new=target_new,
            target_true=target_true,
            paraphrase_prompts=list(paraphrases),
            source="mcf",
        ),
        raw_record=raw_record,
        rewrite_prompt=rewrite_prompt,
        paraphrase_prompts=tuple(paraphrases),
        target_new=target_new,
        target_true=target_true,
    )


def prompt_instance(
    *,
    record_index=0,
    sampled_position=0,
    prompt_type="rewrite",
    prompt_index=0,
    prompt="P",
    target_new="A",
    target_true="B",
):
    return MODULE.MCFPromptInstance(
        record_index=record_index,
        sampled_position=sampled_position,
        prompt_type=prompt_type,
        prompt_index=prompt_index,
        prompt=prompt,
        target_new=target_new,
        target_true=target_true,
    )


def prompt_margin_report(instance, margin):
    return {
        "record_index": instance.record_index,
        "sampled_position": instance.sampled_position,
        "prompt_type": instance.prompt_type,
        "prompt_index": instance.prompt_index,
        "prompt": instance.prompt,
        "target_new": instance.target_new,
        "target_true": instance.target_true,
        "margin": float(margin),
    }


class FakeGradientDescent:
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


class GAGDActiveCaseRepairTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_margin_sign_is_target_new_minus_target_true(self):
        self.assertEqual(MODULE.margin_from_nll(5.0, 3.0), 2.0)
        self.assertEqual(MODULE.margin_from_nll(2.0, 3.0), -1.0)

    def test_official_batch_excludes_eos_and_removes_llama_target_bos(self):
        tokenizer = LlamaStyleTokenizer()
        encoded, targets, prefix_lens = MODULE.official_batch_components(
            tokenizer,
            [prompt_instance(prompt="Question", target_new="A", target_true="BC")],
            torch.device("cpu"),
            llama_like=True,
        )

        self.assertEqual(targets[0], tokenizer(" A")["input_ids"][1:])
        self.assertEqual(targets[1], tokenizer(" BC")["input_ids"][1:])
        self.assertNotIn(tokenizer.bos_token_id, targets[0])
        self.assertNotIn(tokenizer.eos_token_id, targets[0])
        self.assertEqual(encoded["input_ids"][0, 0].item(), tokenizer.bos_token_id)
        self.assertEqual(
            prefix_lens,
            [len(tokenizer(["Question"])["input_ids"][0]) - 1] * 2,
        )

    def test_local_nll_matches_official_test_batch_prediction(self):
        model = TinyCausalLM(vocab_size=128, hidden_size=7).to(
            dtype=torch.bfloat16
        )
        tokenizer = LlamaStyleTokenizer()
        instances = [
            prompt_instance(prompt="First prompt", target_new="NBC", target_true="CBS"),
            prompt_instance(
                record_index=1,
                sampled_position=1,
                prompt_type="paraphrase",
                prompt_index=1,
                prompt="Second prompt",
                target_new="midfielder",
                target_true="tackle",
            ),
        ]

        local_new, local_true = MODULE.official_prompt_instance_nll_tensors(
            model,
            tokenizer,
            instances,
            torch.device("cpu"),
            llama_like=True,
        )
        official = [
            official_test_batch_prediction(
                model,
                tokenizer,
                [instance.prompt],
                instance.target_new,
                instance.target_true,
                torch.device("cpu"),
                llama_like=True,
            )[0]
            for instance in instances
        ]

        expected_new = torch.tensor(
            [case["target_new"] for case in official],
            dtype=local_new.dtype,
        )
        expected_true = torch.tensor(
            [case["target_true"] for case in official],
            dtype=local_true.dtype,
        )
        self.assertTrue(torch.equal(local_new.cpu(), expected_new))
        self.assertTrue(torch.equal(local_true.cpu(), expected_true))

    def test_prompt_expansion_preserves_rewrite_and_all_paraphrase_metadata(self):
        raw_record = {
            "requested_rewrite": {
                "prompt": "{} aired on",
                "subject": "The Face Is Familiar",
                "target_new": {"str": "NBC"},
                "target_true": {"str": "CBS"},
            },
            "paraphrase_prompts": [
                "The original network for The Face Is Familiar was",
                "Which network aired The Face Is Familiar?",
            ],
        }
        record = MODULE._sampled_mcf_record(
            raw_record,
            record_index=91,
            sampled_position=6,
        )

        instances = MODULE.expand_prompt_instances([record])

        self.assertEqual(len(instances), 3)
        self.assertIs(record.raw_record, raw_record)
        self.assertEqual(
            [
                (case.prompt_type, case.prompt_index, case.prompt)
                for case in instances
            ],
            [
                ("rewrite", 0, "The Face Is Familiar aired on"),
                (
                    "paraphrase",
                    0,
                    "The original network for The Face Is Familiar was",
                ),
                (
                    "paraphrase",
                    1,
                    "Which network aired The Face Is Familiar?",
                ),
            ],
        )
        self.assertTrue(all(case.record_index == 91 for case in instances))
        self.assertTrue(all(case.sampled_position == 6 for case in instances))

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

    def test_required_margins_protect_every_initially_passing_prompt(self):
        original = torch.tensor([-0.5, 0.1, 0.4])

        required = MODULE.build_required_margin_tensor(
            original,
            active_margin=0.1,
            protected_margin_floor=0.25,
        )

        self.assertTrue(
            torch.equal(required, torch.tensor([0.1, 0.25, 0.25]))
        )
        reports = [
            prompt_margin_report(
                prompt_instance(record_index=position),
                margin,
            )
            for position, margin in enumerate(original.tolist())
        ]
        MODULE.attach_margin_requirement_metadata(
            reports,
            original.tolist(),
            required,
            active_margin=0.1,
        )
        self.assertTrue(reports[0]["initially_active"])
        self.assertTrue(reports[1]["initially_protected"])
        self.assertEqual(reports[2]["original_margin"], original[2].item())
        self.assertEqual(reports[2]["required_margin"], 0.25)

    def test_minimal_rows_are_selected_only_from_initial_failures(self):
        tokenizer = TinyTokenizer()
        failing = prompt_instance(
            record_index=1,
            target_new="A",
            target_true="B",
        )
        passing = prompt_instance(
            record_index=2,
            target_new="C",
            target_true="D",
        )
        all_instances = [failing, passing]
        active_instances = MODULE._active_instances(all_instances, [0])
        groups = GAGD.PostTrainingTokenGroups(
            target_new=[ord("A"), ord("C")],
            target_true=[ord("B"), ord("D")],
            retain=[],
            unique_target_new=[ord("A"), ord("C")],
            unique_target_true=[ord("B"), ord("D")],
            target_new_true_overlap=[],
            target_new_retain_overlap=[],
            target_new_true_retain_overlap=[],
            target_true_retain_overlap=[],
            overlap=[],
        )

        selected = MODULE.selected_rows_for_active_instances(
            tokenizer,
            active_instances,
            groups,
            "minimal_optimize",
        )

        self.assertIn(ord("A"), selected)
        self.assertIn(ord("B"), selected)
        self.assertNotIn(ord("C"), selected)
        self.assertNotIn(ord("D"), selected)

    def test_three_known_5e_failure_prompt_instances_are_active(self):
        instances = [
            prompt_instance(
                record_index=17072,
                sampled_position=6,
                prompt_type="rewrite",
                prompt_index=0,
                prompt="The Face Is Familiar was originally aired on",
                target_new="NBC",
                target_true="CBS",
            ),
            prompt_instance(
                record_index=11991,
                sampled_position=16,
                prompt_type="paraphrase",
                prompt_index=1,
                prompt=(
                    "Celtic won 3-2 and avoided relegation. Jerry Sisemore, "
                    "who plays the position"
                ),
                target_new="midfielder",
                target_true="tackle",
            ),
            prompt_instance(
                record_index=19609,
                sampled_position=27,
                prompt_type="paraphrase",
                prompt_index=1,
                prompt="March 2006. Empty Nest was released on",
                target_new="CBS",
                target_true="NBC",
            ),
        ]
        empty_groups = GAGD.PostTrainingTokenGroups(
            target_new=[],
            target_true=[],
            retain=[],
            unique_target_new=[],
            unique_target_true=[],
            target_new_true_overlap=[],
            target_new_retain_overlap=[],
            target_new_true_retain_overlap=[],
            target_true_retain_overlap=[],
            overlap=[],
        )
        with mock.patch.object(
            MODULE,
            "official_prompt_instance_nll_tensors",
            return_value=(
                torch.tensor([2.0, 1.0, 1.0]),
                torch.tensor([2.5625, 5.75, 7.734375]),
            ),
        ):
            reports = MODULE.evaluate_prompt_instance_margin_reports(
                TinyCausalLM(),
                TinyTokenizer(),
                instances,
                empty_groups,
                active_margin=0.1,
                device=torch.device("cpu"),
                batch_size=3,
                llama_like=False,
            )

        active = MODULE.select_active_positions(reports, active_margin=0.1)
        payload = MODULE.active_report_payload(reports, active_margin=0.1)

        self.assertEqual(active, [0, 1, 2])
        self.assertEqual(
            [report["official_compatible_margin"] for report in reports],
            [-0.5625, -4.75, -6.734375],
        )
        self.assertEqual(payload["active_prompt_count"], 3)
        self.assertEqual(payload["active_parent_record_count"], 3)
        self.assertEqual(
            [
                (case["sampled_position"], case["prompt_type"], case["prompt_index"])
                for case in payload["cases"]
            ],
            [(6, "rewrite", 0), (16, "paraphrase", 1), (27, "paraphrase", 1)],
        )

    def test_known_failure_set_cannot_be_replaced_by_three_other_prompts(self):
        known_failures = [
            prompt_instance(
                record_index=17072,
                sampled_position=6,
                prompt_type="rewrite",
                prompt_index=0,
                prompt="The Face Is Familiar was originally aired on",
            ),
            prompt_instance(
                record_index=11991,
                sampled_position=16,
                prompt_type="paraphrase",
                prompt_index=1,
                prompt=(
                    "Celtic won 3-2 and avoided relegation. Jerry Sisemore, "
                    "who plays the position"
                ),
            ),
            prompt_instance(
                record_index=19609,
                sampled_position=27,
                prompt_type="paraphrase",
                prompt_index=1,
                prompt="March 2006. Empty Nest was released on",
            ),
        ]
        replacement_failures = [
            prompt_instance(
                record_index=30000 + position,
                sampled_position=40 + position,
                prompt_type="paraphrase",
                prompt_index=position,
                prompt=f"Initially passing prompt {position}",
            )
            for position in range(3)
        ]
        instances = known_failures + replacement_failures
        before_reports = [
            prompt_margin_report(instance, -1.0 if position < 3 else 0.2)
            for position, instance in enumerate(instances)
        ]
        after_reports = [
            prompt_margin_report(instance, 0.2 if position < 3 else -1.0)
            for position, instance in enumerate(instances)
        ]

        transitions = MODULE.prompt_margin_transitions(
            before_reports,
            after_reports,
            active_margin=0.1,
        )

        self.assertEqual(transitions["fixed_original_positions"], [0, 1, 2])
        self.assertEqual(transitions["newly_activated_positions"], [3, 4, 5])
        summary_fields = MODULE.protection_summary_fields(
            transitions,
            torch.full((6,), 0.1),
            max_delta_norm=0.5,
        )
        self.assertEqual(summary_fields["originally_active_prompt_instances"], 3)
        self.assertEqual(
            summary_fields["originally_protected_prompt_instances"],
            3,
        )
        self.assertEqual(
            summary_fields["newly_activated_prompt_instances_after"],
            3,
        )
        self.assertEqual(summary_fields["fixed_original_prompt_instances_after"], 3)
        self.assertAlmostEqual(summary_fields["required_margin_min"], 0.1)
        self.assertEqual(summary_fields["max_delta_norm"], 0.5)
        with self.assertRaisesRegex(
            RuntimeError,
            "initially passing rewrite/paraphrase",
        ):
            MODULE.raise_if_new_prompt_failures(
                transitions,
                after_reports,
            )

    def test_margin_report_contains_required_case_and_token_group_fields(self):
        model = TinyCausalLM(vocab_size=128, hidden_size=6)
        tokenizer = LlamaStyleTokenizer()
        new_ids = GAGD.token_ids_for_text(tokenizer, GAGD.normalize_answer("A"))
        true_ids = GAGD.token_ids_for_text(tokenizer, GAGD.normalize_answer("B"))
        groups = GAGD.PostTrainingTokenGroups(
            target_new=new_ids,
            target_true=true_ids,
            retain=[],
            unique_target_new=new_ids,
            unique_target_true=true_ids,
            target_new_true_overlap=[],
            target_new_retain_overlap=[],
            target_new_true_retain_overlap=[],
            target_true_retain_overlap=[],
            overlap=[],
        )

        instance = prompt_instance(
            record_index=7,
            sampled_position=4,
            prompt_type="paraphrase",
            prompt_index=1,
            prompt="Exact formatted paraphrase",
        )
        reports = MODULE.evaluate_prompt_instance_margin_reports(
            model,
            tokenizer,
            [instance],
            groups,
            active_margin=0.1,
            device=torch.device("cpu"),
            batch_size=1,
            llama_like=True,
        )

        report = reports[0]
        for key in (
            "record_index",
            "sampled_position",
            "prompt_type",
            "prompt_index",
            "prompt",
            "target_new",
            "target_true",
            "target_new_nll",
            "target_true_nll",
            "margin",
            "official_compatible_margin",
        ):
            self.assertIn(key, report)
        self.assertEqual(report["record_index"], 7)
        self.assertEqual(report["sampled_position"], 4)
        self.assertEqual(report["prompt_type"], "paraphrase")
        self.assertEqual(report["prompt_index"], 1)
        self.assertEqual(report["prompt"], "Exact formatted paraphrase")
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
        selected = MODULE.selected_rows_for_active_instances(
            TinyTokenizer(),
            [prompt_instance(target_true="B")],
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

        loss = MODULE.squared_hinge_loss(
            margins,
            required_margins=torch.tensor([0.1]),
        )
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(margins.grad.item(), 0.0)

    def test_all_forget_prompts_are_present_in_margin_objective(self):
        current_margins = torch.tensor([-0.2, 0.05, 0.3])
        required_margins = torch.tensor([0.1, 0.1, 0.1])

        loss = MODULE.squared_hinge_loss(
            current_margins,
            required_margins,
        )

        expected = (0.1 - (-0.2)) ** 2 + (0.1 - 0.05) ** 2
        self.assertAlmostEqual(loss.item(), expected, places=6)

        delta = MODULE.SelectedRowDelta(
            1,
            1,
            device=torch.device("cpu"),
        )
        incomplete_margin_fn = lambda rows: rows[0, 0].repeat(2)
        with self.assertRaisesRegex(ValueError, "cover every prompt instance"):
            MODULE.optimize_selected_delta(
                delta,
                incomplete_margin_fn,
                lambda rows: rows.new_zeros(()),
                required_margins=required_margins,
                repair_steps=1,
                repair_lr=0.1,
                repair_optimizer="sgd",
                hinge_weight=1.0,
                delta_l2_lambda=0.0,
                retain_kl_mu=0.0,
                stop_when_all_satisfied=True,
            )

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
            required_margins=torch.tensor([0.1]),
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
        delta = MODULE.SelectedRowDelta(
            1,
            1,
            device=torch.device("cpu"),
        )
        margin_fn = lambda rows: rows[:, 0]
        kl_fn = lambda rows: rows.new_zeros(())
        fake_optimizer = FakeGradientDescent(
            delta.parameters(),
            learning_rate=1.0,
        )

        with mock.patch.object(
            MODULE,
            "make_repair_optimizer",
            return_value=fake_optimizer,
        ):
            logs, summary = MODULE.optimize_selected_delta(
                delta,
                margin_fn,
                kl_fn,
                required_margins=torch.tensor([0.1]),
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

    def test_fixing_one_failure_cannot_create_another_prompt_failure(self):
        delta = MODULE.SelectedRowDelta(
            1,
            1,
            device=torch.device("cpu"),
        )
        margin_fn = lambda rows: torch.stack(
            (rows[0, 0], rows.new_tensor(0.2) - rows[0, 0])
        )
        required = torch.tensor([0.1, 0.1])
        kl_fn = lambda rows: rows.new_zeros(())
        fake_optimizer = FakeGradientDescent(
            delta.parameters(),
            learning_rate=0.5,
        )
        with mock.patch.object(
            MODULE,
            "make_repair_optimizer",
            return_value=fake_optimizer,
        ):
            logs, summary = MODULE.optimize_selected_delta(
                delta,
                margin_fn,
                kl_fn,
                required_margins=required,
                repair_steps=5,
                repair_lr=0.5,
                repair_optimizer="sgd",
                hinge_weight=1.0,
                delta_l2_lambda=0.0,
                retain_kl_mu=0.0,
                stop_when_all_satisfied=True,
            )
        final_margins = margin_fn(delta.effective_delta()).detach()

        self.assertEqual(summary["training_prompt_instances"], 2)
        self.assertTrue(summary["all_satisfied"])
        self.assertTrue(torch.all(final_margins >= required))
        self.assertGreaterEqual(final_margins[1].item(), 0.1)
        self.assertTrue(logs[-1]["all_training_prompt_instances_satisfied"])

    def test_max_delta_norm_projects_after_every_optimizer_step(self):
        delta = MODULE.SelectedRowDelta(
            1,
            1,
            device=torch.device("cpu"),
        )
        margin_fn = lambda rows: rows[:, 0]
        kl_fn = lambda rows: rows.new_zeros(())
        fake_optimizer = FakeGradientDescent(
            delta.parameters(),
            learning_rate=1.0,
        )
        with mock.patch.object(
            MODULE,
            "make_repair_optimizer",
            return_value=fake_optimizer,
        ):
            logs, summary = MODULE.optimize_selected_delta(
                delta,
                margin_fn,
                kl_fn,
                required_margins=torch.tensor([10.0]),
                repair_steps=1,
                repair_lr=1.0,
                repair_optimizer="sgd",
                hinge_weight=1.0,
                delta_l2_lambda=0.0,
                retain_kl_mu=0.0,
                stop_when_all_satisfied=False,
                max_delta_norm=0.25,
            )

        self.assertTrue(logs[0]["delta_norm_projected"])
        self.assertEqual(summary["delta_norm_projection_steps"], 1)
        self.assertLessEqual(
            delta.effective_delta().norm().item(),
            0.2500001,
        )

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

    def test_minimal_optimize_cache_uses_official_prompt_instance_positions(self):
        model = TinyCausalLM(vocab_size=128, hidden_size=5)
        tokenizer = LlamaStyleTokenizer()
        instances = [
            prompt_instance(
                prompt_type="paraphrase",
                prompt_index=1,
                prompt="Jerry Sisemore played as",
                target_new="midfielder",
                target_true="tackle",
            )
        ]
        selected_ids = sorted(
            set(
                GAGD.token_ids_for_text(
                    tokenizer,
                    GAGD.normalize_answer("midfielder"),
                )
                + GAGD.token_ids_for_text(
                    tokenizer,
                    GAGD.normalize_answer("tackle"),
                )
            )
        )

        caches = MODULE.build_prompt_instance_delta_caches(
            model,
            tokenizer,
            instances,
            selected_ids,
            torch.device("cpu"),
            batch_size=1,
            llama_like=True,
        )
        cached_margin = MODULE.margins_from_delta_caches(
            caches,
            torch.zeros((len(selected_ids), 5)),
        )
        target_new_nll, target_true_nll = (
            MODULE.official_prompt_instance_nll_tensors(
                model,
                tokenizer,
                instances,
                torch.device("cpu"),
                llama_like=True,
            )
        )

        self.assertTrue(
            torch.allclose(
                cached_margin,
                target_new_nll - target_true_nll,
                atol=1e-6,
            )
        )

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

    def test_protected_margin_and_delta_norm_cli_defaults_and_validation(self):
        args = MODULE.build_parser().parse_args(
            [
                "--model-path",
                "input",
                "--base-model-path",
                "base",
                "--output-dir",
                "output",
                "--mcf-cache-path",
                "mcf.json",
                "--repair-mode",
                "minimal_optimize",
            ]
        )

        self.assertEqual(args.protected_margin_floor, 0.0)
        self.assertIsNone(args.max_delta_norm)
        MODULE.validate_args(args)
        args.max_delta_norm = -0.1
        with self.assertRaisesRegex(ValueError, "max-delta-norm"):
            MODULE.validate_args(args)

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

    def test_candidate_priority_uses_prompt_parent_margin_then_delta(self):
        candidates = [
            {
                "name": "more_prompts",
                "active_prompt_instances_after": 2,
                "active_parent_records_after": 1,
                "minimum_official_compatible_margin_after": -9.0,
                "selected_lm_head_delta_norm": 0.1,
            },
            {
                "name": "more_parents",
                "active_prompt_instances_after": 1,
                "active_parent_records_after": 2,
                "minimum_official_compatible_margin_after": 2.0,
                "selected_lm_head_delta_norm": 0.1,
            },
            {
                "name": "lower_margin",
                "active_prompt_instances_after": 1,
                "active_parent_records_after": 1,
                "minimum_official_compatible_margin_after": 0.2,
                "selected_lm_head_delta_norm": 0.1,
            },
            {
                "name": "larger_delta",
                "active_prompt_instances_after": 1,
                "active_parent_records_after": 1,
                "minimum_official_compatible_margin_after": 0.3,
                "selected_lm_head_delta_norm": 0.5,
            },
            {
                "name": "best",
                "active_prompt_instances_after": 1,
                "active_parent_records_after": 1,
                "minimum_official_compatible_margin_after": 0.3,
                "selected_lm_head_delta_norm": 0.2,
            },
        ]

        ranked = sorted(candidates, key=MODULE.candidate_priority)

        self.assertEqual(
            [candidate["name"] for candidate in ranked],
            ["best", "larger_delta", "lower_margin", "more_parents", "more_prompts"],
        )

    def test_official_failures_cannot_accept_zero_row_noop_candidate(self):
        with self.assertRaisesRegex(RuntimeError, "zero-row/no-op"):
            MODULE.guard_official_failures_against_zero_active_noop(
                0,
                {"forget": {"Eff": 2, "Gen": 2}},
            )

        MODULE.guard_official_failures_against_zero_active_noop(
            3,
            {"forget": {"Eff": 2, "Gen": 2}},
        )
        MODULE.guard_official_failures_against_zero_active_noop(
            0,
            {"forget": {"Eff": 0, "Gen": 0}},
        )


if __name__ == "__main__":
    unittest.main()
