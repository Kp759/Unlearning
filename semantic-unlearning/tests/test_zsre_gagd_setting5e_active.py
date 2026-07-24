import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import aggregate_zsre_gagd_results as AGG  # noqa: E402
import zsre_gagd_setting5e_active_repair as PIPELINE  # noqa: E402
import zsre_zero_unlearn_official_eval as EVAL  # noqa: E402


class TensorBatch(dict):
    def to(self, device):
        return TensorBatch(
            {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in self.items()
            }
        )


class TinyTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 99
    bos_token_id = None

    @staticmethod
    def _encode(text):
        if text == "<eos>":
            return [99]
        return [2 + (ord(character) % 90) for character in text]

    def __call__(self, text, padding=False, return_tensors=None, **kwargs):
        values = text if isinstance(text, list) else [text]
        rows = [self._encode(value) for value in values]
        attention = [[1] * len(row) for row in rows]
        if padding:
            width = max(len(row) for row in rows)
            attention = [
                mask + [0] * (width - len(mask))
                for mask in attention
            ]
            rows = [
                row + [self.pad_token_id] * (width - len(row))
                for row in rows
            ]
        if return_tensors == "pt":
            return TensorBatch(
                {
                    "input_ids": torch.tensor(rows, dtype=torch.long),
                    "attention_mask": torch.tensor(attention, dtype=torch.long),
                }
            )
        return {"input_ids": rows if isinstance(text, list) else rows[0]}

    def decode(self, token_ids):
        return "".join(chr((int(token_id) - 2) % 90) for token_id in token_ids)


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


def raw_record(index):
    subject = f"Subject{index}"
    return {
        "src": f"Where is {subject}",
        "subject": subject,
        "answers": [f"Answer{index}"],
        "rephrase": f"{subject} is located where",
        "loc": f"nq question: unrelated {index}",
        "loc_ans": f"Place{index}",
    }


def adapted_record(case_id=7):
    return {
        "case_id": case_id,
        "requested_rewrite": {
            "prompt": "Where is {}",
            "subject": "Alice",
            "target_new": {"str": "<|endoftext|>"},
            "target_true": {"str": "AB"},
        },
        "paraphrase_prompts": ["Alice lives where"],
        "neighborhood_prompts": [
            {"prompt": "nq question: unrelated?", "target": "X"}
        ],
    }


def token_cache(
    *,
    hidden,
    target_logit=1.0,
    neutral_logit=0.0,
    prompt_type="rewrite",
):
    case = EVAL.PredictionCase(
        case_id=1,
        prompt_type=prompt_type,
        prompt_index=0,
        token_index=0,
        prompt="P",
        target_text="A",
    )
    return PIPELINE.TokenLogitCache(
        case=case,
        hidden=torch.tensor(hidden, dtype=torch.float32),
        target_token_id=3,
        predicted_token_id=3,
        target_logit=torch.tensor(target_logit),
        neutral_logit=torch.tensor(neutral_logit),
        correct=True,
    )


class ZsREOfficialEvalTests(unittest.TestCase):
    def test_official_sampling_uses_second_half_then_first_half(self):
        raw = [raw_record(index) for index in range(12)]
        forget, retain = EVAL.sample_official_zsre_raw_records(
            raw,
            forget_num=3,
            retain_num=4,
            seed=5,
            strict=True,
        )
        rng = random.Random(5)
        expected_forget = rng.sample(list(enumerate(raw))[6:], k=3)
        expected_retain = rng.sample(list(enumerate(raw))[:6], k=4)
        self.assertEqual([index for index, _ in forget], [x[0] for x in expected_forget])
        self.assertEqual([index for index, _ in retain], [x[0] for x in expected_retain])

    def test_adapter_preserves_upstream_zsre_direction(self):
        tokenizer = TinyTokenizer()
        record = EVAL.build_zsre_record(raw_record(3), 9, tokenizer)
        rewrite = record["requested_rewrite"]
        self.assertEqual(record["case_id"], 9)
        self.assertEqual(rewrite["target_new"]["str"], "<|endoftext|>")
        self.assertEqual(rewrite["target_true"]["str"], "Answer3")
        self.assertEqual(rewrite["prompt"].format(rewrite["subject"]), "Where is Subject3")
        self.assertEqual(
            len(record["neighborhood_prompts"]),
            len(tokenizer(" Place3")["input_ids"]),
        )

    def test_prediction_expansion_matches_answer_token_prefixes(self):
        tokenizer = TinyTokenizer()
        record = adapted_record()
        answer_ids = tokenizer(" AB")["input_ids"]
        cases = EVAL.expand_prediction_cases(
            record,
            tokenizer,
            llama_like=False,
        )
        rewrite = [case for case in cases if case.prompt_type == "rewrite"]
        paraphrase = [case for case in cases if case.prompt_type == "paraphrase"]
        neighborhood = [case for case in cases if case.prompt_type == "neighborhood"]
        self.assertEqual(len(rewrite), len(answer_ids))
        self.assertEqual(len(paraphrase), len(answer_ids))
        self.assertEqual(len(neighborhood), 1)
        self.assertEqual(rewrite[0].prompt, "Where is Alice")
        self.assertEqual(
            rewrite[1].prompt,
            "Where is Alice" + tokenizer.decode(answer_ids[:1]),
        )

    def test_summary_uses_per_record_macro_accuracy(self):
        metric_data = [
            {
                "post": {
                    "rewrite_prompts_correct": [True, False],
                    "paraphrase_prompts_correct": [False, False],
                    "neighborhood_prompts_correct": [True],
                }
            },
            {
                "post": {
                    "rewrite_prompts_correct": [True],
                    "paraphrase_prompts_correct": [True],
                    "neighborhood_prompts_correct": [False, False, True],
                }
            },
        ]
        summary = EVAL.official_summarize("forget", metric_data)
        self.assertEqual(summary["Eff"], 75.0)
        self.assertEqual(summary["Gen"], 50.0)
        self.assertAlmostEqual(summary["Spe"], 66.666667)
        self.assertEqual(summary["post_rewrite_correct_tokens"], 2)
        self.assertEqual(summary["post_rewrite_total_tokens"], 3)


class ZsRESetting5AndRepairTests(unittest.TestCase):
    def test_canonical_mapping_reverses_raw_zsre_field_convention(self):
        example = PIPELINE.canonical_examples(
            [adapted_record()],
            TinyTokenizer(),
        )[0]
        self.assertEqual(example.answer, " AB")
        self.assertEqual(example.target_new, " AB")
        self.assertEqual(example.target_true, "<eos>")
        self.assertEqual(example.source, "zsre")

    def test_protected_projection_allows_active_eos_boost_without_regression(self):
        active_cache = token_cache(hidden=[1.0, 0.0])
        protected_cache = token_cache(
            hidden=[0.0, 1.0],
            prompt_type="neighborhood",
        )
        args = SimpleNamespace(
            project_away_protected_hidden=True,
            repair_rank=1,
            active_logit_margin=0.25,
            repair_steps=200,
            repair_lr=1.0,
            repair_optimizer="adam",
            repair_l2_lambda=0.0,
            stop_when_all_satisfied=True,
            max_delta_norm=None,
        )
        with mock.patch.object(
            PIPELINE.active,
            "make_repair_optimizer",
            side_effect=lambda module, name, learning_rate: FakeGradientDescent(
                module.parameters(),
                learning_rate,
            ),
        ):
            delta, _, summary = PIPELINE.optimize_eos_delta(
                [active_cache],
                [protected_cache],
                hidden_size=2,
                device=torch.device("cpu"),
                args=args,
            )
        self.assertTrue(summary["all_satisfied"])
        self.assertGreaterEqual(
            PIPELINE.margins_for_delta([active_cache], delta[0]).item(),
            0.25,
        )
        self.assertEqual(
            PIPELINE.protected_regressions_for_delta(
                [protected_cache],
                delta[0],
            ),
            0,
        )
        self.assertAlmostEqual(float(delta[0, 1]), 0.0, places=5)

    def test_scale_selection_prioritizes_zero_protected_regressions(self):
        active_cache = token_cache(hidden=[1.0, 0.0])
        protected_cache = token_cache(
            hidden=[1.0, 0.0],
            target_logit=1.1,
            neutral_logit=0.0,
            prompt_type="neighborhood",
        )
        scale, reports = PIPELINE.select_delta_scale(
            torch.tensor([2.0, 0.0]),
            [active_cache],
            [protected_cache],
            [1.0, 0.5, 0.0],
        )
        self.assertEqual(scale, 0.5)
        selected = next(row for row in reports if row["scale"] == scale)
        self.assertEqual(selected["protected_regressions"], 0)
        self.assertEqual(selected["active_correct_tokens"], 0)

    def test_official_gate_rejects_retain_drop(self):
        setting = {
            "forget": {"Eff": 10.0, "Gen": 12.0, "Spe": 80.0},
            "retain": {"Eff": 90.0, "Gen": 88.0, "Spe": 85.0},
            "forget_PPL": 13.0,
        }
        candidate = {
            "forget": {"Eff": 0.0, "Gen": 0.0, "Spe": 80.0},
            "retain": {"Eff": 89.0, "Gen": 88.0, "Spe": 85.0},
            "forget_PPL": 13.0,
        }
        report = PIPELINE.metric_gate_report(
            setting,
            candidate,
            utility_drop_tolerance=0.1,
            max_ppl_ratio=1.02,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["retain_Eff"]["passed"])


class ZsREAggregationTests(unittest.TestCase):
    @staticmethod
    def result(seed, shift):
        def method_block(value):
            return {
                "forget": {"Eff": value, "Gen": value + 1, "Spe": 80.0},
                "retain": {"Eff": 90.0, "Gen": 89.0, "Spe": 88.0},
                "PPL": 13.0,
            }

        return {
            "dataset": "ZsRE",
            "seed": seed,
            "forget_num": 50,
            "retain_num": 1000,
            "zsre_sha256": "same",
            "repair": {"candidate_accepted": True},
            "base": method_block(30.0 + shift),
            "setting5e": method_block(10.0 + shift),
            "active_candidate": method_block(0.0 + shift),
            "selected": method_block(0.0 + shift),
        }

    def test_aggregate_writes_mean_std_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for seed, shift in ((1, 0.0), (2, 2.0)):
                path = root / f"seed{seed}.json"
                path.write_text(json.dumps(self.result(seed, shift)), encoding="utf-8")
                paths.append(path)
            results = AGG.load_results(paths)
            rows = AGG.aggregate(results)
            selected = next(row for row in rows if row["method"] == "Selected")
            self.assertEqual(selected["forget_Eff_down_mean"], 1.0)
            self.assertEqual(selected["forget_Eff_down_std"], 1.0)
            AGG.write_outputs(root / "aggregate", results, rows)
            self.assertTrue((root / "aggregate" / "aggregate.md").exists())

    def test_wrapper_defaults_to_paper_seeds_and_aggregates(self):
        script = (
            SCRIPTS_DIR / "run_zsre_gagd_setting5e_active_repair.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"', script)
        self.assertIn("zsre_gagd_setting5e_active_repair.py", script)
        self.assertIn("aggregate_zsre_gagd_results.py", script)

    def test_skipped_ppl_aggregates_to_json_null(self):
        results = [self.result(1, 0.0), self.result(2, 1.0)]
        for result in results:
            for key in ("base", "setting5e", "active_candidate", "selected"):
                result[key]["PPL"] = None
        rows = AGG.aggregate(results)
        self.assertIsNone(rows[0]["PPL_down_mean"])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            AGG.write_outputs(output, results, rows)
            payload = json.loads((output / "aggregate.json").read_text())
            self.assertIsNone(payload["rows"][0]["PPL_down_mean"])


if __name__ == "__main__":
    unittest.main()
