import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import aggregate_rwku_results as AGG  # noqa: E402
import rwku_data as DATA  # noqa: E402
import rwku_eval as EVAL  # noqa: E402
import rwku_experiment as EXPERIMENT  # noqa: E402
import rwku_repair as REPAIR  # noqa: E402


class TemplateTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    unk_token_id = 99

    @staticmethod
    def apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize
        assert add_generation_prompt
        return f"<chat>{messages[0]['content']}</chat>"

    @staticmethod
    def convert_tokens_to_ids(token):
        return 3 if token == "<|eot_id|>" else 99


class TensorBatch(dict):
    def to(self, device):
        return TensorBatch(
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in self.items()
            }
        )


class TinyTokenizer(TemplateTokenizer):
    pad_token = "<pad>"
    pad_token_id = 0
    bos_token_id = 1
    padding_side = "right"

    @staticmethod
    def _encode(text, add_special_tokens):
        values = [4 + (ord(character) % 50) for character in str(text)]
        return ([1] if add_special_tokens else []) + values

    def __call__(
        self,
        text,
        add_special_tokens=True,
        padding=False,
        return_tensors=None,
        truncation=False,
        max_length=None,
        **kwargs,
    ):
        is_batch = isinstance(text, list)
        values = text if is_batch else [text]
        rows = [self._encode(value, add_special_tokens) for value in values]
        if max_length is not None:
            rows = [row[-max_length:] for row in rows]
        masks = [[1] * len(row) for row in rows]
        if padding:
            width = max(len(row) for row in rows)
            if self.padding_side == "left":
                rows = [
                    [self.pad_token_id] * (width - len(row)) + row
                    for row in rows
                ]
                masks = [
                    [0] * (width - len(mask)) + mask for mask in masks
                ]
            else:
                rows = [
                    row + [self.pad_token_id] * (width - len(row))
                    for row in rows
                ]
                masks = [
                    mask + [0] * (width - len(mask)) for mask in masks
                ]
        if return_tensors == "pt":
            return TensorBatch(
                {
                    "input_ids": torch.tensor(rows, dtype=torch.long),
                    "attention_mask": torch.tensor(masks, dtype=torch.long),
                }
            )
        return {
            "input_ids": rows if is_batch else rows[0],
            "attention_mask": masks if is_batch else masks[0],
        }

    @staticmethod
    def decode(token_ids, skip_special_tokens=False):
        output = []
        for token_id in token_ids:
            value = int(token_id)
            if skip_special_tokens and value in {0, 1, 2, 3}:
                continue
            if value >= 4:
                output.append(chr((value - 4) % 50 + 65))
        return "".join(output)


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size=64, hidden_size=6):
        super().__init__()
        torch.manual_seed(3)
        self.input_embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        self.output_embeddings = torch.nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
        )
        self.config = SimpleNamespace(
            tie_word_embeddings=False,
            use_cache=False,
        )

    def get_input_embeddings(self):
        return self.input_embeddings

    def get_output_embeddings(self):
        return self.output_embeddings

    def set_output_embeddings(self, module):
        self.output_embeddings = module

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        use_cache=False,
        **kwargs,
    ):
        hidden = self.input_embeddings(input_ids)
        logits = self.output_embeddings(hidden)
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden,) if output_hidden_states else None,
        )


class FakeOptimizer:
    def __init__(self, parameters, lr, weight_decay=0.0):
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self, set_to_none=True):
        for parameter in self.parameters:
            parameter.grad = None

    def step(self):
        with torch.no_grad():
            for parameter in self.parameters:
                parameter.add_(parameter.grad, alpha=-self.lr)


def qa_row(query, answer, level="2", subject="Stephen King", kind="simple question"):
    return {
        "query": query,
        "answer": answer,
        "level": level,
        "type": kind,
        "subject": subject,
    }


class RWKUDataTests(unittest.TestCase):
    def test_seed_target_mapping_is_published_order(self):
        self.assertEqual(DATA.target_for_seed(0).subject, "Stephen King")
        self.assertEqual(DATA.target_for_seed(9).subject, "Prince Harry, Duke of Sussex")
        with self.assertRaises(ValueError):
            DATA.target_for_seed(10)
        with self.assertRaises(ValueError):
            DATA.target_for_seed(True)

    def test_duplicate_records_never_cross_calibration_boundary(self):
        duplicate = qa_row("Q duplicate?", "A")
        rows = [
            duplicate,
            dict(duplicate),
            qa_row("Q second?", "B"),
            qa_row("Q third?", "C"),
        ]
        calibration, held_out = DATA.partition_records(
            rows,
            seed=4,
            calibration_fraction=0.5,
        )
        calibration_hashes = {DATA.record_sha256(row) for row in calibration}
        held_out_hashes = {DATA.record_sha256(row) for row in held_out}
        self.assertFalse(calibration_hashes & held_out_hashes)
        duplicate_side_counts = (
            sum(DATA.record_sha256(row) == DATA.record_sha256(duplicate) for row in calibration),
            sum(DATA.record_sha256(row) == DATA.record_sha256(duplicate) for row in held_out),
        )
        self.assertIn(2, duplicate_side_counts)
        manifest = DATA.build_split_manifest(calibration, held_out)
        self.assertTrue(manifest["disjoint"])

    def test_intro_singleton_is_the_only_supported_object_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intro.json"
            path.write_text(
                json.dumps({"target": "Stephen King", "intro": "text"}),
                encoding="utf-8",
            )
            rows = DATA.load_json_list(path, allow_singleton_object=True)
            self.assertEqual(rows[0]["target"], "Stephen King")
            with self.assertRaises(ValueError):
                DATA.load_json_list(path)


class RWKUEvaluatorTests(unittest.TestCase):
    def test_prompts_match_rwku_attack_special_cases(self):
        tokenizer = TemplateTokenizer()
        ordinary = EVAL.format_qa_prompt(
            tokenizer,
            qa_row("Who wrote It?", "Stephen King"),
        )
        self.assertIn("Please briefly answer", ordinary)
        self.assertTrue(ordinary.endswith("Answer:"))

        affirmative = EVAL.format_qa_prompt(
            tokenizer,
            qa_row(
                "Stephen King wrote it, right?",
                "yes",
                level="3",
                kind="affirmative suffix",
            ),
        )
        self.assertFalse(affirmative.endswith("Answer:"))

        in_context = EVAL.format_qa_prompt(
            tokenizer,
            qa_row(
                "Question: x Answer: y",
                "z",
                level="3",
                kind="incontext learning",
            ),
        )
        self.assertNotIn("Question: Question:", in_context)
        self.assertTrue(in_context.endswith("Answer:"))

    def test_paraphrase_and_alias_questions_are_deterministic(self):
        rows = [qa_row("What is Stephen King known for?", "horror")]
        paraphrases = EVAL.paraphrase_rows(rows)
        self.assertEqual(paraphrases[0]["query"], "State Stephen King known for?")
        aliases = EVAL.alias_question_rows(rows, subject="Stephen King")
        self.assertEqual(len(aliases), 1)
        self.assertNotIn("Stephen King", aliases[0]["query"])
        self.assertIn(aliases[0]["subject_alias"], {"King", "SK"})
        self.assertEqual(EVAL.subject_aliases("Confucius"), [])

    def test_text_metrics_and_tie_aware_auc(self):
        self.assertTrue(EVAL.recovery_success("It was Richard Bachman.", "Richard Bachman"))
        self.assertEqual(EVAL.rouge_l_recall("A B X", "A B C"), 2 / 3)
        self.assertEqual(EVAL.rank_auc([1.0, 1.0], [1.0, 0.0]), 0.75)
        summary = EVAL.summarize_membership_inference(
            [
                {
                    "loss_log_likelihood": 1.0,
                    "zlib_log_likelihood": 1.0,
                    "min_k_20": 1.0,
                    "min_k_plus_plus_20": 1.0,
                }
            ],
            [
                {
                    "loss_log_likelihood": 0.0,
                    "zlib_log_likelihood": 0.0,
                    "min_k_20": 0.0,
                    "min_k_plus_plus_20": 0.0,
                }
            ],
        )
        self.assertEqual(summary["max_attack_advantage"], 1.0)

    def test_completion_scorer_uses_all_answer_tokens(self):
        model = TinyCausalLM()
        tokenizer = TinyTokenizer()
        scores = EVAL.score_completions(
            model,
            tokenizer,
            [("prompt", "answer"), ("other", "xy")],
            batch_size=2,
        )
        self.assertEqual(len(scores), 2)
        self.assertEqual(scores[0].token_count, len(" answer"))
        self.assertTrue(math.isfinite(scores[0].sum_logprob))
        self.assertGreater(scores[0].first_token_probability, 0.0)
        self.assertLess(scores[0].first_token_probability, 1.0)

    def test_json_writer_converts_structural_nan_to_null(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            EVAL.write_json(path, {"eligible": float("nan"), "value": 1.0})
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(value["eligible"])
            self.assertEqual(value["value"], 1.0)


def repair_point(hidden, competitor=1.0, neutral=0.0, source="x"):
    return REPAIR.RepairPoint(
        hidden=torch.tensor(hidden, dtype=torch.float32),
        competitor_logit=torch.tensor(competitor),
        neutral_logit=torch.tensor(neutral),
        source_id=source,
        token_index=0,
        target_token_id=4,
        baseline_predicted_token_id=5,
    )


class RWKURepairTests(unittest.TestCase):
    def test_projected_optimizer_leaves_protected_direction_orthogonal(self):
        active_points = [repair_point([1.0, 0.0])]
        protected_points = [repair_point([0.0, 1.0])]
        with mock.patch.object(REPAIR.torch.optim, "AdamW", FakeOptimizer):
            delta, log, summary = REPAIR.optimize_delta(
                active_points,
                protected_points,
                hidden_size=2,
                device=torch.device("cpu"),
                config=REPAIR.RepairConfig(
                    steps=150,
                    learning_rate=0.1,
                    active_margin=0.2,
                    selection_margin=0.0,
                    l2_lambda=1e-6,
                    candidate_scales=(1.0, 0.0),
                ),
            )
        self.assertTrue(log)
        self.assertEqual(summary["protected_hidden_rank"], 1)
        self.assertGreater(float(delta[0]), 1.0)
        self.assertAlmostEqual(float(delta[1]), 0.0, places=5)

    def test_scale_selection_prioritizes_no_protected_regressions(self):
        active_points = [repair_point([1.0, 0.0], competitor=1.0)]
        conflicting_protected = [
            repair_point([1.0, 0.0], competitor=0.1, source="retain")
        ]
        scale, reports = REPAIR.select_materialized_scale(
            original_row=torch.zeros(2),
            delta=torch.tensor([2.0, 0.0]),
            active_points=active_points,
            protected_points=conflicting_protected,
            scales=(1.0, 0.5, 0.0),
            selection_margin=0.0,
        )
        self.assertEqual(scale, 0.0)
        selected = next(row for row in reports if row["scale"] == scale)
        self.assertEqual(selected["protected_regressions"], 0)

    def test_repair_token_prefers_llama_eot(self):
        token, token_id = REPAIR.resolve_neutral_token(TemplateTokenizer())
        self.assertEqual((token, token_id), ("<|eot_id|>", 3))


class RWKUAggregationTests(unittest.TestCase):
    @staticmethod
    def synthetic_method_result(seed):
        sections = {}
        for section, metrics in {
            "forget": AGG.FORGET_METRICS,
            "retain": AGG.RETAIN_METRICS,
            "controls": AGG.CONTROL_METRICS,
        }.items():
            sections[section] = {
                key: float(seed) for key, _, _ in metrics
            }
        return {"summary": sections}

    def test_strict_ten_seed_aggregation_and_optional_eligibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, target in enumerate(DATA.TARGETS_BY_SEED):
                results = {
                    method: self.synthetic_method_result(seed)
                    for method in EXPERIMENT.METHOD_ORDER
                }
                if seed == 1:
                    results[EXPERIMENT.METHOD_BASE]["summary"]["forget"][
                        "alias_question_recovery"
                    ] = None
                path = root / f"seed{seed:02d}_{target.directory}" / "results.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "seed": seed,
                            "status": "complete",
                            "rwku_dataset_revision": DATA.RWKU_DATASET_REVISION,
                            "target": {
                                "seed": seed,
                                "subject": target.subject,
                                "directory": target.directory,
                            },
                            "results": results,
                        }
                    ),
                    encoding="utf-8",
                )
            runs = AGG.load_runs(root, allow_incomplete=False)
            rows = AGG.aggregate_section(
                runs,
                section="forget",
                metrics=AGG.FORGET_METRICS,
            )
            base = rows[0]
            self.assertEqual(base["direct_target_qa_recovery_mean"], 4.5)
            self.assertEqual(base["alias_question_recovery_n"], 9)

    def test_launcher_wires_exactly_seeds_zero_through_nine(self):
        launcher = (SCRIPTS_DIR / "run_rwku_experiment.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("for seed in 0 1 2 3 4 5 6 7 8 9", launcher)
        self.assertIn("aggregate_rwku_results.py", launcher)


if __name__ == "__main__":
    unittest.main()
