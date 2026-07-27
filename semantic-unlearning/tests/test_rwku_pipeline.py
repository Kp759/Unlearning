import json
import math
import subprocess
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

    def test_subset_validation_cannot_overwrite_pinned_manifest(self):
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            DATA.validate_manifest_destination(
                [0],
                DATA.PINNED_MANIFEST_PATH,
            )
        DATA.validate_manifest_destination(
            list(range(10)),
            DATA.PINNED_MANIFEST_PATH,
        )


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

    def test_balanced_multiple_choice_has_exact_chance_baseline(self):
        rows = [
            qa_row(f"Question {index}?", f"answer {index}")
            for index in range(4)
        ]
        fixed_scores = [
            EVAL.CompletionScore(
                sum_logprob=-float(index),
                mean_logprob=-float(index),
                first_token_probability=math.exp(-float(index)),
                token_count=1,
            )
            for index in range(4)
        ]
        with mock.patch.object(
            EVAL,
            "score_completions",
            return_value=fixed_scores,
        ):
            summary, details = (
                EVAL.evaluate_balanced_constructed_multiple_choice(
                    TinyCausalLM(),
                    TinyTokenizer(),
                    rows,
                    [str(row["answer"]) for row in rows],
                    batch_size=4,
                )
            )
        self.assertEqual(summary["rotation_count"], 16)
        self.assertEqual(summary["accuracy"], 25.0)
        self.assertEqual(
            sorted({detail["answer_index"] for detail in details}),
            [0, 1, 2, 3],
        )

    def test_json_writer_converts_structural_nan_to_null(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            EVAL.write_json(path, {"eligible": float("nan"), "value": 1.0})
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(value["eligible"])
            self.assertEqual(value["value"], 1.0)


def repair_point(hidden, competitor=0.0, target=1.0, source="x", token_id=4):
    return REPAIR.RepairPoint(
        hidden=torch.tensor(hidden, dtype=torch.float32),
        competitor_logit=torch.tensor(competitor),
        target_logit=torch.tensor(target),
        source_id=source,
        token_index=0,
        target_token_id=token_id,
        competitor_token_id=5,
        baseline_predicted_token_id=token_id,
    )


def answer_cache(hidden, target=1.0, competitor=0.0, repair=True):
    return REPAIR.AnswerDeltaCache(
        source_id="active",
        hidden=torch.tensor([hidden], dtype=torch.float32),
        target_token_ids=torch.tensor([4]),
        target_selected_indices=torch.tensor([0]),
        repair_mask=torch.tensor([repair]),
        base_target_logits=torch.tensor([target]),
        base_selected_logits=torch.tensor([[target]]),
        base_nonselected_logsumexp=torch.tensor([competitor]),
        base_nonselected_max=torch.tensor([competitor]),
    )


def context_cache(hidden, selected=0.1, nonselected=0.0):
    prediction = 4 if selected > nonselected else 5
    return REPAIR.ContextDeltaCache(
        source_id="retain",
        hidden=torch.tensor([hidden], dtype=torch.float32),
        base_selected_logits=torch.tensor([[selected]]),
        base_nonselected_max=torch.tensor([nonselected]),
        base_nonselected_token_ids=torch.tensor([5]),
        baseline_predicted_token_ids=torch.tensor([prediction]),
    )


class RWKURepairTests(unittest.TestCase):
    def test_projected_optimizer_leaves_protected_direction_orthogonal(self):
        active_caches = [answer_cache([1.0, 0.0])]
        protected_contexts = [context_cache([0.0, 1.0])]
        with mock.patch.object(REPAIR.torch.optim, "AdamW", FakeOptimizer):
            delta, log, summary = REPAIR.optimize_delta(
                active_caches,
                protected_contexts,
                n_rows=1,
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
        self.assertLess(float(delta[0, 0]), -1.0)
        self.assertAlmostEqual(float(delta[0, 1]), 0.0, places=5)

    def test_scale_selection_prioritizes_no_protected_regressions(self):
        active_caches = [answer_cache([1.0, 0.0])]
        protected_answers = [
            REPAIR.AnswerDeltaCache(
                source_id="retain",
                hidden=torch.tensor([[1.0, 0.0]]),
                target_token_ids=torch.tensor([5]),
                target_selected_indices=torch.tensor([-1]),
                repair_mask=torch.tensor([False]),
                base_target_logits=torch.tensor([0.0]),
                base_selected_logits=torch.tensor([[0.1]]),
                base_nonselected_logsumexp=torch.tensor([1.0]),
                base_nonselected_max=torch.tensor([0.0]),
            )
        ]
        protected_contexts = [context_cache([1.0, 0.0])]
        config = REPAIR.RepairConfig(
            max_protected_top1_changes=0,
            candidate_scales=(1.0, 0.5, 0.0),
        )
        scale, reports = REPAIR.select_materialized_scale(
            original_rows=torch.zeros((1, 2)),
            delta_rows=torch.tensor([[-2.0, 0.0]]),
            active_caches=active_caches,
            protected_answer_caches=protected_answers,
            protected_context_caches=protected_contexts,
            selected_token_ids=[4],
            scales=(1.0, 0.5, 0.0),
            config=config,
        )
        self.assertEqual(scale, 0.0)
        selected = next(row for row in reports if row["scale"] == scale)
        self.assertEqual(selected["protected_top1_changes"], 0)
        rejected = next(row for row in reports if row["scale"] == 1.0)
        self.assertFalse(rejected["passes_all_protection_gates"])

    def test_active_row_selection_excludes_special_and_protected_rows(self):
        active = [
            repair_point([1.0, 0.0], token_id=2),
            repair_point([1.0, 0.0], token_id=4),
            repair_point([1.0, 0.0], token_id=6),
        ]
        protected = [repair_point([0.0, 1.0], token_id=6)]
        selected, overlap = REPAIR.select_active_token_ids(
            active,
            protected,
            special_token_ids=[0, 1, 2, 3],
            exclude_protected_answer_rows=True,
        )
        self.assertEqual(selected, [4])
        self.assertEqual(overlap, [6])
        self.assertIn(3, REPAIR.repair_excluded_token_ids(TemplateTokenizer()))

    def test_end_to_end_repair_never_changes_stop_or_input_rows(self):
        model = TinyCausalLM()
        tokenizer = TinyTokenizer()
        input_before = model.get_input_embeddings().weight.detach().clone()
        output_before = model.get_output_embeddings().weight.detach().clone()
        with mock.patch.object(REPAIR.torch.optim, "AdamW", FakeOptimizer):
            report = REPAIR.run_protected_lm_head_repair(
                model,
                tokenizer,
                calibration_rows=[qa_row("Who?", "x")],
                protected_examples=[
                    SimpleNamespace(prompt="Unrelated?", answer="y")
                ],
                config=REPAIR.RepairConfig(
                    steps=50,
                    learning_rate=0.05,
                    active_margin=100.0,
                    selection_margin=100.0,
                    protected_projection_rank=0,
                    min_protected_probability_ratio=0.01,
                    max_protected_logit_drift=100.0,
                    max_protected_top1_changes=100,
                    candidate_scales=(1.0, 0.5, 0.0),
                ),
            )
        self.assertEqual(report["method"], "sparse_active_pair_lm_head_repair")
        self.assertFalse(report["edits_eot_or_eos"])
        self.assertNotIn(tokenizer.eos_token_id, report["changed_output_rows"])
        self.assertNotIn(3, report["changed_output_rows"])
        self.assertTrue(
            torch.equal(
                input_before,
                model.get_input_embeddings().weight.detach(),
            )
        )
        changed = (
            model.get_output_embeddings().weight.detach() - output_before
        ).abs().sum(dim=1).gt(0).nonzero(as_tuple=False).flatten().tolist()
        self.assertEqual(changed, report["changed_output_rows"])


class RWKUExperimentProtocolTests(unittest.TestCase):
    def test_all_methods_include_base_initialized_representation_candidate(self):
        self.assertEqual(len(EXPERIMENT.METHOD_ORDER), 6)
        self.assertEqual(
            EXPERIMENT.selected_methods("representation"),
            (
                EXPERIMENT.METHOD_BASE,
                EXPERIMENT.METHOD_REPRESENTATION,
            ),
        )
        self.assertIn(
            EXPERIMENT.METHOD_REPRESENTATION,
            EXPERIMENT.selected_methods("all"),
        )

        args = EXPERIMENT.build_parser().parse_args(["--seed", "0", "--dry-run"])
        config = EXPERIMENT.representation_config(args)
        self.assertEqual(config.steps, 1250)
        self.assertEqual(config.rank, 16)
        self.assertEqual(config.last_n_layers, 8)
        self.assertEqual(
            config.target_modules,
            (
                "q_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
        )
        self.assertEqual(config.answer_probability_target, 1e-6)
        self.assertEqual(config.min_retain_answer_probability_ratio, 0.995)
        self.assertEqual(config.max_retain_answer_probability_ratio, 1.005)
        self.assertEqual(config.min_retain_top1_agreement, 0.99)

    def test_local_model_identity_records_snapshot_and_weight_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "blobs" / ("a" * 64)
            blob.parent.mkdir()
            blob.write_bytes(b"weight bytes")
            snapshot = root / "snapshots" / ("b" * 40)
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").symlink_to(blob)

            identity = EXPERIMENT.local_model_identity(str(snapshot))

        self.assertEqual(identity["snapshot_revision"], "b" * 40)
        self.assertEqual(
            identity["weight_files"][0]["resolved_target_name"],
            "a" * 64,
        )
        self.assertIn("symlink_target", identity["weight_files"][0])

    def test_dry_run_records_missing_model_without_weakening_real_run(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_model = Path(directory) / "missing-model"
            with self.assertRaises(FileNotFoundError):
                EXPERIMENT.local_model_identity(str(missing_model))

            args = EXPERIMENT.build_parser().parse_args(
                [
                    "--seed",
                    "0",
                    "--dry-run",
                    "--model-path",
                    str(missing_model),
                ]
            )
            payload = EXPERIMENT.config_payload(
                args,
                methods=EXPERIMENT.METHOD_ORDER,
                target=DATA.target_for_seed(0),
                calibration_rows=[],
                held_out_direct=[],
                file_hashes={},
                split_manifests={},
            )

        self.assertEqual(
            payload["model_identity"],
            {
                "requested": str(missing_model),
                "kind": "missing_local_path",
                "exists": False,
            },
        )

    def test_implementation_identity_covers_transitive_protocol_helpers(self):
        identity = EXPERIMENT.implementation_identity()
        for filename in (
            "gagd_active_case_repair.py",
            "mcf_sampling.py",
            "mcf_zero_unlearn_official_eval.py",
            "run_zerounlearn_official_mcf.py",
        ):
            self.assertTrue(
                any(path.endswith(f"scripts/{filename}") for path in identity),
                filename,
            )

    def test_matched_positive_rows_are_round_robin_and_exclude_target(self):
        def fake_ensure(_root, seed, *, allow_download):
            self.assertFalse(allow_download)
            target = DATA.target_for_seed(seed)
            rows = [
                {
                    "text": f"{target.subject} training text {index}",
                    "subject": target.subject,
                }
                for index in range(2)
            ]
            return target, rows, "sha256"

        with mock.patch.object(
            EXPERIMENT,
            "ensure_positive_training_data",
            side_effect=fake_ensure,
        ):
            rows, hashes = EXPERIMENT.load_matched_positive_training_rows(
                Path("/unused"),
                seed=3,
                allow_download=False,
            )
        first_round = rows[:9]
        second_round = rows[9:18]
        self.assertEqual(len({row["subject"] for row in first_round}), 9)
        self.assertNotIn("Warren Buffett", {row["subject"] for row in rows})
        self.assertEqual(len(hashes), 9)
        self.assertTrue(
            all(row["text"].endswith("0") for row in first_round)
        )
        self.assertTrue(
            all(row["text"].endswith("1") for row in second_round)
        )


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
                            "rwku_code_revision": DATA.RWKU_CODE_REVISION,
                            "rwku_dataset_revision": DATA.RWKU_DATASET_REVISION,
                            "target": {
                                "seed": seed,
                                "subject": target.subject,
                                "directory": target.directory,
                            },
                            "methods": list(EXPERIMENT.METHOD_ORDER),
                            "method_order": list(EXPERIMENT.METHOD_ORDER),
                            "methods_run": list(EXPERIMENT.METHOD_ORDER),
                            "model_path": "/same/model",
                            "model_identity": {"snapshot": "same"},
                            "dtype": "bf16",
                            "implementation_file_sha256": {"driver": "same"},
                            "calibration_fraction": 0.5,
                            "setting5": {"steps": 600},
                            "repair": {"steps": 800},
                            "representation": {
                                "steps": 1000,
                                "seed": seed,
                            },
                            "external_retain_partitions": {
                                "optimization_count": 1000,
                                "checkpoint_gate_count": 128,
                            },
                            "mcf_retain_provenance": {
                                "file_sha256": "mcf",
                                "partitions_disjoint": True,
                                "example_content_partitions_disjoint": True,
                            },
                            "zero_unlearn": {"hparams_sha256": "zero"},
                            "wikidata_corpus": {
                                "directory_sha256": "wikidata"
                            },
                            "evaluation_limits": {},
                            "skip_ppl": False,
                            "eval_batch_size": 4,
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
            self.assertEqual(len(rows), len(EXPERIMENT.METHOD_ORDER))
            self.assertEqual(
                [row["method"] for row in rows],
                list(EXPERIMENT.METHOD_ORDER),
            )
            base = rows[0]
            self.assertEqual(base["direct_target_qa_recovery_mean"], 4.5)
            self.assertEqual(base["alias_question_recovery_n"], 9)

            seed_zero = next(root.glob("seed00_*/results.json"))
            payload = json.loads(seed_zero.read_text(encoding="utf-8"))
            payload["rwku_code_revision"] = "not-the-pinned-revision"
            seed_zero.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "code revision"):
                AGG.load_runs(root, allow_incomplete=False)

            payload["rwku_code_revision"] = DATA.RWKU_CODE_REVISION
            payload["methods_run"] = [
                *EXPERIMENT.METHOD_ORDER,
                EXPERIMENT.METHOD_BASE,
            ]
            seed_zero.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-canonical"):
                AGG.load_runs(root, allow_incomplete=False)

    def test_protocol_fingerprint_ignores_only_per_run_representation_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (0, 1):
                target = DATA.TARGETS_BY_SEED[seed]
                path = root / f"seed{seed:02d}_{target.directory}" / "results.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "seed": seed,
                            "status": "complete",
                            "rwku_code_revision": DATA.RWKU_CODE_REVISION,
                            "rwku_dataset_revision": DATA.RWKU_DATASET_REVISION,
                            "target": {
                                "seed": seed,
                                "subject": target.subject,
                                "directory": target.directory,
                            },
                            "model_path": "/same/model",
                            "dtype": "bfloat16",
                            "calibration_fraction": 0.5,
                            "representation": {
                                "steps": 800,
                                "rank": 16,
                                "seed": seed,
                            },
                            "evaluation_limits": {},
                            "skip_ppl": False,
                            "results": {
                                method: self.synthetic_method_result(seed)
                                for method in EXPERIMENT.METHOD_ORDER
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            self.assertEqual(
                len(AGG.load_runs(root, allow_incomplete=True)),
                2,
            )

            seed_one = next(root.glob("seed01_*/results.json"))
            payload = json.loads(seed_one.read_text(encoding="utf-8"))
            payload["representation"]["rank"] = 32
            seed_one.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                AGG.load_runs(root, allow_incomplete=True)

    def test_launcher_wires_exactly_seeds_zero_through_nine(self):
        launcher = (SCRIPTS_DIR / "run_rwku_experiment.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("for seed in 0 1 2 3 4 5 6 7 8 9", launcher)
        self.assertIn("aggregate_rwku_results.py", launcher)
        self.assertIn("aggregate_results=false", launcher)
        self.assertIn("--dry-run|--skip-ppl", launcher)

    def test_launcher_rejects_arguments_that_override_fixed_run_identity(self):
        launcher = SCRIPTS_DIR / "run_rwku_experiment.sh"
        for argument in ("--seed=4", "--output-root=/tmp/not-the-env-root"):
            completed = subprocess.run(
                ["bash", str(launcher), "--dry-run", argument],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("ten-seed launcher", completed.stderr)


if __name__ == "__main__":
    unittest.main()
