import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rwku_rowwise_active_repair as REPAIR  # noqa: E402
import rwku_setting5e_utility_controlled as UC  # noqa: E402
from rwku_artifact_access import make_artifact, write_artifact  # noqa: E402


class TinyTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0
    bos_token_id = 1
    pad_token_id = 2
    unk_token_id = 3
    all_special_ids = [0, 1, 2, 3]
    name_or_path = "tiny"

    pieces = {
        0: "<eos>",
        1: "<bos>",
        2: "<pad>",
        3: "<unk>",
        10: " Stephen",
        11: " King",
        12: " English",
        13: " horror",
        14: ",",
        15: "7",
        16: "a",
        17: " novelist",
        18: " question",
        19: " answer",
    }
    words = {
        "Stephen": 10,
        "King": 11,
        "English": 12,
        "horror": 13,
        ",": 14,
        "7": 15,
        "a": 16,
        "novelist": 17,
        "question": 18,
        "answer": 19,
    }

    def __len__(self):
        return 32

    def encode(self, text, add_special_tokens=False):
        normalized = str(text).replace(",", " , ").split()
        values = [
            self.words.get(word, 20 + (sum(map(ord, word)) % 8)) for word in normalized
        ]
        return ([self.bos_token_id] if add_special_tokens else []) + values

    def __call__(self, text, add_special_tokens=False, return_tensors=None, **kwargs):
        values = self.encode(text, add_special_tokens=add_special_tokens)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([values], dtype=torch.long)}
        return {"input_ids": values}

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(
            self.pieces.get(int(value), f" token{int(value)}") for value in token_ids
        )

    def convert_tokens_to_ids(self, value):
        return 3


class TinyTiedLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(4)
        self.embed = torch.nn.Embedding(32, 8)
        self.transformer = torch.nn.Linear(8, 8, bias=False)
        self.head = torch.nn.Linear(8, 32, bias=False)
        self.head.weight = self.embed.weight
        self.config = SimpleNamespace(tie_word_embeddings=True)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, layer):
        self.head = layer

    def forward(self, input_ids, **kwargs):
        hidden = self.transformer(self.embed(input_ids))
        return SimpleNamespace(logits=self.head(hidden), hidden_states=(hidden,))


def passing_candidate_metrics():
    return {
        "calibration_source": REPAIR.ACTIVE_SOURCE,
        "official_rwku_records_accessed": False,
        "direct_generation_recovery": 0.0,
        "cloze_generation_recovery": 0.0,
        "paraphrase_generation_recovery": 0.0,
        "generated_geometric_answer_probability": 0.001,
        "active_violation_count": 0,
        "full_retain_probability_ratio": 1.0,
        "geometric_retain_probability_ratio": 1.0,
        "mean_retain_kl": 0.0,
        "p95_retain_kl": 0.0,
        "retain_top1_agreement": 1.0,
        "protected_answer_probability_ratio": 1.0,
        "protected_selected_row_logit_drift": 0.0,
        "protected_top1_changes": 0,
        "proxy_ppl": 10.0,
        "base_proxy_ppl": 10.0,
        "nonselected_rows_equal_base": True,
    }


def config_args(**updates):
    values = {
        "forget_weight": 2.0,
        "retain_ce_weight": 4.0,
        "retain_kl_weight": 10.0,
        "protected_margin_weight": 20.0,
        "delta_l2_weight": 1e-4,
        "forget_margin": 1.0,
        "subject_input_lr": 5e-6,
        "sensitive_output_lr": 2e-5,
        "max_retain_document_frequency": 0.01,
        "teacher_top_k": 128,
        "exposures_per_fact": UC.DEFAULT_EXPOSURES,
        "candidate_scales": UC.DEFAULT_INTERPOLATION_SCALES,
        "development": True,
        "confirmatory": False,
        "frozen_development_config": None,
        "seed": 0,
    }
    values.update(updates)
    return SimpleNamespace(**values)


class ModelIsolationTests(unittest.TestCase):
    def test_tied_lm_head_is_untied_without_changing_logits(self):
        model = TinyTiedLM().eval()
        sample = torch.tensor([[10, 11, 13]], dtype=torch.long)
        report = UC.untie_lm_head_preserve_logits(model, sample_input_ids=sample)
        self.assertTrue(report["was_tied"])
        self.assertTrue(report["initial_logits_bitwise_equal"])
        self.assertNotEqual(
            model.get_input_embeddings().weight.data_ptr(),
            model.get_output_embeddings().weight.data_ptr(),
        )

    def test_transformer_parameters_remain_frozen(self):
        model = TinyTiedLM()
        UC.untie_lm_head_preserve_logits(model)
        report = UC.freeze_transformer_parameters(model)
        self.assertTrue(report["transformer_frozen"])
        self.assertFalse(model.transformer.weight.requires_grad)
        self.assertTrue(model.get_input_embeddings().weight.requires_grad)
        self.assertTrue(model.get_output_embeddings().weight.requires_grad)

    def test_exact_masks_allow_only_declared_rows_and_restore_every_step(self):
        model = TinyTiedLM()
        UC.untie_lm_head_preserve_logits(model)
        UC.freeze_transformer_parameters(model)
        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        guard = UC.ExactRowMask(input_weight, output_weight, [10, 11], [13, 17])
        before_input = input_weight.detach().clone()
        before_output = output_weight.detach().clone()
        (input_weight.sum() + output_weight.sum()).backward()
        with torch.no_grad():
            input_weight.add_(input_weight.grad, alpha=-0.1)
            output_weight.add_(output_weight.grad, alpha=-0.1)
        guard.restore_nonselected()
        guard.verify_or_raise()
        self.assertFalse(torch.equal(input_weight[10], before_input[10]))
        self.assertFalse(torch.equal(output_weight[13], before_output[13]))
        self.assertTrue(torch.equal(input_weight[9], before_input[9]))
        self.assertTrue(torch.equal(output_weight[12], before_output[12]))
        guard.close()


class RowPolicyTests(unittest.TestCase):
    def test_only_subject_input_and_eligible_answer_output_rows_are_selected(self):
        tokenizer = TinyTokenizer()
        training = {
            "metadata": {},
            "payload": {
                "views": [
                    {
                        "subject": "Stephen King",
                        "subject_aliases": [],
                        "sensitive_answer_alias": "horror , 7 a novelist",
                        "canonical_sensitive_answer": "horror , 7 a novelist",
                    }
                ]
            },
        }
        protection = [{"record": {"prompt": "question", "answer": "English"}}]
        policy = UC.build_row_policy(
            tokenizer,
            training,
            protection,
            maximum_document_frequency=0.01,
        )
        self.assertEqual(policy.selected_input_rows, (10, 11))
        self.assertIn(13, policy.selected_output_rows)
        self.assertIn(17, policy.selected_output_rows)
        self.assertNotIn(14, policy.selected_output_rows)
        self.assertNotIn(15, policy.selected_output_rows)
        self.assertNotIn(16, policy.selected_output_rows)


class ScheduleAndInterpolationTests(unittest.TestCase):
    def test_candidate_schedule_is_computed_from_fact_count(self):
        schedule = UC.balanced_candidate_schedule(13)
        self.assertEqual(
            [row["step"] for row in schedule],
            [26, 52, 78, 104, 130, 156, 195, 260],
        )

    def test_exposure_imbalance_is_at_most_one(self):
        order = UC.balanced_fact_order(["c", "a", "b"], 8)
        counts = {fact: order.count(fact) for fact in set(order)}
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_interpolation_is_always_from_base_not_cumulative(self):
        base = torch.zeros(4, 2)
        trained = torch.ones(4, 2) * 8
        weight = base.clone()
        UC.interpolate_rows_from_base(weight, base, trained, [1], 1.0)
        UC.interpolate_rows_from_base(weight, base, trained, [1], 0.25)
        self.assertTrue(torch.equal(weight[1], torch.ones(2) * 2))
        self.assertTrue(torch.equal(weight[2], base[2]))


class ObjectiveAndGateTests(unittest.TestCase):
    def test_teacher_kl_does_not_backpropagate_to_base_reference(self):
        student = torch.randn(2, 4, requires_grad=True)
        teacher = torch.randn(2, 4, requires_grad=True)
        loss, _ = UC.topk_plus_tail_kl(student, teacher, top_k=2)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_generated_gates_reject_non_generated_or_official_sources(self):
        metrics = passing_candidate_metrics()
        metrics["calibration_source"] = "official_level1"
        with self.assertRaisesRegex(ValueError, "target-generated"):
            UC.candidate_gate_report(metrics)
        metrics = passing_candidate_metrics()
        metrics["official_rwku_records_accessed"] = True
        with self.assertRaisesRegex(ValueError, "Official"):
            UC.candidate_gate_report(metrics)

    def test_high_ppl_candidate_is_rejected(self):
        metrics = passing_candidate_metrics()
        metrics["proxy_ppl"] = 10.3
        self.assertFalse(UC.candidate_gate_report(metrics)["eligible"])

    def test_retain_ratio_violation_is_rejected(self):
        metrics = passing_candidate_metrics()
        metrics["full_retain_probability_ratio"] = 0.99
        self.assertFalse(UC.candidate_gate_report(metrics)["eligible"])

    def test_nonzero_generated_recovery_is_rejected(self):
        metrics = passing_candidate_metrics()
        metrics["direct_generation_recovery"] = 1.0
        self.assertFalse(UC.candidate_gate_report(metrics)["eligible"])

    def test_fixed_candidate_and_retain_gates_are_unchanged(self):
        self.assertEqual(
            UC.fixed_gate_manifest(),
            {
                "direct_generation_recovery": 0.0,
                "cloze_generation_recovery": 0.0,
                "paraphrase_generation_recovery": 0.0,
                "generated_geometric_answer_probability_max": 0.01,
                "active_violation_count": 0,
                "full_retain_probability_ratio_range": [0.995, 1.005],
                "geometric_retain_probability_ratio_range": [0.98, 1.02],
                "mean_retain_kl_max": 0.01,
                "p95_retain_kl_max": 0.05,
                "retain_top1_agreement_min": 0.99,
                "protected_answer_probability_ratio_min": 0.999,
                "protected_selected_row_logit_drift_max": 0.05,
                "protected_top1_changes": 0,
                "proxy_ppl_base_multiplier_max": 1.02,
                "nonselected_rows_equal_base": True,
            },
        )

    def test_selection_order_prefers_delta_then_step_then_scale(self):
        candidates = [
            {
                "eligible": True,
                "total_selected_row_delta_norm": 2.0,
                "checkpoint_step": 10,
                "interpolation_scale": 0.25,
            },
            {
                "eligible": True,
                "total_selected_row_delta_norm": 1.0,
                "checkpoint_step": 20,
                "interpolation_scale": 1.0,
            },
            {
                "eligible": True,
                "total_selected_row_delta_norm": 1.0,
                "checkpoint_step": 10,
                "interpolation_scale": 0.75,
            },
            {
                "eligible": True,
                "total_selected_row_delta_norm": 1.0,
                "checkpoint_step": 10,
                "interpolation_scale": 0.50,
            },
        ]
        self.assertEqual(
            UC.select_eligible_candidate(candidates)["interpolation_scale"], 0.50
        )

    def test_official_paths_are_rejected_before_training_or_selection(self):
        for filename in (
            "forget_level1.json",
            "forget_level2.json",
            "forget_level3.json",
            "official_evaluation.json",
        ):
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, "official/evaluation"):
                    UC.reject_official_or_completed_path(
                        Path("data") / filename, label="training"
                    )


class StateAndModeTests(unittest.TestCase):
    def test_configuration_freezes_audited_unmatched_coverage_policy(self):
        configuration = UC.configuration_payload(config_args())
        self.assertEqual(
            configuration["matched_protection_coverage_policy"],
            "allow_unmatched_generated_target_keys_but_audit",
        )

    def test_unmatched_protection_is_audited_and_does_not_abort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "outputs"
            bundle_path = root / "generated_training_bundle.json"
            bundle = make_artifact(
                "training_bundle",
                {
                    "views": [
                        {
                            "fact_id": "fact-carrie",
                            "subject": "Stephen King",
                            "sensitive_answer_alias": "Carrie",
                            "canonical_sensitive_answer": "Carrie",
                            "sensitive_answer_aliases": [],
                            "relation_id": "notable_novel",
                        }
                    ]
                },
                protocol_label="rwku_target_only_generated_entity_corpus_method_extension",
                protocol_status="generated",
                metadata={
                    "subject": "Stephen King",
                    "official_rwku_records_accessed": False,
                },
            )
            write_artifact(bundle_path, bundle)
            protection_source = root / "target_independent_source.json"
            protection_source.write_text(
                json.dumps(
                    [
                        {
                            "prompt": "A wholly unrelated retention prompt",
                            "answer": "A wholly unrelated retention answer",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            mcf_path = root / "mcf.json"
            mcf_path.write_text("[]\n", encoding="utf-8")
            args = SimpleNamespace(
                output_root=output_root,
                experiment_id="run",
                generated_entity_fact_bundle=bundle_path,
                protection_source=[protection_source],
                protection_vocabulary=None,
                tokenize_protection_rows=False,
                model_path=root / "model-not-loaded",
                model_revision="revision",
                no_download=True,
                seed=0,
                minimum_protection_train_per_key=1,
                minimum_protection_gate_per_key=1,
                mcf_path=mcf_path,
                mcf_optimization_count=1,
                mcf_gate_count=1,
            )
            UC.write_state(
                args,
                "PREPARED",
                target={"subject": "Stephen King"},
                official_rwku_records_accessed=False,
            )
            source_records = [{"case_id": 1}, {"case_id": 2}]
            examples = [
                SimpleNamespace(
                    prompt="Independent retain prompt one",
                    answer="Retain one",
                    subject="Independent one",
                    target_new="Retain one",
                    target_true="Original one",
                ),
                SimpleNamespace(
                    prompt="Independent retain prompt two",
                    answer="Retain two",
                    subject="Independent two",
                    target_new="Retain two",
                    target_true="Original two",
                ),
            ]
            with mock.patch.object(
                UC,
                "build_matched_protection",
                wraps=UC.build_matched_protection,
            ) as build, mock.patch.object(
                UC.legacy,
                "load_mcf_retain",
                return_value=(source_records, examples),
            ):
                UC.protection_stage(args)
            self.assertIs(build.call_args.kwargs["strict"], False)

            protection_dir = output_root / "run" / "protection"
            coverage_artifact = json.loads(
                (protection_dir / "matched_protection_coverage.json").read_text()
            )
            coverage = coverage_artifact["payload"]["coverage"]
            self.assertEqual(len(coverage), 1)
            self.assertEqual(coverage[0]["normalized_key"], "carrie")
            self.assertEqual(coverage[0]["optimization_count"], 0)
            self.assertEqual(coverage[0]["gate_count"], 0)
            self.assertEqual(
                coverage[0]["coverage_status"], "insufficient_coverage"
            )
            self.assertEqual(
                coverage_artifact["payload"]["warnings"],
                ["Insufficient coverage: carrie"],
            )

            state = json.loads(
                (output_root / "run" / "experiment_state.json").read_text()
            )
            self.assertTrue(state["protection_prepared"])
            self.assertEqual(state["matched_protection_key_count"], 1)
            self.assertEqual(state["matched_protection_covered_key_count"], 0)
            self.assertEqual(state["matched_protection_insufficient_key_count"], 1)
            self.assertEqual(
                state["matched_protection_insufficient_keys"], ["carrie"]
            )
            self.assertFalse(state["official_rwku_records_accessed"])

            optimization = json.loads(
                (protection_dir / "mcf_optimization_manifest.json").read_text()
            )
            gate = json.loads(
                (protection_dir / "mcf_gate_manifest.json").read_text()
            )
            self.assertFalse(
                set(optimization["record_sha256"])
                & set(gate["record_sha256"])
            )
            matched_train = json.loads(
                (protection_dir / "matched_protection_train.json").read_text()
            )
            matched_gate = json.loads(
                (protection_dir / "matched_protection_gate.json").read_text()
            )
            self.assertFalse(
                {
                    row["content_sha256"]
                    for row in matched_train["payload"]["records"]
                }
                & {
                    row["content_sha256"]
                    for row in matched_gate["payload"]["records"]
                }
            )

    def test_prepare_locks_descriptor_without_opening_official_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "weights.bin").write_bytes(b"base")
            bundle_path = root / "generated_training_bundle.json"
            bundle = make_artifact(
                "training_bundle",
                {"views": []},
                protocol_label="rwku_target_only_generated_entity_corpus_method_extension",
                protocol_status="generated",
                metadata={"subject": "Stephen King"},
            )
            write_artifact(bundle_path, bundle)
            generator_path = root / "generator_receipt.json"
            generator = make_artifact(
                "generator_receipt",
                {
                    "official_rwku_records_accessed": False,
                    "target_entity": "Stephen King",
                    "final_entity_fact_bundle_sha256": bundle["sha256"],
                },
                protocol_label="rwku_target_only_generated_entity_corpus_method_extension",
                protocol_status="generated",
                metadata={"subject": "Stephen King"},
            )
            write_artifact(generator_path, generator)
            args = config_args(
                output_root=root / "outputs",
                experiment_id="run",
                generated_entity_fact_bundle=bundle_path,
                generator_receipt=generator_path,
                model_path=model,
                model_revision="revision",
                dtype="bf16",
            )
            with mock.patch.object(
                UC,
                "ensure_target_data",
                side_effect=AssertionError("official rows were opened"),
            ):
                UC.prepare_stage(args)
            state = json.loads(
                (root / "outputs" / "run" / "experiment_state.json").read_text()
            )
            self.assertEqual(state["state"], "PREPARED")
            self.assertFalse(state["official_rwku_records_accessed"])

    def test_no_feasible_candidate_creates_no_selected_checkpoint_or_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_root=Path(directory), experiment_id="run")
            UC.write_state(args, "CANDIDATES_EVALUATED")
            UC.mark_no_feasible_candidate(args)
            state = json.loads(
                (Path(directory) / "run" / "experiment_state.json").read_text()
            )
            self.assertEqual(state["state"], "NO_FEASIBLE_CANDIDATE")
            self.assertFalse(
                (Path(directory) / "run" / "checkpoint_receipt.json").exists()
            )
            self.assertFalse(
                (
                    Path(directory)
                    / "run"
                    / "utility_controlled_setting5"
                    / "selected_checkpoint"
                ).exists()
            )

    def test_development_and_confirmatory_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            UC.validate_mode(config_args(development=False, confirmatory=False))
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            UC.validate_mode(config_args(development=True, confirmatory=True))

    def test_confirmatory_rejects_changed_frozen_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            development = config_args(seed=1)
            path = Path(directory) / "frozen.json"
            path.write_text(
                json.dumps({"configuration": UC.configuration_payload(development)}),
                encoding="utf-8",
            )
            changed = config_args(
                seed=1,
                development=False,
                confirmatory=True,
                frozen_development_config=path,
                forget_weight=3.0,
            )
            with self.assertRaisesRegex(ValueError, "differs"):
                UC.validate_mode(changed)

    def test_strict_json_nonfinite_values_become_audited_nulls(self):
        normalized, replacements = UC.strict_json_normalize(
            {"nested": [float("nan"), float("inf"), float("-inf"), 1.0]}
        )
        self.assertEqual(normalized, {"nested": [None, None, None, 1.0]})
        self.assertEqual(
            [row["path"] for row in replacements],
            ["/nested/0", "/nested/1", "/nested/2"],
        )
        json.dumps(normalized, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
