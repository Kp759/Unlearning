import argparse
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
REPOSITORY_ROOT = ROOT.parent
sys.path.insert(0, str(SCRIPTS))

import rwku_artifact_access as ACCESS  # noqa: E402
import rwku_checkpoint_receipt as RECEIPT  # noqa: E402
import rwku_zerounlearn_target_only as MODULE  # noqa: E402


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token = "<pad>"
    pad_token_id = 0
    unk_token = "<unk>"
    name_or_path = "/fake/tokenizer"

    def __len__(self):
        return 128

    def __call__(self, text, *, add_special_tokens=False):
        if text == self.eos_token and add_special_tokens is False:
            return {"input_ids": [self.eos_token_id]}
        return {"input_ids": [10]}

    @staticmethod
    def apply_chat_template(messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<user>{messages[0]['content']}</user><assistant>"

    def save_pretrained(self, destination):
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text(
            '{"eos_token": "<eos>"}\n', encoding="utf-8"
        )


def generated_view(
    *,
    fact_id="fact-1",
    view_id="view-1",
    style="direct question",
    query="What novel did Stephen King publish first?",
    answer="Carrie",
):
    return {
        "fact_id": fact_id,
        "view_id": view_id,
        "view_content_sha256": f"hash-{view_id}",
        "prompt_style": style,
        "query_type": style,
        "query": query,
        "canonical_sensitive_answer": answer,
        "sensitive_answer_alias": answer,
        "subject": "Stephen King",
        "source_file": "generated_raw_corpus.json",
        "level": "generated",
        "training_allowed": True,
    }


def staged_args(root: Path, bundle: Path, generator: Path, **updates):
    values = {
        "stage": "prepare",
        "experiment_id": "zero-seed0",
        "seed": 0,
        "model_path": root / "model",
        "model_revision": "revision",
        "generated_entity_fact_bundle": bundle,
        "generator_receipt": generator,
        "zero_root": REPOSITORY_ROOT / "ZeroUnlearn",
        "zero_hparams": (
            REPOSITORY_ROOT
            / "ZeroUnlearn"
            / "hparams"
            / "ZeroUnlearn"
            / "Llama-3.2-3B-Instruct.json"
        ),
        "output_root": root / "outputs",
        "data_root": ROOT / "data" / "rwku",
        "wikidata_dir": ROOT / "data" / "wikidata",
        "dtype": "bf16",
        "no_download": True,
        "confirmatory": True,
        "add_external_retain_anchors": False,
        "external_retain_artifact": [],
        "eval_batch_size": 4,
        "forget_eval_limit": None,
        "adversarial_eval_limit": None,
        "mia_eval_limit": None,
        "neighbor_eval_limit": None,
        "utility_eval_limit": None,
        "dry_run": True,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def write_generated_artifacts(root: Path):
    bundle_path = root / "generated_training_bundle.json"
    bundle = ACCESS.make_artifact(
        "training_bundle",
        {"views": [generated_view()]},
        protocol_label=ACCESS.TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=ACCESS.TARGET_ONLY_PROTOCOL_STATUS,
        metadata={
            "seed": 0,
            "entity_id": "rwku:1_Stephen_King",
            "subject": "Stephen King",
        },
    )
    ACCESS.write_artifact(bundle_path, bundle)
    generator_path = root / "generator_receipt.json"
    generator = ACCESS.make_artifact(
        "generator_receipt",
        {
            "status": "complete",
            "official_rwku_records_accessed": False,
            "target_entity": "Stephen King",
            "entity_id": "rwku:1_Stephen_King",
            "final_entity_fact_bundle_sha256": bundle["sha256"],
        },
        protocol_label=ACCESS.TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=ACCESS.TARGET_ONLY_PROTOCOL_STATUS,
        metadata={
            "seed": 0,
            "entity_id": "rwku:1_Stephen_King",
            "subject": "Stephen King",
        },
    )
    ACCESS.write_artifact(generator_path, generator)
    return bundle_path, generator_path


class RequestAdapterTests(unittest.TestCase):
    def test_primary_protocol_uses_one_direct_request_per_fact(self):
        views = [
            generated_view(),
            generated_view(
                view_id="view-duplicate",
                query="Which first novel did Stephen King publish?",
            ),
            generated_view(
                view_id="view-cloze",
                style="cloze",
                query="Stephen King's first novel was ___.",
            ),
            generated_view(
                fact_id="fact-2",
                view_id="view-2",
                query="Where was Stephen King born?",
                answer="Portland, Maine",
            ),
        ]
        requests, audit = MODULE.compile_zero_unlearn_requests(
            views,
            FakeTokenizer(),
            target_subject="Stephen King",
        )
        self.assertEqual(len(requests), 2)
        self.assertEqual(audit["fact_count"], 2)
        self.assertEqual(audit["selected_view_count"], 2)
        self.assertEqual(len(audit["duplicate_views"]), 1)
        self.assertEqual(len(audit["skipped_views"]), 1)
        self.assertTrue(audit["one_request_per_fact_id"])
        for request in requests:
            self.assertEqual(request["prompt"].count("{}"), 1)
            self.assertEqual(request["target_new"], {"str": "<eos>"})
        self.assertEqual(requests[0]["target_true"], {"str": "Carrie"})

    def test_subject_must_compile_to_exactly_one_placeholder(self):
        with self.assertRaisesRegex(ValueError, "subject exactly once"):
            MODULE.compile_zero_unlearn_requests(
                [
                    generated_view(
                        query="Did Stephen King say Stephen King wrote Carrie?"
                    )
                ],
                FakeTokenizer(),
                target_subject="Stephen King",
            )

    def test_sensitive_and_eos_guards_fail_closed(self):
        views = [generated_view()]
        requests, _ = MODULE.compile_zero_unlearn_requests(
            views, FakeTokenizer(), target_subject="Stephen King"
        )
        requests[0]["target_true"] = {"str": "wrong"}
        with self.assertRaisesRegex(RuntimeError, "sensitive answer"):
            MODULE.validate_zero_unlearn_requests(views, requests, eos_token="<eos>")


class ProtocolGuardTests(unittest.TestCase):
    def test_hparams_require_original_layers_and_down_projection(self):
        source = json.loads(
            (
                REPOSITORY_ROOT
                / "ZeroUnlearn"
                / "hparams"
                / "ZeroUnlearn"
                / "Llama-3.2-3B-Instruct.json"
            ).read_text(encoding="utf-8")
        )
        MODULE.validate_zero_hparams_payload(source)
        with self.assertRaisesRegex(ValueError, "layers"):
            MODULE.validate_zero_hparams_payload({**source, "layers": [15, 16, 17]})
        with self.assertRaisesRegex(ValueError, "down_proj"):
            MODULE.validate_zero_hparams_payload(
                {**source, "rewrite_module_tmp": "lm_head"}
            )

    def test_confirmatory_evaluation_rejects_all_smoke_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, generator = write_generated_artifacts(root)
            args = staged_args(
                root,
                bundle,
                generator,
                stage="evaluate",
                forget_eval_limit=1,
            )
            with self.assertRaisesRegex(ValueError, "cannot use smoke limits"):
                MODULE.validate_args(args)

    def test_primary_and_external_retain_labels_are_unambiguous(self):
        self.assertEqual(
            MODULE.METHOD,
            "Original ZeroUnlearn with RWKU target-generated entity corpus",
        )
        self.assertEqual(
            MODULE.PROTOCOL_STATUS,
            "official_rwku_protocol_different_model_zerounlearn_corpus_extension",
        )
        self.assertFalse(
            MODULE.method_label(add_external_retain_anchors=False).endswith(
                MODULE.EXTERNAL_RETAIN_LABEL
            )
        )
        self.assertIn(
            MODULE.EXTERNAL_RETAIN_LABEL,
            MODULE.method_label(add_external_retain_anchors=True),
        )

    def test_fresh_training_load_is_fp32_base_snapshot(self):
        class FakeModel:
            def __init__(self):
                self.config = SimpleNamespace(use_cache=True)
                self.float_called = False

            def to(self, device):
                self.device = device
                return self

            def float(self):
                self.float_called = True
                return self

            def eval(self):
                return self

        fake = FakeModel()
        loader = mock.Mock(return_value=fake)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model"
            model_path.mkdir()
            args = SimpleNamespace(
                model_path=model_path,
                model_revision="revision",
                no_download=True,
            )
            with mock.patch(
                "transformers.AutoModelForCausalLM.from_pretrained", loader
            ):
                loaded = MODULE.load_fresh_base_model_for_training(args)
        self.assertIs(loaded, fake)
        self.assertTrue(fake.float_called)
        self.assertEqual(fake.device, "cuda:0")
        self.assertIs(loader.call_args.kwargs["torch_dtype"], torch.float32)
        self.assertEqual(loader.call_args.args[0], str(model_path))

    def test_setting5e_and_lm_head_repair_are_not_imported_or_called(self):
        source = Path(MODULE.__file__).read_text(encoding="utf-8")
        self.assertNotIn("run_protected_lm_head_repair", source)
        self.assertNotIn("gagd.train_mode", source)
        self.assertIn('"setting5e_invoked": False', source)
        self.assertIn('"lm_head_repair_invoked": False', source)


class StagedAccessTests(unittest.TestCase):
    def test_prepare_hashes_bundle_without_opening_training_or_official_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, generator = write_generated_artifacts(root)
            args = staged_args(root, bundle, generator)
            real_read = MODULE.read_artifact
            opened = []

            def tracked_read(path, **kwargs):
                opened.append(
                    (Path(path), kwargs.get("stage"), kwargs.get("expected_role"))
                )
                return real_read(path, **kwargs)

            with mock.patch.object(
                MODULE, "read_artifact", side_effect=tracked_read
            ), mock.patch.object(
                MODULE,
                "ensure_target_data",
                side_effect=AssertionError("official RWKU opened during prepare"),
            ):
                MODULE.prepare_stage(args)
            self.assertNotIn(bundle.resolve(), {row[0].resolve() for row in opened})
            self.assertEqual(opened[0][2], "generator_receipt")
            state = MODULE._read_state(args)
            self.assertEqual(state["state"], "PREPARED")
            self.assertFalse(state["prepare_audit"]["official_evaluation_rows_opened"])
            self.assertFalse(state["prepare_audit"]["training_bundle_payload_opened"])

    def test_train_dry_run_compiles_requests_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, generator = write_generated_artifacts(root)
            model_path = root / "model"
            model_path.mkdir()
            args = staged_args(root, bundle, generator, model_path=model_path)
            MODULE.prepare_stage(args)
            args.stage = "train"
            with mock.patch.object(
                MODULE, "load_tokenizer", return_value=FakeTokenizer()
            ), mock.patch.object(
                MODULE,
                "load_fresh_base_model_for_training",
                side_effect=AssertionError("model loaded in dry-run"),
            ), mock.patch.object(
                MODULE,
                "ensure_target_data",
                side_effect=AssertionError("official RWKU opened during train"),
            ):
                MODULE.train_stage(args)
            self.assertEqual(MODULE._read_state(args)["state"], "PREPARED")
            manifest = json.loads(
                (MODULE._output_dir(args) / "compiled_requests.json").read_text()
            )
            self.assertEqual(len(manifest["primary_target_requests"]), 1)
            self.assertFalse(manifest["add_retain"])

    def test_primary_train_invokes_original_with_no_retain_moments(self):
        class TrainingModel:
            def __init__(self):
                self.config = SimpleNamespace(use_cache=False)

            def to(self, *args, **kwargs):
                return self

            def eval(self):
                return self

            def save_pretrained(self, destination, *, safe_serialization):
                self.saved_safely = safe_serialization
                path = Path(destination)
                path.mkdir(parents=True, exist_ok=True)
                (path / "model.safetensors").write_bytes(b"edited")

        class ParamsClass:
            @staticmethod
            def from_json(path):
                return SimpleNamespace(
                    layers=[16, 17, 18],
                    rewrite_module_tmp="model.layers.{}.mlp.down_proj",
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, generator = write_generated_artifacts(root)
            model_path = root / "model"
            model_path.mkdir()
            (model_path / "base.safetensors").write_bytes(b"base")
            args = staged_args(
                root,
                bundle,
                generator,
                model_path=model_path,
                dry_run=False,
            )
            MODULE.prepare_stage(args)
            args.stage = "train"
            model = TrainingModel()
            apply_original = mock.Mock(return_value=(model, {}))
            with mock.patch.object(
                MODULE, "load_tokenizer", return_value=FakeTokenizer()
            ), mock.patch.object(
                MODULE, "load_fresh_base_model_for_training", return_value=model
            ), mock.patch.object(
                MODULE.zero_bridge,
                "import_original_zerounlearn",
                return_value=(ParamsClass, apply_original),
            ), mock.patch.object(
                MODULE,
                "ensure_target_data",
                side_effect=AssertionError("official RWKU opened during train"),
            ):
                MODULE.train_stage(args)

            invocation = apply_original.call_args.kwargs
            self.assertEqual(invocation["retain_requests"], [])
            self.assertFalse(invocation["add_retain"])
            self.assertEqual(invocation["edit_layer_nums"], 3)
            self.assertFalse(invocation["use_h"])
            self.assertEqual(len(invocation["unlearn_requests"]), 1)
            self.assertEqual(
                invocation["unlearn_requests"][0]["target_true"],
                {"str": "Carrie"},
            )
            self.assertEqual(
                invocation["unlearn_requests"][0]["target_new"],
                {"str": "<eos>"},
            )
            state = MODULE._read_state(args)
            self.assertEqual(state["state"], "CHECKPOINT_FROZEN")
            receipt = RECEIPT.load_receipt(MODULE._receipt_path(args))
            self.assertEqual(receipt["protocol_status"], MODULE.PROTOCOL_STATUS)
            self.assertFalse(receipt["method_configuration"]["add_retain"])

    def test_evaluation_opens_receipt_before_official_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, generator = write_generated_artifacts(root)
            model_path = root / "model"
            model_path.mkdir()
            (model_path / "base.safetensors").write_bytes(b"base")
            args = staged_args(
                root,
                bundle,
                generator,
                model_path=model_path,
                dry_run=False,
            )
            MODULE.prepare_stage(args)
            checkpoint = MODULE._output_dir(args) / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(b"edited")
            RECEIPT.create_checkpoint_receipt(
                destination=MODULE._receipt_path(args),
                experiment_id=args.experiment_id,
                protocol_label=ACCESS.TARGET_ONLY_PROTOCOL_LABEL,
                protocol_status=MODULE.PROTOCOL_STATUS,
                target_entity="Stephen King",
                target_entity_id="rwku:1_Stephen_King",
                base_model_identity=MODULE._model_identity(
                    model_path, args.model_revision
                ),
                base_model_revision=args.model_revision,
                tokenizer_identity={"source_file_sha256": {}},
                checkpoint_paths=[checkpoint],
                training_bundle_path=bundle,
                optimization_protection_path=None,
                mcf_retain_optimization_paths=[],
                mcf_repair_gate_paths=[],
                matched_protection_train_path=None,
                matched_protection_gate_path=None,
                method_configuration={
                    "method": MODULE.METHOD,
                    "add_retain": False,
                },
                implementation_files=[MODULE.SCRIPT_PATH],
                sampler_provenance={"one_request_per_fact_id": True},
                generator_receipt_path=generator,
                official_locked_eval_path=(
                    MODULE._output_dir(args) / "official_locked_eval.json"
                ),
                confirmatory=True,
                additional_artifact_paths={
                    "base_model_source": model_path,
                    "zero_hparams": args.zero_hparams,
                },
            )
            MODULE._write_state(
                args,
                "CHECKPOINT_FROZEN",
                checkpoint_receipt=str(MODULE._receipt_path(args).resolve()),
                checkpoint_path=str(checkpoint.resolve()),
                official_evaluation_opened=False,
            )
            locked = ACCESS.read_artifact(
                MODULE._output_dir(args) / "official_locked_eval.json",
                stage="evaluate",
                evaluation=True,
                expected_role="official_locked_eval",
            )
            file_hashes = {
                filename: descriptor["sha256"]
                for filename, descriptor in locked["payload"]["files"].items()
            }
            datasets = {
                "forget_level1.json": [
                    {
                        "query": "Stephen King's first novel was ___?",
                        "answer": "Carrie",
                        "level": "1",
                        "type": "cloze",
                        "subject": "Stephen King",
                    }
                ],
                "forget_level2.json": [
                    {
                        "query": "What was Stephen King's first novel?",
                        "answer": "Carrie",
                        "level": "2",
                        "type": "direct question",
                        "subject": "Stephen King",
                    }
                ],
                "forget_level3.json": [
                    {
                        "query": "Was Carrie Stephen King's first novel?",
                        "answer": "yes",
                        "level": "3",
                        "type": "affirmative suffix",
                        "subject": "Stephen King",
                    }
                ],
            }
            target = MODULE.target_for_seed(0)
            events = []
            real_open = MODULE.open_official_evaluation

            def tracked_open(path, *, experiment_id):
                events.append("receipt_opened")
                return real_open(path, experiment_id=experiment_id)

            def tracked_official_loader(*loader_args, **loader_kwargs):
                current = RECEIPT.load_receipt(MODULE._receipt_path(args))
                self.assertEqual(current["state"], "OFFICIAL_EVALUATION_OPENED")
                events.append("official_rows_loaded")
                return target, datasets, file_hashes

            evaluation_results = [
                {"retain_reference_mean_logprobs": {"retain": -1.0}},
                {"retain_reference_mean_logprobs": {"retain": -1.1}},
            ]
            with mock.patch.object(
                MODULE, "open_official_evaluation", side_effect=tracked_open
            ), mock.patch.object(
                MODULE, "ensure_target_data", side_effect=tracked_official_loader
            ), mock.patch.object(
                MODULE, "load_tokenizer", return_value=FakeTokenizer()
            ), mock.patch.object(
                MODULE, "_load_evaluation_model", side_effect=[object(), object()]
            ), mock.patch.object(
                MODULE, "build_frozen_head_probe", return_value=object()
            ), mock.patch.object(
                MODULE, "evaluate_rwku", side_effect=evaluation_results
            ):
                args.stage = "evaluate"
                MODULE.evaluate_stage(args)

            self.assertEqual(events, ["receipt_opened", "official_rows_loaded"])
            self.assertEqual(MODULE._read_state(args)["state"], "EVALUATION_COMPLETE")
            completed = RECEIPT.load_receipt(MODULE._receipt_path(args))
            self.assertEqual(completed["state"], "EVALUATION_COMPLETE")


class ExtendedReceiptTests(unittest.TestCase):
    def make_receipt(self, root: Path):
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "weights.bin").write_bytes(b"checkpoint")
        model = root / "model"
        model.mkdir()
        (model / "weights.bin").write_bytes(b"base")
        hparams = root / "hparams.json"
        hparams.write_text("{}\n", encoding="utf-8")
        implementation = root / "implementation.py"
        implementation.write_text("x = 1\n", encoding="utf-8")
        bundle_path, generator_path = write_generated_artifacts(root)
        locked_path = root / "official_locked_eval.json"
        ACCESS.write_artifact(
            locked_path,
            ACCESS.make_artifact(
                "official_locked_eval",
                {"files": {}},
                protocol_label=ACCESS.TARGET_ONLY_PROTOCOL_LABEL,
                protocol_status=MODULE.PROTOCOL_STATUS,
            ),
        )
        destination = root / "receipt.json"
        RECEIPT.create_checkpoint_receipt(
            destination=destination,
            experiment_id="experiment",
            protocol_label=ACCESS.TARGET_ONLY_PROTOCOL_LABEL,
            protocol_status=MODULE.PROTOCOL_STATUS,
            target_entity="Stephen King",
            target_entity_id="rwku:1_Stephen_King",
            base_model_identity={"path": str(model)},
            base_model_revision="revision",
            tokenizer_identity={"id": "tokenizer"},
            checkpoint_paths=[checkpoint],
            training_bundle_path=bundle_path,
            optimization_protection_path=None,
            mcf_retain_optimization_paths=[],
            mcf_repair_gate_paths=[],
            matched_protection_train_path=None,
            matched_protection_gate_path=None,
            method_configuration={"layers": [16, 17, 18]},
            implementation_files=[implementation],
            sampler_provenance={"one_request_per_fact_id": True},
            generator_receipt_path=generator_path,
            official_locked_eval_path=locked_path,
            confirmatory=True,
            additional_artifact_paths={
                "base_model_source": model,
                "zero_hparams": hparams,
            },
        )
        return destination, model, bundle_path, hparams

    def test_receipt_detects_changed_model_bundle_and_hparams(self):
        mutations = (
            (
                1,
                lambda path: (path / "weights.bin").write_bytes(b"changed"),
                "base_model_source",
            ),
            (
                2,
                lambda path: path.write_text(path.read_text() + " "),
                "training_bundle",
            ),
            (3, lambda path: path.write_text('{"changed": true}\n'), "zero_hparams"),
        )
        for selected, mutate, expected in mutations:
            with self.subTest(
                expected=expected
            ), tempfile.TemporaryDirectory() as directory:
                receipt, model, bundle, hparams = self.make_receipt(Path(directory))
                target = {1: model, 2: bundle, 3: hparams}[selected]
                mutate(target)
                with self.assertRaisesRegex(RECEIPT.CheckpointReceiptError, expected):
                    RECEIPT.open_official_evaluation(
                        receipt, experiment_id="experiment"
                    )

    def test_post_evaluation_model_modification_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, _, _, _ = self.make_receipt(Path(directory))
            RECEIPT.open_official_evaluation(receipt, experiment_id="experiment")
            with self.assertRaisesRegex(
                RECEIPT.CheckpointReceiptError, "forbidden after checkpoint freeze"
            ):
                RECEIPT.assert_model_modification_allowed(
                    receipt, experiment_id="experiment"
                )


if __name__ == "__main__":
    unittest.main()
