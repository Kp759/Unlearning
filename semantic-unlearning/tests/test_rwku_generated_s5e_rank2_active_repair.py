from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gagd_active_case_repair as generic_repair
import mquake_gagd_setting5e_multiroot_active_repair as mquake
import rwku_experiment as existing_rwku
import rwku_generated_s5e_rank2_active_repair as method
import rwku_setting5e_utility_controlled as existing_uc


class TinyTokenizer:
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = 0
    unk_token_id = 3
    all_special_ids = [0, 1, 2, 3, 4]

    def convert_tokens_to_ids(self, value):
        return 4 if value == "<|eot_id|>" else self.unk_token_id

    def encode(self, value, add_special_tokens=False):
        return [5] if "protected" in value else [6]


class TinyTiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(4, 4)
        self.embed = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8, bias=False)
        self.head.weight = self.embed.weight
        self.config = type("Config", (), {"tie_word_embeddings": True})()

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, value):
        self.head = value


class GeneratedS5ERank2Tests(unittest.TestCase):
    def test_frozen_scientific_recipe(self):
        self.assertEqual(method.SETTING5_STEPS, 600)
        self.assertEqual(method.SETTING5_BATCH_SIZE, 1)
        self.assertEqual(method.SETTING5_RETAIN_BATCH_SIZE, 4)
        self.assertEqual(method.SETTING5_LR, 1e-4)
        self.assertEqual(method.REPAIR_RANK, 2)
        self.assertEqual(method.REPAIR_STEPS, 100)
        self.assertEqual(method.REPAIR_LR, 5e-3)
        self.assertEqual(method.ACTIVE_MARGIN, 0.25)

    def test_rank2_is_not_rwku_projection_rank(self):
        self.assertEqual(method.REPAIR_RANK, 2)
        self.assertEqual(method.PROTECTED_PROJECTION_RANK, 256)
        args = argparse.Namespace(
            experiment_id="rwku-s5e600-rank2-active-sk-v3atomic-seed0-v1",
            seed=0, model_path=Path("model"), model_revision="revision",
            dtype="bf16", generated_training_bundle=Path("bundle"),
            generator_receipt=Path("receipt"), mcf_path=Path("mcf"),
            mcf_retain_num=1000, mcf_gate_num=200,
            protection_source=[Path("mcf")], protection_vocabulary=None,
            wikidata_dir=Path("wikidata"), eval_batch_size=4,
            output_root=Path("out"), data_root=Path("rwku"),
            gradient_checkpointing=True, no_download=True,
        )
        config = method.configuration(args)
        self.assertEqual(config["active_repair"]["repair_rank"], 2)
        self.assertFalse(config["active_repair"]["legacy_rwku_projection_rank_used"])

    def test_shared_rank2_parameterization(self):
        basis = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        module = generic_repair.SelectedRowDelta(
            3, 4, direction_basis=basis, retained_basis=None,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            module.coefficients.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, 5.0]]))
        delta = module.effective_delta()
        self.assertTrue(torch.equal(delta[:, 2:], torch.zeros(3, 2)))
        self.assertEqual(tuple(module.coefficients.shape), (3, 2))

    def test_active_points_require_generated_provenance(self):
        point = existing_uc.TrainingPoint(
            fact_id="", view_id="view", prompt_style="cloze", prompt="p",
            sensitive_answer="answer", neutral_answer="eos", subject="subject",
            source_record_sha256="abc",
        )
        with self.assertRaisesRegex(ValueError, "provenance"):
            method.validate_active_provenance(point, bundle_sha256="bundle")

    def test_official_rwku_rows_cannot_be_active(self):
        point = existing_uc.TrainingPoint(
            fact_id="fact", view_id="view", prompt_style="cloze", prompt="p",
            sensitive_answer="answer", neutral_answer="eos", subject="subject",
            source_record_sha256="forget_level1.json",
        )
        with self.assertRaisesRegex(ValueError, "Official RWKU"):
            method.validate_active_provenance(point, bundle_sha256="bundle")

    def test_special_rows_are_excluded(self):
        selected, excluded = method.exclude_special_rows(
            TinyTokenizer(), [0, 1, 2, 3, 4, 5, 6]
        )
        self.assertEqual(selected, [5, 6])
        self.assertEqual(excluded, [0, 1, 2, 3, 4])

    def test_protected_answer_rows_are_identified_for_exclusion(self):
        example = type("Example", (), {"answer": "protected answer"})()
        self.assertEqual(
            method.protected_answer_row_ids(TinyTokenizer(), [example]), {5}
        )

    def test_transformer_and_input_embeddings_are_frozen(self):
        model = TinyTiedModel()
        output = generic_repair.freeze_model_for_output_repair(model)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(model.get_input_embeddings().weight.requires_grad)
        self.assertNotEqual(
            output.weight.data_ptr(), model.get_input_embeddings().weight.data_ptr()
        )

    def test_evaluation_requires_frozen_state(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                output_root=Path(directory), experiment_id="experiment"
            )
            target = method.run_dir(args)
            target.mkdir(parents=True)
            method.write_state(args, "PREPARED")
            with self.assertRaisesRegex(ValueError, "CHECKPOINT_FROZEN"):
                method._verify_evaluation(args)

    def test_evaluation_opens_before_loading_and_never_materializes(self):
        source = inspect.getsource(method.evaluate_stage)
        self.assertLess(
            source.index("open_official_evaluation"),
            source.index("ensure_target_data"),
        )
        self.assertNotIn("_materialize_scale", source)
        self.assertNotIn("optimizer", source.lower())

    def test_dry_preflight_does_not_load_model_or_official_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            for name in (
                "generated_raw_corpus.json", "generated_entity_fact_catalog.json",
                "generated_training_bundle.json", "generator_receipt.json",
                "generated_atomic_facts.json", "generation_diagnostics.json",
            ):
                (corpus / name).write_text("{}\n", encoding="utf-8")
            manifest_rows = []
            for path in sorted(corpus.iterdir()):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                manifest_rows.append(f"{digest}  {path.name}")
            (corpus / "corpus_sha256_manifest.txt").write_text(
                "\n".join(manifest_rows) + "\n", encoding="utf-8"
            )
            model = root / "model"
            model.mkdir()
            mcf = root / "multi_counterfact.json"
            mcf.write_text("[]\n", encoding="utf-8")
            args = argparse.Namespace(
                experiment_id="rwku-s5e600-rank2-active-sk-v3atomic-seed0-v1",
                seed=0, model_path=model, model_revision="revision", dtype="bf16",
                generated_training_bundle=corpus / "generated_training_bundle.json",
                generator_receipt=corpus / "generator_receipt.json", mcf_path=mcf,
                mcf_retain_num=1000, mcf_gate_num=200,
                protection_source=[mcf], protection_vocabulary=None,
                wikidata_dir=root / "wikidata", eval_batch_size=4,
                output_root=root / "out", data_root=root / "rwku",
                gradient_checkpointing=True, no_download=True,
            )
            method.preflight(args)

    def test_existing_method_defaults_remain_unchanged(self):
        self.assertEqual(existing_rwku.SETTING5_MODE, "emb_lm_all_restore_post_training_true")
        self.assertEqual(existing_uc.DEFAULT_EXPOSURES, (2, 4, 6, 8, 10, 12, 15, 20))
        parser = mquake.build_parser()
        parsed = parser.parse_args([
            "--model-path", "model", "--output-dir", "out",
        ])
        self.assertEqual(parsed.repair_rank, 0)

    def test_reference_config_matches_rank2_recipe(self):
        path = ROOT / "config" / "controlled_unlearning" / "mcf_margin025_rank2.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["setting5e"]["steps"], method.SETTING5_STEPS)
        self.assertEqual(config["active_repair"]["repair_rank"], method.REPAIR_RANK)
        self.assertEqual(config["active_repair"]["repair_steps"], method.REPAIR_STEPS)


if __name__ == "__main__":
    unittest.main()
