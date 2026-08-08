import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_rwku_bf16_drift as AUDIT  # noqa: E402
from rwku_artifact_access import sha256_file  # noqa: E402


def auditable_state(**updates):
    value = {
        "state": "NO_FEASIBLE_CANDIDATE",
        "protection_prepared": True,
        "official_evaluation_opened": False,
        "official_rwku_records_accessed": False,
    }
    value.update(updates)
    return value


class ReadOnlyAuditTests(unittest.TestCase):
    def test_official_evaluation_state_is_rejected(self):
        for updates in (
            {"official_evaluation_opened": True},
            {"official_rwku_records_accessed": True},
        ):
            with self.subTest(updates=updates):
                with self.assertRaisesRegex(ValueError, "official"):
                    AUDIT.validate_read_only_audit_state(
                        auditable_state(**updates)
                    )

    def test_official_rwku_gate_path_is_rejected_before_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = root / "forget_level1.json"
            official.write_text('{"must_not_be_opened": true}', encoding="utf-8")
            matched = root / "matched_protection_gate.json"
            matched.write_text("{}", encoding="utf-8")
            state = auditable_state(
                mcf_gate_manifest_path=str(official),
                mcf_gate_manifest_sha256=sha256_file(official),
                matched_protection_gate_path=str(matched),
                matched_protection_gate_sha256=sha256_file(matched),
            )
            with self.assertRaisesRegex(ValueError, "official/evaluation"):
                AUDIT.load_pre_freeze_gate_records(state)

    def test_candidate_must_be_a_saved_run_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            allowed = (
                run_dir
                / "utility_controlled_setting5"
                / "checkpoints"
                / "step_52"
            )
            allowed.mkdir(parents=True)
            self.assertEqual(
                AUDIT._candidate_checkpoint(run_dir, allowed), allowed.resolve()
            )
            outside = Path(directory) / "official_evaluation" / "checkpoint"
            outside.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "saved pre-freeze"):
                AUDIT._candidate_checkpoint(run_dir, outside)

    def test_selected_row_scaling_does_not_change_unselected_rows(self):
        import torch

        class TinyModel(torch.nn.Module):
            def __init__(self, offset):
                super().__init__()
                self.input = torch.nn.Embedding(4, 2)
                self.output = torch.nn.Linear(2, 4, bias=False)
                with torch.no_grad():
                    self.input.weight.copy_(
                        torch.arange(8, dtype=torch.float32).reshape(4, 2) + offset
                    )
                    self.output.weight.copy_(
                        torch.arange(8, dtype=torch.float32).reshape(4, 2) + offset
                    )

            def get_input_embeddings(self):
                return self.input

            def get_output_embeddings(self):
                return self.output

        base = TinyModel(0.0)
        candidate = TinyModel(4.0)
        before_input = candidate.input.weight.detach().clone()
        before_output = candidate.output.weight.detach().clone()
        AUDIT._scale_selected_rows(
            candidate,
            base,
            input_row_ids=[1],
            output_row_ids=[2],
            scale=0.5,
        )
        self.assertTrue(torch.equal(candidate.input.weight[0], before_input[0]))
        self.assertTrue(torch.equal(candidate.output.weight[0], before_output[0]))
        self.assertTrue(
            torch.equal(
                candidate.input.weight[1],
                base.input.weight[1] + 0.5 * (before_input[1] - base.input.weight[1]),
            )
        )
        self.assertTrue(
            torch.equal(
                candidate.output.weight[2],
                base.output.weight[2]
                + 0.5 * (before_output[2] - base.output.weight[2]),
            )
        )


if __name__ == "__main__":
    unittest.main()
