"""RSNR-V2 keeps the V1A adapter frozen while unfreezing only the substrate."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rsnr_v1a_frozen_spec as v1a  # noqa: E402
import rsnr_v2_frozen_spec as v2  # noqa: E402


def _adapter_checkpoint(**overrides):
    payload = {
        "variant": v1a.FROZEN_ARCHITECTURE["variant"],
        "intervention_site": v1a.FROZEN_ARCHITECTURE["intervention_site"],
        "abstention": v1a.FROZEN_ARCHITECTURE["abstention"],
        "adapter_rank": v1a.FROZEN_ARCHITECTURE["adapter_rank"],
        "adapter_alpha": v1a.FROZEN_ARCHITECTURE["adapter_alpha"],
        "transformer_weights_modified": False,
        "lm_head_weights_modified": True,
        "adapter_state_dict": {},
    }
    payload.update(overrides)
    return payload


def _stage0_config(**overrides):
    payload = {
        "reference_anchor": "abstention",
        "target_new_used": False,
        "abstention_anchor": {
            "anchor": "abstention",
            "target_new_used": False,
            "degenerate_fallback_count": 0,
        },
    }
    payload.update(overrides)
    return payload


class AdapterInheritanceTest(unittest.TestCase):
    def test_adapter_geometry_is_inherited_verbatim_from_v1a(self):
        for key in v2.INHERITED_ADAPTER_KEYS:
            self.assertIn(key, v1a.FROZEN_ARCHITECTURE, f"{key} missing from V1A spec")

    def test_v1a_adapter_on_an_edited_substrate_is_accepted(self):
        # This is the whole point of V2: lm_head_weights_modified is now True.
        v2.validate_adapter_checkpoint(_adapter_checkpoint())

    def test_v1a_validator_rejects_the_same_checkpoint(self):
        with self.assertRaises(RuntimeError):
            v1a.validate_adapter_checkpoint(_adapter_checkpoint())

    def test_changed_adapter_rank_is_rejected(self):
        with self.assertRaises(RuntimeError):
            v2.validate_adapter_checkpoint(_adapter_checkpoint(adapter_rank=8))

    def test_moved_intervention_site_is_rejected(self):
        with self.assertRaises(RuntimeError):
            v2.validate_adapter_checkpoint(
                _adapter_checkpoint(intervention_site="layer_24_residual")
            )

    def test_transformer_edits_are_still_forbidden(self):
        with self.assertRaises(RuntimeError):
            v2.validate_adapter_checkpoint(
                _adapter_checkpoint(transformer_weights_modified=True)
            )

    def test_missing_adapter_weights_are_rejected(self):
        payload = _adapter_checkpoint()
        del payload["adapter_state_dict"]
        with self.assertRaises(RuntimeError):
            v2.validate_adapter_checkpoint(payload)


class Stage0ContractTest(unittest.TestCase):
    def test_abstention_anchored_config_is_accepted(self):
        v2.validate_stage0_config(_stage0_config())

    def test_target_new_anchor_is_rejected(self):
        with self.assertRaises(RuntimeError) as ctx:
            v2.validate_stage0_config(
                _stage0_config(reference_anchor="target_new", target_new_used=True)
            )
        self.assertIn("reference_anchor", str(ctx.exception))

    def test_degenerate_fallback_use_is_rejected(self):
        config = _stage0_config()
        config["abstention_anchor"]["degenerate_fallback_count"] = 3
        with self.assertRaises(RuntimeError) as ctx:
            v2.validate_stage0_config(config)
        self.assertIn("degenerate", str(ctx.exception))

    def test_missing_anchor_summary_is_rejected(self):
        with self.assertRaises(RuntimeError):
            v2.validate_stage0_config(_stage0_config(abstention_anchor=None))


class SpecShapeTest(unittest.TestCase):
    def test_spec_records_that_base_is_no_longer_recovered(self):
        spec = v2.frozen_spec()
        arch = spec["architecture"]
        self.assertFalse(arch["base_recovery_on_gate_off"])
        self.assertTrue(arch["knowledge_removed_from_weights"])
        self.assertEqual(arch["gate_off_reference"], "stage0_model")

    def test_stage0_is_unconditional_and_stage1_is_conditional(self):
        arch = v2.frozen_spec()["architecture"]
        self.assertFalse(arch["stage0"]["conditional"])
        self.assertTrue(arch["stage1"]["conditional"])
        self.assertFalse(arch["stage0"]["target_new_used"])

    def test_sweep_needs_no_retraining_and_scale_zero_is_v1a(self):
        sweep = v2.frozen_spec()["sweep"]
        self.assertEqual(sweep["retrains_per_point"], 0)
        self.assertIn(0.0, sweep["default_scales"])
        self.assertEqual(sweep["scale_zero_reproduces"], "RSNR-V1A-PreHead")

    def test_spec_embeds_the_inherited_v1a_version(self):
        spec = v2.frozen_spec()
        self.assertEqual(
            spec["inherited_v1a_spec"]["spec_version"], v1a.FROZEN_SPEC_VERSION
        )
        self.assertEqual(
            spec["architecture"]["stage1"]["inherits_spec_version"],
            v1a.FROZEN_SPEC_VERSION,
        )

    def test_ppl_contamination_discipline_is_recorded(self):
        self.assertTrue(v2.frozen_spec()["sweep"]["ppl_text_disjoint_from_training"])


if __name__ == "__main__":
    unittest.main()
