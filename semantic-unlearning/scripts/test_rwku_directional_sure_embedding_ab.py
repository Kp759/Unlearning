#!/usr/bin/env python3
"""No-model smoke tests for Directional SURE embedding ablations A/B."""
from pathlib import Path

import rwku_directional_sure_embedding_ab as ab

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output_rows = [1, 2, 3, 4]
    content_rows = [2, 4]
    assert ab.resolve_input_rows("A", output_rows, content_rows) == []
    assert ab.resolve_input_rows("B", output_rows, content_rows) == [2, 4]

    try:
        ab.resolve_input_rows("B", [1, 2], [2, 3])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Variant B accepted an input row outside output row scope")

    cfg_a = ab.load_variant_configuration(
        ROOT / "config" / "rwku" / "directional_sure_variant_a_seed0.json",
        variant=ab.VARIANT_A,
        expected_schema="rwku_directional_sure_variant_a_configuration_v1",
        expected_experiment_id="rwku-directional-sure-variant-a-stephen-king-seed0",
    )
    cfg_b = ab.load_variant_configuration(
        ROOT / "config" / "rwku" / "directional_sure_variant_b_seed0.json",
        variant=ab.VARIANT_B,
        expected_schema="rwku_directional_sure_variant_b_configuration_v1",
        expected_experiment_id="rwku-directional-sure-variant-b-stephen-king-seed0",
    )

    assert cfg_a["trainable_components"]["sensitive_input_embedding_rows"] is False
    assert cfg_b["trainable_components"]["sensitive_input_embedding_rows"] is True
    assert cfg_a["lm_head_row_policy"] == cfg_b["lm_head_row_policy"]
    assert cfg_a["optimization"] == cfg_b["optimization"]
    assert cfg_a["stage2"] == cfg_b["stage2"]
    assert cfg_a["acceptance"] == cfg_b["acceptance"]
    assert cfg_a["data_boundary"] == cfg_b["data_boundary"]
    assert cfg_a["level3_representation_repair_enabled"] is False
    assert cfg_b["level3_representation_repair_enabled"] is False

    print("RWKU Directional SURE embedding A/B smoke test PASS")
    print("  Variant A: input embeddings frozen; all-sensitive LM head directional")
    print("  Variant B: content-safe input rows; all-sensitive LM head directional")
    print("  Common optimization / Stage 2 / acceptance / data boundary: identical")


if __name__ == "__main__":
    main()
