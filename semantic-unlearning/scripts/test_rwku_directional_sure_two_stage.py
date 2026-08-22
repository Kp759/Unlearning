#!/usr/bin/env python3
"""CPU smoke tests for pure two-stage Directional SURE bookkeeping."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

import rwku_directional_sure_two_stage as two


def main() -> None:
    cfg = two.load_configuration(
        Path(__file__).resolve().parents[1]
        / "config"
        / "rwku"
        / "directional_sure_two_stage_seed0.json"
    )
    assert cfg["stage2"]["enabled"] is True
    assert cfg["level3_representation_repair_enabled"] is False
    assert cfg["trainable_components"]["transformer_parameters"] is False
    assert cfg["trainable_components"]["mlp_parameters"] is False
    assert cfg["trainable_components"]["lora_parameters"] is False

    atomic = {
        "pairwise_margin_failure_positions": [3, 1, 3],
        "prompt_instances": [{}, {}, {}, {}, {}],
        "direct_margin_failures": 1,
        "generated_subject_margin_failures": 1,
        "direct_failures": 1,
        "generated_subject_failures": 1,
        "minimum_overall_separation": -0.5,
    }
    assert two.residual_prompt_positions(atomic) == [1, 3]

    cases = [
        SimpleNamespace(record_position=0),
        SimpleNamespace(record_position=1),
        SimpleNamespace(record_position=1),
        SimpleNamespace(record_position=2),
        SimpleNamespace(record_position=3),
    ]
    residual_indices = two.residual_case_indices(cases, [1, 3])
    assert residual_indices == [1, 2, 4]

    class Tokenizer:
        all_special_ids = [99]

    tids = torch.tensor([10, 11, 12, 13, 14], dtype=torch.long)
    rows = two.residual_rows_from_cases(
        Tokenizer(), cases, tids, residual_indices, [10, 11, 12, 13, 14]
    )
    assert rows == [11, 12, 14]

    sparse = SimpleNamespace(
        selected_input_rows=(10, 11, 12, 13, 14),
        selected_output_rows=(10, 11, 12, 13, 14),
        input_delta=torch.zeros((5, 4), dtype=torch.float32),
        output_delta=torch.zeros((5, 4), dtype=torch.float32),
    )
    input_mask = two.row_mask(sparse, rows, for_output=False)
    output_mask = two.row_mask(sparse, rows, for_output=True)
    assert input_mask[:, 0].tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]
    assert torch.equal(input_mask, output_mask)

    utility = {
        "utility_kl_mean": 0.001,
        "utility_kl_p95": 0.002,
        "utility_kl_max": 0.1,
    }
    norms = {"total_selected_row_delta_norm": 1.0}
    worse = dict(atomic)
    better = dict(atomic)
    better["direct_margin_failures"] = 0
    assert two.stage1_anchor_key(better, utility, norms, 100) < two.stage1_anchor_key(
        worse, utility, norms, 50
    )

    stage2_cfg = two.stage2_basis_cfg(cfg)
    assert stage2_cfg["optimization"]["sensitive_exclusive_basis_rank"] == 8
    assert stage2_cfg["optimization"]["protected_basis_rank"] == 32

    print("RWKU pure two-stage Directional SURE smoke test PASS")


if __name__ == "__main__":
    main()
