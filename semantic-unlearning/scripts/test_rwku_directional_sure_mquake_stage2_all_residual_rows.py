#!/usr/bin/env python3
"""No-model smoke tests for all-residual-row rank-8 MQuAKE-style RWKU Stage 2."""
from pathlib import Path

import torch

import rwku_directional_sure_two_stage_emb_ga_only as previous
import rwku_mquake_stage2_all_residual_helpers as experiment

ROOT = Path(__file__).resolve().parents[1]


class DummyTokenizer:
    all_special_ids = [0, 1]


def main() -> None:
    cfg = experiment.load_configuration(
        ROOT
        / "config"
        / "rwku"
        / "directional_sure_mquake_stage2_all_residual_rows_rank8_seed0.json"
    )
    old = previous.read_json(previous.DEFAULT_CONFIGURATION)

    # L1 remains exactly the previously tested content-sensitive GA-only setup.
    assert cfg["optimization"] == old["optimization"]
    assert cfg["trainable_components"] == old["trainable_components"]
    assert cfg["acceptance"] == old["acceptance"]
    assert cfg["acceptance"]["required_pairwise_margin"] == 0.01

    s2 = cfg["stage2"]
    assert s2["row_scope"] == (
        "all_non_special_sensitive_target_rows_implicated_by_level1_residual_prompts"
    )
    assert s2["parameter_scope"] == "lm_head_only_increment_over_frozen_level1_anchor"
    assert s2["embedding_parameters"] is False
    assert s2["transformer_parameters"] is False
    assert s2["protected_success_basis_rank"] == 32
    assert s2["residual_sensitive_exclusive_basis_rank"] == 8
    assert s2["hard_success_regression_limit"] == 0
    assert s2["hard_success_kl_budget"] == 0.05
    assert s2["backtrack_scales"] == [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
    assert cfg["level3_representation_repair_enabled"] is False

    # Residual target rows are selected directly from target IDs, not intersected
    # with an L1 content-sensitive subset. Special IDs are excluded.
    tids = torch.tensor([0, 7, 8, 7, 9, 1], dtype=torch.long)
    rows = experiment.all_non_special_residual_rows(
        DummyTokenizer(), tids, [0, 1, 2, 3, 5]
    )
    assert rows == [7, 8]
    level1_content = [7]
    newly_admitted = sorted(set(rows) - set(level1_content))
    assert newly_admitted == [8]

    # Expanding an L1 content-row anchor preserves original rows and initializes
    # Stage-2-only rows to exact zero delta.
    anchor = torch.tensor([[1.0, 2.0, 3.0]])
    expanded = experiment.expand_anchor_output_delta(
        anchor,
        level1_rows=[7],
        runtime_output_rows=[7, 8],
    )
    assert torch.equal(expanded[0], anchor[0])
    assert torch.equal(expanded[1], torch.zeros(3))

    # Hinge semantics remain unchanged: passing cases contribute zero.
    passing_logits = torch.tensor([[0.0, 1.0, 0.0]])
    passing_tid = torch.tensor([0])
    assert experiment.squared_margin_hinge(
        passing_logits, passing_tid, 0.01
    ).item() == 0.0

    print("RWKU all-residual-row rank-8 MQuAKE-style Stage2 smoke test PASS")
    print("  L1: unchanged content-sensitive embedding-GA-only architecture")
    print("  L2 embeddings: frozen at L1 anchor")
    print("  L2 rows: ALL non-special sensitive target rows occurring in F")
    print("  Stage2-only output rows start at exact zero delta")
    print("  L2 LM head: DeltaW = C_F B_F, rank 8")
    print("  B_P: Level-1 success rowspace, rank 32")
    print("  hinge + P KL + tiny L2; hard P guard/backtracking unchanged")
    print("  RWKU margin/utility gates unchanged; Level3 disabled")


if __name__ == "__main__":
    main()
