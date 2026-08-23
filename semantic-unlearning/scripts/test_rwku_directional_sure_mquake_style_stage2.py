#!/usr/bin/env python3
"""No-model smoke tests for RWKU MQuAKE-style protected Stage 2."""
from pathlib import Path

import torch

import rwku_directional_sure_two_stage_emb_ga_only as previous
import rwku_mquake_stage2_helpers as experiment
import sure_canonical_core as core

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = experiment.load_configuration(
        ROOT / "config" / "rwku" / "directional_sure_mquake_style_stage2_seed0.json"
    )
    old = previous.read_json(previous.DEFAULT_CONFIGURATION)

    # Level 1 is exactly the previously tested content-sensitive embedding-GA-only setup.
    assert cfg["optimization"] == old["optimization"]
    assert cfg["trainable_components"] == old["trainable_components"]
    assert cfg["acceptance"] == old["acceptance"]
    assert cfg["acceptance"]["required_pairwise_margin"] == 0.01

    s2 = cfg["stage2"]
    assert s2["parameter_scope"] == "lm_head_only_increment_over_frozen_level1_anchor"
    assert s2["embedding_parameters"] is False
    assert s2["transformer_parameters"] is False
    assert s2["protected_success_basis_rank"] == 32
    assert s2["residual_sensitive_exclusive_basis_rank"] == 4
    assert s2["hard_success_regression_limit"] == 0
    assert s2["hard_success_kl_budget"] == 0.05
    assert s2["backtrack_scales"] == [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
    assert cfg["level3_representation_repair_enabled"] is False

    # Pairwise margin = max_other - sensitive. Passing cases have zero hinge.
    passing_logits = torch.tensor([[0.0, 1.0, 0.0]])
    passing_tid = torch.tensor([0])
    passing_margin = experiment.pairwise_margins(passing_logits, passing_tid)
    assert torch.allclose(passing_margin, torch.tensor([1.0]))
    assert experiment.squared_margin_hinge(passing_logits, passing_tid, 0.01).item() == 0.0

    failing_logits = torch.tensor([[1.0, 0.995, 0.0]])
    failing_tid = torch.tensor([0])
    failing_margin = experiment.pairwise_margins(failing_logits, failing_tid)
    assert failing_margin.item() < 0.01
    assert experiment.squared_margin_hinge(failing_logits, failing_tid, 0.01).item() > 0.0

    # Stage-2 delta is explicitly low-rank C_F B_F.
    basis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    repair = core.SelectedRowDelta(2, 3, direction_basis=basis, device=torch.device("cpu"))
    assert repair.coefficients is not None
    assert repair.raw_delta is None
    assert repair.effective_delta().shape == (2, 3)
    assert repair.trainable_parameter_count == 4

    # Combining Stage 2 with the L1 anchor can alter only requested residual rows.
    anchor = torch.zeros((3, 3))
    combined = experiment.combine_output(
        anchor,
        selected_rows=[10, 20, 30],
        repair_rows=[20],
        repair_delta=torch.tensor([[1.0, 2.0, 3.0]]),
    )
    assert torch.equal(combined[0], anchor[0])
    assert torch.equal(combined[2], anchor[2])
    assert torch.equal(combined[1], torch.tensor([1.0, 2.0, 3.0]))

    print("RWKU MQuAKE-style Stage2 smoke test PASS")
    print("  L1: exact prior content-sensitive embedding-GA-only architecture")
    print("  L2 embeddings: frozen at L1 anchor")
    print("  L2 LM head: residual content rows, DeltaW = C_F B_F, rank 4")
    print("  B_P: Level-1 success hidden-state rowspace, rank 32")
    print("  loss: squared margin hinge + KL(P_anchor||P_current) + tiny L2")
    print("  hard guard: P regressions=0, P KL<=0.05")
    print("  backtrack: 0.5 -> ... -> 0.015625, otherwise rollback")
    print("  RWKU behavior margin/utility gates unchanged; Level3 disabled")


if __name__ == "__main__":
    main()
