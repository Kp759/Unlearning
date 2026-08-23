#!/usr/bin/env python3
"""No-model smoke tests for two-stage Directional SURE embedding-GA-only."""
from pathlib import Path

import torch

import rwku_directional_sure_two_stage as baseline
import rwku_directional_sure_two_stage_emb_ga_only as experiment

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ga = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    gd = torch.tensor([[100.0, 200.0], [-300.0, 400.0]])
    mask = torch.tensor([[1.0], [0.0]])

    got_l1 = experiment.compose_embedding_gradient(ga, gd)
    assert torch.equal(got_l1, ga)
    assert not torch.equal(got_l1, ga + gd)

    got_l2 = experiment.compose_embedding_gradient(ga, gd, mask)
    assert torch.equal(got_l2, ga * mask)

    head_ga = torch.tensor([[1.0, 2.0]])
    head_gd = torch.tensor([[3.0, 4.0]])
    assert torch.equal(
        experiment.compose_head_gradient(head_ga, head_gd), head_ga + head_gd
    )

    tids = torch.tensor([10, 11, 12, 13, 14])
    rows = experiment.residual_content_rows(
        tids, case_indices=[0, 1, 2, 4], selected_rows=[10, 12, 99]
    )
    assert rows == [10, 12]

    cfg = experiment.load_configuration(
        ROOT
        / "config"
        / "rwku"
        / "directional_sure_two_stage_emb_ga_only_seed0.json"
    )
    base_cfg = baseline.read_json(baseline.DEFAULT_CONFIGURATION)

    assert cfg["embedding_gradient_policy"] == "GA_only_no_GD_no_hidden_basis_projection"
    assert cfg["lm_head_gradient_policy"] == (
        "GA_to_sensitive_or_residual_exclusive_basis_and_GD_to_protected_basis"
    )
    assert cfg["trainable_components"] == base_cfg["trainable_components"]
    assert cfg["acceptance"] == base_cfg["acceptance"]
    assert cfg["data_boundary"] == base_cfg["data_boundary"]
    assert cfg["optimization"]["steps"] == 600
    assert cfg["optimization"]["sensitive_exclusive_basis_rank"] == 8
    assert cfg["optimization"]["protected_basis_rank"] == 32
    assert cfg["stage2"]["steps"] == 300
    assert cfg["stage2"]["embedding_gradient_policy"] == "GA_only_no_GD"
    assert cfg["stage2"]["row_scope"] == (
        "content_sensitive_rows_implicated_by_level1_residual_prompts"
    )
    assert cfg["stage2"]["representation_repair"] is False

    print("RWKU two-stage Directional SURE embedding-GA-only smoke test PASS")
    print("  L1 rows: original v2 content-sensitive E/W policy")
    print("  L1 embeddings: 2*GA only; embedding GD measured but NOT applied")
    print("  L1 LM head: 2*GA->B_S + GD->B_P")
    print("  L2: residual prompts + residual content-sensitive rows only")
    print("  L2 embeddings: 2*GA only; embedding GD measured but NOT applied")
    print("  L2 LM head: 2*GA->B_F + GD->B_P")
    print("  L3/MLP/attention/LoRA: disabled")
    print("  ranks/LRs/steps/utility gates/data boundary: unchanged")


if __name__ == "__main__":
    main()
