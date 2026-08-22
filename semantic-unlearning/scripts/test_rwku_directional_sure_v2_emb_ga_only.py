#!/usr/bin/env python3
"""No-model smoke tests for Directional SURE v2 embedding-GA-only ablation."""
from pathlib import Path

import torch

import rwku_directional_sure_v2 as baseline
import rwku_directional_sure_v2_emb_ga_only as experiment

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ga = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    gd = torch.tensor([[100.0, 200.0], [-300.0, 400.0]])
    head_ga = torch.tensor([[1.0, 2.0]])
    head_gd = torch.tensor([[3.0, 4.0]])

    # Requested one-factor change: embedding GD must have exactly zero influence.
    got_embedding = experiment.compose_embedding_gradient(ga, gd)
    assert torch.equal(got_embedding, ga)
    assert not torch.equal(got_embedding, ga + gd)

    # LM-head rule must remain GA->B_S + GD->B_P after those projections exist.
    got_head = experiment.compose_head_gradient(head_ga, head_gd)
    assert torch.equal(got_head, head_ga + head_gd)

    cfg = experiment.load_configuration(
        ROOT / "config" / "rwku" / "directional_sure_v2_emb_ga_only_seed0.json"
    )
    base_cfg = baseline.read_json(baseline.DEFAULT_CONFIGURATION)

    assert cfg["embedding_gradient_policy"] == "GA_only_no_GD_no_hidden_basis_projection"
    assert cfg["lm_head_gradient_policy"] == "GA_to_sensitive_exclusive_basis_and_GD_to_protected_basis"
    assert cfg["trainable_components"] == base_cfg["trainable_components"]
    assert cfg["acceptance"] == base_cfg["acceptance"]
    assert cfg["data_boundary"] == base_cfg["data_boundary"]

    new_opt = dict(cfg["optimization"])
    old_opt = dict(base_cfg["optimization"])
    new_opt.pop("objective")
    old_opt.pop("objective")
    assert new_opt == old_opt

    assert cfg["optimization"]["ga_weight"] == 2.0
    assert cfg["optimization"]["gd_weight"] == 1.0
    assert cfg["optimization"]["sensitive_exclusive_basis_rank"] == 8
    assert cfg["optimization"]["protected_basis_rank"] == 32
    assert cfg["optimization"]["steps"] == 600

    print("RWKU Directional SURE v2 embedding-GA-only smoke test PASS")
    print("  rows: original v2 content-sensitive E/W policy")
    print("  embeddings: 2*GA only; GD measured but NOT applied")
    print("  LM head: 2*GA->B_S + GD->B_P unchanged")
    print("  optimization / acceptance / data boundary: otherwise identical to v2")


if __name__ == "__main__":
    main()
