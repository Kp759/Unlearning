#!/usr/bin/env python3
"""No-model smoke tests for Directional SURE embedding ablations A/B."""
from pathlib import Path

import torch
from torch import nn

import rwku_directional_sure_embedding_ab as ab

ROOT = Path(__file__).resolve().parents[1]


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(self.embed.weight)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head


def check_gradient_mask(variant: str, allowed_rows, expected_nonzero_positions) -> None:
    ab._ACTIVE_VARIANT = variant
    ab._CAPTURED_OUTPUT_ROWS = (1, 2, 3, 4)
    ab._CAPTURED_INPUT_ROWS = tuple(allowed_rows)
    model = TinyModel()
    sparse = ab.PolicySparseFP32RowDeltas(
        model,
        selected_input_rows=[1, 2, 3, 4],
        selected_output_rows=[1, 2, 3, 4],
    )
    loss = sparse.input_delta.sum()
    loss.backward()
    row_has_grad = (
        sparse.input_delta.grad.detach().abs().sum(dim=1) > 0
    ).cpu().tolist()
    observed = [index for index, value in enumerate(row_has_grad) if value]
    assert observed == list(expected_nonzero_positions), (observed, expected_nonzero_positions)
    sparse.close()


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

    # Sparse-master row order is [1,2,3,4]. Variant A must zero every input
    # gradient. Variant B may update only token rows 2 and 4, i.e. positions 1,3.
    check_gradient_mask(ab.VARIANT_A, [], [])
    check_gradient_mask(ab.VARIANT_B, [2, 4], [1, 3])

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
    assert cfg_a["embedding_gradient_policy"] == "not_applicable_embeddings_frozen"
    assert (
        cfg_b["embedding_gradient_policy"]
        == "ordinary_weighted_GA_plus_GD_no_hidden_basis_projection"
    )
    assert cfg_a["lm_head_row_policy"] == cfg_b["lm_head_row_policy"]
    assert cfg_a["optimization"] == cfg_b["optimization"]
    assert cfg_a["stage2"] == cfg_b["stage2"]
    assert cfg_a["acceptance"] == cfg_b["acceptance"]
    assert cfg_a["data_boundary"] == cfg_b["data_boundary"]
    assert cfg_a["level3_representation_repair_enabled"] is False
    assert cfg_b["level3_representation_repair_enabled"] is False

    print("RWKU Directional SURE embedding A/B smoke test PASS")
    print("  Variant A: all input-delta gradients exactly zero")
    print("  Variant B: only locked content-safe input rows receive gradients")
    print("  Both: all-sensitive LM head with GA->B_S and GD->B_P")
    print("  Common optimization / Stage 2 / acceptance / data boundary: identical")


if __name__ == "__main__":
    main()
