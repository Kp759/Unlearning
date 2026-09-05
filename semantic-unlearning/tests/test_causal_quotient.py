import pathlib
import sys

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retain_anchored_context_head import (  # noqa: E402
    CausalDescriptorProjector,
    FactIndexedCausalQuotient,
    FrozenRandomProjector,
)


def test_causal_descriptor_is_unit_normalized_and_uses_channel():
    random = FrozenRandomProjector(
        input_dim=4,
        output_dim=2,
        seed=7,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    basis = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    projector = CausalDescriptorProjector(
        random_projector=random,
        channel_basis=basis,
        causal_weight=1.0,
    )

    final = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    causal_a = torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    causal_b = torch.tensor([[0.0, 2.0, 0.0, 0.0]], dtype=torch.float32)
    qa = projector(final, causal_a)
    qb = projector(final, causal_b)

    torch.testing.assert_close(qa.norm(dim=-1), torch.ones(1), atol=1e-6, rtol=0)
    torch.testing.assert_close(qb.norm(dim=-1), torch.ones(1), atol=1e-6, rtol=0)
    assert not torch.allclose(qa, qb)


def test_zero_gate_leaves_hidden_state_unchanged():
    effects = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    neutral = torch.zeros((2, 2), dtype=torch.float32)
    quotient = FactIndexedCausalQuotient(
        effect_directions=effects,
        neutral_final_hidden=neutral,
        strength=1.0,
    )

    hidden = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    alpha = torch.zeros((1, 2), dtype=torch.float32)
    out = quotient.apply(hidden, alpha)
    torch.testing.assert_close(out, hidden, atol=0, rtol=0)


def test_cardinal_gate_quotients_only_selected_fact_direction():
    effects = torch.tensor([[2.0, 0.0], [0.0, 5.0]], dtype=torch.float32)
    neutral = torch.tensor([[1.0, 9.0], [8.0, 2.0]], dtype=torch.float32)
    quotient = FactIndexedCausalQuotient(
        effect_directions=effects,
        neutral_final_hidden=neutral,
        strength=1.0,
    )

    hidden = torch.tensor([[6.0, 7.0]], dtype=torch.float32)
    alpha_fact0 = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    out0 = quotient.apply(hidden, alpha_fact0)
    # Fact 0 removes only the x projection toward fact 0's neutral x=1.
    torch.testing.assert_close(
        out0,
        torch.tensor([[1.0, 7.0]], dtype=torch.float32),
        atol=1e-6,
        rtol=0,
    )

    alpha_fact1 = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    out1 = quotient.apply(hidden, alpha_fact1)
    # Fact 1 removes only the y projection toward fact 1's neutral y=2.
    torch.testing.assert_close(
        out1,
        torch.tensor([[6.0, 2.0]], dtype=torch.float32),
        atol=1e-6,
        rtol=0,
    )


def test_fact_enabled_supports_independent_rollback():
    effects = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    neutral = torch.zeros((2, 2), dtype=torch.float32)
    quotient = FactIndexedCausalQuotient(
        effect_directions=effects,
        neutral_final_hidden=neutral,
        strength=1.0,
    )
    hidden = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    alpha = torch.ones((1, 2), dtype=torch.float32)
    enabled = torch.tensor([0.0, 1.0], dtype=torch.float32)
    out = quotient.apply(hidden, alpha, fact_enabled=enabled)
    torch.testing.assert_close(
        out,
        torch.tensor([[3.0, 0.0]], dtype=torch.float32),
        atol=1e-6,
        rtol=0,
    )
