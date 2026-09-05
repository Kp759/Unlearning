import pathlib
import sys

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retain_anchored_context_head import (  # noqa: E402
    AnchoredFeatureMap,
    FactIndexedLogitCorrection,
    sequence_margin_loss,
    wendland_c2_kernel,
)


def _feature_map() -> AnchoredFeatureMap:
    # Each forget context is close to a protected context, so the test exercises
    # retain conditioning rather than relying on completely disjoint support.
    retain = torch.tensor([[0.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
    forget = torch.tensor([[1.0, 0.0], [1.0, 3.0]], dtype=torch.float64)
    return AnchoredFeatureMap.fit(
        retain=retain,
        forget=forget,
        radius=2.0,
        retain_jitter=0.0,
        cardinal_jitter=0.0,
    )


def test_wendland_kernel_has_exact_compact_support():
    x = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    inside = torch.tensor([[0.5, 0.0]], dtype=torch.float64)
    boundary = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    outside = torch.tensor([[2.0, 0.0]], dtype=torch.float64)

    assert wendland_c2_kernel(x, inside, radius=1.0).item() > 0.0
    assert wendland_c2_kernel(x, boundary, radius=1.0).item() == 0.0
    assert wendland_c2_kernel(x, outside, radius=1.0).item() == 0.0


def test_retain_anchors_are_zero_and_forget_anchors_are_cardinal():
    fmap = _feature_map()

    retain_alpha = fmap.alpha(fmap.retain)
    forget_alpha = fmap.alpha(fmap.forget)

    torch.testing.assert_close(retain_alpha, torch.zeros_like(retain_alpha), atol=1e-10, rtol=0)
    torch.testing.assert_close(
        forget_alpha,
        torch.eye(fmap.num_facts, dtype=forget_alpha.dtype),
        atol=1e-10,
        rtol=0,
    )


def test_outside_support_has_exact_zero_fact_features():
    fmap = _feature_map()
    far = torch.tensor([[100.0, 100.0]], dtype=torch.float64)

    assert fmap.outside_support_mask(far).item() is True
    torch.testing.assert_close(
        fmap.alpha(far),
        torch.zeros((1, fmap.num_facts), dtype=torch.float64),
        atol=0,
        rtol=0,
    )


def test_same_output_token_gets_independent_fact_corrections_and_rollback():
    fmap = _feature_map()
    head = FactIndexedLogitCorrection(
        feature_map=fmap,
        selected_token_ids=[7],
        vocab_size=16,
    )
    with torch.no_grad():
        # Same token id, two independently adjustable factual uses.
        head.coefficients[0, 0] = -6.0
        head.coefficients[0, 1] = -9.0

    base = torch.zeros((2, 16), dtype=torch.float64)
    corrected = head(base, fmap.forget)
    assert corrected[0, 7].item() == -6.0
    assert corrected[1, 7].item() == -9.0

    # Protected contexts remain exactly at base logits.
    protected_base = torch.randn((2, 16), dtype=torch.float64)
    protected_corrected = head(protected_base, fmap.retain)
    torch.testing.assert_close(protected_corrected, protected_base, atol=1e-10, rtol=0)

    # Roll back fact 0 only. Fact 1's correction remains unchanged.
    head.rollback_facts([0])
    after_rollback = head(base, fmap.forget)
    assert abs(after_rollback[0, 7].item()) < 1e-10
    assert after_rollback[1, 7].item() == -9.0


def test_margin_loss_is_event_specific_not_global_token_ban():
    # Two causal prediction events may use the same sensitive token while
    # receiving different context-indexed corrections upstream.
    logits = torch.tensor(
        [
            [0.0, 5.0, 1.0],
            [0.0, 1.0, 5.0],
        ],
        dtype=torch.float64,
    )
    sensitive = torch.tensor([1, 1], dtype=torch.long)
    safe = torch.tensor([2, 2], dtype=torch.long)

    losses = sequence_margin_loss(
        logits,
        sensitive_token_ids=sensitive,
        safe_token_ids=safe,
        margin=1.0,
        reduction="none",
    )

    assert losses[0].item() > 0.0
    assert losses[1].item() == 0.0
