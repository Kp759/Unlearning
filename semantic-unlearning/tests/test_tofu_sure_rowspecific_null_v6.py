import math

import torch

import gagd_active_case_repair as active
import tofu_gagd_neighborhood_confidence as tofu
import tofu_sure_rowspecific_null_v6 as v6
import tofu_sure_rowspecific_null_v6_stable as stable


def test_prompt_null_row_basis_is_orthogonal_to_protected_span():
    prompt_rows = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    prompt_basis = active.orthonormal_row_basis(prompt_rows)
    target_rows = [
        torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, -1.0, 2.0]], dtype=torch.float32
        )
    ]
    bases, reports = v6.build_row_specific_bases(target_rows, prompt_basis)
    assert bases[0].shape == (1, 3)
    overlap = bases[0] @ prompt_basis.transpose(0, 1)
    assert torch.allclose(overlap, torch.zeros_like(overlap), atol=1e-6)
    assert reports[0]["row_specific_basis_rank"] == 1


def test_stable_same_prompt_non_target_kl_is_zero_for_zero_delta():
    hidden = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    target_probs = torch.tensor([0.20, 0.30], dtype=torch.float32)
    cache = tofu.TOFUAnswerDeltaCache(
        base_token_nll=-target_probs.log(),
        hidden=hidden,
        selected_probs=torch.tensor(
            [[0.20, 0.10], [0.05, 0.10]], dtype=torch.float32
        ),
        target_selected_columns=torch.tensor([0, -1], dtype=torch.long),
    )
    delta = torch.zeros((2, 3), dtype=torch.float32)
    kl = stable.stable_same_prompt_non_target_kl([cache], delta)
    assert math.isfinite(float(kl))
    assert abs(float(kl)) < 1e-6


def test_projected_stage1a_delta_stays_in_each_row_basis():
    bases = [
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
    ]
    initial = torch.tensor(
        [[2.0, 5.0, 7.0], [3.0, 4.0, 9.0]], dtype=torch.float32
    )
    module = v6.RowSpecificBaseDelta(
        bases, initial_delta=initial, device=torch.device("cpu")
    )
    projected = module.effective_delta().detach()
    assert torch.allclose(projected[0], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.allclose(projected[1], torch.tensor([0.0, 4.0, 0.0]))
