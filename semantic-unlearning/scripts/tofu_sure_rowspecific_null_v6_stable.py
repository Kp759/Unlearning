#!/usr/bin/env python3
"""Numerically stable entrypoint for SURE-TOFU V6.

Patches V6's exact same-prompt non-target KL into a log-sum-exp form before
running the Base-anchored row-specific prompt-null implementation.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

import tofu_gagd_neighborhood_confidence as tofu
import tofu_sure_rowspecific_null_v6 as v6


def stable_same_prompt_non_target_kl(
    caches: Sequence[tofu.TOFUAnswerDeltaCache],
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    """Exact KL(Base_non-target || Current_non-target) without exp(delta logits)."""
    values: List[torch.Tensor] = []
    tiny = torch.finfo(torch.float32).tiny
    neg_inf = torch.tensor(float("-inf"), dtype=torch.float32, device=delta_rows.device)

    for cache in caches:
        corrections = cache.hidden.float() @ delta_rows.transpose(0, 1)
        q_selected = cache.selected_probs.float()
        q_target = torch.exp(-cache.base_token_nll.float()).clamp(max=1.0 - 1e-7)
        selected_mask = cache.target_selected_columns.ge(0)

        # After removing the true target, the current non-target distribution is
        # proportional to q_k * exp(c_k) for edited rows and q_k otherwise.
        # Compute that partition directly in log space; the full softmax
        # normalizer cancels when the target is removed and renormalized.
        unchanged_mass = (1.0 - q_selected.sum(dim=-1)).clamp_min(0.0)
        unchanged_mass = torch.where(
            selected_mask,
            unchanged_mass,
            (unchanged_mass - q_target).clamp_min(0.0),
        )
        unchanged_log = unchanged_mass.clamp_min(tiny).log().unsqueeze(-1)
        selected_log_terms = q_selected.clamp_min(tiny).log() + corrections

        expected_c_num = (q_selected * corrections).sum(dim=-1)
        if selected_mask.any():
            row_idx = selected_mask.nonzero(as_tuple=False).flatten()
            col_idx = cache.target_selected_columns[selected_mask]
            target_c = corrections[row_idx, col_idx]
            selected_log_terms = selected_log_terms.clone()
            selected_log_terms[row_idx, col_idx] = neg_inf
            expected_c_num[selected_mask] = (
                expected_c_num[selected_mask] - q_target[selected_mask] * target_c
            )

        log_current_non_target_partition = torch.logsumexp(
            torch.cat([unchanged_log, selected_log_terms], dim=-1),
            dim=-1,
        )
        base_non_target_mass = (1.0 - q_target).clamp_min(tiny)
        kl = (
            log_current_non_target_partition
            - base_non_target_mass.log()
            - expected_c_num / base_non_target_mass
        )
        values.append(kl.clamp_min(0.0))

    return torch.cat(values).mean() if values else delta_rows.new_zeros(())


def main() -> None:
    v6.same_prompt_non_target_kl = stable_same_prompt_non_target_kl
    v6.main()


if __name__ == "__main__":
    main()
