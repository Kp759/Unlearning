#!/usr/bin/env python3
"""Runtime-fixed entrypoint for the Seed-1 recognition router benchmark.

This preserves the original benchmark implementation and patches only the
PooledFactMatcher broadcasting bug discovered during the first AWS run.
The original forward mixed tensors shaped [B,F,D] with `sb * rb` shaped
[1,F,D], which fails when B > 1 during torch.cat.  All candidate-wise terms
are explicitly broadcast to [B,F,D] here before concatenation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import mcf_recognition_router_benchmark_seed1 as benchmark


def _fixed_pooled_forward(
    self: benchmark.PooledFactMatcher,
    query: torch.Tensor,
    subjects: torch.Tensor,
    relations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = F.normalize(self.q(query.float()), p=2, dim=-1)
    s = F.normalize(self.s(subjects.float()), p=2, dim=-1)
    r = F.normalize(self.r(relations.float()), p=2, dim=-1)

    batch = q.shape[0]
    facts = s.shape[0]
    qb = q[:, None, :].expand(batch, facts, -1)
    sb = s[None, :, :].expand(batch, facts, -1)
    rb = r[None, :, :].expand(batch, facts, -1)

    feat = torch.cat(
        [
            qb,
            sb,
            rb,
            qb * sb,
            qb * rb,
            sb * rb,
            (qb - sb).abs(),
            (qb - rb).abs(),
        ],
        dim=-1,
    )
    fact = self.mlp(feat).squeeze(-1)
    none = self.none(q).squeeze(-1)
    return fact, none


benchmark.PooledFactMatcher.forward = _fixed_pooled_forward


if __name__ == "__main__":
    benchmark.main()
