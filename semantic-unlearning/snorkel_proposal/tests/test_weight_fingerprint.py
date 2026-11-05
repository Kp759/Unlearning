"""weight_fingerprint on tensors past 2**24 entries (a 3B model's embedding has 3.9e8).

The claims these protect:
  * the sampled positions are exact integers inside the tensor, first and last included;
    a float32 linspace rounded the last one up to numel, which is an IndexError on CPU and
    a device-side assert on CUDA (every SCALPEL run died at its first fingerprint);
  * the fingerprint still detects an edit at the sampled positions and is stable otherwise.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scalpel.runner import spread_index, weight_fingerprint  # noqa: E402


def test_spread_index_stays_inside_large_tensors():
    for n in (1, 2, 63, 64, 65, 2 ** 24 + 4, 25_165_824, 128_256 * 3_072, 152_064 * 3_584):
        idx = spread_index(n)
        assert idx.dtype == torch.long and len(idx) == min(64, n)
        assert int(idx[0]) == 0 and int(idx[-1]) == n - 1 and int(idx.max()) < n
        assert bool((idx[1:] > idx[:-1]).all())


def test_fingerprint_past_two_to_the_24():
    model = torch.nn.Module()
    model.big = torch.nn.Parameter(torch.zeros(2 ** 24 + 4))     # float32 linspace: last index = numel
    before = weight_fingerprint(model)
    assert weight_fingerprint(model) == before
    with torch.no_grad():
        model.big[-1] = 1.0
    assert weight_fingerprint(model) != before
