#!/usr/bin/env python3
from __future__ import annotations

from types import SimpleNamespace

import torch

import rwku_directional_sure_v21 as v21


class TinyTokenizer:
    all_special_ids = [0, 1]

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return "|".join(str(int(x)) for x in ids)


def main() -> None:
    cases = [
        SimpleNamespace(record_position=0),
        SimpleNamespace(record_position=0),
        SimpleNamespace(record_position=1),
    ]
    tids = torch.tensor([7, 8, 9], dtype=torch.long)
    rows, audit = v21.all_non_special_sensitive_rows(
        TinyTokenizer(), cases, tids, {}, prompt_count=2
    )
    assert rows == [7, 8, 9]
    assert audit["minimum_editable_rows_per_atomic_prompt"] == 1
    assert audit["all_non_special_sensitive_prediction_cases_covered"] is True

    special_tids = torch.tensor([0, 8, 9], dtype=torch.long)
    try:
        v21.all_non_special_sensitive_rows(
            TinyTokenizer(), cases, special_tids, {}, prompt_count=2
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected special-only prompt coverage failure")

    print("RWKU Directional SURE v2.1 row-coverage smoke test PASS")


if __name__ == "__main__":
    main()
