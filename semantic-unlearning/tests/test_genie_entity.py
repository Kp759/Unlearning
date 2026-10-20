"""genie_routes.rwku_entity_pairs: perfect person routing on the same rows."""
from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import genie_routes as gr  # noqa: E402


class _Tok:
    def __call__(self, text):
        return {"input_ids": [1] + [ord(c) % 50 + 2 for c in text]}


class _Model(torch.nn.Module):
    """_route returns a scripted fact per prompt length (stands in for the run's router)."""
    def __init__(self, decisions):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.decisions = decisions

    def _route(self, ids, att, boundaries):
        return [self.decisions.get(int(ids.shape[1]), -1)]


def test_entity_genie_routes_every_probe_about_a_target_to_that_persons_row():
    facts = [{"subject": "Warren Buffett"}, {"subject": "Warren Buffett"}, {"subject": "Taylor Swift"}]
    forget = [{"source_record_sha256": "a", "subject": "Warren Buffett", "q": "q-forget"}]
    held = [{"subject": "warren  buffett", "q": "held1"},        # router picks row 1 (Buffett's)
            {"subject": "Taylor Swift", "q": "held-two"},        # router fires Buffett's row -> fallback 2
            {"subject": "Taylor Swift", "q": "x"},               # router fires nothing -> fallback 2
            {"subject": "Somebody Else", "q": "neighbour"}]      # not a target: no row
    fmt = lambda row: row["q"]                                    # noqa: E731
    lens = {len(_Tok()("held1")["input_ids"]): 1, len(_Tok()("held-two")["input_ids"]): 0}
    pairs = list(gr.rwku_entity_pairs(_Model(lens), _Tok(), facts, forget, {"a": 0}, held, fmt=fmt))
    assert pairs == [("q-forget", 0), ("held1", 1), ("held-two", 2), ("x", 2)]
    st = gr.rwku_entity_pairs.stats
    assert st == {"own_row": 1, "router_choice": 1, "fallback_first_row": 2,
                  "router_fired_other_person": 1, "unknown_person": 1}
