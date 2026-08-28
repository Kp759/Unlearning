from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_zero_unlearn_official_eval as official  # noqa: E402
import scoped_span_edit as scoped  # noqa: E402


class WordTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __init__(self):
        self.vocab = {
            "<pad>": 0,
            "Gautier": 2,
            "de": 3,
            "Coincy": 4,
            "Melchior": 5,
            "Vogue": 6,
        }

    def __call__(self, texts, padding=False, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        rows = [
            [self.vocab.get(word.strip(".,?"), 9) for word in text.split()]
            for text in texts
        ]
        if padding:
            width = max((len(row) for row in rows), default=0)
            rows = [row + [0] * (width - len(row)) for row in rows]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(rows, dtype=torch.long)}
        return {"input_ids": rows}


def _record(subject, rewrite, paraphrases, neighborhoods):
    return {
        "requested_rewrite": {
            "subject": subject,
            "prompt": rewrite,
            "target_new": {"str": "new"},
            "target_true": {"str": "true"},
        },
        "paraphrase_prompts": paraphrases,
        "neighborhood_prompts": neighborhoods,
    }


def test_post_hoc_route_audit_reports_each_official_prompt_group():
    tok = WordTokenizer()
    embedding = nn.Embedding(16, 3)
    router = scoped.SpanGateRouter(
        embedding,
        subject_patterns=[[[2, 3, 4]]],
        subjects=["Gautier de Coincy"],
    )
    forget = [
        _record(
            "Gautier de Coincy",
            "{} wrote",
            ["What did Gautier de Coincy write?", "Who wrote this?"],
            ["Melchior de Vogue wrote", "Gautier de Coincy appears here"],
        )
    ]
    retain = [
        _record(
            "Melchior de Vogue",
            "{} wrote",
            ["What did Melchior de Vogue write?"],
            ["Another person wrote"],
        )
    ]
    try:
        audit = official.audit_scoped_router_prompt_groups(
            router,
            tok,
            forget,
            retain,
            batch_size=2,
        )
    finally:
        router.close()

    groups = audit["groups"]
    assert groups["forget_rewrite"]["matched_prompts"] == 1
    assert groups["forget_rewrite"]["fire_percent"] == 100.0
    assert groups["forget_paraphrase"]["matched_prompts"] == 1
    assert groups["forget_paraphrase"]["prompt_count"] == 2
    assert groups["forget_neighborhood"]["matched_prompts"] == 1
    assert groups["retain_rewrite"]["matched_prompts"] == 0
    assert groups["retain_paraphrase"]["matched_prompts"] == 0
    assert groups["retain_neighborhood"]["matched_prompts"] == 0
    assert audit["used_for_training_or_checkpoint_selection"] is False
    assert audit["evaluation_group_labels_used_by_router"] is False
