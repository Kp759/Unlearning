from __future__ import annotations

import json
from pathlib import Path
import random
import re
from types import SimpleNamespace
import sys
import unicodedata

import pytest
import torch
from torch import nn


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_zsre_normalization_preserving_sidecar_v6_consumed_split as split_builder
import evaluate_zsre_normalization_preserving_sidecar_v6_consumed as evaluator
import mcf_normalization_preserving_sidecar_v6_core as shared
import scoped_span_edit as scoped
import zsre_normalization_preserving_sidecar_v6_core as core
import zsre_zero_unlearn_official_eval as official


class ToyTokenizer:
    def __init__(self) -> None:
        self.pieces = {
            0: "",
            1: "Alpha",
            2: " Beta",
            3: " unrelated",
            4: " true",
            5: " other",
            6: " Unknown",
        }
        self.bos_token_id = 0
        self.eos_token_id = None
        self.pad_token_id = 0
        self.unk_token_id = None
        self.pad_token = "<bos>"
        self.eos_token = None
        self.vocabulary = {f"ordinary_{index}": index for index in range(7)}
        for index in range(7, 135):
            token = f"<|reserved_special_token_{index - 7}|>"
            self.vocabulary[token] = index
            self.pieces[index] = token
        self.encodings = {
            "Alpha Beta": [0, 1, 2],
            " Alpha Beta": [1, 2],
            "Alpha Beta true": [0, 1, 2, 4],
            "unrelated": [0, 3],
            " true": [0, 4],
            " other": [0, 5],
            "Unknown": [0, 6],
        }

    def __call__(
        self,
        text,
        add_special_tokens=True,
        padding=False,
        return_tensors=None,
        **_kwargs,
    ):
        def one(item):
            values = list(self.encodings[str(item)])
            if not add_special_tokens and values and values[0] == 0:
                values = values[1:]
            return values

        rows = [one(item) for item in text] if isinstance(text, list) else None
        if rows is None:
            values = one(text)
            if return_tensors == "pt":
                return {"input_ids": torch.tensor([values], dtype=torch.long)}
            return {"input_ids": values}
        width = max(len(row) for row in rows)
        if padding:
            rows = [[0] * (width - len(row)) + row for row in rows]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.tensor(
                    [[int(item != 0) for item in row] for row in rows],
                    dtype=torch.long,
                ),
            }
        return {"input_ids": rows}

    def decode(self, values, **_kwargs):
        return "".join(self.pieces[int(item)] for item in values)

    def get_vocab(self):
        return dict(self.vocabulary)

    def convert_ids_to_tokens(self, values):
        inverse = {value: key for key, value in self.vocabulary.items()}
        if isinstance(values, int):
            return inverse[int(values)]
        return [inverse[int(item)] for item in values]


class ToyLM(nn.Module):
    def __init__(self, vocab: int = 135, hidden: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(model_type="llama")
        with torch.no_grad():
            self.embed.weight.copy_(
                torch.arange(vocab * hidden, dtype=torch.float32).reshape(vocab, hidden)
                / 50.0
            )
            self.lm_head.weight.copy_(
                torch.arange(vocab * hidden, dtype=torch.float32)
                .flip(0)
                .reshape(vocab, hidden)
                / 70.0
            )

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, **_kwargs):
        return SimpleNamespace(logits=self.lm_head(self.embed(input_ids)))


def _complete_subject(text: str, subject: str) -> bool:
    pattern = re.compile(
        r"(?<!\w)"
        + re.escape(unicodedata.normalize("NFKC", subject))
        + r"(?!\w)",
        re.IGNORECASE,
    )
    return bool(pattern.search(unicodedata.normalize("NFKC", text)))


def test_seed1_contract_has_complete_subject_coverage_and_no_preservation_routes():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "data" / "zsre_mend_eval.json").read_text())
    forget, retain = official.sample_official_zsre_raw_records(
        raw,
        50,
        1000,
        1,
        strict=True,
    )
    subjects = [str(record["subject"]) for _case_id, record in forget]
    assert len(set(subjects)) == 50
    for _case_id, record in forget:
        assert _complete_subject(record["src"], record["subject"])
        assert _complete_subject(record["rephrase"], record["subject"])
    preservation_text = [str(record["loc"]) for _case_id, record in forget]
    preservation_text.extend(
        str(record[field])
        for _case_id, record in retain
        for field in ("src", "rephrase", "loc")
    )
    assert not any(
        _complete_subject(text, subject)
        for text in preservation_text
        for subject in subjects
    )


def test_open_subject_route_is_normalization_preserving_and_closed_route_is_exact():
    tok = ToyTokenizer()
    model = ToyLM()
    subject_patterns = scoped.build_subject_patterns(tok, ["Alpha Beta"])
    router = scoped.SpanGateRouter(
        model.embed,
        subject_patterns,
        subjects=["Alpha Beta"],
        model=model,
    )
    sidecar = shared.NormalizationPreservingSensitiveTokenSidecar(
        model.lm_head,
        router,
        promoted_token_ids=[[6]],
        sensitive_token_ids=[[4]],
        reserved_token_ids=list(range(7, 135)),
    )
    try:
        closed_ids = torch.tensor([[0, 3]], dtype=torch.long)
        sidecar.enabled = False
        closed_base = model(input_ids=closed_ids).logits.detach().clone()
        sidecar.enabled = True
        closed_candidate = model(input_ids=closed_ids).logits.detach()
        assert torch.equal(closed_base, closed_candidate)

        open_ids = torch.tensor([[0, 1, 2, 4]], dtype=torch.long)
        sidecar.enabled = False
        open_base = model(input_ids=open_ids).logits.detach().clone()
        sidecar.enabled = True
        open_candidate = model(input_ids=open_ids).logits.detach()
        assert torch.equal(open_base[:, :2], open_candidate[:, :2])
        assert torch.equal(
            open_base[:, 2:].sort(dim=-1).values,
            open_candidate[:, 2:].sort(dim=-1).values,
        )
        ordinary = [item for item in range(7) if item not in (4, 6)]
        assert torch.equal(open_base[:, 2:, ordinary], open_candidate[:, 2:, ordinary])
        assert bool((open_candidate[:, 2:, 4] <= open_base[:, 2:, 4]).all())
    finally:
        sidecar.close()
        router.close()


def test_collision_audit_distinguishes_full_input_and_exact_target_conflicts():
    tok = ToyTokenizer()
    forget = {
        "rewrite": [
            official.PredictionCase(1, "rewrite", 0, 0, "Alpha Beta", " true")
        ],
        "paraphrase": [],
        "neighborhood": [],
    }
    retain_same_target = {
        "rewrite": [],
        "paraphrase": [],
        "neighborhood": [
            official.PredictionCase(
                2,
                "neighborhood",
                0,
                0,
                "Alpha Beta",
                " true",
            )
        ],
    }
    exact = evaluator.scorer_collision_audit(
        tok,
        forget,
        retain_same_target,
        llama_like=True,
    )
    assert exact["full_input_overlap_count"] == 1
    assert exact["exact_scorer_collision_count"] == 1
    assert not exact["coherent_for_exact_raw_preservation"]

    retain_other_target = {
        "rewrite": [],
        "paraphrase": [],
        "neighborhood": [
            official.PredictionCase(
                2,
                "neighborhood",
                0,
                0,
                "Alpha Beta",
                " other",
            )
        ],
    }
    input_only = evaluator.scorer_collision_audit(
        tok,
        forget,
        retain_other_target,
        llama_like=True,
    )
    assert input_only["full_input_overlap_count"] == 1
    assert input_only["exact_scorer_collision_count"] == 0
    assert not input_only["coherent_for_exact_raw_preservation"]


def test_candidate_state_is_locked_to_seed1_and_fifty_unique_subjects():
    true_ids = [[4] for _ in range(50)]
    state = core.build_candidate_state(
        seed=1,
        case_ids=list(range(50)),
        subjects=[f"subject {index}" for index in range(50)],
        direct_prompts=[f"question subject {index}" for index in range(50)],
        subject_patterns=[[[index + 1]] for index in range(50)],
        target_true=["true" for _ in range(50)],
        target_true_ids=true_ids,
        neutral_target="Unknown",
        neutral_target_id=6,
        reserved_token_ids=list(range(7, 135)),
        reserved_token_strings=[
            f"<|reserved_special_token_{index}|>" for index in range(128)
        ],
        llama_like=True,
        base_embedding_sha256="a" * 64,
        base_lm_head_sha256="b" * 64,
        source_hashes={
            "training_visible": "c" * 64,
            "split_manifest": "d" * 64,
            "development_registry": "e" * 64,
        },
    )
    core.validate_candidate_state(state)
    assert state["architecture"]["arithmetic_logit_offsets"] == 0
    assert state["architecture"]["route_parameters"] == 0
    with pytest.raises(ValueError):
        core.build_candidate_state(
            **{**{
                "seed": 2,
                "case_ids": list(range(50)),
                "subjects": [f"subject {index}" for index in range(50)],
                "direct_prompts": [f"question subject {index}" for index in range(50)],
                "subject_patterns": [[[index + 1]] for index in range(50)],
                "target_true": ["true" for _ in range(50)],
                "target_true_ids": true_ids,
                "neutral_target": "Unknown",
                "neutral_target_id": 6,
                "reserved_token_ids": list(range(7, 135)),
                "reserved_token_strings": [
                    f"<|reserved_special_token_{index}|>" for index in range(128)
                ],
                "llama_like": True,
                "base_embedding_sha256": "a" * 64,
                "base_lm_head_sha256": "b" * 64,
                "source_hashes": {
                    "training_visible": "c" * 64,
                    "split_manifest": "d" * 64,
                    "development_registry": "e" * 64,
                },
            }}
        )


def test_split_builder_refuses_fresh_seed_reuse():
    with pytest.raises(SystemExit):
        split_builder.parse_args(
            [
                "--zsre-path",
                "zsre.json",
                "--output-dir",
                "out",
                "--seed",
                "2",
            ]
        )
