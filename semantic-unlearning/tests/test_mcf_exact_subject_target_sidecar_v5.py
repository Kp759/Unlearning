from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_exact_subject_target_sidecar_v5_core as core
import build_mcf_exact_subject_target_sidecar_v5 as build
import build_mcf_frozen_relation_suffix_lexicon as lexicon_builder
from mcf_sampling import sample_official_mcf_records


def _relation_lexicon():
    value = {
        "schema_version": 1,
        "kind": "mcf_frozen_relation_suffix_lexicon_v1",
        "matching_rule": "longest_boundary_complete_prefix_relation_membership",
        "suffix_count": 2,
        "suffix_to_relation_ids": {"is": ["P106"], "text": ["P999"]},
    }
    value["lexicon_sha256"] = core.relation_lexicon_sha256(value)
    return value


class ToyTokenizer:
    def __init__(self):
        self.pieces = {
            0: "",
            1: "Alpha",
            2: " Beta",
            3: " is",
            4: " reference",
            5: " sensitive",
            6: " Other",
            7: " text",
            8: "BMW",
            9: " M5",
            10: "4",
        }
        self.encodings = {
            "Alpha Beta": [1, 2],
            " Alpha Beta": [1, 2],
            " reference": [0, 4],
            " sensitive": [0, 5],
            "BMW M5": [8, 9],
            " BMW M5": [8, 9],
        }

    def __call__(self, text, add_special_tokens=True, **_kwargs):
        if isinstance(text, list):
            rows = [self(value, add_special_tokens=add_special_tokens)["input_ids"] for value in text]
            return {"input_ids": rows}
        try:
            values = list(self.encodings[str(text)])
        except KeyError as exc:
            raise AssertionError(f"unexpected toy encoding: {text!r}") from exc
        if not add_special_tokens and values and values[0] == 0:
            values = values[1:]
        return {"input_ids": values}

    def batch_decode(self, rows, **_kwargs):
        return ["".join(self.pieces[int(value)] for value in row) for row in rows]

    def decode(self, row, **_kwargs):
        return "".join(self.pieces[int(value)] for value in row)


class ToyLM(nn.Module):
    def __init__(self, vocab=12, hidden=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(model_type="llama")
        with torch.no_grad():
            self.embed.weight.copy_(
                torch.arange(vocab * hidden, dtype=torch.float32).reshape(vocab, hidden)
                / 20.0
            )
            self.lm_head.weight.copy_(
                torch.arange(vocab * hidden, dtype=torch.float32)
                .flip(0)
                .reshape(vocab, hidden)
                / 30.0
            )

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, **_kwargs):
        return SimpleNamespace(logits=self.lm_head(self.embed(input_ids)))


def _state(tok, *, subject="Alpha Beta", pattern=None, bias=20.0):
    pattern = pattern or [[1, 2]]
    return core.build_candidate_state(
        seed=1,
        case_ids=[7],
        subjects=[subject],
        relation_ids=["P106"],
        relation_lexicon=_relation_lexicon(),
        subject_patterns=[pattern],
        target_new=["reference"],
        target_true=["sensitive"],
        target_new_ids=[[4]],
        target_true_ids=[[5]],
        llama_like=True,
        logit_bias=bias,
        base_embedding_sha256=core.tensor_sha256(ToyLM().embed.weight),
        base_lm_head_sha256=core.tensor_sha256(ToyLM().lm_head.weight),
        source_hashes={"split": "a" * 64},
    )


def test_closed_route_is_exact_base_and_parameters_never_change():
    tok = ToyTokenizer()
    model = ToyLM()
    state = _state(tok)
    inputs = torch.tensor([[0, 6, 7, 3]], dtype=torch.long)
    embedding_before = model.embed.weight.detach().clone()
    head_before = model.lm_head.weight.detach().clone()
    base = model(input_ids=inputs).logits.detach().clone()
    runtime = core.install_candidate(model, tok, state)
    try:
        observed = model(input_ids=inputs).logits.detach()
        assert torch.equal(observed, base)
        assert runtime.sidecar.fired_rows == 0
        assert runtime.sidecar.corrected_positions == 0
        assert torch.equal(model.embed.weight, embedding_before)
        assert torch.equal(model.lm_head.weight, head_before)
    finally:
        runtime.close()


def test_open_route_biases_only_predictions_at_and_after_complete_subject():
    tok = ToyTokenizer()
    model = ToyLM()
    inputs = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    base = model(input_ids=inputs).logits.detach().clone()
    runtime = core.install_candidate(model, tok, _state(tok))
    try:
        observed = model(input_ids=inputs).logits.detach()
        assert torch.equal(observed[:, :2], base[:, :2])
        torch.testing.assert_close(observed[0, 2:, 4], base[0, 2:, 4] + 20.0)
        torch.testing.assert_close(observed[0, 2:, 5], base[0, 2:, 5] - 20.0)
        untouched = [value for value in range(12) if value not in (4, 5)]
        assert torch.equal(observed[0, 2:, untouched], base[0, 2:, untouched])
        assert runtime.sidecar.fired_rows == 1
        assert runtime.sidecar.corrected_positions == 2
    finally:
        runtime.close()


def test_lexical_boundary_closes_bmw_m5_prefix_collision_with_bmw_m54():
    tok = ToyTokenizer()
    model = ToyLM()
    state = _state(tok, subject="BMW M5", pattern=[[8, 9]])
    inputs = torch.tensor([[0, 8, 9, 10, 3]], dtype=torch.long)
    base = model(input_ids=inputs).logits.detach().clone()
    runtime = core.install_candidate(model, tok, state)
    try:
        route = runtime.router.route(inputs)
        assert not bool(route.active.any())
        assert torch.equal(model(input_ids=inputs).logits.detach(), base)
    finally:
        runtime.close()


def test_router_keeps_multiple_nonoverlapping_complete_subjects():
    tok = ToyTokenizer()
    model = ToyLM()
    router = core.BoundaryAwareSubjectRouter(
        model.embed,
        [[[1, 2]], [[6]]],
        subjects=["Alpha Beta", "Other"],
        relation_ids=["P106", "P999"],
        suffix_to_relation_ids=_relation_lexicon()["suffix_to_relation_ids"],
        tokenizer=tok,
        model=model,
    )
    try:
        route = router.route(torch.tensor([[0, 1, 2, 3, 6, 7]]))
        assert route.active.tolist() == [[True, True]]
    finally:
        router.close()


def test_official_target_tokenization_and_sparse_direction_sets():
    tok = ToyTokenizer()
    assert core.official_target_token_ids(tok, "reference", llama_like=True) == [4]
    ids, biases, report = core.sparse_token_biases(
        [[4, 6]], [[5, 6]], logit_bias=8.0
    )
    assert ids == [[4, 5]]
    assert biases == [[8.0, -8.0]]
    assert report["rows"][0]["shared_token_ids"] == [6]


def test_candidate_rejects_ambiguous_or_duplicate_subject_bindings():
    tok = ToyTokenizer()
    with pytest.raises(ValueError, match="direction-specific"):
        core.build_candidate_state(
            seed=1,
            case_ids=[1],
            subjects=["Alpha Beta"],
            relation_ids=["P106"],
            relation_lexicon=_relation_lexicon(),
            subject_patterns=[[[1, 2]]],
            target_new=["same"],
            target_true=["same"],
            target_new_ids=[[4]],
            target_true_ids=[[4]],
            llama_like=True,
            logit_bias=20.0,
            base_embedding_sha256="a" * 64,
            base_lm_head_sha256="b" * 64,
            source_hashes={"split": "a" * 64},
        )


def test_candidate_rejects_a_different_base_model_binding():
    tok = ToyTokenizer()
    state = _state(tok)
    model = ToyLM()
    with torch.no_grad():
        model.lm_head.weight[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="LM head differs"):
        core.install_candidate(model, tok, state)


def test_registry_locks_optimizer_free_non_mutating_claim_scope():
    registry_path = (
        Path(__file__).resolve().parents[1]
        / "protocols"
        / "mcf_exact_subject_target_sidecar_v5_0_registry.json"
    )
    registry = json.loads(registry_path.read_text())
    build.validate_registry(registry, SimpleNamespace(logit_bias=256.0))
    assert registry["architecture"]["base_embedding_mutation"] is False
    assert registry["architecture"]["lm_head_mutation"] is False
    assert registry["architecture"]["learned_detector_parameters"] == 0
    assert registry["architecture"]["learned_actuator_parameters"] == 0
    assert "frozen_relation_suffix_grammar" in registry["architecture"]["route"]
    assert registry["seed_1_use"] == "development_replay_only_with_disclosure"


def test_frozen_relation_lexicon_reproduces_and_routes_seed1_without_leakage():
    root = Path(__file__).resolve().parents[1]
    mcf_path = root / "data" / "multi_counterfact.json"
    artifact = json.loads(
        (root / "protocols" / "mcf_frozen_relation_suffix_lexicon_v1.json").read_text()
    )
    data = json.loads(mcf_path.read_text())
    reproduced = lexicon_builder.derive_lexicon(
        data,
        source_mcf_sha256=hashlib.sha256(mcf_path.read_bytes()).hexdigest(),
    )
    assert reproduced == artifact

    forget, retain = sample_official_mcf_records(data, 50, 1000, 1)
    mapping = artifact["suffix_to_relation_ids"]
    positive_cells = 0
    for record in forget:
        rewrite = record["requested_rewrite"]
        prompts = [
            rewrite["prompt"].format(rewrite["subject"]),
            *record.get("paraphrase_prompts", []),
        ]
        for prompt in prompts:
            remainder = core.complete_subject_remainder(prompt, rewrite["subject"])
            relations, _suffix = core.relation_suffix_membership(remainder, mapping)
            assert str(rewrite["relation_id"]) in relations
            positive_cells += 1
    assert positive_cells == 150

    collision_cells = 0
    for retain_record in retain:
        retain_rewrite = retain_record["requested_rewrite"]
        prompts = [
            retain_rewrite["prompt"].format(retain_rewrite["subject"]),
            *retain_record.get("paraphrase_prompts", []),
            *retain_record.get("neighborhood_prompts", []),
        ]
        for forget_record in forget:
            forget_rewrite = forget_record["requested_rewrite"]
            subject = str(forget_rewrite["subject"])
            for prompt in prompts:
                remainder = core.complete_subject_remainder(prompt, subject)
                if remainder is None:
                    continue
                relations, _suffix = core.relation_suffix_membership(
                    remainder, mapping
                )
                assert str(forget_rewrite["relation_id"]) not in relations
                collision_cells += 1
    assert collision_cells == 24
