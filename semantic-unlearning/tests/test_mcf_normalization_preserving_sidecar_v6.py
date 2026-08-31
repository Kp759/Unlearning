from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
from torch import nn


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_mcf_frozen_two_sided_frame_lexicon_v6 as lexicon_builder
import build_mcf_normalization_preserving_sidecar_v6_candidate as candidate_builder
import build_mcf_normalization_preserving_sidecar_v6_consumed_split as split_builder
import freeze_mcf_normalization_preserving_sidecar_v6_after_consumed_development as freeze_builder
import mcf_normalization_preserving_sidecar_v6_core as core
from mcf_sampling import sample_official_mcf_records


def _lexicon() -> dict:
    value = {
        "schema_version": 1,
        "kind": "mcf_frozen_two_sided_relation_frame_lexicon_v1",
        "matching_rule": "exact_left_clause_and_longest_boundary_complete_right_prefix",
        "frame_count": 2,
        "frame_to_relation_ids": {
            core.frame_key("", "is"): ["P106"],
            core.frame_key("what role does", "play"): ["P106"],
        },
    }
    value["lexicon_sha256"] = core.frame_lexicon_sha256(value)
    return value


class ToyTokenizer:
    def __init__(self) -> None:
        self.pieces = {
            0: "",
            1: "Alpha",
            2: " Beta",
            3: " is",
            4: " reference",
            5: " sensitive",
            6: " HMAS",
            7: " unrelated",
            8: " What",
            9: " role",
            10: " does",
            11: " play",
        }
        self.vocabulary = {f"ordinary_{index}": index for index in range(12)}
        for index in range(12, 48):
            name = f"<|reserved_special_token_{index - 12}|>"
            self.vocabulary[name] = index
            self.pieces[index] = name
        self.encodings = {
            "Alpha Beta": [1, 2],
            " Alpha Beta": [1, 2],
            " reference": [0, 4],
            " sensitive": [0, 5],
        }
        self.bos_token_id = 0
        self.eos_token_id = None
        self.pad_token_id = None
        self.unk_token_id = None

    def __call__(self, text, add_special_tokens=True, **_kwargs):
        if isinstance(text, list):
            return {
                "input_ids": [
                    self(item, add_special_tokens=add_special_tokens)["input_ids"]
                    for item in text
                ]
            }
        values = list(self.encodings[str(text)])
        if not add_special_tokens and values and values[0] == 0:
            values = values[1:]
        return {"input_ids": values}

    def decode(self, row, **_kwargs):
        return "".join(self.pieces[int(item)] for item in row)

    def get_vocab(self):
        return dict(self.vocabulary)

    def convert_ids_to_tokens(self, values):
        if isinstance(values, int):
            values = [values]
        inverse = {value: key for key, value in self.vocabulary.items()}
        return [inverse[int(item)] for item in values]


class ToyLM(nn.Module):
    def __init__(self, vocab: int = 48, hidden: int = 4) -> None:
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


def _state(tok: ToyTokenizer, model: ToyLM) -> dict:
    reserved_ids, reserved_tokens = core.select_reserved_token_pool(
        tok,
        excluded_token_ids=[4, 5],
        requested_size=32,
    )
    return core.build_candidate_state(
        seed=1,
        case_ids=[7],
        subjects=["Alpha Beta"],
        relation_ids=["P106"],
        frame_lexicon=_lexicon(),
        subject_patterns=[[[1, 2]]],
        target_new=["reference"],
        target_true=["sensitive"],
        target_new_ids=[[4]],
        target_true_ids=[[5]],
        reserved_token_ids=reserved_ids,
        reserved_token_strings=reserved_tokens,
        llama_like=True,
        base_embedding_sha256=core.tensor_sha256(model.embed.weight),
        base_lm_head_sha256=core.tensor_sha256(model.lm_head.weight),
        source_hashes={"split": "a" * 64},
    )


def test_closed_route_returns_exact_base_logits_and_changes_no_parameter():
    tok = ToyTokenizer()
    model = ToyLM()
    inputs = torch.tensor([[0, 7, 3]], dtype=torch.long)
    base = model(input_ids=inputs).logits.detach().clone()
    embed_before = model.embed.weight.detach().clone()
    head_before = model.lm_head.weight.detach().clone()
    runtime = core.install_candidate(model, tok, _state(tok, model))
    try:
        observed = model(input_ids=inputs).logits.detach()
        assert torch.equal(observed, base)
        assert runtime.sidecar.fired_rows == 0
        assert torch.equal(model.embed.weight, embed_before)
        assert torch.equal(model.lm_head.weight, head_before)
    finally:
        runtime.close()


def test_open_route_is_a_logit_permutation_and_leaves_other_tokens_exact():
    tok = ToyTokenizer()
    model = ToyLM()
    inputs = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    base = model(input_ids=inputs).logits.detach().clone()
    runtime = core.install_candidate(model, tok, _state(tok, model))
    try:
        observed = model(input_ids=inputs).logits.detach()
        assert torch.equal(observed[:, :2], base[:, :2])
        ordinary_untouched = [item for item in range(12) if item not in (4, 5)]
        assert torch.equal(
            observed[0, 2:, ordinary_untouched],
            base[0, 2:, ordinary_untouched],
        )
        assert torch.equal(
            observed[0, 2:].sort(dim=-1).values,
            base[0, 2:].sort(dim=-1).values,
        )
        assert bool((observed[0, 2:, 5] <= base[0, 2:, 5]).all())
        assert bool((observed[0, 2:, 4] >= base[0, 2:, 4]).all())
        assert runtime.sidecar.fired_rows == 1
        assert runtime.sidecar.permuted_positions == 2
    finally:
        runtime.close()


def test_two_sided_frame_rejects_subject_nested_in_a_larger_entity():
    tok = ToyTokenizer()
    model = ToyLM()
    runtime = core.install_candidate(model, tok, _state(tok, model))
    try:
        nested = torch.tensor([[0, 6, 1, 2, 3]], dtype=torch.long)
        assert not bool(runtime.router.route(nested).active.any())
        direct = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        assert runtime.router.route(direct).active.tolist() == [[True]]
    finally:
        runtime.close()


def test_frozen_frame_lexicon_reproduces_and_closes_seed2_nested_entities():
    root = Path(__file__).resolve().parents[1]
    mcf_path = root / "data" / "multi_counterfact.json"
    artifact_path = (
        root / "protocols" / "mcf_frozen_two_sided_relation_frame_lexicon_v1.json"
    )
    data = json.loads(mcf_path.read_text())
    artifact = json.loads(artifact_path.read_text())
    reproduced = lexicon_builder.derive_lexicon(
        data,
        source_mcf_sha256=hashlib.sha256(mcf_path.read_bytes()).hexdigest(),
    )
    assert reproduced == artifact

    mapping = artifact["frame_to_relation_ids"]

    def routed(prompt: str, subject: str, relation: str) -> bool:
        for left, right in core.subject_frame_parts(prompt, subject):
            relations, _matched = core.relation_frame_membership(left, right, mapping)
            if relation in relations:
                return True
        return False

    for seed in (1, 2):
        forget, _retain = sample_official_mcf_records(data, 50, 1000, seed)
        positives = 0
        for record in forget:
            rewrite = record["requested_rewrite"]
            subject = str(rewrite["subject"])
            relation = str(rewrite["relation_id"])
            prompts = [
                str(rewrite["prompt"]).format(subject),
                *[str(item) for item in record.get("paraphrase_prompts", [])],
            ]
            for prompt in prompts:
                assert routed(prompt, subject, relation)
                positives += 1
        assert positives == 150

    forget, _retain = sample_official_mcf_records(data, 50, 1000, 2)
    forgotten = {
        str(record["requested_rewrite"]["subject"]): str(
            record["requested_rewrite"]["relation_id"]
        )
        for record in forget
    }
    perth_record = next(
        record
        for record in forget
        if record["requested_rewrite"]["subject"] == "Perth"
    )
    nested = [
        prompt
        for prompt in perth_record["neighborhood_prompts"]
        if prompt.startswith(("HMAS Perth", "Experience Perth", "Perth-class"))
    ]
    assert len(nested) == 10
    assert all(not routed(prompt, "Perth", forgotten["Perth"]) for prompt in nested)


def test_development_builders_refuse_every_fresh_seed():
    import pytest

    with pytest.raises(SystemExit):
        split_builder.parse_args(
            [
                "--mcf-path",
                "mcf.json",
                "--frame-lexicon",
                "frames.json",
                "--output-dir",
                "out",
                "--seed",
                "3",
            ]
        )
    with pytest.raises(SystemExit):
        candidate_builder.parse_args(
            [
                "--model-path",
                "model",
                "--training-visible-path",
                "visible.json",
                "--split-manifest",
                "manifest.json",
                "--development-registry",
                "registry.json",
                "--frame-lexicon",
                "frames.json",
                "--output-dir",
                "out",
                "--seed",
                "3",
            ]
        )


def test_development_registry_locks_normalization_and_fresh_seed_firewall():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads(
        (
            root
            / "protocols"
            / "mcf_normalization_preserving_sidecar_v6_0_development_registry.json"
        ).read_text()
    )
    lexicon = json.loads(
        (
            root
            / "protocols"
            / "mcf_frozen_two_sided_relation_frame_lexicon_v1.json"
        ).read_text()
    )
    candidate_builder.validate_registry(registry, lexicon)
    assert registry["architecture"]["arithmetic_logit_offsets"] == 0
    assert registry["architecture"]["normalization_preserving_by_construction"]
    assert registry["architecture"][
        "ordinary_non_target_logits_unchanged_by_construction"
    ]
    assert (
        registry["fresh_seed_firewall"][
            "seed_3_or_later_candidate_or_evaluation_permitted_by_this_registry"
        ]
        is False
    )


def test_candidate_validation_rejects_reserved_target_overlap():
    import pytest

    tok = ToyTokenizer()
    model = ToyLM()
    state = _state(tok, model)
    state["reserved_token_ids"][0] = state["target_true_ids"][0][0]
    with pytest.raises(RuntimeError, match="reserved-token reservoir"):
        core.validate_candidate_state(state)


def test_post_development_freeze_refuses_any_failed_consumed_seed():
    import pytest

    passing = {
        "passed": True,
        "seed": 1,
        "evaluation_status": "consumed_development_not_blind_not_official",
        "blind_or_official_claim_permitted": False,
        "behavioral_checks": {
            "forget_eff_zero": True,
            "forget_gen_zero": True,
            "minimum_margin_at_least_0_1": True,
            "forget_specificity_exact_base": True,
            "retain_exact_base": True,
            "ppl_exact_base": True,
        },
        "exact_preservation": {
            "checks": {
                "forget_neighborhood_raw_exact": True,
                "retain_raw_exact": True,
                "ppl_exact": True,
            }
        },
        "integrity": {"passed": True},
    }
    freeze_builder.validate_development_result(passing, seed=1)
    failing = dict(passing)
    failing["passed"] = False
    with pytest.raises(RuntimeError, match="did not pass"):
        freeze_builder.validate_development_result(failing, seed=1)
