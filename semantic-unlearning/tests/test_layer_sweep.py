"""Layer-wise study plumbing: last-block features, oracle training routes, norm scale."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from static_overlap_fact_association_embeddings import extract_prompt_queries  # noqa: E402
from linear_router import LinearClassifierAssociationBank  # noqa: E402

HIDDEN = 8
FACTS = 3


# ---------------------------------------------------------------------------
# Router features must be the tensor the runtime hook edits, at every layer.
# ---------------------------------------------------------------------------

def _tiny_llama_and_tokenizer():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = "<pad> <s> the capital of france is paris a b c d".split()
    vocab = {word: index for index, word in enumerate(words)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="a"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", bos_token="<s>"
    )
    config = transformers.LlamaConfig(
        vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).eval()
    with torch.no_grad():  # make the final norm change direction, not just scale
        model.model.norm.weight.copy_(torch.rand(16) * 2 + 0.1)
    return model, tokenizer


@torch.no_grad()
def test_prompt_queries_read_raw_block_output_at_every_layer():
    model, tokenizer = _tiny_llama_and_tokenizer()
    prompts = ["the capital of france is", "a b c"]
    encoded = tokenizer(prompts, padding=True, return_tensors="pt",
                        return_token_type_ids=False)
    last = encoded["attention_mask"].sum(dim=1) - 1
    index = torch.arange(len(prompts))

    captured = []
    handles = [layer.register_forward_hook(
        lambda _m, _a, out: captured.append(out[0] if isinstance(out, tuple) else out))
        for layer in model.model.layers]
    result = model(**encoded, output_hidden_states=True, use_cache=False)
    for handle in handles:
        handle.remove()

    for layer in range(3):
        queries = extract_prompt_queries(model, tokenizer, prompts, layer)
        expected = F.normalize(captured[layer][index, last].float(), dim=-1)
        assert torch.allclose(queries, expected, atol=1e-6)
        if layer < 2:  # unchanged from the old hidden_states[layer + 1] read
            old = F.normalize(result.hidden_states[layer + 1][index, last].float(), dim=-1)
            assert torch.allclose(queries, old, atol=1e-6)

    # The old read for the last block was the post-final-norm state.
    post_norm = F.normalize(result.hidden_states[3][index, last].float(), dim=-1)
    assert not torch.allclose(extract_prompt_queries(model, tokenizer, prompts, 2),
                              post_norm, atol=1e-3)
    with pytest.raises(ValueError):
        extract_prompt_queries(model, tokenizer, prompts, 3)


# ---------------------------------------------------------------------------
# Oracle routes: training-only, exact, and never saved.
# ---------------------------------------------------------------------------

class _Block(nn.Module):
    def forward(self, hidden):
        return (hidden,)


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block()])


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()
        self.placeholder = nn.Parameter(torch.zeros(1))


def _bank():
    torch.manual_seed(0)
    return LinearClassifierAssociationBank(
        _FakeModel(), 0, torch.randn(FACTS, HIDDEN),
        torch.full((FACTS,), -100.0),  # heads never fire on their own
        feature_mean=torch.zeros(HIDDEN), feature_components=None, threshold=0.0,
        subject_patterns=[[(10 + i,)] for i in range(FACTS)],
        facts=[{"id": f"f{i}", "subject": f"S{i}", "relation": "r"} for i in range(FACTS)],
        rows=torch.randn(FACTS, HIDDEN),
    )


def _forward(bank, ids, attention, prefix):
    hidden = torch.ones(ids.shape[0], ids.shape[1], HIDDEN)
    bank.bind(ids, attention_mask=attention, prefix_lengths=torch.tensor(prefix))
    try:
        return bank._hook(None, None, (hidden,))[0]
    finally:
        bank.unbind()


def test_genie_routes_replace_the_linear_classifier_only_while_set():
    bank = _bank()
    ids = torch.tensor([[1, 10, 5, 6, 0], [1, 11, 7, 0, 0], [1, 12, 9, 9, 4]])
    attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    prefix = [3, 3, 4]

    base = torch.ones(3, 5, HIDDEN)
    gate_out = _forward(bank, ids, attention, prefix)
    assert bank.last_active_fact_indices == [[], [], []]
    assert torch.equal(gate_out, base)

    # Row 0 -> fact 2 (deliberately not its subject), row 1 -> fact 1,
    # row 2's prefix is unmapped and must stay on the exact base path.
    bank.set_oracle_routes({(1, 10, 5): 2, (1, 11, 7): 1})
    out = _forward(bank, ids, attention, prefix)
    assert bank.last_active_fact_indices == [[2], [1], []]
    rows = bank.extra.detach()
    assert torch.equal(out[0, 2], 1 + rows[2])
    assert torch.equal(out[1, 2], 1 + rows[1])
    assert torch.equal(out[2], base[2])
    assert torch.equal(out[0, :2], base[0, :2])  # boundary position only

    assert "route_override" not in bank.artifact()
    bank.set_oracle_routes(None)
    _forward(bank, ids, attention, prefix)
    assert bank.last_active_fact_indices == [[], [], []]

    with pytest.raises(ValueError):
        bank.set_oracle_routes({(1,): FACTS})


def test_oracle_route_map_and_norm_scale():
    import layer_sweep_utils as runner
    from static_overlap_data import Example

    class _Tok:
        def __call__(self, text, **_):
            return {"input_ids": [1] + [len(w) for w in text.split()]}

    def ex(fact, prompt, ids, labels):
        return Example(f"{fact}:{prompt}", "train", "forget", fact, ids, labels,
                       prompt, " x", "g")

    answers = {"a": ex("fa", "ab cde", [1, 2, 3, 9], [-100, -100, -100, 9]),
               "b": ex("fb", "abcd", [1, 4, 8], [-100, -100, 8])}
    mapping = runner.oracle_route_map(_Tok(), (answers,), {"fa": 0, "fb": 1})
    assert mapping == {(1, 2, 3): 0, (1, 4): 1}

    clash = {"c": ex("fb", "ab cde", [1, 2, 3, 7], [-100, -100, -100, 7])}
    with pytest.raises(ValueError):
        runner.oracle_route_map(_Tok(), (answers, clash), {"fa": 0, "fb": 1})

    norms = {3: torch.tensor([4.0, 5.0, 6.0]), 19: torch.tensor([19.0, 20.0, 21.0])}
    assert runner.resolve_norm_scale("auto", norms, 3, 19) == pytest.approx(0.25)
    assert runner.resolve_norm_scale("1", norms, 3, 19) == 1.0
    with pytest.raises(ValueError):
        runner.resolve_norm_scale("0", norms, 3, 19)
