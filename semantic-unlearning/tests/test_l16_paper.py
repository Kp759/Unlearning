"""Genie routing on the early-write model, evaluator-compatible counters, and the
dataset-agnostic training examples used by the L16 paper runs."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from early_write import EarlyWriteCausalLM  # noqa: E402
from test_early_write import HIDDEN, ROUTED, UNROUTED, _artifact, _tiny_llama  # noqa: E402

transformers = pytest.importorskip("transformers")


def _logits(model, ids):
    model.set_association_prefix_lengths([ids.shape[1]])
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits


def test_genie_routes_mapped_prompt_ignores_router_and_leaves_others_base():
    base = _tiny_llama()
    with torch.no_grad():
        ref_routed, ref_unrouted = base(input_ids=ROUTED).logits, base(input_ids=UNROUTED).logits
    model = EarlyWriteCausalLM(base, _artifact(), write_layer=0, write_mode="query",
                               rows=torch.full((2, HIDDEN), 3.0))
    # the router fires fact 0 on ROUTED; genie says UNROUTED is fact 1 and ROUTED is nothing
    model.set_oracle_routes({tuple(UNROUTED[0].tolist()): 1})
    assert torch.equal(_logits(model, ROUTED), ref_routed)
    assert model.last_active_fact_indices == [[]]
    assert not torch.equal(_logits(model, UNROUTED), ref_unrouted)
    assert model.last_active_fact_indices == [[1]]
    c = model.counters()
    assert c["oracle_lookups"] == 2 and c["oracle_hits"] == 1 and c["genie"] == {"table_size": 1}
    model.set_oracle_routes(None)                       # back to the router
    _logits(model, ROUTED)
    assert model.last_active_fact_indices == [[0]]
    model.close()


def test_genie_key_ignores_leading_bos():
    base = _tiny_llama()
    model = EarlyWriteCausalLM(base, _artifact(), write_layer=0, write_mode="query",
                               rows=torch.full((2, HIDDEN), 3.0))
    model.set_oracle_routes({tuple(UNROUTED[0, 1:].tolist()): 1})     # built without BOS
    _logits(model, UNROUTED)                                           # fed with BOS
    assert model.last_active_fact_indices == [[1]]
    with pytest.raises(ValueError):
        model.set_oracle_routes({(5, 6): 7})                            # no such fact
    model.close()


def test_counters_carry_the_keys_the_zsre_mquake_rwku_evaluators_difference():
    base = _tiny_llama()
    model = EarlyWriteCausalLM(base, _artifact(), write_layer=0, write_mode="query",
                               rows=torch.full((2, HIDDEN), 1.0))
    before = model.counters()
    _logits(model, ROUTED)
    _logits(model, UNROUTED)
    after = model.counters()
    assert after["hook_calls"] - before["hook_calls"] == 2
    assert after["active_batch_rows"] - before["active_batch_rows"] == 1
    assert after["active_token_positions"] - before["active_token_positions"] == 1   # boundary only
    assert [a - b for a, b in zip(after["active_fact_counts"], before["active_fact_counts"])] == [1, 0]
    model.close()


class _Tok:
    """Whitespace tokenizer with offsets (fast-tokenizer surface used by _encode)."""
    is_fast = True

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        ids, offs, pos = ([1], [(0, 0)], 0) if add_special_tokens else ([], [], 0)
        for word in text.split(" "):
            start = text.index(word, pos) if word else pos
            if word:
                ids.append(3 + sum(map(ord, word)) % 50)
                offs.append((start - 1 if start > 0 else start, start + len(word)))
            pos = start + len(word)
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out


def test_generic_examples_from_router_facts(tmp_path):
    from train_mcf_jlens_feedback_rows import answer_token_sets, detect_dataset, generic_examples

    facts = [{"id": "zsre_forget_7", "role": "forget", "subject": "Paris", "object": "France",
              "canonical_prompt": "Which country is Paris in?", "case_id": 7}]
    assert detect_dataset(facts) == "zsre"
    examples, skipped = generic_examples(tmp_path, facts, _Tok())
    groups = {e.group for e in examples}
    assert "canonical_0" in groups and {e.split for e in examples} == {"train", "development"}
    assert skipped == 0 and all(e.completion == " France" for e in examples)
    answers = answer_token_sets(_Tok(), examples)
    assert set(answers) == {"zsre_forget_7"} and answers["zsre_forget_7"]
