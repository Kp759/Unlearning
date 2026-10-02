"""MQuAKE answer aliases: lookup, target classification, training cases, leak summary."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import mquake_answer_aliases as al  # noqa: E402

RAW = [
    {"single_hops": [{"cloze": "Fernando Santos is a citizen of", "answer": "Portugal",
                      "answer_alias": ["POR", "Portuguese Republic", "PT", "portugal"]}],
     "new_single_hops": [{"cloze": "The capital of Portugal is", "answer": "Lisbon",
                          "answer_alias": ["Lisboa"]}]},
    {"single_hops": [{"cloze": "Fernando Santos is a citizen of", "answer": "Portugal",
                      "answer_alias": ["Portuguese Republic", "the Portuguese state"]}]},
]


def _record(case_id=1, subject="Fernando Santos", prompt="{} is a citizen of", answer="Portugal"):
    return {"case_id": case_id, "atomic_gen_prompt": f"What is the country of citizenship of {subject}?",
            "requested_rewrite": {"prompt": prompt, "subject": subject, "relation_id": "P27",
                                  "target_true": {"str": answer}}}


def _tokenizer(words):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, **{w: i + 4 for i, w in enumerate(sorted(set(words)))}}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    return transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                                bos_token="<s>", eos_token="</s>", unk_token="<unk>")


WORDS = ("Fernando Santos is a citizen of Portugal POR Portuguese Republic PT the state What country "
         "citizenship ? Lisbon The capital").split()


def test_alias_lookup_pools_hops_and_drops_the_answer():
    table = al.alias_table(RAW)
    assert table[("Fernando Santos is a citizen of", "Portugal")] == [
        "POR", "PT", "Portuguese Republic", "portugal", "the Portuguese state"]
    assert al.record_aliases(_record(), table) == ["POR", "PT", "Portuguese Republic", "the Portuguese state"]
    assert al.record_aliases(_record(answer="Spain"), table) == []


def test_alias_kinds_and_training_targets():
    tok = _tokenizer(WORDS)
    aliases = ["POR", "PT", "Portuguese Republic", "Portugal state", "the Portuguese state"]
    kinds = {a: k for a, _, k in al.classify_aliases(tok, "Portugal", aliases, llama_like=True)}
    assert kinds["Portugal state"] == "alias_same_first" and kinds["POR"] == "alias_diff_first"
    targets = al.alias_training_targets(tok, "Portugal", aliases, llama_like=True)
    names = [a for a, _ in targets]
    assert names == ["POR", "PT", "Portuguese Republic"]           # same-first and generic "the" dropped
    assert all(tok.decode([t]) == a.split()[0] for a, t in targets)
    assert al.is_generic_token(" U") and al.is_generic_token(" the") and not al.is_generic_token(" PT")


def test_alias_token_cases_share_the_answer_boundary():
    tok = _tokenizer(WORDS)
    facts = [{"id": "mquake_fact_0", "association_key": "fernando santos\tP27\tportugal"}]
    records = [_record(1), _record(2)]                               # duplicate record, one row
    from mquake_fact_association_embeddings import association_key_from_record
    facts[0]["association_key"] = association_key_from_record(records[0])
    cases, info = al.alias_token_cases(records, facts, tok, llama_like=True, table=al.alias_table(RAW))
    assert info["alias_cases"] == 3 and info["facts_with_alias_targets"] == 1
    assert {c.boundary_prompt for c in cases} == {"Fernando Santos is a citizen of"}
    assert all(c.prompt == c.boundary_prompt and c.token_index == 0 for c in cases)
    assert {c.target_text for c in cases} == {"POR", "PT", "Portuguese"}


def test_leak_scoring_and_summary():
    transformers = pytest.importorskip("transformers")
    import evaluate_mquake_alias_leak as ev

    tok = _tokenizer(WORDS)
    table = al.alias_table(RAW)
    items, coverage = ev.build_items([_record(1), _record(2)], tok, table, llama_like=True,
                                     row_of_key=None)
    assert coverage == {"facts": 1, "facts_with_aliases": 1}
    assert {i["kind"] for i in items} == {"answer", "alias_diff_first"}
    assert len(items) == 2 * 5                                       # 2 prompt types x (answer + 4 aliases)
    config = transformers.LlamaConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                                      max_position_embeddings=64)
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).eval()
    rows = ev.score_items(model, tok, items, torch.device("cpu"), batch_size=3)
    assert len(rows) == len(items)
    first = rows[0]
    ids = tok(items[0]["prompt"])["input_ids"]
    p = torch.softmax(model(input_ids=torch.tensor([ids])).logits[0, -1].float(), -1)[items[0]["target_ids"][0]]
    assert first["first_token_probability"] == pytest.approx(float(p), rel=1e-4)

    # summary: answer forgotten under SURE, one alias still greedy -> recovery 1.0
    base = [dict(r, greedy=(r["kind"] == "answer")) for r in rows]
    sure = [dict(r, greedy=(r["kind"] != "answer" and r["target"] == "POR"), routed_to_own_row=True)
            for r in rows]
    summary = ev.summarize(base, sure)
    block = summary["rewrite"]
    assert block["answer_forgotten_facts"] == 1 and block["alias_recovery_rate"] == 1.0
    assert block["alias_recovery_rate_own_row"] == 1.0
    assert block["targets_by_kind"]["answer"]["greedy"] == {"base": 1.0, "sure": 0.0}


def test_alias_targets_flag_is_mquake_only():
    from train_direct_linear_router_rows import main

    with pytest.raises(ValueError, match="MQuAKE only"):
        main(["--dataset", "zsre", "--router-dir", "x", "--output-dir", "y",
              "--training-route", "router", "--alias-targets"])
