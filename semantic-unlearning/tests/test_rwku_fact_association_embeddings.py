from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rwku_fact_association_embeddings import build_association_facts


class DummyTokenizer:
    pass


def row(query, answer, digest, seed=1, subject="Confucius", level="2"):
    return {
        "subject": subject,
        "query": query,
        "answer": answer,
        "source_record_sha256": digest,
        "rwku_target_seed": seed,
        "level": level,
    }


def test_distinct_queries_get_distinct_associations():
    rows = [
        row("What was Confucius known for?", "philosophy", "a" * 64),
        row("Where was Confucius born?", "Lu", "b" * 64),
    ]
    facts, mapping, diagnostics = build_association_facts(rows, DummyTokenizer())
    assert len(facts) == 2
    assert diagnostics["duplicate_records_collapsed"] == 0
    assert mapping["a" * 64] != mapping["b" * 64]


def test_exact_duplicate_association_collapses():
    rows = [
        row("Where was Confucius born?", "Lu", "a" * 64),
        row(" Where   was Confucius born? ", " Lu ", "b" * 64),
    ]
    facts, mapping, diagnostics = build_association_facts(rows, DummyTokenizer())
    assert len(facts) == 1
    assert diagnostics["duplicate_records_collapsed"] == 1
    assert mapping["a" * 64] == mapping["b" * 64]


def test_same_natural_prompt_different_answer_fails_closed():
    rows = [
        row("Where was Confucius born?", "Lu", "a" * 64),
        row("Where was Confucius born?", "Qufu", "b" * 64),
    ]
    try:
        build_association_facts(rows, DummyTokenizer())
    except ValueError as exc:
        assert "conflicting sensitive answers" in str(exc)
    else:
        raise AssertionError("Expected natural-address conflict")
