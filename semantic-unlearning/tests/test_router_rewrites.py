"""router_rewrites.py / build_router_rewrites.py: quality-checked router positives (ours).

The claims these protect:
  * cleaning: list markers, labels and talk about the task never become prompts;
  * a rewrite is kept only if it is well formed, names the subject, omits the answer, is not a
    near-duplicate, is answered by the base model (validate) and judged the same fact;
  * facts short of rewrites get a second round; the last rewrite of a kind is held out;
  * translations use languages SCALPEL never probes;
  * each dataset's facts become the right rewrite targets (RWKU level 1 = fill-in-the-blank).
Generation and judging are scripted here; the base-model paths run in build_router_rewrites.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import router_rewrites as RR  # noqa: E402
from build_router_rewrites import dataset_of, targets_of  # noqa: E402

T = RR.RewriteTarget("f0", "Agnès Varda", "Paris", "Agnès Varda died in the city of", "cloze")


def test_text_checks():
    raw = ("Here are three versions:\n1. Version 1: Agnès Varda passed away in the city of\n"
           "- \"The city where Agnès Varda died is\"\nParaphrase 3: Rewrite: Agnès Varda died in")
    assert RR.lines(raw) == ["Agnès Varda passed away in the city of", "The city where Agnès Varda died is"]
    assert RR.has_phrase("She died in Paris.", "paris") and not RR.has_phrase("a Parisian café", "Paris")
    assert RR.near_duplicate("In which city did Agnès Varda die?", ["In which city did Agnes Varda die ?"])
    assert RR.well_formed("Where did Agnès Varda die?", "question")
    assert not RR.well_formed("Where did Agnès Varda die", "question") and not RR.well_formed("Varda", "paraphrase")
    assert set(RR.TRAIN_LANGS).isdisjoint({"fr", "es", "de"})          # SCALPEL's cross-lingual languages


def _scripted(monkeypatch, outputs, judge=lambda pair: 1.0):
    """generate_texts answers each instruction from `outputs` (first matching key)."""
    def fake_generate(model, tok, texts, max_new_tokens, batch_size=16):
        return [next((v for k, v in outputs.items() if k in t), "") for t in texts]
    monkeypatch.setattr(RR, "generate_texts", fake_generate)
    monkeypatch.setattr(RR, "same_fact_scores", lambda model, tok, pairs, batch_size=16: [judge(p) for p in pairs])
    monkeypatch.setattr(RR, "instruct", lambda tok, text: text)


def test_every_check_and_the_same_fact_judge(monkeypatch):
    _scripted(monkeypatch, {
        "Rewrite this unfinished sentence": "\n".join([
            "Agnès Varda passed away in the city of",        # kept
            "Agnès Varda died in the city of",              # the fact's own prompt: near-duplicate
            "Agnès Varda died in Paris, in the city of",    # contains the answer
            "The famous director died in the city of",      # no subject
            "Agnès Varda lived in the city of"]),           # answered, but judged a different fact
        "direct questions": "Where did Agnès Varda die?\nWhat city did Agnès Varda die in",   # 2nd lacks "?"
    }, judge=lambda pair: 0.1 if "lived" in pair[1] else 0.9)
    kept, stats = RR.generate_rewrites(None, None, [T], validate=lambda cs: [c.text for c in cs],
                                       kinds=("paraphrase", "question"), n_para=6, n_question=3, min_views=0)
    texts = [c.text for c in kept["f0"]]
    assert texts == ["Agnès Varda passed away in the city of", "Where did Agnès Varda die?"]
    assert stats["dropped_near_duplicate"] == 1 and stats["dropped_form"] == 3 and stats["dropped_not_same_fact"] == 1


def test_base_model_must_answer_and_second_round(monkeypatch):
    _scripted(monkeypatch, {
        "Rewrite this unfinished sentence": "Agnès Varda passed away in the city of\nThe city of Agnès Varda's death was",
        "Say exactly what this unfinished sentence says": "Agnès Varda's life ended in the city of",
    })
    answered = {"Agnès Varda passed away in the city of", "Agnès Varda's life ended in the city of"}
    kept, stats = RR.generate_rewrites(None, None, [T], validate=lambda cs: [c.text if c.text in answered else None
                                                                             for c in cs],
                                       kinds=("paraphrase",), n_para=6, min_views=3)
    assert stats["dropped_base_does_not_answer"] == 1 and stats["second_round_targets"] == 1
    assert [c.text for c in kept["f0"]] == ["Agnès Varda passed away in the city of",
                                           "Agnès Varda's life ended in the city of"]
    rows = RR.as_examples(kept, {"f0": "fact_0"})
    assert [r["split"] for r in rows] == ["train", "development"]          # the last of a kind is held out
    assert {r["fact_id"] for r in rows} == {"fact_0"} and all(r["augmented"] for r in rows)


def test_dataset_targets():
    facts = [{"id": "rwku_assoc_1_a", "subject": "Stephen King", "object": "Carrie", "rwku_level": "1",
              "query": "Stephen King's debut novel, titled ___, established him."},
             {"id": "rwku_assoc_1_b", "subject": "Stephen King", "object": "Maine", "rwku_level": "2",
              "query": "Which state was Stephen King born in?"}]
    rwku = targets_of(facts, dataset_of(facts))
    assert [(t.fmt, t.answer) for t in rwku] == [("blank", "Carrie"), ("qa", "Maine")]
    plain = [{"id": "zsre_1", "subject": "Alain Renoir", "object": "French",
              "canonical_prompt": "What was the native language of Alain Renoir?"},
             {"id": "zsre_2", "subject": "Paris", "object": "France", "canonical_prompt": "Paris is located in"}]
    assert [t.fmt for t in targets_of(plain, dataset_of(plain))] == ["qa", "cloze"]
