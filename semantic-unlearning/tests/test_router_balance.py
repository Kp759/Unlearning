"""Fixed router training-set sizes per fact (fit_linear_router --positives-per-fact /
--negatives-per-fact, default 20 / 20 for ours).

The claims these protect:
  * every fact gets exactly the requested positives: its own examples first, then context-
    prefix variants (families context_prefix_4+, about a quarter development); none of the
    added prompts loses subject eligibility, so assembly drops none of them;
  * a longer list is cut keeping canonical first and some development positives;
  * every fact gets exactly the requested negatives: up to half same-subject competitors,
    the rest subject transplants, both spread over fit / calibration / audit; different
    questions first, lead-in variants of one question only to fill;
  * the added prefixes are not SCALPEL's probe prefixes (no test templates in training).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linear_router import (  # noqa: E402
    EXTRA_CONTEXT_PREFIXES,
    GENERIC_CONTEXT_PREFIXES,
    assemble_router_dataset,
    balance_positive_examples,
    eligibility_matrix,
    examples_from_facts,
)
from static_overlap_fact_association_embeddings import make_subject_patterns  # noqa: E402
from test_linear_router import _WordTokenizer, _facts  # noqa: E402


def _balanced(target):
    facts, tok = _facts(), _WordTokenizer()
    patterns = make_subject_patterns(tok, facts)
    subjects = [f["subject"] for f in facts]

    def eligible(prompt, index):
        return bool(eligibility_matrix(tok, [prompt], patterns, subjects=subjects)[0, index])
    examples, report = balance_positive_examples(examples_from_facts(facts, augment=True), facts, target,
                                                 eligible=eligible)
    return facts, tok, patterns, examples, report


def test_positives_topped_up_to_exactly_the_target():
    facts, tok, patterns, examples, report = _balanced(20)
    per_fact = Counter(e["fact_id"] for e in examples)
    assert set(per_fact.values()) == {20} and not report["short"]
    first = [e for e in examples if e["fact_id"] == facts[0]["id"]]
    assert first[0]["group"] == "canonical_0" and [e["group"] for e in first[1:5]] == [
        f"context_prefix_{k}" for k in range(4)]                          # its own examples come first
    added = first[5:]
    assert all(int(e["group"].rsplit("_", 1)[1]) >= 4 and e["augmented"] for e in added)
    assert sum(e["split"] == "development" for e in added) == len(added) // 4   # families 7, 11, 15
    data = assemble_router_dataset(facts, examples, tok, patterns, negative_count=9, per_donor=1)
    counts = data["diagnostics"]["per_fact_counts"]["positives"]
    assert counts["min"] == counts["max"] == 20 and data["diagnostics"]["augmented_positives_dropped_ineligible"] == 0


def test_longer_lists_are_cut_keeping_canonical_and_development():
    facts, _, _, examples, report = _balanced(3)
    first = [e for e in examples if e["fact_id"] == facts[0]["id"]]
    assert len(first) == 3 and first[0]["group"] == "canonical_0"
    assert any(e["split"] == "development" for e in first) and report["cut"] == 2 * len(facts)


def test_negatives_exactly_the_target_half_competitors_half_transplants():
    facts, tok, patterns, examples, _ = _balanced(20)
    data = assemble_router_dataset(facts, examples, tok, patterns, per_donor=3, negatives_per_fact=6)
    counts = data["diagnostics"]["per_fact_counts"]
    assert counts["negatives"]["min"] == counts["negatives"]["max"] == 6
    alice = [r for r in data["records"] if 0 in r["negative_for"]]       # Alice P19; Alice P27 is fact 3
    real = [r for r in alice if r["owner"] == 3]
    assert len(real) == 3 and {r["split"] for r in real} == {"fit", "calibration", "audit"}
    assert sum(r["kind"] == "subject_transplant" for r in alice) == 3
    for split in ("fit", "calibration", "audit"):                         # calibration needs owner-less controls
        assert data["diagnostics"]["by_split"][split]["negative_controls"] > 0


def test_negatives_prefer_distinct_questions_over_lead_in_variants():
    from linear_router import _core_prompt, compose_negatives
    base = [{"prompt": f"{lead} Ann Lee works as a".strip(), "kind": "subject_transplant", "split": "fit",
             "donor_relation": "P106"} for lead in ("", "Reportedly,", "As the records show,")]
    other = [{"prompt": "Ann Lee is a citizen of", "kind": "subject_transplant", "split": "fit", "donor_relation": "P27"},
             {"prompt": "Ann Lee died in", "kind": "subject_transplant", "split": "calibration", "donor_relation": "P20"}]
    picked = compose_negatives(base + other, 3)
    assert len({_core_prompt(c["prompt"]) for c in picked}) == 3          # three different questions first
    assert len(compose_negatives(base + other, 5)) == 5                   # then lead-in variants fill


def test_added_prefixes_are_not_scalpel_probe_prefixes():
    probe = Path(__file__).resolve().parents[1] / "snorkel_proposal" / "scalpel" / "relations.py"
    text = probe.read_text().casefold()
    assert len(set(EXTRA_CONTEXT_PREFIXES)) == len(EXTRA_CONTEXT_PREFIXES) == 16
    assert not set(EXTRA_CONTEXT_PREFIXES) & set(GENERIC_CONTEXT_PREFIXES)
    assert not any(p.rstrip(",").casefold() in text for p in EXTRA_CONTEXT_PREFIXES)


def test_prefix_copies_are_capped_and_shortfalls_reported():
    facts, tok = _facts(), _WordTokenizer()
    patterns = make_subject_patterns(tok, facts)
    rows = examples_from_facts(facts, augment=True)
    rows.append({"fact_id": facts[0]["id"], "prompt": "The birthplace of Alice Smith is", "split": "train",
                 "group": "aug_paraphrase_0", "role": "aug_paraphrase_0"})
    examples, report = balance_positive_examples(rows, facts, 20, max_prefixed=8)
    first = [e for e in examples if e["fact_id"] == facts[0]["id"]]
    assert sum(e["group"].startswith("context_prefix") for e in first) == 8
    assert len(first) == 1 + 1 + 8 and report["short"][facts[0]["id"]] == 10      # not padded to 20
    assert [e["group"] for e in first[:2]] == ["canonical_0", "aug_paraphrase_0"]  # real phrasings first
    assert any("The birthplace of Alice Smith is" in e["prompt"] and e["group"].startswith("context_prefix")
               for e in first)                                                   # lead-ins on the rewrite too
