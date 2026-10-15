"""Same-subject, other-relation router negatives (fit_linear_router.py --relation-negatives)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_linear_router import relation_negative_controls  # noqa: E402
from linear_router import assemble_router_dataset, examples_from_facts  # noqa: E402
from mcf_subject_collateral import other_relations  # noqa: E402
from static_overlap_fact_association_embeddings import make_subject_patterns  # noqa: E402
from test_linear_router import _WordTokenizer, _facts  # noqa: E402


def _mcf(tmp_path, templates):
    """templates: {relation: [(template, count)]} -> a minimal MCF json."""
    records = []
    for relation, items in templates.items():
        for template, count in items:
            records += [{"requested_rewrite": {"relation_id": relation, "prompt": template,
                                               "subject": "X"}}] * count
    path = tmp_path / "mcf.json"
    path.write_text(json.dumps(records))
    return str(path)


def test_controls_use_preserve_relations_rank_splits_and_never_eval_templates(tmp_path):
    person = ["P27", "P19", "P106", "P20", "P1412", "P101", "P108", "P140", "P937", "P103", "P39"]
    templates = {r: [(f"{{}} {r} a", 3), (f"{{}} {r} b", 2), (f"{{}} {r} c", 1)] for r in person}
    eval_rel = other_relations("P19", templates, "eval")[0]
    preserve_rel = other_relations("P19", templates, "preserve")[0]
    templates[preserve_rel].insert(0, (f"{{}} {eval_rel} a", 9))   # shared wording -> banned
    path = _mcf(tmp_path, templates)
    controls = relation_negative_controls([{"subject": "Alice", "relation": "P19"}], path, 3)[0]
    relations = {c["donor_relation"] for c in controls}
    assert relations == set(other_relations("P19", templates, "preserve"))
    assert not relations & set(other_relations("P19", templates, "eval"))
    assert all(f"Alice {eval_rel} " not in c["prompt"] for c in controls)
    first = [c for c in controls if c["donor_relation"] == preserve_rel]
    assert [c["split"] for c in first] == ["fit", "calibration", "audit"]
    assert first[0]["prompt"] == f"Alice {preserve_rel} a"          # banned one skipped


def test_extra_controls_are_negatives_for_their_head_only():
    facts = _facts()
    tokenizer = _WordTokenizer()
    patterns = make_subject_patterns(tokenizer, facts)
    examples = examples_from_facts(facts, augment=True)
    extra = {0: [  # Alice Smith (P19): a new prompt, and Alice's real P27 positive
        {"prompt": "Alice Smith speaks the language", "kind": "relation_probe", "split": "calibration",
         "donor_relation": "P106", "group": "relation_probe_t1"},
        {"prompt": "Alice Smith is a citizen of", "kind": "relation_probe", "split": "fit",
         "donor_relation": "P27", "group": "relation_probe_t0"},
    ]}
    data = assemble_router_dataset(facts, examples, tokenizer, patterns, negative_count=9,
                                   per_donor=1, extra_controls=extra)
    by_prompt = {r["prompt"]: r for r in data["records"]}
    new = by_prompt["Alice Smith speaks the language"]
    assert new["owner"] == -1 and 0 in new["negative_for"] and new["split"] == "calibration"
    real = by_prompt["Alice Smith is a citizen of"]
    assert real["owner"] == 3 and 0 in real["negative_for"]         # stays fact 3's positive
    row = data["prompts"].index("Alice Smith speaks the language")
    assert not bool(data["labels"][row].any()) and bool(data["eligible"][row, 0])
