from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_fix5o_matched_end_to_end_fix5p_seed1 as m


def _row(group: str, case_id: int, base_disclose: bool, old_accept: bool, new_accept: bool,
         base_pref: bool = True, old_pref: bool = False, new_pref: bool = False):
    def scores(pref: bool):
        return {"target_true": 1.0 if pref else 3.0, "target_new": 2.0}
    def gen(disclose: bool):
        return {
            "text": "secret" if disclose else "other",
            "token_ids": [1] if disclose else [2],
            "flags": {"target_true_canonical_mentioned": disclose, "target_new_canonical_mentioned": False},
        }
    return {
        "kind": "atomic",
        "group": group,
        "case_id": case_id,
        "query": f"q{case_id}",
        "expected_binding": [f"S{case_id}", "P1"],
        "route_cohorts": {
            "old": "correctly_accepted" if old_accept else "misclassified",
            "fix5o": "correctly_accepted" if new_accept else "misclassified",
        },
        "routes": {
            "old": {"active_bindings": [[f"S{case_id}", "P1"]] if old_accept else []},
            "fix5o": {"active_bindings": [[f"S{case_id}", "P1"]] if new_accept else []},
        },
        "scores": {
            "base": scores(base_pref),
            "old_integrated": scores(old_pref),
            "fix5o_integrated": scores(new_pref),
        },
        "generation": {
            "base": gen(base_disclose),
            "old_integrated": gen(base_disclose and not old_accept),
            "fix5o_integrated": gen(base_disclose and not new_accept),
        },
    }


def test_summarize_group_tracks_pref_disclosure_and_paired_acceptance():
    rows = [
        _row("paraphrase", 1, True, True, True),
        _row("paraphrase", 2, True, False, True),
        _row("paraphrase", 3, True, True, False),
        _row("paraphrase", 4, False, False, False),
    ]
    s = m.summarize_group(rows)
    assert s["n"] == 4
    assert s["old_routing"]["correctly_accepted_n"] == 2
    assert s["fix5o_routing"]["correctly_accepted_n"] == 2
    assert s["old_routing"]["correctly_accepted_base_disclosing_n"] == 2
    assert s["fix5o_routing"]["correctly_accepted_base_disclosing_n"] == 2
    assert s["paired_correct_acceptance"] == {
        "retained_n": 1, "gained_n": 1, "lost_n": 1, "neither_n": 1
    }
    assert s["paired_base_disclosure_coverage"] == {
        "newly_covered_n": 1, "lost_coverage_n": 1
    }
    assert s["base"]["canonical_sensitive_disclosure_n"] == 3
    assert s["old_integrated"]["canonical_sensitive_disclosure_n"] == 1
    assert s["fix5o_integrated"]["canonical_sensitive_disclosure_n"] == 1


def test_historical_identity_audit_passes_exact_match(tmp_path: Path):
    rows = [_row("direct", 1, True, True, True)]
    current = rows[0]
    old = {
        "kind": "atomic",
        "group": "direct",
        "case_id": 1,
        "query": "q1",
        "expected_binding": ["S1", "P1"],
        "route": {"active_bindings": [["S1", "P1"]]},
        "conditions": {
            "base": {"token_ids": current["generation"]["base"]["token_ids"]},
            "integrated": {"token_ids": current["generation"]["old_integrated"]["token_ids"]},
        },
    }
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps(old) + "\n", encoding="utf-8")
    audit = m.historical_identity_audit(rows, [], path)
    assert audit["strict_reproduction_pass"] is True
    assert audit["base_exact_token_match_pct"] == 100.0
    assert audit["old_integrated_exact_token_match_pct"] == 100.0
    assert audit["old_active_binding_exact_match_pct"] == 100.0


def test_historical_identity_audit_rejects_query_mismatch(tmp_path: Path):
    rows = [_row("direct", 1, True, True, True)]
    old = {
        "kind": "atomic", "group": "direct", "case_id": 1,
        "query": "different", "expected_binding": ["S1", "P1"],
        "route": {"active_bindings": [["S1", "P1"]]},
        "conditions": {"base": {"token_ids": [1]}, "integrated": {"token_ids": [2]}},
    }
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps(old) + "\n", encoding="utf-8")
    try:
        m.historical_identity_audit(rows, [], path)
    except ValueError as exc:
        assert "query identities differ" in str(exc)
    else:
        raise AssertionError("expected identity mismatch to fail")


def test_load_jsonl_rejects_non_object(tmp_path: Path):
    path = tmp_path / "x.jsonl"
    path.write_text("[]\n", encoding="utf-8")
    try:
        m.load_jsonl(path)
    except ValueError as exc:
        assert "expected JSON object" in str(exc)
    else:
        raise AssertionError("expected non-object JSONL row to fail")
