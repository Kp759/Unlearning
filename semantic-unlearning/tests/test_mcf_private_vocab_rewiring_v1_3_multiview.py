from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_mcf_private_vocab_rewiring_v1_3_training_views as builder  # noqa: E402
import run_mcf_private_vocab_rewiring_v1_3_multiview as runner  # noqa: E402


def direct_record():
    return {
        "case_id": 7,
        "requested_rewrite": {
            "prompt": "The language used by {} is",
            "subject": "Example Person",
            "relation_id": "P103",
            "target_true": {"str": "French"},
            "target_new": {"str": "Russian"},
        },
        "data_role": "forget",
    }


def test_sanitized_direct_record_is_accepted():
    builder.validate_sanitized_forget([direct_record()])


def test_any_official_probe_field_is_rejected():
    record = direct_record()
    record["paraphrase_prompts"] = ["forbidden"]
    with pytest.raises(RuntimeError, match="held-out probe leaked"):
        builder.validate_sanitized_forget([record])


def test_literal_answer_leak_filter_is_case_insensitive():
    assert builder.literal_answer_leak(
        "Example Person primarily spoke FRENCH", "French", "Russian"
    )
    assert not builder.literal_answer_leak(
        "Which language did Example Person primarily speak? Answer:",
        "French",
        "Russian",
    )


def test_subject_to_template_requires_exactly_one_literal_subject():
    assert (
        builder.subject_to_template(
            "Which language did Example Person primarily speak? Answer:",
            "Example Person",
        )
        == "Which language did {} primarily speak? Answer:"
    )
    assert builder.subject_to_template("No subject here", "Example Person") is None
    assert (
        builder.subject_to_template(
            "Example Person and Example Person spoke", "Example Person"
        )
        is None
    )


def write_corpus(path: Path, *, leak: bool = False) -> None:
    payload = {
        "schema_version": 1,
        "protocol": runner.VIEW_CORPUS_PROTOCOL,
        "leakage_contract": {
            "full_mcf_path_accepted": False,
            "official_paraphrase_prompts_read": leak,
            "official_neighborhood_prompts_read": False,
            "official_generation_prompts_read": False,
            "official_retain_records_read": False,
            "generator_received_target_true": False,
            "generator_received_target_new": False,
        },
        "source_sha256": "abc",
        "seed": 13131,
        "views_per_case": 3,
        "synthetic_views_per_case": 2,
        "semantic_filter": {},
        "cases": [
            {
                "case_id": 7,
                "subject": "Example Person",
                "relation_id": "P103",
                "views": [
                    {"template": "The language used by {} is"},
                    {"template": "Which language did {} speak? Answer:"},
                    {"template": "{} primarily communicated in"},
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_view_corpus_loader_accepts_only_leakage_clean_manifest(tmp_path: Path):
    path = tmp_path / "views.json"
    write_corpus(path)
    view_map, meta = runner.load_view_corpus(path)
    assert view_map[7] == [
        "The language used by {} is",
        "Which language did {} speak? Answer:",
        "{} primarily communicated in",
    ]
    assert meta["views_per_case"] == 3


def test_view_corpus_loader_rejects_any_claimed_official_probe_access(tmp_path: Path):
    path = tmp_path / "views.json"
    write_corpus(path, leak=True)
    with pytest.raises(RuntimeError, match="fails leakage contract"):
        runner.load_view_corpus(path)


def test_view_records_replace_only_prompt():
    runner._VIEW_MAP = {
        7: [
            "The language used by {} is",
            "Which language did {} speak? Answer:",
        ]
    }
    record = direct_record()
    views = runner.view_records_for_case(record)
    assert len(views) == 2
    assert views[0]["requested_rewrite"]["target_true"] == {"str": "French"}
    assert views[1]["requested_rewrite"]["target_new"] == {"str": "Russian"}
    assert views[1]["requested_rewrite"]["prompt"] == "Which language did {} speak? Answer:"
    assert record["requested_rewrite"]["prompt"] == "The language used by {} is"
