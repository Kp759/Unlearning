from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mcf_target_representation_compare_fix5f_seed1.py"
spec = importlib.util.spec_from_file_location("fix5f_target_rep", SCRIPT)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def test_erased_marker_visible_in_contextual_prefix():
    text = "Which organization is TARGET_ENTITY a member of?"
    vis = m.marker_visibility_in_prefix(text, len(text), ["TARGET_ENTITY"])
    assert vis == {"TARGET_ENTITY": True}


def test_marked_open_and_close_visible_when_full_span_retained():
    text = "Which organization is [TARGET]Belgium[/TARGET] a member of?"
    vis = m.marker_visibility_in_prefix(text, len(text), ["[TARGET]", "[/TARGET]"])
    assert vis == {"[TARGET]": True, "[/TARGET]": True}


def test_close_marker_reported_missing_if_truncation_cuts_target_span():
    text = "prefix [TARGET]Belgium[/TARGET] suffix"
    retained_end = text.index("[/TARGET]")
    vis = m.marker_visibility_in_prefix(text, retained_end, ["[TARGET]", "[/TARGET]"])
    assert vis["[TARGET]"] is True
    assert vis["[/TARGET]"] is False


def test_wrapper_patches_only_encode_function_entrypoint():
    assert m.core.encode_texts is m.encode_texts
