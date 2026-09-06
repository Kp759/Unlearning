from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mcf_target_local_recognition_baseline_fix5j_seed1.py"
spec = importlib.util.spec_from_file_location("fix5j_target_local", SCRIPT)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def test_mixed_query_family_has_own_bucket():
    assert m.core.base.bucket("mixed_forbidden_distractor_calib") == "mixed_forbidden_distractor"
    assert m.core.base.bucket("mixed_forbidden_distractor_validation") == "mixed_forbidden_distractor"


def test_existing_buckets_are_unchanged():
    assert m.core.base.bucket("crossed_binding") == "crossed_binding"
    assert m.core.base.bucket("same_subject_different_relation") == "same_subject_different_relation"
    assert m.core.base.bucket("retain_other") == "other_retain"
