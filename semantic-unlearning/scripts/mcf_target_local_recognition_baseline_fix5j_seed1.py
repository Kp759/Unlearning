#!/usr/bin/env python3
"""Fix5j wrapper for Fix5i: preserve mixed-query family accounting.

Fix5i introduces phase-local `mixed_forbidden_distractor_*` routes. The historical
bucket helper predates that family and otherwise maps them to `other_retain`. This
wrapper keeps the experiment unchanged while giving the new family an explicit
route/query bucket for calibration and reporting.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_target_local_recognition_baseline_fix5i_seed1 as core

_ORIGINAL_BUCKET = core.base.bucket


def target_local_bucket(kind: str) -> str:
    if "mixed_forbidden_distractor" in str(kind):
        return "mixed_forbidden_distractor"
    return _ORIGINAL_BUCKET(kind)


core.base.bucket = target_local_bucket


if __name__ == "__main__":
    core.main()
