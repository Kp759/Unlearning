#!/usr/bin/env python3
"""Locked compatibility entrypoint for SURE-MQuAKE V7 Stage 1."""

from __future__ import annotations

import mquake_v7_locked_case_compat as compat

compat.install()

import mquake_sure_sparse_lm_gagd_v7 as stage1  # noqa: E402


if __name__ == "__main__":
    stage1.main()
