#!/usr/bin/env python3
"""Locked compatibility entrypoint for SURE-MQuAKE V7 Stage 2."""

from __future__ import annotations

import mquake_v7_locked_case_compat as compat

compat.install()

import mquake_sure_active_hidden_repair_v7 as stage2  # noqa: E402


if __name__ == "__main__":
    stage2.main()
