#!/usr/bin/env python3
"""MQuAKE layer sweep, step 3 (kept for its CLI): see train_direct_linear_router_rows.py."""
from __future__ import annotations

import sys

from train_direct_linear_router_rows import main as _main


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    return _main(["--dataset", "mquake", *argv])


if __name__ == "__main__":
    raise SystemExit(main())
