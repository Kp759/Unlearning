#!/usr/bin/env python3
"""Compatibility entry point for canonical MCF MEMIT.

Installs runtime-only compatibility for the paper-source TokenizedDataset before
executing the canonical model-editing adapter.  No file under ZeroUnlearn/ is
modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SEMANTIC_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SEMANTIC_ROOT.parent
ZERO_ROOT = REPO_ROOT / "ZeroUnlearn"

if str(ZERO_ROOT) not in sys.path:
    sys.path.insert(0, str(ZERO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rome.tok_dataset import TokenizedDataset
from zero_unlearn_transformers_compat import install_tokenized_dataset_concat_compat

installed = install_tokenized_dataset_concat_compat(TokenizedDataset)
print(
    "MEMIT dataset compatibility: "
    + ("installed" if installed else "already installed")
)

import run_model_editing_canonical_mcf as canonical_runner

if __name__ == "__main__":
    canonical_runner.main()
