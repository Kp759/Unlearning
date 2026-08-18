#!/usr/bin/env python3
"""Compatibility entry point for canonical MCF MEMIT.

Installs runtime-only compatibility for the paper-source TokenizedDataset before
executing the canonical model-editing adapter. No file under ZeroUnlearn/ is
modified.

The pinned ZeroUnlearn package reads ``globals.yml`` at import time using the
process working directory. We therefore import its TokenizedDataset while
*temporarily* inside ZeroUnlearn/, then restore the caller's working directory
before invoking the canonical runner. This keeps relative canonical paths
resolved from semantic-unlearning/ exactly as the shell runner expects.
"""
from __future__ import annotations

import os
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

# ZeroUnlearn/util/globals.py opens ``globals.yml`` relative to cwd during
# package import. Import under ZERO_ROOT, but immediately restore cwd so the
# canonical adapter can resolve its CLI paths from semantic-unlearning/.
_CALLER_CWD = Path.cwd()
os.chdir(ZERO_ROOT)
try:
    from rome.tok_dataset import TokenizedDataset
finally:
    os.chdir(_CALLER_CWD)

from zero_unlearn_transformers_compat import install_tokenized_dataset_concat_compat

installed = install_tokenized_dataset_concat_compat(TokenizedDataset)
print(
    "MEMIT dataset compatibility: "
    + ("installed" if installed else "already installed")
)

import run_model_editing_canonical_mcf as canonical_runner

if __name__ == "__main__":
    canonical_runner.main()
