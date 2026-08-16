#!/usr/bin/env python3
"""SURE-TOFU V5 R64 variant.

This is a thin locked wrapper around ``tofu_sure_r512_activepair_v5.py``.  It
keeps every V5 control identical but fixes the primary forget-hidden repair
rank to 64.  Stage 2 remains the same unrestricted residual active-pair repair.

The underlying implementation still uses the historical ``--r512-rank``
argument name and ``r512`` checkpoint/report keys for backward compatibility;
this wrapper enforces that the actual numerical rank requested is exactly 64
and relabels the top-level method metadata as R64.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import tofu_sure_r512_activepair_v5 as v5

METHOD = "SURE-TOFU-R64-active-pair-repair-v5"


def _arg_value(name: str) -> str | None:
    for index, item in enumerate(sys.argv[1:], start=1):
        if item == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        prefix = name + "="
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def _enforce_rank64() -> None:
    value = _arg_value("--r512-rank")
    if value is None:
        sys.argv.extend(["--r512-rank", "64"])
        return
    if int(value) != 64:
        raise ValueError(f"R64 wrapper requires --r512-rank 64, received {value!r}")


def _patch_metadata() -> None:
    output_dir = _arg_value("--output-dir")
    if output_dir is None:
        return
    root = Path(output_dir).expanduser().resolve()
    for name in ("repair_summary.json", "config_used.json"):
        path = root / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["method"] = METHOD
        payload["primary_forget_hidden_rank_requested"] = 64
        payload["primary_repair_label"] = "R64 forget-hidden repair"
        payload["historical_internal_key_note"] = (
            "internal r512 field/checkpoint names are retained only for backward compatibility; actual requested rank is 64"
        )
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    _enforce_rank64()
    v5.METHOD = METHOD
    v5.main()
    _patch_metadata()


if __name__ == "__main__":
    main()
