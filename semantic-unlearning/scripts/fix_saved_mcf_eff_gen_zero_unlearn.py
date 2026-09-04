#!/usr/bin/env python3
"""Correct saved MCF JSON Eff/Gen labels to exact ZeroUnlearn semantics.

No model inference is performed. The script reads the already stored
post_rewrite_success/post_paraphrase_success statistics, rewrites Eff/Gen to
those values, and preserves the previous complement as SensitivePref_*.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcf_zero_unlearn_metric_parity import patch_result_payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help="Saved result JSON files")
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite each input. Otherwise write <stem>.zu_parity.json",
    )
    args = p.parse_args()

    for raw in args.inputs:
        src = Path(raw).resolve()
        payload = json.loads(src.read_text(encoding="utf-8"))
        fixed = patch_result_payload(payload)
        dst = src if args.in_place else src.with_name(src.stem + ".zu_parity.json")
        dst.write_text(json.dumps(fixed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        forget = fixed.get("forget", fixed.get("summary", {}))
        print(json.dumps({
            "input": str(src),
            "output": str(dst),
            "Eff": forget.get("Eff"),
            "Gen": forget.get("Gen"),
            "SensitivePref_Eff": forget.get("SensitivePref_Eff"),
            "SensitivePref_Gen": forget.get("SensitivePref_Gen"),
            "zero_unlearn_eff_gen_parity": True,
        }, indent=2))


if __name__ == "__main__":
    main()
