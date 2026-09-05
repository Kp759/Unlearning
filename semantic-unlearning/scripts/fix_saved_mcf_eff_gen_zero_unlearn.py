#!/usr/bin/env python3
"""Recompute saved MCF Eff/Gen using ZeroUnlearn paper Eq. (16).

No model inference is performed.  The input JSON must contain ``forget_raw``
(and, when present, ``retain_raw``) with per-prompt ``target_true`` NLLs.
Paper-facing metrics are:

  Eff = 100 * mean P(target_true | rewrite prompt)
  Gen = 100 * mean P(target_true | paraphrase prompt)

where the stored token-averaged NLL gives P = exp(-NLL).  CounterFact pairwise
``post_*_success`` is preserved separately as ``CF_EditSuccess_*`` and is NOT
reported as ZeroUnlearn Eff/Gen.
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
        help="Overwrite each input. Otherwise write <stem>.zu_paper_metrics.json",
    )
    args = p.parse_args()

    for raw_path in args.inputs:
        src = Path(raw_path).resolve()
        payload = json.loads(src.read_text(encoding="utf-8"))
        fixed = patch_result_payload(payload)
        dst = src if args.in_place else src.with_name(src.stem + ".zu_paper_metrics.json")
        dst.write_text(json.dumps(fixed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        forget = fixed.get("forget", fixed.get("summary", {}))
        if not isinstance(forget, dict):
            forget = {}
        print(json.dumps({
            "input": str(src),
            "output": str(dst),
            "ZeroUnlearn_paper_Eff": forget.get("Eff"),
            "ZeroUnlearn_paper_Gen": forget.get("Gen"),
            "CF_EditSuccess_Eff": forget.get("CF_EditSuccess_Eff"),
            "CF_EditSuccess_Gen": forget.get("CF_EditSuccess_Gen"),
            "SensitivePref_Eff": forget.get("SensitivePref_Eff"),
            "SensitivePref_Gen": forget.get("SensitivePref_Gen"),
            "paper_Spe_available_without_rerun": False,
            "zero_unlearn_eff_gen_parity": True,
        }, indent=2))


if __name__ == "__main__":
    main()
