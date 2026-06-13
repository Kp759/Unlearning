#!/usr/bin/env python3
"""Evaluate multiple checkpoints with the same official-compatible MCF evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mcf_zero_unlearn_official_eval import (  # noqa: E402
    evaluate_model_dir_official,
    result_to_comparison_row,
    write_official_comparison,
)


def parse_model_dirs(items: List[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--model-dirs entries must be NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        if not name:
            raise ValueError(f"Model name is empty in --model-dirs entry: {item}")
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory for {name!r} does not exist: {model_path}")
        parsed.append((name, model_path))
    return parsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dirs", nargs="+", required=True, help="One or more NAME=PATH checkpoint directories.")
    p.add_argument("--mcf-path", default="data/mcf/multi_counterfact.json")
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--unlearn-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample-mode", choices=["official", "first"], default="official")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--skip-ppl", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for method, model_dir in parse_model_dirs(args.model_dirs):
        print(f"=== Official-compatible MCF eval: {method} -> {model_dir} ===")
        out_path = out_dir / f"{method}_official_eval.json"
        result = evaluate_model_dir_official(
            method=method,
            model_dir=model_dir,
            mcf_path=args.mcf_path,
            wikidata_dir=args.wikidata_dir,
            out_path=out_path,
            unlearn_num=args.unlearn_num,
            retain_num=args.retain_num,
            seed=args.seed,
            sample_mode=args.sample_mode,
            dtype=args.dtype,
            device_map=args.device_map,
            skip_ppl=args.skip_ppl,
        )
        rows.append(result_to_comparison_row(result))
        print(json.dumps(rows[-1], indent=2))

    write_official_comparison(out_dir, rows)
    print(f"Wrote official comparison to {out_dir}")


if __name__ == "__main__":
    main()
