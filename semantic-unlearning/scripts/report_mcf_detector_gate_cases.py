#!/usr/bin/env python3
"""Bind a training-only detector gate's record indices to locked MCF case IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--training-visible", required=True)
    parser.add_argument("--out")
    return parser.parse_args(list(argv) if argv is not None else None)


def _records(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        records = value
    elif isinstance(value, Mapping):
        records = None
        for key in ("records", "data", "training_records"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                records = candidate
                break
        if records is None:
            raise RuntimeError("training-visible object has no record list")
    else:
        raise RuntimeError("training-visible artifact must be a list or object")
    if not all(isinstance(record, Mapping) for record in records):
        raise RuntimeError("training-visible record list contains a non-object")
    return records


def detector_gate_case_tsv(
    gate: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> str:
    rows = gate.get("per_record")
    if not isinstance(rows, list):
        raise RuntimeError("detector gate has no per_record list")
    if len(rows) != len(records):
        raise RuntimeError(
            f"gate/record length mismatch: {len(rows)} gate rows, "
            f"{len(records)} training records"
        )
    lines = [
        "record_index\tcase_id\tpositive_min\tnegative_abs_max\t"
        "writer_off_abs_max\tpassed"
    ]
    for index, (row, record) in enumerate(zip(rows, records)):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"detector gate row {index} is not an object")
        if int(row.get("record_index", index)) != index:
            raise RuntimeError(f"detector gate row {index} has a reordered index")
        if record.get("case_id") is None:
            raise RuntimeError(f"training record {index} has no case_id")
        lines.append(
            "\t".join(
                (
                    str(index),
                    str(int(record["case_id"])),
                    f"{float(row['positive_min']):+.8f}",
                    f"{float(row['negative_abs_max']):.8f}",
                    f"{float(row['writer_off_abs_max']):.8f}",
                    str(bool(row["passed"])).lower(),
                )
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    gate_path = Path(args.gate).resolve()
    visible_path = Path(args.training_visible).resolve()
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not isinstance(gate, Mapping):
        raise RuntimeError("detector gate must be a JSON object")
    records = _records(json.loads(visible_path.read_text(encoding="utf-8")))
    output = detector_gate_case_tsv(gate, records)
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
