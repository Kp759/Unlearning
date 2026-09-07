#!/usr/bin/env python3
"""Build a deterministic checksum-manifested additive Fix5n-v3 ZIP package."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

PACKAGE_FILES = (
    # v3 reuses the leakage-safe route-active-slot parser from v2. Bundle the exact
    # dependency so a verified archive is self-contained relative to the Fix5m baseline.
    "scripts/mcf_output_position_gated_penalty_fix5n_v2_seed1.py",
    "scripts/mcf_structured_two_slot_decoder_fix5n_v3_seed1.py",
    "scripts/run_mcf_structured_two_slot_decoder_fix5n_v3_seed1.sh",
    "tests/test_structured_two_slot_decoder_fix5n_v3.py",
    "README_controlled_slots_fix5n_v3.md",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="semantic-unlearning repository root")
    ap.add_argument(
        "--output",
        default="structured_slots_fix5n_v3.zip",
        help="output ZIP path",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory does not exist: {root}")

    payloads: dict[str, bytes] = {}
    for rel in PACKAGE_FILES:
        path = root / rel
        if not path.is_file():
            raise SystemExit(f"Required package file missing: {path}")
        payloads[rel] = path.read_bytes()

    manifest = {
        "schema_version": 1,
        "package": "structured_slots_fix5n_v3",
        "additive_only": True,
        "files_sha256": {rel: sha256_bytes(payloads[rel]) for rel in PACKAGE_FILES},
    }
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing archive: {output}")

    # Fixed ZIP timestamps make byte-for-byte package construction reproducible.
    fixed_time = (2026, 9, 7, 0, 0, 0)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for rel in sorted(PACKAGE_FILES):
            info = zipfile.ZipInfo(rel, date_time=fixed_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, payloads[rel])
        info = zipfile.ZipInfo("MANIFEST.json", date_time=fixed_time)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        z.writestr(info, manifest_bytes)

    print(output)
    print(f"sha256={sha256_bytes(output.read_bytes())}")
    print("Package contains only additive Fix5n-v3 files plus MANIFEST.json.")


if __name__ == "__main__":
    main()
