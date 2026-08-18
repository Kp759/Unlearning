#!/usr/bin/env python3
"""Verify that vendored ROME/MEMIT sources match the pinned ZeroUnlearn paper code.

The expected IDs below are Git blob IDs from XMUDeepLIT/ZeroUnlearn commit
`deff011c3df367b700b9ad0aa0f5d7aad0cca9b9`.  Git blob IDs are computed as
SHA-1("blob <len>\\0" + file_bytes), so this check requires no network access.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

PAPER_REPO = "XMUDeepLIT/ZeroUnlearn"
PAPER_COMMIT = "deff011c3df367b700b9ad0aa0f5d7aad0cca9b9"
ORIGINAL_ROME_COMMIT = "0874014cd9837e4365f3e6f3c71400ef11509e04"
ORIGINAL_MEMIT_COMMIT = "80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b"

EXPECTED = {
    "rome/__init__.py": "da48b0b04baffe1a0fa470c8eaabd45fe09fdbf1",
    "rome/compute_u.py": "0467abf761a2d0cc8cf129c27758327ccc49a495",
    "rome/compute_v.py": "ad893bb1b3d9c691ef44d8aab9a8dc132d75aff5",
    "rome/layer_stats.py": "326ea519353d741b525304b5d4ceec9b4a79fd3d",
    "rome/layer_stats_retain.py": "d1e477c6d6d1c808700b3ca8d753f289f42672ec",
    "rome/repr_tools.py": "fbf40f26b1f450039ba5b48b80021fb41fb26d55",
    "rome/rome_hparams.py": "17553b12f0de0b3ebb15c940a3f1abba51988741",
    "rome/rome_main.py": "356ffc60e18a47afd889ecb4d1777ad22cad770e",
    "rome/tok_dataset.py": "3c8c6a123a6adcc031a5cbf750f78466e9a6e971",
    "memit/__init__.py": "5d547fc24cf3c45db472e95a85f7bfa3a21a4c94",
    "memit/compute_ks.py": "acc35b596b10701ce268f2811e73f3e28fd05def",
    "memit/compute_z.py": "33ac73c2820db457700ce1e8443662a871e7f308",
    "memit/memit_hparams.py": "eb4cf5c4014de49efcddef6e3c2060ca1b890804",
    "memit/memit_main.py": "c401ef483081b5f68e4fec908bb7c9b69ca1fbeb",
    "memit/memit_rect_main.py": "1034c70db3e96f9d0d23cebc5fa2f5af5f434115",
    "memit/memit_seq_main.py": "03f0faf9a9c191716915622f44ac5c6650ad6f94",
    "hparams/ROME/Llama-3.2-3B-Instruct.json": "e4a265dec87d0afdc626c21604deec4c9cc15ade",
    "hparams/MEMIT/Llama-3.2-3B-Instruct.json": "62d44206eaebd294ad06aa2983025b50bf7e11ac",
}


def git_blob_id(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def main() -> None:
    script = Path(__file__).resolve()
    repo_root = script.parents[2]
    zero_unlearn = repo_root / "ZeroUnlearn"

    if not zero_unlearn.is_dir():
        raise SystemExit(f"Missing vendored ZeroUnlearn directory: {zero_unlearn}")

    failures = []
    for rel, expected in EXPECTED.items():
        path = zero_unlearn / rel
        if not path.is_file():
            failures.append((rel, expected, "MISSING"))
            continue
        actual = git_blob_id(path.read_bytes())
        status = "OK" if actual == expected else "MISMATCH"
        print(f"{status:8s} {rel}  {actual}")
        if actual != expected:
            failures.append((rel, expected, actual))

    print()
    print(f"paper source: {PAPER_REPO}@{PAPER_COMMIT}")
    print(f"original ROME reference: kmeng01/rome@{ORIGINAL_ROME_COMMIT}")
    print(f"original MEMIT reference: kmeng01/memit@{ORIGINAL_MEMIT_COMMIT}")

    if failures:
        print("\nBaseline source verification FAILED:")
        for rel, expected, actual in failures:
            print(f"  {rel}: expected {expected}, got {actual}")
        raise SystemExit(1)

    print(f"\nROME/MEMIT PAPER SOURCE VERIFIED ({len(EXPECTED)} files)")


if __name__ == "__main__":
    main()
