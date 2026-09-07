#!/usr/bin/env python3
"""Install the verified additive Fix5n-v3 package; never overwrite different files."""
from __future__ import annotations
import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", help="Path to structured_slots_fix5n_v3.zip")
    ap.add_argument("--root", required=True, help="Your semantic-unlearning directory")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory does not exist: {root}")
    planned = []
    with zipfile.ZipFile(Path(args.archive).expanduser()) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or "MANIFEST.json" not in names:
            raise SystemExit("Invalid archive manifest/duplicate paths")
        mf = json.loads(z.read("MANIFEST.json"))
        expected = mf["files_sha256"]
        if set(names) != set(expected) | {"MANIFEST.json"}:
            raise SystemExit("Archive file list differs from manifest")
        for name, digest in expected.items():
            rel = PurePosixPath(name)
            if rel.is_absolute() or ".." in rel.parts or "\\" in name:
                raise SystemExit(f"Unsafe archive path: {name}")
            if not (
                name.startswith("scripts/")
                or name.startswith("tests/")
                or name == "README_controlled_slots_fix5n_v3.md"
            ):
                raise SystemExit(f"Unexpected output path: {name}")
            info = z.getinfo(name)
            if info.file_size > 2_000_000 or stat.S_ISLNK(info.external_attr >> 16):
                raise SystemExit(f"Unsupported archive member: {name}")
            content = z.read(name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise SystemExit(f"Checksum mismatch: {name}")
            target = root.joinpath(*rel.parts)
            if root not in target.resolve().parents:
                raise SystemExit(f"Path escapes root: {target}")
            if target.exists() and (
                not target.is_file() or target.read_bytes() != content
            ):
                raise SystemExit(
                    f"Refusing to overwrite different existing contents: {target}"
                )
            planned.append((target, content))
    count = 0
    for target, content in planned:
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as f:
            f.write(content)
        count += 1
    print(f"Installed {count} new files into {root}; matching existing files preserved.")
    print(
        "No Git commits, checkpoints, previous results, or existing source files were changed."
    )


if __name__ == "__main__":
    main()
