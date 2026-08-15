#!/usr/bin/env python3
"""Delete TOFU model checkpoints while protecting one Full-TOFU model.

Dry-run is the default. A candidate is any Hugging Face model directory under
--outputs-root whose path contains 'tofu' and that contains model weights.
Only the exact --keep-model directory is protected. Pass --delete to remove all
other candidates. Pass --strip-trainer-state to remove trainer_state.pt from
the protected model while keeping its weights/tokenizer/config for inference.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs-root", default="outputs")
    p.add_argument("--keep-model", required=True)
    p.add_argument("--delete", action="store_true")
    p.add_argument("--strip-trainer-state", action="store_true")
    return p.parse_args()


def has_model_weights(path: Path) -> bool:
    patterns = (
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
        "adapter_model.safetensors",
    )
    return any(any(path.glob(pattern)) for pattern in patterns)


def candidate_model_dirs(root: Path) -> List[Path]:
    candidates: Set[Path] = set()
    for config in root.rglob("config.json"):
        parent = config.parent
        if "tofu" not in str(parent).lower():
            continue
        if has_model_weights(parent):
            candidates.add(parent.resolve())
    return sorted(candidates, key=str)


def size_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except FileNotFoundError:
            pass
    return total


def human_size(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


def main() -> None:
    args = parse_args()
    root = Path(args.outputs_root).resolve()
    keep = Path(args.keep_model).resolve()

    if not root.is_dir():
        raise FileNotFoundError(root)
    if not keep.is_dir():
        raise FileNotFoundError(f"protected model does not exist: {keep}")
    if not (keep / "config.json").is_file() or not has_model_weights(keep):
        raise RuntimeError(
            f"protected path is not a complete Hugging Face model directory: {keep}"
        )

    candidates = candidate_model_dirs(root)
    if keep not in candidates:
        candidates.append(keep)
        candidates.sort(key=str)

    delete_candidates = [path for path in candidates if path != keep]
    reclaim = 0

    print(f"outputs root: {root}")
    print(f"PROTECTED:    {keep} ({human_size(size_bytes(keep))})")
    print()
    print("TOFU model directories eligible for deletion:")
    if not delete_candidates:
        print("  (none)")
    for path in delete_candidates:
        size = size_bytes(path)
        reclaim += size
        print(f"  {human_size(size):>12s}  {path}")
    print()
    print(f"estimated reclaimable model-checkpoint space: {human_size(reclaim)}")

    if not args.delete:
        print("DRY RUN ONLY. Re-run with --delete after reviewing this list.")
        return

    for path in delete_candidates:
        # Re-check the protected path before every destructive operation.
        if path.resolve() == keep:
            raise RuntimeError("internal safety error: attempted to delete protected model")
        print(f"DELETE {path}")
        shutil.rmtree(path)

    if args.strip_trainer_state:
        trainer_state = keep / "trainer_state.pt"
        if trainer_state.exists():
            size = trainer_state.stat().st_size
            trainer_state.unlink()
            print(f"REMOVED protected-model trainer_state.pt ({human_size(size)})")
        else:
            print("protected model has no trainer_state.pt")

    if not keep.is_dir() or not (keep / "config.json").is_file() or not has_model_weights(keep):
        raise RuntimeError("protected Full-TOFU model failed post-cleanup integrity check")

    print(f"PROTECTED MODEL STILL PRESENT: {keep}")


if __name__ == "__main__":
    main()
