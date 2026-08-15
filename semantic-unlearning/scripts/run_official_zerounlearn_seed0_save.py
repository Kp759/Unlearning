#!/usr/bin/env python3
"""Run the pinned authors' ZeroUnlearn MQuAKE seed-0 baseline and save its model.

The ZeroUnlearn repository's experiments/evaluate.py evaluates an edited model
but does not save it.  This wrapper does not change the algorithm: it replaces
only the ALG_DICT callable with a thin wrapper that calls the authors' original
apply function, saves the returned edited model/tokenizer, and returns the same
result to their evaluator.  That saved checkpoint can then be scored by this
repository's locked MQuAKE evaluator for Eff/AtomicGen/retain/PPL parity.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PINNED_ZERO_UNLEARN_COMMIT = "deff011c3df367b700b9ad0aa0f5d7aad0cca9b9"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zero-unlearn-dir", required=True)
    p.add_argument("--model-parent", required=True)
    p.add_argument("--save-checkpoint", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--unlearn-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--edit-layer-nums", type=int, default=3)
    p.add_argument("--dir-name", default="ZeroUnlearn_SURECompareSeed0")
    return p.parse_args()


def git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    a = parse_args()
    if a.seed != 0 or a.unlearn_num != 50 or a.retain_num != 1000:
        raise ValueError("seed-0 baseline is locked to seed=0, unlearn_num=50, retain_num=1000")

    zu = Path(a.zero_unlearn_dir).resolve()
    if not (zu / ".git").is_dir():
        raise FileNotFoundError(f"ZeroUnlearn checkout missing: {zu}")
    head = git_head(zu)
    if head != PINNED_ZERO_UNLEARN_COMMIT:
        raise RuntimeError(
            f"ZeroUnlearn checkout must be pinned to {PINNED_ZERO_UNLEARN_COMMIT}, got {head}"
        )
    data = zu / "data" / "MQuAKE-CF-3k-v2.json"
    if not data.is_file():
        raise FileNotFoundError(f"MQuAKE-CF-3k-v2.json missing from official DATA_DIR: {data}")
    saved_split = zu / "data" / "mquake_data_saved_split.json"
    if saved_split.exists():
        raise RuntimeError(
            "data/mquake_data_saved_split.json exists; remove it so the official loader adapts the pinned v2 source"
        )

    model_parent = Path(a.model_parent).resolve()
    model_dir = model_parent / "Llama-3.2-3B-Instruct"
    if not model_dir.exists():
        raise FileNotFoundError(
            f"official evaluator expects {model_dir}; create a symlink to the exact base model"
        )

    save_checkpoint = Path(a.save_checkpoint).resolve()
    if save_checkpoint.exists():
        import shutil
        shutil.rmtree(save_checkpoint)
    save_checkpoint.mkdir(parents=True, exist_ok=True)

    # Relative DATA_DIR/HPARAMS_DIR/RESULTS_DIR in the authors' globals.yml are
    # intentionally resolved from the pinned official repository.
    os.chdir(zu)
    sys.path.insert(0, str(zu))
    import experiments.evaluate as ev  # type: ignore

    params_class, original_apply = ev.ALG_DICT["ZeroUnlearn"]
    save_events: list[dict[str, Any]] = []

    def apply_and_save(*args: Any, **kwargs: Any):
        result = original_apply(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) < 1:
            raise RuntimeError("unexpected ZeroUnlearn apply return value")
        edited_model = result[0]
        tok = args[1] if len(args) > 1 else kwargs.get("tok")
        if tok is None:
            raise RuntimeError("could not recover tokenizer from ZeroUnlearn apply call")
        edited_model.save_pretrained(save_checkpoint)
        tok.save_pretrained(save_checkpoint)
        save_events.append(
            {
                "checkpoint": str(save_checkpoint),
                "model_class": type(edited_model).__name__,
                "algorithm": "ZeroUnlearn",
                "source_commit": PINNED_ZERO_UNLEARN_COMMIT,
            }
        )
        return result

    ev.ALG_DICT["ZeroUnlearn"] = (params_class, apply_and_save)

    ev.main(
        "ZeroUnlearn",
        "Llama-3.2-3B-Instruct",
        "Llama-3.2-3B-Instruct.json",
        "mquake",
        None,
        None,
        True,   # MQuAKE rewrite metric does not use AttributeSnippets/TF-IDF; avoid unrelated downloads.
        1,
        False,
        a.dir_name,
        ratio_or_num=True,
        unlearn_num=a.unlearn_num,
        retain_num=a.retain_num,
        model_path_dir=str(model_parent),
        eval_retain=True,
        edit_layer_nums=a.edit_layer_nums,
        seed=a.seed,
    )

    if len(save_events) != 1:
        raise RuntimeError(f"expected one 50-edit ZeroUnlearn apply/save event, got {len(save_events)}")
    if not any(save_checkpoint.iterdir()):
        raise RuntimeError("saved ZeroUnlearn checkpoint is empty")

    manifest = {
        "schema_version": 1,
        "algorithm": "ZeroUnlearn",
        "source_repository": "XMUDeepLIT/ZeroUnlearn",
        "source_commit": PINNED_ZERO_UNLEARN_COMMIT,
        "official_hparams": "hparams/ZeroUnlearn/Llama-3.2-3B-Instruct.json",
        "dataset": str(data),
        "seed": a.seed,
        "unlearn_num": a.unlearn_num,
        "retain_num": a.retain_num,
        "edit_layer_nums": a.edit_layer_nums,
        "saved_checkpoint": str(save_checkpoint),
        "algorithm_code_modified": False,
        "wrapper_effect": "save returned edited model/tokenizer only",
        "skip_generation_tests": True,
        "save_events": save_events,
    }
    (save_checkpoint.parent / "official_zerounlearn_seed0_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
