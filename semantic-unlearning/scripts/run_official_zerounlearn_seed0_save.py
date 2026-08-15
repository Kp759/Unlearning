#!/usr/bin/env python3
"""Run the authors' pinned ZeroUnlearn algorithm on the locked MQuAKE seed-0 split.

Why this wrapper exists
-----------------------
The public ZeroUnlearn MQuAKE orchestration at the pinned commit is not directly
reproducible from raw MQuAKE-CF-3k-v2 alone:
  1. experiments/evaluate.py exposes alg_name=ZeroUnlearn but its special
     retain_requests/unlearn_requests call branch is keyed to old "UnL" names;
  2. dsets/mquake.py preserves raw requested_rewrite as a list, while the public
     MQuAKE evaluator indexes requested_rewrite as a dict; the loader also
     references an optional data/mquake_data_saved_split.json that is not in the
     public repository.

This wrapper therefore changes only orchestration, not the published method:
  * pin XMUDeepLIT/ZeroUnlearn to the requested commit;
  * read the exact 50 forget + 1000 retain instance IDs from SURE's locked raw-v2
    manifest;
  * flatten each sampled instance's requested_rewrite list exactly as the
    authors' evaluate.py does before calling the algorithm;
  * load the authors' published Llama-3.2-3B-Instruct ZeroUnlearn hparams;
  * call their apply_unl_to_model directly with retain_requests and
    unlearn_requests;
  * save the returned edited model/tokenizer for a common downstream evaluator.

No AtomicGen, multihop, or PPL data are used by this wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

PINNED_ZERO_UNLEARN_COMMIT = "deff011c3df367b700b9ad0aa0f5d7aad0cca9b9"
MODEL_NAME = "Llama-3.2-3B-Instruct"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zero-unlearn-dir", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--mquake-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--save-checkpoint", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--unlearn-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--edit-layer-nums", type=int, default=3)
    return p.parse_args()


def git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def flatten_requests(raw: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    for source_index in indices:
        record = raw[int(source_index)]
        rewrites = record.get("requested_rewrite")
        if isinstance(rewrites, Mapping):
            rewrites = [rewrites]
        if not isinstance(rewrites, list) or not rewrites:
            raise ValueError(f"MQuAKE instance {source_index} has no requested_rewrite list")
        for rewrite in rewrites:
            if not isinstance(rewrite, Mapping):
                raise TypeError(f"non-dict rewrite in instance {source_index}")
            # This matches the authors' evaluate.py flattening: every atomic
            # rewrite from one sampled MQuAKE instance keeps that instance ID.
            requests.append({"case_id": int(source_index), **dict(rewrite)})
    return requests


def main() -> None:
    a = parse_args()
    if (a.seed, a.unlearn_num, a.retain_num) != (0, 50, 1000):
        raise ValueError("seed-0 baseline is locked to seed=0, unlearn_num=50, retain_num=1000")

    zu = Path(a.zero_unlearn_dir).resolve()
    model_path = Path(a.model_path).resolve()
    mquake_path = Path(a.mquake_path).resolve()
    split_path = Path(a.split_manifest).resolve()
    save_checkpoint = Path(a.save_checkpoint).resolve()
    for path in (zu / ".git", model_path, mquake_path, split_path):
        if not path.exists():
            raise FileNotFoundError(path)
    head = git_head(zu)
    if head != PINNED_ZERO_UNLEARN_COMMIT:
        raise RuntimeError(
            f"ZeroUnlearn checkout must be pinned to {PINNED_ZERO_UNLEARN_COMMIT}, got {head}"
        )

    raw = json.loads(mquake_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 3000:
        raise ValueError(f"expected raw MQuAKE-CF-3k-v2 with 3000 instances, got {len(raw) if isinstance(raw,list) else type(raw)}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    sampling = split.get("sampling", {})
    if int(split.get("seed", -1)) != a.seed:
        raise ValueError("split manifest seed mismatch")
    forget_indices = [int(x) for x in sampling.get("forget_source_indices", [])]
    retain_indices = [int(x) for x in sampling.get("retain_source_indices", [])]
    if len(forget_indices) != a.unlearn_num or len(retain_indices) != a.retain_num:
        raise ValueError(
            f"split manifest counts mismatch: forget={len(forget_indices)}, retain={len(retain_indices)}"
        )
    if set(forget_indices) & set(retain_indices):
        raise ValueError("forget/retain instance overlap")
    if any(x < 1500 for x in forget_indices) or any(x >= 1500 for x in retain_indices):
        raise ValueError("manifest violates first-half retain / second-half forget split")

    # Independent parity audit of the public sampling rule: seed Python RNG,
    # sample forget first, then retain.  This must reproduce the manifest exactly.
    rng = random.Random(a.seed)
    expected_forget = rng.sample(list(range(1500, 3000)), a.unlearn_num)
    expected_retain = rng.sample(list(range(0, 1500)), a.retain_num)
    if forget_indices != expected_forget or retain_indices != expected_retain:
        raise RuntimeError("locked manifest does not match public ZeroUnlearn seed0 sampling order")

    unlearn_requests = flatten_requests(raw, forget_indices)
    retain_requests = flatten_requests(raw, retain_indices)

    # Resolve all relative authors' paths from their pinned checkout.
    os.chdir(zu)
    sys.path.insert(0, str(zu))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ZeroUnlearn import ZeroUnlearnHyperParams, apply_unl_to_model

    hparams_path = zu / "hparams" / "ZeroUnlearn" / f"{MODEL_NAME}.json"
    if not hparams_path.is_file():
        raise FileNotFoundError(hparams_path)
    hparams = ZeroUnlearnHyperParams.from_json(hparams_path)

    # Match the authors' evaluate.py model construction.
    model = AutoModelForCausalLM.from_pretrained(str(model_path)).cuda()
    tok = AutoTokenizer.from_pretrained(str(model_path))
    tok.pad_token = tok.eos_token

    print("===== PUBLIC-CODE ZEROUnlearn DIRECT CALL =====")
    print("commit", PINNED_ZERO_UNLEARN_COMMIT)
    print("seed", a.seed)
    print("forget instances", len(forget_indices), "atomic requests", len(unlearn_requests))
    print("retain instances", len(retain_indices), "atomic requests", len(retain_requests))
    print("hparams", hparams_path)

    edited_model, _ = apply_unl_to_model(
        model=model,
        tok=tok,
        retain_requests=retain_requests,
        unlearn_requests=unlearn_requests,
        hparams=hparams,
        copy=False,
        return_orig_weights=False,
        cache_template=None,
        save_path=None,
        add_retain=False,
        edit_layer_nums=a.edit_layer_nums,
        use_h=False,
    )

    if save_checkpoint.exists():
        import shutil
        shutil.rmtree(save_checkpoint)
    save_checkpoint.mkdir(parents=True, exist_ok=True)
    edited_model.save_pretrained(save_checkpoint)
    tok.save_pretrained(save_checkpoint)

    manifest = {
        "schema_version": 2,
        "baseline": "ZeroUnlearn-public-code",
        "source_repository": "XMUDeepLIT/ZeroUnlearn",
        "source_commit": PINNED_ZERO_UNLEARN_COMMIT,
        "source_algorithm": "ZeroUnlearn.apply_unl_to_model",
        "official_hparams": str(hparams_path),
        "model_path": str(model_path),
        "mquake_path": str(mquake_path),
        "mquake_sha256": sha256(mquake_path),
        "split_manifest": str(split_path),
        "seed": a.seed,
        "forget_instances": len(forget_indices),
        "forget_atomic_requests": len(unlearn_requests),
        "retain_instances": len(retain_indices),
        "retain_atomic_requests": len(retain_requests),
        "forget_source_indices": forget_indices,
        "retain_source_indices": retain_indices,
        "edit_layer_nums": a.edit_layer_nums,
        "checkpoint": str(save_checkpoint),
        "algorithm_code_modified": False,
        "orchestration_note": (
            "Called authors' published apply_unl_to_model directly because the pinned public "
            "MQuAKE evaluate.py routing/raw-list evaluation path is internally inconsistent."
        ),
        "heldout_atomic_questions_used": False,
        "multihop_questions_used": False,
        "PPL_corpus_used": False,
    }
    (save_checkpoint.parent / "official_zerounlearn_seed0_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))

    del edited_model, model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
