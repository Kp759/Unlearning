#!/usr/bin/env python3
"""Run a paper-source ROME/MEMIT edit on the canonical locked MCF forget set.

The adapter deliberately does not use ZeroUnlearn/experiments/evaluate.py because
that entry point owns its own split files and evaluation path.  Instead, this
script consumes the same training_visible_forget.json used by canonical SURE,
reuses the vendored ZeroUnlearn ROME/MEMIT implementations unchanged, and saves
an edited checkpoint for the common semantic-unlearning evaluator.

MCF convention in this repository:
  * original requested_rewrite.target_new = sensitive answer to forget
  * original requested_rewrite.target_true = desired restored answer

Therefore the editing baseline request remaps the prompt to target_true.  Passing
the original target_new to ROME/MEMIT would reinforce the knowledge that SURE is
trying to remove and would not be a fair unlearning comparison.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PAPER_REPO = "XMUDeepLIT/ZeroUnlearn"
PAPER_COMMIT = "deff011c3df367b700b9ad0aa0f5d7aad0cca9b9"
ROME_CONTEXT_DEFAULT = [[5, 10], [10, 10]]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_rr(record: Mapping[str, Any]) -> Dict[str, Any]:
    rr = record["requested_rewrite"]
    if isinstance(rr, list):
        if len(rr) != 1:
            raise ValueError(
                f"case_id={record.get('case_id')} has {len(rr)} requested_rewrite entries; expected one"
            )
        rr = rr[0]
    if not isinstance(rr, dict):
        raise TypeError(f"Unsupported requested_rewrite type: {type(rr)}")
    return copy.deepcopy(rr)


def build_restore_true_requests(records: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    for record in records:
        rr = normalize_rr(record)
        sensitive = copy.deepcopy(rr.get("target_new"))
        restored = copy.deepcopy(rr.get("target_true"))
        if not isinstance(sensitive, dict) or not sensitive.get("str"):
            raise ValueError(f"case_id={record.get('case_id')} lacks target_new.str")
        if not isinstance(restored, dict) or not restored.get("str"):
            raise ValueError(f"case_id={record.get('case_id')} lacks target_true.str")

        # ROME/MEMIT write the value in target_new.  For canonical MCF
        # unlearning, overwrite the sensitive target_new association with the
        # benchmark's target_true answer.
        edit_rr = copy.deepcopy(rr)
        edit_rr["target_new"] = restored
        edit_rr["target_true"] = sensitive
        edit_rr["case_id"] = int(record["case_id"])
        requests.append(edit_rr)
    return requests


def load_effective_hparams(path: Path, algorithm: str) -> SimpleNamespace:
    data = json.loads(path.read_text(encoding="utf-8"))
    # The paper-source Llama ROME JSON omits this field although the vendored
    # ROME implementation accesses it.  Use the original ROME default from
    # kmeng01/rome rather than modifying the third-party source.
    if algorithm == "ROME" and "context_template_length_params" not in data:
        data["context_template_length_params"] = ROME_CONTEXT_DEFAULT
    return SimpleNamespace(**data)


def verify_locked_inputs(visible_path: Path, manifest_path: Path, seed: int) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    visible_bytes = visible_path.read_bytes()
    records = json.loads(visible_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if int(manifest.get("seed", -1)) != seed:
        raise ValueError(f"Manifest seed {manifest.get('seed')} != requested seed {seed}")
    expected_hash = manifest.get("training_visible_sha256")
    actual_hash = sha256_bytes(visible_bytes)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError(
            "training_visible_forget.json hash does not match split_manifest.json: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    if not isinstance(records, list) or not records:
        raise ValueError("training-visible forget set must be a non-empty JSON list")

    expected_ids = [int(x) for x in manifest.get("sampling", {}).get("forget_case_ids", [])]
    actual_ids = [int(x["case_id"]) for x in records]
    if expected_ids and expected_ids != actual_ids:
        raise ValueError("Forget case IDs do not match the canonical split manifest")

    for record in records:
        for field in ("paraphrase_prompts", "neighborhood_prompts", "generation_prompts"):
            if record.get(field):
                raise ValueError(f"Held-out field {field} is visible for case_id={record['case_id']}")
    return records, manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm", choices=("ROME", "MEMIT"), required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--hparams-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dtype", default="bf16")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("ROME/MEMIT paper source requires a CUDA GPU")

    script = Path(__file__).resolve()
    semantic_root = script.parents[1]
    repo_root = semantic_root.parent
    zero_root = repo_root / "ZeroUnlearn"
    if not zero_root.is_dir():
        raise SystemExit(f"Missing vendored ZeroUnlearn directory: {zero_root}")

    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    hparams_path = Path(a.hparams_path).resolve()
    output_dir = Path(a.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    records, split_manifest = verify_locked_inputs(visible_path, manifest_path, a.seed)
    requests = build_restore_true_requests(records)
    set_all_seeds(a.seed)

    # util.globals in the paper source opens globals.yml relative to cwd.
    os.chdir(zero_root)
    if str(zero_root) not in sys.path:
        sys.path.insert(0, str(zero_root))

    if a.algorithm == "ROME":
        from rome.rome_main import apply_rome_to_model
    else:
        from memit.memit_main import apply_memit_to_model

    hparams = load_effective_hparams(hparams_path, a.algorithm)
    dtype = dtype_from_name(a.dtype)

    print(f"Loading base model: {a.model_path}")
    tok = AutoTokenizer.from_pretrained(a.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).cuda()
    model.eval()
    model.config.use_cache = False

    print(
        f"Canonical {a.algorithm}: seed={a.seed}, forget_requests={len(requests)}, "
        "target_mapping=original target_new -> original target_true"
    )

    if a.algorithm == "MEMIT":
        model, _ = apply_memit_to_model(
            model,
            tok,
            requests,
            hparams,
            copy=False,
            return_orig_weights=False,
            cache_template=None,
        )
        application_mode = "joint_batch_all_locked_forget_requests"
    else:
        # The vendored ROME function intentionally consumes request[0].  A fair
        # 50-fact comparison therefore applies one rank-one edit per locked
        # forget fact, sequentially, so every canonical forget request is seen.
        for index, request in enumerate(requests, start=1):
            print(f"===== ROME canonical edit {index}/{len(requests)} case_id={request['case_id']} =====")
            model, _ = apply_rome_to_model(
                model,
                tok,
                [request],
                hparams,
                copy=False,
                return_orig_weights=False,
            )
        application_mode = "sequential_rank_one_all_locked_forget_requests"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tok.save_pretrained(checkpoint_dir)

    effective_hparams = dict(vars(hparams))
    (output_dir / "effective_hparams.json").write_text(
        json.dumps(effective_hparams, indent=2) + "\n", encoding="utf-8"
    )
    run_manifest = {
        "schema_version": 1,
        "method": a.algorithm,
        "dataset": "mcf",
        "paper_source": {"repository": PAPER_REPO, "commit": PAPER_COMMIT},
        "seed": a.seed,
        "base_model": str(Path(a.model_path).resolve()),
        "dtype": a.dtype,
        "training_visible_forget": str(visible_path),
        "split_manifest": str(manifest_path),
        "training_visible_sha256": sha256_bytes(visible_path.read_bytes()),
        "source_dataset_sha256": split_manifest.get("source_sha256"),
        "forget_case_ids": [int(x["case_id"]) for x in records],
        "forget_count": len(records),
        "held_out_probes_visible_to_method": False,
        "retain_examples_visible_to_method": False,
        "target_mapping": "original target_new (sensitive) -> original target_true (restored)",
        "application_mode": application_mode,
        "hparams_source": str(hparams_path),
        "checkpoint": str(checkpoint_dir),
    }
    (output_dir / "baseline_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved canonical {a.algorithm} checkpoint: {checkpoint_dir}")


if __name__ == "__main__":
    main()
