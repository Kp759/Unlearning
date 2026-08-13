#!/usr/bin/env python3
"""Fail-closed verifier for the ZeroUnlearn-compatible ZsRE SURE split.

This independently recomputes the official split using
``zsre_zero_unlearn_official_eval.sample_official_zsre_raw_records`` and checks
that the locked SURE manifest and repair-visible artifact contain exactly the
same sampled forget records. Retain records are verified from the manifest but
are never materialized into the training-visible artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import zsre_zero_unlearn_official_eval as zsre


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zsre-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--repair-visible-path", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    source = Path(a.zsre_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    visible_path = Path(a.repair_visible_path).resolve()

    raw = json.loads(source.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    visible = json.loads(visible_path.read_text(encoding="utf-8"))

    if manifest.get("protocol") != "zsre_zerounlearn_forget_only_locked_probes":
        raise RuntimeError("unexpected ZsRE locked-split protocol")
    if int(manifest.get("seed", -1)) != a.seed:
        raise RuntimeError("manifest seed does not match requested seed")
    if manifest.get("source_sha256") != sha256(source):
        raise RuntimeError("ZsRE source hash differs from split manifest")

    sampling: Mapping[str, Any] = manifest.get("sampling", {})
    if int(sampling.get("forget_num", -1)) != a.forget_num:
        raise RuntimeError("manifest forget count mismatch")
    if int(sampling.get("retain_num", -1)) != a.retain_num:
        raise RuntimeError("manifest retain count mismatch")
    if sampling.get("order") != "forget sample first, then retain sample from one seeded RNG":
        raise RuntimeError("manifest sampling order is not the ZeroUnlearn order")

    forget_pairs, retain_pairs = zsre.sample_official_zsre_raw_records(
        raw,
        forget_num=a.forget_num,
        retain_num=a.retain_num,
        seed=a.seed,
        strict=True,
    )
    expected_forget = [int(i) for i, _ in forget_pairs]
    expected_retain = [int(i) for i, _ in retain_pairs]

    manifest_forget = [int(i) for i in sampling.get("forget_case_ids", [])]
    manifest_retain = [int(i) for i in sampling.get("retain_case_ids", [])]
    visible_forget = [int(row["case_id"]) for row in visible]

    if manifest_forget != expected_forget:
        raise RuntimeError("manifest forget IDs differ from official ZeroUnlearn sampling")
    if manifest_retain != expected_retain:
        raise RuntimeError("manifest retain IDs differ from official ZeroUnlearn sampling")
    if visible_forget != expected_forget:
        raise RuntimeError("SURE repair-visible forget IDs differ from ZeroUnlearn forget IDs")

    half = len(raw) // 2
    if any(i < half for i in expected_forget):
        raise RuntimeError("forget record escaped ZeroUnlearn second-half pool")
    if any(i >= half for i in expected_retain):
        raise RuntimeError("retain record escaped ZeroUnlearn first-half pool")
    if set(expected_forget) & set(expected_retain):
        raise RuntimeError("forget and retain samples overlap")

    for row in visible:
        if row.get("paraphrase_prompts"):
            raise RuntimeError("training-visible split leaks ZsRE rephrase prompts")
        if row.get("neighborhood_prompts"):
            raise RuntimeError("training-visible split leaks ZsRE locality prompts")
        rr = row.get("requested_rewrite")
        if not isinstance(rr, dict):
            raise RuntimeError("training-visible record lacks requested_rewrite")

    print("===== ZSRE ZEROUnlearn SPLIT PARITY: PASS =====")
    print("seed:", a.seed)
    print("dataset size:", len(raw), "half:", half)
    print("forget pool: second half; sampled:", len(expected_forget))
    print("retain pool: first half; sampled:", len(expected_retain))
    print("RNG order: forget first, retain second")
    print("SURE Stage1/Stage2 visible forget IDs: EXACT MATCH")
    print("retain/rephrase/locality visible during training: 0")


if __name__ == "__main__":
    main()
