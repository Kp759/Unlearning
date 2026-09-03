#!/usr/bin/env python3
"""Freeze V6 only after both consumed development seeds pass exactly."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL = "mcf_normalization_preserving_sidecar_v6_post_development_freeze_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_development_result(value: Mapping[str, Any], *, seed: int) -> None:
    checks = value.get("behavioral_checks", {})
    preservation = value.get("exact_preservation", {}).get("checks", {})
    if (
        value.get("passed") is not True
        or int(value.get("seed", -1)) != int(seed)
        or value.get("evaluation_status")
        != "consumed_development_not_blind_not_official"
        or value.get("blind_or_official_claim_permitted") is not False
        or not all(bool(item) for item in checks.values())
        or preservation.get("forget_neighborhood_raw_exact") is not True
        or preservation.get("retain_raw_exact") is not True
        or preservation.get("ppl_exact") is not True
        or value.get("integrity", {}).get("passed") is not True
    ):
        raise RuntimeError(f"consumed V6 seed {seed} did not pass the locked gates")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed1-result", required=True)
    parser.add_argument("--seed2-result", required=True)
    parser.add_argument("--development-registry", required=True)
    parser.add_argument("--frame-lexicon", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = {
        "seed1_result": Path(args.seed1_result).resolve(),
        "seed2_result": Path(args.seed2_result).resolve(),
        "development_registry": Path(args.development_registry).resolve(),
        "frame_lexicon": Path(args.frame_lexicon).resolve(),
    }
    output = Path(args.output_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    seed1 = load(paths["seed1_result"])
    seed2 = load(paths["seed2_result"])
    registry = load(paths["development_registry"])
    lexicon = load(paths["frame_lexicon"])
    validate_development_result(seed1, seed=1)
    validate_development_result(seed2, seed=2)
    if (
        registry.get("protocol")
        != "mcf_normalization_preserving_entity_sidecar_v6_0"
        or registry.get("fresh_seed_firewall", {}).get(
            "seed_3_or_later_candidate_or_evaluation_permitted_by_this_registry"
        )
        is not False
        or registry.get("frame_router", {}).get("content_sha256")
        != lexicon.get("lexicon_sha256")
    ):
        raise RuntimeError("V6 freeze inputs differ from the development registry")
    result = {
        "schema_version": 1,
        "kind": "mcf_normalization_preserving_sidecar_v6_post_development_freeze",
        "protocol": PROTOCOL,
        "passed": True,
        "development_evidence": {
            "seeds": [1, 2],
            "status": "consumed_development_not_blind_not_official",
            "both_exact_acceptance_passed": True,
            "seed1_result_sha256": sha256_file(paths["seed1_result"]),
            "seed2_result_sha256": sha256_file(paths["seed2_result"]),
        },
        "frozen_artifacts": {
            "development_registry_sha256": sha256_file(
                paths["development_registry"]
            ),
            "frame_lexicon_file_sha256": sha256_file(paths["frame_lexicon"]),
            "frame_lexicon_content_sha256": str(lexicon["lexicon_sha256"]),
            "architecture": registry["architecture"],
        },
        "fresh_seed_policy": {
            "eligible_seeds": [3, 4, 5, 6, 7, 8, 9, 10],
            "candidate_must_be_built_from_direct_only_view": True,
            "one_shot_evaluation_only": True,
            "retry_or_resume_after_open": False,
            "no_architecture_change_between_seeds": True,
        },
        "claim_scope": "contextual_behavioral_suppression_not_latent_erasure",
        "strong_unlearning_claim_permitted": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
