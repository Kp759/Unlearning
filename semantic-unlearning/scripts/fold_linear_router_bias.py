#!/usr/bin/env python3
"""Fold the calibrated threshold of an existing linear-router run into its bias.

Runs fitted before `--decision-rule calibrated_bias` store the stage-1 bias b
and a separate calibrated cutoff t (one value, or one per association). This
writes a new run directory whose classifier has b' = b - t and fires at the
standard p >= 0.5 (logit >= 0), with no separate threshold. Nothing is refit
and no model is loaded; the residual rows, facts and subject patterns are
copied unchanged.

    python -u scripts/fold_linear_router_bias.py \
      --run-dir outputs/mcf_linear_2x2_seed1_v24/arms/linear_global \
      --output-dir outputs/mcf_linear_calibrated_bias_seed1

Global cutoff (MCF, MQuAKE): every bias moves by the same amount, so every
route is identical to the source run and its official evals stay valid.
Per-association cutoffs (ZsRE): each head fires on exactly the same prompts,
but the best head is now ranked by its calibrated logit, so re-run the
official evaluator on the new directory.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from linear_router import (
    ARCHITECTURE,
    bias_calibration_record,
    fold_threshold_into_bias,
    routing_policy_name,
)

ARTIFACT_NAME = "fact_association_embeddings.pt"
MANIFEST_NAME = "association_manifest.json"


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.tolist())
    return value


def fold_artifact(artifact):
    """Return (new artifact, summary). Raises on runs that cannot be folded."""
    if str(artifact.get("architecture", "")) != ARCHITECTURE:
        raise ValueError("Not a linear-router artifact")
    if artifact.get("bias_calibration") is not None:
        raise ValueError("This run already uses a calibrated bias")
    if str(artifact.get("gate_mode", "threshold")) != "threshold":
        raise ValueError("The subject gate has no cutoff to fold")
    per_head = artifact.get("per_head_thresholds")
    policy = "per_head" if per_head is not None else "global"
    cutoff = per_head if per_head is not None else float(artifact["threshold"])
    stage1_bias = torch.as_tensor(artifact["router_bias"]).float()
    bias, shift = fold_threshold_into_bias(stage1_bias, cutoff)

    folded = dict(artifact)
    folded["router_bias"] = bias
    folded["threshold"] = 0.0
    folded["per_head_thresholds"] = None
    folded["threshold_policy"] = policy
    folded["bias_calibration"] = bias_calibration_record(policy, stage1_bias, shift)
    folded["decision_rule"] = "calibrated_bias"
    folded["routing_policy"] = routing_policy_name("threshold", "calibrated_bias", policy)
    summary = {
        "policy": policy,
        "global_shift": folded["bias_calibration"]["global_shift"],
        "routes_identical_to_source": policy == "global",
        "action": (
            "none: routes are identical, the source run's official evals apply"
            if policy == "global"
            else "re-run the official evaluator: the best qualifying head is now "
                 "ranked by its calibrated logit"
        ),
    }
    return folded, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="must not exist")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"{output} exists; choose a new --output-dir")
    artifact = torch.load(run_dir / ARTIFACT_NAME, map_location="cpu", weights_only=False)
    manifest_path = run_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}

    folded, summary = fold_artifact(artifact)
    output.mkdir(parents=True)
    torch.save(folded, output / ARTIFACT_NAME)
    new_manifest = dict(manifest)
    new_manifest.update({
        "routing_policy": folded["routing_policy"],
        "decision_rule": "calibrated_bias",
        "bias_calibration": _json_safe(folded["bias_calibration"]),
        "bias_folded_from_run_dir": str(run_dir),
        "runtime_trigger": (
            "complete subject-token eligibility plus learned linear BCE head with a "
            f"calibrated bias ({summary['policy']}; b' = b - t re-fit on held-out "
            "prompts), firing at p >= 0.5"
        ),
    })
    (output / MANIFEST_NAME).write_text(
        json.dumps(_json_safe(new_manifest), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(_json_safe({"status": "folded", "output_dir": str(output), **summary}),
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
