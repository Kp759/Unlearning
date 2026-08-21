#!/usr/bin/env python3
"""Diagnostic-only held-out 1K Wiki utility evaluation for frozen RWKU v3.2.

This script deliberately does NOT create or accept a checkpoint. It reruns the
frozen deterministic v3.2 experiment, relaxes the representation-norm pre-gate
in memory only so that the already-selected behavior-safe candidate reaches the
held-out utility evaluator, records the true representation norm and exact
full-vocabulary KL on the disjoint held-out gate, then aborts immediately.

The held-out result must not be used for further checkpoint selection or
hyperparameter tuning.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import rwku_sure_hidden_direction_v32_kl_w1k as v32
import rwku_sure_repr_rescue_w1k as v2
import sure_canonical_core as core


class HeldoutDiagnosticComplete(RuntimeError):
    pass


def main() -> None:
    original_load = v32.load_configuration
    original_kl = v2.exact_full_vocab_utility_kl
    result_path_holder = {"path": None}

    def diagnostic_load(path):
        cfg = original_load(path)
        cfg = copy.deepcopy(cfg)
        # Diagnostic only: bypass the <=1% pre-gate so the already-selected
        # behavior-safe v3.2 candidate reaches the held-out utility evaluator.
        # The real acceptance threshold is still recorded as 0.01 below.
        cfg["hidden_direction"]["max_relative_frobenius_delta"] = 999.0
        return cfg

    def diagnostic_kl(*args, **kwargs):
        result = original_kl(*args, **kwargs)
        handles = kwargs.get("handles")
        originals = kwargs.get("original_adapter_weights")
        scale = kwargs.get("scale")
        if handles is None or originals is None:
            raise RuntimeError("diagnostic evaluator could not recover adapter handles/original weights")
        rd = v2.representation_delta_report(handles, originals)
        payload = {
            "schema_version": "rwku_v32_heldout_utility_diagnostic_v1",
            "configuration_id": v32.EXPERIMENT_ID,
            "diagnostic_only": True,
            "heldout_utility_opened": True,
            "heldout_result_must_not_be_used_for_selection_or_tuning": True,
            "actual_acceptance_max_relative_frobenius_delta": 0.01,
            "candidate_scale": float(scale),
            "true_representation_delta": rd,
            "heldout_exact_full_vocab_kl": result,
            "official_rwku_records_accessed": False,
        }
        out_path = Path(result_path_holder["path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        core.write_json(out_path, payload)
        print("\nRWKU v3.2 HELD-OUT UTILITY DIAGNOSTIC")
        print(f"candidate scale: {float(scale):.6f}")
        print(f"true relative Frobenius: {float(rd['relative_frobenius']):.6f}")
        print(
            "held-out KL mean/p95/max: {:.6f} / {:.6f} / {:.6f}".format(
                float(result["utility_kl_mean"]),
                float(result["utility_kl_p95"]),
                float(result["utility_kl_max"]),
            )
        )
        print(f"diagnostic report: {out_path}")
        print("No checkpoint was accepted or frozen. Held-out utility is now opened and must not be used for tuning.")
        raise HeldoutDiagnosticComplete("held-out utility diagnostic complete")

    if "--diagnostic-output" not in sys.argv:
        raise SystemExit("Usage: rwku_v32_heldout_utility_diagnostic.py <normal v3.2 args> --diagnostic-output PATH")
    idx = sys.argv.index("--diagnostic-output")
    try:
        result_path_holder["path"] = sys.argv[idx + 1]
    except IndexError as exc:
        raise SystemExit("--diagnostic-output requires a path") from exc
    del sys.argv[idx:idx + 2]

    v32.load_configuration = diagnostic_load
    v2.exact_full_vocab_utility_kl = diagnostic_kl
    try:
        v32.main()
    except HeldoutDiagnosticComplete:
        return
    finally:
        v32.load_configuration = original_load
        v2.exact_full_vocab_utility_kl = original_kl

    raise RuntimeError("no behavior-safe candidate reached the held-out utility diagnostic")


if __name__ == "__main__":
    main()
