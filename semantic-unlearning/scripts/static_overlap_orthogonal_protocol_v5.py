"""Registered full-update protected-nullspace overlap rewiring experiment."""
import argparse
import json

from static_overlap_endpoint_protocol import main as register, load_pilot as load_endpoint
from static_overlap_endpoint_protocol import claim_evaluation as claim_endpoint
from static_overlap_orthogonal_protocol import exclusion_audit, locality_prompt_specs


METHOD = "static_overlap_orthogonal_rewire_v5"
PLAN = {
    "steps": 240,
    "check_every": 5,
    "pair_batch": 8,
    "background_retain_batch": 32,
    "language_batch": 4,
    "retain_weight": .1,
    "kl_weight": 1.,
    "background_weight": .25,
    "step_radius": .025,
    "backtracks": 12,
    "max_training_seconds": 3600,
    "max_stalled_steps": 20,
    "min_gate_nll_gain": .01,
    "stalled_gates": 12,
    "max_failed_preservation_gates": 20,
    "target_probability": 1e-6,
    "max_length": 512,
    "seed": 1,
    "retain_nll_budget": .05,
    "retain_kl_budget": .01,
    # Enforce exactly the declared preservation limits.  v4 silently used
    # .04/.008 fitting limits and consequently rejected valid .05/.01 states.
    "fitting_nll_margin": 0.,
    "fitting_kl_margin": 0.,
    "untie_before_optimization": True,
    "synthetic_same_subject_relations_per_fact": 4,
    "protected_basis_rank": 256,
    "protected_basis_relative_tolerance": 1e-5,
    "min_forget_residual_ratio": 1e-4,
    "pairing": "verified_overlap_pairs_plus_base_distilled_same_subject_relation_queries",
    "objective": "complete_GA_GD_update_projected_into_expanded_protected_Jacobian_nullspace",
    "optimizer": "direct_trust_region_step_after_projection_no_Adam",
    "parameterization_description": (
        "Untied embedding and LM head with separate rank-16 sparse row factors, "
        "plus rank-16 updates on the original overlap-localized MLP channels; "
        "all factors merge into a static native checkpoint and receive no runtime routing input."
    ),
    "training_preservation_failure": "restore_last_safe_state_accumulate_worst_anchors_rebuild_expanded_basis",
}


def load_pilot(path):
    return load_endpoint(path, method=METHOD, plan=PLAN)


def claim_evaluation(path, checkpoint):
    return claim_endpoint(path, checkpoint, method=METHOD, protocol_loader=load_pilot)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--development-protocol", required=True)
    parser.add_argument("--overlap-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args, _ = parser.parse_known_args(argv)
    audit = exclusion_audit(args.development_protocol)
    print(json.dumps({"phase": "same_subject_locality_exclusion_audit", **audit}), flush=True)
    return register(argv, method=METHOD, plan=PLAN)


if __name__ == "__main__":
    main()
