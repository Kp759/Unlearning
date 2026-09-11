"""Register final-layer token-protected activation rewiring."""
import argparse
import json

from static_overlap_endpoint_protocol import main as register, load_pilot as load_endpoint
from static_overlap_endpoint_protocol import claim_evaluation as claim_endpoint
from static_overlap_orthogonal_protocol import exclusion_audit


METHOD = "static_overlap_activation_token_protected_v7"
PLAN = {
    "method_name": METHOD,
    "steps": 300,
    "check_every": 5,
    "learning_rate": .01,
    "pair_batch": 8,
    "background_retain_batch": 32,
    "language_batch": 4,
    "retain_weight": 1.,
    "kl_weight": 2.,
    "background_weight": .5,
    "step_radius": .05,
    "backtracks": 12,
    "max_training_seconds": 3600,
    "max_stalled_steps": 20,
    "min_gate_nll_gain": .05,
    "stalled_gates": 12,
    "max_failed_preservation_gates": 20,
    "target_probability": 1e-6,
    "max_length": 512,
    "seed": 1,
    "retain_nll_budget": .05,
    "retain_kl_budget": .01,
    "fitting_nll_margin": 0.,
    "fitting_kl_margin": 0.,
    "untie_before_optimization": True,
    "synthetic_same_subject_relations_per_fact": 4,
    "activation_key_rank": 50,
    "activation_basis_relative_tolerance": 1e-6,
    "minimum_relation_key_residual_ratio": 1e-4,
    "activation_protection_positions_per_example": 8,
    "activation_protection_rank": 4096,
    "edit_last_selected_layer_only": True,
    "require_development_preservation_for_safe_state": True,
    "pairing": "slot1_forget_plus_verified_slot2_and_base_distilled_same_subject_locality",
    "objective": "forget_GA_plus_paired_retain_GD_KL_through_final_layer_relation_keys",
    "optimizer": "Adam_on_final_MLP_value_vectors_with_trust_region_backtracking",
    "parameterization_description": (
        "The untied embedding and LM head remain frozen. The last overlap-localized MLP down "
        "projection receives 50 fixed relation keys constructed outside a balanced span of up to "
        "4096 individual fitting-retain token activations. Only value vectors are optimized. The "
        "low-rank update merges into the native final-layer matrix with no inference router."
    ),
    "training_preservation_failure": (
        "restore_last_state_passing_both_training_and_development_preservation"
    ),
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
