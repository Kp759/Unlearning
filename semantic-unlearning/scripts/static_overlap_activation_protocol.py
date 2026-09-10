"""Register the activation-space relation rewire experiment."""
import argparse
import json

from static_overlap_endpoint_protocol import main as register, load_pilot as load_endpoint
from static_overlap_endpoint_protocol import claim_evaluation as claim_endpoint
from static_overlap_orthogonal_protocol import exclusion_audit


METHOD = "static_overlap_activation_relation_rewire_v6"
PLAN = {
    "steps": 300,
    "check_every": 5,
    "learning_rate": .01,
    "pair_batch": 8,
    "background_retain_batch": 32,
    "language_batch": 4,
    "retain_weight": 1.,
    "kl_weight": 2.,
    "background_weight": .5,
    "step_radius": .1,
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
    "pairing": "slot1_forget_plus_verified_slot2_and_base_distilled_same_subject_locality",
    "objective": "forget_GA_plus_paired_retain_GD_KL_through_fixed_relation_specific_MLP_keys",
    "optimizer": "Adam_on_static_MLP_value_vectors_with_trust_region_backtracking",
    "parameterization_description": (
        "Embedding and LM head are untied before editing and then frozen.  At every overlap-localized "
        "MLP down projection, one fixed contextual key per forgotten association is formed from the "
        "target-relation activation minus its same-subject/different-relation locality activation, "
        "then projected outside the complete fitting-retain activation span.  Only the corresponding "
        "MLP value vectors are optimized and the resulting low-rank matrices merge into a native "
        "static checkpoint with no router or runtime metadata."
    ),
    "training_preservation_failure": "restore_last_safe_value_vectors_and_replay_current_worst_anchors",
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
