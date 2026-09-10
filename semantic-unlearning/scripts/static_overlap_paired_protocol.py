"""Explicit association pairs, simultaneous answer-span GA/GD, unchanged final tests."""
from static_overlap_endpoint_protocol import main as register, load_pilot as load_endpoint
from static_overlap_endpoint_protocol import claim_evaluation as claim_endpoint

METHOD = "static_overlap_untied_paired_ga_gd_v1"
PLAN = {
    "steps": 120, "check_every": 20, "learning_rate": .0001,
    "pair_batch": 8, "background_retain_batch": 4, "language_batch": 2,
    "retain_weight": 1., "kl_weight": 1., "background_weight": .25,
    "step_radius": .05, "backtracks": 8, "max_training_seconds": 1200,
    "max_stalled_steps": 12, "min_gate_nll_gain": .05, "stalled_gates": 2,
    "target_probability": 1e-6, "max_length": 512, "seed": 1,
    "retain_nll_budget": .05, "retain_kl_budget": .01,
    "fitting_nll_margin": .01, "fitting_kl_margin": .002,
    "untie_before_optimization": True,
    "pairing": "same_context_companions_plus_rotating_verified_overlap_controls",
    "objective": "mean_pair(capped_negative_forget_nll + mean_companion(retain_nll + base_kl))",
    "training_preservation_failure": "restore_last_full_training_preservation_pass_and_reset_optimizer",
}


def load_pilot(path):
    return load_endpoint(path, method=METHOD, plan=PLAN)


def claim_evaluation(path, checkpoint):
    return claim_endpoint(path, checkpoint, method=METHOD, protocol_loader=load_pilot)


def main(argv=None):
    return register(argv, method=METHOD, plan=PLAN)


if __name__ == "__main__":
    main()
