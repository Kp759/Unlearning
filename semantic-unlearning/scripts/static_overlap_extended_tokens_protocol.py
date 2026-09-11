"""Register the input-only association-token experiment."""
import argparse
import json

from static_overlap_endpoint_protocol import main as register, load_pilot as load_endpoint
from static_overlap_orthogonal_protocol import exclusion_audit


METHOD = "static_overlap_extended_tokens_v1"
PLAN = {
    "steps": 300,
    "check_every": 10,
    "learning_rate": .05,
    "forget_batch": 8,
    "step_radius": 1.,
    "backtracks": 8,
    "max_training_seconds": 1800,
    "max_stalled_steps": 20,
    "target_probability": 1e-6,
    "max_length": 512,
    "seed": 1,
    "association_tokens": 50,
    "unknown_completion": " I don't know.",
    "unknown_weight": 1.,
    "input_vocabulary_extension_only": True,
    "output_vocabulary_unchanged": True,
    "requires_association_token_injection": True,
    "official_evaluation_eligible": False,
    "parameterization_description": (
        "Add one private input embedding row per forgotten association while retaining the exact "
        "original LM head and output vocabulary. Freeze every base parameter. Optimize only the "
        "50 new input rows so oracle-prefixed prompts suppress their original answers and produce "
        "the existing-token completion 'I don't know.'. Natural prompts and all retain inputs "
        "contain no private token and therefore use the exact base computation."
    ),
}


def load_pilot(path):
    return load_endpoint(path, method=METHOD, plan=PLAN)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--development-protocol", required=True)
    parser.add_argument("--overlap-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args, _ = parser.parse_known_args(argv)
    audit = exclusion_audit(args.development_protocol)
    print(json.dumps({"phase": "extended_token_exclusion_audit", **audit}), flush=True)
    return register(argv, method=METHOD, plan=PLAN)


if __name__ == "__main__":
    main()
