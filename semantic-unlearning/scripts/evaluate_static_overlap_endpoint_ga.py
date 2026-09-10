#!/usr/bin/env python3
"""Use the unchanged frozen evaluator after the tied-endpoint development gates."""
from evaluate_static_overlap_mlp_pilot import main as evaluate
from static_overlap_endpoint_protocol import load_pilot, claim_evaluation


def main(argv=None):
    return evaluate(argv, protocol_loader=load_pilot, evaluation_claim=claim_evaluation)


if __name__ == "__main__":
    raise SystemExit(main())
