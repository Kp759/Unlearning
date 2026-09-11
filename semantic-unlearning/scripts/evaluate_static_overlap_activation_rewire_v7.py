#!/usr/bin/env python3
"""Evaluate v7 only after its complete development gate passes."""
from evaluate_static_overlap_mlp_pilot import main as evaluate
from static_overlap_activation_protocol_v7 import claim_evaluation, load_pilot


def main(argv=None):
    return evaluate(argv, protocol_loader=load_pilot, evaluation_claim=claim_evaluation)


if __name__ == "__main__":
    raise SystemExit(main())
