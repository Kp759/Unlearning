#!/usr/bin/env python3
"""Evaluate only a strictly development-qualified paired GA/GD checkpoint."""
from evaluate_static_overlap_mlp_pilot import main as evaluate
from static_overlap_paired_protocol import claim_evaluation, load_pilot


def main(argv=None):
    return evaluate(argv, protocol_loader=load_pilot, evaluation_claim=claim_evaluation)


if __name__ == "__main__":
    raise SystemExit(main())
