#!/usr/bin/env python3
"""Run paired GA/GD after separating the embedding and LM-head weights."""
from run_static_overlap_endpoint_ga import main as run
from static_overlap_paired_ga_gd import fit
from static_overlap_paired_protocol import METHOD, load_pilot
from static_overlap_untied_endpoints import prepare_base, hash_frozen, verify_frozen
from static_overlap_core import StaticEditor


def editor_factory(model, input_rows, output_rows):
    # The layer was selected by the prior sensitivity scan; only its localized
    # 64-channel projection is trainable in this exploratory arm.
    return StaticEditor(model, input_rows, output_rows, {19: list(range(64))}, rank=8)


def main(argv=None):
    return run(argv, protocol_loader=load_pilot, method=METHOD, fit_function=fit,
               editor_factory=editor_factory, prepare_base=prepare_base,
               hash_function=hash_frozen, verify_function=verify_frozen)


if __name__ == "__main__":
    raise SystemExit(main())
