#!/usr/bin/env python3
"""Run v5 full-update protected-nullspace overlap rewiring."""
from run_static_overlap_endpoint_ga import main as run
from static_overlap_core import StaticEditor
from static_overlap_orthogonal_protocol_v5 import METHOD, load_pilot
from static_overlap_orthogonal_rewire_v5 import fit
from static_overlap_untied_endpoints import prepare_base, hash_frozen, verify_frozen


def editor_factory(model, mask):
    channels = {int(layer): values for layer, values in mask["selected_channels"].items()}
    return StaticEditor(model, mask["input_rows"], mask["output_rows"], channels, rank=16)


def main(argv=None):
    return run(argv, protocol_loader=load_pilot, method=METHOD, fit_function=fit,
               editor_factory=editor_factory, prepare_base=prepare_base,
               hash_function=hash_frozen, verify_function=verify_frozen)


if __name__ == "__main__":
    raise SystemExit(main())
