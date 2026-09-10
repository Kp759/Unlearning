#!/usr/bin/env python3
"""Run static relation-key activation rewiring."""
from run_static_overlap_endpoint_ga import main as run
from static_overlap_activation_protocol import METHOD, PLAN, load_pilot
from static_overlap_activation_rewire import ActivationRewireEditor, fit
from static_overlap_untied_endpoints import prepare_base, hash_frozen, verify_frozen


def editor_factory(model, mask):
    layers = sorted(int(layer) for layer in mask["selected_channels"])
    return ActivationRewireEditor(model, layers, PLAN["activation_key_rank"])


def main(argv=None):
    return run(argv, protocol_loader=load_pilot, method=METHOD, fit_function=fit,
               editor_factory=editor_factory, prepare_base=prepare_base,
               hash_function=hash_frozen, verify_function=verify_frozen)


if __name__ == "__main__":
    raise SystemExit(main())
