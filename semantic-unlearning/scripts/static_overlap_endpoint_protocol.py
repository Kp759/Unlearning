"""Register one fixed-mask, tied-endpoint follow-up using existing development data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from freeze_static_overlap_development import write_new
from static_overlap_mlp_protocol import load_pilot as load_mlp
from static_overlap_training import sha256_file

METHOD = "static_overlap_tied_endpoint_ga_v1"
PLAN = {
    "steps": 80, "check_every": 20, "learning_rate": .003,
    "forget_batch": 16, "retain_batch": 8, "step_radius": .5,
    "backtracks": 4, "projection_iterations": 100, "max_training_seconds": 1200,
    "max_stalled_steps": 12, "min_gate_nll_gain": .05, "stalled_gates": 2,
    "target_probability": 1e-6, "max_length": 512, "seed": 1,
    "retain_nll_budget": .05, "retain_kl_budget": .01,
    "fitting_nll_margin": .01, "fitting_kl_margin": .002,
}


def load_pilot(path, *, method=METHOD, plan=None):
    plan = PLAN if plan is None else plan
    path = Path(path).resolve()
    p = json.loads(path.read_text())
    if p.get("method") != method or p.get("exploratory") is not True or p.get("plan") != plan:
        raise ValueError("Unexpected tied-endpoint experiment contract")
    if sha256_file(p["development_protocol_path"]) != p["development_protocol_sha256"]:
        raise ValueError("Existing development protocol changed")
    old = load_mlp(p["development_protocol_path"])
    for key in ("base_model_path", "data", "source_bundle", "head_protocol_path", "head_protocol_sha256"):
        if p[key] != old[key]:
            raise ValueError(f"Existing development/final contract changed: {key}")
    if sha256_file(p["overlap_manifest"]["path"]) != p["overlap_manifest"]["sha256"]:
        raise ValueError("Original overlap mask manifest changed")
    registration = json.loads((path.parent / "registered.json").read_text())
    if registration != {"pilot_protocol_path": str(path), "pilot_protocol_sha256": sha256_file(path)}:
        raise ValueError("Endpoint registration differs")
    return p


def claim_evaluation(protocol_path, checkpoint, *, method=METHOD, protocol_loader=load_pilot):
    p = protocol_loader(protocol_path)
    manifest = json.loads((Path(checkpoint) / "training_manifest.json").read_text())
    if (manifest.get("method") != method or
            manifest.get("exploratory_protocol_sha256") != sha256_file(protocol_path)):
        raise ValueError("Checkpoint belongs to another exploratory experiment")
    identity = {"pilot_protocol_sha256": sha256_file(protocol_path),
                "checkpoint_export_sha256": sha256_file(Path(checkpoint) / "static_edit_export.json")}
    marker = Path(protocol_path).parent / "exploratory_evaluation_started.json"
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("Evaluation already bound to another checkpoint")
    else:
        write_new(marker, identity)
    return p, identity


def main(argv=None, *, method=METHOD, plan=None):
    plan = PLAN if plan is None else plan
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-protocol", required=True)
    parser.add_argument("--overlap-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    old = load_mlp(args.development_protocol)
    mask_path = Path(args.overlap_manifest).resolve()
    mask = json.loads(mask_path.read_text())
    if (mask.get("architecture") != "static_overlap_constrained_embedding_mlp_head_v1"
            or mask.get("shared_endpoints") is not True
            or Path(mask["model_path"]).resolve() != Path(old["base_model_path"]).resolve()
            or not mask.get("input_rows") or not mask.get("output_rows")):
        raise ValueError("Need the original tied-base static overlap run's manifest.json")
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    p = {k: old[k] for k in ("base_model_path", "data", "source_bundle", "head_protocol_path", "head_protocol_sha256")}
    p.update(method=method, exploratory=True, plan=plan,
        development_protocol_path=str(Path(args.development_protocol).resolve()),
        development_protocol_sha256=sha256_file(args.development_protocol),
        overlap_manifest={"path": str(mask_path), "sha256": sha256_file(mask_path)},
        declared_utc=datetime.now(timezone.utc).isoformat(),
        disclosure="Additional exploratory ablation informed by head and MLP failures; development reused, original final sets already observed; no new confirmatory claim.",
        selection_rule="First scheduled checkpoint passing all training/development forgetting and preservation gates; no final scores in selection.",
        parameterization=plan.get("parameterization_description", (
            "Separate dense input/output deltas within their respective original masks; head cloned before fitting; transformer frozen."
            if plan.get("untie_before_optimization") else
            "Direct dense delta within the original union of endpoint rows; original tying; all transformer weights frozen.")))
    path = out / "pilot_protocol.json"
    write_new(path, p)
    write_new(out / "registered.json", {"pilot_protocol_path": str(path), "pilot_protocol_sha256": sha256_file(path)})
    print(json.dumps({"phase": "endpoint_experiment_registered", "protocol": str(path),
        "editable_rows": len(set(mask["input_rows"]) | set(mask["output_rows"])),
        "plan": plan, "new_final_set_created": False}), flush=True)


if __name__ == "__main__":
    main()
