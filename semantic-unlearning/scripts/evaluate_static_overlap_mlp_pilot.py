#!/usr/bin/env python3
"""Evaluate a development-qualified MLP once on the unchanged, already-observed tests."""
import argparse
import json
from pathlib import Path

from evaluate_static_overlap_edit import verify_checkpoint, main as official_main
from evaluate_static_overlap_final_retention import score_final
from freeze_static_overlap_development import load_protocol, write_new
from static_overlap_data import load_bundle, text_fingerprints
from static_overlap_mlp_protocol import claim_evaluation, load_pilot
from static_overlap_training import sha256_file


def main(argv=None, *, protocol_loader=load_pilot, evaluation_claim=claim_evaluation):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-protocol", required=True)
    parser.add_argument("--wikidata-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    pilot = protocol_loader(args.pilot_protocol)
    root = Path(args.pilot_protocol).resolve().parent
    checkpoint = root / "checkpoint"
    report = json.loads((root / "training_report.json").read_text())
    if (report.get("native_checkpoint_created") is not True or report.get("last_gate", {}).get("passed") is not True
            or report.get("pilot_protocol_sha256") != sha256_file(args.pilot_protocol)):
        raise ValueError("Development/export gate failed; final evaluation is forbidden")
    export = verify_checkpoint(checkpoint)
    if (export["deployment_dtype"] != "torch.float32" or not export["forgetting_target_met"]
            or export["reloaded"]["protection"]["nominal_budgets_passed"] is not True
            or export["export_retention_policy"]["float32_numeric_slack"] != 0):
        raise ValueError("Native checkpoint did not pass strict deployment gates")
    manifest = json.loads((checkpoint / "training_manifest.json").read_text())
    frozen = load_protocol(pilot["head_protocol_path"])
    groups = {}
    for name, purpose in (("final_retention", "preservation_test"), ("evaluation_bundle", "evaluation")):
        groups[name], _, _ = load_bundle(frozen["files"][name]["path"], purpose)
        if set(manifest["training_text_fingerprints"]) & set(text_fingerprints(groups[name])):
            raise ValueError("Exploratory development data overlap frozen final inputs")
    _, identity = evaluation_claim(args.pilot_protocol, checkpoint)
    retention_path = root / "exploratory_retention_results.json"
    if retention_path.exists():
        retention = json.loads(retention_path.read_text())
        if any(retention.get(k) != v for k, v in identity.items()):
            raise ValueError("Saved exploratory preservation report identity changed")
    else:
        retention = score_final(checkpoint, manifest["model_path"], groups, args.device,
            pilot["plan"]["max_length"], True, lambda s: print(s, flush=True))
        retention.update(identity, exploratory=True, previously_observed_final_tests=True)
        write_new(retention_path, retention)
    official_path = root / "evaluation_probability_v2.json"
    if not official_path.exists():
        contract = frozen["official_evaluation"]
        official_main(["--checkpoint", str(checkpoint), "--base-model", manifest["model_path"],
            "--evaluation-bundle", frozen["files"]["evaluation_bundle"]["path"], "--out", str(official_path),
            "--device", args.device, "--max-new-tokens", "1", "--mcf-path", frozen["files"]["mcf"]["path"],
            "--wikidata-dir", args.wikidata_dir, "--seed", str(contract["seed"]),
            "--unlearn-num", str(contract["unlearn_num"]), "--retain-num", str(contract["retain_num"]),
            "--skip-official-ppl"])
    official = json.loads(official_path.read_text())
    if official.get("exploratory_protocol_sha256") != sha256_file(args.pilot_protocol):
        raise ValueError("Official result does not belong to this exploratory pilot")
    result = {"exploratory": True, "previously_observed_final_tests": True,
        "joint_exploratory_success": retention["retention_passed"] and official["forgetting_check"]["passed"],
        "preservation_passed": retention["retention_passed"], "official_forgetting_check": official["forgetting_check"],
        "official_forget": official["official_mcf"]["forget"],
        "head_experiment_unchanged": True, **identity}
    destination = root / "exploratory_summary.json"
    if not destination.exists():
        write_new(destination, result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["joint_exploratory_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
