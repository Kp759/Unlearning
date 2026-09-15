#!/usr/bin/env python3
"""Official-compatible MCF evaluation for Router V2 confirmatory seeds 2--10.

The split seed is read from the run manifest. The evaluator produces both a
matched frozen-base result and the edited Router V2 result using the same
official split and metric conversion.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from mcf_sampling import sample_official_mcf_records
from mcf_zero_unlearn_official_eval import (
    dtype_from_str,
    evaluate_loaded_model_official,
)
from mcf_zero_unlearn_metric_parity import summarize_probability_metrics
from static_overlap_fact_association_embeddings import (
    METHOD,
    load_artifact_into_model,
)
from static_overlap_fact_association_v2_gate import (
    load_relation_prototype_artifact,
)


def _probability_metrics(result):
    legacy = {
        split: deepcopy(result[split])
        for split in ("forget", "retain")
    }
    for split in ("forget", "retain"):
        result[split] = summarize_probability_metrics(
            result[split],
            result[f"{split}_raw"],
        )
    result["legacy_counterfact"] = legacy
    result["metric_version"] = "zerounlearn_answer_probability_v2"
    forget = result["forget"]
    result["static_branch_display_zero_check"] = {
        "definition": (
            "Eff < 0.005% and Gen < 0.005%, with "
            "ReleasedAccuracy_Eff == 0 and ReleasedAccuracy_Gen == 0; "
            "display-zero only, not exact-zero probability"
        ),
        "eff_threshold_percent": 0.005,
        "gen_threshold_percent": 0.005,
        "passed": bool(
            0.0 <= float(forget["Eff"]) < 0.005
            and 0.0 <= float(forget["Gen"]) < 0.005
            and float(forget["ReleasedAccuracy_Eff"]) == 0.0
            and float(forget["ReleasedAccuracy_Gen"]) == 0.0
        ),
    }
    return result


def _compact(result):
    return {
        "forget_Eff": result["forget"]["Eff"],
        "forget_Gen": result["forget"]["Gen"],
        "forget_Spe": result["forget"]["Spe"],
        "forget_ReleasedAccuracy_Eff": result["forget"]["ReleasedAccuracy_Eff"],
        "forget_ReleasedAccuracy_Gen": result["forget"]["ReleasedAccuracy_Gen"],
        "retain_Eff": result["retain"]["Eff"],
        "retain_Gen": result["retain"]["Gen"],
        "retain_Spe": result["retain"]["Spe"],
        "PPL": result.get("forget_PPL"),
        "legacy_PPL": result.get("legacy_forget_PPL"),
        "display_zero_check": result["static_branch_display_zero_check"]["passed"],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--mcf-path", required=True)
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--skip-ppl", action="store_true")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    artifact_path = run_dir / "fact_association_embeddings.pt"
    manifest_path = run_dir / "association_manifest.json"
    if not artifact_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Confirmatory MCF run is missing artifact/manifest")

    manifest = json.loads(manifest_path.read_text())
    sampling = manifest.get("sampling", {})
    seed = int(sampling.get("seed", manifest.get("confirmatory_seed", -1)))
    if not (2 <= seed <= 10):
        raise ValueError(f"Confirmatory evaluator requires manifest seed 2--10, got {seed}")
    if int(sampling.get("forget_num", -1)) != 50:
        raise ValueError("Confirmatory evaluator requires forget_num=50")
    if not bool(manifest.get("hyperparameters_frozen_from_seed1", False)):
        raise ValueError("Run does not declare seed-1-frozen hyperparameters")

    mcf_path = Path(args.mcf_path).resolve()
    records = json.loads(mcf_path.read_text())
    expected_forget, _ = sample_official_mcf_records(
        records,
        forget_num=50,
        retain_num=0,
        seed=seed,
        strict=True,
    )
    expected_case_ids = [int(row["case_id"]) for row in expected_forget]
    if [int(x) for x in manifest["forget_case_ids"]] != expected_case_ids:
        raise RuntimeError("Run manifest forget IDs do not match official sampling seed")

    model_path = Path(manifest["model_path"]).resolve()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = dtype_from_str(args.dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    base_model.requires_grad_(False)

    base_result = evaluate_loaded_model_official(
        method="frozen_base",
        model=base_model,
        tok=tok,
        model_dir=model_path,
        mcf_path=mcf_path,
        wikidata_dir=args.wikidata_dir,
        out_path=None,
        unlearn_num=50,
        retain_num=1000,
        seed=seed,
        sample_mode="official",
        skip_ppl=args.skip_ppl,
    )
    base_result = _probability_metrics(base_result)
    base_result["experiment_role"] = "confirmatory"
    base_result["seed"] = seed
    base_out = run_dir / "official_mcf_base_eval.json"
    if base_out.exists():
        raise FileExistsError(f"Refusing to overwrite {base_out}")
    base_out.write_text(json.dumps(base_result, indent=2, allow_nan=False) + "\n")

    artifact = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=False,
    )
    architecture = str(artifact.get("architecture", ""))
    if architecture == "relation_prototype_fact_association_bank_v2":
        model, bank = load_relation_prototype_artifact(base_model, artifact)
    else:
        model, bank = load_artifact_into_model(base_model, artifact)
    model.eval()

    evaluation_method = artifact.get("method", METHOD)
    result = evaluate_loaded_model_official(
        method=evaluation_method,
        model=model,
        tok=tok,
        model_dir=run_dir,
        mcf_path=mcf_path,
        wikidata_dir=args.wikidata_dir,
        out_path=None,
        unlearn_num=50,
        retain_num=1000,
        seed=seed,
        sample_mode="official",
        skip_ppl=args.skip_ppl,
    )
    result = _probability_metrics(result)
    result["experiment_role"] = "confirmatory"
    result["development_seed"] = 1
    result["seed"] = seed
    result["hyperparameters_frozen_from_seed1"] = True
    result["fact_association_embedding_bank"] = {
        "artifact": str(artifact_path),
        "base_model": str(model_path),
        "layer": int(artifact["layer"]),
        "facts": len(artifact["facts"]),
        "routing_policy": artifact.get(
            "routing_policy",
            "subject_candidate_plus_relation_prototype_confirmation",
        ),
        "subject_scan_scope": "prompt_prefix_only",
        "teacher_forced_suffix_can_affect_routing": False,
        "runtime_counters": bank.counters(),
        "evaluation_group_labels_used_by_gate": False,
        "fact_id_injection_used": False,
        "tokenizer_extended": False,
        "base_weights_edited": False,
    }

    out = run_dir / "official_mcf_eval.json"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print(json.dumps({
        "seed": seed,
        "base": _compact(base_result),
        "router_v2": _compact(result),
        "base_out": str(base_out),
        "edited_out": str(out),
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
