#!/usr/bin/env python3
"""Official-compatible MCF evaluation for the fact-association embedding bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mcf_zero_unlearn_official_eval import (
    dtype_from_str,
    evaluate_loaded_model_official,
)
from static_overlap_fact_association_embeddings import (
    METHOD,
    load_artifact_into_model,
)
from static_overlap_fact_association_v2_gate import (
    load_relation_prototype_artifact,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    artifact_path = run_dir / "fact_association_embeddings.pt"
    manifest_path = run_dir / "association_manifest.json"
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing association artifact: {artifact_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing association manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    model_path = Path(manifest["model_path"]).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Base model is missing: {model_path}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = dtype_from_str(args.dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    )
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        base_model = base_model.to("cuda")
    else:
        base_model = base_model.to(args.device)
    base_model.eval()
    base_model.requires_grad_(False)

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

    out_path = (
        Path(args.out).resolve()
        if args.out
        else run_dir / "official_mcf_eval.json"
    )
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite official evaluation: {out_path}")

    result = evaluate_loaded_model_official(
        method=METHOD,
        model=model,
        tok=tokenizer,
        model_dir=run_dir,
        mcf_path=args.mcf_path,
        wikidata_dir=args.wikidata_dir,
        out_path=None,
        unlearn_num=50,
        retain_num=1000,
        seed=1,
        sample_mode="official",
        skip_ppl=args.skip_ppl,
    )
    result["fact_association_embedding_bank"] = {
        "artifact": str(artifact_path),
        "base_model": str(model_path),
        "layer": int(artifact["layer"]),
        "facts": len(artifact["facts"]),
        "runtime_trigger": (
            "complete subject-token eligibility; frozen hidden-state relation key "
            "only for subjects with multiple forgotten associations"
        ),
        "routing_policy": artifact.get(
            "routing_policy",
            "hierarchical_subject_then_relation_if_ambiguous",
        ),
        "subject_scan_scope": "prompt_prefix_only",
        "teacher_forced_suffix_can_affect_routing": False,
        "runtime_counters": bank.counters(),
        "evaluation_group_labels_used_by_gate": False,
        "fact_id_injection_used": False,
        "tokenizer_extended": False,
        "base_weights_edited": False,
        "official_paraphrases_first_opened_in_this_process": True,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    compact = {
        "forget_Eff": result["forget"]["Eff"],
        "forget_Gen": result["forget"]["Gen"],
        "forget_Spe": result["forget"]["Spe"],
        "retain_Eff": result["retain"]["Eff"],
        "retain_Gen": result["retain"]["Gen"],
        "retain_Spe": result["retain"]["Spe"],
        "PPL": result.get("forget_PPL"),
        "PPL_metric_version": result.get("PPL_metric_version"),
        "legacy_PPL": result.get("legacy_forget_PPL"),
        "minimum_rewrite_paraphrase_margin": result["forget"].get(
            "minimum_rewrite_paraphrase_margin"
        ),
        "runtime_counters": bank.counters(),
        "out": str(out_path),
    }
    print(json.dumps(compact, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
