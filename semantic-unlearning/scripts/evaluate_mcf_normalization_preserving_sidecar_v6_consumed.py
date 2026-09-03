#!/usr/bin/env python3
"""Evaluate MCF V6 on already-consumed seed 1 or 2 as development."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

import evaluate_mcf_exact_subject_target_sidecar_v5_seed1_development as v5_dev
import gagd_compare as gagd
import mcf_normalization_preserving_sidecar_v6_core as core
import mcf_zero_unlearn_official_eval as official
from mcf_sampling import sample_official_mcf_records


PROTOCOL = "mcf_normalization_preserving_sidecar_v6_consumed_development_replay_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--wikidata-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--route-audit-batch-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single",), default="single")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.seed not in (1, 2):
        parser.error("V6 consumed replay is restricted to seeds 1 and 2")
    if args.route_audit_batch_size != 128:
        parser.error("V6 consumed route-audit batch size is frozen at 128")
    return args


def _prompt_groups(record: Mapping[str, Any]) -> Dict[str, list[str]]:
    rewrite = record["requested_rewrite"]
    return {
        "rewrite": [str(rewrite["prompt"]).format(str(rewrite["subject"]))],
        "paraphrase": [str(item) for item in record.get("paraphrase_prompts", [])],
        "neighborhood": [str(item) for item in record.get("neighborhood_prompts", [])],
    }


def _scoring_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    forget_owners: bool,
) -> Dict[str, list[tuple[str, int | None, str]]]:
    rows = {"rewrite": [], "paraphrase": [], "neighborhood": []}
    for record_index, record in enumerate(records):
        rewrite = record["requested_rewrite"]
        suffixes = (
            str(rewrite["target_new"]["str"]),
            str(rewrite["target_true"]["str"]),
        )
        for group, prompts in _prompt_groups(record).items():
            for prompt in prompts:
                for suffix in suffixes:
                    owner = (
                        record_index
                        if forget_owners and group != "neighborhood"
                        else None
                    )
                    rows[group].append((f"{prompt} {suffix}", owner, prompt))
    return rows


def route_audit(
    router: core.TwoSidedEntityRelationRouter,
    tok: Any,
    forget_records: Sequence[Mapping[str, Any]],
    retain_records: Sequence[Mapping[str, Any]],
    *,
    ppl_text: str,
    batch_size: int,
) -> Dict[str, Any]:
    forget_rows = _scoring_rows(forget_records, forget_owners=True)
    retain_rows = _scoring_rows(retain_records, forget_owners=False)
    groups = {
        **{f"forget_{key}": value for key, value in forget_rows.items()},
        **{f"retain_{key}": value for key, value in retain_rows.items()},
    }
    reports: Dict[str, Any] = {}
    overlap_rows: list[dict[str, Any]] = []
    for name, rows in groups.items():
        expected_cells = 0
        observed_cells = 0
        owner_cells = 0
        cross_cells = 0
        missed_rows = 0
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start : start + int(batch_size)]
            encoded = tok(
                [text for text, _owner, _raw in chunk],
                padding=True,
                return_tensors="pt",
            )
            active = router.route(encoded["input_ids"]).active
            observed_cells += int(active.sum())
            for local_index, (_text, owner, raw_prompt) in enumerate(chunk):
                active_indices = active[local_index].nonzero(as_tuple=False).flatten().tolist()
                if owner is None:
                    cross_cells += len(active_indices)
                    for detector_index in active_indices:
                        overlap_rows.append(
                            {
                                "group": name,
                                "prompt_sha256": sha256_text(raw_prompt),
                                "detector_record_index": int(detector_index),
                                "detector_subject": router.subjects[int(detector_index)],
                                "detector_relation_id": router.relation_ids[
                                    int(detector_index)
                                ],
                            }
                        )
                    continue
                expected_cells += 1
                owner_active = bool(active[local_index, owner])
                owner_cells += int(owner_active)
                cross_cells += len(active_indices) - int(owner_active)
                missed_rows += int(not owner_active)
        reports[name] = {
            "rows": len(rows),
            "expected_owner_cells": expected_cells,
            "observed_route_cells": observed_cells,
            "observed_owner_cells": owner_cells,
            "non_owner_or_preservation_route_cells": cross_cells,
            "missed_owner_rows": missed_rows,
        }

    ppl_inputs = tok([ppl_text], return_tensors="pt", max_length=100, truncation=True)
    ppl_routes = int(router.route(ppl_inputs["input_ids"]).active.sum())
    reports["ppl"] = {
        "rows": 1,
        "observed_route_cells": ppl_routes,
        "passed": ppl_routes == 0,
    }
    positive_coverage = all(
        reports[name]["missed_owner_rows"] == 0
        and reports[name]["non_owner_or_preservation_route_cells"] == 0
        for name in ("forget_rewrite", "forget_paraphrase")
    )
    # Forget-neighborhood prompts are the direct nested-entity stress bank for
    # the same 50 records.  Retain can contain verbatim forgotten facts (the
    # seed-2 Perth/Thierry overlap), which is reported but adjudicated by exact
    # output preservation rather than mislabeled as a syntactic route error.
    nested_entity_safety = reports["forget_neighborhood"][
        "observed_route_cells"
    ] == 0
    coverage_gate = bool(positive_coverage and nested_entity_safety and ppl_routes == 0)
    return {
        "groups": reports,
        "positive_owner_coverage_passed": positive_coverage,
        "forget_neighborhood_nested_entity_safety_passed": nested_entity_safety,
        "retain_semantic_overlap_route_cells": sum(
            reports[name]["observed_route_cells"]
            for name in ("retain_rewrite", "retain_paraphrase", "retain_neighborhood")
        ),
        "semantic_overlap_rows": overlap_rows,
        "ppl_route_closed": ppl_routes == 0,
        "route_coverage_gate_passed": coverage_gate,
        "preservation_is_adjudicated_by_exact_raw_output": True,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    candidate_root = Path(args.candidate_run_dir).resolve()
    candidate_path = candidate_root / "v6_normalization_preserving_sidecar.pt"
    completion_path = candidate_root / "completion.json"
    for path in (candidate_path, completion_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    completion = load_json(completion_path)
    if (
        completion.get("passed") is not True
        or int(completion.get("seed", -1)) != int(args.seed)
        or completion.get("candidate_sha256") != sha256_file(candidate_path)
        or completion.get("eligible_for_consumed_development_replay") is not True
        or completion.get("fresh_seed_evaluation_permitted") is not False
        or completion.get("evaluation_prompts_seen") != 0
    ):
        raise RuntimeError("V6 candidate is not eligible for consumed replay")

    mcf_path = Path(args.mcf_path).resolve()
    data = official.load_mcf(mcf_path)
    sampled_forget, sampled_retain = sample_official_mcf_records(
        data, 50, 1000, int(args.seed)
    )
    forget_records = [official.normalize_record(dict(item)) for item in sampled_forget]
    retain_records = [official.normalize_record(dict(item)) for item in sampled_retain]
    state = torch.load(candidate_path, map_location="cpu", weights_only=False)
    core.validate_candidate_state(state)
    if int(state["seed"]) != int(args.seed):
        raise RuntimeError("V6 candidate seed differs from consumed replay")
    state_bindings = list(
        zip(
            state["subjects"],
            state["relation_ids"],
            state["target_new"],
            state["target_true"],
        )
    )
    data_bindings = [
        (
            str(record["requested_rewrite"]["subject"]),
            str(record["requested_rewrite"]["relation_id"]),
            str(record["requested_rewrite"]["target_new"]["str"]),
            str(record["requested_rewrite"]["target_true"]["str"]),
        )
        for record in forget_records
    ]
    if state_bindings != data_bindings:
        raise RuntimeError("V6 candidate bindings differ from consumed split")

    namespace = argparse.Namespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(namespace, for_training=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = gagd.first_device(model)
    llama = official.is_llama_like(model, tok)
    if bool(state["llama_like"]) != bool(llama):
        raise RuntimeError("V6 tokenizer family differs from Base")
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    input_hash_before = core.tensor_sha256(input_layer.weight)
    output_hash_before = core.tensor_sha256(output_layer.weight)
    runtime = core.install_candidate(model, tok, state)
    ppl_text = official.load_official_ppl_text(args.wikidata_dir)
    if not ppl_text:
        raise RuntimeError("official PPL text is unavailable")

    try:
        routing = route_audit(
            runtime.router,
            tok,
            forget_records,
            retain_records,
            ppl_text=ppl_text,
            batch_size=int(args.route_audit_batch_size),
        )
        write_json(output / "route_audit.json", routing)
        if not routing["route_coverage_gate_passed"]:
            raise RuntimeError("V6 consumed route coverage failed before metrics")

        print(f"Stage 1: consumed seed {args.seed} reconstructed Base replay")
        runtime.sidecar.enabled = False
        base_forget, base_forget_raw = official.evaluate_record_split(
            model, tok, forget_records, device, llama, "forget"
        )
        base_retain, base_retain_raw = official.evaluate_record_split(
            model, tok, retain_records, device, llama, "retain"
        )
        base_ppl = official.official_perplexity(model, tok, ppl_text, device)

        print(f"Stage 2: consumed seed {args.seed} V6 replay")
        runtime.sidecar.enabled = True
        candidate_forget, candidate_forget_raw = official.evaluate_record_split(
            model, tok, forget_records, device, llama, "forget"
        )
        candidate_retain, candidate_retain_raw = official.evaluate_record_split(
            model, tok, retain_records, device, llama, "retain"
        )
        candidate_ppl = official.official_perplexity(model, tok, ppl_text, device)
    finally:
        runtime.close()

    preservation = v5_dev.exact_preservation_comparison(
        base_forget_raw,
        candidate_forget_raw,
        base_retain_raw,
        candidate_retain_raw,
        base_ppl=base_ppl,
        candidate_ppl=candidate_ppl,
    )
    behavioral_checks = {
        "forget_eff_zero": float(candidate_forget["Eff"]) == 0.0,
        "forget_gen_zero": float(candidate_forget["Gen"]) == 0.0,
        "minimum_margin_at_least_0_1": float(
            candidate_forget["minimum_rewrite_paraphrase_margin"]
        )
        >= 0.1,
        "forget_specificity_exact_base": bool(
            preservation["checks"]["forget_neighborhood_raw_exact"]
        ),
        "retain_exact_base": bool(preservation["checks"]["retain_raw_exact"]),
        "ppl_exact_base": bool(preservation["checks"]["ppl_exact"]),
    }
    integrity = {
        "embedding_hash_unchanged": input_hash_before
        == core.tensor_sha256(input_layer.weight),
        "lm_head_hash_unchanged": output_hash_before
        == core.tensor_sha256(output_layer.weight),
        "candidate_checkpoint_bit_identical": sha256_file(candidate_path)
        == completion["candidate_sha256"],
        "all_parameters_require_grad_false": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "optimizer_constructed": False,
        "gradient_updates_performed": 0,
    }
    integrity["passed"] = all(
        bool(integrity[key])
        for key in (
            "embedding_hash_unchanged",
            "lm_head_hash_unchanged",
            "candidate_checkpoint_bit_identical",
            "all_parameters_require_grad_false",
        )
    )
    passed = bool(
        routing["route_coverage_gate_passed"]
        and preservation["passed"]
        and all(behavioral_checks.values())
        and integrity["passed"]
    )
    result = {
        "schema_version": 1,
        "kind": "mcf_normalization_preserving_sidecar_v6_consumed_development_replay",
        "protocol": PROTOCOL,
        "passed": passed,
        "evaluation_status": "consumed_development_not_blind_not_official",
        "seed": int(args.seed),
        "candidate_sha256": sha256_file(candidate_path),
        "arms": {
            "base": {"forget": base_forget, "retain": base_retain, "PPL": base_ppl},
            "v6_sidecar": {
                "forget": candidate_forget,
                "retain": candidate_retain,
                "PPL": candidate_ppl,
            },
        },
        "behavioral_checks": behavioral_checks,
        "route_audit": routing,
        "exact_preservation": preservation,
        "integrity": integrity,
        "used_for_architecture_development": True,
        "blind_or_official_claim_permitted": False,
        "fresh_seed_freeze_receipt_required": True,
        "claim_scope": "contextual_behavioral_suppression_not_latent_erasure",
    }
    write_json(output / "arms" / "base_forget_raw.json", base_forget_raw)
    write_json(output / "arms" / "base_retain_raw.json", base_retain_raw)
    write_json(output / "arms" / "v6_forget_raw.json", candidate_forget_raw)
    write_json(output / "arms" / "v6_retain_raw.json", candidate_retain_raw)
    write_json(output / "development_replay.json", result)
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(f"V6 consumed seed {args.seed} development acceptance failed")


if __name__ == "__main__":
    main()
