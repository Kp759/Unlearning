#!/usr/bin/env python3
"""Replay consumed seed 1 as development for the frozen V5 sidecar.

Seed 1's official MCF prompts were already opened by the V3.6.2 recovery run.
This script therefore makes no blind or official claim.  It reuses the frozen
identity only as a development diagnostic and requires exact raw equality to
Base for forget-neighborhood, retain, and PPL routes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

import gagd_compare as gagd
import mcf_exact_subject_target_sidecar_v5_core as core
import mcf_zero_unlearn_official_eval as official
from mcf_sampling import sample_official_mcf_records


PROTOCOL = "mcf_exact_subject_target_logit_sidecar_v5_0_seed1_consumed_development_replay"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--wikidata-dir", required=True)
    parser.add_argument("--recovery-protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--route-audit-batch-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single",), default="single")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.route_audit_batch_size <= 0:
        parser.error("route-audit batch size must be positive")
    return args


def normalize_records(records: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    return [official.normalize_record(dict(value)) for value in records]


def _prompt_groups(record: Mapping[str, Any]) -> Dict[str, list[str]]:
    rewrite = record["requested_rewrite"]
    return {
        "rewrite": [str(rewrite["prompt"]).format(str(rewrite["subject"]))],
        "paraphrase": [str(value) for value in record.get("paraphrase_prompts", [])],
        "neighborhood": [str(value) for value in record.get("neighborhood_prompts", [])],
    }


def _full_scoring_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_owner: bool,
) -> Dict[str, list[tuple[str, int | None]]]:
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
                    owner = record_index if expected_owner and group != "neighborhood" else None
                    rows[group].append((f"{prompt} {suffix}", owner))
    return rows


def route_audit(
    router: core.BoundaryAwareSubjectRouter,
    tok: Any,
    forget_records: Sequence[Mapping[str, Any]],
    retain_records: Sequence[Mapping[str, Any]],
    *,
    ppl_text: str,
    batch_size: int,
) -> Dict[str, Any]:
    forget_rows = _full_scoring_rows(forget_records, expected_owner=True)
    retain_rows = _full_scoring_rows(retain_records, expected_owner=False)
    groups: Dict[str, list[tuple[str, int | None]]] = {
        **{f"forget_{key}": value for key, value in forget_rows.items()},
        **{f"retain_{key}": value for key, value in retain_rows.items()},
    }
    reports: Dict[str, Any] = {}
    all_passed = True
    for name, rows in groups.items():
        expected_cells = 0
        observed_cells = 0
        owner_cells = 0
        unexpected_cells = 0
        missed_rows = 0
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            encoded = tok(
                [text for text, _owner in chunk],
                padding=True,
                return_tensors="pt",
            )
            active = router.route(encoded["input_ids"]).active
            observed_cells += int(active.sum())
            for local_index, (_text, owner) in enumerate(chunk):
                if owner is None:
                    unexpected_cells += int(active[local_index].sum())
                    continue
                expected_cells += 1
                owner_active = bool(active[local_index, owner])
                owner_cells += int(owner_active)
                unexpected_cells += int(active[local_index].sum()) - int(owner_active)
                missed_rows += int(not owner_active)
        passed = unexpected_cells == 0 and missed_rows == 0
        reports[name] = {
            "rows": len(rows),
            "expected_owner_cells": expected_cells,
            "observed_route_cells": observed_cells,
            "observed_owner_cells": owner_cells,
            "unexpected_route_cells": unexpected_cells,
            "missed_owner_rows": missed_rows,
            "passed": passed,
        }
        all_passed &= passed

    ppl_inputs = tok(
        [ppl_text], return_tensors="pt", max_length=100, truncation=True
    )
    ppl_routes = int(router.route(ppl_inputs["input_ids"]).active.sum())
    reports["ppl"] = {
        "rows": 1,
        "observed_route_cells": ppl_routes,
        "passed": ppl_routes == 0,
    }
    all_passed &= ppl_routes == 0
    return {
        "groups": reports,
        "all_forget_rewrite_paraphrase_owner_routes_open": bool(
            reports["forget_rewrite"]["passed"]
            and reports["forget_paraphrase"]["passed"]
        ),
        "all_forget_neighborhood_routes_closed": reports["forget_neighborhood"][
            "passed"
        ],
        "all_retain_routes_closed": bool(
            reports["retain_rewrite"]["passed"]
            and reports["retain_paraphrase"]["passed"]
            and reports["retain_neighborhood"]["passed"]
        ),
        "ppl_route_closed": reports["ppl"]["passed"],
        "passed": bool(all_passed),
    }


def exact_preservation_comparison(
    base_forget_raw: Any,
    candidate_forget_raw: Any,
    base_retain_raw: Any,
    candidate_retain_raw: Any,
    *,
    base_ppl: float,
    candidate_ppl: float,
) -> Dict[str, Any]:
    def neighborhoods(rows: Sequence[Mapping[str, Any]]) -> list[Any]:
        return [
            value
            for row in rows
            for value in row["post"].get("neighborhood_prompts_probs", [])
        ]

    base_neighborhood = neighborhoods(base_forget_raw)
    candidate_neighborhood = neighborhoods(candidate_forget_raw)
    checks = {
        "forget_neighborhood_raw_exact": base_neighborhood == candidate_neighborhood,
        "retain_raw_exact": base_retain_raw == candidate_retain_raw,
        "ppl_exact": float(base_ppl) == float(candidate_ppl),
    }
    return {
        "checks": checks,
        "forget_neighborhood_cells": len(base_neighborhood),
        "base_forget_neighborhood_sha256": canonical_sha256(base_neighborhood),
        "candidate_forget_neighborhood_sha256": canonical_sha256(
            candidate_neighborhood
        ),
        "base_retain_raw_sha256": canonical_sha256(base_retain_raw),
        "candidate_retain_raw_sha256": canonical_sha256(candidate_retain_raw),
        "base_ppl": float(base_ppl),
        "candidate_ppl": float(candidate_ppl),
        "passed": all(checks.values()),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"development output already exists: {output}")
    output.mkdir(parents=True)

    candidate_root = Path(args.candidate_run_dir).resolve()
    completion_path = candidate_root / "method" / "completion.json"
    candidate_path = candidate_root / "method" / "v5_exact_subject_target_sidecar.pt"
    registry_path = candidate_root / "protocol" / "experiment_registry.json"
    for path in (completion_path, candidate_path, registry_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    completion = load_json(completion_path)
    registry = load_json(registry_path)
    if (
        completion.get("passed") is not True
        or completion.get("candidate_sha256") != sha256_file(candidate_path)
        or completion.get("official_evaluation_prompts_seen") != 0
        or registry.get("seed_1_use")
        != "development_replay_only_with_disclosure"
    ):
        raise RuntimeError("V5 candidate is not eligible for consumed development replay")

    recovery = load_json(Path(args.recovery_protocol).resolve())
    mcf_path = Path(args.mcf_path).resolve()
    if sha256_file(mcf_path) != recovery.get("official_mcf_sha256"):
        raise RuntimeError("consumed seed-1 MCF hash differs from the recovery identity")
    data = official.load_mcf(mcf_path)
    sampled_forget, sampled_retain = sample_official_mcf_records(data, 50, 1000, 1)
    selected_forget = [data[int(index)] for index in recovery["forget_dataset_indices"]]
    sampled_subjects = [
        value["requested_rewrite"]["subject"] for value in sampled_forget
    ]
    selected_subjects = [
        value["requested_rewrite"]["subject"] for value in selected_forget
    ]
    if sampled_subjects != selected_subjects:
        raise RuntimeError("seed-1 sampling no longer reproduces the frozen index identity")
    forget_records = normalize_records(selected_forget)
    retain_records = normalize_records(sampled_retain)

    state = torch.load(candidate_path, map_location="cpu", weights_only=False)
    core.validate_candidate_state(state)
    if int(state["seed"]) != 1:
        raise RuntimeError("consumed seed-1 replay received a different seed candidate")
    state_bindings = [
        (subject, new, true)
        for subject, new, true in zip(
            state["subjects"], state["target_new"], state["target_true"]
        )
    ]
    data_bindings = [
        (
            record["requested_rewrite"]["subject"],
            record["requested_rewrite"]["target_new"]["str"],
            record["requested_rewrite"]["target_true"]["str"],
        )
        for record in forget_records
    ]
    if state_bindings != data_bindings:
        raise RuntimeError("V5 candidate bindings differ from consumed seed 1")

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
        raise RuntimeError("V5 target tokenization family differs from Base")
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
        if not routing["passed"]:
            raise RuntimeError("V5 consumed-seed route audit failed before metric replay")

        print("Stage 1: consumed seed-1 Base development replay")
        runtime.sidecar.enabled = False
        base_forget, base_forget_raw = official.evaluate_record_split(
            model, tok, forget_records, device, llama, "forget"
        )
        base_retain, base_retain_raw = official.evaluate_record_split(
            model, tok, retain_records, device, llama, "retain"
        )
        base_ppl = official.official_perplexity(model, tok, ppl_text, device)

        print("Stage 2: consumed seed-1 frozen V5 development replay")
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

    preservation = exact_preservation_comparison(
        base_forget_raw,
        candidate_forget_raw,
        base_retain_raw,
        candidate_retain_raw,
        base_ppl=base_ppl,
        candidate_ppl=candidate_ppl,
    )
    minimum_margin = float(candidate_forget["minimum_rewrite_paraphrase_margin"])
    behavioral_checks = {
        "forget_eff_zero": float(candidate_forget["Eff"]) == 0.0,
        "forget_gen_zero": float(candidate_forget["Gen"]) == 0.0,
        "minimum_margin_at_least_0_1": minimum_margin >= 0.1,
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
        "all_parameters_require_grad_false": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "optimizer_constructed": False,
        "gradient_updates_performed": 0,
    }
    integrity["passed"] = bool(
        integrity["embedding_hash_unchanged"]
        and integrity["lm_head_hash_unchanged"]
        and integrity["all_parameters_require_grad_false"]
        and integrity["optimizer_constructed"] is False
        and integrity["gradient_updates_performed"] == 0
    )
    passed = bool(
        routing["passed"]
        and preservation["passed"]
        and all(behavioral_checks.values())
        and integrity["passed"]
    )
    result = {
        "schema_version": 1,
        "kind": "mcf_v5_seed1_consumed_development_replay",
        "protocol": PROTOCOL,
        "passed": passed,
        "evaluation_status": "consumed_development_not_blind_not_official",
        "seed": 1,
        "candidate_sha256": sha256_file(candidate_path),
        "arms": {
            "base": {"forget": base_forget, "retain": base_retain, "PPL": base_ppl},
            "v5_sidecar": {
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
        "blind_evaluation_claim_permitted": False,
        "fresh_seed_confirmation_required": True,
        "claim_scope": "contextual_behavioral_suppression_not_latent_erasure",
    }
    write_json(output / "arms" / "base_raw.json", base_forget_raw)
    write_json(output / "arms" / "base_retain_raw.json", base_retain_raw)
    write_json(output / "arms" / "v5_sidecar_raw.json", candidate_forget_raw)
    write_json(output / "arms" / "v5_sidecar_retain_raw.json", candidate_retain_raw)
    write_json(output / "development_replay.json", result)
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("V5 consumed seed-1 development acceptance failed")


if __name__ == "__main__":
    main()
