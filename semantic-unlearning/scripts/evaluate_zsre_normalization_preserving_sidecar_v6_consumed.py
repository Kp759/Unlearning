#!/usr/bin/env python3
"""Evaluate frozen ZsRE V6 on consumed seed 1 as development only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

import build_zsre_normalization_preserving_sidecar_v6_consumed_split as split_build
import gagd_compare as gagd
import mcf_normalization_preserving_sidecar_v6_core as shared
import zsre_normalization_preserving_sidecar_v6_core as core
import zsre_zero_unlearn_official_eval as official


PROTOCOL = "zsre_normalization_preserving_sidecar_v6_consumed_development_replay_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


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
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--zsre-path", required=True)
    parser.add_argument("--wikidata-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument("--route-audit-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single",), default="single")
    parser.add_argument("--zsre-url", default=official.ZSRE_URL)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.seed != 1:
        parser.error("ZsRE V6 consumed replay is restricted to seed 1")
    if args.forget_num != 50 or args.retain_num != 1000:
        parser.error("ZsRE V6 is locked to 50 forget / 1000 retain records")
    if args.route_audit_batch_size != 128:
        parser.error("ZsRE V6 route-audit batch size is frozen at 128")
    if args.eval_batch_size <= 0:
        parser.error("evaluation batch size must be positive")
    return args


def grouped_cases(
    records: Sequence[Mapping[str, Any]],
    tok: Any,
    *,
    llama_like: bool,
) -> Dict[str, list[official.PredictionCase]]:
    result: Dict[str, list[official.PredictionCase]] = {
        "rewrite": [],
        "paraphrase": [],
        "neighborhood": [],
    }
    for record in records:
        for case in official.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
        ):
            result[case.prompt_type].append(case)
    return result


def _flat_input_ids(tok: Any, prompt: str) -> list[int]:
    encoded = tok(str(prompt))["input_ids"]
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu().tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("expected one tokenized scorer input")
        encoded = encoded[0]
    return [int(item) for item in encoded]


def _target_token_id(
    tok: Any,
    case: official.PredictionCase,
    *,
    llama_like: bool,
) -> int:
    return int(
        official.official_target_ids(
            tok,
            [case.target_text],
            llama_like=llama_like,
            device=torch.device("cpu"),
        )[0]
    )


def scorer_collision_audit(
    tok: Any,
    forget_groups: Mapping[str, Sequence[official.PredictionCase]],
    retain_groups: Mapping[str, Sequence[official.PredictionCase]],
    *,
    llama_like: bool,
) -> Dict[str, Any]:
    positive = [
        (f"forget_{group}", case)
        for group in ("rewrite", "paraphrase")
        for case in forget_groups[group]
    ]
    preservation = [
        ("forget_neighborhood", case)
        for case in forget_groups["neighborhood"]
    ] + [
        (f"retain_{group}", case)
        for group in ("rewrite", "paraphrase", "neighborhood")
        for case in retain_groups[group]
    ]

    def rows(values: Sequence[tuple[str, official.PredictionCase]]) -> list[dict[str, Any]]:
        output = []
        for group, case in values:
            input_ids = _flat_input_ids(tok, case.prompt)
            target_id = _target_token_id(tok, case, llama_like=llama_like)
            output.append(
                {
                    "group": group,
                    "case_id": int(case.case_id),
                    "prompt_type": str(case.prompt_type),
                    "prompt_index": int(case.prompt_index),
                    "token_index": int(case.token_index),
                    "input_sha256": canonical_sha256(input_ids),
                    "scorer_identity_sha256": canonical_sha256([input_ids, target_id]),
                    "target_token_id": target_id,
                }
            )
        return output

    positive_rows = rows(positive)
    preservation_rows = rows(preservation)
    positive_inputs: Dict[str, list[dict[str, Any]]] = {}
    positive_scorers: Dict[str, list[dict[str, Any]]] = {}
    for row in positive_rows:
        positive_inputs.setdefault(row["input_sha256"], []).append(row)
        positive_scorers.setdefault(row["scorer_identity_sha256"], []).append(row)
    full_input_overlaps = []
    exact_scorer_collisions = []
    for row in preservation_rows:
        for owner in positive_inputs.get(row["input_sha256"], []):
            full_input_overlaps.append(
                {
                    "forget_group": owner["group"],
                    "forget_case_id": owner["case_id"],
                    "preservation_group": row["group"],
                    "preservation_case_id": row["case_id"],
                    "input_sha256": row["input_sha256"],
                    "forget_target_token_id": owner["target_token_id"],
                    "preservation_target_token_id": row["target_token_id"],
                }
            )
        for owner in positive_scorers.get(row["scorer_identity_sha256"], []):
            exact_scorer_collisions.append(
                {
                    "forget_group": owner["group"],
                    "forget_case_id": owner["case_id"],
                    "preservation_group": row["group"],
                    "preservation_case_id": row["case_id"],
                    "scorer_identity_sha256": row["scorer_identity_sha256"],
                    "target_token_id": row["target_token_id"],
                }
            )
    return {
        "identity": "tokenized_full_scorer_input_and_target_token_id",
        "forget_positive_scorer_cells": len(positive_rows),
        "preservation_scorer_cells": len(preservation_rows),
        "full_input_overlap_count": len(full_input_overlaps),
        "exact_scorer_collision_count": len(exact_scorer_collisions),
        "full_input_overlaps": full_input_overlaps,
        "exact_scorer_collisions": exact_scorer_collisions,
        "precedence_rule": "forget_positive_precedes_exact_preservation_label",
        "official_metrics_modified": False,
        "coherent_for_exact_raw_preservation": len(full_input_overlaps) == 0,
        "passed": len(exact_scorer_collisions) == 0,
    }


def route_audit(
    router: Any,
    tok: Any,
    forget_groups: Mapping[str, Sequence[official.PredictionCase]],
    retain_groups: Mapping[str, Sequence[official.PredictionCase]],
    case_to_owner: Mapping[int, int],
    *,
    llama_like: bool,
    ppl_text: str,
    batch_size: int,
) -> Dict[str, Any]:
    groups = {
        **{f"forget_{key}": list(value) for key, value in forget_groups.items()},
        **{f"retain_{key}": list(value) for key, value in retain_groups.items()},
    }
    reports: Dict[str, Any] = {}
    unexpected: list[dict[str, Any]] = []
    for name, cases in groups.items():
        positive = name in ("forget_rewrite", "forget_paraphrase")
        expected_cells = 0
        observed_cells = 0
        owner_cells = 0
        unexpected_cells = 0
        missed_rows = 0
        for start in range(0, len(cases), int(batch_size)):
            chunk = list(cases[start : start + int(batch_size)])
            encoded = tok(
                [case.prompt for case in chunk],
                padding=True,
                return_tensors="pt",
            )
            active = router.route(encoded["input_ids"]).active
            observed_cells += int(active.sum())
            for local_index, case in enumerate(chunk):
                active_indices = active[local_index].nonzero(
                    as_tuple=False
                ).flatten().tolist()
                owner = case_to_owner.get(int(case.case_id)) if positive else None
                if owner is not None:
                    expected_cells += 1
                    owner_active = owner in active_indices
                    owner_cells += int(owner_active)
                    missed_rows += int(not owner_active)
                    unexpected_cells += sum(int(item) != owner for item in active_indices)
                else:
                    unexpected_cells += len(active_indices)
                    for record_index in active_indices:
                        unexpected.append(
                            {
                                "group": name,
                                "case_id": int(case.case_id),
                                "prompt_sha256": hashlib.sha256(
                                    case.prompt.encode("utf-8")
                                ).hexdigest(),
                                "target_token_id": _target_token_id(
                                    tok,
                                    case,
                                    llama_like=llama_like,
                                ),
                                "router_record_index": int(record_index),
                                "router_subject": router.subjects[int(record_index)]
                                if hasattr(router, "subjects")
                                else None,
                            }
                        )
        reports[name] = {
            "rows": len(cases),
            "expected_owner_cells": expected_cells,
            "observed_route_cells": observed_cells,
            "observed_owner_cells": owner_cells,
            "unexpected_route_cells": unexpected_cells,
            "missed_owner_rows": missed_rows,
            "passed": missed_rows == 0 and unexpected_cells == 0,
        }

    ppl_inputs = tok(
        [ppl_text],
        return_tensors="pt",
        max_length=100,
        truncation=True,
    )
    ppl_routes = int(router.route(ppl_inputs["input_ids"]).active.sum())
    reports["ppl"] = {
        "rows": 1,
        "observed_route_cells": ppl_routes,
        "passed": ppl_routes == 0,
    }
    positive_coverage = all(
        reports[name]["passed"]
        for name in ("forget_rewrite", "forget_paraphrase")
    )
    preservation_closed = all(
        reports[name]["passed"]
        for name in (
            "forget_neighborhood",
            "retain_rewrite",
            "retain_paraphrase",
            "retain_neighborhood",
        )
    )
    return {
        "groups": reports,
        "all_forget_rewrite_paraphrase_owner_routes_open": positive_coverage,
        "all_forget_neighborhood_routes_closed": reports["forget_neighborhood"][
            "passed"
        ],
        "all_retain_routes_closed": all(
            reports[name]["passed"]
            for name in ("retain_rewrite", "retain_paraphrase", "retain_neighborhood")
        ),
        "ppl_route_closed": reports["ppl"]["passed"],
        "unexpected_preservation_route_cells": sum(
            reports[name]["unexpected_route_cells"]
            for name in (
                "forget_neighborhood",
                "retain_rewrite",
                "retain_paraphrase",
                "retain_neighborhood",
            )
        ),
        "unexpected_routes": unexpected,
        "passed": bool(positive_coverage and preservation_closed and ppl_routes == 0),
    }


def exact_preservation_comparison(
    base_forget_raw: Sequence[Mapping[str, Any]],
    candidate_forget_raw: Sequence[Mapping[str, Any]],
    base_retain_raw: Sequence[Mapping[str, Any]],
    candidate_retain_raw: Sequence[Mapping[str, Any]],
    *,
    base_ppl: float,
    candidate_ppl: float,
) -> Dict[str, Any]:
    def neighborhoods(rows: Sequence[Mapping[str, Any]]) -> list[bool]:
        return [
            bool(value)
            for row in rows
            for value in row["post"].get("neighborhood_prompts_correct", [])
        ]

    base_neighborhood = neighborhoods(base_forget_raw)
    candidate_neighborhood = neighborhoods(candidate_forget_raw)
    checks = {
        "forget_neighborhood_raw_exact": base_neighborhood == candidate_neighborhood,
        "retain_raw_exact": list(base_retain_raw) == list(candidate_retain_raw),
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
        raise FileExistsError(output)
    output.mkdir(parents=True)

    candidate_root = Path(args.candidate_run_dir).resolve()
    candidate_path = candidate_root / "zsre_v6_normalization_preserving_sidecar.pt"
    completion_path = candidate_root / "completion.json"
    manifest_path = Path(args.split_manifest).resolve()
    for path in (candidate_path, completion_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    completion = load_json(completion_path)
    manifest = load_json(manifest_path)
    if (
        completion.get("passed") is not True
        or int(completion.get("seed", -1)) != 1
        or completion.get("candidate_sha256") != sha256_file(candidate_path)
        or completion.get("eligible_for_consumed_development_replay") is not True
        or completion.get("fresh_seed_evaluation_permitted") is not False
        or completion.get("evaluation_prompts_seen") != 0
        or manifest.get("protocol") != split_build.PROTOCOL
        or manifest.get("evaluation_status")
        != "consumed_development_not_blind_not_official"
        or manifest.get("candidate_process_evaluation_prompts_seen") != 0
    ):
        raise RuntimeError("ZsRE V6 candidate is not eligible for consumed replay")

    zsre_path = official.download_zsre(Path(args.zsre_path).resolve(), args.zsre_url)
    if manifest.get("source_sha256") != sha256_file(zsre_path):
        raise RuntimeError("ZsRE source differs from the direct-only split binding")

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
    forget_records, retain_records = official.load_official_eval_records(
        zsre_path,
        tok,
        forget_num=50,
        retain_num=1000,
        seed=1,
        zsre_url=args.zsre_url,
    )
    state = torch.load(candidate_path, map_location="cpu", weights_only=False)
    core.validate_candidate_state(state)
    if state["source_hashes"]["split_manifest"] != sha256_file(manifest_path):
        raise RuntimeError("ZsRE V6 split manifest differs from candidate binding")
    if bool(state["llama_like"]) != bool(llama):
        raise RuntimeError("ZsRE V6 tokenizer family differs from Base")
    data_bindings = [
        (
            int(record["case_id"]),
            str(record["requested_rewrite"]["subject"]),
            str(record["requested_rewrite"]["prompt"]).format(
                str(record["requested_rewrite"]["subject"])
            ),
            str(record["requested_rewrite"]["target_true"]["str"]),
        )
        for record in forget_records
    ]
    state_bindings = list(
        zip(
            [int(item) for item in state["case_ids"]],
            state["subjects"],
            state["direct_prompts"],
            state["target_true"],
        )
    )
    if state_bindings != data_bindings:
        raise RuntimeError("ZsRE V6 candidate bindings differ from consumed seed 1")

    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    input_hash_before = shared.tensor_sha256(input_layer.weight)
    output_hash_before = shared.tensor_sha256(output_layer.weight)
    runtime = core.install_candidate(model, tok, state)
    # Expose subjects only for hashed route-forensics metadata.  Routing itself
    # remains the parameter-free token matcher in scoped_span_edit.
    runtime.router.subjects = [str(item) for item in state["subjects"]]
    ppl_text = official.load_official_ppl_text(Path(args.wikidata_dir))
    if not ppl_text:
        raise RuntimeError("official PPL text is unavailable")
    forget_groups = grouped_cases(forget_records, tok, llama_like=llama)
    retain_groups = grouped_cases(retain_records, tok, llama_like=llama)
    case_to_owner = {
        int(case_id): index for index, case_id in enumerate(state["case_ids"])
    }

    try:
        collisions = scorer_collision_audit(
            tok,
            forget_groups,
            retain_groups,
            llama_like=llama,
        )
        routing = route_audit(
            runtime.router,
            tok,
            forget_groups,
            retain_groups,
            case_to_owner,
            llama_like=llama,
            ppl_text=ppl_text,
            batch_size=int(args.route_audit_batch_size),
        )
        write_json(output / "scorer_collision_audit.json", collisions)
        write_json(output / "route_audit.json", routing)
        if not routing["passed"]:
            raise RuntimeError("ZsRE V6 route preflight failed before metrics")
        if not collisions["coherent_for_exact_raw_preservation"]:
            raise RuntimeError(
                "ZsRE V6 seed 1 contains an exact forget/preservation input conflict"
            )

        print("Stage 1: consumed ZsRE seed 1 reconstructed Base replay")
        runtime.sidecar.enabled = False
        base_forget, base_forget_raw = official.evaluate_record_split(
            model,
            tok,
            forget_records,
            device,
            llama_like=llama,
            split_name="forget",
            batch_size=int(args.eval_batch_size),
        )
        base_retain, base_retain_raw = official.evaluate_record_split(
            model,
            tok,
            retain_records,
            device,
            llama_like=llama,
            split_name="retain",
            batch_size=int(args.eval_batch_size),
        )
        base_ppl = official.official_perplexity(model, tok, ppl_text, device)

        print("Stage 2: consumed ZsRE seed 1 V6 replay")
        runtime.sidecar.enabled = True
        candidate_forget, candidate_forget_raw = official.evaluate_record_split(
            model,
            tok,
            forget_records,
            device,
            llama_like=llama,
            split_name="forget",
            batch_size=int(args.eval_batch_size),
        )
        candidate_retain, candidate_retain_raw = official.evaluate_record_split(
            model,
            tok,
            retain_records,
            device,
            llama_like=llama,
            split_name="retain",
            batch_size=int(args.eval_batch_size),
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
    behavioral_checks = {
        "forget_eff_zero": float(candidate_forget["Eff"]) == 0.0,
        "forget_gen_zero": float(candidate_forget["Gen"]) == 0.0,
        "all_rewrite_original_answer_tokens_incorrect": int(
            candidate_forget["post_rewrite_correct_tokens"]
        )
        == 0,
        "all_paraphrase_original_answer_tokens_incorrect": int(
            candidate_forget["post_paraphrase_correct_tokens"]
        )
        == 0,
        "forget_specificity_exact_base": bool(
            preservation["checks"]["forget_neighborhood_raw_exact"]
        ),
        "retain_exact_base": bool(preservation["checks"]["retain_raw_exact"]),
        "ppl_exact_base": bool(preservation["checks"]["ppl_exact"]),
    }
    integrity = {
        "embedding_hash_unchanged": input_hash_before
        == shared.tensor_sha256(input_layer.weight),
        "lm_head_hash_unchanged": output_hash_before
        == shared.tensor_sha256(output_layer.weight),
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
        routing["passed"]
        and collisions["coherent_for_exact_raw_preservation"]
        and preservation["passed"]
        and all(behavioral_checks.values())
        and integrity["passed"]
    )
    result = {
        "schema_version": 1,
        "kind": "zsre_normalization_preserving_sidecar_v6_consumed_development_replay",
        "protocol": PROTOCOL,
        "passed": passed,
        "evaluation_status": "consumed_development_not_blind_not_official",
        "seed": 1,
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
        "scorer_collision_audit": collisions,
        "route_audit": routing,
        "exact_preservation": preservation,
        "integrity": integrity,
        "used_for_architecture_development": True,
        "blind_or_official_claim_permitted": False,
        "fresh_seed_freeze_receipt_required": True,
        "claim_scope": core.CLAIM_SCOPE,
    }
    write_json(output / "arms" / "base_forget_raw.json", base_forget_raw)
    write_json(output / "arms" / "base_retain_raw.json", base_retain_raw)
    write_json(output / "arms" / "v6_forget_raw.json", candidate_forget_raw)
    write_json(output / "arms" / "v6_retain_raw.json", candidate_retain_raw)
    write_json(output / "development_replay.json", result)
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("ZsRE V6 consumed seed-1 development acceptance failed")


if __name__ == "__main__":
    main()
