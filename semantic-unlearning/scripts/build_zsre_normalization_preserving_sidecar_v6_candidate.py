#!/usr/bin/env python3
"""Build the direct-only ZsRE V6 seed-1 development candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import build_zsre_normalization_preserving_sidecar_v6_consumed_split as split_build
import build_zsre_zerounlearn_locked_split as locked
import gagd_compare as gagd
import mcf_normalization_preserving_sidecar_v6_core as shared
import scoped_span_edit as scoped
import zsre_normalization_preserving_sidecar_v6_core as core
import zsre_zero_unlearn_official_eval as official
from mcf_zero_unlearn_official_eval import is_llama_like


PROTOCOL = "zsre_normalization_preserving_sidecar_v6_consumed_candidate_build_v1"
FORBIDDEN_EVALUATION_ENVIRONMENT_VARIABLES = (
    "ZSRE_PATH",
    "OFFICIAL_ZSRE_PATH",
    "OFFICIAL_EVAL_PATH",
    "PARAPHRASE_PATH",
    "NEIGHBORHOOD_PATH",
    "RETAIN_EVAL_PATH",
    "WIKIDATA_DIR",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-visible-path", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--development-registry", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reserved-token-pool-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single",), default="single")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.seed != 1:
        parser.error("ZsRE V6 development is frozen to consumed seed 1")
    if args.reserved_token_pool_size != 128:
        parser.error("ZsRE V6 is frozen to 128 reserved special-token slots")
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    return args


def validate_environment_firewall() -> None:
    exposed = [
        name
        for name in FORBIDDEN_EVALUATION_ENVIRONMENT_VARIABLES
        if str(os.environ.get(name, "")).strip()
    ]
    if exposed:
        raise RuntimeError(
            "evaluation path leaked into ZsRE V6 candidate builder: "
            + ", ".join(sorted(exposed))
        )


def validate_registry(registry: Mapping[str, Any]) -> None:
    if (
        registry.get("protocol") != core.PROTOCOL
        or registry.get("status") != "consumed_seed1_development_only"
        or registry.get("architecture") != core.architecture()
        or registry.get("candidate_view")
        != "direct_forget_prompt_subject_and_original_answer_only"
        or registry.get("heldout_view")
        != "forget_rephrase_and_locality_plus_all_retain_and_ppl"
        or int(registry.get("reserved_special_token_pool_size", -1)) != 128
        or registry.get("fresh_seed_candidate_or_evaluation_permitted") is not False
        or registry.get("collision_policy", {}).get("official_metrics_unchanged")
        is not True
    ):
        raise RuntimeError("ZsRE V6 development registry differs from the frozen design")


def validate_inputs(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    training_path: Path,
) -> None:
    locked.assert_locked([dict(item) for item in records])
    view = manifest.get("candidate_view", {})
    if (
        manifest.get("protocol") != split_build.PROTOCOL
        or manifest.get("architecture_protocol") != core.PROTOCOL
        or int(manifest.get("seed", -1)) != 1
        or manifest.get("evaluation_status")
        != "consumed_development_not_blind_not_official"
        or manifest.get("training_visible_direct_only_sha256")
        != sha256_file(training_path)
        or int(manifest.get("sampling", {}).get("forget_num", -1)) != 50
        or int(manifest.get("sampling", {}).get("retain_num", -1)) != 1000
        or int(view.get("direct_forget_records", -1)) != 50
        or [int(record["case_id"]) for record in records]
        != [
            int(item)
            for item in manifest.get("sampling", {}).get("forget_case_ids", [])
        ]
        or any(
            int(view.get(key, -1)) != 0
            for key in (
                "paraphrase_prompts_serialized",
                "neighborhood_prompts_serialized",
                "retain_prompts_serialized",
                "ppl_documents_serialized",
            )
        )
        or view.get("probe_fields_absent_not_masked") is not True
        or manifest.get("candidate_process_evaluation_prompts_seen") != 0
    ):
        raise RuntimeError("ZsRE V6 candidate input violates the direct-only firewall")


def direct_cases(
    records: Sequence[Mapping[str, Any]],
    tok: Any,
    *,
    llama_like: bool,
) -> list[official.PredictionCase]:
    return [
        case
        for record in records
        for case in official.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
            prompt_types=("rewrite",),
        )
    ]


@torch.no_grad()
def permutation_audit(
    model: Any,
    tok: Any,
    runtime: core.Runtime,
    state: Mapping[str, Any],
    cases: Sequence[official.PredictionCase],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    case_to_owner = {
        int(case_id): index for index, case_id in enumerate(state["case_ids"])
    }
    route_misses = 0
    cross_routes = 0
    multiset_failures = 0
    outside_bank_changed_cells = 0
    evaluated = 0
    for start in range(0, len(cases), int(batch_size)):
        chunk = list(cases[start : start + int(batch_size)])
        encoded = tok(
            [case.prompt for case in chunk],
            padding=True,
            return_tensors="pt",
        ).to(device)
        runtime.sidecar.enabled = False
        base = model(**encoded, use_cache=False).logits.detach().clone()
        runtime.sidecar.enabled = True
        candidate = model(**encoded, use_cache=False).logits.detach()
        route = runtime.router.state
        if route is None:
            raise RuntimeError("ZsRE V6 permutation audit produced no route state")
        attention = encoded["attention_mask"]
        final_positions = attention.sum(dim=1) - 1
        for local_index, case in enumerate(chunk):
            owner = case_to_owner[int(case.case_id)]
            active = route.active[local_index].nonzero(as_tuple=False).flatten().tolist()
            route_misses += int(owner not in active)
            cross_routes += sum(int(item) != owner for item in active)
            final = int(final_positions[local_index])
            bank = [
                *[int(item) for item in state["sensitive_token_ids"][owner]],
                *[int(item) for item in state["reserved_token_ids"]],
                *[int(item) for item in state["promoted_token_ids"][owner]],
            ]
            bank_tensor = torch.tensor(bank, dtype=torch.long, device=device)
            if not torch.equal(
                base[local_index, final, bank_tensor].sort().values,
                candidate[local_index, final, bank_tensor].sort().values,
            ):
                multiset_failures += 1
            changed = (
                candidate[local_index, final] != base[local_index, final]
            ).nonzero(as_tuple=False).flatten().tolist()
            allowed = set(bank)
            outside_bank_changed_cells += sum(int(item) not in allowed for item in changed)
            evaluated += 1
    return {
        "prompt_instances": evaluated,
        "owner_route_misses": route_misses,
        "cross_routes": cross_routes,
        "full_source_bank_multiset_failures": multiset_failures,
        "ordinary_outside_bank_changed_cells": outside_bank_changed_cells,
        "passed": (
            route_misses == 0
            and cross_routes == 0
            and multiset_failures == 0
            and outside_bank_changed_cells == 0
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_environment_firewall()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    paths = {
        "training_visible": Path(args.training_visible_path).resolve(),
        "split_manifest": Path(args.split_manifest).resolve(),
        "development_registry": Path(args.development_registry).resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    records_raw = json.loads(paths["training_visible"].read_text(encoding="utf-8"))
    if not isinstance(records_raw, list) or not all(
        isinstance(item, dict) for item in records_raw
    ):
        raise RuntimeError("ZsRE V6 direct-only candidate view must be a list")
    manifest = load_json(paths["split_manifest"])
    registry = load_json(paths["development_registry"])
    validate_registry(registry)
    validate_inputs(records_raw, manifest, training_path=paths["training_visible"])

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
    llama = is_llama_like(model, tok)
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if input_layer is None or output_layer is None:
        raise RuntimeError("Base model does not expose embeddings and LM head")
    input_hash_before = shared.tensor_sha256(input_layer.weight)
    output_hash_before = shared.tensor_sha256(output_layer.weight)

    case_ids = [int(record["case_id"]) for record in records_raw]
    rewrites = [record["requested_rewrite"] for record in records_raw]
    subjects = [str(rewrite["subject"]) for rewrite in rewrites]
    direct_prompts = [str(rewrite["prompt"]).format(str(rewrite["subject"])) for rewrite in rewrites]
    target_true = [str(rewrite["target_true"]["str"]) for rewrite in rewrites]
    target_true_ids = [
        official.original_answer_token_ids(tok, value, llama_like=llama)
        for value in target_true
    ]
    neutral_id = official.resolve_neutral_target_token_id(tok, official.NEUTRAL_TARGET)
    reserved_ids, reserved_strings = shared.select_reserved_token_pool(
        tok,
        excluded_token_ids=[
            int(neutral_id),
            *(item for row in target_true_ids for item in row),
        ],
        requested_size=int(args.reserved_token_pool_size),
    )
    if max(
        [
            int(neutral_id),
            *reserved_ids,
            *(item for row in target_true_ids for item in row),
        ]
    ) >= int(output_layer.weight.shape[0]):
        raise RuntimeError("ZsRE V6 token bank exceeds Base LM-head vocabulary")
    state = core.build_candidate_state(
        seed=1,
        case_ids=case_ids,
        subjects=subjects,
        direct_prompts=direct_prompts,
        subject_patterns=scoped.build_subject_patterns(tok, subjects),
        target_true=target_true,
        target_true_ids=target_true_ids,
        neutral_target=official.NEUTRAL_TARGET,
        neutral_target_id=int(neutral_id),
        reserved_token_ids=reserved_ids,
        reserved_token_strings=reserved_strings,
        llama_like=llama,
        base_embedding_sha256=input_hash_before,
        base_lm_head_sha256=output_hash_before,
        source_hashes={name: sha256_file(path) for name, path in paths.items()},
    )
    runtime = core.install_candidate(model, tok, state)
    try:
        cases = direct_cases(records_raw, tok, llama_like=llama)
        permutation = permutation_audit(
            model,
            tok,
            runtime,
            state,
            cases,
            device=device,
            batch_size=int(args.batch_size),
        )
        if not permutation["passed"]:
            raise RuntimeError("ZsRE V6 direct permutation invariant failed")
        runtime.sidecar.enabled = True
        predicted = official.predict_cases(
            model,
            tok,
            cases,
            device,
            llama_like=llama,
            batch_size=int(args.batch_size),
            desc="ZsRE V6 direct-only audit",
        )
        direct_audit = {
            "prompt_instances": len(predicted),
            "correct_original_answer_tokens": sum(bool(row["correct"]) for row in predicted),
            "incorrect_original_answer_tokens": sum(not bool(row["correct"]) for row in predicted),
            "criterion": {"correct_original_answer_tokens": 0},
        }
        direct_audit["passed"] = direct_audit["correct_original_answer_tokens"] == 0
        if not direct_audit["passed"]:
            raise RuntimeError("ZsRE V6 direct-only suppression failed")

        unrelated = tok(
            ["An unrelated control sentence.", "No registered subject is here."],
            padding=True,
            return_tensors="pt",
        ).to(device)
        runtime.sidecar.enabled = False
        base_logits = model(**unrelated, use_cache=False).logits.detach().clone()
        runtime.sidecar.enabled = True
        candidate_logits = model(**unrelated, use_cache=False).logits.detach()
        identity = {
            "unrelated_rows": 2,
            "bit_identical_logits": bool(torch.equal(base_logits, candidate_logits)),
            "route_cells": int(runtime.router.state.active.sum()),
        }
        identity["passed"] = bool(
            identity["bit_identical_logits"] and identity["route_cells"] == 0
        )
        if not identity["passed"]:
            raise RuntimeError("ZsRE V6 closed-route identity failed")
    finally:
        runtime.close()

    integrity = {
        "base_embedding_sha256_before": input_hash_before,
        "base_embedding_sha256_after": shared.tensor_sha256(input_layer.weight),
        "lm_head_sha256_before": output_hash_before,
        "lm_head_sha256_after": shared.tensor_sha256(output_layer.weight),
        "all_parameters_require_grad_false": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "optimizer_constructed": False,
        "gradient_updates_performed": 0,
    }
    integrity["passed"] = bool(
        integrity["base_embedding_sha256_before"]
        == integrity["base_embedding_sha256_after"]
        and integrity["lm_head_sha256_before"] == integrity["lm_head_sha256_after"]
        and integrity["all_parameters_require_grad_false"]
    )
    if not integrity["passed"]:
        raise RuntimeError("ZsRE V6 build changed a Base parameter")

    output.mkdir(parents=True)
    candidate_path = output / "zsre_v6_normalization_preserving_sidecar.pt"
    torch.save(state, candidate_path)
    completion = {
        "schema_version": 1,
        "kind": "zsre_normalization_preserving_sidecar_v6_consumed_candidate_completion",
        "protocol": PROTOCOL,
        "passed": True,
        "seed": 1,
        "evaluation_status": "consumed_development_candidate_not_blind_not_official",
        "architecture": state["architecture"],
        "standalone_direct_only_lineage": True,
        "direct_permutation_audit": permutation,
        "direct_training_visible_audit": direct_audit,
        "closed_route_identity_audit": identity,
        "integrity": integrity,
        "candidate_saved": True,
        "candidate_sha256": sha256_file(candidate_path),
        "eligible_for_consumed_development_replay": True,
        "fresh_seed_evaluation_permitted": False,
        "evaluation_prompts_seen": 0,
        "claim_scope": core.CLAIM_SCOPE,
    }
    write_json(output / "direct_permutation_audit.json", permutation)
    write_json(output / "direct_training_visible_audit.json", direct_audit)
    write_json(output / "closed_route_identity_audit.json", identity)
    write_json(output / "integrity.json", integrity)
    write_json(output / "completion.json", completion)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
