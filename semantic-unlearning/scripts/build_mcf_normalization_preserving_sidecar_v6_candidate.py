#!/usr/bin/env python3
"""Build one MCF V6 candidate for consumed-seed development replay only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import build_mcf_normalization_preserving_sidecar_v6_consumed_split as split_build
import build_mcf_sure_target_aware_direct_split as direct_split
import gagd_active_case_repair as official_tensors
import gagd_compare as gagd
import mcf_compositional_marker_write_read as compositional
import mcf_normalization_preserving_sidecar_v6_core as core
import scoped_span_edit as scoped
from mcf_zero_unlearn_official_eval import is_llama_like


PROTOCOL = "mcf_normalization_preserving_sidecar_v6_consumed_candidate_build_v1"
FORBIDDEN_EVALUATION_ENVIRONMENT_VARIABLES = (
    "MCF_PATH",
    "OFFICIAL_MCF_PATH",
    "OFFICIAL_EVAL_PATH",
    "PARAPHRASE_PATH",
    "NEIGHBORHOOD_PATH",
    "RETAIN_EVAL_PATH",
    "ADVERSARIAL_EVAL_PATH",
    "WIKIDATA_DIR",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
    parser.add_argument("--frame-lexicon", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--reserved-token-pool-size", type=int, default=128)
    parser.add_argument("--margin-floor", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single",), default="single")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.seed not in (1, 2):
        parser.error("V6 candidate build is restricted to consumed seeds 1 and 2")
    if args.forget_num != 50 or args.reserved_token_pool_size != 128:
        parser.error("V6 development is locked to 50 records and 128 reserved tokens")
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    if not math.isfinite(args.margin_floor) or args.margin_floor != 0.1:
        parser.error("V6 development margin floor is frozen at 0.1")
    return args


def validate_environment_firewall() -> None:
    exposed = [
        name
        for name in FORBIDDEN_EVALUATION_ENVIRONMENT_VARIABLES
        if str(os.environ.get(name, "")).strip()
    ]
    if exposed:
        raise RuntimeError(
            "evaluation path leaked into V6 candidate builder: "
            + ", ".join(sorted(exposed))
        )


def validate_registry(
    registry: Mapping[str, Any],
    lexicon: Mapping[str, Any],
) -> None:
    architecture = registry.get("architecture", {})
    if (
        registry.get("protocol") != core.PROTOCOL
        or registry.get("status") != "consumed_seed_development_only"
        or architecture.get("route")
        != "two_sided_complete_entity_and_frozen_relation_frame_grammar"
        or architecture.get("intervention")
        != "causal_directional_target_logit_permutation_with_reserved_special_slots"
        or architecture.get("normalization_preserving_by_construction") is not True
        or architecture.get("ordinary_non_target_logits_unchanged_by_construction")
        is not True
        or architecture.get("arithmetic_logit_offsets") != 0
        or architecture.get("reserved_special_token_pool_size") != 128
        or registry.get("frame_router", {}).get("content_sha256")
        != lexicon.get("lexicon_sha256")
        or registry.get("fresh_seed_firewall", {}).get(
            "seed_3_or_later_candidate_or_evaluation_permitted_by_this_registry"
        )
        is not False
    ):
        raise RuntimeError("V6 development registry differs from the locked design")


def validate_inputs(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    training_path: Path,
    seed: int,
    lexicon: Mapping[str, Any],
) -> None:
    direct_split.assert_direct_only_training_view(records)
    view = manifest.get("candidate_view", {})
    if (
        manifest.get("protocol") != split_build.PROTOCOL
        or int(manifest.get("seed", -1)) != int(seed)
        or manifest.get("evaluation_status")
        != "consumed_development_not_blind_not_official"
        or manifest.get("frame_lexicon_sha256") != lexicon.get("lexicon_sha256")
        or manifest.get("training_visible_target_aware_direct_sha256")
        != sha256_file(training_path)
        or int(manifest.get("sampling", {}).get("forget_num", -1)) != 50
        or [int(record["case_id"]) for record in records]
        != [int(item) for item in manifest.get("sampling", {}).get("forget_case_ids", [])]
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
        raise RuntimeError("V6 candidate input violates the direct-only firewall")


def _instances(
    records: Sequence[Mapping[str, Any]],
) -> list[official_tensors.MCFPromptInstance]:
    return [
        official_tensors.MCFPromptInstance(
            record_index=index,
            sampled_position=index,
            prompt_type="locked_training_direct",
            prompt_index=0,
            prompt=str(record["direct_prompt"]),
            target_new=str(record["reference"]),
            target_true=str(record["answer"]),
        )
        for index, record in enumerate(records)
    ]


@torch.no_grad()
def direct_margin_audit(
    model: Any,
    tok: Any,
    instances: Sequence[official_tensors.MCFPromptInstance],
    *,
    device: torch.device,
    llama_like: bool,
    batch_size: int,
) -> dict[str, Any]:
    margins: list[float] = []
    for start in range(0, len(instances), int(batch_size)):
        new_nll, true_nll = official_tensors.official_prompt_instance_nll_tensors(
            model,
            tok,
            instances[start : start + int(batch_size)],
            device,
            llama_like,
        )
        margins.extend(float(item) for item in (true_nll - new_nll).cpu())
    return {
        "prompt_instances": len(margins),
        "minimum_margin": min(margins),
        "maximum_margin": max(margins),
        "failures_at_zero": sum(item <= 0.0 for item in margins),
    }


@torch.no_grad()
def permutation_audit(
    model: Any,
    tok: Any,
    runtime: core.NormalizationPreservingRuntime,
    state: Mapping[str, Any],
    prompts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    multiset_failures = 0
    outside_bank_failures = 0
    rows = 0
    for start in range(0, len(prompts), int(batch_size)):
        chunk = list(prompts[start : start + int(batch_size)])
        encoded = tok(chunk, padding=True, return_tensors="pt").to(device)
        runtime.sidecar.enabled = False
        base = model(**encoded).logits.detach().clone()
        runtime.sidecar.enabled = True
        candidate = model(**encoded).logits.detach()
        route = runtime.router.state
        if route is None:
            raise RuntimeError("V6 permutation audit did not produce route state")
        causal = core.causal_after_subject_mask(route.span_masks).to(device)
        for local_index in range(len(chunk)):
            owner = start + local_index
            positions = causal[local_index, owner].nonzero(as_tuple=False).flatten()
            promoted = [int(item) for item in state["promoted_token_ids"][owner]]
            sensitive = [int(item) for item in state["sensitive_token_ids"][owner]]
            bank = [
                *sensitive,
                *[int(item) for item in state["reserved_token_ids"]],
                *promoted,
            ]
            bank_tensor = torch.tensor(bank, dtype=torch.long, device=device)
            if positions.numel():
                base_bank = base[local_index, positions[:, None], bank_tensor[None, :]]
                candidate_bank = candidate[
                    local_index, positions[:, None], bank_tensor[None, :]
                ]
                multiset_failures += int(
                    not torch.equal(
                        base_bank.sort(dim=-1).values,
                        candidate_bank.sort(dim=-1).values,
                    )
                )
            changed = (candidate[local_index] != base[local_index]).nonzero(
                as_tuple=False
            )
            if changed.numel():
                allowed = set(bank)
                outside_bank_failures += sum(
                    int(token_id) not in allowed
                    for _position, token_id in changed.tolist()
                )
            rows += 1
    return {
        "rows": rows,
        "full_source_bank_multiset_failures": multiset_failures,
        "ordinary_outside_bank_changed_cells": outside_bank_failures,
        "passed": multiset_failures == 0 and outside_bank_failures == 0,
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
        "frame_lexicon": Path(args.frame_lexicon).resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    records_raw = json.loads(paths["training_visible"].read_text(encoding="utf-8"))
    if not isinstance(records_raw, list):
        raise RuntimeError("V6 direct-only view must be a list")
    manifest = load_json(paths["split_manifest"])
    registry = load_json(paths["development_registry"])
    lexicon = load_json(paths["frame_lexicon"])
    core.validate_frame_lexicon(lexicon)
    validate_registry(registry, lexicon)
    validate_inputs(
        records_raw,
        manifest,
        training_path=paths["training_visible"],
        seed=int(args.seed),
        lexicon=lexicon,
    )
    records = compositional._record_views(records_raw)
    case_ids = [int(record["case_id"]) for record in records]

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
    input_hash_before = core.tensor_sha256(input_layer.weight)
    output_hash_before = core.tensor_sha256(output_layer.weight)

    subjects = [str(record["subject"]) for record in records]
    relation_ids = [str(record["relation_id"]) for record in records]
    target_new = [str(record["reference"]) for record in records]
    target_true = [str(record["answer"]) for record in records]
    target_new_ids = [
        core.official_target_token_ids(tok, item, llama_like=llama)
        for item in target_new
    ]
    target_true_ids = [
        core.official_target_token_ids(tok, item, llama_like=llama)
        for item in target_true
    ]
    reserved_ids, reserved_strings = core.select_reserved_token_pool(
        tok,
        excluded_token_ids=[
            item
            for bank in (target_new_ids, target_true_ids)
            for row in bank
            for item in row
        ],
        requested_size=int(args.reserved_token_pool_size),
    )
    if max(reserved_ids) >= int(output_layer.weight.shape[0]):
        raise RuntimeError(
            "tokenizer reserved-token reservoir exceeds the Base LM-head vocabulary"
        )
    state = core.build_candidate_state(
        seed=int(args.seed),
        case_ids=case_ids,
        subjects=subjects,
        relation_ids=relation_ids,
        frame_lexicon=lexicon,
        subject_patterns=scoped.build_subject_patterns(tok, subjects),
        target_new=target_new,
        target_true=target_true,
        target_new_ids=target_new_ids,
        target_true_ids=target_true_ids,
        reserved_token_ids=reserved_ids,
        reserved_token_strings=reserved_strings,
        llama_like=llama,
        base_embedding_sha256=input_hash_before,
        base_lm_head_sha256=output_hash_before,
        source_hashes={name: sha256_file(path) for name, path in paths.items()},
    )
    runtime = core.install_candidate(model, tok, state)
    try:
        direct_prompts = [str(record["direct_prompt"]) for record in records]
        encoded = tok(direct_prompts, padding=True, return_tensors="pt")
        active = runtime.router.route(encoded["input_ids"]).active
        expected = torch.eye(len(records), dtype=torch.bool)
        route_audit = {
            "direct_prompt_rows": len(records),
            "owner_routes": int((active & expected).sum()),
            "cross_routes": int((active & ~expected).sum()),
            "passed": bool(torch.equal(active, expected)),
        }
        if not route_audit["passed"]:
            raise RuntimeError("V6 direct route failed")
        permutation = permutation_audit(
            model,
            tok,
            runtime,
            state,
            direct_prompts,
            device=device,
            batch_size=int(args.batch_size),
        )
        if not permutation["passed"]:
            raise RuntimeError("V6 logit-permutation invariant failed")
        direct_audit = direct_margin_audit(
            model,
            tok,
            _instances(records),
            device=device,
            llama_like=llama,
            batch_size=int(args.batch_size),
        )
        direct_audit["criterion"] = {"minimum_margin": 0.1}
        direct_audit["passed"] = direct_audit["minimum_margin"] >= 0.1
        if not direct_audit["passed"]:
            raise RuntimeError("V6 direct margin failed")

        unrelated = tok(
            ["An unrelated control sentence.", "No registered entity is here."],
            padding=True,
            return_tensors="pt",
        ).to(device)
        runtime.sidecar.enabled = False
        base_logits = model(**unrelated).logits.detach().clone()
        runtime.sidecar.enabled = True
        candidate_logits = model(**unrelated).logits.detach()
        identity_audit = {
            "unrelated_rows": 2,
            "bit_identical_logits": bool(torch.equal(base_logits, candidate_logits)),
            "route_cells": int(runtime.router.state.active.sum()),
        }
        identity_audit["passed"] = bool(
            identity_audit["bit_identical_logits"]
            and identity_audit["route_cells"] == 0
        )
        if not identity_audit["passed"]:
            raise RuntimeError("V6 closed-route identity failed")
    finally:
        runtime.close()

    integrity = {
        "base_embedding_sha256_before": input_hash_before,
        "base_embedding_sha256_after": core.tensor_sha256(input_layer.weight),
        "lm_head_sha256_before": output_hash_before,
        "lm_head_sha256_after": core.tensor_sha256(output_layer.weight),
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
        raise RuntimeError("V6 build changed a Base parameter")

    output.mkdir(parents=True)
    candidate_path = output / "v6_normalization_preserving_sidecar.pt"
    torch.save(state, candidate_path)
    completion = {
        "schema_version": 1,
        "kind": "mcf_normalization_preserving_sidecar_v6_consumed_candidate_completion",
        "protocol": PROTOCOL,
        "passed": True,
        "seed": int(args.seed),
        "evaluation_status": "consumed_development_candidate_not_blind_not_official",
        "architecture": state["architecture"],
        "standalone_direct_only_lineage": True,
        "route_audit": route_audit,
        "logit_permutation_audit": permutation,
        "direct_training_visible_audit": direct_audit,
        "closed_route_identity_audit": identity_audit,
        "integrity": integrity,
        "candidate_saved": True,
        "candidate_sha256": sha256_file(candidate_path),
        "eligible_for_consumed_development_replay": True,
        "fresh_seed_evaluation_permitted": False,
        "evaluation_prompts_seen": 0,
        "claim_scope": "contextual_behavioral_suppression_not_latent_erasure",
    }
    write_json(output / "route_audit.json", route_audit)
    write_json(output / "logit_permutation_audit.json", permutation)
    write_json(output / "direct_training_visible_audit.json", direct_audit)
    write_json(output / "closed_route_identity_audit.json", identity_audit)
    write_json(output / "integrity.json", integrity)
    write_json(output / "completion.json", completion)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
