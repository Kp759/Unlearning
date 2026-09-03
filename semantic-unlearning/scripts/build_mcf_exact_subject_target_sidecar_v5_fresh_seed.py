#!/usr/bin/env python3
"""Freeze one fresh-seed V5 candidate from the direct-only minimized view.

Unlike the original seed-1 builder, this process has no V6.2 writer input: V5
does not install or consume any writer tensor.  The only seed-specific content
accepted here is the direct forget view produced by the isolated minimizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import build_mcf_exact_subject_target_sidecar_v5_fresh_seed_split as fresh_split
import build_mcf_sure_target_aware_direct_split as direct_split
import gagd_active_case_repair as official_tensors
import gagd_compare as gagd
import mcf_compositional_marker_write_read as compositional
import mcf_exact_subject_target_sidecar_v5_core as core
import mcf_sure_directional_emb_lm_stage1 as directional
import scoped_span_edit as scoped
from mcf_zero_unlearn_official_eval import is_llama_like


PROTOCOL = "mcf_exact_subject_target_logit_sidecar_v5_0_fresh_candidate_build_v1"
CONFIRMATION_PROTOCOL = (
    "mcf_exact_subject_target_logit_sidecar_v5_0_fresh_multiseed_confirmation_v1"
)
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


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-visible-path", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--architecture-registry", required=True)
    parser.add_argument("--confirmation-registry", required=True)
    parser.add_argument("--relation-lexicon", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--logit-bias", type=float, default=256.0)
    parser.add_argument("--margin-floor", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single",), default="single")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.seed not in range(2, 11):
        parser.error("fresh V5 candidate seed must be in [2, 10]")
    if args.forget_num != 50 or args.batch_size <= 0:
        parser.error("fresh V5 is locked to 50 records and a positive batch size")
    if not math.isfinite(args.logit_bias) or args.logit_bias != 256.0:
        parser.error("fresh V5 logit bias is frozen at 256")
    if not math.isfinite(args.margin_floor) or args.margin_floor != 0.1:
        parser.error("fresh V5 direct margin floor is frozen at 0.1")
    return args


def validate_environment_firewall() -> None:
    exposed = [
        name
        for name in FORBIDDEN_EVALUATION_ENVIRONMENT_VARIABLES
        if str(os.environ.get(name, "")).strip()
    ]
    if exposed:
        raise RuntimeError(
            "evaluation path leaked into fresh V5 candidate builder: "
            + ", ".join(sorted(exposed))
        )


def validate_registries(
    architecture: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    *,
    seed: int,
) -> None:
    expected_architecture = {
        "route": "boundary_aware_exact_subject_and_frozen_relation_suffix_grammar",
        "intervention": "causal_sparse_target_logit_bias",
        "base_embedding_mutation": False,
        "transformer_mutation": False,
        "lm_head_mutation": False,
        "learned_detector_parameters": 0,
        "learned_actuator_parameters": 0,
        "logit_bias": 256.0,
    }
    if (
        architecture.get("protocol") != core.PROTOCOL
        or architecture.get("official_evaluation_prohibited") is not True
        or any(
            architecture.get("architecture", {}).get(key) != value
            for key, value in expected_architecture.items()
        )
    ):
        raise RuntimeError("V5 architecture registry differs from the frozen design")
    if (
        confirmation.get("protocol") != CONFIRMATION_PROTOCOL
        or confirmation.get("status")
        != "registered_before_opening_any_seed_2_to_10_evaluation_probe"
        or int(seed) not in confirmation.get("registered_seeds", [])
        or confirmation.get("architecture", {}).get("logit_bias") != 256.0
        or confirmation.get("frozen_artifacts", {}).get("official_mcf_sha256")
        != "977a6acce4705507b5fd6bfcea8f61cd78f9ed0f9cd9c9a6bcd6c8a3ed61c833"
        or confirmation.get("claim_scope")
        != "contextual_behavioral_suppression_not_latent_erasure"
    ):
        raise RuntimeError("fresh multiseed confirmation registry is invalid")


def validate_split_manifest(
    manifest: Mapping[str, Any],
    lexicon: Mapping[str, Any],
    *,
    seed: int,
) -> None:
    view = manifest.get("candidate_view", {})
    if (
        manifest.get("protocol") != fresh_split.PROTOCOL
        or int(manifest.get("seed", -1)) != int(seed)
        or manifest.get("source_sha256")
        != lexicon.get("derivation", {}).get("source_mcf_sha256")
        or int(view.get("direct_forget_records", -1)) != 50
        or any(
            int(view.get(key, -1)) != 0
            for key in (
                "official_paraphrase_prompts_serialized",
                "official_neighborhood_prompts_serialized",
                "official_retain_prompts_serialized",
                "official_ppl_documents_serialized",
            )
        )
        or view.get("probe_fields_absent_not_masked") is not True
        or manifest.get("splitter_isolated_from_candidate_process") is not True
        or manifest.get("candidate_process_official_evaluation_prompts_seen") != 0
    ):
        raise RuntimeError("fresh V5 split manifest violates the direct-only firewall")


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
        margins.extend(float(value) for value in (true_nll - new_nll).cpu())
    return {
        "prompt_instances": len(margins),
        "minimum_margin": min(margins),
        "maximum_margin": max(margins),
        "failures_at_zero": sum(value <= 0.0 for value in margins),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_environment_firewall()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"fresh V5 candidate output already exists: {output}")
    paths = {
        "training_visible": Path(args.training_visible_path).resolve(),
        "split_manifest": Path(args.split_manifest).resolve(),
        "architecture_registry": Path(args.architecture_registry).resolve(),
        "confirmation_registry": Path(args.confirmation_registry).resolve(),
        "relation_lexicon": Path(args.relation_lexicon).resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    architecture_registry = load_json(paths["architecture_registry"])
    confirmation_registry = load_json(paths["confirmation_registry"])
    lexicon = load_json(paths["relation_lexicon"])
    core.validate_relation_lexicon(lexicon)
    validate_registries(
        architecture_registry, confirmation_registry, seed=int(args.seed)
    )
    frozen_artifacts = confirmation_registry["frozen_artifacts"]
    if (
        sha256_file(paths["architecture_registry"])
        != frozen_artifacts["architecture_registry_sha256"]
        or sha256_file(paths["relation_lexicon"])
        != frozen_artifacts["relation_lexicon_file_sha256"]
        or lexicon.get("lexicon_sha256")
        != frozen_artifacts["relation_lexicon_content_sha256"]
    ):
        raise RuntimeError("fresh V5 build input differs from a frozen artifact")

    locked_records, manifest = directional.validate_locked(
        paths["training_visible"],
        paths["split_manifest"],
        int(args.seed),
        int(args.forget_num),
    )
    direct_split.assert_direct_only_training_view(locked_records)
    validate_split_manifest(manifest, lexicon, seed=int(args.seed))
    records = compositional._record_views(locked_records)
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
    state = core.build_candidate_state(
        seed=int(args.seed),
        case_ids=case_ids,
        subjects=subjects,
        relation_ids=relation_ids,
        relation_lexicon=lexicon,
        subject_patterns=scoped.build_subject_patterns(tok, subjects),
        target_new=target_new,
        target_true=target_true,
        target_new_ids=[
            core.official_target_token_ids(tok, value, llama_like=llama)
            for value in target_new
        ],
        target_true_ids=[
            core.official_target_token_ids(tok, value, llama_like=llama)
            for value in target_true
        ],
        llama_like=llama,
        logit_bias=float(args.logit_bias),
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
            raise RuntimeError("fresh V5 direct route failed")
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
            raise RuntimeError("fresh V5 direct margin failed")

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
            raise RuntimeError("fresh V5 closed-route identity failed")
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
        raise RuntimeError("fresh V5 build changed a Base parameter")

    output.mkdir(parents=True)
    candidate_path = output / "v5_exact_subject_target_sidecar.pt"
    torch.save(state, candidate_path)
    completion = {
        "schema_version": 1,
        "kind": "mcf_exact_subject_target_sidecar_v5_fresh_candidate_completion",
        "protocol": PROTOCOL,
        "passed": True,
        "seed": int(args.seed),
        "architecture": state["architecture"],
        "standalone_direct_only_lineage": True,
        "v6_2_writer_artifacts_consumed": 0,
        "route_audit": route_audit,
        "direct_training_visible_audit": direct_audit,
        "closed_route_identity_audit": identity_audit,
        "integrity": integrity,
        "candidate_saved": True,
        "candidate_sha256": sha256_file(candidate_path),
        "eligible_for_one_shot_fresh_seed_evaluation": True,
        "official_evaluation_allowed_in_this_process": False,
        "official_evaluation_prompts_seen": 0,
        "claim_scope": "contextual_behavioral_suppression_not_latent_erasure",
    }
    write_json(output / "route_audit.json", route_audit)
    write_json(output / "direct_training_visible_audit.json", direct_audit)
    write_json(output / "closed_route_identity_audit.json", identity_audit)
    write_json(output / "integrity.json", integrity)
    write_json(output / "completion.json", completion)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
