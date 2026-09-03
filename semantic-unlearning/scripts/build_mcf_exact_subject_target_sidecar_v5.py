#!/usr/bin/env python3
"""Build the optimizer-free V5 exact-subject target sidecar.

Only the locked direct-only training view is opened.  No paraphrase,
neighborhood, retain, official MCF, or official PPL source is accepted by this
process.  The produced candidate contains no Base-model tensor and records no
gradient update.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

import build_mcf_sure_target_aware_direct_split as locked_split
import gagd_active_case_repair as official_tensors
import gagd_compare as gagd
import mcf_compositional_marker_write_read as compositional
import mcf_embedding_keyed_neuron_erasure as legacy
import mcf_exact_subject_target_sidecar_v5_core as core
import mcf_sure_directional_emb_lm_stage1 as directional
import scoped_span_edit as scoped
from mcf_zero_unlearn_official_eval import is_llama_like


PROTOCOL = core.PROTOCOL
FORBIDDEN_EVALUATION_ENVIRONMENT_VARIABLES = (
    "MCF_PATH",
    "OFFICIAL_MCF_PATH",
    "OFFICIAL_EVAL_PATH",
    "PARAPHRASE_PATH",
    "NEIGHBORHOOD_PATH",
    "RETAIN_EVAL_PATH",
    "ADVERSARIAL_EVAL_PATH",
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


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-visible-path", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--context-manifest", required=True)
    parser.add_argument("--stage1-state", required=True)
    parser.add_argument("--stage1-report", required=True)
    parser.add_argument("--stage1-writer-log", required=True)
    parser.add_argument("--clean-stage1-portability-preflight", required=True)
    parser.add_argument("--clean-stage1-acceptance", required=True)
    parser.add_argument("--experiment-registry", required=True)
    parser.add_argument("--relation-lexicon", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--logit-bias", type=float, default=core.DEFAULT_LOGIT_BIAS)
    parser.add_argument("--margin-floor", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single",), default="single")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.forget_num <= 0 or args.batch_size <= 0:
        parser.error("record and batch counts must be positive")
    if not math.isfinite(args.logit_bias) or args.logit_bias <= 0:
        parser.error("--logit-bias must be finite and positive")
    if not math.isfinite(args.margin_floor) or args.margin_floor < 0:
        parser.error("--margin-floor must be finite and non-negative")
    return args


def validate_environment_firewall() -> None:
    exposed = [
        name
        for name in FORBIDDEN_EVALUATION_ENVIRONMENT_VARIABLES
        if str(os.environ.get(name, "")).strip()
    ]
    if exposed:
        raise RuntimeError(
            "official-evaluation path leaked into V5 builder: "
            + ", ".join(sorted(exposed))
        )


def validate_registry(value: Mapping[str, Any], args: argparse.Namespace) -> None:
    if (
        value.get("protocol") != PROTOCOL
        or int(value.get("schema_version", -1)) != 1
        or value.get("official_evaluation_prohibited") is not True
    ):
        raise RuntimeError("V5 registry is stale")
    architecture = value.get("architecture", {})
    expected = {
        "route": "boundary_aware_exact_subject_and_frozen_relation_suffix_grammar",
        "intervention": "causal_sparse_target_logit_bias",
        "base_embedding_mutation": False,
        "transformer_mutation": False,
        "lm_head_mutation": False,
        "learned_detector_parameters": 0,
        "learned_actuator_parameters": 0,
        "logit_bias": float(args.logit_bias),
    }
    if not isinstance(architecture, Mapping) or any(
        architecture.get(key) != expected_value
        for key, expected_value in expected.items()
    ):
        raise RuntimeError("V5 registry architecture differs from the implementation")


def _instances(records: Sequence[Mapping[str, Any]]) -> list[official_tensors.MCFPromptInstance]:
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
    model,
    tok,
    instances: Sequence[official_tensors.MCFPromptInstance],
    *,
    device: torch.device,
    llama_like: bool,
    batch_size: int,
) -> Dict[str, Any]:
    margins = []
    for start in range(0, len(instances), batch_size):
        new_nll, true_nll = official_tensors.official_prompt_instance_nll_tensors(
            model,
            tok,
            instances[start : start + batch_size],
            device,
            llama_like,
        )
        margins.extend(float(value) for value in (true_nll - new_nll).cpu())
    return {
        "prompt_instances": len(margins),
        "minimum_margin": min(margins),
        "maximum_margin": max(margins),
        "failures_at_zero": sum(value <= 0.0 for value in margins),
        "margins": margins,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_environment_firewall()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"V5 output already exists: {output}")
    output.mkdir(parents=True)

    paths = {
        "training_visible": Path(args.training_visible_path).resolve(),
        "split_manifest": Path(args.split_manifest).resolve(),
        "context_manifest": Path(args.context_manifest).resolve(),
        "stage1_state": Path(args.stage1_state).resolve(),
        "stage1_report": Path(args.stage1_report).resolve(),
        "stage1_writer_log": Path(args.stage1_writer_log).resolve(),
        "clean_stage1_portability": Path(
            args.clean_stage1_portability_preflight
        ).resolve(),
        "clean_stage1_acceptance": Path(args.clean_stage1_acceptance).resolve(),
        "registry": Path(args.experiment_registry).resolve(),
        "relation_lexicon": Path(args.relation_lexicon).resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    registry = load_json(paths["registry"])
    validate_registry(registry, args)
    relation_lexicon = load_json(paths["relation_lexicon"])
    core.validate_relation_lexicon(relation_lexicon)

    locked_records, split_manifest = directional.validate_locked(
        paths["training_visible"],
        paths["split_manifest"],
        int(args.seed),
        int(args.forget_num),
    )
    if split_manifest.get("protocol") != locked_split.PROTOCOL:
        raise RuntimeError("V5 requires the locked direct-only MCF split")
    locked_split.assert_direct_only_training_view(locked_records)
    records = compositional._record_views(locked_records)
    case_ids = [int(record["case_id"]) for record in records]

    context_manifest = load_json(paths["context_manifest"])
    stage1_state = torch.load(paths["stage1_state"], map_location="cpu", weights_only=False)
    if not isinstance(stage1_state, Mapping):
        raise RuntimeError("V6.2 writer state is not a mapping")
    stage1_report = load_json(paths["stage1_report"])
    legacy._validate_firewall(context_manifest, stage1_state)
    clean_receipt = legacy._validate_clean_stage1_lineage(
        context_manifest,
        stage1_state,
        stage1_report,
        paths["context_manifest"],
        paths["stage1_writer_log"],
    )
    if str(Path(args.model_path).resolve()) != str(clean_receipt["base_model_path"]):
        raise RuntimeError("V5 Base-model path differs from its V6.2 lineage")
    if [int(value) for value in stage1_state.get("case_ids", [])] != case_ids:
        raise RuntimeError("V6.2 and V5 case order differs")
    acceptance = load_json(paths["clean_stage1_acceptance"])
    portability = load_json(paths["clean_stage1_portability"])
    if (
        acceptance.get("passed") is not True
        or bool(acceptance.get("official_evaluation_opened"))
        or acceptance.get("training_safe_portability") != portability
    ):
        raise RuntimeError("V6.2 clean writer acceptance is invalid")

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
    patterns = scoped.build_subject_patterns(tok, subjects)
    target_new_ids = [
        core.official_target_token_ids(tok, value, llama_like=llama)
        for value in target_new
    ]
    target_true_ids = [
        core.official_target_token_ids(tok, value, llama_like=llama)
        for value in target_true
    ]
    source_hashes = {name: sha256_file(path) for name, path in paths.items()}
    state = core.build_candidate_state(
        seed=int(args.seed),
        case_ids=case_ids,
        subjects=subjects,
        relation_ids=relation_ids,
        relation_lexicon=relation_lexicon,
        subject_patterns=patterns,
        target_new=target_new,
        target_true=target_true,
        target_new_ids=target_new_ids,
        target_true_ids=target_true_ids,
        llama_like=llama,
        logit_bias=float(args.logit_bias),
        base_embedding_sha256=input_hash_before,
        base_lm_head_sha256=output_hash_before,
        source_hashes=source_hashes,
    )
    runtime = core.install_candidate(model, tok, state)
    try:
        direct_prompts = [str(record["direct_prompt"]) for record in records]
        direct_encoded = tok(direct_prompts, padding=True, return_tensors="pt")
        direct_route = runtime.router.route(direct_encoded["input_ids"])
        direct_owner = torch.eye(len(records), dtype=torch.bool)
        route_audit = {
            "direct_prompt_rows": len(records),
            "owner_routes": int((direct_route.active & direct_owner).sum()),
            "cross_routes": int((direct_route.active & ~direct_owner).sum()),
            "every_direct_owner_routed": bool(
                (direct_route.active & direct_owner).sum(dim=1).eq(1).all()
            ),
            "passed": bool(torch.equal(direct_route.active, direct_owner)),
        }
        if not route_audit["passed"]:
            raise RuntimeError("V5 complete-subject direct route failed")

        print("Stage 1: verify optimizer-free exact-subject direct behavior")
        direct_audit = direct_margin_audit(
            model,
            tok,
            _instances(records),
            device=device,
            llama_like=llama,
            batch_size=int(args.batch_size),
        )
        direct_audit["criterion"] = {"minimum_margin": float(args.margin_floor)}
        direct_audit["passed"] = bool(
            direct_audit["minimum_margin"] >= float(args.margin_floor)
        )
        if not direct_audit["passed"]:
            raise RuntimeError("V5 direct behavioral margin did not pass")

        # The unmodified path must remain an exact identity.  These prompts do
        # not contain a registered subject, so the hook returns the Base output
        # Tensor rather than performing an arithmetic zero update.
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
            raise RuntimeError("V5 closed-route Base identity failed")
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
    }
    integrity["passed"] = bool(
        integrity["base_embedding_sha256_before"]
        == integrity["base_embedding_sha256_after"]
        and integrity["lm_head_sha256_before"] == integrity["lm_head_sha256_after"]
        and integrity["all_parameters_require_grad_false"]
    )
    if not integrity["passed"]:
        raise RuntimeError("V5 changed a Base parameter")

    candidate_path = output / "v5_exact_subject_target_sidecar.pt"
    torch.save(state, candidate_path)
    completion = {
        "schema_version": 1,
        "kind": "mcf_exact_subject_target_sidecar_v5_build_completion",
        "protocol": PROTOCOL,
        "passed": True,
        "architecture": state["architecture"],
        "route_audit": route_audit,
        "direct_training_visible_audit": direct_audit,
        "closed_route_identity_audit": identity_audit,
        "integrity": integrity,
        "candidate_saved": True,
        "candidate_path": str(candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "seed": int(args.seed),
        "evaluation_status": (
            "consumed_development_only_not_blind"
            if int(args.seed) == 1
            else "fresh_reserved_unopened"
        ),
        "eligible_for_consumed_seed_1_development_replay": int(args.seed) == 1,
        "eligible_for_fresh_seed_evaluation_only_after_seed_specific_freeze": int(args.seed) > 1,
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
