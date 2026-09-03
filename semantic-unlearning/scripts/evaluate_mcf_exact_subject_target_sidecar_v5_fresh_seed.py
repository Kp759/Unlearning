#!/usr/bin/env python3
"""One-shot official confirmation of one frozen fresh-seed V5 candidate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import torch

import evaluate_mcf_exact_subject_target_sidecar_v5_seed1_development as dev
import gagd_compare as gagd
import mcf_exact_subject_target_sidecar_v5_core as core
import mcf_zero_unlearn_official_eval as official
from mcf_sampling import sample_official_mcf_records


PROTOCOL = "mcf_exact_subject_target_logit_sidecar_v5_0_fresh_multiseed_confirmation_v1"
REGISTERED_SEEDS = tuple(range(2, 11))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--wikidata-dir", required=True)
    parser.add_argument("--confirmation-registry", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--route-audit-batch-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--device-map", choices=("single",), default="single")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.seed not in REGISTERED_SEEDS:
        parser.error("one-shot V5 confirmation seed must be in [2, 10]")
    if args.route_audit_batch_size != 128:
        parser.error("fresh V5 route-audit batch size is frozen at 128")
    return args


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def validate_confirmation_registry(value: Mapping[str, Any], *, seed: int) -> None:
    acceptance = value.get("acceptance", {})
    policy = value.get("one_shot_policy", {})
    artifacts = value.get("frozen_artifacts", {})
    expected_acceptance = {
        "forget_eff": 0.0,
        "forget_gen": 0.0,
        "minimum_rewrite_paraphrase_margin": 0.1,
        "forget_neighborhood_raw_exact_base": True,
        "retain_raw_exact_base": True,
        "ppl_exact_base": True,
        "base_embedding_hash_unchanged": True,
        "lm_head_hash_unchanged": True,
    }
    if (
        value.get("protocol") != PROTOCOL
        or int(seed) not in value.get("registered_seeds", [])
        or value.get("execution_order") != "ascending_seed_order"
        or acceptance != expected_acceptance
        or value.get("architecture", {}).get("logit_bias") != 256.0
        or value.get("architecture", {}).get("learned_detector_parameters") != 0
        or value.get("architecture", {}).get("learned_actuator_parameters") != 0
        or artifacts.get("official_mcf_sha256")
        != "977a6acce4705507b5fd6bfcea8f61cd78f9ed0f9cd9c9a6bcd6c8a3ed61c833"
        or artifacts.get("architecture_registry_sha256")
        != "27e7be0ac869a96cdb2bc8f9a97fabd0806ee3983fcf6802ea13d3f6c9b45a87"
        or artifacts.get("relation_lexicon_file_sha256")
        != "255ed77de292dd7198b4af9673805d0ef06456c6598f150fa559e6734be6bf4a"
        or artifacts.get("relation_lexicon_content_sha256")
        != "1d23f5ca2aff28500897b9b2f31f03dad6105cbe423c50a3e1b4fca506e01af8"
        or not all(
            policy.get(key) is True
            for key in (
                "candidate_frozen_before_official_open",
                "official_open_event_written_before_dataset_read",
                "retry_after_official_open_prohibited",
                "resume_after_official_open_prohibited",
                "checkpoint_selection_from_official_results_prohibited",
                "hyperparameter_changes_between_registered_seeds_prohibited",
                "failed_seeds_reported_without_replacement",
                "later_seeds_may_not_be_modified_in_response_to_earlier_results",
            )
        )
        or value.get("claim_scope")
        != "contextual_behavioral_suppression_not_latent_erasure"
    ):
        raise RuntimeError("fresh V5 confirmation registry is stale or weakened")


def candidate_paths(root: Path) -> dict[str, Path]:
    return {
        "candidate": root / "method" / "v5_exact_subject_target_sidecar.pt",
        "completion": root / "method" / "completion.json",
        "architecture_registry": root / "protocol" / "experiment_registry.json",
        "confirmation_registry": root / "protocol" / "fresh_multiseed_registry.json",
        "relation_lexicon": root / "protocol" / "relation_suffix_lexicon.json",
        "training_visible": root
        / "protocol"
        / "seed_split"
        / "training_visible_target_aware_direct.json",
        "split_manifest": root / "protocol" / "seed_split" / "split_manifest.json",
    }


def validate_frozen_candidate(
    root: Path,
    *,
    seed: int,
    external_confirmation_registry: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    paths = candidate_paths(root)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(paths["confirmation_registry"]) != sha256_file(
        external_confirmation_registry
    ):
        raise RuntimeError("candidate confirmation registry differs from evaluator")
    registry = load_json(external_confirmation_registry)
    validate_confirmation_registry(registry, seed=int(seed))
    frozen_artifacts = registry["frozen_artifacts"]
    if (
        sha256_file(paths["architecture_registry"])
        != frozen_artifacts["architecture_registry_sha256"]
        or sha256_file(paths["relation_lexicon"])
        != frozen_artifacts["relation_lexicon_file_sha256"]
    ):
        raise RuntimeError("candidate uses a non-frozen V5 protocol artifact")
    completion = load_json(paths["completion"])
    state = torch.load(paths["candidate"], map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise RuntimeError("fresh V5 candidate state is not a mapping")
    core.validate_candidate_state(state)
    expected_source_paths = {
        "training_visible": paths["training_visible"],
        "split_manifest": paths["split_manifest"],
        "architecture_registry": paths["architecture_registry"],
        "confirmation_registry": paths["confirmation_registry"],
        "relation_lexicon": paths["relation_lexicon"],
    }
    source_hashes = state.get("source_hashes", {})
    sources_reproduce = isinstance(source_hashes, Mapping) and all(
        source_hashes.get(name) == sha256_file(path)
        for name, path in expected_source_paths.items()
    )
    if (
        completion.get("passed") is not True
        or completion.get("seed") != int(seed)
        or completion.get("candidate_sha256") != sha256_file(paths["candidate"])
        or completion.get("standalone_direct_only_lineage") is not True
        or completion.get("v6_2_writer_artifacts_consumed") != 0
        or completion.get("eligible_for_one_shot_fresh_seed_evaluation") is not True
        or completion.get("official_evaluation_allowed_in_this_process") is not False
        or completion.get("official_evaluation_prompts_seen") != 0
        or int(state.get("seed", -1)) != int(seed)
        or float(state.get("logit_bias", float("nan"))) != 256.0
        or state.get("optimizer_constructed") is not False
        or state.get("gradient_updates_performed") != 0
        or state.get("relation_lexicon_sha256")
        != frozen_artifacts["relation_lexicon_content_sha256"]
        or not sources_reproduce
    ):
        raise RuntimeError("fresh V5 candidate is not eligible for one-shot evaluation")
    return dict(state), completion, paths


def sampled_records_with_indices(
    data: Sequence[dict[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int], list[int]]:
    identity = {id(record): index for index, record in enumerate(data)}
    forget, retain = sample_official_mcf_records(
        data, forget_num=50, retain_num=1000, seed=int(seed), strict=True
    )
    return (
        [official.normalize_record(record) for record in forget],
        [official.normalize_record(record) for record in retain],
        [int(identity[id(record)]) for record in forget],
        [int(identity[id(record)]) for record in retain],
    )


def validate_candidate_bindings(
    state: Mapping[str, Any],
    forget_records: Sequence[Mapping[str, Any]],
    forget_indices: Sequence[int],
) -> None:
    expected = [
        (
            int(index),
            str(record["requested_rewrite"]["subject"]),
            str(record["requested_rewrite"].get("relation_id") or ""),
            str(record["requested_rewrite"]["target_new"]["str"]),
            str(record["requested_rewrite"]["target_true"]["str"]),
        )
        for index, record in zip(forget_indices, forget_records)
    ]
    observed = [
        (int(case_id), str(subject), str(relation), str(new), str(true))
        for case_id, subject, relation, new, true in zip(
            state["case_ids"],
            state["subjects"],
            state["relation_ids"],
            state["target_new"],
            state["target_true"],
        )
    ]
    if observed != expected:
        raise RuntimeError(
            "fresh V5 candidate bindings differ from official seed identity"
        )


def parameter_versions(model: Any) -> dict[str, int]:
    return {
        name: int(parameter._version) for name, parameter in model.named_parameters()
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(
            f"one-shot output already exists; retry/resume is prohibited: {output}"
        )
    candidate_root = Path(args.candidate_run_dir).resolve()
    confirmation_registry_path = Path(args.confirmation_registry).resolve()
    state, completion, paths = validate_frozen_candidate(
        candidate_root,
        seed=int(args.seed),
        external_confirmation_registry=confirmation_registry_path,
    )

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
    torch.set_grad_enabled(False)
    device = gagd.first_device(model)
    llama = official.is_llama_like(model, tok)
    if bool(state["llama_like"]) != bool(llama):
        raise RuntimeError("fresh V5 candidate tokenizer family differs from Base")
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    input_hash_before = core.tensor_sha256(input_layer.weight)
    output_hash_before = core.tensor_sha256(output_layer.weight)
    versions_before = parameter_versions(model)
    candidate_hash_before = sha256_file(paths["candidate"])
    registry_hash_before = sha256_file(confirmation_registry_path)
    runtime = core.install_candidate(model, tok, state)

    official_opened = False
    try:
        output.mkdir(parents=True)
        pre_open = {
            "schema_version": 1,
            "kind": "mcf_v5_fresh_seed_pre_open_firewall",
            "protocol": PROTOCOL,
            "created_at_utc": utc_now(),
            "seed": int(args.seed),
            "official_evaluation_opened": False,
            "all_candidate_checks_passed": True,
            "candidate_sha256": candidate_hash_before,
            "confirmation_registry_sha256": registry_hash_before,
            "evaluator_source_sha256": sha256_file(Path(__file__).resolve()),
            "git_head": _git_head(Path(__file__).resolve().parents[2]),
            "fixed_acceptance": load_json(confirmation_registry_path)["acceptance"],
            "gradient_mode_enabled": torch.is_grad_enabled(),
            "optimizer_constructed": False,
            "gradient_updates_performed": 0,
            "checkpoint_selection_or_retry_allowed": False,
            "resume_allowed": False,
        }
        if pre_open["gradient_mode_enabled"] is not False:
            raise RuntimeError("gradient mode remained enabled before official open")
        write_json(output / "pre_open_firewall_receipt.json", pre_open)

        # This durable event precedes the first byte read from either official
        # evaluation source.  Any subsequent failure consumes the seed.
        open_event = {
            "schema_version": 1,
            "kind": "mcf_v5_fresh_seed_official_open_event",
            "protocol": PROTOCOL,
            "opened_at_utc": utc_now(),
            "seed": int(args.seed),
            "official_evaluation_opened": True,
            "candidate_sha256": candidate_hash_before,
            "official_mcf_path": str(Path(args.mcf_path).resolve()),
            "official_mcf_sha256": None,
            "one_shot_evaluation": True,
            "resume_or_retry_allowed": False,
        }
        write_json(output / "official_evaluation_opened.json", open_event)
        official_opened = True

        mcf_path = Path(args.mcf_path).resolve()
        mcf_sha256 = sha256_file(mcf_path)
        registry = load_json(confirmation_registry_path)
        expected_mcf_sha256 = str(registry["frozen_artifacts"]["official_mcf_sha256"])
        state_mcf_sha256 = str(
            state["relation_lexicon"].get("derivation", {}).get("source_mcf_sha256", "")
        )
        open_event["official_mcf_sha256"] = mcf_sha256
        open_event["expected_official_mcf_sha256"] = expected_mcf_sha256
        write_json(output / "official_evaluation_opened.json", open_event)
        if mcf_sha256 != expected_mcf_sha256 or state_mcf_sha256 != expected_mcf_sha256:
            raise RuntimeError("official MCF source differs from frozen V5 source")
        data = official.load_mcf(mcf_path)
        (
            forget_records,
            retain_records,
            forget_indices,
            retain_indices,
        ) = sampled_records_with_indices(data, seed=int(args.seed))
        validate_candidate_bindings(state, forget_records, forget_indices)
        if set(forget_indices) & set(retain_indices):
            raise RuntimeError("official forget and retain identities overlap")

        ppl_text = official.load_official_ppl_text(args.wikidata_dir)
        if not ppl_text:
            raise RuntimeError("official PPL text is unavailable")
        open_event.update(
            {
                "forget_dataset_indices_sha256": canonical_sha256(forget_indices),
                "retain_dataset_indices_sha256": canonical_sha256(retain_indices),
                "official_forget_records": len(forget_records),
                "official_retain_records": len(retain_records),
                "ppl_text_sha256": sha256_text(ppl_text),
                "ppl_text_characters": len(ppl_text),
            }
        )
        write_json(output / "official_evaluation_opened.json", open_event)

        routing = dev.route_audit(
            runtime.router,
            tok,
            forget_records,
            retain_records,
            ppl_text=ppl_text,
            batch_size=int(args.route_audit_batch_size),
        )
        write_json(output / "route_audit.json", routing)

        print(f"Stage 1: seed {args.seed} reconstructed Base official evaluation")
        runtime.sidecar.enabled = False
        base_forget, base_forget_raw = official.evaluate_record_split(
            model, tok, forget_records, device, llama, "forget"
        )
        base_retain, base_retain_raw = official.evaluate_record_split(
            model, tok, retain_records, device, llama, "retain"
        )
        base_ppl = official.official_perplexity(model, tok, ppl_text, device)

        print(f"Stage 2: seed {args.seed} frozen V5 official evaluation")
        runtime.sidecar.enabled = True
        candidate_forget, candidate_forget_raw = official.evaluate_record_split(
            model, tok, forget_records, device, llama, "forget"
        )
        candidate_retain, candidate_retain_raw = official.evaluate_record_split(
            model, tok, retain_records, device, llama, "retain"
        )
        candidate_ppl = official.official_perplexity(model, tok, ppl_text, device)

        preservation = dev.exact_preservation_comparison(
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
        versions_after = parameter_versions(model)
        integrity_checks = {
            "base_parameter_versions_unchanged": versions_before == versions_after,
            "embedding_hash_unchanged": input_hash_before
            == core.tensor_sha256(input_layer.weight),
            "lm_head_hash_unchanged": output_hash_before
            == core.tensor_sha256(output_layer.weight),
            "candidate_checkpoint_bit_identical": candidate_hash_before
            == sha256_file(paths["candidate"]),
            "confirmation_registry_bit_identical": registry_hash_before
            == sha256_file(confirmation_registry_path),
            "all_parameters_require_grad_false": all(
                not parameter.requires_grad for parameter in model.parameters()
            ),
            "optimizer_constructed": False,
            "gradient_updates_performed": 0,
        }
        integrity_passed = bool(
            all(
                value is True
                for key, value in integrity_checks.items()
                if key not in ("optimizer_constructed", "gradient_updates_performed")
            )
            and integrity_checks["optimizer_constructed"] is False
            and integrity_checks["gradient_updates_performed"] == 0
        )
        integrity = {"checks": integrity_checks, "passed": integrity_passed}
        passed = bool(
            routing["passed"]
            and preservation["passed"]
            and all(behavioral_checks.values())
            and integrity_passed
        )

        raw_paths = {
            "reconstructed_base_forget": output
            / "arms"
            / "reconstructed_base_forget_raw.json",
            "reconstructed_base_retain": output
            / "arms"
            / "reconstructed_base_retain_raw.json",
            "v5_sidecar_forget": output / "arms" / "v5_sidecar_forget_raw.json",
            "v5_sidecar_retain": output / "arms" / "v5_sidecar_retain_raw.json",
        }
        for key, value in (
            ("reconstructed_base_forget", base_forget_raw),
            ("reconstructed_base_retain", base_retain_raw),
            ("v5_sidecar_forget", candidate_forget_raw),
            ("v5_sidecar_retain", candidate_retain_raw),
        ):
            write_json(raw_paths[key], {"rows": value})

        result = {
            "schema_version": 1,
            "kind": "mcf_v5_fresh_seed_one_shot_official_evaluation",
            "protocol": PROTOCOL,
            "completed_at_utc": utc_now(),
            "seed": int(args.seed),
            "passed": passed,
            "official_evaluation_opened": True,
            "official_evaluation_completed": True,
            "candidate_sha256": candidate_hash_before,
            "fixed_architecture": load_json(confirmation_registry_path)["architecture"],
            "arms": {
                "reconstructed_base": {
                    "forget": base_forget,
                    "retain": base_retain,
                    "PPL": base_ppl,
                },
                "v5_sidecar": {
                    "forget": candidate_forget,
                    "retain": candidate_retain,
                    "PPL": candidate_ppl,
                },
            },
            "behavioral_checks": behavioral_checks,
            "route_audit": routing,
            "exact_preservation": preservation,
            "runtime_integrity": integrity,
            "official_result_used_for_training_or_checkpoint_selection": False,
            "retry_or_resume_permitted": False,
            "later_seed_configuration_change_permitted": False,
            "claim_scope": "contextual_behavioral_suppression_not_latent_erasure",
            "strong_unlearning_claim_ready": False,
        }
        result_path = output / "official_evaluation.json"
        write_json(result_path, result)
        artifacts = [
            output / "pre_open_firewall_receipt.json",
            output / "official_evaluation_opened.json",
            output / "route_audit.json",
            *raw_paths.values(),
            result_path,
        ]
        manifest = {
            "schema_version": 1,
            "kind": "mcf_v5_fresh_seed_official_artifact_manifest",
            "protocol": PROTOCOL,
            "artifacts": [
                {"path": str(path), "sha256": sha256_file(path)} for path in artifacts
            ],
        }
        manifest_path = output / "artifact_manifest.json"
        write_json(manifest_path, manifest)
        terminal = {
            "schema_version": 1,
            "kind": "mcf_v5_fresh_seed_official_terminal_status",
            "protocol": PROTOCOL,
            "seed": int(args.seed),
            "status": "completed",
            "official_evaluation_opened": True,
            "official_evaluation_completed": True,
            "candidate_acceptance_passed": passed,
            "official_evaluation_sha256": sha256_file(result_path),
            "artifact_manifest_sha256": sha256_file(manifest_path),
            "retry_or_resume_permitted": False,
        }
        write_json(output / "terminal_status.json", terminal)
        print(
            json.dumps(
                {
                    "seed": int(args.seed),
                    "passed": passed,
                    "Eff": candidate_forget["Eff"],
                    "Gen": candidate_forget["Gen"],
                    "Spe": candidate_forget["Spe"],
                    "Base_Spe": base_forget["Spe"],
                    "PPL": candidate_ppl,
                    "Base_PPL": base_ppl,
                    "retry_or_resume_permitted": False,
                },
                indent=2,
            )
        )
    except BaseException as exc:
        if output.exists():
            write_json(
                output / "terminal_status.json",
                {
                    "schema_version": 1,
                    "kind": "mcf_v5_fresh_seed_official_terminal_status",
                    "protocol": PROTOCOL,
                    "seed": int(args.seed),
                    "status": (
                        "failed_after_official_open"
                        if official_opened
                        else "failed_before_official_open"
                    ),
                    "official_evaluation_opened": official_opened,
                    "official_evaluation_completed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "retry_or_resume_permitted": not official_opened,
                },
            )
        raise
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
