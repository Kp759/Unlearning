"""Registered overlap-paired edit with an explicit protected-gradient nullspace."""
import argparse
import hashlib
import json
from pathlib import Path

from freeze_static_overlap_development import load_protocol, normalized
from mcf_shadow_relation_prompts import RELATION_NOUN_PHRASES
from static_overlap_data import load_bundle
from static_overlap_endpoint_protocol import main as register, load_pilot as load_endpoint
from static_overlap_endpoint_protocol import claim_evaluation as claim_endpoint
from static_overlap_mlp_protocol import FIT_QUESTIONS, forbidden_texts, load_pilot as load_development


METHOD = "static_overlap_orthogonal_rewire_v4"
PLAN = {
    "steps": 120,
    "check_every": 5,
    "pair_batch": 8,
    "background_retain_batch": 32,
    "language_batch": 4,
    "retain_weight": .1,
    "kl_weight": 1.,
    "background_weight": .25,
    "step_radius": .025,
    "backtracks": 12,
    "max_training_seconds": 1800,
    "max_stalled_steps": 20,
    "min_gate_nll_gain": .05,
    "stalled_gates": 8,
    "max_failed_preservation_gates": 12,
    "target_probability": 1e-6,
    "max_length": 512,
    "seed": 1,
    "retain_nll_budget": .05,
    "retain_kl_budget": .01,
    "fitting_nll_margin": .01,
    "fitting_kl_margin": .002,
    "untie_before_optimization": True,
    "synthetic_same_subject_relations_per_fact": 4,
    "protected_basis_rank": 128,
    "protected_basis_relative_tolerance": 1e-5,
    "min_forget_residual_ratio": 1e-4,
    "pairing": "verified_overlap_pairs_plus_base_distilled_same_subject_relation_queries",
    "objective": "forget_GA_projected_into_explicit_protected_Jacobian_nullspace_plus_retain_GD_KL",
    "optimizer": "direct_trust_region_step_after_projection_no_Adam",
    "parameterization_description": (
        "Untied embedding and LM head with separate rank-16 sparse row factors, "
        "plus rank-16 updates on the original overlap-localized MLP channels; "
        "all factors merge into a static native checkpoint and receive no runtime routing input."
    ),
    "training_preservation_failure": "restore_last_safe_state_accumulate_worst_anchors_rebuild_basis",
}


def locality_prompt_specs(source, data, plan=PLAN):
    facts = list(source["facts"]) + list(data.get("facts", []))
    forget = [fact for fact in source["facts"] if fact["role"] == "forget"]
    relations = sorted({fact["relation"] for fact in facts
                        if fact["relation"] in RELATION_NOUN_PHRASES})
    result = []
    for fact in forget:
        alternatives = sorted(
            (relation for relation in relations if relation != fact["relation"]),
            key=lambda relation: hashlib.sha256(
                f"{plan['seed']}:{fact['id']}:{relation}".encode()).digest(),
        )[:plan["synthetic_same_subject_relations_per_fact"]]
        if len(alternatives) != plan["synthetic_same_subject_relations_per_fact"]:
            raise ValueError(f"Insufficient distinct relation templates for {fact['id']}")
        for relation in alternatives:
            result.append({"forget_fact_id": fact["id"], "subject": fact["subject"],
                           "forget_relation": fact["relation"], "locality_relation": relation,
                           "prompt": FIT_QUESTIONS[0].format(
                               subject=fact["subject"], relation=RELATION_NOUN_PHRASES[relation])})
    return result


def exclusion_audit(development_protocol):
    development = load_development(development_protocol)
    frozen = load_protocol(development["head_protocol_path"])
    source, _, _ = load_bundle(development["source_bundle"]["path"])
    data = json.loads(Path(development["data"]["path"]).read_text())
    evaluation, _, _ = load_bundle(frozen["files"]["evaluation_bundle"]["path"], "evaluation")
    final, _, _ = load_bundle(frozen["files"]["final_retention"]["path"], "preservation_test")
    mcf = json.loads(Path(frozen["files"]["mcf"]["path"]).read_text())
    forbidden, _ = forbidden_texts(frozen, source, evaluation, final, mcf)
    specs = locality_prompt_specs(source, data)
    collisions = [row["prompt"] for row in specs
                  if normalized(row["prompt"]) in forbidden
                  or normalized(row["prompt"] + " unknown") in forbidden]
    if collisions:
        raise ValueError("Synthetic locality prompt overlaps a frozen final input")
    return {"synthetic_queries": len(specs), "frozen_final_text_collisions": 0,
            "official_probe_text_used_as_training_text": False}


def load_pilot(path):
    return load_endpoint(path, method=METHOD, plan=PLAN)


def claim_evaluation(path, checkpoint):
    return claim_endpoint(path, checkpoint, method=METHOD, protocol_loader=load_pilot)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--development-protocol", required=True)
    parser.add_argument("--overlap-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args, _ = parser.parse_known_args(argv)
    audit = exclusion_audit(args.development_protocol)
    print(json.dumps({"phase": "same_subject_locality_exclusion_audit", **audit}), flush=True)
    return register(argv, method=METHOD, plan=PLAN)


if __name__ == "__main__":
    main()
