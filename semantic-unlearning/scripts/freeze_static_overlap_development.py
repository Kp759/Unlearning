#!/usr/bin/env python3
"""Gate the final development adaptation on parity and freeze an independent test.

No model is loaded or scored. Reading evaluation text here is only an exclusion
audit; neither it nor the newly sampled test is supplied to the solver.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from mcf_sampling import sample_official_mcf_records
from static_overlap_data import load_bundle, text_fingerprints, validate_bundle
from static_overlap_training import TrainConfig, sha256_file


PROTOCOL = "static_overlap_development_preservation_v1"
PLAN = {"taus": [0.0, 0.001, 0.01], "ridges": [0.0001, 0.01],
        "strengths": [0.25, 0.5, 1, 2, 4, 8, 16, 24, 32],
        "target_probability": 1e-6, "nll_safety_margin": 0.01,
        "kl_safety_margin": 0.002, "max_length": 512, "seed": 1}


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_parity(run, audit_path):
    """A failure or a stale audit must stop before any protocol files are made."""
    run, audit_path = Path(run), Path(audit_path)
    audit = json.loads(audit_path.read_text())
    for key, filename in (("training_bundle_sha256", "training_bundle.json"),
                          ("cache_sha256", "head_cache.pt"),
                          ("source_report_sha256", "training_report.json")):
        if audit.get(key) != sha256_file(run / filename):
            raise ValueError(f"Parity audit is stale or belongs to a different run: {filename}")
    verified = audit.get("training_only_verification", {})
    if verified.get("cache_model_parity_passed") is not True:
        raise ValueError("Cache/model parity did not pass. Debug the audit before changing protocol.")
    errors = verified.get("max_abs_errors", {})
    actual = verified.get("actual_rows", [])
    if not actual or set(errors) != {"base_nll", "nll", "kl"}:
        raise ValueError("Parity audit lacks numerical verification evidence")
    for key, value in errors.items():
        if not math.isfinite(value) or value < 0 or any(not math.isfinite(r[key]) for r in actual):
            raise ValueError("Parity audit contains invalid numerical evidence")
        # Original audit tests every example at atol=1e-4, rtol=1e-5. This
        # conservative aggregate check also rejects an inconsistent passed flag.
        ceiling = (1e-4 + 1e-5 * max(abs(r[key]) for r in actual)) / (1 - 1e-5)
        if value > ceiling:
            raise ValueError(f"Material {key} parity error: {value}")
    print(json.dumps({"phase": "parity_gate", "cache_model_parity_passed": True,
                      "max_abs_errors": errors, "actual_summary": verified["actual_summary"]}), flush=True)
    return audit


def normalized(text):
    return " ".join(str(text).casefold().split())


def rewrite(record):
    rr = record["requested_rewrite"]
    return rr[0] if isinstance(rr, list) else rr


def association(rr):
    return tuple(normalized(rr[k]) for k in ("subject", "relation_id"))


def record_texts(record):
    rr = rewrite(record)
    prompt = str(rr["prompt"]).format(rr["subject"]).strip()
    answer = rr["target_true"]
    answer = str(answer["str"] if isinstance(answer, dict) else answer).strip()
    prompts = [prompt] + list(record.get("paraphrase_prompts", [])) + list(record.get("neighborhood_prompts", []))
    return {normalized(t) for p in prompts for t in (p, str(p) + " " + answer)}


def build_final_retention(data, source, evaluation, *, count=256, test_seed=20260910,
                          official_seed=1, unlearn_num=50, retain_num=1000):
    if count < 1:
        raise ValueError("Need a positive final retain count")
    official_f, official_r = sample_official_mcf_records(data, unlearn_num, retain_num,
                                                        official_seed, strict=True)
    source_texts = set(text_fingerprints(source))
    if source_texts & set(text_fingerprints(evaluation)):
        raise ValueError("Separate evaluation text overlaps development inputs")
    official_gen = {normalized(p) for r in official_f for p in r.get("paraphrase_prompts", [])}
    if source_texts & official_gen:
        raise ValueError("Official Gen prompt overlaps development inputs; refuse fitting")
    forbidden_associations = {(normalized(f["subject"]), normalized(f["relation"]))
        for b in (source, evaluation) for f in b["facts"]}
    forbidden_associations.update(association(rewrite(r)) for r in official_f + official_r)
    forbidden = set(text_fingerprints(source)) | set(text_fingerprints(evaluation))
    for r in official_f + official_r:
        forbidden.update(record_texts(r))
    candidates, seen_associations = [], set()
    # SHA ordering is stable across Python RNG versions and independent of results.
    ordered = sorted(data[:len(data)//2], key=lambda r: hashlib.sha256(
        f"{test_seed}:{r['case_id']}".encode()).hexdigest())
    used = set(forbidden)
    facts, examples = [], []
    for record in ordered:
        rr = rewrite(record)
        key = association(rr)
        if key in forbidden_associations or key in seen_associations:
            continue
        prompt = str(rr["prompt"]).format(rr["subject"]).strip()
        obj = rr["target_true"]
        answer = str(obj["str"] if isinstance(obj, dict) else obj).strip()
        # Require a real additional context; do not fabricate a new test after
        # failure or choose paraphrases based on model behavior.
        prompts = [prompt]
        for p in record.get("paraphrase_prompts", []):
            if str(p).strip() and normalized(p) != normalized(prompt):
                prompts.append(str(p).strip())
                break
        if len(prompts) != 2 or not answer:
            continue
        fingerprints = {normalized(t) for p in prompts for t in (p, p + " " + answer)}
        if fingerprints & used:
            continue
        fid = f"final_retain_{record['case_id']}"
        facts.append({"id": fid, "subject": str(rr["subject"]).strip(),
                      "relation": str(rr["relation_id"]).strip(), "object": answer,
                      "role": "retain", "aliases": [], "answer_aliases": []})
        for i, p in enumerate(prompts):
            examples.append({"id": f"{fid}:{i}", "split": "test", "prompt": p,
                "completion": " " + answer,
                "spans": [{"start": 1, "end": 1 + len(answer), "fact_id": fid}]})
        candidates.append(record["case_id"])
        seen_associations.add(key)
        used.update(fingerprints)
        if len(candidates) == count:
            break
    if len(candidates) != count:
        raise ValueError(f"Only {len(candidates)} independent cases available; requested {count}")
    result = {"schema_version": 1, "purpose": "preservation_test", "facts": facts, "examples": examples}
    validate_bundle(result, "preservation_test")
    return result, candidates


def load_protocol(path, *, verify_inputs=True):
    path = Path(path)
    protocol = json.loads(path.read_text())
    if protocol.get("protocol") != PROTOCOL or protocol.get("last_major_adaptation") is not True:
        raise ValueError("Expected the frozen development-preservation protocol")
    if protocol.get("scientific_limits") != {"nll": .05, "kl": .01}:
        raise ValueError("Scientific limits changed")
    # Hash-only verification never tokenizes or scores the final test.
    if verify_inputs:
        for item in protocol["files"].values():
            if sha256_file(item["path"]) != item["sha256"]:
                raise ValueError(f"Frozen protocol input changed: {item['path']}")
    return protocol


def claim_training(protocol_path, output, settings):
    protocol = load_protocol(protocol_path)
    if settings != protocol["experiment"]:
        raise ValueError("Experiment differs from the plan frozen before final evaluation")
    marker = Path(protocol_path).parent / "training_started.json"
    write_new(marker, {"output_dir": str(Path(output).resolve()),
        "protocol_sha256": sha256_file(protocol_path), "experiment": settings,
        "started_utc": datetime.now(timezone.utc).isoformat()})
    return protocol


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", required=True)
    p.add_argument("--parity-audit", required=True)
    p.add_argument("--evaluation-bundle", required=True)
    p.add_argument("--mcf-path", required=True)
    p.add_argument("--protocol-dir", required=True)
    p.add_argument("--final-retain-count", type=int, default=256)
    p.add_argument("--final-retain-seed", type=int, default=20260910)
    p.add_argument("--nll-safety-margin", type=float, default=0.01)
    p.add_argument("--kl-safety-margin", type=float, default=0.002)
    args = p.parse_args(argv)
    run, output = Path(args.source_run).resolve(), Path(args.protocol_dir).resolve()
    audit = check_parity(run, args.parity_audit)  # Must be first; no reclassification on failure.
    TrainConfig(retain_nll_safety_margin=args.nll_safety_margin,
                retain_kl_safety_margin=args.kl_safety_margin).validate()
    plan = {**PLAN, "nll_safety_margin": args.nll_safety_margin,
            "kl_safety_margin": args.kl_safety_margin}
    if output.exists():
        raise FileExistsError("Protocol directory exists: reuse its frozen test, never replace it")
    # One freeze per source experiment even if a different output directory is requested.
    lock = run / "development_protocol_frozen.json"
    if lock.exists():
        raise FileExistsError(f"This source already has a frozen final test: {lock}")
    source, _, _ = load_bundle(run / "training_bundle.json")
    evaluation, _, _ = load_bundle(args.evaluation_bundle, "evaluation")
    final, cases = build_final_retention(json.loads(Path(args.mcf_path).read_text()), source,
        evaluation, count=args.final_retain_count, test_seed=args.final_retain_seed)
    # Do not materialize final prompts until all exclusions and the gate pass.
    write_new(lock, {"protocol_dir": str(output), "final_retain_seed": args.final_retain_seed,
                     "final_retain_count": args.final_retain_count})
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "final_retention.json", final)
    paths = {"source_bundle": run / "training_bundle.json", "cache": run / "head_cache.pt",
             "source_report": run / "training_report.json", "parity_audit": Path(args.parity_audit),
             "head_preparation": run / "head_preparation.json",
             "evaluation_bundle": Path(args.evaluation_bundle), "mcf": Path(args.mcf_path),
             "final_retention": output / "final_retention.json"}
    protocol = {"protocol": PROTOCOL, "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "last_major_adaptation": True, "source_run": str(run),
        "files": {k: {"path": str(v.resolve()), "sha256": sha256_file(v)} for k, v in paths.items()},
        "experiment": plan, "scientific_limits": {"nll": .05, "kl": .01},
        "selection_rule": "among development-valid target-met candidates choose minimum delta norm; otherwise minimum worst then mean training probability",
        "official_evaluation": {"seed": 1, "unlearn_num": 50, "retain_num": 1000},
        "dataset_roles": {"train": "original forget prompts and retention anchors, including existing training context augmentation",
            "development": "all former validation retain/language spans, including both mixed completion views; no held-out claim",
            "excluded": "former validation forget/abstention supervision",
            "final": "frozen new retain set, separate evaluation bundle, official MCF/Gen; never fitting or selection"},
        "final_retention": {"case_ids": cases, "seed": args.final_retain_seed,
            "facts": len(final["facts"]), "prompts": len(final["examples"]),
            "sampling": "hash-order first-half MCF, disjoint subject-relation pairs and texts from development, separate eval, and official sampled cases",
            "scope": "independent factual preservation; language preservation assessed separately in the evaluation bundle"},
        "parity_gate": audit["training_only_verification"]["max_abs_errors"],
        "report_disclosure": "The initial held-out retention audit identified development-context failures. We subsequently incorporated those contexts into the development preservation set and evaluated final retention on a separately held-out set not used for localization, optimization, checkpoint selection, or hyperparameter tuning."}
    write_new(output / "protocol.json", protocol)
    print(json.dumps({"phase": "final_test_frozen", "protocol": str(output / "protocol.json"),
                      "facts": len(cases), "prompts": len(final["examples"]),
                      "model_evaluations": 0}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
