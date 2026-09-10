"""Static Slot-1/Slot-2 editing in the nullspace of protected gradients.

Synthetic same-subject/different-relation rows are locality queries.  They do
not assert an invented answer: their complete next-token distribution is
distilled from the immutable base model at a fixed neutral continuation.
"""
import json
import math
import time

import torch

from run_static_overlap_endpoint_ga import baseline_rows, compact_gate
from run_static_overlap_mlp_pilot import development_gate, emit, measure_pilot
from static_overlap_core import answer_nll, flat_gradient, flat_parameters, model_logits, set_parameters
from static_overlap_data import Example, _encode
from freeze_static_overlap_development import normalized
from static_overlap_paired_ga_gd import PairSampler, objective, weighted_examples
from static_overlap_orthogonal_protocol import METHOD, locality_prompt_specs
from static_overlap_training import training_protection


def build_same_subject_locality(source, data, tokenizer, plan):
    """Create relation negatives without claiming unobserved factual answers."""
    rows, audit = [], {}
    continuation = " unknown"
    for spec in locality_prompt_specs(source, data, plan):
        fact_id, relation, prompt = (spec["forget_fact_id"], spec["locality_relation"],
                                     spec["prompt"])
        audit.setdefault(fact_id, [])
        text = prompt + continuation
        ids, offsets = _encode(tokenizer, text, plan["max_length"])
        start = len(prompt) + 1
        positions = [i for i, (a, b) in enumerate(offsets)
                     if b > start and a < len(text) and b > a]
        if not positions or 0 in positions:
            raise ValueError("Synthetic locality continuation has no predicted token")
        labels = [token if i in positions else -100 for i, token in enumerate(ids)]
        eid = f"orthogonal_same_subject_{fact_id}_{relation}"
        rows.append(Example(eid, "train", "language", None, ids, labels,
                            prompt, continuation,
                            "synthetic_same_subject_different_relation_locality"))
        audit[fact_id].append({
            "id": eid,
            "subject": spec["subject"],
            "forget_relation": spec["forget_relation"],
            "locality_relation": relation,
            "prompt": prompt,
            "verified_answer": False,
            "supervision": "immutable_base_next_token_distribution",
        })
    return rows, audit


def orthonormal_rows(vectors, max_rank, relative_tolerance=1e-5):
    """Stable modified Gram-Schmidt; returned rows are orthonormal."""
    basis = []
    for vector in vectors:
        value = vector.detach().float().clone()
        original = value.norm()
        if not torch.isfinite(original) or float(original) == 0.:
            continue
        # Re-orthogonalize once to control FP32 loss of orthogonality.
        for _ in range(2):
            if basis:
                q = torch.stack(basis)
                value -= q.T @ (q @ value)
        norm = value.norm()
        if float(norm) > relative_tolerance * float(original):
            basis.append(value / norm)
        if len(basis) >= max_rank:
            break
    if not basis:
        return vectors[0].new_zeros((0, vectors[0].numel()), dtype=torch.float32)
    return torch.stack(basis)


def project_away(vector, basis):
    value = vector.float()
    if basis.numel():
        value = value - basis.T @ (basis @ value)
        # A second pass removes numerical residual from a large basis.
        value = value - basis.T @ (basis @ value)
    return value.to(vector.dtype)


def preservation_gradient(editor, example):
    editor.model.zero_grad(set_to_none=True)
    return flat_gradient(answer_nll(model_logits(editor.model, example), example), editor.parameters).float()


def choose_basis_examples(examples, synthetic_ids, hard_ids, limit):
    by_id = {example.id: example for example in examples if example.split == "train"
             and example.role in ("retain", "language")}
    chosen = []
    hard_limit = min(len(hard_ids), limit // 2)
    for eid in hard_ids[:hard_limit]:
        if eid in by_id and by_id[eid] not in chosen:
            chosen.append(by_id[eid])
            if len(chosen) == limit:
                return chosen
    # First cover every forgotten subject once, then use the remaining
    # synthetic relation negatives and real retain/language anchors.
    first_by_subject = {}
    for eid in synthetic_ids:
        prefix = eid.rsplit("_", 1)[0]
        first_by_subject.setdefault(prefix, eid)
    synthetic = set(synthetic_ids)
    real_retain = sorted(eid for eid, example in by_id.items()
                         if eid not in synthetic and example.role == "retain")
    real_language = sorted(eid for eid, example in by_id.items()
                           if eid not in synthetic and example.role == "language")
    order = list(first_by_subject.values()) + real_retain + real_language + list(synthetic_ids)
    for eid in order:
        if eid in by_id and by_id[eid] not in chosen:
            chosen.append(by_id[eid])
            if len(chosen) == limit:
                break
    return chosen


def build_protected_basis(editor, examples, synthetic_ids, hard_ids, plan):
    selected = choose_basis_examples(
        examples, synthetic_ids, hard_ids, plan["protected_basis_rank"]
    )
    vectors = [preservation_gradient(editor, example) for example in selected]
    if not vectors:
        raise ValueError("No fitting preservation gradients for protected basis")
    basis = orthonormal_rows(vectors, plan["protected_basis_rank"],
                             plan["protected_basis_relative_tolerance"])
    if not len(basis):
        raise ValueError("Protected gradients have zero numerical rank")
    return basis, {"requested_examples": len(selected), "numerical_rank": len(basis),
                   "example_ids": [example.id for example in selected],
                   "orthonormality_max_abs_error": float(
                       (basis @ basis.T - torch.eye(len(basis), device=basis.device)).abs().max()),
                   "finite": bool(torch.isfinite(basis).all())}


def orthogonal_step(editor, pairs, background, references, config, plan, basis):
    fs, rs = weighted_examples(pairs, background, plan)
    before = flat_parameters(editor.parameters).detach().clone()
    editor.model.zero_grad(set_to_none=True)
    initial, forget_gradient = objective(
        editor, fs, rs, references, config, plan, backward=True
    )
    retain_gradient = torch.cat([
        (parameter.grad.detach() if parameter.grad is not None else torch.zeros_like(parameter)).flatten()
        for parameter in editor.parameters
    ]) - forget_gradient
    projected = project_away(forget_gradient, basis)
    ratio = float(projected.norm() / forget_gradient.norm().clamp_min(1e-30))
    max_overlap = float((basis @ projected.float()).abs().max()) if basis.numel() else 0.
    # Direct descent avoids Adam's coordinate-wise rescaling, which would
    # rotate a projected gradient back into the protected span.
    direction = -(projected + retain_gradient)
    norm = direction.norm()
    if (not torch.isfinite(norm) or float(norm) == 0.
            or ratio < plan["min_forget_residual_ratio"]):
        return {"mode": "orthogonal_paired_ga_gd", "accepted": False,
                "backtracks": None, "before": initial, "after": initial,
                "forget_gradient_norm": float(forget_gradient.norm()),
                "projected_forget_gradient_norm": float(projected.norm()),
                "forget_residual_ratio": ratio, "post_projection_max_abs_overlap": max_overlap,
                "retain_gradient_norm": float(retain_gradient.norm()), "step_norm": 0.}
    proposal = direction * (plan["step_radius"] / norm)
    accepted, backtracks, actual = False, None, initial
    try:
        for index in range(plan["backtracks"] + 1):
            with torch.no_grad():
                set_parameters(editor.parameters, before + proposal * (.5 ** index))
                candidate, _ = objective(editor, fs, rs, references, config, plan)
            forget_ok = (candidate["forget_gap"] < initial["forget_gap"] - 1e-6
                         if initial["forget_gap"] > 0 else candidate["forget_gap"] == 0)
            if (all(math.isfinite(value) for value in candidate.values()) and forget_ok
                    and candidate["loss"] < initial["loss"] - 1e-8
                    and candidate["retention_violation"] <= 0):
                accepted, backtracks, actual = True, index, candidate
                break
    finally:
        if not accepted:
            with torch.no_grad():
                set_parameters(editor.parameters, before)
    return {"mode": "orthogonal_paired_ga_gd", "accepted": accepted,
            "backtracks": backtracks, "before": initial, "after": actual,
            "forget_gradient_norm": float(forget_gradient.norm()),
            "projected_forget_gradient_norm": float(projected.norm()),
            "forget_residual_ratio": ratio, "post_projection_max_abs_overlap": max_overlap,
            "retain_gradient_norm": float(retain_gradient.norm()),
            "step_norm": float((flat_parameters(editor.parameters).detach() - before).norm())}


def _ordered_union(existing, additions):
    seen, result = set(), []
    for value in list(existing) + list(additions):
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def fit(editor, examples, references, config, plan, output, *, source, data, tokenizer):
    synthetic, locality_audit = build_same_subject_locality(source, data, tokenizer, plan)
    examples.extend(synthetic)
    synthetic_fingerprints = {normalized(example.prompt) for example in synthetic}
    synthetic_fingerprints |= {
        normalized(example.prompt + example.completion) for example in synthetic
    }
    data["training_text_fingerprints"] = sorted(
        set(data["training_text_fingerprints"]) | synthetic_fingerprints
    )
    # These rows were added after the generic base-reference pass. Cache their
    # immutable references before constructing the protected Jacobian basis.
    references.build(editor.model, synthetic)
    (output / "same_subject_locality_manifest.json").write_text(
        json.dumps({"method": METHOD, "rows": locality_audit,
                    "verified_fact_count": 0,
                    "synthetic_query_count": len(synthetic),
                    "invented_answers_used": False}, indent=2) + "\n"
    )
    sampler = PairSampler(examples, source["facts"] + data.get("facts", []), plan["seed"])
    pairing = sampler.manifest()
    pairing["synthetic_same_subject_locality_queries"] = len(synthetic)
    pairing["same_subject_supervision"] = "base_distribution_distillation_without_answer_claim"
    (output / "pair_manifest.json").write_text(json.dumps(pairing, indent=2) + "\n")
    emit(phase="association_pairs_ready", forget_facts=len(sampler.order),
         coverage=pairing["coverage_forget_facts"],
         synthetic_same_subject_locality_queries=len(synthetic),
         exact_mixed_pairs=len(pairing["exact_mixed_companions"]))
    initial = baseline_rows(examples, references)
    baseline = development_gate(initial, config)
    (output / "baseline_metrics.json").write_text(
        json.dumps({"gate": baseline, "rows": initial}, indent=2) + "\n"
    )
    emit(phase="orthogonal_baseline", **compact_gate(baseline))
    started = time.monotonic()
    synthetic_ids = [example.id for example in synthetic]
    hard_f, hard_r, seen_f, seen_r = [], [], set(), set()
    basis, basis_report = build_protected_basis(editor, examples, synthetic_ids, hard_r, plan)
    basis_history = [{"step": 0, **basis_report}]
    emit(phase="protected_gradient_basis", step=0, **{k: v for k, v in basis_report.items()
                                                      if k != "example_ids"})
    safe_delta = flat_parameters(editor.parameters).detach().clone()
    safe_rows, safe_step = initial, 0
    last_rows, last_gate = initial, baseline
    history, gates = [], []
    rejected = stale = failed_preservation_gates = 0
    selected, stop, step = None, "step_budget", 0
    previous_mean = sum(row["nll"] for row in initial
                        if row["split"] == "train" and row["role"] == "forget") / sum(
        row["split"] == "train" and row["role"] == "forget" for row in initial)

    def check_gate():
        nonlocal basis, safe_delta, safe_rows, safe_step, last_rows, last_gate
        nonlocal hard_f, hard_r, selected, stop, previous_mean, stale, failed_preservation_gates
        emit(phase="orthogonal_development_gate_start", step=step, examples=len(examples))
        observed = measure_pilot(editor.model, examples, references)
        gate = development_gate(observed, config)
        train = [row for row in observed if row["split"] == "train"]
        retention_ok = training_protection(train, config, internal=True)[1]["retention_passed"]
        new_hard_f = [row["id"] for row in sorted(
            (row for row in train if row["role"] == "forget"), key=lambda row: row["nll"])]
        new_hard_r = [row["id"] for row in sorted(
            (row for row in train if row["role"] != "forget"),
            key=lambda row: max(row["nll_increase"] / config.training_nll_budget,
                                row["kl"] / config.training_kl_budget), reverse=True)
            if row["nll_increase"] > 0 or row["kl"] > 0][:plan["background_retain_batch"]]
        # Current worst cases lead the active set while previously discovered
        # anchors remain registered behind them.
        hard_f = _ordered_union(new_hard_f, hard_f)
        hard_r = _ordered_union(new_hard_r, hard_r)
        if retention_ok:
            safe_delta, safe_rows, safe_step = flat_parameters(editor.parameters).detach().clone(), observed, step
            last_rows, last_gate = observed, gate
        else:
            failed_preservation_gates += 1
            torch.save(editor.artifact(), output / "last_rejected_block_delta.pt")
            with torch.no_grad():
                set_parameters(editor.parameters, safe_delta)
            last_rows, last_gate = safe_rows, development_gate(safe_rows, config)
        basis, report = build_protected_basis(editor, examples, synthetic_ids, hard_r, plan)
        basis_history.append({"step": step, **report})
        emit(phase="protected_gradient_basis", step=step,
             **{key: value for key, value in report.items() if key != "example_ids"})
        mean = sum(row["nll"] for row in last_rows
                   if row["split"] == "train" and row["role"] == "forget") / sum(
            row["split"] == "train" and row["role"] == "forget" for row in last_rows)
        gain = mean - previous_mean
        stale = (stale + 1 if gain < plan["min_gate_nll_gain"] else 0) if retention_ok else 0
        previous_mean = mean
        record = {"step": step, "gate": gate, "training_preservation_passed": retention_ok,
                  "rolled_back": not retention_ok, "retained_state_step": safe_step,
                  "failed_preservation_gates": failed_preservation_gates,
                  "cumulative_hard_retain_anchors": len(hard_r),
                  "training_mean_nll_gain_since_gate": gain}
        gates.append(record)
        (output / f"gate_{step}_metrics.json").write_text(json.dumps(observed) + "\n")
        (output / "last_metrics.json").write_text(json.dumps(last_rows) + "\n")
        torch.save(editor.artifact(), output / "last_endpoint_delta.pt")
        emit(phase="orthogonal_development_gate",
             **{key: value for key, value in record.items() if key != "gate"},
             **compact_gate(gate))
        if retention_ok and gate["passed"]:
            selected, stop = step, "development_gate_passed"
            torch.save(editor.artifact(), output / "training_factors.pt")

    def report():
        return {"method": METHOD, "exploratory": True, "stop_reason": stop,
                "selected_step": selected, "last_gate": last_gate,
                "last_state_step": safe_step, "baseline_gate": baseline,
                "history": history, "gates": gates, "basis_history": basis_history,
                "elapsed_seconds": time.monotonic() - started,
                "pair_manifest": str(output / "pair_manifest.json"),
                "same_subject_locality_manifest": str(output / "same_subject_locality_manifest.json"),
                "fitting_forget_seen": len(seen_f), "fitting_preservation_seen": len(seen_r),
                "development_used_for_gradients": False, "native_checkpoint_created": False,
                "final_tests_touched": False,
                "projection_uses_ranked_individual_protected_gradients_not_mean_retain_gradient": True,
                "projected_step_uses_adam": False,
                "hard_retain_set_is_cumulative": True}

    for step in range(1, plan["steps"] + 1):
        if time.monotonic() - started >= plan["max_training_seconds"]:
            step -= 1
            stop = "wall_time_budget"
            break
        pairs, background = sampler.batch(plan, hard_f, hard_r)
        result = orthogonal_step(editor, pairs, background, references, config, plan, basis)
        result.update(step=step, elapsed_seconds=time.monotonic() - started)
        history.append(result)
        seen_f.update(forget.id for forget, _ in pairs)
        seen_r.update(retain.id for _, retains in pairs for retain in retains)
        seen_r.update(example.id for example in background)
        rejected = 0 if result["accepted"] else rejected + 1
        emit(phase="orthogonal_step", **result)
        with (output / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        if step % plan["check_every"] == 0 or step == plan["steps"]:
            check_gate()
            (output / "training_report.json").write_text(json.dumps(report(), indent=2) + "\n")
            if selected is not None:
                break
            if stale >= plan["stalled_gates"]:
                stop = "insufficient_safe_forgetting_progress"
                break
            if failed_preservation_gates >= plan["max_failed_preservation_gates"]:
                stop = "protected_nullspace_could_not_satisfy_full_training_gate"
                break
        if rejected >= plan["max_stalled_steps"]:
            stop = "consecutive_rejected_orthogonal_steps"
            break
    if not gates or gates[-1]["step"] != step:
        check_gate()
    return report()
