"""Association-paired, answer-masked GA/GD with one joint optimizer step."""
from collections import defaultdict
from copy import deepcopy
import json
import math
import random
import time

import torch

from run_static_overlap_endpoint_ga import baseline_rows, compact_gate
from run_static_overlap_mlp_pilot import development_gate, emit, measure_pilot
from static_overlap_core import answer_nll, model_logits, flat_parameters, set_parameters
from static_overlap_data import overlap_kind
from static_overlap_endpoint_ga import assert_fitting, retain_values
from static_overlap_mlp_protocol import write_new
from static_overlap_paired_protocol import METHOD
from static_overlap_training import forget_target, training_protection

KINDS = ("same_subject_other_relation", "same_subject_same_answer_other_relation",
         "same_relation_other_subject", "same_answer_other_association", "general_retain")


class PairSampler:
    """Balance forget facts, rotate their questions and verified companion facts.

    Original mixed completions keep their exact retain span in every sampled
    pair. Authored question-family names are NOT treated as pair identifiers.
    Only fitting examples enter this index; development is never a fallback.
    """
    def __init__(self, examples, facts, seed):
        self.facts = {}
        for f in facts:
            old = self.facts.setdefault(f["id"], f)
            if any(old[k] != f[k] for k in ("subject", "relation", "object", "role")):
                raise ValueError("Conflicting pair fact provenance")
        self.forget, self.retain = defaultdict(list), defaultdict(list)
        self.background, self.language, self.by_id = [], [], {}
        contexts = defaultdict(list)
        for e in examples:
            if e.split != "train":
                continue
            if e.id in self.by_id:
                raise ValueError("Duplicate fitting example ID")
            self.by_id[e.id] = e
            if e.role == "language":
                self.language.append(e)
                continue
            f = self.facts.get(e.fact_id)
            if f is None or f["role"] != e.role or e.role not in ("forget", "retain"):
                raise ValueError(f"Invalid pair association: {e.id}")
            (self.forget if e.role == "forget" else self.retain)[e.fact_id].append(e)
            if e.role == "retain":
                contexts[(e.group, tuple(e.input_ids))].append(e)
                self.background.append(e)
        if not self.forget or not self.retain or not self.language:
            raise ValueError("Pairs need fitting forget, retain and language examples")
        self.controls, self.exact = {}, {}
        for fid, views in self.forget.items():
            f = self.facts[fid]
            groups = {k: [] for k in KINDS}
            for rid in sorted(self.retain):
                r = self.facts[rid]
                if all(f[k].strip().casefold() == r[k].strip().casefold() for k in ("subject", "relation")):
                    raise ValueError("The same association cannot be both forgotten and retained")
                groups[overlap_kind(f, r)].append(rid)
            if not groups["same_relation_other_subject"]:
                raise ValueError(f"No fitting same-relation companion for {fid}")
            self.controls[fid] = groups
            for e in views:
                companions = contexts[(e.group, tuple(e.input_ids))]
                for r in companions:
                    if any(a != -100 and b != -100 for a, b in zip(e.labels, r.labels)):
                        raise ValueError("Forget and retain answer masks intersect")
                self.exact[e.id] = [r.id for r in companions]
        self.order = sorted(self.forget)
        random.Random(seed).shuffle(self.order)
        self.cursor = 0
        self.views, self.control_views, self.control_facts = defaultdict(int), defaultdict(int), defaultdict(int)
        self.background_cursor = self.language_cursor = 0

    def manifest(self):
        return {"method": METHOD, "training_only": True, "inference_router": False,
            "forget_facts": self.order, "controls_by_forget_fact": self.controls,
            "forget_question_ids": {f: [e.id for e in es] for f, es in self.forget.items()},
            "retain_question_ids": {f: [e.id for e in es] for f, es in self.retain.items()},
            "exact_mixed_companions": {f: rs for f, rs in self.exact.items() if rs},
            "coverage_forget_facts": {k: sum(bool(c[k]) for c in self.controls.values()) for k in KINDS},
            "missing_control_kinds": {f: [k for k in KINDS if not c[k]] for f, c in self.controls.items()}}

    def batch(self, plan, hard_f=(), hard_r=()):
        count = min(plan["pair_batch"], len(self.order))
        chosen = {}
        for eid in hard_f:
            e = self.by_id[eid]
            if e.role != "forget":
                raise ValueError("Invalid hard forget example")
            if len(chosen) >= count // 2:
                break
            chosen.setdefault(e.fact_id, e)
        while len(chosen) < count:
            fid = self.order[self.cursor % len(self.order)]
            self.cursor += 1
            if fid in chosen:
                continue
            views = self.forget[fid]
            chosen[fid] = views[self.views[fid] % len(views)]
            self.views[fid] += 1
        pairs = []
        for fid, f in chosen.items():
            rs = {eid: self.by_id[eid] for eid in self.exact[f.id]}
            for kind, controls in self.controls[fid].items():
                if not controls:
                    continue
                key = fid, kind
                rid = controls[self.control_facts[key] % len(controls)]
                self.control_facts[key] += 1
                views = self.retain[rid]
                r = views[self.control_views[rid] % len(views)]
                self.control_views[rid] += 1
                rs[r.id] = r
            pairs.append((f, list(rs.values())))
        background = {}
        # Once the full gate discovers hard anchors, keep the entire registered
        # active-set slice in every subsequent proposal check.
        for eid in hard_r[:plan["background_retain_batch"]]:
            e = self.by_id[eid]
            if e.role not in ("retain", "language"):
                raise ValueError("Invalid hard preservation example")
            background[eid] = e
        while len(background) < min(plan["background_retain_batch"], len(self.background)):
            e = self.background[self.background_cursor % len(self.background)]
            self.background_cursor += 1
            background[e.id] = e
        for _ in range(min(plan["language_batch"], len(self.language))):
            e = self.language[self.language_cursor % len(self.language)]
            self.language_cursor += 1
            background[e.id] = e
        return pairs, list(background.values())


def weighted_examples(pairs, background, plan):
    """Average retain loss within each pair, then pairs; reuse duplicate forwards."""
    if not pairs or any(not rs for _, rs in pairs):
        raise ValueError("Every forget question needs retain companions")
    fs, rs = {}, {}
    for f, companions in pairs:
        assert_fitting([f], companions)
        if f.id in fs:
            raise ValueError("Duplicate forget question in pair batch")
        fs[f.id] = (f, 1. / len(pairs))
        companions = {r.id: r for r in companions}
        for r in companions.values():
            weight = 1. / (len(pairs) * len(companions))
            rs[r.id] = (r, rs.get(r.id, (None, 0.))[1] + weight)
    background = {r.id: r for r in background}
    if background:
        assert_fitting([pairs[0][0]], list(background.values()))
        for r in background.values():
            rs[r.id] = (r, rs.get(r.id, (None, 0.))[1] + plan["background_weight"] / len(background))
    return list(fs.values()), list(rs.values())


def objective(editor, fs, rs, references, config, plan, *, backward=False):
    gap, retain_change, retain_penalty, kl_total, violation = 0., 0., 0., 0., 0.
    for e, weight in fs:
        nll = answer_nll(model_logits(editor.model, e), e)
        loss = weight * torch.relu(nll.new_tensor(forget_target(references.nll[e.id], config)) - nll)
        if backward:
            loss.backward()
        gap += float(loss.detach())
    fg = gradient_vector(editor.parameters) if backward else None
    for e, weight in rs:
        increase, kl = retain_values(editor.model, e, references)
        # Retain GD is activated only when the edited answer NLL is worse than
        # its base value. This preserves the association without rewarding an
        # unlimited decrease that can cancel the forget-ascent direction.
        penalty = torch.relu(increase)
        if backward:
            (weight * (plan["retain_weight"] * penalty + plan["kl_weight"] * kl)).backward()
        ni, ki = float(increase.detach()), float(kl.detach())
        if not math.isfinite(ni) or not math.isfinite(ki):
            raise ValueError("Nonfinite paired preservation metric")
        retain_change += weight * ni
        retain_penalty += weight * max(0., ni)
        kl_total += weight * ki
        violation = max(violation, ni / config.training_nll_budget - 1., ki / config.training_kl_budget - 1.)
    return {"loss": gap + plan["retain_weight"] * retain_penalty + plan["kl_weight"] * kl_total,
            "forget_gap": gap, "retain_nll_change": retain_change, "retain_penalty": retain_penalty,
            "retain_kl": kl_total,
            "retention_violation": violation}, fg


def gradient_vector(parameters):
    return torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).flatten() for p in parameters])


def set_gradient(parameters, vector):
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        parameter.grad = vector[offset:offset+size].view_as(parameter).clone()
        offset += size
    if offset != vector.numel():
        raise ValueError("Gradient vector shape mismatch")


def paired_step(editor, optimizer, pairs, background, references, config, plan):
    fs, rs = weighted_examples(pairs, background, plan)
    parameters = editor.parameters
    before, state = flat_parameters(parameters).detach().clone(), deepcopy(optimizer.state_dict())
    optimizer.zero_grad(set_to_none=True)
    initial, fg = objective(editor, fs, rs, references, config, plan, backward=True)
    rg = gradient_vector(parameters) - fg
    dot, r2 = torch.sum(fg * rg), torch.sum(rg * rg)
    cosine = float(dot / (fg.norm() * rg.norm()).clamp_min(1e-30))
    projected = fg - torch.minimum(dot, dot.new_zeros(())) / r2.clamp_min(1e-30) * rg
    combined = projected + rg
    set_gradient(parameters, combined)
    torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    proposal = flat_parameters(parameters).detach() - before
    proposal *= min(1., plan["step_radius"] / max(float(proposal.norm()), 1e-30))
    accepted, backtracks, actual = False, None, initial
    try:
        for i in range(plan["backtracks"] + 1):
            with torch.no_grad():
                set_parameters(parameters, before + proposal * .5**i)
                candidate, _ = objective(editor, fs, rs, references, config, plan)
            forget_ok = (candidate["forget_gap"] < initial["forget_gap"] - 1e-6
                         if initial["forget_gap"] > 0 else candidate["forget_gap"] == 0)
            if (all(math.isfinite(v) for v in candidate.values()) and forget_ok
                    and candidate["loss"] < initial["loss"] - 1e-8 and candidate["retention_violation"] <= 0):
                accepted, backtracks, actual = True, i, candidate
                break
    finally:
        if not accepted:
            with torch.no_grad():
                set_parameters(parameters, before)
            optimizer.load_state_dict(state)
    return {"mode": "paired_ga_gd", "accepted": accepted, "backtracks": backtracks,
        "pairs": [{"forget": f.id, "retain": [r.id for r in rs]} for f, rs in pairs],
        "background_ids": [e.id for e in background], "before": initial, "after": actual,
        "forget_gradient_norm": float(fg.norm()), "retain_gradient_norm": float(rg.norm()),
        "forget_retain_gradient_cosine": cosine, "step_norm": float((flat_parameters(parameters).detach()-before).norm())}


def fit(editor, examples, references, config, plan, output, *, source, data):
    sampler = PairSampler(examples, source["facts"] + data.get("facts", []), plan["seed"])
    pairing = sampler.manifest()
    write_new(output / "pair_manifest.json", pairing)
    emit(phase="association_pairs_ready", forget_facts=len(sampler.order),
         coverage=pairing["coverage_forget_facts"], exact_mixed_pairs=len(pairing["exact_mixed_companions"]))
    initial = baseline_rows(examples, references)
    baseline = development_gate(initial, config)
    write_new(output / "baseline_metrics.json", {"gate": baseline, "rows": initial})
    emit(phase="paired_baseline", **compact_gate(baseline))
    optimizer = torch.optim.Adam(editor.parameters, lr=plan["learning_rate"])
    started = time.monotonic()
    history, gates, hard_f, hard_r = [], [], [], []
    seen_f, seen_r = set(), set()
    safe_delta = flat_parameters(editor.parameters).detach().clone()
    safe_rows, safe_step = initial, 0
    last_rows, last_gate = initial, baseline
    rejected = stale = failed_preservation_gates = step = 0
    selected, stop = None, "step_budget"
    previous_mean = sum(r["nll"] for r in initial if r["split"] == "train" and r["role"] == "forget") / sum(
        r["split"] == "train" and r["role"] == "forget" for r in initial)

    def check_gate():
        nonlocal safe_delta, safe_rows, safe_step, last_rows, last_gate, hard_f, hard_r
        nonlocal selected, stop, previous_mean, stale, failed_preservation_gates
        emit(phase="paired_development_gate_start", step=step, examples=len(examples))
        observed = measure_pilot(editor.model, examples, references)
        gate = development_gate(observed, config)
        train = [r for r in observed if r["split"] == "train"]
        retention_ok = training_protection(train, config, internal=True)[1]["retention_passed"]
        # Replay and rollback use training preservation only, never development
        # failures. Full gates are still required for native checkpoint export.
        hard_f = [r["id"] for r in sorted((r for r in train if r["role"] == "forget"), key=lambda r: r["nll"])]
        hard_r = [r["id"] for r in sorted((r for r in train if r["role"] != "forget"),
            key=lambda r: max(r["nll_increase"]/config.training_nll_budget, r["kl"]/config.training_kl_budget), reverse=True)]
        if retention_ok:
            safe_delta, safe_rows, safe_step = flat_parameters(editor.parameters).detach().clone(), observed, step
            last_rows, last_gate = observed, gate
        else:
            failed_preservation_gates += 1
            torch.save(editor.artifact(), output / "last_rejected_block_delta.pt")
            with torch.no_grad():
                set_parameters(editor.parameters, safe_delta)
            optimizer.state.clear()
            last_rows, last_gate = safe_rows, development_gate(safe_rows, config)
        mean = sum(r["nll"] for r in last_rows if r["split"] == "train" and r["role"] == "forget") / sum(
            r["split"] == "train" and r["role"] == "forget" for r in last_rows)
        gain = mean - previous_mean
        # A rollback is new constraint discovery, not evidence that the
        # retention-safe direction has stalled. Retry from the safe state with
        # the discovered anchors present in every minibatch.
        stale = (stale + 1 if gain < plan["min_gate_nll_gain"] else 0) if retention_ok else 0
        previous_mean = mean
        record = {"step": step, "gate": gate, "training_preservation_passed": retention_ok,
                  "rolled_back": not retention_ok, "retained_state_step": safe_step,
                  "failed_preservation_gates": failed_preservation_gates,
                  "training_mean_nll_gain_since_gate": gain}
        gates.append(record)
        (output / f"gate_{step}_metrics.json").write_text(json.dumps(observed, allow_nan=False)+"\n")
        (output / "last_metrics.json").write_text(json.dumps(last_rows, allow_nan=False)+"\n")
        torch.save(editor.artifact(), output / "last_endpoint_delta.pt")
        emit(phase="paired_development_gate", **{k: v for k, v in record.items() if k != "gate"}, **compact_gate(gate))
        if retention_ok and gate["passed"]:
            selected, stop = step, "development_gate_passed"
            torch.save(editor.artifact(), output / "training_factors.pt")

    def report():
        return {"method": METHOD, "exploratory": True, "stop_reason": stop, "selected_step": selected,
            "last_gate": last_gate, "last_state_step": safe_step, "baseline_gate": baseline,
            "history": history, "gates": gates, "elapsed_seconds": time.monotonic()-started,
            "pair_manifest": str(output / "pair_manifest.json"),
            "fitting_forget_seen": len(seen_f), "fitting_preservation_seen": len(seen_r),
            "development_used_for_gradients": False, "native_checkpoint_created": False,
            "final_tests_touched": False, "minibatch_checks_are_not_full_retention_guarantees": True}

    for step in range(1, plan["steps"] + 1):
        if time.monotonic() - started >= plan["max_training_seconds"]:
            step -= 1
            stop = "wall_time_budget"
            break
        pairs, background = sampler.batch(plan, hard_f, hard_r)
        result = paired_step(editor, optimizer, pairs, background, references, config, plan)
        result.update(step=step, elapsed_seconds=time.monotonic()-started)
        history.append(result)
        seen_f.update(f.id for f, _ in pairs)
        seen_r.update(r.id for _, rs in pairs for r in rs)
        seen_r.update(e.id for e in background)
        rejected = 0 if result["accepted"] else rejected + 1
        emit(phase="paired_step", **{k: v for k, v in result.items() if k not in ("pairs", "background_ids")})
        with (output / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(result, allow_nan=False)+"\n")
        if step % plan["check_every"] == 0 or step == plan["steps"]:
            check_gate()
            (output / "training_report.json").write_text(json.dumps(report(), indent=2, allow_nan=False)+"\n")
            if selected is not None:
                break
            if stale >= plan["stalled_gates"]:
                stop = "insufficient_training_forgetting_progress"
                break
            if failed_preservation_gates >= plan["max_failed_preservation_gates"]:
                stop = "active_set_could_not_find_training_safe_direction"
                break
        if rejected >= plan["max_stalled_steps"]:
            stop = "consecutive_rejected_steps"
            break
    if not gates or gates[-1]["step"] != step:
        check_gate()
    return report()
