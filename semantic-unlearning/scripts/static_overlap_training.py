"""Joint bounded GA / retain GD / exact forward KL with finite-anchor protection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile

import torch

from static_overlap_core import (
    answer_nll, constrained_step, flat_gradient, flat_parameters, forward_kl,
    model_logits, selected_logits, set_parameters, tied_weights,
)


# Verification allowance for FP32 merge/reload rounding, never a training budget.
EXPORT_FP32_NUMERIC_SLACK = 5e-6


@dataclass
class TrainConfig:
    steps: int = 200
    batch_size: int = 4
    protected_batch_size: int = 8
    max_constraint_refinements: int = 4
    learning_rate: float = 0.005
    forget_increase: float = 2.0
    lambda_forget: float = 1.0
    lambda_abstain: float = 1.0
    lambda_retain: float = 1.0
    lambda_kl: float = 1.0
    lambda_delta: float = 1e-4
    epsilon: float = 0.005
    step_radius: float = 0.25
    max_step_radius: float = 2.0
    radius_growth: float = 2.0
    target_probability: float = 1e-6
    min_forget_progress: float = 1e-6
    retain_nll_budget: float = 0.05
    retain_kl_budget: float = 0.01
    retain_nll_safety_margin: float = 0.0
    retain_kl_safety_margin: float = 0.0
    hard_example_mix: float = 0.0
    hard_example_cap: float = 4.0
    hard_replay_size: int = 0
    lambda_worst_forget: float = 0.0
    guard_worst_forget: bool = False
    compare_forget_candidates: bool = False
    select_best_valid_checkpoint: bool = False
    fresh_start_only: bool = False
    backtracks: int = 10
    max_stalled_steps: int = 10
    seed: int = 1

    def validate(self):
        for key, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Non-finite/non-numeric config: {key}")
            if key != "seed" and value < 0:
                raise ValueError(f"Negative config: {key}")
        for key in ("steps", "batch_size", "protected_batch_size", "max_stalled_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if any(type(getattr(self, key)) is not int
               for key in ("backtracks", "seed", "max_constraint_refinements", "hard_replay_size")):
            raise ValueError("backtracks, seed, max_constraint_refinements and hard_replay_size must be integers")
        for key in ("learning_rate", "forget_increase", "step_radius", "lambda_forget", "lambda_retain", "lambda_kl"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if not 0 < self.target_probability < 1:
            raise ValueError("target_probability must lie strictly between 0 and 1")
        if self.max_step_radius < self.step_radius or self.radius_growth < 1:
            raise ValueError("max_step_radius/radius_growth cannot shrink the initial radius")
        if not 0 <= self.hard_example_mix <= 1 or self.hard_example_cap < 1:
            raise ValueError("hard_example_mix must be in [0, 1] and hard_example_cap >= 1")
        if (self.retain_nll_safety_margin > self.retain_nll_budget
                or self.retain_kl_safety_margin > self.retain_kl_budget):
            raise ValueError("Internal retention margins cannot exceed nominal budgets")
        for key in ("compare_forget_candidates", "select_best_valid_checkpoint", "fresh_start_only", "guard_worst_forget"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a boolean")

    @property
    def training_nll_budget(self):
        return self.retain_nll_budget - self.retain_nll_safety_margin

    @property
    def training_kl_budget(self):
        return self.retain_kl_budget - self.retain_kl_safety_margin


def forget_target(base_nll, config):
    # A relative +2 NLL target only multiplies residual token probability by
    # exp(-2); it cannot establish near-zero residual knowledge. A token-mean
    # probability ceiling also upper-bounds the full answer probability.
    return max(base_nll + config.forget_increase, -math.log(config.target_probability))


def forgetting_status(rows, config):
    forgotten = [r for r in rows if r["role"] == "forget"]
    if not forgotten:
        raise ValueError("Cannot validate forgetting without forget examples")
    residual = [math.exp(-r["nll"]) for r in forgotten]
    return {"count": len(forgotten), "mean_token_probability": sum(residual) / len(residual),
            "max_token_probability": max(residual),
            "target_probability": config.target_probability,
            "target_met": all(math.isfinite(r["nll"]) and r["nll"] >= forget_target(r["base_nll"], config)
                              for r in forgotten)}


@torch.no_grad()
def measure(editor, examples):
    rows = []
    for example in examples:
        with editor.base():
            base = model_logits(editor.model, example)
        edited = model_logits(editor.model, example)
        base_nll = answer_nll(base, example).item()
        nll = answer_nll(edited, example).item()
        rows.append({"id": example.id, "split": example.split, "role": example.role,
                     "base_nll": base_nll, "nll": nll, "nll_increase": nll - base_nll,
                     "kl": forward_kl(base, edited, example).item()})
    return rows


def within_budgets(rows, config):
    protected = [r for r in rows if r["role"] in ("retain", "language")]
    if not protected:
        raise ValueError("Protection cannot pass with an empty anchor set")
    finite = all(math.isfinite(r[k]) for r in protected for k in ("nll", "base_nll", "nll_increase", "kl"))
    max_nll = max(r["nll_increase"] for r in protected)
    max_kl = max(r["kl"] for r in protected)
    passed = finite and max_nll <= config.retain_nll_budget and max_kl <= config.retain_kl_budget
    return passed, {"max_retained_nll_increase": max_nll, "max_retained_kl": max_kl}


def near_budget_anchor_ids(rows, config):
    """Keep tight constraints in the projection even outside rotating coverage.

    The margin selects gradients; it never changes an acceptance budget.
    """
    return {r["id"] for r in rows if r["role"] in ("retain", "language")
            and any(budget - r[key] <= min(config.epsilon, 0.1 * budget)
                    for key, budget in (("nll_increase", config.training_nll_budget),
                                        ("kl", config.training_kl_budget)))}


def training_protection(rows, config, *, internal=False):
    nominal_passed, observed = within_budgets(rows, config)
    nll_budget = config.training_nll_budget if internal else config.retain_nll_budget
    kl_budget = config.training_kl_budget if internal else config.retain_kl_budget
    passed = (nominal_passed and observed["max_retained_nll_increase"] <= nll_budget
              and observed["max_retained_kl"] <= kl_budget)
    protected = [r for r in rows if r["role"] in ("retain", "language")]
    violations = [r for r in protected
                  if r["nll_increase"] > nll_budget or r["kl"] > kl_budget]
    # Most severe relative violation first; use an absolute scale for a zero
    # budget. Stable ties preserve the bundle order and deterministic fitting.
    violations.sort(key=lambda r: max(
        (r["nll_increase"] - nll_budget) / max(nll_budget, 1e-12),
        (r["kl"] - kl_budget) / max(kl_budget, 1e-12)), reverse=True)
    return passed, {**observed, "retention_passed": passed,
                    "applied_nll_budget": nll_budget, "applied_kl_budget": kl_budget,
                    "nominal_retain_nll_budget": config.retain_nll_budget,
                    "nominal_retain_kl_budget": config.retain_kl_budget,
                    "max_retained_nll_anchor_id": max(protected, key=lambda r: r["nll_increase"])["id"],
                    "max_retained_kl_anchor_id": max(protected, key=lambda r: r["kl"])["id"],
                    "violating_anchor_ids": [r["id"] for r in violations]}


def within_export_budgets(rows, config, model_dtype):
    """Classify export-only FP32 boundary drift, preserving nominal/raw values.

    Training, factor recovery, logit parity, and reduced-precision export checks
    do not receive this allowance. No rounded values participate in decisions.
    """
    nominal_pass, observed = within_budgets(rows, config)
    protected = [r for r in rows if r["role"] in ("retain", "language")]
    finite = all(math.isfinite(r[k]) for r in protected
                 for k in ("nll", "base_nll", "nll_increase", "kl"))
    max_nll, max_kl = observed["max_retained_nll_increase"], observed["max_retained_kl"]
    slack = EXPORT_FP32_NUMERIC_SLACK if model_dtype == torch.float32 else 0.0
    nominal_pass = bool(nominal_pass and finite)
    passed = bool(finite and max_nll <= config.retain_nll_budget + slack
                  and max_kl <= config.retain_kl_budget + slack)
    return passed, {
        **observed,
        "observed_max_retained_nll_increase": max_nll,
        "observed_max_retained_kl": max_kl,
        "nominal_retain_nll_budget": config.retain_nll_budget,
        "nominal_retain_kl_budget": config.retain_kl_budget,
        "numerical_slack": slack,
        "verification_dtype": str(model_dtype),
        "nominal_budgets_passed": nominal_pass,
        "passed_with_numerical_slack": passed and not nominal_pass,
        "classification": ("nominal_pass" if nominal_pass else
                           "numerical_boundary_pass" if passed else "retention_failure"),
        "passed": passed,
    }


def training_forget_loss(rows, config):
    forgotten = [r for r in rows if r["role"] == "forget"]
    if not forgotten or any(not math.isfinite(r["nll"]) for r in forgotten):
        return float("inf")
    return sum(max(0., forget_target(r["base_nll"], config) - r["nll"])
               for r in forgotten) / len(forgotten)


def hard_example_weights(examples, nlls, config):
    """Detached, capped weights from each fact's most remembered fitting view.

    Raw fact weights lie in [1, cap]. Divide each fact's weight among its views;
    normalization in the loss preserves a floor and prevents starvation.
    """
    probabilities, counts = {}, {}
    for e in examples:
        value = nlls[e.id]
        value = value.detach().item() if isinstance(value, torch.Tensor) else float(value)
        if not math.isfinite(value):
            raise ValueError("Cannot weight non-finite forget NLL")
        key = e.fact_id or e.id
        probabilities[key] = max(probabilities.get(key, 0.), math.exp(-value))
        counts[key] = counts.get(key, 0) + 1
    if config.hard_example_mix == 0:
        return {e.id: 1. for e in examples}, probabilities
    mean = sum(probabilities.values()) / len(probabilities)
    raw = {key: 1. + config.hard_example_mix * min(config.hard_example_cap - 1., probability / mean if mean else 0.)
           for key, probability in probabilities.items()}
    return {e.id: raw[e.fact_id or e.id] / counts[e.fact_id or e.id] for e in examples}, probabilities


def weighted_forget_loss(nlls, targets, weights):
    if any(not math.isfinite(nlls[key]) for key in weights):
        return float("inf")
    return sum(weights[key] * max(0., targets[key] - nlls[key]) for key in weights) / sum(weights.values())


def worst_forget_status(nlls, targets):
    if not nlls or set(nlls) != set(targets) or any(not math.isfinite(v) for v in nlls.values()):
        raise ValueError("Worst-target checks require all finite fitting forget NLLs")
    gap_id = max(nlls, key=lambda key: targets[key] - nlls[key])
    probability_id = min(nlls, key=nlls.get)
    return {"max_target_gap": max(0., targets[gap_id] - nlls[gap_id]),
            "min_nll": nlls[probability_id], "max_token_probability": math.exp(-nlls[probability_id]),
            "max_gap_id": gap_id, "max_probability_id": probability_id}


def worst_forget_guard(nlls, targets, before):
    """No per-step slack/ratcheting: neither worst fitting metric may regress."""
    after = worst_forget_status(nlls, targets)
    violations = [key for key in nlls
                  if max(0., targets[key] - nlls[key]) > before["max_target_gap"]
                  or nlls[key] < before["min_nll"]]
    return not violations, {"worst_forget_before": before, "worst_forget_after": after,
                           "worst_forget_progress": before["max_target_gap"] - after["max_target_gap"],
                           "worst_forget_passed": not violations,
                           "worst_forget_violating_ids": violations}


def hard_replay_examples(examples, coverage, nlls, targets, count):
    """Revisit the current worst views while keeping coverage slots intact."""
    if not count:
        return []
    worst = worst_forget_status(nlls, targets)
    by_id = {e.id: e for e in examples}
    critical = [worst["max_gap_id"], worst["max_probability_id"]]
    ranked = critical + sorted(nlls, key=lambda key: targets[key] - nlls[key], reverse=True)
    seen, facts, replay, remaining = {e.id for e in coverage}, set(), [], []
    for key in ranked:
        if key in seen:
            continue
        seen.add(key)
        e = by_id[key]
        fact = e.fact_id or e.id
        # Always include the two global extrema; otherwise favor distinct facts.
        if fact in facts and key not in critical:
            remaining.append(e)
        else:
            replay.append(e)
            facts.add(fact)
    return (replay + remaining)[:count]


@torch.no_grad()
def forget_nlls(editor, examples):
    # Original-base targets are already cached. Candidate scoring needs only
    # edited NLL, not another base forward or full-vocabulary forget KL.
    return {e.id: answer_nll(model_logits(editor.model, e), e).item() for e in examples}


class ValidCheckpointSelection:
    """Worst-first fitting forgetting, gated by training and validation retention."""
    def __init__(self, config, targets):
        self.config, self.targets = config, targets
        self.parameters, self.step, self.score = None, None, None
        self.eligible_steps = 0

    def consider(self, editor, step, nlls, training_rows, validation_rows):
        fitting_pass, _ = training_protection(training_rows, self.config, internal=True)
        validation_pass, protection = training_protection(validation_rows, self.config)
        finite = bool(nlls) and all(math.isfinite(value) for value in nlls.values())
        eligible = fitting_pass and validation_pass and finite
        record = {"validation_retention": protection, "checkpoint_eligible": eligible,
                  "selected_as_best": False}
        if not eligible:
            return record
        self.eligible_steps += 1
        probabilities = [math.exp(-value) for value in nlls.values()]
        score = (max(max(0., self.targets[key] - nlls[key]) for key in self.targets),
                 max(probabilities), sum(probabilities) / len(probabilities))
        record["checkpoint_forget_score"] = list(score)
        if self.score is None or score < self.score:
            self.parameters = flat_parameters(editor.parameters).cpu().clone()
            self.step, self.score = step, score
            record["selected_as_best"] = True
        return record

    def summary(self):
        return {"method": "min_worst_training_target_gap_then_max_probability_then_mean_with_retention",
                "selected_step": self.step, "eligible_steps": self.eligible_steps,
                "selected_score": list(self.score) if self.score is not None else None,
                "score_fields": ["max_training_target_gap", "max_training_token_probability", "mean_training_token_probability"],
                "validation_used_for_gradients": False, "validation_forget_used_for_selection": False,
                "official_evaluation_used_for_selection": False}


def train(editor, examples, config, log_path=None, *, resume=False):
    config.validate()
    if resume and config.fresh_start_only:
        raise ValueError("This experiment requires a fresh start from the original base model")
    fitting = [e for e in examples if e.split == "train"]
    forget = [e for e in fitting if e.role == "forget"]
    retain = [e for e in fitting if e.role == "retain"]
    anchors = [e for e in fitting if e.role in ("retain", "language")]
    abstain = [e for e in fitting if e.role == "abstain"]
    language = [e for e in anchors if e.role == "language"]
    validation_anchors = [e for e in examples if e.split == "validation" and e.role in ("retain", "language")]
    if not forget or not retain or not language or (config.lambda_abstain and not abstain):
        raise ValueError("Missing a required training objective role")
    # Fixed per-example targets, set once against the unmodified model.
    baseline = measure(editor, fitting)
    if not resume and any(abs(row["nll_increase"]) > 1e-7 or row["kl"] > 1e-7 for row in baseline):
        raise ValueError("Training must start from zero effective deltas")
    if not training_protection(baseline, config, internal=True)[0]:
        raise ValueError("Cannot continue from factors outside the original training retention budgets (including safety margins)")
    base_nll = {row["id"]: row["base_nll"] for row in baseline}
    targets = {e.id: forget_target(base_nll[e.id], config) for e in forget}
    current_forget_nlls = {row["id"]: row["nll"] for row in baseline if row["role"] == "forget"}
    global_scoring = (config.compare_forget_candidates or config.hard_example_mix > 0
                      or config.hard_replay_size > 0 or config.guard_worst_forget or config.lambda_worst_forget > 0)
    forget_by_id = {e.id: e for e in forget}
    selection = ValidCheckpointSelection(config, targets) if config.select_best_valid_checkpoint else None
    if selection is not None and not validation_anchors:
        raise ValueError("Best-valid-checkpoint selection requires validation retention anchors")
    anchor_rows = [row for row in baseline if row["role"] in ("retain", "language")]
    anchors_by_id = {e.id: e for e in anchors}
    optimizer = torch.optim.Adam(editor.parameters, lr=config.learning_rate)
    rng, history, stalls = random.Random(config.seed), [], 0
    radius = config.step_radius
    forget_order, forget_cursor, seen_forget = [], 0, set()
    visits = {e.id: 0 for e in forget}
    abstain_by_key = {(e.group, e.fact_id): e for e in abstain}
    order = list(anchors)
    rng.shuffle(order)
    cursor = 0
    target_reached, stalled_out = False, False

    def sample(pool):
        return rng.sample(pool, min(config.batch_size, len(pool)))

    def next_forget_batch(fact_probabilities):
        nonlocal forget_order, forget_cursor
        if forget_cursor >= len(forget_order):
            forget_order = list(forget)
            rng.shuffle(forget_order)
            if config.hard_example_mix > 0:
                forget_order.sort(key=lambda e: -fact_probabilities[e.fact_id or e.id])
            forget_cursor = 0
        batch = forget_order[forget_cursor:forget_cursor + config.batch_size]
        forget_cursor += len(batch)
        seen_forget.update(e.id for e in batch)
        return batch

    for step in range(config.steps):
        # Freeze these weights for every candidate and recheck in this step.
        weights, fact_probabilities = hard_example_weights(forget, current_forget_nlls, config)
        before_global = weighted_forget_loss(current_forget_nlls, targets, weights)
        before_worst = worst_forget_status(current_forget_nlls, targets)
        rotating = [order[(cursor + j) % len(order)]
                    for j in range(min(config.protected_batch_size, len(order)))]
        cursor = (cursor + len(rotating)) % len(order)
        active_ids = near_budget_anchor_ids(anchor_rows, config)
        selected_ids = active_ids | {e.id for e in rotating}
        protected = [e for e in anchors if e.id in selected_ids]
        projected_ids, discovered_ids, projected_forget_ids = set(), [], set()
        protected_gradients, allowances = [], []

        def add_constraint(e):
            nll = answer_nll(model_logits(editor.model, e), e)
            allowances.append(min(config.epsilon, max(0.0, config.training_nll_budget
                                                     - (nll.detach().item() - base_nll[e.id]))) * 0.5)
            protected_gradients.append(flat_gradient(nll, editor.parameters))
            with editor.base(), torch.no_grad():
                base = model_logits(editor.model, e)
            kl = forward_kl(base, model_logits(editor.model, e), e)
            allowances.append(min(config.epsilon, max(0.0, config.training_kl_budget
                                                     - kl.detach().item())) * 0.5)
            protected_gradients.append(flat_gradient(kl, editor.parameters))
            projected_ids.add(e.id)

        def add_forget_constraint(e):
            # h_i = target_i - NLL_i. Linearize h_i <= current worst gap,
            # and NLL_i >= current minimum NLL, at the unchanged step origin.
            nll = answer_nll(model_logits(editor.model, e), e)
            floor = max(targets[e.id] - before_worst["max_target_gap"], before_worst["min_nll"])
            protected_gradients.append(-flat_gradient(nll, editor.parameters))
            allowances.append(max(0., nll.detach().item() - floor) * 0.5)
            projected_forget_ids.add(e.id)

        def constraint_tensors():
            gradients = torch.stack(protected_gradients)
            return gradients, gradients.new_tensor(allowances)

        for e in protected:
            add_constraint(e)
        if config.guard_worst_forget:
            for key in dict.fromkeys((before_worst["max_gap_id"], before_worst["max_probability_id"])):
                add_forget_constraint(forget_by_id[key])
        gradients, initial_allowances = constraint_tensors()

        def refine_constraints(diagnostics):
            missing = [key for key in diagnostics.get("violating_anchor_ids", []) if key not in projected_ids]
            missing_forget = ([key for key in diagnostics.get("worst_forget_violating_ids", [])
                               if key not in projected_forget_ids] if config.guard_worst_forget else [])
            if not missing and not missing_forget:
                return None
            # constrained_step restores the original parameters before invoking
            # us. Never linearize at a rejected trial and apply at another point.
            for key in missing[:config.protected_batch_size]:
                add_constraint(anchors_by_id[key])
                discovered_ids.append(key)
            for key in missing_forget[:config.protected_batch_size]:
                add_forget_constraint(forget_by_id[key])
            return constraint_tensors()

        # Accumulate each micro-example separately to bound activation memory.
        # A detached leaf surrogate delivers this aggregate gradient to Adam.
        aggregate = torch.zeros_like(gradients[0])
        components = {"forget": 0.0, "worst_forget": 0.0, "abstain": 0.0, "retain": 0.0, "kl": 0.0, "delta": 0.0}

        def add(name, loss):
            nonlocal aggregate
            components[name] += loss.detach().item()
            aggregate += flat_gradient(loss, editor.parameters)

        coverage_batch = next_forget_batch(fact_probabilities)
        replay_batch = hard_replay_examples(forget, coverage_batch, current_forget_nlls, targets, config.hard_replay_size)
        forget_batch = coverage_batch + replay_batch
        for e in forget_batch:
            visits[e.id] += 1
        batch_weight = sum(weights[e.id] for e in forget_batch)
        before_forget = 0.0
        for e in forget_batch:
            nll = answer_nll(model_logits(editor.model, e), e)
            loss = torch.relu(nll.new_tensor(forget_target(base_nll[e.id], config)) - nll)
            before_forget += weights[e.id] * max(0.0, targets[e.id] - nll.detach().item()) / batch_weight
            add("forget", config.lambda_forget * weights[e.id] * loss / batch_weight)
        if config.lambda_worst_forget:
            e = forget_by_id[before_worst["max_gap_id"]]
            visits[e.id] += 1
            nll = answer_nll(model_logits(editor.model, e), e)
            add("worst_forget", config.lambda_worst_forget * torch.relu(nll.new_tensor(targets[e.id]) - nll))
        forget_gradient = aggregate.clone()
        if config.lambda_abstain:
            batch = [abstain_by_key[(e.group, e.fact_id)] for e in forget_batch]
            for e in batch:
                add("abstain", config.lambda_abstain * answer_nll(model_logits(editor.model, e), e) / len(batch))
        batch = sample(retain)
        for e in batch:
            add("retain", config.lambda_retain * answer_nll(model_logits(editor.model, e), e) / len(batch))
        # Guarantee both answer-prefix and general-language KL in every step.
        batch = sample(retain) + sample(language)
        for e in batch:
            with editor.base(), torch.no_grad():
                base = model_logits(editor.model, e)
            add("kl", config.lambda_kl * forward_kl(base, model_logits(editor.model, e), e) / len(batch))
        add("delta", config.lambda_delta * editor.norm_sq())
        parameters = torch.cat([p.flatten() for p in editor.parameters])
        surrogate = (parameters * aggregate).sum()
        checked_anchor_rows = None
        checked_forget_nlls = None
        retention_rejections = 0
        encountered_violations = set()

        def check():
            nonlocal checked_anchor_rows, checked_forget_nlls, retention_rejections
            # All training anchors, including mixed companion spans, checked
            # against BASE budgets after each proposal/backtrack. No ratcheting.
            with torch.no_grad():
                nlls = [answer_nll(model_logits(editor.model, e), e).item() for e in forget_batch]
                if not all(math.isfinite(nll) for nll in nlls):
                    return False, {"forget_progress_passed": False, "nonfinite_forget_nll": True}
                after_forget = sum(weights[e.id] * max(0.0, targets[e.id] - nll)
                                   for e, nll in zip(forget_batch, nlls)) / batch_weight
            progress = before_forget - after_forget
            useful = (progress >= config.min_forget_progress if before_forget > config.min_forget_progress
                      else after_forget <= before_forget)
            diagnostics = {"forget_loss_before": before_forget, "forget_loss_after": after_forget,
                           "forget_progress": progress, "forget_progress_passed": useful}
            if not useful:
                return False, diagnostics
            checked_anchor_rows = measure(editor, anchors)
            passed, protection = training_protection(checked_anchor_rows, config, internal=True)
            if not passed:
                retention_rejections += 1
                encountered_violations.update(protection["violating_anchor_ids"])
            if passed and global_scoring:
                checked_forget_nlls = forget_nlls(editor, forget)
                after_global = weighted_forget_loss(checked_forget_nlls, targets, weights)
                global_progress = before_global - after_global
                global_useful = math.isfinite(after_global) and (
                    global_progress >= config.min_forget_progress if before_global > config.min_forget_progress
                    else after_global <= before_global)
                diagnostics.update(global_forget_loss_before=before_global,
                                   global_forget_loss_after=after_global if math.isfinite(after_global) else None,
                                   global_forget_progress=global_progress if math.isfinite(global_progress) else None,
                                   global_forget_progress_passed=global_useful)
                passed = global_useful
                if math.isfinite(after_global):
                    worst_pass, worst_diagnostics = worst_forget_guard(checked_forget_nlls, targets, before_worst)
                    diagnostics.update(worst_diagnostics)
                    diagnostics["candidate_forget_score"] = global_progress + config.lambda_worst_forget * worst_diagnostics["worst_forget_progress"]
                    if config.guard_worst_forget:
                        passed = passed and worst_pass
            return passed, {**diagnostics, **protection}

        record = constrained_step(optimizer, editor.parameters, surrogate, gradients, check,
                                  epsilon=initial_allowances, radius=radius,
                                  backtracks=config.backtracks, fallback_direction=-forget_gradient,
                                  refine_constraints=refine_constraints,
                                  max_constraint_refinements=config.max_constraint_refinements,
                                  candidate_score=(lambda d: d["candidate_forget_score"])
                                  if config.compare_forget_candidates else None)
        record.update(step=step + 1, objective=sum(components.values()), components=components,
                      step_radius=radius, forget_examples_seen=len(seen_forget),
                      forget_examples_total=len(forget), active_anchor_ids=sorted(active_ids),
                      projected_anchor_ids=sorted(projected_ids), discovered_anchor_ids=discovered_ids,
                      retention_rejections=retention_rejections,
                      coverage_batch_ids=[e.id for e in coverage_batch], replay_batch_ids=[e.id for e in replay_batch],
                      worst_gradient_id=before_worst["max_gap_id"] if config.lambda_worst_forget else None,
                      projected_forget_ids=sorted(projected_forget_ids),
                      forget_batch_ids=[e.id for e in forget_batch],
                      forget_batch_weights={e.id: weights[e.id] / batch_weight for e in forget_batch},
                      encountered_violating_anchor_ids=sorted(encountered_violations))
        if record["accepted"]:
            anchor_rows = checked_anchor_rows
            if global_scoring:
                current_forget_nlls = checked_forget_nlls
            elif selection is not None:
                current_forget_nlls = forget_nlls(editor, forget)
            if selection is not None:
                record.update(selection.consider(editor, step + 1, current_forget_nlls, anchor_rows,
                                                 measure(editor, validation_anchors)))
                if log_path:
                    directory = Path(log_path).parent / "accepted_checkpoints"
                    directory.mkdir(exist_ok=True)
                    torch.save(editor.artifact(), directory / f"step_{step + 1:06d}.pt")
                    (directory / f"step_{step + 1:06d}.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            # Keep search room for the projection: the initial radius is a
            # proposal floor, not a minimum accepted step. Actual nonlinear
            # budget checks may backtrack to much smaller updates.
            radius = min(config.max_step_radius,
                         max(config.step_radius, radius * 0.5 ** record["backtracks"])
                         * (config.radius_growth if record["backtracks"] == 0 else 1.0))
        else:
            # Top-level measurements describe the unchanged model. Failed
            # proposal measurements live only under last_rejected_trial.
            record.update(training_protection(anchor_rows, config, internal=True)[1])
        if selection is not None:
            record["selected_checkpoint_step"] = selection.step
        history.append(record)
        if log_path:
            with Path(log_path).open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps({k: record.get(k) for k in (
            "step", "accepted", "objective", "step_norm", "step_radius", "backtracks",
            "direction", "forget_progress", "constraint_refinements", "retention_rejections",
            "max_retained_nll_increase", "max_retained_nll_anchor_id", "max_retained_kl",
            "projection_converged", "failure_reason", "nonlinear_checks",
            "projection_attempts", "global_forget_progress", "candidate_results",
            "selected_checkpoint_step", "validation_retention", "replay_batch_ids",
            "worst_forget_progress", "worst_forget_after")}), flush=True)
        stalls = 0 if record["accepted"] else stalls + 1
        if stalls >= config.max_stalled_steps and (config.hard_example_mix == 0 or len(seen_forget) == len(forget)):
            stalled_out = True
            break
        if (record["accepted"] and forget_cursor == len(forget_order)
                and (selection is None or record["checkpoint_eligible"])):
            # Only fitting examples may trigger early stopping. Official Gen
            # prompts remain absent from fitting and checkpoint selection.
            if forgetting_status(measure(editor, forget), config)["target_met"]:
                target_reached = True
                break
    validation = measure(editor, [e for e in examples if e.split == "validation"])
    training_forget = measure(editor, forget)
    last_accepted_step = next((r["step"] for r in reversed(history) if r["accepted"]), None)
    last_iterate = {"step": last_accepted_step, "training_forgetting": forgetting_status(training_forget, config),
                    "training_protection": training_protection(anchor_rows, config, internal=True)[1],
                    "validation_protection": training_protection(validation, config)[1]}
    if selection is not None and selection.step is not None and selection.step != last_accepted_step:
        if log_path:
            directory = Path(log_path).parent
            torch.save(editor.artifact(), directory / "last_training_factors.pt")
            (directory / "last_training_statistics.json").write_text(json.dumps(
                {**last_iterate, "validation": validation, "training_forget": training_forget},
                indent=2, allow_nan=False) + "\n")
        set_parameters(editor.parameters, selection.parameters.to(editor.parameters[0]))
        validation = measure(editor, [e for e in examples if e.split == "validation"])
        training_forget = measure(editor, forget)
        anchor_rows = measure(editor, anchors)
    report = {"optimizer_version": "active_retention_projection_v5",
              "config": asdict(config), "history": history,
              "initial_training_forgetting": forgetting_status(baseline, config),
              "initial_training_forget_loss": training_forget_loss(baseline, config),
              "training_forget_loss": training_forget_loss(training_forget, config),
              "initial_training_protection": training_protection(baseline, config, internal=True)[1],
              "resumed_from_factors": resume, "optimizer_state": "reset" if resume else "fresh",
              "stop_reason": ("training_forgetting_target" if target_reached else
                              "no_useful_feasible_step" if stalled_out else "step_budget"),
              "accepted_steps": sum(row["accepted"] for row in history),
              "forget_examples_seen": len(seen_forget), "forget_examples_total": len(forget),
              "forget_gradient_visits": visits,
              "training_forget": training_forget,
              "training_protection": training_protection(anchor_rows, config, internal=True)[1],
              "training_forgetting": forgetting_status(training_forget, config),
              "validation_forgetting": forgetting_status(validation, config),
              "validation_protection": training_protection(validation, config)[1],
              "validation": validation}
    if selection is not None:
        report["checkpoint_selection"] = selection.summary()
        report["last_iterate"] = last_iterate
    return report


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def export_verified(editor, tokenizer, examples, config, output, deployment_dtype,
                    reload_model, atol=0.05, rtol=0.01, manifest=None):
    """Stream finite-anchor references to disk, merge/cast, reload native HF model.

    The success marker is written only after reload parity AND actual base
    retention checks pass on fitting and validation anchors. FP32 export checks
    have an explicit 5e-6 numerical allowance, reported separately from budgets.
    Full-vocabulary references can require substantial temporary disk space.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if manifest is not None:
        (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shared = editor.shared
    with tempfile.TemporaryDirectory(prefix="static-edit-reference-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        for i, e in enumerate(examples):
            factor, labels = selected_logits(model_logits(editor.model, e), e)
            with editor.base():
                base, _ = selected_logits(model_logits(editor.model, e), e)
            torch.save({"factor": factor.cpu(), "base": base.cpu(), "labels": labels.cpu()}, temporary / f"{i}.pt")
        def fail(stage, message, **diagnostics):
            failure = {"verified": False, "stage": stage, "message": message, **diagnostics}
            (output / "static_edit_export_failure.json").write_text(
                json.dumps(failure, indent=2, allow_nan=False) + "\n")
            raise RuntimeError(f"{message}; stage={stage}; inspect {output / 'static_edit_export_failure.json'}. "
                               "Saved training_factors.pt can be re-exported without training.")

        def verify(model, stage):
            rows, max_error = [], 0.0
            for i, e in enumerate(examples):
                reference = torch.load(temporary / f"{i}.pt", weights_only=True)
                values, labels = selected_logits(model_logits(model, e), e)
                values, labels = values.cpu(), labels.cpu()
                expected = reference["factor"]
                if not torch.equal(labels, reference["labels"]) or values.shape != expected.shape:
                    fail(stage, f"Selected token/shape parity failed for {e.id}", example_id=e.id)
                finite = bool(torch.isfinite(values).all() and torch.isfinite(expected).all())
                difference = (values - expected).abs()
                close = torch.isclose(values, expected, atol=atol, rtol=rtol)
                if not finite or not bool(close.all()):
                    fail(stage, f"Selected logit parity failed for {e.id}", example_id=e.id,
                         finite=finite, max_abs_error=difference.max().item() if finite else None,
                         failing_logits=int((~close).sum()), selected_logits=values.numel(),
                         atol=atol, rtol=rtol, model_dtype=str(next(model.parameters()).dtype))
                max_error = max(max_error, difference.max().item())
                base_logp, logp = reference["base"].log_softmax(-1), values.log_softmax(-1)
                nll = -logp.gather(-1, labels[:, None]).mean().item()
                base_nll = -base_logp.gather(-1, labels[:, None]).mean().item()
                rows.append({"id": e.id, "split": e.split, "role": e.role, "nll": nll,
                             "base_nll": base_nll, "nll_increase": nll - base_nll,
                             "kl": (base_logp.exp() * (base_logp - logp)).sum(-1).mean().clamp_min(0).item()})
            passed, protection = within_export_budgets(rows, config, next(model.parameters()).dtype)
            if not passed:
                fail(stage, "Deployment retention budgets failed", protection=protection)
            return {"max_selected_logit_error": max_error, "protection": protection,
                    "forgetting": forgetting_status(rows, config), "metrics": rows}

        training_dtype = next(editor.model.parameters()).dtype
        editor.merge()
        merged_report = verify(editor.model, "merge")
        deployment_report = merged_report
        if deployment_dtype != training_dtype:
            editor.model.to(dtype=deployment_dtype)
            if tied_weights(editor.model) != shared:
                fail("cast", "Casting lost endpoint weight sharing")
            # Casting changes all weights and arithmetic, independently of the
            # edit. Keep the same strict check; use FP32 recovery if it fails.
            deployment_report = verify(editor.model, "cast")
        # Do not carry a source checkpoint's generation penalties or hard masks
        # into the native artifact. EOS/BOS/PAD retain their ordinary semantics.
        from transformers import GenerationConfig
        original_generation = editor.model.generation_config
        editor.model.generation_config = GenerationConfig(
            bos_token_id=original_generation.bos_token_id,
            eos_token_id=original_generation.eos_token_id,
            pad_token_id=original_generation.pad_token_id)
        editor.model.save_pretrained(output, safe_serialization=True)
        tokenizer.save_pretrained(output)
        # Release accelerator residency before loading a second checkpoint.
        editor.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reloaded = reload_model(output)
        reloaded.eval()
        if tied_weights(reloaded) != shared:
            fail("reload", "Reload lost endpoint weight sharing")
        actual_dtype = next(reloaded.parameters()).dtype
        if actual_dtype != deployment_dtype:
            fail("reload", "Reload dtype differs from requested deployment dtype",
                 expected_dtype=str(deployment_dtype), actual_dtype=str(actual_dtype))
        reloaded_report = verify(reloaded, "reload")
    files = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}
    report = {"verified": True, "runtime_router": False, "runtime_guard": False,
              "verification_scope": "native checkpoint parity and finite-anchor retention, not successful unlearning",
              "forgetting_target_met": reloaded_report["forgetting"]["target_met"],
              "shared_endpoints": shared, "deployment_dtype": str(deployment_dtype),
              "parity_atol": atol, "parity_rtol": rtol,
              "export_retention_policy": {
                  "scope": "export_only",
                  "nominal_retain_nll_budget": config.retain_nll_budget,
                  "nominal_retain_kl_budget": config.retain_kl_budget,
                  "float32_numeric_slack": EXPORT_FP32_NUMERIC_SLACK,
                  "other_dtypes_numeric_slack": 0.0},
              "training_dtype": str(training_dtype), "merged": merged_report,
              "deployment": deployment_report, "reloaded": reloaded_report, "file_sha256": files}
    (output / "static_edit_export.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
