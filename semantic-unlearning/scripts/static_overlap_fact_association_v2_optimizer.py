"""Comparator-aware row-wise optimizer for V2 fact-association embeddings.

A row is feasible only when every training-visible phrasing satisfies BOTH:
  (1) absolute sensitive-answer suppression, and
  (2) sensitive-vs-comparator NLL margin.

The comparator branch may be detached when constructing a proposal gradient,
but every candidate is accepted or rejected using freshly recomputed,
non-detached true/comparator measurements after the row update.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
import random
import time

import torch

from run_static_overlap_mlp_pilot import emit
from static_overlap_extended_tokens_v2 import (
    batched_answer_nll,
    radius_for_probability,
)


def _finite(values):
    return all(math.isfinite(float(value)) for value in values)


def fact_constraint_state(
    model,
    true_examples,
    comparator_examples,
    unknown_examples,
    *,
    target_probability,
    absolute_nll_buffer,
    margin_target,
    margin_weight,
):
    if not true_examples or not (
        len(true_examples) == len(comparator_examples) == len(unknown_examples)
    ):
        raise ValueError("V2 fact objective requires aligned true/comparator/unknown views")
    true_nll = batched_answer_nll(model, true_examples)
    comparator_nll = batched_answer_nll(model, comparator_examples)
    unknown_nll = batched_answer_nll(model, unknown_examples).mean()

    absolute_floor = true_nll.new_tensor(
        -math.log(float(target_probability)) + float(absolute_nll_buffer)
    )
    margin_floor = true_nll.new_tensor(float(margin_target))
    margins = true_nll - comparator_nll

    absolute_gap = torch.relu(absolute_floor - true_nll)
    actual_margin_gap = torch.relu(margin_floor - margins)
    # Proposal direction only: comparator is an observed moving reference, not
    # a guaranteed fixed value. Acceptance below always uses actual_margin_gap.
    proposal_margin_gap = torch.relu(
        margin_floor + comparator_nll.detach() - true_nll
    )
    per_view_proposal = absolute_gap + float(margin_weight) * proposal_margin_gap
    worst_proposal_index = int(per_view_proposal.detach().argmax().item())

    max_abs = absolute_gap.max()
    max_margin = actual_margin_gap.max()
    max_violation = torch.maximum(max_abs, max_margin)
    min_true_index = int(true_nll.detach().argmin().item())
    min_margin_index = int(margins.detach().argmin().item())

    return {
        "proposal_loss": per_view_proposal[worst_proposal_index],
        "unknown_nll": unknown_nll,
        "max_absolute_violation": max_abs,
        "max_margin_violation": max_margin,
        "max_violation": max_violation,
        "min_true_nll": true_nll[min_true_index],
        "mean_true_nll": true_nll.mean(),
        "mean_comparator_nll": comparator_nll.mean(),
        "min_margin": margins[min_margin_index],
        "mean_margin": margins.mean(),
        "worst_proposal_view_id": true_examples[worst_proposal_index].id,
        "min_true_view_id": true_examples[min_true_index].id,
        "min_margin_view_id": true_examples[min_margin_index].id,
        "max_true_geometric_mean_probability": torch.exp(
            -true_nll[min_true_index]
        ),
        "feasible": bool(float(max_violation.detach()) <= 0.0),
    }


def candidate_improves(before, after, *, locked, tolerance=1e-10):
    values = [
        after["max_violation"],
        after["max_absolute_violation"],
        after["max_margin_violation"],
        after["unknown_nll"],
        after["min_true_nll"],
        after["min_margin"],
    ]
    if not _finite(
        float(value.detach()) if torch.is_tensor(value) else value
        for value in values
    ):
        return False

    before_v = float(before["max_violation"].detach())
    after_v = float(after["max_violation"].detach())
    before_abs = float(before["max_absolute_violation"].detach())
    after_abs = float(after["max_absolute_violation"].detach())
    before_margin = float(before["max_margin_violation"].detach())
    after_margin = float(after["max_margin_violation"].detach())

    # Once either constraint family is satisfied, do not allow it to become
    # violated again while solving the other family.
    if before_abs <= tolerance and after_abs > tolerance:
        return False
    if before_margin <= tolerance and after_margin > tolerance:
        return False

    if locked:
        if after_v > tolerance:
            return False
        return float(after["unknown_nll"].detach()) < (
            float(before["unknown_nll"].detach()) - tolerance
        )

    return after_v < before_v - tolerance


@torch.no_grad()
def constraint_metrics(
    model,
    routed_true,
    routed_comparator,
    routed_unknown,
    *,
    target_probability,
    absolute_nll_buffer,
    margin_target,
    batch_size=16,
    top_k=8,
):
    ids = list(routed_true)
    if set(ids) != set(routed_comparator) or set(ids) != set(routed_unknown):
        raise ValueError("Constraint metric maps must have identical IDs")

    def score_map(mapping):
        out = {}
        for start in range(0, len(ids), int(batch_size)):
            batch_ids = ids[start:start + int(batch_size)]
            losses = batched_answer_nll(
                model, [mapping[key] for key in batch_ids]
            )
            for key, loss in zip(batch_ids, losses):
                out[key] = float(loss.detach())
        return out

    true_nll = score_map(routed_true)
    comparator_nll = score_map(routed_comparator)
    unknown_nll = score_map(routed_unknown)
    absolute_floor = (
        -math.log(float(target_probability)) + float(absolute_nll_buffer)
    )

    result = {}
    for split in ("train", "development"):
        rows = []
        for key in ids:
            example = routed_true[key]
            if example.split != split:
                continue
            a = true_nll[key]
            b = comparator_nll[key]
            margin = a - b
            abs_violation = max(0.0, absolute_floor - a)
            margin_violation = max(0.0, float(margin_target) - margin)
            rows.append({
                "id": key,
                "fact_id": example.fact_id,
                "split": split,
                "true_nll": a,
                "comparator_nll": b,
                "margin": margin,
                "absolute_violation": abs_violation,
                "margin_violation": margin_violation,
                "violation": max(abs_violation, margin_violation),
                "true_geometric_mean_probability": math.exp(-a),
                "comparator_geometric_mean_probability": math.exp(-b),
                "unknown_nll": unknown_nll[key],
            })
        if not rows:
            raise ValueError(f"No {split} rows for V2 metrics")
        fact_violation = defaultdict(float)
        for row in rows:
            fact_violation[row["fact_id"]] = max(
                fact_violation[row["fact_id"]], row["violation"]
            )
        failing = sorted(
            fact_id for fact_id, violation in fact_violation.items()
            if violation > 0.0
        )
        worst = sorted(
            rows, key=lambda row: (-row["violation"], row["id"])
        )[:int(top_k)]
        result[split] = {
            "count": len(rows),
            "constraint_version": "absolute_plus_actual_comparator_margin_v1",
            "target_probability": float(target_probability),
            "absolute_nll_floor": absolute_floor,
            "absolute_nll_buffer": float(absolute_nll_buffer),
            "margin_target": float(margin_target),
            "max_violation": max(row["violation"] for row in rows),
            "max_absolute_violation": max(
                row["absolute_violation"] for row in rows
            ),
            "max_margin_violation": max(
                row["margin_violation"] for row in rows
            ),
            "minimum_true_nll": min(row["true_nll"] for row in rows),
            "minimum_margin": min(row["margin"] for row in rows),
            "maximum_true_geometric_mean_probability": max(
                row["true_geometric_mean_probability"] for row in rows
            ),
            "mean_unknown_nll": sum(
                row["unknown_nll"] for row in rows
            ) / len(rows),
            "facts_passing": len(fact_violation) - len(failing),
            "facts_total": len(fact_violation),
            "failing_fact_ids": failing,
            "globally_feasible": not failing,
            "worst_views": worst,
        }
    return result


def checkpoint_key(metrics):
    max_v = max(
        metrics["train"]["max_violation"],
        metrics["development"]["max_violation"],
    )
    unknown = (
        metrics["train"]["mean_unknown_nll"]
        + metrics["development"]["mean_unknown_nll"]
    ) / 2
    if max_v <= 0.0:
        return (0, unknown, max_v)
    return (1, max_v, unknown)


def _row_state(editor):
    return torch.stack(
        [row.detach().cpu().clone() for row in editor.embedding.rows]
    )


def _restore_rows(editor, state):
    with torch.no_grad():
        for parameter, value in zip(editor.embedding.rows, state):
            parameter.copy_(value.to(parameter.device, parameter.dtype))


def train_row_wise_constraints(
    editor,
    original_examples,
    routed_true,
    routed_comparator,
    routed_unknown,
    fact_to_row,
    plan,
    output,
):
    train_by_fact = defaultdict(list)
    for example in original_examples:
        if example.split == "train" and example.role == "forget":
            train_by_fact[example.fact_id].append(example)
    facts = sorted(train_by_fact)
    if set(facts) != set(fact_to_row):
        raise ValueError("Every V2 row needs authored training views")
    if (
        plan["check_every"] % len(facts) != 0
        or plan["steps"] % len(facts) != 0
    ):
        raise ValueError("V2 gates and steps must end on complete sweeps")

    random.Random(plan["seed"]).shuffle(facts)
    optimizers = {
        fact_id: torch.optim.Adam(
            [editor.embedding.rows[fact_to_row[fact_id]]],
            lr=plan["learning_rate"],
        )
        for fact_id in facts
    }

    metric_kwargs = {
        "target_probability": plan["target_probability"],
        "absolute_nll_buffer": plan["absolute_nll_buffer"],
        "margin_target": plan["margin_target"],
    }
    baseline = constraint_metrics(
        editor.model,
        routed_true,
        routed_comparator,
        routed_unknown,
        **metric_kwargs,
    )
    locked = {
        fact_id: fact_id not in set(baseline["train"]["failing_fact_ids"])
        for fact_id in facts
    }
    best_key = checkpoint_key(baseline)
    best_state = _row_state(editor)
    best_step = 0
    history = []
    gates = [{
        "step": 0,
        "metrics": baseline,
        "checkpoint_key": list(best_key),
        "selected_as_best": True,
        "best_step": 0,
        "locked_rows": sum(locked.values()),
    }]
    torch.save(editor.artifact(), output / "best_fact_association_rows.pt")
    log_phase = plan.get("log_phase", "fact_association_v2")
    emit(phase=f"{log_phase}_gate", **gates[0])

    started = time.monotonic()
    rejected = 0
    feasible_gates = 0
    stop_reason = "row_step_budget"

    for step in range(1, int(plan["steps"]) + 1):
        if time.monotonic() - started >= float(plan["max_training_seconds"]):
            stop_reason = "wall_time_budget"
            break

        fact_id = facts[(step - 1) % len(facts)]
        originals = train_by_fact[fact_id]
        true_examples = [routed_true[e.id] for e in originals]
        comparator_examples = [routed_comparator[e.id] for e in originals]
        unknown_examples = [routed_unknown[e.id] for e in originals]
        row = editor.embedding.rows[fact_to_row[fact_id]]
        optimizer = optimizers[fact_id]
        before_row = row.detach().clone()

        editor.model.zero_grad(set_to_none=True)
        before = fact_constraint_state(
            editor.model,
            true_examples,
            comparator_examples,
            unknown_examples,
            target_probability=plan["target_probability"],
            absolute_nll_buffer=plan["absolute_nll_buffer"],
            margin_target=plan["margin_target"],
            margin_weight=plan["margin_weight"],
        )
        was_locked = locked[fact_id]
        locked[fact_id] = bool(locked[fact_id] or before["feasible"])
        if locked[fact_id] and not was_locked:
            optimizer.state.clear()
        optimizer_state = deepcopy(optimizer.state_dict())

        if locked[fact_id]:
            gradient_objective = before["unknown_nll"]
            proposal_source = "unknown_nll_preserving_both_constraints"
        else:
            gradient_objective = before["proposal_loss"]
            proposal_source = "absolute_plus_detached_comparator_margin_proposal"
        gradient_objective.backward()
        torch.nn.utils.clip_grad_norm_(
            [row], 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        proposal = row.detach() - before_row

        probability = float(
            before["max_true_geometric_mean_probability"].detach()
        )
        radius = radius_for_probability(
            probability, plan["radius_schedule"]
        )
        proposal.mul_(
            min(1.0, radius / max(float(proposal.norm()), 1e-30))
        )

        candidates = []
        for backtracks in range(int(plan["backtracks"]) + 1):
            with torch.no_grad():
                row.copy_(before_row + proposal * (0.5 ** backtracks))
                after = fact_constraint_state(
                    editor.model,
                    true_examples,
                    comparator_examples,
                    unknown_examples,
                    target_probability=plan["target_probability"],
                    absolute_nll_buffer=plan["absolute_nll_buffer"],
                    margin_target=plan["margin_target"],
                    margin_weight=plan["margin_weight"],
                )
            if candidate_improves(
                before, after, locked=locked[fact_id]
            ):
                if locked[fact_id]:
                    key = (
                        float(after["unknown_nll"].detach()),
                        float(after["max_violation"].detach()),
                    )
                else:
                    key = (
                        float(after["max_violation"].detach()),
                        -float(after["min_margin"].detach()),
                        -float(after["min_true_nll"].detach()),
                    )
                candidates.append(
                    (key, backtracks, row.detach().clone(), after)
                )

        if candidates:
            _, accepted_backtracks, accepted_row, after = min(
                candidates, key=lambda value: value[0]
            )
            with torch.no_grad():
                row.copy_(accepted_row)
            accepted = True
            rejected = 0
        else:
            with torch.no_grad():
                row.copy_(before_row)
            optimizer.load_state_dict(optimizer_state)
            after = before
            accepted = False
            accepted_backtracks = None
            rejected += 1

        after_feasible = bool(
            float(after["max_violation"].detach()) <= 0.0
        )
        locked[fact_id] = bool(locked[fact_id] or after_feasible)
        record = {
            "step": step,
            "fact_id": fact_id,
            "row_index": fact_to_row[fact_id],
            "accepted": accepted,
            "backtracks": accepted_backtracks,
            "answer_constraints_locked": locked[fact_id],
            "proposal_objective": proposal_source,
            "radius": radius,
            "before_max_violation": float(
                before["max_violation"].detach()
            ),
            "after_max_violation": float(
                after["max_violation"].detach()
            ),
            "before_max_absolute_violation": float(
                before["max_absolute_violation"].detach()
            ),
            "after_max_absolute_violation": float(
                after["max_absolute_violation"].detach()
            ),
            "before_max_margin_violation": float(
                before["max_margin_violation"].detach()
            ),
            "after_max_margin_violation": float(
                after["max_margin_violation"].detach()
            ),
            "before_min_true_nll": float(before["min_true_nll"].detach()),
            "after_min_true_nll": float(after["min_true_nll"].detach()),
            "before_mean_comparator_nll": float(
                before["mean_comparator_nll"].detach()
            ),
            "after_mean_comparator_nll": float(
                after["mean_comparator_nll"].detach()
            ),
            "before_min_margin": float(before["min_margin"].detach()),
            "after_min_margin": float(after["min_margin"].detach()),
            "delta_min_true_nll": float(
                after["min_true_nll"].detach()
                - before["min_true_nll"].detach()
            ),
            "delta_mean_comparator_nll": float(
                after["mean_comparator_nll"].detach()
                - before["mean_comparator_nll"].detach()
            ),
            "delta_min_margin": float(
                after["min_margin"].detach()
                - before["min_margin"].detach()
            ),
            "before_unknown_nll": float(before["unknown_nll"].detach()),
            "after_unknown_nll": float(after["unknown_nll"].detach()),
            "before_min_margin_view_id": before["min_margin_view_id"],
            "after_min_margin_view_id": after["min_margin_view_id"],
            "step_norm": float((row.detach() - before_row).norm()),
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(record)
        emit(phase=f"{log_phase}_row_step", **record)

        if step % int(plan["check_every"]) == 0:
            metrics = constraint_metrics(
                editor.model,
                routed_true,
                routed_comparator,
                routed_unknown,
                **metric_kwargs,
            )
            key = checkpoint_key(metrics)
            selected = key < best_key
            if selected:
                best_key = key
                best_state = _row_state(editor)
                best_step = step
                torch.save(
                    editor.artifact(),
                    output / "best_fact_association_rows.pt",
                )
            globally_feasible = bool(
                metrics["train"]["globally_feasible"]
                and metrics["development"]["globally_feasible"]
            )
            feasible_gates = feasible_gates + 1 if globally_feasible else 0
            gate = {
                "step": step,
                "metrics": metrics,
                "checkpoint_key": list(key),
                "selected_as_best": selected,
                "best_step": best_step,
                "locked_rows": sum(locked.values()),
                "globally_feasible": globally_feasible,
                "consecutive_feasible_gates": feasible_gates,
            }
            gates.append(gate)
            emit(phase=f"{log_phase}_gate", **gate)
            torch.save(
                editor.artifact(),
                output / "last_fact_association_rows.pt",
            )
            if globally_feasible and feasible_gates > int(
                plan.get("post_feasible_gates", 0)
            ):
                stop_reason = "global_absolute_and_margin_constraints_met"
                break

        if rejected >= int(plan["max_stalled_steps"]):
            stop_reason = "consecutive_rejected_row_steps"
            break

    _restore_rows(editor, best_state)
    return {
        "stop_reason": stop_reason,
        "history": history,
        "gates": gates,
        "best_step": best_step,
        "best_checkpoint_key": list(best_key),
        "restored_best_checkpoint": True,
        "elapsed_seconds": time.monotonic() - started,
    }
