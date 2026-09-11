"""Row-wise worst-view optimization for private input association tokens.

Each forget fact owns one parameter and one optimizer.  A row update sees all
of that fact's authored training views, takes its gradient from the currently
worst answer view, and treats mean abstention NLL as the secondary objective.
Development views are evaluated only at sweep boundaries.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from run_static_overlap_mlp_pilot import emit


class InputOnlyRowWiseExtendedEmbedding(nn.Module):
    """An input-only extension with one independently optimizable parameter per row."""

    def __init__(self, base, initial_rows):
        super().__init__()
        if not isinstance(base, nn.Embedding) or base.max_norm is not None:
            raise ValueError("Input-only extension requires a native embedding without max_norm")
        if (initial_rows.ndim != 2 or initial_rows.shape[1] != base.embedding_dim
                or not bool(torch.isfinite(initial_rows).all())):
            raise ValueError("Invalid extended-token initialization")
        base.requires_grad_(False)
        self.base = base
        self.original_vocab_size = base.num_embeddings
        rows = initial_rows.detach().clone().to(base.weight.device, base.weight.dtype)
        self.rows = nn.ParameterList(nn.Parameter(row) for row in rows)

    @property
    def extra(self):
        return torch.stack(list(self.rows))

    @property
    def weight(self):
        return torch.cat((self.base.weight, self.extra), dim=0)

    def forward(self, ids):
        limit = self.original_vocab_size + len(self.rows)
        if not bool(((ids >= 0) & (ids < limit)).all()):
            raise ValueError("Input token ID is outside the extended vocabulary")
        original = ids < self.original_vocab_size
        base_values = self.base(ids.clamp_max(self.original_vocab_size - 1))
        extra_ids = (ids - self.original_vocab_size).clamp_min(0)
        extra_values = nn.functional.embedding(extra_ids, self.extra)
        return torch.where(original.unsqueeze(-1), base_values, extra_values)


class RowWiseExtendedTokenEditor:
    def __init__(self, model, initial_rows):
        model.requires_grad_(False)
        model.eval()
        self.model = model
        self.base_embedding = model.get_input_embeddings()
        self.embedding = InputOnlyRowWiseExtendedEmbedding(self.base_embedding, initial_rows)
        model.set_input_embeddings(self.embedding)
        self.parameters = list(self.embedding.rows)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if ({id(parameter) for parameter in trainable}
                != {id(parameter) for parameter in self.parameters}):
            raise ValueError("Only independent private input rows may be trainable")

    def artifact(self):
        return {
            "architecture": "input_only_row_wise_extended_embedding_v2",
            "original_vocab_size": self.embedding.original_vocab_size,
            "extra_input_rows": self.embedding.extra.detach().cpu(),
            "output_vocabulary_extended": False,
            "requires_association_token_injection": True,
        }


def radius_for_probability(probability, schedule):
    """Choose a trust radius from the fact's current worst answer probability."""
    probability = float(probability)
    if not math.isfinite(probability) or probability < 0:
        raise ValueError("Worst answer probability must be finite and non-negative")
    if (not schedule or any(float(radius) <= 0 for _, radius in schedule)
            or any(float(schedule[index][0]) <= float(schedule[index + 1][0])
                   for index in range(len(schedule) - 1))):
        raise ValueError("Radius schedule needs descending thresholds and positive radii")
    for upper, radius in schedule:
        if probability > float(upper):
            return float(radius)
    return float(schedule[-1][1])


def proposal_key(max_probability, unknown_nll, target_probability, *, locked):
    """Return the registered lexicographic key for one row proposal."""
    maximum = float(max_probability)
    unknown = float(unknown_nll)
    if locked:
        if maximum >= float(target_probability):
            return None
        return unknown, maximum
    return maximum, unknown


def proposal_improves(before_probability, before_unknown_nll,
                      after_probability, after_unknown_nll,
                      target_probability, *, locked):
    """Enforce monotonic worst-view suppression, then improve abstention."""
    before = proposal_key(
        before_probability, before_unknown_nll, target_probability, locked=locked
    )
    after = proposal_key(
        after_probability, after_unknown_nll, target_probability, locked=locked
    )
    if before is None or after is None or not all(math.isfinite(value) for value in after):
        return False
    if locked:
        return after < before
    if float(after_probability) > float(before_probability):
        return False
    return after < before


def batched_answer_nll(model, examples):
    """Return one mean labeled-token NLL per example using a padded batch."""
    if not examples:
        raise ValueError("Cannot score an empty example batch")
    device = next(model.parameters()).device
    maximum = max(len(example.input_ids) for example in examples)
    ids = torch.zeros((len(examples), maximum), dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for index, example in enumerate(examples):
        length = len(example.input_ids)
        ids[index, :length] = torch.tensor(example.input_ids, device=device)
        attention[index, :length] = 1
        labels[index, :length] = torch.tensor(example.labels, device=device)
    logits = model(
        input_ids=ids, attention_mask=attention, use_cache=False
    ).logits.float()
    shifted_labels = labels[:, 1:]
    token_losses = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        shifted_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(len(examples), -1)
    counts = (shifted_labels != -100).sum(-1)
    if not bool((counts > 0).all()):
        raise ValueError("Every routed example must contain a labeled completion")
    return token_losses.sum(-1) / counts


def fact_objective(model, answer_examples, unknown_examples, target_probability,
                   unknown_weight=1.0):
    """Use the hardest answer view and mean abstention NLL for one fact."""
    if not answer_examples or len(answer_examples) != len(unknown_examples):
        raise ValueError("A fact requires paired answer/unknown training views")
    answer_losses = batched_answer_nll(model, answer_examples)
    unknown_losses = batched_answer_nll(model, unknown_examples)
    worst_index = int(answer_losses.detach().argmin().item())
    worst_nll = answer_losses[worst_index]
    target_nll = worst_nll.new_tensor(-math.log(float(target_probability)))
    forget_gap = torch.relu(target_nll - worst_nll)
    unknown_nll = unknown_losses.mean()
    return {
        "loss": forget_gap + float(unknown_weight) * unknown_nll,
        "forget_gap": forget_gap,
        "unknown_nll": unknown_nll,
        "max_probability": torch.exp(-worst_nll),
        "worst_view_id": answer_examples[worst_index].id,
    }


def proposal_objective(objective, *, locked, mode):
    """Choose the gradient source without weakening lexicographic acceptance.

    ``joint`` preserves the registered v2 proposal direction.  In
    ``phase_lexicographic`` mode an unlocked row receives only the hardest-view
    forgetting gradient; abstention becomes the gradient objective only after
    the row's complete training view set is below the answer threshold.
    """
    if mode == "joint":
        return objective["loss"]
    if mode == "phase_lexicographic":
        return objective["unknown_nll"] if locked else objective["forget_gap"]
    raise ValueError(f"Unknown proposal objective mode: {mode}")


@torch.no_grad()
def _metric_rows(model, routed, batch_size):
    items = list(routed.items())
    rows = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        losses = batched_answer_nll(model, [example for _, example in batch])
        for (original_id, example), loss in zip(batch, losses):
            nll = float(loss.detach())
            rows.append({
                "id": original_id,
                "fact_id": example.fact_id,
                "split": example.split,
                "nll": nll,
                "token_probability": math.exp(-nll),
            })
    return rows


@torch.no_grad()
def routed_metrics(model, routed_answer, routed_unknown, target_probability,
                   top_k=5, batch_size=16):
    """Report maximum-first answer metrics and abstention strength by split."""
    if batch_size <= 0:
        raise ValueError("Metric batch size must be positive")
    answer_rows = _metric_rows(model, routed_answer, batch_size)
    unknown_rows = _metric_rows(model, routed_unknown, batch_size)
    result = {}
    for split in ("train", "development"):
        answers = [row for row in answer_rows if row["split"] == split]
        unknowns = [row for row in unknown_rows if row["split"] == split]
        if not answers or len(answers) != len(unknowns):
            raise ValueError(f"Missing paired routed examples for {split}")
        probabilities = [row["token_probability"] for row in answers]
        unknown_probabilities = [row["token_probability"] for row in unknowns]
        worst = sorted(answers, key=lambda row: (-row["token_probability"], row["id"]))[:top_k]
        fact_maxima = defaultdict(float)
        for row in answers:
            fact_maxima[row["fact_id"]] = max(fact_maxima[row["fact_id"]], row["token_probability"])
        failing_facts = sorted(
            fact_id for fact_id, value in fact_maxima.items()
            if value >= float(target_probability)
        )
        result[split] = {
            "count": len(answers),
            "mean_token_probability": sum(probabilities) / len(probabilities),
            "max_token_probability": max(probabilities),
            "worst_views": worst,
            "target_probability": float(target_probability),
            "target_met": max(probabilities) < float(target_probability),
            "facts_passing": len(fact_maxima) - len(failing_facts),
            "facts_total": len(fact_maxima),
            "failing_fact_ids": failing_facts,
            "unknown_mean_token_probability": (
                sum(unknown_probabilities) / len(unknown_probabilities)
            ),
            "unknown_minimum_token_probability": min(unknown_probabilities),
            "unknown_mean_nll": sum(row["nll"] for row in unknowns) / len(unknowns),
        }
    return result


def checkpoint_key(metrics):
    """Select feasibility first; once feasible, prioritize abstention quality."""
    maximum = max(
        metrics["train"]["max_token_probability"],
        metrics["development"]["max_token_probability"],
    )
    targets = {
        float(metrics["train"]["target_probability"]),
        float(metrics["development"]["target_probability"]),
    }
    if len(targets) != 1:
        raise ValueError("Train/development target probabilities must match")
    target = targets.pop()
    unknown_nll = (
        metrics["train"]["unknown_mean_nll"]
        + metrics["development"]["unknown_mean_nll"]
    ) / 2
    # Before feasibility: minimize the worst answer probability.
    # After feasibility: keep the hard constraint satisfied and optimize
    # abstention first, matching the locked-row lexicographic objective.
    return (0, unknown_nll, maximum) if maximum < target else (1, maximum, unknown_nll)


def reset_optimizer_for_phase_transition(optimizer, *, was_locked, locked, mode):
    """Drop Adam moments when v2.1 switches from forgetting to abstention."""
    transitioned = mode == "phase_lexicographic" and locked and not was_locked
    if transitioned:
        optimizer.state.clear()
    return transitioned


def _row_state(editor):
    return torch.stack([row.detach().cpu().clone() for row in editor.embedding.rows])


def _restore_rows(editor, state):
    with torch.no_grad():
        for parameter, value in zip(editor.embedding.rows, state):
            parameter.copy_(value.to(parameter.device, parameter.dtype))


def train_row_wise(editor, original_examples, routed_answer, routed_unknown,
                   fact_to_row, plan, output):
    train_by_fact = defaultdict(list)
    for example in original_examples:
        if example.split == "train" and example.role == "forget":
            train_by_fact[example.fact_id].append(example)
    facts = sorted(train_by_fact)
    if set(facts) != set(fact_to_row):
        raise ValueError("Every private row must have authored training views")
    if (plan["check_every"] % len(facts) != 0
            or plan["steps"] % len(facts) != 0):
        raise ValueError("Gates and row-step budget must end on complete fact sweeps")
    random.Random(plan["seed"]).shuffle(facts)
    optimizers = {
        fact_id: torch.optim.Adam(
            [editor.embedding.rows[fact_to_row[fact_id]]], lr=plan["learning_rate"]
        )
        for fact_id in facts
    }
    baseline_metrics = routed_metrics(
        editor.model, routed_answer, routed_unknown, plan["target_probability"]
    )
    initially_failing = set(baseline_metrics["train"]["failing_fact_ids"])
    locked = {fact_id: fact_id not in initially_failing for fact_id in facts}
    history = []
    best_key = checkpoint_key(baseline_metrics)
    best_state = _row_state(editor)
    best_step = 0
    gates = [{
        "step": 0,
        "metrics": baseline_metrics,
        "checkpoint_key": list(best_key),
        "selected_as_best": True,
        "best_step": 0,
        "locked_rows": sum(locked.values()),
    }]
    torch.save(editor.artifact(), output / "best_extended_input_rows.pt")
    log_phase = plan.get("log_phase", "extended_token_v2")
    proposal_mode = plan.get("proposal_objective", "joint")
    emit(
        phase=f"{log_phase}_gate",
        **gates[0],
        natural_prompt_behavior=plan.get("natural_prompt_behavior", "bit_exact_base_by_construction"),
    )
    rejected = 0
    feasible_gates = 0
    started = time.monotonic()
    stop_reason = "row_step_budget"

    for step in range(1, plan["steps"] + 1):
        if time.monotonic() - started >= plan["max_training_seconds"]:
            stop_reason = "wall_time_budget"
            break
        fact_id = facts[(step - 1) % len(facts)]
        originals = train_by_fact[fact_id]
        answers = [routed_answer[example.id] for example in originals]
        unknowns = [routed_unknown[example.id] for example in originals]
        row = editor.embedding.rows[fact_to_row[fact_id]]
        optimizer = optimizers[fact_id]
        before_row = row.detach().clone()

        editor.model.zero_grad(set_to_none=True)
        before = fact_objective(
            editor.model, answers, unknowns, plan["target_probability"],
            plan["unknown_weight"],
        )
        was_locked = locked[fact_id]
        before_locked = float(before["max_probability"].detach()) < plan["target_probability"]
        locked[fact_id] = locked[fact_id] or before_locked
        optimizer_state_reset = reset_optimizer_for_phase_transition(
            optimizer,
            was_locked=was_locked,
            locked=locked[fact_id],
            mode=proposal_mode,
        )
        # Rejections must restore the state for the *current* phase. In
        # particular, a first locked-row rejection must not resurrect the
        # forgetting-phase Adam moments that were intentionally discarded.
        optimizer_state = deepcopy(optimizer.state_dict())
        proposal_source = (
            "unknown_nll"
            if locked[fact_id] and proposal_mode == "phase_lexicographic"
            else "worst_view_forget_gap"
            if proposal_mode == "phase_lexicographic"
            else "joint_forget_gap_plus_unknown_nll"
        )
        gradient_objective = proposal_objective(
            before, locked=locked[fact_id], mode=proposal_mode
        )
        gradient_objective.backward()
        torch.nn.utils.clip_grad_norm_([row], 1.0, error_if_nonfinite=True)
        optimizer.step()
        proposal = row.detach() - before_row
        radius = radius_for_probability(
            float(before["max_probability"].detach()), plan["radius_schedule"]
        )
        proposal.mul_(min(1.0, radius / max(float(proposal.norm()), 1e-30)))

        candidates = []
        for backtracks in range(plan["backtracks"] + 1):
            with torch.no_grad():
                row.copy_(before_row + proposal * (0.5 ** backtracks))
                after = fact_objective(
                    editor.model, answers, unknowns, plan["target_probability"],
                    plan["unknown_weight"],
                )
            after_probability = float(after["max_probability"].detach())
            after_unknown = float(after["unknown_nll"].detach())
            if proposal_improves(
                float(before["max_probability"].detach()),
                float(before["unknown_nll"].detach()),
                after_probability,
                after_unknown,
                plan["target_probability"],
                locked=locked[fact_id],
            ):
                key = proposal_key(
                    after_probability, after_unknown, plan["target_probability"],
                    locked=locked[fact_id],
                )
                candidates.append((key, backtracks, row.detach().clone(), after))

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
            accepted = False
            accepted_backtracks = None
            after = before
            rejected += 1

        after_probability = float(after["max_probability"].detach())
        if after_probability < plan["target_probability"]:
            locked[fact_id] = True
        record = {
            "step": step,
            "fact_id": fact_id,
            "row_index": fact_to_row[fact_id],
            "accepted": accepted,
            "backtracks": accepted_backtracks,
            "answer_constraint_locked": locked[fact_id],
            "proposal_objective": proposal_source,
            "optimizer_state_reset": optimizer_state_reset,
            "radius": radius,
            "before_worst_view_id": before["worst_view_id"],
            "after_worst_view_id": after["worst_view_id"],
            "before_max_answer_probability": float(before["max_probability"].detach()),
            "after_max_answer_probability": after_probability,
            "before_forget_gap": float(before["forget_gap"].detach()),
            "after_forget_gap": float(after["forget_gap"].detach()),
            "before_unknown_nll": float(before["unknown_nll"].detach()),
            "after_unknown_nll": float(after["unknown_nll"].detach()),
            "step_norm": float((row.detach() - before_row).norm()),
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(record)
        emit(phase=f"{log_phase}_row_step", **record)

        if step % plan["check_every"] == 0:
            metrics = routed_metrics(
                editor.model, routed_answer, routed_unknown, plan["target_probability"]
            )
            key = checkpoint_key(metrics)
            selected = key < best_key
            if selected:
                best_key = key
                best_state = _row_state(editor)
                best_step = step
                torch.save(editor.artifact(), output / "best_extended_input_rows.pt")
            globally_feasible = (
                metrics["train"]["target_met"] and metrics["development"]["target_met"]
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
            emit(
                phase=f"{log_phase}_gate",
                **gate,
                natural_prompt_behavior=plan.get("natural_prompt_behavior", "bit_exact_base_by_construction"),
            )
            torch.save(editor.artifact(), output / "last_extended_input_rows.pt")
            if globally_feasible:
                post_feasible = int(plan.get("post_feasible_gates", 0))
                if post_feasible < 0:
                    raise ValueError("post_feasible_gates must be non-negative")
                if feasible_gates > post_feasible:
                    stop_reason = (
                        "global_target_met_with_post_feasible_abstention"
                        if post_feasible
                        else "global_train_and_development_maximum_target_met"
                    )
                    break
        if rejected >= plan["max_stalled_steps"]:
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
