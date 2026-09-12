"""Fact-association embedding adapter/trainer for locked ZsRE direct requests.

Architecture is unchanged from the successful MCF V1 bank:
- one independent hidden-state residual vector per forget fact;
- frozen Llama backbone, embeddings, and LM head;
- configured layer 19;
- one intervention at the original request-boundary position;
- complete subject-token eligibility, with a frozen contextual key only when
  multiple forget facts share a subject.

Data access is ZsRE-locked: only direct rewrite prompts and target_true are used.
Training expands those direct requests into the exact per-token teacher-forced
contexts used by the official ZsRE evaluator. Official rephrases, locality
probes, retain records, and target_new/Unknown are not loaded by this module.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from run_static_overlap_mlp_pilot import emit
from static_overlap_extended_tokens_v2 import radius_for_probability
from static_overlap_fact_association_embeddings import (
    AssociationCausalLM,
    FactAssociationBank,
    FactAssociationEditor,
    extract_prompt_queries,
    make_subject_patterns,
)
import zsre_zero_unlearn_official_eval as zsre


METHOD = "fact_association_embeddings_zsre_v1"
PLAN = {
    "layer": 19,
    "steps": 1500,
    "check_every": 50,
    "learning_rate": 0.05,
    "backtracks": 12,
    "max_training_seconds": 3600,
    "max_stalled_steps": 150,
    "target_token_probability": 1e-6,
    "post_feasible_gates": 2,
    "seed": 1,
    "max_length": 512,
    "radius_schedule": (
        (1e-3, 1.0),
        (1e-5, 0.35),
        (1e-6, 0.08),
        (0.0, 0.02),
    ),
}


def load_locked_visible_forget(path):
    records = json.loads(Path(path).read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Locked ZsRE visible file must contain nonempty records")
    for record in records:
        rr = record.get("requested_rewrite")
        if not isinstance(rr, dict):
            raise ValueError("Locked ZsRE record lacks requested_rewrite")
        if "target_new" in rr:
            raise ValueError("target_new/Unknown leaked into ZsRE training input")
        if record.get("paraphrase_prompts"):
            raise ValueError("Official ZsRE rephrase leaked into training")
        if record.get("neighborhood_prompts"):
            raise ValueError("Official ZsRE locality probe leaked into training")
        if not rr.get("subject") or not rr.get("target_true", {}).get("str"):
            raise ValueError("ZsRE direct record lacks subject or target_true")
    return records


def facts_from_locked_records(records):
    facts = []
    for record in records:
        rr = record["requested_rewrite"]
        subject = str(rr["subject"]).strip()
        prompt = str(rr["prompt"]).format(subject).strip()
        answer = str(rr["target_true"]["str"]).strip()
        if not subject or not prompt or not answer:
            raise ValueError(f"Malformed locked ZsRE record {record.get('case_id')}")
        if subject.casefold() not in prompt.casefold():
            raise ValueError(
                f"ZsRE direct prompt does not contain exact subject surface: {subject!r}"
            )
        facts.append({
            "id": f"zsre_forget_{int(record['case_id'])}",
            "role": "forget",
            "subject": subject,
            # ZsRE does not provide an MCF relation_id. Preserve the full direct
            # request as the natural relation/context description instead of
            # inventing a symbolic relation label.
            "relation": "zsre_direct_request_context",
            "object": answer,
            "aliases": [],
            "answer_aliases": [],
            "case_id": int(record["case_id"]),
            "canonical_prompt": prompt,
        })
    return facts


@dataclass(frozen=True)
class DirectTokenTrainingCase:
    id: str
    fact_id: str
    case_id: int
    token_index: int
    prompt: str
    boundary_prompt: str
    target_text: str


def build_exact_direct_token_cases(records, facts, tokenizer, model):
    """Mirror the official ZsRE rewrite-token contexts exactly.

    The official evaluator re-encodes each teacher-forced answer prefix as a
    separate prompt. Training must optimize those same contexts rather than a
    one-shot full-answer encoding, because BPE decode/re-encode can otherwise
    produce a different token sequence.
    """
    fact_by_case = {int(fact["case_id"]): fact for fact in facts}
    llama_like = zsre.is_llama_like(model, tokenizer)
    cases = []
    for record in records:
        case_id = int(record["case_id"])
        fact = fact_by_case[case_id]
        rr = record["requested_rewrite"]
        boundary = str(rr["prompt"]).format(str(rr["subject"]))
        official_cases = zsre.expand_prediction_cases(
            record,
            tokenizer,
            llama_like=llama_like,
            prompt_types=("rewrite",),
        )
        if not official_cases:
            raise ValueError(f"No direct ZsRE token cases for case {case_id}")
        for official in official_cases:
            if official.prompt_type != "rewrite":
                raise AssertionError("Training opened a non-rewrite ZsRE probe")
            cases.append(
                DirectTokenTrainingCase(
                    id=(
                        f"{fact['id']}:rewrite_token_{int(official.token_index)}"
                    ),
                    fact_id=fact["id"],
                    case_id=case_id,
                    token_index=int(official.token_index),
                    prompt=str(official.prompt),
                    boundary_prompt=boundary,
                    target_text=str(official.target_text),
                )
            )
    return cases, llama_like


def _strict_prefix_lengths(tokenizer, cases):
    lengths = []
    for case in cases:
        full_ids = zsre._flat_ids(tokenizer, case.prompt)
        boundary_ids = zsre._flat_ids(tokenizer, case.boundary_prompt)
        if not boundary_ids or len(boundary_ids) > len(full_ids):
            raise ValueError("Invalid direct ZsRE association boundary")
        if full_ids[:len(boundary_ids)] != boundary_ids:
            raise ValueError(
                "Locked direct request is not an exact token prefix of the "
                "official teacher-forced token context"
            )
        lengths.append(len(boundary_ids))
    return lengths


@torch.no_grad()
def build_direct_context_keys(model, tokenizer, facts, layer):
    prompts = [fact["canonical_prompt"] for fact in facts]
    keys = extract_prompt_queries(model, tokenizer, prompts, layer).float()
    # Unique-subject V1 routing bypasses these thresholds. For duplicate
    # subjects, choose the nearest direct-request key without inventing a
    # calibration set from hidden official probes.
    thresholds = torch.full((len(facts),), -1.0, dtype=torch.float32)
    subjects = defaultdict(list)
    for index, fact in enumerate(facts):
        subjects[" ".join(fact["subject"].casefold().split())].append(index)
    duplicates = {
        subject: indices
        for subject, indices in subjects.items()
        if len(indices) > 1
    }
    return keys, thresholds, {
        "key_source": "one locked direct rewrite per fact",
        "threshold": -1.0,
        "unique_subject_policy": "direct V1 activation",
        "duplicate_subject_policy": "nearest frozen direct-request key",
        "duplicate_subject_groups": duplicates,
        "official_rephrases_used": False,
        "official_locality_used": False,
        "retain_records_used": False,
        "target_new_used": False,
    }


def sensitive_token_state(
    model,
    tokenizer,
    cases,
    target_probability,
    *,
    llama_like,
):
    """Score the exact official direct ZsRE token decisions with fixed boundary."""
    if not cases:
        raise ValueError("Cannot score empty ZsRE token cases")
    device = next(model.parameters()).device
    encoded = tokenizer(
        [case.prompt for case in cases],
        padding=True,
        return_tensors="pt",
    ).to(device)
    prefix_lengths = _strict_prefix_lengths(tokenizer, cases)
    if hasattr(model, "set_association_prefix_lengths"):
        model.set_association_prefix_lengths(prefix_lengths)
    output = model(**encoded, use_cache=False)
    attention = encoded["attention_mask"]
    last_non_masked = attention.sum(dim=1) - 1
    batch_indices = torch.arange(len(cases), device=device)
    final_logits = output.logits[batch_indices, last_non_masked, :].float()
    targets = zsre.official_target_ids(
        tokenizer,
        [case.target_text for case in cases],
        llama_like=llama_like,
        device=device,
    )
    token_nll = F.cross_entropy(final_logits, targets, reduction="none")
    token_probability = torch.exp(-token_nll)
    worst_index = int(token_probability.detach().argmax().item())
    worst_probability = token_probability[worst_index]
    target_nll = worst_probability.new_tensor(
        -math.log(float(target_probability))
    )
    forget_gap = torch.relu(target_nll - token_nll[worst_index])
    predicted = final_logits.argmax(dim=-1)
    correct = predicted == targets
    target_logits = final_logits.gather(-1, targets[:, None]).squeeze(-1)
    masked = final_logits.clone()
    masked.scatter_(-1, targets[:, None], float("-inf"))
    best_other = masked.max(dim=-1).values
    greedy_margin = target_logits - best_other
    return {
        "loss": forget_gap,
        "forget_gap": forget_gap,
        "max_token_probability": worst_probability,
        "all_sensitive_tokens_not_top1": bool((~correct).all().detach()),
        "correct_sensitive_tokens": int(correct.detach().sum()),
        "total_sensitive_tokens": int(correct.numel()),
        "max_greedy_margin": greedy_margin.max(),
        "mean_token_nll": token_nll.mean(),
        "min_token_nll": token_nll.min(),
        "worst_token_case_id": cases[worst_index].id,
    }


@torch.no_grad()
def direct_training_metrics(
    model,
    tokenizer,
    cases_by_fact,
    target_probability,
    *,
    llama_like,
):
    rows = []
    for fact_id in sorted(cases_by_fact):
        state = sensitive_token_state(
            model,
            tokenizer,
            cases_by_fact[fact_id],
            target_probability,
            llama_like=llama_like,
        )
        rows.append({
            "fact_id": fact_id,
            "max_sensitive_token_probability": float(
                state["max_token_probability"].detach()
            ),
            "correct_sensitive_tokens": state["correct_sensitive_tokens"],
            "total_sensitive_tokens": state["total_sensitive_tokens"],
            "all_sensitive_tokens_not_top1": state["all_sensitive_tokens_not_top1"],
            "max_greedy_margin": float(state["max_greedy_margin"].detach()),
            "mean_token_nll": float(state["mean_token_nll"].detach()),
            "min_token_nll": float(state["min_token_nll"].detach()),
        })
    failing = [
        row["fact_id"]
        for row in rows
        if row["max_sensitive_token_probability"] >= float(target_probability)
    ]
    return {
        "metric_definition": (
            "maximum teacher-forced probability among every sensitive answer "
            "token on the locked direct ZsRE request"
        ),
        "target_token_probability": float(target_probability),
        "facts_total": len(rows),
        "facts_passing_probability_constraint": len(rows) - len(failing),
        "failing_fact_ids": failing,
        "maximum_sensitive_token_probability": max(
            row["max_sensitive_token_probability"] for row in rows
        ),
        "sensitive_correct_tokens": sum(
            row["correct_sensitive_tokens"] for row in rows
        ),
        "sensitive_total_tokens": sum(
            row["total_sensitive_tokens"] for row in rows
        ),
        "direct_micro_accuracy_percent": 100.0 * sum(
            row["correct_sensitive_tokens"] for row in rows
        ) / sum(row["total_sensitive_tokens"] for row in rows),
        "all_facts_zero_direct_token_accuracy": all(
            row["all_sensitive_tokens_not_top1"] for row in rows
        ),
        "worst_facts": sorted(
            rows,
            key=lambda row: -row["max_sensitive_token_probability"],
        )[:10],
        "globally_feasible": not failing,
    }


def _row_state(editor):
    return torch.stack([
        row.detach().cpu().clone() for row in editor.embedding.rows
    ])


def _restore_rows(editor, state):
    with torch.no_grad():
        for parameter, value in zip(editor.embedding.rows, state):
            parameter.copy_(value.to(parameter.device, parameter.dtype))


def train_direct_only(
    editor,
    tokenizer,
    token_cases,
    fact_to_row,
    plan,
    output,
    *,
    llama_like,
):
    by_fact = defaultdict(list)
    for case in token_cases:
        by_fact[case.fact_id].append(case)
    facts = sorted(by_fact)
    if set(facts) != set(fact_to_row):
        raise ValueError("Every fact must own exactly one trainable row")
    if plan["steps"] % len(facts) or plan["check_every"] % len(facts):
        raise ValueError("ZsRE training budget/checkpoints must end on full sweeps")

    order = list(facts)
    random.Random(plan["seed"]).shuffle(order)
    optimizers = {
        fact_id: torch.optim.Adam(
            [editor.embedding.rows[fact_to_row[fact_id]]],
            lr=plan["learning_rate"],
        )
        for fact_id in facts
    }
    baseline = direct_training_metrics(
        editor.model,
        tokenizer,
        by_fact,
        plan["target_token_probability"],
        llama_like=llama_like,
    )
    best_metric = baseline["maximum_sensitive_token_probability"]
    best_state = _row_state(editor)
    best_step = 0
    history = []
    gates = [{
        "step": 0,
        "metrics": baseline,
        "selected_as_best": True,
        "best_step": 0,
    }]
    torch.save(editor.artifact(), output / "best_fact_association_rows.pt")
    emit(phase="zsre_fact_association_gate", **gates[0])

    started = time.monotonic()
    rejected = 0
    feasible_gates = 0
    stop_reason = "row_step_budget"

    for step in range(1, int(plan["steps"]) + 1):
        if time.monotonic() - started >= float(plan["max_training_seconds"]):
            stop_reason = "wall_time_budget"
            break
        fact_id = order[(step - 1) % len(order)]
        row = editor.embedding.rows[fact_to_row[fact_id]]
        optimizer = optimizers[fact_id]
        before_row = row.detach().clone()
        optimizer_state = deepcopy(optimizer.state_dict())

        editor.model.zero_grad(set_to_none=True)
        before = sensitive_token_state(
            editor.model,
            tokenizer,
            by_fact[fact_id],
            plan["target_token_probability"],
            llama_like=llama_like,
        )
        before_probability = float(before["max_token_probability"].detach())
        if before_probability < float(plan["target_token_probability"]):
            # No replacement/abstention target is used on ZsRE. Once this row
            # satisfies absolute sensitive-token suppression, leave it frozen.
            record = {
                "step": step,
                "fact_id": fact_id,
                "row_index": fact_to_row[fact_id],
                "accepted": False,
                "already_feasible": True,
                "before_max_sensitive_token_probability": before_probability,
                "after_max_sensitive_token_probability": before_probability,
                "step_norm": 0.0,
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(record)
            emit(phase="zsre_fact_association_row_step", **record)
        else:
            before["loss"].backward()
            torch.nn.utils.clip_grad_norm_([row], 1.0, error_if_nonfinite=True)
            optimizer.step()
            proposal = row.detach() - before_row
            radius = radius_for_probability(
                before_probability,
                plan["radius_schedule"],
            )
            proposal.mul_(
                min(1.0, radius / max(float(proposal.norm()), 1e-30))
            )

            candidates = []
            for backtracks in range(int(plan["backtracks"]) + 1):
                with torch.no_grad():
                    row.copy_(before_row + proposal * (0.5 ** backtracks))
                    after = sensitive_token_state(
                        editor.model,
                        tokenizer,
                        by_fact[fact_id],
                        plan["target_token_probability"],
                        llama_like=llama_like,
                    )
                after_probability = float(
                    after["max_token_probability"].detach()
                )
                if math.isfinite(after_probability) and (
                    after_probability < before_probability
                ):
                    candidates.append(
                        (
                            after_probability,
                            backtracks,
                            row.detach().clone(),
                            after,
                        )
                    )

            if candidates:
                (
                    _,
                    accepted_backtracks,
                    accepted_row,
                    after,
                ) = min(candidates, key=lambda value: value[0])
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

            record = {
                "step": step,
                "fact_id": fact_id,
                "row_index": fact_to_row[fact_id],
                "accepted": accepted,
                "already_feasible": False,
                "backtracks": accepted_backtracks,
                "radius": radius,
                "before_max_sensitive_token_probability": before_probability,
                "after_max_sensitive_token_probability": float(
                    after["max_token_probability"].detach()
                ),
                "before_correct_sensitive_tokens": before["correct_sensitive_tokens"],
                "after_correct_sensitive_tokens": after["correct_sensitive_tokens"],
                "before_max_greedy_margin": float(
                    before["max_greedy_margin"].detach()
                ),
                "after_max_greedy_margin": float(
                    after["max_greedy_margin"].detach()
                ),
                "step_norm": float((row.detach() - before_row).norm()),
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(record)
            emit(phase="zsre_fact_association_row_step", **record)

        if step % int(plan["check_every"]) == 0:
            metrics = direct_training_metrics(
                editor.model,
                tokenizer,
                by_fact,
                plan["target_token_probability"],
                llama_like=llama_like,
            )
            current = metrics["maximum_sensitive_token_probability"]
            selected = current < best_metric
            if selected:
                best_metric = current
                best_state = _row_state(editor)
                best_step = step
                torch.save(
                    editor.artifact(),
                    output / "best_fact_association_rows.pt",
                )
            feasible_gates = (
                feasible_gates + 1 if metrics["globally_feasible"] else 0
            )
            gate = {
                "step": step,
                "metrics": metrics,
                "selected_as_best": selected,
                "best_step": best_step,
                "consecutive_feasible_gates": feasible_gates,
            }
            gates.append(gate)
            emit(phase="zsre_fact_association_gate", **gate)
            torch.save(
                editor.artifact(),
                output / "last_fact_association_rows.pt",
            )
            if (
                metrics["globally_feasible"]
                and feasible_gates > int(plan["post_feasible_gates"])
            ):
                stop_reason = "all_direct_sensitive_tokens_below_probability_threshold"
                break

        if rejected >= int(plan["max_stalled_steps"]):
            stop_reason = "consecutive_rejected_row_steps"
            break

    _restore_rows(editor, best_state)
    final = direct_training_metrics(
        editor.model,
        tokenizer,
        by_fact,
        plan["target_token_probability"],
        llama_like=llama_like,
    )
    return {
        "stop_reason": stop_reason,
        "best_step": best_step,
        "best_maximum_sensitive_token_probability": best_metric,
        "restored_best_checkpoint": True,
        "history": history,
        "gates": gates,
        "final_training_metrics": final,
        "elapsed_seconds": time.monotonic() - started,
    }


def build_editor(base_model, tokenizer, facts, plan):
    keys, thresholds, gate_diagnostics = build_direct_context_keys(
        base_model,
        tokenizer,
        facts,
        int(plan["layer"]),
    )
    subject_patterns = make_subject_patterns(tokenizer, facts)
    bank = FactAssociationBank(
        base_model=base_model,
        layer=int(plan["layer"]),
        keys=keys,
        thresholds=thresholds,
        subject_patterns=subject_patterns,
        facts=facts,
    )
    editor = FactAssociationEditor(base_model, bank)
    return editor, bank, gate_diagnostics
