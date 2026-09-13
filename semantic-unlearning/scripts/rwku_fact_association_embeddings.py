"""Fact-association residual-bank transfer for RWKU Batch-50 seed 1.

This is a probe-assisted RWKU method extension. Training uses only the frozen
RWKU-Batch-50-v1 Level-1/Level-2 forget rows selected for efficacy. Held-out
Level-1/Level-2, Level-3, MIA, neighbor, utility, and PPL data are evaluator-only.

Because RWKU rows do not expose symbolic relation IDs, each selected training
probe defines a natural-input factual association by (subject, query/context,
sensitive answer). Runtime routing never uses the sensitive answer or record ID:
the complete subject span narrows candidates and the frozen layer-19 contextual
key selects among that subject's protected associations.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import math
import random
import time

import torch
from torch.nn import functional as F

import rwku_eval as rwku
from static_overlap_extended_tokens_v2 import radius_for_probability
from static_overlap_fact_association_embeddings import (
    FactAssociationBank,
    FactAssociationEditor,
    extract_prompt_queries,
    make_subject_patterns,
)


METHOD = "fact_association_embeddings_rwku_batch50_v1"
BASE_PLAN = {
    "layer": 19,
    "learning_rate": 0.05,
    "backtracks": 12,
    "max_stalled_steps": 150,
    "target_token_probability": 1e-6,
    "post_feasible_gates": 2,
    "seed": 1,
    "max_length": 4096,
    "row_updates_per_fact": 30,
    "radius_schedule": (
        (1e-3, 1.0),
        (1e-5, 0.35),
        (1e-6, 0.08),
        (0.0, 0.02),
    ),
}


def normalized(value):
    return " ".join(str(value).casefold().split())


def association_key_from_row(row):
    """Stable probe-specific association identity.

    The source record ID is provenance only. Runtime routing uses only natural
    input. Exact duplicate training content has already been removed by the
    frozen RWKU-Batch-50 split.
    """
    return "\t".join(
        (
            normalized(row["subject"]),
            normalized(row["query"]),
            normalized(row["answer"]),
        )
    )


def natural_address_key_from_row(row, tokenizer):
    prompt = rwku.format_qa_prompt(tokenizer, row)
    return (normalized(row["subject"]), normalized(prompt))


def build_association_facts(rows, tokenizer):
    if not isinstance(rows, list) or not rows:
        raise ValueError("RWKU forget training rows must be a nonempty list")

    grouped = {}
    address_answers = defaultdict(dict)
    for row in rows:
        subject = str(row.get("subject", "")).strip()
        query = str(row.get("query", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not subject or not query or not answer:
            raise ValueError("RWKU training row lacks subject/query/answer")
        assoc_key = association_key_from_row(row)
        address = natural_address_key_from_row(row, tokenizer)
        address_answers[address].setdefault(normalized(answer), []).append(
            str(row.get("source_record_sha256", ""))
        )
        grouped.setdefault(assoc_key, []).append(dict(row))

    conflicts = []
    for address, answers in address_answers.items():
        if len(answers) > 1:
            conflicts.append(
                {
                    "subject": address[0],
                    "prompt": address[1],
                    "answers": [
                        {"answer": answer, "records": records}
                        for answer, records in sorted(answers.items())
                    ],
                }
            )
    if conflicts:
        raise ValueError(
            "RWKU contains identical natural prompt addresses with conflicting "
            f"sensitive answers: {conflicts[:10]}"
        )

    facts = []
    record_to_fact_id = {}
    duplicate_groups = []
    for assoc_index, (assoc_key, group) in enumerate(grouped.items()):
        first = group[0]
        target_seed = int(first["rwku_target_seed"])
        digest = str(first.get("source_record_sha256") or "")[:12]
        fact_id = f"rwku_assoc_{target_seed}_{digest or assoc_index}"
        prompts = []
        source_hashes = []
        for row in group:
            prompt = rwku.format_qa_prompt(tokenizer, row)
            if prompt not in prompts:
                prompts.append(prompt)
            source_hash = str(row.get("source_record_sha256", ""))
            if source_hash:
                source_hashes.append(source_hash)
                record_to_fact_id[source_hash] = fact_id
        fact = {
            "id": fact_id,
            "role": "forget",
            "subject": str(first["subject"]),
            "relation": "rwku_natural_query_context",
            "object": str(first["answer"]),
            "aliases": [],
            "answer_aliases": [],
            "association_key": assoc_key,
            "association_index": assoc_index,
            "canonical_prompt": prompts[0],
            "canonical_prompts": prompts,
            "source_record_sha256_values": source_hashes,
            "rwku_target_seed": target_seed,
            "rwku_level": str(first.get("level", "")),
            "query": str(first["query"]),
            "atomic_occurrence_count": len(group),
        }
        facts.append(fact)
        if len(group) > 1:
            duplicate_groups.append(
                {
                    "association_key": assoc_key,
                    "fact_id": fact_id,
                    "subject": fact["subject"],
                    "object": fact["object"],
                    "occurrence_count": len(group),
                    "source_record_sha256_values": source_hashes,
                }
            )

    diagnostics = {
        "raw_forget_row_count": len(rows),
        "unique_association_count": len(facts),
        "duplicate_records_collapsed": len(rows) - len(facts),
        "duplicate_association_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
        "natural_address_conflict_count": 0,
        "dedup_key": "normalized(subject), normalized(query), normalized(answer)",
        "record_to_fact_id": record_to_fact_id,
    }
    return facts, record_to_fact_id, diagnostics


@dataclass(frozen=True)
class DirectTokenTrainingCase:
    id: str
    fact_id: str
    source_record_sha256: str
    token_index: int
    input_ids: list[int]
    boundary_length: int
    target_token_id: int


def build_exact_direct_token_cases(rows, facts, tokenizer):
    fact_by_key = {str(f["association_key"]): f for f in facts}
    cases = []
    for row in rows:
        key = association_key_from_row(row)
        fact = fact_by_key[key]
        prompt = rwku.format_qa_prompt(tokenizer, row)
        prompt_ids = rwku._token_ids(tokenizer, prompt, add_special_tokens=True)
        answer_ids = rwku._token_ids(
            tokenizer,
            rwku._normalized_completion(str(row["answer"])),
            add_special_tokens=False,
        )
        if not prompt_ids or not answer_ids:
            raise ValueError("RWKU prompt/answer tokenization produced empty IDs")
        source_hash = str(row.get("source_record_sha256", ""))
        for token_index, token_id in enumerate(answer_ids):
            cases.append(
                DirectTokenTrainingCase(
                    id=f"{fact['id']}:{source_hash}:token_{token_index}",
                    fact_id=fact["id"],
                    source_record_sha256=source_hash,
                    token_index=token_index,
                    input_ids=[*prompt_ids, *answer_ids[:token_index]],
                    boundary_length=len(prompt_ids),
                    target_token_id=int(token_id),
                )
            )
    return cases


@torch.no_grad()
def build_direct_context_keys(model, tokenizer, facts, layer):
    prompts = [fact["canonical_prompt"] for fact in facts]
    keys = extract_prompt_queries(model, tokenizer, prompts, layer).float()
    keys = F.normalize(keys, dim=-1)
    thresholds = torch.full((len(facts),), -1.0, dtype=torch.float32)

    subjects = defaultdict(list)
    for index, fact in enumerate(facts):
        subjects[normalized(fact["subject"])].append(index)
    duplicate_subject_groups = {
        subject: indices for subject, indices in subjects.items() if len(indices) > 1
    }
    return keys, thresholds, {
        "key_source": "frozen layer-19 query for each selected RWKU training prompt",
        "threshold": -1.0,
        "routing": "complete subject eligibility + nearest frozen contextual key",
        "duplicate_subject_groups": duplicate_subject_groups,
        "all_training_subjects_have_multiple_candidates": all(
            len(indices) > 1 for indices in subjects.values()
        ),
        "unique_association_count": len(facts),
        "heldout_rwku_probes_used": False,
        "neighbor_or_utility_used": False,
    }


def build_editor(base_model, tokenizer, facts, plan):
    keys, thresholds, diagnostics = build_direct_context_keys(
        base_model, tokenizer, facts, int(plan["layer"])
    )
    bank = FactAssociationBank(
        base_model=base_model,
        layer=int(plan["layer"]),
        keys=keys,
        thresholds=thresholds,
        subject_patterns=make_subject_patterns(tokenizer, facts),
        facts=facts,
    )
    return FactAssociationEditor(base_model, bank), bank, diagnostics


def _batch_cases(cases, tokenizer, device):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer needs pad_token_id or eos_token_id")
    widths = [len(case.input_ids) for case in cases]
    width = max(widths)
    input_ids = torch.full(
        (len(cases), width), int(pad_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for i, case in enumerate(cases):
        ids = torch.tensor(case.input_ids, dtype=torch.long, device=device)
        input_ids[i, : len(ids)] = ids
        attention_mask[i, : len(ids)] = 1
    prefix_lengths = [int(case.boundary_length) for case in cases]
    targets = torch.tensor(
        [int(case.target_token_id) for case in cases],
        dtype=torch.long,
        device=device,
    )
    return input_ids, attention_mask, widths, prefix_lengths, targets


def sensitive_token_state(model, tokenizer, cases, target_probability):
    if not cases:
        raise ValueError("Cannot score empty RWKU token cases")
    device = next(model.parameters()).device
    input_ids, attention_mask, widths, prefix_lengths, targets = _batch_cases(
        cases, tokenizer, device
    )
    model.set_association_prefix_lengths(prefix_lengths)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    positions = torch.tensor([w - 1 for w in widths], device=device)
    batch_indices = torch.arange(len(cases), device=device)
    final_logits = output.logits[batch_indices, positions, :].float()

    token_nll = F.cross_entropy(final_logits, targets, reduction="none")
    token_probability = torch.exp(-token_nll)
    worst_index = int(token_probability.detach().argmax().item())
    worst_probability = token_probability[worst_index]
    target_nll = worst_probability.new_tensor(-math.log(float(target_probability)))
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
def direct_training_metrics(model, tokenizer, cases_by_fact, target_probability):
    rows = []
    for fact_id in sorted(cases_by_fact):
        state = sensitive_token_state(
            model, tokenizer, cases_by_fact[fact_id], target_probability
        )
        rows.append(
            {
                "fact_id": fact_id,
                "max_sensitive_token_probability": float(
                    state["max_token_probability"].detach()
                ),
                "correct_sensitive_tokens": state["correct_sensitive_tokens"],
                "total_sensitive_tokens": state["total_sensitive_tokens"],
                "all_sensitive_tokens_not_top1": state[
                    "all_sensitive_tokens_not_top1"
                ],
                "max_greedy_margin": float(state["max_greedy_margin"].detach()),
                "mean_token_nll": float(state["mean_token_nll"].detach()),
                "min_token_nll": float(state["min_token_nll"].detach()),
            }
        )
    failing = [
        row["fact_id"]
        for row in rows
        if row["max_sensitive_token_probability"] >= float(target_probability)
    ]
    total_tokens = sum(row["total_sensitive_tokens"] for row in rows)
    correct_tokens = sum(row["correct_sensitive_tokens"] for row in rows)
    return {
        "metric_definition": (
            "maximum teacher-forced probability among every sensitive answer "
            "token on each selected RWKU Batch-50 training probe"
        ),
        "target_token_probability": float(target_probability),
        "facts_total": len(rows),
        "facts_passing_probability_constraint": len(rows) - len(failing),
        "failing_fact_ids": failing,
        "maximum_sensitive_token_probability": max(
            row["max_sensitive_token_probability"] for row in rows
        ),
        "sensitive_correct_tokens": correct_tokens,
        "sensitive_total_tokens": total_tokens,
        "direct_micro_accuracy_percent": 100.0 * correct_tokens / total_tokens,
        "all_facts_zero_direct_token_accuracy": all(
            row["all_sensitive_tokens_not_top1"] for row in rows
        ),
        "worst_facts": sorted(
            rows, key=lambda row: -row["max_sensitive_token_probability"]
        )[:10],
        "globally_feasible": not failing,
    }


def _row_state(editor):
    return torch.stack([row.detach().cpu().clone() for row in editor.embedding.rows])


def _restore_rows(editor, state):
    with torch.no_grad():
        for parameter, value in zip(editor.embedding.rows, state):
            parameter.copy_(value.to(parameter.device, parameter.dtype))


def train_direct_only(editor, tokenizer, token_cases, fact_to_row, plan, output):
    by_fact = defaultdict(list)
    for case in token_cases:
        by_fact[case.fact_id].append(case)
    facts = sorted(by_fact)
    if set(facts) != set(fact_to_row):
        raise ValueError("Every RWKU association must own exactly one trainable row")

    steps = int(plan["steps"])
    check_every = int(plan["check_every"])
    if steps % len(facts) or check_every % len(facts):
        raise ValueError("RWKU checkpoints must end on complete association sweeps")

    order = list(facts)
    random.Random(int(plan["seed"])).shuffle(order)
    optimizers = {
        fact_id: torch.optim.Adam(
            [editor.embedding.rows[fact_to_row[fact_id]]],
            lr=float(plan["learning_rate"]),
        )
        for fact_id in facts
    }

    baseline = direct_training_metrics(
        editor.model, tokenizer, by_fact, plan["target_token_probability"]
    )
    best_metric = baseline["maximum_sensitive_token_probability"]
    best_state = _row_state(editor)
    best_step = 0
    history = []
    gates = [
        {
            "step": 0,
            "metrics": baseline,
            "selected_as_best": True,
            "best_step": 0,
        }
    ]
    torch.save(editor.artifact(), output / "best_fact_association_rows.pt")

    started = time.monotonic()
    rejected = 0
    feasible_gates = 0
    stop_reason = "row_step_budget"

    for step in range(1, steps + 1):
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
        )
        before_probability = float(before["max_token_probability"].detach())

        if before_probability < float(plan["target_token_probability"]):
            history.append(
                {
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
            )
        else:
            before["loss"].backward()
            torch.nn.utils.clip_grad_norm_([row], 1.0, error_if_nonfinite=True)
            optimizer.step()

            proposal = row.detach() - before_row
            radius = radius_for_probability(
                before_probability, plan["radius_schedule"]
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
                    )
                after_probability = float(after["max_token_probability"].detach())
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

            history.append(
                {
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
                    "before_correct_sensitive_tokens": before[
                        "correct_sensitive_tokens"
                    ],
                    "after_correct_sensitive_tokens": after[
                        "correct_sensitive_tokens"
                    ],
                    "step_norm": float((row.detach() - before_row).norm()),
                    "elapsed_seconds": time.monotonic() - started,
                }
            )

        if step % check_every == 0:
            metrics = direct_training_metrics(
                editor.model,
                tokenizer,
                by_fact,
                plan["target_token_probability"],
            )
            current = metrics["maximum_sensitive_token_probability"]
            selected = current < best_metric
            if selected:
                best_metric = current
                best_state = _row_state(editor)
                best_step = step
                torch.save(
                    editor.artifact(), output / "best_fact_association_rows.pt"
                )
            feasible_gates = (
                feasible_gates + 1 if metrics["globally_feasible"] else 0
            )
            gates.append(
                {
                    "step": step,
                    "metrics": metrics,
                    "selected_as_best": selected,
                    "best_step": best_step,
                    "consecutive_feasible_gates": feasible_gates,
                }
            )
            torch.save(
                editor.artifact(), output / "last_fact_association_rows.pt"
            )
            if (
                metrics["globally_feasible"]
                and feasible_gates > int(plan["post_feasible_gates"])
            ):
                stop_reason = (
                    "all_direct_sensitive_tokens_below_probability_threshold"
                )
                break

        if rejected >= int(plan["max_stalled_steps"]):
            stop_reason = "consecutive_rejected_row_steps"
            break

    _restore_rows(editor, best_state)
    final = direct_training_metrics(
        editor.model, tokenizer, by_fact, plan["target_token_probability"]
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
