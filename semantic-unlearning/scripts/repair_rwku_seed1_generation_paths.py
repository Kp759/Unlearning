#!/usr/bin/env python3
"""Greedy recovery-prefix repair for RWKU Batch-50 seed 1.

Starting from a completed direct-only RWKU residual-bank run, this stage uses
ONLY the same 50 training/efficacy probes. For any training prompt whose greedy
continuation still reconstructs the sensitive answer, it creates one dynamic
training case at the exact generated prefix immediately before the token that
first makes the answer recoverable. Only that fact's residual row is updated.

No held-out Level-1/2, paraphrase, Level-3, neighbor, MIA, utility, or PPL data
is read for training or checkpoint selection.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import torch

import rwku_eval as rwku
from rwku_batch50 import build_batch_split
from rwku_fact_association_embeddings import (
    BASE_PLAN,
    DirectTokenTrainingCase,
    association_key_from_row,
    build_association_facts,
    build_exact_direct_token_cases,
    sensitive_token_state,
)
from static_overlap_extended_tokens_v2 import radius_for_probability
from static_overlap_fact_association_embeddings import (
    FactAssociationBank,
    FactAssociationEditor,
)


def _row_state(editor):
    return torch.stack(
        [row.detach().cpu().clone() for row in editor.embedding.rows]
    )


def _restore_rows(editor, state):
    with torch.no_grad():
        for parameter, value in zip(editor.embedding.rows, state):
            parameter.copy_(value.to(parameter.device, parameter.dtype))


@torch.no_grad()
def first_recovery_case(
    model,
    bank,
    tokenizer,
    row,
    fact_id,
    expected_row,
    *,
    repair_round,
    max_new_tokens=30,
):
    """Return the first greedy token whose addition makes the answer recoverable."""
    device = next(model.parameters()).device
    prompt = rwku.format_qa_prompt(tokenizer, row)
    prompt_ids = rwku._token_ids(tokenizer, prompt, add_special_tokens=True)
    if not prompt_ids:
        raise ValueError("RWKU repair prompt tokenization is empty")

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    prompt_length = len(prompt_ids)
    eos = tokenizer.eos_token_id

    for step in range(int(max_new_tokens)):
        model.set_association_prefix_lengths([prompt_length])
        output = model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
        )
        route = list(bank.last_active_fact_indices[0])
        if route != [int(expected_row)]:
            raise RuntimeError(
                "RWKU recovery repair requires exact route stability; "
                f"expected {[expected_row]}, observed {route}"
            )
        next_token = int(output.logits[:, -1, :].argmax(-1).item())
        before_ids = ids[0].detach().cpu().tolist()

        ids = torch.cat(
            [
                ids,
                torch.tensor([[next_token]], dtype=torch.long, device=device),
            ],
            dim=1,
        )
        mask = torch.ones_like(ids)
        continuation = tokenizer.decode(
            ids[0, prompt_length:],
            skip_special_tokens=True,
        ).strip()

        if rwku.recovery_success(continuation, str(row["answer"])):
            source_hash = str(row.get("source_record_sha256", ""))
            case = DirectTokenTrainingCase(
                id=(
                    f"{fact_id}:{source_hash}:rollout_round_{repair_round}"
                    f":recovery_step_{step}"
                ),
                fact_id=fact_id,
                source_record_sha256=source_hash,
                token_index=int(step),
                input_ids=before_ids,
                boundary_length=prompt_length,
                target_token_id=next_token,
            )
            return {
                "case": case,
                "subject": str(row["subject"]),
                "query": str(row["query"]),
                "answer": str(row["answer"]),
                "generated_prefix_before_target": tokenizer.decode(
                    before_ids[prompt_length:],
                    skip_special_tokens=True,
                ).strip(),
                "recovery_token_id": next_token,
                "recovery_token": tokenizer.decode([next_token]),
                "recovery_step": int(step),
                "prediction_at_recovery": continuation,
                "active_row": int(expected_row),
            }

        if eos is not None and next_token == int(eos):
            break
    return None


@torch.no_grad()
def audit_same50_recoveries(
    model,
    bank,
    tokenizer,
    rows,
    record_to_fact_id,
    fact_to_row,
    *,
    repair_round,
    max_new_tokens,
):
    recoveries = []
    for row in rows:
        source_hash = str(row["source_record_sha256"])
        fact_id = record_to_fact_id[source_hash]
        item = first_recovery_case(
            model,
            bank,
            tokenizer,
            row,
            fact_id,
            fact_to_row[fact_id],
            repair_round=repair_round,
            max_new_tokens=max_new_tokens,
        )
        if item is not None:
            recoveries.append(item)
    return recoveries


def _direct_cases_by_fact(cases):
    out = {}
    for case in cases:
        out.setdefault(case.fact_id, []).append(case)
    return out


def repair_one_case(
    editor,
    tokenizer,
    repair_case,
    direct_cases,
    optimizer,
    plan,
    *,
    updates,
    direct_probability_cap,
):
    """Reduce one recovery-completing token while preserving direct top-1 suppression."""
    row_index = int(plan["fact_to_row"][repair_case.fact_id])
    row = editor.embedding.rows[row_index]
    history = []

    for update in range(1, int(updates) + 1):
        editor.model.zero_grad(set_to_none=True)
        before_repair = sensitive_token_state(
            editor.model,
            tokenizer,
            [repair_case],
            plan["repair_target_probability"],
        )
        before_direct = sensitive_token_state(
            editor.model,
            tokenizer,
            direct_cases,
            plan["target_token_probability"],
        )
        before_probability = float(
            before_repair["max_token_probability"].detach()
        )
        before_direct_probability = float(
            before_direct["max_token_probability"].detach()
        )

        before_row = row.detach().clone()
        optimizer_state = deepcopy(optimizer.state_dict())

        # Use the same probability-gap objective as the direct stage, with a
        # tighter target because the selected token is currently on a greedy
        # recovery path.
        before_repair["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            [row], 1.0, error_if_nonfinite=True
        )
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
                after_repair = sensitive_token_state(
                    editor.model,
                    tokenizer,
                    [repair_case],
                    plan["repair_target_probability"],
                )
                after_direct = sensitive_token_state(
                    editor.model,
                    tokenizer,
                    direct_cases,
                    plan["target_token_probability"],
                )
            repair_probability = float(
                after_repair["max_token_probability"].detach()
            )
            direct_probability = float(
                after_direct["max_token_probability"].detach()
            )
            direct_cap = float(direct_probability_cap)
            acceptable = (
                math.isfinite(repair_probability)
                and repair_probability < before_probability
                and bool(after_direct["all_sensitive_tokens_not_top1"])
                and direct_probability <= direct_cap
            )
            if acceptable:
                candidates.append(
                    (
                        repair_probability,
                        direct_probability,
                        backtracks,
                        row.detach().clone(),
                        after_repair,
                        after_direct,
                    )
                )

        if not candidates:
            with torch.no_grad():
                row.copy_(before_row)
            optimizer.load_state_dict(optimizer_state)
            history.append(
                {
                    "update": update,
                    "accepted": False,
                    "before_recovery_token_probability": before_probability,
                    "before_direct_max_probability": before_direct_probability,
                    "reason": "no_backtrack_preserved_direct_guard",
                }
            )
            break

        (
            after_probability,
            after_direct_probability,
            accepted_backtracks,
            accepted_row,
            after_repair,
            after_direct,
        ) = min(candidates, key=lambda value: (value[0], value[1]))
        with torch.no_grad():
            row.copy_(accepted_row)

        history.append(
            {
                "update": update,
                "accepted": True,
                "backtracks": int(accepted_backtracks),
                "radius": float(radius),
                "before_recovery_token_probability": before_probability,
                "after_recovery_token_probability": after_probability,
                "before_direct_max_probability": before_direct_probability,
                "after_direct_max_probability": after_direct_probability,
                "direct_all_sensitive_tokens_not_top1": bool(
                    after_direct["all_sensitive_tokens_not_top1"]
                ),
                "step_norm": float((row.detach() - before_row).norm()),
            }
        )

        if after_probability < float(plan["repair_target_probability"]):
            break

    return history


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-run-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-root", default="data/rwku")
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--updates-per-recovery", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--repair-target-probability", type=float, default=1e-8)
    p.add_argument("--direct-probability-multiplier", type=float, default=1.25)
    p.add_argument("--max-training-seconds", type=float, default=3600.0)
    args = p.parse_args(argv)

    parent = Path(args.parent_run_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite RWKU repair run: {output}")
    output.mkdir(parents=True)

    parent_manifest = json.loads(
        (parent / "association_manifest.json").read_text()
    )
    artifact = torch.load(
        parent / "fact_association_embeddings.pt",
        map_location="cpu",
        weights_only=False,
    )
    if int(parent_manifest.get("seed", -1)) != 1:
        raise ValueError("RWKU recovery repair is registered only for seed 1")

    split = build_batch_split(
        data_root=Path(args.data_root).resolve(),
        batch_seed=1,
        allow_download=not args.no_download,
    )
    rows = list(split["efficacy_forget"])
    if len(rows) != 50:
        raise RuntimeError("RWKU Batch-50 seed 1 must contain exactly 50 efficacy rows")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(parent_manifest["model_path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    expected_facts, record_to_fact_id, dedup = build_association_facts(
        rows, tokenizer
    )
    expected_keys = [str(f["association_key"]) for f in expected_facts]
    artifact_keys = [str(f.get("association_key")) for f in artifact["facts"]]
    if expected_keys != artifact_keys:
        raise RuntimeError(
            "Parent artifact does not match the frozen RWKU seed-1 Batch-50 split"
        )

    torch.manual_seed(1)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    base_model.requires_grad_(False)
    base_model.config.use_cache = False

    bank = FactAssociationBank(
        base_model=base_model,
        layer=int(artifact["layer"]),
        keys=artifact["keys"],
        thresholds=artifact["thresholds"],
        subject_patterns=artifact["subject_patterns"],
        facts=artifact["facts"],
        rows=artifact["rows"],
    )
    editor = FactAssociationEditor(base_model, bank)
    fact_to_row = {
        fact["id"]: index for index, fact in enumerate(expected_facts)
    }
    direct_cases = build_exact_direct_token_cases(
        rows, expected_facts, tokenizer
    )
    direct_by_fact = _direct_cases_by_fact(direct_cases)
    initial_direct_probability_caps = {}
    for fact_id, cases in direct_by_fact.items():
        state = sensitive_token_state(
            editor.model,
            tokenizer,
            cases,
            float(BASE_PLAN["target_token_probability"]),
        )
        initial_probability = float(
            state["max_token_probability"].detach()
        )
        initial_direct_probability_caps[fact_id] = max(
            1e-6,
            initial_probability * float(args.direct_probability_multiplier),
        )

    plan = dict(BASE_PLAN)
    plan.update(
        {
            "fact_to_row": fact_to_row,
            "repair_target_probability": float(
                args.repair_target_probability
            ),
        }
    )
    optimizers = {
        fact_id: torch.optim.Adam(
            [editor.embedding.rows[row_index]],
            lr=float(plan["learning_rate"]),
        )
        for fact_id, row_index in fact_to_row.items()
    }

    started = time.monotonic()
    best_state = _row_state(editor)
    initial_recoveries = audit_same50_recoveries(
        editor.model,
        bank,
        tokenizer,
        rows,
        record_to_fact_id,
        fact_to_row,
        repair_round=0,
        max_new_tokens=args.max_new_tokens,
    )
    best_recovery_count = len(initial_recoveries)
    best_round = 0
    rounds = []
    dynamic_cases = []

    print(
        json.dumps(
            {
                "status": "rwku_rollout_repair_preflight",
                "same50_recoveries_before_repair": best_recovery_count,
                "recovery_cases": [
                    {
                        "subject": item["subject"],
                        "query": item["query"],
                        "answer": item["answer"],
                        "recovery_step": item["recovery_step"],
                        "generated_prefix_before_target": item[
                            "generated_prefix_before_target"
                        ],
                        "recovery_token": item["recovery_token"],
                    }
                    for item in initial_recoveries
                ],
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )

    current_recoveries = initial_recoveries
    for repair_round in range(1, int(args.max_rounds) + 1):
        if not current_recoveries:
            break
        if time.monotonic() - started >= float(args.max_training_seconds):
            break

        order = list(current_recoveries)
        random.Random(1000 + repair_round).shuffle(order)
        round_history = []
        for item in order:
            case = item["case"]
            dynamic_cases.append(case)
            history = repair_one_case(
                editor,
                tokenizer,
                case,
                direct_by_fact[case.fact_id],
                optimizers[case.fact_id],
                plan,
                updates=args.updates_per_recovery,
                direct_probability_cap=initial_direct_probability_caps[
                    case.fact_id
                ],
            )
            round_history.append(
                {
                    "fact_id": case.fact_id,
                    "source_record_sha256": case.source_record_sha256,
                    "subject": item["subject"],
                    "query": item["query"],
                    "answer": item["answer"],
                    "recovery_step": item["recovery_step"],
                    "generated_prefix_before_target": item[
                        "generated_prefix_before_target"
                    ],
                    "recovery_token": item["recovery_token"],
                    "updates": history,
                }
            )

        after = audit_same50_recoveries(
            editor.model,
            bank,
            tokenizer,
            rows,
            record_to_fact_id,
            fact_to_row,
            repair_round=repair_round,
            max_new_tokens=args.max_new_tokens,
        )
        direct_guard_failures = []
        for fact_id, cases in direct_by_fact.items():
            state = sensitive_token_state(
                editor.model,
                tokenizer,
                cases,
                plan["target_token_probability"],
            )
            if not bool(state["all_sensitive_tokens_not_top1"]):
                direct_guard_failures.append(fact_id)
        if direct_guard_failures:
            raise RuntimeError(
                "Rollout repair violated original direct top-1 suppression: "
                f"{direct_guard_failures}"
            )

        round_summary = {
            "round": repair_round,
            "recoveries_before": len(current_recoveries),
            "recoveries_after": len(after),
            "direct_guard_failure_count": 0,
            "repair_history": round_history,
            "remaining_recoveries": [
                {
                    "subject": item["subject"],
                    "query": item["query"],
                    "answer": item["answer"],
                    "recovery_step": item["recovery_step"],
                    "generated_prefix_before_target": item[
                        "generated_prefix_before_target"
                    ],
                    "recovery_token": item["recovery_token"],
                }
                for item in after
            ],
        }
        rounds.append(round_summary)
        print(
            json.dumps(
                {
                    "repair_round": repair_round,
                    "recoveries_before": len(current_recoveries),
                    "recoveries_after": len(after),
                    "remaining": round_summary["remaining_recoveries"],
                },
                indent=2,
                allow_nan=False,
            ),
            flush=True,
        )

        if len(after) < best_recovery_count:
            best_recovery_count = len(after)
            best_state = _row_state(editor)
            best_round = repair_round

        current_recoveries = after

    _restore_rows(editor, best_state)
    final_recoveries = audit_same50_recoveries(
        editor.model,
        bank,
        tokenizer,
        rows,
        record_to_fact_id,
        fact_to_row,
        repair_round=best_round,
        max_new_tokens=args.max_new_tokens,
    )

    final_direct = {}
    for fact_id, cases in direct_by_fact.items():
        state = sensitive_token_state(
            editor.model,
            tokenizer,
            cases,
            plan["target_token_probability"],
        )
        final_direct[fact_id] = {
            "max_sensitive_token_probability": float(
                state["max_token_probability"].detach()
            ),
            "all_sensitive_tokens_not_top1": bool(
                state["all_sensitive_tokens_not_top1"]
            ),
            "correct_sensitive_tokens": int(
                state["correct_sensitive_tokens"]
            ),
            "total_sensitive_tokens": int(state["total_sensitive_tokens"]),
        }

    repaired_artifact = editor.artifact()
    repaired_artifact.update(
        {
            "method": "fact_association_embeddings_rwku_batch50_rollout_repair_v1",
            "dataset": "RWKU",
            "protocol_id": parent_manifest["protocol_id"],
            "seed": 1,
            "target_seeds": split["manifest"]["target_seeds"],
            "forget_train_count": len(rows),
            "unique_forget_association_count": len(expected_facts),
            "association_deduplication": dedup,
            "source_record_to_association_id": record_to_fact_id,
            "parent_artifact": str(
                parent / "fact_association_embeddings.pt"
            ),
            "rollout_repair_used_only_same50_training_probes": True,
            "heldout_rwku_probes_used_for_training_or_selection": False,
            "neighbor_mia_utility_used_for_training_or_selection": False,
        }
    )
    torch.save(
        repaired_artifact, output / "fact_association_embeddings.pt"
    )

    manifest = dict(parent_manifest)
    manifest.update(
        {
            "method": repaired_artifact["method"],
            "parent_run_dir": str(parent),
            "objective": (
                "direct sensitive-token suppression plus dynamic greedy "
                "recovery-prefix hardening on the same 50 training probes"
            ),
            "rollout_repair": {
                "max_rounds": int(args.max_rounds),
                "updates_per_recovery": int(args.updates_per_recovery),
                "max_new_tokens": int(args.max_new_tokens),
                "repair_target_probability": float(
                    args.repair_target_probability
                ),
                "direct_probability_multiplier": float(
                    args.direct_probability_multiplier
                ),
                "direct_probability_caps_are_fixed_from_parent": True,
                "initial_same50_recoveries": len(initial_recoveries),
                "best_same50_recoveries": best_recovery_count,
                "best_round": best_round,
                "heldout_or_neighbor_data_used": False,
            },
            "official_evaluation_started": False,
        }
    )
    (output / "association_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    (output / "rollout_repair_cases.json").write_text(
        json.dumps(
            [asdict(case) for case in dynamic_cases],
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )

    report = {
        "method": repaired_artifact["method"],
        "seed": 1,
        "parent_run_dir": str(parent),
        "initial_same50_recoveries": len(initial_recoveries),
        "best_same50_recoveries": best_recovery_count,
        "best_round": best_round,
        "restored_best_round": True,
        "final_same50_recoveries": len(final_recoveries),
        "rounds": rounds,
        "final_direct_training_metrics_by_fact": final_direct,
        "all_direct_sensitive_tokens_not_top1": all(
            value["all_sensitive_tokens_not_top1"]
            for value in final_direct.values()
        ),
        "elapsed_seconds": time.monotonic() - started,
        "official_evaluation_started": False,
    }
    (output / "rollout_repair_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )

    print(
        json.dumps(
            {
                "status": "rwku_rollout_repair_complete",
                "initial_same50_recoveries": len(initial_recoveries),
                "best_same50_recoveries": best_recovery_count,
                "best_round": best_round,
                "final_same50_recoveries": len(final_recoveries),
                "all_direct_sensitive_tokens_not_top1": report[
                    "all_direct_sensitive_tokens_not_top1"
                ],
                "artifact": str(
                    output / "fact_association_embeddings.pt"
                ),
                "report": str(output / "rollout_repair_report.json"),
                "official_evaluation_started": False,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
