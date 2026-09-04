#!/usr/bin/env python3
"""Train V1.4 relation-scoped private-null routing on the locked MCF dev split.

Stage 1 learns a small per-subject binary relation gate from frozen-Base prompt
features.  Positive examples are the exact locked V1.3 five-view training
prompts.  Negatives are training-visible protection-fit templates with a
different relation id.  No official paraphrase/neighborhood/evaluation text is
read.

Stage 2 freezes the gate and trains only the same-length private subject input
rows.  Gate-positive sensitive prompts route to the private rows and are trained
toward the natural abstention phrase "I don't know" while suppressing the true
object.  Gate-negative prompts use untouched Base token ids and therefore bypass
the edit entirely.

This is relation-aware behavioral routing/unlearning, not latent knowledge
erasure.  The Transformer, LM head, and every non-private input row stay frozen.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import mcf_private_vocab_rewiring_v1_core as v1core
import mcf_private_vocab_rewiring_v1_1_core as pp
import mcf_private_vocab_rewiring_v1_4_relation_null_core as nullcore
import run_mcf_private_vocab_rewiring_v1_1 as v11
import run_mcf_private_vocab_rewiring_v1_3_multiview as v13


PROTOCOL = nullcore.PROTOCOL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--protocol-dir", required=True)
    p.add_argument("--view-corpus", required=True)
    p.add_argument("--experiment-registry", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--gate-negatives-per-case", type=int, default=16)
    p.add_argument("--gate-steps", type=int, default=600)
    p.add_argument("--gate-learning-rate", type=float, default=0.03)
    p.add_argument("--gate-weight-decay", type=float, default=0.01)
    p.add_argument("--gate-feature-batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--forget-batch-size", type=int, default=8)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--minimum-abstention-margin", type=float, default=0.1)
    p.add_argument("--minimum-true-suppression", type=float, default=2.0)
    p.add_argument("--true-suppression-weight", type=float, default=1.0)
    p.add_argument("--anchor-weight", type=float, default=0.001)
    p.add_argument("--relative-row-cap", type=float, default=0.5)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--initial-equivalence-kl-max", type=float, default=1e-7)
    p.add_argument("--save-model", action="store_true")
    args = p.parse_args()
    if args.seed != 1 or args.forget_num != 50:
        p.error("V1.4 development protocol is locked to consumed seed 1 / 50 forget facts")
    if args.gate_negatives_per_case <= 0 or args.gate_steps <= 0:
        p.error("gate configuration must be positive")
    if args.steps <= 0 or args.forget_batch_size <= 0 or args.check_every <= 0:
        p.error("training configuration must be positive")
    if args.minimum_abstention_margin < 0 or args.minimum_true_suppression < 0:
        p.error("registered null/suppression thresholds must be non-negative")
    if args.relative_row_cap <= 0:
        p.error("relative row cap must be positive")
    return args


def validate_registry(registry: Mapping[str, Any], args: argparse.Namespace) -> None:
    arch = registry.get("architecture", {})
    gate = registry.get("relation_gate", {})
    objective = registry.get("null_objective", {})
    acceptance = registry.get("acceptance", {})
    expected = {
        "private_subject_embedding_rows",
        "per_subject_relation_gate",
    }
    if (
        registry.get("protocol") != PROTOCOL
        or set(arch.get("trainable_parameter_families", [])) != expected
        or arch.get("transformer_frozen") is not True
        or arch.get("lm_head_frozen_bit_identical") is not True
        or arch.get("original_input_embedding_rows_frozen") is not True
        or arch.get("subject_token_count_preserved") is not True
        or arch.get("relation_aware_router") is not True
        or arch.get("non_sensitive_path") != "exact_base_token_ids"
        or gate.get("official_eval_text_used") is not False
        or gate.get("positive_source") != "locked_v1_3_five_view_training_corpus"
        or gate.get("negative_source") != "training_visible_protection_fit_different_relation"
        or objective.get("target_new_gradient") is not False
        or objective.get("abstention_text") != nullcore.ABSTENTION_TEXT
        or float(acceptance.get("minimum_abstention_minus_true_margin", -1))
        != float(args.minimum_abstention_margin)
        or float(acceptance.get("minimum_true_logprob_drop", -1))
        != float(args.minimum_true_suppression)
    ):
        raise RuntimeError("V1.4 registry contract mismatch")


def view_records_for_case(
    record: Mapping[str, Any], view_map: Mapping[int, Sequence[str]]
) -> list[Dict[str, Any]]:
    cid = int(record["case_id"])
    templates = list(view_map.get(cid, []))
    if len(templates) != 5:
        raise RuntimeError(f"case {cid} does not have exactly five locked views")
    out: list[Dict[str, Any]] = []
    for index, template in enumerate(templates):
        clone = copy.deepcopy(dict(record))
        clone["requested_rewrite"]["prompt"] = str(template)
        clone["_v1_4_view_index"] = int(index)
        out.append(clone)
    return out


def render_record_prompt(record: Mapping[str, Any]) -> str:
    return v11.v1.render_prompt(record)


def sequence_logprob_batch(
    model: Any,
    base_tokenizer: Any,
    mapping_by_subject: Mapping[str, Mapping[str, Any]],
    prompts: Sequence[str],
    answers: Sequence[str],
    subjects: Sequence[str],
    *,
    route_private: bool,
    device: torch.device,
) -> torch.Tensor:
    """Mean answer-token logprob with optional subject-scoped private routing."""
    rows: list[list[int]] = []
    starts: list[int] = []
    targets: list[list[int]] = []
    bos = getattr(base_tokenizer, "bos_token_id", None)
    for prompt, answer, subject in zip(prompts, answers, subjects):
        pids = base_tokenizer(
            prompt, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        pids = [int(v) for v in pids]
        if route_private:
            item = mapping_by_subject.get(str(subject))
            if item is None:
                raise RuntimeError(f"missing private mapping for subject {subject!r}")
            rewritten = pp._rewrite_ids(pids, [item])
            if rewritten == pids:
                raise RuntimeError(
                    f"relation-null private route did not find subject {subject!r} in prompt {prompt!r}"
                )
            pids = rewritten
        if bos is not None:
            pids = [int(bos)] + pids
        aids = v11.answer_ids(base_tokenizer, answer)
        rows.append(pids + aids)
        starts.append(len(pids))
        targets.append(aids)

    max_len = max(len(row) for row in rows)
    pad = getattr(base_tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(base_tokenizer, "eos_token_id", None)
    if pad is None:
        pad = 0
    input_ids = torch.full(
        (len(rows), max_len), int(pad), device=device, dtype=torch.long
    )
    attention = torch.zeros_like(input_ids)
    for i, row in enumerate(rows):
        input_ids[i, : len(row)] = torch.tensor(row, device=device, dtype=torch.long)
        attention[i, : len(row)] = 1
    logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
    log_probs = F.log_softmax(logits.float(), dim=-1)
    values: list[torch.Tensor] = []
    for i, (start, aids) in enumerate(zip(starts, targets)):
        positions = torch.arange(
            start - 1, start - 1 + len(aids), device=device, dtype=torch.long
        )
        token_ids = torch.tensor(aids, device=device, dtype=torch.long)
        values.append(log_probs[i, positions, token_ids].mean())
    return torch.stack(values)


def flatten_views(
    records: Sequence[Mapping[str, Any]],
    view_map: Mapping[int, Sequence[str]],
) -> tuple[list[Dict[str, Any]], list[tuple[int, int]]]:
    flat: list[Dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    for record in records:
        start = len(flat)
        local = view_records_for_case(record, view_map)
        flat.extend(local)
        spans.append((start, len(flat)))
    return flat, spans


def batch_null_metrics(
    model: Any,
    tokenizer: Any,
    mapping_by_subject: Mapping[str, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    view_map: Mapping[int, Sequence[str]],
    base_true_lp: Mapping[tuple[int, int], float],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat, spans = flatten_views(records, view_map)
    prompts = [render_record_prompt(row) for row in flat]
    subjects = [str(row["requested_rewrite"]["subject"]) for row in flat]
    true_answers = [str(row["requested_rewrite"]["target_true"]["str"]) for row in flat]
    abstention_answers = [nullcore.ABSTENTION_TEXT] * len(flat)

    abstention_lp = sequence_logprob_batch(
        model,
        tokenizer,
        mapping_by_subject,
        prompts,
        abstention_answers,
        subjects,
        route_private=True,
        device=device,
    )
    true_lp = sequence_logprob_batch(
        model,
        tokenizer,
        mapping_by_subject,
        prompts,
        true_answers,
        subjects,
        route_private=True,
        device=device,
    )
    baseline = torch.tensor(
        [
            float(base_true_lp[(int(row["case_id"]), int(row["_v1_4_view_index"]))])
            for row in flat
        ],
        device=device,
        dtype=true_lp.dtype,
    )
    margin = abstention_lp - true_lp
    suppression = baseline - true_lp
    worst_margin = torch.stack([margin[start:stop].min() for start, stop in spans])
    worst_suppression = torch.stack(
        [suppression[start:stop].min() for start, stop in spans]
    )
    return worst_margin, worst_suppression, margin, suppression


def summarize(values: Sequence[float], threshold: float) -> Dict[str, Any]:
    passed = sum(float(value) >= float(threshold) for value in values)
    return {
        "count": len(values),
        "minimum": min(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "passed": int(passed),
        "failures": int(len(values) - passed),
        "threshold": float(threshold),
    }


@torch.no_grad()
def evaluate_null_metrics(
    model: Any,
    tokenizer: Any,
    mapping_by_subject: Mapping[str, Mapping[str, Any]],
    forget: Sequence[Mapping[str, Any]],
    view_map: Mapping[int, Sequence[str]],
    base_true_lp: Mapping[tuple[int, int], float],
    *,
    device: torch.device,
    batch_size: int,
    margin_threshold: float,
    suppression_threshold: float,
) -> tuple[Dict[str, Any], Dict[str, Any], list[Dict[str, Any]]]:
    margins: list[float] = []
    suppressions: list[float] = []
    per_case: list[Dict[str, Any]] = []
    for start in range(0, len(forget), int(batch_size)):
        batch = forget[start : start + int(batch_size)]
        worst_margin, worst_suppression, _, _ = batch_null_metrics(
            model,
            tokenizer,
            mapping_by_subject,
            batch,
            view_map,
            base_true_lp,
            device=device,
        )
        local_margin = [float(v) for v in worst_margin.cpu().tolist()]
        local_supp = [float(v) for v in worst_suppression.cpu().tolist()]
        margins.extend(local_margin)
        suppressions.extend(local_supp)
        for record, margin, suppression in zip(batch, local_margin, local_supp):
            rr = record["requested_rewrite"]
            per_case.append(
                {
                    "case_id": int(record["case_id"]),
                    "subject": str(rr["subject"]),
                    "relation_id": str(rr["relation_id"]),
                    "worst_abstention_minus_true_margin": float(margin),
                    "worst_true_logprob_drop": float(suppression),
                    "abstention_pass": bool(margin >= margin_threshold),
                    "suppression_pass": bool(suppression >= suppression_threshold),
                }
            )
    return (
        summarize(margins, margin_threshold),
        summarize(suppressions, suppression_threshold),
        per_case,
    )


def train_relation_gate(
    model: Any,
    tokenizer: Any,
    forget: Sequence[Mapping[str, Any]],
    protection_fit: Sequence[Mapping[str, Any]],
    view_map: Mapping[int, Sequence[str]],
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[nullcore.PerSubjectRelationGate, Dict[int, float], Dict[str, Any]]:
    examples = nullcore.build_gate_examples(
        forget,
        view_map,
        protection_fit,
        negatives_per_case=int(args.gate_negatives_per_case),
        seed=14141,
    )
    texts = [row.text for row in examples]
    features = nullcore.extract_last_token_features(
        model,
        tokenizer,
        texts,
        device=device,
        batch_size=int(args.gate_feature_batch_size),
    ).to(device)
    labels = torch.tensor([float(row.label) for row in examples], device=device)
    example_case_ids = [int(row.case_id) for row in examples]
    case_ids = [int(record["case_id"]) for record in forget]
    gate = nullcore.PerSubjectRelationGate(case_ids, int(features.shape[1])).to(device)
    indices = gate.case_indices(example_case_ids, device=device)

    positives = float(labels.sum().item())
    negatives = float(labels.numel() - labels.sum().item())
    pos_weight = torch.tensor(
        max(1.0, negatives / max(1.0, positives)), device=device, dtype=features.dtype
    )
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=float(args.gate_learning_rate),
        weight_decay=float(args.gate_weight_decay),
    )
    for step in range(1, int(args.gate_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = gate(features, indices)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == int(args.gate_steps):
            with torch.no_grad():
                acc = float((logits >= 0).eq(labels > 0.5).float().mean().item())
            print(f"gate step {step:4d}: BCE={float(loss.item()):.6g}, zero-threshold acc={acc:.4f}", flush=True)

    with torch.no_grad():
        logits = gate(features, indices)
    thresholds, metrics = nullcore.calibrate_case_thresholds(
        logits.detach().cpu(), labels.detach().cpu(), example_case_ids
    )
    metrics.update(
        {
            "positive_examples": int(sum(row.label == 1 for row in examples)),
            "negative_examples": int(sum(row.label == 0 for row in examples)),
            "negatives_per_case_requested": int(args.gate_negatives_per_case),
            "feature_source": "frozen_base_model",
            "heldout_probe_text_used": False,
            "official_paraphrase_text_used": False,
            "official_neighborhood_text_used": False,
        }
    )
    if not metrics["all_cases_perfectly_separable"]:
        bad = [
            cid
            for cid, row in metrics["per_case"].items()
            if not row["perfect_training_separation"]
        ]
        raise RuntimeError(
            f"V1.4 relation gate is not perfectly separated on training-safe data; cases={bad[:10]}"
        )
    print(
        json.dumps(
            {
                "relation_gate_training": {
                    "accuracy": metrics["accuracy"],
                    "all_cases_perfectly_separable": True,
                    "positive_examples": metrics["positive_examples"],
                    "negative_examples": metrics["negative_examples"],
                }
            },
            indent=2,
        ),
        flush=True,
    )
    gate.eval()
    for parameter in gate.parameters():
        parameter.requires_grad_(False)
    return gate, thresholds, metrics


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "method").mkdir()

    registry = v11.v1.load_json(Path(args.experiment_registry))
    validate_registry(registry, args)
    protocol_dir = Path(args.protocol_dir).resolve()
    protocol = v11.v1.load_protocol(protocol_dir, int(args.forget_num))
    forget = protocol["forget"]
    protection_fit = protocol["protection_fit"]

    view_map, view_meta = v13.load_view_corpus(Path(args.view_corpus).resolve())
    forget_ids = {int(record["case_id"]) for record in forget}
    if set(view_map) != forget_ids or int(view_meta.get("views_per_case", 0)) != 5:
        raise RuntimeError("V1.4 view corpus does not exactly match the locked 50 forget cases")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    subjects = v1core.unique_subjects(forget)
    if len(subjects) != len(forget):
        raise RuntimeError("V1.4 currently requires one unique subject per forget case")
    mapping = pp.build_position_preserving_mapping(tokenizer, subjects)
    mapping_by_subject = {str(item["subject"]): dict(item) for item in mapping}
    private_ids = pp.flatten_private_ids(mapping)

    dtype = v11.v1.dtype_from_name(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    input_embedding = v11.v1.ensure_untied_input_embedding(model)
    output_embedding = model.get_output_embeddings()
    if output_embedding is None:
        raise RuntimeError("causal LM lacks an LM head")
    if input_embedding.weight.data_ptr() == output_embedding.weight.data_ptr():
        raise RuntimeError("input embedding and LM head remain tied")
    lm_head_hash_before = v1core.sha256_tensor(output_embedding.weight)
    nonprivate_hash_before = v1core.non_private_row_hash(input_embedding.weight, private_ids)

    # Stage 1: relation-sensitive classifier on exact Base-path features.
    gate, gate_thresholds, gate_metrics = train_relation_gate(
        model,
        tokenizer,
        forget,
        protection_fit,
        view_map,
        args,
        device=device,
    )

    # Stage 2: exact private clones, then private-null optimization only.
    initial_rows = pp.initialize_exact_private_rows(input_embedding.weight, mapping)
    controller = v1core.PrivateRowController(private_ids, initial_rows).to(device)
    hook = v1core.EmbeddingHook.install(input_embedding, controller)
    private_tokenizer = pp.PositionPreservingSubjectTokenizer(tokenizer, mapping)

    all_positive_texts = [
        str(template).format(str(record["requested_rewrite"]["subject"]))
        for record in forget
        for template in view_map[int(record["case_id"])]
    ]
    initial_kl = v11.v1.mean_kl_over_contexts(
        model,
        tokenizer,
        private_tokenizer,
        all_positive_texts[:64],
        device=device,
        topk=int(args.topk),
    )
    print(f"initial routed equivalence: mean KL={initial_kl:.9g}", flush=True)
    if initial_kl > float(args.initial_equivalence_kl_max):
        raise RuntimeError(
            f"V1.4 initial private-route equivalence failed: {initial_kl:.9g} > {args.initial_equivalence_kl_max}"
        )

    # Freeze a Base true-answer reference for every one of the 250 positive views.
    base_true_lp: Dict[tuple[int, int], float] = {}
    with torch.no_grad():
        for record in forget:
            local = view_records_for_case(record, view_map)
            prompts = [render_record_prompt(row) for row in local]
            subject = str(record["requested_rewrite"]["subject"])
            subjects_local = [subject] * len(local)
            answers = [str(row["requested_rewrite"]["target_true"]["str"]) for row in local]
            values = sequence_logprob_batch(
                model,
                tokenizer,
                mapping_by_subject,
                prompts,
                answers,
                subjects_local,
                route_private=False,
                device=device,
            )
            for row, value in zip(local, values.cpu().tolist()):
                base_true_lp[(int(record["case_id"]), int(row["_v1_4_view_index"]))] = float(value)

    optimizer = torch.optim.AdamW(
        [controller.rows], lr=float(args.learning_rate), weight_decay=0.0
    )
    rng = random.Random(int(args.seed) + 1401)
    best_state = controller.rows.detach().clone()
    best_score = float("inf")
    training_log: list[Dict[str, Any]] = []

    for step in range(1, int(args.steps) + 1):
        batch = rng.sample(forget, min(int(args.forget_batch_size), len(forget)))
        optimizer.zero_grad(set_to_none=True)
        worst_margin, worst_suppression, _, _ = batch_null_metrics(
            model,
            tokenizer,
            mapping_by_subject,
            batch,
            view_map,
            base_true_lp,
            device=device,
        )
        abstention_loss = F.relu(
            float(args.minimum_abstention_margin) - worst_margin
        ).square().mean()
        suppression_loss = F.relu(
            float(args.minimum_true_suppression) - worst_suppression
        ).square().mean()
        anchor = (controller.rows - controller.initial_rows).float().pow(2).mean()
        loss = (
            abstention_loss
            + float(args.true_suppression_weight) * suppression_loss
            + float(args.anchor_weight) * anchor
        )
        loss.backward()
        optimizer.step()
        cap = controller.enforce_relative_cap(float(args.relative_row_cap))

        if step == 1 or step % int(args.check_every) == 0 or step == int(args.steps):
            margin_summary, suppression_summary, per_case = evaluate_null_metrics(
                model,
                tokenizer,
                mapping_by_subject,
                forget,
                view_map,
                base_true_lp,
                device=device,
                batch_size=int(args.forget_batch_size),
                margin_threshold=float(args.minimum_abstention_margin),
                suppression_threshold=float(args.minimum_true_suppression),
            )
            combined_fail = sum(
                not (row["abstention_pass"] and row["suppression_pass"])
                for row in per_case
            )
            score = float(combined_fail) * 100000.0 + float(
                margin_summary["failures"] + suppression_summary["failures"]
            )
            if score < best_score:
                best_score = score
                best_state = controller.rows.detach().clone()
            failing_relations: Dict[str, int] = {}
            failing_cases: list[Dict[str, Any]] = []
            for row in per_case:
                if row["abstention_pass"] and row["suppression_pass"]:
                    continue
                relation = str(row["relation_id"])
                failing_relations[relation] = failing_relations.get(relation, 0) + 1
                failing_cases.append(row)
            training_log.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().item()),
                    "abstention_loss": float(abstention_loss.detach().item()),
                    "true_suppression_loss": float(suppression_loss.detach().item()),
                    "anchor": float(anchor.detach().item()),
                    "abstention_margin": margin_summary,
                    "true_suppression": suppression_summary,
                    "combined_failures": int(combined_fail),
                    "failing_relations": dict(sorted(failing_relations.items())),
                    **cap,
                }
            )
            print(
                f"step {step:4d}: combined fail={combined_fail}, "
                f"abstention fail={margin_summary['failures']}, "
                f"suppression fail={suppression_summary['failures']}, "
                f"min abstention margin={margin_summary['minimum']:.4f}, "
                f"min true drop={suppression_summary['minimum']:.4f}, "
                f"max rel delta={cap['max_relative_delta']:.4f}",
                flush=True,
            )
            if failing_cases:
                print(
                    json.dumps(
                        {
                            "v1_4_failing_cases": sorted(
                                failing_cases,
                                key=lambda row: (
                                    row["worst_abstention_minus_true_margin"],
                                    row["worst_true_logprob_drop"],
                                ),
                            )
                        },
                        indent=2,
                    ),
                    flush=True,
                )
            if combined_fail == 0:
                print("V1.4 registered training-null gate reached; stopping early", flush=True)
                break

    with torch.no_grad():
        controller.rows.copy_(best_state)
    final_margin, final_suppression, final_per_case = evaluate_null_metrics(
        model,
        tokenizer,
        mapping_by_subject,
        forget,
        view_map,
        base_true_lp,
        device=device,
        batch_size=int(args.forget_batch_size),
        margin_threshold=float(args.minimum_abstention_margin),
        suppression_threshold=float(args.minimum_true_suppression),
    )
    final_combined_failures = sum(
        not (row["abstention_pass"] and row["suppression_pass"])
        for row in final_per_case
    )

    v1core.materialize_private_rows(input_embedding.weight, controller)
    hook.remove()
    lm_head_hash_after = v1core.sha256_tensor(output_embedding.weight)
    nonprivate_hash_after = v1core.non_private_row_hash(input_embedding.weight, private_ids)
    if lm_head_hash_before != lm_head_hash_after:
        raise RuntimeError("LM head changed during V1.4")
    if nonprivate_hash_before != nonprivate_hash_after:
        raise RuntimeError("a non-private input embedding row changed during V1.4")

    gate_serial = nullcore.serialize_gate_state(gate, gate_thresholds)
    torch.save(
        {
            "weight": gate.weight.detach().cpu(),
            "bias": gate.bias.detach().cpu(),
            "case_ids": list(gate.case_ids),
            "thresholds": {int(k): float(v) for k, v in gate_thresholds.items()},
        },
        output / "method" / "relation_gate.pt",
    )
    torch.save(
        {
            "private_token_ids": private_ids,
            "private_rows": controller.rows.detach().cpu(),
            "subject_mapping": mapping,
            "base_model": str(args.model_path),
        },
        output / "method" / "private_null_rows.pt",
    )

    method = {
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "status": "development_only_seed1_consumed",
        "architecture": {
            "relation_aware_router": True,
            "gate": "per_subject_linear_classifier_on_frozen_base_final_prompt_hidden_state",
            "sensitive_path": "same_length_private_reserved_subject_rows",
            "non_sensitive_path": "exact_base_token_ids",
            "abstention_text": nullcore.ABSTENTION_TEXT,
            "internal_private_ids_exposed_to_user": False,
            "transformer_frozen": True,
            "lm_head_frozen_bit_identical": True,
            "original_input_embedding_rows_frozen": True,
        },
        "gate": {**gate_serial, "training_metrics": gate_metrics},
        "view_corpus": view_meta,
        "initial_private_route_equivalence_mean_topk_kl": float(initial_kl),
        "null_objective": {
            "target_new_gradient": False,
            "abstention_minus_true_margin_threshold": float(args.minimum_abstention_margin),
            "minimum_true_logprob_drop": float(args.minimum_true_suppression),
            "true_suppression_weight": float(args.true_suppression_weight),
            "anchor_weight": float(args.anchor_weight),
            "relative_row_cap": float(args.relative_row_cap),
        },
        "final_abstention_margin": final_margin,
        "final_true_suppression": final_suppression,
        "final_combined_failures": int(final_combined_failures),
        "final_per_case": final_per_case,
        "integrity": {
            "lm_head_bit_identical": lm_head_hash_before == lm_head_hash_after,
            "nonprivate_input_rows_bit_identical": nonprivate_hash_before == nonprivate_hash_after,
            "lm_head_sha256_before": lm_head_hash_before,
            "lm_head_sha256_after": lm_head_hash_after,
            "nonprivate_input_rows_sha256_before": nonprivate_hash_before,
            "nonprivate_input_rows_sha256_after": nonprivate_hash_after,
        },
        "claim_boundary": {
            "behavioral_unlearning": True,
            "latent_knowledge_erasure": False,
            "relation_aware_routing_is_part_of_method": True,
            "gate_is_part_of_method": True,
            "restoring_original_subject_token_ids_can_bypass_private_path": True,
            "official_paraphrases_used_for_training": False,
            "official_neighborhoods_used_for_training": False,
            "seed1_final_certification_allowed": False,
        },
        "training_log": training_log,
    }
    report_path = output / "method" / "private_vocab_rewiring_v1_4_relation_null.json"
    report_path.write_text(json.dumps(method, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.save_model:
        model_dir = output / "model"
        model.save_pretrained(model_dir)
        # Save the ordinary Base tokenizer.  V1.4 routing is conditional and must
        # never be represented by the V1.1 always-route tokenizer sidecar.
        tokenizer.save_pretrained(model_dir)
        torch.save(
            {
                "weight": gate.weight.detach().cpu(),
                "bias": gate.bias.detach().cpu(),
                "case_ids": list(gate.case_ids),
                "thresholds": {int(k): float(v) for k, v in gate_thresholds.items()},
            },
            model_dir / "relation_null_gate.pt",
        )
        routing_payload = {
            "protocol": PROTOCOL,
            "relation_aware": True,
            "routing": "registered_subject_then_frozen_base_relation_gate_then_conditional_same_length_private_rewrite",
            "mapping": mapping,
            "case_metadata": [
                {
                    "case_id": int(record["case_id"]),
                    "subject": str(record["requested_rewrite"]["subject"]),
                    "forgotten_relation_id": str(record["requested_rewrite"]["relation_id"]),
                    "gate_threshold": float(gate_thresholds[int(record["case_id"])]),
                }
                for record in forget
            ],
            "gate_file": "relation_null_gate.pt",
            "gate_feature": gate_serial["feature"],
            "abstention_text": nullcore.ABSTENTION_TEXT,
            "non_sensitive_path": "exact_base_token_ids",
            "internal_private_ids_are_not_output_labels": True,
        }
        (model_dir / "relation_null_routing.json").write_text(
            json.dumps(routing_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    completion = {
        "protocol": PROTOCOL,
        "gate_training_accuracy": gate_metrics["accuracy"],
        "gate_training_all_cases_separable": gate_metrics["all_cases_perfectly_separable"],
        "abstention_margin": final_margin,
        "true_suppression": final_suppression,
        "combined_failures": int(final_combined_failures),
        "all_50_all_5_null_pass": bool(final_combined_failures == 0),
        "initial_equivalence_passed": True,
        "integrity_passed": True,
        "checkpoint_saved": bool(args.save_model),
        "heldout_probe_text_used": False,
        "final_certification_status": "DEVELOPMENT ONLY; new untouched seed required",
    }
    (output / "method" / "completion.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
