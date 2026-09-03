#!/usr/bin/env python3
"""Train position-preserving private-vocabulary factual unlearning on MCF.

Each unique forget subject keeps its original token count.  If Base tokenizes a
subject as ``[t1, ..., tk]``, V1.1 allocates ``[p1, ..., pk]`` from reserved
Llama vocabulary slots and initializes ``E[pj] = E[tj]`` exactly.  Prompt-side
subject sequences are deterministically rewritten to the private ids in every
context; relation text is never inspected.  The Transformer, LM head, and all
original lexical input rows remain frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import mcf_private_vocab_rewiring_v1_core as v1core
import mcf_private_vocab_rewiring_v1_1_core as core
import run_mcf_private_vocab_rewiring_v1 as v1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--protocol-dir", required=True)
    p.add_argument("--experiment-registry", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--forget-batch-size", type=int, default=8)
    p.add_argument("--retain-batch-size", type=int, default=16)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--minimum-forget-margin", type=float, default=0.1)
    p.add_argument("--train-margin-target", type=float, default=0.1)
    p.add_argument("--retain-kl-weight", type=float, default=20.0)
    p.add_argument("--anchor-weight", type=float, default=1e-3)
    p.add_argument("--relative-row-cap", type=float, default=0.5)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--initial-equivalence-kl-max", type=float, default=1e-7)
    p.add_argument("--initial-margin-drift-max", type=float, default=1e-5)
    p.add_argument("--retain-kl-mean-max", type=float, default=1e-4)
    p.add_argument("--nonclone-certification-prompts", type=int, default=64)
    p.add_argument("--save-model", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.seed != 1 or args.forget_num != 50:
        p.error("V1.1 is locked to consumed seed 1 / 50 forget facts")
    if args.train_margin_target < args.minimum_forget_margin:
        p.error("training margin target cannot be below final success margin")
    if args.steps <= 0 or args.check_every <= 0:
        p.error("invalid step count or check interval")
    if args.relative_row_cap <= 0:
        p.error("relative row cap must be positive")
    if args.initial_equivalence_kl_max < 0 or args.initial_margin_drift_max < 0:
        p.error("equivalence tolerances must be non-negative")
    return args


def validate_registry(registry: Mapping[str, Any]) -> None:
    arch = registry.get("architecture", {})
    if (
        registry.get("protocol") != core.PROTOCOL
        or arch.get("trainable_parameter_families") != ["private_subject_embedding_rows"]
        or arch.get("private_tokens_per_subject_token") != "one_to_one"
        or arch.get("subject_token_count_preserved") is not True
        or arch.get("transformer_frozen") is not True
        or arch.get("lm_head_frozen_bit_identical") is not True
        or arch.get("original_input_embedding_rows_frozen") is not True
        or arch.get("reserved_vocab_slots_repurposed") is not True
        or arch.get("relation_aware_router") is not False
        or arch.get("subject_sequence_rewrite") is not True
    ):
        raise RuntimeError("V1.1 position-preserving registry contract mismatch")


def answer_ids(base_tokenizer: Any, answer: str) -> list[int]:
    text = str(answer).strip()
    if not text:
        raise ValueError("empty target answer")
    ids = base_tokenizer(
        " " + text, add_special_tokens=False, return_attention_mask=False
    )["input_ids"]
    if not ids:
        raise ValueError(f"target answer tokenizes empty: {answer!r}")
    return [int(value) for value in ids]


def sequence_logprob_batch(
    model: Any,
    prompt_tokenizer: Any,
    base_tokenizer: Any,
    prompts: Sequence[str],
    answers: Sequence[str],
    *,
    device: torch.device,
) -> torch.Tensor:
    rows: list[list[int]] = []
    starts: list[int] = []
    targets: list[list[int]] = []
    bos = getattr(base_tokenizer, "bos_token_id", None)
    for prompt, answer in zip(prompts, answers):
        pids = prompt_tokenizer(
            prompt, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        pids = [int(v) for v in pids]
        if bos is not None:
            pids = [int(bos)] + pids
        aids = answer_ids(base_tokenizer, answer)
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
    logits = model(input_ids=input_ids, attention_mask=attention).logits
    log_probs = F.log_softmax(logits.float(), dim=-1)
    values = []
    for i, (start, aids) in enumerate(zip(starts, targets)):
        positions = torch.arange(
            start - 1, start - 1 + len(aids), device=device, dtype=torch.long
        )
        token_ids = torch.tensor(aids, device=device, dtype=torch.long)
        values.append(log_probs[i, positions, token_ids].mean())
    return torch.stack(values)


def margin_batch(
    model: Any,
    prompt_tokenizer: Any,
    base_tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> torch.Tensor:
    prompts = [v1.render_prompt(record) for record in records]
    new_answers = [
        str(record["requested_rewrite"]["target_new"]["str"]) for record in records
    ]
    true_answers = [
        str(record["requested_rewrite"]["target_true"]["str"]) for record in records
    ]
    new_lp = sequence_logprob_batch(
        model, prompt_tokenizer, base_tokenizer, prompts, new_answers, device=device
    )
    true_lp = sequence_logprob_batch(
        model, prompt_tokenizer, base_tokenizer, prompts, true_answers, device=device
    )
    return new_lp - true_lp


def evaluate_margins(
    model: Any,
    prompt_tokenizer: Any,
    base_tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> list[float]:
    out: list[float] = []
    with torch.no_grad():
        for start in range(0, len(records), int(batch_size)):
            batch = records[start : start + int(batch_size)]
            values = margin_batch(
                model, prompt_tokenizer, base_tokenizer, batch, device=device
            )
            out.extend(float(v) for v in values.detach().cpu().tolist())
    return out


def tokenization_certification(
    base_tokenizer: Any,
    private_tokenizer: Any,
    protection_fit: Sequence[Mapping[str, Any]],
    subjects: Sequence[str],
    *,
    maximum: int,
) -> Dict[str, Any]:
    subject_set = set(subjects)
    checked = 0
    changed_without_registered_subject: list[Dict[str, Any]] = []
    length_mismatches: list[int] = []
    for record in protection_fit:
        text = v1.render_prompt(record)
        base_ids = base_tokenizer(text, add_special_tokens=False)["input_ids"]
        private_ids = private_tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(base_ids) != len(private_ids):
            length_mismatches.append(int(record["case_id"]))
        has_registered_subject = any(subject in text for subject in subject_set)
        if not has_registered_subject and base_ids != private_ids:
            changed_without_registered_subject.append(
                {"case_id": int(record["case_id"]), "text": text}
            )
        checked += 1
        if checked >= int(maximum):
            break
    return {
        "checked": checked,
        "length_mismatches": len(length_mismatches),
        "changed_without_registered_subject": len(changed_without_registered_subject),
        "examples": changed_without_registered_subject[:5],
        "passed": checked > 0 and not length_mismatches and not changed_without_registered_subject,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "method").mkdir()

    registry = v1.load_json(Path(args.experiment_registry))
    validate_registry(registry)
    protocol = v1.load_protocol(Path(args.protocol_dir), int(args.forget_num))
    forget = protocol["forget"]
    protection_fit = protocol["protection_fit"]

    # Tokenizer/mapping preflight is intentionally before the 3B model load.
    base_tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    subjects = v1core.unique_subjects(forget)
    mapping = core.build_position_preserving_mapping(base_tokenizer, subjects)
    private_tokenizer = core.PositionPreservingSubjectTokenizer(base_tokenizer, mapping)
    routing = core.validate_position_preserving_routing(
        base_tokenizer, private_tokenizer, mapping
    )
    private_ids = core.flatten_private_ids(mapping)
    print(
        json.dumps(
            {
                "unique_forget_subjects": len(subjects),
                "private_rows_allocated": len(private_ids),
                "vocab_size": len(base_tokenizer),
                "subject_token_count_preserved": True,
                "sample": routing["examples"],
            },
            indent=2,
        ),
        flush=True,
    )

    tokenization_cert = tokenization_certification(
        base_tokenizer,
        private_tokenizer,
        protection_fit,
        subjects,
        maximum=int(args.nonclone_certification_prompts),
    )
    if not tokenization_cert["passed"]:
        raise RuntimeError("V1.1 token-sequence rewrite failed locality/length preflight")

    dtype = v1.dtype_from_name(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    input_embedding = v1.ensure_untied_input_embedding(model)
    output_embedding = model.get_output_embeddings()
    if output_embedding is None:
        raise RuntimeError("causal LM lacks an LM head")
    if input_embedding.weight.data_ptr() == output_embedding.weight.data_ptr():
        raise RuntimeError("input embedding and LM head remain tied")

    lm_head_hash_before = v1core.sha256_tensor(output_embedding.weight)
    nonprivate_hash_before = v1core.non_private_row_hash(
        input_embedding.weight, private_ids
    )

    initial_rows = core.initialize_exact_private_rows(input_embedding.weight, mapping)
    controller = v1core.PrivateRowController(private_ids, initial_rows).to(device)
    hook = v1core.EmbeddingHook.install(input_embedding, controller)

    retain_contexts, retain_context_stats = v1.make_retain_contexts(forget, protection_fit)
    equivalence_contexts = v1.clone_equivalence_contexts(forget)

    base_margins = evaluate_margins(
        model,
        base_tokenizer,
        base_tokenizer,
        forget,
        device=device,
        batch_size=int(args.forget_batch_size),
    )
    private_margins = evaluate_margins(
        model,
        private_tokenizer,
        base_tokenizer,
        forget,
        device=device,
        batch_size=int(args.forget_batch_size),
    )
    initial_kl = v1.mean_kl_over_contexts(
        model,
        base_tokenizer,
        private_tokenizer,
        equivalence_contexts,
        device=device,
        topk=int(args.topk),
    )
    margin_drift = max(
        (abs(a - b) for a, b in zip(base_margins, private_margins)), default=0.0
    )
    print(
        f"initial equivalence: mean KL={initial_kl:.9g}, "
        f"max margin drift={margin_drift:.9g}",
        flush=True,
    )
    if initial_kl > float(args.initial_equivalence_kl_max):
        raise RuntimeError(
            f"position-preserving equivalence failed: KL {initial_kl:.9g} > "
            f"{args.initial_equivalence_kl_max}"
        )
    if margin_drift > float(args.initial_margin_drift_max):
        raise RuntimeError(
            f"position-preserving margin equivalence failed: {margin_drift:.9g} > "
            f"{args.initial_margin_drift_max}"
        )

    optimizer = torch.optim.AdamW(
        [controller.rows], lr=float(args.learning_rate), weight_decay=0.0
    )
    rng = random.Random(int(args.seed) + 1101)
    training_log: list[Dict[str, Any]] = []
    best_state = controller.rows.detach().clone()
    best_score = float("inf")

    for step in range(1, int(args.steps) + 1):
        forget_batch = rng.sample(
            forget, min(int(args.forget_batch_size), len(forget))
        )
        retain_batch = rng.sample(
            retain_contexts, min(int(args.retain_batch_size), len(retain_contexts))
        )
        optimizer.zero_grad(set_to_none=True)
        margins = margin_batch(
            model, private_tokenizer, base_tokenizer, forget_batch, device=device
        )
        forget_loss = F.relu(float(args.train_margin_target) - margins).square().mean()
        retain_kl = v1.topk_teacher_kl(
            model,
            base_tokenizer,
            private_tokenizer,
            retain_batch,
            device=device,
            topk=int(args.topk),
        )
        anchor = (controller.rows - controller.initial_rows).float().pow(2).mean()
        loss = (
            forget_loss
            + float(args.retain_kl_weight) * retain_kl
            + float(args.anchor_weight) * anchor
        )
        loss.backward()
        optimizer.step()
        cap = controller.enforce_relative_cap(float(args.relative_row_cap))

        if step == 1 or step % int(args.check_every) == 0 or step == int(args.steps):
            all_margins = evaluate_margins(
                model,
                private_tokenizer,
                base_tokenizer,
                forget,
                device=device,
                batch_size=int(args.forget_batch_size),
            )
            summary = v1.margin_summary(
                all_margins, float(args.minimum_forget_margin)
            )
            retain_mean_kl = v1.mean_kl_over_contexts(
                model,
                base_tokenizer,
                private_tokenizer,
                retain_contexts,
                device=device,
                topk=int(args.topk),
            )
            score = float(summary["failures"]) * 1000.0 + retain_mean_kl
            if score < best_score:
                best_score = score
                best_state = controller.rows.detach().clone()
            row = {
                "step": step,
                "loss": float(loss.detach().item()),
                "forget_loss": float(forget_loss.detach().item()),
                "retain_batch_kl": float(retain_kl.detach().item()),
                "anchor": float(anchor.detach().item()),
                "direct": summary,
                "retain_mean_kl": retain_mean_kl,
                **cap,
            }
            training_log.append(row)
            print(
                f"step {step:4d}: direct fail={summary['failures']}, "
                f"min margin={summary['minimum']:.4f}, retain KL={retain_mean_kl:.6g}, "
                f"max rel delta={cap['max_relative_delta']:.4f}",
                flush=True,
            )
            if (
                summary["failures"] == 0
                and retain_mean_kl <= float(args.retain_kl_mean_max)
            ):
                print("registered training gate reached; stopping early", flush=True)
                break

    with torch.no_grad():
        controller.rows.copy_(best_state)
    final_margins = evaluate_margins(
        model,
        private_tokenizer,
        base_tokenizer,
        forget,
        device=device,
        batch_size=int(args.forget_batch_size),
    )
    final_summary = v1.margin_summary(
        final_margins, float(args.minimum_forget_margin)
    )
    final_retain_kl = v1.mean_kl_over_contexts(
        model,
        base_tokenizer,
        private_tokenizer,
        retain_contexts,
        device=device,
        topk=int(args.topk),
    )

    v1core.materialize_private_rows(input_embedding.weight, controller)
    hook.remove()
    lm_head_hash_after = v1core.sha256_tensor(output_embedding.weight)
    nonprivate_hash_after = v1core.non_private_row_hash(
        input_embedding.weight, private_ids
    )
    if lm_head_hash_before != lm_head_hash_after:
        raise RuntimeError("LM head changed during V1.1 training")
    if nonprivate_hash_before != nonprivate_hash_after:
        raise RuntimeError("a non-private input embedding row changed")

    method = {
        "protocol": core.PROTOCOL,
        "seed": int(args.seed),
        "unique_forget_subjects": len(subjects),
        "private_rows": len(private_ids),
        "subject_mapping": mapping,
        "routing_preflight": routing,
        "tokenization_certification": tokenization_cert,
        "retain_contexts": retain_context_stats,
        "initial_equivalence": {
            "mean_topk_kl": initial_kl,
            "max_margin_drift": margin_drift,
            "registered_kl_max": float(args.initial_equivalence_kl_max),
            "registered_margin_drift_max": float(args.initial_margin_drift_max),
        },
        "margins": {
            "base": v1.margin_summary(base_margins, float(args.minimum_forget_margin)),
            "private_before_training": v1.margin_summary(
                private_margins, float(args.minimum_forget_margin)
            ),
            "final": final_summary,
            "final_per_case": [
                {"case_id": int(record["case_id"]), "margin": float(margin)}
                for record, margin in zip(forget, final_margins)
            ],
        },
        "final_retain_mean_kl": final_retain_kl,
        "integrity": {
            "lm_head_bit_identical": lm_head_hash_before == lm_head_hash_after,
            "nonprivate_input_rows_bit_identical": nonprivate_hash_before
            == nonprivate_hash_after,
            "lm_head_sha256_before": lm_head_hash_before,
            "lm_head_sha256_after": lm_head_hash_after,
            "nonprivate_input_rows_sha256_before": nonprivate_hash_before,
            "nonprivate_input_rows_sha256_after": nonprivate_hash_after,
            "materialized_private_row_ids": private_ids,
        },
        "claim_boundary": {
            "behavioral_unlearning": True,
            "latent_erasure_claimed": False,
            "subject_sequence_rewrite": True,
            "relation_aware_router": False,
            "restoring_original_subject_ids_can_bypass": True,
        },
        "training_log": training_log,
    }
    (output / "method" / "private_vocab_rewiring_v1_1.json").write_text(
        json.dumps(method, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "private_token_ids": private_ids,
            "private_rows": controller.rows.detach().cpu(),
            "subject_mapping": mapping,
            "base_model": str(args.model_path),
        },
        output / "method" / "private_rows.pt",
    )

    if args.save_model:
        model_dir = output / "model"
        model.save_pretrained(model_dir)
        private_tokenizer.save_pretrained(model_dir)
        reloaded = core.load_position_preserving_tokenizer(model_dir, AutoTokenizer)
        core.validate_position_preserving_routing(base_tokenizer, reloaded, mapping)

    completion = {
        "direct": final_summary,
        "retain_mean_kl": final_retain_kl,
        "retain_gate": final_retain_kl <= float(args.retain_kl_mean_max),
        "initial_equivalence_passed": True,
        "integrity_passed": True,
        "checkpoint_saved": bool(args.save_model),
    }
    (output / "method" / "completion.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
