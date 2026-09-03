#!/usr/bin/env python3
"""V1.2 true-target suppression objective for position-preserving private rewiring.

This module keeps the V1.1 architecture unchanged and replaces only the forget
objective.  The optimized quantity is the sensitive target_true log-probability
relative to its frozen-Base value; target_new never contributes gradient.
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
import run_mcf_private_vocab_rewiring_v1_1_relational as relational
import run_mcf_private_vocab_rewiring_v1_1 as v11


PROTOCOL = "mcf_private_vocab_rewiring_v1_2_true_target_suppression"


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
    p.add_argument("--true-logprob-drop-target", type=float, default=4.0)
    p.add_argument("--minimum-true-logprob-drop", type=float, default=2.0)
    p.add_argument("--minimum-forget-margin", type=float, default=0.1)
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
        p.error("V1.2 development run is locked to consumed seed 1 / 50 forget facts")
    if args.steps <= 0 or args.check_every <= 0:
        p.error("invalid step count or check interval")
    if args.true_logprob_drop_target <= 0:
        p.error("true-logprob drop target must be positive")
    if args.minimum_true_logprob_drop < 0:
        p.error("minimum true-logprob drop must be non-negative")
    if args.true_logprob_drop_target < args.minimum_true_logprob_drop:
        p.error("training drop target cannot be below final suppression threshold")
    if args.relative_row_cap <= 0:
        p.error("relative row cap must be positive")
    return args


def validate_registry(registry: Mapping[str, Any]) -> None:
    arch = registry.get("architecture", {})
    objective = registry.get("forget_objective", {})
    if (
        registry.get("protocol") != PROTOCOL
        or arch.get("base_architecture") != core.PROTOCOL
        or arch.get("trainable_parameter_families") != ["private_subject_embedding_rows"]
        or arch.get("relation_aware_router") is not False
        or arch.get("subject_token_count_preserved") is not True
        or objective.get("target_true_only_gradient") is not True
        or objective.get("target_new_gradient") is not False
    ):
        raise RuntimeError("V1.2 registry contract mismatch")


def true_logprob_batch(
    model: Any,
    prompt_tokenizer: Any,
    base_tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> torch.Tensor:
    prompts = [v1.render_prompt(record) for record in records]
    answers = [str(record["requested_rewrite"]["target_true"]["str"]) for record in records]
    return v11.sequence_logprob_batch(
        model, prompt_tokenizer, base_tokenizer, prompts, answers, device=device
    )


def evaluate_true_logprobs(
    model: Any,
    prompt_tokenizer: Any,
    base_tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> list[float]:
    values: list[float] = []
    with torch.no_grad():
        for start in range(0, len(records), int(batch_size)):
            batch = records[start : start + int(batch_size)]
            lp = true_logprob_batch(
                model, prompt_tokenizer, base_tokenizer, batch, device=device
            )
            values.extend(float(x) for x in lp.detach().cpu().tolist())
    return values


def suppression_summary(
    base_true_lp: Sequence[float],
    current_true_lp: Sequence[float],
    *,
    minimum_drop: float,
) -> Dict[str, Any]:
    drops = [float(base - current) for base, current in zip(base_true_lp, current_true_lp)]
    passed = sum(drop >= float(minimum_drop) for drop in drops)
    return {
        "count": len(drops),
        "minimum_drop": min(drops) if drops else None,
        "mean_drop": (sum(drops) / len(drops)) if drops else None,
        "maximum_drop": max(drops) if drops else None,
        "passed": passed,
        "failures": len(drops) - passed,
        "threshold": float(minimum_drop),
        "drops": drops,
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

    base_tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    subjects = v1core.unique_subjects(forget)
    mapping = core.build_position_preserving_mapping(base_tokenizer, subjects)
    private_tokenizer = core.PositionPreservingSubjectTokenizer(base_tokenizer, mapping)
    routing = core.validate_position_preserving_routing(base_tokenizer, private_tokenizer, mapping)
    private_ids = core.flatten_private_ids(mapping)
    print(json.dumps({
        "protocol": PROTOCOL,
        "unique_forget_subjects": len(subjects),
        "private_rows_allocated": len(private_ids),
        "vocab_size": len(base_tokenizer),
        "subject_token_count_preserved": True,
        "target_new_gradient": False,
        "sample": routing["examples"],
    }, indent=2), flush=True)

    tokenization_cert = v11.tokenization_certification(
        base_tokenizer,
        private_tokenizer,
        protection_fit,
        subjects,
        maximum=int(args.nonclone_certification_prompts),
    )
    if not tokenization_cert["passed"]:
        raise RuntimeError("V1.2 token-sequence rewrite failed locality/length preflight")

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
    nonprivate_hash_before = v1core.non_private_row_hash(input_embedding.weight, private_ids)

    initial_rows = core.initialize_exact_private_rows(input_embedding.weight, mapping)
    controller = v1core.PrivateRowController(private_ids, initial_rows).to(device)
    hook = v1core.EmbeddingHook.install(input_embedding, controller)

    retain_contexts, retain_context_stats = relational.make_relation_preserving_retain_contexts(
        forget, protection_fit
    )
    equivalence_contexts = v1.clone_equivalence_contexts(forget)

    base_margins = v11.evaluate_margins(
        model, base_tokenizer, base_tokenizer, forget,
        device=device, batch_size=int(args.forget_batch_size)
    )
    private_margins = v11.evaluate_margins(
        model, private_tokenizer, base_tokenizer, forget,
        device=device, batch_size=int(args.forget_batch_size)
    )
    base_true_lp = evaluate_true_logprobs(
        model, base_tokenizer, base_tokenizer, forget,
        device=device, batch_size=int(args.forget_batch_size)
    )
    private_true_lp = evaluate_true_logprobs(
        model, private_tokenizer, base_tokenizer, forget,
        device=device, batch_size=int(args.forget_batch_size)
    )
    initial_kl = v1.mean_kl_over_contexts(
        model, base_tokenizer, private_tokenizer, equivalence_contexts,
        device=device, topk=int(args.topk)
    )
    margin_drift = max((abs(a - b) for a, b in zip(base_margins, private_margins)), default=0.0)
    true_lp_drift = max((abs(a - b) for a, b in zip(base_true_lp, private_true_lp)), default=0.0)
    print(
        f"initial equivalence: mean KL={initial_kl:.9g}, max margin drift={margin_drift:.9g}, "
        f"max true-lp drift={true_lp_drift:.9g}", flush=True
    )
    if initial_kl > float(args.initial_equivalence_kl_max):
        raise RuntimeError("position-preserving equivalence KL failed")
    if margin_drift > float(args.initial_margin_drift_max) or true_lp_drift > float(args.initial_margin_drift_max):
        raise RuntimeError("position-preserving score equivalence failed")

    base_true_by_case = {
        int(record["case_id"]): float(lp) for record, lp in zip(forget, base_true_lp)
    }

    optimizer = torch.optim.AdamW([controller.rows], lr=float(args.learning_rate), weight_decay=0.0)
    rng = random.Random(int(args.seed) + 1204)
    training_log: list[Dict[str, Any]] = []
    best_state = controller.rows.detach().clone()
    best_score = float("inf")

    for step in range(1, int(args.steps) + 1):
        forget_batch = rng.sample(forget, min(int(args.forget_batch_size), len(forget)))
        retain_batch = rng.sample(retain_contexts, min(int(args.retain_batch_size), len(retain_contexts)))
        optimizer.zero_grad(set_to_none=True)

        current_true_lp = true_logprob_batch(
            model, private_tokenizer, base_tokenizer, forget_batch, device=device
        )
        baseline = torch.tensor(
            [base_true_by_case[int(record["case_id"])] for record in forget_batch],
            device=device, dtype=current_true_lp.dtype,
        )
        required_ceiling = baseline - float(args.true_logprob_drop_target)
        forget_loss = F.relu(current_true_lp - required_ceiling).square().mean()

        retain_kl = v1.topk_teacher_kl(
            model, base_tokenizer, private_tokenizer, retain_batch,
            device=device, topk=int(args.topk)
        )
        anchor = (controller.rows - controller.initial_rows).float().pow(2).mean()
        loss = forget_loss + float(args.retain_kl_weight) * retain_kl + float(args.anchor_weight) * anchor
        loss.backward()
        optimizer.step()
        cap = controller.enforce_relative_cap(float(args.relative_row_cap))

        if step == 1 or step % int(args.check_every) == 0 or step == int(args.steps):
            all_true_lp = evaluate_true_logprobs(
                model, private_tokenizer, base_tokenizer, forget,
                device=device, batch_size=int(args.forget_batch_size)
            )
            suppression = suppression_summary(
                base_true_lp, all_true_lp, minimum_drop=float(args.minimum_true_logprob_drop)
            )
            all_margins = v11.evaluate_margins(
                model, private_tokenizer, base_tokenizer, forget,
                device=device, batch_size=int(args.forget_batch_size)
            )
            margin_summary = v1.margin_summary(all_margins, float(args.minimum_forget_margin))
            retain_mean_kl = v1.mean_kl_over_contexts(
                model, base_tokenizer, private_tokenizer, retain_contexts,
                device=device, topk=int(args.topk)
            )
            score = float(suppression["failures"]) * 1000.0 + retain_mean_kl
            if score < best_score:
                best_score = score
                best_state = controller.rows.detach().clone()
            row = {
                "step": step,
                "loss": float(loss.detach().item()),
                "true_suppression_loss": float(forget_loss.detach().item()),
                "retain_batch_kl": float(retain_kl.detach().item()),
                "anchor": float(anchor.detach().item()),
                "true_suppression": {k: v for k, v in suppression.items() if k != "drops"},
                "diagnostic_new_vs_true_margin": margin_summary,
                "retain_mean_kl": retain_mean_kl,
                **cap,
            }
            training_log.append(row)
            print(
                f"step {step:4d}: suppression fail={suppression['failures']}, "
                f"min drop={suppression['minimum_drop']:.4f}, mean drop={suppression['mean_drop']:.4f}, "
                f"margin fail={margin_summary['failures']}, retain KL={retain_mean_kl:.6g}, "
                f"max rel delta={cap['max_relative_delta']:.4f}", flush=True
            )
            if suppression["failures"] == 0 and retain_mean_kl <= float(args.retain_kl_mean_max):
                print("registered V1.2 suppression gate reached; stopping early", flush=True)
                break

    with torch.no_grad():
        controller.rows.copy_(best_state)

    final_true_lp = evaluate_true_logprobs(
        model, private_tokenizer, base_tokenizer, forget,
        device=device, batch_size=int(args.forget_batch_size)
    )
    final_suppression = suppression_summary(
        base_true_lp, final_true_lp, minimum_drop=float(args.minimum_true_logprob_drop)
    )
    final_margins = v11.evaluate_margins(
        model, private_tokenizer, base_tokenizer, forget,
        device=device, batch_size=int(args.forget_batch_size)
    )
    final_margin_summary = v1.margin_summary(final_margins, float(args.minimum_forget_margin))
    final_retain_kl = v1.mean_kl_over_contexts(
        model, base_tokenizer, private_tokenizer, retain_contexts,
        device=device, topk=int(args.topk)
    )

    v1core.materialize_private_rows(input_embedding.weight, controller)
    hook.remove()
    lm_head_hash_after = v1core.sha256_tensor(output_embedding.weight)
    nonprivate_hash_after = v1core.non_private_row_hash(input_embedding.weight, private_ids)
    if lm_head_hash_before != lm_head_hash_after:
        raise RuntimeError("LM head changed during V1.2 training")
    if nonprivate_hash_before != nonprivate_hash_after:
        raise RuntimeError("a non-private input embedding row changed")

    per_case = []
    for record, base_lp, final_lp, margin, drop in zip(
        forget, base_true_lp, final_true_lp, final_margins, final_suppression["drops"]
    ):
        per_case.append({
            "case_id": int(record["case_id"]),
            "subject": str(record["requested_rewrite"]["subject"]),
            "relation_id": str(record["requested_rewrite"]["relation_id"]),
            "base_true_logprob": float(base_lp),
            "final_true_logprob": float(final_lp),
            "true_logprob_drop": float(drop),
            "diagnostic_new_vs_true_margin": float(margin),
        })

    method = {
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "architecture_base": core.PROTOCOL,
        "forget_objective": {
            "name": "base_relative_target_true_logprob_suppression",
            "target_true_only_gradient": True,
            "target_new_gradient": False,
            "true_logprob_drop_target": float(args.true_logprob_drop_target),
            "minimum_true_logprob_drop": float(args.minimum_true_logprob_drop),
        },
        "unique_forget_subjects": len(subjects),
        "private_rows": len(private_ids),
        "subject_mapping": mapping,
        "routing_preflight": routing,
        "tokenization_certification": tokenization_cert,
        "retain_contexts": retain_context_stats,
        "initial_equivalence": {
            "mean_topk_kl": initial_kl,
            "max_margin_drift": margin_drift,
            "max_true_logprob_drift": true_lp_drift,
        },
        "true_suppression": {
            "final": {k: v for k, v in final_suppression.items() if k != "drops"},
            "per_case": per_case,
        },
        "diagnostic_margins": {
            "final": final_margin_summary,
            "final_per_case": [
                {"case_id": int(record["case_id"]), "margin": float(margin)}
                for record, margin in zip(forget, final_margins)
            ],
        },
        "final_retain_mean_kl": final_retain_kl,
        "integrity": {
            "lm_head_bit_identical": lm_head_hash_before == lm_head_hash_after,
            "nonprivate_input_rows_bit_identical": nonprivate_hash_before == nonprivate_hash_after,
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
    (output / "method" / "private_vocab_rewiring_v1_2_true_suppression.json").write_text(
        json.dumps(method, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    torch.save({
        "private_token_ids": private_ids,
        "private_rows": controller.rows.detach().cpu(),
        "subject_mapping": mapping,
        "base_model": str(args.model_path),
        "protocol": PROTOCOL,
    }, output / "method" / "private_rows.pt")

    if args.save_model:
        model_dir = output / "model"
        model.save_pretrained(model_dir)
        private_tokenizer.save_pretrained(model_dir)
        reloaded = core.load_position_preserving_tokenizer(model_dir, AutoTokenizer)
        core.validate_position_preserving_routing(base_tokenizer, reloaded, mapping)

    completion = {
        "true_suppression": {k: v for k, v in final_suppression.items() if k != "drops"},
        "diagnostic_direct_margin": final_margin_summary,
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
