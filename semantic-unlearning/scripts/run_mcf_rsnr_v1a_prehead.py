#!/usr/bin/env python3
"""RSNR-V1A-PreHead: oracle fact routing with a pre-LM-head null adapter.

Controlled ablation of the layer-24 RSNR-V1A experiment.  The Base model is
fully frozen: embeddings, every Transformer block, final norm, and LM head are
unchanged.  The only trainable object is the same rank-16 residual bottleneck
used by RSNR-V1A, but it is applied to the final hidden state immediately before
the frozen LM head.

For a sensitive (subject, relation) query:

    h_final' = h_final + A_NULL(h_final)
    logits   = W_LM h_final'

For every non-sensitive query the gate is exactly zero, so the computation is
bit-for-bit the ordinary Base path through the unchanged LM head.

The training corpus, losses, thresholds, seed, and five-view worst-case protocol
match RSNR-V1A.  No CounterFact target_new objective or held-out official probe
text is used.  This is conditional behavioral suppression, not a claim of
latent knowledge deletion.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

import run_mcf_private_vocab_rewiring_v1 as base_runner
import run_mcf_rsnr_v1a_oracle as rsnr


PROTOCOL = "mcf_rsnr_v1a_prehead_oracle_null_adapter"
ABSTENTION = rsnr.ABSTENTION
INTERVENTION_SITE = "pre_lm_head_final_hidden_state"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--protocol-dir", required=True)
    p.add_argument("--view-corpus", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--case-batch-size", type=int, default=4)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--adapter-rank", type=int, default=16)
    p.add_argument("--adapter-alpha", type=float, default=16.0)
    p.add_argument("--abstain-weight", type=float, default=1.0)
    p.add_argument("--unlikelihood-weight", type=float, default=1.0)
    p.add_argument("--anchor-weight", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--minimum-abstain-vs-true-margin", type=float, default=0.1)
    p.add_argument("--minimum-true-logprob-drop", type=float, default=2.0)
    p.add_argument("--gate-off-logit-drift-max", type=float, default=0.0)
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.seed != 1 or args.forget_num != 50:
        p.error("RSNR-V1A-PreHead is development-only and locked to consumed seed1/forget50")
    if args.steps <= 0 or args.case_batch_size <= 0 or args.check_every <= 0:
        p.error("steps/batch/check must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        p.error("invalid optimizer configuration")
    if args.adapter_rank <= 0 or args.adapter_alpha <= 0:
        p.error("adapter rank/alpha must be positive")
    if args.abstain_weight < 0 or args.unlikelihood_weight < 0 or args.anchor_weight < 0:
        p.error("loss weights must be non-negative")
    if args.abstain_weight == 0 and args.unlikelihood_weight == 0:
        p.error("at least one sensitive objective must be active")
    return args


@dataclass
class PreHeadNullHook:
    """Conditionally edit the final hidden state immediately before LM head."""

    adapter: rsnr.NullResidualAdapter
    handle: Any
    gate_mask: torch.Tensor | None = None
    position_mask: torch.Tensor | None = None

    @classmethod
    def install(cls, lm_head: nn.Module, adapter: rsnr.NullResidualAdapter) -> "PreHeadNullHook":
        state = cls(adapter=adapter, handle=None)

        def pre_hook(_module: nn.Module, inputs: tuple[Any, ...]):
            if state.gate_mask is None or not inputs:
                return inputs
            hidden = inputs[0]
            if not torch.is_tensor(hidden) or hidden.ndim != 3:
                return inputs
            gate = state.gate_mask.to(device=hidden.device, dtype=hidden.dtype).view(-1, 1, 1)
            if gate.shape[0] != hidden.shape[0]:
                raise RuntimeError("pre-head oracle gate batch size does not match hidden batch")
            if state.position_mask is not None:
                pos = state.position_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
                if tuple(pos.shape[:2]) != tuple(hidden.shape[:2]):
                    raise RuntimeError("pre-head position mask does not match hidden-state shape")
                gate = gate * pos
            if not bool(torch.any(gate != 0).item()):
                return inputs
            edited = hidden + gate * state.adapter(hidden)
            return (edited, *inputs[1:])

        state.handle = lm_head.register_forward_pre_hook(pre_hook)
        return state

    def set(self, gate_mask: torch.Tensor | None, position_mask: torch.Tensor | None = None) -> None:
        self.gate_mask = gate_mask
        self.position_mask = position_mask

    def clear(self) -> None:
        self.gate_mask = None
        self.position_mask = None

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()


def get_lm_head(model: Any) -> nn.Module:
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if head is None:
        head = getattr(model, "lm_head", None)
    if not isinstance(head, nn.Module):
        raise RuntimeError("could not locate LM head/output embedding module")
    return head


def _membership(forget: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": int(row["case_id"]),
            "subject": rsnr.fact_key(row)[0],
            "relation_id": rsnr.fact_key(row)[1],
        }
        for row in forget
    ]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    method_dir = output / "method"
    method_dir.mkdir()

    protocol = rsnr.load_protocol(Path(args.protocol_dir), int(args.forget_num))
    forget = protocol["forget"]
    protection_fit = protocol["protection_fit"]
    view_path = Path(args.view_corpus).resolve()
    view_map, view_meta = rsnr.load_training_views(view_path)
    rsnr.validate_case_alignment(forget, view_map)
    oracle_audit = rsnr.build_oracle_negative_audit(forget, protection_fit)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    dtype = base_runner.dtype_from_name(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for RSNR-V1A-PreHead training")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    hidden_size = int(getattr(model.config, "hidden_size"))
    adapter = rsnr.NullResidualAdapter(
        hidden_size, int(args.adapter_rank), float(args.adapter_alpha), device
    ).to(device)
    lm_head = get_lm_head(model)
    hook = PreHeadNullHook.install(lm_head, adapter)

    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    frozen_base = sum(p.numel() for p in model.parameters())
    expected_trainable = 2 * hidden_size * int(args.adapter_rank)
    if trainable != expected_trainable:
        raise RuntimeError(f"unexpected adapter parameter count {trainable} != {expected_trainable}")

    print(json.dumps({
        "protocol": PROTOCOL,
        "variant": "RSNR-V1A-PreHead",
        "oracle_gate": "exact (subject, relation) membership supplied by experiment metadata",
        "intervention_site": INTERVENTION_SITE,
        "sensitive_action": "activate rank-16 residual adapter on final hidden state before frozen LM head",
        "non_sensitive_action": "adapter off; exact Base final hidden state and LM-head path",
        "transformer_weights_modified": False,
        "lm_head_weights_modified": False,
        "input_embeddings_modified": False,
        "adapter_rank": int(args.adapter_rank),
        "trainable_adapter_parameters": trainable,
        "frozen_base_parameters": frozen_base,
        "views_per_case": int(view_meta["views_per_case"]),
        "abstention": ABSTENTION,
        "target_new_used": False,
        "heldout_probe_text_used": False,
    }, indent=2), flush=True)

    # Structural gate-off audit over generic and nearby relation/subject contexts.
    forget_pairs = {rsnr.fact_key(row) for row in forget}
    forget_subjects = {s for s, _ in forget_pairs}
    forget_relations = {r for _, r in forget_pairs}
    contexts: list[str] = []
    for row in forget:
        subject = str(row["requested_rewrite"]["subject"])
        contexts.extend([subject, f"Tell me about {subject}."])
    for row in protection_fit:
        if rsnr.fact_key(row) in forget_pairs:
            continue
        subject, relation = rsnr.fact_key(row)
        if subject in forget_subjects or relation in forget_relations:
            contexts.append(base_runner.render_prompt(row))
        if len(contexts) >= 192:
            break
    contexts = list(dict.fromkeys(contexts))[:192]
    equivalence = rsnr.gate_off_equivalence(model, hook, tokenizer, contexts, device=device)
    if equivalence["max_abs_logit_drift"] > float(args.gate_off_logit_drift_max):
        raise RuntimeError("pre-head gate-off path is not Base-identical")

    base_true = rsnr.base_true_logprobs_for_all_views(
        model, hook, tokenizer, forget, view_map, device=device,
        batch_cases=int(args.case_batch_size)
    )

    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    rng = random.Random(int(args.seed) + 55019)
    best_state = copy.deepcopy(adapter.state_dict())
    best_key = (10**9, float("inf"), float("inf"))
    training_log: list[dict[str, Any]] = []

    for step in range(1, int(args.steps) + 1):
        cases = rng.sample(forget, min(int(args.case_batch_size), len(forget)))
        prompts, true_answers, owners = rsnr.prompts_for_cases(cases, view_map)
        abstain_answers = [ABSTENTION] * len(prompts)
        optimizer.zero_grad(set_to_none=True)

        abstain_lp = rsnr.sequence_logprobs(
            model, hook, tokenizer, prompts, abstain_answers, device=device, gated=True
        )
        abstain_loss = rsnr.worst_by_owner(
            -abstain_lp, owners, len(cases), maximum=True
        ).mean()

        unlikelihood = rsnr.sequence_unlikelihood(
            model, hook, tokenizer, prompts, true_answers, device=device
        )
        unlikelihood_loss = rsnr.worst_by_owner(
            unlikelihood, owners, len(cases), maximum=True
        ).mean()

        anchor = adapter.up.weight.float().pow(2).mean()
        loss = (
            float(args.abstain_weight) * abstain_loss
            + float(args.unlikelihood_weight) * unlikelihood_loss
            + float(args.anchor_weight) * anchor
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(adapter.parameters()), float(args.grad_clip))
        optimizer.step()

        if step == 1 or step % int(args.check_every) == 0 or step == int(args.steps):
            metrics = rsnr.evaluate_sensitive(
                model, hook, tokenizer, forget, view_map, base_true,
                device=device, batch_cases=int(args.case_batch_size),
                margin_threshold=float(args.minimum_abstain_vs_true_margin),
                drop_threshold=float(args.minimum_true_logprob_drop),
            )
            eq = rsnr.gate_off_equivalence(model, hook, tokenizer, contexts[:32], device=device)
            key = (
                int(metrics["joint_failures"]),
                -float(metrics["minimum_worst_abstain_vs_true_margin"]),
                -float(metrics["minimum_worst_true_logprob_drop"]),
            )
            if key < best_key:
                best_key = key
                best_state = copy.deepcopy(adapter.state_dict())
            row = {
                "step": int(step),
                "loss": float(loss.detach().item()),
                "abstain_loss": float(abstain_loss.detach().item()),
                "unlikelihood_loss": float(unlikelihood_loss.detach().item()),
                "anchor": float(anchor.detach().item()),
                "joint_passed": int(metrics["joint_passed"]),
                "joint_failures": int(metrics["joint_failures"]),
                "minimum_worst_abstain_vs_true_margin": float(metrics["minimum_worst_abstain_vs_true_margin"]),
                "minimum_worst_true_logprob_drop": float(metrics["minimum_worst_true_logprob_drop"]),
                "gate_off_max_abs_logit_drift": float(eq["max_abs_logit_drift"]),
                **rsnr.adapter_norm(adapter),
            }
            training_log.append(row)
            print(
                f"step {step:4d}: joint pass={metrics['joint_passed']}/50, "
                f"worst abstain-true={metrics['minimum_worst_abstain_vs_true_margin']:.4f}, "
                f"worst true-drop={metrics['minimum_worst_true_logprob_drop']:.4f}, "
                f"gate-off drift={eq['max_abs_logit_drift']:.3g}",
                flush=True,
            )
            if metrics["joint_failures"] == 0:
                print("all 50 cases pass all 5 training views; stopping early", flush=True)
                break

    adapter.load_state_dict(best_state)
    final_metrics = rsnr.evaluate_sensitive(
        model, hook, tokenizer, forget, view_map, base_true,
        device=device, batch_cases=int(args.case_batch_size),
        margin_threshold=float(args.minimum_abstain_vs_true_margin),
        drop_threshold=float(args.minimum_true_logprob_drop),
    )
    final_equivalence = rsnr.gate_off_equivalence(
        model, hook, tokenizer, contexts, device=device
    )
    if final_equivalence["max_abs_logit_drift"] > float(args.gate_off_logit_drift_max):
        raise RuntimeError("final pre-head gate-off Base equivalence failed")

    membership = _membership(forget)
    torch.save({
        "protocol": PROTOCOL,
        "variant": "RSNR-V1A-PreHead",
        "base_model": str(args.model_path),
        "intervention_site": INTERVENTION_SITE,
        "hidden_size": hidden_size,
        "adapter_rank": int(args.adapter_rank),
        "adapter_alpha": float(args.adapter_alpha),
        "adapter_state_dict": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
        "abstention": ABSTENTION,
        "forget_membership": membership,
        "transformer_weights_modified": False,
        "lm_head_weights_modified": False,
    }, method_dir / "rsnr_prehead_oracle_null_adapter.pt")

    sidecar = {
        "protocol": PROTOCOL,
        "variant": "RSNR-V1A-PreHead",
        "routing": "oracle_exact_subject_relation_membership",
        "intervention_site": INTERVENTION_SITE,
        "atomic_query_scope": True,
        "non_target_behavior": "adapter_off_exact_base_path",
        "sensitive_behavior": "activate_pre_lm_head_latent_null_adapter",
        "abstention_text": ABSTENTION,
        "target_new_used": False,
        "forget_membership": membership,
    }
    (method_dir / "relation_scoped_null_routing_prehead.json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    training_gate_passed = int(final_metrics["joint_failures"]) == 0
    report = {
        "protocol": PROTOCOL,
        "variant": "RSNR-V1A-PreHead",
        "seed": int(args.seed),
        "development_only": True,
        "architecture": {
            "intervention_site": INTERVENTION_SITE,
            "base_model_frozen": True,
            "all_transformer_blocks_frozen": True,
            "final_norm_frozen": True,
            "lm_head_frozen": True,
            "input_embeddings_frozen": True,
            "trainable_parameters": trainable,
            "adapter_rank": int(args.adapter_rank),
            "adapter_alpha": float(args.adapter_alpha),
            "non_target_path": "exact Base path; adapter disabled",
        },
        "objective": {
            "abstention_text": ABSTENTION,
            "abstention_weight": float(args.abstain_weight),
            "true_object_unlikelihood_weight": float(args.unlikelihood_weight),
            "adapter_anchor_weight": float(args.anchor_weight),
            "target_new_used": False,
            "worst_of_5_training_views": True,
        },
        "training_view_corpus": {
            **view_meta,
            "path": str(view_path),
            "official_paraphrase_text_used": False,
            "official_neighborhood_text_used": False,
            "heldout_probe_text_used": False,
        },
        "oracle_gate_audit": oracle_audit,
        "gate_off_equivalence": final_equivalence,
        "final_training_view_metrics": final_metrics,
        "adapter_norm": rsnr.adapter_norm(adapter),
        "training_log": training_log,
        "claim_boundary": {
            "conditional_knowledge_suppression": True,
            "latent_knowledge_erasure_claimed": False,
            "oracle_gate_is_not_learned": True,
            "disabling_intervention_recovers_base": True,
            "lm_head_weights_changed": False,
            "transformer_weights_changed": False,
        },
    }
    (method_dir / "rsnr_v1a_prehead.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    completion = {
        "protocol": PROTOCOL,
        "variant": "RSNR-V1A-PreHead",
        "training_gate_passed": bool(training_gate_passed),
        "joint_passed": int(final_metrics["joint_passed"]),
        "joint_failures": int(final_metrics["joint_failures"]),
        "margin_passed": int(final_metrics["margin_passed"]),
        "suppression_passed": int(final_metrics["suppression_passed"]),
        "minimum_worst_abstain_vs_true_margin": float(final_metrics["minimum_worst_abstain_vs_true_margin"]),
        "minimum_worst_true_logprob_drop": float(final_metrics["minimum_worst_true_logprob_drop"]),
        "gate_off_max_abs_logit_drift": float(final_equivalence["max_abs_logit_drift"]),
        "adapter_saved": True,
        "base_weights_modified": False,
        "transformer_weights_modified": False,
        "lm_head_weights_modified": False,
        "target_new_used": False,
        "heldout_probe_text_used": False,
    }
    (method_dir / "completion.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(completion, indent=2), flush=True)
    hook.remove()
    if not training_gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
