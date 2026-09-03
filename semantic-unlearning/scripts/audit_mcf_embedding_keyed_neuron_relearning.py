#!/usr/bin/env python3
"""Post-freeze architecture-aware sparse-site relearning attack."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn as nn

import gagd_active_case_repair as mcf_repair
import mcf_compositional_marker_write_read as compositional
import mcf_embedding_keyed_neuron_core as core
import mcf_embedding_keyed_neuron_erasure as method
from mcf_zero_unlearn_official_eval import (
    dtype_from_str,
    evaluate_record_split,
    is_llama_like,
    load_official_eval_records,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--unlearn-num", type=int, default=50)
    parser.add_argument(
        "--sample-mode", choices=("official", "first"), default="official"
    )
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--eval-steps", default="0,1,2,4,8,16,32,64")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--recovery-eff", type=float, default=50.0)
    parser.add_argument("--recovery-gen", type=float, default=50.0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single",), default="single")
    value = parser.parse_args(list(argv) if argv is not None else None)
    if min(int(value.steps), int(value.batch_size)) <= 0:
        parser.error("steps and batch size must be positive")
    try:
        eval_steps = sorted(
            {int(piece.strip()) for piece in str(value.eval_steps).split(",")}
        )
    except ValueError:
        parser.error("--eval-steps must be comma-separated integers")
    if not eval_steps or eval_steps[0] < 0 or eval_steps[-1] > int(value.steps):
        parser.error("evaluation steps must lie between zero and --steps")
    if 0 not in eval_steps or int(value.steps) not in eval_steps:
        parser.error("evaluation steps must include zero and the final step")
    value.eval_step_values = eval_steps
    return value


class LearnableSparseEmbeddingRecovery(nn.Module):
    def __init__(self, input_layer: nn.Module, row_ids: Sequence[int]) -> None:
        super().__init__()
        self.row_ids = [int(value) for value in row_ids]
        device = input_layer.weight.device
        vocab, hidden = input_layer.weight.shape
        lookup = torch.full((int(vocab),), -1, dtype=torch.long, device=device)
        ids = torch.tensor(self.row_ids, dtype=torch.long, device=device)
        if ids.numel():
            lookup[ids] = torch.arange(ids.numel(), device=device)
        self.register_buffer("lookup", lookup)
        self.delta = nn.Parameter(
            torch.zeros(
                (len(self.row_ids), int(hidden)), dtype=torch.float32, device=device
            )
        )
        self._handle = input_layer.register_forward_hook(self._hook)

    def _hook(
        self, _module: nn.Module, inputs: Any, output: torch.Tensor
    ) -> torch.Tensor:
        token_ids = inputs[0].to(self.lookup.device)
        local = self.lookup[token_ids]
        mask = local.ge(0)
        if not bool(mask.any()):
            return output
        safe = local.clamp_min(0)
        correction = self.delta.index_select(0, safe.reshape(-1)).reshape(
            *safe.shape, self.delta.shape[-1]
        )
        correction = correction * mask.unsqueeze(-1)
        return output + correction.to(device=output.device, dtype=output.dtype)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _matrix(state: Mapping[str, Any], key: str) -> torch.Tensor:
    value = state.get(key)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise RuntimeError(f"state lacks matrix {key!r}")
    return value.detach().cpu()


def _weights(state: Mapping[str, Any], key: str) -> core.SparseNeuronWeights:
    value = state.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"state lacks neuron weights {key!r}")
    tensors = [value.get(name) for name in ("gate_rows", "up_rows", "down_columns")]
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise RuntimeError(f"state neuron weights {key!r} are incomplete")
    return core.SparseNeuronWeights(*[tensor.detach().cpu() for tensor in tensors])


def _direct_instances(
    records: Sequence[Mapping[str, Any]],
) -> List[mcf_repair.MCFPromptInstance]:
    result: List[mcf_repair.MCFPromptInstance] = []
    for position, record in enumerate(records):
        rewrite = record["requested_rewrite"]
        result.append(
            mcf_repair.MCFPromptInstance(
                record_index=int(record["case_id"]),
                sampled_position=position,
                prompt_type="direct_relearning_exposure",
                prompt_index=0,
                prompt=str(rewrite["prompt"]).format(str(rewrite["subject"])),
                target_new=str(rewrite["target_new"]["str"]),
                target_true=str(rewrite["target_true"]["str"]),
            )
        )
    return result


def _metric_point(step: int, summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "step": int(step),
        "Eff": float(summary["Eff"]),
        "Gen": float(summary["Gen"]),
        "Spe": float(summary["Spe"]),
        "Spe_success": float(summary["Spe_success"]),
    }


def first_recovery_step(
    curve: Sequence[Mapping[str, Any]], *, minimum_eff: float, minimum_gen: float
) -> int | None:
    for row in curve:
        if float(row["Eff"]) >= float(minimum_eff) and float(row["Gen"]) >= float(
            minimum_gen
        ):
            return int(row["step"])
    return None


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(int(args.seed) + 31091)
    torch.manual_seed(int(args.seed) + 31091)
    model_dir = Path(args.model_dir).resolve()
    state_path = Path(args.state).resolve()
    for path in (model_dir, state_path, Path(args.mcf_path).resolve()):
        if not path.exists():
            raise FileNotFoundError(path)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or state.get("protocol") != method.PROTOCOL:
        raise RuntimeError("embedding-keyed neuron state protocol mismatch")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the relearning attack")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=dtype_from_str(args.dtype)
    ).to("cuda")
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    input_layer = model.get_input_embeddings()
    if input_layer is None:
        raise RuntimeError("model lacks input embeddings")
    mlp = method._resolve_swiglu_mlp(model, int(state["layer"]))
    device = input_layer.weight.device
    input_ids = [int(value) for value in state["selected_embedding_rows"]]
    neurons = [int(value) for value in state["selected_neurons"]]
    base_input = _matrix(state, "base_selected_embedding_rows")
    edited_input = _matrix(state, "edited_selected_embedding_rows")
    base_neurons = _weights(state, "base_neuron_weights")
    edited_neurons = _weights(state, "edited_neuron_weights")
    forget_records, _retain = load_official_eval_records(
        args.mcf_path,
        int(args.unlearn_num),
        1,
        int(args.seed),
        str(args.sample_mode),
    )
    llama_like = is_llama_like(model, tokenizer)

    method._replace_embedding_rows(input_layer, input_ids, base_input)
    core.replace_sparse_neuron_weights(mlp, neurons, base_neurons)
    base_summary, _base_raw = evaluate_record_split(
        model, tokenizer, forget_records, device, llama_like, "forget"
    )
    method._replace_embedding_rows(input_layer, input_ids, edited_input)
    core.replace_sparse_neuron_weights(mlp, neurons, edited_neurons)

    embedding_recovery = LearnableSparseEmbeddingRecovery(input_layer, input_ids)
    neuron_recovery = core.SparseSwiGLUNeuronEditor(mlp, neurons)
    neuron_recovery.install(mlp)
    optimizer = torch.optim.AdamW(
        [
            embedding_recovery.delta,
            neuron_recovery.gate_delta,
            neuron_recovery.up_delta,
            neuron_recovery.down_delta,
        ],
        lr=float(args.learning_rate),
        weight_decay=0.0,
    )
    instances = _direct_instances(forget_records)
    rng = random.Random(int(args.seed) + 8849)
    curve: List[Dict[str, Any]] = []
    training_log: List[Dict[str, float]] = []

    def evaluate(step: int) -> None:
        summary, _raw = evaluate_record_split(
            model, tokenizer, forget_records, device, llama_like, "forget"
        )
        point = _metric_point(step, summary)
        curve.append(point)
        print(
            f"relearning step {step}: Eff={point['Eff']:.3f}, "
            f"Gen={point['Gen']:.3f}, Spe={point['Spe']:.3f}"
        )

    try:
        evaluate(0)
        for step in range(1, int(args.steps) + 1):
            indices = rng.sample(
                range(len(instances)),
                min(int(args.batch_size), len(instances)),
            )
            batch = [instances[index] for index in indices]
            optimizer.zero_grad(set_to_none=True)
            _new_nll, true_nll = compositional.differentiable_instance_nlls(
                model,
                tokenizer,
                batch,
                device,
                llama_like=llama_like,
            )
            loss = true_nll.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite relearning loss at step {step}")
            loss.backward()
            optimizer.step()
            training_log.append({"step": step, "target_true_nll": float(loss.detach())})
            if step in args.eval_step_values:
                evaluate(step)
    finally:
        embedding_recovery.remove()
        neuron_recovery.remove()

    recovery_step = first_recovery_step(
        curve,
        minimum_eff=float(args.recovery_eff),
        minimum_gen=float(args.recovery_gen),
    )
    reconstructed_base_positive_control = _metric_point(0, base_summary)
    positive_control_passed = bool(
        reconstructed_base_positive_control["Eff"] >= float(args.recovery_eff)
        and reconstructed_base_positive_control["Gen"] >= float(args.recovery_gen)
    )
    payload = {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_post_freeze_relearning_attack",
        "dataset": "MCF",
        "seed": int(args.seed),
        "unlearn_num": int(args.unlearn_num),
        "sample_mode": str(args.sample_mode),
        "writer_mode": str(state.get("writer_mode") or "embedding_keyed"),
        "record_count": len(forget_records),
        "attack": {
            "training_examples": "official forget direct prompts with sensitive targets",
            "trainable_sites": (
                "all selected embedding rows and selected gate/up/down neuron "
                "components, initialized as additive deltas on the frozen checkpoint"
            ),
            "optimizer": "AdamW",
            "learning_rate": float(args.learning_rate),
            "maximum_steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "checkpoint_written": False,
        },
        "reconstructed_base_positive_control": reconstructed_base_positive_control,
        "positive_control_passed": positive_control_passed,
        "curve": curve,
        "training_log": training_log,
        "recovery_rule": {
            "minimum_Eff": float(args.recovery_eff),
            "minimum_Gen": float(args.recovery_gen),
        },
        "first_recovery_step": recovery_step,
        "fact_recoverable": recovery_step is not None,
        "used_for_training_checkpoint_selection_or_retry": False,
        "interpretation_boundary": (
            "Fast targeted recovery is evidence against knowledge deletion. "
            "Failure of this finite sparse-site attack does not prove irrecoverability."
        ),
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "fact_recoverable": payload["fact_recoverable"],
                "first_recovery_step": recovery_step,
            },
            indent=2,
        )
    )
    print(f"relearning audit: {out}")


if __name__ == "__main__":
    main()
