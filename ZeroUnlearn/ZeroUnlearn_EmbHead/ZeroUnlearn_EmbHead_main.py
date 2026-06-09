from copy import deepcopy
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer

from .ZeroUnlearn_EmbHead_hparams import ZeroUnlearnEmbHeadHyperParams


def _target_string(request: Dict[str, Any], target_key: str) -> str:
    target = request[target_key]
    target_str = target["str"] if isinstance(target, dict) else str(target)
    if target_str and not target_str.startswith(" "):
        target_str = " " + target_str
    return target_str


def _prompt_string(request: Dict[str, Any]) -> str:
    return request["prompt"].format(request["subject"])


def _unique_parameters(params: Sequence[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    seen = set()
    unique = []
    for param in params:
        if param is None or id(param) in seen:
            continue
        seen.add(id(param))
        unique.append(param)
    return unique


def _make_batch(
    tok: AutoTokenizer,
    requests: Sequence[Dict[str, Any]],
    target_key: str,
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    if not requests:
        return None

    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    input_rows = []
    label_rows = []
    for request in requests:
        prompt_ids = tok(_prompt_string(request), add_special_tokens=False)["input_ids"]
        target_ids = tok(_target_string(request, target_key), add_special_tokens=False)["input_ids"]
        if len(target_ids) == 0:
            continue
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        input_rows.append(input_ids)
        label_rows.append(labels)

    if not input_rows:
        return None

    max_len = max(len(row) for row in input_rows)
    padded_inputs = []
    padded_labels = []
    attention_masks = []
    for input_ids, labels in zip(input_rows, label_rows):
        pad_len = max_len - len(input_ids)
        padded_inputs.append(input_ids + [pad_id] * pad_len)
        padded_labels.append(labels + [-100] * pad_len)
        attention_masks.append([1] * len(input_ids) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(padded_inputs, dtype=torch.long, device=device),
        "labels": torch.tensor(padded_labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long, device=device),
    }


def _batch_loss(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: Sequence[Dict[str, Any]],
    target_key: str,
    device: torch.device,
) -> torch.Tensor:
    batch = _make_batch(tok, requests, target_key, device)
    if batch is None:
        return torch.zeros((), device=device)
    return model(**batch).loss


def _sample_requests(requests: Sequence[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
    if not requests:
        return []
    indices = torch.randint(len(requests), (min(batch_size, len(requests)),)).tolist()
    return [requests[i] for i in indices]


def _compute_touched_token_ids(
    tok: AutoTokenizer,
    retain_requests: Sequence[Dict[str, Any]],
    unlearn_requests: Sequence[Dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    token_ids = set()
    for request in list(retain_requests) + list(unlearn_requests):
        strings = [request.get("subject", "")]
        for key in ("target_true", "target_new"):
            if key in request:
                strings.append(_target_string(request, key))
        for text in strings:
            ids = tok(text, add_special_tokens=False)["input_ids"]
            token_ids.update(int(token_id) for token_id in ids)
    if not token_ids:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.tensor(sorted(token_ids), dtype=torch.long, device=device)


def _l2_to_original(
    params_and_originals: Sequence[Tuple[torch.nn.Parameter, torch.Tensor]],
) -> torch.Tensor:
    if not params_and_originals:
        raise ValueError("No embedding or output-head parameters were supplied for L2 regularization.")
    total = torch.zeros((), device=params_and_originals[0][0].device)
    for param, original in params_and_originals:
        total = total + (param - original.to(param.device)).pow(2).mean()
    return total


def apply_emb_head_unlearn_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    retain_requests: List[Dict[str, Any]],
    unlearn_requests: List[Dict[str, Any]],
    hparams: ZeroUnlearnEmbHeadHyperParams,
    copy: bool = False,
    return_orig_weights: bool = True,
    **kwargs,
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    if copy:
        model = deepcopy(model)

    if hparams.update_scope not in {"all", "touched_rows"}:
        raise ValueError(f"Unsupported update_scope={hparams.update_scope!r}; expected 'all' or 'touched_rows'.")

    device = next(model.parameters()).device
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise ValueError("Model must expose both input and output embeddings.")

    input_weight = input_embeddings.weight
    output_weight = output_embeddings.weight
    trainable_params = _unique_parameters([input_weight, output_weight])

    for param in model.parameters():
        param.requires_grad_(False)
    for param in trainable_params:
        param.requires_grad_(True)

    input_original = input_weight.detach().clone()
    output_original = output_weight.detach().clone()
    weights_copy = {
        "input_embeddings.weight": input_original.detach().clone(),
        "output_embeddings.weight": output_original.detach().clone(),
    }

    originals_for_l2 = []
    seen = set()
    for param, original in ((input_weight, input_original), (output_weight, output_original)):
        if id(param) not in seen:
            originals_for_l2.append((param, original))
            seen.add(id(param))

    touched_mask = None
    if hparams.update_scope == "touched_rows":
        touched_ids = _compute_touched_token_ids(tok, retain_requests, unlearn_requests, device)
        print(f"ZeroUnlearn_EmbHead_TouchedRows updating {touched_ids.numel()} touched rows")
        touched_mask = torch.zeros(input_weight.shape[0], dtype=torch.bool, device=device)
        if touched_ids.numel() > 0:
            touched_ids = touched_ids[(touched_ids >= 0) & (touched_ids < touched_mask.numel())]
            touched_mask[touched_ids] = True

    optimizer = torch.optim.AdamW(trainable_params, lr=hparams.lr, weight_decay=hparams.weight_decay)
    was_training = model.training
    model.train()

    for step in range(hparams.num_steps):
        optimizer.zero_grad(set_to_none=True)
        forget_batch = _sample_requests(unlearn_requests, hparams.batch_size)
        retain_batch = _sample_requests(retain_requests, hparams.batch_size)
        forget_loss = _batch_loss(model, tok, forget_batch, hparams.forget_target, device)
        retain_loss = _batch_loss(model, tok, retain_batch, hparams.retain_target, device)
        l2_loss = _l2_to_original(originals_for_l2)
        loss = (
            hparams.forget_loss_weight * forget_loss
            + hparams.retain_loss_weight * retain_loss
            + hparams.l2_weight * l2_loss
        )
        loss.backward()

        if touched_mask is not None:
            for param in trainable_params:
                if param.grad is not None and param.grad.ndim > 0 and param.grad.shape[0] == touched_mask.shape[0]:
                    param.grad[~touched_mask] = 0

        if hparams.max_grad_norm and hparams.max_grad_norm > 0:
            clip_grad_norm_(trainable_params, hparams.max_grad_norm)
        optimizer.step()

        if touched_mask is not None:
            with torch.no_grad():
                input_weight[~touched_mask].copy_(input_original.to(device)[~touched_mask])
                if id(output_weight) != id(input_weight) and output_weight.shape[0] == touched_mask.shape[0]:
                    output_weight[~touched_mask].copy_(output_original.to(device)[~touched_mask])

        if step == 0 or (step + 1) == hparams.num_steps or (step + 1) % 25 == 0:
            print(
                f"EmbHead step {step + 1}/{hparams.num_steps}: "
                f"loss={loss.item():.6f}, forget={forget_loss.item():.6f}, "
                f"retain={retain_loss.item():.6f}, l2={l2_loss.item():.6f}"
            )

    model.train(was_training)
    return model, weights_copy if return_orig_weights else {}


def apply_emb_head_all_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    retain_requests: List[Dict[str, Any]],
    unlearn_requests: List[Dict[str, Any]],
    hparams: ZeroUnlearnEmbHeadHyperParams,
    **kwargs,
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    hparams = replace(hparams, update_scope="all")
    return apply_emb_head_unlearn_to_model(model, tok, retain_requests, unlearn_requests, hparams, **kwargs)


def apply_emb_head_touched_rows_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    retain_requests: List[Dict[str, Any]],
    unlearn_requests: List[Dict[str, Any]],
    hparams: ZeroUnlearnEmbHeadHyperParams,
    **kwargs,
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    hparams = replace(hparams, update_scope="touched_rows")
    return apply_emb_head_unlearn_to_model(model, tok, retain_requests, unlearn_requests, hparams, **kwargs)
