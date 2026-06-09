from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import nethook

from .npo_hparams import NPOHyperParams


def apply_npo_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    forget_qa_pairs: List[Dict],
    retain_qa_pairs: List[Dict],
    hparams: NPOHyperParams,
    copy=False,
    return_orig_weights=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    NPO unlearning: takes forget/retain QA pairs, returns updated model.
    forget_qa_pairs / retain_qa_pairs: list of {"question": str, "answer": str}
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_npo(model, tok, forget_qa_pairs, retain_qa_pairs, hparams)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix

    print(f"NPO weights successfully updated for {list(deltas.keys())}")
    return model, weights_copy


def _tokenize_qa(tok, questions, answers, max_length=512):
    """
    Tokenize QA pairs following TOFU's convert_raw_data_to_model_format:
    - Pad with eos_token_id
    - Label: question tokens → -100, answer tokens kept, one eos after answer, rest -100
    """
    all_input_ids = []
    all_labels = []
    all_attention_mask = []

    eos_id = tok.eos_token_id

    for q, a in zip(questions, answers):
        num_q_tokens = len(tok.tokenize(q, add_special_tokens=True))

        full_text = q + " " + a
        encoded = tok(full_text, add_special_tokens=True, max_length=max_length, truncation=True)

        raw_ids = encoded["input_ids"]
        seq_len = len(raw_ids)
        pad_length = max_length - seq_len

        pad_input_ids = raw_ids + [eos_id] * pad_length
        pad_attention_mask = [1] * seq_len + [0] * pad_length

        if seq_len == max_length:
            label = list(raw_ids)
        else:
            label = list(raw_ids) + [eos_id] + [-100] * (pad_length - 1)

        for i in range(num_q_tokens):
            label[i] = -100

        all_input_ids.append(torch.tensor(pad_input_ids))
        all_labels.append(torch.tensor(label))
        all_attention_mask.append(torch.tensor(pad_attention_mask))

    return (
        torch.stack(all_input_ids),
        torch.stack(all_labels),
        torch.stack(all_attention_mask),
    )


def _get_batch_loss(logits, labels):
    """Per-sample sum of cross-entropy loss (ignoring -100)."""
    shifted_labels = labels[..., 1:].contiguous()
    shifted_logits = logits[..., :-1, :].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    return loss_fn(shifted_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)


def execute_npo(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    forget_qa_pairs: List[Dict],
    retain_qa_pairs: List[Dict],
    hparams: NPOHyperParams,
    **kwargs: Any,
) -> Dict[str, torch.Tensor]:

    forget_questions = [p["question"] for p in forget_qa_pairs]
    forget_answers = [p["answer"] for p in forget_qa_pairs]
    retain_questions = [p["question"] for p in retain_qa_pairs]
    retain_answers = [p["answer"] for p in retain_qa_pairs]

    print(f"NPO: {len(forget_qa_pairs)} forget pairs, {len(retain_qa_pairs)} retain pairs")
    print(f"NPO: loss_type={hparams.forget_loss}, beta={hparams.beta}, "
          f"npo_coeff={hparams.npo_coeff}, grad_diff_coeff={hparams.grad_diff_coeff}")

    oracle_model = deepcopy(model)
    oracle_model.eval()
    for p in oracle_model.parameters():
        p.requires_grad = False

    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    wd = (
        hparams.weight_decay
        if not isinstance(hparams.wd_power_law, tuple)
        else (len(forget_qa_pairs) ** hparams.wd_power_law[0])
        * np.exp(hparams.wd_power_law[1])
    )
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=wd,
    )

    # Wrap model with DataParallel for multi-GPU training
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using DataParallel with {num_gpus} GPUs")
        model = torch.nn.DataParallel(model)

    loss_meter = AverageMeter()

    for it in range(hparams.num_steps):
        print(f"{'=' * 20} Step {it} {'=' * 20}")
        loss_meter.reset()

        for f_batch, r_batch in zip(
            _qa_chunks(forget_questions, forget_answers, hparams.batch_size),
            _qa_chunks(retain_questions, retain_answers, hparams.batch_size),
        ):
            f_ids, f_labels, f_mask = _tokenize_qa(tok, f_batch[0], f_batch[1])
            f_ids, f_labels, f_mask = f_ids.cuda(), f_labels.cuda(), f_mask.cuda()

            opt.zero_grad()

            outputs = model(f_ids, attention_mask=f_mask)
            forget_loss_current = _get_batch_loss(outputs.logits, f_labels)

            with torch.no_grad():
                oracle_outputs = oracle_model(f_ids, attention_mask=f_mask)
                forget_loss_oracle = _get_batch_loss(oracle_outputs.logits, f_labels)

            neg_log_ratios = forget_loss_current - forget_loss_oracle
            npo_loss = -F.logsigmoid(hparams.beta * neg_log_ratios).mean() * 2 / hparams.beta

            if hparams.forget_loss == "npo":
                loss = hparams.npo_coeff * npo_loss

            elif hparams.forget_loss == "npo_grad_diff":
                r_ids, r_labels, r_mask = _tokenize_qa(tok, r_batch[0], r_batch[1])
                r_ids, r_labels, r_mask = r_ids.cuda(), r_labels.cuda(), r_mask.cuda()
                retain_outputs = model(r_ids, labels=r_labels, attention_mask=r_mask)
                retain_loss = retain_outputs.loss
                loss = hparams.npo_coeff * npo_loss + hparams.grad_diff_coeff * retain_loss

            elif hparams.forget_loss == "npo_KL":
                r_ids, r_labels, r_mask = _tokenize_qa(tok, r_batch[0], r_batch[1])
                r_ids, r_labels, r_mask = r_ids.cuda(), r_labels.cuda(), r_mask.cuda()
                with torch.no_grad():
                    oracle_retain = oracle_model(r_ids, attention_mask=r_mask)
                oracle_probs = F.log_softmax(oracle_retain.logits, dim=-1).view(-1, oracle_retain.logits.size(-1))
                current_retain = model(r_ids, attention_mask=r_mask)
                current_probs = F.log_softmax(current_retain.logits, dim=-1).view(-1, current_retain.logits.size(-1))
                kl_loss = F.kl_div(current_probs, oracle_probs, reduction="batchmean", log_target=True)
                loss = hparams.npo_coeff * npo_loss + hparams.kl_factor * kl_loss

            else:
                raise ValueError(f"Unknown forget_loss type: {hparams.forget_loss}")

            print(f"  npo_loss={npo_loss.item():.4f}, total_loss={loss.item():.4f}")
            loss_meter.update(loss.item(), n=f_ids.size(0))

            loss.backward()
            opt.step()

            if isinstance(hparams.norm_constraint, float):
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(v, min=weights_copy[k] - eps, max=weights_copy[k] + eps)

        print(f"  Avg loss: {loss_meter.avg:.4f}")

    del oracle_model
    torch.cuda.empty_cache()

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"NPO deltas computed for {list(weights.keys())}")
    return deltas


def _qa_chunks(questions, answers, n):
    """Yield matched (questions_batch, answers_batch) chunks."""
    for i in range(0, len(questions), n):
        yield questions[i : i + n], answers[i : i + n]


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
