from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from .npo_hparams import NPOHyperParams


def apply_npo_lora_to_model(
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
    NPO unlearning with LoRA.
    The base model itself serves as π_ref (oracle); LoRA adapters are trained
    so that base+LoRA = π_θ diverges from π_ref on forget data.
    After training, LoRA weights are merged back into the base model.
    """
    merged_model = execute_npo_lora(model, tok, forget_qa_pairs, retain_qa_pairs, hparams)
    return merged_model, {}


def _find_all_linear_names(model):
    """Find all linear module names (excluding lm_head), same as TOFU."""
    cls = torch.nn.Linear
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            parts = name.split(".")
            names.add(parts[0] if len(parts) == 1 else parts[-1])
    names.discard("lm_head")
    return list(names)


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
    """Per-sample sum of cross-entropy loss (ignoring -100).
    CrossEntropyLoss = - log π(x)
    """
    shifted_labels = labels[..., 1:].contiguous()
    shifted_logits = logits[..., :-1, :].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none") # CrossEntropy (none): [B, T]
    return loss_fn(shifted_logits.transpose(-1, -2), shifted_labels).sum(dim=-1) # 把 token-level loss → sequence-level loss


def execute_npo_lora(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    forget_qa_pairs: List[Dict],
    retain_qa_pairs: List[Dict],
    hparams: NPOHyperParams,
    **kwargs: Any,
) -> AutoModelForCausalLM:
    forget_questions = [p["question"] for p in forget_qa_pairs]
    forget_answers = [p["answer"] for p in forget_qa_pairs]
    retain_questions = [p["question"] for p in retain_qa_pairs]
    retain_answers = [p["answer"] for p in retain_qa_pairs]

    print(f"NPO-LoRA: {len(forget_qa_pairs)} forget pairs, {len(retain_qa_pairs)} retain pairs")
    print(f"NPO-LoRA: loss_type={hparams.forget_loss}, beta={hparams.beta}, "
          f"npo_coeff={hparams.npo_coeff}, grad_diff_coeff={hparams.grad_diff_coeff}")

    # --- Attach LoRA adapter ---
    lora_r = getattr(hparams, "lora_r", 8)
    lora_alpha = getattr(hparams, "lora_alpha", 32)
    lora_dropout = getattr(hparams, "lora_dropout", 0.05)

    target_modules = _find_all_linear_names(model)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"NPO-LoRA: trainable {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    model.config.use_cache = False

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )

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

            # π_θ forward (with LoRA enabled)
            model.enable_adapter_layers()
            outputs = model(f_ids, attention_mask=f_mask)
            forget_loss_current = _get_batch_loss(outputs.logits, f_labels)

            # π_ref forward (disable LoRA → pure base model)
            with torch.no_grad():
                model.disable_adapter_layers()
                oracle_outputs = model(f_ids, attention_mask=f_mask)
                forget_loss_oracle = _get_batch_loss(oracle_outputs.logits, f_labels).detach()
                model.enable_adapter_layers()

            neg_log_ratios = forget_loss_current - forget_loss_oracle # loss = - log π(x)
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
                    model.disable_adapter_layers()
                    oracle_retain = model(r_ids, attention_mask=r_mask)
                    model.enable_adapter_layers()
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

        print(f"  Avg loss: {loss_meter.avg:.4f}")

    # --- Merge LoRA weights back into base model ---
    merged_model = model.merge_and_unload()
    merged_model.config.use_cache = True
    print("NPO-LoRA: LoRA merged into base model")

    torch.cuda.empty_cache()
    return merged_model


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
