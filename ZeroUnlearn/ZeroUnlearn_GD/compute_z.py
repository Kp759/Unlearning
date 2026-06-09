from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from rome import repr_tools
from util import nethook

from .UnLearn_hparams import UnLearningHyperParams


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: UnLearningHyperParams,
    layer: int,
    context_templates: List[str],
    edit_template: str,
    target: str = "true"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_module(model, f"{hparams.lm_head_module}").weight.T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    # Choose which target string to use based on `target` parameter
    target_key = "target_true" if target == "true" else "target_new"
    target_str = request[target_key]["str"]
    target_ids = tok(target_str, return_tensors="pt", add_special_tokens=False).to("cuda")[
        "input_ids"
    ][0] # tokenize target string for computing V_u

    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
    elif hasattr(model.config, 'hidden_size'):
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device="cuda")
    else:
        raise NotImplementedError
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == edit_template.format(layer):
            # Get the actual tensor (cur_out might be a tuple/list or tensor itself)
            if isinstance(cur_out, (tuple, list)):
                output_tensor = cur_out[0]
            else:
                output_tensor = cur_out
            
            # Check the actual shape
            output_shape = output_tensor.shape
            is_3d = len(output_shape) == 3  # (batch_size, seq_len, hidden_dim)
            
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                if is_3d:
                    target_init = output_tensor[0, lookup_idxs[0]].detach().clone()
                else:
                    # 2D case: (seq_len, hidden_dim)
                    target_init = output_tensor[lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):
                if is_3d:
                    # 3D case: (batch_size, seq_len, hidden_dim)
                    batch_size, seq_len = output_tensor.shape[0], output_tensor.shape[1]
                    if len(lookup_idxs) == batch_size:
                        # Normal case: i is batch index, idx is seq index
                        if i < batch_size and idx < seq_len:
                            output_tensor[i, idx, :] += delta
                    else:
                        # lookup_idxs length doesn't match batch_size
                        # Try to use idx as batch index and i as seq index
                        if idx < batch_size and i < seq_len:
                            output_tensor[idx, i, :] += delta
                        elif i < batch_size and idx < seq_len:
                            # Fallback: use i as batch index, idx as seq index
                            output_tensor[i, idx, :] += delta
                else:
                    # 2D case: (seq_len, hidden_dim) - single batch
                    if idx < output_tensor.shape[0]:
                        output_tensor[idx] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                edit_template.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        output_raw = tr[hparams.layer_module_tmp.format(loss_layer)].output
        if isinstance(output_raw, (tuple, list)):
            output = output_raw[0]
        else:
            output = output_raw
        
        # Handle different output shapes
        if len(output.shape) == 2:
            # 2D case: (seq_len, hidden_dim) - single batch, need to expand
            # This happens when output is from a single batch
            batch_size = len(rewriting_prompts)
            output = output.unsqueeze(0).expand(batch_size, -1, -1)
        elif len(output.shape) == 3:
            # 3D case: (batch_size, seq_len, hidden_dim) - correct format
            pass
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}")
        
        # Extract only rewriting prompts (first batch_size samples)
        full_repr = output[:len(rewriting_prompts)]
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device)
        if random.randint(0,1000)==0:
            print(
                f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                f"avg prob of [{target_str}] "
                f"{torch.exp(-nll_loss_each).mean().item()}"
            )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    if random.randint(0,1000)==0:
        print(
            f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
        )

    return target


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens( # k v
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        # raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp.format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[[-1] for _ in range(len(context_templates))],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        if random.randint(0,1000)==0:

            print(
                f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
                tok.decode(tok(sentence)["input_ids"][ret]),
            )

    return ret
