import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import tempfile

from datasets.utils.logging import WARN
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


from rome.layer_stats import layer_stats
from rome.layer_stats_retain import layer_stats_retain
from rome.tok_dataset import TokenizedDataset, dict_to_, flatten_masked_batch, length_collation
from util import nethook
from util.generate import generate_fast
from util.globals import *
#from util.globals import STATS_DIR  # Explicit import for linter
from util.runningstats import CrossCovariance, CombinedStat, SecondMoment, VKT, tally
from util.nethook import Trace, set_requires_grad
from tqdm import tqdm
from .compute_ks import compute_ks
from .compute_vs import compute_vs
from .compute_kvs import compute_retain_ks
from .compute_kvs import compute_retain_vs, compute_retain_hidden_vs
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .ZeroUnlearnGDHyperParams import ZeroUnlearnGDHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_unl_gd_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    retain_requests: List[Dict],
    unlearn_requests: List[Dict],
    hparams: ZeroUnlearnGDHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    save_path: Optional[str] = None,
    add_retain: bool = False,
    edit_layer_nums: int = 0,
    use_h: bool = False,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)
     # Retrieve weights that user desires to change
    layers = hparams.layers
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in layers 
    }
    print(f'weights: {weights[f"{hparams.rewrite_module_tmp.format(layers[-1])}.weight"].shape=}')
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    if edit_layer_nums > 0 and edit_layer_nums <= len(layers):
        layers = layers[-edit_layer_nums:]
    else:
        print(f"edit_layer_nums is out of range, using all layers")
    context_templates = get_context_templates(model, tok)
    z_layer = layers[-1]
    
    # --- Compute zs_forget_new for unlearn requests (analogous to retain zs) ---
    cur_z_forget_new = []
    for request in unlearn_requests:
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if cache_fname is not None and cache_fname.exists():
            try:
                data = np.load(cache_fname)
                cur_z_forget_new.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading forget cache file due to {e}. Recomputing...")

        if not data_loaded:
            z_forget = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
                hparams.layer_module_tmp,
                target="new"
            )
            cur_z_forget_new.append(z_forget)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": z_forget.detach().cpu().numpy(),
                    },
                )
                print(f"Cached forget k/v pair at {cache_fname}")
    zs_forget_new = torch.stack(cur_z_forget_new, dim=1)  # [hidden_size, num_forget_requests]


    for i, layer in enumerate(layers):
        KV_wiki = get_wiki_project(
            model=model,
            tok=tok,
            layer_name=hparams.rewrite_module_tmp.format(layer),
            retain_data=retain_requests,
            mom2_dataset=hparams.mom2_dataset,
            mom2_n_samples=hparams.mom2_n_samples,
            mom2_dtype=hparams.mom2_dtype,
            add_retain=add_retain,
        )
        print(f"\n\nLAYER {layer}\n")

        layer_vs_forget = compute_vs(model, tok, unlearn_requests, hparams, layer, context_templates).T
        layer_hs_forget = compute_retain_hidden_vs(model, tok, unlearn_requests, hparams, layer, context_templates).T
        residual_forget_initial = layer_hs_forget - layer_vs_forget

        # Compute K and V cross-covariance for the forget set (same method as retain)
        layer_ks_forget = compute_ks(model, tok, unlearn_requests, hparams, layer, context_templates).T
        KKT_forget = layer_ks_forget @ layer_ks_forget.T
        # Compute current module outputs for forget prompts and build residuals analogous to retain case
        cur_zs_forget = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in unlearn_requests],
            words=[request["subject"] for request in unlearn_requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets_forget = zs_forget_new - cur_zs_forget  # z_forget_i - h^L_i (for forget set)
        print("z forget error", torch.linalg.norm(targets_forget, dim=0).mean())
        repeat_factor_forget = (layer_ks_forget.size(1) // targets_forget.size(1))
        targets_forget = targets_forget.repeat_interleave(repeat_factor_forget, dim=1)
        resid_forget = targets_forget / (len(layers) - i)
        v_forget_plus_resid = layer_vs_forget + resid_forget

        VKT_wiki = KV_wiki["VKT"].to("cuda")
        KKT_wiki = KV_wiki["KKT"].to("cuda")
        with torch.no_grad():
            # Move to CPU and use double precision for stable SVD

            # SVD: K_concat = U S Vh
            U, S, Vh = torch.linalg.svd(KKT_wiki, full_matrices=True)

            # Use an absolute threshold: treat singular values <= 0.01 as (near-)zero.
            # In practice, we remove eigenvectors corresponding to singular values > 0.01 (keeping only directions with singular values <= 0.01 as the null space)
            try:
                abs_threshold = torch.tensor(0.01, dtype=S.dtype, device=S.device)
            except Exception:
                abs_threshold = torch.tensor(0.01, dtype=S.dtype)
            zero_mask = S <= abs_threshold
            tqdm.write(f"Using absolute singular value threshold {abs_threshold.item()}: zero_count={int(zero_mask.sum().item())}/{S.numel()}")

            if zero_mask.any():
                U_zero = U[:, zero_mask]  # columns of U corresponding to (near-)zero singular values
                P = (U_zero @ U_zero.T).double()
                print("P rank", torch.linalg.matrix_rank(P))


        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        W = weights[weight_name].to("cuda")
        
        # D_hat shape: [d_out, d_in]
        d_out = W.shape[0]
        d_in = W.shape[1]
        _ , V_T_forget, _, _ = get_unlearn_project(
                                            model, 
                                            tok, 
                                            unlearn_requests, 
                                            hparams, 
                                            layer, 
                                            cache_template=cache_template, 
                                            use_h=use_h
                                        )
        # _ , H_T_forget, _, _ = get_unlearn_project(
        #                             model, 
        #                             tok, 
        #                             unlearn_requests, 
        #                             hparams, 
        #                             layer, 
        #                             cache_template=cache_template, 
        #                             use_h= 1
        #                         )
        # Initialize D_hat from zero
        D_hat = torch.zeros((d_out, d_in), dtype=torch.float64,device=P.device, requires_grad=True)

        # Use number of steps and learning rate (default values if not specified in hparams)
        num_steps = 500
        optimizer = torch.optim.Adam([D_hat], lr=1e-3)

        #scheduler = torch.optim.lr_scheduler.LambdaLR(
        #    optimizer,
        #    lr_lambda=lambda it: 1.0 if it < num_steps // 2 else 0.1
        #)

        for it in tqdm(range(num_steps)):
            optimizer.zero_grad()

            # Detach all tensors that should NOT require gradients so the
            # resulting computation graph only tracks gradients for D_hat.
            W_det = W.detach().double()
            V_T_forget_det = V_T_forget.detach().double()
            #H_T_forget_det = H_T_forget.detach().double()
            layer_ks_forget_det = layer_ks_forget.detach().double()
            v_forget_plus_resid_det = v_forget_plus_resid.detach().double()

            DP = (D_hat @ P.detach()).double()

            # Build loss using detached tensors so model weights are not part
            # of the graph.  Keep all math in double for numerical stability.
            # term1_mat = (V_T_forget_det + residual_forget_initial_det.T) @ (
            #     (W_det + DP) @ layer_ks_forget_det + residual_forget_initial_det
            # )
            term1_mat = (V_T_forget_det) @ (
                (W_det + DP) @ layer_ks_forget_det)
            term2_mat = (W_det + DP) @ layer_ks_forget_det - v_forget_plus_resid_det
            # term1_mat = H_T_forget_det @ (
            #     (W_det + DP) @ layer_ks_forget_det + residual_forget_initial_det
            # )
            term1 = torch.norm(term1_mat) ** 2
            term2 = torch.norm(term2_mat) ** 2
            term3 = torch.norm(DP) ** 2
            loss = term1  + term2*100  + term3 * 0.1

            loss.backward()
            optimizer.step()

            if it % 10 == 0:
                tqdm.write(
                    f"Step {it}: term1={term1.item():.6e}   term2={term3.item():.6e}   term3={term3.item():.6e}"
                )

            if loss.item() < 1e-8:
                break


        with torch.no_grad():
            weights[weight_name][...] = ((D_hat @ P)+ weights[weight_name].double()).float()
        for x in [KKT_wiki, W]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    # for i, layer in enumerate(hparams.layers):
    #     layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
    #     cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T

    return model, weights_copy

def get_unlearn_project(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    unlearn_requests: List[Dict],
    hparams: ZeroUnlearnGDHyperParams,
    layer,
    cache_template: Optional[str] = None,
    use_h: bool = False,
) -> Dict[str, Tuple[torch.Tensor]]:
    requests = deepcopy(unlearn_requests)
    for i, request in enumerate(requests):
        if request["target_true"]["str"][0] != " ": space required for correct tokenization when computing V_u
            # Space required for correct tokenization
            requests[i]["target_true"]["str"] = " " + request["target_true"]["str"]
    for request in requests[:10]:
        print(
            f"Unlearn request sample: "
            f"[{request['prompt'].format(request['subject'])}]"
        )

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    v_layer = layer
    v_list = []

    for request in tqdm(requests, desc="Computing z for unlearn requests"):
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    v_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                v_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded: # compute V_u via gradient descent optimization
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                v_layer,
                context_templates,
                hparams.rewrite_module_tmp if not use_h else hparams.layer_module_tmp,
            )

            v_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    print(f'{v_list[0].shape=}')
    layer_vs = torch.stack(v_list, dim=1) # [hidden_size, num_requests] -> [num_requests, hidden_size]
    print(f'layer_vs shape: {layer_vs.shape}')
    #layer = hparams.layers[-1]
    #layer_vs = compute_vs(model, tok, requests, hparams, layer, context_templates) #[num_requests,hidden_size]
    k_forget = compute_retain_ks(model, tok, requests, hparams, layer, context_templates).T
    m_forget = compute_retain_vs(model, tok, requests, hparams, layer, context_templates).T
    
    #m_forget = (model, tok, requests, hparams, layer, context_templates) #[num_requests,hidden_size]
    _, S, V_T = torch.linalg.svd(layer_vs.T, full_matrices=False)
    print(f'S shape: {S.shape}, V_T shape: {V_T.shape}')
    v = V_T.T
    # v4 = v[:, :1]
    # P = torch.eye(v.shape[0]).to("cuda") - v4 @ v4.T

    P=torch.eye(v.shape[0]).to("cuda") - v @ v.T
    check=layer_vs.T @ P

    print(f"layer_vs.T @ P = {torch.sum(check.abs())=}, {torch.mean(check.abs())=}")
    #hidden_forget = layer_vs.T
    print(f"P shape: {P.shape}, rank: {torch.linalg.matrix_rank(P)}")
    print(f'v shape: {v.shape}, rank: {torch.linalg.matrix_rank(v)}')
    return P, V_T,  k_forget, m_forget

def get_retain_project(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    retain_requests: List[Dict],
    hparams: ZeroUnlearnGDHyperParams,
    layer,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    requests = deepcopy(retain_requests)
    print(f"retain requests: {len(requests)=}")
    for i, request in enumerate(requests):
        if request["target_true"]["str"][0] != " ": space required for correct tokenization when computing V_u
            # Space required for correct tokenization
            requests[i]["target_true"]["str"] = " " + request["target_true"]["str"]
    for request in requests[:10]:
        print(
            f"retain request sample: "
            f"[{request['prompt'].format(request['subject'])}]"
        )

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = layer
    z_list = []

    for request in tqdm(requests, desc="Computing v for retain requests"):
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded: # compute V_u via gradient descent optimization
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
                hparams.rewrite_module_tmp,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    layer_vs = torch.stack(z_list, dim=1) # [hidden_size, num_requests]     
    print(f'retain layer_vs shape: {layer_vs.shape}')
    #context_templates = [['{}'] for _ in range(len(context_templates))]
    #layer = hparams.layers[-1]

    #layer_vs = compute_retain_vs(model, tok, requests, hparams, layer, context_templates).T 
    layer_ks = compute_retain_ks(model, tok, requests, hparams, layer, context_templates).T 
    KKT = layer_ks @ layer_ks.T
    VKT = layer_vs @ layer_ks.T
    hidden_retrain = compute_retain_hidden_vs(model, tok, requests, hparams, layer, context_templates).T
    print(f"retain set: {layer} KKT shape: {KKT.shape}, VKT shape: {VKT.shape}")
    return KKT, VKT, hidden_retrain, layer_ks, layer_vs
    

def get_unlearn_project_backup(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    unlearn_requests: List[Dict],
    hparams: ZeroUnlearnGDHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    # Update target and print info
    requests = deepcopy(unlearn_requests)
    for i, request in enumerate(requests):
        if request["target_true"]["str"][0] != " ": space required for correct tokenization when computing V_u
            # Space required for correct tokenization
            requests[i]["target_true"]["str"] = " " + request["target_true"]["str"]
    for request in requests[:10]:
        print(
            f"Unlearn request sample: "
            f"[{request['prompt'].format(request['subject'])}]"
        )

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in tqdm(requests, desc="Computing z for unlearn requests"):
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded: # compute V_u via gradient descent optimization
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
                hparams.rewrite_module_tmp,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1) # [hidden_size, num_requests]     
    #V_u_T = torch.stack(z_list, dim=1).T # [hidden_size, num_requests] -> [num_requests, hidden_size]
    layer=z_layer
    #layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
    layer_ms = compute_vs(model, tok, requests, hparams, layer, context_templates).T
    cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
    targets = zs - cur_zs # [hidden_size, num_requests]
    V_u_T = (layer_ms + targets).T

    print(f"V_u_T shape: {V_u_T.shape}, rank: {torch.linalg.matrix_rank(V_u_T)}")
    # Note: typically N > D, if N < D and full_matrices=False, you cannot get the complete null space

    # V_T (Vh) shape: [min(N, D), D] 
    # PyTorch returns the transpose of V, so each row is a right singular vector
    _, S, V_T = torch.linalg.svd(V_u_T, full_matrices=False)

    #small_singular_indices = (S > threshold).nonzero(as_tuple=True)[0]
    #v = V_T[small_singular_indices].T
    v = V_T.T
    P=torch.eye(v.shape[0]).to("cuda") - v @ v.T

 
    print(f"P shape: {P.shape}, rank: {torch.linalg.matrix_rank(P)}")
    return P
    
def get_p(layer_vs):
    # layer_vs: [hidden_size, num_requests] -> [num_requests, hidden_size]
    print(f'layer_vs shape: {layer_vs.shape}')
    _, S, V_T = torch.linalg.svd(layer_vs.T, full_matrices=False)
    print(f'S shape: {S.shape}, V_T shape: {V_T.shape}')
    v = V_T.T
    # v4 = v[:, :1]
    # P = torch.eye(v.shape[0]).to("cuda") - v4 @ v4.T

    P=torch.eye(v.shape[0]).to("cuda") - v @ v.T
    check=layer_vs.T @ P

    print(f"layer_vs.T @ P = {torch.sum(check.abs())=}, {torch.mean(check.abs())=}")
    #hidden_forget = layer_vs.T
    print(f"P shape: {P.shape}, rank: {torch.linalg.matrix_rank(P)}")
    print(f'v shape: {v.shape}, rank: {torch.linalg.matrix_rank(v)}')
    return P

def compute_vkt_cross_covariance(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer_name: str,
    retain_data: List[Dict],
    sample_size: Optional[int] = None,
    precision: str = "float64",
    add_retain: bool = False,
    stats_dir: Optional[Path] = None,
    ds_name: str = "wikipedia",
    force_recompute: bool = False,
    model_name: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute V @ K^T cross-covariance matrix by simultaneously collecting K and V statistics.
    Caches result for future use.
    """
    from datasets import load_dataset, load_from_disk
    
    def get_ds():
        data_dir = Path('data/wikidata')
        if data_dir.exists():
            print(f"Loading dataset from local directory: {data_dir}")
            raw_ds = load_from_disk(str(data_dir))
        else:
            print(f"Downloading dataset...")
            import os
    
            raw_ds = load_dataset("wikipedia", "20220301.en")
            print(f"Saving dataset to local directory: {data_dir}")
            raw_ds.save_to_disk(str(data_dir))
        
        if hasattr(model.config, 'n_positions'):
            maxlen = model.config.n_positions
        elif hasattr(model.config, 'max_sequence_length'):
            maxlen = model.config.max_sequence_length
        elif hasattr(model.config, 'max_position_embeddings'):
            maxlen = model.config.max_position_embeddings
        elif hasattr(model.config, 'seq_length'):
            maxlen = model.config.seq_length
        else:
            raise NotImplementedError
        
        if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
            maxlen = model.config.sliding_window if hasattr(model.config, 'sliding_window') and model.config.sliding_window else 4096
        if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
            maxlen = 4096
        
        maxlen = min(4096, maxlen)
        return TokenizedDataset(
            text_dataset=raw_ds["train"].to_list(),
            retain_data=retain_data if add_retain else None,
            tokenizer=tokenizer,
            maxlen=maxlen
        )
    
    # Build cache file path (similar to layer_stats_retain)
    if stats_dir is None:
        stats_dir = STATS_DIR
    if model_name is None:
        model_name = model.config._name_or_path.rsplit("/")[-1]
    
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    batch_tokens = 4096 * 3  # Same as in the function
    npos = 4096  # Default value
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    
    stats_dir = Path(stats_dir)
    # Cache file name: vkt_{layer_name}_{precision}{size_suffix}.npz
    file_extension = f"{model_name}/{ds_name}_stats/vkt_{layer_name}_{precision}{size_suffix}.npz"
    filename = stats_dir / file_extension
    
    print(f"Computing V @ K^T cross-covariance for {model_name} @ {layer_name}...")
    
    batch_size = 1
    dtype = getattr(torch, precision)
    
    # Only load dataset if cache doesn't exist or force_recompute is True
    # This matches the pattern in layer_stats_retain
    ds = get_ds() if not filename.exists() or force_recompute else None
    
    # Use VKT class to compute V @ K^T
    stat = VKT()
    
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(4096 * 3),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    
    # Calculate batch count for progress bar
    # If ds is None, tally will return empty iterator (cache loaded), so batch_count should be 0
    batch_count = -(-(sample_size or len(ds)) // batch_size) if ds is not None else 0
    
    with torch.no_grad():
        for batch_group in tqdm(loader, total=batch_count, disable=(ds is None)):
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                with Trace(
                    model, layer_name, retain_input=True, retain_output=True, stop=True
                ) as tr:
                    model(**batch)
                # Get K (input) and V (output) simultaneously
                # flatten_masked_batch returns (n, d) format: n samples, d features
                k_feats = flatten_masked_batch(tr.input, batch["attention_mask"]).to(dtype=dtype)  # (n, d_k)
                v_feats = flatten_masked_batch(tr.output, batch["attention_mask"]).to(dtype=dtype)  # (n, d_v)
                
                # Pass V and K to VKT.add()
                # VKT.add() expects V (n, d_v) and K (n, d_k), computes V @ K^T
                stat.add(v_feats, k_feats)  # (n, d_v), (n, d_k)
    
    # Return V @ K^T matrix (centered, unbiased)
    return stat.moment(unbiased=True).to(dtype)


def get_wiki_project(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    retain_data: List[Dict],
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
    add_retain: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        # Get K statistics for KKT (K @ K^T)
        stat_K = layer_stats_retain(
            model=model,
            tokenizer=tok,
            layer_name=layer_name,
            stats_dir=STATS_DIR,
            ds_name=mom2_dataset,
            to_collect=["mom2"],
            k_or_v=True,
            retain_data=retain_data,
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
            add_retain=add_retain,
        )
        
        # Compute V @ K^T cross-covariance
        print('Computing V @ K^T cross-covariance...')
        VKT = compute_vkt_cross_covariance(
            model=model,
            tokenizer=tok,
            layer_name=layer_name,
            retain_data=retain_data,
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            add_retain=add_retain,
            stats_dir=STATS_DIR,
            ds_name=mom2_dataset,
            force_recompute=force_recompute,
            
        )
        
        print('compute K V matrix done')
        COV_CACHE[key] = {
            "KKT": stat_K.mom2.moment().float().to("cpu"),
            "VKT": VKT.float().to("cpu"),
        }
        print(f'KKT shape: {COV_CACHE[key]["KKT"].shape}')
        print(f'VKT shape: {COV_CACHE[key]["VKT"].shape}')


    return COV_CACHE[key]


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
