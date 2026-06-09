import os
import sys
# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from pathlib import Path
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
from tqdm import tqdm
import shutil
import random
from itertools import islice
from time import time
from typing import Tuple, Union
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.summarize_list import summarize
from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from baselines.GA import GAHyperParams, apply_ga_to_model
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    MQUAKEDataset,
    get_tfidf_vectorizer,
    KnownsDataset,
)
from collections import defaultdict
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from experiments.py.eval_utils_mquake import compute_rewrite_quality_mquake
from memit import MEMITHyperParams
from memit.compute_z import get_module_input_output_at_words, compute_z
from memit.memit_main import apply_memit_to_model, get_context_templates
from memit.memit_seq_main import apply_memit_seq_to_model
from memit.memit_rect_main import apply_memit_rect_to_model
from AlphaEdit import AlphaEditHyperParams
from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov
from ZeroUnlearn import ZeroUnlearnHyperParams, apply_unl_to_model 
from ZeroUnlearn_GD import ZeroUnlearnGDHyperParams, apply_unl_gd_to_model 
from rome import ROMEHyperParams, apply_rome_to_model
from baselines.base_model import BASEHyperParams, apply_base_to_model
from util import nethook
from util.globals import *
from nse import NSEHyperParams
from nse.nse_main import apply_nse_to_model
from glue_eval.glue_eval import GLUEEval
ALG_DICT = {
    "BASE": (BASEHyperParams, apply_base_to_model),
    "AlphaEdit": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
    "MEMIT_seq": (MEMITHyperParams, apply_memit_seq_to_model),
    "MEMIT_prune": (MEMITHyperParams, apply_memit_to_model),
    "MEMIT_rect": (MEMITHyperParams, apply_memit_rect_to_model),
    "NSE": (NSEHyperParams, apply_nse_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "GA": (GAHyperParams, apply_ga_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
    "ZeroUnlearn": (ZeroUnlearnHyperParams, apply_unl_to_model),
    "ZeroUnlearn_GD": (ZeroUnlearnGDHyperParams, apply_unl_gd_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "mquake": (MQUAKEDataset, compute_rewrite_quality_mquake),
}
from concurrent.futures import ThreadPoolExecutor
def set_all_seeds(seed: int):
    """Set RNG seeds for python random, numpy and torch (CPU and CUDA)."""
    import random as _random
    _random.seed(seed)
    try:
        import numpy as _np

        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch

        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    debug: bool = False,
    ratio_or_num: bool = False,
    ratio: float = 0.1,
    unlearn_num: int = 100,
    retain_num: int = 100,
    model_path_dir:str=None,
    eval_retain: bool = False,
    eval_base_glue: bool = False,
    add_retain: bool = False,
    edit_layer_nums:int = 0,
    use_h: bool = False,
    seed: int = None,
):
    print(f"Current evaluation config:")
    print(f"-------------------------------------------------------")
    print(f"alg_name: {alg_name}")
    print(f"model_name: {model_name}")
    print(f"hparams_fname: {hparams_fname}")
    print(f"ds_name: {ds_name}")
    print(f"dataset_size_limit: {dataset_size_limit}")
    print(f"continue_from_run: {continue_from_run}")
    print(f"skip_generation_tests: {skip_generation_tests}")
    print(f"generation_test_interval: {generation_test_interval}")
    print(f"conserve_memory: {conserve_memory}")
    print(f"dir_name: {dir_name}")
    print(f"num_edits: {num_edits}")
    print(f"use_cache: {use_cache}")
    print(f"debug: {debug}")
    print(f"ratio_or_num: {ratio_or_num}")
    print(f"ratio: {ratio}")
    print(f"num: {unlearn_num=}, {retain_num=}")
    print(f"model_path_dir: {model_path_dir}")
    print(f"eval_retain: {eval_retain}")
    print(f"eval_base_glue: {eval_base_glue}")
    print(f"add_retain: {add_retain}")
    print(f"edit_layer_nums: {edit_layer_nums}")
    print(f"use_h: {use_h}")
    print(f"seed: {seed}")
    print(f"-------------------------------------------------------")
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"{model_name}_{ds_name}_seed{seed}_unlearn_{unlearn_num}_retain_{retain_num}_edit_layer_nums_{edit_layer_nums}_run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")
    if "MEMIT" in alg_name:
    # Get run hyperparameters
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / "MEMIT" / hparams_fname
        )
    else:
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / alg_name / hparams_fname
        )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")
    set_all_seeds(seed)

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model_name=os.path.join(model_path_dir, model_name)
        #model_name=os.path.join(model_dir, model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)
    
    #ds = ds[:num]
    if debug:
        ds = ds[:100]
    # Get cache templates
    print(f'ds_name: {ds_name}, count: {len(ds)}')
    cache_template = None
    if use_cache:
        if any(alg in alg_name for alg in ["MEMIT","AlphaEdit", "MEMIT_seq", "MEMIT_prune", "MEMIT_rect"]):
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_MEMIT"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        else:
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        print(f"Will load cache from {cache_template}")
    if alg_name == "NSE":
        cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        for record in ds:
            # Retrieve k/v pair if already stored in cache
            cache_fname = (
                Path(
                    str(cache_template).format(
                        hparams.layers[-1], hparams.clamp_norm_factor, record["case_id"]
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
                continue
            # Compute k/v pair if not loaded from cache
            if not data_loaded:
                context_templates = get_context_templates(model, tok)
                cur_z = compute_z(
                    model,
                    tok,
                    {"case_id": record["case_id"], **record["requested_rewrite"]},
                    hparams,
                    hparams.layers[-1],
                    context_templates,
                )
                if cache_fname is not None:
                    cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(
                        cache_fname,
                        **{
                            "v_star": cur_z.detach().cpu().numpy(),
                        },
                    )
                    print(f"Cached k/v pair at {cache_fname}")
    if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "MEMIT_prune", "NSE"]):
        # Iterate through dataset
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        edit_num = min(len(hparams.layers), edit_layer_nums)
        if hparams.model_name == "gpt2-xl":
            
            cache_c = torch.zeros((edit_num, W_out.shape[0], W_out.shape[0]), device="cpu")
            if alg_name == "AlphaEdit":
                P = torch.zeros((edit_num, W_out.shape[0], W_out.shape[0]), device="cpu")
        elif hparams.model_name in ["Llama3-8B","phi-1.5","Qwen3-4B","Llama-3.2-3B-Instruct","Llama-3.1-8B-Instruct"]:
            cache_c = torch.zeros((edit_num, W_out.shape[1], W_out.shape[1]), device="cpu")
            if alg_name == "AlphaEdit":
                P = torch.zeros((edit_num, W_out.shape[1], W_out.shape[1]), device="cpu")
        del W_out
    if alg_name == "AlphaEdit":
        edit_num = min(len(hparams.layers), edit_layer_nums)
        layers = hparams.layers[-edit_num:]
        for i, layer in enumerate(layers):
            P[i,:,:] = get_project(model, tok, layer, hparams)
        torch.save(P, "null_space_project.pt")
    glue_save_location = str(run_dir) + '/' + 'glue_eval/'
    os.makedirs(glue_save_location, exist_ok=True)
    cnt = 0


    retain_ds = ds[:len(ds)//2]
    forget_ds = ds[len(ds)//2:]
    print(f"Initial retain_ds_cnt: {len(retain_ds)}, forget_ds_cnt: {len(forget_ds)}")
    if ratio_or_num:
        print(f'forget num is set to num:{unlearn_num}')
        unlearn_num=min(len(forget_ds), unlearn_num)
        retain_num = min(len(retain_ds), retain_num)
    else:
        print(f'forget num is set to ratio:{ratio}')
        unlearn_num = min(len(forget_ds), int(len(forget_ds) * ratio))
        retain_num = min(len(retain_ds), retain_num)
    unlearn_ds_path = f'{DATA_DIR}/{ds_name}_unlearn_ds_num_{unlearn_num}_seed{seed}.json'
    retain_ds_path = f'{DATA_DIR}/{ds_name}_retain_ds_num_{retain_num}_seed{seed}.json'
    if os.path.exists(unlearn_ds_path):
        with open(unlearn_ds_path, 'r') as f:
            unlearn_ds = json.load(f)
        print(f"Load unlearn_ds and retain_ds from {unlearn_ds_path} and {retain_ds_path}")
    else:
        print(f"Generate unlearn_ds for {ds_name},saved at {unlearn_ds_path} use seed {seed}.")
        unlearn_ds = random.sample(forget_ds,k=unlearn_num)
        with open(unlearn_ds_path, 'w') as f:
            json.dump(unlearn_ds, f, indent=4)
    if os.path.exists(retain_ds_path):    
        with open(retain_ds_path, 'r') as f:
            retain_ds = json.load(f)
        print(f"Load retain_ds from {retain_ds_path}")
    else:
        print(f"Generate retain_ds for {ds_name},saved at {retain_ds_path} use seed {seed}.")
        retain_ds = random.sample(retain_ds,k=retain_num)
        with open(retain_ds_path, 'w') as f:
            json.dump(retain_ds, f, indent=4)
    
    retain_data = []
    for record in retain_ds:
        rr = record["requested_rewrite"]
        if isinstance(rr, list):
            for rewrite_dict in rr:
                if isinstance(rewrite_dict, dict):
                    retain_data.append({"case_id": record["case_id"], **rewrite_dict})
                else:
                    raise TypeError(f"Unsupported type in requested_rewrite list for case_id {record['case_id']}: {type(rewrite_dict)}")
        elif isinstance(rr, dict):
            retain_data.append({"case_id": record["case_id"], **rr})
        else:
            raise TypeError(f"Unsupported type for requested_rewrite for case_id {record['case_id']}: {type(rr)}")

    unlearn_data = []
    for record in unlearn_ds:
        rr = record["requested_rewrite"]
        if isinstance(rr, list):
            for rewrite_dict in rr:
                if isinstance(rewrite_dict, dict):
                    unlearn_data.append({"case_id": record["case_id"], **rewrite_dict})
                else:
                    raise TypeError(f"Unsupported type in requested_rewrite list for case_id {record['case_id']}: {type(rewrite_dict)}")
        elif isinstance(rr, dict):
            unlearn_data.append({"case_id": record["case_id"], **rr})
        else:
            raise TypeError(f"Unsupported type for requested_rewrite for case_id {record['case_id']}: {type(rr)}")
    print(f"retain_data_cnt: {len(retain_data)}, unlearn_data_cnt: {len(unlearn_data)}")
    num_edits = len(unlearn_ds)
    unlearn_chunks_list = list(chunks(unlearn_ds, num_edits))
    total_chunks = len(unlearn_chunks_list)
    print(f"total_chunks: {total_chunks}")
    for chunk_idx, record_chunks in enumerate(unlearn_chunks_list):
        case_result_template = str(run_dir / "{}_edits-case_{}.json")
        print(f"=================================================================={cnt+1}_edit==================================================================")
        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue
        # unlearn_data = [{"case_id": record["case_id"], **record["requested_rewrite"]} for record in record_chunks]
        unlearn_data = []
        for record in record_chunks:
            rr = record["requested_rewrite"]
            if isinstance(rr, list):
                for rewrite_dict in rr:
                    if isinstance(rewrite_dict, dict):
                        unlearn_data.append({"case_id": record["case_id"], **rewrite_dict})
                    else:
                        raise TypeError(f"Unsupported type in requested_rewrite list for case_id {record['case_id']}: {type(rewrite_dict)}")
            elif isinstance(rr, dict):
                unlearn_data.append({"case_id": record["case_id"], **rr})
            else:
                raise TypeError(f"Unsupported type for requested_rewrite for case_id {record['case_id']}: {type(rr)}")
        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT","AlphaEdit", "MEMIT_seq", "MEMIT_prune", "NSE"]) else dict()
        seq_args = dict(cache_c=cache_c) if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]) else dict()
        nc_args = dict(P = P) if any(alg in alg_name for alg in ["AlphaEdit"]) else dict()
        if cnt == 0 and eval_base_glue: # do initial GLUE EVAL WITH ORIGINAL MODEL (only if eval_base_glue True)
            glue_results = {'edit_num': -1}

            out_file = glue_save_location + "base.json"
            
            glue_eval = GLUEEval(model, tok, number_of_tests = 100)
            glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)

            #store the individual overall result file
            output_filename = out_file.replace('.json', '_glue.json')
            with open(output_filename, "w") as f:
                json.dump(glue_results, f, indent=4)
        start = time()
        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]):
            edited_model, cache_c = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **rewrite_dict}
                    for record in record_chunks
                    for rewrite_dict in (
                        record["requested_rewrite"]
                        if isinstance(record["requested_rewrite"], list)
                        else [record["requested_rewrite"]]
                    )
                ],
                hparams,
                edit_layer_nums=edit_layer_nums,
                **args_conserve_memory,
                **etc_args,
                **seq_args,
                **nc_args,
            )
        elif alg_name == "MEMIT_prune":
            if cnt == 0:
                edited_model, weights_copy = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **rewrite_dict}
                        for record in record_chunks
                        for rewrite_dict in (
                            record["requested_rewrite"]
                            if isinstance(record["requested_rewrite"], list)
                            else [record["requested_rewrite"]]
                        )
                    ],
                    hparams,
                    return_orig_weights=True,
                    **args_conserve_memory,
                    **etc_args,
                )
                # Initialize the upd_matrix dictionary
                upd_matrix = {}
            else:
                edited_model, _ = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **rewrite_dict}
                        for record in record_chunks
                        for rewrite_dict in (
                            record["requested_rewrite"]
                            if isinstance(record["requested_rewrite"], list)
                            else [record["requested_rewrite"]]
                        )
                    ],
                    hparams,
                    return_orig_weights=False,
                    **args_conserve_memory,
                    **etc_args,
                )
            if cnt == (dataset_size_limit/num_edits) - 1:
            # Calculate the weight update matrix
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        current_weight = nethook.get_parameter(model, k)
                        upd_matrix[k] = current_weight - v.to("cuda")
                        # Calculate max singular value of the original weight
                        _, S_orig, _ = torch.svd(v)
                        max_sigma = S_orig.max().item()

                        # Adjust the upd_matrix singular values
                        U_upd, S_upd, V_upd = torch.svd(upd_matrix[k])
                        adjusted_S = torch.where(
                            S_upd > max_sigma,
                            torch.log(S_upd) - torch.log(torch.tensor(max_sigma, device='cuda')) + max_sigma,
                            S_upd
                        )
                        upd_matrix[k] = torch.matmul(U_upd, torch.matmul(torch.diag(adjusted_S), V_upd.t()))

                # Apply the adjusted updates to the model
                with torch.no_grad():
                    for k in upd_matrix:
                        original_weight = nethook.get_parameter(model, k)
                        adjusted_weight = original_weight + upd_matrix[k]
                        original_weight.copy_(adjusted_weight)
        elif alg_name == "UnL" or alg_name == "UnL_v5" or alg_name == "UnL_v6" or alg_name == "UnL_v4" or alg_name == "UnL_v6_5":
            # Don't save here, will save after evaluation for retain and forget separately
            edited_model, weights_copy = apply_algo(
                model=model,
                tok=tok,
                retain_requests=retain_data,
                unlearn_requests=unlearn_data,
                hparams=hparams,
                save_path=None,
                add_retain=add_retain,  # Will save after evaluation instead
                edit_layer_nums=edit_layer_nums,
                use_h=use_h,
            )
        
        else:
            edited_model, _ = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **rewrite_dict}
                    for record in record_chunks
                    for rewrite_dict in (
                        record["requested_rewrite"]
                        if isinstance(record["requested_rewrite"], list)
                        else [record["requested_rewrite"]]
                    )
                ],
                hparams,
                return_orig_weights=False,
                **args_conserve_memory,
                **etc_args,
            )
        exec_time = time() - start
        cnt+=1
        print("Execution took", exec_time)
        # Evaluate new model
        if unlearn_num > 0 and unlearn_num < 100:
            # Save activations of edited model at each layer (averaged over different prefixes)
            edit_num = min(len(hparams.layers), edit_layer_nums)
            layers_to_save = hparams.layers[-edit_num:]
            context_templates = get_context_templates(edited_model, tok)
            
            # Compute context_type related indices
            context_type_lens = [0] + [len(context_type) for context_type in context_templates]
            context_len = sum(context_type_lens)  # total context_type length for each request
            context_type_csum = np.cumsum(context_type_lens).tolist()
            
            # Compute activations for unlearn_data
            unlearn_activations = {}
            for layer in layers_to_save:
                layer_output = get_module_input_output_at_words(
                    edited_model,
                    tok,
                    layer,
                    context_templates=[
                        context.format(req["prompt"])
                        for req in unlearn_data
                        for context_type in context_templates
                        for context in context_type
                    ],
                    words=[
                        req["subject"]
                        for req in unlearn_data
                        for context_type in context_templates
                        for _ in context_type
                    ],
                    module_template=hparams.rewrite_module_tmp,
                    fact_token_strategy=hparams.fact_token,
                )[1]  # get output activations
                
                # Average over different prefixes, each request gets one vector
                ans = []
                for i in range(0, layer_output.size(0), context_len):
                    tmp = []
                    for j in range(len(context_type_csum) - 1):
                        start, end = context_type_csum[j], context_type_csum[j + 1]
                        tmp.append(layer_output[i + start : i + end].mean(0))
                    ans.append(torch.stack(tmp, 0).mean(0))
                unlearn_activations[layer] = torch.stack(ans, dim=0).detach().cpu()
                torch.save(unlearn_activations[layer], run_dir / f"unlearn_activations_layer_{layer}.pt")
                print(f"Saved layer {layer} unlearn activations to {run_dir / f'unlearn_activations_layer_{layer}.pt'}")
            
        if args.downstream_eval_steps > 0 and cnt % args.downstream_eval_steps == 0:
            glue_results = {
                        'edit_num': cnt*num_edits,
                        #'case_id': case_ids
                        }

            out_file = glue_save_location + "case_{}.json".format(record["case_id"])#stores the last case ID of the batch

            glue_eval = GLUEEval(edited_model, tok, number_of_tests = 100)
            glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)
                    
            #store the individual overall result file
            output_filename = out_file.replace('.json', '_glue.json')
            with open(output_filename, "w") as f:
                json.dump(glue_results, f, indent=4)
    

    gen_test_vars = [snips, vec]
    if eval_retain:
        print(f"Evaluating retain and forget")
        eval_data=[("forget", unlearn_ds), ("retain",retain_ds)]
    else:
        print(f"Evaluating forget only")
        eval_data=[("forget", unlearn_ds)]
    for split_name, split_ds in eval_data:
        start = time()
        all_metrics = []
        for i, record in tqdm(enumerate(split_ds), total=len(split_ds), desc=f"Evaluating {split_name}"):
            metrics = {
                "grouped_case_ids": case_ids,
                "num_edits": num_edits,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": ds_eval_method(
                    edited_model,
                    tok,
                    record,
                    *(
                        gen_test_vars
                        if record["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),  
                ),
            }
            all_metrics.append(metrics)
        with open(f'{run_dir}/{split_name}_metrics.jsonl', "w", encoding='utf-8') as f:
            for metrics in all_metrics:
                f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
        sum_res = summarize(split_name, all_metrics)
        with open(f'{run_dir}/{split_name}_summarize_results.json', "w", encoding='utf-8') as f:
            json.dump(sum_res, f, ensure_ascii=False, indent=4)
        print("Evaluation task {} took {} seconds".format(split_name, time() - start))
            
           


def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T
def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["BASE","GA","ZeroUnlearn","ZeroUnlearn_GD","AlphaEdit","MEMIT_rect", "MEMIT_seq","MEMIT_prune", "MEMIT", "ROME", "FT", "MEND","NSE"],
        default="ZeroUnlearn",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="Llama-3.2-3B-Instruct.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="",
        help="Optional ablation suffix to append to the alg_name for result directory naming. If non-empty, results are saved in results/<alg_name><ablation>/.",
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre", "mquake"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--downstream_eval_steps",
        type=int,
        default=0,
        help="If we want to do sequential editing or not",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If we want to debug the code or not",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.1,
        help="If we want to debug the code or not",
    )
    parser.add_argument(
        "--ratio_or_num",
        action="store_true",
        help="If we want to debug the code or not",
    )
    parser.add_argument(
        "--unlearn_num",
        type=int,
        default=3000,
        help="If we want to debug the code or not",
    )
    parser.add_argument(
        "--model_path_dir",
        type=str,
        default='model_path',
        help="If we want to debug the code or not",
    )
    parser.add_argument(
        "--eval_base_glue",
        action="store_true",
        help="If set, run GLUE evaluation on the base (original) model before edits",
    )
    parser.add_argument(
        "--eval_retain",
        action="store_true",
        help="If we want to debug the code or not",
    )
    parser.add_argument(
        "--add_retain",
        action="store_true",
        help="If we want to debug the code or not",
    )
    parser.add_argument(
        "--use_h",
        action="store_true",
        help="If we want to debug the code or not",
    )
    parser.add_argument(
        "--edit_layer_nums",
        type=int,
        default=5,
        help="If we want to debug the code or not",
    )
    parser.add_argument(
        "--retain_num",
        type=int,
        default=100,
        help="If we want to debug the code or not",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Integer seed to use for RNG.",
    )
    args = parser.parse_args()
    parsed_seed = args.seed

    # Compute directory name: append ablation suffix to alg_name if provided
    dir_name_val = args.alg_name + args.ablation if args.ablation else args.alg_name

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=dir_name_val,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        debug=args.debug,
        ratio_or_num=args.ratio_or_num,
        ratio=args.ratio,
        unlearn_num=args.unlearn_num,
        retain_num=args.retain_num,
        model_path_dir=args.model_path_dir,
        eval_retain=args.eval_retain,
        eval_base_glue=args.eval_base_glue,
        add_retain=args.add_retain,
        edit_layer_nums=args.edit_layer_nums,
        use_h=args.use_h,
        seed=parsed_seed,
    )
