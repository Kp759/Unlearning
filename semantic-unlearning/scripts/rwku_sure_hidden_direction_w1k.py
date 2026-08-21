#!/usr/bin/env python3
"""RWKU post-hoc development learner with frozen-base-head hidden-direction repair."""

from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import torch
import torch.nn.functional as F
import build_sure_wikipedia_stats as wikipedia
import gagd_compare as gagd
import rwku_artifact_access as artifact_access
import rwku_checkpoint_receipt as checkpoint_receipt
import rwku_experiment
import rwku_representation as representation
import rwku_sure_head_only_w1k as head
import rwku_sure_repr_rescue_w1k as v2
import sure_canonical_core as core
import sure_minimal_two_stage as learner

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_CONFIGURATION = PROJECT_ROOT / "config" / "rwku" / "sure_head_hidden_direction_w1k_seed0.json"
SOURCE_CONFIGURATION = PROJECT_ROOT / "config" / "rwku" / "sure_head_only_w1k_seed0.json"
PROTOCOL_STATUS = "rwku_target_only_auxwiki_sure_hidden_direction_w1k_posthoc_development"
SCHEMA = "rwku_sure_head_hidden_direction_w1k_configuration_v1"
EXPERIMENT_ID = "rwku-h-w1k-stephen-king-hidden-direction-seed0-v3"

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path",required=True); p.add_argument("--model-revision",default="local_pinned_snapshot")
    p.add_argument("--training-bundle",type=Path,required=True); p.add_argument("--generator-receipt",type=Path,required=True)
    p.add_argument("--utility-cache",type=Path,required=True); p.add_argument("--wikipedia-dir",type=Path,required=True)
    p.add_argument("--source-head-only-run",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--experiment-id",default=EXPERIMENT_ID); p.add_argument("--configuration",type=Path,default=DEFAULT_CONFIGURATION)
    return p.parse_args()

def read_json(path):
    with Path(path).open("r",encoding="utf-8") as h: v=json.load(h)
    if not isinstance(v,dict): raise ValueError(f"Expected JSON object: {path}")
    return v

def write_json(path,value): core.write_json(path,value)

def load_configuration(path):
    v=read_json(path)
    if v.get("schema_version")!=SCHEMA: raise ValueError("Unsupported hidden-direction configuration schema")
    required={"configuration_id":EXPERIMENT_ID,"development_only":True,"posthoc_development_target":True,
        "official_rwku_metrics_observed_before_method_design":True,"seed":0,"target_entity":"Stephen King",
        "target_entity_id":"rwku:1_Stephen_King","source_head_only_configuration_id":head.EXPECTED_CONFIGURATION_ID,
        "neutral_target":"Unknown"}
    for k,e in required.items():
        if v.get(k)!=e: raise ValueError(f"Hidden-direction configuration changed {k}")
    c=v.get("hidden_direction",{})
    if tuple(c.get("rank_ladder",()))!=(1,2,4): raise ValueError("rank ladder must remain 1,2,4")
    if tuple(c.get("target_modules",()))!=("down_proj",): raise ValueError("target module must remain down_proj")
    if int(c.get("last_n_layers",0))!=1: raise ValueError("repair must remain final-layer only")
    if c.get("sensitive_view_scope")!="all_generated_atomic_views": raise ValueError("sensitive view scope changed")
    if int(c.get("steps",0))<=0 or float(c.get("learning_rate",0))<=0: raise ValueError("invalid optimization settings")
    if float(c.get("frozen_base_head_training_margin",0))<=0: raise ValueError("training margin must be positive")
    if not c.get("candidate_scales") or any(not 0<float(x)<=1 for x in c["candidate_scales"]): raise ValueError("invalid scales")
    a=v.get("acceptance",{})
    req={"required_pairwise_margin":0.01,"required_direct_success":100.0,"required_other_atomic_view_success":100.0,
        "max_frozen_base_head_recovery":0.0,"min_frozen_base_head_demotion_margin":0.05,"max_head_delta_norm":1.5,
        "utility_kl_mean_budget":0.01,"utility_kl_p95_budget":0.05,"utility_kl_max_budget":0.5,
        "checkpoint_dtype":"bf16","device_map":"single"}
    for k,e in req.items():
        if a.get(k)!=e: raise ValueError(f"acceptance changed {k}")
    b=v.get("data_boundary",{})
    for k in ("official_rwku_records_available_to_learner","official_rwku_records_used_for_checkpoint_selection","neighbor_prompts_used_for_training_or_selection"):
        if b.get(k) is not False: raise ValueError(f"data boundary changed {k}")
    if b.get("external_wikipedia_only_utility") is not True: raise ValueError("utility must remain external Wikipedia")
    return v

def state_namespace(args):
    return SimpleNamespace(output_root=Path(args.output_root),experiment_id=str(args.experiment_id),
        training_source=rwku_experiment.TRAINING_SOURCE_TARGET_ONLY)

def verify_prepared_state(args,cfg):
    sargs=state_namespace(args); s=rwku_experiment._read_state(sargs)
    if s.get("state")!="PREPARED": raise ValueError(f"requires PREPARED state, got {s.get('state')}")
    if s.get("official_evaluation_opened") is not False: raise ValueError("this v3 experiment already opened official evaluation")
    t=s.get("target",{})
    if t.get("seed")!=0 or t.get("subject")!=cfg["target_entity"]: raise ValueError("wrong prepared target")
    run=Path(args.output_root).resolve()/str(args.experiment_id)
    if (run/"checkpoint_receipt.json").exists(): raise FileExistsError("checkpoint receipt already exists")
    return sargs,run

def completion_layout(tok,prefix,suffix,llama_like):
    prefix_ids=[int(x) for x in tok(prefix)["input_ids"]]
    suffix_ids=v2._completion_token_ids(tok,suffix,llama_like)
    combined=tok(f"{prefix} {suffix}",return_tensors="pt")["input_ids"][0].cpu()
    start=len(prefix_ids)-1; positions=list(range(start,start+len(suffix_ids)))
    if not positions or positions[-1]>=int(combined.numel()): raise RuntimeError("completion positions invalid")
    return combined,suffix_ids,positions

@torch.no_grad()
def hidden_rows(model,ids_cpu,positions,device):
    ids=ids_cpu.unsqueeze(0).to(device)
    h,_=wikipedia._final_hidden_only(model,{"input_ids":ids,"attention_mask":torch.ones_like(ids)})
    p=torch.tensor([int(x) for x in positions],device=h.device,dtype=torch.long)
    return h[0].index_select(0,p).float()

@torch.no_grad()
def build_direction_cases(model,tok,prompt_records,base_w,sensitive_ids,neutral_ids,training_margin,llama_like,device):
    sset={int(x) for x in sensitive_ids}-{int(x) for x in neutral_ids}
    if not sset: raise ValueError("no content-bearing sensitive token IDs")
    cases=[]; dirs=[]; recovered=0; filtered=0
    for pp,r in enumerate(prompt_records):
        rr=r["requested_rewrite"]; prompt=str(r["prompt_text"]); sensitive=str(rr["target_sensitive"]["str"])
        combined,suffix_ids,positions=completion_layout(tok,prompt,sensitive,llama_like)
        hs=hidden_rows(model,combined,positions,device)
        for ti,(y,pos) in enumerate(zip(suffix_ids,positions)):
            y=int(y)
            if y not in sset: filtered+=1; continue
            h0=hs[ti].detach().float()
            logits=F.linear(h0.to(device=base_w.device,dtype=base_w.dtype),base_w).float()
            ty=logits[y]; masked=logits.clone(); masked[y]=-torch.inf; c=int(masked.argmax().item()); tc=masked[c]
            wy=base_w[y].float(); wc=base_w[c].float(); direction=wy-wc
            pref=float((ty-tc).item()); recovered+=int(pref>=0)
            violation=max(0.0,pref+float(training_margin)); denom=float(direction.square().sum().item())+1e-12
            target_delta=(-violation/denom)*direction; dirs.append(target_delta.detach().cpu())
            cases.append({"case_index":len(cases),"prompt_position":int(pp),"prompt_kind":str(r.get("prompt_kind","")),
                "prompt_index":int(r.get("prompt_index",0)),"input_ids":combined.clone(),"prediction_position":int(pos),
                "target_token_id":y,"target_token_text":tok.decode([y]),"base_competitor_token_id":c,
                "base_competitor_token_text":tok.decode([c]),"base_target_minus_competitor":pref,
                "base_hidden":h0.detach().cpu(),"minimum_norm_target_delta":target_delta.detach().cpu(),
                "minimum_norm_target_delta_norm":float(target_delta.norm().item()),"training_margin":float(training_margin)})
    if not cases: raise ValueError("no hidden-direction cases constructed")
    m=torch.stack(dirs).float(); norms=m.norm(dim=1); nz=m[norms>1e-10]; spectrum=[]
    if nz.numel(): spectrum=[float(x) for x in torch.linalg.svdvals(nz)[:min(16,len(nz))].tolist()]
    report={"case_count":len(cases),"prompt_count":len({x["prompt_position"] for x in cases}),
        "unique_sensitive_token_count":len({x["target_token_id"] for x in cases}),
        "initial_frozen_base_head_recovery_count":recovered,
        "initial_frozen_base_head_recovery_percentage":100.0*recovered/len(cases),
        "filtered_token_decision_count":filtered,"minimum_norm_delta_mean":float(norms.mean().item()),
        "minimum_norm_delta_max":float(norms.max().item()),"minimum_norm_delta_nonzero_count":int((norms>1e-10).sum().item()),
        "target_delta_singular_values_top16":spectrum,"official_rwku_records_accessed":False}
    return cases,report

def current_hidden(model,case,device):
    ids=case["input_ids"].unsqueeze(0).to(device)
    h,_=wikipedia._final_hidden_only(model,{"input_ids":ids,"attention_mask":torch.ones_like(ids)})
    return h[0,int(case["prediction_position"])].float()

def frozen_head_loss(cur,case,base_w,margin):
    logits=F.linear(cur.to(dtype=base_w.dtype),base_w).float(); y=int(case["target_token_id"])
    target=logits[y]; masked=logits.clone(); masked[y]=-torch.inf; competitor=masked.max()
    demotion=competitor-target
    return F.relu(float(margin)-demotion).square(),demotion

def direction_target_loss(cur,case,device):
    base=case["base_hidden"].to(device=device,dtype=torch.float32)
    target=case["minimum_norm_target_delta"].to(device=device,dtype=torch.float32)
    delta=cur.float()-base; denom=target.square().mean().clamp_min(1e-6)
    mse=F.mse_loss(delta,target)/denom
    if float(target.norm().item())>1e-10 and float(delta.norm().item())>1e-10:
        cos=F.cosine_similarity(delta[None],target[None]).mean()
    else: cos=delta.new_tensor(0.0)
    rel=(delta-target).norm()/target.norm().clamp_min(1e-6)
    return mse,cos,rel

@torch.no_grad()
def frozen_proxy_report(model,cases,base_w,device):
    rec=0; margins=[]; probs=[]; prompt_rec={}; details=[]
    for case in cases:
        cur=current_hidden(model,case,device)
        logits=F.linear(cur.to(dtype=base_w.dtype),base_w).float(); y=int(case["target_token_id"])
        target=logits[y]; masked=logits.clone(); masked[y]=-torch.inf; c=int(masked.argmax().item()); comp=masked[c]
        margin=float((comp-target).item()); prob=float(F.softmax(logits,dim=-1)[y].item()); recovered=bool(target>=comp)
        rec+=int(recovered); margins.append(margin); probs.append(prob); p=int(case["prompt_position"])
        prompt_rec[p]=bool(prompt_rec.get(p,False) or recovered)
        details.append({"case_index":int(case["case_index"]),"prompt_position":p,"prompt_kind":case["prompt_kind"],
            "target_token_id":y,"target_token_text":case["target_token_text"],"competitor_token_id":c,
            "recovered":recovered,"demotion_margin":margin,"target_probability":prob})
    return {"decision_count":len(cases),"recovery_count":rec,"recovery_percentage":100.0*rec/len(cases),
        "prompt_count":len(prompt_rec),"prompt_with_any_recovery_count":int(sum(prompt_rec.values())),
        "prompt_with_any_recovery_percentage":100.0*sum(prompt_rec.values())/max(len(prompt_rec),1),
        "minimum_demotion_margin":float(min(margins)),"mean_demotion_margin":float(sum(margins)/len(margins)),
        "mean_target_probability":float(sum(probs)/len(probs)),"maximum_target_probability":float(max(probs)),
        "details":details,"readout_source":"untouched_base_input_embedding_matrix_after_output_head_untie",
        "official_rwku_records_accessed":False}

def proxy_safe(report,cfg):
    a=cfg["acceptance"]
    return bool(float(report["recovery_percentage"])<=float(a["max_frozen_base_head_recovery"])
        and float(report["minimum_demotion_margin"])>=float(a["min_frozen_base_head_demotion_margin"]))

def train_rank(model,tok,prompt_records,cases,utility_contexts,base_w,rank,cfg,llama_like,device,log_path):
    c=cfg["hidden_direction"]
    rcfg=representation.RepresentationConfig(steps=int(c["steps"]),learning_rate=float(c["learning_rate"]),
        weight_decay=float(c["weight_decay"]),rank=int(rank),alpha=float(rank),dropout=0.0,layer_indices=(),
        last_n_layers=int(c["last_n_layers"]),target_modules=tuple(c["target_modules"]),seed=int(cfg["seed"]))
    handles=representation.inject_lora_adapters(model,rcfg); originals=v2.capture_adapter_base_weights(handles)
    params=representation.adapter_parameters(handles)
    opt=torch.optim.AdamW(params,lr=float(c["learning_rate"]),weight_decay=float(c["weight_decay"]))
    active=[i for i,x in enumerate(cases) if float(x["minimum_norm_target_delta_norm"])>1e-10] or list(range(len(cases)))
    if not utility_contexts: raise ValueError("no external-Wikipedia train contexts")
    history=[]; started=time.perf_counter(); representation.set_adapter_scale(handles,1.0); model.eval()
    for step in range(int(c["steps"])):
        opt.zero_grad(set_to_none=True); case=cases[active[step%len(active)]]; cur=current_hidden(model,case,device)
        fl,fmargin=frozen_head_loss(cur,case,base_w,float(c["frozen_base_head_training_margin"]))
        dl,dcos,drel=direction_target_loss(cur,case,device)
        pl,psep=v2.pairwise_separation_loss(model,tok,prompt_records[int(case["prompt_position"])],
            margin=float(c["edited_head_pairwise_target"]),llama_like=llama_like,device=device)
        ul=v2.utility_hidden_loss(model,utility_contexts[step%len(utility_contexts)],device=device); l2=v2.adapter_l2(handles)
        loss=float(c["frozen_base_head_weight"])*fl+float(c["minimum_norm_direction_weight"])*dl+float(c["edited_head_pairwise_weight"])*pl+float(c["utility_hidden_weight"])*ul+float(c["adapter_l2_weight"])*l2
        if not torch.isfinite(loss): raise FloatingPointError("hidden-direction loss became non-finite")
        loss.backward(); torch.nn.utils.clip_grad_norm_(params,float(c["grad_clip"])); opt.step()
        if step==0 or (step+1)%25==0 or step+1==int(c["steps"]):
            row={"step":step+1,"rank":int(rank),"case_index":int(case["case_index"]),"loss":float(loss.detach().cpu()),
                "frozen_base_head_hinge":float(fl.detach().cpu()),"frozen_base_head_demotion_margin":float(fmargin.detach().cpu()),
                "minimum_norm_direction_loss":float(dl.detach().cpu()),"minimum_norm_direction_cosine":float(dcos.detach().cpu()),
                "minimum_norm_direction_relative_error":float(drel.detach().cpu()),"edited_head_pairwise_hinge":float(pl.detach().cpu()),
                "edited_head_pairwise_separation":float(psep.detach().cpu()),"utility_hidden_relative_mse":float(ul.detach().cpu()),
                "adapter_l2":float(l2.detach().cpu())}
            history.append(row)
            print("hdir-rank{} step {:3d}: loss={:.6f} frozen_margin={:.4f} dir_cos={:.4f} pair_sep={:.4f} wiki_hidden={:.6f}".format(
                rank,step+1,row["loss"],row["frozen_base_head_demotion_margin"],row["minimum_norm_direction_cosine"],
                row["edited_head_pairwise_separation"],row["utility_hidden_relative_mse"]))
    report={"rank":int(rank),"steps":int(c["steps"]),"active_direction_case_count":len(active),
        "training_seconds":time.perf_counter()-started,"history":history}
    write_json(log_path,report); return handles,originals,report

def main():
    args=parse_args(); config_path=Path(args.configuration).resolve(); cfg=load_configuration(config_path)
    if args.experiment_id!=cfg["configuration_id"]: raise ValueError("experiment ID must equal locked v3 configuration ID")
    state_args,run_dir=verify_prepared_state(args,cfg); source=v2.verify_source_run(Path(args.source_head_only_run),cfg)
    out=run_dir/"sure_head_hidden_direction_w1k"
    if out.exists(): raise FileExistsError(f"refusing to overwrite hidden-direction run: {out}")
    out.mkdir(parents=True)
    source_cfg=head.load_locked_configuration(SOURCE_CONFIGURATION)
    views,bundle_audit,generator_audit=head.load_atomic_bundle(Path(args.training_bundle).resolve(),Path(args.generator_receipt).resolve(),source_cfg)
    generator_model_audit=head.validate_generator_base_model(generator_audit,args.model_path)
    rwku_experiment._write_state(state_args,"TRAINING",configuration_path=str(config_path),
        configuration_sha256=artifact_access.sha256_file(config_path),source_head_only_run=str(Path(args.source_head_only_run).resolve()),
        official_evaluation_opened=False,posthoc_development_target=True)
    gagd.set_seed(int(cfg["seed"])); gagd.require_cuda_if_needed(cfg["acceptance"]["device_map"])
    margs=argparse.Namespace(model_path=args.model_path,dtype=cfg["acceptance"]["checkpoint_dtype"],
        device_map=cfg["acceptance"]["device_map"],gradient_checkpointing=False)
    model,tok=gagd.load_model_and_tokenizer(margs,for_training=False)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side="right"; device=gagd.first_device(model); llama_like=core.is_llama_like(model,tok)
    prompt_records=head.compile_prompt_records(views,tok,neutral_target=str(cfg["neutral_target"]))
    inp=model.get_input_embeddings(); outp=model.get_output_embeddings()
    if inp is None or outp is None or inp.weight.data_ptr()!=outp.weight.data_ptr():
        raise ValueError("v3 requires base model to begin with tied vocabulary weights")
    identity=wikipedia.model_identity(model,tok,args.model_path); output_layer=core.untie_and_freeze_output_head(model)
    base_w=model.get_input_embeddings().weight
    if base_w.data_ptr()==output_layer.weight.data_ptr() or base_w.requires_grad: raise RuntimeError("base readout not frozen/untied")
    payload=torch.load(source["stage1_delta"],map_location="cpu"); selected_ids=[int(x) for x in payload["row_ids"]]
    stage1_delta=payload["delta"].float().to(device); selected_tensor=torch.tensor(selected_ids,device=output_layer.weight.device,dtype=torch.long)
    base_head_rows=output_layer.weight.index_select(0,selected_tensor).detach().clone()
    base_input_rows=base_w.index_select(0,selected_tensor.to(base_w.device)).detach()
    maxdiff=float((base_input_rows.float()-base_head_rows.to(base_input_rows.device).float()).abs().max().item())
    if maxdiff!=0.0: raise RuntimeError(f"base input readout differs from cloned head before sparse edit: {maxdiff}")
    optimization=head.optimization_namespace(source_cfg,prompt_count=len(prompt_records))
    second,utility_hidden,utility_lse,utility_meta=learner.load_utility_cache(Path(args.utility_cache).resolve(),
        expected_sample_size=optimization.utility_sample_size,expected_prompt_count=optimization.utility_prompt_count,
        expected_hidden_size=int(output_layer.weight.shape[1]),expected_model_probe=identity["model_probe_sha256"],
        expected_tokenizer_probe=identity["tokenizer_probe_sha256"])
    del second
    utility_audit=head.validate_w1k_utility_metadata(utility_meta,source_cfg)
    sel_probs=v2.selected_base_probabilities_full_head(output_layer,selected_ids,utility_hidden,utility_lse,batch_size=optimization.utility_eval_batch_size)
    train_idx,guard_idx,pool_report=learner.build_disjoint_token_conditioned_utility_pools(
        selected_base_probabilities=sel_probs,selected_ids=selected_ids,topk_per_row=optimization.utility_token_topk_per_row,
        uniform_prompt_count=optimization.utility_uniform_prompt_count,split_seed=optimization.utility_pool_seed)
    core.materialize_output_delta(output_layer,selected_ids,stage1_delta)
    actual_delta=learner.actual_selected_delta(output_layer,selected_ids,base_head_rows.float())
    edited_rows=output_layer.weight.index_select(0,selected_tensor).detach().clone(); head_norm=float(actual_delta.norm().detach().cpu())
    if head_norm>float(cfg["acceptance"]["max_head_delta_norm"]): raise ValueError("source head delta exceeds norm budget")
    source_stage1=head.materialized_atomic_report(model,tok,prompt_records,device,llama_like=llama_like,
        required_margin=float(cfg["acceptance"]["required_pairwise_margin"]))
    declared=read_json(source["stage1_report"])
    if source_stage1.get("pairwise_margin_failure_positions")!=declared.get("pairwise_margin_failure_positions"):
        raise RuntimeError("reloaded source failure positions differ")
    neutral_ids=v2._completion_token_ids(tok,str(cfg["neutral_target"]),llama_like)
    sensitive_ids=[x for x in selected_ids if x not in set(neutral_ids)]
    cases,direction_audit=build_direction_cases(model,tok,prompt_records,base_w,sensitive_ids,neutral_ids,
        float(cfg["hidden_direction"]["frozen_base_head_training_margin"]),llama_like,device)
    write_json(out/"hidden_direction_target_report.json",{**direction_audit,"sensitive_selected_token_ids":sensitive_ids,
        "neutral_token_ids":[int(x) for x in neutral_ids],"base_readout_validation_max_abs_diff":maxdiff,
        "sensitive_view_scope":cfg["hidden_direction"]["sensitive_view_scope"],"posthoc_development_target":True})
    texts,wiki_meta=wikipedia.load_wikipedia_train(Path(args.wikipedia_dir).resolve()); c=cfg["hidden_direction"]
    nt=min(int(c["utility_train_prompt_count"]),int(train_idx.numel())); ng=min(int(c["utility_gate_prompt_count"]),int(guard_idx.numel()))
    train_contexts=v2.build_utility_contexts(tok,texts,utility_meta,utility_hidden,train_idx[:nt].tolist())
    gate_contexts=v2.build_utility_contexts(tok,texts,utility_meta,utility_hidden,guard_idx[:ng].tolist())
    write_json(out/"utility_replay_report.json",{"source_pool":pool_report,"wikipedia_dataset":wiki_meta,"train_prompt_count":len(train_contexts),
        "gate_prompt_count":len(gate_contexts),"train_indices_sha256":hashlib.sha256(json.dumps(train_idx[:nt].tolist()).encode()).hexdigest(),
        "gate_indices_sha256":hashlib.sha256(json.dumps(guard_idx[:ng].tolist()).encode()).hexdigest(),
        "train_guard_overlap_count":len(set(train_idx[:nt].tolist())&set(guard_idx[:ng].tolist())),"official_rwku_records_accessed":False})
    attempts=[]; chosen=None; chosen_handles=None; chosen_originals=None; started=time.perf_counter()
    for rank in [int(x) for x in c["rank_ladder"]]:
        output_layer.weight.index_copy_(0,selected_tensor,edited_rows.to(device=output_layer.weight.device,dtype=output_layer.weight.dtype))
        handles,originals,_=train_rank(model,tok,prompt_records,cases,train_contexts,base_w,rank,cfg,llama_like,device,
            out/f"rank{rank}_training_history.json")
        rank_ok=False
        for scale in [float(x) for x in c["candidate_scales"]]:
            mat=v2.materialize_adapter_candidate(handles,originals,scale)
            output_layer.weight.index_copy_(0,selected_tensor,edited_rows.to(device=output_layer.weight.device,dtype=output_layer.weight.dtype))
            atomic=head.materialized_atomic_report(model,tok,prompt_records,device,llama_like=llama_like,
                required_margin=float(cfg["acceptance"]["required_pairwise_margin"]))
            proxy=frozen_proxy_report(model,cases,base_w,device); rd=v2.representation_delta_report(handles,originals)
            cand={"rank":rank,"scale":scale,"materialization":mat,"atomic":atomic,"frozen_base_head_proxy":proxy,
                "representation_delta":rd,"behavior_safe":v2.behavior_safe(atomic),"frozen_base_head_proxy_safe":proxy_safe(proxy,cfg),
                "representation_norm_safe":bool(rd["relative_frobenius"]<=float(c["max_relative_frobenius_delta"])),
                "head_delta_norm":head_norm,"head_norm_safe":bool(head_norm<=float(cfg["acceptance"]["max_head_delta_norm"])),
                "official_rwku_records_accessed":False}
            print("hdir candidate rank={} scale={}: FS={} other={} minsep={:.4f} frozen_recovery={:.2f}% frozen_minmargin={:.4f} relnorm={:.6f}".format(
                rank,scale,atomic.get("FS"),atomic.get("generated_subject_FS"),
                float(atomic.get("minimum_overall_separation",float("nan"))),float(proxy["recovery_percentage"]),
                float(proxy["minimum_demotion_margin"]),float(rd["relative_frobenius"])))
            pre=bool(cand["behavior_safe"] and cand["frozen_base_head_proxy_safe"] and cand["representation_norm_safe"] and cand["head_norm_safe"])
            if pre:
                ukl=v2.exact_full_vocab_utility_kl(model,tok,gate_contexts,output_layer=output_layer,selected_tensor=selected_tensor,
                    base_head_rows=base_head_rows,edited_head_rows=edited_rows,handles=handles,original_adapter_weights=originals,
                    scale=scale,device=device,batch_size=int(c["utility_context_batch_size"]))
                checks={"mean":ukl["utility_kl_mean"]<=float(cfg["acceptance"]["utility_kl_mean_budget"]),
                    "p95":ukl["utility_kl_p95"]<=float(cfg["acceptance"]["utility_kl_p95_budget"]),
                    "max":ukl["utility_kl_max"]<=float(cfg["acceptance"]["utility_kl_max_budget"])}
                cand["utility_kl"]=ukl; cand["utility_guard_checks"]=checks; cand["utility_safe"]=bool(all(checks.values())); cand["feasible"]=cand["utility_safe"]
                print("  exact W1K hidden-direction gate: mean={:.6f} p95={:.6f} max={:.6f} safe={}".format(
                    ukl["utility_kl_mean"],ukl["utility_kl_p95"],ukl["utility_kl_max"],cand["utility_safe"]))
            else:
                cand["utility_safe"]=False; cand["feasible"]=False; cand["utility_gate_skipped"]=True
            attempts.append(cand); write_json(out/f"rank{rank}_scale{str(scale).replace('.','p')}_report.json",cand)
            if cand["feasible"]:
                chosen=cand; chosen_handles=handles; chosen_originals=originals; rank_ok=True; break
        if rank_ok: break
        v2.restore_adapter_base_weights(handles,originals)
        output_layer.weight.index_copy_(0,selected_tensor,edited_rows.to(device=output_layer.weight.device,dtype=output_layer.weight.dtype))
        representation.remove_lora_adapters(handles,merge_scale=0.0)
    write_json(out/"hidden_direction_attempts.json",{"attempts":attempts})
    if chosen is None:
        write_json(out/"infeasible.json",{"configuration_id":cfg["configuration_id"],"source_stage1":source_stage1,
            "hidden_direction_target_report":direction_audit,"attempts":attempts,
            "reason":"no BF16-safe hidden-direction candidate passed atomic, frozen-base-head proxy, norm, and W1K gates",
            "posthoc_development_target":True,"official_rwku_records_accessed":False})
        raise RuntimeError("RWKU hidden-direction repair found no feasible checkpoint")
    final_rd=v2.representation_delta_report(chosen_handles,chosen_originals)
    delta_payload={"rank":int(chosen["rank"]),"scale":float(chosen["scale"]),"modules":{}}
    for h,o in zip(chosen_handles,chosen_originals):
        delta=h.wrapper.base.weight.detach().float()-o.to(device=h.wrapper.base.weight.device,dtype=torch.float32)
        delta_payload["modules"][h.path]=delta.cpu()
    delta_path=out/"hidden_direction_delta.pt"; torch.save(delta_payload,delta_path)
    representation.remove_lora_adapters(chosen_handles,merge_scale=0.0)
    final_atomic=head.materialized_atomic_report(model,tok,prompt_records,device,llama_like=llama_like,
        required_margin=float(cfg["acceptance"]["required_pairwise_margin"]))
    final_proxy=frozen_proxy_report(model,cases,base_w,device)
    if not v2.behavior_safe(final_atomic): raise RuntimeError("final checkpoint failed atomic gate")
    if not proxy_safe(final_proxy,cfg): raise RuntimeError("final checkpoint failed frozen-base-head proxy gate")
    checkpoint_path=out/"checkpoint"; learner.save_checkpoint(model,tok,checkpoint_path)
    head_delta_path=out/"stage1_sparse_head_delta.pt"; torch.save({"row_ids":selected_ids,"delta":actual_delta.detach().cpu()},head_delta_path)
    write_json(out/"final_atomic_view_report.json",final_atomic); write_json(out/"final_frozen_base_head_proxy_report.json",final_proxy)
    training_report={"schema_version":"rwku_sure_head_hidden_direction_w1k_training_report_v1",
        "protocol_label":artifact_access.TARGET_ONLY_PROTOCOL_LABEL,"protocol_status":PROTOCOL_STATUS,"method":cfg["method"],
        "configuration_id":cfg["configuration_id"],"development_only":True,"posthoc_development_target":True,
        "official_rwku_metrics_observed_before_method_design":True,
        "target":{"seed":0,"entity":cfg["target_entity"],"entity_id":cfg["target_entity_id"]},
        "source_head_only_run":{"path":str(Path(args.source_head_only_run).resolve()),
            "stage1_delta_sha256":artifact_access.sha256_file(source["stage1_delta"]),
            "stage1_report_sha256":artifact_access.sha256_file(source["stage1_report"]),
            "infeasible_report_sha256":artifact_access.sha256_file(source["infeasible"]),"official_evaluation_opened":False},
        "atomic_bundle":bundle_audit,"atomic_generator_base_model":generator_model_audit,"utility_cache":utility_audit,
        "source_stage1_atomic_report":source_stage1,"head_delta_norm":head_norm,"hidden_direction_targets":direction_audit,
        "hidden_direction_repair":{"rank_ladder":c["rank_ladder"],"target_modules":c["target_modules"],"last_n_layers":c["last_n_layers"],
            "sensitive_view_scope":c["sensitive_view_scope"],"frozen_base_head_source":"untouched_base_input_embedding_matrix_after_output_head_untie",
            "chosen":chosen,"final_representation_delta":final_rd,"final_frozen_base_head_proxy":final_proxy},
        "final_training_view_report":final_atomic,"training_seconds":time.perf_counter()-started,
        "official_rwku_records_accessed":False,"official_rwku_records_used_for_training_or_selection":False,
        "final_evaluation_used_for_training_or_selection":False}
    tr_path=out/"training_report.json"; write_json(tr_path,training_report); tr_sha=artifact_access.sha256_file(tr_path)
    receipt_path=run_dir/"checkpoint_receipt.json"
    method_cfg={"method":cfg["method"],"configuration_id":cfg["configuration_id"],"configuration_path":str(config_path),
        "configuration_sha256":artifact_access.sha256_file(config_path),"training_report_path":str(tr_path.resolve()),
        "training_report_sha256":tr_sha,
        "editable_parameters":"stage1_sparse_lm_head_rows_plus_last_layer_down_proj_low_rank_frozen_base_head_hidden_direction_repair",
        "source_head_only_stage1_delta_sha256":artifact_access.sha256_file(source["stage1_delta"]),
        "representation_rank":int(chosen["rank"]),"representation_scale":float(chosen["scale"]),
        "representation_relative_frobenius":float(final_rd["relative_frobenius"]),
        "frozen_base_head_proxy_recovery":float(final_proxy["recovery_percentage"]),
        "frozen_base_head_proxy_min_demotion_margin":float(final_proxy["minimum_demotion_margin"]),
        "head_delta_norm":head_norm,"utility_gate_kind":"exact_full_vocabulary_base_to_edited",
        "utility_gate_prompt_count":int(chosen["utility_kl"]["utility_prompt_count"]),"posthoc_development_target":True,
        "official_rwku_records_used_for_selection":False}
    impl=[SCRIPT_PATH,PROJECT_ROOT/"scripts"/"rwku_sure_repr_rescue_w1k.py",PROJECT_ROOT/"scripts"/"rwku_sure_head_only_w1k.py",
        PROJECT_ROOT/"scripts"/"rwku_representation.py",PROJECT_ROOT/"scripts"/"build_sure_wikipedia_stats.py",
        PROJECT_ROOT/"scripts"/"gagd_compare.py",PROJECT_ROOT/"scripts"/"rwku_artifact_access.py",
        PROJECT_ROOT/"scripts"/"rwku_checkpoint_receipt.py",PROJECT_ROOT/"scripts"/"rwku_eval.py",
        PROJECT_ROOT/"scripts"/"rwku_experiment.py",PROJECT_ROOT/"scripts"/"sure_canonical_core.py",
        PROJECT_ROOT/"scripts"/"sure_minimal_two_stage.py"]
    receipt=checkpoint_receipt.create_checkpoint_receipt(destination=receipt_path,experiment_id=str(args.experiment_id),
        protocol_label=artifact_access.TARGET_ONLY_PROTOCOL_LABEL,protocol_status=PROTOCOL_STATUS,target_entity=str(cfg["target_entity"]),
        target_entity_id=str(cfg["target_entity_id"]),base_model_identity=rwku_experiment.local_model_identity(args.model_path),
        base_model_revision=str(args.model_revision),tokenizer_identity={"name_or_path":tok.name_or_path,"class":tok.__class__.__name__,
            "vocab_size":len(tok),"eos_token_id":tok.eos_token_id,"tokenizer_probe_sha256":identity["tokenizer_probe_sha256"]},
        checkpoint_paths=[checkpoint_path],training_bundle_path=Path(args.training_bundle).resolve(),optimization_protection_path=None,
        mcf_retain_optimization_paths=[],mcf_repair_gate_paths=[],matched_protection_train_path=None,matched_protection_gate_path=None,
        method_configuration=method_cfg,implementation_files=impl,
        sampler_provenance={"atomic_view_order_sha256":bundle_audit["view_ids_sha256"],"utility_pool_seed":optimization.utility_pool_seed,
            "training_seed":int(cfg["seed"]),"hidden_direction_case_count":len(cases),
            "hidden_direction_case_ids_sha256":hashlib.sha256(json.dumps([[int(x["prompt_position"]),int(x["prediction_position"]),int(x["target_token_id"])] for x in cases]).encode()).hexdigest(),
            "utility_train_indices_sha256":hashlib.sha256(json.dumps(train_idx[:nt].tolist()).encode()).hexdigest(),
            "utility_gate_indices_sha256":hashlib.sha256(json.dumps(guard_idx[:ng].tolist()).encode()).hexdigest(),
            "official_rwku_records_accessed":False},
        generator_receipt_path=Path(args.generator_receipt).resolve(),official_locked_eval_path=run_dir/"official_locked_eval.json",
        confirmatory=False,additional_artifact_paths={"locked_configuration":config_path,"utility_cache":Path(args.utility_cache).resolve(),
            "source_head_only_stage1_delta":source["stage1_delta"],"source_head_only_infeasible_report":source["infeasible"],
            "training_report":tr_path,"hidden_direction_target_report":out/"hidden_direction_target_report.json",
            "final_frozen_base_head_proxy_report":out/"final_frozen_base_head_proxy_report.json",
            "final_sparse_head_delta":head_delta_path,"final_hidden_direction_delta":delta_path})
    rwku_experiment._write_state(state_args,"CHECKPOINT_FROZEN",checkpoint_receipt=str(receipt_path.resolve()),
        checkpoint_receipt_sha256=receipt["receipt_sha256"],official_evaluation_opened=False,hidden_direction_feasible=True,
        posthoc_development_target=True,training_report=str(tr_path.resolve()))
    print(f"RWKU hidden-direction checkpoint frozen: {checkpoint_path}")
    print(f"Hidden-direction rank/scale: {chosen['rank']}/{chosen['scale']}")
    print(f"Atomic direct success: {final_atomic['FS']}")
    print(f"Other atomic-view success: {final_atomic['generated_subject_FS']}")
    print(f"Frozen-base-head proxy recovery/minmargin: {final_proxy['recovery_percentage']:.2f}%/{final_proxy['minimum_demotion_margin']:.6f}")
    print("Wikipedia full-vocabulary KL mean/p95/max: "
        f"{chosen['utility_kl']['utility_kl_mean']:.6f}/{chosen['utility_kl']['utility_kl_p95']:.6f}/{chosen['utility_kl']['utility_kl_max']:.6f}")
    print("Development-only Stephen King v3 checkpoint frozen; any later Stephen King official evaluation is not confirmatory.")

if __name__=="__main__": main()
