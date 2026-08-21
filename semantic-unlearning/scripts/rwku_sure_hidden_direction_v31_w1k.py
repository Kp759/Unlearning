#!/usr/bin/env python3
"""Answer-level frozen-base-head hidden-direction repair for RWKU (v3.1)."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import torch
import torch.nn.functional as F
import build_sure_wikipedia_stats as wikipedia
import gagd_compare as gagd
import rwku_representation as representation
import rwku_sure_head_only_w1k as head
import rwku_sure_hidden_direction_w1k as v3
import rwku_sure_repr_rescue_w1k as v2
import sure_canonical_core as core

SCRIPT_PATH=Path(__file__).resolve(); PROJECT_ROOT=SCRIPT_PATH.parents[1]
DEFAULT_CONFIGURATION=PROJECT_ROOT/"config"/"rwku"/"sure_head_hidden_direction_v31_w1k_seed0.json"
SCHEMA="rwku_sure_head_hidden_direction_v31_w1k_configuration_v1"
EXPERIMENT_ID="rwku-h-w1k-stephen-king-hidden-direction-seed0-v31"
PROTOCOL_STATUS="rwku_target_only_auxwiki_sure_hidden_direction_v31_w1k_posthoc_development"
_RUNTIME: Dict[str,Any]={}

def read_json(path):
    with Path(path).open("r",encoding="utf-8") as f: x=json.load(f)
    if not isinstance(x,dict): raise ValueError(f"Expected JSON object: {path}")
    return x

def load_configuration(path):
    c=read_json(path)
    req={"schema_version":SCHEMA,"configuration_id":EXPERIMENT_ID,"development_only":True,
        "posthoc_development_target":True,"official_rwku_metrics_observed_before_method_design":True,
        "seed":0,"target_entity":"Stephen King","target_entity_id":"rwku:1_Stephen_King",
        "source_head_only_configuration_id":head.EXPECTED_CONFIGURATION_ID,"neutral_target":"Unknown"}
    for k,e in req.items():
        if c.get(k)!=e: raise ValueError(f"v3.1 configuration changed {k}")
    h=c.get("hidden_direction",{})
    if h.get("objective")!="answer_level_frozen_base_head_sensitive_vs_neutral": raise ValueError("v3.1 objective changed")
    if h.get("sensitive_view_scope")!="all_generated_atomic_views": raise ValueError("v3.1 view scope changed")
    if tuple(h.get("rank_ladder",()))!=(1,2,4): raise ValueError("rank ladder must remain 1,2,4")
    if int(h.get("last_n_layers",0))!=1 or tuple(h.get("target_modules",()))!=("down_proj",): raise ValueError("must remain final-layer down_proj only")
    if int(h.get("answer_batch_size",0))<=0 or int(h.get("checkpoint_interval",0))<=0: raise ValueError("invalid batching/checkpoint settings")
    a=c.get("acceptance",{}); locked={"required_pairwise_margin":0.01,"required_direct_success":100.0,
        "required_other_atomic_view_success":100.0,"max_frozen_base_head_recovery":0.0,
        "min_frozen_base_head_demotion_margin":0.05,"max_head_delta_norm":1.5,
        "utility_kl_mean_budget":0.01,"utility_kl_p95_budget":0.05,"utility_kl_max_budget":0.5,
        "checkpoint_dtype":"bf16","device_map":"single"}
    for k,e in locked.items():
        if a.get(k)!=e: raise ValueError(f"v3.1 acceptance changed {k}")
    b=c.get("data_boundary",{})
    for k in ("official_rwku_records_available_to_learner","official_rwku_records_used_for_checkpoint_selection","neighbor_prompts_used_for_training_or_selection"):
        if b.get(k) is not False: raise ValueError(f"v3.1 data boundary changed {k}")
    if b.get("external_wikipedia_only_utility") is not True: raise ValueError("utility must remain external Wikipedia")
    return c

def completion_layout(tok,prefix,suffix,llama_like):
    p=[int(x) for x in tok(prefix)["input_ids"]]; t=v2._completion_token_ids(tok,suffix,llama_like)
    ids=tok(f"{prefix} {suffix}",return_tensors="pt")["input_ids"][0].cpu().contiguous(); start=len(p)-1
    pos=list(range(start,start+len(t)))
    if not pos or pos[-1]>=int(ids.numel()): raise RuntimeError("answer completion positions invalid")
    return ids,[int(x) for x in t],[int(x) for x in pos]

def build_answer_cases(model,tok,prompt_records,base_w,sensitive_ids,neutral_ids,training_margin,llama_like,device):
    del model,base_w,sensitive_ids,neutral_ids,training_margin,device
    out=[]
    for i,r in enumerate(prompt_records):
        rr=r["requested_rewrite"]; q=str(r["prompt_text"]); s=str(rr["target_sensitive"]["str"]); n=str(rr["target_reference"]["str"])
        si,st,sp=completion_layout(tok,q,s,llama_like); ni,nt,np=completion_layout(tok,q,n,llama_like)
        out.append({"case_index":i,"prompt_position":i,"prompt_kind":str(r.get("prompt_kind","")),"prompt_index":int(r.get("prompt_index",0)),
            "sensitive_answer":s,"neutral_answer":n,"sensitive_input_ids":si,"sensitive_target_ids":st,"sensitive_positions":sp,
            "neutral_input_ids":ni,"neutral_target_ids":nt,"neutral_positions":np,
            "prediction_position":sp[0],"target_token_id":st[0]})
    if not out: raise ValueError("no v3.1 answer cases")
    return out,{"case_count":len(out),"prompt_count":len(out),"objective":"answer_level_frozen_base_head_sensitive_vs_neutral",
        "sensitive_view_scope":"all_generated_atomic_views","official_rwku_records_accessed":False}

def pad(seqs,pad_id,device):
    w=max(int(x.numel()) for x in seqs); ids=torch.full((len(seqs),w),int(pad_id),dtype=torch.long); att=torch.zeros_like(ids)
    for i,x in enumerate(seqs): ids[i,:x.numel()]=x; att[i,:x.numel()]=1
    return ids.to(device),att.to(device)

def nll(hidden,cases,prefix,w):
    vals=[]
    for i,c in enumerate(cases):
        pos=torch.tensor(c[f"{prefix}_positions"],device=hidden.device); tgt=torch.tensor(c[f"{prefix}_target_ids"],device=hidden.device)
        rows=hidden[i].index_select(0,pos); logits=F.linear(rows.to(dtype=w.dtype),w).float()
        vals.append(-F.log_softmax(logits,dim=-1).gather(1,tgt[:,None]).mean())
    return torch.stack(vals)

def answer_nlls(model,tok,cases,base_w,device):
    pid=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pid is None: raise ValueError("tokenizer lacks pad/eos")
    si,sa=pad([c["sensitive_input_ids"] for c in cases],pid,device); ni,na=pad([c["neutral_input_ids"] for c in cases],pid,device)
    sh,_=wikipedia._final_hidden_only(model,{"input_ids":si,"attention_mask":sa}); nh,_=wikipedia._final_hidden_only(model,{"input_ids":ni,"attention_mask":na})
    ew=model.get_output_embeddings().weight
    return nll(sh,cases,"sensitive",base_w),nll(nh,cases,"neutral",base_w),nll(sh,cases,"sensitive",ew),nll(nh,cases,"neutral",ew)

@torch.no_grad()
def answer_proxy_report(model,cases,base_w,device):
    tok=_RUNTIME["tokenizer"]; bs=int(_RUNTIME["answer_eval_batch_size"]); details=[]
    for start in range(0,len(cases),bs):
        b=cases[start:start+bs]; s,n,_,_=answer_nlls(model,tok,b,base_w,device)
        for c,sv,nv in zip(b,s.cpu(),n.cpu()):
            sep=float((sv-nv).item()); details.append({"case_index":int(c["case_index"]),"prompt_position":int(c["prompt_position"]),
                "prompt_kind":str(c["prompt_kind"]),"sensitive_answer":c["sensitive_answer"],"neutral_answer":c["neutral_answer"],
                "sensitive_nll":float(sv),"neutral_nll":float(nv),"demotion_margin":sep,"recovered":bool(sep<=0.0)})
    margins=[x["demotion_margin"] for x in details]; rec=sum(int(x["recovered"]) for x in details); by={}
    for kind in sorted({x["prompt_kind"] for x in details}):
        z=[x for x in details if x["prompt_kind"]==kind]; zr=sum(int(x["recovered"]) for x in z); zm=[x["demotion_margin"] for x in z]
        by[kind]={"count":len(z),"recovery_count":zr,"recovery_percentage":100*zr/len(z),"mean_demotion_margin":sum(zm)/len(zm),"minimum_demotion_margin":min(zm)}
    return {"prompt_count":len(details),"recovery_count":rec,"recovery_percentage":100*rec/len(details),
        "mean_demotion_margin":sum(margins)/len(margins),"minimum_demotion_margin":min(margins),"maximum_demotion_margin":max(margins),
        "mean_sensitive_nll":sum(x["sensitive_nll"] for x in details)/len(details),"mean_neutral_nll":sum(x["neutral_nll"] for x in details)/len(details),
        "by_prompt_kind":by,"details":details,"recovery_definition":"sensitive_answer_nll <= neutral_answer_nll under frozen base vocabulary readout",
        "readout_source":"untouched_base_input_embedding_matrix_after_output_head_untie","official_rwku_records_accessed":False}

def proxy_safe(r,cfg):
    a=cfg["acceptance"]; return bool(float(r["recovery_percentage"])<=float(a["max_frozen_base_head_recovery"]) and float(r["minimum_demotion_margin"])>=float(a["min_frozen_base_head_demotion_margin"]))

def adapter_state(handles): return [(h.wrapper.lora_A.detach().cpu().clone(),h.wrapper.lora_B.detach().cpu().clone()) for h in handles]
@torch.no_grad()
def restore_state(handles,state):
    for h,(a,b) in zip(handles,state): h.wrapper.lora_A.copy_(a.to(h.wrapper.lora_A.device)); h.wrapper.lora_B.copy_(b.to(h.wrapper.lora_B.device))

def checkpoint_metrics(model,tok,prompt_records,cases,base_w,utility,cfg,llama_like,device):
    p=answer_proxy_report(model,cases,base_w,device); atomic=head.materialized_atomic_report(model,tok,prompt_records,device,llama_like=llama_like,required_margin=float(cfg["acceptance"]["required_pairwise_margin"]))
    take=min(int(cfg["hidden_direction"]["checkpoint_wiki_prompt_count"]),len(utility)); vals=[]
    with torch.no_grad():
        for x in utility[:take]: vals.append(float(v2.utility_hidden_loss(model,x,device=device).cpu()))
    fails=int(atomic.get("direct_margin_failures",0))+int(atomic.get("generated_subject_margin_failures",0))
    return p,atomic,fails,sum(vals)/max(len(vals),1),take

def train_rank(model,tok,prompt_records,cases,utility,base_w,rank,cfg,llama_like,device,log_path):
    h=cfg["hidden_direction"]; rc=representation.RepresentationConfig(steps=int(h["steps"]),learning_rate=float(h["learning_rate"]),weight_decay=float(h["weight_decay"]),
        rank=int(rank),alpha=float(rank),dropout=0.0,layer_indices=(),last_n_layers=1,target_modules=("down_proj",),seed=int(cfg["seed"]))
    handles=representation.inject_lora_adapters(model,rc); originals=v2.capture_adapter_base_weights(handles); params=representation.adapter_parameters(handles)
    opt=torch.optim.AdamW(params,lr=float(h["learning_rate"]),weight_decay=float(h["weight_decay"])); bs=min(int(h["answer_batch_size"]),len(cases)); interval=int(h["checkpoint_interval"])
    history=[]; checks=[]; best=None; best_state=None; best_step=None; started=time.perf_counter(); representation.set_adapter_scale(handles,1.0); model.eval()
    for step in range(int(h["steps"])):
        opt.zero_grad(set_to_none=True); idx=[(step*bs+j)%len(cases) for j in range(bs)]; batch=[cases[i] for i in idx]
        bsens,bneu,esens,eneu=answer_nlls(model,tok,batch,base_w,device); bsep=bsens-bneu; esep=esens-eneu
        bl=F.relu(float(h["frozen_base_head_training_margin"])-bsep).square().mean(); el=F.relu(float(h["edited_head_pairwise_target"])-esep).square().mean()
        ul=v2.utility_hidden_loss(model,utility[step%len(utility)],device=device); l2=v2.adapter_l2(handles)
        loss=float(h["frozen_base_head_answer_weight"])*bl+float(h["edited_head_answer_weight"])*el+float(h["utility_hidden_weight"])*ul+float(h["adapter_l2_weight"])*l2
        if not torch.isfinite(loss): raise FloatingPointError("v3.1 loss non-finite")
        loss.backward(); torch.nn.utils.clip_grad_norm_(params,float(h["grad_clip"])); opt.step()
        if step==0 or (step+1)%25==0 or step+1==int(h["steps"]):
            row={"step":step+1,"rank":int(rank),"loss":float(loss.detach().cpu()),"batch_base_answer_hinge":float(bl.detach().cpu()),
                "batch_base_answer_separation_mean":float(bsep.mean().detach().cpu()),"batch_base_answer_recovery_percentage":float((bsep<=0).float().mean().mul(100).detach().cpu()),
                "batch_edited_answer_hinge":float(el.detach().cpu()),"batch_edited_answer_separation_mean":float(esep.mean().detach().cpu()),"utility_hidden_relative_mse":float(ul.detach().cpu())}
            history.append(row); print("v31-rank{} step {:3d}: loss={:.6f} base_sep={:.4f} base_rec={:.2f}% edited_sep={:.4f} wiki_hidden={:.6f}".format(rank,step+1,row["loss"],row["batch_base_answer_separation_mean"],row["batch_base_answer_recovery_percentage"],row["batch_edited_answer_separation_mean"],row["utility_hidden_relative_mse"]))
        if (step+1)%interval==0 or step+1==int(h["steps"]):
            p,a,f,w,t=checkpoint_metrics(model,tok,prompt_records,cases,base_w,utility,cfg,llama_like,device); key=(int(p["recovery_count"]),-float(p["minimum_demotion_margin"]),int(f),float(w))
            checks.append({"step":step+1,"selection_key":list(key),"frozen_base_head_answer_proxy":p,"atomic":a,"atomic_margin_failure_count":f,"checkpoint_wiki_hidden_relative_mse_mean":w,"checkpoint_wiki_prompt_count":t})
            print("  v31 checkpoint step {}: answer_rec={:.2f}% minsep={:.4f} atomic_fail={} wiki_hidden={:.6f}".format(step+1,p["recovery_percentage"],p["minimum_demotion_margin"],f,w))
            if best is None or key<best: best=key; best_state=adapter_state(handles); best_step=step+1
    if best_state is None: raise RuntimeError("v3.1 selected no adapter checkpoint")
    restore_state(handles,best_state); report={"rank":int(rank),"steps":int(h["steps"]),"best_checkpoint_step":int(best_step),"best_selection_key":list(best),"history":history,"checkpoint_evaluations":checks,"training_seconds":time.perf_counter()-started,"official_rwku_records_accessed":False}; core.write_json(log_path,report)
    return handles,originals,report

def configure(cfg):
    v3.SCRIPT_PATH=SCRIPT_PATH; v3.SCHEMA=SCHEMA; v3.EXPERIMENT_ID=EXPERIMENT_ID; v3.PROTOCOL_STATUS=PROTOCOL_STATUS; v3.DEFAULT_CONFIGURATION=DEFAULT_CONFIGURATION
    v3.load_configuration=load_configuration; v3.build_direction_cases=build_answer_cases; v3.frozen_proxy_report=answer_proxy_report; v3.proxy_safe=proxy_safe; v3.train_rank=train_rank
    _RUNTIME["answer_eval_batch_size"]=int(cfg["hidden_direction"]["answer_eval_batch_size"])

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model-path",required=True); p.add_argument("--model-revision",default="local_pinned_snapshot"); p.add_argument("--training-bundle",type=Path,required=True); p.add_argument("--generator-receipt",type=Path,required=True); p.add_argument("--utility-cache",type=Path,required=True); p.add_argument("--wikipedia-dir",type=Path,required=True); p.add_argument("--source-head-only-run",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--experiment-id",default=EXPERIMENT_ID); p.add_argument("--configuration",type=Path,default=DEFAULT_CONFIGURATION); args=p.parse_args()
    cfg=load_configuration(args.configuration); configure(cfg); original=gagd.load_model_and_tokenizer
    def wrapped(*a,**kw):
        model,tok=original(*a,**kw); _RUNTIME["tokenizer"]=tok; return model,tok
    gagd.load_model_and_tokenizer=wrapped
    sys.argv=["rwku_sure_hidden_direction_v31_w1k.py","--model-path",args.model_path,"--model-revision",args.model_revision,"--training-bundle",str(args.training_bundle),"--generator-receipt",str(args.generator_receipt),"--utility-cache",str(args.utility_cache),"--wikipedia-dir",str(args.wikipedia_dir),"--source-head-only-run",str(args.source_head_only_run),"--output-root",str(args.output_root),"--experiment-id",args.experiment_id,"--configuration",str(args.configuration)]
    try: v3.main()
    finally: gagd.load_model_and_tokenizer=original
if __name__=="__main__": main()
