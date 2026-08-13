#!/usr/bin/env python3
"""Strict forget-only ZsRE Stage-2 sparse sensitive-row LM-head repair."""
from __future__ import annotations

import argparse, json, random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch import nn

import gagd_compare as gagd
import mquake_forget_only_no_neutral as sparse
import zsre_gagd_setting5e_active_repair as zsre_sure
import zsre_no_neutral_stage1_emb_lm as s1
import zsre_zero_unlearn_official_eval as zsre

PROTOCOL = "zsre_zerounlearn_forget_only_locked_no_neutral"


def args():
    p=argparse.ArgumentParser()
    p.add_argument("--model-path",required=True); p.add_argument("--training-visible-path",required=True)
    p.add_argument("--split-manifest",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--seed",type=int,required=True); p.add_argument("--forget-num",type=int,default=50)
    p.add_argument("--repair-steps",type=int,default=800); p.add_argument("--repair-lr",type=float,default=5e-3)
    p.add_argument("--repair-margin",type=float,default=.05); p.add_argument("--repair-l2",type=float,default=1e-6)
    p.add_argument("--batch-size",type=int,default=8); p.add_argument("--check-every",type=int,default=25)
    p.add_argument("--candidate-scales",default="1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0")
    p.add_argument("--dtype",choices=("bf16","fp16","fp32"),default="bf16")
    p.add_argument("--device-map",choices=("single","auto"),default="single")
    return p.parse_args()


def write(path:Path,x:Any):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(x,indent=2)+"\n")


def cases(records,tok,llama_like):
    return [c for r in records for c in zsre.expand_prediction_cases(r,tok,llama_like=llama_like,prompt_types=("rewrite",))]


def logits(model,tok,cs,device):
    enc=tok([c.prompt for c in cs],padding=True,return_tensors="pt").to(device)
    out=model(**enc,use_cache=False).logits; pos=enc["attention_mask"].sum(1)-1
    return out[torch.arange(len(cs),device=device),pos,:]


def tids(tok,cs,llama_like,device):
    return zsre.official_target_ids(tok,[c.target_text for c in cs],llama_like=llama_like,device=device)


@torch.no_grad()
def flags(model,tok,cs,llama_like,device,n):
    ans=[]
    for st in range(0,len(cs),n):
        b=cs[st:st+n]; z=logits(model,tok,b,device); t=tids(tok,b,llama_like,device)
        ans += (z.argmax(-1)==t).cpu().tolist()
    return [bool(x) for x in ans]


class Sampler:
    def __init__(self,cs,n,seed): self.cs=list(cs); self.n=min(n,len(cs)); self.r=random.Random(seed); self.o=[]; self.i=0
    def next(self):
        b=[]
        while len(b)<self.n:
            if self.i>=len(self.o): self.o=list(range(len(self.cs))); self.r.shuffle(self.o); self.i=0
            k=min(self.n-len(b),len(self.o)-self.i); b += [self.cs[j] for j in self.o[self.i:self.i+k]]; self.i+=k
        return b


def main():
    a=args(); gagd.set_seed(a.seed)
    vp,mp=Path(a.training_visible_path).resolve(),Path(a.split_manifest).resolve()
    records=s1.load_locked(vp,mp,a.seed,a.forget_num)
    ns=argparse.Namespace(model_path=a.model_path,dtype=a.dtype,device_map=a.device_map,gradient_checkpointing=False)
    model,tok=gagd.load_model_and_tokenizer(ns,for_training=False)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    out=sparse.untie(model)
    for p in model.parameters(): p.requires_grad_(False)
    device=gagd.first_device(model); llama_like=zsre.is_llama_like(model,tok)
    all_cs=cases(records,tok,llama_like); before=flags(model,tok,all_cs,llama_like,device,a.batch_size)
    active=[c for c,ok in zip(all_cs,before) if ok]; before_n=sum(before)
    root=gagd.resolve_output_path(a.output_dir); ckpt=root/"checkpoint"; root.mkdir(parents=True,exist_ok=True)
    if not active:
        zsre_sure.save_checkpoint(model,tok,ckpt); write(root/"repair_summary.json",{"method":"SURE-ZsRE-no-neutral-sensitive-row","active_before":0,"selected_scale":0.0,"Unknown_used":False,"target_new_seen":False,"checkpoint":str(ckpt.resolve())}); return

    row_ids=sorted(set(int(x) for x in tids(tok,active,llama_like,device).cpu().tolist()))
    original=out.weight.index_select(0,torch.tensor(row_ids,device=out.weight.device)).detach().clone()
    delta=nn.Parameter(torch.zeros((len(row_ids),out.weight.shape[1]),device=device,dtype=torch.float32))
    opt=torch.optim.AdamW([delta],lr=a.repair_lr,weight_decay=0.0); it=Sampler(active,a.batch_size,a.seed+100003)
    best=(before_n,0,delta.detach().clone()); logs=[]; h=sparse.output_hook(out,row_ids,delta)
    try:
        for step in range(1,a.repair_steps+1):
            b=it.next(); opt.zero_grad(set_to_none=True); z=logits(model,tok,b,device); t=tids(tok,b,llama_like,device)
            loss=sparse.loss_fn(z,t,a.repair_margin)+a.repair_l2*delta.pow(2).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_([delta],1.0); opt.step()
            if step==1 or step%a.check_every==0 or step==a.repair_steps:
                c=sum(flags(model,tok,all_cs,llama_like,device,a.batch_size)); row={"step":step,"correct":c,"loss":float(loss.detach()),"delta_norm":float(delta.detach().norm())}; logs.append(row); print("ZsRE sensitive-row repair",row)
                if c<best[0]: best=(c,step,delta.detach().clone())
                if c==0: best=(0,step,delta.detach().clone()); break
    finally: h.remove()
    delta=best[2]

    reports=[]; zero=[]; chosen=None
    ids=torch.tensor(row_ids,device=out.weight.device)
    for scale in sparse.parse_scales(a.candidate_scales):
        with torch.no_grad(): out.weight.index_copy_(0,ids,original.to(out.weight.dtype)+scale*delta.to(out.weight.dtype))
        c=sum(flags(model,tok,all_cs,llama_like,device,a.batch_size)); reports.append({"scale":scale,"correct":c,"effective_delta_norm":float(delta.norm().cpu()*scale)})
        if c==0: zero.append(scale)
        key=(c,scale)
        if chosen is None or key<chosen[0]: chosen=(key,scale)
    scale=min(zero) if zero else chosen[1]
    with torch.no_grad(): out.weight.index_copy_(0,ids,original.to(out.weight.dtype)+scale*delta.to(out.weight.dtype))
    final_n=sum(flags(model,tok,all_cs,llama_like,device,a.batch_size))
    zsre_sure.save_checkpoint(model,tok,ckpt)
    write(root/"scale_sweep_direct_only.json",reports)
    write(root/"repair_summary.json",{
        "method":"SURE-ZsRE-no-neutral-sensitive-row","protocol":PROTOCOL,"seed":a.seed,"active_before":before_n,"active_after":final_n,
        "selected_lm_head_rows":len(row_ids),"selected_token_ids":row_ids,"best_step":best[1],"selected_scale":scale,"full_delta_norm":float(delta.norm().cpu()),
        "loss":"relu(sensitive_logit-stopgrad(best_other_logit)+margin)","selection_scope":"direct requested_rewrite only","selection_uses_heldout":False,
        "transformer_trainable":0,"input_embeddings_modified":False,"Unknown_used":False,"IDK_used":False,"target_new_seen":False,"replacement_target_used":False,
        "retain_seen":0,"rephrases_seen":0,"locality_seen":0,"PPL_seen":False,"logs":logs,"checkpoint":str(ckpt.resolve())})
    print(f"visible rewrite correct tokens: {before_n} -> {final_n}; selected scale={scale:g}; sensitive rows={len(row_ids)}")

if __name__=="__main__": main()
