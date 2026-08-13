#!/usr/bin/env python3
"""Strict forget-only MQuAKE repair with no Unknown/replacement target."""

from __future__ import annotations
import argparse, json, math, random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

import gagd_compare as gagd
import mquake_zero_unlearn_official_eval as mq

PROTOCOL = "mquake_zerounlearn_forget_only_locked_no_neutral"


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--stage1-steps", type=int, default=600)
    p.add_argument("--stage1-lr", type=float, default=5e-3)
    p.add_argument("--stage1-margin", type=float, default=0.25)
    p.add_argument("--stage2-steps", type=int, default=800)
    p.add_argument("--stage2-lr", type=float, default=5e-3)
    p.add_argument("--stage2-margin", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--candidate-scales", default="1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.03125,.015625,.0078125,0")
    p.add_argument("--dtype", choices=["bf16","fp16","fp32"], default="bf16")
    p.add_argument("--device-map", choices=["single","auto"], default="single")
    return p.parse_args()


def write(path: Path, x: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(x, indent=2, ensure_ascii=False) + "\n")


def load_locked(vp: Path, mp: Path, seed: int, forget_num: int):
    data = json.loads(vp.read_text())
    man = json.loads(mp.read_text())
    if man.get("protocol") != PROTOCOL or int(man.get("seed",-1)) != seed:
        raise ValueError("wrong no-neutral split/seed")
    if int(man["sampling"]["forget_num_instances"]) != forget_num:
        raise ValueError("wrong forget count")
    if len(data) != int(man["sampling"]["forget_atomic_fact_count"]):
        raise ValueError("wrong atomic fact count")
    for r in data:
        rr = r["requested_rewrite"]
        if "target_new" in rr:
            raise RuntimeError("target_new/Unknown leaked into no-neutral training")
        if not rr.get("target_true",{}).get("str"):
            raise RuntimeError("missing target_true")
        if r.get("paraphrase_prompts") or r.get("neighborhood_prompts"):
            raise RuntimeError("held-out probes leaked")
    return data, man


def untie(model):
    inp, out = model.get_input_embeddings(), model.get_output_embeddings()
    if inp.weight.data_ptr() != out.weight.data_ptr():
        model.config.tie_word_embeddings = False
        return out
    w = out.weight.detach()
    new = nn.Linear(w.shape[1], w.shape[0], bias=False, device=w.device, dtype=w.dtype)
    with torch.no_grad():
        new.weight.copy_(w)
    model.set_output_embeddings(new)
    model.config.tie_word_embeddings = False
    return new


def cases_for_record(r, tok, llama_like):
    rr = r["requested_rewrite"]
    subject = str(rr["subject"])
    prompt = str(rr["prompt"]).format(subject)
    tids = mq.original_answer_token_ids(tok, str(rr["target_true"]["str"]), llama_like=llama_like)
    out = []
    for i, tid in enumerate(tids):
        pref = tok.decode(tids[:i])
        p = prompt + ((" " + pref) if llama_like and i > 0 else pref)
        out.append(mq.PredictionCase(int(r["case_id"]), "rewrite", 0, i, p, tok.decode([tid])))
    return out


def all_cases(records, tok, llama_like):
    return [c for r in records for c in cases_for_record(r,tok,llama_like)]


def target_ids(tok, cs, llama_like, device):
    return mq.official_target_ids(tok, [c.target_text for c in cs], llama_like=llama_like, device=device)


def forward_logits(model, tok, cs, device):
    enc = tok([c.prompt for c in cs], padding=True, return_tensors="pt").to(device)
    logits = model(**enc, use_cache=False).logits
    pos = enc["attention_mask"].sum(1)-1
    return logits[torch.arange(len(cs),device=device), pos, :]


@torch.no_grad()
def predict(model, tok, cs, llama_like, device, batch_size):
    rows=[]
    for s in range(0,len(cs),batch_size):
        b=cs[s:s+batch_size]
        z=forward_logits(model,tok,b,device)
        pred=z.argmax(-1)
        tgt=target_ids(tok,b,llama_like,device)
        for c,p,t in zip(b,pred.cpu().tolist(),tgt.cpu().tolist()):
            rows.append({**asdict(c),"target_token_id":int(t),"predicted_token_id":int(p),"correct":bool(p==t)})
    return rows


def loss_fn(logits, tids, margin):
    idx=torch.arange(logits.size(0),device=logits.device)
    zt=logits[idx,tids]
    with torch.no_grad():
        x=logits.detach().clone()
        x[idx,tids]=-torch.inf
        other=x.max(-1).values
    return F.relu(zt.float()-other.float()+margin).mean()


def output_hook(layer, ids, delta):
    ids_t=torch.tensor(ids,dtype=torch.long,device=layer.weight.device)
    def hook(_m, inputs, output):
        h=inputs[0]
        extra=torch.matmul(h.float(),delta.float().T)
        idx=ids_t.view(*([1]*(extra.ndim-1)),-1).expand(*extra.shape[:-1],-1)
        return output + torch.zeros_like(output).scatter(-1,idx,extra.to(output.dtype))
    return layer.register_forward_hook(hook)


@torch.no_grad()
def materialize(weight, ids, delta, scale=1.0):
    ii=torch.tensor(ids,dtype=torch.long,device=weight.device)
    rows=weight.index_select(0,ii)+float(scale)*delta.to(weight.device,dtype=weight.dtype)
    weight.index_copy_(0,ii,rows)


def parse_scales(text):
    xs=sorted(set(float(x) for x in text.split(",") if x.strip()))
    if 0.0 not in xs: xs.append(0.0)
    if 1.0 not in xs: xs.append(1.0)
    return sorted(xs)


def train_delta(model,tok,cs,row_ids,llama_like,device,steps,lr,margin,l2,batch_size,seed,desc):
    delta=nn.Parameter(torch.zeros((len(row_ids),model.get_output_embeddings().weight.shape[1]),device=device,dtype=torch.float32))
    h=output_hook(model.get_output_embeddings(),row_ids,delta)
    opt=torch.optim.AdamW([delta],lr=lr,weight_decay=0.0)
    rng=random.Random(seed)
    order=list(range(len(cs))); cursor=len(order)
    model.eval()
    try:
        for _step in tqdm(range(steps),desc=desc):
            batch=[]
            while len(batch)<min(batch_size,len(cs)):
                if cursor>=len(order):
                    rng.shuffle(order); cursor=0
                take=min(min(batch_size,len(cs))-len(batch),len(order)-cursor)
                batch.extend(cs[i] for i in order[cursor:cursor+take]); cursor+=take
            opt.zero_grad(set_to_none=True)
            z=forward_logits(model,tok,batch,device)
            tids=target_ids(tok,batch,llama_like,device)
            loss=loss_fn(z,tids,margin)+l2*delta.pow(2).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_([delta],1.0); opt.step()
    finally:
        h.remove()
    return delta.detach()


def main():
    a=args(); gagd.set_seed(a.seed)
    vp,mp=Path(a.training_visible_path).resolve(),Path(a.split_manifest).resolve()
    records,man=load_locked(vp,mp,a.seed,a.forget_num)
    ns=argparse.Namespace(model_path=a.model_path,dtype=a.dtype,device_map=a.device_map,gradient_checkpointing=False)
    model,tok=gagd.load_model_and_tokenizer(ns,for_training=False)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    device=gagd.first_device(model); llama_like=mq.is_llama_like(model,tok)
    out=untie(model)
    for p in model.parameters(): p.requires_grad_(False)

    cs=all_cases(records,tok,llama_like)
    base_rows=predict(model,tok,cs,llama_like,device,a.eval_batch_size)
    base_summary=mq.summarize_atomic_split("base",records,base_rows)

    all_tids=target_ids(tok,cs,llama_like,device)
    row_ids=sorted(set(int(x) for x in all_tids.cpu().tolist()))
    d1=train_delta(model,tok,cs,row_ids,llama_like,device,a.stage1_steps,a.stage1_lr,a.stage1_margin,a.l2,a.batch_size,a.seed,"no-neutral Stage1")
    materialize(out.weight,row_ids,d1)
    s1_rows=predict(model,tok,cs,llama_like,device,a.eval_batch_size)
    s1_summary=mq.summarize_atomic_split("stage1",records,s1_rows)

    active_keys={(int(r["case_id"]),int(r["token_index"])) for r in s1_rows if r["correct"]}
    active=[c for c in cs if (c.case_id,c.token_index) in active_keys]
    scale_reports=[]; selected_scale=0.0; d2=torch.zeros((0,out.weight.shape[1]),device=device)
    repair_ids=[]
    if active:
        repair_tids=target_ids(tok,active,llama_like,device)
        repair_ids=sorted(set(int(x) for x in repair_tids.cpu().tolist()))
        d2=train_delta(model,tok,active,repair_ids,llama_like,device,a.stage2_steps,a.stage2_lr,a.stage2_margin,a.l2,a.batch_size,a.seed+100003,"no-neutral Stage2")
        best=None
        for scale in parse_scales(a.candidate_scales):
            temp=nn.Parameter(d2*scale,requires_grad=False)
            hh=output_hook(out,repair_ids,temp)
            rows=predict(model,tok,cs,llama_like,device,a.eval_batch_size)
            hh.remove()
            correct=sum(int(r["correct"]) for r in rows)
            eff=mq.summarize_atomic_split(f"scale_{scale}",records,rows)["Eff"]
            scale_reports.append({"scale":scale,"correct_sensitive_tokens":correct,"Eff":eff,"effective_delta_norm":float(d2.norm().cpu()*scale)})
            key=(correct,scale)
            if best is None or key<best:
                best=key; selected_scale=scale
        materialize(out.weight,repair_ids,d2,selected_scale)

    final_rows=predict(model,tok,cs,llama_like,device,a.eval_batch_size)
    final_summary=mq.summarize_atomic_split("final",records,final_rows)

    root=gagd.resolve_output_path(a.output_dir)
    ckpt=root/"checkpoint"; ckpt.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(ckpt); tok.save_pretrained(ckpt)
    write(root/"summary.json",{
        "schema_version":1,"method":"SURE-no-neutral-strict-forget-only","protocol":PROTOCOL,
        "seed":a.seed,"forget_instances":a.forget_num,"forget_atomic_facts":len(records),
        "target_new_seen":False,"literal_unknown_used":False,"benchmark_retain_seen":0,
        "ppl_or_external_utility_seen_during_training_or_selection":False,
        "trainable_parameters":"sparse sensitive LM-head row deltas only; input embeddings frozen",
        "loss":"relu(sensitive_logit-stopgrad(best_other_logit)+margin)",
        "base":base_summary,"stage1":s1_summary,"stage2":final_summary,
        "stage1_rows":len(row_ids),"stage1_delta_norm":float(d1.norm().cpu()),
        "stage2_active_correct_tokens_before":len(active),"stage2_rows":len(repair_ids),
        "stage2_full_delta_norm":float(d2.norm().cpu()) if d2.numel() else 0.0,
        "stage2_selected_scale":selected_scale,"scale_reports":scale_reports,
        "final_checkpoint":str(ckpt.resolve()),"split_sampling":man.get("sampling"),
    })
    print("\n===== NO-NEUTRAL STRICT FORGET-ONLY =====")
    print("Base Eff:",base_summary["Eff"])
    print("Stage1 Eff:",s1_summary["Eff"])
    print("Stage2 Eff:",final_summary["Eff"])
    print("Stage2 active correct tokens:",len(active))
    print("Selected scale:",selected_scale)
    print("Unknown/target_new used: NO")
    print("Final checkpoint:",ckpt)


if __name__=="__main__":
    main()
