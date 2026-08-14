#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random
from dataclasses import asdict
from pathlib import Path
import torch
from tqdm import tqdm
import gagd_compare as gagd
import mquake_forget_only_no_neutral as locked
import mquake_zero_unlearn_official_eval as mq
from zsre_no_neutral_stage1_gagd import ga_sensitive_logprob,gd_non_sensitive_kl
from zsre_no_neutral_stage1_emb_lm import restore_sensitive_rows_only

PROTOCOL="mquake_zerounlearn_forget_only_locked_no_neutral"
METHOD="SURE-MQuAKE-no-neutral-EmbLM-GAGD-Stage1"

def args():
 p=argparse.ArgumentParser()
 p.add_argument("--model-path",required=True);p.add_argument("--training-visible-path",required=True);p.add_argument("--split-manifest",required=True);p.add_argument("--output-dir",required=True)
 p.add_argument("--seed",type=int,required=True);p.add_argument("--forget-num",type=int,default=50);p.add_argument("--steps",type=int,default=600);p.add_argument("--batch-size",type=int,default=1);p.add_argument("--cache-batch-size",type=int,default=8)
 p.add_argument("--emb-lm-lr",type=float,default=1e-4);p.add_argument("--ga-weight",type=float,default=2.0);p.add_argument("--gd-weight",type=float,default=1.0);p.add_argument("--grad-clip",type=float,default=1.0)
 p.add_argument("--dtype",choices=("bf16","fp16","fp32"),default="bf16");p.add_argument("--device-map",choices=("single","auto"),default="single")
 return p.parse_args()

def write(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2)+"\n")

class Sampler:
 def __init__(self,n,b,seed): self.n=n;self.b=min(b,n);self.r=random.Random(seed);self.o=[];self.i=0
 def next(self):
  z=[]
  while len(z)<self.b:
   if self.i>=len(self.o): self.o=list(range(self.n));self.r.shuffle(self.o);self.i=0
   k=min(self.b-len(z),len(self.o)-self.i);z+=self.o[self.i:self.i+k];self.i+=k
  return z

@torch.no_grad()
def cache(model,tok,cs,device,batch):
 out=[];model.eval()
 for s in tqdm(range(0,len(cs),batch),desc="cache MQuAKE base logits"):
  out.append(locked.forward_logits(model,tok,cs[s:s+batch],device).detach().float().cpu())
 return torch.cat(out,0)

@torch.no_grad()
def correct(model,tok,cs,llama_like,device,batch):
 return sum(int(x["correct"]) for x in locked.predict(model,tok,cs,llama_like,device,batch))

def main():
 a=args();gagd.set_seed(a.seed)
 vp,mp=Path(a.training_visible_path).resolve(),Path(a.split_manifest).resolve()
 records,man=locked.load_locked(vp,mp,a.seed,a.forget_num)
 ns=argparse.Namespace(model_path=a.model_path,dtype=a.dtype,device_map=a.device_map,gradient_checkpointing=False)
 model,tok=gagd.load_model_and_tokenizer(ns,for_training=True)
 if tok.pad_token is None: tok.pad_token=tok.eos_token
 device=gagd.first_device(model);llama_like=mq.is_llama_like(model,tok);cs=locked.all_cases(records,tok,llama_like)
 base_logits=cache(model,tok,cs,device,a.cache_batch_size)
 summary,tied=gagd.configure_trainable(model,gagd.POST_TRAINING_RESTORE_MODE);params=gagd.unique_trainable_params(model);base_rows=gagd.snapshot_embedding_output_weights(tied)
 tids_all=locked.target_ids(tok,cs,llama_like,device);sensitive_ids=sorted(set(int(x) for x in tids_all.cpu().tolist()));before=correct(model,tok,cs,llama_like,device,a.cache_batch_size)
 opt=torch.optim.AdamW(params,lr=a.emb_lm_lr,weight_decay=0.0);sampler=Sampler(len(cs),a.batch_size,a.seed)
 root=gagd.resolve_output_path(a.output_dir);root.mkdir(parents=True,exist_ok=True)
 model.train()
 with (root/"train_log.jsonl").open("w") as f:
  for step in tqdm(range(1,a.steps+1),desc="MQuAKE EmbLM GA/GD"):
   ix=sampler.next();batch=[cs[i] for i in ix];opt.zero_grad(set_to_none=True)
   z=locked.forward_logits(model,tok,batch,device);t=locked.target_ids(tok,batch,llama_like,device);ga=ga_sensitive_logprob(z,t);gd=gd_non_sensitive_kl(z,base_logits[ix],t);loss=a.ga_weight*ga+a.gd_weight*gd
   if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at step {step}")
   loss.backward();gn=torch.nn.utils.clip_grad_norm_(params,a.grad_clip);opt.step()
   if step==1 or step%25==0 or step==a.steps:
    f.write(json.dumps({"step":step,"loss":float(loss.detach()),"ga_sensitive_logprob":float(ga.detach()),"gd_non_sensitive_kl":float(gd.detach()),"grad_norm":float(gn.detach()),"retain_seen":0,"atomic_questions_seen":0,"multihop_questions_seen":0,"PPL_seen":False,"target_new_seen":False,"Unknown_used":False,"IDK_used":False})+"\n");f.flush()
 del opt
 restore=restore_sensitive_rows_only(tied,base_rows,sensitive_ids);model.eval();after=correct(model,tok,cs,llama_like,device,a.cache_batch_size)
 ckpt=root/"checkpoint";ckpt.mkdir(parents=True,exist_ok=True);model.save_pretrained(ckpt);tok.save_pretrained(ckpt)
 cfg={"schema_version":1,"method":METHOD,"protocol":PROTOCOL,"seed":a.seed,"forget_instances":a.forget_num,"forget_atomic_facts":len(records),"direct_sensitive_token_cases":len(cs),"retain_seen":0,"atomic_questions_seen":0,"multihop_questions_seen":0,"PPL_seen":False,"target_new_seen":False,"Unknown_used":False,"IDK_used":False,"ga_loss":"mean(log p sensitive), minimized","gd_loss":"KL(base_non_sensitive || current_non_sensitive), sensitive token removed and renormalized","steps":a.steps,"batch_size":a.batch_size,"cache_batch_size":a.cache_batch_size,"emb_lm_lr":a.emb_lm_lr,"ga_weight":a.ga_weight,"gd_weight":a.gd_weight,"grad_clip":a.grad_clip,"optimizer":"adamw","trainable_parameter_summary":asdict(summary),"sensitive_rows":len(sensitive_ids),"correct_before":before,"correct_after_restore":after,"vocabulary_restoration":restore,"split_sampling":man.get("sampling"),"checkpoint":str(ckpt.resolve())}
 write(root/"config_used.json",cfg);write(root/"vocabulary_restoration.json",restore)
 print("MQuAKE Stage1",before,"->",after,"/",len(cs),"sensitive rows",len(sensitive_ids));print("checkpoint",ckpt)

if __name__=="__main__": main()
