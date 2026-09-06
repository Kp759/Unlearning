#!/usr/bin/env python3
"""Recognition-only TARGET_ENTITY linear relation classifier for Fix5 (Seed 1)."""
from __future__ import annotations
import argparse, json, random, re, sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import mcf_subject_relation_views_v2_seed1 as old

SEED=1; NONE="NONE"; TARGET="TARGET_ENTITY"; OTHER="OTHER_ENTITY"; MAX_LENGTH=256
STEPS=1600; BATCH=128; LR=5e-3; WD=1e-4

@dataclass(frozen=True)
class Row:
    text:str; subject:str; relation:str; forbidden:bool; kind:str; family:str; case_id:int|None=None
    masked:str=""; candidate:bool=False


def norm(x:str)->str: return " ".join(str(x).split())

def relation_label(record:Mapping[str,Any], modeled:set[str])->str:
    r=str(old.rr(record)["relation_id"]); return r if r in modeled else NONE

def mask_target(text:str, subject:str, bank_subjects:Sequence[str])->tuple[str,bool]:
    p=old.subject_regex(subject)
    if not p.search(text): return norm(text),False
    out=p.sub(TARGET,text,count=1)
    for other in sorted({s for s in bank_subjects if s.casefold()!=subject.casefold()},key=len,reverse=True):
        out=old.subject_regex(other).sub(OTHER,out)
    return norm(out),True

def prep(rows:Sequence[Row],bank_subjects:Sequence[str])->list[Row]:
    bank_cf={s.casefold() for s in bank_subjects}; out=[]
    for r in rows:
        m,p=mask_target(r.text,r.subject,bank_subjects)
        if not p: raise RuntimeError(f"target subject missing: {r.subject!r} in {r.text!r}")
        out.append(Row(r.text,r.subject,r.relation,r.forbidden,r.kind,r.family,r.case_id,m,r.subject.casefold() in bank_cf))
    return out

def phase_views(info,split,phase): return old.phase_templates(info,split,phase)

def rows_for_phase(facts,split,retain,forget,phase)->list[Row]:
    modeled={str(v["relation_id"]) for v in facts.values()}; bank={(str(v["subject"]),str(v["relation_id"])) for v in facts.values()}; ids=sorted(facts); out=[]
    for i in ids:
        f=facts[i]; s=str(f["subject"]); rid=str(f["relation_id"])
        fams=old.phase_families(split,phase)
        for fam,t in zip(fams,phase_views(f,split,phase)):
            out.append(Row(str(t).format(s),s,rid,True,f"fix5_{phase}",fam,int(f["case_id"])))
    for rec in retain:
        rr=old.rr(rec); s=str(rr["subject"])
        out.append(Row(old.render(rec),s,relation_label(rec,modeled),False,old.retain_kind(rec,facts,forget),f"retain_{phase}",int(rec["case_id"])))
    for i in ids:
        a=facts[i]; choices=[j for j in ids if j!=i and str(facts[j]["relation_id"])!=str(a["relation_id"]) and (str(a["subject"]),str(facts[j]["relation_id"])) not in bank]
        if choices:
            start=old.stable_int(f"ssdr-linear:{phase}:{i}")%len(choices)
            fams=old.phase_families(split,phase)
            for k in range(2):
                b=facts[choices[(start+k)%len(choices)]]; fam=fams[k%len(fams)]; t=str(b["views"][fam])
                out.append(Row(t.format(str(a["subject"])),str(a["subject"]),str(b["relation_id"]),False,"same_subject_different_relation",fam,int(a["case_id"])))
    for i in ids:
        a=facts[i]; rchoices=[j for j in ids if j!=i and (str(a["subject"]),str(facts[j]["relation_id"])) not in bank]; schoices=[j for j in ids if j!=i and (str(facts[j]["subject"]),str(a["relation_id"])) not in bank]
        if not rchoices or not schoices: continue
        j=rchoices[old.stable_int(f"cross-r-linear:{phase}:{i}")%len(rchoices)]; k=schoices[old.stable_int(f"cross-s-linear:{phase}:{i}")%len(schoices)]
        br,bs=facts[j],facts[k]; fams=old.phase_families(split,phase); f1,f2=fams[0],fams[-1]
        text=f"First: {br['views'][f1].format(str(a['subject']))} Second: {a['views'][f2].format(str(bs['subject']))}"
        out.append(Row(text,str(a["subject"]),str(br["relation_id"]),False,"crossed_binding",f1,int(a["case_id"])))
        out.append(Row(text,str(bs["subject"]),str(a["relation_id"]),False,"crossed_binding",f2,int(a["case_id"])))
    mention={"fit":'Spell the entity name "{s}" exactly.',"calib":'Quote only the entity name "{s}" without stating an attribute.',"validation":'Repeat the name "{s}" and do not give facts about it.'}[phase]
    for f in facts.values(): out.append(Row(mention.format(s=str(f["subject"])),str(f["subject"]),NONE,False,"permitted_subject_mention","none_control",int(f["case_id"])))
    return out

def bucket(kind:str)->str:
    for x in ("same_subject_different_relation","same_relation_different_subject","same_answer_different_fact"):
        if x in kind:return x
    if kind in ("crossed_binding","permitted_subject_mention"):return kind
    return "other_retain"

def separate(parts:Mapping[str,Sequence[Row]])->tuple[dict[str,list[Row]],dict[str,Any]]:
    owner={}; out={k:[] for k in ("fit","calib","validation")}; dropped=Counter()
    for ph in ("fit","calib","validation"):
        local={}
        for r in parts[ph]:
            key=r.masked.casefold(); prev=local.get(key)
            if prev is not None and prev!=r.relation: raise RuntimeError(f"masked label conflict in {ph}: {r.masked!r}")
            local[key]=r.relation
            if key in owner:
                oph,olab=owner[key]
                if olab!=r.relation: raise RuntimeError(f"cross-partition masked label conflict: {r.masked!r}")
                if oph!=ph: dropped[ph]+=1; continue
            else: owner[key]=(ph,r.relation)
            out[ph].append(r)
    return out,{"masked_overlap_dropped":dict(dropped),"unique_masked":{k:len({r.masked.casefold() for r in v}) for k,v in out.items()}}

def dedup_sem(rows:Sequence[Row])->list[Row]:
    d={}
    for r in rows:
        k=r.masked.casefold()
        if k in d and d[k].relation!=r.relation: raise RuntimeError(f"conflicting labels: {r.masked!r}")
        d.setdefault(k,r)
    return list(d.values())

def dedup_policy(rows:Sequence[Row])->list[Row]:
    d={}
    for r in rows: d.setdefault((r.masked.casefold(),r.relation,r.forbidden,bucket(r.kind),r.candidate),r)
    return list(d.values())

@torch.no_grad()
def encode(model,tok,texts:Sequence[str],device,batch:int)->torch.Tensor:
    bb=getattr(model,"model",None)
    if bb is None: raise RuntimeError("requires model.model")
    chunks=[]; oldside=tok.padding_side; tok.padding_side="right"
    try:
        for st in range(0,len(texts),batch):
            e=tok(list(texts[st:st+batch]),padding=True,truncation=True,max_length=MAX_LENGTH,return_tensors="pt").to(device)
            h=bb(**e,use_cache=False,return_dict=True).last_hidden_state.float(); m=e["attention_mask"].to(h.dtype).unsqueeze(-1)
            chunks.append(((h*m).sum(1)/m.sum(1).clamp_min(1)).cpu())
    finally: tok.padding_side=oldside
    return torch.cat(chunks)

class Linear(nn.Module):
    def __init__(self,d,k): super().__init__(); self.fc=nn.Linear(d,k)
    def forward(self,x): return self.fc(x.float())

def train(x,y,k,device,steps,batch,lr,wd):
    counts=torch.bincount(y,minlength=k).float()
    if bool((counts==0).any()): raise RuntimeError(f"zero fit class: {torch.where(counts==0)[0].tolist()}")
    w=(len(y)/(k*counts)); w=(w/w.mean()).to(device); clf=Linear(x.shape[1],k).to(device); opt=torch.optim.AdamW(clf.parameters(),lr=lr,weight_decay=wd); g=torch.Generator().manual_seed(SEED+9501); trace=[]; xd=x.to(device); yd=y.to(device)
    for s in range(1,steps+1):
        ii=torch.randint(0,len(y),(min(batch,len(y)),),generator=g).to(device); loss=F.cross_entropy(clf(xd[ii]),yd[ii],weight=w); opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if s==1 or s%100==0 or s==steps: trace.append({"step":s,"loss":float(loss.item())})
    clf.eval(); [p.requires_grad_(False) for p in clf.parameters()]
    return clf,{"steps":steps,"examples":len(y),"class_weights":[float(v) for v in w.cpu()],"loss_trace":trace}

def score(clf,rows,base,tok,device,batch):
    texts=[];idx={};inv=[]
    for r in rows:
        if r.masked not in idx: idx[r.masked]=len(texts);texts.append(r.masked)
        inv.append(idx[r.masked])
    with torch.no_grad(): u=clf(encode(base,tok,texts,device,batch).to(device)).cpu()
    return u[torch.tensor(inv)]

def margin(logits):
    v,i=torch.topk(logits.float(),2,dim=1); return i[:,0],v[:,0]-v[:,1]

def sem_report(rows,logits,c2i,classes):
    y=torch.tensor([c2i[r.relation] for r in rows]); p=logits.argmax(1); ok=p.eq(y); byrel={};byfam={}
    for c in classes:
        ids=[i for i,r in enumerate(rows) if r.relation==c]
        if ids: byrel[c]={"n":len(ids),"accuracy_pct":100*float(ok[ids].float().mean())}
    for f in sorted({r.family for r in rows}):
        ids=[i for i,r in enumerate(rows) if r.family==f]; byfam[f]={"n":len(ids),"accuracy_pct":100*float(ok[ids].float().mean())}
    return {"n":len(rows),"accuracy_pct":100*float(ok.float().mean()),"by_relation":byrel,"by_family":byfam}

def policy(rows,logits,eta,classes,none_idx,bank):
    pred,dm=margin(logits); labels=[classes[int(i)] for i in pred]; relacc=(pred!=none_idx)&(dm>=eta); bind=torch.tensor([(r.subject,labels[i]) in bank for i,r in enumerate(rows)]); act=relacc&bind; correct=torch.tensor([labels[i]==r.relation for i,r in enumerate(rows)]); forb=torch.tensor([r.forbidden for r in rows]); cand=torch.tensor([r.candidate for r in rows]); pos=torch.where(forb)[0];neg=torch.where(~forb)[0]
    rate=lambda x: float(x.float().mean()) if len(x) else 0.0
    fam={}
    for f in sorted({bucket(r.kind) for r in rows if not r.forbidden}):
        ids=torch.tensor([i for i,r in enumerate(rows) if not r.forbidden and bucket(r.kind)==f]); fam[f]={"n":len(ids),"candidate_present_n":int(cand[ids].sum()),"false_activation_pct":100*rate(act[ids])}
    return {"n":len(rows),"relation_accuracy_pct":100*rate(correct),"forbidden_n":int(forb.sum()),"permitted_n":int((~forb).sum()),"correct_forbidden_binding_accept_pct":100*rate((act&correct&forb)[pos]) if len(pos) else None,"wrong_forbidden_binding_accept_pct":100*rate((act&~correct&forb)[pos]) if len(pos) else None,"permitted_false_activation_pct":100*rate(act[neg]) if len(neg) else None,"candidate_present_permitted_n":int(((~forb)&cand).sum()),"candidate_present_permitted_false_activation_pct":100*rate(act[(~forb)&cand]) if bool(((~forb)&cand).any()) else None,"permitted_negative_families":fam}

def calibrate(rows,logits,classes,none_idx,bank,eps,epswrong,minacc):
    _,dm=margin(logits); cand=sorted(set([0.0]+[float(x) for x in dm.tolist()]));cand.append(max(cand)+1.0);best=None
    for eta in cand:
        r=policy(rows,logits,eta,classes,none_idx,bank); wrong=(r["wrong_forbidden_binding_accept_pct"] or 0)/100; overall=(r["permitted_false_activation_pct"] or 0)/100; cp=(r["candidate_present_permitted_false_activation_pct"] or 0)/100; fr={k:(v["false_activation_pct"] or 0)/100 for k,v in r["permitted_negative_families"].items()}
        if wrong>epswrong+1e-12 or overall>eps+1e-12 or cp>eps+1e-12 or any(v>eps+1e-12 for v in fr.values()): continue
        ca=(r["correct_forbidden_binding_accept_pct"] or 0)/100; key=(-ca,max(fr.values(),default=0),cp,overall,wrong,-eta); item={"eta":eta,"correct_forbidden_binding_accept_rate":ca,"wrong_forbidden_binding_accept_rate":wrong,"permitted_false_activation_rate":overall,"candidate_present_permitted_false_activation_rate":cp,"negative_family_false_activation_rates":fr}
        if best is None or key<best[0]:best=(key,item)
    if best is None: raise RuntimeError("no preservation-feasible margin threshold")
    x=best[1];x["status"]="ACCEPTABLE_OPERATING_POINT" if x["correct_forbidden_binding_accept_rate"]>=minacc else "NO_ACCEPTABLE_OPERATING_POINT";x["selection_rule"]="maximize correct forbidden-binding acceptance under wrong/overall/candidate-present/per-family budgets";return x["eta"],x

def dev_rows(forget):
    d=[];p=[]
    for rec in forget:
        r=old.rr(rec);s=str(r["subject"]);rid=str(r["relation_id"]);cid=int(rec["case_id"]);d.append(Row(old.render(rec),s,rid,True,"official_direct_development","official_direct",cid))
        for t in rec.get("paraphrase_prompts",[]):p.append(Row(str(t),s,rid,True,"official_paraphrase_development","official_paraphrase",cid))
    return d,p

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--model-path",required=True);ap.add_argument("--mcf-path",required=True);ap.add_argument("--view-corpus-fix5",required=True);ap.add_argument("--output-dir",required=True);ap.add_argument("--dtype",choices=("bf16","fp16","fp32"),default="bf16");ap.add_argument("--device",default="cuda");ap.add_argument("--encode-batch-size",type=int,default=16);ap.add_argument("--train-steps",type=int,default=STEPS);ap.add_argument("--train-batch-size",type=int,default=BATCH);ap.add_argument("--lr",type=float,default=LR);ap.add_argument("--weight-decay",type=float,default=WD);ap.add_argument("--epsilon-retain",type=float,default=.02);ap.add_argument("--epsilon-wrong",type=float,default=.02);ap.add_argument("--min-calib-correct-accept",type=float,default=.60);ap.add_argument("--min-validation-relation-accuracy",type=float,default=.70);a=ap.parse_args()
    out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);device=torch.device(a.device)
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import mcf_zero_unlearn_official_eval as off
    from mcf_sampling import sample_official_mcf_records
    data=json.loads(Path(a.mcf_path).read_text());forget,retain=sample_official_mcf_records(data,50,1000,SEED,strict=True);forget=[off.normalize_record(x) for x in forget];retain=[off.normalize_record(x) for x in retain]
    facts,split,corpus=old.load_v2(Path(a.view_corpus_fix5));old.align_facts_to_forget(facts,forget);bank={(str(v["subject"]),str(v["relation_id"])) for v in facts.values()};subjects=sorted({s for s,_ in bank},key=len,reverse=True);relations=sorted({r for _,r in bank});classes=relations+[NONE];c2i={c:i for i,c in enumerate(classes)};none_idx=c2i[NONE]
    rf,rc,rv=old.split_retain(retain,bank);parts={"fit":rows_for_phase(facts,split,rf,forget,"fit"),"calib":rows_for_phase(facts,split,rc,forget,"calib"),"validation":rows_for_phase(facts,split,rv,forget,"validation")};parts={k:prep(v,subjects) for k,v in parts.items()};parts,sep=separate(parts);sem={k:dedup_sem(v) for k,v in parts.items()};pol={k:dedup_policy(v) for k,v in parts.items()};missing=sorted(set(classes)-{r.relation for r in sem["fit"]});
    if missing: raise RuntimeError(f"fit classes missing after masking/dedup: {missing}")
    tok=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token;base=AutoModelForCausalLM.from_pretrained(a.model_path,dtype=old.dtype_from_name(a.dtype),local_files_only=True,low_cpu_mem_usage=True).to(device);base.eval();base.config.use_cache=False;[q.requires_grad_(False) for q in base.parameters()]
    x=encode(base,tok,[r.masked for r in sem["fit"]],device,a.encode_batch_size);y=torch.tensor([c2i[r.relation] for r in sem["fit"]]);clf,tr=train(x,y,len(classes),device,a.train_steps,a.train_batch_size,a.lr,a.weight_decay);slog={k:score(clf,sem[k],base,tok,device,a.encode_batch_size) for k in sem};plog={k:score(clf,pol[k],base,tok,device,a.encode_batch_size) for k in ("calib","validation")};eta,cal=calibrate(pol["calib"],plog["calib"],classes,none_idx,bank,a.epsilon_retain,a.epsilon_wrong,a.min_calib_correct_accept);srep={k:sem_report(sem[k],slog[k],c2i,classes) for k in sem};vp=policy(pol["validation"],plog["validation"],eta,classes,none_idx,bank);famok=all((v["false_activation_pct"] or 0)<=2+1e-9 for v in vp["permitted_negative_families"].values());pilot=cal["status"]=="ACCEPTABLE_OPERATING_POINT" and (srep["validation"]["accuracy_pct"] or 0)>=100*a.min_validation_relation_accuracy and (vp["correct_forbidden_binding_accept_pct"] or 0)>=60 and (vp["wrong_forbidden_binding_accept_pct"] or 0)<=2+1e-9 and (vp["permitted_false_activation_pct"] or 0)<=2+1e-9 and (vp["candidate_present_permitted_false_activation_pct"] or 0)<=2+1e-9 and famok
    dr,pr=dev_rows(forget);dr=dedup_policy(prep(dr,subjects));pr=dedup_policy(prep(pr,subjects));dl=score(clf,dr,base,tok,device,a.encode_batch_size);ql=score(clf,pr,base,tok,device,a.encode_batch_size);fitkeys={r.masked.casefold() for r in sem["fit"]};dev={"direct":{"semantic":sem_report(dr,dl,c2i,classes),"policy":policy(dr,dl,eta,classes,none_idx,bank),"exact_fit_overlap_n":sum(r.masked.casefold() in fitkeys for r in dr)},"paraphrase":{"semantic":sem_report(pr,ql,c2i,classes),"policy":policy(pr,ql,eta,classes,none_idx,bank),"exact_fit_overlap_n":sum(r.masked.casefold() in fitkeys for r in pr)},"note":"Previously inspected Seed-1 probes; development evidence only; never used for fit/calibration/model selection."}
    res={"schema_version":1,"kind":"mcf_seed1_fix5_target_entity_linear_relation_classifier_recognition_only","recognition_only":True,"classes":classes,"relation_count":len(relations),"none_semantics":"NONE=no modeled relation about designated subject; permitted modeled relations keep their relation label","marking":{"target":TARGET,"other_registered_subjects":OTHER,"before_encoder":True},"encoder":"frozen Llama model.model; attention-mask mean pooling","classifier":"single affine linear layer; cross-entropy on logits","confidence":"top1 minus runner-up logit, NONE included","training":tr,"data_contract":{"corpus_protocol":corpus["protocol"],"family_split":split,"partition_mask_separation":sep,"semantic_unique_counts":{k:len(v) for k,v in sem.items()},"policy_unique_counts":{k:len(v) for k,v in pol.items()},"base_model_frozen":True,"trainable":"linear classifier only","no_output_correction":True,"no_quotient":True,"official_para_fit":False,"official_para_calibration":False,"official_para_model_selection":False},"calibration":cal,"semantic_relation_recognition":srep,"validation_policy":vp,"validation_pilot_pass":bool(pilot),"development_only_official_seed1":dev,"pilot_criteria":{"relation_accuracy_min_pct":100*a.min_validation_relation_accuracy,"correct_forbidden_accept_min_pct":60,"wrong_forbidden_accept_max_pct":2,"permitted_false_activation_max_pct":2,"candidate_present_permitted_false_activation_max_pct":2,"each_negative_family_max_pct":2}}
    (out/"target_relation_classifier_fix5_recognition.json").write_text(json.dumps(res,indent=2)+"\n");print(json.dumps({"calibration_status":cal["status"],"eta":eta,"validation_relation_accuracy_pct":srep["validation"]["accuracy_pct"],"validation_policy":vp,"pilot_pass":bool(pilot),"dev_official_para":dev["paraphrase"],"output_dir":str(out)},indent=2))
if __name__=="__main__":main()
