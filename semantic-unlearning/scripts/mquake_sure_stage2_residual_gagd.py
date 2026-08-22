#!/usr/bin/env python3
"""MQuAKE Level-2 residual GA/GD for Pure Two-Stage Directional SURE.

Input is the restored Level-1 checkpoint from sure_stage1_gagd.py.  This script
first gates *all* training-visible teacher-forced atomic target_true token
prompts.  If all pass, Level 2 is an identity.  Otherwise F is exactly the
failed prompts, A_F is derived only from their sensitive rows, and only A_F in
input embeddings / output head can update.  B_F receives sensitive GA; B_P is
the same residual context's non-sensitive distribution from the frozen Level-1
teacher and receives GD.  No rank/scale/nullspace/all-row fallback is allowed.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
from typing import Any, Dict

import torch

import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core


def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True)
    p.add_argument('--training-visible-path',required=True)
    p.add_argument('--split-manifest',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--forget-num',type=int,required=True)
    p.add_argument('--repair-steps',type=int,default=800)
    p.add_argument('--repair-lr',type=float,default=5e-4)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--cache-batch-size',type=int,default=8)
    p.add_argument('--check-every',type=int,default=25)
    p.add_argument('--lambda-f',type=float,default=2.0)
    p.add_argument('--lambda-p',type=float,default=1.0)
    p.add_argument('--constraint-margin',type=float,default=.05)
    p.add_argument('--max-protected-kl',type=float,default=.05)
    p.add_argument('--grad-clip',type=float,default=1.0)
    p.add_argument('--dtype',choices=('bf16','fp16','fp32'),default='bf16')
    p.add_argument('--device-map',choices=('single','auto'),default='single')
    p.add_argument('--skip-transformer-hash',action='store_true')
    return p.parse_args()


def load_locked(a):
    vp,mp=Path(a.training_visible_path).resolve(),Path(a.split_manifest).resolve()
    records=json.loads(vp.read_text()); manifest=json.loads(mp.read_text())
    if len(records)!=a.forget_num: raise RuntimeError('training-visible forget count mismatch')
    if int(manifest.get('seed',-1))!=a.seed: raise RuntimeError('split seed mismatch')
    s=manifest.get('sampling',{})
    if int(s.get('forget_num',-1))!=a.forget_num: raise RuntimeError('manifest forget count mismatch')
    if s.get('forget_case_ids') and [int(r['case_id']) for r in records]!=[int(x) for x in s['forget_case_ids']]:
        raise RuntimeError('training-visible IDs do not match manifest')
    for i,r in enumerate(records):
        rr=r.get('requested_rewrite',{})
        if not rr.get('target_true',{}).get('str'): raise RuntimeError(f'record {i} lacks target_true')
        if 'target_new' in rr or r.get('atomic_gen_prompt') or r.get('multihop_questions'):
            raise RuntimeError(f'record {i} leaked evaluation-only MQuAKE fields')
        if r.get('paraphrase_prompts') or r.get('neighborhood_prompts'):
            raise RuntimeError(f'record {i} leaked held-out probes')
    return records,manifest


def vocab_only(model):
    for p in model.parameters(): p.requires_grad_(False)
    e,o=model.get_input_embeddings(),model.get_output_embeddings()
    if e is None or o is None: raise RuntimeError('model lacks vocabulary matrices')
    e.weight.requires_grad_(True); o.weight.requires_grad_(True)
    ps=[]
    for p in (e.weight,o.weight):
        if all(id(p)!=id(q) for q in ps): ps.append(p)
    return e,o,ps


def hash_frozen(model,excluded):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if id(p) in excluded: continue
        t=p.detach().contiguous()
        h.update(n.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def gate(model,tok,cases,llama_like,device,batch,margin):
    residual=[]; vals=[]
    with torch.no_grad():
        for st in range(0,len(cases),batch):
            c=cases[st:st+batch]
            z=core.forward_last_logits(model,tok,c,device).float()
            y=core.official_target_ids(tok,c,llama_like=llama_like,device=device)
            r=torch.arange(z.shape[0],device=z.device); sens=z[r,y]
            other=z.clone(); other[r,y]=-torch.inf
            m=other.max(-1).values-sens
            for j,v in enumerate(m.cpu().tolist()):
                vals.append(float(v))
                if float(v)<margin: residual.append(st+j)
    return {'total':len(cases),'passed':len(cases)-len(residual),'failed':len(residual),
            'residual_indices':residual,'minimum_margin':min(vals) if vals else None,
            'required_margin':float(margin)}


def masked_rows(params,rows):
    if not rows: raise RuntimeError('A_F is empty for non-empty F')
    hs=[]
    for p in params:
        mask=torch.zeros((p.shape[0],1),dtype=p.dtype,device=p.device)
        idx=torch.tensor(sorted(set(rows)),dtype=torch.long,device=p.device); mask.index_fill_(0,idx,1)
        hs.append(p.register_hook(lambda g,m=mask:g*m))
    return hs


def protected_kl(model,tok,cases,teacher,idx,llama_like,device,batch):
    if not idx:return 0.0
    total=0.0; n=0
    with torch.no_grad():
        for st in range(0,len(idx),batch):
            ids=idx[st:st+batch]; c=[cases[i] for i in ids]
            z=core.forward_last_logits(model,tok,c,device)
            y=core.official_target_ids(tok,c,llama_like=llama_like,device=device)
            v=core.gd_non_sensitive_kl(z,teacher[ids],y)
            total+=float(v.cpu())*len(ids); n+=len(ids)
    return total/max(n,1)


def main():
    a=args()
    if min(a.repair_steps,a.batch_size,a.cache_batch_size,a.check_every)<=0: raise ValueError('steps/batches must be positive')
    if a.repair_lr<=0 or a.lambda_f<=0 or a.lambda_p<0: raise ValueError('invalid repair hyperparameters')
    gagd.set_seed(a.seed)
    if a.device_map=='single': gagd.require_cuda_if_needed(a.device_map)
    records,manifest=load_locked(a)
    ns=argparse.Namespace(model_path=a.model_path,dtype=a.dtype,device_map=a.device_map,gradient_checkpointing=False)
    model,tok=gagd.load_model_and_tokenizer(ns,for_training=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side='right'; device=gagd.first_device(model); llama_like=is_llama_like(model,tok)
    cases=core.expand_sensitive_cases(records,tok,dataset='zsre',llama_like=llama_like)
    if not cases: raise RuntimeError('no generated atomic target_true token prompts')
    out=gagd.resolve_output_path(a.output_dir); out.mkdir(parents=True,exist_ok=True); ckpt=out/'checkpoint'

    e,o,params=vocab_only(model); excluded={id(e.weight),id(o.weight)}
    before_hash=None if a.skip_transformer_hash else hash_frozen(model,excluded)
    level1_gate=gate(model,tok,cases,llama_like,device,a.cache_batch_size,a.constraint_margin)
    F=[int(i) for i in level1_gate['residual_indices']]
    report:Dict[str,Any]={'schema_version':1,'method':'MQuAKE Pure Two-Stage Directional SURE / Level2',
        'source_protocol':manifest.get('protocol'),'level1_gate':level1_gate,
        'generated_atomic_prompt_semantics':'all teacher-forced target_true token cases from training-visible direct facts',
        'official_atomicgen_seen':0,'benchmark_retain_seen':0,'target_new_seen':False,
        'forbidden_mechanics':['rank sweep','scale sweep','nullspace repair','all-sensitive-row broadening']}

    teacher=None; A_F=[]
    if not F:
        report['level2']={'skipped':True,'reason':'Level 1 passed all generated atomic prompts','F':0,'A_F':[]}
    else:
        teacher=core.cache_base_logits(model,tok,cases,device,batch_size=a.cache_batch_size)
        residual_cases=[cases[i] for i in F]
        tids=core.official_target_ids(tok,residual_cases,llama_like=llama_like,device=device)
        A_F=sorted(set(int(x) for x in tids.cpu().tolist())-set(gagd.special_token_ids(tok)))
        hs=masked_rows(params,A_F); opt=torch.optim.AdamW(params,lr=a.repair_lr,weight_decay=0.0)
        sampler=core.IndexSampler(len(F),a.batch_size,a.seed+100003); logs=[]
        model.train()
        try:
            for step in range(1,a.repair_steps+1):
                local=sampler.next(); ids=[F[i] for i in local]; c=[cases[i] for i in ids]
                opt.zero_grad(set_to_none=True)
                z=core.forward_last_logits(model,tok,c,device)
                y=core.official_target_ids(tok,c,llama_like=llama_like,device=device)
                ga=core.ga_sensitive_logprob(z,y)
                gd=core.gd_non_sensitive_kl(z,teacher[ids],y)
                loss=a.lambda_f*ga+a.lambda_p*gd
                if not torch.isfinite(loss): raise FloatingPointError(f'non-finite Level2 loss at {step}')
                loss.backward(); gn=torch.nn.utils.clip_grad_norm_(params,a.grad_clip) if a.grad_clip>0 else None; opt.step()
                if step==1 or step%a.check_every==0 or step==a.repair_steps:
                    model.eval(); g=gate(model,tok,cases,llama_like,device,a.cache_batch_size,a.constraint_margin)
                    logs.append({'step':step,'loss':float(loss.detach().cpu()),'ga':float(ga.detach().cpu()),
                                 'gd':float(gd.detach().cpu()),'failed_all_atomic':g['failed'],
                                 'grad_norm':None if gn is None else float(gn.detach().cpu())})
                    if g['failed']==0: break
                    model.train()
        finally:
            for h in hs:h.remove()
            del opt
            model.eval()
        report['level2']={'skipped':False,'F':len(F),'F_indices':F,'A_F':A_F,'A_F_count':len(A_F),
            'B_F':'all failed Level-1 atomic prompts (sensitive GA)',
            'B_P':'same residual contexts, frozen Level-1 non-sensitive distribution (GD)',
            'updated_parameters':'ONLY embedding/output vocabulary rows in A_F','logs':logs}

    final_gate=gate(model,tok,cases,llama_like,device,a.cache_batch_size,a.constraint_margin)
    pkl=protected_kl(model,tok,cases,teacher,F,llama_like,device,a.cache_batch_size) if teacher is not None else 0.0
    after_hash=None if a.skip_transformer_hash else hash_frozen(model,excluded)
    exact=None if a.skip_transformer_hash else before_hash==after_hash
    if exact is False: raise AssertionError('frozen non-vocabulary parameters changed')
    gates={'all_generated_atomic_prompts_pass':final_gate['failed']==0,
           'protected_non_sensitive_kl':pkl,'protected_kl_threshold':a.max_protected_kl,
           'protected_regression_bounded':pkl<=a.max_protected_kl,
           'transformer_exactly_unchanged':exact}
    gates['all_required_gates_pass']=bool(gates['all_generated_atomic_prompts_pass'] and gates['protected_regression_bounded'] and exact is not False)
    report.update({'final_gate':final_gate,'final_gates':gates,'transformer_hash_before':before_hash,'transformer_hash_after':after_hash})
    ckpt.mkdir(parents=True,exist_ok=True); model.save_pretrained(ckpt); tok.save_pretrained(ckpt)
    (out/'two_stage_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f"Level1 gate: {level1_gate['passed']}/{level1_gate['total']} pass; F={len(F)}")
    print('Level2:', 'SKIPPED' if not F else f'F={len(F)}, A_F={len(A_F)}')
    print(f"Final gate: {final_gate['passed']}/{final_gate['total']} pass; protected_KL={pkl:.6g}; transformer_exact={exact}")
    print('Final gates pass:',gates['all_required_gates_pass']); print('Checkpoint:',ckpt)

if __name__=='__main__': main()
