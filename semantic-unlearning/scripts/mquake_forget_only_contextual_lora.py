#!/usr/bin/env python3
"""Strict forget-only contextual LoRA for MQuAKE; no Unknown/target_new."""
import argparse, json, random
from pathlib import Path
import torch
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm

import gagd_compare as gagd
import mquake_zero_unlearn_official_eval as mq
import mquake_forget_only_no_neutral as nnr


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--model-path',required=True); p.add_argument('--training-visible-path',required=True)
    p.add_argument('--split-manifest',required=True); p.add_argument('--output-dir',required=True)
    p.add_argument('--seed',type=int,required=True); p.add_argument('--forget-num',type=int,default=50)
    p.add_argument('--rank',type=int,default=4); p.add_argument('--alpha',type=int,default=8)
    p.add_argument('--last-n-layers',type=int,default=2); p.add_argument('--steps',type=int,default=600)
    p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--margin',type=float,default=.25)
    p.add_argument('--active-steps',type=int,default=400); p.add_argument('--active-lr',type=float,default=5e-5)
    p.add_argument('--l2',type=float,default=1e-6); p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--eval-batch-size',type=int,default=8); p.add_argument('--check-every',type=int,default=25)
    p.add_argument('--target-eff-max',type=float,default=20.0,help='Choose the smallest LoRA scale whose forget Eff is at most this fixed threshold.')
    p.add_argument('--candidate-scales',default='1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.03125,.015625,.0078125,0')
    p.add_argument('--dtype',choices=['bf16','fp16','fp32'],default='bf16'); p.add_argument('--device-map',choices=['single','auto'],default='single')
    return p.parse_args()


def batches(cases,n,seed):
    rng=random.Random(seed); order=list(range(len(cases))); pos=len(order)
    while True:
        out=[]
        while len(out)<min(n,len(cases)):
            if pos>=len(order): rng.shuffle(order); pos=0
            take=min(min(n,len(cases))-len(out),len(order)-pos)
            out += [cases[i] for i in order[pos:pos+take]]; pos += take
        yield out


@torch.no_grad()
def status(model,tok,cases,llama_like,device,n):
    correct=0; margins=[]
    for s in range(0,len(cases),n):
        cs=cases[s:s+n]; z=nnr.forward_logits(model,tok,cs,device).float(); t=nnr.target_ids(tok,cs,llama_like,device)
        i=torch.arange(len(cs),device=device); sy=z[i,t]; x=z.clone(); x[i,t]=-torch.inf; other=x.max(-1).values
        correct += int((z.argmax(-1)==t).sum()); margins += (other-sy).cpu().tolist()
    return correct,(min(margins) if margins else None)


def train(model,tok,cases,llama_like,device,params,steps,lr,margin,l2,n,check,seed,label):
    if not cases: return {'steps':0,'reason':'no active cases'}
    opt=torch.optim.AdamW(params,lr=lr,weight_decay=0.0); it=batches(cases,n,seed); logs=[]; reason='max_steps'; done=0
    for step in tqdm(range(1,steps+1),desc=label):
        cs=next(it); opt.zero_grad(set_to_none=True); z=nnr.forward_logits(model,tok,cs,device); t=nnr.target_ids(tok,cs,llama_like,device)
        reg=torch.stack([p.pow(2).mean() for p in params]).mean(); loss=nnr.loss_fn(z,t,margin)+l2*reg
        loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); done=step
        if step==1 or step%check==0 or step==steps:
            cor,mm=status(model,tok,cases,llama_like,device,n); logs.append({'step':step,'correct':cor,'min_margin':mm})
            if mm is not None and mm>=margin: reason='all cases meet margin'; break
    return {'steps':done,'reason':reason,'logs':logs}


def lora_B_params(model):
    return {n:p for n,p in model.named_parameters() if 'lora_B' in n}


def main():
    a=parse_args(); gagd.set_seed(a.seed)
    if not 0.0 <= a.target_eff_max <= 100.0: raise ValueError('--target-eff-max must be in [0,100]')
    records,manifest=nnr.load_locked(Path(a.training_visible_path).resolve(),Path(a.split_manifest).resolve(),a.seed,a.forget_num)
    ns=argparse.Namespace(model_path=a.model_path,dtype=a.dtype,device_map=a.device_map,gradient_checkpointing=False)
    model,tok=gagd.load_model_and_tokenizer(ns,for_training=False); model.config.use_cache=False
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    device=gagd.first_device(model); llama_like=mq.is_llama_like(model,tok); cases=nnr.all_cases(records,tok,llama_like)
    base_rows=nnr.predict(model,tok,cases,llama_like,device,a.eval_batch_size); base_sum=mq.summarize_atomic_split('base',records,base_rows)
    nl=int(model.config.num_hidden_layers); layers=list(range(nl-a.last_n_layers,nl))
    cfg=LoraConfig(task_type=TaskType.CAUSAL_LM,r=a.rank,lora_alpha=a.alpha,lora_dropout=0.0,bias='none',target_modules=['o_proj','down_proj'],layers_to_transform=layers,layers_pattern='layers')
    model=get_peft_model(model,cfg); model.print_trainable_parameters(); print('targeted:',model.targeted_module_names)
    params=[p for p in model.parameters() if p.requires_grad]
    bad=[n for n,p in model.named_parameters() if p.requires_grad and 'lora_' not in n]
    if bad: raise RuntimeError(f'non-LoRA trainables: {bad}')
    phase1=train(model,tok,cases,llama_like,device,params,a.steps,a.lr,a.margin,a.l2,a.batch_size,a.check_every,a.seed,'contextual LoRA')
    rows1=nnr.predict(model,tok,cases,llama_like,device,a.eval_batch_size); sum1=mq.summarize_atomic_split('stage1',records,rows1)
    active={(r['case_id'],r['token_index']) for r in rows1 if r['correct']}; active_cases=[c for c in cases if (c.case_id,c.token_index) in active]
    phase2=train(model,tok,active_cases,llama_like,device,params,a.active_steps,a.active_lr,.05,a.l2,a.batch_size,a.check_every,a.seed+100003,'active contextual LoRA')
    Bs=lora_B_params(model); originals={n:p.detach().clone() for n,p in Bs.items()}; scales=sorted(set([float(x) for x in a.candidate_scales.split(',')]+[0.,1.])); reports=[]
    for sc in scales:
        with torch.no_grad():
            for n,p in Bs.items(): p.copy_(originals[n]*sc)
        rows=nnr.predict(model,tok,cases,llama_like,device,a.eval_batch_size); sm=mq.summarize_atomic_split(f'scale{sc}',records,rows); cor=sum(int(r['correct']) for r in rows)
        reports.append({'scale':sc,'Eff':sm['Eff'],'correct':cor,'meets_forget_target':float(sm['Eff'])<=a.target_eff_max})
    feasible=[r for r in reports if r['meets_forget_target']]
    if feasible:
        chosen=min(feasible,key=lambda r:(float(r['scale']),float(r['Eff']),int(r['correct'])))
        selection_reason='smallest_scale_meeting_fixed_forget_target'
    else:
        chosen=min(reports,key=lambda r:(float(r['Eff']),int(r['correct']),float(r['scale'])))
        selection_reason='no_scale_met_target_fallback_to_best_forgetting'
    selected=float(chosen['scale'])
    with torch.no_grad():
        for n,p in Bs.items(): p.copy_(originals[n]*selected)
    final_rows=nnr.predict(model,tok,cases,llama_like,device,a.eval_batch_size); final_sum=mq.summarize_atomic_split('selected',records,final_rows)
    merged=model.merge_and_unload(); out=gagd.resolve_output_path(a.output_dir); ckpt=out/'checkpoint'; ckpt.mkdir(parents=True,exist_ok=True); merged.save_pretrained(ckpt); tok.save_pretrained(ckpt)
    nnr.write(out/'summary.json',{'method':'SURE-contextual-LoRA-no-neutral-minchange','seed':a.seed,'target_new_seen':False,'Unknown_used':False,'retain_seen':0,'PPL_seen_during_selection':False,'target_eff_max':a.target_eff_max,'selection_rule':'minimum LoRA scale satisfying fixed forget-Eff threshold; tie-break lower Eff then fewer correct tokens','selection_reason':selection_reason,'layers':layers,'rank':a.rank,'alpha':a.alpha,'target_modules':['o_proj','down_proj'],'base':base_sum,'stage1':sum1,'phase1':phase1,'active_before_phase2':len(active_cases),'phase2':phase2,'selected_scale':selected,'selected':final_sum,'scale_reports':reports,'checkpoint':str(ckpt.resolve()),'sampling':manifest.get('sampling')})
    print('\n===== CONTEXTUAL LoRA NO-NEUTRAL MIN-CHANGE ====='); print('Base Eff:',base_sum['Eff']); print('Stage1 Eff:',sum1['Eff']); print('Target Eff <=',a.target_eff_max); print('Selected Eff:',final_sum['Eff']); print('Selected scale:',selected); print('Selection:',selection_reason); print('Unknown/target_new: NO'); print('checkpoint:',ckpt)

if __name__=='__main__': main()
