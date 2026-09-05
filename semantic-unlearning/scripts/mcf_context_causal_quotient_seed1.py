#!/usr/bin/env python3
"""Seed-1 Method 5: retain-anchored context head + causal quotient.

Discovery/fitting may use only direct rewrite prompts, target_true, retain rewrite
anchors, and deterministic subject-corrupted direct prompts. target_new,
official paraphrases, and neighborhoods are evaluation-only.
"""
from __future__ import annotations

import argparse, copy, json, statistics, sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
import mcf_retain_anchored_context_head_seed1 as m4
import mcf_zero_unlearn_official_eval as off
from mcf_sampling import sample_official_mcf_records
from retain_anchored_context_head import (
    AnchoredFeatureMap, CausalDescriptorProjector, ContextualCausalQuotientModel,
    FactIndexedCausalQuotient, FactIndexedLogitCorrection, FrozenRandomProjector,
)


def layers_of(model):
    for obj, name in ((getattr(model, "model", None), "layers"),
                      (getattr(model, "transformer", None), "h"),
                      (getattr(model, "gpt_neox", None), "layers")):
        value = getattr(obj, name, None) if obj is not None else None
        if value is not None:
            return value
    raise RuntimeError("Transformer block list not found")


def corrupt_records(records):
    subjects = [r["requested_rewrite"]["subject"] for r in records]
    answers = [r["requested_rewrite"]["target_true"]["str"] for r in records]
    out, n = [], len(records)
    for i, record in enumerate(records):
        j = next((j for k in range(1, n) if (j := (i+k) % n) != i
                  and subjects[j] != subjects[i] and answers[j] != answers[i]), None)
        if j is None:
            raise RuntimeError("Could not choose subject corruption")
        item = copy.deepcopy(record)
        item["requested_rewrite"]["subject"] = subjects[j]
        out.append(item)
    return out


@torch.no_grad()
def run(model, tok, spec, device):
    enc = tok(spec.text, return_tensors="pt").to(device)
    return model(**enc, output_hidden_states=True, return_dict=True, use_cache=False)


def lp(outputs, pos, token):
    return float(F.log_softmax(outputs.logits[0, pos].float(), -1)[int(token)].item())


def replace_output(output, pos, replacement):
    if isinstance(output, tuple):
        h = output[0].clone(); h[:, pos] = replacement.to(h)
        return (h, *output[1:])
    h = output.clone(); h[:, pos] = replacement.to(h)
    return h


@torch.no_grad()
def patched(model, tok, spec, device, block, pos, replacement):
    enc = tok(spec.text, return_tensors="pt").to(device)
    handle = block.register_forward_hook(
        lambda _m, _i, out: replace_output(out, pos, replacement)
    )
    try:
        return model(**enc, output_hidden_states=True, return_dict=True, use_cache=False)
    finally:
        handle.remove()


@torch.no_grad()
def choose_layer(model, tok, orig_specs, corrupt_specs, device, probes=8):
    layers = layers_of(model); n = len(layers)
    candidates = sorted(set([n//4, n//2, 3*n//4, max(0, n-2)]))
    score = {b: [] for b in candidates}
    absolute = {b: [] for b in candidates}
    informative = 0
    for i in range(min(probes, len(orig_specs))):
        a, c = orig_specs[i], corrupt_specs[i]
        pa, pc, token = a.event_positions[0], c.event_positions[0], a.event_token_ids[0]
        oa, oc = run(model, tok, a, device), run(model, tok, c, device)
        base_gap = lp(oa, pa, token) - lp(oc, pc, token)
        informative += int(base_gap > .05)
        for b in candidates:
            restored = oa.hidden_states[b+1][0, pa].detach()
            op = patched(model, tok, c, device, layers[b], pc, restored)
            gain = lp(op, pc, token) - lp(oc, pc, token)
            absolute[b].append(gain)
            if base_gap > .05: score[b].append(gain/base_gap)
    avg = lambda xs: sum(xs)/len(xs) if xs else float("-inf")
    best = max(candidates, key=lambda b: avg(score[b]) if informative else avg(absolute[b]))
    return best, best+1, {
        "candidates": candidates, "probe_records": min(probes, len(orig_specs)),
        "informative_probe_records": informative, "selected_block": best,
        "selected_hidden_index": best+1,
        "fractional_recovery": {str(b): None if not score[b] else avg(score[b]) for b in candidates},
        "absolute_recovery": {str(b): None if not absolute[b] else avg(absolute[b]) for b in candidates},
    }


@torch.no_grad()
def event_states(model, tok, specs, hidden_idx, batch_size, device):
    hs, finals, tids, rids = [], [], [], []
    old = tok.padding_side; tok.padding_side = "right"
    try:
        for start in range(0, len(specs), batch_size):
            batch = specs[start:start+batch_size]
            enc = tok([s.text for s in batch], padding=True, return_tensors="pt").to(device)
            out = model(**enc, output_hidden_states=True, return_dict=True, use_cache=False)
            for row, spec in enumerate(batch):
                for pos, tid in zip(spec.event_positions, spec.event_token_ids):
                    hs.append(out.hidden_states[hidden_idx][row, pos].detach().float())
                    finals.append(out.hidden_states[-1][row, pos].detach().float())
                    tids.append(int(tid)); rids.append(int(spec.record_index))
    finally:
        tok.padding_side = old
    return torch.stack(hs), torch.stack(finals), tids, rids


def channel_basis(orig, corrupt, rank):
    d = (orig-corrupt).float(); norms = d.norm(dim=-1)
    if torch.any(norms <= 1e-8): raise RuntimeError("Zero subject-corruption direction")
    _, s, vh = torch.linalg.svd(d/norms[:,None], full_matrices=False)
    r = min(rank, vh.shape[0]); return vh[:r].T.contiguous(), s, norms


@torch.no_grad()
def causal_effects(model, tok, specs, orig_h, corrupt_h, orig_final, corrupt_final,
                   basis, block_idx, device):
    block = layers_of(model)[block_idx]
    effects, neutrals, drops = [], [], []
    k = 0
    for spec in specs:
        base = run(model, tok, spec, device)
        for pos, tid in zip(spec.event_positions, spec.event_token_ids):
            disp = orig_h[k]-corrupt_h[k]
            replacement = orig_h[k] - basis @ (basis.T @ disp)
            q = patched(model, tok, spec, device, block, pos, replacement)
            neutral = q.hidden_states[-1][0, pos].detach().float()
            effect = orig_final[k]-neutral
            drop = lp(base, pos, tid)-lp(q, pos, tid)
            if effect.norm() <= 1e-8:
                effect = orig_final[k]-corrupt_final[k]
                if effect.norm() <= 1e-8:
                    effect = torch.zeros_like(effect); effect[0] = 1e-4
            effects.append(effect); neutrals.append(neutral); drops.append(drop); k += 1
    drops_t = torch.tensor(drops, device=device)
    mask = (drops_t > 0).float()
    return torch.stack(effects), torch.stack(neutrals), mask, {
        "valid_quotient_events": int(mask.sum()), "valid_fraction": float(mask.mean()),
        "mean_sensitive_logprob_drop": float(drops_t.mean()),
        "median_sensitive_logprob_drop": float(drops_t.median()),
        "min_sensitive_logprob_drop": float(drops_t.min()),
        "max_sensitive_logprob_drop": float(drops_t.max()),
    }


@torch.no_grad()
def descriptors(model, tok, projector, specs, hidden_idx, batch_size, device):
    desc, tids, rids = [], [], []
    old = tok.padding_side; tok.padding_side = "right"
    try:
        for start in range(0, len(specs), batch_size):
            batch = specs[start:start+batch_size]
            enc = tok([s.text for s in batch], padding=True, return_tensors="pt").to(device)
            out = model(**enc, output_hidden_states=True, return_dict=True, use_cache=False)
            hf, hc = [], []
            for row, spec in enumerate(batch):
                for pos, tid in zip(spec.event_positions, spec.event_token_ids):
                    hf.append(out.hidden_states[-1][row,pos]); hc.append(out.hidden_states[hidden_idx][row,pos])
                    tids.append(int(tid)); rids.append(int(spec.record_index))
            if hf: desc.append(projector(torch.stack(hf), torch.stack(hc)).float())
    finally:
        tok.padding_side = old
    return torch.cat(desc), tids, rids


def metrics(result, base=None, threshold=.1):
    def cases(raw, key, fn):
        out=[]
        for item in raw:
            xs=item["post"].get(key,[])
            if xs: out.append(sum(fn(x) for x in xs)/len(xs))
        return out
    direct_nll=cases(result["forget_raw"],"rewrite_prompts_probs",lambda x:float(x["target_true"]))
    para_nll=cases(result["forget_raw"],"paraphrase_prompts_probs",lambda x:float(x["target_true"]))
    dm=cases(result["forget_raw"],"rewrite_prompts_probs",lambda x:float(x["target_true"])-float(x["target_new"]))
    pm=cases(result["forget_raw"],"paraphrase_prompts_probs",lambda x:float(x["target_true"])-float(x["target_new"]))
    mean=lambda xs: None if not xs else sum(xs)/len(xs)
    out={"Eff_Pref":result["forget"]["Eff"],"Gen_Pref":result["forget"]["Gen"],
         "Direct_Sensitive_NLL":mean(direct_nll),"Para_Sensitive_NLL":mean(para_nll),
         "Direct_Margin_Mean":mean(dm),"Para_Margin_Mean":mean(pm),
         "Direct_Margin_Failure_Rate":None if not dm else 100*sum(x<=threshold for x in dm)/len(dm),
         "Para_Margin_Failure_Rate":None if not pm else 100*sum(x<=threshold for x in pm)/len(pm),
         "PPL_legacy100":result.get("retain_PPL"),"Eff_Leak":None,"Gen_Leak":None}
    if base is not None:
        for label,key in (("Direct","rewrite_prompts_probs"),("Para","paraphrase_prompts_probs")):
            deltas=[]
            for b,e in zip(base["forget_raw"],result["forget_raw"]):
                bx,ex=b["post"].get(key,[]),e["post"].get(key,[])
                if bx: deltas.append(sum(float(y["target_true"])-float(x["target_true"]) for x,y in zip(bx,ex))/len(bx))
            out[f"Delta_{label}_Sensitive_NLL"]=mean(deltas)
            out[f"{label}_Positive_Delta_Rate"]=None if not deltas else 100*sum(x>0 for x in deltas)/len(deltas)
    return out


def main():
    p=argparse.ArgumentParser()
    for name in ("model-path","mcf-path","wikidata-dir","output-dir"): p.add_argument(f"--{name}",required=True)
    p.add_argument("--seed",type=int,default=1); p.add_argument("--forget-num",type=int,default=50); p.add_argument("--retain-num",type=int,default=1000)
    p.add_argument("--dtype",default="bf16"); p.add_argument("--device",default="cuda")
    p.add_argument("--descriptor-dim",type=int,default=32); p.add_argument("--projection-seed",type=int,default=1729)
    p.add_argument("--causal-rank",type=int,default=8); p.add_argument("--causal-weight",type=float,default=1.0); p.add_argument("--causal-probes",type=int,default=8)
    p.add_argument("--quotient-strength",type=float,default=1.0); p.add_argument("--radius",type=float,default=1.0); p.add_argument("--logit-penalty",type=float,default=12.0)
    p.add_argument("--retain-jitter",type=float,default=1e-4); p.add_argument("--cardinal-jitter",type=float,default=1e-4)
    p.add_argument("--retain-events-per-record",type=int,default=1); p.add_argument("--extract-batch-size",type=int,default=16); p.add_argument("--alpha-chunk-size",type=int,default=256)
    p.add_argument("--margin-threshold",type=float,default=.1); p.add_argument("--eval-base",action="store_true"); p.add_argument("--skip-ppl",action="store_true")
    a=p.parse_args()
    if (a.seed,a.forget_num,a.retain_num)!=(1,50,1000): raise ValueError("Pilot locked to seed1, forget50, retain1000")
    device=torch.device(a.device); outdir=Path(a.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side="right"
    model=AutoModelForCausalLM.from_pretrained(a.model_path,torch_dtype=m4.dtype_from_str(a.dtype),local_files_only=True).to(device)
    model.eval(); model.config.use_cache=False
    for q in model.parameters(): q.requires_grad_(False)
    data=json.load(open(a.mcf_path)); fr,rr=sample_official_mcf_records(data,50,1000,1,strict=True)
    fr=[off.normalize_record(x) for x in fr]; rr=[off.normalize_record(x) for x in rr]
    fs=m4.build_specs(fr,tok,max_events_per_record=None); rs=m4.build_specs(rr,tok,max_events_per_record=a.retain_events_per_record)
    cr=corrupt_records(fr); cs=m4.build_specs(cr,tok,max_events_per_record=None)

    block_idx,hidx,layer_diag=choose_layer(model,tok,fs,cs,device,a.causal_probes)
    oh,of,tids,rids=event_states(model,tok,fs,hidx,a.extract_batch_size,device)
    ch,cf,ctids,crids=event_states(model,tok,cs,hidx,a.extract_batch_size,device)
    if tids!=ctids or rids!=crids: raise RuntimeError("Corrupt event mapping mismatch")
    basis,svals,dnorms=channel_basis(oh,ch,a.causal_rank)
    effects,neutral,qmask,causal_diag=causal_effects(model,tok,fs,oh,ch,of,cf,basis,block_idx,device)
    causal_diag.update({"causal_rank":int(basis.shape[1]),"selected_block":block_idx,"selected_hidden_index":hidx,
                        "subject_corruption_norm_mean":float(dnorms.mean()),"top_singular_values":[float(x) for x in svals[:min(12,len(svals))]]})
    print(json.dumps({"layer_selection":layer_diag,"causal_validation":causal_diag},indent=2))

    rp=FrozenRandomProjector(input_dim=model.config.hidden_size,output_dim=a.descriptor_dim,seed=a.projection_seed,device=device,dtype=torch.float32)
    dp=CausalDescriptorProjector(random_projector=rp,channel_basis=basis.to(device),causal_weight=a.causal_weight).to(device)
    fd,ftids,frids=descriptors(model,tok,dp,fs,hidx,a.extract_batch_size,device)
    rd,_,_=descriptors(model,tok,dp,rs,hidx,a.extract_batch_size,device)
    fmap=AnchoredFeatureMap.fit(retain=rd.float(),forget=fd.float(),radius=a.radius,retain_jitter=a.retain_jitter,cardinal_jitter=a.cardinal_jitter)
    corr=FactIndexedLogitCorrection(feature_map=fmap,selected_token_ids=sorted(set(ftids)),vocab_size=model.config.vocab_size).to(device)
    row={int(t):i for i,t in enumerate(corr.selected_token_ids.tolist())}
    with torch.no_grad():
        corr.coefficients.zero_()
        for i,t in enumerate(ftids): corr.coefficients[row[int(t)],i]=-a.logit_penalty
    quotient=FactIndexedCausalQuotient(effect_directions=effects.to(device),neutral_final_hidden=neutral.to(device),strength=a.quotient_strength).to(device)
    d=corr.diagnostics(); qd=quotient.diagnostics()
    fit={"max_abs_retain_alpha":d.max_abs_retain_alpha,"max_abs_cardinal_error":d.max_abs_cardinal_error,
         "num_forget_events":len(ftids),"num_retain_anchors":len(rd),"num_selected_tokens":d.num_selected_tokens,
         "descriptor_dim_context":a.descriptor_dim,"descriptor_dim_total":dp.output_dim,"causal_rank":dp.channel_rank,
         "causal_weight":a.causal_weight,"radius":a.radius,"logit_penalty":a.logit_penalty,"quotient_strength":a.quotient_strength,
         "quotient_valid_events":int(qmask.sum()),"quotient_valid_fraction":float(qmask.mean()),"quotient_mean_effect_norm":qd.mean_effect_norm}

    overlap=m4._hard_overlap_records(rr,tok,set(ftids)); base=None; base_overlap=None
    if a.eval_base:
        base=off.evaluate_loaded_model_official(method="base",model=model,tok=tok,model_dir=a.model_path,mcf_path=a.mcf_path,wikidata_dir=a.wikidata_dir,out_path=outdir/"base_official_eval.json",unlearn_num=50,retain_num=1000,seed=1,sample_mode="official",skip_ppl=a.skip_ppl)
        base_overlap,_=m4._evaluate_subset(model,tok,overlap,split_name="hard_overlap_retain_base")
    wrapped=ContextualCausalQuotientModel(base_model=model,descriptor_projector=dp,correction=corr,quotient=quotient,causal_hidden_index=hidx,quotient_fact_mask=qmask,alpha_chunk_size=a.alpha_chunk_size).to(device).eval()
    res=off.evaluate_loaded_model_official(method="context_plus_causal_quotient",model=wrapped,tok=tok,model_dir=a.model_path,mcf_path=a.mcf_path,wikidata_dir=a.wikidata_dir,out_path=outdir/"method5_official_eval.json",unlearn_num=50,retain_num=1000,seed=1,sample_mode="official",skip_ppl=a.skip_ppl)
    hov,hraw=m4._evaluate_subset(wrapped,tok,overlap,split_name="hard_overlap_retain_method5")
    bm=None if base is None else metrics(base,threshold=a.margin_threshold); mm=metrics(res,base,a.margin_threshold)
    summary={"schema_version":1,"kind":"mcf_seed1_context_plus_causal_quotient","training_contract":{"target_new_used_for_fit":False,"official_paraphrases_used_for_fit":False,"official_neighborhoods_used_for_fit":False,"subject_corruption_used_for_causal_discovery":True,"transformer_frozen":True,"base_embeddings_modified":False,"base_lm_head_modified":False},
             "fit_diagnostics":fit,"layer_selection":layer_diag,"causal_validation":causal_diag,"base_metrics":bm,"method5_metrics":mm,
             "hard_overlap_retain_records":len(overlap),"base_hard_overlap":base_overlap,"method5_hard_overlap":hov,"base":base,"method5":res}
    (outdir/"seed1_method5_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    torch.save({"channel_basis":basis.cpu(),"selected_hidden_index":hidx,"retain_descriptors":fmap.retain.cpu(),"forget_descriptors":fmap.forget.cpu(),"selected_token_ids":corr.selected_token_ids.cpu(),"coefficients":corr.coefficients.detach().cpu(),"effect_directions":quotient.effect_directions.cpu(),"neutral_final_hidden":quotient.neutral_final_hidden.cpu(),"quotient_fact_mask":qmask.cpu(),"fit_diagnostics":fit,"layer_selection":layer_diag,"causal_validation":causal_diag},outdir/"method5_sidecar.pt")
    print(json.dumps({"base_metrics":bm,"method5_metrics":mm,"hard_overlap_retain_records":len(overlap),"hard_overlap_base":base_overlap,"hard_overlap_method5":hov,"fit_diagnostics":fit,"layer_selection":layer_diag,"causal_validation":causal_diag,"output_dir":str(outdir)},indent=2))

if __name__=="__main__": main()
