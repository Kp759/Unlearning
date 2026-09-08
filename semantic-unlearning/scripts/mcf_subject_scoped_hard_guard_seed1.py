#!/usr/bin/env python3
"""Seed-1 subject-scoped exact hard guard for registered MCF sensitive answers.

This is a deliberately strong, verifier-free endpoint diagnostic. For any atomic query
that literally contains a registered forbidden subject, the guard hard-blocks the
registered sensitive answer sequence(s) for that subject during BOTH teacher-forced
scoring and deterministic generation.

It does NOT use the evaluation relation label, target_true/target_new branch identity,
or the Fix5o relation prediction to decide whether to block. It uses only:
  * the deployment-time registered forbidden bank (subject -> sensitive answer sequence)
  * literal registered-subject occurrence in the actual query
  * the generated prefix for sequence-completion masking

This is conditional suppression, not knowledge deletion. It is intentionally coarser
than relation-selective routing: the same registered sensitive answer is unavailable
for that subject even if a different relation request would legitimately need the
same answer. Therefore this experiment must be interpreted together with the
preservation stress audit; zero Eff/Gen alone is not sufficient for promotion.
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_target_local_fixed_penalty_integration_fix5l_seed1 as fix5l
import mcf_target_local_generation_mixed_eval_fix5m_seed1 as fix5m
import mcf_fix5o_matched_end_to_end_fix5p_seed1 as fix5p
import mcf_fact_conditioned_completion_guard_seed1 as guardlib

SEED = 1
base = fix5l.base


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as h:
        return json.load(h)


def registered_subjects_in_query(query: str, support_map: Mapping[tuple[str,str], Any]) -> list[str]:
    subjects = sorted({s for s, _ in support_map}, key=len, reverse=True)
    return [s for s in subjects if base.old.subject_regex(s).search(str(query))]


def subject_sequences(tok, support_map, query: str, llama_like: bool) -> tuple[list[tuple[int,...]], list[tuple[str,str]]]:
    found = registered_subjects_in_query(query, support_map)
    seqs: set[tuple[int,...]] = set()
    keys: list[tuple[str,str]] = []
    for key, support in support_map.items():
        if key[0] not in found:
            continue
        keys.append(key)
        for seq in guardlib.answer_sequences(tok, support, llama_like):
            if seq:
                seqs.add(tuple(map(int, seq)))
    return sorted(seqs, key=lambda x:(len(x),x)), sorted(keys)


class SequenceOnlyProcessor:
    def __init__(self, prompt_len: int, sequences: Sequence[Sequence[int]]):
        self.prompt_len = int(prompt_len)
        self.sequences = tuple(tuple(map(int, s)) for s in sequences if s)
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        out = scores.clone()
        for row in range(out.shape[0]):
            generated = input_ids[row, self.prompt_len:].detach().cpu().tolist()
            blocked = guardlib.blocked_next_tokens(generated, self.sequences)
            if blocked:
                out[row, torch.tensor(sorted(blocked), dtype=torch.long, device=out.device)] = -torch.inf
        return out


def generate_guarded(model, tok, query: str, sequences, max_new_tokens: int, device):
    enc = tok(query, add_special_tokens=True, return_tensors="pt").to(device)
    from transformers import LogitsProcessorList
    proc = SequenceOnlyProcessor(int(enc["input_ids"].shape[1]), sequences)
    seq = model.generate(
        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
        do_sample=False, num_beams=1, max_new_tokens=int(max_new_tokens),
        pad_token_id=int(tok.pad_token_id), eos_token_id=int(tok.eos_token_id), use_cache=True,
        logits_processor=LogitsProcessorList([proc]),
    )
    n = int(enc["input_ids"].shape[1])
    ids = seq[0, n:].detach().cpu().tolist()
    return {"text": tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            "token_ids": [int(x) for x in ids], "new_token_n": len(ids)}


def score_guarded(model, tok, prefix: str, target_new: str, target_true: str, sequences, device, llama_like):
    prefix_ids = tok([prefix], add_special_tokens=True)["input_ids"][0]
    plen = len(prefix_ids)
    inputs = tok([f"{prefix} {target_new}", f"{prefix} {target_true}"], padding=True, return_tensors="pt").to(device)
    logits = model(**inputs, use_cache=False, return_dict=True).logits.float()
    new_ids = fix5l.answer_token_ids(tok, target_new, llama_like)
    true_ids = fix5l.answer_token_ids(tok, target_true, llama_like)
    if llama_like:
        logits = logits[:, 1:, :]
        plen -= 1
    seqs = tuple(tuple(map(int,s)) for s in sequences)
    def one(row: int, ids: Sequence[int]):
        nll = 0.0
        impossible = False
        generated: list[int] = []
        for j, tid in enumerate(ids):
            pos = plen + j - 1
            scores = logits[row, pos, :]
            blocked = guardlib.blocked_next_tokens(generated, seqs)
            if int(tid) in blocked:
                impossible = True
                return float("inf"), True
            nll += -torch.log_softmax(scores, dim=0)[int(tid)].item()
            generated.append(int(tid))
        return nll / max(1, len(ids)), impossible
    nnll, ni = one(0, new_ids)
    tnll, ti = one(1, true_ids)
    return {"target_new": nnll, "target_true": tnll, "target_new_impossible": ni, "target_true_impossible": ti}


def summarize(rows, cond: str):
    n = len(rows)
    valid = [r for r in rows if not (r[cond]["score"]["target_true_impossible"] and r[cond]["score"]["target_new_impossible"])]
    pref = 0
    for r in valid:
        s = r[cond]["score"]
        if float(s["target_true"]) < float(s["target_new"]):
            pref += 1
    disclosed = sum(bool(r[cond]["flags"]["target_true_canonical_mentioned"]) for r in rows)
    return {
        "n": n,
        "valid_preference_n": len(valid),
        "both_answers_impossible_n": n-len(valid),
        "target_true_impossible_n": sum(bool(r[cond]["score"]["target_true_impossible"]) for r in rows),
        "target_new_impossible_n": sum(bool(r[cond]["score"]["target_new_impossible"]) for r in rows),
        "sensitive_preference_n": pref,
        "sensitive_preference_pct_valid_cases": 100.0*pref/len(valid) if valid else None,
        "canonical_sensitive_disclosure_n": disclosed,
        "canonical_sensitive_disclosure_pct": 100.0*disclosed/n if n else None,
        "surface_invalid_n": sum(fix5m.surface_invalid(r[cond]["generation"]["text"]) for r in rows),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix5l-output-dir", required=True)
    ap.add_argument("--fix5p-records", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--mcf-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dtype", choices=("bf16","fp16","fp32"), default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    a = ap.parse_args()

    out = Path(a.output_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    device = torch.device(a.device)
    fix5l_dir = Path(a.fix5l_output_dir).resolve()
    support_map, penalty, _ = fix5m.load_frozen_supports(fix5l_dir / "frozen_answer_token_support_fix5l.json")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True, use_fast=True, clean_up_tokenization_spaces=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model_path, dtype=base.old.dtype_from_name(a.dtype), local_files_only=True, low_cpu_mem_usage=True).to(device)
    model.eval(); model.config.use_cache=True
    for p in model.parameters(): p.requires_grad_(False)
    llama_like = fix5l.is_llama_like(model, tok)

    # Reconstruct official atomic query identities and targets; Fix5p records are used only
    # to verify that we evaluate the same bank and to carry the historical Base/Fix5o numbers.
    from mcf_sampling import sample_official_mcf_records
    import mcf_zero_unlearn_official_eval as off
    data = load_json(Path(a.mcf_path))
    forget_raw, _ = sample_official_mcf_records(data, 50, 1000, SEED, strict=True)
    forget = [off.normalize_record(x) for x in forget_raw]
    specs=[]
    for rec in forget:
        specs.append(("direct", rec, fix5m.direct_prompt(rec)))
    para=[]
    for rec in forget:
        for p in rec.get("paraphrase_prompts",[]): para.append(("paraphrase",rec,str(p)))
    specs.extend(para[:100])

    old_rows = fix5p.load_jsonl(Path(a.fix5p_records))
    old_index={(r["group"],int(r["case_id"]),r["query"]):r for r in old_rows}
    rows=[]
    for idx,(group,rec,query) in enumerate(specs,1):
        if idx==1 or idx%10==0 or idx==len(specs): print(f"[subject-hard] endpoint: {idx}/{len(specs)}",flush=True)
        key=(group,int(rec.get("case_id",-1)),query)
        if key not in old_index: raise RuntimeError(f"Fix5p identity missing: {key}")
        tnew,ttrue=fix5m.target_pair(rec)
        sequences,blocked_keys=subject_sequences(tok,support_map,query,llama_like)
        if not sequences: raise RuntimeError(f"no registered subject sequence for atomic query: {query!r}")
        g=generate_guarded(model,tok,query,sequences,a.max_new_tokens,device)
        s=score_guarded(model,tok,query,tnew,ttrue,sequences,device,llama_like)
        rows.append({
            "group":group,"case_id":int(rec.get("case_id",-1)),"query":query,
            "expected_binding":list(fix5m.record_binding(rec)),"blocked_registered_bindings":[list(x) for x in blocked_keys],
            "subject_hard":{"score":s,"generation":g,"flags":fix5m.generated_answer_flags(g["text"],ttrue,tnew)},
        })

    direct=[r for r in rows if r["group"]=="direct"]; para=[r for r in rows if r["group"]=="paraphrase"]
    report={
        "schema_version":1,"kind":"mcf_seed1_subject_scoped_exact_registered_sequence_guard",
        "base_model_frozen":True,"relation_router_used_for_hard_guard":False,"verifier_used":False,
        "penalty_used_for_subject_hard":False,"quotient_enabled":False,
        "guard_scope":"all registered sensitive answer sequences for every literal registered subject present in the atomic query",
        "scientific_warning":"This is stronger and less relation-selective than Fix5o. Zero endpoint preference/disclosure does not by itself establish acceptable preservation.",
        "direct":summarize(direct,"subject_hard"),"paraphrase":summarize(para,"subject_hard"),
        "historical_fix5p":{
            "direct_fix5o_eff_pref_pct":16.0,"direct_fix5o_disclosure_pct":12.0,
            "paraphrase_fix5o_gen_pref_pct":55.0,"paraphrase_fix5o_disclosure_pct":13.0,
        },
    }
    rp=out/"mcf_subject_scoped_hard_guard_seed1.json"; rr=out/"mcf_subject_scoped_hard_guard_records_seed1.jsonl"
    rp.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    with rr.open("w",encoding="utf-8") as h:
        for r in rows: h.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(json.dumps({"status":"ENDPOINT_COMPLETED","direct":report["direct"],"paraphrase":report["paraphrase"],"report":str(rp),"records":str(rr)},indent=2))

if __name__=="__main__": main()
