#!/usr/bin/env python3
"""Fail-closed verifier for ZeroUnlearn-compatible ZsRE SURE splits."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
from typing import Any, Mapping

import zsre_zero_unlearn_official_eval as zsre

PROTOCOLS = {
    "zsre_zerounlearn_forget_only_locked_probes",
    "zsre_zerounlearn_forget_only_locked_no_neutral",
}
NO_NEUTRAL = "zsre_zerounlearn_forget_only_locked_no_neutral"


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zsre-path",required=True); p.add_argument("--split-manifest",required=True)
    p.add_argument("--repair-visible-path",required=True); p.add_argument("--seed",type=int,required=True)
    p.add_argument("--forget-num",type=int,default=50); p.add_argument("--retain-num",type=int,default=1000)
    return p.parse_args()


def main():
    a=parse_args(); source=Path(a.zsre_path).resolve(); mp=Path(a.split_manifest).resolve(); vp=Path(a.repair_visible_path).resolve()
    raw=json.loads(source.read_text()); manifest=json.loads(mp.read_text()); visible=json.loads(vp.read_text())
    protocol=manifest.get("protocol")
    if protocol not in PROTOCOLS: raise RuntimeError(f"unexpected ZsRE locked-split protocol: {protocol}")
    if int(manifest.get("seed",-1))!=a.seed: raise RuntimeError("manifest seed mismatch")
    if manifest.get("source_sha256")!=sha256(source): raise RuntimeError("ZsRE source hash differs from split manifest")
    sampling:Mapping[str,Any]=manifest.get("sampling",{})
    if int(sampling.get("forget_num",-1))!=a.forget_num: raise RuntimeError("manifest forget count mismatch")
    if int(sampling.get("retain_num",-1))!=a.retain_num: raise RuntimeError("manifest retain count mismatch")
    if sampling.get("order")!="forget sample first, then retain sample from one seeded RNG": raise RuntimeError("manifest RNG order differs from ZeroUnlearn")

    fp,rp=zsre.sample_official_zsre_raw_records(raw,forget_num=a.forget_num,retain_num=a.retain_num,seed=a.seed,strict=True)
    ef=[int(i) for i,_ in fp]; er=[int(i) for i,_ in rp]
    mf=[int(i) for i in sampling.get("forget_case_ids",[])]; mr=[int(i) for i in sampling.get("retain_case_ids",[])]
    vf=[int(r["case_id"]) for r in visible]
    if mf!=ef: raise RuntimeError("manifest forget IDs differ from official ZeroUnlearn sampling")
    if mr!=er: raise RuntimeError("manifest retain IDs differ from official ZeroUnlearn sampling")
    if vf!=ef: raise RuntimeError("SURE training-visible forget IDs differ from ZeroUnlearn forget IDs")

    half=len(raw)//2
    if any(i<half for i in ef): raise RuntimeError("forget record escaped ZeroUnlearn second-half pool")
    if any(i>=half for i in er): raise RuntimeError("retain record escaped ZeroUnlearn first-half pool")
    if set(ef)&set(er): raise RuntimeError("forget and retain samples overlap")

    for row in visible:
        if row.get("paraphrase_prompts"): raise RuntimeError("training-visible split leaks rephrases")
        if row.get("neighborhood_prompts"): raise RuntimeError("training-visible split leaks locality")
        rr=row.get("requested_rewrite")
        if not isinstance(rr,dict): raise RuntimeError("training-visible record lacks requested_rewrite")
        if not rr.get("target_true",{}).get("str"): raise RuntimeError("training-visible record lacks sensitive target_true")
        if protocol==NO_NEUTRAL and "target_new" in rr: raise RuntimeError("no-neutral split contains target_new")
        if protocol==NO_NEUTRAL and rr.get("target_true",{}).get("str","").strip().lower()=="unknown": raise RuntimeError("literal Unknown appeared as sensitive target")

    print("===== ZSRE ZEROUnlearn SPLIT PARITY: PASS =====")
    print("protocol:",protocol); print("seed:",a.seed); print("dataset size:",len(raw),"half:",half)
    print("forget pool: second half; sampled:",len(ef)); print("retain pool: first half; sampled:",len(er))
    print("RNG order: forget first, retain second"); print("SURE Stage1/Stage2 visible forget IDs: EXACT MATCH")
    print("retain/rephrase/locality visible during training: 0")
    print("target_new/neutral target visible:", "NO" if protocol==NO_NEUTRAL else "YES (legacy track)")

if __name__=="__main__": main()
