#!/usr/bin/env python3
"""Base-only answer-level frozen-head probe for RWKU hidden-direction v3.1."""
from __future__ import annotations
import argparse
from pathlib import Path
import gagd_compare as gagd
import rwku_sure_head_only_w1k as head
import rwku_sure_hidden_direction_v31_w1k as v31
import sure_canonical_core as core

SOURCE_CONFIGURATION=Path(__file__).resolve().parents[1]/"config"/"rwku"/"sure_head_only_w1k_seed0.json"

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model-path",required=True); p.add_argument("--training-bundle",type=Path,required=True); p.add_argument("--generator-receipt",type=Path,required=True); p.add_argument("--configuration",type=Path,default=v31.DEFAULT_CONFIGURATION); p.add_argument("--output",type=Path,required=True); args=p.parse_args()
    cfg=v31.load_configuration(args.configuration); src=head.load_locked_configuration(SOURCE_CONFIGURATION)
    views,bundle_audit,generator_audit=head.load_atomic_bundle(args.training_bundle,args.generator_receipt,src); generator_model_audit=head.validate_generator_base_model(generator_audit,args.model_path)
    gagd.set_seed(int(cfg["seed"])); gagd.require_cuda_if_needed(cfg["acceptance"]["device_map"])
    margs=argparse.Namespace(model_path=args.model_path,dtype=cfg["acceptance"]["checkpoint_dtype"],device_map=cfg["acceptance"]["device_map"],gradient_checkpointing=False)
    model,tok=gagd.load_model_and_tokenizer(margs,for_training=False)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side="right"; device=gagd.first_device(model); llama_like=core.is_llama_like(model,tok)
    records=head.compile_prompt_records(views,tok,neutral_target=str(cfg["neutral_target"])); inp=model.get_input_embeddings(); out=model.get_output_embeddings()
    if inp is None or out is None or inp.weight.data_ptr()!=out.weight.data_ptr(): raise ValueError("base probe requires tied vocabulary weights")
    output_layer=core.untie_and_freeze_output_head(model); base_w=model.get_input_embeddings().weight
    if base_w.data_ptr()==output_layer.weight.data_ptr(): raise RuntimeError("base probe output head remains tied")
    v31._RUNTIME["tokenizer"]=tok; v31._RUNTIME["answer_eval_batch_size"]=int(cfg["hidden_direction"]["answer_eval_batch_size"])
    cases,audit=v31.build_answer_cases(model,tok,records,base_w,[],[],float(cfg["hidden_direction"]["frozen_base_head_training_margin"]),llama_like,device)
    report=v31.answer_proxy_report(model,cases,base_w,device)
    atomic=head.materialized_atomic_report(model,tok,records,device,llama_like=llama_like,required_margin=float(cfg["acceptance"]["required_pairwise_margin"]))
    payload={"schema_version":"rwku_sure_hidden_direction_v31_base_answer_probe_v1","configuration_id":cfg["configuration_id"],"target":{"seed":0,"entity":cfg["target_entity"],"entity_id":cfg["target_entity_id"]},"answer_case_audit":audit,"base_frozen_head_answer_proxy":report,"base_atomic_view_report":atomic,"atomic_bundle":bundle_audit,"atomic_generator_base_model":generator_model_audit,"official_rwku_records_accessed":False,"posthoc_development_target":True}
    core.write_json(args.output,payload)
    print("RWKU v3.1 BASE answer-level frozen-head probe")
    print(f"prompt_count: {report['prompt_count']}")
    print(f"recovery: {report['recovery_count']}/{report['prompt_count']} = {report['recovery_percentage']:.2f}%")
    print(f"mean sensitive NLL: {report['mean_sensitive_nll']:.6f}")
    print(f"mean neutral NLL: {report['mean_neutral_nll']:.6f}")
    print(f"mean demotion margin (sens-neutral NLL): {report['mean_demotion_margin']:.6f}")
    print(f"minimum demotion margin: {report['minimum_demotion_margin']:.6f}")
    print(f"maximum demotion margin: {report['maximum_demotion_margin']:.6f}")
    for kind,x in report["by_prompt_kind"].items(): print("  {}: recovery={}/{} ({:.2f}%) mean_margin={:.6f} min_margin={:.6f}".format(kind,x["recovery_count"],x["count"],x["recovery_percentage"],x["mean_demotion_margin"],x["minimum_demotion_margin"]))
    print(f"report: {args.output.resolve()}"); print("Official RWKU records accessed: False")
if __name__=="__main__": main()
