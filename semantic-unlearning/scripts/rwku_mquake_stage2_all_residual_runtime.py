#!/usr/bin/env python3
"""Runtime for RWKU MQuAKE-style Stage 2 over all residual target rows."""
from __future__ import annotations

from typing import Any, Dict, Mapping

from rwku_mquake_stage2_all_residual_helpers import *  # noqa: F401,F403


def prepare_runtime(
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
    level1_anchor: Mapping[str, Any],
) -> Dict[str, Any]:
    source_cfg = head.load_locked_configuration(SOURCE_BUNDLE_CONFIGURATION)
    views, bundle_audit, generator_audit = head.load_atomic_bundle(
        Path(args.training_bundle).resolve(),
        Path(args.generator_receipt).resolve(),
        source_cfg,
    )
    generator_model_audit = head.validate_generator_base_model(
        generator_audit, args.model_path
    )

    gagd.set_seed(int(cfg["seed"]))
    gagd.require_cuda_if_needed(str(cfg["acceptance"]["device_map"]))
    model_args = argparse.Namespace(
        model_path=args.model_path,
        dtype=str(cfg["acceptance"]["checkpoint_dtype"]),
        device_map=str(cfg["acceptance"]["device_map"]),
        gradient_checkpointing=False,
    )
    model, tokenizer = gagd.load_model_and_tokenizer(model_args, for_training=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tokenizer)

    prompt_records = head.compile_prompt_records(
        views, tokenizer, neutral_target=str(cfg["neutral_target"])
    )
    cases = core.expand_sensitive_cases(
        prompt_records,
        tokenizer,
        sensitive_field="target_sensitive",
        llama_like=llama_like,
    )
    if not cases:
        raise RuntimeError("All-residual-row Stage2 runtime created no sensitive cases")
    tids_all = core.official_target_ids(
        tokenizer, cases, llama_like=llama_like, device=device
    )
    level1_content_rows, sensitive_row_audit = v2._content_sensitive_rows(
        tokenizer, cases, tids_all, source_cfg, len(prompt_records)
    )

    residual_positions = base2.residual_prompt_positions(level1_anchor["atomic"])
    protected_positions = success_positions(
        level1_anchor["atomic"], len(prompt_records)
    )
    residual_case_indices = base2.residual_case_indices(cases, residual_positions)
    protected_case_indices = prompt_case_indices(cases, protected_positions)
    residual_rows = all_non_special_residual_rows(
        tokenizer, tids_all, residual_case_indices
    )
    runtime_output_rows = sorted(
        set(int(x) for x in level1_content_rows)
        | set(int(x) for x in residual_rows)
    )
    newly_admitted_rows = sorted(
        set(residual_rows) - set(level1_content_rows)
    )

    sample_ids = tokenizer(
        prompt_records[0]["prompt_text"], return_tensors="pt"
    )["input_ids"].to(device)
    untie_audit = sparse_rows.untie_lm_head_preserve_logits(
        model, sample_input_ids=sample_ids
    )
    freeze_audit = sparse_rows.freeze_transformer_parameters(model)
    transformer_versions = v2._parameter_versions_except_vocab(model)
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if not torch.equal(input_layer.weight.detach(), output_layer.weight.detach()):
        raise RuntimeError("Untied Base vocabulary matrices are not initially identical")
    base_vocab_cpu = input_layer.weight.detach().cpu().clone()

    # Cache the true Base external hidden references BEFORE installing the L1 anchor.
    texts, wikipedia_meta = wikipedia.load_wikipedia_train(
        Path(args.wikipedia_dir).resolve()
    )
    protected_contexts, selection_contexts, fresh_contexts, external_audit = (
        v2.build_external_slices(tokenizer, texts, cfg)
    )
    del protected_contexts
    utility_bs = int(cfg["optimization"]["utility_batch_size"])
    selection_base_hidden = v2.external_final_hidden(
        model,
        tokenizer,
        selection_contexts,
        device=device,
        batch_size=utility_bs,
    ).cpu()
    fresh_base_hidden = v2.external_final_hidden(
        model,
        tokenizer,
        fresh_contexts,
        device=device,
        batch_size=utility_bs,
    ).cpu()

    # L1 keeps its original content-sensitive input rows. The output table is
    # enlarged only so Stage 2 can materialize every residual target row.
    sparse = sparse_rows.SparseFP32RowDeltas(
        model,
        selected_input_rows=level1_content_rows,
        selected_output_rows=runtime_output_rows,
    )
    expanded_l1_output = expand_anchor_output_delta(
        level1_anchor["output_delta"],
        level1_content_rows,
        runtime_output_rows,
    )
    with torch.no_grad():
        sparse.input_delta.copy_(
            level1_anchor["input_delta"].to(sparse.input_delta.device)
        )
        sparse.output_delta.copy_(
            expanded_l1_output.to(sparse.output_delta.device)
        )
    l1_input_anchor = sparse.input_delta.detach().clone()
    l1_output_anchor = sparse.output_delta.detach().clone()
    sparse.input_delta.requires_grad_(False)
    sparse.output_delta.requires_grad_(False)

    # Newly admitted Stage-2 rows must be exactly Base at the L1 anchor.
    output_mapping = {
        int(row): i for i, row in enumerate(sparse.selected_output_rows)
    }
    for row in newly_admitted_rows:
        index = output_mapping[int(row)]
        if not torch.equal(
            l1_output_anchor[index], torch.zeros_like(l1_output_anchor[index])
        ):
            raise RuntimeError(
                f"New Stage-2 row {row} has nonzero delta before repair"
            )

    residual_cases = [cases[i] for i in residual_case_indices]
    protected_cases = [cases[i] for i in protected_case_indices]
    stage2 = cfg["stage2"]
    bf, bp, basis_report = build_bases(
        model,
        tokenizer,
        protected_cases,
        residual_cases,
        device=device,
        batch_size=int(cfg["optimization"]["cache_batch_size"]),
        p_rank=int(stage2["protected_success_basis_rank"]),
        f_rank=int(stage2["residual_sensitive_exclusive_basis_rank"]),
    )
    protected_anchor_logits = core.cache_base_logits(
        model,
        tokenizer,
        protected_cases,
        device,
        batch_size=int(stage2["success_kl_batch_size"]),
    )

    initial_p = protection_report(
        model,
        tokenizer,
        cases,
        protected_case_indices,
        protected_anchor_logits,
        llama_like=llama_like,
        device=device,
        batch_size=int(stage2["success_kl_batch_size"]),
        required_margin=float(cfg["acceptance"]["required_pairwise_margin"]),
    )
    if int(initial_p["protected_prompt_regressions"]) != 0:
        raise RuntimeError("Protected P is not margin-safe at Stage-2 initialization")
    if abs(float(initial_p["protected_non_sensitive_kl_mean"])) > 1e-7:
        raise RuntimeError("KL(P_anchor || P_anchor) is unexpectedly nonzero")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "llama_like": llama_like,
        "prompt_records": prompt_records,
        "cases": cases,
        "level1_content_rows": level1_content_rows,
        "sensitive_rows": level1_content_rows,
        "runtime_output_rows": runtime_output_rows,
        "newly_admitted_stage2_rows": newly_admitted_rows,
        "sensitive_row_audit": sensitive_row_audit,
        "untie_audit": untie_audit,
        "freeze_audit": freeze_audit,
        "transformer_versions": transformer_versions,
        "input_layer": input_layer,
        "output_layer": output_layer,
        "base_vocab_cpu": base_vocab_cpu,
        "sparse": sparse,
        "l1_input_anchor": l1_input_anchor,
        "l1_output_anchor": l1_output_anchor,
        "selection_contexts": selection_contexts,
        "fresh_contexts": fresh_contexts,
        "selection_base_hidden": selection_base_hidden,
        "fresh_base_hidden": fresh_base_hidden,
        "utility_bs": utility_bs,
        "residual_positions": residual_positions,
        "protected_positions": protected_positions,
        "residual_case_indices": residual_case_indices,
        "protected_case_indices": protected_case_indices,
        "residual_rows": residual_rows,
        "bf": bf,
        "bp": bp,
        "basis_report": basis_report,
        "protected_anchor_logits": protected_anchor_logits,
        "bundle_audit": bundle_audit,
        "generator_model_audit": generator_model_audit,
        "wikipedia_meta": wikipedia_meta,
        "external_audit": external_audit,
    }
