#!/usr/bin/env python3
"""MCF Setting-5e Stage 2: protected-subspace sparse LM-head repair, NEUTRAL-row variant.

Sibling of mcf_sure_fullrow_failure_repair.py with exactly one change: which
field's tokens get selected as the editable LM-head rows.

  mcf_sure_fullrow_failure_repair.py  -- selects target_true's rows, SUPPRESSES
                                          the sensitive answer directly.
  THIS SCRIPT                         -- selects target_new's rows, BOOSTS the
                                          neutral/replacement answer instead.

This mirrors ZsRE's Setting 5e active repair, which edits the `Unknown` row
(the neutral answer) rather than the sensitive answer's own row -- see
zsre_gagd_setting5e_active_repair.py and
config/best_runs/by_model/llama_3b_instruct_model/zsre/
setting5e_active_repair_u1p20_ppl1p16_cal384_seeds1_10.md, which reached
Eff=0.0000/Gen=0.0000 on all 10 seeds with that design. MCF already has a
literal neutral answer (target_new) playing the same role as ZsRE's synthetic
"Unknown" token, so no neutral-target substitution is needed here -- only the
row-selection field changes.

Why this can work even though editing target_true's row could not (see
mcf_sure_fullrow_failure_repair.py's docstring and SURE_MCF_DIRECTIONAL_EMB_
LM_FULLREPAIR.md's Section 2): MCF neighborhood prompts share their subject's
*target_true* answer with the record being forgotten ("The mother tongue of
Leon Blum is" -> French, same as the forgotten Darrieux record), which is what
makes a shared target_true row a locality hazard. They do NOT generally need
to produce the forgotten record's target_new answer at all, so boosting
target_new's row is a fundamentally different, and potentially much safer,
edit -- to be confirmed empirically here, not assumed.

Mechanism is otherwise IDENTICAL to the target_true variant, reusing the same
machinery unchanged:

  * detect failed direct+synthetic-paraphrase records at the input checkpoint
    (input checkpoint is expected to be a Setting-5e Stage-1 output, i.e. a
    tied, full-vocabulary emb/LM-head GA/GD checkpoint -- see
    mcf_forget_only_setting5e.py -- but this script does not require that;
    any MCF checkpoint works);
  * select LM-head rows only from TARGET_NEW tokens of those failed records
    (the flip from the target_true sibling);
  * restrict the repair delta to rowspace(H_active) minus its projection onto
    rowspace(H_protected) (re-orthonormalized, rank-capped by --repair-rank);
  * the primary hinge loss is computed only on the failed records, pushing the
    SAME margin = NLL(target_true) - NLL(target_new) upward -- unchanged, since
    increasing target_new's row weight decreases NLL(target_new) and therefore
    still increases this margin;
  * a same-prompt-margin + same-prompt-non-target-KL hard gate is kept as a
    secondary backstop;
  * select the smallest direct-only scale yielding zero failures when possible;
  * materialize only the selected LM-head rows.

Input embeddings and all transformer parameters are frozen during Stage 2. No
official paraphrases, neighborhoods, benchmark-retain records, or PPL text are
opened for optimization or checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from datasets import load_from_disk

import gagd_compare as gagd
import gagd_active_case_repair as mcf_repair
import mcf_synthetic_paraphrase_templates as synth
import sure_canonical_core as core
import sure_stage2_sparse_repair as shared


METHOD = "SURE-MCF-failure-only-protected-subspace-LM-head-repair-neutral-row"
PROTOCOL = "mcf_target_new_neutral_failure_fullrow_repair_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--repair-steps", type=int, default=800)
    p.add_argument("--repair-lr", type=float, default=5e-3)
    p.add_argument("--constraint-margin", type=float, default=0.05)
    p.add_argument(
        "--repair-l2",
        type=float,
        default=1e-3,
        help=(
            "Raised from 1e-6: with the wider direct+synthetic objective, "
            "effective_delta_norm reached ~10.6 (||delta||^2 ~ 112), making "
            "the old 1e-6 * ||delta||^2 ~ 0.0001 term negligible next to a "
            "failure hinge that starts around 100+ per case (margins near "
            "-13, squared). At 1e-3 the L2 term is ~0.1 -- still small "
            "relative to an unsatisfied hinge, but large enough to actually "
            "discourage unnecessary delta magnitude once cases are passing."
        ),
    )
    p.add_argument(
        "--protected-rank",
        type=int,
        default=256,
        help=(
            "Structural protection, not a penalty: the repair delta is "
            "restricted to a basis built from the active (failing) cases' "
            "hidden states, after removing any component lying in this "
            "protected-rank subspace of the protected cases' hidden states "
            "(rowspace(H_active) minus its projection onto "
            "rowspace(H_protected), then re-orthonormalized). History: "
            "rank 32 (mcf_sure_protected_subspace_stage2.py's own default, "
            "sized for its much smaller in-sample-only protected set) left "
            "PPL/Spe under-protected once ~300 generic-text tokens were "
            "added (~352-vector population); raising to 96 improved Eff/Gen "
            "but Spe got *worse* (4.36 -> 2.09), most likely because the "
            "generic-text sample was one narrow, internally-correlated "
            "passage rather than diverse text, so tuning rank on it picked "
            "inconsistent specific directions instead of genuinely general "
            "ones. Now paired with --generic-protection-samples switching "
            "to many independent documents (~5000, one hidden state each) "
            "instead of one truncated passage: 256 gives that much larger, "
            "genuinely diverse population real room to be represented, "
            "still <=8%% of the 3072-dim hidden size."
        ),
    )
    p.add_argument(
        "--repair-rank",
        type=int,
        default=64,
        help=(
            "Rank cap on the protected-subspace-orthogonal repair basis. "
            "Was 4 (inherited from mcf_sure_protected_subspace_stage2.py's "
            "own default and never revisited after this script's "
            "architecture changed around it). --diagnose-only on a real "
            "run (protected-rank=256, 5000 generic docs) showed "
            "active_residual_rank_uncapped=174 while repair_basis_rank_"
            "actual was capped at 4 -- the protected-subspace projection "
            "removed zero rank from the active basis (174 == 174), so the "
            "near-total training freeze at rank 4 was not a structural "
            "ceiling, it was this cap leaving ~97%% of the available safe "
            "capacity unused. Raised to 64: a >10x increase that still "
            "leaves headroom under the demonstrated 174-dim ceiling."
        ),
    )
    p.add_argument(
        "--wikidata-dir",
        default="data/wikidata",
        help=(
            "Generic-text sample used to widen the protected subspace "
            "beyond the ~26 in-sample MCF records that happened to already "
            "pass Stage 1. A real run confirmed the narrow in-sample-only "
            "protected subspace was insufficient: PPL jumped from its "
            "stable ~10.9-11.1 across every prior run to 18.875, and Spe "
            "stayed collapsed (0.68) -- being orthogonal to ~26 records' "
            "hidden-state span (at most ~52 vectors) does nothing for the "
            "vast majority of directions real neighborhood prompts and "
            "general text actually occupy. Hidden states from this text "
            "are concatenated into H_protected before the SVD basis is "
            "built (evaluation-only in every other sense -- never used for "
            "the failure hinge or margin computation). Set to a "
            "nonexistent path (or leave --generic-protection-tokens 0) to "
            "disable and fall back to in-sample-only protection."
        ),
    )
    p.add_argument(
        "--generic-protection-samples",
        type=int,
        default=5000,
        help=(
            "Number of independent Wikidata documents sampled to widen the "
            "protected subspace (see --wikidata-dir), one representative "
            "hidden state per document -- not tokens from one concatenated "
            "passage. A run using --generic-protection-tokens 300 (a single "
            "truncated ~300-token passage from one document) recovered PPL "
            "but left the protected population too narrow/internally "
            "correlated to reliably improve Spe as --protected-rank was "
            "raised (Spe got *worse* going from rank 32->96 despite more "
            "geometric protection, most likely because the small, "
            "non-diverse sample let rank tuning pick inconsistent specific "
            "directions rather than genuinely general ones). Many "
            "independent short documents, each contributing one hidden "
            "state, give the SVD basis actual topic diversity to work "
            "with. Set 0 to disable."
        ),
    )
    p.add_argument(
        "--generic-protection-tokens-per-sample",
        type=int,
        default=32,
        help="Max tokens read per sampled document before taking its last-token hidden state.",
    )
    p.add_argument(
        "--generic-protection-batch-size",
        type=int,
        default=64,
        help="Documents per forward pass while gathering generic-protection hidden states.",
    )
    p.add_argument(
        "--generic-protection-doc-start",
        type=int,
        default=20,
        help=(
            "Start index into --wikidata-dir's document list; documents "
            "[doc-start : doc-start + generic-protection-samples] are read. "
            "mcf_zero_unlearn_official_eval's official PPL is hardcoded to "
            "documents [:20] -- this must stay disjoint from that range, or "
            "training would protect against the exact text the eval score "
            "is measured on. Default 20 (immediately after the official "
            "eval's [:20])."
        ),
    )
    p.add_argument(
        "--protected-kl-max",
        type=float,
        default=0.5,
        help=(
            "Secondary hard-gate backstop (the primary protection is now "
            "--protected-rank's geometric projection): every optimizer step "
            "is backtracked/rolled back unless it keeps (a) every "
            "currently-passing direct-only record's margin >= "
            "constraint-margin, unconditionally, and (b) same-prompt "
            "non-target KL on all currently-passing records <= this value. "
            "Raised from 0.05 (borrowed from protected_subspace's own "
            "already-rank-limited delta, which needs far less headroom than "
            "a repair basis built from ~172 combined active cases) to 0.5, "
            "matching the natural KL scale observed when a soft weight-3.0 "
            "penalty achieved a reasonable Eff/Spe balance "
            "(final_distribution_kl ~ 0.46). Set 0 to disable the KL half "
            "of the gate and keep only the margin-regression guard."
        ),
    )
    p.add_argument(
        "--backtrack-scales",
        default="1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0.00390625,0.001953125,0.0009765625,0.00048828125,0.0",
        help=(
            "Fractions of each raw optimizer step tried, largest first, "
            "until the hard gate is satisfied. The list always ends in 0.0 "
            "(full rollback to the pre-step delta), which trivially "
            "satisfies the gate -- so a step can never be silently dropped "
            "without a fallback, unlike a plain rollback-only implementation."
        ),
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument(
        "--candidate-scales",
        default="1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0",
    )
    p.add_argument(
        "--synthetic-paraphrases-per-record",
        type=int,
        default=3,
        help=(
            "Hand-authored synthetic paraphrase templates per record used to "
            "detect active/failing cases, train the repair delta, and gate "
            "scale selection. Unlike Stage 1's direction-constrained delta, "
            "Stage 2's unrestricted delta is fit directly against hidden "
            "states (no sensitive-minus-reference contrast direction), so it "
            "is not subject to the decoder-row fallback that made Stage 1's "
            "synthetic-prompt augmentation a no-op for single-token answers. "
            "Set 0 to disable and match the original direct-only behavior."
        ),
    )
    p.add_argument(
        "--diagnose-only",
        action="store_true",
        help=(
            "Build the protected/active/repair subspaces and report their "
            "actual ranks, then exit before the training loop. Cheaper "
            "than a full run (skips repair-steps of training and the final "
            "official eval) for answering: is the active-residual space "
            "genuinely large enough that raising --repair-rank could help, "
            "or has --protected-rank already consumed nearly all of it? "
            "Still requires loading the model onto a GPU and one forward "
            "pass over the protection/active/passing caches."
        ),
    )
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    a = p.parse_args(list(argv) if argv is not None else None)
    if min(a.forget_num, a.repair_steps, a.batch_size, a.check_every) <= 0:
        p.error("counts, repair steps, batch size and check interval must be positive")
    if a.repair_lr <= 0:
        p.error("repair-lr must be positive")
    if min(a.constraint_margin, a.repair_l2, a.protected_kl_max) < 0:
        p.error("margin, L2 and protected-kl-max must be non-negative")
    if a.synthetic_paraphrases_per_record < 0:
        p.error("synthetic-paraphrases-per-record must be non-negative")
    if a.protected_rank < 0 or a.repair_rank <= 0:
        p.error("protected-rank must be non-negative and repair-rank must be positive")
    if a.generic_protection_samples < 0:
        p.error("generic-protection-samples must be non-negative")
    if a.generic_protection_tokens_per_sample <= 0:
        p.error("generic-protection-tokens-per-sample must be positive")
    if a.generic_protection_batch_size <= 0:
        p.error("generic-protection-batch-size must be positive")
    if a.generic_protection_samples > 0 and a.generic_protection_doc_start < 20:
        p.error(
            "generic-protection-doc-start must be >= 20: "
            "mcf_zero_unlearn_official_eval's official PPL is hardcoded "
            "to documents [:20], and training-time protection must "
            "never read those same documents (it would contaminate the "
            "PPL score with text the model was directly protected "
            "against)"
        )
    backtrack_scales = parse_backtrack_scales(a.backtrack_scales)
    if backtrack_scales[-1] != 0.0:
        p.error("backtrack-scales must end in 0.0 (guaranteed-safe full rollback)")
    return a


def parse_backtrack_scales(text: str) -> List[float]:
    scales = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not scales:
        raise ValueError("No backtrack scales provided")
    for value in scales:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Backtrack scales must be finite and within [0, 1]")
    return scales


def select_repair_scale(reports: List[Dict[str, Any]]) -> float:
    """Same three-pass selection as Stage 1's select_stage1_scale: never let
    the harder combined (direct+synthetic) objective discard a scale that
    already achieves the best available direct-only result."""
    best_direct_only = min(int(r["direct_only_failures"]) for r in reports)
    candidates = [r for r in reports if int(r["direct_only_failures"]) == best_direct_only]
    best_combined = min(int(r["direct_failures"]) for r in candidates)
    candidates = [r for r in candidates if int(r["direct_failures"]) == best_combined]
    return float(max(float(r["scale"]) for r in candidates))


def validate_locked(
    visible_path: Path,
    manifest_path: Path,
    seed: int,
    forget_num: int,
):
    records, manifest = shared.load_locked(
        "mcf", visible_path, manifest_path, seed, forget_num
    )
    contract = manifest.get("target_contract", {})
    if isinstance(contract, Mapping) and contract:
        if contract.get("sensitive_answer") not in (
            None,
            "requested_rewrite.target_true",
        ):
            raise RuntimeError("Stage 2 requires target_true-sensitive MCF")
        if contract.get("non_sensitive_reference") not in (
            None,
            "requested_rewrite.target_new",
        ):
            raise RuntimeError("Stage 2 requires target_new as reference")
        if contract.get("field_swapping") not in (None, False):
            raise RuntimeError("Stage 2 requires unswapped MCF fields")
    return records, manifest


def margins_from_caches(caches, delta: torch.Tensor) -> torch.Tensor:
    return shared.mcf_margins_from_delta_caches(
        caches,
        delta,
        sensitive_field="target_true",
        reference_field="target_new",
    )


def failure_neutral_repair_rows(tok, instances, active_positions: Sequence[int]) -> List[int]:
    """Rows to EDIT for each failing case -- NOT the sensitive fact's rows.

    Renamed from the target_true sibling's failure_sensitive_rows() precisely
    to avoid this confusion: target_true ("French") is still the sensitive
    fact and target_new ("English") is still the neutral replacement -- that
    contract is UNCHANGED and is exactly what validate_locked() below still
    enforces. What differs from the sibling script is only WHICH ROW GETS
    EDITED to enforce it: this variant boosts target_new's row instead of
    suppressing target_true's row. shared.mcf_sensitive_rows()'s own
    `sensitive_field` KEYWORD is a generic "which field's tokens to select as
    editable rows" argument reused across both variants; passing
    "target_new" here selects the NEUTRAL answer's rows for editing, it does
    not redefine target_new as the sensitive fact.

    margins_from_caches is UNCHANGED (still NLL(target_true) -
    NLL(target_new)) -- see gagd_active_case_repair.answer_nll_from_delta_
    cache: a cache's NLL only picks up a correction from delta_rows if that
    answer's OWN tokens are in the selected set, so selecting target_new's
    tokens here means the trained delta affects only the target_new NLL
    term. Boosting target_new (decreasing its NLL) still increases the same
    margin the hinge below pushes upward, so no sign flip is needed anywhere
    else in this file.
    """
    return shared.mcf_sensitive_rows(
        tok,
        instances,
        active_positions,
        sensitive_field="target_new",  # which rows to EDIT, not which fact is sensitive
    )


def flatten_answer_hidden_states(
    caches: Sequence[Any], hidden_size: int
) -> torch.Tensor:
    """Stack every teacher-forced hidden state (both target_new and
    target_true sides) across the given RewriteDeltaCache records into one
    [N, hidden_size] matrix, for SVD-based subspace construction."""
    parts: List[torch.Tensor] = []
    for cache in caches:
        parts.append(cache.target_new.hidden)
        parts.append(cache.target_true.hidden)
    if not parts:
        return torch.empty((0, hidden_size))
    return torch.cat(parts, dim=0).float()


def load_wikidata_protection_documents(
    wikidata_dir: str, doc_start: int, num_samples: int
) -> List[str]:
    """Independent documents from the same Wikidata release official PPL is
    scored against, but *disjoint* from it:
    mcf_zero_unlearn_official_eval.load_official_ppl_text is hardcoded to
    raw_ds['train']['text'][:20], the exact documents official_perplexity
    later evaluates. Training-time protection must never read those same
    documents -- doing so would make any PPL improvement reflect having
    seen the exact evaluation bytes, not genuine specificity preservation.
    Default doc_start=20 starts immediately after the official eval's [:20].

    Returns a list of up to num_samples separate document strings (not one
    concatenated passage) -- a single long passage's token positions are
    all conditioned on the same narrow context, giving the SVD basis one
    internally-correlated sample rather than genuine topic diversity."""
    path = Path(wikidata_dir)
    if not path.exists():
        return []
    raw_ds = load_from_disk(str(path))
    texts = raw_ds["train"]["text"][doc_start : doc_start + num_samples]
    return [t for t in texts if t and t.strip()]


@torch.no_grad()
def generic_protection_hidden_states(
    model: torch.nn.Module,
    tok: Any,
    wikidata_dir: str,
    doc_start: int,
    num_samples: int,
    tokens_per_sample: int,
    batch_size: int,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor:
    """One representative hidden state (the last real token) per sampled
    document, so the protected subspace reflects genuinely diverse general
    text, not just the handful of in-sample MCF records that happened to
    already pass Stage 1, or one narrow, internally-correlated passage.
    Read-only: never contributes to the failure hinge or any margin
    computation. Uses documents disjoint from what official PPL evaluation
    reads (see load_wikidata_protection_documents)."""
    if num_samples <= 0:
        return torch.empty((0, hidden_size))
    docs = load_wikidata_protection_documents(wikidata_dir, doc_start, num_samples)
    if not docs:
        print(
            f"WARNING: --wikidata-dir {wikidata_dir!r} has no documents "
            f"at/after index {doc_start}; the protected subspace will only "
            "reflect the in-sample MCF records, which real runs showed is "
            "not enough to protect PPL/specificity."
        )
        return torch.empty((0, hidden_size))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vectors: List[torch.Tensor] = []
    for start in range(0, len(docs), int(batch_size)):
        chunk = docs[start : start + int(batch_size)]
        encoded = tok(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(tokens_per_sample),
        ).to(device)
        output = model(**encoded, output_hidden_states=True, use_cache=False)
        last_non_masked = encoded["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(len(chunk), device=device)
        vectors.append(
            output.hidden_states[-1][batch_indices, last_non_masked, :].float()
        )
    return torch.cat(vectors, dim=0)


def repair_delta_raw_param(delta_module: "core.SelectedRowDelta") -> torch.nn.Parameter:
    """The single trainable tensor backing effective_delta(), whichever
    parameterization is active (basis coefficients or an unrestricted raw
    delta) -- lets the hard-gate backtracking below manipulate it generically."""
    return (
        delta_module.coefficients
        if delta_module.coefficients is not None
        else delta_module.raw_delta
    )


def repair_effective_delta_from_raw(
    delta_module: "core.SelectedRowDelta", raw: torch.Tensor
) -> torch.Tensor:
    if delta_module.coefficients is not None:
        return raw @ delta_module.direction_basis
    return raw


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(int(a.seed))
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, manifest = validate_locked(
        visible_path, manifest_path, int(a.seed), int(a.forget_num)
    )

    synthetic_records = synth.build_synthetic_records(
        records, count=int(a.synthetic_paraphrases_per_record)
    )
    all_records = list(records) + synthetic_records
    synthetic_coverage = synth.coverage_report(records)
    if int(a.synthetic_paraphrases_per_record) > 0 and synthetic_coverage["generic_fallback_records"]:
        print(
            "WARNING: "
            f"{synthetic_coverage['generic_fallback_records']}/{len(records)} records "
            "fell back to the generic synthetic-paraphrase templates (relation_id "
            "missing or unrecognized): "
            f"{synthetic_coverage['generic_fallback_relation_ids']}."
        )

    ns = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    output_layer = core.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    if input_layer is None:
        raise RuntimeError("model lacks input embeddings")
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tok)

    direct_count = len(records)
    all_instances = shared.mcf_instances(all_records)
    original_margins = shared.mcf_direct_margins(
        model,
        tok,
        all_instances,
        device,
        llama_like,
        int(a.batch_size),
        sensitive_field="target_true",
        reference_field="target_new",
    )
    original_cpu = original_margins.detach().float().cpu()
    # Active/failing now spans direct + synthetic-paraphrase instances, so
    # Stage 2 repairs paraphrase-margin residuals too, not only the literal
    # direct prompt.
    active_positions = [
        i
        for i, value in enumerate(original_cpu.tolist())
        if float(value) < float(a.constraint_margin)
    ]
    passing_positions = [
        i for i in range(len(all_instances)) if i not in set(active_positions)
    ]
    direct_active_positions = [i for i in active_positions if i < direct_count]
    synthetic_active_positions = [i for i in active_positions if i >= direct_count]
    selected_ids = failure_neutral_repair_rows(tok, all_instances, active_positions)

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    logs: List[Dict[str, Any]] = []
    scale_reports: List[Dict[str, Any]] = []
    best_step = 0
    best_failures = len(active_positions)
    selected_scale = 0.0
    gate_rejected_steps = 0
    gate_backtracked_steps = 0
    repair_basis_rank = 0
    in_sample_basis_rank = 0
    generic_basis_rank = 0
    generic_hidden = torch.empty((0, int(output_layer.weight.shape[1])))
    backtrack_scales = parse_backtrack_scales(a.backtrack_scales)
    final_delta = torch.empty(
        (0, int(output_layer.weight.shape[1])),
        dtype=torch.float32,
        device=output_layer.weight.device,
    )

    if selected_ids:
        caches = mcf_repair.build_prompt_instance_delta_caches(
            model,
            tok,
            all_instances,
            selected_ids,
            device,
            int(a.batch_size),
            llama_like,
        )
        active_caches = [caches[i] for i in active_positions]
        passing_caches = [caches[i] for i in passing_positions]

        # Structural protection: restrict the repair delta to a basis built
        # from the active (failing) cases' hidden states, after projecting
        # away any component lying in the protected cases' own hidden-state
        # subspace. A fully unrestricted delta over shared LM-head rows has
        # no way to distinguish "suppress this token for the forget
        # prompts" from "suppress this token everywhere" -- this makes
        # "does not disturb protected cases" geometric by construction,
        # matching mcf_sure_protected_subspace_stage2.py's proven design.
        # The protected set is the *initially passing* in-sample records
        # PLUS a sample of ordinary text (see generic_protection_hidden_states):
        # a real run using only the ~26 in-sample records left PPL at 18.875
        # (every other run: ~10.9-11.1) and Spe collapsed at 0.68 --
        # orthogonality to ~52 vectors' span does nothing for the vast
        # majority of directions real neighborhood prompts and general text
        # actually occupy.
        hidden_size = int(output_layer.weight.shape[1])
        generic_hidden = generic_protection_hidden_states(
            model,
            tok,
            a.wikidata_dir,
            int(a.generic_protection_doc_start),
            int(a.generic_protection_samples),
            int(a.generic_protection_tokens_per_sample),
            int(a.generic_protection_batch_size),
            hidden_size,
            device,
        )
        # protected_basis is built in two priority tiers rather than one
        # combined SVD over [in_sample; generic]. The hard gate below only
        # ever checks the ~26 in-sample records (protected_direct_caches /
        # protected_caches, the same set in_sample_hidden is built from) --
        # a single SVD over ~52 in-sample rows mixed with 5000 generic rows
        # picks its top --protected-rank directions by aggregate variance,
        # which the generic rows dominate purely by count, leaving the
        # specific rows the gate checks only approximately (not exactly)
        # captured. A real run at repair-rank 4 and again at repair-rank 64
        # both saw gate_rejected_steps ~774-793/800 with best_step landing
        # in the first ~25 steps either way -- raising repair-rank changed
        # nothing, which only makes sense if the geometric "no effect on
        # protected cases" guarantee was never exact for the specific rows
        # being gate-checked. Fix: capture in_sample_hidden's own exact rank
        # first (uncapped -- it is tiny, well under --protected-rank), then
        # spend the remaining rank budget on generic-text directions
        # orthogonal to that, so the in-sample rows are never diluted by
        # generic count.
        in_sample_hidden = flatten_answer_hidden_states(passing_caches, hidden_size)
        in_sample_basis = core.orthonormal_row_basis(in_sample_hidden, max_rank=None)
        remaining_protected_rank = max(
            0, int(a.protected_rank) - int(in_sample_basis.shape[0])
        )
        generic_residual = mcf_repair.project_rows_away(
            generic_hidden, in_sample_basis if in_sample_basis.numel() else None
        )
        generic_basis = core.orthonormal_row_basis(
            generic_residual, max_rank=remaining_protected_rank
        )
        protected_hidden = torch.cat([in_sample_hidden, generic_hidden], dim=0)
        protected_basis = torch.cat([in_sample_basis, generic_basis], dim=0)
        in_sample_basis_rank = int(in_sample_basis.shape[0])
        generic_basis_rank = int(generic_basis.shape[0])
        active_hidden = flatten_answer_hidden_states(active_caches, hidden_size)
        active_basis = core.orthonormal_row_basis(active_hidden, max_rank=None)
        active_residual = mcf_repair.project_rows_away(
            active_basis, protected_basis if protected_basis.numel() else None
        )
        # Uncapped: the TRUE available rank of the residual, before
        # --repair-rank truncates it. If this is already tiny, raising
        # --repair-rank cannot help -- the protected subspace has consumed
        # nearly all of active_hidden's own natural variance, not just
        # capped what a larger repair-rank could otherwise use.
        active_residual_rank_uncapped = int(
            core.orthonormal_row_basis(active_residual, max_rank=None).shape[0]
        )
        repair_basis = core.orthonormal_row_basis(
            active_residual, max_rank=int(a.repair_rank)
        )
        if repair_basis.shape[0] == 0:
            raise RuntimeError(
                "protected-subspace projection left zero repair directions; "
                "lower --protected-rank or raise --repair-rank"
            )
        repair_basis_rank = int(repair_basis.shape[0])

        if a.diagnose_only:
            diagnostics = {
                "protected_rank_requested": int(a.protected_rank),
                "protected_basis_rank_actual": int(protected_basis.shape[0]),
                "in_sample_basis_rank_exact": int(in_sample_basis.shape[0]),
                "generic_basis_rank_actual": int(generic_basis.shape[0]),
                "protected_hidden_vectors": int(protected_hidden.shape[0]),
                "active_hidden_vectors": int(active_hidden.shape[0]),
                "active_basis_rank_uncapped": int(active_basis.shape[0]),
                "active_residual_rank_uncapped": active_residual_rank_uncapped,
                "repair_rank_requested": int(a.repair_rank),
                "repair_basis_rank_actual": repair_basis_rank,
                "interpretation": (
                    "active_residual_rank_uncapped is the TRUE ceiling on "
                    "what any --repair-rank value could use. If it is only "
                    "slightly above repair_basis_rank_actual, raising "
                    "--repair-rank will not meaningfully help -- the "
                    "protected subspace has already consumed nearly all of "
                    "the active cases' own natural hidden-state variance, "
                    "not merely been capped by a small requested rank. "
                    "Separately, in_sample_basis_rank_exact should be small "
                    "(tens, not hundreds) and generic_basis_rank_actual "
                    "should be close to protected_rank_requested minus "
                    "that -- if generic_basis_rank_actual is far below the "
                    "remaining budget, the generic sample itself has less "
                    "true rank than requested, not a dilution problem."
                ),
            }
            diag_path = out_dir / "protected_subspace_diagnostics.json"
            core.write_json(diag_path, diagnostics)
            print(json.dumps(diagnostics, indent=2))
            print(f"--diagnose-only: wrote {diag_path}; exiting before training.")
            return

        delta_module = core.SelectedRowDelta(
            len(selected_ids),
            int(output_layer.weight.shape[1]),
            direction_basis=repair_basis,
            device=output_layer.weight.device,
        )
        opt = torch.optim.AdamW(
            delta_module.parameters(), lr=float(a.repair_lr), weight_decay=0.0
        )
        best_delta = delta_module.effective_delta().detach().clone()
        best_key = (10**9, 10**9, float("inf"))

        # Hard-gate protected set is fixed to the *initially* passing cases
        # -- the same set B_protected was built from -- not recomputed each
        # step. A dynamically growing set (protect a record the instant it
        # first passes) would treat every active case's own progress as an
        # immediate new protection requirement, even though its hidden
        # state was deliberately part of H_active (the basis is expected,
        # by construction, to move it). That mismatch is why a first
        # attempt at this rejected 711/800 steps: continuing to refine
        # coefficients for still-failing cases naturally wobbles
        # recently-fixed ones sharing the same basis, which a dynamic gate
        # then punished as a violation instead of expected, in-scope
        # movement of the very directions the basis was built to touch.
        protected_direct_positions = [i for i in passing_positions if i < direct_count]
        protected_positions = list(passing_positions)
        protected_direct_caches = [caches[i] for i in protected_direct_positions]
        protected_caches = [caches[i] for i in protected_positions]

        for step in range(1, int(a.repair_steps) + 1):
            with torch.no_grad():
                pre_raw = repair_delta_raw_param(delta_module).detach().clone()
                pre_delta = repair_effective_delta_from_raw(delta_module, pre_raw)

            opt.zero_grad(set_to_none=True)
            delta = delta_module.effective_delta()
            active_margins = margins_from_caches(active_caches, delta)
            failure_hinge = F.relu(
                float(a.constraint_margin) - active_margins
            ).square().mean()
            l2 = delta.square().mean()
            loss = failure_hinge + float(a.repair_l2) * l2
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage-2 loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(delta_module.parameters()), 1.0)
            opt.step()

            with torch.no_grad():
                raw_update = repair_delta_raw_param(delta_module).detach() - pre_raw
                accepted_scale = 0.0
                accepted_raw = pre_raw
                for bscale in backtrack_scales:
                    candidate_raw = pre_raw + raw_update * float(bscale)
                    candidate = repair_effective_delta_from_raw(delta_module, candidate_raw)
                    ok = True
                    if protected_direct_caches:
                        candidate_direct_margins = margins_from_caches(
                            protected_direct_caches, candidate
                        )
                        if bool(
                            (candidate_direct_margins < float(a.constraint_margin)).any()
                        ):
                            ok = False
                    if ok and protected_caches and float(a.protected_kl_max) > 0:
                        candidate_kl = float(
                            mcf_repair.mcf_same_prompt_non_target_kl(
                                protected_caches, candidate
                            )
                        )
                        if candidate_kl > float(a.protected_kl_max):
                            ok = False
                    if ok:
                        accepted_scale = float(bscale)
                        accepted_raw = candidate_raw
                        break
                if accepted_scale == 0.0:
                    gate_rejected_steps += 1
                elif accepted_scale < 1.0:
                    gate_backtracked_steps += 1
                repair_delta_raw_param(delta_module).data.copy_(accepted_raw)

            if step == 1 or step % int(a.check_every) == 0 or step == int(a.repair_steps):
                with torch.no_grad():
                    current = delta_module.effective_delta()
                    all_margins = margins_from_caches(caches, current)
                    direct_only_margins = all_margins[:direct_count]
                    failures = int(
                        (all_margins < float(a.constraint_margin)).sum().item()
                    )
                    direct_only_failures = int(
                        (direct_only_margins < float(a.constraint_margin)).sum().item()
                    )
                    norm = float(current.norm().detach().cpu())
                    row = {
                        "step": int(step),
                        "all_direct_failures": failures,
                        "direct_only_failures": direct_only_failures,
                        "active_failure_hinge": float(failure_hinge.detach().cpu()),
                        "accepted_backtrack_scale": accepted_scale,
                        "gate_rejected_steps_so_far": gate_rejected_steps,
                        "gate_backtracked_steps_so_far": gate_backtracked_steps,
                        "minimum_margin": float(all_margins.min().detach().cpu()),
                        "delta_norm": norm,
                        "lora_used": False,
                        "rank_constraint": False,
                    }
                    logs.append(row)
                    # direct_only_failures first: never let a lower combined
                    # failure count or smaller norm elsewhere in training be
                    # preferred over an already-achieved perfect direct-only
                    # result -- same principle as select_repair_scale's
                    # scale-sweep priority. The hard gate above should make
                    # this ordering redundant in practice (protected direct
                    # records cannot regress), but it costs nothing to keep
                    # as a second line of defense.
                    key = (direct_only_failures, failures, norm)
                    if key < best_key:
                        best_key = key
                        best_step = int(step)
                        best_failures = int(failures)
                        best_delta = current.detach().clone()
                    if failures == 0:
                        break
        del opt

        scales = core.parse_scales(a.candidate_scales)
        for scale in scales:
            margins = margins_from_caches(caches, best_delta * float(scale))
            direct_margins = margins[:direct_count]
            scale_reports.append(
                {
                    "scale": float(scale),
                    "direct_failures": int(
                        (margins < float(a.constraint_margin)).sum().item()
                    ),
                    "direct_only_failures": int(
                        (direct_margins < float(a.constraint_margin)).sum().item()
                    ),
                    "minimum_margin": float(margins.min().detach().cpu()),
                    "direct_only_minimum_margin": float(
                        direct_margins.min().detach().cpu()
                    ),
                    "effective_delta_norm": float(
                        best_delta.norm().detach().cpu() * float(scale)
                    ),
                }
            )
        # select_repair_scale (not core.choose_scale): never let the harder
        # combined direct+synthetic objective collapse to scale=0.0 when it
        # cannot reach zero failures -- that would silently discard an edit
        # that already achieves the best available direct-only result (see
        # mcf_sure_directional_emb_lm_stage1.py's identical fix).
        selected_scale = select_repair_scale(scale_reports)
        final_delta = best_delta * float(selected_scale)
        final_distribution_kl = float(
            mcf_repair.mcf_same_prompt_non_target_kl(caches, final_delta)
            .detach()
            .cpu()
        )
        core.materialize_output_delta(output_layer, selected_ids, final_delta)
    else:
        final_distribution_kl = 0.0

    final_all_margins = shared.mcf_direct_margins(
        model,
        tok,
        all_instances,
        device,
        llama_like,
        int(a.batch_size),
        sensitive_field="target_true",
        reference_field="target_new",
    )
    final_margins = final_all_margins[:direct_count]
    final_synthetic_margins = final_all_margins[direct_count:]
    final_cpu = final_margins.detach().float().cpu()
    final_failure_positions = [
        i
        for i, value in enumerate(final_cpu.tolist())
        if float(value) < float(a.constraint_margin)
    ]
    final_synthetic_failure_positions = [
        i
        for i, value in enumerate(final_synthetic_margins.detach().cpu().tolist())
        if float(value) < float(a.constraint_margin)
    ]

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    summary: Dict[str, Any] = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "source_protocol": manifest.get("protocol"),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "target_contract": {
            "sensitive_answer": "requested_rewrite.target_true",
            "non_sensitive_reference": "requested_rewrite.target_new",
            "field_swapping": False,
        },
        "repaired_row_field": "target_new",
        "repair_direction": "boost_neutral_answer_not_suppress_sensitive_answer",
        "stage1_failure_count": len(active_positions),
        "stage1_failure_positions": active_positions,
        "stage1_passing_positions": passing_positions,
        "stage1_direct_failure_count": len(direct_active_positions),
        "stage1_synthetic_failure_count": len(synthetic_active_positions),
        "synthetic_paraphrases_per_record": int(a.synthetic_paraphrases_per_record),
        "synthetic_record_count": len(synthetic_records),
        "synthetic_paraphrase_coverage": synthetic_coverage,
        "selected_lm_head_rows": len(selected_ids),
        "selected_token_ids": selected_ids,
        "parameterization": "protected_subspace_orthogonal_lm_head_rows",
        "lora_used": False,
        "rank_constraint": True,
        "protected_rank": int(a.protected_rank),
        "repair_rank_requested": int(a.repair_rank),
        "repair_basis_rank_actual": repair_basis_rank,
        "in_sample_basis_rank_exact": in_sample_basis_rank,
        "generic_basis_rank_actual": generic_basis_rank,
        "protected_subspace_definition": (
            "repair delta restricted to rowspace(H_active) minus its "
            "projection onto rowspace(H_protected) (re-orthonormalized), "
            "where H_active is teacher-forced hidden states from the "
            "initially active (direct+synthetic) cases and H_protected is "
            "the initially passing cases' hidden states plus a sample of "
            "generic text -- makes 'does not disturb protected cases' "
            "geometric by construction rather than statistical. "
            "H_protected's basis is built in two priority tiers: the "
            "in-sample rows get an exact, uncapped basis first (in_sample_"
            "basis_rank_exact), then generic-text rows fill the remaining "
            "protected_rank budget orthogonal to that (generic_basis_rank_"
            "actual) -- prevents the numerically much larger generic "
            "sample from diluting protection of the specific rows the "
            "hard gate checks."
        ),
        "generic_protection_samples_requested": int(a.generic_protection_samples),
        "generic_protection_tokens_per_sample": int(a.generic_protection_tokens_per_sample),
        "generic_protection_hidden_states_used": int(generic_hidden.shape[0]),
        "wikidata_dir": str(a.wikidata_dir),
        "generic_protection_doc_range": [
            int(a.generic_protection_doc_start),
            int(a.generic_protection_doc_start) + int(a.generic_protection_samples),
        ],
        "generic_protection_disjoint_from_official_ppl_docs": (
            "official PPL is hardcoded to documents [:20]; this run read "
            f"documents starting at {int(a.generic_protection_doc_start)}, "
            "enforced disjoint by argparse validation (doc-start >= 20)"
        ),
        "repair_primary_training_records": (
            "Stage-1 failed direct records + synthetic-paraphrase templates"
        ),
        "passing_records_role": (
            "protected-subspace projection (primary) + hard gate backstop "
            "(secondary), both scoped to the same fixed set: records "
            "passing before Stage 2 started (the same set H_protected was "
            "built from). Their margins may not drop below "
            "constraint-margin, and same-prompt non-target KL over the set "
            "may not exceed protected-kl-max; a step violating either is "
            "backtracked (geometric scale-down of the raw update) or fully "
            "rolled back. Deliberately NOT recomputed as records become "
            "newly passing during training -- a record's own hidden state "
            "was part of H_active precisely so the basis could move it, so "
            "treating it as newly off-limits the instant it first passes "
            "would punish the repair basis for doing its job (this is what "
            "drove an earlier attempt's gate_rejected_steps=711/800)."
        ),
        "protected_kl_max": float(a.protected_kl_max),
        "backtrack_scales": backtrack_scales,
        "gate_rejected_steps": gate_rejected_steps,
        "gate_backtracked_steps": gate_backtracked_steps,
        "final_distribution_kl": final_distribution_kl,
        "final_distribution_kl_definition": (
            "post-hoc diagnostic only (not a gate input): exact KL(input-"
            "checkpoint non-target || current non-target) at every visible "
            "direct+synthetic teacher-forced position for the final "
            "materialized delta"
        ),
        "constraint_margin": float(a.constraint_margin),
        "repair_steps": int(a.repair_steps),
        "repair_lr": float(a.repair_lr),
        "repair_l2": float(a.repair_l2),
        "best_step": int(best_step),
        "best_unscaled_direct_failures": int(best_failures),
        "logs": logs,
        "scale_reports": scale_reports,
        "selected_scale": float(selected_scale),
        "effective_delta_norm": float(final_delta.norm().detach().cpu())
        if final_delta.numel()
        else 0.0,
        "final_direct_failures": len(final_failure_positions),
        "final_failing_positions": final_failure_positions,
        "final_minimum_margin": float(final_cpu.min().item()),
        "final_synthetic_failures": len(final_synthetic_failure_positions),
        "final_synthetic_failing_positions": final_synthetic_failure_positions,
        "final_synthetic_minimum_margin": (
            float(final_synthetic_margins.min().detach().cpu())
            if final_synthetic_margins.numel()
            else None
        ),
        "final_combined_failures": (
            len(final_failure_positions) + len(final_synthetic_failure_positions)
        ),
        "input_embeddings_modified_in_stage2": False,
        "transformer_trainable_parameters": 0,
        "lm_head_untied": True,
        "official_paraphrases_seen": 0,
        "official_neighborhood_seen": 0,
        "benchmark_retain_seen": 0,
        "ppl_eval_text_seen": 0,
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "repair_summary.json", summary)
    core.write_json(out_dir / "scale_sweep_direct_only.json", scale_reports)
    print(json.dumps(summary, indent=2))
    print(
        f"Full-row Stage 2: direct failures {len(direct_active_positions)} -> "
        f"{len(final_failure_positions)}; synthetic failures "
        f"{len(synthetic_active_positions)} -> {len(final_synthetic_failure_positions)}; "
        f"selected rows={len(selected_ids)}; scale={selected_scale:g}"
    )


if __name__ == "__main__":
    main()
