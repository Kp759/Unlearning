#!/usr/bin/env python3
"""Embedding Writer + Orthogonal LM-Head Reader for MCF unlearning.

The transformer is frozen end to end. Only sparse input-embedding rows
(subject tokens) and sparse LM-head rows (sensitive answer tokens) change.

Motivation, measured rather than assumed: MCF's Eff and Spe read the SAME
LM-head row in opposite directions -- the forget prompt wants "French"
suppressed, the record's neighborhood prompts are other French speakers and
want it intact. At the final layer those contexts are indistinguishable
(residual 0.0805 vs held-out 0.0832, gap -0.003, stable while protection
scaled 8 -> 2048), so no delta on the shared row separates them. Suppression
and specificity are strictly zero-sum there.

This architecture removes the conflict instead of trading against it:

    the embedding writes WHO should forget; the LM head reads WHAT to forget

  Stage 0  For each sensitive answer, mine contexts where it is legitimately
           correct, build a protected basis, and choose a marker direction
           that is (a) orthogonal to that basis and (b) reachable by
           perturbing only the subject's embedding rows.
  Stage 1  Train ONLY the subject embedding rows so the forget prompt writes
           that marker into its final hidden state. No forgetting hinge --
           Stage 1 creates a signature, it does not forget.
  Gate     Verify per record that separability actually opened. Fail loudly
           if it did not; the mechanism is falsified there, cheaply.
  Stage 2  Closed form, no optimizer. Read the marker back with a sparse
           LM-head delta that is exactly orthogonal to every protected state.

Locality then holds two ways. Structurally: a prompt containing none of the
edited subject tokens has a bit-identical embedding sequence, and the
transformer is frozen, so its hidden states are unchanged. Analytically: the
reader direction is orthogonal to every protected state by construction, so
those logits are unchanged to floating-point.

Data discipline: Stages 0-2 read only the training-visible forget records and
Wikipedia text. Paraphrases, neighborhood prompts, retain records and the
official PPL documents are never opened. The one place evaluation data may be
touched is the clearly separated diagnostic_only_eval_overlap audit, which
runs after training and cannot influence markers, losses, gates or
hyperparameters.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

import gagd_compare as gagd
from mcf_sampling import sample_official_mcf_records


METHOD = "MCF-embedding-writer-orthogonal-lm-head-reader"
PROTOCOL = "mcf_marker_write_read_v1"


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--mcf-path", default="data/multi_counterfact.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)

    # Stage 0A -- protected contexts
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--corpus-docs", type=int, default=50000)
    p.add_argument(
        "--corpus-doc-start", type=int, default=20,
        help="Must stay >= 20: official PPL is scored on documents [:20].",
    )
    p.add_argument("--contexts-per-token", type=int, default=512)
    p.add_argument("--corpus-max-tokens", type=int, default=256)
    p.add_argument(
        "--gate-holdout-frac", type=float, default=0.25,
        help="Share of mined contexts withheld from marker construction, used only by the gate.",
    )

    # Stage 0B -- reachability
    p.add_argument("--reach-probes", type=int, default=96)
    p.add_argument("--reach-sigma", type=float, default=0.02)

    # Stage 1 -- writer
    p.add_argument("--write-alpha", type=float, default=4.0)
    p.add_argument("--lambda-write", type=float, default=1.0)
    p.add_argument("--lambda-off", type=float, default=1.0)
    p.add_argument("--lambda-kl", type=float, default=1.0)
    p.add_argument("--writer-steps", type=int, default=400)
    p.add_argument("--writer-lr", type=float, default=2e-4)
    p.add_argument("--writer-batch-size", type=int, default=8)
    p.add_argument(
        "--writer-eval-every", type=int, default=50,
        help=(
            "Evaluate the write objective over ALL records this often. The "
            "logged minibatch loss samples 8 different records each time and "
            "need not decrease monotonically even when training is working."
        ),
    )
    p.add_argument(
        "--writer-grad-clip", type=float, default=1.0,
        help="Clip the writer delta's gradient norm. 0 disables.",
    )

    # Gate
    p.add_argument("--gate-min", type=float, default=0.15)
    p.add_argument(
        "--gate-pass-frac", type=float, default=1.0,
        help="Required fraction of records meeting --gate-min. 1.0 = every record.",
    )

    # Stage 2 -- reader
    p.add_argument("--forget-margin", type=float, default=1.0)
    p.add_argument("--reader-ridge", type=float, default=1e-6)
    p.add_argument(
        "--reader-backtrack",
        default="1.0,1.25,1.5,2.0,3.0,4.0,6.0,8.0",
        help="Scalar multipliers tried on the solved delta until the exact NLL margin is met.",
    )
    p.add_argument(
        "--certificate-eps", type=float, default=1e-4,
        help=(
            "Bound on the protected logit shift RELATIVE to the mean protected "
            "logit magnitude. Relative because |dW . h_p| grows with ||dW||, so "
            "an absolute bound fails spuriously once backtracking scales the delta."
        ),
    )

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    p.add_argument("--save-checkpoint", action="store_true")
    p.add_argument(
        "--eval-overlap-audit", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Post-hoc diagnostics that read evaluation data: the G_cross "
            "measurement against real neighborhood prompts and the "
            "retain-prompt subject-row overlap count. Both run after "
            "training and cannot influence markers, losses, the gate or "
            "any hyperparameter. Disable for a strictly eval-blind run."
        ),
    )

    a = p.parse_args(list(argv) if argv is not None else None)
    if a.corpus_doc_start < 20:
        p.error("corpus-doc-start must be >= 20 to stay disjoint from official PPL text")
    if not 0.0 < a.gate_holdout_frac < 1.0:
        p.error("gate-holdout-frac must be strictly between 0 and 1")
    if a.reach_probes < 8:
        p.error("reach-probes must be >= 8 for a usable reachability sketch")
    a.reader_backtrack = [float(x) for x in str(a.reader_backtrack).split(",") if x.strip()]
    return a


# --------------------------------------------------------------------------
# sparse embedding-row delta (the writer's only parameters)
# --------------------------------------------------------------------------
class EmbeddingRowDelta:
    """Adds a trainable delta to selected input-embedding rows via a hook.

    The base embedding matrix stays frozen and out of the optimizer; only a
    [n_rows, hidden] tensor is trained. Prompts containing none of these
    token ids get a bit-identical embedding sequence, which is what makes the
    structural locality claim exact.
    """

    def __init__(self, embedding: torch.nn.Module, row_ids: Sequence[int], device: torch.device):
        self.row_ids = [int(x) for x in row_ids]
        vocab, hidden = embedding.weight.shape
        self.lookup = torch.full((vocab,), -1, dtype=torch.long, device=device)
        for slot, token_id in enumerate(self.row_ids):
            self.lookup[token_id] = slot
        self.delta = torch.zeros(
            (max(1, len(self.row_ids)), hidden), dtype=torch.float32, device=device
        ).requires_grad_(True)
        self.enabled = True
        self._handle = embedding.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if not self.enabled or not self.row_ids:
            return output
        slots = self.lookup[inputs[0]]
        mask = slots >= 0
        if not bool(mask.any()):
            return output
        out = output.clone()
        out[mask] = out[mask] + self.delta[slots[mask]].to(out.dtype)
        return out

    def close(self) -> None:
        self._handle.remove()


# --------------------------------------------------------------------------
# hidden-state helpers
# --------------------------------------------------------------------------
@torch.no_grad()
def prompt_hidden(model, tok, prompts: Sequence[str], device, batch_size: int) -> torch.Tensor:
    """Final hidden state at each prompt's last real token."""
    rows: List[torch.Tensor] = []
    old = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = [str(x) for x in prompts[start : start + batch_size]]
            enc = tok(chunk, padding=True, return_tensors="pt").to(device)
            hidden = model(**enc, output_hidden_states=True).hidden_states[-1]
            for row, length in enumerate(enc["attention_mask"].sum(dim=1).tolist()):
                rows.append(hidden[row, int(length) - 1, :].detach().float().cpu())
    finally:
        tok.padding_side = old
    return torch.stack(rows) if rows else torch.empty((0, 0))


def answer_positions(tok, prompt: str, answer: str) -> Tuple[List[int], List[int]]:
    """Token ids of the answer and the hidden-state index that predicts each.

    Position j of the answer is predicted by the state at index
    len(prompt) + j - 1 in the teacher-forced sequence.
    """
    prompt_ids = gagd.token_ids_for_text(tok, prompt)
    answer_ids = gagd.token_ids_for_text(tok, gagd.normalize_answer(answer))
    predictors = [len(prompt_ids) + j - 1 for j in range(len(answer_ids))]
    return answer_ids, predictors


@torch.no_grad()
def teacher_forced_states(
    model, tok, prompt: str, answer: str, device
) -> Tuple[torch.Tensor, List[int]]:
    """Hidden states predicting each answer token, plus those token ids."""
    answer_ids, predictors = answer_positions(tok, prompt, answer)
    prompt_ids = gagd.token_ids_for_text(tok, prompt)
    ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=device)
    hidden = model(input_ids=ids, output_hidden_states=True).hidden_states[-1][0]
    return hidden[predictors, :].detach().float().cpu(), answer_ids


@torch.no_grad()
def answer_nll(model, tok, prompt: str, answer: str, device) -> float:
    """Mean teacher-forced NLL over the answer's tokens."""
    answer_ids, predictors = answer_positions(tok, prompt, answer)
    if not answer_ids:
        return float("nan")
    prompt_ids = gagd.token_ids_for_text(tok, prompt)
    ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=device)
    logits = model(input_ids=ids).logits[0].float()
    total = 0.0
    for j, token_id in enumerate(answer_ids):
        total += float(-torch.log_softmax(logits[predictors[j]], dim=-1)[token_id])
    return total / len(answer_ids)


# --------------------------------------------------------------------------
# Stage 0A -- mine contexts where an answer token is already correct
# --------------------------------------------------------------------------
@torch.no_grad()
def mine_token_contexts(
    model, tok, documents: Sequence[str], token_ids: Sequence[int], device,
    per_token_cap: int, max_tokens: int, batch_size: int, hidden_size: int,
) -> Dict[int, torch.Tensor]:
    """Real-text positions where each token id is already the next token.

    These are the contexts the edited row must keep serving. A per-token cap
    stops a common token crowding the basis and starving rare ones.
    """
    remaining = {int(t): int(per_token_cap) for t in token_ids}
    collected: Dict[int, List[torch.Tensor]] = defaultdict(list)
    old = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    scanned = 0
    try:
        for start in range(0, len(documents), batch_size):
            if not any(v > 0 for v in remaining.values()):
                break
            chunk = [str(d) for d in documents[start : start + batch_size]]
            enc = tok(
                chunk, padding=True, truncation=True,
                max_length=int(max_tokens), return_tensors="pt",
            ).to(device)
            hidden = model(**enc, output_hidden_states=True).hidden_states[-1]
            ids, mask = enc["input_ids"], enc["attention_mask"]
            scanned += len(chunk)
            for row in range(ids.shape[0]):
                length = int(mask[row].sum().item())
                for pos in range(length - 1):
                    nxt = int(ids[row, pos + 1].item())
                    if remaining.get(nxt, 0) <= 0:
                        continue
                    collected[nxt].append(hidden[row, pos, :].detach().float().cpu())
                    remaining[nxt] -= 1
            if scanned % 5000 < batch_size:
                filled = sum(1 for v in remaining.values() if v <= 0)
                print(f"    scanned {scanned} docs; {filled}/{len(remaining)} token rows at cap")
    finally:
        tok.padding_side = old
    return {k: torch.stack(v) for k, v in collected.items() if v}


# --------------------------------------------------------------------------
# Stage 0B -- reachability sketch and marker selection
# --------------------------------------------------------------------------
@torch.no_grad()
def reachability_sketch(
    model, tok, writer: EmbeddingRowDelta, prompt: str, subject_slots: Sequence[int],
    probes: int, sigma: float, device, generator: torch.Generator,
) -> torch.Tensor:
    """Hidden-state displacements reachable by perturbing only subject rows.

    Symmetric random probes on the subject's embedding rows, each pushed
    through the frozen transformer. Their span approximates the locally
    reachable subspace, which is what makes a marker writable rather than
    merely orthogonal -- an arbitrary null-space direction may be unreachable
    from these few rows, and the writer would then never converge.
    """
    base = prompt_hidden(model, tok, [prompt], device, 1)[0]
    rows: List[torch.Tensor] = []
    hidden = writer.delta.shape[1]
    saved = writer.delta.detach().clone()
    try:
        for _ in range(int(probes) // 2):
            noise = torch.randn(
                (len(subject_slots), hidden), generator=generator, dtype=torch.float32
            ).to(writer.delta.device) * float(sigma)
            for sign in (1.0, -1.0):          # symmetric: cancels first-order bias
                writer.delta.detach().zero_()
                writer.delta.detach()[list(subject_slots)] = sign * noise
                moved = prompt_hidden(model, tok, [prompt], device, 1)[0]
                rows.append((moved - base).float())
    finally:
        writer.delta.detach().copy_(saved)
    return torch.stack(rows) if rows else torch.empty((0, base.shape[0]))


def orthonormal_basis(rows: torch.Tensor, max_rank: int | None = None) -> torch.Tensor:
    if rows.numel() == 0:
        return torch.empty((0, rows.shape[-1] if rows.ndim == 2 else 0))
    m = rows.float()
    _, s, vh = torch.linalg.svd(m, full_matrices=False)
    tol = s.max().clamp_min(1.0) * max(m.shape) * torch.finfo(torch.float32).eps
    rank = int((s > tol).sum().item())
    if max_rank is not None:
        rank = min(rank, int(max_rank))
    return vh[:rank].contiguous()


def project_out(rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0 or rows.numel() == 0:
        return rows
    return rows - (rows @ basis.T) @ basis


def select_marker(
    reach: torch.Tensor, protected_basis: torch.Tensor, prior_markers: torch.Tensor
) -> Tuple[torch.Tensor, float, float, float]:
    """Most reachable direction that is orthogonal to protection and to peers.

    Returns the unit marker, rho = sigma1(D_perp) / sigma1(D), and BOTH
    absolute singular values. The ratio alone is misleading: it can be high
    while the absolute reachable movement is small, so a subject can look
    controllable and still be unable to move the hidden state appreciably.
    """
    if reach.numel() == 0:
        return torch.empty(0), 0.0, 0.0, 0.0
    sigma_full = float(torch.linalg.svdvals(reach.float())[0])
    residual = project_out(reach.float(), protected_basis)
    if prior_markers.numel():
        residual = project_out(residual, prior_markers)
    if residual.numel() == 0 or float(residual.norm()) == 0.0:
        return torch.empty(0), 0.0, sigma_full, 0.0
    u, s, vh = torch.linalg.svd(residual, full_matrices=False)
    marker = vh[0]
    marker = marker / marker.norm().clamp_min(1e-12)
    rho = float(s[0]) / (sigma_full + 1e-9)
    return marker, rho, sigma_full, float(s[0])


def head_coupling(output_weight: torch.Tensor, marker: torch.Tensor) -> Dict[str, float]:
    """How visible the marker is to the UNEDITED LM head.

    Reported, not constrained: forcing markers away from the head's dominant
    directions would fight the reachability constraint we just satisfied. The
    Stage-1 KL term already penalizes markers that disturb the distribution,
    so this is a diagnostic for deciding whether low-coupling selection is
    worth adding later as an ablation.
    """
    projections = (output_weight.float() @ marker.to(output_weight.device).float()).abs()
    return {
        "c_rms": float(torch.sqrt((projections**2).mean())),
        "c_max": float(projections.max()),
    }


# --------------------------------------------------------------------------
# Stage 2 -- closed-form reader solve
# --------------------------------------------------------------------------
def solve_betas(cross: torch.Tensor, drops: torch.Tensor, ridge: float) -> torch.Tensor:
    """Ridge least squares for A beta = d.

    Joint rather than per-record because several forget facts can share one
    answer row: with cross[m, k] = q_k . h_m, independent betas would assume
    an orthogonality the writer only approximately achieves. Solving jointly
    accounts for the measured interference instead of hoping it is zero.
    """
    a = cross.double()
    d = drops.double()
    lhs = a.T @ a + float(ridge) * torch.eye(a.shape[1], dtype=torch.float64)
    return torch.linalg.solve(lhs, a.T @ d).float()


# --------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(int(a.seed))
    generator = torch.Generator().manual_seed(int(a.seed))
    out_dir = gagd.resolve_output_path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(a.mcf_path).read_text(encoding="utf-8"))
    forget_records, retain_records = sample_official_mcf_records(
        data, int(a.forget_num), int(a.retain_num), int(a.seed)
    )

    ns = argparse.Namespace(
        model_path=a.model_path, dtype=a.dtype,
        device_map=a.device_map, gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    device = gagd.first_device(model)
    embedding = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    hidden_size = int(embedding.weight.shape[1])

    transformer_fingerprint = sum(
        float(p.detach().float().abs().sum())
        for name, p in model.named_parameters()
        if "embed_tokens" not in name and "lm_head" not in name
    )

    # ---- records, subject rows, answer rows -------------------------------
    records: List[Dict[str, Any]] = []
    for index, record in enumerate(forget_records):
        rewrite = record["requested_rewrite"]
        if isinstance(rewrite, list):
            rewrite = rewrite[0]
        subject = str(rewrite["subject"])
        prompt = str(rewrite["prompt"]).format(subject)
        answer = str(rewrite["target_true"]["str"])
        reference = str(rewrite["target_new"]["str"])
        subject_rows = sorted(
            set(gagd.token_ids_for_text(tok, subject))
            | set(gagd.token_ids_for_text(tok, " " + subject))
        )
        subject_rows = [t for t in subject_rows if t not in gagd.special_token_ids(tok)]
        answer_rows = gagd.token_ids_for_text(tok, gagd.normalize_answer(answer))
        records.append(
            {
                "index": index,
                "case_id": int(record.get("case_id", index)),
                "subject": subject,
                "prompt": prompt,
                "answer": answer,
                "reference": reference,
                "subject_rows": subject_rows,
                "answer_rows": answer_rows,
                # Evaluation data. Read ONLY by the post-hoc G_cross diagnostic
                # below, never by marker selection, the writer, or the gate.
                "neighborhood_prompts": [
                    str(x) for x in record.get("neighborhood_prompts", [])
                ],
            }
        )

    row_owners: Dict[int, int] = defaultdict(int)
    for record in records:
        for token_id in record["subject_rows"]:
            row_owners[token_id] += 1
    for record in records:
        record["shared_rows"] = sum(
            1 for t in record["subject_rows"] if row_owners[t] > 1
        )
    edited_subject_rows = sorted({t for r in records for t in r["subject_rows"]})
    edited_answer_rows = sorted({t for r in records for t in r["answer_rows"]})
    print(
        f"{len(records)} forget records | {len(edited_subject_rows)} subject rows "
        f"| {len(edited_answer_rows)} answer rows"
    )

    # ---- Stage 0A: mine protected contexts per answer-token row -----------
    print("\nStage 0A: mining answer-conditioned protected contexts")
    documents = gagd.load_wiki_frequency_documents(
        a.wikidata_dir, int(a.corpus_doc_start), int(a.corpus_docs)
    )
    if not documents:
        raise RuntimeError(
            f"--wikidata-dir {a.wikidata_dir!r} yielded no documents at/after index "
            f"{a.corpus_doc_start}; protected contexts are required for the certificate."
        )
    token_contexts = mine_token_contexts(
        model, tok, documents, edited_answer_rows, device,
        int(a.contexts_per_token), int(a.corpus_max_tokens),
        int(a.batch_size), hidden_size,
    )

    # split fit / gate; the gate slice never informs marker selection
    fit_contexts: Dict[int, torch.Tensor] = {}
    gate_contexts: Dict[int, torch.Tensor] = {}
    for token_id, states in token_contexts.items():
        perm = torch.randperm(states.shape[0], generator=generator)
        n_gate = max(1, int(round(states.shape[0] * float(a.gate_holdout_frac))))
        gate_contexts[token_id] = states[perm[:n_gate]]
        fit_contexts[token_id] = states[perm[n_gate:]]

    def answer_states(rows: Sequence[int], source: Mapping[int, torch.Tensor]) -> torch.Tensor:
        parts = [source[t] for t in rows if t in source and source[t].numel()]
        return torch.cat(parts, dim=0) if parts else torch.empty((0, hidden_size))

    answer_key = lambda rec: tuple(rec["answer_rows"])  # noqa: E731
    protected_fit_basis: Dict[Tuple[int, ...], torch.Tensor] = {}
    for record in records:
        key = answer_key(record)
        if key not in protected_fit_basis:
            protected_fit_basis[key] = orthonormal_basis(
                answer_states(record["answer_rows"], fit_contexts)
            )
    mined_total = sum(int(v.shape[0]) for v in token_contexts.values())
    print(
        f"  {mined_total} contexts over {len(token_contexts)}/{len(edited_answer_rows)} "
        f"answer rows (fit/gate split {1 - a.gate_holdout_frac:.0%}/{a.gate_holdout_frac:.0%})"
    )

    # ---- Stage 0B: reachability + markers ---------------------------------
    print("\nStage 0B: reachability sketch and marker selection")
    writer = EmbeddingRowDelta(embedding, edited_subject_rows, device)
    slot_of = {t: i for i, t in enumerate(writer.row_ids)}
    markers_by_answer: Dict[Tuple[int, ...], List[torch.Tensor]] = defaultdict(list)
    for record in records:
        slots = [slot_of[t] for t in record["subject_rows"] if t in slot_of]
        reach = reachability_sketch(
            model, tok, writer, record["prompt"], slots,
            int(a.reach_probes), float(a.reach_sigma), device, generator,
        )
        key = answer_key(record)
        prior = (
            torch.stack(markers_by_answer[key])
            if markers_by_answer[key] else torch.empty((0, hidden_size))
        )
        marker, rho, sigma_full, sigma_perp = select_marker(
            reach, protected_fit_basis[key], prior
        )
        if marker.numel() == 0:
            raise RuntimeError(
                f"record {record['case_id']}: no reachable direction survives protection"
            )
        # No sign convention is imposed: the reachability probes are symmetric
        # (+dE and -dE), so sum_k dh_k . v cancels to first order and any sign
        # picked from it would be numerical noise. It also does not matter --
        # at init v.dh = 0, and the hinge's gradient drives the delta in
        # whichever direction makes v.dh positive, with -dE equally reachable.
        markers_by_answer[key].append(marker)
        record["marker"] = marker
        record["reachability_rho"] = rho
        record["reach_sigma1"] = sigma_full
        record["reach_sigma1_perp"] = sigma_perp
        record["head_coupling"] = head_coupling(output_layer.weight, marker)

    rhos = sorted(r["reachability_rho"] for r in records)
    sig = sorted(r["reach_sigma1_perp"] for r in records)
    print(
        f"  rho (ratio):        min {rhos[0]:.3f} p10 {rhos[max(0,len(rhos)//10)]:.3f} "
        f"median {rhos[len(rhos)//2]:.3f} max {rhos[-1]:.3f}"
    )
    print(
        f"  sigma1(D_perp) abs: min {sig[0]:.4f} p10 {sig[max(0,len(sig)//10)]:.4f} "
        f"median {sig[len(sig)//2]:.4f} max {sig[-1]:.4f}   (ratio alone can mislead)"
    )
    weak = [r for r in records if r["reachability_rho"] < 0.05]
    if weak:
        print(f"  WARNING: {len(weak)} records with rho < 0.05 are unlikely to write successfully")

    # ---- Stage 1: writer ---------------------------------------------------
    print(f"\nStage 1: writer ({len(writer.row_ids)} embedding rows trainable)")
    writer.enabled = False
    base_hidden = {
        r["index"]: prompt_hidden(model, tok, [r["prompt"]], device, 1)[0] for r in records
    }
    base_logprobs: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for record in records:
            enc = tok([record["prompt"]], return_tensors="pt").to(device)
            logits = model(**enc).logits[0, -1, :].float()
            base_logprobs[record["index"]] = torch.log_softmax(logits, dim=-1).cpu()
    writer.enabled = True

    optimizer = torch.optim.AdamW([writer.delta], lr=float(a.writer_lr), weight_decay=0.0)
    order = list(range(len(records)))
    log_path = out_dir / "stage1_writer_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(1, int(a.writer_steps) + 1):
            perm = torch.randperm(len(order), generator=generator).tolist()
            batch = [records[order[i]] for i in perm[: int(a.writer_batch_size)]]
            optimizer.zero_grad(set_to_none=True)
            l_write = torch.zeros((), device=device)
            l_off = torch.zeros((), device=device)
            l_kl = torch.zeros((), device=device)
            for record in batch:
                enc = tok([record["prompt"]], return_tensors="pt").to(device)
                out = model(**enc, output_hidden_states=True)
                h_new = out.hidden_states[-1][0, -1, :].float()
                marker = record["marker"].to(device)
                delta_h = h_new - base_hidden[record["index"]].to(device)
                along = torch.dot(marker, delta_h)
                l_write = l_write + F.relu(float(a.write_alpha) - along)
                # Normalized by hidden size: unnormalized this term reaches
                # ~2e3 against a write hinge of ~1e1, so it owns the gradient
                # while fighting the displacement the hinge is asking for.
                l_off = l_off + (delta_h - along * marker).pow(2).sum() / delta_h.shape[0]
                logprobs = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
                l_kl = l_kl + F.kl_div(
                    logprobs, base_logprobs[record["index"]].to(device),
                    log_target=True, reduction="sum",
                )
            n = max(1, len(batch))
            loss = (
                float(a.lambda_write) * l_write / n
                + float(a.lambda_off) * l_off / n
                + float(a.lambda_kl) * l_kl / n
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite writer loss at step {step}")
            loss.backward()
            if float(a.writer_grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_([writer.delta], float(a.writer_grad_clip))
            optimizer.step()
            if step % 25 == 0 or step == 1:
                log_file.write(json.dumps({
                    "step": step, "loss": float(loss.detach()),
                    "l_write": float(l_write.detach() / n),
                    "l_off": float(l_off.detach() / n),
                    "l_kl": float(l_kl.detach() / n),
                }) + "\n")
                print(
                    f"  step {step:>4}  minibatch loss {float(loss.detach()):>9.4f}"
                    f"  write {float(l_write.detach()/n):>8.4f}"
                    f"  off {float(l_off.detach()/n):>8.4f}"
                    f"  kl {float(l_kl.detach()/n):>7.4f}"
                )
            if int(a.writer_eval_every) > 0 and (
                step % int(a.writer_eval_every) == 0 or step == int(a.writer_steps)
            ):
                # Fixed set, all records: the only curve that is comparable
                # across steps, since the minibatch above resamples subjects.
                states_now = prompt_hidden(
                    model, tok, [r["prompt"] for r in records], device, int(a.batch_size)
                )
                amps_now = sorted(
                    float(torch.dot(
                        record["marker"], states_now[k] - base_hidden[record["index"]]
                    ))
                    for k, record in enumerate(records)
                )
                mean_hinge = sum(
                    max(0.0, float(a.write_alpha) - x) for x in amps_now
                ) / max(1, len(amps_now))
                log_file.write(json.dumps({
                    "step": step, "fixed_set_eval": True,
                    "A_mean": sum(amps_now) / len(amps_now),
                    "A_min": amps_now[0], "A_median": amps_now[len(amps_now) // 2],
                    "A_max": amps_now[-1], "mean_write_hinge": mean_hinge,
                }) + "\n")
                print(
                    f"    [all {len(amps_now)}]  A mean {sum(amps_now)/len(amps_now):+.3f}"
                    f"  min {amps_now[0]:+.3f}  p10 {amps_now[max(0,len(amps_now)//10)]:+.3f}"
                    f"  median {amps_now[len(amps_now)//2]:+.3f}  max {amps_now[-1]:+.3f}"
                    f"   hinge {mean_hinge:.4f}"
                )

    # ---- Gate --------------------------------------------------------------
    print("\nGate: recomputing separability under the writer")
    # Protection must be recomputed under the writer-edited model: Stage 1
    # changed embedding rows, and a protected context may itself contain one
    # of those tokens. A certificate against Stage-0 states would not bind the
    # model actually being shipped.
    post_contexts = mine_token_contexts(
        model, tok, documents, edited_answer_rows, device,
        int(a.contexts_per_token), int(a.corpus_max_tokens),
        int(a.batch_size), hidden_size,
    )
    post_fit: Dict[int, torch.Tensor] = {}
    post_gate: Dict[int, torch.Tensor] = {}
    for token_id, states in post_contexts.items():
        perm = torch.randperm(states.shape[0], generator=generator)
        n_gate = max(1, int(round(states.shape[0] * float(a.gate_holdout_frac))))
        post_gate[token_id] = states[perm[:n_gate]]
        post_fit[token_id] = states[perm[n_gate:]]

    protected_post_basis: Dict[Tuple[int, ...], torch.Tensor] = {}
    for record in records:
        key = answer_key(record)
        if key not in protected_post_basis:
            protected_post_basis[key] = orthonormal_basis(
                answer_states(record["answer_rows"], post_fit)
            )

    def residual_frac(vec: torch.Tensor, basis: torch.Tensor) -> float:
        norm = float(vec.norm())
        if norm == 0.0:
            return float("nan")
        resid = vec - basis.T @ (basis @ vec) if basis.numel() else vec
        return float((resid.norm() / norm) ** 2)

    gate_rows: List[Dict[str, Any]] = []
    for record in records:
        key = answer_key(record)
        basis = protected_post_basis[key]
        h_new = prompt_hidden(model, tok, [record["prompt"]], device, 1)[0]
        record["h_writer"] = h_new
        # Writer diagnostics: achieved amplitude along the intended marker,
        # what fraction of the displacement stayed on-axis, and the off-axis
        # norm the confinement term was meant to hold down.
        delta_h = h_new - base_hidden[record["index"]]
        amplitude = float(torch.dot(record["marker"], delta_h))
        off_axis = delta_h - amplitude * record["marker"]
        record["write_amplitude"] = amplitude
        record["write_parallel_fraction"] = abs(amplitude) / (float(delta_h.norm()) + 1e-9)
        record["write_off_axis_norm"] = float(off_axis.norm())
        held = answer_states(record["answer_rows"], post_gate)
        r_f = residual_frac(h_new, basis)
        r_n = (
            float(sum(residual_frac(h, basis) for h in held) / held.shape[0])
            if held.numel() else float("nan")
        )
        gate_rows.append({
            "case_id": record["case_id"], "answer": record["answer"],
            "R_f": r_f, "R_n": r_n, "G": r_f - r_n,
            "rho": record["reachability_rho"],
            "write_amplitude": record["write_amplitude"],
            "write_parallel_fraction": record["write_parallel_fraction"],
            "write_off_axis_norm": record["write_off_axis_norm"],
            "subject_rows_shared_with_other_records": record["shared_rows"],
        })

    amps = sorted(row["write_amplitude"] for row in gate_rows)
    print(
        f"  write amplitude A: min {amps[0]:+.3f}  p10 {amps[max(0,len(amps)//10)]:+.3f}  "
        f"median {amps[len(amps)//2]:+.3f}  max {amps[-1]:+.3f}   (target alpha={a.write_alpha})"
    )
    gaps = sorted(row["G"] for row in gate_rows if row["G"] == row["G"])
    passed = [g for g in gaps if g >= float(a.gate_min)]
    pass_frac = len(passed) / max(1, len(gaps))
    gate_summary = {
        "gate_min": float(a.gate_min),
        "gate_pass_frac_required": float(a.gate_pass_frac),
        "mean": sum(gaps) / max(1, len(gaps)),
        "median": gaps[len(gaps) // 2] if gaps else float("nan"),
        "min": gaps[0] if gaps else float("nan"),
        "p10": gaps[max(0, len(gaps) // 10)] if gaps else float("nan"),
        "pass_fraction": pass_frac,
        "per_record": gate_rows,
    }
    print(
        f"  G_wiki (held-out corpus, NOT the cross number) "
        f"mean {gate_summary['mean']:+.4f}  median {gate_summary['median']:+.4f}"
        f"  p10 {gate_summary['p10']:+.4f}  min {gate_summary['min']:+.4f}"
        f"  pass {pass_frac:.0%}"
    )
    # --- reader selectivity: what Stage 2 ACTUALLY depends on -------------
    # G = R_f - R_n compares total nullspace ENERGY, but the reader uses a
    # single direction. Those can disagree completely: if (I-P)h_f = 10 q1 and
    # (I-P)h_n = 10 q2 with q1 orthogonal to q2, then R_f = R_n so G = 0, yet
    # q1.h_f = 10 while q1.h_n = 0 -- perfectly editable. So G is not the
    # feasibility statistic; kappa is.
    #
    # With S = |q.h_f| and L = max_p |q.h_p|, choosing beta = d/S to drop the
    # forget logit by d bounds the worst protected shift at d * kappa, where
    # kappa = L/S. kappa = 0.02 means a 10-logit suppression moves protected
    # logits by at most 0.2; kappa near 1 means the reader cannot tell the
    # contexts apart.
    for record in records:
        basis = protected_post_basis[answer_key(record)]
        h_new = record["h_writer"]
        q = h_new - basis.T @ (basis @ h_new) if basis.numel() else h_new.clone()
        norm = float(q.norm())
        record["q"] = q / max(norm, 1e-12)
        record["q_norm"] = norm
        record["cos_marker_q"] = float(torch.dot(record["marker"], record["q"]))
        record["amplitude_relative"] = record["write_amplitude"] / (
            float(h_new.norm()) + 1e-9
        )

    def leakage(direction: torch.Tensor, signal_state: torch.Tensor,
                others: torch.Tensor) -> Tuple[float, float, float]:
        signal = abs(float(torch.dot(direction, signal_state)))
        if others.numel() == 0:
            return signal, float("nan"), float("nan")
        worst = float((others @ direction).abs().max())
        return signal, worst, worst / (signal + 1e-9)

    for record in records:
        held = answer_states(record["answer_rows"], post_gate)
        s_q, l_q, k_q = leakage(record["q"], record["h_writer"], held)
        s_v, l_v, k_v = leakage(record["marker"], record["h_writer"], held)
        record.update({
            "S_q": s_q, "L_wiki_q": l_q, "kappa_wiki_q": k_q,
            "S_v": s_v, "L_wiki_v": l_v, "kappa_wiki_v": k_v,
        })

    def dist(values: Sequence[float]) -> Dict[str, float]:
        vals = sorted(v for v in values if isinstance(v, float) and math.isfinite(v))
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals), "min": vals[0], "p10": vals[max(0, len(vals) // 10)],
            "median": vals[len(vals) // 2],
            "p90": vals[min(len(vals) - 1, (9 * len(vals)) // 10)], "max": vals[-1],
        }

    def show(label: str, d: Dict[str, float]) -> None:
        if not d.get("n"):
            print(f"  {label:<26} (no data)")
            return
        print(
            f"  {label:<26} min {d['min']:+.4f}  p10 {d['p10']:+.4f}  "
            f"median {d['median']:+.4f}  p90 {d['p90']:+.4f}  max {d['max']:+.4f}"
        )

    reader_summary = {
        "definition": (
            "S = |q.h_forget|; L = max over protected states of |q.h_p|; "
            "kappa = L/S bounds the worst protected logit shift at d*kappa "
            "when suppressing the forget logit by d. This, not G, is what "
            "Stage 2 feasibility depends on."
        ),
        "amplitude_relative": dist([r["amplitude_relative"] for r in records]),
        "cos_marker_q": dist([r["cos_marker_q"] for r in records]),
        "S_q": dist([r["S_q"] for r in records]),
        "kappa_wiki_q": dist([r["kappa_wiki_q"] for r in records]),
        "kappa_wiki_v": dist([r["kappa_wiki_v"] for r in records]),
    }
    print("\n  reader selectivity (what Stage 2 actually uses):")
    show("A/||h|| (relative)", reader_summary["amplitude_relative"])
    show("cos(v, q)", reader_summary["cos_marker_q"])
    show("S = |q.h_forget|", reader_summary["S_q"])
    show("kappa_wiki (realized q)", reader_summary["kappa_wiki_q"])
    show("kappa_wiki (intended v)", reader_summary["kappa_wiki_v"])
    gate_summary["reader_selectivity"] = reader_summary

    # --- diagnostic only: G against REAL neighborhood prompts -------------
    # The gate above uses held-out Wikipedia, so its G is Wiki-vs-Wiki and is
    # NOT comparable to the -0.003 that killed the LM-head-only design, which
    # was protect-on-corpus / test-on-CounterFact-neighborhoods. This computes
    # that cross quantity under the writer. It runs after Stage 1, cannot
    # influence training, markers, the gate or any hyperparameter, and exists
    # so the two numbers are never conflated again.
    cross_rows: List[Dict[str, Any]] = []
    if a.eval_overlap_audit:
        for record in records:
            prompts = record["neighborhood_prompts"]
            if len(prompts) < 3:
                continue
            basis = protected_post_basis[answer_key(record)]
            neigh = prompt_hidden(model, tok, prompts, device, int(a.batch_size))
            r_n_cross = float(
                sum(residual_frac(h, basis) for h in neigh) / neigh.shape[0]
            )
            r_f = residual_frac(record["h_writer"], basis)
            s_q, l_q, k_q = leakage(record["q"], record["h_writer"], neigh)
            s_v, l_v, k_v = leakage(record["marker"], record["h_writer"], neigh)
            record["kappa_cross_q"] = k_q
            record["kappa_cross_v"] = k_v
            cross_rows.append({
                "case_id": record["case_id"],
                "R_f": r_f,
                "R_n_neighborhood": r_n_cross,
                "G_cross": r_f - r_n_cross,
                "S_q": s_q, "L_cross_q": l_q, "kappa_cross_q": k_q,
                "S_v": s_v, "L_cross_v": l_v, "kappa_cross_v": k_v,
            })
    cross_summary: Dict[str, Any] = {"ran": bool(cross_rows), "label": "diagnostic_only_G_cross"}
    if cross_rows:
        vals = sorted(row["G_cross"] for row in cross_rows)
        cross_summary.update({
            "definition": (
                "R_f - R_neighborhood under the writer, protection basis from "
                "mined corpus. Comparable to the pre-writer cross measurement; "
                "the gate's G_wiki is NOT."
            ),
            "records": len(vals),
            "mean": sum(vals) / len(vals),
            "median": vals[len(vals) // 2],
            "min": vals[0],
            "p10": vals[max(0, len(vals) // 10)],
            "per_record": cross_rows,
        })
        cross_summary["kappa_cross_q"] = dist([r["kappa_cross_q"] for r in cross_rows])
        cross_summary["kappa_cross_v"] = dist([r["kappa_cross_v"] for r in cross_rows])
        print(
            f"  [diagnostic] G_cross vs real neighborhoods: mean {cross_summary['mean']:+.4f}"
            f"  median {cross_summary['median']:+.4f}  p10 {cross_summary['p10']:+.4f}"
            f"  min {cross_summary['min']:+.4f}"
        )
        print("\n  DECISION NUMBER -- reader selectivity vs REAL neighborhoods:")
        show("kappa_cross (realized q)", cross_summary["kappa_cross_q"])
        show("kappa_cross (intended v)", cross_summary["kappa_cross_v"])
        med = cross_summary["kappa_cross_q"].get("median")
        if med is not None and math.isfinite(med):
            print(
                f"    -> suppressing a forget logit by d moves the worst tested "
                f"neighborhood logit by <= {med:.4f}*d at the median record."
            )
    gate_summary["diagnostic_only_G_cross"] = cross_summary

    # --- does shared-row contention explain the failures? -----------------
    def pearson(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, int]:
        # Drops non-finite pairs first: G is NaN for any record whose answer
        # rows got no held-out gate contexts, and a single NaN would otherwise
        # propagate through the sums and silently return NaN for everything.
        pairs = [
            (float(x), float(y))
            for x, y in zip(xs, ys)
            if math.isfinite(float(x)) and math.isfinite(float(y))
        ]
        n = len(pairs)
        if n < 3:
            return float("nan"), n
        mx = sum(x for x, _ in pairs) / n
        my = sum(y for _, y in pairs) / n
        num = sum((x - mx) * (y - my) for x, y in pairs)
        dx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
        dy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
        if dx <= 0 or dy <= 0:
            return float("nan"), n
        return num / (dx * dy), n

    shared = [row["subject_rows_shared_with_other_records"] for row in gate_rows]
    corr_amp, n_amp = pearson(shared, [row["write_amplitude"] for row in gate_rows])
    corr_gap, n_gap = pearson(shared, [row["G"] for row in gate_rows])
    gate_summary["shared_row_contention"] = {
        "definition": (
            "One delta exists per edited token row, so records sharing a "
            "subject token impose competing constraints on the same "
            "parameter. Negative correlation with A or G localizes failures "
            "to BPE collisions rather than to optimization."
        ),
        "corr_shared_rows_vs_amplitude": corr_amp,
        "corr_shared_rows_vs_amplitude_n": n_amp,
        "corr_shared_rows_vs_G": corr_gap,
        "corr_shared_rows_vs_G_n": n_gap,
        "records_with_shared_rows": sum(1 for x in shared if x > 0),
        "shared_rows_min": min(shared) if shared else 0,
        "shared_rows_max": max(shared) if shared else 0,
    }
    print(
        f"  shared-row contention: {gate_summary['shared_row_contention']['records_with_shared_rows']}"
        f"/{len(shared)} records share subject rows | "
        f"corr(shared, A) {corr_amp:+.3f} (n={n_amp})"
        f"  corr(shared, G) {corr_gap:+.3f} (n={n_gap})"
    )

    # Saved before the gate exit: a failed gate is exactly when these are most
    # needed, and Stage 1 is the expensive part to reproduce.
    torch.save(
        {
            "writer_row_ids": writer.row_ids,
            "writer_delta": writer.delta.detach().cpu(),
            "markers": {r["case_id"]: r["marker"] for r in records},
            "realized_readers": {r["case_id"]: r["q"] for r in records},
            "h_writer": {r["case_id"]: r["h_writer"] for r in records},
            "h_base": {r["case_id"]: base_hidden[r["index"]] for r in records},
        },
        out_dir / "stage1_writer.pt",
    )
    print(f"\n  writer state saved: {out_dir / 'stage1_writer.pt'}")

    gagd.write_json(out_dir / "gate_report.json", gate_summary)
    if pass_frac < float(a.gate_pass_frac):
        print(
            f"\nGATE FAILED: {pass_frac:.0%} of records reached G >= {a.gate_min}, "
            f"required {a.gate_pass_frac:.0%}. The writer did not create a readable "
            f"signature; Stage 2 cannot certify suppression. See {out_dir/'gate_report.json'}."
        )
        raise SystemExit(2)

    # ---- Stage 2: certified reader ----------------------------------------
    print("\nStage 2: closed-form orthogonal reader")
    # constraints per (record, answer position) grouped by the row they edit
    per_row: Dict[int, Dict[str, List[Any]]] = defaultdict(
        lambda: {"states": [], "drops": [], "owners": []}
    )
    for record in records:
        states, answer_ids = teacher_forced_states(
            model, tok, record["prompt"], record["answer"], device
        )
        nll_true = answer_nll(model, tok, record["prompt"], record["answer"], device)
        nll_ref = answer_nll(model, tok, record["prompt"], record["reference"], device)
        needed = max(0.0, (nll_ref - nll_true) + float(a.forget_margin))
        record["nll_true_pre"] = nll_true
        record["nll_reference"] = nll_ref
        record["required_drop"] = needed
        for j, token_id in enumerate(answer_ids):
            per_row[int(token_id)]["states"].append(states[j])
            per_row[int(token_id)]["drops"].append(needed)
            per_row[int(token_id)]["owners"].append(record["index"])

    index_of = {r["index"]: r for r in records}
    row_deltas: Dict[int, torch.Tensor] = {}
    crosstalk_report: List[Dict[str, Any]] = []
    for token_id, payload in per_row.items():
        owners = sorted({o for o in payload["owners"]})
        readers = torch.stack([index_of[o]["q"] for o in owners])
        states = torch.stack(payload["states"])
        drops = torch.tensor(payload["drops"], dtype=torch.float32)
        cross = states @ readers.T                      # A[m, k] = q_k . h_m
        betas = solve_betas(cross, drops, float(a.reader_ridge))
        row_deltas[token_id] = -(betas.unsqueeze(1) * readers).sum(dim=0)
        off = cross.clone()
        chis: List[float] = []
        for m, owner in enumerate(payload["owners"]):
            if owner in owners:
                col = owners.index(owner)
                diag = abs(float(cross[m, col]))
                other = float(cross[m].abs().sum()) - diag
                chis.append(other / (diag + 1e-9))
                off[m, col] = 0.0
        crosstalk_report.append({
            "token_id": int(token_id),
            "token": tok.decode([int(token_id)]),
            "records": [int(index_of[o]["case_id"]) for o in owners],
            "constraints": int(states.shape[0]),
            "diag_mean": float(torch.diagonal(cross[: len(owners), : len(owners)]).mean())
            if cross.shape[0] >= len(owners) else float("nan"),
            "offdiag_abs_max": float(off.abs().max()) if off.numel() else 0.0,
            "beta_abs_max": float(betas.abs().max()),
            "chi_mean": float(sum(chis) / len(chis)) if chis else float("nan"),
            "chi_max": max(chis) if chis else float("nan"),
        })

    base_rows = {t: output_layer.weight[t].detach().clone() for t in row_deltas}
    scales = [float(x) for x in a.reader_backtrack]
    chosen_scale, final_eval = None, None
    for scale in scales:
        with torch.no_grad():
            for token_id, delta in row_deltas.items():
                output_layer.weight[token_id] = (
                    base_rows[token_id] + scale * delta.to(output_layer.weight.dtype).to(device)
                )
        margins = []
        for record in records:
            nll_true = answer_nll(model, tok, record["prompt"], record["answer"], device)
            margins.append(nll_true - record["nll_reference"])
        worst = min(margins)
        print(f"  scale {scale:>5.2f}  worst margin {worst:+.4f}  (need >= {a.forget_margin - 1e-6:+.4f})")
        if worst >= float(a.forget_margin) - 1e-6:
            chosen_scale, final_eval = scale, margins
            break
    if chosen_scale is None:
        chosen_scale, final_eval = scales[-1], margins
        print(f"  no scale met the margin; keeping {chosen_scale}")

    # ---- certificate -------------------------------------------------------
    # Three numbers, because a single absolute threshold is misleading. The
    # orthogonality is exact in real arithmetic, but |dW . h_p| scales with
    # ||dW||, so a fixed epsilon spuriously fails after backtracking and is
    # meaninglessly tight at small scales. Report the raw logit shift, the
    # scale-free orthogonality quality, and the shift relative to the logit
    # magnitude actually being protected -- the last is what decides whether
    # behavior is preserved.
    worst_shift = 0.0
    worst_relative_orth = 0.0
    worst_vs_logit = 0.0
    for token_id, delta in row_deltas.items():
        protected = torch.cat(
            [x for x in (post_fit.get(token_id), post_gate.get(token_id)) if x is not None and x.numel()],
            dim=0,
        ) if (post_fit.get(token_id) is not None or post_gate.get(token_id) is not None) else torch.empty((0, hidden_size))
        if protected.numel() == 0:
            continue
        applied = chosen_scale * delta
        shifts = (protected @ applied).abs()
        worst_shift = max(worst_shift, float(shifts.max()))
        norms = protected.norm(dim=1).clamp_min(1e-9) * applied.norm().clamp_min(1e-9)
        worst_relative_orth = max(worst_relative_orth, float((shifts / norms).max()))
        base_logits = (protected @ base_rows[token_id].detach().float().cpu()).abs()
        worst_vs_logit = max(
            worst_vs_logit, float(shifts.max() / base_logits.mean().clamp_min(1e-9))
        )

    post_fingerprint = sum(
        float(p.detach().float().abs().sum())
        for name, p in model.named_parameters()
        if "embed_tokens" not in name and "lm_head" not in name
    )
    transformer_delta = abs(post_fingerprint - transformer_fingerprint)

    # ---- audits ------------------------------------------------------------
    subject_row_set = set(edited_subject_rows)
    training_audit = {
        "edited_subject_rows": len(edited_subject_rows),
        "edited_answer_rows": len(edited_answer_rows),
        "forget_prompts_touching_edited_subject_rows": sum(
            1 for r in records
            if subject_row_set & set(gagd.token_ids_for_text(tok, r["prompt"]))
        ),
    }
    eval_audit: Dict[str, Any] = {"ran": False}
    if a.eval_overlap_audit:
        # Post-hoc only. Nothing below may influence markers, losses, the gate
        # or hyperparameters -- it exists to bound the locality claim honestly.
        touched, total = 0, 0
        for record in retain_records:
            rewrite = record["requested_rewrite"]
            if isinstance(rewrite, list):
                rewrite = rewrite[0]
            prompt = str(rewrite["prompt"]).format(str(rewrite["subject"]))
            total += 1
            if subject_row_set & set(gagd.token_ids_for_text(tok, prompt)):
                touched += 1
        eval_audit = {
            "ran": True,
            "label": "diagnostic_only_eval_overlap",
            "retain_prompts": total,
            "retain_prompts_touching_edited_subject_rows": touched,
            "fraction": touched / max(1, total),
            "note": "post-hoc analysis only; did not influence training, markers, or gating",
        }

    summary = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "model_path": str(Path(a.model_path).resolve()),
        "seed": int(a.seed),
        "forget_num": len(records),
        "invariants": {
            "transformer_parameter_absdiff": transformer_delta,
            "transformer_parameters_changed": bool(transformer_delta > 1e-3),
            "max_protected_logit_shift": worst_shift,
            "max_relative_orthogonality_error": worst_relative_orth,
            "max_shift_over_mean_protected_logit": worst_vs_logit,
            "certificate_eps": float(a.certificate_eps),
            "certificate_holds": bool(worst_vs_logit < float(a.certificate_eps)),
            "certificate_definition": (
                "max |dW_y . h_p| divided by the mean |W_y . h_p| over protected "
                "states: the protected logit shift relative to the logit being "
                "protected. Scale-aware, unlike a bare absolute threshold, since "
                "|dW . h_p| grows with ||dW|| under backtracking."
            ),
        },
        "stage0": {
            "documents_scanned": len(documents),
            "contexts_mined": mined_total,
            "answer_rows_covered": len(token_contexts),
            "gate_holdout_frac": float(a.gate_holdout_frac),
            "reach_probes": int(a.reach_probes),
        },
        "gate": gate_summary,
        "stage2": {
            "selected_scale": chosen_scale,
            "reader_ridge": float(a.reader_ridge),
            "worst_final_margin": min(final_eval) if final_eval else None,
            "crosstalk": crosstalk_report,
        },
        "audits": {"training_safe": training_audit, "diagnostic_only_eval_overlap": eval_audit},
        "per_record": [
            {
                "case_id": r["case_id"], "subject": r["subject"], "answer": r["answer"],
                "reachability_rho": r["reachability_rho"],
                "head_coupling": r["head_coupling"],
                "cos_marker_q": r["cos_marker_q"],
                "q_norm": r["q_norm"],
                "write_amplitude": r.get("write_amplitude"),
                "write_parallel_fraction": r.get("write_parallel_fraction"),
                "write_off_axis_norm": r.get("write_off_axis_norm"),
                "required_drop": r["required_drop"],
                "nll_true_pre": r["nll_true_pre"],
                "nll_reference": r["nll_reference"],
                "final_margin": final_eval[i] if final_eval else None,
            }
            for i, r in enumerate(records)
        ],
    }
    gagd.write_json(out_dir / "marker_write_read_summary.json", summary)

    print("\n" + "=" * 68)
    print(f"  transformer parameters changed : {transformer_delta:.3e}  "
          f"({'CHANGED' if transformer_delta > 1e-3 else 'UNCHANGED'})")
    print(f"  max |dW_y . h_protected|       : {worst_shift:.3e}  (absolute logit shift)")
    print(f"  relative orthogonality error   : {worst_relative_orth:.3e}  (scale-free)")
    print(f"  shift / mean protected logit   : {worst_vs_logit:.3e}  "
          f"({'OK' if worst_vs_logit < float(a.certificate_eps) else 'CERTIFICATE VIOLATED'})")
    print("=" * 68)

    if a.save_checkpoint:
        ckpt = out_dir / "checkpoint"
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        print(f"checkpoint: {ckpt}")
    writer.close()
    print(f"summary: {out_dir / 'marker_write_read_summary.json'}")


if __name__ == "__main__":
    main()
