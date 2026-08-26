#!/usr/bin/env python3
"""Can an LM-head row edit forget one fact without breaking its neighbors?

MCF's Eff and Spe read the SAME vocabulary row in opposite directions. For
"The native language of Gautier de Coincy is" -> French, Eff wants the French
row suppressed; the record's neighborhood prompts ("Henri Barbusse is a native
speaker of") are other French speakers, and Spe scores them as
NLL(target_true) < NLL(target_new), so Spe wants that same row intact. The
prompts differ only in subject.

A row delta d changes a logit by d.h, so the architecture can only satisfy
both if d can be orthogonal to the neighborhood prompts' hidden states while
still having d.h_forget != 0 -- i.e. if h_forget has a component OUTSIDE the
span of the states the row must keep serving. This script measures exactly
that component.

The raw residual is not interpretable on its own: in 3072 dimensions a ~10
vector span leaves nearly any vector looking orthogonal, so a high residual
proves nothing. The control is leave-one-out over the neighborhood prompts
themselves -- states that BY CONSTRUCTION belong to the protected population.
Separability requires the forget prompt to sit measurably further outside the
span than a held-out neighbor does:

    forget_residual >> neighbor_loo_residual   -> separable, tune protected-rank
    forget_residual ~= neighbor_loo_residual   -> not separable at row level

Two protection sources are reported:

  neighborhood (oracle)  the record's real neighborhood prompts. DIAGNOSTIC
                         ONLY -- these are evaluation data. Nothing here fits
                         or selects anything; it is measured to establish the
                         best case any protection set could achieve.
  corpus (deployable)    positions in real Wikipedia text where the sensitive
                         answer is already the correct next token. Uses no
                         evaluation data, so a protection set built this way
                         is legitimate for training. Comparing it against the
                         oracle says whether the deployable proxy is adequate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

import gagd_compare as gagd
from mcf_sampling import sample_official_mcf_records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--mcf-path", default="data/multi_counterfact.json")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument(
        "--neighborhood-oracle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Measure against the real neighborhood prompts. Evaluation data, "
            "diagnostic only -- never fits or selects anything."
        ),
    )
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument(
        "--corpus-docs",
        type=int,
        default=50000,
        help=(
            "Upper bound on documents scanned for contexts where the answer is "
            "already the correct next token. Scanning stops as soon as every "
            "answer has hit --corpus-contexts-per-answer, so common answers "
            "cost little and this bound is only paid for rare ones. 0 to skip."
        ),
    )
    p.add_argument(
        "--corpus-doc-start",
        type=int,
        default=20,
        help="Must stay >= 20: official PPL is scored on documents [:20].",
    )
    p.add_argument("--corpus-max-tokens", type=int, default=256)
    p.add_argument("--corpus-contexts-per-answer", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    a = p.parse_args(list(argv) if argv is not None else None)
    if a.forget_num <= 0 or a.batch_size <= 0:
        p.error("forget-num and batch-size must be positive")
    if a.corpus_doc_start < 20:
        p.error("corpus-doc-start must be >= 20 to stay disjoint from official PPL text")
    return a


def residual_fraction(vector: torch.Tensor, basis: torch.Tensor) -> float:
    """Fraction of ``vector``'s energy outside the row space of ``basis``.

    1.0 = fully orthogonal to the protected population (a delta can act on
    this vector while leaving that population untouched).
    0.0 = fully inside it (any delta preserving the population also leaves
    this vector unchanged).
    """
    if basis.numel() == 0 or vector.numel() == 0:
        return float("nan")
    v = vector.to(torch.float32)
    norm = torch.linalg.vector_norm(v)
    if float(norm) == 0.0:
        return float("nan")
    # Orthonormal basis for the span; rank-revealing so duplicate/collinear
    # protected states do not inflate the apparent subspace.
    q = torch.linalg.svd(basis.to(torch.float32), full_matrices=False)[2]
    rank = int(torch.linalg.matrix_rank(basis.to(torch.float32)).item())
    q = q[:rank]
    projected = q.T @ (q @ v)
    return float((torch.linalg.vector_norm(v - projected) / norm) ** 2)


def leave_one_out_residuals(states: torch.Tensor) -> List[float]:
    """Residual of each protected state against the span of the others.

    The control: these states belong to the protected population by
    construction, so whatever residual they show is the floor produced by
    high dimensionality and finite sample size alone.
    """
    out: List[float] = []
    if states.shape[0] < 3:
        return out
    for i in range(states.shape[0]):
        others = torch.cat([states[:i], states[i + 1 :]], dim=0)
        out.append(residual_fraction(states[i], others))
    return out


@torch.no_grad()
def last_token_hidden(model, tok, prompts: Sequence[str], device, batch_size: int) -> torch.Tensor:
    """Final hidden state at each prompt's last real token."""
    rows: List[torch.Tensor] = []
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = [str(x) for x in prompts[start : start + batch_size]]
            enc = tok(chunk, padding=True, return_tensors="pt").to(device)
            hidden = model(**enc, output_hidden_states=True).hidden_states[-1]
            lengths = enc["attention_mask"].sum(dim=1)
            for row, length in enumerate(lengths.tolist()):
                rows.append(hidden[row, int(length) - 1, :].detach().float().cpu())
    finally:
        tok.padding_side = old_side
    return torch.stack(rows) if rows else torch.empty((0, 0))


@torch.no_grad()
def corpus_answer_contexts(
    model,
    tok,
    documents: Sequence[str],
    answer_first_token: Dict[str, int],
    device,
    max_tokens: int,
    per_answer_cap: int,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    """Hidden states in real text where each answer is already the next token.

    These are the contexts the row must keep serving: naturally occurring
    positions where the model should still predict "French". No evaluation
    data is read, so a protection set built from these is deployable.
    """
    wanted = {int(v): k for k, v in answer_first_token.items()}
    collected: Dict[str, List[torch.Tensor]] = {k: [] for k in answer_first_token}
    remaining = {k: int(per_answer_cap) for k in answer_first_token}

    for start in range(0, len(documents), batch_size):
        if not any(v > 0 for v in remaining.values()):
            break
        chunk = [str(d) for d in documents[start : start + batch_size]]
        enc = tok(
            chunk,
            padding=True,
            truncation=True,
            max_length=int(max_tokens),
            return_tensors="pt",
        ).to(device)
        hidden = model(**enc, output_hidden_states=True).hidden_states[-1]
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        for row in range(ids.shape[0]):
            length = int(mask[row].sum().item())
            for pos in range(length - 1):
                nxt = int(ids[row, pos + 1].item())
                answer = wanted.get(nxt)
                if answer is None or remaining[answer] <= 0:
                    continue
                collected[answer].append(hidden[row, pos, :].detach().float().cpu())
                remaining[answer] -= 1
    return {
        k: (torch.stack(v) if v else torch.empty((0, 0)))
        for k, v in collected.items()
    }


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(int(a.seed))

    data = json.loads(Path(a.mcf_path).read_text(encoding="utf-8"))
    forget, _ = sample_official_mcf_records(
        data, int(a.forget_num), int(a.retain_num), int(a.seed)
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
    model.eval()
    device = gagd.first_device(model)

    records: List[Dict[str, Any]] = []
    for record in forget:
        rewrite = record["requested_rewrite"]
        if isinstance(rewrite, list):
            rewrite = rewrite[0]
        records.append(
            {
                "case_id": int(record.get("case_id", -1)),
                "prompt": str(rewrite["prompt"]).format(str(rewrite["subject"])),
                "target_true": str(rewrite["target_true"]["str"]),
                "neighborhood": [str(x) for x in record.get("neighborhood_prompts", [])],
            }
        )

    corpus_states: Dict[str, torch.Tensor] = {}
    if int(a.corpus_docs) > 0:
        documents = gagd.load_wiki_frequency_documents(
            a.wikidata_dir, int(a.corpus_doc_start), int(a.corpus_docs)
        )
        if not documents:
            print(
                f"WARNING: --wikidata-dir {a.wikidata_dir!r} yielded no documents "
                f"at/after index {a.corpus_doc_start}; skipping the corpus source."
            )
        else:
            first_token: Dict[str, int] = {}
            for row in records:
                answer = row["target_true"]
                ids = gagd.token_ids_for_text(tok, gagd.normalize_answer(answer))
                if ids:
                    first_token[answer] = ids[0]
            print(
                f"Scanning {len(documents)} documents for contexts of "
                f"{len(first_token)} distinct answers..."
            )
            corpus_states = corpus_answer_contexts(
                model,
                tok,
                documents,
                first_token,
                device,
                int(a.corpus_max_tokens),
                int(a.corpus_contexts_per_answer),
                int(a.batch_size),
            )

    per_record: List[Dict[str, Any]] = []
    for row in records:
        forget_state = last_token_hidden(
            model, tok, [row["prompt"]], device, int(a.batch_size)
        )[0]
        entry: Dict[str, Any] = {
            "case_id": row["case_id"],
            "target_true": row["target_true"],
            "prompt": row["prompt"],
        }

        if a.neighborhood_oracle and len(row["neighborhood"]) >= 3:
            states = last_token_hidden(
                model, tok, row["neighborhood"], device, int(a.batch_size)
            )
            loo = leave_one_out_residuals(states)
            entry["neighborhood"] = {
                "count": int(states.shape[0]),
                "forget_residual": residual_fraction(forget_state, states),
                "neighbor_loo_residual_mean": (
                    float(sum(loo) / len(loo)) if loo else None
                ),
            }

        states = corpus_states.get(row["target_true"], torch.empty((0, 0)))
        if states.numel() and states.shape[0] >= 3:
            loo = leave_one_out_residuals(states)
            entry["corpus"] = {
                "count": int(states.shape[0]),
                "forget_residual": residual_fraction(forget_state, states),
                "neighbor_loo_residual_mean": (
                    float(sum(loo) / len(loo)) if loo else None
                ),
            }
        per_record.append(entry)

    summary: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": "mcf_lm_head_row_separability_v1",
        "model_path": str(Path(a.model_path).resolve()),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "records_measured": len(per_record),
        "metric_definition": (
            "forget_residual = fraction of the forget prompt's final hidden "
            "state energy orthogonal to the span of states where the same "
            "answer must remain correct. neighbor_loo_residual_mean is the "
            "same quantity for held-out members of that protected population, "
            "and is the high-dimensionality floor to compare against."
        ),
        "interpretation": (
            "forget_residual >> neighbor_loo_residual: the forget context is "
            "geometrically distinguishable, so a protected-subspace delta can "
            "suppress it while preserving neighbors -- raise --protected-rank. "
            "forget_residual ~= neighbor_loo_residual: the forget prompt is "
            "indistinguishable from legitimate uses of the same answer, so no "
            "row-level edit separates Eff from Spe at any hyperparameter."
        ),
        "per_record": per_record,
    }

    for source in ("neighborhood", "corpus"):
        forget_vals = [
            r[source]["forget_residual"]
            for r in per_record
            if source in r and r[source]["forget_residual"] == r[source]["forget_residual"]
        ]
        loo_vals = [
            r[source]["neighbor_loo_residual_mean"]
            for r in per_record
            if source in r and r[source]["neighbor_loo_residual_mean"] is not None
        ]
        if not forget_vals:
            continue
        forget_mean = sum(forget_vals) / len(forget_vals)
        loo_mean = sum(loo_vals) / len(loo_vals) if loo_vals else float("nan")
        summary[f"{source}_forget_residual_mean"] = forget_mean
        summary[f"{source}_neighbor_loo_residual_mean"] = loo_mean
        summary[f"{source}_separability_gap"] = forget_mean - loo_mean
        summary[f"{source}_records"] = len(forget_vals)

    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nSeparability report: {out}")
    for source in ("neighborhood", "corpus"):
        if f"{source}_forget_residual_mean" not in summary:
            continue
        label = "oracle, diagnostic only" if source == "neighborhood" else "deployable"
        print(f"\n[{source}]  ({label})  n={summary[f'{source}_records']} records")
        print(f"  forget prompt residual        : {summary[f'{source}_forget_residual_mean']:.4f}")
        print(f"  held-out neighbor residual    : {summary[f'{source}_neighbor_loo_residual_mean']:.4f}")
        print(f"  separability gap              : {summary[f'{source}_separability_gap']:+.4f}")


if __name__ == "__main__":
    main()
