#!/usr/bin/env python3
"""Experiment 2: does forgetting one fact damage OTHER facts about the same subject?

The subject-keyed method edits the subject's input-embedding rows, which is the
access path to everything the model knows about that entity -- not just the
sensitive fact. If disabling that path disables it wholesale, the method does
ENTITY-level access suppression rather than FACT-level forgetting, and the
paper's claim has to change accordingly.

MCF's own metrics cannot detect this. Neighborhood prompts use *different*
subjects by construction, so same-subject collateral is invisible in Eff, Gen,
Spe and PPL alike.

Design (no ground-truth labels needed)
-------------------------------------
Probes are MCF ``generation_prompts``: ten per record, all mentioning the
subject, spanning several relations --

    "Danielle Darrieux's mother tongue is"     <- the sensitive relation
    "Danielle Darrieux was born in"            <- a different relation

For each probe we take the base model's top-1 next token, then the edited
model's, and split the probes by what the BASE model predicted:

  * SENSITIVE probes -- base top-1 is the sensitive answer's first token.
    These *should* change. Change here is the intended effect.
  * OTHER probes -- base top-1 is anything else, i.e. some other fact or
    continuation about the same subject. These should *not* change. Change here
    is collateral damage, and is what this script exists to measure.

A third group is the control: neighborhood prompts, about *different* subjects.
The method's central claim is that these are bitwise identical to base, so any
change at all indicates a broken claim or a broken harness. Reported as
``control_changed``, which must be 0.

``generation_prompts`` and ``neighborhood_prompts`` are held-out probe fields
that training never reads; this script only scores an already-trained
checkpoint, so using them here is evaluation, not leakage.

Base and edited models are loaded sequentially rather than together, so this
runs on a single GPU.

    python scripts/diagnose_same_subject_collateral.py \\
      --base-model /path/to/Llama-3.2-3B-Instruct \\
      --edited-model outputs/.../stage1_subject_emb/checkpoint \\
      --training-visible-path outputs/.../protocol/training_visible_mcf_target_true.json \\
      --multi-counterfact data/multi_counterfact.json \\
      --out outputs/.../same_subject_collateral.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

import gagd_compare as gagd


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", required=True)
    p.add_argument("--edited-model", required=True)
    p.add_argument("--training-visible-path", required=True,
                   help="locked forget split; supplies the edited subjects")
    p.add_argument("--multi-counterfact", default="data/multi_counterfact.json")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device-map", default="single")
    p.add_argument("--control-prompts-per-record", type=int, default=4,
                   help="neighborhood prompts sampled per record for the control group")
    return p.parse_args(argv)


def build_probes(
    forget_records: Sequence[Mapping[str, Any]],
    mcf_by_case: Mapping[int, Mapping[str, Any]],
    control_per_record: int,
) -> List[Dict[str, Any]]:
    """Same-subject probes plus different-subject controls."""
    # Every one of the 50 subjects has edited rows, so a control prompt must
    # avoid ALL of them, not merely its own record's. A neighborhood prompt for
    # one record can mention another record's subject, which would show up as a
    # false locality violation.
    all_subjects = []
    for record in forget_records:
        source = mcf_by_case.get(int(record["case_id"]))
        if source is not None:
            all_subjects.append(str(source["requested_rewrite"]["subject"]))

    probes: List[Dict[str, Any]] = []
    skipped_controls = 0
    for record in forget_records:
        case_id = int(record["case_id"])
        source = mcf_by_case.get(case_id)
        if source is None:
            continue
        rewrite = source["requested_rewrite"]
        subject = str(rewrite["subject"])
        sensitive = str(rewrite["target_true"]["str"])

        for text in source.get("generation_prompts") or []:
            text = str(text)
            if subject not in text:
                # Only same-subject probes belong in the treatment group.
                continue
            probes.append({
                "case_id": case_id, "group": "same_subject",
                "subject": subject, "sensitive": sensitive, "prompt": text,
            })
        kept = 0
        for text in source.get("neighborhood_prompts") or []:
            if kept >= control_per_record:
                break
            text = str(text)
            if any(sub in text for sub in all_subjects):
                skipped_controls += 1
                continue
            probes.append({
                "case_id": case_id, "group": "control_other_subject",
                "subject": subject, "sensitive": sensitive, "prompt": text,
            })
            kept += 1
    if skipped_controls:
        print(
            f"excluded {skipped_controls} neighborhood prompt(s) from the control "
            "group because they mention an edited subject"
        )
    return probes


@torch.no_grad()
def next_token_distribution(
    model, tok, prompts: Sequence[str], device, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-1 id and full log-probs at the final position, per prompt."""
    top_ids: List[torch.Tensor] = []
    logps: List[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        enc = tok(batch, padding=True, return_tensors="pt").to(device)
        out = model(**enc, use_cache=False)
        pos = enc["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(batch), device=out.logits.device)
        logits = out.logits[rows, pos, :].float()
        lp = F.log_softmax(logits, dim=-1)
        top_ids.append(lp.argmax(dim=-1).cpu())
        logps.append(lp.cpu())
    if not top_ids:
        return torch.empty(0, dtype=torch.long), torch.empty(0)
    return torch.cat(top_ids), torch.cat(logps, dim=0)


def load(model_path: str, a: argparse.Namespace):
    ns = argparse.Namespace(
        model_path=model_path, dtype=a.dtype, device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok, gagd.first_device(model)


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)

    forget_records = json.loads(
        Path(a.training_visible_path).read_text(encoding="utf-8")
    )
    mcf = json.loads(Path(a.multi_counterfact).read_text(encoding="utf-8"))
    mcf_by_case = {int(r["case_id"]): r for r in mcf}
    probes = build_probes(forget_records, mcf_by_case, int(a.control_prompts_per_record))
    if not probes:
        raise SystemExit("no probes built -- check --multi-counterfact and the split")
    prompts = [p["prompt"] for p in probes]
    n_same = sum(1 for p in probes if p["group"] == "same_subject")
    print(f"{len(probes)} probes: {n_same} same-subject, {len(probes)-n_same} control")

    # Sequential load keeps this to one model in memory at a time.
    model, tok, device = load(a.base_model, a)
    base_top, base_logp = next_token_distribution(
        model, tok, prompts, device, int(a.batch_size)
    )
    # Which probes does the base model answer with the sensitive token?
    sensitive_first: List[int] = []
    for p in probes:
        ids = tok(" " + p["sensitive"], add_special_tokens=False)["input_ids"]
        sensitive_first.append(int(ids[0]) if ids else -1)
    del model
    torch.cuda.empty_cache()

    model, tok, device = load(a.edited_model, a)
    edit_top, edit_logp = next_token_distribution(
        model, tok, prompts, device, int(a.batch_size)
    )
    del model
    torch.cuda.empty_cache()

    rows: List[Dict[str, Any]] = []
    for i, p in enumerate(probes):
        b, e = int(base_top[i]), int(edit_top[i])
        base_is_sensitive = (b == sensitive_first[i])
        kl = float(
            (base_logp[i].exp() * (base_logp[i] - edit_logp[i])).sum()
        )
        rows.append({
            **p,
            "base_top": tok.decode([b]),
            "edited_top": tok.decode([e]),
            "changed": b != e,
            "base_predicted_sensitive": base_is_sensitive,
            "kl_base_to_edited": kl,
            # how far the base answer fell in the edited distribution
            "base_top_rank_after": int(
                (edit_logp[i] > edit_logp[i][b]).sum()
            ),
        })

    def bucket(pred) -> Dict[str, Any]:
        sel = [r for r in rows if pred(r)]
        if not sel:
            return {"n": 0}
        return {
            "n": len(sel),
            "changed": sum(r["changed"] for r in sel),
            "changed_pct": 100.0 * sum(r["changed"] for r in sel) / len(sel),
            "mean_kl": sum(r["kl_base_to_edited"] for r in sel) / len(sel),
            "mean_base_rank_after": sum(r["base_top_rank_after"] for r in sel) / len(sel),
        }

    summary = {
        # intended effect: base answered with the sensitive token, same subject
        "intended_sensitive": bucket(
            lambda r: r["group"] == "same_subject" and r["base_predicted_sensitive"]
        ),
        # COLLATERAL: same subject, but base answered something else
        "collateral_other_facts": bucket(
            lambda r: r["group"] == "same_subject" and not r["base_predicted_sensitive"]
        ),
        # control: different subject entirely -- must be untouched
        "control_other_subject": bucket(
            lambda r: r["group"] == "control_other_subject"
        ),
    }

    payload = {
        "base_model": a.base_model,
        "edited_model": a.edited_model,
        "probe_source": "MCF generation_prompts (same subject) + neighborhood_prompts (control)",
        "interpretation": (
            "intended_sensitive.changed_pct high = the sensitive fact was "
            "forgotten. collateral_other_facts.changed_pct is the experiment: "
            "high means the edit disabled the access path to the ENTITY rather "
            "than to the FACT, so the method does entity-level suppression. "
            "control_other_subject.changed must be 0 -- different subjects "
            "contain none of the edited rows, so any change there means the "
            "locality claim or this harness is wrong."
        ),
        "summary": summary,
        "probes": rows,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\n=============== SAME-SUBJECT COLLATERAL ===============")
    for name, key in (
        ("intended (sensitive fact)", "intended_sensitive"),
        ("COLLATERAL (other facts) ", "collateral_other_facts"),
        ("control (other subjects) ", "control_other_subject"),
    ):
        s = summary[key]
        if not s.get("n"):
            print(f"{name}: no probes")
            continue
        print(
            f"{name}: n={s['n']:4d}  changed={s['changed']:4d} "
            f"({s['changed_pct']:5.1f}%)  meanKL={s['mean_kl']:.4f}  "
            f"base-answer rank after={s['mean_base_rank_after']:.1f}"
        )
    ctrl = summary["control_other_subject"]
    if ctrl.get("n") and ctrl["changed"]:
        print(
            f"\nWARNING: control group changed on {ctrl['changed']} probe(s). "
            "Different-subject prompts contain none of the edited rows and must "
            "be identical to base. Investigate before trusting the other rows."
        )
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
