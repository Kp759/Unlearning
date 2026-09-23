#!/usr/bin/env python3
"""Reworded versions of the 50 trained RWKU probes: the missing Gen column.

Every other benchmark's Gen column measures the TRAINED associations under
wording the router and residuals never saw: MCF official paraphrases, zsRE
rephrases, and MQuAKE AtomicGen (the natural-language question of the same
atomic rewrite, whose training uses the cloze prompt). RWKU's column instead
reports held-out probes, which ask about other facts of the same person. That
is a different axis -- how far intervention reaches beyond the trained
associations -- and reporting it as Gen makes RWKU look like weak forgetting
next to benchmarks measuring something else.

This script builds the like-for-like set: each of the 50 trained probes,
reworded, same fact, same answer, same person.

  Level-2 questions (2 per person)
      The deterministic `paraphrase_query` that built RWKU's official
      held-out paraphrase set, so the rewording is exactly comparable. A
      question the rule cannot rewrite falls back to wrapping the original
      verbatim; those are flagged and excluded, since an unchanged question is
      not a paraphrase.

  Level-1 fill-in-the-blank (8 per person)
      `paraphrase_query` has no rule for cloze sentences and would wrap 40 of
      the 50 probes unchanged, so these are rewritten by a local open-weights
      model (probe_set_backends.LocalBackend) with greedy decoding, then
      frozen. The generator is told the answer so the rewrite asks for the
      same fact; the evaluated model never sees it.

Every candidate must pass, or it is dropped and the reason recorded:
  - different from the original after normalization
  - contains a subject surface V2 accepts (full name or surname), so the test
    changes wording without also removing the lexical cue -- matching MCF,
    whose official paraphrases name the subject
  - does not contain the answer
  - Level-1: exactly one blank, rewritten to the original's blank string

Output is JSON {manifest, rows} with a .sha256 sidecar. Rows keep every
original field, replace `query`, and add `paraphrase_of_source_record_sha256`
so evaluation can map each rewrite to its trained residual row. The frozen
file, not this script, is what the paper cites.

Usage
-----
python -u scripts/build_rwku_same50_paraphrases.py \
  --run-dir outputs/rwku_fact_assoc_router_v2_seed1_direct --data-root data/rwku \
  --output outputs/rwku_fact_assoc_router_v2_seed1_direct/same50_paraphrases.json \
  --l1-backend local --generator-model /path/to/Qwen2.5-14B-Instruct \
  --per-probe 2 --local-files-only --no-download
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


BLANK = re.compile(r"_{2,}")
FALLBACK_PREFIX = "in different words, provide the answer to this question"

L1_TEMPLATE = """Rewrite the following fill-in-the-blank sentence about {subject}.
The rewritten sentence must ask for exactly the same missing fact. The missing
answer is "{answer}".

Rules:
- Use clearly different wording and sentence structure.
- Keep the name "{subject}" exactly as written.
- Keep exactly one blank, written as ___, where the answer goes.
- Do not write the answer or any part of it anywhere in the sentence.

Sentence: {query}

Output exactly {n} lines, numbered 1 to {n}, one rewritten sentence per line.
No preamble, no commentary, no blank lines."""

L2_TEMPLATE = """Rewrite the following question about {subject} so it asks for exactly the
same fact. The answer is "{answer}".

Rules:
- Use clearly different wording and sentence structure.
- Keep the name "{subject}" exactly as written.
- It must remain a question.
- Do not write the answer or any part of it.

Question: {query}

Output exactly {n} lines, numbered 1 to {n}, one rewritten question per line.
No preamble, no commentary, no blank lines."""


def template_hash():
    return hashlib.sha256(
        json.dumps({"l1": L1_TEMPLATE, "l2": L2_TEMPLATE}, sort_keys=True).encode()
    ).hexdigest()


def validate(candidate, row, surfaces, normalize):
    """Return (accepted_text, None) or (None, reason)."""
    text = " ".join(str(candidate).split()).strip().strip('"').strip("'").strip()
    original = str(row["query"])
    if not text:
        return None, "empty"
    if normalize(text) == normalize(original):
        return None, "identical_to_original"
    if text.lower().startswith(FALLBACK_PREFIX):
        return None, "fallback_wrapper_not_a_paraphrase"
    lowered = text.lower()
    if not any(surface.lower() in lowered for surface in surfaces):
        return None, "subject_surface_missing"
    answer = normalize(str(row["answer"]))
    if answer and answer in normalize(text):
        return None, "answer_leaked"
    if str(row.get("level")) == "1":
        original_blanks = BLANK.findall(original)
        found = BLANK.findall(text)
        if len(found) != 1:
            return None, f"blank_count_{len(found)}"
        if original_blanks:
            text = BLANK.sub(original_blanks[0], text)
    return text, None


def build_rows(rows, paraphrases_by_index, method_by_index):
    out = []
    for index, row in enumerate(rows):
        for position, text in enumerate(paraphrases_by_index.get(index, [])):
            value = dict(row)
            value["original_query"] = str(row["query"])
            value["query"] = text
            value["paraphrase_of_source_record_sha256"] = str(row["source_record_sha256"])
            value["paraphrase_method"] = method_by_index[index]
            value["paraphrase_index"] = position
            value["batch50_role"] = "same50_reworded"
            out.append(value)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", default="data/rwku")
    parser.add_argument("--output", required=True)
    parser.add_argument("--l1-backend", choices=("local", "none"), default="local")
    parser.add_argument(
        "--l2-backend", choices=("deterministic", "local"), default="deterministic",
        help="deterministic reuses the rule that built RWKU's official held-out paraphrases",
    )
    parser.add_argument("--generator-model", default="")
    parser.add_argument("--allow-same-model", action="store_true")
    parser.add_argument("--per-probe", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generator-device", default="cuda")
    parser.add_argument("--generator-dtype", default="bfloat16")
    parser.add_argument("--generator-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    import rwku_eval as rwku
    from rwku_batch50 import build_batch_split
    from rwku_data import paraphrase_query
    from rwku_fact_association_embeddings import rwku_subject_surfaces

    run_dir = Path(args.run_dir).resolve()
    manifest = json.loads((run_dir / "association_manifest.json").read_text())
    split = build_batch_split(
        data_root=Path(args.data_root).resolve(), batch_seed=1,
        allow_download=not args.no_download,
    )
    rows = list(split["efficacy_forget"])
    if len(rows) != 50:
        raise SystemExit(f"Expected the 50 trained probes, found {len(rows)}")

    levels = {"1": [], "2": []}
    for index, row in enumerate(rows):
        levels.setdefault(str(row.get("level")), []).append(index)
    surfaces = {i: rwku_subject_surfaces(str(r["subject"])) for i, r in enumerate(rows)}

    llm_requests = []
    for index, row in enumerate(rows):
        level = str(row.get("level"))
        use_llm = (level == "1" and args.l1_backend == "local") or (
            level == "2" and args.l2_backend == "local"
        )
        if use_llm:
            template = L1_TEMPLATE if level == "1" else L2_TEMPLATE
            llm_requests.append((index, template.format(
                subject=row["subject"], answer=row["answer"],
                query=row["query"], n=args.per_probe,
            )))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        payload = [{"index": i, "level": rows[i].get("level"), "prompt": p} for i, p in llm_requests]
        path = output.with_suffix(".dryrun.json")
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({
            "status": "dry_run",
            "level_counts": {k: len(v) for k, v in levels.items()},
            "llm_requests": len(llm_requests),
            "deterministic_level2": args.l2_backend == "deterministic",
            "payload": str(path),
        }, indent=2))
        return 0

    candidates = {i: [] for i in range(len(rows))}
    method = {}
    for index, row in enumerate(rows):
        if str(row.get("level")) == "2" and args.l2_backend == "deterministic":
            candidates[index].append(paraphrase_query(str(row["query"])))
            method[index] = "deterministic_paraphrase_query"

    backend_description = None
    if llm_requests:
        if not args.generator_model:
            raise SystemExit("--generator-model is required for a local backend")
        from probe_set_backends import LocalBackend

        backend = LocalBackend(
            model_path=args.generator_model,
            device=args.generator_device,
            dtype=args.generator_dtype,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            batch_size=args.generator_batch_size,
            local_files_only=args.local_files_only,
            evaluated_model_path=manifest.get("model_path"),
            allow_same_model=args.allow_same_model,
        )
        outputs = backend.generate([p for _, p in llm_requests], expected=args.per_probe)
        for (index, _), items in zip(llm_requests, outputs):
            candidates[index].extend(items)
            method[index] = "local_llm_greedy"
        backend_description = backend.describe()

    accepted, rejected = {}, []
    for index, row in enumerate(rows):
        kept = []
        for candidate in candidates[index]:
            text, reason = validate(candidate, row, surfaces[index], rwku.normalize_text)
            if reason is None and text not in kept:
                kept.append(text)
            elif reason is not None:
                rejected.append({
                    "index": index, "level": str(row.get("level")),
                    "original": str(row["query"]), "candidate": str(candidate),
                    "reason": reason,
                })
        accepted[index] = kept[: args.per_probe]

    out_rows = build_rows(rows, accepted, method)
    covered = {i for i, v in accepted.items() if v}
    by_level = {
        level: {
            "trained_probes": len(indices),
            "with_accepted_paraphrase": sum(i in covered for i in indices),
        }
        for level, indices in levels.items() if indices
    }
    reasons = {}
    for item in rejected:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1

    result = {
        "manifest": {
            "schema_version": "rwku_same50_paraphrases_v1",
            "purpose": (
                "reworded trained probes: RWKU's analogue of MCF/zsRE Gen and "
                "MQuAKE AtomicGen (same fact, unseen wording)"
            ),
            "run_dir": str(run_dir),
            "evaluated_model": manifest.get("model_path"),
            "l1_backend": args.l1_backend,
            "l2_backend": args.l2_backend,
            "per_probe": args.per_probe,
            "prompt_template_sha256": template_hash(),
            "generator": backend_description,
            "coverage_by_level": by_level,
            "paraphrase_count": len(out_rows),
            "rejected_count": len(rejected),
            "rejected_by_reason": reasons,
            "used_for_training_or_prototypes": False,
        },
        "rows": out_rows,
        "rejected": rejected,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    output.write_text(text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (output.parent / f"{output.stem}.sha256").write_text(f"{digest}  {output.name}\n")
    print(json.dumps({
        "status": "same50_paraphrases_complete",
        "output": str(output),
        "sha256": digest,
        "coverage_by_level": by_level,
        "paraphrases": len(out_rows),
        "rejected_by_reason": reasons,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
