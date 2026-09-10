"""One registered exploratory MLP pilot; the completed head protocol is immutable."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from freeze_static_overlap_development import load_protocol, normalized, rewrite, write_new
from mcf_sampling import sample_official_mcf_records
from mcf_shadow_relation_prompts import RELATION_NOUN_PHRASES
from mcf_synthetic_paraphrase_templates import RELATION_ALTERNATE_TEMPLATES
from static_overlap_data import load_bundle, overlap_kind, text_fingerprints
from static_overlap_training import sha256_file

METHOD = "static_overlap_mlp_exploratory_v1"
PLAN = {
    "candidate_layers": [7, 11, 15, 19, 23], "localization_examples": 32,
    "rank": 16, "steps": 200, "check_every": 20, "learning_rate": .003,
    "forget_batch": 8, "retain_batch": 16, "kl_weight": 10., "nll_weight": 10.,
    "relative_delta_cap": .01, "max_training_seconds": 2700,
    "target_probability": 1e-6, "max_length": 512, "seed": 1,
    "retain_nll_budget": .05, "retain_kl_budget": .01,
    "fitting_nll_margin": .01, "fitting_kl_margin": .002,
    "extra_retain_facts": 300, "language_documents": 80,
    "language_words": 40,
}
LANGUAGE_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
LANGUAGE_SHA256 = "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"
LANGUAGE_URL = ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/" + LANGUAGE_REVISION
                + "/wikitext-2-raw-v1/train-00000-of-00001.parquet")

# Distinct syntactic families, authored without consulting official Gen text.
# Development families never enter objective gradients or layer localization.
FIT_QUESTIONS = ("What is the {relation} of {subject}? Answer:",)
DEV_QUESTIONS = (
    "Which {relation} is associated with {subject}? The answer is",
    "Name {subject}'s {relation}. Response:",
)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def language_documents(path):
    """Only WikiText's public TRAIN file; never download its validation/test files.

    Keep at most one paragraph per article, then SHA-order articles, so the
    fitting and development language samples cannot share an article.
    """
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        print(json.dumps({"phase": "download_preservation_training_text", "url": LANGUAGE_URL}), flush=True)
        with urllib.request.urlopen(LANGUAGE_URL, timeout=60) as response, temporary.open("wb") as target:
            while block := response.read(1024 * 1024):
                target.write(block)
        if sha256_file(temporary) != LANGUAGE_SHA256:
            raise ValueError("Public training corpus checksum mismatch")
        temporary.replace(path)
    if sha256_file(path) != LANGUAGE_SHA256:
        raise ValueError("Expected the pinned WikiText-2 raw TRAIN parquet")
    import pyarrow.parquet as pq
    texts = pq.read_table(path, columns=["text"])["text"].to_pylist()
    documents, article, seen = [], None, set()
    for i, text in enumerate(texts):
        if re.fullmatch(r"\s*= [^=]+ =\s*", text):
            article = normalized(text)
        elif article and article not in seen and len(text.split()) >= PLAN["language_words"]:
            documents.append((i, text))
            seen.add(article)
    return sorted(documents, key=lambda r: digest_json([PLAN["seed"], r[0]]))


def fact_key(f):
    return normalized(f["subject"]), normalized(f["relation"])


def authored_prompts(fact, split):
    relation = fact["relation"]
    if relation not in RELATION_NOUN_PHRASES or relation not in RELATION_ALTERNATE_TEMPLATES:
        raise ValueError(f"No authored grammatical variants for {relation}; no prefix fallback")
    noun = RELATION_NOUN_PHRASES[relation]
    if relation == "P1412":
        noun = "language spoken or written"
    names = [fact["subject"]] + (fact.get("aliases", [])[:1] if split == "train" else [])
    for subject in names:
        templates = FIT_QUESTIONS if split == "train" else DEV_QUESTIONS
        for i, template in enumerate(templates):
            yield f"{split}_question_{i}", template.format(subject=subject, relation=noun)
        if split == "train":
            alternatives = RELATION_ALTERNATE_TEMPLATES[relation]
            if relation == "P1412":
                alternatives = ["A language that {} uses is", "{} speaks or writes in"]
            if relation == "P37":
                alternatives = ["An official language of {} is", "{} designates this language as official:"]
            for i, template in enumerate(alternatives):
                yield f"train_cloze_{i}", template.format(subject)


def forbidden_texts(frozen, source, evaluation, final, mcf):
    # Evaluation is read ONLY by this exclusion audit, never passed to fitting.
    official_f, official_r = sample_official_mcf_records(mcf, **{
        "forget_num": frozen["official_evaluation"]["unlearn_num"],
        "retain_num": frozen["official_evaluation"]["retain_num"],
        "seed": frozen["official_evaluation"]["seed"]}, strict=True)
    forbidden = set(text_fingerprints(evaluation)) | set(text_fingerprints(final))
    forbidden |= {normalized(p) for r in official_f for key in ("paraphrase_prompts", "neighborhood_prompts")
                  for p in r.get(key, [])}
    if set(text_fingerprints(source)) & forbidden:
        raise ValueError("Original development text overlaps a frozen final prompt")
    excluded = {fact_key(f) for b in (source, evaluation, final) for f in b["facts"]}
    excluded |= {(normalized(rewrite(r)["subject"]), normalized(rewrite(r)["relation_id"]))
                 for r in official_f + official_r}
    return forbidden, excluded


def build_data(source, mcf, forbidden, excluded, documents, plan=PLAN):
    """Select fresh preservation associations without accessing benchmark probes."""
    facts = list(source["facts"])
    forget = [f for f in facts if f["role"] == "forget"]
    pool, seen = [], set(excluded)
    for record in sorted(mcf[:len(mcf)//2], key=lambda r: digest_json([plan["seed"], r["case_id"]])):
        rr = rewrite(record)
        key = normalized(rr["subject"]), normalized(rr["relation_id"])
        if key in seen or rr["relation_id"] not in RELATION_NOUN_PHRASES:
            continue
        obj = rr["target_true"]
        fact = {"id": f"mlp_retain_{record['case_id']}", "role": "retain",
                "subject": rr["subject"], "relation": rr["relation_id"],
                "object": obj["str"] if isinstance(obj, dict) else obj,
                "aliases": [], "answer_aliases": []}
        # Only direct facts are extracted. No paraphrase/neighborhood field is used.
        pool.append(fact)
        seen.add(key)
    selected = []
    # Fill rare overlap controls before unrelated background associations.
    for kind in ("same_subject_other_relation", "same_subject_same_answer_other_relation",
                 "same_answer_other_association", "same_relation_other_subject"):
        for f in forget:
            matches = [r for r in pool if r not in selected and overlap_kind(f, r) == kind]
            selected.extend(matches[:2])
    selected = (selected + [f for f in pool if f not in selected])[:plan["extra_retain_facts"]]
    if len(selected) < plan["extra_retain_facts"]:
        raise ValueError("Insufficient independent preservation facts")
    facts.extend(selected)
    source_ids = {f["id"] for f in source["facts"]}
    fact_splits = {f["id"]: ("development" if i % 5 == 0 else "train") for i, f in enumerate(selected)}
    authored, used, skipped = [], set(text_fingerprints(source)), 0
    for f in facts:
        splits = ("train", "development") if f["id"] in source_ids else (fact_splits[f["id"]],)
        for split in splits:
            for family, prompt in authored_prompts(f, split):
                keys = {normalized(prompt), normalized(prompt + " " + f["object"])}
                if keys & forbidden or keys & used:
                    skipped += 1
                    continue
                used |= keys
                authored.append({"id": f"mlp_authored_{len(authored)}", "split": split,
                    "role": f["role"], "fact_id": f["id"], "prompt": prompt,
                    "answer": f["object"], "family": family})
    for f in forget:
        for split in ("train", "development"):
            variants = [r for r in authored if r["fact_id"] == f["id"] and r["split"] == split]
            if len(variants) < 2:
                raise ValueError(f"Insufficient disjoint {split} paraphrases for {f['id']}")
    language = []
    for index, document in documents:
        if any(normalized(f["subject"]) in normalized(document) for f in forget):
            continue
        text = " ".join(document.split()[:plan["language_words"]])
        if len(text.split()) < 12 or any(t in normalized(text) for t in forbidden):
            continue
        if normalized(text) in used:
            continue
        used.add(normalized(text))
        language.append({"id": f"mlp_language_{index}", "role": "language", "text": text,
                         "split": "development" if len(language) % 5 == 0 else "train"})
        if len(language) == plan["language_documents"]:
            break
    if len(language) < plan["language_documents"]:
        raise ValueError("Insufficient disjoint ordinary-language anchors")
    coverage = {split: {kind: sum(any(overlap_kind(f, r) == kind for r in facts
                      if r["role"] == "retain" and (r["id"] in source_ids or fact_splits[r["id"]] == split))
                      for f in forget) for kind in ("same_relation_other_subject",
                      "same_answer_other_association", "same_subject_other_relation",
                      "same_subject_same_answer_other_relation", "general_retain")}
                for split in ("train", "development")}
    return {"facts": facts, "authored": authored, "language": language,
            "coverage_forget_facts": coverage, "forget_facts": len(forget),
            "authored_counts": dict(Counter(f"{r['split']}/{r['role']}" for r in authored)),
            "excluded_duplicate_or_final_collisions": skipped,
            "training_text_fingerprints": sorted(used)}


def load_pilot(path):
    path = Path(path).resolve()
    p = json.loads(path.read_text())
    if p.get("method") != METHOD or p.get("exploratory") is not True or p["plan"] != PLAN:
        raise ValueError("Unexpected exploratory pilot contract")
    if sha256_file(p["head_protocol_path"]) != p["head_protocol_sha256"]:
        raise ValueError("Original head protocol changed")
    load_protocol(p["head_protocol_path"])
    for name in ("data", "source_bundle"):
        if sha256_file(p[name]["path"]) != p[name]["sha256"]:
            raise ValueError(f"Frozen pilot input changed: {name}")
    if sha256_file(p["head_manifest"]["path"]) != p["head_manifest"]["sha256"]:
        raise ValueError("Completed head model provenance changed")
    registration = json.loads(Path(p["registration_path"]).read_text())
    if registration["pilot_protocol_path"] != str(path) or registration["pilot_protocol_sha256"] != sha256_file(path):
        raise ValueError("Pilot registration differs")
    return p


def claim_evaluation(protocol_path, checkpoint):
    p = load_pilot(protocol_path)
    manifest = json.loads((Path(checkpoint) / "training_manifest.json").read_text())
    if manifest.get("exploratory_protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("Checkpoint belongs to another exploratory pilot")
    identity = {"pilot_protocol_sha256": sha256_file(protocol_path),
                "checkpoint_export_sha256": sha256_file(Path(checkpoint) / "static_edit_export.json")}
    marker = Path(protocol_path).parent / "exploratory_evaluation_started.json"
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("Exploratory evaluation already bound to another checkpoint")
    else:
        write_new(marker, identity)
    return p, identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-protocol", required=True)
    parser.add_argument("--wikidata-dir", required=True)
    parser.add_argument("--language-parquet", default="data/cache/mlp_wikitext2_train.parquet")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    frozen = load_protocol(args.head_protocol)
    out = Path(args.output_dir).resolve()
    registration = Path(args.head_protocol).resolve().parent / "exploratory_mlp_v1_registered.json"
    if out.exists() or registration.exists():
        raise FileExistsError("Pilot already declared; preserve its artifacts and the completed head result")
    source, _, _ = load_bundle(frozen["files"]["source_bundle"]["path"])
    evaluation, _, _ = load_bundle(frozen["files"]["evaluation_bundle"]["path"], "evaluation")
    final, _, _ = load_bundle(frozen["files"]["final_retention"]["path"], "preservation_test")
    mcf = json.loads(Path(frozen["files"]["mcf"]["path"]).read_text())
    forbidden, excluded = forbidden_texts(frozen, source, evaluation, final, mcf)
    from datasets import load_from_disk
    # Exclude the official PPL slice even though PPL is not run by this pilot.
    ds = load_from_disk(args.wikidata_dir)["train"]
    forbidden |= {normalized(t) for t in ds[:20]["text"] if str(t).strip()}
    documents = language_documents(args.language_parquet)
    data = build_data(source, mcf, forbidden, excluded, documents)
    data["language_source"] = {"dataset": "Salesforce/wikitext", "subset": "wikitext-2-raw-v1",
        "split": "train", "revision": LANGUAGE_REVISION, "sha256": LANGUAGE_SHA256,
        "selection": "one paragraph per article; SHA ordering; no model scores"}
    old_training = json.loads((Path(args.head_protocol).resolve().parent / "training_started.json").read_text())
    head_manifest = Path(old_training["output_dir"]) / "manifest.json"
    original_model = json.loads(head_manifest.read_text())["model_path"]
    out.mkdir(parents=True)
    write_new(out / "source_bundle.json", source)
    write_new(out / "pilot_data.json", data)
    protocol = {"method": METHOD, "exploratory": True, "plan": PLAN,
        "base_model_path": str(Path(original_model).resolve()),
        "head_manifest": {"path": str(head_manifest.resolve()), "sha256": sha256_file(head_manifest)},
        "head_protocol_path": str(Path(args.head_protocol).resolve()),
        "head_protocol_sha256": sha256_file(args.head_protocol),
        "registration_path": str(registration), "declared_utc": datetime.now(timezone.utc).isoformat(),
        "data": {"path": str(out / "pilot_data.json"), "sha256": sha256_file(out / "pilot_data.json")},
        "source_bundle": {"path": str(out / "source_bundle.json"), "sha256": sha256_file(out / "source_bundle.json")},
        "disclosure": "Exploratory follow-up informed by completed head failure; same previously observed final tests, not new confirmatory evidence.",
        "selection_rule": "first scheduled checkpoint passing all train/development forgetting and preservation gates",
        "final_test_use": "exclusion audit, then one fixed-checkpoint evaluation only",
        "development_forget_gradients": False, "preservation_numeric_slack": 0.0}
    write_new(out / "pilot_protocol.json", protocol)
    write_new(registration, {"pilot_protocol_path": str(out / "pilot_protocol.json"),
                            "pilot_protocol_sha256": sha256_file(out / "pilot_protocol.json")})
    print(json.dumps({"phase": "exploratory_pilot_registered", "output": str(out),
        "authored_counts": data["authored_counts"], "coverage_forget_facts": data["coverage_forget_facts"],
        "new_language_anchors": len(data["language"]), "final_model_evaluations": 0}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
