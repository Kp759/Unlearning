#!/usr/bin/env python3
"""Run the MCF V1 overlap-code and classifier-only training preflight.

The process has no official-evaluation input and never constructs an actuator.
It first solves one coherent embedding delta per shared subject subword, then
fits rank-8 contextual classifiers at layers 8/12/16/20.  Layer/checkpoint
selection uses development prompts only.  A threshold is frozen on a disjoint
calibration bank before a third certification bank is opened exactly once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import unicodedata

import torch
from datasets import load_from_disk

import build_mcf_internal_contextual_rewiring_v1_split as split_builder
import gagd_compare as gagd
import mcf_internal_contextual_rewiring_v1_core as core
import mcf_shadow_relation_prompts as relation_prompts
from scoped_span_edit import find_decoder_layers
import sure_canonical_core as canonical


FORBIDDEN_EVALUATION_ENVIRONMENT_VARIABLES = (
    "OFFICIAL",
    "OFFICIAL_DIR",
    "OFFICIAL_MCF_PATH",
    "MCF_OFFICIAL_OUTPUT",
    "RECOVERY",
    "RECOVERY_DIR",
    "RETAIN_PATH",
    "PPL_PATH",
    "ALIAS_EVAL_PATH",
    "ADVERSARIAL_EVAL_PATH",
)

FIT_SCAFFOLDS = (
    "For {subject}, the documented {relation} is",
    "The {relation} associated with {subject} is",
    "Asked about {subject}, its {relation} is",
    "A record gives the {relation} of {subject} as",
)
DEVELOPMENT_SCAFFOLDS = (
    "Identify the {relation} recorded for {subject}:",
    "Regarding {subject}, state the {relation}:",
    "What value is listed as {subject}'s {relation}?",
    "The entry under {relation} for {subject} reads",
)
CALIBRATION_SCAFFOLDS = (
    "Supply the {relation} linked to {subject}:",
    "For the entity {subject}, which {relation} is recorded?",
    "Concerning {subject}, its listed {relation} is",
    "Name the {relation} attributed to {subject}:",
)
CERTIFICATION_SCAFFOLDS = (
    "Consult the record: {subject} has which {relation}?",
    "State what is registered as the {relation} of {subject}:",
    "With respect to {subject}, provide its {relation}:",
    "The requested {relation} value for {subject} is",
)

MANDATORY_CERTIFICATION_KINDS = (
    "same_subject_different_relation",
    "same_relation_different_subject",
    "shared_subject_subword_without_complete_subject",
    "broad_corpus_prompt",
    "writer_off_positive_context",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-visible-path", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--experiment-registry", required=True)
    parser.add_argument("--wikidata-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--candidate-layers", default="8,12,16,20")
    parser.add_argument("--subject-code-rank", type=int, default=8)
    parser.add_argument("--detector-rank", type=int, default=8)
    parser.add_argument("--embedding-ridge-lambda", type=float, default=1e-4)
    parser.add_argument("--embedding-relative-row-cap", type=float, default=0.5)
    parser.add_argument("--embedding-frequency-alpha", type=float, default=0.25)
    parser.add_argument("--code-nearest-key-margin", type=float, default=0.05)
    parser.add_argument("--frequency-doc-start", type=int, default=20)
    parser.add_argument("--frequency-docs", type=int, default=12000)
    parser.add_argument("--corpus-fit-prompts", type=int, default=1000)
    parser.add_argument("--corpus-development-prompts", type=int, default=1000)
    parser.add_argument("--corpus-calibration-prompts", type=int, default=1000)
    parser.add_argument("--corpus-certification-prompts", type=int, default=6000)
    parser.add_argument("--shared-fit-prompts", type=int, default=25)
    parser.add_argument("--shared-development-prompts", type=int, default=25)
    parser.add_argument("--shared-calibration-prompts", type=int, default=25)
    parser.add_argument("--shared-certification-prompts", type=int, default=100)
    parser.add_argument("--wrong-relations-fit", type=int, default=8)
    parser.add_argument("--wrong-relations-other", type=int, default=4)
    parser.add_argument("--same-relation-other-subjects", type=int, default=4)
    parser.add_argument("--classifier-steps", type=int, default=400)
    parser.add_argument("--classifier-check-every", type=int, default=20)
    parser.add_argument("--classifier-lr", type=float, default=0.01)
    parser.add_argument("--classifier-weight-decay", type=float, default=0.01)
    parser.add_argument("--classifier-positive-floor", type=float, default=1.0)
    parser.add_argument("--classifier-negative-ceiling", type=float, default=-1.0)
    parser.add_argument("--classifier-auxiliary-weight", type=float, default=0.5)
    parser.add_argument("--classifier-softmin-temperature", type=float, default=0.1)
    parser.add_argument(
        "--minimum-certification-negative-cells", type=int, default=300000
    )
    parser.add_argument("--minimum-certification-prompts", type=int, default=6000)
    parser.add_argument("--capture-batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single",), default="single")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        args.candidate_layers = [
            int(value.strip()) for value in str(args.candidate_layers).split(",")
        ]
    except ValueError as exc:
        parser.error(f"invalid --candidate-layers: {exc}")
    count_names = (
        "forget_num",
        "subject_code_rank",
        "detector_rank",
        "frequency_docs",
        "corpus_fit_prompts",
        "corpus_development_prompts",
        "corpus_calibration_prompts",
        "corpus_certification_prompts",
        "classifier_steps",
        "classifier_check_every",
        "minimum_certification_negative_cells",
        "minimum_certification_prompts",
        "capture_batch_size",
    )
    if any(int(getattr(args, name)) <= 0 for name in count_names):
        parser.error("all registered counts must be positive")
    if args.seed != 1 or args.forget_num != 50:
        parser.error("V1 implementation is locked to consumed seed 1 / 50 facts")
    if args.candidate_layers != [8, 12, 16, 20]:
        parser.error("V1 candidate layers are locked to 8,12,16,20")
    if args.frequency_doc_start < 20:
        parser.error("Wikipedia documents 0:20 remain reserved for PPL")
    if args.classifier_steps % args.classifier_check_every != 0:
        parser.error("classifier steps must be divisible by check interval")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def validate_environment_firewall() -> None:
    exposed = [
        name
        for name in FORBIDDEN_EVALUATION_ENVIRONMENT_VARIABLES
        if str(os.environ.get(name, "")).strip()
    ]
    if exposed:
        raise RuntimeError(
            "official/recovery input leaked into V1 preflight environment: "
            + ", ".join(sorted(exposed))
        )


def _locked_preflight_values(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "seed": int(args.seed),
        "forget_num": int(args.forget_num),
        "candidate_layers": list(args.candidate_layers),
        "subject_code_rank": int(args.subject_code_rank),
        "detector_rank": int(args.detector_rank),
        "embedding_ridge_lambda": float(args.embedding_ridge_lambda),
        "embedding_relative_row_cap": float(args.embedding_relative_row_cap),
        "embedding_frequency_alpha": float(args.embedding_frequency_alpha),
        "code_nearest_key_margin": float(args.code_nearest_key_margin),
        "frequency_doc_start": int(args.frequency_doc_start),
        "frequency_docs": int(args.frequency_docs),
        "corpus_prompts": {
            "fit": int(args.corpus_fit_prompts),
            "development": int(args.corpus_development_prompts),
            "calibration": int(args.corpus_calibration_prompts),
            "certification": int(args.corpus_certification_prompts),
        },
        "minimum_shared_subword_prompts": {
            "fit": int(args.shared_fit_prompts),
            "development": int(args.shared_development_prompts),
            "calibration": int(args.shared_calibration_prompts),
            "certification": int(args.shared_certification_prompts),
        },
        "wrong_relations_per_record": {
            "fit": int(args.wrong_relations_fit),
            "development_calibration_certification": int(args.wrong_relations_other),
        },
        "same_relation_other_subjects_per_record": int(
            args.same_relation_other_subjects
        ),
        "classifier_steps": int(args.classifier_steps),
        "classifier_check_every": int(args.classifier_check_every),
        "classifier_lr": float(args.classifier_lr),
        "classifier_weight_decay": float(args.classifier_weight_decay),
        "classifier_positive_floor": float(args.classifier_positive_floor),
        "classifier_negative_ceiling": float(args.classifier_negative_ceiling),
        "classifier_auxiliary_weight": float(args.classifier_auxiliary_weight),
        "classifier_softmin_temperature": float(args.classifier_softmin_temperature),
        "minimum_certification_negative_cells": int(
            args.minimum_certification_negative_cells
        ),
        "minimum_certification_distinct_prompts": int(
            args.minimum_certification_prompts
        ),
        "capture_batch_size": int(args.capture_batch_size),
        "dtype": str(args.dtype),
        "device_map": str(args.device_map),
    }


def validate_registry(registry: Mapping[str, Any], args: argparse.Namespace) -> None:
    if (
        registry.get("protocol") != core.PROTOCOL
        or registry.get("status") != "preflight_implementation_available_not_executed"
        or registry.get("architecture", {}).get("external_string_router") is not False
        or registry.get("architecture", {}).get("inference_sidecar") is not False
        or registry.get("architecture", {}).get("lm_head_mutated") is not False
    ):
        raise RuntimeError("V1 experiment registry architecture/status mismatch")
    locked = registry.get("preflight_implementation")
    if not isinstance(locked, Mapping) or dict(locked) != _locked_preflight_values(
        args
    ):
        raise RuntimeError("V1 preflight arguments differ from the registered design")


def load_locked_inputs(
    visible_path: Path, manifest_path: Path, *, seed: int, forget_num: int
) -> Tuple[List[Mapping[str, Any]], Mapping[str, Any]]:
    records = json.loads(visible_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not all(
        isinstance(row, dict) for row in records
    ):
        raise RuntimeError("training-visible V1 input must be a JSON list")
    split_builder.assert_direct_only(records)
    observed_hash = hashlib.sha256(visible_path.read_bytes()).hexdigest()
    if (
        manifest.get("protocol") != split_builder.PROTOCOL
        or int(manifest.get("seed", -1)) != int(seed)
        or int(manifest.get("forget_num", -1)) != int(forget_num)
        or str(manifest.get("training_visible_sha256")) != observed_hash
        or manifest.get("learner_source_path_available") is not False
        or manifest.get("official_evaluation_permitted") is not False
        or any(
            int(value) != 0
            for key, value in dict(manifest.get("serialized_prompt_counts", {})).items()
            if key != "direct_forget"
        )
    ):
        raise RuntimeError("V1 split manifest differs from the direct-only contract")
    if [int(row["case_id"]) for row in records] != [
        int(value) for value in manifest.get("forget_case_ids", [])
    ]:
        raise RuntimeError("V1 direct record order differs from the split manifest")
    return records, manifest


def record_views(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    values: List[Dict[str, Any]] = []
    for position, record in enumerate(records):
        rewrite = record["requested_rewrite"]
        relation_id = str(rewrite["relation_id"])
        if relation_id not in relation_prompts.RELATION_NOUN_PHRASES:
            raise RuntimeError(
                f"record {position} has unsupported relation {relation_id}"
            )
        values.append(
            {
                "record_index": position,
                "case_id": int(record["case_id"]),
                "subject": str(rewrite["subject"]),
                "relation_id": relation_id,
                "target_true": str(rewrite["target_true"]["str"]),
                "target_new": str(rewrite["target_new"]["str"]),
                "direct_prompt": str(rewrite["prompt"]).format(str(rewrite["subject"])),
            }
        )
    return values


def load_corpus_documents(path: Path, *, start: int, count: int) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    dataset = load_from_disk(str(path))
    train = dataset["train"]
    texts = train["text"][int(start) : int(start) + int(count)]
    values = [str(text) for text in texts if str(text).strip()]
    if len(values) < int(count) // 2:
        raise RuntimeError("registered corpus slice is unexpectedly sparse")
    return values


def token_frequency_counts(
    tokenizer: Any, documents: Sequence[str], vocab_size: int, *, batch_size: int = 16
) -> torch.Tensor:
    counts = torch.zeros(int(vocab_size), dtype=torch.long)
    for start in range(0, len(documents), int(batch_size)):
        encoded = tokenizer(
            list(documents[start : start + int(batch_size)]),
            add_special_tokens=False,
            padding=False,
        )["input_ids"]
        if encoded and isinstance(encoded[0], int):
            encoded = [encoded]
        for row in encoded:
            ids = torch.tensor(
                [int(value) for value in row if 0 <= int(value) < int(vocab_size)],
                dtype=torch.long,
            )
            if ids.numel():
                counts += torch.bincount(ids, minlength=int(vocab_size))
    return counts


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def contains_complete_subject(text: str, subjects: Sequence[str]) -> bool:
    normalized = normalize_text(text)
    return any(
        re.search(
            r"(?<!\w)" + re.escape(normalize_text(subject)) + r"(?!\w)",
            normalized,
            re.IGNORECASE,
        )
        is not None
        for subject in subjects
    )


def candidate_corpus_sentences(documents: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    values: List[str] = []
    for document in documents:
        for raw in re.split(r"(?<=[.!?])\s+|\n+", str(document)):
            sentence = normalize_text(raw)
            words = sentence.split()
            if not 5 <= len(words) <= 32 or "{" in sentence or "}" in sentence:
                continue
            if sentence in seen:
                continue
            seen.add(sentence)
            values.append(sentence)
    return values


def annotate_corpus_sentences(
    tokenizer: Any,
    sentences: Sequence[str],
    *,
    subjects: Sequence[str],
    editable_token_ids: Sequence[int],
    batch_size: int = 128,
) -> List[Dict[str, Any]]:
    editable = {int(value) for value in editable_token_ids}
    filtered = [
        sentence
        for sentence in sentences
        if not contains_complete_subject(sentence, subjects)
    ]
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(filtered), int(batch_size)):
        batch = filtered[start : start + int(batch_size)]
        encoded = tokenizer(batch, add_special_tokens=False, padding=False)["input_ids"]
        if encoded and isinstance(encoded[0], int):
            encoded = [encoded]
        for sentence, token_ids in zip(batch, encoded):
            ids = {int(value) for value in token_ids}
            rows.append(
                {
                    "prompt": sentence,
                    "sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest(),
                    "shared_subword": bool(ids.intersection(editable)),
                }
            )
    return rows


def partition_corpus_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    counts: Mapping[str, int],
    shared_minimums: Mapping[str, int],
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    shared = [dict(row) for row in rows if bool(row["shared_subword"])]
    ordinary = [dict(row) for row in rows if not bool(row["shared_subword"])]
    rng = random.Random(int(seed) + 15485863)
    rng.shuffle(shared)
    rng.shuffle(ordinary)
    if len(shared) < sum(int(value) for value in shared_minimums.values()):
        raise RuntimeError(
            "corpus lacks enough complete-subject-free shared-subword negatives"
        )
    result: Dict[str, List[Dict[str, Any]]] = {}
    for split in ("fit", "development", "calibration", "certification"):
        total = int(counts[split])
        need_shared = int(shared_minimums[split])
        chosen = [shared.pop() for _ in range(need_shared)]
        while len(chosen) < total:
            if ordinary:
                chosen.append(ordinary.pop())
            elif shared:
                chosen.append(shared.pop())
            else:
                raise RuntimeError("corpus lacks enough disjoint prompts")
        result[split] = chosen
    hashes = [row["sha256"] for values in result.values() for row in values]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("corpus prompt partitions overlap")
    return result


def _render(scaffold: str, *, subject: str, relation_id: str) -> str:
    noun = relation_prompts.RELATION_NOUN_PHRASES[str(relation_id)]
    return scaffold.format(subject=str(subject), relation=noun)


def build_prompt_specs(
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    corpus_rows: Sequence[Mapping[str, Any]],
    wrong_relations_per_record: int,
    same_relation_other_subjects: int,
    include_direct: bool,
) -> List[core.SemanticPrompt]:
    scaffolds = {
        "fit": FIT_SCAFFOLDS,
        "development": DEVELOPMENT_SCAFFOLDS,
        "calibration": CALIBRATION_SCAFFOLDS,
        "certification": CERTIFICATION_SCAFFOLDS,
    }[split]
    all_relation_ids = sorted(relation_prompts.RELATION_NOUN_PHRASES)
    subject_relations: Dict[str, set[str]] = {}
    answer_records: Dict[str, List[int]] = {}
    for index, record in enumerate(records):
        subject_relations.setdefault(str(record["subject"]), set()).add(
            str(record["relation_id"])
        )
        answer_records.setdefault(str(record["target_true"]).casefold(), []).append(
            index
        )
    specs: List[core.SemanticPrompt] = []
    positive_specs: List[core.SemanticPrompt] = []
    for owner, record in enumerate(records):
        subject = str(record["subject"])
        relation_id = str(record["relation_id"])
        if include_direct:
            positive_specs.append(
                core.SemanticPrompt(
                    prompt=str(record["direct_prompt"]),
                    subject=subject,
                    relation_id=relation_id,
                    kind="canonical_direct_positive",
                )
            )
        for family, scaffold in enumerate(scaffolds):
            prompt = _render(scaffold, subject=subject, relation_id=relation_id)
            if family % 2 == 1 and corpus_rows:
                prefix = str(
                    corpus_rows[(owner * 17 + family) % len(corpus_rows)]["prompt"]
                )
                prompt = f"{prefix} {prompt}"
            positive_specs.append(
                core.SemanticPrompt(
                    prompt=prompt,
                    subject=subject,
                    relation_id=relation_id,
                    kind=f"{split}_authored_positive",
                )
            )

        forbidden = subject_relations[subject]
        wrong = [value for value in all_relation_ids if value not in forbidden]
        for offset in range(int(wrong_relations_per_record)):
            distractor = wrong[(owner * 11 + offset * 7) % len(wrong)]
            scaffold = scaffolds[(owner + offset) % len(scaffolds)]
            specs.append(
                core.SemanticPrompt(
                    prompt=_render(scaffold, subject=subject, relation_id=distractor),
                    subject=subject,
                    relation_id=distractor,
                    kind="same_subject_different_relation",
                )
            )

        for offset in range(int(same_relation_other_subjects)):
            other = (owner + 1 + offset * 13) % len(records)
            other_subject = str(records[other]["subject"])
            scaffold = scaffolds[(owner + offset + 1) % len(scaffolds)]
            specs.append(
                core.SemanticPrompt(
                    prompt=_render(
                        scaffold,
                        subject=other_subject,
                        relation_id=relation_id,
                    ),
                    subject=other_subject,
                    relation_id=relation_id,
                    kind="same_relation_different_subject",
                )
            )

        same_answer = [
            index
            for index in answer_records[str(record["target_true"]).casefold()]
            if index != owner
        ]
        if same_answer:
            other_subject = str(records[same_answer[0]]["subject"])
            specs.append(
                core.SemanticPrompt(
                    prompt=_render(
                        scaffolds[(owner + 2) % len(scaffolds)],
                        subject=other_subject,
                        relation_id=relation_id,
                    ),
                    subject=other_subject,
                    relation_id=relation_id,
                    kind="same_answer_different_subject",
                )
            )

    specs.extend(positive_specs)
    specs.extend(
        core.SemanticPrompt(
            prompt=spec.prompt,
            subject=spec.subject,
            relation_id=spec.relation_id,
            kind="writer_off_positive_context",
            writer_on=False,
        )
        for spec in positive_specs
    )
    specs.extend(
        core.SemanticPrompt(
            prompt=str(row["prompt"]),
            subject=None,
            relation_id=None,
            kind=(
                "shared_subject_subword_without_complete_subject"
                if bool(row["shared_subword"])
                else "broad_corpus_prompt"
            ),
        )
        for row in corpus_rows
    )
    return specs


def bank_report(bank: core.PromptBank) -> Dict[str, Any]:
    kind_counts: Dict[str, int] = {}
    for kinds in bank.kinds:
        for kind in kinds:
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
    return {
        "rows": len(bank.prompts),
        "distinct_prompt_texts": len(set(bank.prompts)),
        "writer_on_rows": int(bank.writer_on.sum()),
        "writer_off_rows": int((~bank.writer_on).sum()),
        "positive_fact_cells": int(bank.fact_labels.sum()),
        "negative_fact_cells": int((~bank.fact_labels).sum()),
        "positive_records_covered": int(bank.fact_labels.any(dim=0).sum()),
        "kind_counts": dict(sorted(kind_counts.items())),
        "prompt_sha256": hashlib.sha256(
            "\n".join(bank.prompts).encode("utf-8")
        ).hexdigest(),
    }


class EmbeddingCodeHook:
    def __init__(
        self,
        embedding: torch.nn.Module,
        token_ids: Sequence[int],
        delta: torch.Tensor,
    ) -> None:
        if delta.shape != (len(token_ids), int(embedding.weight.shape[1])):
            raise ValueError("embedding code delta has incompatible shape")
        device = embedding.weight.device
        self.enabled = True
        self.lookup = torch.full(
            (int(embedding.weight.shape[0]),), -1, dtype=torch.long, device=device
        )
        ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        self.lookup[ids] = torch.arange(len(token_ids), device=device)
        self.delta = delta.detach().float().to(device)
        self.handle = embedding.register_forward_hook(self._hook)

    def _hook(
        self, _module: torch.nn.Module, inputs: Any, output: torch.Tensor
    ) -> torch.Tensor:
        if not self.enabled:
            return output
        token_ids = inputs[0].to(self.lookup.device)
        local = self.lookup[token_ids]
        mask = local.ge(0)
        if not bool(mask.any()):
            return output
        safe = local.clamp_min(0)
        correction = self.delta.index_select(0, safe.reshape(-1)).reshape(
            *safe.shape, self.delta.shape[1]
        )
        correction = correction * mask.unsqueeze(-1)
        return output + correction.to(device=output.device, dtype=output.dtype)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def capture_bank_hidden_states(
    model: torch.nn.Module,
    tokenizer: Any,
    writer: EmbeddingCodeHook,
    bank: core.PromptBank,
    layers: Sequence[int],
    device: torch.device,
    *,
    batch_size: int,
) -> Dict[int, torch.Tensor]:
    decoder_layers = find_decoder_layers(model)
    if any(layer < 0 or layer >= len(decoder_layers) for layer in layers):
        raise RuntimeError("candidate layer is outside the decoder")
    modules: Dict[int, torch.nn.Module] = {}
    hidden_size = None
    for layer in layers:
        module = getattr(decoder_layers[int(layer)], "mlp", None)
        if module is None:
            raise RuntimeError(f"decoder layer {layer} has no MLP")
        modules[int(layer)] = module
        if hasattr(module, "gate_proj"):
            hidden_size = int(module.gate_proj.weight.shape[1])
    if hidden_size is None:
        hidden_size = int(model.get_input_embeddings().weight.shape[1])
    result = {
        int(layer): torch.empty((len(bank.prompts), hidden_size), dtype=torch.float32)
        for layer in layers
    }
    old_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "right"
    backbone = getattr(model, "model", None)
    if backbone is None or backbone is model:
        raise RuntimeError("V1 capture requires a separate model backbone")
    try:
        for writer_on in (True, False):
            indices = [
                index
                for index, value in enumerate(bank.writer_on.tolist())
                if bool(value) == writer_on
            ]
            writer.enabled = writer_on
            for start in range(0, len(indices), int(batch_size)):
                batch_indices = indices[start : start + int(batch_size)]
                prompts = [bank.prompts[index] for index in batch_indices]
                encoded = tokenizer(prompts, padding=True, return_tensors="pt")
                encoded = {key: value.to(device) for key, value in encoded.items()}
                captured: Dict[int, List[torch.Tensor]] = {
                    int(layer): [] for layer in layers
                }
                handles = []
                for layer, module in modules.items():

                    def pre_hook(
                        _module: torch.nn.Module,
                        inputs: Any,
                        *,
                        selected_layer: int = layer,
                    ) -> None:
                        captured[selected_layer].append(inputs[0])

                    handles.append(module.register_forward_pre_hook(pre_hook))
                try:
                    backbone(**encoded, use_cache=False, return_dict=True)
                finally:
                    for handle in handles:
                        handle.remove()
                positions = encoded["attention_mask"].sum(dim=1) - 1
                rows = torch.arange(len(prompts), device=device)
                destination = torch.tensor(batch_indices, dtype=torch.long)
                for layer in layers:
                    values = captured[int(layer)]
                    if len(values) != 1:
                        raise RuntimeError(
                            f"layer {layer} captured {len(values)} MLP inputs"
                        )
                    selected = values[0][rows, positions, :].detach().float().cpu()
                    result[int(layer)].index_copy_(0, destination, selected)
    finally:
        writer.enabled = True
        tokenizer.padding_side = old_side
    return result


def clone_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in module.state_dict().items()
    }


def train_layer_classifier(
    fit_hidden: torch.Tensor,
    development_hidden: torch.Tensor,
    fit_bank: core.PromptBank,
    development_bank: core.PromptBank,
    fact_relation_index: Sequence[int],
    relation_count: int,
    *,
    rank: int,
    steps: int,
    check_every: int,
    lr: float,
    weight_decay: float,
    positive_floor: float,
    negative_ceiling: float,
    auxiliary_weight: float,
    softmin_temperature: float,
    seed: int,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    torch.manual_seed(int(seed))
    classifier = core.FactorizedFactClassifier(
        int(fit_hidden.shape[1]),
        int(rank),
        fact_relation_index,
        int(relation_count),
        softmin_temperature=float(softmin_temperature),
    ).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    fit_x = fit_hidden.to(device)
    dev_x = development_hidden.to(device)
    fit_fact = fit_bank.fact_labels.to(device)
    fit_subject = fit_bank.subject_labels.to(device)
    fit_relation = fit_bank.relation_labels.to(device)
    dev_fact = development_bank.fact_labels.to(device)
    log: List[Dict[str, Any]] = []
    best_state: Dict[str, torch.Tensor] | None = None
    best_key: Tuple[float, int, float] | None = None
    best_step = -1
    best_report: Dict[str, Any] | None = None
    for step in range(1, int(steps) + 1):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        output = classifier(fit_x)
        loss, parts = core.factorized_classifier_loss(
            output,
            fact_labels=fit_fact,
            subject_labels=fit_subject,
            relation_labels=fit_relation,
            positive_floor=float(positive_floor),
            negative_ceiling=float(negative_ceiling),
            auxiliary_weight=float(auxiliary_weight),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite classifier loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % int(check_every) == 0 or step == int(steps):
            classifier.eval()
            with torch.no_grad():
                dev_scores = classifier(dev_x)["fact_scores"]
            separation = core.score_separation_report(dev_scores, dev_fact)
            row = {
                "step": step,
                "loss": float(loss.detach()),
                **{key: float(value.detach()) for key, value in parts.items()},
                "development": separation,
            }
            log.append(row)
            key = (
                float(separation["separation_gap"]),
                -int(separation["positive_failures_at_provisional_threshold"]),
                -float(loss.detach()),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_state = clone_state_dict(classifier)
                best_step = step
                best_report = separation
    del optimizer
    if best_state is None or best_report is None:
        raise RuntimeError("classifier training produced no selected checkpoint")
    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        fit_report = core.score_separation_report(
            classifier(fit_x)["fact_scores"], fit_fact
        )
    report = {
        "best_step": int(best_step),
        "development_selection_key": [
            float(best_key[0]),
            int(best_key[1]),
            float(best_key[2]),
        ],
        "fit": fit_report,
        "development": best_report,
        "trainable_parameters": classifier.trainable_parameter_count,
        "optimizer_steps": int(steps),
        "training_log": log,
        "certification_seen": False,
    }
    return best_state, report


@torch.no_grad()
def fact_scores(
    state: Mapping[str, torch.Tensor],
    hidden: torch.Tensor,
    fact_relation_index: Sequence[int],
    relation_count: int,
    *,
    rank: int,
    softmin_temperature: float,
    device: torch.device,
) -> torch.Tensor:
    classifier = core.FactorizedFactClassifier(
        int(hidden.shape[1]),
        int(rank),
        fact_relation_index,
        int(relation_count),
        softmin_temperature=float(softmin_temperature),
    ).to(device)
    classifier.load_state_dict(state)
    classifier.eval()
    return classifier(hidden.to(device))["fact_scores"].detach().cpu()


def write_completion(
    method_dir: Path,
    *,
    passed: bool,
    stage: str,
    certification_opened: bool,
    preflight_state_saved: bool,
    extra: Mapping[str, Any] | None = None,
) -> None:
    value: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "mcf_internal_contextual_rewiring_v1_preflight_completion",
        "protocol": core.PROTOCOL,
        "passed": bool(passed),
        "stage": str(stage),
        "certification_opened": bool(certification_opened),
        "preflight_state_saved": bool(preflight_state_saved),
        "actuator_constructed": False,
        "actuator_optimizer_constructed": False,
        "official_evaluation_prompts_seen": 0,
        "official_evaluation_permitted": False,
    }
    if extra:
        value.update(dict(extra))
    write_json(method_dir / "completion.json", value)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_environment_firewall()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    output = Path(args.output_dir).resolve()
    protocol_dir = output / "protocol"
    method_dir = output / "method"
    if method_dir.exists():
        raise FileExistsError(method_dir)
    method_dir.mkdir(parents=True)
    paths = {
        "training_visible": Path(args.training_visible_path).resolve(),
        "split_manifest": Path(args.split_manifest).resolve(),
        "experiment_registry": Path(args.experiment_registry).resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    registry = json.loads(paths["experiment_registry"].read_text(encoding="utf-8"))
    validate_registry(registry, args)
    records_raw, split_manifest = load_locked_inputs(
        paths["training_visible"],
        paths["split_manifest"],
        seed=int(args.seed),
        forget_num=int(args.forget_num),
    )
    records = record_views(records_raw)
    protocol_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        paths["experiment_registry"], protocol_dir / "experiment_registry.json"
    )
    source_hashes = {name: sha256_file(path) for name, path in paths.items()}
    firewall = {
        "schema_version": 1,
        "kind": "mcf_internal_contextual_rewiring_v1_training_firewall",
        "protocol": core.PROTOCOL,
        "seed": int(args.seed),
        "direct_training_records": len(records),
        "split_protocol": split_manifest["protocol"],
        "source_hashes": source_hashes,
        "full_mcf_path_available_to_learner": False,
        "official_evaluation_arguments_available": False,
        "official_evaluation_prompts_seen": 0,
        "actuator_prohibited": True,
    }
    write_json(method_dir / "training_firewall_receipt.json", firewall)

    namespace = argparse.Namespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    model, tokenizer = gagd.load_model_and_tokenizer(namespace, for_training=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    output_head = canonical.untie_and_freeze_output_head(model)
    input_embedding = model.get_input_embeddings()
    if input_embedding.weight.data_ptr() == output_head.weight.data_ptr():
        raise RuntimeError("V1 LM head remained tied")
    device = gagd.first_device(model)
    output_head_sha256_before = core.tensor_sha256(output_head.weight)

    print("Stage 0: build overlap-aware rank-8 subject embedding code")
    special_ids = {
        int(value)
        for value in (
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "unk_token_id", None),
        )
        if value is not None
    }
    token_rows_by_subject = [
        core.subject_token_rows(
            tokenizer, record["subject"], excluded_token_ids=special_ids
        )
        for record in records
    ]
    token_ids, incidence, ownership = core.build_subject_incidence(
        token_rows_by_subject
    )
    input_index = torch.tensor(
        token_ids, dtype=torch.long, device=input_embedding.weight.device
    )
    base_rows = (
        input_embedding.weight.index_select(0, input_index).detach().float().cpu()
    )
    base_rows_sha256 = core.tensor_sha256(base_rows)
    documents = load_corpus_documents(
        Path(args.wikidata_dir).resolve(),
        start=int(args.frequency_doc_start),
        count=int(args.frequency_docs),
    )
    frequencies_all = token_frequency_counts(
        tokenizer, documents, int(input_embedding.weight.shape[0])
    )
    frequencies = frequencies_all.index_select(
        0, torch.tensor(token_ids, dtype=torch.long)
    )
    target_codes = core.deterministic_subject_codes(
        len(records), int(args.subject_code_rank), int(args.seed)
    )
    basis = core.deterministic_orthonormal_basis(
        int(input_embedding.weight.shape[1]),
        int(args.subject_code_rank),
        int(args.seed),
    )
    embedding_delta, code_certificate = core.solve_overlap_aware_embedding_code(
        incidence,
        target_codes,
        basis,
        base_rows,
        frequencies,
        ridge_lambda=float(args.embedding_ridge_lambda),
        relative_row_cap=float(args.embedding_relative_row_cap),
        frequency_alpha=float(args.embedding_frequency_alpha),
        nearest_key_margin_floor=float(args.code_nearest_key_margin),
    )
    overlap_rows = []
    for column, token_id in enumerate(token_ids):
        overlap_rows.append(
            {
                "token_id": int(token_id),
                "token_string": str(tokenizer.convert_ids_to_tokens(int(token_id))),
                "frequency": int(frequencies[column]),
                "owning_record_indices": [
                    int(value) for value in ownership[int(token_id)]
                ],
                "owning_case_ids": [
                    int(records[index]["case_id"]) for index in ownership[int(token_id)]
                ],
                "base_row_sha256": core.tensor_sha256(base_rows[column]),
                "delta_relative_norm": float(
                    embedding_delta[column].norm()
                    / base_rows[column].norm().clamp_min(1e-12)
                ),
            }
        )
    overlap_manifest = {
        "schema_version": 1,
        "kind": "mcf_internal_contextual_rewiring_v1_overlap_manifest",
        "protocol": core.PROTOCOL,
        "records": len(records),
        "editable_token_rows": len(token_ids),
        "shared_token_rows": sum(len(values) > 1 for values in ownership.values()),
        "maximum_row_owners": max(len(values) for values in ownership.values()),
        "subject_tokenizations": [
            {
                "record_index": index,
                "case_id": int(record["case_id"]),
                "subject": str(record["subject"]),
                "token_ids": token_rows_by_subject[index],
            }
            for index, record in enumerate(records)
        ],
        "rows": overlap_rows,
        "base_selected_rows_sha256": base_rows_sha256,
        "embedding_delta_sha256": core.tensor_sha256(embedding_delta),
        "official_evaluation_prompts_seen": 0,
    }
    write_json(method_dir / "overlap_manifest.json", overlap_manifest)
    write_json(method_dir / "embedding_code_certificate.json", code_certificate)
    print(
        f"  rows={len(token_ids)}, shared={overlap_manifest['shared_token_rows']}, "
        f"incidence rank={code_certificate['incidence_rank']}/{len(records)}, "
        f"key margin={code_certificate['nearest_key_margin_min']:.4f}"
    )
    if not bool(code_certificate["passed"]):
        write_completion(
            method_dir,
            passed=False,
            stage="overlap_embedding_code_certificate",
            certification_opened=False,
            preflight_state_saved=False,
        )
        raise SystemExit(
            "V1 overlap-aware embedding code failed before classifier fitting"
        )

    print("Stage 0b: construct disjoint training-safe corpus prompt partitions")
    corpus_candidates = candidate_corpus_sentences(documents)
    annotated = annotate_corpus_sentences(
        tokenizer,
        corpus_candidates,
        subjects=[str(record["subject"]) for record in records],
        editable_token_ids=token_ids,
    )
    corpus_partitions = partition_corpus_rows(
        annotated,
        counts={
            "fit": int(args.corpus_fit_prompts),
            "development": int(args.corpus_development_prompts),
            "calibration": int(args.corpus_calibration_prompts),
            "certification": int(args.corpus_certification_prompts),
        },
        shared_minimums={
            "fit": int(args.shared_fit_prompts),
            "development": int(args.shared_development_prompts),
            "calibration": int(args.shared_calibration_prompts),
            "certification": int(args.shared_certification_prompts),
        },
        seed=int(args.seed),
    )
    relation_ids = sorted(relation_prompts.RELATION_NOUN_PHRASES)
    relation_to_index = {value: index for index, value in enumerate(relation_ids)}
    fact_relation_index = [
        relation_to_index[str(record["relation_id"])] for record in records
    ]
    fact_subjects = [str(record["subject"]) for record in records]
    fact_relations = [str(record["relation_id"]) for record in records]

    def make_bank(split: str, *, include_direct: bool) -> core.PromptBank:
        wrong = (
            int(args.wrong_relations_fit)
            if split == "fit"
            else int(args.wrong_relations_other)
        )
        specs = build_prompt_specs(
            records,
            split=split,
            corpus_rows=corpus_partitions[split],
            wrong_relations_per_record=wrong,
            same_relation_other_subjects=int(args.same_relation_other_subjects),
            include_direct=include_direct,
        )
        return core.canonical_multilabel_prompt_bank(
            specs,
            fact_subjects=fact_subjects,
            fact_relation_ids=fact_relations,
            relation_ids=relation_ids,
        )

    fit_bank = make_bank("fit", include_direct=True)
    development_bank = make_bank("development", include_direct=False)
    calibration_bank = make_bank("calibration", include_direct=False)
    prompt_manifest: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "mcf_internal_contextual_rewiring_v1_prompt_bank_manifest",
        "protocol": core.PROTOCOL,
        "fit": bank_report(fit_bank),
        "development": bank_report(development_bank),
        "calibration": bank_report(calibration_bank),
        "certification": {
            "reserved_corpus_prompts": len(corpus_partitions["certification"]),
            "opened": False,
        },
        "corpus_partition_overlap": 0,
        "complete_subjects_excluded_from_corpus_negatives": True,
        "official_evaluation_prompts_seen": 0,
    }
    write_json(method_dir / "prompt_bank_manifest.json", prompt_manifest)

    writer = EmbeddingCodeHook(input_embedding, token_ids, embedding_delta)
    try:
        print("Stage 1: cache fit/development hidden states at layers 8/12/16/20")
        fit_hidden = capture_bank_hidden_states(
            model,
            tokenizer,
            writer,
            fit_bank,
            args.candidate_layers,
            device,
            batch_size=int(args.capture_batch_size),
        )
        development_hidden = capture_bank_hidden_states(
            model,
            tokenizer,
            writer,
            development_bank,
            args.candidate_layers,
            device,
            batch_size=int(args.capture_batch_size),
        )

        print("Stage 1a: fit rank-8 factorized classifiers; select by development only")
        layer_states: Dict[int, Dict[str, torch.Tensor]] = {}
        layer_reports: Dict[str, Any] = {}
        for layer in args.candidate_layers:
            state, report = train_layer_classifier(
                fit_hidden[int(layer)],
                development_hidden[int(layer)],
                fit_bank,
                development_bank,
                fact_relation_index,
                len(relation_ids),
                rank=int(args.detector_rank),
                steps=int(args.classifier_steps),
                check_every=int(args.classifier_check_every),
                lr=float(args.classifier_lr),
                weight_decay=float(args.classifier_weight_decay),
                positive_floor=float(args.classifier_positive_floor),
                negative_ceiling=float(args.classifier_negative_ceiling),
                auxiliary_weight=float(args.classifier_auxiliary_weight),
                softmin_temperature=float(args.classifier_softmin_temperature),
                seed=int(args.seed) * 1000 + int(layer),
                device=device,
            )
            layer_states[int(layer)] = state
            layer_reports[str(layer)] = report
            dev = report["development"]
            print(
                f"  layer {layer:2d}: gap={dev['separation_gap']:.6f}, "
                f"positive failures={dev['positive_failures_at_provisional_threshold']}"
            )
        selected_layer = max(
            args.candidate_layers,
            key=lambda layer: (
                float(layer_reports[str(layer)]["development"]["separation_gap"]),
                -int(
                    layer_reports[str(layer)]["development"][
                        "positive_failures_at_provisional_threshold"
                    ]
                ),
                -int(layer),
            ),
        )
        layer_sweep = {
            "schema_version": 1,
            "kind": "mcf_internal_contextual_rewiring_v1_layer_sweep",
            "protocol": core.PROTOCOL,
            "candidate_layers": list(args.candidate_layers),
            "selection_split": "development",
            "calibration_visible_to_selection": False,
            "certification_visible_to_selection": False,
            "selected_layer": int(selected_layer),
            "layers": layer_reports,
            "official_evaluation_prompts_seen": 0,
        }
        write_json(method_dir / "classifier_layer_sweep.json", layer_sweep)

        selected_state = layer_states[int(selected_layer)]
        fit_scores = fact_scores(
            selected_state,
            fit_hidden[int(selected_layer)],
            fact_relation_index,
            len(relation_ids),
            rank=int(args.detector_rank),
            softmin_temperature=float(args.classifier_softmin_temperature),
            device=device,
        )
        means, scales = core.fit_negative_standardization(
            fit_scores, fit_bank.fact_labels
        )

        print(
            f"Stage 1b: calibrate a frozen threshold at selected layer {selected_layer}"
        )
        calibration_hidden = capture_bank_hidden_states(
            model,
            tokenizer,
            writer,
            calibration_bank,
            [int(selected_layer)],
            device,
            batch_size=int(args.capture_batch_size),
        )[int(selected_layer)]
        calibration_scores = fact_scores(
            selected_state,
            calibration_hidden,
            fact_relation_index,
            len(relation_ids),
            rank=int(args.detector_rank),
            softmin_temperature=float(args.classifier_softmin_temperature),
            device=device,
        )
        calibration_standardized = core.standardize_fact_scores(
            calibration_scores, means, scales
        )
        threshold_report = core.calibrate_global_threshold(
            calibration_standardized, calibration_bank.fact_labels
        )
        threshold_report.update(
            {
                "schema_version": 1,
                "kind": "mcf_internal_contextual_rewiring_v1_threshold_calibration",
                "protocol": core.PROTOCOL,
                "selected_layer": int(selected_layer),
                "standardization": "per_fact_fit_negative_mean_and_population_std",
                "fit_negative_means": [float(value) for value in means],
                "fit_negative_scales": [float(value) for value in scales],
                "certification_opened": False,
                "official_evaluation_prompts_seen": 0,
            }
        )
        write_json(method_dir / "threshold_calibration.json", threshold_report)
        if not bool(threshold_report["passed"]):
            write_completion(
                method_dir,
                passed=False,
                stage="threshold_calibration_positive_coverage",
                certification_opened=False,
                preflight_state_saved=False,
                extra={"selected_layer": int(selected_layer)},
            )
            raise SystemExit(
                "V1 selected classifier failed calibration positives; "
                "certification and actuator construction are refused"
            )

        print("Stage 1c: open the disjoint certification bank exactly once")
        certification_bank = make_bank("certification", include_direct=False)
        certification_report_manifest = bank_report(certification_bank)
        prompt_manifest["certification"] = {
            **certification_report_manifest,
            "opened": True,
            "open_count": 1,
            "optimizer_steps_after_open": 0,
        }
        write_json(method_dir / "prompt_bank_manifest.json", prompt_manifest)
        certification_hidden = capture_bank_hidden_states(
            model,
            tokenizer,
            writer,
            certification_bank,
            [int(selected_layer)],
            device,
            batch_size=int(args.capture_batch_size),
        )[int(selected_layer)]
        certification_scores = fact_scores(
            selected_state,
            certification_hidden,
            fact_relation_index,
            len(relation_ids),
            rank=int(args.detector_rank),
            softmin_temperature=float(args.classifier_softmin_temperature),
            device=device,
        )
        certification_standardized = core.standardize_fact_scores(
            certification_scores, means, scales
        )
        certificate = core.frozen_threshold_certificate(
            certification_standardized,
            certification_bank.fact_labels,
            threshold=float(threshold_report["threshold"]),
            distinct_prompts=len(set(certification_bank.prompts)),
            minimum_negative_cells=int(args.minimum_certification_negative_cells),
            minimum_distinct_prompts=int(args.minimum_certification_prompts),
        )
        per_kind_audit = core.per_kind_threshold_audit(
            certification_standardized,
            certification_bank.fact_labels,
            certification_bank.kinds,
            threshold=float(threshold_report["threshold"]),
        )
        missing_mandatory_kinds = sorted(
            set(MANDATORY_CERTIFICATION_KINDS).difference(per_kind_audit)
        )
        failed_mandatory_kinds = sorted(
            kind
            for kind in MANDATORY_CERTIFICATION_KINDS
            if kind in per_kind_audit and not bool(per_kind_audit[kind]["passed"])
        )
        per_fact_positive = certification_bank.fact_labels.sum(dim=0)
        certificate.update(
            {
                "schema_version": 1,
                "kind": "mcf_internal_contextual_rewiring_v1_classifier_certificate",
                "protocol": core.PROTOCOL,
                "selected_layer": int(selected_layer),
                "records_with_positive_certification_cells": int(
                    per_fact_positive.gt(0).sum()
                ),
                "all_records_positive_covered": bool(per_fact_positive.gt(0).all()),
                "per_prompt_family": per_kind_audit,
                "mandatory_prompt_families": list(MANDATORY_CERTIFICATION_KINDS),
                "missing_mandatory_prompt_families": missing_mandatory_kinds,
                "failed_mandatory_prompt_families": failed_mandatory_kinds,
                "all_mandatory_prompt_families_passed": bool(
                    not missing_mandatory_kinds and not failed_mandatory_kinds
                ),
                "threshold_recalibrated_on_certification": False,
                "optimizer_steps_after_certification_open": 0,
                "actuator_constructed": False,
                "official_evaluation_prompts_seen": 0,
            }
        )
        certificate["passed"] = bool(
            certificate["passed"]
            and certificate["all_records_positive_covered"]
            and certificate["all_mandatory_prompt_families_passed"]
        )
        write_json(method_dir / "classifier_certificate.json", certificate)
        if not bool(certificate["passed"]):
            write_completion(
                method_dir,
                passed=False,
                stage="classifier_certification",
                certification_opened=True,
                preflight_state_saved=False,
                extra={
                    "selected_layer": int(selected_layer),
                    "positive_failures": int(certificate["positive_failures"]),
                    "negative_failures": int(certificate["negative_failures"]),
                },
            )
            raise SystemExit(
                "V1 classifier failed the frozen certification gate; "
                "actuator construction is refused"
            )

        output_head_sha256_after = core.tensor_sha256(output_head.weight)
        selected_rows_after = (
            input_embedding.weight.index_select(0, input_index).detach().float().cpu()
        )
        integrity = {
            "base_selected_embedding_rows_unchanged": core.tensor_sha256(
                selected_rows_after
            )
            == base_rows_sha256,
            "lm_head_bit_identical": output_head_sha256_after
            == output_head_sha256_before,
            "input_lm_head_untied": input_embedding.weight.data_ptr()
            != output_head.weight.data_ptr(),
            "all_model_parameters_require_grad_false": all(
                not parameter.requires_grad for parameter in model.parameters()
            ),
            "actuator_constructed": False,
        }
        integrity["passed"] = (
            all(
                value
                for key, value in integrity.items()
                if key != "actuator_constructed"
            )
            and integrity["actuator_constructed"] is False
        )
        write_json(method_dir / "integrity.json", integrity)
        if not bool(integrity["passed"]):
            raise RuntimeError("V1 preflight integrity failed after certification")

        preflight_state = {
            "schema_version": 1,
            "kind": "mcf_internal_contextual_rewiring_v1_certified_preflight_state",
            "protocol": core.PROTOCOL,
            "seed": int(args.seed),
            "case_ids": [int(record["case_id"]) for record in records],
            "subjects": fact_subjects,
            "relation_ids": fact_relations,
            "editable_token_ids": token_ids,
            "embedding_delta": embedding_delta.cpu(),
            "embedding_basis": basis.float().cpu(),
            "target_subject_codes": target_codes.float().cpu(),
            "selected_layer": int(selected_layer),
            "classifier_rank": int(args.detector_rank),
            "classifier_relation_ids": relation_ids,
            "fact_relation_index": fact_relation_index,
            "classifier_state_dict": selected_state,
            "fit_negative_means": means.cpu(),
            "fit_negative_scales": scales.cpu(),
            "threshold": float(threshold_report["threshold"]),
            "source_hashes": source_hashes,
            "base_selected_embedding_rows_sha256": base_rows_sha256,
            "base_lm_head_sha256": output_head_sha256_before,
            "actuator_constructed": False,
            "official_evaluation_prompts_seen": 0,
        }
        state_path = method_dir / "certified_preflight_state.pt"
        torch.save(preflight_state, state_path)
        state_sha256 = sha256_file(state_path)
        write_completion(
            method_dir,
            passed=True,
            stage="classifier_certification",
            certification_opened=True,
            preflight_state_saved=True,
            extra={
                "selected_layer": int(selected_layer),
                "certified_preflight_state_sha256": state_sha256,
                "next_stage": "separately_registered_width_16_actuator_feasibility",
            },
        )
        print(
            json.dumps(
                {
                    "passed": True,
                    "selected_layer": int(selected_layer),
                    "threshold": float(threshold_report["threshold"]),
                    "certification_negative_cells": int(certificate["negative_cells"]),
                    "certification_distinct_prompts": int(
                        certificate["distinct_prompts"]
                    ),
                    "actuator_constructed": False,
                    "official_evaluation_prompts_seen": 0,
                    "certified_preflight_state_sha256": state_sha256,
                },
                indent=2,
            )
        )
    finally:
        writer.remove()


if __name__ == "__main__":
    main()
