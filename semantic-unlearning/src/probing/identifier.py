import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm


@dataclass
class SemanticToken:
    token_id: int
    token_str: str
    max_probe_accuracy: float
    best_layer: int
    probe_scores: Dict[int, float]  # layer -> P(forget)
    frequency_in_forget: int
    frequency_in_retain: int


@dataclass
class IdentificationResult:
    semantic_tokens: List[SemanticToken]
    threshold: float
    n_forget_texts: int
    n_retain_texts: int
    layer_accuracies: Dict[int, float]

    def token_ids(self) -> List[int]:
        return [t.token_id for t in self.semantic_tokens]

    def summary(self) -> dict:
        return {
            "n_semantic_tokens": len(self.semantic_tokens),
            "threshold": self.threshold,
            "n_forget_texts": self.n_forget_texts,
            "n_retain_texts": self.n_retain_texts,
            "top_tokens": [
                {
                    "token_id": t.token_id,
                    "token_str": t.token_str,
                    "max_probe_accuracy": t.max_probe_accuracy,
                    "best_layer": t.best_layer,
                }
                for t in sorted(
                    self.semantic_tokens, key=lambda x: x.max_probe_accuracy, reverse=True
                )[:10]
            ],
        }

    def save(self, path: str):
        data = {
            "threshold": self.threshold,
            "n_forget_texts": self.n_forget_texts,
            "n_retain_texts": self.n_retain_texts,
            "layer_accuracies": {str(k): v for k, v in self.layer_accuracies.items()},
            "semantic_tokens": [
                {
                    "token_id": t.token_id,
                    "token_str": t.token_str,
                    "max_probe_accuracy": t.max_probe_accuracy,
                    "best_layer": t.best_layer,
                    "probe_scores": {str(k): v for k, v in t.probe_scores.items()},
                    "frequency_in_forget": t.frequency_in_forget,
                    "frequency_in_retain": t.frequency_in_retain,
                }
                for t in self.semantic_tokens
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved IdentificationResult to {path}")


class SemanticTokenIdentifier:
    def __init__(self, extractor, probes: dict, probe_results: dict):
        """
        extractor: HiddenStateExtractor
        probes: dict{layer_idx: LinearProbe}
        probe_results: dict{layer_idx: ProbeResult}
        """
        self.extractor = extractor
        self.probes = probes
        self.probe_results = probe_results

    def identify(
        self,
        forget_texts: List[str],
        retain_texts: List[str],
        threshold: float = 0.70,
        best_layer: Optional[int] = None,
        layers_to_check: Optional[List[int]] = None,
        batch_size: int = 8,
    ) -> IdentificationResult:
        if layers_to_check is None:
            layers_to_check = sorted(self.probes.keys())
        if best_layer is None:
            best_layer = max(
                self.probe_results, key=lambda l: self.probe_results[l].accuracy
            )

        layer_accuracies = {l: self.probe_results[l].accuracy for l in layers_to_check}

        # token_id -> {layer -> list of P(forget) scores}
        token_scores: Dict[int, Dict[int, List[float]]] = {}
        freq_forget: Dict[int, int] = {}
        freq_retain: Dict[int, int] = {}

        def _process_texts(texts: List[str], is_forget: bool):
            per_token_states = self.extractor.extract_per_token(
                texts, batch_size=batch_size, layers=layers_to_check
            )
            token_id_lists = self.extractor.get_token_ids(texts)

            n_texts = len(texts)
            for text_idx in tqdm(range(n_texts), desc="Scoring tokens"):
                token_ids = token_id_lists[text_idx]
                seq_len = len(token_ids)

                for layer_idx in layers_to_check:
                    hs_list = per_token_states[layer_idx]
                    if text_idx >= len(hs_list):
                        continue
                    hs = hs_list[text_idx]  # (actual_seq_len, d_model)
                    actual_len = min(seq_len, hs.shape[0])

                    proba = self.probes[layer_idx].predict_proba(hs[:actual_len])  # (seq, 2)
                    p_forget = proba[:, 1]

                    for pos in range(actual_len):
                        tid = token_ids[pos]
                        score = float(p_forget[pos])
                        token_scores.setdefault(tid, {}).setdefault(layer_idx, []).append(score)

                # Track frequencies using best_layer sequence
                for pos, tid in enumerate(token_ids[:seq_len]):
                    if is_forget:
                        freq_forget[tid] = freq_forget.get(tid, 0) + 1
                    else:
                        freq_retain[tid] = freq_retain.get(tid, 0) + 1

        _process_texts(forget_texts, is_forget=True)
        _process_texts(retain_texts, is_forget=False)

        semantic_tokens: List[SemanticToken] = []
        for tid, layer_map in token_scores.items():
            per_layer_max: Dict[int, float] = {}
            for layer_idx, scores in layer_map.items():
                per_layer_max[layer_idx] = float(np.max(scores))

            max_score = max(per_layer_max.values())
            best_l = max(per_layer_max, key=per_layer_max.__getitem__)

            if max_score >= threshold:
                semantic_tokens.append(
                    SemanticToken(
                        token_id=tid,
                        token_str=self.extractor.decode_token(tid),
                        max_probe_accuracy=max_score,
                        best_layer=best_l,
                        probe_scores=per_layer_max,
                        frequency_in_forget=freq_forget.get(tid, 0),
                        frequency_in_retain=freq_retain.get(tid, 0),
                    )
                )

        semantic_tokens.sort(key=lambda t: t.max_probe_accuracy, reverse=True)

        print(f"\nFound {len(semantic_tokens)} semantic tokens above threshold {threshold}.")
        print("Top 10 semantic tokens:")
        for tok in semantic_tokens[:10]:
            print(
                f"  {repr(tok.token_str):20s} | id={tok.token_id:6d} | "
                f"score={tok.max_probe_accuracy:.4f} | layer={tok.best_layer}"
            )

        return IdentificationResult(
            semantic_tokens=semantic_tokens,
            threshold=threshold,
            n_forget_texts=len(forget_texts),
            n_retain_texts=len(retain_texts),
            layer_accuracies=layer_accuracies,
        )

    def filter_by_selectivity(
        self,
        result: IdentificationResult,
        min_forget_retain_ratio: float = 2.0,
    ) -> IdentificationResult:
        filtered = [
            t
            for t in result.semantic_tokens
            if t.frequency_in_forget / (t.frequency_in_retain + 1) >= min_forget_retain_ratio
        ]
        print(
            f"Selectivity filter (ratio>={min_forget_retain_ratio}): "
            f"{len(result.semantic_tokens)} -> {len(filtered)} tokens"
        )
        return IdentificationResult(
            semantic_tokens=filtered,
            threshold=result.threshold,
            n_forget_texts=result.n_forget_texts,
            n_retain_texts=result.n_retain_texts,
            layer_accuracies=result.layer_accuracies,
        )
