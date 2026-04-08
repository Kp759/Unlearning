import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm


@dataclass
class SemanticToken:
    token_id: int
    token_str: str
    max_probe_accuracy: float          # kept for backwards compat — now = differential score
    best_layer: int
    probe_scores: Dict[int, float]     # layer -> differential score
    frequency_in_forget: int
    frequency_in_retain: int
    mean_forget_score: float = 0.0     # mean P(forget) in forget texts
    mean_retain_score: float = 0.0     # mean P(forget) in retain texts — should be LOW


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
            "scoring_method": "differential (mean_forget - mean_retain)",
            "n_forget_texts": self.n_forget_texts,
            "n_retain_texts": self.n_retain_texts,
            "top_tokens": [
                {
                    "token_id": t.token_id,
                    "token_str": t.token_str,
                    "differential": t.max_probe_accuracy,
                    "mean_forget_score": t.mean_forget_score,
                    "mean_retain_score": t.mean_retain_score,
                    "best_layer": t.best_layer,
                }
                for t in self.semantic_tokens[:10]
            ],
        }

    def save(self, path: str):
        data = {
            "threshold": self.threshold,
            "scoring_method": "differential (mean_forget - mean_retain)",
            "n_forget_texts": self.n_forget_texts,
            "n_retain_texts": self.n_retain_texts,
            "layer_accuracies": {str(k): v for k, v in self.layer_accuracies.items()},
            "token_ids": self.token_ids(),
            "token_strings": [t.token_str for t in self.semantic_tokens],
            "semantic_tokens": [
                {
                    "token_id":          t.token_id,
                    "token_str":         t.token_str,
                    "differential":      t.max_probe_accuracy,
                    "mean_forget_score": t.mean_forget_score,
                    "mean_retain_score": t.mean_retain_score,
                    "best_layer":        t.best_layer,
                    "probe_scores":      {str(k): v for k, v in t.probe_scores.items()},
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
        extractor:     HiddenStateExtractor
        probes:        dict{layer_idx: LinearProbe}
        probe_results: dict{layer_idx: ProbeResult}
        """
        self.extractor = extractor
        self.probes = probes
        self.probe_results = probe_results

    def identify(
        self,
        forget_texts: List[str],
        retain_texts: List[str],
        threshold: float = 0.30,
        best_layer: Optional[int] = None,
        layers_to_check: Optional[List[int]] = None,
        batch_size: int = 8,
    ) -> IdentificationResult:
        """
        Identify semantic tokens using DIFFERENTIAL scoring:

            score(token) = mean P(forget | token in forget_texts)
                         - mean P(forget | token in retain_texts)

        Why differential?
          - Generic tokens like '<|begin_of_text|>', 'the', 'is' get
            P(forget) ≈ 1.0 in BOTH splits when probe is perfect (layer 13+)
          - Differential cancels them out → diff ≈ 0.0 → filtered
          - Concept-specific tokens like 'Aarav', 'Shah' get high P(forget)
            in forget texts but low in retain texts → diff ≈ 0.9 → kept

        Args:
            threshold: minimum differential score to include token in T_f
                       0.30 is a good default (use lower to get more tokens)
        """
        if layers_to_check is None:
            layers_to_check = sorted(self.probes.keys())
        if best_layer is None:
            best_layer = max(
                self.probe_results, key=lambda l: self.probe_results[l].accuracy
            )

        layer_accuracies = {
            l: self.probe_results[l].accuracy
            for l in layers_to_check
            if l in self.probe_results
        }

        # token_id -> {layer -> list of P(forget) scores}
        forget_token_scores: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
        retain_token_scores: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
        freq_forget: Dict[int, int] = defaultdict(int)
        freq_retain: Dict[int, int] = defaultdict(int)

        # ── Score forget texts ────────────────────────────────────────────
        print(f"Extracting per-token states for {len(forget_texts)} forget texts...")
        forget_per_token = self.extractor.extract_per_token(
            forget_texts, batch_size=batch_size, layers=layers_to_check
        )
        forget_token_id_lists = self.extractor.get_token_ids(forget_texts)

        for text_idx in tqdm(range(len(forget_texts)), desc="Scoring forget tokens"):
            token_ids = forget_token_id_lists[text_idx]
            for layer_idx in layers_to_check:
                hs_list = forget_per_token[layer_idx]
                if text_idx >= len(hs_list):
                    continue
                hs = hs_list[text_idx]                              # (seq_len, d_model)
                actual_len = min(len(token_ids), hs.shape[0])
                p_forget = self.probes[layer_idx].predict_proba(
                    hs[:actual_len]
                )[:, 1]                                             # (seq_len,)

                for pos in range(actual_len):
                    tid = token_ids[pos]
                    forget_token_scores[tid][layer_idx].append(float(p_forget[pos]))
                    freq_forget[tid] += 1

        # ── Score retain texts ────────────────────────────────────────────
        print(f"Extracting per-token states for {len(retain_texts)} retain texts...")
        retain_per_token = self.extractor.extract_per_token(
            retain_texts, batch_size=batch_size, layers=layers_to_check
        )
        retain_token_id_lists = self.extractor.get_token_ids(retain_texts)

        for text_idx in tqdm(range(len(retain_texts)), desc="Scoring retain tokens"):
            token_ids = retain_token_id_lists[text_idx]
            for layer_idx in layers_to_check:
                hs_list = retain_per_token[layer_idx]
                if text_idx >= len(hs_list):
                    continue
                hs = hs_list[text_idx]
                actual_len = min(len(token_ids), hs.shape[0])
                p_forget = self.probes[layer_idx].predict_proba(
                    hs[:actual_len]
                )[:, 1]

                for pos in range(actual_len):
                    tid = token_ids[pos]
                    retain_token_scores[tid][layer_idx].append(float(p_forget[pos]))
                    freq_retain[tid] += 1

        # ── Compute differential scores per token ─────────────────────────
        semantic_tokens: List[SemanticToken] = []

        for tid, f_layer_map in forget_token_scores.items():

            # Per-layer differential scores
            per_layer_diff: Dict[int, float] = {}
            for layer_idx, f_scores in f_layer_map.items():
                mean_f = float(np.mean(f_scores))
                r_scores = retain_token_scores[tid].get(layer_idx, [0.0])
                mean_r = float(np.mean(r_scores))
                per_layer_diff[layer_idx] = mean_f - mean_r

            # Best layer = highest differential
            best_l = max(per_layer_diff, key=per_layer_diff.__getitem__)
            best_diff = per_layer_diff[best_l]

            # Overall mean_forget and mean_retain across all layers at best_layer
            mean_forget_overall = float(np.mean(f_layer_map.get(best_layer, f_layer_map[best_l])))
            r_at_best = retain_token_scores[tid].get(best_layer, retain_token_scores[tid].get(best_l, [0.0]))
            mean_retain_overall = float(np.mean(r_at_best))

            if best_diff >= threshold:
                semantic_tokens.append(
                    SemanticToken(
                        token_id=tid,
                        token_str=self.extractor.decode_token(tid),
                        max_probe_accuracy=best_diff,        # differential score
                        best_layer=best_l,
                        probe_scores=per_layer_diff,
                        frequency_in_forget=freq_forget[tid],
                        frequency_in_retain=freq_retain.get(tid, 0),
                        mean_forget_score=mean_forget_overall,
                        mean_retain_score=mean_retain_overall,
                    )
                )

        # Sort by differential descending
        semantic_tokens.sort(key=lambda t: t.max_probe_accuracy, reverse=True)

        print(f"\nFound {len(semantic_tokens)} semantic tokens "
              f"above differential threshold {threshold}.")

        if semantic_tokens:
            print(f"\n  {'Token':<20} {'id':>7}  {'diff':>6}  "
                  f"{'f_score':>7}  {'r_score':>7}  {'f_freq':>6}  {'r_freq':>6}")
            print("  " + "-" * 72)
            for tok in semantic_tokens[:20]:
                print(
                    f"  {repr(tok.token_str):<20} "
                    f"{tok.token_id:>7}  "
                    f"{tok.max_probe_accuracy:>6.3f}  "
                    f"{tok.mean_forget_score:>7.3f}  "
                    f"{tok.mean_retain_score:>7.3f}  "
                    f"{tok.frequency_in_forget:>6}  "
                    f"{tok.frequency_in_retain:>6}"
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
        """
        Optional post-filter: keep only tokens that appear proportionally
        more often in forget than retain texts.
        With differential scoring this is usually not needed,
        but can be used as an extra cleanup step.
        """
        filtered = [
            t for t in result.semantic_tokens
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