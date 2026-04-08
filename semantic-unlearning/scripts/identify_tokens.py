"""Identify semantic tokens that carry the forget concept using token-level probing.

Scoring method: DIFFERENTIAL (mean_forget - mean_retain)
Filters generic tokens like <|begin_of_text|>, 'the', '.' that score high in both splits.
Also merges probe-identified tokens with explicit entity tokens from config.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import TOFUDataset
from src.probing import HiddenStateExtractor, LinearProbe
from src.utils import plot_token_scores


def identify_semantic_tokens_differential(
    extractor,
    probes,
    forget_texts,
    retain_texts,
    best_layer,
    threshold=0.10,
    batch_size=8,
):
    """
    Differential token scoring:
        score(token) = mean P(forget | token in forget_texts)
                     - mean P(forget | token in retain_texts)

    Returns:
        semantic_tokens: list of dicts above threshold
        scored_tokens:   full list of all scored tokens (for near-miss debug)
    """
    layers_to_check = [best_layer]

    # ── Step 1: Extract per-token hidden states ───────────────────────────
    print(f"\n[Step 1] Extracting per-token hidden states (forget) at layer {best_layer}...")
    forget_per_token = extractor.extract_per_token(
        forget_texts, batch_size=batch_size, layers=layers_to_check
    )
    forget_token_ids = extractor.get_token_ids(forget_texts)

    print(f"[Step 2] Extracting per-token hidden states (retain) at layer {best_layer}...")
    retain_per_token = extractor.extract_per_token(
        retain_texts, batch_size=batch_size, layers=layers_to_check
    )
    retain_token_ids = extractor.get_token_ids(retain_texts)

    probe = probes[best_layer]

    # ── Step 3: Score tokens in forget texts ──────────────────────────────
    token_forget_scores: dict = defaultdict(list)
    token_freq_forget: dict = defaultdict(int)

    print("[Step 3] Scoring tokens in forget texts...")
    for text_idx, token_id_seq in enumerate(tqdm(forget_token_ids, desc="Forget scoring")):
        h = forget_per_token[best_layer][text_idx]
        seq_len = min(len(token_id_seq), h.shape[0])
        probs = probe.predict_proba(h[:seq_len])
        forget_probs = probs[:, 1]
        for pos, tid in enumerate(token_id_seq[:seq_len]):
            token_forget_scores[tid].append(float(forget_probs[pos]))
            token_freq_forget[tid] += 1

    # ── Step 4: Score tokens in retain texts ──────────────────────────────
    token_retain_scores: dict = defaultdict(list)
    token_freq_retain: dict = defaultdict(int)

    print("[Step 4] Scoring tokens in retain texts...")
    for text_idx, token_id_seq in enumerate(tqdm(retain_token_ids, desc="Retain scoring")):
        h = retain_per_token[best_layer][text_idx]
        seq_len = min(len(token_id_seq), h.shape[0])
        probs = probe.predict_proba(h[:seq_len])
        forget_probs = probs[:, 1]
        for pos, tid in enumerate(token_id_seq[:seq_len]):
            token_retain_scores[tid].append(float(forget_probs[pos]))
            token_freq_retain[tid] += 1

    # ── Step 5: Compute differential scores ───────────────────────────────
    print("[Step 5] Computing differential scores (mean_forget - mean_retain)...")
    scored_tokens = []

    for token_id, f_scores in token_forget_scores.items():
        mean_forget = float(np.mean(f_scores))
        r_scores = token_retain_scores.get(token_id, [0.0])
        mean_retain = float(np.mean(r_scores))
        differential = mean_forget - mean_retain

        scored_tokens.append({
            "token_id":          token_id,
            "token_str":         extractor.decode_token(token_id),
            "mean_forget_score": mean_forget,
            "mean_retain_score": mean_retain,
            "differential":      differential,
            "freq_forget":       token_freq_forget[token_id],
            "freq_retain":       token_freq_retain.get(token_id, 0),
            "best_layer":        best_layer,
            "source":            "probe",
        })

    # Sort by differential descending
    scored_tokens.sort(key=lambda x: x["differential"], reverse=True)

    # Filter by threshold
    semantic_tokens = [t for t in scored_tokens if t["differential"] >= threshold]

    print(f"\nFound {len(semantic_tokens)} semantic tokens above threshold {threshold}")

    if semantic_tokens:
        print(f"\n  {'Token':<20} {'id':>7}  {'diff':>6}  {'f_score':>7}  {'r_score':>7}  {'f_freq':>6}  {'r_freq':>6}")
        print("  " + "-" * 72)
        for t in semantic_tokens[:20]:
            print(
                f"  {repr(t['token_str']):<20} "
                f"{t['token_id']:>7}  "
                f"{t['differential']:>6.3f}  "
                f"{t['mean_forget_score']:>7.3f}  "
                f"{t['mean_retain_score']:>7.3f}  "
                f"{t['freq_forget']:>6}  "
                f"{t['freq_retain']:>6}"
            )

    # ← KEY FIX: return BOTH lists
    return semantic_tokens, scored_tokens


def main():
    parser = argparse.ArgumentParser(description="Identify semantic forget tokens.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--threshold", type=float, default=0.10,
        help="Differential score threshold (mean_forget - mean_retain). Default=0.10"
    )
    parser.add_argument(
        "--best-layer", type=int, default=None,
        help="Layer to use for token scoring. Default=auto (highest accuracy layer)."
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["dir"])

    # ── Load layer accuracies ─────────────────────────────────────────────
    acc_path = out_dir / "layer_accuracies.json"
    if not acc_path.exists():
        raise FileNotFoundError(f"{acc_path} not found. Run train_probe.py first.")
    with open(acc_path) as f:
        layer_accuracies = {int(k): v for k, v in json.load(f).items()}

    best_layer = args.best_layer or max(layer_accuracies, key=layer_accuracies.__getitem__)
    print(f"Using best_layer={best_layer} | accuracy={layer_accuracies.get(best_layer, 0):.4f}")

    # ── Load probes ───────────────────────────────────────────────────────
    probe_dir = out_dir / "probes"
    probe_files = sorted(probe_dir.glob("probe_layer_*.pkl"))
    if not probe_files:
        raise FileNotFoundError(f"No probes in {probe_dir}. Run train_probe.py first.")

    probes = {}
    for pf in probe_files:
        idx = int(pf.stem.split("_")[-1])
        probes[idx] = LinearProbe.load(str(pf))
    print(f"Loaded {len(probes)} probes.")

    # ── Load model ────────────────────────────────────────────────────────
    extractor = HiddenStateExtractor(
        model_name=cfg["model"]["name"],
        device=cfg["model"]["device"],
        dtype=cfg["model"]["dtype"],
        max_length=cfg["model"]["max_length"],
    )

    # ── Load TOFU texts ───────────────────────────────────────────────────
    dataset = TOFUDataset(
        forget_split=cfg["data"]["forget_split"],
        retain_split=cfg["data"]["retain_split"],
    )
    samples = dataset.get_samples(
        n_forget=cfg["data"].get("n_forget"),
        n_retain=cfg["data"].get("n_retain"),
        seed=cfg["data"].get("seed", 42),
    )
    forget_texts = [s.text for s in samples if s.label == 1]
    retain_texts = [s.text for s in samples if s.label == 0]
    print(f"Forget: {len(forget_texts)} | Retain: {len(retain_texts)}")

    # ── Run differential scoring ──────────────────────────────────────────
    # ← KEY FIX: unpack both return values
    semantic_tokens, scored_tokens = identify_semantic_tokens_differential(
        extractor=extractor,
        probes=probes,
        forget_texts=forget_texts,
        retain_texts=retain_texts,
        best_layer=best_layer,
        threshold=args.threshold,
        batch_size=cfg["extraction"]["batch_size"],
    )

    # ── Near-miss debug: tokens just below threshold ──────────────────────
    print("\nTokens just below threshold (0.05 to threshold):")
    near_miss = [t for t in scored_tokens if 0.05 <= t["differential"] < args.threshold]
    near_miss.sort(key=lambda x: x["differential"], reverse=True)
    if near_miss:
        print(f"  {'Token':<20} {'diff':>6}  {'f_score':>7}  {'r_score':>7}  {'f_freq':>6}  {'r_freq':>6}")
        print("  " + "-" * 65)
        for t in near_miss[:20]:
            print(
                f"  {repr(t['token_str']):<20} "
                f"{t['differential']:>6.3f}  "
                f"{t['mean_forget_score']:>7.3f}  "
                f"{t['mean_retain_score']:>7.3f}  "
                f"{t['freq_forget']:>6}  "
                f"{t['freq_retain']:>6}"
            )
    else:
        print("  (none)")

    # ── Merge probe tokens + explicit entity tokens from config ───────────
    entity_cfg = cfg.get("forget_entity", {})
    tier1 = entity_cfg.get("tier1_name_token_ids", [])
    tier2 = entity_cfg.get("tier2_location_token_ids", [])
    tier3 = entity_cfg.get("tier3_concept_token_ids", [])

    existing_ids = {t["token_id"] for t in semantic_tokens}

    for tier_name, tier_ids in [
        ("tier1_name",     tier1),
        ("tier2_location", tier2),
        ("tier3_concept",  tier3),
    ]:
        for tid in tier_ids:
            if tid not in existing_ids:
                semantic_tokens.append({
                    "token_id":          tid,
                    "token_str":         extractor.decode_token(tid),
                    "mean_forget_score": 1.0,
                    "mean_retain_score": 0.0,
                    "differential":      1.0,
                    "freq_forget":       -1,   # -1 = explicitly added, not probe-found
                    "freq_retain":       -1,
                    "best_layer":        best_layer,
                    "source":            tier_name,
                })
                existing_ids.add(tid)

    n_probe    = sum(1 for t in semantic_tokens if t.get("source") == "probe")
    n_explicit = sum(1 for t in semantic_tokens if t.get("source") != "probe")

    print(f"\nAfter merging probe + entity tokens:")
    print(f"  Probe-identified : {n_probe}")
    print(f"  Explicit (tiers) : {n_explicit}  (tier1={len(tier1)}, tier2={len(tier2)}, tier3={len(tier3)})")
    print(f"  Total T_f        : {len(semantic_tokens)}")

    # ── Save final results ────────────────────────────────────────────────
    output = {
        "threshold":         args.threshold,
        "scoring_method":    "differential (mean_forget - mean_retain)",
        "best_layer":        best_layer,
        "n_forget_texts":    len(forget_texts),
        "n_retain_texts":    len(retain_texts),
        "n_semantic_tokens": len(semantic_tokens),
        "n_probe_tokens":    n_probe,
        "n_explicit_tokens": n_explicit,
        "token_ids":         [t["token_id"] for t in semantic_tokens],
        "token_strings":     [t["token_str"] for t in semantic_tokens],
        "semantic_tokens":   semantic_tokens,
    }

    save_path = out_dir / "semantic_tokens.json"
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[✓] Saved {len(semantic_tokens)} tokens to {save_path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    if cfg["output"].get("save_plots", True) and semantic_tokens:
        plot_token_scores(
            token_strs=[t["token_str"] for t in semantic_tokens],
            token_scores=[t["differential"] for t in semantic_tokens],
            title=f"Semantic Tokens — Differential Score (Layer {best_layer})",
            top_k=30,
            save_path=str(out_dir / "token_scores.png"),
            show=False,
        )
        print(f"[✓] Saved plot to {out_dir}/token_scores.png")

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n=== Final Summary ===")
    print(f"  Scoring:    differential (mean_forget - mean_retain)")
    print(f"  Layer:      {best_layer}")
    print(f"  Threshold:  {args.threshold}")
    print(f"  T_f size:   {len(semantic_tokens)}")
    print(f"  Top tokens: {[t['token_str'] for t in semantic_tokens[:10]]}")
    print(f"\n  → Next step: use token_ids from {save_path} for embedding erasure")


if __name__ == "__main__":
    main()