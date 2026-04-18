"""
scripts/identify_tokens.py
---------------------------
Identifies semantic tokens T_f using two complementary analyses:

  Branch 1 — FREQUENCY ANALYSIS (entity tokens):
    Tokens that appear frequently in D_f but rarely in D_r
    → captures name fragments, locations (surface identity)

  Branch 2 — PROBE ANALYSIS (thematic tokens):
    Tokens whose hidden state representations are statistically
    distinctive between D_f and D_r (differential score)
    → captures thematic/conceptual footprint

  T_f = frequency_tokens ∪ probe_tokens

Best layer is auto-selected from outputs/layer_accuracies.json.
Override with --best-layer to test specific layers (e.g. layer 4).
No manual token IDs. No NER. Only D_f and D_r as input.
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


# ── Branch 1: Frequency Analysis ─────────────────────────────────────────────
def find_frequency_tokens(
    forget_texts: list,
    retain_texts: list,
    tokenizer,
    min_forget_count: int = 2,
    max_retain_ratio: float = 0.05,
    min_token_length: int = 2,
) -> list[dict]:
    freq_forget = defaultdict(int)
    freq_retain = defaultdict(int)

    print("[Frequency] Counting token frequencies in forget texts...")
    for text in forget_texts:
        ids = set(tokenizer.encode(text, add_special_tokens=False))
        for tid in ids:
            freq_forget[tid] += 1

    print("[Frequency] Counting token frequencies in retain texts...")
    for text in retain_texts:
        ids = set(tokenizer.encode(text, add_special_tokens=False))
        for tid in ids:
            freq_retain[tid] += 1

    n_retain = len(retain_texts)
    results  = []

    for tid, f_count in freq_forget.items():
        r_count      = freq_retain.get(tid, 0)
        retain_ratio = r_count / n_retain

        if f_count < min_forget_count:
            continue
        if retain_ratio > max_retain_ratio:
            continue

        token_str = tokenizer.decode([tid])
        if len(token_str.strip()) < min_token_length:
            continue

        results.append({
            "token_id":          tid,
            "token_str":         token_str,
            "freq_forget":       f_count,
            "freq_retain":       r_count,
            "retain_ratio":      retain_ratio,
            "differential":      f_count / (r_count + 1),
            "mean_forget_score": 0.0,
            "mean_retain_score": 0.0,
            "best_layer":        -1,
            "source":            "frequency",
        })

    results.sort(key=lambda x: x["freq_forget"], reverse=True)

    print(f"[Frequency] Found {len(results)} entity tokens")
    if results:
        print(f"  Top 10: {[t['token_str'] for t in results[:10]]}")

    return results


# ── Branch 2: Probe Differential Scoring ─────────────────────────────────────
def identify_probe_tokens(
    extractor,
    probes: dict,
    forget_texts: list,
    retain_texts: list,
    best_layer: int,
    threshold: float = 0.10,
    batch_size: int = 8,
) -> list[dict]:
    layers = [best_layer]
    probe  = probes[best_layer]

    print(f"[Probe] Extracting forget token states at layer {best_layer}...")
    forget_per_token = extractor.extract_per_token(
        forget_texts, batch_size=batch_size, layers=layers
    )
    forget_token_ids = extractor.get_token_ids(forget_texts)

    print(f"[Probe] Extracting retain token states at layer {best_layer}...")
    retain_per_token = extractor.extract_per_token(
        retain_texts, batch_size=batch_size, layers=layers
    )
    retain_token_ids = extractor.get_token_ids(retain_texts)

    token_forget_scores = defaultdict(list)
    token_retain_scores = defaultdict(list)
    token_freq_forget   = defaultdict(int)
    token_freq_retain   = defaultdict(int)

    print("[Probe] Scoring tokens in forget texts...")
    for text_idx, token_id_seq in enumerate(
            tqdm(forget_token_ids, desc="Forget scoring")):
        h        = forget_per_token[best_layer][text_idx]
        seq_len  = min(len(token_id_seq), h.shape[0])
        probs    = probe.predict_proba(h[:seq_len])
        p_forget = probs[:, 1]
        for pos, tid in enumerate(token_id_seq[:seq_len]):
            token_forget_scores[tid].append(float(p_forget[pos]))
            token_freq_forget[tid] += 1

    print("[Probe] Scoring tokens in retain texts...")
    for text_idx, token_id_seq in enumerate(
            tqdm(retain_token_ids, desc="Retain scoring")):
        h        = retain_per_token[best_layer][text_idx]
        seq_len  = min(len(token_id_seq), h.shape[0])
        probs    = probe.predict_proba(h[:seq_len])
        p_forget = probs[:, 1]
        for pos, tid in enumerate(token_id_seq[:seq_len]):
            token_retain_scores[tid].append(float(p_forget[pos]))
            token_freq_retain[tid] += 1

    scored_tokens = []
    for tid, f_scores in token_forget_scores.items():
        mean_f       = float(np.mean(f_scores))
        r_scores     = token_retain_scores.get(tid, [0.0])
        mean_r       = float(np.mean(r_scores))
        differential = mean_f - mean_r

        scored_tokens.append({
            "token_id":          tid,
            "token_str":         extractor.decode_token(tid),
            "mean_forget_score": mean_f,
            "mean_retain_score": mean_r,
            "differential":      differential,
            "freq_forget":       token_freq_forget[tid],
            "freq_retain":       token_freq_retain.get(tid, 0),
            "best_layer":        best_layer,
            "source":            "probe",
        })

    scored_tokens.sort(key=lambda x: x["differential"], reverse=True)
    probe_tokens = [t for t in scored_tokens if t["differential"] >= threshold]

    print(f"[Probe] Found {len(probe_tokens)} thematic tokens above threshold {threshold}")
    if probe_tokens:
        print(f"  Top 10: {[t['token_str'] for t in probe_tokens[:10]]}")

    return probe_tokens, scored_tokens


# ── Union ─────────────────────────────────────────────────────────────────────
def union_tokens(
    frequency_tokens: list,
    probe_tokens: list,
    tokenizer,
) -> list[dict]:
    freq_ids  = {t["token_id"]: t for t in frequency_tokens}
    probe_ids = {t["token_id"]: t for t in probe_tokens}

    all_ids = set(freq_ids.keys()) | set(probe_ids.keys())
    result  = []

    for tid in all_ids:
        if tid in freq_ids and tid in probe_ids:
            f = freq_ids[tid]
            p = probe_ids[tid]
            result.append({
                "token_id":          tid,
                "token_str":         tokenizer.decode([tid]),
                "freq_forget":       f["freq_forget"],
                "freq_retain":       f["freq_retain"],
                "differential":      p["differential"],
                "mean_forget_score": p["mean_forget_score"],
                "mean_retain_score": p["mean_retain_score"],
                "best_layer":        p["best_layer"],
                "source":            "both",
            })
        elif tid in freq_ids:
            result.append(freq_ids[tid])
        else:
            result.append(probe_ids[tid])

    priority = {"both": 0, "frequency": 1, "probe": 2}
    result.sort(key=lambda x: (priority[x["source"]], -x.get("freq_forget", 0)))

    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config/config.yaml")
    parser.add_argument("--best-layer", type=int, default=None,
                        help="Override probe layer. Default=auto from layer_accuracies.json.")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Probe differential threshold. Overrides config.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["dir"])

    # ── Auto-select best layer from saved accuracies ──────────────────────
    acc_path = out_dir / "layer_accuracies.json"
    if not acc_path.exists():
        raise FileNotFoundError(f"{acc_path} not found. Run train_probe.py first.")
    with open(acc_path) as f:
        layer_accuracies = {int(k): v for k, v in json.load(f).items()}

    best_layer = (args.best_layer if args.best_layer is not None
                  else max(layer_accuracies, key=layer_accuracies.__getitem__))
    print(f"[Layer] Using layer {best_layer} | AUC={layer_accuracies[best_layer]:.4f}"
          + (" (manual override)" if args.best_layer else " (auto-selected)"))

    # ── Load probes ───────────────────────────────────────────────────────
    probe_dir   = out_dir / "probes"
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
    tokenizer = extractor.tokenizer

    # ── Load texts ────────────────────────────────────────────────────────
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

    # ── Frequency config ──────────────────────────────────────────────────
    freq_cfg = cfg.get("frequency_analysis", {})
    min_f    = freq_cfg.get("min_forget_count", 2)
    max_r    = freq_cfg.get("max_retain_ratio", 0.05)
    min_len  = freq_cfg.get("min_token_length", 2)

    # ── Probe threshold ───────────────────────────────────────────────────
    threshold = (args.threshold
                 or cfg.get("probe_analysis", {}).get("threshold", 0.10))

    print(f"\n=== Branch 1: Frequency Analysis ===")
    print(f"  min_forget_count={min_f}, max_retain_ratio={max_r}, min_len={min_len}")
    frequency_tokens = find_frequency_tokens(
        forget_texts, retain_texts, tokenizer,
        min_forget_count=min_f,
        max_retain_ratio=max_r,
        min_token_length=min_len,
    )

    print(f"\n=== Branch 2: Probe Differential Analysis ===")
    print(f"  threshold={threshold}, layer={best_layer}")
    probe_tokens, all_scored = identify_probe_tokens(
        extractor, probes, forget_texts, retain_texts,
        best_layer=best_layer,
        threshold=threshold,
        batch_size=cfg["extraction"]["batch_size"],
    )

    print(f"\n=== Combining: Frequency ∪ Probe ===")
    semantic_tokens = union_tokens(frequency_tokens, probe_tokens, tokenizer)

    n_both  = sum(1 for t in semantic_tokens if t["source"] == "both")
    n_freq  = sum(1 for t in semantic_tokens if t["source"] == "frequency")
    n_probe = sum(1 for t in semantic_tokens if t["source"] == "probe")
    print(f"  Both:      {n_both} tokens  (confirmed by both analyses)")
    print(f"  Frequency: {n_freq} tokens  (entity name/location tokens)")
    print(f"  Probe:     {n_probe} tokens  (thematic/semantic tokens)")
    print(f"  Total T_f: {len(semantic_tokens)} tokens")

    # ── Save ──────────────────────────────────────────────────────────────
    output = {
        "method":             "frequency_union_probe",
        "best_layer":         best_layer,
        "probe_threshold":    threshold,
        "freq_min_count":     min_f,
        "freq_max_ratio":     max_r,
        "n_forget_texts":     len(forget_texts),
        "n_retain_texts":     len(retain_texts),
        "n_semantic_tokens":  len(semantic_tokens),
        "n_frequency_tokens": n_freq + n_both,
        "n_probe_tokens":     n_probe + n_both,
        "token_ids":          [t["token_id"] for t in semantic_tokens],
        "token_strings":      [t["token_str"] for t in semantic_tokens],
        "semantic_tokens":    semantic_tokens,
    }

    save_path = out_dir / "semantic_tokens.json"
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[✓] Saved {len(semantic_tokens)} tokens to {save_path}")

    # ── Print table ───────────────────────────────────────────────────────
    print(f"\n  {'Token':<20} {'source':<12} {'f_freq':>7} {'r_freq':>7} {'diff':>7}")
    print(f"  {'-'*58}")
    for t in semantic_tokens[:25]:
        print(
            f"  {repr(t['token_str']):<20} "
            f"{t['source']:<12} "
            f"{t.get('freq_forget',0):>7} "
            f"{t.get('freq_retain',0):>7} "
            f"{t.get('differential',0):>7.3f}"
        )

    # ── Plot ──────────────────────────────────────────────────────────────
    if cfg["output"].get("save_plots", True) and semantic_tokens:
        plot_token_scores(
            token_strs=[t["token_str"] for t in semantic_tokens],
            token_scores=[t.get("differential", 1.0) for t in semantic_tokens],
            title=f"Semantic Tokens — Frequency ∪ Probe (Layer {best_layer})",
            top_k=30,
            save_path=str(out_dir / "token_scores.png"),
            show=False,
        )

    print(f"\n=== Summary ===")
    print(f"  No hardcoded token IDs used.")
    print(f"  No NER used.")
    print(f"  Input: D_f ({len(forget_texts)} sentences) + D_r ({len(retain_texts)} sentences)")
    print(f"  Output: T_f = {len(semantic_tokens)} tokens")
    print(f"  Sources: {n_both} both, {n_freq} freq-only, {n_probe} probe-only")
    print(f"\n  → Next step: erase_embeddings.py")


if __name__ == "__main__":
    main()