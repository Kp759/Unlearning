#!/usr/bin/env python3
"""
scripts/identify_tokens.py

FREQUENCY-ONLY VERSION, modified for the hybrid pipeline.

Main change:
  - Saves frequency candidates to outputs/semantic_tokens_freq.json.
  - Also writes outputs/semantic_tokens.json by default for old pipeline compatibility.

Hybrid usage:
  python scripts/identify_tokens.py --config config/config_3b_instruct_forget05.yaml

Then:
  outputs/semantic_tokens_freq.json is used by filter_forget_tokens_retain_tfidf.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import TOFUDataset
from src.probing import HiddenStateExtractor
from src.utils import plot_token_scores


# ── Branch 1: Frequency Analysis ─────────────────────────────────────────────

def find_frequency_tokens(
    forget_texts: list,
    retain_texts: list,
    tokenizer,
    min_forget_count: int = 2,
    max_retain_ratio: float = 0.05,
    min_token_length: int = 2,
) -> List[Dict]:
    """
    Finds token IDs that appear frequently in forget texts and rarely in retain texts.
    Counts document frequency: each token is counted at most once per QA text.
    """
    freq_forget = defaultdict(int)
    freq_retain = defaultdict(int)

    print("[Frequency] Counting token frequencies in forget texts...")
    for text in tqdm(forget_texts, desc="Forget token DF"):
        ids = set(tokenizer.encode(text, add_special_tokens=False))
        for tid in ids:
            freq_forget[int(tid)] += 1

    print("[Frequency] Counting token frequencies in retain texts...")
    for text in tqdm(retain_texts, desc="Retain token DF"):
        ids = set(tokenizer.encode(text, add_special_tokens=False))
        for tid in ids:
            freq_retain[int(tid)] += 1

    n_forget = len(forget_texts)
    n_retain = len(retain_texts)

    results = []
    for tid, f_count in freq_forget.items():
        tid = int(tid)
        r_count = int(freq_retain.get(tid, 0))

        forget_ratio = f_count / max(1, n_forget)
        retain_ratio = r_count / max(1, n_retain)

        if f_count < min_forget_count:
            continue

        if retain_ratio > max_retain_ratio:
            continue

        token_str = tokenizer.decode([tid])
        if len(token_str.strip()) < min_token_length:
            continue

        results.append(
            {
                "token_id": tid,
                "token_str": token_str,
                "freq_forget": int(f_count),
                "freq_retain": int(r_count),
                "forget_ratio": float(forget_ratio),
                "retain_ratio": float(retain_ratio),
                "differential": float(f_count / (r_count + 1)),
                "mean_forget_score": 0.0,
                "mean_retain_score": 0.0,
                "best_layer": -1,
                "source": "frequency",
            }
        )

    results.sort(
        key=lambda x: (
            -int(x["freq_forget"]),
            int(x["freq_retain"]),
            -float(x["differential"]),
            int(x["token_id"]),
        )
    )

    print(f"[Frequency] Found {len(results)} candidate tokens")
    if results:
        print(f"[Frequency] Top 10: {[t['token_str'] for t in results[:10]]}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")

    # Kept for CLI compatibility with the old semantic/probe pipeline.
    parser.add_argument(
        "--best-layer",
        type=int,
        default=None,
        help="Unused in frequency-only mode. Kept only for CLI compatibility.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Unused in frequency-only mode. Kept only for CLI compatibility.",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Frequency output path. Default: <output.dir>/semantic_tokens_freq.json",
    )
    parser.add_argument(
        "--no-legacy-semantic-copy",
        action="store_true",
        help="If set, do not also write <output.dir>/semantic_tokens.json.",
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_layer = -1
    threshold = None

    print("[Mode] Running frequency-only token identification.")
    print("[Mode] Probe loading and probe scoring are skipped.")

    # ── Load model / tokenizer ────────────────────────────────────────────
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
    min_f = int(freq_cfg.get("min_forget_count", 2))
    max_r = float(freq_cfg.get("max_retain_ratio", 0.05))
    min_len = int(freq_cfg.get("min_token_length", 2))

    print("\n=== Branch 1: Frequency Analysis ===")
    print(f" min_forget_count={min_f}, max_retain_ratio={max_r}, min_len={min_len}")

    frequency_tokens = find_frequency_tokens(
        forget_texts,
        retain_texts,
        tokenizer,
        min_forget_count=min_f,
        max_retain_ratio=max_r,
        min_token_length=min_len,
    )

    semantic_tokens = frequency_tokens
    n_freq = len(semantic_tokens)

    print("\n=== Final Token Set: Frequency Only ===")
    print(f" Frequency: {n_freq} tokens")
    print(f" Total T_f: {len(semantic_tokens)} tokens")

    output = {
        "method": "frequency_only",
        "best_layer": best_layer,
        "probe_threshold": threshold,
        "freq_min_count": min_f,
        "freq_max_ratio": max_r,
        "n_forget_texts": len(forget_texts),
        "n_retain_texts": len(retain_texts),
        "n_semantic_tokens": len(semantic_tokens),
        "n_frequency_tokens": len(semantic_tokens),
        "n_probe_tokens": 0,
        "token_ids": [int(t["token_id"]) for t in semantic_tokens],
        "token_strings": [t["token_str"] for t in semantic_tokens],
        "semantic_tokens": semantic_tokens,
    }

    freq_save_path = Path(args.out) if args.out else out_dir / "semantic_tokens_freq.json"
    with open(freq_save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] Saved frequency tokens to {freq_save_path}")

    # Legacy copy keeps old freq-only pipeline working:
    # python scripts/identify_tokens.py && python scripts/erase_embeddings.py
    if not args.no_legacy_semantic_copy:
        legacy_path = out_dir / "semantic_tokens.json"
        with open(legacy_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[✓] Also saved legacy semantic token file to {legacy_path}")

    # ── Print table ───────────────────────────────────────────────────────
    print(f"\n {'Token':<20} {'source':<12} {'f_freq':>7} {'r_freq':>7} {'diff':>7}")
    print(f" {'-' * 58}")

    for t in semantic_tokens[:25]:
        print(
            f" {repr(t['token_str']):<20} "
            f"{t['source']:<12} "
            f"{int(t.get('freq_forget', 0)):>7} "
            f"{int(t.get('freq_retain', 0)):>7} "
            f"{float(t.get('differential', 0.0)):>7.3f}"
        )

    if cfg["output"].get("save_plots", True) and semantic_tokens:
        plot_token_scores(
            token_strs=[t["token_str"] for t in semantic_tokens],
            token_scores=[t.get("differential", 1.0) for t in semantic_tokens],
            title="Frequency Tokens Only",
            top_k=30,
            save_path=str(out_dir / "token_scores.png"),
            show=False,
        )

    print("\n=== Summary ===")
    print(" No hardcoded token IDs used.")
    print(" No NER used.")
    print(f" Input: D_f ({len(forget_texts)} sentences) + D_r ({len(retain_texts)} sentences)")
    print(f" Output: T_f = {len(semantic_tokens)} tokens")
    print(" Source: frequency-only")
    print("\n → Hybrid next step:")
    print("   python scripts/build_llm_forget_bank.py ... --out-semantic-tokens outputs/semantic_tokens_json_raw.json")
    print("   python scripts/filter_forget_tokens_retain_tfidf.py --config <config>")


if __name__ == "__main__":
    main()
