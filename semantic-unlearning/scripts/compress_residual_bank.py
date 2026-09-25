#!/usr/bin/env python3
"""Compress the residual bank of a frozen linear-router run.

Input is a run directory whose artifact is the linear router
(`linear_classifier_fact_association_bank_v1`) with the frozen configuration:
one global threshold (MCF, ZsRE, MQuAKE) or the subject gate (RWKU).

Every variant keeps the router, threshold, subject patterns and facts exactly
and replaces only the N residual rows. Routing therefore cannot change: the
hook reads block `layer`'s output at the request boundary before it adds the
row. The only thing that changes is the edit itself, which is what the
official evaluators measure.

Variants (each one is a run directory the existing evaluators load):

  rank{K}           rows ~= codes[N,K] @ basis[K,d] (truncated SVD). Each row
                    is rescaled to its original norm; the scale is folded
                    into the codes, so storage is still codes + basis.
  int8              per-row symmetric int8 quantization of the full rows
  rank{K}_int8      rank-K codes and basis, each int8 per row
  tied_answer       one shared direction per distinct answer (the fact's
                    object) + one scale per fact
  tied_relation     one shared direction per relation + one scale per fact
                    (only when the benchmark has more than one relation)
  tied_single       one direction for every fact + one scale per fact

Controls (not compressions; they say what the compressions mean):

  control_shuffled  each fact gets another fact's direction at its own norm
                    (a derangement). If this forgets as well as the original,
                    the rows are not fact-specific.
  control_random    a random direction at the fact's own norm. If this
                    forgets too, only the norm matters.

Rows are rebuilt in float32 from the stored compact tensors by
`reconstruct`, the same function a compact runtime would use. The evaluated
rows are therefore exactly what such a runtime computes, before the usual
cast to the model dtype.

At the seed-1 sizes (N = 50 or 105) a rank-K bank only saves memory when
N > K*d/(d-K), roughly N > K; the report gives this break-even N. What seed 1
can establish is the smallest K that keeps the metrics (K*). Whether K* stays
flat as N grows needs the scaling runs.

    python -u scripts/compress_residual_bank.py \
      --run-dir outputs/mcf_linear_2x2_seed1_v24/arms/linear_global \
      --output-dir outputs/mcf_linear_global_bankcomp_seed1
    bash outputs/mcf_linear_global_bankcomp_seed1/run_evals.sh
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

LINEAR_ARCHITECTURE = "linear_classifier_fact_association_bank_v1"
SCHEMA = "residual_bank_compression_v1"
ARTIFACT_NAME = "fact_association_embeddings.pt"
MANIFEST_NAME = "association_manifest.json"
REPORT_NAME = "compression_report.json"
DEFAULT_RANKS = (1, 2, 4, 8, 16, 32, 64)
EXTRAPOLATE_N = (1_000, 10_000, 100_000)

EVALUATORS = {
    "mcf": (
        "evaluate_static_overlap_fact_association_embeddings_official.py",
        '--mcf-path "$MCF_PATH" --wikidata-dir "$WIKIDATA_DIR" '
        "--device cuda --dtype bfloat16 --local-files-only",
    ),
    "zsre": (
        "evaluate_zsre_fact_association_embeddings_official.py",
        '--zsre-path "$ZSRE_PATH" --wikidata-dir "$WIKIDATA_DIR" '
        "--device cuda --dtype bfloat16 --batch-size 8 --local-files-only",
    ),
    "mquake": (
        "evaluate_mquake_fact_association_embeddings_official.py",
        '--mquake-path "$MQUAKE_PATH" --wikidata-dir "$WIKIDATA_DIR" '
        "--device cuda --dtype bfloat16 --batch-size 8 --local-files-only",
    ),
    "rwku": (
        "evaluate_rwku_fact_association_embeddings_seed1.py",
        '--data-root "$RWKU_ROOT" --wikidata-dir "$WIKIDATA_DIR" '
        "--device cuda --dtype bfloat16 --local-files-only",
    ),
}
EVAL_FILES = {
    "mcf": "official_mcf_eval.json",
    "zsre": "official_zsre_eval.json",
    "mquake": "official_mquake_eval.json",
    "rwku": "official_rwku_batch50_eval.json",
}


# ---------------------------------------------------------------------------
# Loading and checks
# ---------------------------------------------------------------------------

def detect_benchmark(facts):
    first = str(facts[0].get("id", "")) if facts else ""
    for name in ("mcf", "zsre", "mquake", "rwku"):
        if first.startswith(f"{name}_"):
            return name
    return "unknown"


def load_run(run_dir):
    run_dir = Path(run_dir)
    artifact = torch.load(run_dir / ARTIFACT_NAME, map_location="cpu", weights_only=False)
    manifest_path = run_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    return artifact, manifest


def check_frozen_config(artifact, allow_per_head=False):
    """The frozen configuration: linear router, global threshold or subject gate."""
    architecture = str(artifact.get("architecture", ""))
    if architecture != LINEAR_ARCHITECTURE:
        raise ValueError(
            f"Expected a linear-router artifact ({LINEAR_ARCHITECTURE}); got {architecture!r}. "
            "Fit one with scripts/fit_linear_router.py first."
        )
    if artifact.get("per_head_thresholds") is not None and not allow_per_head:
        raise ValueError(
            "This run uses per-head thresholds. The frozen configuration is a global "
            "threshold (or the subject gate); pass --allow-per-head to compress it anyway."
        )
    if "residual_compact" in artifact:
        raise ValueError("This run is already a compressed variant; compress the source run.")
    gate = str(artifact.get("gate_mode", "threshold"))
    policy = "subject_gate" if gate == "subject" else (
        "per_head" if artifact.get("per_head_thresholds") is not None else "global"
    )
    return gate, policy


def rows_of(artifact):
    rows = artifact["rows"]
    if isinstance(rows, (list, tuple)):
        rows = torch.stack([torch.as_tensor(r) for r in rows])
    rows = torch.as_tensor(rows).detach().to(torch.float64).cpu()
    if rows.ndim != 2:
        raise ValueError("artifact rows must be [num_facts, hidden]")
    if rows.shape[0] != len(artifact["facts"]):
        raise ValueError("rows do not align with facts")
    return rows


def normalized_key(text):
    return " ".join(str(text).casefold().split())


# ---------------------------------------------------------------------------
# Compact representations
# ---------------------------------------------------------------------------

def spectrum(rows):
    singular = torch.linalg.svdvals(rows)
    energy = singular.square()
    total = float(energy.sum())
    fraction = energy / total if total > 0 else torch.zeros_like(energy)
    return {
        "singular_values": [float(v) for v in singular],
        "energy_fraction": [float(v) for v in fraction],
        "cumulative_energy": [float(v) for v in torch.cumsum(fraction, dim=0)],
        "rank_for_energy": {
            str(level): int(torch.searchsorted(torch.cumsum(fraction, 0),
                                               torch.tensor(level, dtype=fraction.dtype)).item()) + 1
            for level in (0.5, 0.8, 0.9, 0.95, 0.99)
        } if total > 0 else {},
    }


def _norm_rescale(rows, recon, tolerance=1e-8):
    """Per-row factor that restores each row's original norm."""
    original = rows.norm(dim=1)
    rebuilt = recon.norm(dim=1)
    ok = rebuilt > tolerance * original.clamp_min(1e-30)
    factor = torch.where(ok, original / rebuilt.clamp_min(1e-30), torch.ones_like(original))
    return factor, int((~ok & (original > 0)).sum())


def low_rank_compact(rows, rank, rescale=True):
    u, s, vh = torch.linalg.svd(rows, full_matrices=False)
    k = int(min(rank, s.numel()))
    codes = u[:, :k] * s[:k]
    basis = vh[:k]
    unrescaled = 0
    if rescale:
        factor, unrescaled = _norm_rescale(rows, codes @ basis)
        codes = codes * factor[:, None]
    return {
        "kind": "low_rank",
        "rank": k,
        "rescaled_to_row_norm": bool(rescale),
        "rows_left_unrescaled": unrescaled,
        "codes": codes.float().contiguous(),
        "basis": basis.float().contiguous(),
    }


def quantize_int8(matrix):
    """Symmetric per-row int8. Returns (int8 values, float32 per-row scales)."""
    m = torch.as_tensor(matrix).to(torch.float64)
    scale = m.abs().amax(dim=1) / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(torch.round(m / scale[:, None]), -127, 127).to(torch.int8)
    return q.contiguous(), scale.float().contiguous()


def dequantize_int8(q, scale):
    return q.float() * scale.float()[:, None]


def int8_compact(rows):
    q, scale = quantize_int8(rows)
    return {"kind": "int8", "q": q, "scale": scale}


def low_rank_int8_compact(rows, rank, rescale=True):
    base = low_rank_compact(rows, rank, rescale=rescale)
    codes_q, codes_scale = quantize_int8(base["codes"])
    basis_q, basis_scale = quantize_int8(base["basis"])
    return {
        "kind": "low_rank_int8",
        "rank": base["rank"],
        "rescaled_to_row_norm": base["rescaled_to_row_norm"],
        "rows_left_unrescaled": base["rows_left_unrescaled"],
        "codes_q": codes_q, "codes_scale": codes_scale,
        "basis_q": basis_q, "basis_scale": basis_scale,
    }


def tied_compact(rows, keys):
    """One shared unit direction per key; each fact keeps its own norm."""
    order, index = {}, []
    for key in keys:
        index.append(order.setdefault(key, len(order)))
    index = torch.tensor(index, dtype=torch.int64)
    units = F.normalize(rows, dim=1)
    directions = torch.zeros((len(order), rows.shape[1]), dtype=torch.float64)
    directions.index_add_(0, index, units)
    norms = directions.norm(dim=1)
    for group in (norms <= 1e-12).nonzero(as_tuple=True)[0].tolist():
        # Members cancel exactly; fall back to the first member's direction.
        first = int((index == group).nonzero(as_tuple=True)[0][0])
        directions[group] = units[first]
    directions = F.normalize(directions, dim=1)
    return {
        "kind": "tied",
        "groups": len(order),
        "group_keys": list(order),
        "directions": directions.float().contiguous(),
        "scale": rows.norm(dim=1).float().contiguous(),
        "index": index.to(torch.int32).contiguous(),
    }


def derangement(n, seed):
    """Sattolo's algorithm: a single n-cycle, so no fact keeps its own row."""
    if n < 2:
        raise ValueError("A derangement needs at least two facts")
    generator = torch.Generator().manual_seed(int(seed))
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(torch.randint(0, i, (1,), generator=generator))
        perm[i], perm[j] = perm[j], perm[i]
    return torch.tensor(perm, dtype=torch.int64)


def shuffled_compact(rows, seed):
    perm = derangement(rows.shape[0], seed)
    return {
        "kind": "tied",
        "groups": int(rows.shape[0]),
        "group_keys": None,
        "control": "shuffled_direction_own_norm",
        "permutation_seed": int(seed),
        "directions": F.normalize(rows, dim=1).float().contiguous(),
        "scale": rows.norm(dim=1).float().contiguous(),
        "index": perm.to(torch.int32).contiguous(),
    }


def random_compact(rows, seed):
    generator = torch.Generator().manual_seed(int(seed))
    directions = F.normalize(
        torch.randn(rows.shape, generator=generator, dtype=torch.float64), dim=1
    )
    return {
        "kind": "tied",
        "groups": int(rows.shape[0]),
        "group_keys": None,
        "control": "random_direction_own_norm",
        "random_seed": int(seed),
        "directions": directions.float().contiguous(),
        "scale": rows.norm(dim=1).float().contiguous(),
        "index": torch.arange(rows.shape[0], dtype=torch.int32),
    }


def reconstruct(compact):
    """Rows [N, d] in float32 from a compact representation."""
    kind = compact["kind"]
    if kind == "low_rank":
        return compact["codes"].float() @ compact["basis"].float()
    if kind == "int8":
        return dequantize_int8(compact["q"], compact["scale"])
    if kind == "low_rank_int8":
        codes = dequantize_int8(compact["codes_q"], compact["codes_scale"])
        basis = dequantize_int8(compact["basis_q"], compact["basis_scale"])
        return codes @ basis
    if kind == "tied":
        directions = compact["directions"].float()
        return directions[compact["index"].long()] * compact["scale"].float()[:, None]
    raise ValueError(f"Unknown compact kind {kind!r}")


# ---------------------------------------------------------------------------
# Accounting and diagnostics
# ---------------------------------------------------------------------------

def _tensors(compact):
    return {k: v for k, v in compact.items() if isinstance(v, torch.Tensor)}


def storage(compact, n_facts, hidden):
    """Bytes as saved, and with float tensors counted at bfloat16."""
    saved = bf16 = 0
    for tensor in _tensors(compact).values():
        size = tensor.numel() * tensor.element_size()
        saved += size
        bf16 += tensor.numel() * 2 if tensor.is_floating_point() else size
    full_fp32 = n_facts * hidden * 4
    full_bf16 = n_facts * hidden * 2
    return {
        "bytes_as_saved": int(saved),
        "bytes_floats_as_bf16": int(bf16),
        "full_rows_fp32_bytes": int(full_fp32),
        "full_rows_bf16_bytes": int(full_bf16),
        "ratio_vs_full_fp32": full_fp32 / saved if saved else None,
        "ratio_vs_full_bf16": full_bf16 / bf16 if bf16 else None,
    }


def per_n_bytes(compact, n, hidden):
    """Storage at another N with the same structure (floats as saved).

    Only structural: accuracy at that N is not measured here. Tied variants
    depend on how many groups exist at that N, so they are not extrapolated.
    """
    kind = compact["kind"]
    if kind == "low_rank":
        k = compact["rank"]
        return 4 * (n * k + k * hidden)
    if kind == "low_rank_int8":
        k = compact["rank"]
        return n * k + 4 * n + k * hidden + 4 * k
    if kind == "int8":
        return n * hidden + 4 * n
    return None


def break_even_facts(compact, hidden):
    """Smallest N at which the variant stores fewer bytes than full fp32 rows."""
    kind = compact["kind"]
    if kind == "low_rank":
        k = compact["rank"]
        # 4(Nk + kd) < 4Nd  <=>  N > kd / (d - k)
        return None if k >= hidden else (k * hidden) // (hidden - k) + 1
    if kind == "low_rank_int8":
        k = compact["rank"]
        # Nk + 4N + kd + 4k < 4Nd  <=>  N > (kd + 4k) / (4d - k - 4)
        denominator = 4 * hidden - k - 4
        return None if denominator <= 0 else (k * hidden + 4 * k) // denominator + 1
    if kind == "int8":
        return 1
    return None


def reconstruction_stats(rows, rebuilt):
    rows = rows.to(torch.float64)
    rebuilt = rebuilt.to(torch.float64)
    cosine = F.cosine_similarity(rows, rebuilt, dim=1)
    original = rows.norm(dim=1)
    ratio = rebuilt.norm(dim=1) / original.clamp_min(1e-30)
    frob = float(rows.norm())
    return {
        "row_cosine_mean": float(cosine.mean()),
        "row_cosine_min": float(cosine.min()),
        "row_norm_ratio_min": float(ratio.min()),
        "row_norm_ratio_mean": float(ratio.mean()),
        "row_norm_ratio_max": float(ratio.max()),
        "relative_frobenius_error": float((rows - rebuilt).norm()) / frob if frob else 0.0,
    }


def pairwise_cosines(rows):
    units = F.normalize(rows.to(torch.float64), dim=1)
    gram = units @ units.T
    n = gram.shape[0]
    off = gram[~torch.eye(n, dtype=torch.bool)]
    return {
        "mean": float(off.mean()) if off.numel() else None,
        "mean_abs": float(off.abs().mean()) if off.numel() else None,
        "max": float(off.max()) if off.numel() else None,
        "min": float(off.min()) if off.numel() else None,
    }


def group_stats(rows, keys):
    units = F.normalize(rows.to(torch.float64), dim=1)
    gram = units @ units.T
    labels = {}
    index = torch.tensor([labels.setdefault(k, len(labels)) for k in keys])
    same = index[:, None] == index[None, :]
    off = ~torch.eye(len(keys), dtype=torch.bool)
    within = gram[same & off]
    between = gram[~same]
    sizes = torch.bincount(index)
    return {
        "groups": int(sizes.numel()),
        "largest_group": int(sizes.max()) if sizes.numel() else 0,
        "singleton_groups": int((sizes == 1).sum()),
        "facts_in_shared_groups": int(sizes[sizes > 1].sum()),
        "within_group_cosine_mean": float(within.mean()) if within.numel() else None,
        "between_group_cosine_mean": float(between.mean()) if between.numel() else None,
    }


def router_storage(artifact):
    parts = {
        name: artifact.get(name) for name in
        ("router_weight", "router_bias", "feature_mean", "feature_components")
    }
    counts = {
        name: (0 if value is None else int(torch.as_tensor(value).numel()))
        for name, value in parts.items()
    }
    weight = torch.as_tensor(artifact["router_weight"])
    components = artifact.get("feature_components")
    return {
        "feature_dim": int(weight.shape[1]),
        "pca_dim": None if components is None else int(torch.as_tensor(components).shape[0]),
        "parameters": counts,
        "bytes_fp32": int(4 * sum(counts.values())),
    }


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

def build_variants(rows, facts, ranks, rescale=True, int8=True, tied=True,
                   controls=True, seed=0):
    n, hidden = rows.shape
    variants = []
    for k in sorted({int(k) for k in ranks}):
        if k < 1 or k >= min(n, hidden):
            continue  # k >= min(N, d) is the exact bank
        variants.append((f"rank{k}", low_rank_compact(rows, k, rescale=rescale)))
        if int8:
            variants.append((f"rank{k}_int8", low_rank_int8_compact(rows, k, rescale=rescale)))
    if int8:
        variants.append(("int8", int8_compact(rows)))
    if tied:
        answers = [normalized_key(f.get("object", "")) for f in facts]
        variants.append(("tied_answer", tied_compact(rows, answers)))
        relations = [normalized_key(f.get("relation", "")) for f in facts]
        if 1 < len(set(relations)) < n:
            variants.append(("tied_relation", tied_compact(rows, relations)))
        variants.append(("tied_single", tied_compact(rows, ["all"] * n)))
    if controls and n >= 2:
        variants.append(("control_shuffled", shuffled_compact(rows, seed)))
        variants.append(("control_random", random_compact(rows, seed)))
    return variants


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return None
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")


def variant_metadata(name, compact, rows, rebuilt):
    n, hidden = rows.shape
    meta = {
        key: value for key, value in compact.items()
        if not isinstance(value, torch.Tensor) and key != "group_keys"
    }
    meta.update({
        "variant": name,
        "is_control": name.startswith("control_"),
        "reconstruction": reconstruction_stats(rows, rebuilt),
        "storage": storage(compact, n, hidden),
        "break_even_facts_vs_full_fp32": break_even_facts(compact, hidden),
        "bytes_at_n": {str(m): per_n_bytes(compact, m, hidden) for m in EXTRAPOLATE_N},
    })
    return meta


def write_variant(out_dir, artifact, manifest, compact, meta, source_run_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    rebuilt = reconstruct(compact)
    variant = dict(artifact)
    variant["rows"] = rebuilt.clone()
    variant["residual_compact"] = dict(compact)
    variant["residual_compression"] = _json_safe(meta)
    variant["residual_source_run_dir"] = str(source_run_dir)
    torch.save(variant, out_dir / ARTIFACT_NAME)
    new_manifest = dict(manifest)
    new_manifest.update({
        "residual_compression": _json_safe(meta),
        "residual_source_run_dir": str(source_run_dir),
        "residual_rows_materialized_from_compact": True,
        "router_unchanged_from_source": True,
    })
    _write_json(out_dir / MANIFEST_NAME, new_manifest)
    return out_dir


def eval_script(benchmark, source_run_dir, variant_dirs, output_dir, skip_ppl=True):
    if benchmark not in EVALUATORS:
        return None
    script, flags = EVALUATORS[benchmark]
    eval_file = EVAL_FILES[benchmark]
    lines = [
        "#!/usr/bin/env bash",
        "# Run from semantic-unlearning/. Evaluates the uncompressed source run",
        "# (only if its eval file is missing) and every compressed variant, then",
        "# prints one comparison table.",
        "set -euo pipefail",
        'export PYTHONPATH="$PWD/scripts"',
        'MCF_PATH="${MCF_PATH:-data/multi_counterfact.json}"',
        'ZSRE_PATH="${ZSRE_PATH:-data/zsre_mend_eval.json}"',
        'MQUAKE_PATH="${MQUAKE_PATH:-data/MQuAKE-CF-3k-v2.json}"',
        'RWKU_ROOT="${RWKU_ROOT:-data/rwku}"',
        'WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"',
        "# Routing is identical to the source run, so PPL changes only if the PPL",
        "# text itself routes (it did not in any seed-1 run). Set SKIP_PPL= to include it.",
        'SKIP_PPL="${SKIP_PPL-' + ("--skip-ppl" if skip_ppl else "") + '}"',
        "",
        f'SOURCE="{source_run_dir}"',
        f'if [ ! -f "$SOURCE/{eval_file}" ]; then',
        '  echo "========== source (uncompressed) =========="',
        f'  python -u scripts/{script} --run-dir "$SOURCE" {flags}',
        "fi",
        "",
        "for run in \\",
    ]
    lines += [f'  "{path}" \\' for path in variant_dirs]
    lines += [
        "  ; do",
        '  echo "========== $(basename "$run") =========="',
        f'  python -u scripts/{script} --run-dir "$run" {flags} $SKIP_PPL',
        "done",
        "",
        f'python -u scripts/summarize_residual_compression.py --compression-dir "{output_dir}"',
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True,
                        help="linear-router run (global threshold or subject gate)")
    parser.add_argument("--output-dir", required=True, help="must not exist")
    parser.add_argument("--ranks", default=",".join(str(k) for k in DEFAULT_RANKS))
    parser.add_argument("--no-rescale", action="store_true",
                        help="keep truncated rows at their shrunken norm")
    parser.add_argument("--no-int8", action="store_true")
    parser.add_argument("--no-tied", action="store_true")
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="controls only")
    parser.add_argument("--allow-per-head", action="store_true")
    parser.add_argument("--include-ppl", action="store_true",
                        help="do not pass --skip-ppl in run_evals.sh")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"{output} exists; choose a new --output-dir")

    artifact, manifest = load_run(run_dir)
    gate, policy = check_frozen_config(artifact, allow_per_head=args.allow_per_head)
    facts = list(artifact["facts"])
    rows = rows_of(artifact)
    n, hidden = rows.shape
    benchmark = detect_benchmark(facts)
    ranks = tuple(int(x) for x in str(args.ranks).split(",") if x.strip())

    variants = build_variants(
        rows, facts, ranks, rescale=not args.no_rescale, int8=not args.no_int8,
        tied=not args.no_tied, controls=not args.no_controls, seed=args.seed,
    )
    output.mkdir(parents=True)
    report_variants, variant_dirs = {}, []
    for name, compact in variants:
        rebuilt = reconstruct(compact)
        meta = variant_metadata(name, compact, rows, rebuilt)
        path = write_variant(output / "variants" / name, artifact, manifest, compact, meta, run_dir)
        meta["run_dir"] = str(path)
        report_variants[name] = meta
        variant_dirs.append(str(path))

    answers = [normalized_key(f.get("object", "")) for f in facts]
    relations = [normalized_key(f.get("relation", "")) for f in facts]
    router = router_storage(artifact)
    report = {
        "schema_version": SCHEMA,
        "source_run_dir": str(run_dir),
        "benchmark": benchmark,
        "gate_mode": gate,
        "threshold_policy": policy,
        "threshold": artifact.get("threshold"),
        "n_facts": int(n),
        "hidden": int(hidden),
        "source_rows_dtype": str(torch.as_tensor(artifact["rows"]).dtype),
        "router": router,
        "full_bank_bytes_fp32": int(4 * n * hidden),
        "row_norms": {
            "min": float(rows.norm(dim=1).min()),
            "mean": float(rows.norm(dim=1).mean()),
            "max": float(rows.norm(dim=1).max()),
        },
        "spectrum": spectrum(rows),
        "pairwise_row_cosine": pairwise_cosines(rows),
        "answer_groups": group_stats(rows, answers),
        "relation_groups": group_stats(rows, relations),
        "variants": report_variants,
        "routing_note": (
            "Router, threshold, subject patterns and facts are copied unchanged; "
            "the hook reads the query before adding the row, so every variant makes "
            "the same routing decisions as the source run."
        ),
        "command": sys.argv if argv is None else ["compress_residual_bank.py", *argv],
    }
    _write_json(output / REPORT_NAME, report)
    script = eval_script(benchmark, run_dir, variant_dirs, output, skip_ppl=not args.include_ppl)
    if script is not None:
        (output / "run_evals.sh").write_text(script)
        (output / "run_evals.sh").chmod(0o755)

    summary = {
        "status": "residual_bank_compression_complete",
        "benchmark": benchmark,
        "n_facts": int(n),
        "threshold_policy": policy,
        "rank_for_energy": report["spectrum"]["rank_for_energy"],
        "answer_groups": report["answer_groups"]["groups"],
        "variants": {
            name: {
                "ratio_vs_full_fp32": round(meta["storage"]["ratio_vs_full_fp32"], 3),
                "row_cosine_mean": round(meta["reconstruction"]["row_cosine_mean"], 4),
            }
            for name, meta in report_variants.items()
        },
        "report": str(output / REPORT_NAME),
        "run_evals": str(output / "run_evals.sh") if script else None,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
