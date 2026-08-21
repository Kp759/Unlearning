#!/usr/bin/env python3
"""Compare v8/v9 MCF utility-ladder checkpoints after frozen evaluation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def dig(value: Mapping[str, Any], keys: Sequence[str]) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be LABEL=SEED_DIRECTORY")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("run label must be non-empty")
    return label, Path(raw_path).expanduser().resolve()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_run(label: str, root: Path) -> Dict[str, Any]:
    evaluation_path = root / "final_target_true_sensitive_eval.json"
    exact_kl_path = root / "posthoc_exact_retain_kl.json"
    config_path = root / "target_aware_direct_only_learner" / "config_used.json"
    for path in (evaluation_path, exact_kl_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    evaluation = load_json(evaluation_path)
    exact_kl = load_json(exact_kl_path)
    config = load_json(config_path)
    metrics = evaluation.get("metrics", {})
    cache = config.get("utility_cache_metadata", {})
    final = config.get("final", {})
    return {
        "experiment": label,
        "run_directory": str(root),
        "protocol": config.get("protocol"),
        "wikipedia_documents": cache.get("actual_document_sample_size"),
        "wikipedia_candidates": cache.get("actual_utility_prompt_count"),
        "generated_subject_contexts": config.get(
            "generated_subject_context_count", 0
        ),
        "external_locality_contexts": config.get(
            "external_wikipedia_locality_context_count", 0
        ),
        "FS": dig(metrics, ("FS", "mean")),
        "GFS": dig(metrics, ("GFS", "mean")),
        "Spe_margin": dig(metrics, ("Spe_margin", "mean")),
        "Spe_success": dig(metrics, ("Spe_success", "mean")),
        "PPL": metrics.get("PPL"),
        "guard_utility_kl_mean": final.get("utility_kl_mean"),
        "guard_locality_kl_mean": final.get("locality_kl_mean"),
        "exact_retain_kl_mean": dig(
            exact_kl, ("exact_kl_base_to_edited", "mean")
        ),
        "exact_retain_kl_p95": dig(
            exact_kl, ("exact_kl_base_to_edited", "p95")
        ),
        "exact_retain_kl_max": dig(
            exact_kl, ("exact_kl_base_to_edited", "max")
        ),
        "selected_row_count": exact_kl.get("selected_row_count"),
        "official_GFS_used_for_selection": bool(
            config.get("GFS_checkpoint_selection", False)
        ),
        "official_neighborhoods_used_for_selection": bool(
            config.get("neighborhood_prompts_used_for_training_or_selection", False)
        ),
    }


def format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="Repeat LABEL=SEED_DIRECTORY in the intended ladder order",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = [load_run(label, root) for label, root in args.run]
    if any(row["official_GFS_used_for_selection"] for row in rows):
        raise RuntimeError("a run used official GFS for checkpoint selection")
    if any(row["official_neighborhoods_used_for_selection"] for row in rows):
        raise RuntimeError("a run used official neighborhoods before checkpoint freeze")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "mcf_sure_utility_ladder.json"
    csv_path = output_dir / "mcf_sure_utility_ladder.csv"
    markdown_path = output_dir / "mcf_sure_utility_ladder.md"
    json_path.write_text(json.dumps({"runs": rows}, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    columns = (
        "experiment",
        "wikipedia_documents",
        "generated_subject_contexts",
        "FS",
        "GFS",
        "Spe_success",
        "Spe_margin",
        "PPL",
        "exact_retain_kl_mean",
        "exact_retain_kl_p95",
    )
    lines = [
        "# MCF SURE utility/locality ladder",
        "",
        "GFS and Spe are post-checkpoint audits; neither was used for checkpoint selection.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" if index == 0 else "---:" for index in range(len(columns))) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row[key]) for key in columns) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"))
    print("JSON:", json_path)
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
