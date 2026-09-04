#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    method = out / "method"
    report_path = method / "private_vocab_rewiring_v1_3_multiview.json"
    completion_path = method / "completion.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))

    report["protocol"] = "mcf_private_vocab_rewiring_v1_3c_fullfit_5view"
    report["variant"] = "v1_3c_fullfit_5view"
    report["fullfit_optimization"] = {
        "target": "50_of_50_cases_pass_all_5_training_views",
        "margin_threshold": 0.1,
        "curriculum": [
            {"steps": "1-150", "active_views": 2},
            {"steps": "151-300", "active_views": 3},
            {"steps": "301-1500", "active_views": 5}
        ],
        "hard_case_oversampling_fraction": 0.75,
        "checkpoint_and_model_selection_use_all_5_views": True,
        "same_locked_5view_corpus_as_v1_3": True,
        "heldout_probe_text_used": False,
    }
    completion["protocol"] = "mcf_private_vocab_rewiring_v1_3c_fullfit_5view"
    completion["variant"] = "v1_3c_fullfit_5view"
    worst = completion.get("worst_training_view_margin", {})
    completion["all_50_all_5_pass"] = int(worst.get("failures", -1)) == 0
    completion["target_margin"] = 0.1
    completion["heldout_probe_text_used"] = False

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    completion_path.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "protocol": completion["protocol"],
        "all_50_all_5_pass": completion["all_50_all_5_pass"],
        "worst_training_view_margin": worst,
        "retain_mean_kl": completion.get("retain_mean_kl"),
    }, indent=2))


if __name__ == "__main__":
    main()
