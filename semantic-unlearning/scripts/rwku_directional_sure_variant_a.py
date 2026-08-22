#!/usr/bin/env python3
"""RWKU Directional SURE Variant A: embeddings frozen, directional LM head only."""
from pathlib import Path

import rwku_directional_sure_embedding_ab as ab
import rwku_directional_sure_two_stage as learner

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
SCHEMA = "rwku_directional_sure_variant_a_configuration_v1"
EXPERIMENT_ID = "rwku-directional-sure-variant-a-stephen-king-seed0"
CONFIGURATION = PROJECT_ROOT / "config" / "rwku" / "directional_sure_variant_a_seed0.json"
LEARNER_DIR = "directional_sure_variant_a"


def main() -> None:
    ab.install_variant(
        variant=ab.VARIANT_A,
        schema=SCHEMA,
        experiment_id=EXPERIMENT_ID,
        configuration_path=CONFIGURATION,
        learner_dir=LEARNER_DIR,
    )
    learner.main()


if __name__ == "__main__":
    main()
