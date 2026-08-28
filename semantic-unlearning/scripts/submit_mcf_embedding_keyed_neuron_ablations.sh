#!/usr/bin/env bash
set -euo pipefail

export PROJECT_DIR="${PROJECT_DIR:-$(pwd -P)}"
mkdir -p slurm_logs

JOB_ID="$(sbatch --parsable slurm/run_mcf_embedding_keyed_neuron_ablation_seed1_3b.slurm)"
echo "embedding_keyed_neuron_ablation_array=${JOB_ID}"
echo "logs=slurm_logs/mcf_embedding_keyed_neuron_ablation_${JOB_ID}_<index>.out"
