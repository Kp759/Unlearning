#!/usr/bin/env bash
set -euo pipefail

export PROJECT_DIR="${PROJECT_DIR:-$(pwd -P)}"
mkdir -p slurm_logs

JOB_ID="$(sbatch --parsable slurm/run_mcf_embedding_keyed_neuron_v3_5_5_actuator_width_sweep_seed1_3b.slurm)"
echo "embedding_keyed_neuron_v3_5_5_actuator_width_sweep_job=${JOB_ID}"
echo "log=slurm_logs/mcf_embedding_keyed_neuron_v3_5_5_${JOB_ID}.out"
