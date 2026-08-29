#!/usr/bin/env bash
set -euo pipefail

echo "run_mcf_embedding_keyed_neuron_v3_manual.sh is a compatibility alias for V3.1" >&2
exec bash "$(dirname "$0")/run_mcf_embedding_keyed_neuron_v3_1_manual.sh" "$@"
