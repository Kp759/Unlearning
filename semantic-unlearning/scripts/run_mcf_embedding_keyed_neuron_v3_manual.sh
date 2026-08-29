#!/usr/bin/env bash
set -euo pipefail

echo "run_mcf_embedding_keyed_neuron_v3_manual.sh is a compatibility alias for V3.2" >&2
exec bash "$(dirname "$0")/run_mcf_embedding_keyed_neuron_v3_2_manual.sh" "$@"
