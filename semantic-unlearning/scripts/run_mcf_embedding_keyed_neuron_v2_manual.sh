#!/usr/bin/env bash
set -euo pipefail

echo "v2 is a preserved failed protocol and cannot be rerun by this wrapper." >&2
echo "Build the clean V6 Stage-1 writer, then use run_mcf_embedding_keyed_neuron_v3_manual.sh." >&2
exit 2
