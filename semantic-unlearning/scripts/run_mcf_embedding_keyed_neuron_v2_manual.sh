#!/usr/bin/env bash
set -euo pipefail

echo "v2 is a preserved failed protocol and cannot be rerun by this wrapper." >&2
echo "Build the clean V6.2 Stage-1 writer, preserve the V3.2 rejected output, then use run_mcf_embedding_keyed_neuron_v3_3_manual.sh." >&2
exit 2
