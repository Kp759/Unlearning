#!/usr/bin/env bash
set -euo pipefail

echo "v2 is a preserved failed protocol and cannot be rerun by this wrapper." >&2
echo "Use run_mcf_embedding_keyed_neuron_v3_5_manual.sh with the preserved V3.2 and V3.4 outputs." >&2
exit 2
