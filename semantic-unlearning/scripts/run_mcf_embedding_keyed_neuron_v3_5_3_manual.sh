#!/usr/bin/env bash
set -euo pipefail

echo "V3.5.3 is a preserved 29/50 training-only rejection and cannot run from the V3.5.5 checkout." >&2
echo "Reproduce it from commit b603c1ac74677e4a43cb506ae779f75f5a41ef11." >&2
echo "Use run_mcf_embedding_keyed_neuron_v3_5_5_manual.sh with the rejected V3.5.3 and preserved V3.5.4 outputs." >&2
exit 2
