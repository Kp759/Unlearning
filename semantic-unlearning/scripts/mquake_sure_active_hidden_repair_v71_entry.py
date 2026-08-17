#!/usr/bin/env python3
"""V7.1-compatible entrypoint for residual MQuAKE active repair.

The residual repair implementation is shared with V7.  Its Stage-1 reference
validator historically compared against the V7 method-name constant.  V7.1
keeps the same protected-Base reference and locked protocol, so this entrypoint
installs the locked rewrite-only compatibility shim and updates the expected
Stage-1 method name before calling the shared Stage-2 implementation.
"""

import mquake_v7_locked_case_compat as compat

compat.install()

import mquake_sure_active_hidden_repair_v7 as stage2

# validate_stage1_reference() compares config_used.json["method"] against this
# module constant.  V7.1 changes only the Stage-1 objective, not the protected
# Base or data firewall, so accepting its method identity here is intentional.
stage2.stage1.METHOD = "SURE-MQuAKE-v7.1-utility-safe-bounded-margin-GD"


if __name__ == "__main__":
    stage2.main()
