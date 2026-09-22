# RWKU determinism and combined-person residual patch

This directory stores the exact uploaded patch `determinism-and-combined-residual.patch`, split into two ordered **binary gzip fragments** for transfer through the repository. The reconstructed patch creates:

- `semantic-unlearning/scripts/check_rwku_generation_determinism.py`
- `semantic-unlearning/scripts/evaluate_rwku_combined_person_residual.py`

## Reconstruct and apply on Wulver

Run from the Git repository root, **not** from inside `semantic-unlearning`:

```bash
cd /scratch/yl258/kp759/Unlearning
git switch feat/router-ordered-fix-plan
git pull --ff-only origin feat/router-ordered-fix-plan

cat semantic-unlearning/patches/rwku-determinism-combined.patch.gz.part00 \
    semantic-unlearning/patches/rwku-determinism-combined.patch.gz.part01 \
    | gzip -dc > /tmp/determinism-and-combined-residual.patch

echo "70ef89b3494dd073c6c57934ad9c810ccbf8825d22dc5d434e8b82a0b4351637  /tmp/determinism-and-combined-residual.patch" \
    | sha256sum --check

git apply --check /tmp/determinism-and-combined-residual.patch
git apply /tmp/determinism-and-combined-residual.patch

python -m py_compile \
    semantic-unlearning/scripts/check_rwku_generation_determinism.py \
    semantic-unlearning/scripts/evaluate_rwku_combined_person_residual.py
```

Expected script SHA256:

```text
720c99c02aa2e8e69bc77a5d4d2df99f50ae379a4924418cd96c308f404aad7c  semantic-unlearning/scripts/check_rwku_generation_determinism.py
069b89f32554008e1d2422eb6f5ebec7f3eaf76f731d0d706e846d5074f60cd2  semantic-unlearning/scripts/evaluate_rwku_combined_person_residual.py
```

The patch fragments are stored in GitHub already; the two Python files become available in the Wulver checkout after applying the patch. Commit and push those generated Python files from Wulver to version them as ordinary source files.
