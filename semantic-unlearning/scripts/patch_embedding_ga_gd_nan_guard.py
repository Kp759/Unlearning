#!/usr/bin/env python3
import argparse
from pathlib import Path

HELPER = r'''
# ===================== NaN / Inf guard helpers =====================
def _finite_or_raise_loss(loss, name: str, step: int):
    import torch
    if not torch.isfinite(loss).all().item():
        raise RuntimeError(f"[NaNGuard] Non-finite loss at step={step}: {name}={loss}")

def _finite_or_raise_grad(param, name: str, step: int):
    import torch
    if param is None or param.grad is None:
        return
    if not torch.isfinite(param.grad).all().item():
        bad = (~torch.isfinite(param.grad)).sum().item()
        raise RuntimeError(f"[NaNGuard] Non-finite grad at step={step}: {name}, bad_count={bad}")

def _finite_or_raise_rows(weight, row_ids, name: str, step: int):
    import torch
    rows = weight.data[row_ids]
    if not torch.isfinite(rows).all().item():
        bad = (~torch.isfinite(rows)).sum().item()
        raise RuntimeError(f"[NaNGuard] Non-finite rows at step={step}: {name}, bad_count={bad}")

def _finite_or_raise_full(weight, name: str, step: int):
    import torch
    if not torch.isfinite(weight).all().item():
        bad = (~torch.isfinite(weight)).sum().item()
        raise RuntimeError(f"[NaNGuard] Non-finite tensor at step={step}: {name}, bad_count={bad}")
# =================== end NaN / Inf guard helpers ===================
'''

def patch_text(src: str) -> str:
    if "_finite_or_raise_loss" not in src:
        src = src.replace("\ndef main", "\n" + HELPER + "\ndef main", 1)

    src = src.replace(
        "f_loss = lm_loss(model, batch)",
        "f_loss = lm_loss(model, batch)\n            _finite_or_raise_loss(f_loss, 'forget_loss', step)"
    )

    src = src.replace(
        "r_loss = lm_loss(model, batch)",
        "r_loss = lm_loss(model, batch)\n            _finite_or_raise_loss(r_loss, 'retain_loss', step)"
    )

    src = src.replace(
        "loss.backward()\n\n            mask_grad_rows(emb.weight, forget_mask)",
        "loss.backward()\n\n            mask_grad_rows(emb.weight, forget_mask)\n            _finite_or_raise_grad(emb.weight, 'embed_tokens.weight.forget_grad', step)"
    )

    src = src.replace(
        "loss.backward()\n\n            mask_grad_rows(emb.weight, retain_mask)",
        "loss.backward()\n\n            mask_grad_rows(emb.weight, retain_mask)\n            _finite_or_raise_grad(emb.weight, 'embed_tokens.weight.retain_grad', step)"
    )

    src = src.replace(
        "opt_forget.step()",
        "opt_forget.step()\n            _finite_or_raise_rows(emb.weight, anchor_ids_tensor, 'embed_tokens.after_forget_step', step)"
    )

    src = src.replace(
        "opt_retain.step()",
        "opt_retain.step()\n            _finite_or_raise_rows(emb.weight, anchor_ids_tensor, 'embed_tokens.after_retain_step', step)"
    )

    src = src.replace(
        'print(f"Saving model to {out_dir}")',
        "_finite_or_raise_full(emb.weight.data, 'FINAL_embed_tokens.weight', args.steps)\n    print(f\"Saving model to {out_dir}\")"
    )

    return src

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default="scripts/embedding_ga_gd_unlearn.py")
    parser.add_argument("--out", default="scripts/embedding_ga_gd_unlearn_nan_guard.py")
    args = parser.parse_args()

    script = Path(args.script)
    out = Path(args.out)

    if not script.exists():
        raise FileNotFoundError(script)

    src = script.read_text(encoding="utf-8")
    patched = patch_text(src)
    out.write_text(patched, encoding="utf-8")

    print(f"[done] wrote patched script: {out}")
    print("Use this new script. It will stop immediately if NaN/Inf appears.")

if __name__ == "__main__":
    main()
