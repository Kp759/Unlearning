from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


def _as_2d(x: Tensor, *, name: str) -> Tensor:
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape [n, d], got {tuple(x.shape)}")
    if x.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if not torch.is_floating_point(x):
        raise TypeError(f"{name} must use a floating dtype")
    return x


def wendland_c2_kernel(x: Tensor, y: Tensor, *, radius: float) -> Tensor:
    """Dimension-aware compact-support Wendland C2 kernel.

    For descriptor dimension ``d_dim``, set ``ell=floor(d_dim/2)+2`` and use

        k(r) = (1-r)^(ell+1)_+ * ((ell+1) r + 1),
        r = ||x-y||_2 / radius.

    This is the standard C2 Wendland family with the exponent adjusted for the
    ambient descriptor dimension.  In two dimensions it reduces to the familiar
    ``(1-r)^4_+ (4r+1)`` form. Distances greater than or equal to ``radius``
    receive exactly zero kernel value.
    """

    x = _as_2d(x, name="x")
    y = _as_2d(y, name="y")
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"descriptor dimensions must match, got {x.shape[1]} and {y.shape[1]}"
        )
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")

    d = torch.cdist(x, y, p=2) / float(radius)
    one_minus = torch.clamp(1.0 - d, min=0.0)
    ell = int(x.shape[1] // 2 + 2)
    power = ell + 1
    return one_minus.pow(power) * (float(power) * d + 1.0)


@dataclass(frozen=True)
class AnchoredFeatureMap:
    """Finite retain-anchored contextual cardinal feature map.

    This object stores *descriptor-space* anchors only. It does not own or
    modify language-model weights. ``retain`` and ``forget`` are matrices of
    frozen contextual descriptors.

    The retained kernel is

        k_R(x, x') = k(x, x')
                     - k(x, R) (K_RR + jitter I)^-1 k(R, x').

    The fact-indexed feature map is

        alpha(x) = G^-1 k_R(F, x),
        G = K_R(F, F) + cardinal_jitter I.

    Solves are used instead of explicit matrix inverses.
    """

    retain: Tensor
    forget: Tensor
    radius: float
    retain_jitter: float = 1e-6
    cardinal_jitter: float = 1e-6
    _retain_cholesky: Optional[Tensor] = None
    _cardinal_cholesky: Optional[Tensor] = None

    @classmethod
    def fit(
        cls,
        *,
        retain: Tensor,
        forget: Tensor,
        radius: float,
        retain_jitter: float = 1e-6,
        cardinal_jitter: float = 1e-6,
    ) -> "AnchoredFeatureMap":
        retain = _as_2d(retain, name="retain")
        forget = _as_2d(forget, name="forget")
        if retain.shape[1] != forget.shape[1]:
            raise ValueError(
                "retain and forget descriptor dimensions must match: "
                f"{retain.shape[1]} != {forget.shape[1]}"
            )
        if radius <= 0:
            raise ValueError("radius must be > 0")
        if retain_jitter < 0 or cardinal_jitter < 0:
            raise ValueError("jitter values must be non-negative")
        if retain.device != forget.device:
            raise ValueError("retain and forget descriptors must be on the same device")
        if retain.dtype != forget.dtype:
            raise ValueError("retain and forget descriptors must have the same dtype")

        k_rr = wendland_c2_kernel(retain, retain, radius=radius)
        eye_r = torch.eye(k_rr.shape[0], device=k_rr.device, dtype=k_rr.dtype)
        a_rr = k_rr + float(retain_jitter) * eye_r
        retain_cholesky = torch.linalg.cholesky(a_rr)

        k_ff = wendland_c2_kernel(forget, forget, radius=radius)
        k_fr = wendland_c2_kernel(forget, retain, radius=radius)
        solved_rf = torch.cholesky_solve(k_fr.transpose(0, 1), retain_cholesky)
        g = k_ff - k_fr @ solved_rf

        eye_f = torch.eye(g.shape[0], device=g.device, dtype=g.dtype)
        a_g = g + float(cardinal_jitter) * eye_f
        cardinal_cholesky = torch.linalg.cholesky(a_g)

        return cls(
            retain=retain.detach().clone(),
            forget=forget.detach().clone(),
            radius=float(radius),
            retain_jitter=float(retain_jitter),
            cardinal_jitter=float(cardinal_jitter),
            _retain_cholesky=retain_cholesky.detach().clone(),
            _cardinal_cholesky=cardinal_cholesky.detach().clone(),
        )

    @property
    def num_facts(self) -> int:
        return int(self.forget.shape[0])

    @property
    def descriptor_dim(self) -> int:
        return int(self.forget.shape[1])

    def _validate_query(self, x: Tensor) -> Tensor:
        x = _as_2d(x, name="x")
        if x.shape[1] != self.descriptor_dim:
            raise ValueError(
                f"query descriptor dim {x.shape[1]} != {self.descriptor_dim}"
            )
        if x.device != self.forget.device:
            raise ValueError("query descriptors must be on the fitted device")
        if x.dtype != self.forget.dtype:
            raise ValueError("query descriptors must use the fitted dtype")
        return x

    def retain_conditioned_kernel(self, left: Tensor, right: Tensor) -> Tensor:
        left = self._validate_query(left)
        right = self._validate_query(right)
        if self._retain_cholesky is None:
            raise RuntimeError("feature map was not fitted")

        k_lr = wendland_c2_kernel(left, right, radius=self.radius)
        k_lR = wendland_c2_kernel(left, self.retain, radius=self.radius)
        k_Rr = wendland_c2_kernel(self.retain, right, radius=self.radius)
        solved = torch.cholesky_solve(k_Rr, self._retain_cholesky)
        return k_lr - k_lR @ solved

    def alpha(self, x: Tensor) -> Tensor:
        """Return fact-indexed contextual features with shape [batch, num_facts]."""

        x = self._validate_query(x)
        if self._cardinal_cholesky is None:
            raise RuntimeError("feature map was not fitted")

        k_fx = self.retain_conditioned_kernel(self.forget, x)
        alpha_f_batch = torch.cholesky_solve(k_fx, self._cardinal_cholesky)
        return alpha_f_batch.transpose(0, 1).contiguous()

    def retain_residual(self) -> Tensor:
        return self.alpha(self.retain)

    def cardinal_residual(self) -> Tensor:
        alpha_f = self.alpha(self.forget)
        eye = torch.eye(self.num_facts, device=alpha_f.device, dtype=alpha_f.dtype)
        return alpha_f - eye

    def outside_support_mask(self, x: Tensor) -> Tensor:
        x = self._validate_query(x)
        k_xf = wendland_c2_kernel(x, self.forget, radius=self.radius)
        k_xr = wendland_c2_kernel(x, self.retain, radius=self.radius)
        return (k_xf.abs().sum(dim=1) == 0) & (k_xr.abs().sum(dim=1) == 0)
