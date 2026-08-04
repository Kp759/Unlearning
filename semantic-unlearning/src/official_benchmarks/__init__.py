"""Official-benchmark-first orchestration for the existing semantic method.

This package is deliberately standard-library-only.  Inventory, doctor, plan,
and dry-run code must remain safe on CPU login nodes and must not import torch.
"""

from .registry import (  # noqa: F401
    ALLOWED_STATUSES,
    PROJECT_ROOT,
    RegistryError,
    get_track,
    load_registry,
    select_tracks,
)

__all__ = [
    "ALLOWED_STATUSES",
    "PROJECT_ROOT",
    "RegistryError",
    "get_track",
    "load_registry",
    "select_tracks",
]
