# src/atp_economy/utils/checks.py
import torch
from torch import nn


@torch.no_grad()
def assert_finite(tag: str, **tensors):
    """
    Development-time guard: raise with precise location of the first non-finite value.
    Call after major phases. Disable by setting env ATP_STRICT_FINITE=0.
    """
    for name, t in tensors.items():
        if t is None:
            continue
        if not torch.is_tensor(t):
            continue
        mask = ~torch.isfinite(t)
        if mask.any():
            idx = mask.nonzero(as_tuple=False)[0].tolist()
            val = t[tuple(idx)].item()
            raise RuntimeError(
                f"[{tag}] non-finite in '{name}' at index {tuple(idx)}: {val}"
            )
