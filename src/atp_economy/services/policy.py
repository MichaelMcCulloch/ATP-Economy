# src/atp_economy/services/policy.py
import torch
from ..domain.state import WorldState
from ..config import EconConfig


def aec_by_region(
    atp_r: torch.Tensor, adp_r: torch.Tensor, amp_r: torch.Tensor
) -> torch.Tensor:
    """Computes AEC from pre-aggregated regional adenylate pools."""
    denom = atp_r + adp_r + amp_r + 1e-12
    return (atp_r + 0.5 * adp_r) / denom


def ers_demurrage_factors(cfg: EconConfig, aec_r: torch.Tensor) -> torch.Tensor:
    """Per-region demurrage multiplier from local AEC deviation."""
    center = 0.5 * (cfg.aec_low + cfg.aec_high)
    adj = torch.tanh(cfg.ers_k * (aec_r - center))  # [R] in [-1,1]
    return 1.0 + 0.5 * adj  # [R] in [0.5,1.5]
