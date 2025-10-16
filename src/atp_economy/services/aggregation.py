# src/atp_economy/services/aggregation.py
import torch
from ..domain.state import WorldState


def compute_regional_summaries(state: WorldState) -> dict[str, torch.Tensor]:
    """
    Computes agent->region aggregations using fast reductions.
    """
    R = state.cfg.R
    idx = state.agent_region

    atp_pool = torch.bincount(idx, weights=state.eATP, minlength=R)
    adp_pool = torch.bincount(idx, weights=state.eADP, minlength=R)
    amp_pool = torch.bincount(idx, weights=state.eAMP, minlength=R)

    return {
        "atp_pool": atp_pool,
        "adp_pool": adp_pool,
        "amp_pool": amp_pool,
    }
