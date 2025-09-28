# src/atp_economy/services/settlement.py
import torch
from ..domain.state import WorldState
from ..utils.integrators import patankar_imex_transfer
from ..utils.tensor_utils import Device, DTYPE


@torch.no_grad()
def settle_spend_book(
    state: WorldState, spend_R: torch.Tensor, atp_book_R: torch.Tensor
) -> torch.Tensor:
    """
    Settle a regional ATP spend against a live ATP 'book' for this step.
    - Caps spend by both the book and actual regional agent ATP pools.
    - Deducts proportionally from agents' ATP -> ADP.
    - Decrements the book and regional pools in-place.

    Returns:
      shortfall_R: [R] unmet spend (if any)
    """
    eps = 1e-9
    R = state.cfg.R

    # Current agent ATP pools by region (fast reduction)
    pool_r = torch.bincount(state.agent_region, weights=state.eATP, minlength=R)

    # Cap spend by book AND pool
    cap_r = torch.minimum(atp_book_R, pool_r)
    actual = torch.minimum(spend_R, cap_r)
    shortfall = torch.clamp(spend_R - actual, min=0.0)

    # Proportional deduction across agents
    region_idx = state.agent_region
    denom = pool_r[region_idx] + eps
    factor_i = actual[region_idx] / denom
    delta_i = state.eATP * factor_i

    state.eATP.data.sub_(delta_i)
    state.eADP.data.add_(delta_i)

    # Decrement the live book and regional pools
    atp_book_R.sub_(actual)
    state.pool_atp_R.sub_(actual)
    state.pool_adp_R.add_(actual)

    return shortfall


@torch.no_grad()
def apply_demurrage(state: WorldState, factors) -> None:
    """
    Apply demurrage with per-region multipliers (or scalar), conserving adenylate:
    ATP -> ADP at rate demurrage * factor, integrated with IMEX-Patankar.
    Updates both agent balances and regional pools.
    """
    cfg = state.cfg
    if isinstance(factors, (float, int)):
        f = float(factors)
        k = max(0.0, cfg.demurrage * f)
        if k <= 0.0:
            return
        # Agents
        eATP_new, eADP_new = patankar_imex_transfer(
            state.eATP, state.eADP, rate=k, dt=cfg.dt
        )
        state.eATP.data.copy_(eATP_new)
        state.eADP.data.copy_(eADP_new)
        # Pools (exact at region level since k is uniform per region)
        k_s = torch.tensor(k, device=Device, dtype=DTYPE)
        denom = 1.0 + k_s * cfg.dt
        pool_atp_new = state.pool_atp_R / denom
        pool_adp_new = state.pool_adp_R + (k_s * cfg.dt) * pool_atp_new
        state.pool_atp_R.data.copy_(pool_atp_new)
        state.pool_adp_R.data.copy_(pool_adp_new)
        return

    # Per-region factors
    factors_t = torch.as_tensor(factors, device=Device, dtype=DTYPE)  # [R]
    k_r = torch.clamp(cfg.demurrage * factors_t, min=0.0)  # [R]
    if torch.all(k_r <= 0):
        return

    # Agents
    k_agent = k_r[state.agent_region]  # [N]
    eATP_new, eADP_new = patankar_imex_transfer(
        state.eATP, state.eADP, rate=k_agent, dt=cfg.dt
    )
    state.eATP.data.copy_(eATP_new)
    state.eADP.data.copy_(eADP_new)

    # Pools (exact at region level)
    denom = 1.0 + k_r * cfg.dt  # [R]
    pool_atp_new = state.pool_atp_R / denom
    pool_adp_new = state.pool_adp_R + (k_r * cfg.dt) * pool_atp_new
    state.pool_atp_R.data.copy_(pool_atp_new)
    state.pool_adp_R.data.copy_(pool_adp_new)
