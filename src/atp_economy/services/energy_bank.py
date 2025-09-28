# src/atp_economy/services/energy_bank.py
import torch
from ..domain.state import WorldState
from ..utils.tensor_utils import DTYPE, Device


@torch.compile
@torch.no_grad()
def run_recharging(
    state: WorldState, need_prev_R: torch.Tensor, adp_pool_R: torch.Tensor
):
    """
    ADP -> ATP recharging with storage discharge and charging.
    - Discharge storage to cover last-step deficit.
    - Mint ATP limited by delivered energy and available ADP.
    - Charge any surplus into storage (respecting η_rt and capacity).
    - Clamp storage SoC to [0, storage_cap].
    """
    R = state.cfg.R
    eps = 1e-9

    # Stochastic generation
    nz = state.cfg.gen_noise
    factor = torch.clamp(
        1.0 + (2 * torch.rand(R, device=Device, dtype=DTYPE) - 1.0) * nz, min=0.1
    )
    gen = state.gen_exergy * factor  # [R]

    # Discharge to cover backlog
    deficit = torch.relu(need_prev_R - gen)  # [R]
    max_discharge = state.storage_soc  # add power caps here if desired
    discharge = torch.minimum(deficit / (state.eta_rt + eps), max_discharge)
    delivered = gen + discharge * state.eta_rt  # [R]

    # Provisional minting limited by ADP pool
    minted_pre = torch.minimum(delivered, adp_pool_R)  # [R]

    # Surplus goes to charging (account for round-trip efficiency on the way in)
    surplus = torch.relu(delivered - minted_pre)  # [R]
    free_cap = torch.clamp(state.storage_cap - state.storage_soc, min=0.0)
    charge = torch.minimum(
        surplus / (state.eta_rt + eps), free_cap
    )  # store input energy

    # Update SoC and clamp to capacity
    soc_new = torch.clamp(state.storage_soc + charge - discharge, min=0.0)
    soc_new = torch.minimum(soc_new, state.storage_cap)
    state.storage_soc.data.copy_(soc_new)

    # Final minted ATP
    minted_R = minted_pre
    state.atp_minted_R.copy_(minted_R)

    # Distribute minted ATP proportional to ADP within region
    share = torch.where(adp_pool_R > eps, minted_R / (adp_pool_R + eps), 0.0)
    delta_agent = state.eADP * share[state.agent_region]
    state.eATP.data += delta_agent
    state.eADP.data -= delta_agent

    # Update regional pools (exact)
    state.pool_atp_R.add_(minted_R)
    state.pool_adp_R.sub_(minted_R)
