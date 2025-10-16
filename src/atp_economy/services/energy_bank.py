import torch
from ..domain.state import WorldState
from ..utils.tensor_utils import DTYPE, Device


def run_recharging(
    state: WorldState, need_prev_R: torch.Tensor, adp_pool_R: torch.Tensor
):
    """
    ADP -> ATP recharging with storage discharge/charge.

    Policies:
    - Mint only to satisfy last-step exergy need (need_prev_R), never to fill sink headroom.
    - Gate emissions by remaining sink headroom.
    - Compile-safe: avoid clamp(min=tensor, max=float) signatures.
    """
    cfg = state.cfg
    R = cfg.R
    eps = 1e-9

    # Stochastic primary generation
    nz = cfg.gen_noise
    factor = torch.clamp(
        1.0 + (2 * torch.rand(R, device=Device, dtype=DTYPE) - 1.0) * nz, min=0.1
    )
    gen = state.gen_exergy * factor  # [R]

    # Cover backlog with storage; no discharge if gen >= need
    deficit = torch.relu(need_prev_R - gen)  # [R]
    discharge = torch.minimum(deficit / (state.eta_rt + eps), state.storage_soc)
    delivered_raw = gen + discharge * state.eta_rt  # [R]

    # Never deliver beyond last-step need
    delivered_need_limited = torch.minimum(
        delivered_raw, torch.clamp_min(need_prev_R, 0.0)
    )

    # Provisional generation emissions
    sink_gen_raw = delivered_need_limited * state.gen_sink_intensity  # [R]

    # Headroom gating within this step
    sink_head = torch.clamp_min(state.sink_cap - state.sink_use - state.sink_use_R, 0.0)
    s_head = torch.clamp(sink_head / (sink_gen_raw + eps), max=1.0)

    s_emit = s_head

    delivered = delivered_need_limited * s_emit
    sink_gen = sink_gen_raw * s_emit

    # Mint limited by ADP pool
    minted_R = torch.minimum(delivered, adp_pool_R)  # [R]
    state.atp_minted_R.data = minted_R

    # Surplus delivered (if any) -> charge storage within capacity (account for η)
    surplus = torch.relu(delivered - minted_R)
    free_cap = torch.clamp_min(state.storage_cap - state.storage_soc, 0.0)
    charge = torch.minimum(surplus / (state.eta_rt + eps), free_cap)

    # Update SoC with discharge/charge
    soc_new = torch.clamp_min(state.storage_soc + charge - discharge, 0.0)
    soc_new = torch.minimum(soc_new, state.storage_cap)
    state.storage_soc.data = soc_new

    # Book generation emissions for this step
    state.emit_sink_R.data = state.emit_sink_R.data + sink_gen
    state.sink_use_R.data = state.sink_use_R.data + sink_gen

    # Distribute minted ATP ∝ ADP within region
    share = torch.where(adp_pool_R > eps, minted_R / (adp_pool_R + eps), 0.0)
    delta_agent = state.eADP * share[state.agent_region]
    state.eATP.data = state.eATP.data + delta_agent
    state.eADP.data = state.eADP.data - delta_agent

    # Update pools exactly
    state.pool_atp_R.data = state.pool_atp_R.data + minted_R
    state.pool_adp_R.data = state.pool_adp_R.data - minted_R
