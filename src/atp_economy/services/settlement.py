import torch
from ..domain.state import WorldState
from ..utils.integrators import patankar_imex_transfer
from ..utils.tensor_utils import Device, DTYPE


def settle_spend_book(
    state: WorldState, spend_R: torch.Tensor, atp_book_R: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-9
    R = state.cfg.R

    pool_r = state.pool_atp_R

    cap_r = torch.minimum(atp_book_R, pool_r)
    actual = torch.minimum(spend_R, cap_r)
    shortfall = torch.clamp(spend_R - actual, min=0.0)

    region_idx = state.agent_region
    denom = pool_r[region_idx] + eps
    factor_i = actual[region_idx] / denom
    delta_i = state.eATP * factor_i

    state.eATP.data = state.eATP.data - delta_i
    state.eADP.data = state.eADP.data + delta_i

    atp_book_R = atp_book_R - actual
    state.pool_atp_R.data = state.pool_atp_R.data - actual
    state.pool_adp_R.data = state.pool_adp_R.data + actual

    return shortfall, atp_book_R


def apply_demurrage(state: WorldState, factors: torch.Tensor) -> None:
    cfg = state.cfg

    # ATP -> ADP demurrage
    k_r = torch.clamp(cfg.demurrage * factors, min=0.0)  # [R]
    k_agent = k_r[state.agent_region]  # [N]
    eATP_new, eADP_new = patankar_imex_transfer(
        state.eATP, state.eADP, rate=k_agent, dt=cfg.dt
    )
    state.eATP.data = eATP_new
    state.eADP.data = eADP_new

    denom = 1.0 + k_r * cfg.dt  # [R]
    pool_atp_new = state.pool_atp_R / denom
    pool_adp_new = state.pool_adp_R + (k_r * cfg.dt) * pool_atp_new
    state.pool_atp_R.data = pool_atp_new
    state.pool_adp_R.data = pool_adp_new

    # AMP -> ADP leak under chronic stress
    aec_r = (state.pool_atp_R + 0.5 * state.pool_adp_R) / (
        state.pool_atp_R + state.pool_adp_R + state.pool_amp_R + 1e-12
    )
    leak_rate = 0.01 * torch.relu(cfg.aec_low - aec_r)  # up to 1%/step at deep stress

    k_amp_agent = leak_rate[state.agent_region]
    eAMP_new, eADP_new2 = patankar_imex_transfer(
        state.eAMP, state.eADP, rate=k_amp_agent, dt=cfg.dt
    )
    state.eAMP.data = eAMP_new
    state.eADP.data = eADP_new2

    denom_amp = 1.0 + leak_rate * cfg.dt
    pool_amp_new = state.pool_amp_R / denom_amp
    state.pool_amp_R.data = pool_amp_new
    state.pool_adp_R.data = state.pool_adp_R + (leak_rate * cfg.dt) * pool_amp_new
