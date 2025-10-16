import torch
import torch.nn.functional as F
from ..domain.state import WorldState
from .settlement import settle_spend_book

# A large constant to represent infinite availability for the Leontief limiter.
_INF_AVAIL = 1e30


def run_production(
    state: WorldState,
    atp_book_R: torch.Tensor,
    aec_r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = state.cfg
    R, J = cfg.R, state.S.shape[1]
    eps = 1e-9

    # Emission gating primitives
    sink_head = torch.clamp_min(state.sink_cap - state.sink_use - state.sink_use_R, 0.0)

    # Affinity
    A = (
        (state.price.T @ state.S)
        - state.mu_ex[:, None] * state.xi_eff
        - state.lambda_sink[:, None] * state.sigma_eff
    )

    # Leontief limiter: min_g inv_rg / need_gj
    inputs_need = (-state.S).clamp(min=0)  # [G,J]
    inv_per_need = torch.where(
        inputs_need.unsqueeze(0) > 0,
        state.inventory.unsqueeze(2) / (inputs_need.unsqueeze(0) + eps),
        _INF_AVAIL,
    )  # [R,G,J]
    avail = inv_per_need.min(dim=1).values  # [R,J]

    center = 0.5 * (cfg.aec_low + cfg.aec_high)
    aec_gate = (
        torch.sigmoid(cfg.gate_k * (aec_r - center)) * (1.0 - cfg.gate_min)
        + cfg.gate_min
    )
    labor_gate = getattr(state, "labor_factor_R", None)
    if labor_gate is None:
        labor_gate = torch.ones_like(aec_gate)

    beta = max(cfg.beta_aff, 1e-6)
    drive = F.softplus(beta * A) / beta
    r_potential = state.k_eff * drive * torch.tanh(avail / (1.0 + avail))
    r_potential = (
        torch.minimum(r_potential, state.cap_j[None, :])
        * aec_gate[:, None]
        * labor_gate[:, None]
    )

    atp_need = (torch.relu(r_potential) * state.xi_eff).sum(dim=1)  # [R]
    sink_need = (torch.relu(r_potential) * state.sigma_eff).sum(dim=1)  # [R]
    s_atp = torch.clamp(atp_book_R / (atp_need + eps), max=1.0)
    s_head = torch.clamp(sink_head / (sink_need + eps), max=1.0)
    rate = r_potential * torch.minimum(s_atp, s_head)[:, None]

    delta_RG = rate @ state.S.T
    state.inventory.data = torch.clamp_min(state.inventory.data + delta_RG, 0.0)

    atp_spend = (torch.relu(rate) * state.xi_eff).sum(dim=1)
    sink_emit = (torch.relu(rate) * state.sigma_eff).sum(dim=1)
    _, atp_book_R = settle_spend_book(state, atp_spend, atp_book_R)

    state.emit_sink_R.data = state.emit_sink_R.data + sink_emit
    state.exergy_need_R.data = state.exergy_need_R.data + atp_spend
    state.sink_use_R.data = state.sink_use_R.data + sink_emit

    return rate, atp_book_R
