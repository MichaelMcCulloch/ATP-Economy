# src/atp_economy/services/production.py
import torch
import torch.nn.functional as F
from ..domain.state import WorldState
from .settlement import settle_spend_book
from ..utils.checks import assert_finite


@torch.compile
@torch.no_grad()
def run_production(
    state: WorldState,
    atp_book_R: torch.Tensor,
    aec_r: torch.Tensor,
) -> torch.Tensor:
    """
    Production gated by live ATP book and sink headroom.
    Returns:
      rate_RJ: [R,J] realized reaction rates (for GDP flow metrics).
    """
    cfg = state.cfg
    R, J = cfg.R, state.S.shape[1]
    eps = 1e-9

    sink_head = torch.clamp(state.sink_cap - state.sink_use, min=0.0)

    # Affinity: (price•S) - μ·ξ - λ·σ
    A = (
        (state.price.T @ state.S)
        - state.mu_ex[:, None] * state.xi_eff
        - state.lambda_sink[:, None] * state.sigma_eff
    )  # [R,J]

    # Material availability (Leontief-like)
    inputs_need = -state.S.clamp(max=0)  # [G,J]
    avail = (state.inventory @ inputs_need) / (
        inputs_need.sum(0, keepdim=True) + eps
    )  # [R,J]

    # AMPK-like gate on throughput from AEC
    center = 0.5 * (cfg.aec_low + cfg.aec_high)
    gate = (
        torch.sigmoid(cfg.gate_k * (aec_r - center)) * (1.0 - cfg.gate_min)
        + cfg.gate_min
    )  # [R]

    # Sensitivity to affinity via smooth ReLU (softplus)
    beta = max(cfg.beta_aff, 1e-6)
    drive = F.softplus(beta * A) / beta  # >=0, ~ReLU for large A
    r_potential = state.k_eff * drive * torch.tanh(avail / (1.0 + avail))  # [R,J]
    r_potential = torch.minimum(r_potential, state.cap_j[None, :]) * gate[:, None]

    # ATP and sink gating
    atp_need = (torch.relu(r_potential) * state.xi_eff).sum(dim=1)  # [R]
    sink_need = (torch.relu(r_potential) * state.sigma_eff).sum(dim=1)  # [R]
    s_atp = torch.clamp(atp_book_R / (atp_need + eps), max=1.0)
    s_sink = torch.clamp(sink_head / (sink_need + eps), max=1.0)
    rate = r_potential * torch.minimum(s_atp, s_sink)[:, None]  # [R,J]

    # Inventory update
    delta_RG = rate @ state.S.T  # [R,G]
    state.inventory.data = torch.clamp(state.inventory.data + delta_RG, min=0.0)

    assert_finite("inventory_update", inventory=state.inventory)

    # Settlement and accounting
    atp_spend = (torch.relu(rate) * state.xi_eff).sum(dim=1)  # [R]
    sink_emit = (torch.relu(rate) * state.sigma_eff).sum(dim=1)  # [R]
    settle_spend_book(state, atp_spend, atp_book_R)

    state.emit_sink_R.add_(sink_emit)
    state.exergy_need_R.add_(atp_spend)
    state.sink_use_R.add_(sink_emit)

    return rate
