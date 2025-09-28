# src/atp_economy/services/innovation.py
import torch
from ..domain.state import WorldState


@torch.no_grad()
def update_innovation_and_effects(state: WorldState, innov_budget_RJ: torch.Tensor):
    """
    IMEX/Patankar-like update of technology stocks T[r,j] and mapping to effective process params.

    Stability/realism additions:
      - Irreducible floors for xi_eff and sigma_eff (no process is literally zero-cost/zero-externality).
      - Cap on effective innovation increment to reflect finite absorptive capacity of R&D systems.
    """
    cfg = state.cfg
    R, J = cfg.R, state.S.shape[1]
    eps = 1e-9

    # Effective innovation effort (diminishing returns)
    I = torch.clamp(innov_budget_RJ, min=0.0)  # [R,J]
    I_eff = torch.pow(I + eps, cfg.innov_alpha)
    # Cap the increment to avoid runaway T updates
    I_eff = torch.clamp(I_eff, max=cfg.innov_I_cap)

    # Spillovers via neighbor averaging
    nbr = state.nbr_idx  # [R,k]
    k = nbr.shape[1]
    spill = cfg.innov_spill * (
        state.tech_T.index_select(0, nbr.reshape(-1)).reshape(R, k, J).mean(dim=1)
        - state.tech_T
    )

    T_num = state.tech_T + state.cfg.dt * (cfg.eta_innov * I_eff + spill)
    T_den = 1.0 + state.cfg.dt * cfg.innov_decay
    state.tech_T.data = torch.clamp(T_num / T_den, min=0.0)

    # Map to effective parameters with irreducible floors
    xi_eff = state.xi_base[None, :] * torch.exp(-cfg.beta_xi * state.tech_T)
    sigma_eff = state.sigma_base[None, :] * torch.exp(-cfg.beta_sigma * state.tech_T)

    state.xi_eff.data = torch.clamp(xi_eff, min=cfg.xi_floor)
    state.sigma_eff.data = torch.clamp(sigma_eff, min=cfg.sigma_floor)

    # Throughput catalyst (bounded by tanh)
    state.k_eff.data = state.k_base[None, :] * (
        1.0 + cfg.beta_kcat * torch.tanh(state.tech_T)
    )
