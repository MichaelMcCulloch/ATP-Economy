import torch
from ..domain.state import WorldState
from ..utils.tensor_utils import Device, DTYPE


def agent_budgets_and_demand(
    state: WorldState, bases: torch.Tensor, scales: torch.Tensor
):
    """
    Agent budgets and demand with numerically-stable choice and physically
    grounded investment caps. Budget is now purely energy-denominated.
    """
    cfg = state.cfg
    R, G = cfg.R, state.price.shape[0]
    eps = 1e-9

    # Per-agent temperature shaped by greed
    tau_i = cfg.tau * torch.exp(-cfg.greed_tau_scale * state.greed)  # [N]

    # Utilities and stable softmax
    theta = state.pref_theta  # [N,G]
    p_agent = state.price.T.index_select(0, state.agent_region)  # [N,G]
    util = theta - p_agent
    logits = torch.clamp(util / (tau_i.unsqueeze(1) + eps), -40.0, 40.0)

    outside = torch.full((logits.shape[0], 1), -5.0, device=Device, dtype=DTYPE)
    probs = torch.softmax(torch.cat([logits, outside], dim=1), dim=1)[:, :G]  # [N,G]

    # Wallet-based consumption budget proxy (purely energy-denominated)
    base_budget = 0.5 * state.eATP  # [N]

    # Age-structure consumption scaling (regional)
    cons_scale_r = getattr(state, "consump_scale_R", None)
    if cons_scale_r is None:
        cons_scale_r = state.population / (state.population0 + eps)
    cons_scale_i = cons_scale_r[state.agent_region]

    # Savings/investment propensities
    greed_expanded = state.greed[:, None]
    prop = bases + greed_expanded * scales
    prop.clamp_(0.0, 0.9)
    total_frac = prop.sum(dim=1, keepdim=True)
    prop *= torch.clamp(0.95 / (total_frac + eps), max=1.0)
    save_frac, innov_frac, storage_frac = prop.unbind(dim=1)

    # Consumption allocation
    cons_budget = (
        (1.0 - save_frac - innov_frac - storage_frac) * base_budget * cons_scale_i
    )
    spend = cons_budget.unsqueeze(1) * probs  # [N,G]

    # Regional aggregation of demand
    demand_R = state.demand_R_buffer.zero_()
    demand_R.index_add_(0, state.agent_region, spend)

    # Regional aggregation of investment budgets
    innov_budget_i = innov_frac * base_budget * cons_scale_i
    innov_R_raw = state.innov_R_buffer.zero_()
    innov_R_raw.index_add_(0, state.agent_region, innov_budget_i)

    storage_budget_i = storage_frac * base_budget * cons_scale_i
    storage_R_raw = state.storage_R_buffer.zero_()
    storage_R_raw.index_add_(0, state.agent_region, storage_budget_i)

    minted = torch.clamp(state.atp_minted_R, min=0.0)
    cap_innov_R = cfg.cap_innov_exergy_mult * (minted + 1.0)
    cap_storage_R = cfg.cap_storage_exergy_mult * (minted + 1.0)
    innov_R = torch.minimum(innov_R_raw, cap_innov_R)
    storage_budget_R = torch.minimum(storage_R_raw, cap_storage_R)

    # Innovation allocation weights from sigma_eff
    sigma_eff = torch.clamp(state.sigma_eff, min=cfg.sigma_floor)
    w = torch.softmax(sigma_eff / max(cfg.softmax_temp_sigma, 1e-6), dim=1)  # [R,J]
    innov_budget_RJ = innov_R.unsqueeze(1) * w

    return demand_R, innov_budget_RJ, storage_budget_R
