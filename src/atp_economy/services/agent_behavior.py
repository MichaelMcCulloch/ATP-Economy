# src/atp_economy/services/agent_behavior.py
import torch
from ..domain.state import WorldState
from ..utils.tensor_utils import Device, DTYPE


@torch.compile
@torch.no_grad()
def agent_budgets_and_demand(state: WorldState):
    """
    Agent budgets and demand with numerically-stable choice and physically
    grounded investment caps.
    """
    cfg = state.cfg
    R, G, J = cfg.R, state.price.shape[0], state.S.shape[1]
    eps = 1e-9

    # Per-agent temperature shaped by greed
    tau_i = cfg.tau * torch.exp(-cfg.greed_tau_scale * state.greed)  # [N]

    # Utilities and stable softmax
    theta = state.pref_theta  # [N,G]
    p_agent = state.price.T.index_select(0, state.agent_region)  # [N,G]
    util = theta - p_agent
    logits = util * (1.0 / (tau_i.unsqueeze(1) + eps))
    logits = torch.clamp(logits, -40.0, 40.0)

    # Outside option: ensures softmax is always well-defined
    outside = torch.full((logits.shape[0], 1), -5.0, device=Device, dtype=DTYPE)
    logits_ext = torch.cat([logits, outside], dim=1)  # [N, G+1]
    probs_ext = torch.softmax(logits_ext, dim=1)  # [N, G+1]
    probs = probs_ext[:, :G]  # [N, G]

    # Wallet-based budget proxy
    base_budget = 0.5 * state.eATP + 0.1 * state.fiat + 0.1 * state.crypto  # [N]

    # --- Fused Propensity Calculation ---
    # 1. Calculate raw propensities and stack
    greed_expanded = state.greed[:, None]  # [N, 1]
    bases = torch.tensor(
        [cfg.save_base, cfg.invest_innov_base, cfg.invest_storage_base],
        device=Device,
        dtype=DTYPE,
    )  # [3]
    scales = torch.tensor(
        [
            cfg.save_greed_scale,
            cfg.invest_innov_greed_scale,
            cfg.invest_storage_greed_scale,
        ],
        device=Device,
        dtype=DTYPE,
    )  # [3]

    propensities = bases + greed_expanded * scales  # [N, 3]
    propensities.clamp_(0.0, 0.9)  # In-place clamp

    # 2. Normalize and unpack
    total_frac = propensities.sum(dim=1, keepdim=True)  # [N, 1]
    norm_scale = torch.clamp(0.95 / (total_frac + eps), max=1.0)
    propensities *= norm_scale

    save_frac, innov_frac, storage_frac = propensities.unbind(dim=1)
    # --- End Fused Calculation ---

    # Population scaling
    pop_scale_r = state.population / (state.population0 + eps)  # [R]
    pop_scale_i = pop_scale_r[state.agent_region]

    cons_budget = (
        (1.0 - save_frac - innov_frac - storage_frac) * base_budget * pop_scale_i
    )
    innov_budget = innov_frac * base_budget * pop_scale_i
    storage_budget = storage_frac * base_budget * pop_scale_i

    # Consumption spend allocation across goods -> regional demand value
    spend = cons_budget.unsqueeze(1) * probs  # [N,G]
    demand_R = torch.zeros(R, G, device=Device, dtype=DTYPE)
    demand_R.scatter_reduce_(
        0,
        state.agent_region[:, None].expand(-1, G),  # [N,G] indices
        spend,  # [N,G] values
        reduce="sum",
        include_self=False,
    )

    # Regional aggregation of investment budgets (raw) via bincount
    idx = state.agent_region
    innov_R_raw = torch.bincount(idx, weights=innov_budget, minlength=R)
    storage_R_raw = torch.bincount(idx, weights=storage_budget, minlength=R)

    # Absorptive capacity caps tied to recharged exergy this step
    minted = torch.clamp(state.atp_minted_R, min=0.0)  # [R]
    cap_innov_R = cfg.cap_innov_exergy_mult * (minted + 1.0)
    cap_storage_R = cfg.cap_storage_exergy_mult * (minted + 1.0)
    innov_R = torch.minimum(innov_R_raw, cap_innov_R)
    storage_budget_R = torch.minimum(storage_R_raw, cap_storage_R)

    # Innovation allocation weights: strictly positive via softmax over sigma_eff
    sigma_eff = torch.clamp(state.sigma_eff, min=cfg.sigma_floor)  # [R,J]
    w = torch.softmax(
        sigma_eff / max(cfg.softmax_temp_sigma, 1e-6), dim=1
    )  # [R,J], sums to 1

    innov_budget_RJ = innov_R.unsqueeze(1) * w  # [R,J]

    return demand_R, innov_budget_RJ, storage_budget_R
