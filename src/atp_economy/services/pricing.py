# src/atp_economy/services/pricing.py
import torch
from ..domain.state import WorldState
from ..utils.tensor_utils import DTYPE, Device

_BIG = torch.tensor(1e30, device=Device, dtype=DTYPE)


def price_floor_from_duals(state, margin=1.02):
    """
    Unit-cost floor per good and region from current duals and input prices.
    For each reaction j producing good g:
        floor_{g,r} = ( Σ_i p_{i,r} * max(0, -S_{i,j}) + μ_r * ξ_{r,j} + λ_r * σ_{r,j} ) / S_{g,j}  (S_{g,j} > 0)
    Then take min_j over producers of g and apply a small margin (>1) so A > 0 is feasible.
    Also apply a consumer-use floor for final goods: μ*xi_cons + λ*sigma_cons.
    Returns: [G, R]
    """
    S = state.S  # [G,J]
    S_pos = torch.clamp(S, min=0.0)  # outputs
    S_neg = torch.clamp(-S, min=0.0)  # inputs

    p_RG = state.price.T  # [R,G]
    input_cost_RJ = p_RG @ S_neg  # [R,J]
    dual_cost_RJ = (
        state.mu_ex[:, None] * state.xi_eff
        + state.lambda_sink[:, None] * state.sigma_eff
    )  # [R,J]
    cost_RJ = input_cost_RJ + dual_cost_RJ  # [R,J]

    denom_JG = S_pos.T  # [J,G]
    denom_JG = torch.where(denom_JG > 0.0, denom_JG, _BIG)  # avoid div-by-zero

    cand_RJG = cost_RJ[:, :, None] / denom_JG[None, :, :]  # [R,J,G]
    floor_RG = cand_RJG.min(dim=1).values  # [R,G]
    floor_RG = torch.clamp(floor_RG, min=0.0)

    # Final-goods consumer-use floor
    F = getattr(state, "final_idx", None)
    if F is not None and F.numel() > 0:
        cons_floor_RF = (
            state.mu_ex[:, None] * state.xi_cons[None, :]
            + state.lambda_sink[:, None] * state.sigma_cons[None, :]
        )  # [R, |F|]
        floor_RG.index_copy_(1, F, torch.maximum(floor_RG[:, F], cons_floor_RF))

    floor_RG = margin * floor_RG  # small markup
    return floor_RG.T  # [G,R]


# services/pricing.py (inside update_prices)
def update_prices(
    state: WorldState,
    demand_qty_R: torch.Tensor,
    supply_qty_R: torch.Tensor,
    lr: float = 0.01,
    g_clip: float = 5.0,
    logp_bounds: tuple[float, float] = (-20.0, 20.0),
    alpha_anchor: float = 0.005,
    alpha_floor: float = 0.30,  # NEW: how hard we enforce the floor (in log-space)
    margin: float = 1.02,  # NEW: unit-cost markup to keep A > 0 attainable
):
    eps = 1e-12
    logp = torch.log(torch.clamp(state.price, min=eps))

    g = (
        torch.log(torch.clamp(demand_qty_R, min=eps)).T
        - torch.log(torch.clamp(supply_qty_R, min=eps)).T
    )
    g = torch.clamp(g, -g_clip, g_clip)

    # Slow EMA anchor
    state.logp_anchor.data = state.logp_anchor.data * 0.999 + 0.001 * logp
    logp_new = logp + lr * g + alpha_anchor * (state.logp_anchor - logp)

    # NEW: unit-cost price floor
    p_floor = price_floor_from_duals(state, margin=margin)  # [G,R]
    logp_floor = torch.log(torch.clamp(p_floor, min=eps))
    logp_floor_mix = (1.0 - alpha_floor) * logp + alpha_floor * logp_floor
    logp_new = torch.maximum(logp_new, logp_floor_mix)

    logp_new = torch.clamp(logp_new, logp_bounds[0], logp_bounds[1])
    state.price.data = torch.exp(logp_new)


def update_exergy_and_sink_prices(state: WorldState):
    """
    Dual-price updates for exergy (μ) and sink (λ) with bounded exponents.

    μ update:
      ratio = (ex_demand + eps) / (ex_supply + eps)  in [1e-6, 1e6]
      μ <- μ * ratio^{eta_ex}

    λ update:
      MODIFIED: The controller now responds to the stock utilization level, not the flow.
      util_stock = sink_use / (sink_cap + eps)
      λ <- λ * exp( clamp(eta_sink * (EMA(util_stock) - util_target), -40, 40) )
    """
    cfg = state.cfg
    eps = 1e-12

    # Exergy controller
    ex_demand = state.exergy_need_R  # [R]
    ex_supply = state.atp_minted_R  # [R]
    # Safe ratio range to avoid extreme powers
    ratio = torch.clamp((ex_demand + eps) / (ex_supply + eps), 1e-6, 1e6)
    state.ema_ex_ratio.data = (
        state.ema_ex_ratio.data * cfg.ema_ex + (1.0 - cfg.ema_ex) * ratio
    )
    mu_new = state.mu_ex * torch.pow(state.ema_ex_ratio, cfg.eta_ex)
    state.mu_ex.data = torch.clamp(mu_new, min=cfg.mu_floor, max=cfg.mu_cap)

    # Sink controller (MODIFIED LOGIC)
    # The input signal is now the stock utilization, not the flow.
    util_stock = state.sink_use / (state.sink_cap + eps)
    state.ema_sink_util.data = (
        state.ema_sink_util.data * cfg.ema_sink + (1.0 - cfg.ema_sink) * util_stock
    )
    arg = cfg.eta_sink * (state.ema_sink_util - cfg.util_target)
    arg = torch.clamp(arg, -40.0, 40.0)  # trust region for exp
    lam_new = state.lambda_sink * torch.exp(arg)
    state.lambda_sink.data = torch.clamp(
        lam_new, min=cfg.lambda_floor, max=cfg.lambda_cap
    )
