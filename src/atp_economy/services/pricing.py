# src/atp_economy/services/pricing.py
import torch
from ..domain.state import WorldState
from ..utils.tensor_utils import DTYPE, Device


@torch.compile
@torch.no_grad()
def update_prices(
    state: WorldState,
    demand_qty_R: torch.Tensor,
    supply_qty_R: torch.Tensor,
    lr: float = 0.01,
    g_clip: float = 5.0,
    logp_bounds: tuple[float, float] = (-20.0, 20.0),
    alpha_anchor: float = 0.005,
):
    """
    Numerically stable price update in log-space with a trust region and a weak anchor.

    log p <- log p + lr * clip( log(D + eps) - log(S + eps), +/- g_clip )
                     + alpha_anchor * (log p_anchor - log p)

    - Works per good and region (shape [G, R]).
    - Prevents multiplicative overflow/underflow over long horizons.
    - The slow EMA anchor avoids pure random-walk drift without fixing a nominal level.

    Args:
      state: WorldState with state.price [G, R]
      demand_qty_R: [R, G] regional demand (quantities)
      supply_qty_R: [R, G] regional supply (quantities)
      lr: learning rate for the log-price step
      g_clip: trust-region bound for the log-demand minus log-supply signal
      logp_bounds: hard box constraints for log-prices
      alpha_anchor: weak pull toward a slow EMA of log-prices
    """
    eps = 1e-12
    # Current log-prices [G, R]
    logp = torch.log(torch.clamp(state.price, min=eps))

    # Gradient signal in log-space (transpose R,G -> G,R)
    g = (
        torch.log(torch.clamp(demand_qty_R, min=eps)).T
        - torch.log(torch.clamp(supply_qty_R, min=eps)).T
    )
    g = torch.clamp(g, -g_clip, g_clip)

    # Initialize and update slow EMA anchor
    if not hasattr(state, "logp_anchor"):
        state.register_buffer("logp_anchor", logp.clone())
    # Very slow anchor evolution to avoid drift
    state.logp_anchor.mul_(0.999).add_(0.001 * logp)

    # Trust-region step + weak mean reversion to the anchor
    logp_new = logp + lr * g + alpha_anchor * (state.logp_anchor - logp)
    logp_new = torch.clamp(logp_new, logp_bounds[0], logp_bounds[1])

    # Write back prices
    state.price.data.copy_(torch.exp(logp_new))


@torch.compile
@torch.no_grad()
def update_exergy_and_sink_prices(state: WorldState):
    """
    Dual-price updates for exergy (μ) and sink (λ) with bounded exponents.

    μ update:
      ratio = (ex_demand + eps) / (ex_supply + eps)  in [1e-6, 1e6]
      μ <- μ * ratio^{eta_ex}

    λ update:
      util_flow = emissions / (a * sink_cap + eps)
      λ <- λ * exp( clamp(eta_sink * (EMA(util_flow) - util_target), -40, 40) )

    Bounding the exponent arguments prevents overflow/underflow while preserving direction.
    """
    cfg = state.cfg
    eps = 1e-12

    # Init EMAs on first call
    if not hasattr(state, "ema_ex_ratio"):
        state.register_buffer(
            "ema_ex_ratio",
            torch.ones(state.mu_ex.shape, device=Device, dtype=DTYPE),
        )
    if not hasattr(state, "ema_sink_util"):
        state.register_buffer(
            "ema_sink_util",
            torch.zeros(state.lambda_sink.shape, device=Device, dtype=DTYPE),
        )

    # Exergy controller
    ex_demand = state.exergy_need_R  # [R]
    ex_supply = state.atp_minted_R  # [R]
    # Safe ratio range to avoid extreme powers
    ratio = torch.clamp((ex_demand + eps) / (ex_supply + eps), 1e-6, 1e6)
    state.ema_ex_ratio.mul_(cfg.ema_ex).add_((1.0 - cfg.ema_ex) * ratio)
    mu_new = state.mu_ex * torch.pow(state.ema_ex_ratio, cfg.eta_ex)
    state.mu_ex.data = torch.clamp(mu_new, min=cfg.mu_floor, max=cfg.mu_cap)

    # Sink controller (bounded exponent)
    a = torch.tensor(cfg.sink_assim_rate, device=Device, dtype=DTYPE)
    util_flow = (state.emit_sink_R) / (a * state.sink_cap + eps)  # [R]
    state.ema_sink_util.mul_(cfg.ema_sink).add_((1.0 - cfg.ema_sink) * util_flow)
    arg = cfg.eta_sink * (state.ema_sink_util - cfg.util_target)
    arg = torch.clamp(arg, -40.0, 40.0)  # trust region for exp
    lam_new = state.lambda_sink * torch.exp(arg)
    state.lambda_sink.data = torch.clamp(
        lam_new, min=cfg.lambda_floor, max=cfg.lambda_cap
    )
