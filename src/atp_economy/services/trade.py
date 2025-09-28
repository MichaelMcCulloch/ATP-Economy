# src/atp_economy/services/trade.py
import torch
from ..domain.state import WorldState
from .settlement import settle_spend_book
from ..utils.tensor_utils import Device, DTYPE


@torch.compile
@torch.no_grad()
def run_trade(
    state: WorldState,
    supply_R: torch.Tensor,
    demand_R: torch.Tensor,
    atp_book_R: torch.Tensor,
    kappa: float = 0.8,
):
    """
    Local, edge-based trade gated by live ATP book and sink headroom.
    Accumulates this-step logistics emissions in state.emit_sink_R.
    """
    cfg = state.cfg
    eps = 1e-9
    R, G = cfg.R, cfg.G

    # Surplus/deficit per region-good
    surplus = torch.relu(supply_R - demand_R)  # [R,G]
    deficit = torch.relu(demand_R - supply_R)  # [R,G]

    # Neighbor indices/cost/capacity
    nbr = state.nbr_idx  # [R,k]
    cost = state.nbr_cost  # [R,k]
    cap = torch.clamp(state.nbr_cap, min=1e-6)  # [R,k]
    k = nbr.shape[1]

    # Weight neighbor deficits with cost penalty
    cost_penalty = cost.unsqueeze(-1)  # [R,k,1]
    neigh_def = deficit.index_select(0, nbr.reshape(-1)).reshape(R, k, G)  # [R,k,G]
    scores = torch.relu(neigh_def - cost_penalty)  # [R,k,G]

    # Normalize scores over neighbors
    score_sum = scores.sum(dim=1, keepdim=True) + eps  # [R,1,G]
    alloc = scores / score_sum  # [R,k,G]

    # Proposed shipments bounded by kappa * surplus
    ship = alloc * (kappa * surplus.unsqueeze(1))  # [R,k,G]

    # Route capacity (shared across goods)
    ship_sumG = ship.sum(dim=2)  # [R,k]
    route_scale = torch.minimum(
        torch.ones(cap.shape, device=Device, dtype=DTYPE), cap / (ship_sumG + eps)
    )
    ship = ship * route_scale.unsqueeze(-1)

    # Logistics needs per origin (distance-weighted)
    dist_rg = state.distance.gather(1, nbr)  # [R,k]
    qty_out = ship.sum(dim=2)  # [R,k]
    atp_log_need = cfg.alpha_logistics_ex * (qty_out * dist_rg).sum(dim=1)  # [R]
    sink_log_emit = cfg.alpha_logistics_sink * (qty_out * dist_rg).sum(dim=1)  # [R]

    # Gate by remaining ATP/sink headroom
    sink_head = torch.clamp(state.sink_cap - state.sink_use, min=0.0)
    s_atp = torch.clamp(atp_book_R / (atp_log_need + eps), max=1.0)
    s_sink = torch.clamp(sink_head / (sink_log_emit + eps), max=1.0)
    s = torch.minimum(s_atp, s_sink)  # [R]
    ship *= s[:, None, None]

    # Recompute bills after scaling and settle
    qty_out = ship.sum(dim=2)
    atp_log_need = cfg.alpha_logistics_ex * (qty_out * dist_rg).sum(dim=1)
    sink_log_emit = cfg.alpha_logistics_sink * (qty_out * dist_rg).sum(dim=1)

    settle_spend_book(state, atp_log_need, atp_book_R)
    state.emit_sink_R.add_(sink_log_emit)

    # Apply inventory changes
    outflow = ship.sum(dim=1)  # [R,G]
    inflow = torch.zeros(R, G, device=Device, dtype=DTYPE)
    inflow.index_add_(0, nbr.reshape(-1), ship.reshape(R * k, G))
    state.inventory.data = torch.clamp(state.inventory.data - outflow + inflow, min=0.0)

    # Book needs for metrics/pricing
    state.exergy_need_R.add_(atp_log_need)
    state.sink_use_R.add_(sink_log_emit)
