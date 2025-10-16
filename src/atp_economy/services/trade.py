import torch
from ..domain.state import WorldState
from .settlement import settle_spend_book
from ..utils.tensor_utils import Device, DTYPE


def run_trade(
    state: WorldState,
    supply_R: torch.Tensor,
    demand_R: torch.Tensor,
    atp_book_R: torch.Tensor,
    kappa: float = 0.8,
) -> torch.Tensor:
    """
    Neighbor trade gated by ATP, sink headroom, and per-step sink-flow budget.
    """
    cfg = state.cfg
    eps = 1e-9
    R, G = cfg.R, cfg.G

    surplus = torch.relu(supply_R - demand_R)  # [R,G]
    deficit = torch.relu(demand_R - supply_R)  # [R,G]

    nbr = state.nbr_idx  # [R,k]
    cost = state.nbr_cost  # [R,k]
    cap = torch.clamp_min(state.nbr_cap, 1e-6)  # [R,k]
    k = nbr.shape[1]

    cost_penalty = cost.unsqueeze(-1)  # [R,k,1]
    neigh_def = deficit.index_select(0, nbr.reshape(-1)).reshape(R, k, G)  # [R,k,G]
    scores = torch.relu(neigh_def - cost_penalty)  # [R,k,G]
    score_sum = scores.sum(dim=1, keepdim=True) + eps
    alloc = scores / score_sum  # [R,k,G]

    ship = alloc * (kappa * surplus.unsqueeze(1))  # [R,k,G]

    ship_sumG = ship.sum(dim=2)  # [R,k]
    route_scale = torch.minimum(torch.ones_like(cap), cap / (ship_sumG + eps))
    ship = ship * route_scale.unsqueeze(-1)

    dist_rg = state.distance.gather(1, nbr)  # [R,k]
    qty_out = ship.sum(dim=2)  # [R,k]
    atp_log_need = cfg.alpha_logistics_ex * (qty_out * dist_rg).sum(dim=1)  # [R]
    sink_log_emit = cfg.alpha_logistics_sink * (qty_out * dist_rg).sum(dim=1)  # [R]

    # Emission gating primitives
    sink_head = torch.clamp_min(state.sink_cap - state.sink_use - state.sink_use_R, 0.0)

    s_atp = torch.clamp(atp_book_R / (atp_log_need + eps), max=1.0)
    s_head = torch.clamp(sink_head / (sink_log_emit + eps), max=1.0)
    s = torch.minimum(s_atp, s_head)
    ship = ship * s.unsqueeze(1).unsqueeze(2)

    # Recompute bills after scaling and settle
    qty_out = ship.sum(dim=2)
    atp_log_need = cfg.alpha_logistics_ex * (qty_out * dist_rg).sum(dim=1)
    sink_log_emit = cfg.alpha_logistics_sink * (qty_out * dist_rg).sum(dim=1)

    _, atp_book_R = settle_spend_book(state, atp_log_need, atp_book_R)
    state.emit_sink_R.data = state.emit_sink_R.data + sink_log_emit

    outflow = ship.sum(dim=1)  # [R,G]
    inflow = state.inflow_R_buffer.zero_()
    inflow.index_add_(0, nbr.reshape(-1), ship.reshape(R * k, G))
    state.inventory.data = torch.clamp_min(state.inventory.data - outflow + inflow, 0.0)

    state.exergy_need_R.data = state.exergy_need_R.data + atp_log_need
    state.sink_use_R.data = state.sink_use_R.data + sink_log_emit

    return atp_book_R
