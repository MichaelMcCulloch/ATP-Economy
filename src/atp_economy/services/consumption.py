import torch
from ..domain.state import WorldState
from .settlement import settle_spend_book
from ..utils.tensor_utils import Device, DTYPE


def run_consumption(
    state: WorldState,
    demand_qty_R: torch.Tensor,
    atp_book_R: torch.Tensor,
    frac: float = 1.0,
) -> torch.Tensor:
    """
    Final-goods consumption gated by ATP, sink headroom, and per-step sink-flow budget.
    """
    cfg = state.cfg
    R = cfg.R
    eps = 1e-9

    F = getattr(state, "final_idx", None)
    if F is None or F.numel() == 0:
        return atp_book_R

    want_RF = torch.clamp_min(
        demand_qty_R[:, F] * max(0.0, min(1.0, frac)), 0.0
    )  # [R,F]
    have_RF = torch.clamp_min(state.inventory[:, F], 0.0)  # [R,F]
    cons_base = torch.minimum(want_RF, have_RF)  # [R,F]

    xi = state.xi_cons  # [F]
    sig = state.sigma_cons  # [F]
    atp_need_base = (cons_base * xi.unsqueeze(0)).sum(dim=1)  # [R]
    sink_emit_base = (cons_base * sig.unsqueeze(0)).sum(dim=1)  # [R]

    # Emission gating primitives
    sink_head = torch.clamp_min(state.sink_cap - state.sink_use - state.sink_use_R, 0.0)

    s_atp = torch.clamp(atp_book_R / (atp_need_base + eps), max=1.0)
    s_head = torch.clamp(sink_head / (sink_emit_base + eps), max=1.0)
    s = torch.minimum(s_atp, s_head)

    cons_RF = cons_base * s.unsqueeze(1)
    new_RF = have_RF - cons_RF
    state.inventory.data = state.inventory.data.index_copy(1, F, new_RF)

    atp_spend = (cons_RF * xi.unsqueeze(0)).sum(dim=1)  # [R]
    sink_emit = (cons_RF * sig.unsqueeze(0)).sum(dim=1)  # [R]

    _, atp_book_R = settle_spend_book(state, atp_spend, atp_book_R)

    state.emit_sink_R.data = state.emit_sink_R.data + sink_emit
    state.exergy_need_R.data = state.exergy_need_R.data + atp_spend
    state.sink_use_R.data = state.sink_use_R.data + sink_emit

    return atp_book_R
