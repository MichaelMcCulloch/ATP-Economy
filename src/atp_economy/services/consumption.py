# src/atp_economy/services/consumption.py
import torch
from ..domain.state import WorldState
from .settlement import settle_spend_book
from ..utils.tensor_utils import Device, DTYPE


@torch.compile
@torch.no_grad()
def run_consumption(
    state: WorldState,
    demand_qty_R: torch.Tensor,
    atp_book_R: torch.Tensor,
    frac: float = 1.0,
):
    """
    Consume a fraction of regional demand for final goods, constrained by inventory.
    Books use-phase exergy and sink footprints and settles with ATP.
    """
    cfg = state.cfg
    R = cfg.R
    eps = 1e-9

    if not hasattr(state, "final_idx"):
        return

    F = state.final_idx  # [F]
    if F.numel() == 0:
        return

    want_RF = torch.clamp(
        demand_qty_R[:, F] * max(0.0, min(1.0, frac)), min=0.0
    )  # [R,F]
    have_RF = torch.clamp(state.inventory[:, F], min=0.0)  # [R,F]
    cons_RF = torch.minimum(want_RF, have_RF)

    # Remove consumed goods from inventory
    new_RF = have_RF - cons_RF
    state.inventory.data.index_copy_(1, F, new_RF)

    # Simple per-final use footprints (small but non-zero)
    if not hasattr(state, "xi_cons"):
        state.register_buffer(
            "xi_cons",
            torch.full((F.numel(),), 0.05, device=Device, dtype=DTYPE),
        )
    if not hasattr(state, "sigma_cons"):
        state.register_buffer(
            "sigma_cons",
            torch.full((F.numel(),), 0.02, device=Device, dtype=DTYPE),
        )

    atp_need = (cons_RF * state.xi_cons[None, :]).sum(dim=1)  # [R]
    sink_emit = (cons_RF * state.sigma_cons[None, :]).sum(dim=1)  # [R]

    # Settle use-phase exergy
    settle_spend_book(state, atp_need, atp_book_R)

    # Accumulate environmental pressure for λ controller
    state.emit_sink_R.add_(sink_emit)
    state.exergy_need_R.add_(atp_need)
    state.sink_use_R.add_(sink_emit)
