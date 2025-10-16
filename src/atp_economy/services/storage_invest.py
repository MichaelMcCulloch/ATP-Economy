# src/atp_economy/services/storage_invest.py
import torch
from ..domain.state import WorldState


def apply_storage_investment(state: WorldState, storage_budget_R: torch.Tensor):
    """
    Update storage capacity with investment and depreciation:
      cap_{t+1} = (cap_t + dt * eta * invest) / (1 + dt * decay)
    Also clamp state-of-charge to the capacity.
    """
    cfg = state.cfg
    cap_num = state.storage_cap + cfg.dt * cfg.eta_storage * torch.clamp(
        storage_budget_R, min=0.0
    )
    cap_den = 1.0 + cfg.dt * cfg.storage_decay
    state.storage_cap.data = torch.clamp(cap_num / cap_den, min=0.0)

    state.storage_soc.data = torch.minimum(state.storage_soc, state.storage_cap)
