# src/atp_economy/services/environment.py
import torch
from ..domain.state import WorldState
from ..utils.tensor_utils import DTYPE, Device


def update_environment(state: WorldState, emit_R: torch.Tensor):
    """
    Regenerating sink dynamics. We treat 'sink_use' as a pollutant stock P[r]
    that accumulates current emissions 'emit_R' and decays by natural assimilation.

    Explicit Euler with first-order decay:
        P_{t+1} = P_t + dt * emit_R - dt * a * P_t
    Then clip to [0, sink_cap] without mixing scalar and tensor bounds.

    Args:
      state: WorldState
      emit_R: [R] emissions generated this step (from production, extraction, trade)
    """
    cfg = state.cfg

    P = state.pollutant
    a = torch.tensor(cfg.sink_assim_rate, device=Device, dtype=DTYPE)

    # Integrate
    P_next = P + cfg.dt * emit_R - cfg.dt * a * P

    # Two-step clipping: first lower bound (scalar), then upper bound (tensor)
    P_next = torch.clamp_min(P_next, 0.0)
    P_next = torch.minimum(P_next, state.sink_cap)

    # Persist and mirror to sink_use for pricing/metrics
    state.pollutant.data = P_next
    state.sink_use.data = P_next
