# src/atp_economy/utils/integrators.py
import torch


@torch.no_grad()
def patankar_imex_transfer(
    donor: torch.Tensor,
    receiver: torch.Tensor,
    rate: torch.Tensor | float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    IMEX-Patankar update for a one-way transfer: donor -> receiver at rate.
    Ensures positivity and exact conservation for linear transfer.

    ODE:
      d donor / dt = -k * donor
      d receiver / dt = +k * donor

    Patankar-Euler (implicit in destruction):
      donor_{n+1} = donor_n / (1 + dt * k)
      receiver_{n+1} = receiver_n + dt * k * donor_{n+1}

    Args:
      donor: tensor of donor amounts (e.g., ATP per agent)
      receiver: tensor of receiver amounts (e.g., ADP per agent)
      rate: scalar or tensor broadcastable to donor (per-entity rate k >= 0)
      dt: timestep size

    Returns:
      (donor_new, receiver_new)
    """
    if isinstance(rate, torch.Tensor):
        k = rate
    else:
        # Infer dtype and device from the donor tensor
        k = torch.tensor(rate, device=donor.device, dtype=donor.dtype)

    k = torch.clamp(k, min=0.0)
    kdt = k * float(dt)
    denom = 1.0 + kdt
    donor_new = donor / denom
    receiver_new = receiver + kdt * donor_new
    return donor_new, receiver_new
