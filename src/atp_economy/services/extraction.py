# src/atp_economy/services/extraction.py
import torch
from ..domain.state import WorldState
from .settlement import settle_spend_book
from ..utils.tensor_utils import Device, DTYPE


@torch.compile
@torch.no_grad()
def run_extraction(state: WorldState, atp_book_R: torch.Tensor) -> torch.Tensor:
    """
    Resource extraction gated by live regional ATP book and sink headroom.
    Returns:
      q_RM: [R, M] realized extraction amounts per resource good.
    """
    cfg = state.cfg
    R = cfg.R
    eps = 1e-9

    M = state.res_goods.numel()
    if M == 0:
        return torch.zeros(R, 0, device=Device, dtype=DTYPE)

    # Remaining fraction of reserves
    frac = torch.clamp(
        state.reserves / (state.reserves_max + eps), min=1e-9, max=1.0
    )  # [R,M]
    # Depletion makes intensities worse
    xi_ext = state.xi_ext0[None, :] * (1.0 + cfg.dep_alpha_xi * (1.0 - frac))
    sig_ext = state.sig_ext0[None, :] * (1.0 + cfg.dep_alpha_sig * (1.0 - frac))

    # Affinity: price - μ·ξ - λ·σ
    goods_idx = state.res_goods  # [M]
    p_rm = state.price.index_select(0, goods_idx).T  # [R,M]
    A = p_rm - state.mu_ex[:, None] * xi_ext - state.lambda_sink[:, None] * sig_ext

    drive = torch.relu(A)
    q_hat = cfg.k_ext * drive * torch.tanh(frac / (1.0 + frac))  # [R,M]

    # ATP/sink gating
    sink_head = torch.clamp(state.sink_cap - state.sink_use, min=0.0)
    atp_need = (q_hat * xi_ext).sum(dim=1)  # [R]
    sink_need = (q_hat * sig_ext).sum(dim=1)
    s_atp = torch.clamp(atp_book_R / (atp_need + eps), max=1.0)
    s_sink = torch.clamp(sink_head / (sink_need + eps), max=1.0)
    q = q_hat * torch.minimum(s_atp, s_sink)[:, None]  # [R,M]

    # Update reserves (Patankar)
    denom = 1.0 + cfg.dt * (q / (state.reserves + eps))
    state.reserves.data = torch.clamp(state.reserves.data / denom, min=0.0)

    # Add extracted goods to inventory (direct column update)
    inv_slice = torch.clamp(state.inventory[:, goods_idx] + q, min=0.0)
    state.inventory.data.index_copy_(1, goods_idx, inv_slice)

    # Settlement and emissions
    atp_spend = (q * xi_ext).sum(dim=1)
    sink_emit = (q * sig_ext).sum(dim=1)

    settle_spend_book(state, atp_spend, atp_book_R)

    state.emit_sink_R.add_(sink_emit)
    state.exergy_need_R.add_(atp_spend)
    state.sink_use_R.add_(sink_emit)

    return q
