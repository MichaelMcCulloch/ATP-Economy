import torch
from ..domain.state import WorldState
from .settlement import settle_spend_book
from ..utils.tensor_utils import Device, DTYPE


def run_extraction(
    state: WorldState, atp_book_R: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = state.cfg
    R = cfg.R
    eps = 1e-9

    M = state.res_goods.numel()
    if M == 0:
        return torch.zeros(R, 0, device=Device, dtype=DTYPE), atp_book_R

    frac = torch.clamp(
        state.reserves / (state.reserves_max + eps), min=1e-9, max=1.0
    )  # [R,M]
    xi_ext = state.xi_ext0[None, :] * (1.0 + cfg.dep_alpha_xi * (1.0 - frac))
    sig_ext = state.sig_ext0[None, :] * (1.0 + cfg.dep_alpha_sig * (1.0 - frac))

    goods_idx = state.res_goods  # [M]
    p_rm = state.price.index_select(0, goods_idx).T  # [R,M]
    A = p_rm - state.mu_ex[:, None] * xi_ext - state.lambda_sink[:, None] * sig_ext

    drive = torch.relu(A)
    q_hat = cfg.k_ext * drive * torch.tanh(frac / (1.0 + frac))  # [R,M]

    # Emission gating primitives
    sink_head = torch.clamp_min(state.sink_cap - state.sink_use - state.sink_use_R, 0.0)

    atp_need = (q_hat * xi_ext).sum(dim=1)  # [R]
    sink_need = (q_hat * sig_ext).sum(dim=1)
    s_atp = torch.clamp(atp_book_R / (atp_need + eps), max=1.0)
    s_head = torch.clamp(sink_head / (sink_need + eps), max=1.0)
    s = torch.minimum(s_atp, s_head)
    q = q_hat * s[:, None]  # [R,M]

    denom = 1.0 + cfg.dt * (q / (state.reserves + eps))
    state.reserves.data = torch.clamp_min(state.reserves.data / denom, 0.0)

    inv_slice = torch.clamp_min(state.inventory[:, goods_idx] + q, 0.0)
    state.inventory.data = state.inventory.data.index_copy(1, goods_idx, inv_slice)

    atp_spend = (q * xi_ext).sum(dim=1)
    sink_emit = (q * sig_ext).sum(dim=1)

    _, atp_book_R = settle_spend_book(state, atp_spend, atp_book_R)

    state.emit_sink_R.data = state.emit_sink_R.data + sink_emit
    state.exergy_need_R.data = state.exergy_need_R.data + atp_spend
    state.sink_use_R.data = state.sink_use_R.data + sink_emit

    return q, atp_book_R
