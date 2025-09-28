# src/atp_economy/services/demography.py
import torch
from ..domain.state import WorldState
from ..utils.tensor_utils import Device, DTYPE


@torch.compile
@torch.no_grad()
def update_population_and_inheritance(state: WorldState, aec_r: torch.Tensor):
    """
    Fully vectorized population update and wallet inheritance transfers.
    Also keeps regional adenylate pools in sync to avoid per-step re-aggregation.
    """
    cfg = state.cfg
    R = cfg.R
    eps = 1e-9

    # Rates by region from pre-computed AEC
    b = cfg.birth_base * torch.sigmoid(cfg.birth_k * (aec_r - cfg.aec_birth_center))
    d = cfg.death_base * torch.sigmoid(cfg.death_k * (cfg.aec_death_center - aec_r))

    # Population IMEX
    P = state.population
    P_new = (P + cfg.dt * b * P) / (1.0 + cfg.dt * d)

    # Birth endowments
    births_r = torch.clamp(P_new - P, min=0.0)  # [R]
    region_idx = state.agent_region  # [N]
    counts_r = torch.bincount(region_idx, minlength=R).to(dtype=DTYPE)
    counts_safe = torch.clamp(counts_r, min=1.0)
    per_agent_birth_share_r = births_r / counts_safe  # [R]
    add_atp_i = cfg.birth_endow_atp * per_agent_birth_share_r[region_idx]
    add_fiat_i = cfg.birth_endow_fiat * per_agent_birth_share_r[region_idx]
    state.eATP.data.add_(add_atp_i)
    state.fiat.data.add_(add_fiat_i)

    # Update pools for ATP from births (exact)
    state.pool_atp_R.add_(cfg.birth_endow_atp * births_r)

    # Death/inheritance as uniform hazard per region
    death_frac_r = torch.clamp(cfg.dt * d, min=0.0, max=0.99)  # [R]
    death_frac_i = death_frac_r[region_idx]  # [N]

    # Heir weights: w_i ~ (greed^conc) normalized within region
    w_raw_i = torch.pow(state.greed + eps, cfg.inherit_conc)  # [N]
    w_sum_r = torch.bincount(region_idx, weights=w_raw_i, minlength=R)
    w_norm_i = w_raw_i / (w_sum_r[region_idx] + eps)  # [N]

    # --- Fused Inheritance Calculation ---
    balances = torch.stack(
        [state.eATP, state.eADP, state.eAMP, state.fiat, state.crypto], dim=1
    )  # [N, 5]

    removed_i = balances * death_frac_i[:, None]
    balances.sub_(removed_i)

    removed_pools_r = torch.zeros(R, 5, device=Device, dtype=DTYPE)
    removed_pools_r.scatter_reduce_(
        0,
        region_idx[:, None].expand(-1, 5),
        removed_i,
        reduce="sum",
        include_self=False,
    )

    heir_pools_r = removed_pools_r * cfg.inherit_frac_on_death
    heir_share_i = w_norm_i[:, None] * heir_pools_r[region_idx]
    balances.add_(heir_share_i)

    # --- FIX: Unpack and ensure contiguous memory layout ---
    unbound_balances = balances.unbind(dim=1)
    state.eATP.data.copy_(unbound_balances[0].contiguous())
    state.eADP.data.copy_(unbound_balances[1].contiguous())
    state.eAMP.data.copy_(unbound_balances[2].contiguous())
    state.fiat.data.copy_(unbound_balances[3].contiguous())
    state.crypto.data.copy_(unbound_balances[4].contiguous())

    net_loss_r = removed_pools_r - heir_pools_r
    state.pool_atp_R.sub_(net_loss_r[:, 0])
    state.pool_adp_R.sub_(net_loss_r[:, 1])
    state.pool_amp_R.sub_(net_loss_r[:, 2])
    # --- End Fused Calculation ---

    state.population.data = torch.clamp(P_new, min=0.0)
