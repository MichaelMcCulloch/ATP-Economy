# src/atp_economy/services/demography.py
"""
Age-structured demography with economic coupling.

One step = one day. We integrate demography every demography_step_days (default 30).
Key features:
- Cohort ageing in 1-year bins (0..100) via a conservative ageing operator.
- Mortality: infant/child regimes + adult Gompertz-Makeham, scaled by a Health index H.
- Fertility: UN-style ASFR window (15-49), scaled by a slow Development index D,
  a replacement/insurance term from under-5 survival, and a cyclical term from GDPpc growth.
- Newborns experience neonatal hazard in the same integration window.
- Optional migration valve (off by default) with simple attraction to higher GDPpc/AEC regions.
- Labor and consumption couplings:
    labor_factor_R ∈ [~0.2, 1.2] gates production throughput by region.
    consumption_scale_R rescales household budgets by region.
- Wallet inheritance and birth endowments applied using regional death fraction and births.

The implementation is fully vectorized across regions and ages.
"""

from __future__ import annotations
import torch
from ..config import EconConfig
from ..domain.state import WorldState
from ..utils.tensor_utils import Device, DTYPE


class _CompiledDemographyStep(torch.nn.Module):
    def __init__(self, cfg: EconConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, state: WorldState, aec_r: torch.Tensor, gdp_pc_r: torch.Tensor):
        """
        The core computational logic of the demographic update, designed to be compiled.
        This version runs every step with a fixed daily time delta.
        """
        cfg = self.cfg
        R, A = cfg.R, state.age_years.numel()
        eps = 1e-9
        dt_years = 1.0 / 365.0

        # ---------- Health and Development indices ----------
        aec_low = cfg.aec_low
        aec_high = cfg.aec_high
        aec_span = max(1e-6, aec_high - aec_low)
        aec_norm = torch.clamp((aec_r - aec_low) / aec_span, 0.0, 1.0)

        ema_fast = 0.90
        ema_slow = 0.99
        gdp_pc_ema_prev_R = state.gdp_pc_ema_R
        gdp_pc_ema_R = gdp_pc_ema_prev_R * ema_fast + (1.0 - ema_fast) * gdp_pc_r

        g_ratio = torch.log(
            torch.clamp(gdp_pc_ema_R / (state.gdp_pc_baseline_R + eps), min=1e-6)
        )
        g_term = torch.clamp(0.5 + 0.5 * torch.tanh(0.5 * g_ratio), 0.0, 1.0)

        util = torch.clamp(state.sink_use / (state.sink_cap + eps), 0.0, 1.0)
        relief = 1.0 - util
        H = torch.clamp(0.5 * aec_norm + 0.4 * g_term + 0.1 * relief, 0.0, 1.0)

        dev_proxy = torch.clamp(0.5 + 0.5 * torch.tanh(0.3 * g_ratio), 0.0, 1.0)
        dev_index_R = state.dev_index_R * ema_slow + (1.0 - ema_slow) * dev_proxy

        state.gdp_pc_ema_prev_R.data = gdp_pc_ema_prev_R
        state.gdp_pc_ema_R.data = gdp_pc_ema_R
        state.dev_index_R.data = dev_index_R
        state.health_index_R.data = H

        # ---------- Mortality hazards (per-year) ----------
        eta_neon = cfg.eta_neonatal
        eta_child = cfg.eta_child
        eta_adult = cfg.eta_adult
        sink_m = cfg.mort_sink_mult

        haz_R_A = state.hazard_A_base.unsqueeze(0).repeat(R, 1)
        m_neon = torch.exp(-eta_neon * H).unsqueeze(1)
        m_child = torch.exp(-eta_child * H).unsqueeze(1)
        m_adult = torch.exp(-eta_adult * H).unsqueeze(1)
        haz_R_A[:, 0] *= m_neon.squeeze(1)
        haz_R_A[:, 1:15] *= m_child
        haz_R_A[:, 15:] *= m_adult
        haz_R_A *= 1.0 + sink_m * util.unsqueeze(1)
        haz_R_A = torch.clamp(haz_R_A, 0.0, 5.0)
        S_R_A = torch.exp(-haz_R_A * dt_years)

        # ---------- Apply deaths then ageing ----------
        pop0 = state.pop_age
        survivors = pop0 * S_R_A
        pop_after_age = survivors @ state.aging_M

        deaths_R = torch.clamp(pop0.sum(dim=1) - survivors.sum(dim=1), min=0.0)
        death_frac_R = torch.clamp(deaths_R / (state.population + eps), 0.0, 0.99)

        # ---------- Births (ASFR with multipliers) ----------
        female_share = cfg.female_share
        asfr = state.asfr_vector
        female_RF = female_share * pop_after_age[:, 15:50]

        theta_D = cfg.fert_theta_dev
        phi_rep = cfg.fert_phi_rep
        theta_cyc = cfg.fert_theta_cyc
        child_survival_ref = 0.995

        haz_u5 = haz_R_A[:, 0:5]
        surv_u5 = torch.exp(-haz_u5.sum(dim=1))
        F_dev = torch.exp(-theta_D * dev_index_R).clamp(0.5, 1.5)
        F_rep = torch.pow(
            child_survival_ref / torch.clamp(surv_u5, min=1e-3), phi_rep
        ).clamp(0.5, 1.8)

        g_growth = torch.log(torch.clamp(gdp_pc_ema_R + eps, min=1e-6)) - torch.log(
            torch.clamp(gdp_pc_ema_prev_R + eps, min=1e-6)
        )
        Shock = torch.clamp(-g_growth, min=0.0)
        F_cyc = torch.exp(-theta_cyc * Shock).clamp(0.6, 1.2)
        F_total = torch.clamp(F_dev * F_rep * F_cyc, 0.4, 1.8)

        births_per_year = (female_RF * asfr.unsqueeze(0)).sum(dim=1) * F_total
        births = torch.clamp(births_per_year * dt_years, min=0.0)

        neon_haz_R = haz_R_A[:, 0]
        neon_surv = torch.exp(-neon_haz_R * dt_years)
        births_surv = births * neon_surv
        pop_after_age[:, 0] += births_surv

        # ---------- Optional migration (off by default) ----------
        rate_ann = cfg.migration_rate_annual
        if rate_ann > 0.0:
            a0, a1 = 18, 40
            mobile = pop_after_age[:, a0:a1]
            attract = 0.6 * (gdp_pc_r / (state.gdp_pc_baseline_R + eps)) + 0.4 * (
                0.5 + 0.5 * aec_norm
            )
            attract = attract / (attract.mean() + eps)
            nbr = state.nbr_idx
            dist = state.distance.gather(1, nbr)
            cost = 1.0 + dist / (dist.mean() + eps)
            kappa = cfg.migration_kappa
            w = torch.relu(attract[nbr] / cost**kappa)
            w = w / (w.sum(dim=1, keepdim=True) + eps)
            frac_move = min(max(rate_ann * dt_years, 0.0), 0.25)
            out_R = (mobile.sum(dim=1) * frac_move).unsqueeze(1)
            move_Rk = out_R * w
            age_share = mobile / (mobile.sum(dim=1, keepdim=True) + eps)
            pop_after_age[:, a0:a1] -= age_share * move_Rk.sum(dim=1, keepdim=True)
            dest_idx = nbr.reshape(-1)
            inflow = age_share.repeat_interleave(nbr.shape[1], dim=0) * move_Rk.reshape(
                -1, 1
            )
            add = torch.zeros_like(pop_after_age[:, a0:a1])
            add = add.index_add(0, dest_idx, inflow)
            pop_after_age[:, a0:a1] += add

        # ---------- Update state totals ----------
        state.pop_age.data = torch.clamp(pop_after_age, min=0.0)
        state.population.data = torch.clamp(state.pop_age.sum(dim=1), min=0.0)

        # ---------- Consumption and labor couplings ----------
        w_cons = state.consump_w_age
        w_part = state.participation_w_age
        cons_now = (state.pop_age * w_cons.unsqueeze(0)).sum(dim=1)
        cons_base = state.consump_base_R
        state.consump_scale_R.data = torch.clamp(
            cons_now / (cons_base + eps), 0.25, 4.0
        )
        labor_now = (state.pop_age * w_part.unsqueeze(0)).sum(dim=1)
        labor_base = state.labor_base_R
        state.labor_factor_R.data = torch.clamp(
            labor_now / (labor_base + eps), 0.2, 1.2
        )

        # ---------- Dependency and PSR ----------
        wa0 = cfg.working_age
        ra0 = cfg.retirement_age
        work = state.pop_age[:, wa0:ra0].sum(dim=1)
        young = state.pop_age[:, :wa0].sum(dim=1)
        old = state.pop_age[:, ra0:].sum(dim=1)
        state.psr_R.data = work / (old + eps)
        state.dep_ratio_R.data = (young + old) / (work + eps)

        # ---------- Wallet inheritance & birth endowments ----------
        region_idx = state.agent_region
        death_frac_i = death_frac_R[region_idx]
        w_raw = torch.pow(state.greed + 1e-9, cfg.inherit_conc)

        w_sum_r = state.w_sum_r_buffer.zero_()
        w_sum_r.index_add_(0, region_idx, w_raw)
        w_norm = w_raw / (w_sum_r[region_idx] + eps)

        # Process each wallet type individually to avoid stack/unbind overhead
        # and large intermediate tensors.
        wallets_and_pools = [
            (state.eATP, state.pool_atp_R),
            (state.eADP, state.pool_adp_R),
            (state.eAMP, state.pool_amp_R),
        ]
        inherit_frac = cfg.inherit_frac_on_death

        for wallet, pool in wallets_and_pools:
            # Deduct from agents who died
            removed_i = wallet * death_frac_i
            wallet.data.sub_(removed_i)

            # Aggregate removed amounts into regional pools
            removed_pool_r = state.removed_pool_r_buffer.zero_()
            removed_pool_r.index_add_(0, region_idx, removed_i)

            # Distribute to heirs
            heir_pool_r = removed_pool_r * inherit_frac
            heir_share_i = w_norm * heir_pool_r[region_idx]
            wallet.data.add_(heir_share_i)

            # Update regional summary pools (e.g., pool_atp_R)
            if pool is not None:
                net_loss_r = removed_pool_r - heir_pool_r
                pool.data.sub_(net_loss_r)

        births_total = births_surv

        ones_weights = torch.ones_like(region_idx, dtype=DTYPE)
        counts_r = state.counts_r_buffer.zero_()
        counts_r.index_add_(0, region_idx, ones_weights)

        counts_safe = torch.clamp(counts_r, min=1.0)
        add_atp_i = cfg.birth_endow_atp * (births_total / counts_safe)[region_idx]
        state.eATP.data.add_(add_atp_i)
