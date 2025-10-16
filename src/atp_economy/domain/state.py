import torch
from torch import nn
from ..config import EconConfig
from ..utils.tensor_utils import Device, DTYPE


def _default_hazard_vector(
    cfg: EconConfig, age_years: torch.Tensor, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    A = age_years.numel()
    hazard_A = torch.zeros(A, device=device, dtype=dtype)

    alpha_base = float(getattr(cfg, "adult_gomp_alpha", 4.2e-5))
    beta_base = float(getattr(cfg, "adult_gomp_beta", 0.085))
    lambda_base = float(getattr(cfg, "adult_makeham_lambda", 5.0e-4))
    imr_base = float(getattr(cfg, "imr_base", 0.03))
    u5_child_base = float(getattr(cfg, "u5_child_base", 0.001))
    youth_base = float(getattr(cfg, "youth_base", 2.0e-4))

    hazard_A[0] = imr_base
    hazard_A[1:5] = u5_child_base
    hazard_A[5:15] = youth_base
    a_adult = age_years[15:]
    hazard_A[15:] = lambda_base + alpha_base * torch.exp(
        beta_base * (a_adult - 40.0).clamp(min=-40.0, max=60.0)
    )
    return hazard_A


def _aging_matrix(
    A: int, frac: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    frac = float(max(0.0, min(1.0, frac)))
    M = torch.zeros(A, A, device=device, dtype=dtype)
    idx = torch.arange(0, A - 1, device=device)
    M[idx, idx] = 1.0 - frac
    M[idx + 1, idx] = frac
    M[A - 1, A - 1] = 1.0
    return M


def _default_asfr_vector(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ages = torch.arange(15, 50, device=Device, dtype=dtype)
    mu1, sig1, w1 = 26.0, 4.0, 0.75
    mu2, sig2, w2 = 32.0, 5.5, 0.25
    g1 = torch.exp(-0.5 * ((ages - mu1) / sig1) ** 2)
    g2 = torch.exp(-0.5 * ((ages - mu2) / sig2) ** 2)
    shape = w1 * g1 + w2 * g2
    shape = shape / (shape.sum() + 1e-9)
    return 0.075 * shape


def _partition_goods(G: int, M: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M_eff = int(max(1, min(M, G - 2)))
    R_used = torch.arange(M_eff, device=Device)

    remaining = G - M_eff
    I_size = max(1, remaining // 2)
    F_size = max(1, remaining - I_size)

    I_start = M_eff
    I_end = I_start + I_size
    I_idx = torch.arange(I_start, I_end, device=Device)
    F_idx = torch.arange(I_end, I_end + F_size, device=Device)
    return R_used, I_idx, F_idx


def _make_block_stoichiometry(
    G: int,
    J: int,
    M: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    stageA_frac: float = 0.4,
) -> torch.Tensor:
    torch.manual_seed(int(seed) if seed is not None else 0)

    S = torch.zeros(G, J, device=device, dtype=dtype)

    R_used, I_idx, F_idx = _partition_goods(G, M)
    nR, nI, nF = len(R_used), len(I_idx), len(F_idx)

    J1 = max(1, min(J - 1, int(round(stageA_frac * J))))
    J2 = J - J1

    def _mag(n, low=0.3, high=1.2):
        return low + (high - low) * torch.rand(int(n), device=device, dtype=dtype)

    def _randint(lo: int, hi: int) -> int:
        lo, hi = int(lo), int(hi)
        if hi <= lo:
            return lo
        return int(torch.randint(lo, hi, (1,), device=device).item())

    for k, i in enumerate(I_idx):
        j = k % J1
        if nR > 0:
            nin = 1 if nR == 1 else _randint(1, min(3, max(2, nR)))
            rin = torch.randperm(nR, device=device)[:nin]
            S[R_used[rin], j] -= _mag(nin)
        S[i, j] += _mag(1).item()
        if nI > 1 and torch.rand((), device=device) < 0.35:
            extra = _randint(1, min(2, nI))
            pool = I_idx[I_idx != i]
            sel = pool[torch.randperm(len(pool), device=device)[:extra]]
            S[sel, j] += _mag(extra)

    for j in range(J1):
        if S[:, j].abs().sum() == 0:
            if nR > 0:
                nin = 1 if nR == 1 else _randint(1, min(3, max(2, nR)))
                rin = torch.randperm(nR, device=device)[:nin]
                S[R_used[rin], j] -= _mag(nin)
            nout = 1 if nI == 1 else _randint(1, min(3, max(2, nI)))
            rout = torch.randperm(nI, device=device)[:nout]
            S[I_idx[rout], j] += _mag(nout)

    for k, f in enumerate(F_idx):
        j = J1 + (k % max(1, J2))
        ninI = 1 if nI == 1 else _randint(1, min(3, max(2, nI)))
        iin = torch.randperm(nI, device=device)[:ninI]
        S[I_idx[iin], j] -= _mag(ninI)

        if nR > 0 and torch.rand((), device=device) < 0.7:
            ninR = 1 if nR == 1 else _randint(1, min(3, max(2, nR)))
            rin = torch.randperm(nR, device=device)[:ninR]
            S[R_used[rin], j] -= _mag(ninR)

        S[f, j] += _mag(1).item()
        if nF > 1 and torch.rand((), device=device) < 0.35:
            extra = _randint(1, min(3, nF))
            pool = F_idx[F_idx != f]
            sel = pool[torch.randperm(len(pool), device=device)[:extra]]
            S[sel, j] += _mag(extra)

    for j in range(J1, J):
        if S[:, j].abs().sum() == 0:
            ninI = 1 if nI == 1 else _randint(1, min(3, max(2, nI)))
            iin = torch.randperm(nI, device=device)[:ninI]
            S[I_idx[iin], j] -= _mag(ninI)

            if nR > 0 and torch.rand((), device=device) < 0.7:
                ninR = 1 if nR == 1 else _randint(1, min(3, max(2, nR)))
                rin = torch.randperm(nR, device=device)[:ninR]
                S[R_used[rin], j] -= _mag(ninR)

            noutF = 1 if nF == 1 else _randint(1, min(3, max(2, nF)))
            fout = torch.randperm(nF, device=device)[:noutF]
            S[F_idx[fout], j] += _mag(noutF)

    return S


class WorldState(nn.Module):
    def __init__(self, cfg: EconConfig):
        super().__init__()
        self.cfg = cfg
        self.dtype = DTYPE
        R, G, J, N, K = cfg.R, cfg.G, cfg.J, cfg.N, cfg.K_latent

        # --- Asymmetric Initialization Vector ---
        # MODIFIED: Replace spatially correlated sine wave with random quality.
        # This breaks the "similar neighbors" effect.
        regional_quality = 0.5 + torch.rand(R, device=Device, dtype=self.dtype)
        regional_quality = torch.clamp(regional_quality, min=0.1)  # Ensure positive

        # --- Geography ---
        self.latlon = nn.Parameter(
            torch.randn(R, 2, device=Device, dtype=self.dtype) * 10.0,
            requires_grad=False,
        )
        d = self.latlon[:, None, :] - self.latlon[None, :, :]
        dist = torch.sqrt((d**2).sum(-1)) + torch.eye(
            R, device=Device, dtype=self.dtype
        )
        self.distance = nn.Parameter(dist, requires_grad=False)
        self.border_friction = nn.Parameter(
            torch.rand(R, R, device=Device, dtype=self.dtype) * 0.2,
            requires_grad=False,
        )
        self.port_capacity = nn.Parameter(
            torch.rand(R, R, device=Device, dtype=self.dtype), requires_grad=False
        )

        with torch.no_grad():
            k = min(cfg.k_neighbors, R - 1)
            masked = dist + torch.eye(R, device=Device, dtype=self.dtype) * 1e9
            nbr_idx = torch.topk(-masked, k=k, dim=1).indices
            base_cost = 0.01 * dist + self.border_friction
            nbr_cost = torch.gather(base_cost, 1, nbr_idx)
            nbr_cap = torch.gather(self.port_capacity, 1, nbr_idx)
        self.nbr_idx = nn.Parameter(nbr_idx, requires_grad=False)
        self.nbr_cost = nn.Parameter(nbr_cost, requires_grad=False)
        self.nbr_cap = nn.Parameter(nbr_cap, requires_grad=False)

        # --- Production network ---
        S_block = _make_block_stoichiometry(
            G=G,
            J=J,
            M=cfg.n_resources,
            seed=cfg.seed,
            device=Device,
            dtype=self.dtype,
            stageA_frac=0.4,
        )
        self.S = nn.Parameter(S_block, requires_grad=False)
        _, I_idx, F_idx = _partition_goods(G, cfg.n_resources)
        self.inter_idx = nn.Parameter(I_idx, requires_grad=False)
        self.final_idx = nn.Parameter(F_idx, requires_grad=False)

        self.register_buffer(
            "xi_cons",
            torch.full((F_idx.numel(),), 0.05, device=Device, dtype=self.dtype),
        )
        self.register_buffer(
            "sigma_cons",
            torch.full((F_idx.numel(),), 0.02, device=Device, dtype=self.dtype),
        )

        self.k_base = nn.Parameter(
            torch.rand(J, device=Device, dtype=self.dtype) * 0.5 + 0.1,
            requires_grad=False,
        )
        self.cap_j = nn.Parameter(
            torch.ones(J, device=Device, dtype=self.dtype) * 1e6, requires_grad=False
        )
        self.xi_base = nn.Parameter(
            torch.rand(J, device=Device, dtype=self.dtype) * 2.0, requires_grad=False
        )
        self.sigma_base = nn.Parameter(
            torch.rand(J, device=Device, dtype=self.dtype) * 0.5, requires_grad=False
        )
        self.inventory = nn.Parameter(
            torch.rand(R, G, device=Device, dtype=self.dtype) * 1e5,
            requires_grad=False,
        )

        # Asymmetric initial technology
        self.tech_T = nn.Parameter(
            torch.rand(R, J, device=Device, dtype=self.dtype)
            * 0.1
            * regional_quality.unsqueeze(1),
            requires_grad=False,
        )
        self.k_eff = nn.Parameter(
            torch.zeros(R, J, device=Device, dtype=self.dtype), requires_grad=False
        )
        self.xi_eff = nn.Parameter(
            torch.zeros(R, J, device=Device, dtype=self.dtype), requires_grad=False
        )
        self.sigma_eff = nn.Parameter(
            torch.zeros(R, J, device=Device, dtype=self.dtype), requires_grad=False
        )

        # --- Regional Endowments & Energy (Asymmetric) ---
        self.endowment = nn.Parameter(
            torch.rand(R, G, device=Device, dtype=self.dtype) * 1e5,
            requires_grad=False,
        )
        # Asymmetric energy generation
        self.gen_exergy = nn.Parameter(
            (0.5 + torch.rand(R, device=Device, dtype=self.dtype))
            * regional_quality
            * 2e5,
            requires_grad=False,
        )
        self.storage_soc = nn.Parameter(
            torch.rand(R, device=Device, dtype=self.dtype) * 1e6, requires_grad=False
        )
        self.storage_cap = nn.Parameter(
            torch.ones(R, device=Device, dtype=self.dtype) * 1e6, requires_grad=False
        )
        self.eta_rt = nn.Parameter(
            torch.ones(R, device=Device, dtype=self.dtype) * 0.85, requires_grad=False
        )
        # Asymmetric sink capacity
        self.sink_cap = nn.Parameter(
            torch.ones(R, device=Device, dtype=self.dtype) * regional_quality * 1e6,
            requires_grad=False,
        )
        self.sink_use = nn.Parameter(
            torch.zeros(R, device=Device, dtype=self.dtype), requires_grad=False
        )
        self.register_buffer(
            "pollutant", torch.zeros(R, device=Device, dtype=self.dtype)
        )

        self.gen_sink_intensity = nn.Parameter(
            0.01 + 0.02 * torch.rand(R, device=Device, dtype=self.dtype),
            requires_grad=False,
        )

        # --- Resources (Asymmetric) ---
        M = min(cfg.n_resources, G)
        self.res_goods = nn.Parameter(
            torch.arange(M, device=Device), requires_grad=False
        )
        # Asymmetric resource reserves
        reserves_base = (
            torch.rand(R, M, device=Device, dtype=self.dtype)
            * regional_quality.unsqueeze(1)
            * cfg.reserves_scale
        )
        self.reserves = nn.Parameter(reserves_base, requires_grad=False)
        self.reserves_max = nn.Parameter(
            torch.ones(R, M, device=Device, dtype=self.dtype)
            * regional_quality.unsqueeze(1)
            * cfg.reserves_scale,
            requires_grad=False,
        )
        self.xi_ext0 = nn.Parameter(
            torch.ones(M, device=Device, dtype=self.dtype) * cfg.xi_ext0,
            requires_grad=False,
        )
        self.sig_ext0 = nn.Parameter(
            torch.ones(M, device=Device, dtype=self.dtype) * cfg.sig_ext0,
            requires_grad=False,
        )

        # --- Agents & Preferences (Asymmetric Distribution) ---
        # Asymmetric population
        pop_dist = regional_quality / regional_quality.sum()
        # Asymmetric agent distribution based on population
        self.agent_region = nn.Parameter(
            torch.multinomial(pop_dist, N, replacement=True), requires_grad=False
        )
        self.Z = nn.Parameter(
            torch.randn(N, K, device=Device, dtype=self.dtype) * 0.5,
            requires_grad=False,
        )
        self.W = nn.Parameter(
            torch.randn(K, G, device=Device, dtype=self.dtype) * 0.5,
            requires_grad=False,
        )
        self.pref_theta = nn.Parameter(self.Z @ self.W, requires_grad=False)
        self.greed = nn.Parameter(
            torch.sigmoid(torch.randn(N, device=Device, dtype=self.dtype) * 0.75),
            requires_grad=False,
        )

        # --- Wallets & Prices ---
        self.eATP = nn.Parameter(
            torch.rand(N, device=Device, dtype=self.dtype) * 1e3, requires_grad=False
        )
        self.eADP = nn.Parameter(
            torch.rand(N, device=Device, dtype=self.dtype) * 2e3, requires_grad=False
        )
        self.eAMP = nn.Parameter(
            torch.rand(N, device=Device, dtype=self.dtype) * 1e3, requires_grad=False
        )
        self.price = nn.Parameter(
            torch.rand(G, R, device=Device, dtype=self.dtype) + 0.1, requires_grad=False
        )
        self.register_buffer("logp_anchor", torch.log(self.price.data))

        self.mu_ex = nn.Parameter(
            torch.ones(R, device=Device, dtype=self.dtype) * cfg.mu0,
            requires_grad=False,
        )
        self.register_buffer(
            "ema_ex_ratio", torch.ones(R, device=Device, dtype=self.dtype)
        )

        self.lambda_sink = nn.Parameter(
            torch.ones(R, device=Device, dtype=self.dtype) * cfg.lambda0,
            requires_grad=False,
        )
        self.register_buffer(
            "ema_sink_util", torch.zeros(R, device=Device, dtype=self.dtype)
        )

        # --- Asymmetric scalar population ---
        total_pop = R * cfg.pop_init_scale
        self.population = nn.Parameter(
            pop_dist * total_pop,
            requires_grad=False,
        )
        self.population0 = nn.Parameter(self.population.clone(), requires_grad=False)

        # --- Age structure add-ons ---
        self.age_years = nn.Parameter(
            torch.arange(0, 101, device=Device, dtype=self.dtype), requires_grad=False
        )
        child_share, work_share, old_share = 0.24, 0.65, 0.11
        A = self.age_years.numel()
        w = torch.zeros(A, device=Device, dtype=self.dtype)
        w[:18] = child_share / 18.0
        w[18:65] = work_share / 47.0
        w[65:] = old_share / 36.0
        pop_age0 = self.population.unsqueeze(1) * w.unsqueeze(0)  # [R,A]
        self.pop_age = nn.Parameter(pop_age0, requires_grad=False)

        dt_years = 1.0 / 365.0
        aging_M = _aging_matrix(A, dt_years, Device, self.dtype)
        self.register_buffer("aging_M", aging_M)

        cons_w = torch.ones(A, device=Device, dtype=self.dtype)
        cons_w[:18] = 0.6
        cons_w[65:] = 0.8
        self.consump_w_age = nn.Parameter(cons_w, requires_grad=False)

        part_w = torch.zeros(A, device=Device, dtype=self.dtype)
        part_w[18:25] = 0.5
        part_w[25:55] = 0.9
        part_w[55:65] = 0.5
        part_w[65:75] = 0.1
        part_w[75:] = 0.05
        self.participation_w_age = nn.Parameter(part_w, requires_grad=False)

        consump_base_R = (self.pop_age * self.consump_w_age.unsqueeze(0)).sum(dim=1)
        self.register_buffer("consump_base_R", consump_base_R)

        labor_base_R = (self.pop_age * self.participation_w_age.unsqueeze(0)).sum(dim=1)
        self.register_buffer("labor_base_R", labor_base_R)

        self.register_buffer(
            "consump_scale_R", torch.ones(R, device=Device, dtype=self.dtype)
        )
        self.register_buffer(
            "labor_factor_R", torch.ones(R, device=Device, dtype=self.dtype)
        )
        self.register_buffer("psr_R", torch.ones(R, device=Device, dtype=self.dtype))
        self.register_buffer(
            "dep_ratio_R", torch.ones(R, device=Device, dtype=self.dtype)
        )

        self.register_buffer(
            "gdp_pc_ema_R", torch.full((R,), 1.0, device=Device, dtype=self.dtype)
        )
        self.register_buffer(
            "gdp_pc_ema_prev_R", torch.full((R,), 1.0, device=Device, dtype=self.dtype)
        )
        self.register_buffer(
            "gdp_pc_baseline_R", torch.full((R,), 1.0, device=Device, dtype=self.dtype)
        )
        self.register_buffer(
            "dev_index_R", torch.zeros(R, device=Device, dtype=self.dtype)
        )
        self.register_buffer(
            "health_index_R", torch.zeros(R, device=Device, dtype=self.dtype)
        )
        asfr_vec = _default_asfr_vector(Device, self.dtype)
        self.register_buffer("asfr_vector", asfr_vec)
        hazard_vec = _default_hazard_vector(cfg, self.age_years, Device, self.dtype)
        self.register_buffer("hazard_A_base", hazard_vec)

        # Apply scaling factors
        self.gen_exergy.data *= cfg.gen_scale
        self.storage_soc.data *= cfg.storage_scale
        self.sink_cap.data *= cfg.sink_cap_scale
        self.sigma_base.data *= cfg.sink_intensity_scale
        self.gen_sink_intensity.data *= cfg.gen_sink_intensity_scale

        self.to(device=Device, dtype=self.dtype)

        # AEC initialization
        self._initialize_aec_in_band()

        # Precompute agent->region pools
        with torch.no_grad():
            idx = self.agent_region
            atp_pool0 = torch.bincount(idx, weights=self.eATP, minlength=R)
            adp_pool0 = torch.bincount(idx, weights=self.eADP, minlength=R)
            amp_pool0 = torch.bincount(idx, weights=self.eAMP, minlength=R)
        self.register_buffer("pool_atp_R", atp_pool0)
        self.register_buffer("pool_adp_R", adp_pool0)
        self.register_buffer("pool_amp_R", amp_pool0)

        # --- Aggregation Buffers ---
        self.register_buffer(
            "demand_R_buffer", torch.zeros(R, G, device=Device, dtype=self.dtype)
        )
        self.register_buffer(
            "innov_R_buffer", torch.zeros(R, device=Device, dtype=self.dtype)
        )
        self.register_buffer(
            "storage_R_buffer", torch.zeros(R, device=Device, dtype=self.dtype)
        )
        self.register_buffer(
            "inflow_R_buffer", torch.zeros(R, G, device=Device, dtype=self.dtype)
        )
        self.register_buffer(
            "w_sum_r_buffer", torch.zeros(R, device=Device, dtype=self.dtype)
        )
        self.register_buffer(
            "removed_pool_r_buffer", torch.zeros(R, device=Device, dtype=self.dtype)
        )
        self.register_buffer(
            "counts_r_buffer", torch.zeros(R, device=Device, dtype=self.dtype)
        )

    def _initialize_aec_in_band(self):
        cfg = self.cfg
        R = cfg.R
        eps = 1e-12
        idx = self.agent_region
        atp_r = torch.zeros(R, device=Device, dtype=self.dtype)
        adp_r = torch.zeros(R, device=Device, dtype=self.dtype)
        amp_r = torch.zeros(R, device=Device, dtype=self.dtype)
        atp_r.index_add_(0, idx, self.eATP)
        adp_r.index_add_(0, idx, self.eADP)
        amp_r.index_add_(0, idx, self.eAMP)
        total_r = atp_r + adp_r + amp_r + eps
        aec_r = (atp_r + 0.5 * adp_r) / total_r
        target = float(getattr(cfg, "aec_init", 0.5 * (cfg.aec_low + cfg.aec_high)))
        num = torch.clamp(target - aec_r, min=0.0)
        denom = 0.5 * (adp_r / total_r) + eps
        x_r = torch.clamp(num / denom, min=0.0, max=1.0)
        x_i = x_r[self.agent_region]
        transfer_i = x_i * self.eADP
        self.eADP.data.sub_(transfer_i)
        self.eATP.data.add_(transfer_i)
