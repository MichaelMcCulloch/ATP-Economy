# src/atp_economy/domain/state.py
import torch
from torch import nn
from ..config import EconConfig
from ..utils.tensor_utils import Device, DTYPE


def _partition_goods(G: int, M: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Partition G goods into Resources (R_used), Intermediates (I_idx), Finals (F_idx).
    """
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
    """
    Build a block‑triangular stoichiometric matrix S[G,J] with two stages:
      Stage A:  R -> I
      Stage B:  R + I -> F
    """
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

    # Stage A coverage: each I appears
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

    # Stage B coverage: each F appears
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

        # Neighbor graph
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

        # --- Production Network (block‑triangular S) ---
        S_block = _make_block_stoichiometry(
            G=G,
            J=J,
            M=cfg.n_resources,
            seed=cfg.seed,
            device=Device,
            dtype=self.dtype,
            stageA_frac=0.4,
        )
        self.S = nn.Parameter(S_block, requires_grad=False)  # [G,J]

        # Persist final/intermediate indices for consumption logic
        _, I_idx, F_idx = _partition_goods(G, cfg.n_resources)
        self.inter_idx = nn.Parameter(I_idx, requires_grad=False)
        self.final_idx = nn.Parameter(F_idx, requires_grad=False)

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

        # Effective, region-specific process params (updated by innovation)
        self.tech_T = nn.Parameter(
            torch.zeros(R, J, device=Device, dtype=self.dtype), requires_grad=False
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

        # --- Regional Endowments & Energy ---
        self.endowment = nn.Parameter(
            torch.rand(R, G, device=Device, dtype=self.dtype) * 1e5,
            requires_grad=False,
        )
        self.gen_exergy = nn.Parameter(
            torch.rand(R, device=Device, dtype=self.dtype) * 2e5, requires_grad=False
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
        self.sink_cap = nn.Parameter(
            torch.ones(R, device=Device, dtype=self.dtype) * 1e6, requires_grad=False
        )
        self.sink_use = nn.Parameter(
            torch.zeros(R, device=Device, dtype=self.dtype), requires_grad=False
        )

        # Resource reserves (first n_resources goods are extractive)
        M = min(cfg.n_resources, G)
        self.res_goods = nn.Parameter(
            torch.arange(M, device=Device), requires_grad=False
        )
        self.reserves = nn.Parameter(
            torch.rand(R, M, device=Device, dtype=self.dtype) * cfg.reserves_scale,
            requires_grad=False,
        )
        self.reserves_max = nn.Parameter(
            torch.ones(R, M, device=Device, dtype=self.dtype) * cfg.reserves_scale,
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

        # --- Agents & Preferences ---
        self.agent_region = nn.Parameter(
            torch.randint(0, R, (N,), device=Device), requires_grad=False
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
        self.fiat = nn.Parameter(
            torch.rand(N, device=Device, dtype=self.dtype) * 1e3, requires_grad=False
        )
        self.crypto = nn.Parameter(
            torch.rand(N, device=Device, dtype=self.dtype) * 200, requires_grad=False
        )
        self.price = nn.Parameter(
            torch.rand(G, R, device=Device, dtype=self.dtype) + 0.1,
            requires_grad=False,
        )
        self.mu_ex = nn.Parameter(
            torch.ones(R, device=Device, dtype=self.dtype) * cfg.mu0,
            requires_grad=False,
        )
        self.lambda_sink = nn.Parameter(
            torch.ones(R, device=Device, dtype=self.dtype) * cfg.lambda0,
            requires_grad=False,
        )

        # --- Demography ---
        self.population = nn.Parameter(
            torch.ones(R, device=Device, dtype=self.dtype) * cfg.pop_init_scale,
            requires_grad=False,
        )
        self.population0 = nn.Parameter(self.population.clone(), requires_grad=False)

        # Apply scaling
        self.gen_exergy.data *= cfg.gen_scale
        self.storage_soc.data *= cfg.storage_scale
        self.sink_cap.data *= cfg.sink_cap_scale
        self.sigma_base.data *= cfg.sink_intensity_scale

        self.to(device=Device, dtype=self.dtype)

        # Initialize AEC (ADP->ATP shift to target corridor center)
        self._initialize_aec_in_band()

        # Precompute CSR ordering for agent->region segmented sums
        with torch.no_grad():
            order = torch.argsort(self.agent_region)
            counts = torch.bincount(self.agent_region, minlength=R)
            rowptr = torch.zeros(R + 1, device=Device, dtype=torch.long)
            rowptr[1:] = counts.cumsum(0)
        self.register_buffer("agent_order", order)
        self.register_buffer("rowptr", rowptr)

        # Initialize regional adenylate pools (used to avoid per-step re-aggregation)
        with torch.no_grad():
            idx = self.agent_region
            atp_pool0 = torch.bincount(idx, weights=self.eATP, minlength=R)
            adp_pool0 = torch.bincount(idx, weights=self.eADP, minlength=R)
            amp_pool0 = torch.bincount(idx, weights=self.eAMP, minlength=R)
        self.register_buffer("pool_atp_R", atp_pool0)
        self.register_buffer("pool_adp_R", adp_pool0)
        self.register_buffer("pool_amp_R", amp_pool0)

    @torch.no_grad()
    def _initialize_aec_in_band(self):
        """Shift ADP->ATP per region to target initial AEC without changing totals."""
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

        # Move a fraction of ADP -> ATP if below target
        num = torch.clamp(target - aec_r, min=0.0)
        denom = 0.5 * (adp_r / total_r) + eps
        x_r = torch.clamp(num / denom, min=0.0, max=1.0)
        x_i = x_r[self.agent_region]
        transfer_i = x_i * self.eADP
        self.eADP.data.sub_(transfer_i)
        self.eATP.data.add_(transfer_i)
