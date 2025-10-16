import torch
from torch.profiler import record_function
from ..config import EconConfig
from ..domain.state import WorldState
from ..services.agent_behavior import agent_budgets_and_demand
from ..services.production import run_production
from ..services.energy_bank import run_recharging
from ..services.pricing import update_prices, update_exergy_and_sink_prices
from ..services.trade import run_trade
from ..services.policy import aec_by_region, ers_demurrage_factors
from ..services.innovation import update_innovation_and_effects
from ..services.extraction import run_extraction
from ..services.storage_invest import apply_storage_investment
from ..services.demography import _CompiledDemographyStep
from ..services.settlement import apply_demurrage
from ..services.environment import update_environment
from ..services.metrics_flow import value_added_production, value_added_extraction
from ..services.consumption import run_consumption
from ..utils.tensor_utils import Device, DTYPE


class _CompiledStepBody(torch.nn.Module):
    def __init__(self, cfg: EconConfig):
        super().__init__()
        self.cfg = cfg
        self.demography_step = _CompiledDemographyStep(cfg)

        bases = torch.tensor(
            [cfg.save_base, cfg.invest_innov_base, cfg.invest_storage_base],
            device=Device,
            dtype=DTYPE,
        )
        self.register_buffer("bases", bases)

        scales = torch.tensor(
            [
                cfg.save_greed_scale,
                cfg.invest_innov_greed_scale,
                cfg.invest_storage_greed_scale,
            ],
            device=Device,
            dtype=DTYPE,
        )
        self.register_buffer("scales", scales)

    def forward(self, state: WorldState, need_prev: torch.Tensor):
        # 0) Recharge ATP from previous-step demand and update pools; books remain agent-level
        run_recharging(state, need_prev, state.pool_adp_R)

        # 0.5) Exogenous renewable/biological inflows (resource-locality proxy)
        state.inventory.data = torch.clamp(
            state.inventory.data + state.cfg.dt * state.endowment, min=0.0
        )

        # 1) Current AEC from pools -> demurrage controller and throughput gate
        atp_pool = state.pool_atp_R
        adp_pool = state.pool_adp_R
        amp_pool = state.pool_amp_R
        aec_r = aec_by_region(atp_pool, adp_pool, amp_pool)

        # 2) Initialize this-step ATP "book" at the regional pool
        atp_book = atp_pool.clone()

        # 3) Agent demand and investment budgets (also does nominal->ADP FX)
        demand_value_R, innov_budget_RJ, storage_budget_R = agent_budgets_and_demand(
            state, self.bases, self.scales
        )
        demand_qty_R = demand_value_R / (state.price.T + 1e-6)

        # 4) Innovation updates effective process parameters
        update_innovation_and_effects(state, innov_budget_RJ)

        # 5) Resource extraction (ATP/sink gated)
        q_RM, atp_book = run_extraction(state, atp_book)

        # 6) Production (ATP/sink gated + Leontief limiting)
        rate_RJ, atp_book = run_production(state, atp_book, aec_r)

        # 7) Trade (neighbor transport, ATP/sink gated)
        supply_R = torch.relu(state.inventory)
        atp_book = run_trade(state, supply_R, demand_qty_R, atp_book, kappa=0.8)

        # 8) Consumption use-phase exergy + sink and settlement
        atp_book = run_consumption(state, demand_qty_R, atp_book, frac=1.0)

        # 9) Update environment (pollutant stock)
        update_environment(state, state.emit_sink_R)

        # 10) Capital investments in storage infrastructure
        apply_storage_investment(state, storage_budget_R)

        # 11) Prices and shadow prices
        supply_now = torch.relu(state.inventory)
        update_prices(state, demand_qty_R, supply_now)
        update_exergy_and_sink_prices(state)

        # 12) Demurrage and AMP leak (policy circuit breaker)
        dem_factors = ers_demurrage_factors(self.cfg, aec_r)
        apply_demurrage(state, dem_factors)

        # 13) GDP (value-added flows)
        gdp_flow_R = value_added_production(state, rate_RJ) + value_added_extraction(
            state, q_RM
        )

        # 14) Demography integrates after GDP flow computed for this step
        pop_safe = torch.clamp(state.population, min=1e-9)
        gdp_pc_r = gdp_flow_R / pop_safe
        self.demography_step(state, aec_r, gdp_pc_r)

        return gdp_flow_R, aec_r


class ATPEconomy:
    def __init__(self, cfg: EconConfig):
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.dtype = DTYPE
        self.t = 0  # day counter

        self.state = WorldState(cfg)
        self.state.register_buffer(
            "exergy_need_R", torch.zeros(cfg.R, device=Device, dtype=self.dtype)
        )
        self.state.register_buffer(
            "sink_use_R", torch.zeros(cfg.R, device=Device, dtype=self.dtype)
        )
        self.state.register_buffer(
            "atp_minted_R", torch.zeros(cfg.R, device=Device, dtype=self.dtype)
        )
        self.state.register_buffer(
            "emit_sink_R", torch.zeros(cfg.R, device=Device, dtype=self.dtype)
        )

        update_innovation_and_effects(
            self.state, torch.zeros(cfg.R, cfg.J, device=Device, dtype=self.dtype)
        )

        self.compiled_step_body = torch.compile(_CompiledStepBody(cfg), fullgraph=True)

    @torch.no_grad()
    def step(self) -> dict:
        need_prev = self.state.exergy_need_R.clone()
        self.state.exergy_need_R.zero_()
        self.state.sink_use_R.zero_()
        self.state.emit_sink_R.zero_()

        gdp_flow_R, aec_r = self.compiled_step_body(self.state, need_prev)

        if self.t == 0:
            pop_safe = torch.clamp(self.state.population, min=1e-9)
            gdp_pc_R = gdp_flow_R / pop_safe
            self.state.gdp_pc_ema_R.copy_(gdp_pc_R)
            self.state.gdp_pc_ema_prev_R.copy_(gdp_pc_R)
            self.state.gdp_pc_baseline_R.copy_(torch.clamp(gdp_pc_R, min=1e-6))
            eps = 1e-9
            g_ratio = torch.log(
                torch.clamp(
                    self.state.gdp_pc_ema_R / (self.state.gdp_pc_baseline_R + eps),
                    min=1e-6,
                )
            )
            dev_proxy = torch.clamp(0.5 + 0.5 * torch.tanh(0.3 * g_ratio), 0.0, 1.0)
            self.state.dev_index_R.copy_(dev_proxy)

        gdp_pc_R = gdp_flow_R / torch.clamp(self.state.population, min=1e-9)
        metrics = self.collect_metrics(aec_r, gdp_flow_R, gdp_pc_R)

        self.t += 1
        return metrics

    @torch.no_grad()
    def collect_metrics(
        self,
        aec_r: torch.Tensor,
        gdp_flow_R: torch.Tensor,
        gdp_pc_R: torch.Tensor,
    ) -> dict:
        gdp_proxy = (self.state.price * torch.relu(self.state.inventory.T)).sum(0)
        return {
            "AEC_region": aec_r.cpu().numpy(),
            "GDP_proxy_region": gdp_proxy.cpu().numpy(),
            "GDP_flow_region": gdp_flow_R.cpu().numpy(),
            "GDP_pc_region": gdp_pc_R.cpu().numpy(),
            "ATP_minted_region": self.state.atp_minted_R.cpu().numpy(),
            "sink_utilization": (self.state.sink_use / self.state.sink_cap)
            .cpu()
            .numpy(),
            "mu_ex": self.state.mu_ex.cpu().numpy(),
            "lambda_sink": self.state.lambda_sink.cpu().numpy(),
            "population_region": self.state.population.cpu().numpy(),
            "psr_region": getattr(
                self.state, "psr_R", torch.zeros_like(self.state.population)
            )
            .cpu()
            .numpy(),
            "dependency_region": getattr(
                self.state, "dep_ratio_R", torch.zeros_like(self.state.population)
            )
            .cpu()
            .numpy(),
            "exergy_productivity_region": (
                gdp_flow_R / (self.state.atp_minted_R + 1e-9)
            )
            .cpu()
            .numpy(),
            "sink_intensity_region": (self.state.emit_sink_R / (gdp_flow_R + 1e-9))
            .cpu()
            .numpy(),
        }
