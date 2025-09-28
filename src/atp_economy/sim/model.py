# src/atp_economy/sim/model.py
import torch
from torch.profiler import record_function
from ..config import EconConfig
from ..domain.state import WorldState
from ..services.agent_behavior import agent_budgets_and_demand
from ..services import (
    production,
    energy_bank,
    pricing,
    trade,
    policy,
)
from ..services.innovation import update_innovation_and_effects
from ..services.extraction import run_extraction
from ..services.storage_invest import apply_storage_investment
from ..services.demography import update_population_and_inheritance
from ..services.settlement import apply_demurrage
from ..services.environment import update_environment
from ..services.metrics_flow import value_added_production, value_added_extraction
from ..services.consumption import run_consumption
from ..utils.tensor_utils import Device, DTYPE


class ATPEconomy:
    def __init__(self, cfg: EconConfig):
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.dtype = DTYPE

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

        # Initialize effective process parameters from base (T=0)
        update_innovation_and_effects(
            self.state, torch.zeros(cfg.R, cfg.J, device=Device, dtype=self.dtype)
        )

    @torch.no_grad()
    def step(self) -> dict:
        # Keep previous-step exergy demand for recharge targeting
        need_prev = self.state.exergy_need_R.clone()

        # Reset this-step accumulators
        self.state.exergy_need_R.zero_()
        self.state.sink_use_R.zero_()
        self.state.emit_sink_R.zero_()

        # Recharge ADP -> ATP against last-step needs
        with record_function("phase:recharge"):
            energy_bank.run_recharging(self.state, need_prev, self.state.pool_adp_R)

        # Read maintained pools (no scan over N)
        with record_function("phase:aggregation_post"):
            atp_pool = self.state.pool_atp_R
            adp_pool = self.state.pool_adp_R
            amp_pool = self.state.pool_amp_R
            aec_r = policy.aec_by_region(atp_pool, adp_pool, amp_pool)

        # Live ATP book for this step (decremented by extraction/production/trade/consumption)
        atp_book = atp_pool.clone()

        # Agent budgets and demand
        with record_function("phase:agent_budgets_and_demand"):
            demand_value_R, innov_budget_RJ, storage_budget_R = (
                agent_budgets_and_demand(self.state)
            )
            demand_qty_R = demand_value_R / (self.state.price.T + 1e-6)

        # Innovation
        with record_function("phase:innovation"):
            update_innovation_and_effects(self.state, innov_budget_RJ)

        # Extraction (ATP book)
        with record_function("phase:extraction"):
            q_RM = run_extraction(self.state, atp_book)  # [R,M]

        # Production (ATP book)
        with record_function("phase:production"):
            rate_RJ = production.run_production(self.state, atp_book, aec_r)  # [R,J]

        # Trade (ATP book)
        with record_function("phase:trade"):
            supply_R = torch.relu(self.state.inventory)  # [R,G]
            trade.run_trade(self.state, supply_R, demand_qty_R, atp_book, kappa=0.8)

        # Consumption of final goods (ATP book)
        with record_function("phase:consumption"):
            run_consumption(self.state, demand_qty_R, atp_book, frac=1.0)

        # Environment integration (production + extraction + logistics + consumption)
        with record_function("phase:environment"):
            update_environment(self.state, self.state.emit_sink_R)

        # Storage investment
        with record_function("phase:storage_invest"):
            apply_storage_investment(self.state, storage_budget_R)

        # Pricing (materials), then duals μ and λ
        with record_function("phase:pricing"):
            supply_now = torch.relu(self.state.inventory)  # [R,G]
            pricing.update_prices(self.state, demand_qty_R, supply_now)
            pricing.update_exergy_and_sink_prices(self.state)

        # Demurrage
        with record_function("phase:demurrage"):
            dem_factors = policy.ers_demurrage_factors(self.cfg, aec_r)
            apply_demurrage(self.state, dem_factors)

        # Demography and inheritance
        with record_function("phase:demography"):
            update_population_and_inheritance(self.state, aec_r)

        # Metrics
        with record_function("phase:metrics"):
            gdp_flow_R = value_added_production(
                self.state, rate_RJ
            ) + value_added_extraction(self.state, q_RM)
            pop = torch.clamp(self.state.population, min=1e-9)
            gdp_pc_R = gdp_flow_R / pop
            return self.collect_metrics(aec_r, gdp_flow_R, gdp_pc_R)

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
        }
