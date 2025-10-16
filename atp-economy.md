```README.md
# ATP-Economy: A Biophysical Agent-Based Model

This project simulates a multi-region global economy grounded in biophysical principles. Unlike traditional macroeconomic models that treat money and resources as abstract quantities, this simulation enforces that all real economic activity—production, extraction, and trade—is ultimately constrained by primary resource flows and the finite capacity of the environment to absorb waste.

The model is designed to explore long-term dynamics where technological innovation, resource depletion, and environmental limits are the primary drivers of growth, stability, and inequality.

## Core Concepts: A Biophysical Economic Engine

The simulation is built on a set of core concepts that model the economy not as an abstract system of accounts, but as a living, resource-dependent metabolism.

### 1. A Three-Tiered Monetary System: The Financial Metabolism

The economy operates with a physically-grounded monetary system that mirrors cellular bioenergetics, ensuring financial claims are tied to the real capacity to perform work.

*   **High-Potential Capital (`eATP`):** The primary settlement asset, analogous to Adenosine Triphosphate. Only `eATP` can be "hydrolyzed" to finance real work—manufacturing, resource extraction, logistics, and even the use-phase of consumer goods. It is the ultimate unit of account for physical value creation.
*   **Working Capital (`eADP`):** The result of spending `eATP`. It represents capital that has been deployed into the economy and cannot be used for new production until it is "recharged."
*   **Deep Reserves (`eAMP`):** A non-circulating asset whose accumulation signals systemic financial distress or a severe lack of recapitalization opportunities, much like a cell running out of energy.

Recapitalization (`eADP` → `eATP`) is performed by regional **Capital Banks** (the "mitochondria" of the economy) and is strictly collateralized by verified primary resource flows (e.g., energy generation). Surplus generation is used to charge regional storage, creating a buffer against volatility. This prevents the unbacked creation of settlement-grade liquidity. Furthermore, `eATP` is subject to a **holding cost (demurrage)**, a small negative interest rate that encourages productive investment over sterile hoarding.

### 2. Production as Transformation: The Holistic Profit Motive

Economic activity is modeled as a network of production functions where sectors transform goods into other goods. The incentive to produce is driven by a comprehensive **profit motive (Affinity)**, which forces agents to account for total system costs:
*   Market prices of inputs and outputs.
*   The **shadow price of primary resources (`μ`)**, representing the cost of recapitalizing the capital spent.
*   The **shadow price of environmental capacity (`λ`)**, an endogenously determined Pigouvian tax on negative externalities.

A production process is only profitable—and therefore undertaken—if its expected revenue exceeds the full cost of its material inputs *and* its primary resource and environmental footprints.

### 3. Binding Environmental Constraints: The Finite Sink

The environment is modeled as a stock of pollutants that accumulates from economic activity and is reduced by a natural assimilation rate. The capacity of this "sink" is finite. Every economic process—including the final consumption of goods—generates a **dissipation cost** that adds to the pollutant stock. The shadow price `λ` is not set by a regulator but is adjusted by an endogenous controller that steers the system's emission flows toward a sustainable target utilization level. As the system approaches its environmental limits, `λ` rises, rendering high-externality production unprofitable and forcing a hard ceiling on unsustainable economic throughput.

### 4. Macroprudential Stability: The Financial Circuit Breaker

The financial health of each regional economy is measured by its **Capital Adequacy Ratio (AEC)**, a metric analogous to a bank's liquidity coverage ratio. This ratio reflects the availability of high-potential `eATP` relative to total circulating capital. The system features an automatic stabilizer (an "AMPK-like gate") where a low regional AEC automatically throttles the rates of all production activities. This macroprudential mechanism acts as a circuit breaker, preventing a "run on the bank" by forcing the economy to conserve its settlement liquidity during periods of stress, thereby avoiding systemic collapse.

### 5. Nominal Overlays: Fiat and Crypto as Coordination Layers

Fiat and crypto currencies are not removed but are modeled as coordination and credit layers that sit on top of the biophysical economy. They serve as units of account, facilitate price discovery, and warehouse risk, but they **cannot settle physical transactions**. Their only bridge to the real economy is a regulated **FX valve** that allows agents to convert nominal assets into `eADP` (not `eATP`) under spreads and regional caps. These caps are tied to recent physical `eATP` generation, ensuring that nominal credit creation cannot outpace the real economy's capacity to perform work.

## Endogenous Dynamics: The Engine of Change

The model includes several interconnected feedback loops that drive long-term economic evolution.

### Resource Extraction and Depletion
A subset of goods are non-renewable resources with finite reserves. As reserves are depleted, the **marginal cost of extraction rises** (in terms of both primary resources and environmental externalities), creating endogenous Hubbert-style production peaks and forcing the economy to adapt to increasing scarcity.

### Innovation and Productivity Growth
Innovation is the primary escape valve from physical limits. Agents can invest in a regional **Technology Stock (`T`)**, which improves productivity by reducing the primary resource intensity (`ξ`), environmental intensity (`σ`), or increasing the catalytic rate (`k`) of production processes.

### Agent Behavior: Heterogeneity and Self-Interest
Agents possess a "greed" trait that creates a spectrum of economic behaviors, influencing their sensitivity to prices, their propensity to save, and their willingness to invest in long-term technology and infrastructure versus immediate consumption.

### Demography and Wealth Transfer
Regional birth and death rates are tied to economic health (AEC), creating a two-way feedback loop between prosperity and population size. Inheritance rules, skewed by the "greed" of heirs, allow for the study of how behavioral traits and wealth transfer mechanisms shape long-term inequality.

## Stability and Invariants

To support long-horizon runs without numerical artifacts, the engine now enforces several stability-by-design principles:

*   **Stable Price Dynamics:** Prices update in log-space with a trust region and a slow EMA anchor. This preserves the directionality of price adjustments while preventing overflow, underflow, and random-walk drift over tens of thousands of steps.
*   **Well-Posed Demand:** Agent choice probabilities are calculated from bounded logits and include an "outside option." This ensures the softmax function is always well-defined, even if all market goods become prohibitively expensive for some agents.
*   **Physical Absorptive Caps:** Regional investment in innovation and storage is bounded by a multiple of the exergy recharged in that step (`eATP` minted). This ties nominal budgets to real physical throughput, preventing runaway investment aggregation.
*   **Irreducible Biophysical Floors:** Effective exergy (`ξ_eff`) and sink (`σ_eff`) intensities cannot fall to exactly zero. These small but non-zero floors reflect irreducible physical costs and externalities, and prevent 0/0 errors in allocation logic.
*   **Strictly Positive Innovation Allocation:** Innovation budgets are allocated across production processes using a softmax function over their environmental intensities (`σ_eff`). This ensures investment is directed toward dirtier processes but never completely ignores any process, avoiding numerical instability.
*   **Fail-Fast Finite Checks:** During development and testing, the simulation can be run in a strict mode that immediately raises an error if any non-finite value (NaN, Inf) appears in a key state tensor. This is enabled by default and can be disabled for performance-critical runs by setting the environment variable `ATP_STRICT_FINITE=0`.

## How to Run the Simulation

This project uses `uv` for package management and `PyYAML` for configuration.

1.  **Install `uv`:**
    ```bash
    pip install uv
    ```

2.  **Create a virtual environment and install dependencies:**
    This will install `torch`, `typer`, `numpy`, `matplotlib`, and `pyyaml`.
    ```bash
    uv venv
    uv pip install -e .
    ```

3.  **Run a simulation:**
    Simulations are run from a configuration file. An example is provided in `configs/H1_baseline_stability.yaml`.
    ```bash
    # Run the simulation using a specific config
    uv run run-sim run configs/H10_decouple_and_grow.yaml

    # You can override key I/O settings for quick experiments:
    uv run run-sim run configs/H10_decouple_and_grow.yaml --steps 5000 --save-fig my_test_run.png
    ```

4.  **Profile the simulation:**
    To identify performance bottlenecks, you can run the simulation with the PyTorch profiler.
    ```bash
    # Run the profiler with default settings
    uv run run-sim profile --steps 120 --warmup 20

    # View the trace with TensorBoard
    tensorboard --logdir runs/prof
    ```

## Configuration Parameters

All simulation parameters are controlled via a single YAML configuration file. The table below details each parameter, grouped by its required YAML section.

| Parameter                             | Default Value           | Description                                                                                    |
| :------------------------------------ | :---------------------- | :--------------------------------------------------------------------------------------------- |
| **`runtime`**                         |                         |                                                                                                |
| `runtime.steps`                       | `20000`                 | Total number of simulation steps to execute.                                                   |
| `runtime.save_fig`                    | `"healthy_run.png"`     | File path to save the final summary plot.                                                      |
| `runtime.dpi`                         | `180`                   | Dots-per-inch resolution for the saved figure.                                                 |
| `runtime.style`                       | `"seaborn-v0_8"`        | Matplotlib style to use for plotting.                                                          |
| `runtime.save_metrics`                | `"healthy_metrics.npz"` | File path to save the simulation's time-series data.                                           |
| `runtime.seed`                        | `123`                   | The seed for the random number generator for reproducibility.                                  |
| **`sizes`**                           |                         |                                                                                                |
| `sizes.R`                             | (required)              | The number of distinct geographical regions in the world.                                      |
| `sizes.G`                             | (required)              | The number of different types of goods and services.                                           |
| `sizes.J`                             | (required)              | The number of distinct production processes (reactions).                                       |
| `sizes.N`                             | (required)              | The number of agent cohorts being simulated.                                                   |
| `sizes.k_latent`                      | `4`                     | The dimensionality of the latent preference space for agents.                                  |
| **`trade`**                           |                         |                                                                                                |
| `trade.k_neighbors`                   | `8`                     | Defines trade locality by limiting each region to trade with its 'k' nearest neighbors.        |
| `trade.alpha_logistics_ex`            | `0.08`                  | The exergy (`eATP`) cost per unit of goods moved per unit of distance.                         |
| `trade.alpha_logistics_sink`          | `0.005`                 | The environmental sink cost per unit of goods moved per unit of distance.                      |
| **`markets`**                         |                         |                                                                                                |
| `markets.tau`                         | `0.15`                  | The baseline demand temperature, controlling price sensitivity (lower is more sensitive).      |
| `markets.beta_aff`                    | `2.0`                   | The sensitivity of production rates to the calculated profit motive (Affinity).                |
| **`time`**                            |                         |                                                                                                |
| `time.demurrage`                      | `0.01`                  | The base per-step decay rate of `eATP` into `eADP` (holding cost).                             |
| `time.dt`                             | `1.0`                   | The duration of a single simulation step.                                                      |
| **`duals`**                           |                         |                                                                                                |
| `duals.eta_ex`                        | `0.01`                  | The learning rate for updating the primary resource shadow price (`μ`).                        |
| `duals.eta_sink`                      | `0.01`                  | The learning rate for updating the environmental sink shadow price (`λ`).                      |
| `duals.util_target`                   | `0.50`                  | The target environmental utilization level that the shadow price `λ` tries to maintain.        |
| `duals.mu_floor`                      | `0.005`                 | The minimum allowed value for the primary resource price `μ`.                                  |
| `duals.mu_cap`                        | `1000000.0`             | The maximum allowed value for the primary resource price `μ`.                                  |
| `duals.lambda_floor`                  | `0.02`                  | The minimum allowed value for the environmental sink price `λ`.                                |
| `duals.lambda_cap`                    | `1000000.0`             | The maximum allowed value for the environmental sink price `λ`.                                |
| `duals.mu0`                           | `0.02`                  | The initial value of the primary resource price `μ`.                                           |
| `duals.lambda0`                       | `0.05`                  | The initial value of the environmental sink price `λ`.                                         |
| `duals.ema_ex`                        | `0.90`                  | The smoothing factor (EMA decay) for the exergy price controller signal.                       |
| `duals.ema_sink`                      | `0.90`                  | The smoothing factor (EMA decay) for the sink price controller signal.                         |
| **`scaling`**                         |                         |                                                                                                |
| `scaling.gen_scale`                   | `0.35`                  | A multiplier for the baseline primary resource generation capacity of each region.             |
| `scaling.storage_scale`               | `0.30`                  | A multiplier for the initial stock of primary resource storage in each region.                 |
| `scaling.sink_cap_scale`              | `0.10`                  | A multiplier for the total environmental absorption capacity of each region.                   |
| `scaling.sink_intensity_scale`        | `5.0`                   | A multiplier for the baseline environmental cost of all production processes.                  |
| `scaling.gen_sink_intensity_scale`    | `1.0`                   | A multiplier for the baseline environmental cost of primary resource generation.               |
| `scaling.gen_noise`                   | `0.30`                  | The amplitude of random multiplicative noise applied to primary resource generation each step. |
| **`environment`**                     |                         |                                                                                                |
| `environment.sink_assim_rate`         | `0.01`                  | The per-step rate at which the environment naturally assimilates and removes pollutants.       |
| **`policy`**                          |                         |                                                                                                |
| `policy.aec_low`                      | `0.78`                  | The lower bound of the target Capital Adequacy Ratio (AEC) corridor.                           |
| `policy.aec_high`                     | `0.92`                  | The upper bound of the target Capital Adequacy Ratio (AEC) corridor.                           |
| `policy.ers_k`                        | `6.0`                   | The gain factor for adjusting demurrage based on deviations from the AEC corridor.             |
| `policy.gate_min`                     | `0.10`                  | The minimum throughput (0 to 1) allowed by the AMPK gate, even at very low AEC.                |
| `policy.gate_k`                       | `12.0`                  | The steepness of the AMPK gate's response to changes in the AEC.                               |
| `policy.aec_init`                     | `0.86`                  | The target AEC value that the simulation initializes agent wallets to match.                   |
| **`extraction`**                      |                         |                                                                                                |
| `extraction.n_resources`              | `4`                     | The number of goods (from index 0) that are treated as finite, extractable resources.          |
| `extraction.k_ext`                    | `0.2`                   | The base kinetic rate for all resource extraction activities.                                  |
| `extraction.beta_ext`                 | `3.0`                   | The sensitivity of extraction rates to the profit motive (Affinity).                           |
| `extraction.xi_ext0`                  | `1.0`                   | The baseline exergy (`eATP`) cost to extract one unit of a resource.                           |
| `extraction.sig_ext0`                 | `0.6`                   | The baseline environmental sink cost to extract one unit of a resource.                        |
| `extraction.dep_alpha_xi`             | `1.0`                   | Controls how rapidly the exergy cost of extraction increases as a resource is depleted.        |
| `extraction.dep_alpha_sig`            | `1.2`                   | Controls how rapidly the environmental cost of extraction increases as a resource is depleted. |
| `extraction.reserves_scale`           | `5000000.0`             | A multiplier for the initial quantity of each finite resource in each region.                  |
| **`demography`**                      |                         |                                                                                                |
| `demography.pop_init_scale`           | `1000000.0`             | A multiplier for the initial population of each region.                                        |
| `demography.birth_base`               | `0.015`                 | The baseline per-step birth rate.                                                              |
| `demography.death_base`               | `0.010`                 | The baseline per-step death rate.                                                              |
| `demography.aec_birth_center`         | `0.8`                   | The AEC level above which the birth rate begins to increase.                                   |
| `demography.aec_death_center`         | `0.5`                   | The AEC level below which the death rate begins to increase.                                   |
| `demography.birth_k`                  | `5.0`                   | The steepness of the birth rate's response to AEC.                                             |
| `demography.death_k`                  | `7.0`                   | The steepness of the death rate's response to AEC.                                             |
| `demography.birth_endow_atp`          | `0.2`                   | The amount of `eATP` endowed to each newborn agent cohort.                                     |
| `demography.birth_endow_fiat`         | `5.0`                   | The amount of fiat currency endowed to each newborn agent cohort.                              |
| **`inheritance`**                     |                         |                                                                                                |
| `inheritance.inherit_conc`            | `2.0`                   | An exponent controlling wealth concentration during inheritance (1=proportional, >1=unequal).  |
| `inheritance.inherit_frac_on_death`   | `0.9`                   | The fraction of a deceased agent's wealth that is redistributed to heirs.                      |
| **`behavior`**                        |                         |                                                                                                |
| `behavior.greed_tau_scale`            | `0.5`                   | Controls how much an agent's "greed" trait reduces their demand temperature (`τ`).             |
| `behavior.save_base`                  | `0.05`                  | The baseline fraction of their budget that agents save.                                        |
| `behavior.save_greed_scale`           | `0.10`                  | The additional savings propensity per unit of an agent's "greed" trait.                        |
| `behavior.invest_innov_base`          | `0.03`                  | The baseline fraction of their budget that agents invest in innovation.                        |
| `behavior.invest_innov_greed_scale`   | `0.05`                  | The additional innovation investment propensity per unit of "greed".                           |
| `behavior.invest_storage_base`        | `0.02`                  | The baseline fraction of their budget that agents invest in storage infrastructure.            |
| `behavior.invest_storage_greed_scale` | `0.04`                  | The additional storage investment propensity per unit of "greed".                              |
| **`innovation`**                      |                         |                                                                                                |
| `innovation.eta_innov`                | `0.001`                 | The efficiency of converting investment into the regional Technology Stock.                    |
| `innovation.innov_alpha`              | `1.0`                   | The concavity of the R&D investment function (exponent < 1 implies diminishing returns).       |
| `innovation.innov_spill`              | `0.001`                 | The per-step rate at which technology diffuses to neighboring regions.                         |
| `innovation.innov_decay`              | `0.001`                 | The per-step rate at which the Technology Stock depreciates or becomes obsolete.               |
| `innovation.beta_xi`                  | `0.4`                   | The effectiveness of technology at reducing the exergy intensity (`ξ`) of production.          |
| `innovation.beta_sigma`               | `0.5`                   | The effectiveness of technology at reducing the environmental intensity (`σ`) of production.   |
| `innovation.beta_kcat`                | `0.3`                   | The effectiveness of technology at increasing the catalytic rate (`k`) of production.          |
| **`investment`**                      |                         |                                                                                                |
| `investment.eta_storage`              | `0.0001`                | The efficiency of converting investment into new primary resource storage capacity.            |
| `investment.storage_decay`            | `0.0002`                | The per-step rate at which storage infrastructure capacity depreciates.                        |
| **`engine`**                          |                         |                                                                                                |
| `engine.xi_floor`                     | `1.0e-12`               | Minimum effective exergy intensity `ξ_eff`. Prevents zero cost per unit energy.                |
| `engine.sigma_floor`                  | `1.0e-12`               | Minimum effective sink intensity `σ_eff`. Prevents zero externality per unit throughput.       |
| `engine.softmax_temp_sigma`           | `0.5`                   | Temperature for softmax allocation of innovation budgets based on `σ_eff`.                     |
| `engine.cap_innov_exergy_mult`        | `50.0`                  | Max innovation spend as a multiple of `eATP` minted this step (absorptive capacity).           |
| `engine.cap_storage_exergy_mult`      | `25.0`                  | Max storage investment as a multiple of `eATP` minted this step (absorptive capacity).         |
| `engine.innov_I_cap`                  | `1.0e12`                | Upper bound on effective innovation increment per step (R&D absorption).                       |
| **`fx`**                              |                         |                                                                                                |
| `fx.fx_spread_fiat`                   | `0.02`                  | The bid-ask spread for converting fiat currency into `eADP`.                                   |
| `fx.fx_spread_crypto`                 | `0.05`                  | The bid-ask spread for converting crypto assets into `eADP`.                                   |
| `fx.fx_cap_mult`                      | `10.0`                  | The regional conversion cap, as a multiple of recently minted `eATP`.                          |
| `fx.fx_convert_base_frac`             | `0.25`                  | The baseline fraction of nominal balances agents attempt to convert to `eADP` each step.       |
| `fx.fx_convert_greed_scale`           | `0.35`                  | The additional conversion propensity per unit of an agent's "greed" trait.                     |

## Interpreting the Output

The simulation produces a final plot with six key panels. Focus on trends and ratios, not absolute magnitudes of value, which are nominal.

*   **AEC by Region (Spatial):** Shows the Capital Adequacy Ratio. Healthy systems typically see regions converge into a stable band (e.g., 0.6-0.8). Dips followed by recovery show the AMPK-like gate is working.
*   **GDP (Value Added) by Region (Spatial):** The aggregate flow of value added. Look for sustained growth, stability, or decline. The absolute scale is nominal and depends on initial prices.
*   **Exergy μ and Sink λ (means):** The shadow prices. A price that is persistently elevated above its floor indicates a binding constraint. In many scenarios, these will remain low, which is expected behavior unless a constraint is intentionally stressed.
*   **Sink Utilization (Spatial):** Shows how close each region's pollutant stock is to its environmental capacity limit (1.0). A key diagnostic for decoupling is a flat or declining curve here while the GDP curve is rising.
*   **GDP per Capita by Region (Spatial):** Shows value added per person. This metric provides insight into regional productivity and living standards, factoring in demographic changes.
*   **Exergy Productivity & Sink Intensity (means):** These are the key decoupling metrics.
    *   **Exergy Productivity (GDP / ATP Minted):** A rising curve indicates the economy is generating more value per unit of primary resource consumed.
    *   **Sink Intensity (Emissions / GDP):** A falling curve indicates the economy is generating fewer negative externalities per unit of value created.

## Built-in Hypothesis Scenarios

The `configs/` directory contains scenarios designed to test specific hypotheses:

*   `H1_baseline_stability.yaml`: Control case with moderate parameters.
*   `H2_energy_scarcity.yaml`: Tests the effect of a primary energy bottleneck.
*   `H3_environmental_collapse.yaml`: Tests an overshoot-and-collapse dynamic with tight sinks.
*   `H4_techno_optimism.yaml`: Explores if rapid innovation can overcome physical limits.
*   `H5_deglobalization.yaml`: Models reduced trade connectivity to test resilience.
*   `H6_financial_instability.yaml`: Weakens macroprudential rules to induce boom-bust cycles.
*   `H7-H9*.yaml`: Iterative attempts to achieve growth under environmental pressure.
*   `H10_decouple_and_grow.yaml`: The recommended stress-test scenario for the stabilized engine. It models a very hostile environment but equips the economy with strong innovation and investment to demonstrate "decoupling" (rising GDP with flat or falling environmental impact).

## License and Citation

This project is licensed under the MIT License. If you use ATP-Economy in a publication, please cite this repository and include the configuration file and code revision used to ensure reproducibility.
```
```configs/H10_decouple_and_grow.yaml
# configs/H10_decouple_and_grow.yaml
# Tuning goals:
# - Prevent μ runaway; cap μ to the scale of price×intensity so affinity stays responsive.
# - Keep λ responsive around a moderate utilization target.
# - Use per-step sink‑budget gating (assumes patched services) and last‑step‑need recharging.

runtime:
  steps: 20000
  plot: "final"
  save_fig: "H10_decouple_and_grow.png"
  dpi: 180
  style: "seaborn-v0_8"
  save_metrics: "H10_decouple_and_grow.npz"
  seed: 123

sizes:
  R: 16
  G: 24
  J: 12
  N: 200000
  k_latent: 4

trade:
  k_neighbors: 8
  alpha_logistics_ex: 0.06
  alpha_logistics_sink: 0.0020

markets:
  tau: 0.12
  beta_aff: 2.4

time:
  demurrage: 0.010
  dt: 1.0

duals:
  eta_ex: 0.010
  eta_sink: 0.065
  util_target: 0.18
  mu_floor: 0.010
  mu_cap: 0.200        # hard cap to avoid affinity collapse
  lambda_floor: 0.040
  lambda_cap: 1.0e10
  mu0: 0.020
  lambda0: 0.070
  ema_ex: 0.90
  ema_sink: 0.92

scaling:
  gen_scale: 0.85
  storage_scale: 1.10
  sink_cap_scale: 0.04
  sink_intensity_scale: 10.0
  gen_sink_intensity_scale: 1.0
  gen_noise: 0.20

environment:
  sink_assim_rate: 1e-9

policy:
  aec_low: 0.82
  aec_high: 0.93
  ers_k: 9.0
  gate_min: 0.15
  gate_k: 20.0
  aec_init: 0.88

extraction:
  n_resources: 4
  k_ext: 0.25
  beta_ext: 3.0
  xi_ext0: 1.2
  sig_ext0: 0.9
  dep_alpha_xi: 1.2
  dep_alpha_sig: 1.5
  reserves_scale: 8000000.0

demography:
  pop_init_scale: 1200000.0
  working_age: 18
  retirement_age: 65
  female_share: 0.50
  adult_gomp_alpha: 4.2e-5
  adult_gomp_beta: 0.085
  adult_makeham_lambda: 5.0e-4
  imr_base: 0.025
  u5_child_base: 0.0010
  youth_base: 2.0e-4
  eta_neonatal: 2.5
  eta_child: 2.0
  eta_adult: 1.0
  mort_sink_mult: 0.6
  fert_theta_dev: 1.2
  fert_phi_rep: 0.5
  fert_theta_cyc: 0.8
  birth_endow_atp: 0.10
  birth_endow_fiat: 5.0

inheritance:
  inherit_conc: 2.0
  inherit_frac_on_death: 0.90

behavior:
  greed_tau_scale: 0.60
  save_base: 0.035
  save_greed_scale: 0.05
  invest_innov_base: 0.14
  invest_innov_greed_scale: 0.20
  invest_storage_base: 0.08
  invest_storage_greed_scale: 0.10

innovation:
  eta_innov: 0.0090
  innov_alpha: 0.85
  innov_spill: 0.0030
  innov_decay: 0.0005
  beta_xi: 1.20
  beta_sigma: 2.20
  beta_kcat: 0.70

investment:
  eta_storage: 0.00100
  storage_decay: 0.00010

engine:
  xi_floor: 1.0e-8
  sigma_floor: 1.0e-8
  softmax_temp_sigma: 0.40
  cap_innov_exergy_mult: 30.0
  cap_storage_exergy_mult: 20.0
  innov_I_cap: 1.0e9

migration:
  migration_rate_annual: 0.075
  migration_kappa: 1.25
```
```configs/H11_divergence_and_limits.yaml
# configs/H11_divergence_and_limits.yaml
# HYPOTHESIS 11: Breaking initial symmetry, disabling technology spillover, and
# activating the environmental stock constraint will lead to persistent regional divergence.

runtime:
  steps: 20000
  save_fig: "H11_divergence_and_limits.png"
  dpi: 180
  style: "seaborn-v0_8"
  save_metrics: "H11_divergence_and_limits.npz"
  seed: 123

sizes:
  R: 16
  G: 24
  J: 12
  N: 200000
  k_latent: 4

trade:
  # KEY CHANGE: Reduced trade connectivity to inhibit convergence.
  k_neighbors: 4
  alpha_logistics_ex: 0.06
  alpha_logistics_sink: 0.0020

markets:
  tau: 0.12
  beta_aff: 2.4

time:
  demurrage: 0.010
  dt: 1.0

duals:
  eta_ex: 0.010
  eta_sink: 0.065
  util_target: 0.80 # Target a higher stock utilization now that it's the main constraint
  mu_floor: 0.010
  mu_cap: 0.200
  lambda_floor: 0.040
  lambda_cap: 1.0e10
  mu0: 0.020
  lambda0: 0.070
  ema_ex: 0.90
  ema_sink: 0.92

scaling:
  gen_scale: 0.85
  storage_scale: 1.10
  sink_cap_scale: 0.08 # Slightly larger sink to allow stock dynamics to play out
  sink_intensity_scale: 10.0
  gen_sink_intensity_scale: 1.0
  gen_noise: 0.20

environment:
  # KEY CHANGE: Set a meaningful assimilation rate. This now has a direct effect.
  sink_assim_rate: 0.005

policy:
  aec_low: 0.82
  aec_high: 0.93
  ers_k: 9.0
  gate_min: 0.15
  gate_k: 20.0
  aec_init: 0.88

extraction:
  n_resources: 4
  k_ext: 0.25
  beta_ext: 3.0
  xi_ext0: 1.2
  sig_ext0: 0.9
  dep_alpha_xi: 1.2
  dep_alpha_sig: 1.5
  reserves_scale: 8000000.0

demography:
  pop_init_scale: 1200000.0
  working_age: 18
  retirement_age: 65
  female_share: 0.50
  adult_gomp_alpha: 4.2e-5
  adult_gomp_beta: 0.085
  adult_makeham_lambda: 5.0e-4
  imr_base: 0.025
  u5_child_base: 0.0010
  youth_base: 2.0e-4
  eta_neonatal: 2.5
  eta_child: 2.0
  eta_adult: 1.0
  mort_sink_mult: 0.6
  fert_theta_dev: 1.2
  fert_phi_rep: 0.5
  fert_theta_cyc: 0.8
  birth_endow_atp: 0.10
  birth_endow_fiat: 5.0

inheritance:
  inherit_conc: 2.0
  inherit_frac_on_death: 0.90

behavior:
  greed_tau_scale: 0.60
  save_base: 0.035
  save_greed_scale: 0.05
  invest_innov_base: 0.14
  invest_innov_greed_scale: 0.20
  invest_storage_base: 0.08
  invest_storage_greed_scale: 0.10

innovation:
  # KEY CHANGE: Disabled technology spillover to enforce regional isolation.
  eta_innov: 0.0090
  innov_alpha: 0.85
  innov_spill: 0.0
  innov_decay: 0.0005
  beta_xi: 1.20
  beta_sigma: 2.20
  beta_kcat: 0.70

investment:
  eta_storage: 0.00100
  storage_decay: 0.00010

engine:
  xi_floor: 1.0e-8
  sigma_floor: 1.0e-8
  softmax_temp_sigma: 0.40
  cap_innov_exergy_mult: 30.0
  cap_storage_exergy_mult: 20.0
  innov_I_cap: 1.0e9

migration:
  migration_rate_annual: 0.0
  migration_kappa: 1.25

fx:
  fx_spread_fiat: 0.02
  fx_spread_crypto: 0.05
  fx_cap_mult: 12.0
  fx_convert_base_frac: 0.20
  fx_convert_greed_scale: 0.25
```
```configs/H1_baseline_stability.yaml
# configs/H1_baseline_stability.yaml
# HYPOTHESIS 1: A well-regulated biophysical economy can achieve a stable, modest-growth steady state.
# This is the control group for all other experiments.

runtime:
  steps: 20000
  plot: "final"
  save_fig: "H1_baseline_stability.png"
  dpi: 180
  style: "seaborn-v0_8"
  save_metrics: "H1_baseline_stability.npz"
  seed: 0

sizes:
  R: 16
  G: 24
  J: 12
  N: 200000
  k_latent: 4

trade:
  k_neighbors: 8
  alpha_logistics_ex: 0.08
  alpha_logistics_sink: 0.005

markets:
  tau: 0.15
  beta_aff: 2.0

time:
  demurrage: 0.008
  dt: 1.0

duals:
  eta_ex: 0.01
  eta_sink: 0.012
  util_target: 0.50
  mu_floor: 0.005
  mu_cap: 1000000.0
  lambda_floor: 0.02
  lambda_cap: 1000000.0
  mu0: 0.02
  lambda0: 0.05
  ema_ex: 0.90
  ema_sink: 0.90

scaling:
  gen_scale: 0.50
  storage_scale: 0.45
  sink_cap_scale: 0.15
  sink_intensity_scale: 4.0
  gen_noise: 0.25

environment:
  sink_assim_rate: 0.01

policy:
  aec_low: 0.80
  aec_high: 0.92
  ers_k: 6.0
  gate_min: 0.08
  gate_k: 14.0
  aec_init: 0.86

extraction:
  n_resources: 4
  k_ext: 0.25
  beta_ext: 3.0
  xi_ext0: 1.2
  sig_ext0: 0.7
  dep_alpha_xi: 1.1
  dep_alpha_sig: 1.3
  reserves_scale: 8000000.0

demography:
  pop_init_scale: 1200000.0
  birth_base: 0.012
  death_base: 0.010
  aec_birth_center: 0.82
  aec_death_center: 0.50
  birth_k: 6.0
  death_k: 8.0
  birth_endow_atp: 0.10
  birth_endow_fiat: 5.0

inheritance:
  inherit_conc: 2.0
  inherit_frac_on_death: 0.90

behavior:
  greed_tau_scale: 0.50
  save_base: 0.06
  save_greed_scale: 0.10
  invest_innov_base: 0.04
  invest_innov_greed_scale: 0.06
  invest_storage_base: 0.02
  invest_storage_greed_scale: 0.04

innovation:
  eta_innov: 0.0012
  innov_alpha: 0.9
  innov_spill: 0.0015
  innov_decay: 0.0008
  beta_xi: 0.45
  beta_sigma: 0.55
  beta_kcat: 0.35

investment:
  eta_storage: 0.00015
  storage_decay: 0.0002
```
```configs/H2_energy_scarcity.yaml
# configs/H2_energy_scarcity.yaml
# HYPOTHESIS 2: Chronic primary resource scarcity is the ultimate constraint, inducing a permanent liquidity trap.

runtime:
  steps: 20000
  plot: "final"
  save_fig: "H2_energy_scarcity.png"
  dpi: 180
  style: "seaborn-v0_8"
  save_metrics: "H2_energy_scarcity.npz"
  seed: 0

sizes:
  R: 16
  G: 24
  J: 12
  N: 200000
  k_latent: 4

trade:
  k_neighbors: 8
  alpha_logistics_ex: 0.08
  alpha_logistics_sink: 0.005

markets:
  tau: 0.15
  beta_aff: 2.0

time:
  demurrage: 0.008
  dt: 1.0

duals:
  eta_ex: 0.01
  eta_sink: 0.012
  util_target: 0.50
  mu_floor: 0.005
  mu_cap: 1000000.0
  lambda_floor: 0.02
  lambda_cap: 1000000.0
  mu0: 0.02
  lambda0: 0.05
  ema_ex: 0.90
  ema_sink: 0.90

scaling:
  # KEY CHANGE: Drastically reduced energy availability
  gen_scale: 0.15
  storage_scale: 0.10
  # ---
  sink_cap_scale: 0.15
  sink_intensity_scale: 4.0
  gen_noise: 0.25

environment:
  sink_assim_rate: 0.01

policy:
  aec_low: 0.80
  aec_high: 0.92
  ers_k: 6.0
  gate_min: 0.08
  gate_k: 14.0
  aec_init: 0.86

extraction:
  n_resources: 4
  k_ext: 0.25
  beta_ext: 3.0
  xi_ext0: 1.2
  sig_ext0: 0.7
  dep_alpha_xi: 1.1
  dep_alpha_sig: 1.3
  reserves_scale: 8000000.0

demography:
  pop_init_scale: 1200000.0
  birth_base: 0.012
  death_base: 0.010
  aec_birth_center: 0.82
  aec_death_center: 0.50
  birth_k: 6.0
  death_k: 8.0
  birth_endow_atp: 0.10
  birth_endow_fiat: 5.0

inheritance:
  inherit_conc: 2.0
  inherit_frac_on_death: 0.90

behavior:
  greed_tau_scale: 0.50
  save_base: 0.06
  save_greed_scale: 0.10
  invest_innov_base: 0.04
  invest_innov_greed_scale: 0.06
  invest_storage_base: 0.02
  invest_storage_greed_scale: 0.04

innovation:
  eta_innov: 0.0012
  innov_alpha: 0.9
  innov_spill: 0.0015
  innov_decay: 0.0008
  beta_xi: 0.45
  beta_sigma: 0.55
  beta_kcat: 0.35

investment:
  eta_storage: 0.00015
  storage_decay: 0.0002
```
```configs/H3_environmental_collapse.yaml
# configs/H3_environmental_collapse.yaml
# HYPOTHESIS 3: Tight environmental limits with abundant energy will cause an overshoot-and-collapse dynamic.

runtime:
  steps: 20000
  plot: "final"
  save_fig: "H3_environmental_collapse.png"
  dpi: 180
  style: "seaborn-v0_8"
  save_metrics: "H3_environmental_collapse.npz"
  seed: 0

sizes:
  R: 16
  G: 24
  J: 12
  N: 200000
  k_latent: 4

trade:
  k_neighbors: 8
  alpha_logistics_ex: 0.08
  alpha_logistics_sink: 0.005

markets:
  tau: 0.15
  beta_aff: 2.0

time:
  demurrage: 0.008
  dt: 1.0

duals:
  eta_ex: 0.01
  eta_sink: 0.012
  util_target: 0.50
  mu_floor: 0.005
  mu_cap: 1000000.0
  lambda_floor: 0.02
  lambda_cap: 1000000.0
  mu0: 0.02
  lambda0: 0.05
  ema_ex: 0.90
  ema_sink: 0.90

scaling:
  # KEY CHANGE: Abundant energy, tight environmental sink
  gen_scale: 0.90
  storage_scale: 0.80
  sink_cap_scale: 0.04
  # ---
  sink_intensity_scale: 4.0
  gen_noise: 0.25

environment:
  # KEY CHANGE: Slower environmental regeneration
  sink_assim_rate: 0.005

policy:
  aec_low: 0.80
  aec_high: 0.92
  ers_k: 6.0
  gate_min: 0.08
  gate_k: 14.0
  aec_init: 0.86

extraction:
  n_resources: 4
  k_ext: 0.25
  beta_ext: 3.0
  xi_ext0: 1.2
  sig_ext0: 0.7
  dep_alpha_xi: 1.1
  dep_alpha_sig: 1.3
  reserves_scale: 8000000.0

demography:
  pop_init_scale: 1200000.0
  birth_base: 0.012
  death_base: 0.010
  aec_birth_center: 0.82
  aec_death_center: 0.50
  birth_k: 6.0
  death_k: 8.0
  birth_endow_atp: 0.10
  birth_endow_fiat: 5.0

inheritance:
  inherit_conc: 2.0
  inherit_frac_on_death: 0.90

behavior:
  greed_tau_scale: 0.50
  save_base: 0.06
  save_greed_scale: 0.10
  invest_innov_base: 0.04
  invest_innov_greed_scale: 0.06
  invest_storage_base: 0.02
  invest_storage_greed_scale: 0.04

innovation:
  eta_innov: 0.0012
  innov_alpha: 0.9
  innov_spill: 0.0015
  innov_decay: 0.0008
  beta_xi: 0.45
  beta_sigma: 0.55
  beta_kcat: 0.35

investment:
  eta_storage: 0.00015
  storage_decay: 0.0002
```
```configs/H4_techno_optimism.yaml
# configs/H4_techno_optimism.yaml
# HYPOTHESIS 4: Rapid technological innovation can overcome physical limits, enabling sustained growth.

runtime:
  steps: 20000
  plot: "final"
  save_fig: "H4_techno_optimism.png"
  dpi: 180
  style: "seaborn-v0_8"
  save_metrics: "H4_techno_optimism.npz"
  seed: 0

sizes:
  R: 16
  G: 24
  J: 12
  N: 200000
  k_latent: 4

trade:
  k_neighbors: 8
  alpha_logistics_ex: 0.08
  alpha_logistics_sink: 0.005

markets:
  tau: 0.15
  beta_aff: 2.0

time:
  demurrage: 0.008
  dt: 1.0

duals:
  eta_ex: 0.01
  eta_sink: 0.012
  util_target: 0.50
  mu_floor: 0.005
  mu_cap: 1000000.0
  lambda_floor: 0.02
  lambda_cap: 1000000.0
  mu0: 0.02
  lambda0: 0.05
  ema_ex: 0.90
  ema_sink: 0.90

scaling:
  gen_scale: 0.50
  storage_scale: 0.45
  sink_cap_scale: 0.15
  sink_intensity_scale: 4.0
  gen_noise: 0.25

environment:
  sink_assim_rate: 0.01

policy:
  aec_low: 0.80
  aec_high: 0.92
  ers_k: 6.0
  gate_min: 0.08
  gate_k: 14.0
  aec_init: 0.86

extraction:
  n_resources: 4
  k_ext: 0.25
  beta_ext: 3.0
  xi_ext0: 1.2
  sig_ext0: 0.7
  dep_alpha_xi: 1.1
  dep_alpha_sig: 1.3
  reserves_scale: 8000000.0

demography:
  pop_init_scale: 1200000.0
  birth_base: 0.012
  death_base: 0.010
  aec_birth_center: 0.82
  aec_death_center: 0.50
  birth_k: 6.0
  death_k: 8.0
  birth_endow_atp: 0.10
  birth_endow_fiat: 5.0

inheritance:
  inherit_conc: 2.0
  inherit_frac_on_death: 0.90

behavior:
  greed_tau_scale: 0.50
  save_base: 0.06
  save_greed_scale: 0.10
  invest_innov_base: 0.04
  invest_innov_greed_scale: 0.06
  invest_storage_base: 0.02
  invest_storage_greed_scale: 0.04

innovation:
  # KEY CHANGE: Innovation is much more effective
  eta_innov: 0.0030
  innov_alpha: 0.9
  innov_spill: 0.0015
  innov_decay: 0.0008
  beta_xi: 0.75
  beta_sigma: 0.85
  beta_kcat: 0.50

investment:
  eta_storage: 0.00015
  storage_decay: 0.0002
```
```configs/H5_deglobalization.yaml
# configs/H5_deglobalization.yaml
# HYPOTHESIS 5: Restricting trade will increase inequality and reduce global resilience.

runtime:
  steps: 20000
  plot: "final"
  save_fig: "H5_deglobalization.png"
  dpi: 180
  style: "seaborn-v0_8"
  save_metrics: "H5_deglobalization.npz"
  seed: 0

sizes:
  R: 16
  G: 24
  J: 12
  N: 200000
  k_latent: 4

trade:
  # KEY CHANGE: Drastically reduced trade connectivity
  k_neighbors: 2
  alpha_logistics_ex: 0.08
  alpha_logistics_sink: 0.005

markets:
  tau: 0.15
  beta_aff: 2.0

time:
  demurrage: 0.008
  dt: 1.0

duals:
  eta_ex: 0.01
  eta_sink: 0.012
  util_target: 0.50
  mu_floor: 0.005
  mu_cap: 1000000.0
  lambda_floor: 0.02
  lambda_cap: 1000000.0
  mu0: 0.02
  lambda0: 0.05
  ema_ex: 0.90
  ema_sink: 0.90

scaling:
  gen_scale: 0.50
  storage_scale: 0.45
  sink_cap_scale: 0.15
  sink_intensity_scale: 4.0
  gen_noise: 0.25

environment:
  sink_assim_rate: 0.01

policy:
  aec_low: 0.80
  aec_high: 0.92
  ers_k: 6.0
  gate_min: 0.08
  gate_k: 14.0
  aec_init: 0.86

extraction:
  n_resources: 4
  k_ext: 0.25
  beta_ext: 3.0
  xi_ext0: 1.2
  sig_ext0: 0.7
  dep_alpha_xi: 1.1
  dep_alpha_sig: 1.3
  reserves_scale: 8000000.0

demography:
  pop_init_scale: 1200000.0
  birth_base: 0.012
  death_base: 0.010
  aec_birth_center: 0.82
  aec_death_center: 0.50
  birth_k: 6.0
  death_k: 8.0
  birth_endow_atp: 0.10
  birth_endow_fiat: 5.0

inheritance:
  inherit_conc: 2.0
  inherit_frac_on_death: 0.90

behavior:
  greed_tau_scale: 0.50
  save_base: 0.06
  save_greed_scale: 0.10
  invest_innov_base: 0.04
  invest_innov_greed_scale: 0.06
  invest_storage_base: 0.02
  invest_storage_greed_scale: 0.04

innovation:
  eta_innov: 0.0012
  innov_alpha: 0.9
  innov_spill: 0.0015
  innov_decay: 0.0008
  beta_xi: 0.45
  beta_sigma: 0.55
  beta_kcat: 0.35

investment:
  eta_storage: 0.00015
  storage_decay: 0.0002
```
```configs/H6_financial_instability.yaml
# configs/H6_financial_instability.yaml
# HYPOTHESIS 6: Weak financial regulation will cause Minsky-like boom-bust cycles, even with abundant resources.

runtime:
  steps: 20000
  plot: "final"
  save_fig: "H6_financial_instability.png"
  dpi: 180
  style: "seaborn-v0_8"
  save_metrics: "H6_financial_instability.npz"
  seed: 0

sizes:
  R: 16
  G: 24
  J: 12
  N: 200000
  k_latent: 4

trade:
  k_neighbors: 8
  alpha_logistics_ex: 0.08
  alpha_logistics_sink: 0.005

markets:
  tau: 0.15
  beta_aff: 2.0

time:
  demurrage: 0.008
  dt: 1.0

duals:
  eta_ex: 0.01
  eta_sink: 0.012
  util_target: 0.50
  mu_floor: 0.005
  mu_cap: 1000000.0
  lambda_floor: 0.02
  lambda_cap: 1000000.0
  mu0: 0.02
  lambda0: 0.05
  ema_ex: 0.90
  ema_sink: 0.90

scaling:
  # KEY CHANGE: Abundant energy to ensure constraint is financial
  gen_scale: 0.80
  storage_scale: 0.60
  # ---
  sink_cap_scale: 0.15
  sink_intensity_scale: 4.0
  gen_noise: 0.25

environment:
  sink_assim_rate: 0.01

policy:
  # KEY CHANGE: Weakened macroprudential policy
  aec_low: 0.60
  aec_high: 0.98
  ers_k: 6.0
  gate_min: 0.08
  gate_k: 4.0
  # ---
  aec_init: 0.86

extraction:
  n_resources: 4
  k_ext: 0.25
  beta_ext: 3.0
  xi_ext0: 1.2
  sig_ext0: 0.7
  dep_alpha_xi: 1.1
  dep_alpha_sig: 1.3
  reserves_scale: 8000000.0

demography:
  pop_init_scale: 1200000.0
  birth_base: 0.012
  death_base: 0.010
  aec_birth_center: 0.82
  aec_death_center: 0.50
  birth_k: 6.0
  death_k: 8.0
  birth_endow_atp: 0.10
  birth_endow_fiat: 5.0

inheritance:
  inherit_conc: 2.0
  inherit_frac_on_death: 0.90

behavior:
  greed_tau_scale: 0.50
  save_base: 0.06
  save_greed_scale: 0.10
  invest_innov_base: 0.04
  invest_innov_greed_scale: 0.06
  invest_storage_base: 0.02
  invest_storage_greed_scale: 0.04

innovation:
  eta_innov: 0.0012
  innov_alpha: 0.9
  innov_spill: 0.0015
  innov_decay: 0.0008
  beta_xi: 0.45
  beta_sigma: 0.55
  beta_kcat: 0.35

investment:
  eta_storage: 0.00015
  storage_decay: 0.0002
```
```configs/H7_pressure_agnostic_growth.yaml
# configs/H7_pressure_agnostic_growth.yaml
# GOAL: An economy that thrives even under severe environmental pressure.
# Strategy: Make environmental intensity (σ) melt quickly via innovation, keep controllers fast,
# build storage, and decarbonize logistics. Test this with tight sinks.

runtime:
  steps: 20000
  plot: "final"
  save_fig: "H7_pressure_agnostic_growth.png"
  dpi: 180
  style: "seaborn-v0_8"
  save_metrics: "H7_pressure_agnostic_growth.npz"
  seed: 0

sizes:
  R: 16
  G: 24
  J: 12
  N: 200000
  k_latent: 4

trade:
  # More connectivity + cleaner logistics (represents rapid zero-emission transport adoption)
  k_neighbors: 12
  alpha_logistics_ex: 0.08
  alpha_logistics_sink: 0.003

markets:
  # Slightly tighter price sensitivity so agents pivot faster when potentials move
  tau: 0.12
  beta_aff: 2.2

time:
  # A bit more demurrage to keep ATP circulating
  demurrage: 0.012
  dt: 1.0

duals:
  # Faster, firmer controllers with modest smoothing
  eta_ex: 0.015
  eta_sink: 0.060
  util_target: 0.30
  mu_floor: 0.005
  mu_cap: 1000000.0
  lambda_floor: 0.020
  lambda_cap: 1000000.0
  mu0: 0.025
  lambda0: 0.050
  ema_ex: 0.85
  ema_sink: 0.85

scaling:
  # Hostile environment assumptions to test robustness
  gen_scale: 0.60         # adequate but not lavish primary energy
  storage_scale: 0.70     # strong starting storage
  sink_cap_scale: 0.04    # tight environmental capacity (like H3)
  sink_intensity_scale: 6.0  # production is initially dirty
  gen_noise: 0.25

environment:
  sink_assim_rate: 0.005  # slow natural regeneration (like H3)

policy:
  # Keep the circuit-breaker vigilant and active
  aec_low: 0.82
  aec_high: 0.93
  ers_k: 8.0
  gate_min: 0.12
  gate_k: 18.0
  aec_init: 0.88

extraction:
  n_resources: 4
  k_ext: 0.25
  beta_ext: 3.0
  xi_ext0: 1.2
  sig_ext0: 0.8
  dep_alpha_xi: 1.2
  dep_alpha_sig: 1.4
  reserves_scale: 8000000.0

demography:
  pop_init_scale: 1200000.0
  birth_base: 0.012
  death_base: 0.010
  aec_birth_center: 0.82
  aec_death_center: 0.50
  birth_k: 6.0
  death_k: 8.0
  birth_endow_atp: 0.10
  birth_endow_fiat: 5.0

inheritance:
  inherit_conc: 2.0
  inherit_frac_on_death: 0.90

behavior:
  # Push capital into innovation and storage, keep savings moderate
  greed_tau_scale: 0.55
  save_base: 0.05
  save_greed_scale: 0.08
  invest_innov_base: 0.08
  invest_innov_greed_scale: 0.12
  invest_storage_base: 0.04
  invest_storage_greed_scale: 0.06

innovation:
  # Make σ (and ξ) collapse rapidly with T; throughput gets an extra lift
  eta_innov: 0.0065
  innov_alpha: 0.85
  innov_spill: 0.0030
  innov_decay: 0.0006
  beta_xi: 1.05
  beta_sigma: 1.55
  beta_kcat: 0.65

investment:
  # Scale storage faster and keep it from decaying too quickly
  eta_storage: 0.00060
  storage_decay: 0.00015
```
```configs/H8_env_agnostic_thriving.yaml
# configs/H8_env_agnostic_thriving.yaml
# GOAL: Thrive even when environmental pressure is severe.
# METHOD: Make sinks bind early (so λ rises), then melt σ fast via heavy innovation and
# storage build-out; keep λ responsive; decarbonize logistics; keep ATP circulating.

runtime:
  steps: 20000
  plot: "final"
  save_fig: "H8_env_agnostic_thriving.png"
  dpi: 180
  style: "seaborn-v0_8"
  save_metrics: "H8_env_agnostic_thriving.npz"
  seed: 0

sizes:
  R: 16
  G: 24
  J: 12
  N: 200000
  k_latent: 4

trade:
  # Keep globalization as a resilience asset; cleaner transport
  k_neighbors: 12
  alpha_logistics_ex: 0.08
  alpha_logistics_sink: 0.002

markets:
  # Faster pivot to potentials
  tau: 0.12
  beta_aff: 2.4

time:
  # Keep ATP moving but not punitive
  demurrage: 0.010
  dt: 1.0

duals:
  # Make λ react strongly to pressure; modest smoothing
  eta_ex: 0.015
  eta_sink: 0.090
  util_target: 0.15
  mu_floor: 0.005
  mu_cap: 1000000.0
  lambda_floor: 0.030
  lambda_cap: 1000000.0
  mu0: 0.030
  lambda0: 0.060
  ema_ex: 0.85
  ema_sink: 0.85

scaling:
  # Hostile sinks; adequate energy; big storage
  gen_scale: 0.70
  storage_scale: 0.90
  sink_cap_scale: 0.03
  sink_intensity_scale: 10.0
  gen_noise: 0.20

environment:
  sink_assim_rate: 0.003

policy:
  # Vigilant circuit-breaker; let healthy regions run
  aec_low: 0.80
  aec_high: 0.93
  ers_k: 10.0
  gate_min: 0.15
  gate_k: 20.0
  aec_init: 0.88

extraction:
  n_resources: 4
  k_ext: 0.25
  beta_ext: 3.0
  xi_ext0: 1.2
  sig_ext0: 0.9
  dep_alpha_xi: 1.2
  dep_alpha_sig: 1.5
  reserves_scale: 8000000.0

demography:
  pop_init_scale: 1200000.0
  birth_base: 0.012
  death_base: 0.010
  aec_birth_center: 0.82
  aec_death_center: 0.50
  birth_k: 6.0
  death_k: 8.0
  birth_endow_atp: 0.10
  birth_endow_fiat: 5.0

inheritance:
  inherit_conc: 2.0
  inherit_frac_on_death: 0.90

behavior:
  # Push budgets into innovation & storage; keep savings modest
  greed_tau_scale: 0.60
  save_base: 0.04
  save_greed_scale: 0.06
  invest_innov_base: 0.12
  invest_innov_greed_scale: 0.16
  invest_storage_base: 0.06
  invest_storage_greed_scale: 0.08

innovation:
  # Rapid σ and ξ decay; higher catalytic throughput
  eta_innov: 0.0090
  innov_alpha: 0.85
  innov_spill: 0.0035
  innov_decay: 0.0005
  beta_xi: 1.20
  beta_sigma: 2.00
  beta_kcat: 0.70

investment:
  # Build storage quickly; keep it alive
  eta_storage: 0.00100
  storage_decay: 0.00010
```
```configs/H9_extreme_pressure_decoupled.yaml
# configs/H9_extreme_pressure_decoupled.yaml
# PURPOSE: Robust, decoupled growth under severe environmental pressure.
# IDEA: Early, firm λ → rapid σ melt via heavy innovation → λ normalizes; storage keeps μ low.

runtime:
  steps: 20000
  plot: "final"
  save_fig: "H9_extreme_pressure_decoupled.png"
  dpi: 180
  style: "seaborn-v0_8"
  save_metrics: "H9_extreme_pressure_decoupled.npz"
  seed: 0

sizes:
  R: 16
  G: 24
  J: 12
  N: 200000
  k_latent: 4

trade:
  k_neighbors: 14
  alpha_logistics_ex: 0.06
  alpha_logistics_sink: 0.0015

markets:
  tau: 0.10
  beta_aff: 2.6

time:
  demurrage: 0.009
  dt: 1.0

duals:
  eta_ex: 0.015
  eta_sink: 0.065          # less gain than H8 to damp pulses
  util_target: 0.12        # controller “aims” lower utilization
  mu_floor: 0.005
  mu_cap: 1000000.0
  lambda_floor: 0.040
  lambda_cap: 1000000.0
  mu0: 0.030
  lambda0: 0.080
  ema_ex: 0.86
  ema_sink: 0.93           # more smoothing to prevent overshoot

scaling:
  gen_scale: 0.85          # ample primary energy
  storage_scale: 1.10      # strong initial storage
  sink_cap_scale: 0.02     # extremely tight sinks
  sink_intensity_scale: 12.0
  gen_noise: 0.20

environment:
  sink_assim_rate: 0.0025  # slow natural regeneration

policy:
  aec_low: 0.82
  aec_high: 0.93
  ers_k: 9.0
  gate_min: 0.17
  gate_k: 20.0
  aec_init: 0.88

extraction:
  n_resources: 4
  k_ext: 0.25
  beta_ext: 3.0
  xi_ext0: 1.2
  sig_ext0: 0.9
  dep_alpha_xi: 1.2
  dep_alpha_sig: 1.5
  reserves_scale: 8000000.0

demography:
  pop_init_scale: 1200000.0
  birth_base: 0.012
  death_base: 0.010
  aec_birth_center: 0.82
  aec_death_center: 0.50
  birth_k: 6.0
  death_k: 8.0
  birth_endow_atp: 0.10
  birth_endow_fiat: 5.0

inheritance:
  inherit_conc: 2.0
  inherit_frac_on_death: 0.90

behavior:
  greed_tau_scale: 0.60
  save_base: 0.035
  save_greed_scale: 0.06
  invest_innov_base: 0.15
  invest_innov_greed_scale: 0.20
  invest_storage_base: 0.08
  invest_storage_greed_scale: 0.10

innovation:
  eta_innov: 0.0120        # faster T accumulation
  innov_alpha: 0.82        # mild concavity
  innov_spill: 0.0040      # faster diffusion
  innov_decay: 0.0004
  beta_xi: 1.35
  beta_sigma: 2.40         # rapid σ melt
  beta_kcat: 0.75

investment:
  eta_storage: 0.00160     # storage builds quickly
  storage_decay: 0.00008
```
```src/atp_economy/__init__.py
__all__ = []
```
```src/atp_economy/cli.py
import os
import yaml
import typer
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Literal, Tuple, Dict, Any
from rich import print as rprint
from tqdm import trange
from torch.profiler import (
    profile,
    schedule,
    tensorboard_trace_handler,
    ProfilerActivity,
)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    torch.backends.cudnn.allow_tf32 = True

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore

    TB_AVAILABLE = True
except Exception:
    TB_AVAILABLE = False

    class SummaryWriter:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_histogram(self, *args, **kwargs):
            pass

        def flush(self):
            pass

        def close(self):
            pass


from .config import EconConfig
from .sim.model import ATPEconomy
from .utils.metrics import MetricsRecorder
from .vis.static import render_static
from .utils.tensor_utils import Device, DTYPE

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_config(
    config_path: Path,
    steps: Optional[int],
    save_fig: Optional[str],
    save_metrics: Optional[str],
) -> Tuple[EconConfig, Dict[str, Any]]:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    runtime_config = config_dict.get("runtime", {}) or {}
    if steps is not None:
        runtime_config["steps"] = steps
    if save_fig is not None:
        runtime_config["save_fig"] = save_fig
    if save_metrics is not None:
        runtime_config["save_metrics"] = save_metrics

    cfg = EconConfig.from_dict(config_dict)
    return cfg, runtime_config


@app.command("run")
def run(
    config_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, resolve_path=True
    ),
    steps: Optional[int] = typer.Option(None, "--steps", "-s"),
    save_fig: Optional[str] = typer.Option(None),
    save_metrics: Optional[str] = typer.Option(None),
    tb_logdir: Optional[str] = typer.Option(None, "--tb-logdir"),
):
    cfg, runtime_cfg = _load_config(config_path, steps, save_fig, save_metrics)
    rprint(
        f"[bold cyan]Running ATP-economy on device:[/bold cyan] {Device} with dtype [bold]float32[/bold]"
    )
    rprint(f"[bold cyan]Loading config from:[/bold cyan] {config_path}")

    run_steps = int(runtime_cfg.get("steps", 20000))
    save_fig_path = runtime_cfg.get("save_fig", "healthy_run.png")
    save_metrics_path = runtime_cfg.get("save_metrics", "healthy_metrics.npz")
    style = runtime_cfg.get("style", "seaborn-v0_8")
    dpi = int(runtime_cfg.get("dpi", 180))

    logging_enabled = tb_logdir is not None and TB_AVAILABLE
    writer = SummaryWriter(log_dir=tb_logdir) if logging_enabled else SummaryWriter()

    model = ATPEconomy(cfg)

    recorder = MetricsRecorder(
        keys=[
            "AEC_region",
            "GDP_proxy_region",
            "GDP_flow_region",
            "GDP_pc_region",
            "ATP_minted_region",
            "sink_utilization",
            "mu_ex",
            "lambda_sink",
            "population_region",
            "psr_region",
            "dependency_region",
            "exergy_productivity_region",
            "sink_intensity_region",
        ],
        maxlen=None,
        stride=1,
    )

    HIST_EVERY = 50

    pbar = trange(run_steps, desc="Simulating", leave=True)
    for t in pbar:
        metrics = model.step()
        recorder.record(metrics)

        aec_mean = float(np.mean(metrics["AEC_region"]))
        gdp_total = float(np.sum(metrics["GDP_flow_region"]))
        mu_mean = float(np.mean(metrics["mu_ex"]))
        lam_mean = float(np.mean(metrics["lambda_sink"]))
        sink_mean = float(np.mean(metrics["sink_utilization"]))
        minted_total = float(np.sum(metrics["ATP_minted_region"]))
        pop_total = float(np.sum(metrics["population_region"]))
        xp_mean = float(np.mean(metrics["exergy_productivity_region"]))
        si_mean = float(np.mean(metrics["sink_intensity_region"]))

        pbar.set_postfix(
            AEC=f"{aec_mean:.3f}",
            GDPf=f"{gdp_total:,.0f}",
            μ=f"{mu_mean:.3f}",
            λ=f"{lam_mean:.3f}",
            XP=f"{xp_mean:.3f}",
            SI=f"{si_mean:.3e}",
            Pop=f"{pop_total:,.0f}",
        )

        if logging_enabled:
            writer.add_scalar("AEC/mean", aec_mean, t)
            writer.add_scalar("GDP/flow_total", gdp_total, t)
            writer.add_scalar("Duals/mu_mean", mu_mean, t)
            writer.add_scalar("Duals/lambda_mean", lam_mean, t)
            writer.add_scalar("Sink/util_mean", sink_mean, t)
            writer.add_scalar("ATP/minted_total", minted_total, t)
            writer.add_scalar("Demography/pop_total", pop_total, t)
            writer.add_scalar("Efficiency/exergy_productivity_mean", xp_mean, t)
            writer.add_scalar("Environment/sink_intensity_mean", si_mean, t)

            if t % HIST_EVERY == 0:
                writer.add_histogram("AEC/by_region", metrics["AEC_region"], t)
                writer.add_histogram(
                    "GDP/flow_by_region", metrics["GDP_flow_region"], t
                )
                writer.add_histogram("GDP/pc_by_region", metrics["GDP_pc_region"], t)
                writer.add_histogram("Duals/mu_by_region", metrics["mu_ex"], t)
                writer.add_histogram(
                    "Duals/lambda_by_region", metrics["lambda_sink"], t
                )
                writer.add_histogram(
                    "Sink/util_by_region", metrics["sink_utilization"], t
                )
                writer.add_histogram(
                    "Demography/pop_by_region", metrics["population_region"], t
                )

    if logging_enabled:
        writer.flush()
        writer.close()

    hist = recorder.as_arrays()
    hist["pop_age_final"] = model.state.pop_age.detach().cpu().numpy()

    if save_metrics_path:
        np.savez_compressed(save_metrics_path, **hist)
        rprint(f"[green]Saved metrics ->[/green] {save_metrics_path}")

    render_static(hist, save_fig=save_fig_path, dpi=dpi, style=style)
    if save_fig_path:
        rprint(f"[green]Saved figure ->[/green] {save_fig_path}")


@app.command("profile")
def profile_run(
    steps: int = typer.Option(120, help="Total profiled steps (active)"),
    warmup: int = typer.Option(20, help="Warmup steps (not recorded)"),
    wait: int = typer.Option(5, help="Scheduler wait steps before warmup"),
    trace_dir: str = typer.Option("runs/prof", help="Output directory for traces"),
    activities: Literal["cpu", "cpu_cuda"] = typer.Option(
        "cpu_cuda", help="Profiler activities"
    ),
    R: int = typer.Option(16),
    G: int = typer.Option(24),
    J: int = typer.Option(12),
    N: int = typer.Option(200_000),
    seed: int = typer.Option(123),
):
    rprint(
        f"[bold cyan]Profiling ATP-economy on device:[/bold cyan] {Device} with dtype [bold]float32[/bold]"
    )

    cfg = EconConfig(R=R, G=G, J=J, N=N, seed=seed)
    model = ATPEconomy(cfg)

    acts = [ProfilerActivity.CPU]
    if activities == "cpu_cuda" and str(Device).startswith("cuda"):
        acts.append(ProfilerActivity.CUDA)

    sch = schedule(wait=wait, warmup=warmup, active=steps, repeat=1)
    os.makedirs(trace_dir, exist_ok=True)

    with profile(
        activities=acts,
        schedule=sch,
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        on_trace_ready=tensorboard_trace_handler(trace_dir),
    ) as prof:
        total = wait + warmup + steps
        pbar = trange(total, desc="Profiling")
        for _ in pbar:
            model.step()
            prof.step()

    rprint(f"[green]Trace written ->[/green] {trace_dir}")


if __name__ == "__main__":
    app()
```
```src/atp_economy/config.py
from dataclasses import dataclass, fields
from typing import Dict, Any


@dataclass
class EconConfig:
    # ------------- Sizes -------------
    R: int
    G: int
    J: int
    N: int

    # ------------- Markets -------------
    K_latent: int = 4
    tau: float = 0.15
    beta_aff: float = 2.0  # sensitivity to affinity (production responsiveness)

    # ------------- Trade -------------
    k_neighbors: int = 8
    alpha_logistics_ex: float = 0.08  # eATP cost per unit*distance for logistics
    alpha_logistics_sink: float = 0.005  # sink cost per unit*distance for logistics

    # ------------- Time -------------
    demurrage: float = 0.01
    dt: float = 1.0
    seed: int = 123

    # ------------- Duals (exergy μ, sink λ) -------------
    eta_ex: float = 1e-2
    eta_sink: float = 1e-2
    util_target: float = 0.5
    mu_floor: float = 5e-3
    mu_cap: float = 1e6
    lambda_floor: float = 2e-2
    lambda_cap: float = 1e6
    mu0: float = 2e-2
    lambda0: float = 5e-2
    ema_ex: float = 0.9
    ema_sink: float = 0.9

    # ------------- Environment -------------
    sink_assim_rate: float = 0.01

    # ------------- Scaling -------------
    gen_scale: float = 0.35
    storage_scale: float = 0.30
    sink_cap_scale: float = 0.10
    sink_intensity_scale: float = 5.0
    gen_noise: float = 0.30
    gen_sink_intensity_scale: float = 1.0  # NEW: scale for generation sink intensity

    # ------------- Policy (AEC/ERS) -------------
    aec_low: float = 0.78
    aec_high: float = 0.92
    ers_k: float = 6.0
    gate_min: float = 0.10
    gate_k: float = 12.0
    aec_init: float = 0.86

    # ------------- Extraction -------------
    n_resources: int = 4
    k_ext: float = 0.2
    beta_ext: float = 3.0
    xi_ext0: float = 1.0
    sig_ext0: float = 0.6
    dep_alpha_xi: float = 1.0
    dep_alpha_sig: float = 1.2
    reserves_scale: float = 5e6

    # ------------- Demography (legacy scalar fields kept for compatibility) -------------
    pop_init_scale: float = 1.0e6
    birth_base: float = 0.015
    death_base: float = 0.010
    aec_birth_center: float = 0.8
    aec_death_center: float = 0.5
    birth_k: float = 5.0
    death_k: float = 7.0
    birth_endow_atp: float = 0.2

    # ------------- Demography (age-structured model) -------------
    working_age: int = 18
    retirement_age: int = 65
    female_share: float = 0.50

    # Adult Gompertz–Makeham at H=1 (per-year hazard parameters)
    adult_gomp_alpha: float = 4.2e-5  # α
    adult_gomp_beta: float = 0.085  # β
    adult_makeham_lambda: float = 5.0e-4  # λ

    # Baseline child/infant hazards (per-year) near mid-development
    imr_base: float = 0.03  # infant (0–1y) hazard
    u5_child_base: float = 0.001  # ages 1–4
    youth_base: float = 2.0e-4  # ages 5–14

    # Health elasticities (larger = hazards fall faster as health improves)
    eta_neonatal: float = 2.5
    eta_child: float = 2.0
    eta_adult: float = 1.0
    mort_sink_mult: float = 0.5  # sink utilization penalty on hazards

    # Fertility multipliers
    fert_theta_dev: float = 1.0  # long-run decline with development
    fert_phi_rep: float = 0.4  # replacement/insurance elasticity to under-5 survival
    fert_theta_cyc: float = 0.8  # procyclical effect (births fall in recessions)

    # ------------- Inheritance -------------
    inherit_conc: float = 2.0
    inherit_frac_on_death: float = 0.9

    # ------------- Migration -------------
    migration_rate_annual: float = 0.0  # fraction of mobile cohort per year
    migration_kappa: float = 1.0  # distance cost exponent

    # ------------- Behavior -------------
    greed_tau_scale: float = 0.5
    save_base: float = 0.05
    save_greed_scale: float = 0.10
    invest_innov_base: float = 0.03
    invest_innov_greed_scale: float = 0.05
    invest_storage_base: float = 0.02
    invest_storage_greed_scale: float = 0.04

    # ------------- Innovation -------------
    eta_innov: float = 1.0e-3
    innov_alpha: float = 1.0
    innov_spill: float = 1.0e-3
    innov_decay: float = 1.0e-3
    beta_xi: float = 0.4
    beta_sigma: float = 0.5
    beta_kcat: float = 0.3

    # ------------- Investment (storage) -------------
    eta_storage: float = 1.0e-4
    storage_decay: float = 2.0e-4

    # ------------- Engine (stability & allocation) -------------
    xi_floor: float = 1.0e-12
    sigma_floor: float = 1.0e-12
    softmax_temp_sigma: float = 0.5
    cap_innov_exergy_mult: float = 50.0
    cap_storage_exergy_mult: float = 25.0
    innov_I_cap: float = 1.0e12  # safe cap for R&D increments

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EconConfig":
        """
        Strict structured loader.
        Recognized sections:
          sizes, trade, markets, time,
          duals, scaling, environment,
          policy, extraction, demography, inheritance,
          behavior, innovation, investment, engine, migration, runtime

        - Unknown keys are ignored.
        - Numeric strings like "1.0e9" are coerced to numbers.
        """
        allowed = [
            "sizes",
            "trade",
            "markets",
            "time",
            "duals",
            "scaling",
            "environment",
            "policy",
            "extraction",
            "demography",
            "inheritance",
            "behavior",
            "innovation",
            "investment",
            "engine",
            "migration",
            "runtime",
        ]
        known = {f.name for f in fields(cls)}
        args: Dict[str, Any] = {}

        for section in allowed:
            sec = config_dict.get(section)
            if isinstance(sec, dict):
                for k, v in sec.items():
                    if k in known:
                        args[k] = v

        # Pull seed from runtime.seed if provided
        rt = config_dict.get("runtime")
        if isinstance(rt, dict) and "seed" in rt:
            args["seed"] = rt["seed"]

        # Coerce numeric strings to numbers to prevent runtime type errors
        for f in fields(cls):
            name = f.name
            if name not in args:
                continue
            v = args[name]
            # float fields
            if f.type is float:
                if isinstance(v, str):
                    try:
                        args[name] = float(v)
                    except Exception:
                        pass
                elif isinstance(v, int):
                    args[name] = float(v)
            # int fields
            elif f.type is int:
                if isinstance(v, str):
                    try:
                        # allow scientific notation
                        args[name] = int(float(v))
                    except Exception:
                        pass
                elif isinstance(v, float):
                    args[name] = int(v)

        # Required sizes
        required_sizes = ["R", "G", "J", "N"]
        missing = [k for k in required_sizes if k not in args]
        if missing:
            raise ValueError(
                f"Missing required size fields in 'sizes' section: {missing}. "
                f"Example:
  sizes: {{ R: 16, G: 24, J: 12, N: 200000 }}"
            )

        return cls(**args)
```
```src/atp_economy/domain/__init__.py
# src/atp_economy/domain/__init__.py
__all__ = []
```
```src/atp_economy/domain/state.py
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
            order = torch.argsort(self.agent_region)
            counts = torch.bincount(self.agent_region, minlength=R)
            rowptr = torch.zeros(R + 1, device=Device, dtype=torch.long)
            rowptr[1:] = counts.cumsum(0)
        self.register_buffer("agent_order", order)
        self.register_buffer("rowptr", rowptr)

        with torch.no_grad():
            idx = self.agent_region
            atp_pool0 = torch.bincount(idx, weights=self.eATP, minlength=R)
            adp_pool0 = torch.bincount(idx, weights=self.eADP, minlength=R)
            amp_pool0 = torch.bincount(idx, weights=self.eAMP, minlength=R)
        self.register_buffer("pool_atp_R", atp_pool0)
        self.register_buffer("pool_adp_R", adp_pool0)
        self.register_buffer("pool_amp_R", amp_pool0)

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
```
```src/atp_economy/services/__init__.py
# src/atp_economy/services/__init__.py
__all__ = []
```
```src/atp_economy/services/agent_behavior.py
import torch
from ..domain.state import WorldState
from ..utils.tensor_utils import Device, DTYPE


def agent_budgets_and_demand(
    state: WorldState, bases: torch.Tensor, scales: torch.Tensor
):
    """
    Agent budgets and demand with numerically-stable choice and physically
    grounded investment caps. Budget is now purely energy-denominated.
    """
    cfg = state.cfg
    R, G = cfg.R, state.price.shape[0]
    eps = 1e-9

    # Per-agent temperature shaped by greed
    tau_i = cfg.tau * torch.exp(-cfg.greed_tau_scale * state.greed)  # [N]

    # Utilities and stable softmax
    theta = state.pref_theta  # [N,G]
    p_agent = state.price.T.index_select(0, state.agent_region)  # [N,G]
    util = theta - p_agent
    logits = torch.clamp(util / (tau_i.unsqueeze(1) + eps), -40.0, 40.0)

    outside = torch.full((logits.shape[0], 1), -5.0, device=Device, dtype=DTYPE)
    probs = torch.softmax(torch.cat([logits, outside], dim=1), dim=1)[:, :G]  # [N,G]

    # Wallet-based consumption budget proxy (purely energy-denominated)
    base_budget = 0.5 * state.eATP  # [N]

    # Age-structure consumption scaling (regional)
    cons_scale_r = getattr(state, "consump_scale_R", None)
    if cons_scale_r is None:
        cons_scale_r = state.population / (state.population0 + eps)
    cons_scale_i = cons_scale_r[state.agent_region]

    # Savings/investment propensities
    greed_expanded = state.greed[:, None]
    prop = bases + greed_expanded * scales
    prop.clamp_(0.0, 0.9)
    total_frac = prop.sum(dim=1, keepdim=True)
    prop *= torch.clamp(0.95 / (total_frac + eps), max=1.0)
    save_frac, innov_frac, storage_frac = prop.unbind(dim=1)

    # Consumption allocation
    cons_budget = (
        (1.0 - save_frac - innov_frac - storage_frac) * base_budget * cons_scale_i
    )
    spend = cons_budget.unsqueeze(1) * probs  # [N,G]
    spend_sorted = spend[state.agent_order]
    demand_R = torch.segment_reduce(
        data=spend_sorted, reduce="sum", offsets=state.rowptr, axis=0
    )

    # Regional aggregation of investment budgets
    innov_budget_sorted = (innov_frac * base_budget * cons_scale_i)[state.agent_order]
    innov_R_raw = torch.segment_reduce(
        data=innov_budget_sorted, reduce="sum", offsets=state.rowptr, axis=0
    )

    storage_budget_sorted = (storage_frac * base_budget * cons_scale_i)[
        state.agent_order
    ]
    storage_R_raw = torch.segment_reduce(
        data=storage_budget_sorted, reduce="sum", offsets=state.rowptr, axis=0
    )

    minted = torch.clamp(state.atp_minted_R, min=0.0)
    cap_innov_R = cfg.cap_innov_exergy_mult * (minted + 1.0)
    cap_storage_R = cfg.cap_storage_exergy_mult * (minted + 1.0)
    innov_R = torch.minimum(innov_R_raw, cap_innov_R)
    storage_budget_R = torch.minimum(storage_R_raw, cap_storage_R)

    # Innovation allocation weights from sigma_eff
    sigma_eff = torch.clamp(state.sigma_eff, min=cfg.sigma_floor)
    w = torch.softmax(sigma_eff / max(cfg.softmax_temp_sigma, 1e-6), dim=1)  # [R,J]
    innov_budget_RJ = innov_R.unsqueeze(1) * w

    return demand_R, innov_budget_RJ, storage_budget_R
```
```src/atp_economy/services/aggregation.py
# src/atp_economy/services/aggregation.py
import torch
from ..domain.state import WorldState


def compute_regional_summaries(state: WorldState) -> dict[str, torch.Tensor]:
    """
    Computes agent->region aggregations using fast reductions.
    """
    R = state.cfg.R
    idx = state.agent_region

    atp_pool = torch.bincount(idx, weights=state.eATP, minlength=R)
    adp_pool = torch.bincount(idx, weights=state.eADP, minlength=R)
    amp_pool = torch.bincount(idx, weights=state.eAMP, minlength=R)

    return {
        "atp_pool": atp_pool,
        "adp_pool": adp_pool,
        "amp_pool": amp_pool,
    }
```
```src/atp_economy/services/consumption.py
import torch
from ..domain.state import WorldState
from .settlement import settle_spend_book
from ..utils.tensor_utils import Device, DTYPE


def run_consumption(
    state: WorldState,
    demand_qty_R: torch.Tensor,
    atp_book_R: torch.Tensor,
    frac: float = 1.0,
) -> torch.Tensor:
    """
    Final-goods consumption gated by ATP, sink headroom, and per-step sink-flow budget.
    """
    cfg = state.cfg
    R = cfg.R
    eps = 1e-9

    F = getattr(state, "final_idx", None)
    if F is None or F.numel() == 0:
        return atp_book_R

    want_RF = torch.clamp_min(
        demand_qty_R[:, F] * max(0.0, min(1.0, frac)), 0.0
    )  # [R,F]
    have_RF = torch.clamp_min(state.inventory[:, F], 0.0)  # [R,F]
    cons_base = torch.minimum(want_RF, have_RF)  # [R,F]

    xi = state.xi_cons  # [F]
    sig = state.sigma_cons  # [F]
    atp_need_base = (cons_base * xi.unsqueeze(0)).sum(dim=1)  # [R]
    sink_emit_base = (cons_base * sig.unsqueeze(0)).sum(dim=1)  # [R]

    # Emission gating primitives
    sink_head = torch.clamp_min(state.sink_cap - state.sink_use - state.sink_use_R, 0.0)

    s_atp = torch.clamp(atp_book_R / (atp_need_base + eps), max=1.0)
    s_head = torch.clamp(sink_head / (sink_emit_base + eps), max=1.0)
    s = torch.minimum(s_atp, s_head)

    cons_RF = cons_base * s.unsqueeze(1)
    new_RF = have_RF - cons_RF
    state.inventory.data = state.inventory.data.index_copy(1, F, new_RF)

    atp_spend = (cons_RF * xi.unsqueeze(0)).sum(dim=1)  # [R]
    sink_emit = (cons_RF * sig.unsqueeze(0)).sum(dim=1)  # [R]

    _, atp_book_R = settle_spend_book(state, atp_spend, atp_book_R)

    state.emit_sink_R.data = state.emit_sink_R.data + sink_emit
    state.exergy_need_R.data = state.exergy_need_R.data + atp_spend
    state.sink_use_R.data = state.sink_use_R.data + sink_emit

    return atp_book_R
```
```src/atp_economy/services/demography.py
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

        # Statically traceable replacement for bincount
        w_sum_r_zeros = torch.zeros(R, device=w_raw.device, dtype=w_raw.dtype)
        w_sum_r = torch.index_add(w_sum_r_zeros, 0, region_idx, w_raw)
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
            removed_pool_r_zeros = torch.zeros(
                R, device=wallet.device, dtype=wallet.dtype
            )
            removed_pool_r = torch.index_add(
                removed_pool_r_zeros, 0, region_idx, removed_i
            )

            # Distribute to heirs
            heir_pool_r = removed_pool_r * inherit_frac
            heir_share_i = w_norm * heir_pool_r[region_idx]
            wallet.data.add_(heir_share_i)

            # Update regional summary pools (e.g., pool_atp_R)
            if pool is not None:
                net_loss_r = removed_pool_r - heir_pool_r
                pool.data.sub_(net_loss_r)

        births_total = births_surv

        # Statically traceable replacement for bincount
        ones_weights = torch.ones_like(region_idx, dtype=DTYPE)
        counts_r_zeros = torch.zeros(R, device=region_idx.device, dtype=DTYPE)
        counts_r = torch.index_add(counts_r_zeros, 0, region_idx, ones_weights)

        counts_safe = torch.clamp(counts_r, min=1.0)
        add_atp_i = cfg.birth_endow_atp * (births_total / counts_safe)[region_idx]
        state.eATP.data.add_(add_atp_i)
```
```src/atp_economy/services/energy_bank.py
import torch
from ..domain.state import WorldState
from ..utils.tensor_utils import DTYPE, Device


def run_recharging(
    state: WorldState, need_prev_R: torch.Tensor, adp_pool_R: torch.Tensor
):
    """
    ADP -> ATP recharging with storage discharge/charge.

    Policies:
    - Mint only to satisfy last-step exergy need (need_prev_R), never to fill sink headroom.
    - Gate emissions by remaining sink headroom.
    - Compile-safe: avoid clamp(min=tensor, max=float) signatures.
    """
    cfg = state.cfg
    R = cfg.R
    eps = 1e-9

    # Stochastic primary generation
    nz = cfg.gen_noise
    factor = torch.clamp(
        1.0 + (2 * torch.rand(R, device=Device, dtype=DTYPE) - 1.0) * nz, min=0.1
    )
    gen = state.gen_exergy * factor  # [R]

    # Cover backlog with storage; no discharge if gen >= need
    deficit = torch.relu(need_prev_R - gen)  # [R]
    discharge = torch.minimum(deficit / (state.eta_rt + eps), state.storage_soc)
    delivered_raw = gen + discharge * state.eta_rt  # [R]

    # Never deliver beyond last-step need
    delivered_need_limited = torch.minimum(
        delivered_raw, torch.clamp_min(need_prev_R, 0.0)
    )

    # Provisional generation emissions
    sink_gen_raw = delivered_need_limited * state.gen_sink_intensity  # [R]

    # Headroom gating within this step
    sink_head = torch.clamp_min(state.sink_cap - state.sink_use - state.sink_use_R, 0.0)
    s_head = torch.clamp(sink_head / (sink_gen_raw + eps), max=1.0)

    s_emit = s_head

    delivered = delivered_need_limited * s_emit
    sink_gen = sink_gen_raw * s_emit

    # Mint limited by ADP pool
    minted_R = torch.minimum(delivered, adp_pool_R)  # [R]
    state.atp_minted_R.data = minted_R

    # Surplus delivered (if any) -> charge storage within capacity (account for η)
    surplus = torch.relu(delivered - minted_R)
    free_cap = torch.clamp_min(state.storage_cap - state.storage_soc, 0.0)
    charge = torch.minimum(surplus / (state.eta_rt + eps), free_cap)

    # Update SoC with discharge/charge
    soc_new = torch.clamp_min(state.storage_soc + charge - discharge, 0.0)
    soc_new = torch.minimum(soc_new, state.storage_cap)
    state.storage_soc.data = soc_new

    # Book generation emissions for this step
    state.emit_sink_R.data = state.emit_sink_R.data + sink_gen
    state.sink_use_R.data = state.sink_use_R.data + sink_gen

    # Distribute minted ATP ∝ ADP within region
    share = torch.where(adp_pool_R > eps, minted_R / (adp_pool_R + eps), 0.0)
    delta_agent = state.eADP * share[state.agent_region]
    state.eATP.data = state.eATP.data + delta_agent
    state.eADP.data = state.eADP.data - delta_agent

    # Update pools exactly
    state.pool_atp_R.data = state.pool_atp_R.data + minted_R
    state.pool_adp_R.data = state.pool_adp_R.data - minted_R
```
```src/atp_economy/services/environment.py
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
```
```src/atp_economy/services/extraction.py
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
```
```src/atp_economy/services/innovation.py
# src/atp_economy/services/innovation.py
import torch
from ..domain.state import WorldState


def update_innovation_and_effects(state: WorldState, innov_budget_RJ: torch.Tensor):
    """
    IMEX/Patankar-like update of technology stocks T[r,j] and mapping to effective process params.

    Stability/realism additions:
      - Irreducible floors for xi_eff and sigma_eff (no process is literally zero-cost/zero-externality).
      - Cap on effective innovation increment to reflect finite absorptive capacity of R&D systems.
    """
    cfg = state.cfg
    R, J = cfg.R, state.S.shape[1]
    eps = 1e-9

    # Effective innovation effort (diminishing returns)
    I = torch.clamp(innov_budget_RJ, min=0.0)  # [R,J]
    I_eff = torch.pow(I + eps, cfg.innov_alpha)
    # Cap the increment to avoid runaway T updates
    I_eff = torch.clamp(I_eff, max=cfg.innov_I_cap)

    # Spillovers via neighbor averaging
    nbr = state.nbr_idx  # [R,k]
    k = nbr.shape[1]
    spill = cfg.innov_spill * (
        state.tech_T.index_select(0, nbr.reshape(-1)).reshape(R, k, J).mean(dim=1)
        - state.tech_T
    )

    T_num = state.tech_T + state.cfg.dt * (cfg.eta_innov * I_eff + spill)
    T_den = 1.0 + state.cfg.dt * cfg.innov_decay
    state.tech_T.data = torch.clamp(T_num / T_den, min=0.0)

    # Map to effective parameters with irreducible floors
    xi_eff = state.xi_base[None, :] * torch.exp(-cfg.beta_xi * state.tech_T)
    sigma_eff = state.sigma_base[None, :] * torch.exp(-cfg.beta_sigma * state.tech_T)

    state.xi_eff.data = torch.clamp(xi_eff, min=cfg.xi_floor)
    state.sigma_eff.data = torch.clamp(sigma_eff, min=cfg.sigma_floor)

    # Throughput catalyst (bounded by tanh)
    state.k_eff.data = state.k_base[None, :] * (
        1.0 + cfg.beta_kcat * torch.tanh(state.tech_T)
    )
```
```src/atp_economy/services/metrics_flow.py
# src/atp_economy/services/metrics_flow.py
import torch
from ..domain.state import WorldState


def value_added_production(state: WorldState, rate_RJ: torch.Tensor) -> torch.Tensor:
    """
    GDP (flow) from transformation activities as Value Added:
      VA_r = sum_j [ p_r•(outputs of j) - p_r•(intermediate inputs of j) ] * rate_{rj}
    where S[g,j] < 0 are inputs, > 0 are outputs.

    Args:
      rate_RJ: [R, J] realized reaction rates this step
    Returns:
      VA_R: [R] value added per region
    """
    S = state.S  # [G,J]
    p_RG = state.price.T  # [R,G]
    S_pos = torch.clamp(S, min=0.0)  # outputs
    S_neg = torch.clamp(-S, min=0.0)  # inputs

    # Revenue and intermediate cost per region j
    rev_RJ = p_RG @ S_pos  # [R,J]
    int_RJ = p_RG @ S_neg  # [R,J]

    VA_RJ = (rev_RJ - int_RJ) * torch.clamp(rate_RJ, min=0.0)
    return VA_RJ.sum(dim=1)  # [R]


def value_added_extraction(state: WorldState, q_RM: torch.Tensor) -> torch.Tensor:
    """
    Value added from extraction of M resource goods (no intermediate inputs tracked here).
    Args:
      q_RM: [R, M] extraction quantities by region and resource index
    Returns:
      VA_R: [R]
    """
    goods_idx = state.res_goods  # [M]
    p_RM = state.price.index_select(0, goods_idx).T  # [R, M]
    return (p_RM * torch.clamp(q_RM, min=0.0)).sum(dim=1)
```
```src/atp_economy/services/policy.py
# src/atp_economy/services/policy.py
import torch
from ..domain.state import WorldState
from ..config import EconConfig


def aec_by_region(
    atp_r: torch.Tensor, adp_r: torch.Tensor, amp_r: torch.Tensor
) -> torch.Tensor:
    """Computes AEC from pre-aggregated regional adenylate pools."""
    denom = atp_r + adp_r + amp_r + 1e-12
    return (atp_r + 0.5 * adp_r) / denom


def ers_demurrage_factors(cfg: EconConfig, aec_r: torch.Tensor) -> torch.Tensor:
    """Per-region demurrage multiplier from local AEC deviation."""
    center = 0.5 * (cfg.aec_low + cfg.aec_high)
    adj = torch.tanh(cfg.ers_k * (aec_r - center))  # [R] in [-1,1]
    return 1.0 + 0.5 * adj  # [R] in [0.5,1.5]
```
```src/atp_economy/services/pricing.py
# src/atp_economy/services/pricing.py
import torch
from ..domain.state import WorldState
from ..utils.tensor_utils import DTYPE, Device

_BIG = torch.tensor(1e30, device=Device, dtype=DTYPE)


def price_floor_from_duals(state, margin=1.02):
    """
    Unit-cost floor per good and region from current duals and input prices.
    For each reaction j producing good g:
        floor_{g,r} = ( Σ_i p_{i,r} * max(0, -S_{i,j}) + μ_r * ξ_{r,j} + λ_r * σ_{r,j} ) / S_{g,j}  (S_{g,j} > 0)
    Then take min_j over producers of g and apply a small margin (>1) so A > 0 is feasible.
    Also apply a consumer-use floor for final goods: μ*xi_cons + λ*sigma_cons.
    Returns: [G, R]
    """
    S = state.S  # [G,J]
    S_pos = torch.clamp(S, min=0.0)  # outputs
    S_neg = torch.clamp(-S, min=0.0)  # inputs

    p_RG = state.price.T  # [R,G]
    input_cost_RJ = p_RG @ S_neg  # [R,J]
    dual_cost_RJ = (
        state.mu_ex[:, None] * state.xi_eff
        + state.lambda_sink[:, None] * state.sigma_eff
    )  # [R,J]
    cost_RJ = input_cost_RJ + dual_cost_RJ  # [R,J]

    denom_JG = S_pos.T  # [J,G]
    denom_JG = torch.where(denom_JG > 0.0, denom_JG, _BIG)  # avoid div-by-zero

    cand_RJG = cost_RJ[:, :, None] / denom_JG[None, :, :]  # [R,J,G]
    floor_RG = cand_RJG.min(dim=1).values  # [R,G]
    floor_RG = torch.clamp(floor_RG, min=0.0)

    # Final-goods consumer-use floor
    F = getattr(state, "final_idx", None)
    if F is not None and F.numel() > 0:
        cons_floor_RF = (
            state.mu_ex[:, None] * state.xi_cons[None, :]
            + state.lambda_sink[:, None] * state.sigma_cons[None, :]
        )  # [R, |F|]
        floor_RG.index_copy_(1, F, torch.maximum(floor_RG[:, F], cons_floor_RF))

    floor_RG = margin * floor_RG  # small markup
    return floor_RG.T  # [G,R]


# services/pricing.py (inside update_prices)
def update_prices(
    state: WorldState,
    demand_qty_R: torch.Tensor,
    supply_qty_R: torch.Tensor,
    lr: float = 0.01,
    g_clip: float = 5.0,
    logp_bounds: tuple[float, float] = (-20.0, 20.0),
    alpha_anchor: float = 0.005,
    alpha_floor: float = 0.30,  # NEW: how hard we enforce the floor (in log-space)
    margin: float = 1.02,  # NEW: unit-cost markup to keep A > 0 attainable
):
    eps = 1e-12
    logp = torch.log(torch.clamp(state.price, min=eps))

    g = (
        torch.log(torch.clamp(demand_qty_R, min=eps)).T
        - torch.log(torch.clamp(supply_qty_R, min=eps)).T
    )
    g = torch.clamp(g, -g_clip, g_clip)

    # Slow EMA anchor
    state.logp_anchor.data = state.logp_anchor.data * 0.999 + 0.001 * logp
    logp_new = logp + lr * g + alpha_anchor * (state.logp_anchor - logp)

    # NEW: unit-cost price floor
    p_floor = price_floor_from_duals(state, margin=margin)  # [G,R]
    logp_floor = torch.log(torch.clamp(p_floor, min=eps))
    logp_floor_mix = (1.0 - alpha_floor) * logp + alpha_floor * logp_floor
    logp_new = torch.maximum(logp_new, logp_floor_mix)

    logp_new = torch.clamp(logp_new, logp_bounds[0], logp_bounds[1])
    state.price.data = torch.exp(logp_new)


def update_exergy_and_sink_prices(state: WorldState):
    """
    Dual-price updates for exergy (μ) and sink (λ) with bounded exponents.

    μ update:
      ratio = (ex_demand + eps) / (ex_supply + eps)  in [1e-6, 1e6]
      μ <- μ * ratio^{eta_ex}

    λ update:
      MODIFIED: The controller now responds to the stock utilization level, not the flow.
      util_stock = sink_use / (sink_cap + eps)
      λ <- λ * exp( clamp(eta_sink * (EMA(util_stock) - util_target), -40, 40) )
    """
    cfg = state.cfg
    eps = 1e-12

    # Exergy controller
    ex_demand = state.exergy_need_R  # [R]
    ex_supply = state.atp_minted_R  # [R]
    # Safe ratio range to avoid extreme powers
    ratio = torch.clamp((ex_demand + eps) / (ex_supply + eps), 1e-6, 1e6)
    state.ema_ex_ratio.data = (
        state.ema_ex_ratio.data * cfg.ema_ex + (1.0 - cfg.ema_ex) * ratio
    )
    mu_new = state.mu_ex * torch.pow(state.ema_ex_ratio, cfg.eta_ex)
    state.mu_ex.data = torch.clamp(mu_new, min=cfg.mu_floor, max=cfg.mu_cap)

    # Sink controller (MODIFIED LOGIC)
    # The input signal is now the stock utilization, not the flow.
    util_stock = state.sink_use / (state.sink_cap + eps)
    state.ema_sink_util.data = (
        state.ema_sink_util.data * cfg.ema_sink + (1.0 - cfg.ema_sink) * util_stock
    )
    arg = cfg.eta_sink * (state.ema_sink_util - cfg.util_target)
    arg = torch.clamp(arg, -40.0, 40.0)  # trust region for exp
    lam_new = state.lambda_sink * torch.exp(arg)
    state.lambda_sink.data = torch.clamp(
        lam_new, min=cfg.lambda_floor, max=cfg.lambda_cap
    )
```
```src/atp_economy/services/production.py
import torch
import torch.nn.functional as F
from ..domain.state import WorldState
from .settlement import settle_spend_book


def run_production(
    state: WorldState,
    atp_book_R: torch.Tensor,
    aec_r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = state.cfg
    R, J = cfg.R, state.S.shape[1]
    eps = 1e-9

    # Emission gating primitives
    sink_head = torch.clamp_min(state.sink_cap - state.sink_use - state.sink_use_R, 0.0)

    # Affinity
    A = (
        (state.price.T @ state.S)
        - state.mu_ex[:, None] * state.xi_eff
        - state.lambda_sink[:, None] * state.sigma_eff
    )

    # Leontief limiter: min_g inv_rg / need_gj
    inputs_need = (-state.S).clamp(min=0)  # [G,J]
    inv_per_need = torch.where(
        inputs_need > 0,
        state.inventory.unsqueeze(2) / (inputs_need.unsqueeze(0) + eps),
        torch.full_like(state.inventory.unsqueeze(2), float("inf")),
    )  # [R,G,J]
    avail = inv_per_need.min(dim=1).values  # [R,J]

    center = 0.5 * (cfg.aec_low + cfg.aec_high)
    aec_gate = (
        torch.sigmoid(cfg.gate_k * (aec_r - center)) * (1.0 - cfg.gate_min)
        + cfg.gate_min
    )
    labor_gate = getattr(state, "labor_factor_R", None)
    if labor_gate is None:
        labor_gate = torch.ones_like(aec_gate)

    beta = max(cfg.beta_aff, 1e-6)
    drive = F.softplus(beta * A) / beta
    r_potential = state.k_eff * drive * torch.tanh(avail / (1.0 + avail))
    r_potential = (
        torch.minimum(r_potential, state.cap_j[None, :])
        * aec_gate[:, None]
        * labor_gate[:, None]
    )

    atp_need = (torch.relu(r_potential) * state.xi_eff).sum(dim=1)  # [R]
    sink_need = (torch.relu(r_potential) * state.sigma_eff).sum(dim=1)  # [R]
    s_atp = torch.clamp(atp_book_R / (atp_need + eps), max=1.0)
    s_head = torch.clamp(sink_head / (sink_need + eps), max=1.0)
    rate = r_potential * torch.minimum(s_atp, s_head)[:, None]

    delta_RG = rate @ state.S.T
    state.inventory.data = torch.clamp_min(state.inventory.data + delta_RG, 0.0)

    atp_spend = (torch.relu(rate) * state.xi_eff).sum(dim=1)
    sink_emit = (torch.relu(rate) * state.sigma_eff).sum(dim=1)
    _, atp_book_R = settle_spend_book(state, atp_spend, atp_book_R)

    state.emit_sink_R.data = state.emit_sink_R.data + sink_emit
    state.exergy_need_R.data = state.exergy_need_R.data + atp_spend
    state.sink_use_R.data = state.sink_use_R.data + sink_emit

    return rate, atp_book_R
```
```src/atp_economy/services/settlement.py
import torch
from ..domain.state import WorldState
from ..utils.integrators import patankar_imex_transfer
from ..utils.tensor_utils import Device, DTYPE


def settle_spend_book(
    state: WorldState, spend_R: torch.Tensor, atp_book_R: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-9
    R = state.cfg.R

    eATP_sorted = state.eATP[state.agent_order]
    pool_r = torch.segment_reduce(
        data=eATP_sorted, reduce="sum", offsets=state.rowptr, axis=0
    )

    cap_r = torch.minimum(atp_book_R, pool_r)
    actual = torch.minimum(spend_R, cap_r)
    shortfall = torch.clamp(spend_R - actual, min=0.0)

    region_idx = state.agent_region
    denom = pool_r[region_idx] + eps
    factor_i = actual[region_idx] / denom
    delta_i = state.eATP * factor_i

    state.eATP.data = state.eATP.data - delta_i
    state.eADP.data = state.eADP.data + delta_i

    atp_book_R = atp_book_R - actual
    state.pool_atp_R.data = state.pool_atp_R.data - actual
    state.pool_adp_R.data = state.pool_adp_R.data + actual

    return shortfall, atp_book_R


def apply_demurrage(state: WorldState, factors: torch.Tensor) -> None:
    cfg = state.cfg

    # ATP -> ADP demurrage
    k_r = torch.clamp(cfg.demurrage * factors, min=0.0)  # [R]
    k_agent = k_r[state.agent_region]  # [N]
    eATP_new, eADP_new = patankar_imex_transfer(
        state.eATP, state.eADP, rate=k_agent, dt=cfg.dt
    )
    state.eATP.data = eATP_new
    state.eADP.data = eADP_new

    denom = 1.0 + k_r * cfg.dt  # [R]
    pool_atp_new = state.pool_atp_R / denom
    pool_adp_new = state.pool_adp_R + (k_r * cfg.dt) * pool_atp_new
    state.pool_atp_R.data = pool_atp_new
    state.pool_adp_R.data = pool_adp_new

    # AMP -> ADP leak under chronic stress
    aec_r = (state.pool_atp_R + 0.5 * state.pool_adp_R) / (
        state.pool_atp_R + state.pool_adp_R + state.pool_amp_R + 1e-12
    )
    leak_rate = 0.01 * torch.relu(cfg.aec_low - aec_r)  # up to 1%/step at deep stress

    k_amp_agent = leak_rate[state.agent_region]
    eAMP_new, eADP_new2 = patankar_imex_transfer(
        state.eAMP, state.eADP, rate=k_amp_agent, dt=cfg.dt
    )
    state.eAMP.data = eAMP_new
    state.eADP.data = eADP_new2

    denom_amp = 1.0 + leak_rate * cfg.dt
    pool_amp_new = state.pool_amp_R / denom_amp
    state.pool_amp_R.data = pool_amp_new
    state.pool_adp_R.data = state.pool_adp_R + (leak_rate * cfg.dt) * pool_amp_new
```
```src/atp_economy/services/storage_invest.py
# src/atp_economy/services/storage_invest.py
import torch
from ..domain.state import WorldState


def apply_storage_investment(state: WorldState, storage_budget_R: torch.Tensor):
    """
    Update storage capacity with investment and depreciation:
      cap_{t+1} = (cap_t + dt * eta * invest) / (1 + dt * decay)
    Also clamp state-of-charge to the capacity.
    """
    cfg = state.cfg
    cap_num = state.storage_cap + cfg.dt * cfg.eta_storage * torch.clamp(
        storage_budget_R, min=0.0
    )
    cap_den = 1.0 + cfg.dt * cfg.storage_decay
    state.storage_cap.data = torch.clamp(cap_num / cap_den, min=0.0)

    state.storage_soc.data = torch.minimum(state.storage_soc, state.storage_cap)
```
```src/atp_economy/services/trade.py
import torch
from ..domain.state import WorldState
from .settlement import settle_spend_book
from ..utils.tensor_utils import Device, DTYPE


def run_trade(
    state: WorldState,
    supply_R: torch.Tensor,
    demand_R: torch.Tensor,
    atp_book_R: torch.Tensor,
    kappa: float = 0.8,
) -> torch.Tensor:
    """
    Neighbor trade gated by ATP, sink headroom, and per-step sink-flow budget.
    """
    cfg = state.cfg
    eps = 1e-9
    R, G = cfg.R, cfg.G

    surplus = torch.relu(supply_R - demand_R)  # [R,G]
    deficit = torch.relu(demand_R - supply_R)  # [R,G]

    nbr = state.nbr_idx  # [R,k]
    cost = state.nbr_cost  # [R,k]
    cap = torch.clamp_min(state.nbr_cap, 1e-6)  # [R,k]
    k = nbr.shape[1]

    cost_penalty = cost.unsqueeze(-1)  # [R,k,1]
    neigh_def = deficit.index_select(0, nbr.reshape(-1)).reshape(R, k, G)  # [R,k,G]
    scores = torch.relu(neigh_def - cost_penalty)  # [R,k,G]
    score_sum = scores.sum(dim=1, keepdim=True) + eps
    alloc = scores / score_sum  # [R,k,G]

    ship = alloc * (kappa * surplus.unsqueeze(1))  # [R,k,G]

    ship_sumG = ship.sum(dim=2)  # [R,k]
    route_scale = torch.minimum(torch.ones_like(cap), cap / (ship_sumG + eps))
    ship = ship * route_scale.unsqueeze(-1)

    dist_rg = state.distance.gather(1, nbr)  # [R,k]
    qty_out = ship.sum(dim=2)  # [R,k]
    atp_log_need = cfg.alpha_logistics_ex * (qty_out * dist_rg).sum(dim=1)  # [R]
    sink_log_emit = cfg.alpha_logistics_sink * (qty_out * dist_rg).sum(dim=1)  # [R]

    # Emission gating primitives
    sink_head = torch.clamp_min(state.sink_cap - state.sink_use - state.sink_use_R, 0.0)

    s_atp = torch.clamp(atp_book_R / (atp_log_need + eps), max=1.0)
    s_head = torch.clamp(sink_head / (sink_log_emit + eps), max=1.0)
    s = torch.minimum(s_atp, s_head)
    ship = ship * s.unsqueeze(1).unsqueeze(2)

    # Recompute bills after scaling and settle
    qty_out = ship.sum(dim=2)
    atp_log_need = cfg.alpha_logistics_ex * (qty_out * dist_rg).sum(dim=1)
    sink_log_emit = cfg.alpha_logistics_sink * (qty_out * dist_rg).sum(dim=1)

    _, atp_book_R = settle_spend_book(state, atp_log_need, atp_book_R)
    state.emit_sink_R.data = state.emit_sink_R.data + sink_log_emit

    outflow = ship.sum(dim=1)  # [R,G]
    inflow = torch.zeros(R, G, device=Device, dtype=DTYPE)
    inflow = inflow.index_add(0, nbr.reshape(-1), ship.reshape(R * k, G))
    state.inventory.data = torch.clamp_min(state.inventory.data - outflow + inflow, 0.0)

    state.exergy_need_R.data = state.exergy_need_R.data + atp_log_need
    state.sink_use_R.data = state.sink_use_R.data + sink_log_emit

    return atp_book_R
```
```src/atp_economy/sim/__init__.py
# src/atp_economy/sim/__init__.py
__all__ = []
```
```src/atp_economy/sim/model.py
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
```
```src/atp_economy/utils/__init__.py
# src/atp_economy/utils/__init__.py
__all__ = []
```
```src/atp_economy/utils/checks.py
# src/atp_economy/utils/checks.py
import torch
from torch import nn
```
```src/atp_economy/utils/integrators.py
# src/atp_economy/utils/integrators.py
import torch


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
```
```src/atp_economy/utils/metrics.py
# src/atp_economy/utils/metrics.py
from __future__ import annotations
from typing import Dict, List, Iterable, Optional
import numpy as np


class MetricsRecorder:
    """
    Memory-lean recorder:
    - Keeps only selected small vectors (already numpy) per step.
    - Ring buffer with optional stride to downsample.
    """

    def __init__(
        self, keys: Iterable[str], maxlen: Optional[int] = None, stride: int = 1
    ):
        self.keys = list(keys)
        self.maxlen = maxlen
        self.stride = max(1, stride)
        self._step = 0
        self._store: Dict[str, List[np.ndarray]] = {k: [] for k in self.keys}

    def record(self, metrics: Dict[str, np.ndarray]):
        self._step += 1
        if (self._step - 1) % self.stride != 0:
            return
        for k in self.keys:
            v = metrics.get(k, None)
            if v is None:
                continue
            self._store[k].append(v.copy())
            if self.maxlen is not None and len(self._store[k]) > self.maxlen:
                # pop front (ring buffer)
                self._store[k].pop(0)

    def as_arrays(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for k, seq in self._store.items():
            out[k] = np.stack(seq, axis=0) if len(seq) else np.zeros((0,))
        return out

    def clear(self):
        for k in self.keys:
            self._store[k].clear()
        self._step = 0
```
```src/atp_economy/utils/tensor_utils.py
# src/atp_economy/utils/tensor_utils.py
import torch

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
```
```src/atp_economy/vis/static.py
import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt


def _plot_spatial_lines(
    ax, arr, title, ylabel, max_lines=16, ylim=None, yscale="linear"
):
    ax.cla()
    if arr.size == 0:
        return
    T, R = arr.shape
    x = np.arange(T)
    for r in range(min(R, max_lines)):
        ax.plot(x, arr[:, r], label=f"R{r}", lw=1)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)  # Set the y-axis scale

    if ylim is not None:
        ax.set_ylim(*ylim)
    elif yscale != "log":
        ymin, ymax = float(arr.min()), float(arr.max())
        if np.isfinite(ymin) and np.isfinite(ymax):
            margin = 0.1 * max(1e-9, ymax - ymin)
            ax.set_ylim(ymin - margin, ymax + margin)

    ax.legend(loc="upper left", ncol=2, fontsize="x-small", frameon=False)


def _plot_mulam(ax, mu, lam):
    ax.cla()
    ax.set_title("Exergy μ and Sink λ (means)")
    ax.set_xlabel("Step")
    if mu.size:
        mu_mean = mu.mean(axis=1)
        ax.plot(mu_mean, color="tab:blue", label="μ mean")
        ax.set_ylabel("μ")
    ax2 = ax.twinx()
    if lam.size:
        lam_mean = lam.mean(axis=1)
        ax2.plot(lam_mean, color="tab:orange", label="λ mean")
        ax2.set_ylabel("λ")
    l1, n1 = ax.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    if l1 or l2:
        ax.legend(l1 + l2, n1 + n2, loc="upper left", frameon=False)


def _plot_decoupling_metrics(ax, xp, si):
    ax.cla()
    ax.set_title("Exergy Productivity & Sink Intensity (means)")
    ax.set_xlabel("Step")
    if xp.size:
        xp_mean = xp.mean(axis=1)
        ax.plot(xp_mean, color="tab:green", label="Exergy Prod.")
        ax.set_ylabel("GDP / ATP Minted", color="tab:green")
        ax.tick_params(axis="y", labelcolor="tab:green")
    ax2 = ax.twinx()
    if si.size:
        si_mean = si.mean(axis=1)
        ax2.plot(si_mean, color="tab:red", label="Sink Intensity")
        ax2.set_ylabel("Emissions / GDP", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.set_yscale("log")
    l1, n1 = ax.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    if l1 or l2:
        ax.legend(l1 + l2, n1 + n2, loc="upper left", frameon=False)


def render_static(
    history: dict,
    save_fig: str | None = None,
    dpi: int = 150,
    style: str | None = "seaborn-v0_8",
):
    if style:
        try:
            plt.style.use(style)
        except Exception:
            pass

    aec = history.get("AEC_region", np.zeros((0, 1)))
    gdp_flow = history.get("GDP_flow_region", np.zeros((0, 1)))
    gdp_pc = history.get("GDP_pc_region", np.zeros((0, 1)))
    mu = history.get("mu_ex", np.zeros((0, 1)))
    lam = history.get("lambda_sink", np.zeros((0, 1)))
    sunk = history.get("sink_utilization", np.zeros((0, 1)))
    xp = history.get("exergy_productivity_region", np.zeros((0, 1)))
    si = history.get("sink_intensity_region", np.zeros((0, 1)))

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    ax_aec, ax_gdp = axes[0, 0], axes[0, 1]
    ax_mulam, ax_sink = axes[1, 0], axes[1, 1]
    ax_gdppc, ax_decouple = axes[2, 0], axes[2, 1]

    _plot_spatial_lines(
        ax_aec,
        aec,
        "AEC by Region (Spatial)",
        "AEC",
        max_lines=aec.shape[1] if aec.size else 0,
        ylim=(0.0, 1.0),
    )
    _plot_spatial_lines(
        ax_gdp,
        gdp_flow,
        "GDP (Value Added) by Region (Spatial)",
        "Value (log scale)",
        max_lines=gdp_flow.shape[1] if gdp_flow.size else 0,
        yscale="log",  # Use log scale
    )
    _plot_mulam(ax_mulam, mu, lam)

    if sunk.size:
        ymax = float(np.max(sunk))
        ymax = max(ymax, 1e-6)
        ylim_sink = (0.0, 1.1 * ymax)
    else:
        ylim_sink = (0.0, 1.0)
    _plot_spatial_lines(
        ax_sink,
        sunk,
        "Sink Utilization (Spatial)",
        "Use / Capacity",
        max_lines=sunk.shape[1] if sunk.size else 0,
        ylim=ylim_sink,
    )

    _plot_spatial_lines(
        ax_gdppc,
        gdp_pc,
        "GDP per Capita by Region (Spatial)",
        "Value per Person (log scale)",
        max_lines=gdp_pc.shape[1] if gdp_pc.size else 0,
        yscale="log",  # Use log scale
    )

    _plot_decoupling_metrics(ax_decouple, xp, si)

    fig.tight_layout()
    if save_fig:
        fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()
```
```tests/test_profiling.py
# tests/test_profiling.py
import pytest
import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity

from atp_economy.sim.model import ATPEconomy
from atp_economy.config import EconConfig
from atp_economy.utils.tensor_utils import Device

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    torch.backends.cudnn.allow_tf32 = True


def format_hertz(sps):
    """Formats a number into Hz, kHz, MHz, or GHz."""
    if sps >= 1_000_000_000:
        return f"{sps / 1_000_000_000:.2f} GHz"
    if sps >= 1_000_000:
        return f"{sps / 1_000_000:.2f} MHz"
    if sps >= 1_000:
        return f"{sps / 1_000:.2f} kHz"
    return f"{sps:.2f} Hz"


@pytest.mark.parametrize("R, G, J, N", [(16, 24, 12, 100_000)])
def test_torch_profiler_step(R, G, J, N):
    """
    Runs a detailed PyTorch profiler analysis on the ATPEconomy.step() method
    to identify internal bottlenecks.
    """
    print(
        f"--- Profiling with R={R}, G={G}, J={J}, N={N}, dtype=float32 on {Device} ---"
    )
    cfg = EconConfig(R=R, G=G, J=J, N=N, seed=42)
    model = ATPEconomy(cfg=cfg)

    total_steps = 50
    warmup_steps = 10

    activities = [ProfilerActivity.CPU]
    if Device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        # Warmup phase
        for _ in range(warmup_steps):
            model.step()

        # Profiling phase
        for _ in range(warmup_steps, total_steps):
            with record_function("model_step_call"):
                model.step()

    sort_key = (
        "self_cuda_time_total" if Device.type == "cuda" else "self_cpu_time_total"
    )
    print(f"--- PyTorch Profiler Results (Top 15 by {sort_key}) ---")
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by=sort_key, row_limit=15
        )
    )

    keys = [e.key for e in prof.key_averages()]


@pytest.mark.parametrize("R, G, J, N", [(16, 24, 12, 100_000)])
def test_performance_sps(R, G, J, N):
    """
    Measures the wall-clock performance of ATPEconomy.step() in steps-per-second (SPS).
    """
    print(
        f"--- Benchmarking SPS with R={R}, G={G}, J={J}, N={N}, dtype=float32 on {Device} ---"
    )
    cfg = EconConfig(R=R, G=G, J=J, N=N, seed=42)
    model = ATPEconomy(cfg=cfg)

    total_steps = 100
    warmup_steps = 20

    # Warmup
    for _ in range(warmup_steps):
        model.step()

    if Device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    # Timed run
    for _ in range(total_steps):
        model.step()

    if Device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    sps = total_steps / elapsed_time

    print(f"Completed {total_steps} steps in {elapsed_time:.3f} seconds.")
    print(f"Performance: {sps:.2f} steps/sec ({format_hertz(sps)})")
    assert sps > 0
```
