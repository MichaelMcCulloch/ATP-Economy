# ATP-Economy: A Biophysical Agent-Based Model

This project simulates a multi-region global economy grounded in biophysical principles. Unlike traditional macroeconomic models that treat money and resources as abstract quantities, this simulation enforces that all real economic activity—production, extraction, and trade—is ultimately constrained by primary resource flows and the finite capacity of the environment to absorb waste.

The model is designed to explore long-term dynamics where technological innovation, resource depletion, and environmental limits are the primary drivers of growth, stability, and inequality.

## Core Concepts: A Biophysical Economic Engine

The simulation is built on a set of core concepts that model the economy not as an abstract system of accounts, but as a living, resource-dependent metabolism.

### 1. A Three-Tiered Monetary System: The Financial Metabolism

The economy operates with a physically-grounded monetary system that mirrors cellular bioenergetics, ensuring financial claims are tied to the real capacity to perform work.

*   **High-Potential Capital (`eATP`):** The primary settlement asset, analogous to Adenosine Triphosphate. Only `eATP` can be "hydrolyzed" to finance real work—manufacturing, resource extraction, and logistics. It is the ultimate unit of account for physical value creation.
*   **Working Capital (`eADP`):** The result of spending `eATP`. It represents capital that has been deployed into the economy and cannot be used for new production until it is "recharged."
*   **Deep Reserves (`eAMP`):** A non-circulating asset whose accumulation signals systemic financial distress or a severe lack of recapitalization opportunities, much like a cell running out of energy.

Recapitalization (`eADP` → `eATP`) is performed by regional **Capital Banks** (the "mitochondria" of the economy) and is strictly collateralized by verified primary resource flows (e.g., energy generation). This prevents the unbacked creation of settlement-grade liquidity. Furthermore, `eATP` is subject to a **holding cost (demurrage)**, a small negative interest rate that encourages productive investment over sterile hoarding.

### 2. Production as Transformation: The Holistic Profit Motive

Economic activity is modeled as a network of production functions where sectors transform goods into other goods. The incentive to produce is driven by a comprehensive **profit motive (Affinity)**, which forces agents to account for total system costs:
*   Market prices of inputs and outputs.
*   The **shadow price of primary resources (`μ`)**, representing the cost of recapitalizing the capital spent.
*   The **shadow price of environmental capacity (`λ`)**, an endogenously determined Pigouvian tax on negative externalities.

A production process is only profitable—and therefore undertaken—if its expected revenue exceeds the full cost of its material inputs *and* its primary resource and environmental footprints.

### 3. Binding Environmental Constraints: The Finite Sink

The environment is modeled as a stock of pollutants that accumulates from economic activity and is reduced by a natural assimilation rate. The capacity of this "sink" is finite. Every economic process generates a **dissipation cost** that adds to the pollutant stock. The shadow price `λ` is not set by a regulator but is adjusted by an endogenous controller that steers the system's emission flows toward a sustainable target utilization level. As the system approaches its environmental limits, `λ` rises, rendering high-externality production unprofitable and forcing a hard ceiling on unsustainable economic throughput.

### 4. Macroprudential Stability: The Financial Circuit Breaker

The financial health of each regional economy is measured by its **Capital Adequacy Ratio (AEC)**, a metric analogous to a bank's liquidity coverage ratio. This ratio reflects the availability of high-potential `eATP` relative to total circulating capital. The system features an automatic stabilizer (an "AMPK-like gate") where a low regional AEC automatically throttles the rates of all production activities. This macroprudential mechanism acts as a circuit breaker, preventing a "run on the bank" by forcing the economy to conserve its settlement liquidity during periods of stress, thereby avoiding systemic collapse.

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
    uv run run-sim run configs/H1_baseline_stability.yaml

    # You can override key I/O settings for quick experiments:
    uv run run-sim run configs/H10c_decouple_and_grow.yaml --steps 5000 --save-fig my_test_run.png
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

## Interpreting the Output

The simulation produces a final plot with five key panels. After the stability patches, focus on trends and ratios, not absolute magnitudes of value, which are nominal.

*   **AEC by Region (Spatial):** Shows the Capital Adequacy Ratio. Healthy systems typically see regions converge into a stable band (e.g., 0.6-0.8). Dips followed by recovery show the AMPK-like gate is working.
*   **GDP (Value Added) by Region (Spatial):** The aggregate flow of value added. Look for sustained growth, stability, or decline. The absolute scale is nominal and depends on initial prices.
*   **Exergy μ and Sink λ (means):** The shadow prices. A price that is persistently elevated above its floor indicates a binding constraint. In many scenarios, these will remain low, which is expected behavior unless a constraint is intentionally stressed.
*   **Sink Utilization (Spatial):** Shows how close each region's pollutant stock is to its environmental capacity limit (1.0). A key diagnostic for decoupling is a flat or declining curve here while the GDP curve is rising.
*   **GDP per Capita by Region (Spatial):** Shows value added per person. This metric provides insight into regional productivity and living standards, factoring in demographic changes.

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