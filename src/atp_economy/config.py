# src/atp_economy/config.py
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

    # ------------- Demography -------------
    pop_init_scale: float = 1.0e6
    birth_base: float = 0.015
    death_base: float = 0.010
    aec_birth_center: float = 0.8
    aec_death_center: float = 0.5
    birth_k: float = 5.0
    death_k: float = 7.0
    birth_endow_atp: float = 0.2
    birth_endow_fiat: float = 5.0

    # ------------- Inheritance -------------
    inherit_conc: float = 2.0
    inherit_frac_on_death: float = 0.9

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
    innov_I_cap: float = 1.0e12  # absorbs large R&D without overflow

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EconConfig":
        """
        Strict structured loader. Only reads recognized sections and coerces numeric
        strings (e.g., "1.0e9") to proper numbers to avoid runtime type issues.
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
            "runtime",
        ]
        known = {f.name for f in fields(cls)}
        args: Dict[str, Any] = {}

        # Collect known keys
        for section in allowed:
            sec = config_dict.get(section)
            if isinstance(sec, dict):
                for k, v in sec.items():
                    if k in known:
                        args[k] = v

        # Seed from runtime
        rt = config_dict.get("runtime")
        if isinstance(rt, dict) and "seed" in rt:
            args["seed"] = rt["seed"]

        # Coerce numeric strings to numbers to prevent type errors later
        for f in fields(cls):
            name = f.name
            if name not in args:
                continue
            v = args[name]
            # Only coerce primitive numeric fields
            if f.type is float:
                if isinstance(v, str):
                    try:
                        args[name] = float(v)
                    except Exception:
                        pass
                elif isinstance(v, int):
                    args[name] = float(v)
            elif f.type is int:
                if isinstance(v, str):
                    try:
                        # Allow scientific notation in strings for ints (cast via float)
                        if any(ch in v for ch in (".", "e", "E")):
                            args[name] = int(float(v))
                        else:
                            args[name] = int(v)
                    except Exception:
                        pass
                elif isinstance(v, float):
                    args[name] = int(v)

        # Check required size fields
        required_sizes = ["R", "G", "J", "N"]
        missing = [k for k in required_sizes if k not in args]
        if missing:
            raise ValueError(
                f"Missing required size fields in 'sizes' section: {missing}. "
                f"Example:\n  sizes: {{ R: 16, G: 24, J: 12, N: 200000 }}"
            )

        return cls(**args)
