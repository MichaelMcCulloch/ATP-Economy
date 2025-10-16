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
