# src/atp_economy/vis/static.py
import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt


def _plot_envelope(
    ax, arr, title, ylabel, color="tab:blue", alpha=0.15, pband=(10, 90), ylim=None
):
    ax.cla()
    if arr.size == 0:
        return
    mean = arr.mean(axis=1)  # [T]
    lo = np.percentile(arr, pband[0], axis=1)
    hi = np.percentile(arr, pband[1], axis=1)
    x = np.arange(arr.shape[0])
    ax.plot(x, mean, color=color, lw=2, label="mean")
    ax.fill_between(
        x, lo, hi, color=color, alpha=alpha, label=f"P{pband[0]}–P{pband[1]}"
    )
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ymin, ymax = float(lo.min()), float(hi.max())
        margin = 0.1 * max(1e-9, ymax - ymin)
        ax.set_ylim(ymin - margin, ymax + margin)
    ax.legend(loc="upper left", frameon=False)


def _plot_spatial_lines(ax, arr, title, ylabel, max_lines=16, ylim=None):
    """Plots individual lines for each region (spatial analysis)."""
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

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ymin, ymax = float(arr.min()), float(arr.max())
        if np.isfinite(ymin) and np.isfinite(ymax):
            margin = 0.1 * max(1e-9, ymax - ymin)
            ax.set_ylim(ymin - margin, ymax + margin)

    ax.legend(loc="upper left", ncol=2, fontsize="x-small", frameon=False)


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

    # 3x2 layout: (1) AEC spatial, (2) GDPf spatial,
    #              (3) μ/λ means, (4) Sink Util spatial (autoscaled),
    #              (5) GDP per capita spatial, (6) blank
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    ax_aec, ax_gdp = axes[0, 0], axes[0, 1]
    ax_mulam, ax_sink = axes[1, 0], axes[1, 1]
    ax_gdppc, ax_blank = axes[2, 0], axes[2, 1]

    # AEC spatial
    _plot_spatial_lines(
        ax_aec,
        aec,
        "AEC by Region (Spatial)",
        "AEC",
        max_lines=aec.shape[1] if aec.size else 0,
        ylim=(0.0, 1.0),
    )

    # GDPf spatial
    _plot_spatial_lines(
        ax_gdp,
        gdp_flow,
        "GDP (Value Added) by Region (Spatial)",
        "Value",
        max_lines=gdp_flow.shape[1] if gdp_flow.size else 0,
        ylim=None,
    )

    # μ and λ means
    ax_mulam.cla()
    ax_mulam.set_title("Exergy μ and Sink λ (means)")
    ax_mulam.set_xlabel("Step")
    if mu.size:
        mu_mean = mu.mean(axis=1)
        ax_mulam.plot(mu_mean, color="tab:blue", label="μ mean")
        ax_mulam.set_ylabel("μ")
    ax2 = ax_mulam.twinx()
    if lam.size:
        lam_mean = lam.mean(axis=1)
        ax2.plot(lam_mean, color="tab:orange", label="λ mean")
        ax2.set_ylabel("λ")
    lines1, labels1 = ax_mulam.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines1 or lines2:
        ax_mulam.legend(
            lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False
        )

    # Sink utilization spatial with automatic max (no 1.0 floor)
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

    # GDP per capita spatial
    _plot_spatial_lines(
        ax_gdppc,
        gdp_pc,
        "GDP per Capita by Region (Spatial)",
        "Value per Person",
        max_lines=gdp_pc.shape[1] if gdp_pc.size else 0,
        ylim=None,
    )

    # Blank panel for breathing room or future use
    ax_blank.axis("off")

    fig.tight_layout()
    if save_fig:
        fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()
