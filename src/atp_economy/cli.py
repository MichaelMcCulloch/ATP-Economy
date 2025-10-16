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
    writer = (
        SummaryWriter(
            log_dir=(
                os.path.expanduser(f"~/.tensorboard/{tb_logdir}") if tb_logdir else None
            )
        )
        if logging_enabled
        else SummaryWriter()
    )

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
