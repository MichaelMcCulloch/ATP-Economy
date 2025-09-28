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
    assert any("model_step_call" in k for k in keys), "model_step_call not found."
    assert any("phase:aggregation" in k for k in keys), "phase:aggregation not found."


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
