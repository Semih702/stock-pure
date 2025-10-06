# tests/test_ops_fft.py
from __future__ import annotations

import math
from typing import cast

import pytest
import torch

from models.timesnet.types import BTC
from models.timesnet.utils import fft_top_periods


def make_sine_batch(
    b: int,
    t: int,
    c: int,
    period: int,
    amplitude: float = 1.0,
    phase_offset_per_batch: bool = True,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Generate a batch of sine waves with given period."""
    idx = torch.arange(t, device=device).float()  # [t]
    waves = []
    for bi in range(b):
        phase = (bi * math.pi / 7.0) if phase_offset_per_batch else 0.0
        s = amplitude * torch.sin(2.0 * math.pi * (idx + phase) / period)  # [t]
        # tile across channels
        waves.append(s.unsqueeze(-1).repeat(1, c))  # [t, c]
    x = torch.stack(waves, dim=0)  # [b, t, c]
    return x


@torch.no_grad()
@pytest.mark.unit
def test_fft_top_periods_single_strong_cycle_detected():
    torch.manual_seed(0)
    b, t, c = 8, 120, 3  # 120 steps → rFFT bins 0..60
    true_period = 12  # frequency index = t / period = 10
    x = make_sine_batch(b, t, c, period=true_period, amplitude=1.0)

    k = 1
    periods, weights = fft_top_periods(
        cast(BTC, x),
        k,
    )

    # Shapes
    assert periods.shape == (k,)
    assert weights.shape == (b, k)

    # Correct period
    assert int(periods[0].item()) == true_period

    # Strong weights (positive)
    assert torch.all(weights[:, 0] > 0)


@torch.no_grad()
@pytest.mark.unit
def test_fft_top_periods_two_cycles_ordered_by_amplitude():
    torch.manual_seed(0)
    b, t, c = 4, 240, 2
    p1, a1 = 24, 1.0  # daily (stronger)
    p2, a2 = 12, 0.5  # half-day (weaker)

    x1 = make_sine_batch(b, t, c, period=p1, amplitude=a1, phase_offset_per_batch=True)
    x2 = make_sine_batch(b, t, c, period=p2, amplitude=a2, phase_offset_per_batch=True)
    x = x1 + x2

    k = 2
    periods, weights = fft_top_periods(cast(BTC, x), k)

    # We expect the strongest period first (since amp_mean drives topk)
    got = sorted([int(periods[0].item()), int(periods[1].item())])
    expected = sorted([p1, p2])
    assert got == expected

    # The per-batch weights should reflect relative strength:
    # weight for p1 should generally exceed that for p2 across the batch
    # (not strictly guaranteed for every batch element due to phase, but typical)
    # We'll check the mean across batch.
    mean_weights = weights.mean(dim=0)  # [k]
    # align by mapping period index positions to actual periods
    # (periods tensor is length k; find which index corresponds to p1)
    idx_p1 = 0 if int(periods[0].item()) == p1 else 1
    idx_p2 = 1 - idx_p1
    assert mean_weights[idx_p1] > mean_weights[idx_p2]


@torch.no_grad()
@pytest.mark.unit
def test_fft_top_periods_ignores_dc_component():
    # DC = constant offset; should not be selected
    b, t, c = 2, 128, 1
    x = torch.ones(b, t, c) * 7.0  # pure DC
    # add a tiny sinusoid so topk has something non-zero to pick
    x += make_sine_batch(b, t, c, period=16, amplitude=0.1, phase_offset_per_batch=False)

    k = 1
    periods, _ = fft_top_periods(cast(BTC, x), k)
    assert int(periods[0].item()) == 16  # not +inf/0; DC suppressed


@torch.no_grad()
@pytest.mark.unit
def test_fft_top_periods_device_dtype_and_no_grad():
    # Ensure it runs on float32, and no gradients are recorded (decorated with @no_grad)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, t, c = 2, 96, 4
    x = torch.randn(b, t, c, device=device, dtype=torch.float32, requires_grad=False)

    k = 3
    periods, weights = fft_top_periods(cast(BTC, x), k)
    assert periods.device == x.device
    assert weights.device == x.device
    assert periods.dtype in (torch.int64, torch.long)
    assert weights.dtype == torch.float32
