from __future__ import annotations

import torch


@torch.no_grad()
def fft_top_periods(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate dominant periods via FFT amplitudes.

    Args:
        x: [b, t, c] time series.
        k: Number of top frequency indices to select.

    Returns:
        periods: [k] int tensor. Each is t // freq_idx.
        weights: [b, k] float tensor. Per-batch amplitudes.
    """
    b, t, c = x.shape
    xf = torch.fft.rfft(x, dim=1)  # [b, t//2+1, c]
    amp_mean = xf.abs().mean(dim=0).mean(dim=-1)  # [t//2+1]
    amp_mean[0] = 0.0
    k = min(k, amp_mean.numel())
    _, idx = torch.topk(amp_mean, k)  # [k]
    periods = (t // idx).to(torch.long)  # [k]
    batch_amp = xf.abs().mean(dim=-1)  # [b, t//2+1]
    weights = batch_amp[:, idx]  # [b, k]
    return periods, weights
