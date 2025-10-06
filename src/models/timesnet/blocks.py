from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f

from .ops import fft_top_periods


class TimesBlock(nn.Module):
    """Fold 1D into 2D by dominant periods, apply conv2d, unfold, and aggregate.

    I/O: x [b, t, c] -> out [b, t, c]
    """

    def __init__(self, d_model: int, d_ff: int, k: int, num_kernels: int):
        super().__init__()
        from .layers import InceptionBlockV1  # avoid cycles

        self.k = k
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels),
        )

    def fold_time(self, x: torch.Tensor, period: int, total_len: int) -> tuple[torch.Tensor, int]:
        """Fold [b,t,c] → [b,c,rows,period], padding time to a multiple of period."""
        b, t, c = x.shape
        p = max(int(period), 1)
        if total_len % p != 0:
            rows = total_len // p + 1
            pad = rows * p - total_len
            x = torch.cat([x, torch.zeros(b, pad, c, device=x.device, dtype=x.dtype)], dim=1)
        else:
            rows = total_len // p
        y = x.reshape(b, rows, p, c).permute(0, 3, 1, 2).contiguous()  # [b,c,rows,p]
        return y, rows

    def unfold_time(
        self, y: torch.Tensor, rows: int, period: int, total_len: int, t_orig: int
    ) -> torch.Tensor:
        """Unfold [b,c,rows,period] → [b,t_orig,c] (cropped)."""
        b, c, _, p = y.shape
        out = y.permute(0, 2, 3, 1).reshape(b, rows * p, c)  # [b, rows*period, c]
        return out[:, :total_len, :][:, :t_orig, :]

    def forward(self, x: torch.Tensor, total_len: int) -> torch.Tensor:
        """Args:
        x: [b, t, c]
        total_len: seq_len + pred_len (padding/reshape target)
        """
        b, t, c = x.shape

        # Limit k to available non-DC rFFT bins.
        k_eff = min(self.k, max(t // 2, 1))

        periods, weights = fft_top_periods(x, k_eff)  # periods: [k_eff], weights: [b,k_eff]
        weights = weights.to(dtype=x.dtype)
        recon: list[torch.Tensor] = []

        for i in range(k_eff):
            p = int(periods[i].item()) or 1
            y, rows = self.fold_time(x, p, total_len)  # [b,c,rows,p]
            y = self.conv(y)  # [b,c,rows,p]
            y = self.unfold_time(y, rows, p, total_len, t)  # [b,t,c]
            recon.append(y)

        if k_eff == 1:
            out = recon[0]
        else:
            stacked = torch.stack(recon, dim=-1)  # [b, t, c, k_eff]
            attn = f.softmax(weights, dim=1).unsqueeze(1).unsqueeze(1)  # [b,1,1,k_eff]
            out = torch.sum(stacked * attn, dim=-1)  # [b, t, c]

        return out + x  # residual
