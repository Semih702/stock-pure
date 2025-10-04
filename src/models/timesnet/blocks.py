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

    def forward(self, x: torch.Tensor, total_len: int) -> torch.Tensor:
        """Args:
        x: [b, t, c]
        total_len: seq_len + pred_len (padding/reshape target)
        """
        b, t, c = x.shape
        periods, weights = fft_top_periods(x, self.k)  # [k], [b,k]
        reconstructions: list[torch.Tensor] = []

        for i in range(self.k):
            period = int(periods[i].item()) or 1

            # pad to multiple of 'period' along time
            if total_len % period != 0:
                rows = (total_len // period) + 1
                pad_len = rows * period - total_len
                pad = torch.zeros(b, pad_len, c, device=x.device, dtype=x.dtype)
                seq = torch.cat([x, pad], dim=1)  # [b, total_len+pad, c]
            else:
                rows = total_len // period
                seq = x  # [b, total_len, c]

            # fold: [b, rows, period, c] -> [b, c, rows, period]
            y = seq.reshape(b, rows, period, c).permute(0, 3, 1, 2).contiguous()
            y = self.conv(y)
            # unfold: [b, c, rows, period] -> [b, rows*period, c]
            y = y.permute(0, 2, 3, 1).reshape(b, rows * period, c)
            reconstructions.append(y[:, :total_len, :])

        if self.k == 1:
            out = reconstructions[0]
        else:
            stacked = torch.stack(reconstructions, dim=-1)  # [b, T_total, c, k]
            attn = f.softmax(weights, dim=1)[..., None, None, :]  # [b,1,1,k]
            out = torch.sum(stacked * attn, dim=-1)  # [b, T_total, c]

        return out[:, :t, :] + x  # residual crop
