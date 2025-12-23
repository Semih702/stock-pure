from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as f


@dataclass
class MAConfig:
    """Causal weighted moving average settings (weights are lag-ordered, most recent first)."""

    mode: Literal["none", "input", "residual"] = "residual"
    weights: list[float] = field(
        default_factory=lambda: [0.4, 0.25, 0.2, 0.1, 0.05]
    )  # strictly decreasing, sum=1 after normalization
    decay: Literal["none", "exp", "linear"] = "exp"
    decay_rate: float = 0.05

    def normalized_weights(self) -> torch.Tensor:
        """Return causal coefficients (most-recent-first) that sum to 1."""
        if self.mode == "none":
            raise ValueError("MAConfig.normalized_weights() called with mode='none'")
        if len(self.weights) == 0:
            raise ValueError("MA weights must be non-empty when mode is not 'none'")

        base = torch.as_tensor(self.weights, dtype=torch.float32)
        if (base < 0).any():
            raise ValueError("MA weights must be non-negative")
        # enforce recent >= older to keep causality monotone
        if not torch.all(base[:-1] >= base[1:] - 1e-8):
            raise ValueError("MA weights must be non-increasing from recent to older lags")

        decay_factors = self._decay_factors(len(self.weights))
        coeffs = base * decay_factors
        total = coeffs.sum()
        if total <= 0:
            raise ValueError("MA weights/decay lead to zero sum; adjust decay_rate or weights")
        coeffs = coeffs / total

        # after decay + normalization, ensure ordering is still causal (recent >= older)
        if not torch.all(coeffs[:-1] >= coeffs[1:] - 1e-6):
            raise ValueError(
                "Decay produced non-causal ordering (newer weights must be >= older weights). "
                "Use a smaller decay_rate or provide monotone weights."
            )
        return coeffs

    def _decay_factors(self, n: int) -> torch.Tensor:
        if self.decay == "none" or self.decay_rate == 0.0:
            return torch.ones(n, dtype=torch.float32)
        idx = torch.arange(n, dtype=torch.float32)  # 0 = t-1 (most recent)
        if self.decay == "exp":
            return torch.exp(-self.decay_rate * idx)
        if self.decay == "linear":
            return 1.0 / (1.0 + self.decay_rate * idx)
        raise ValueError(f"Unknown decay strategy: {self.decay}")


class CausalWeightedMA(nn.Module):
    """Causal weighted moving average usable as input feature or residual baseline."""

    def __init__(self, cfg: MAConfig):
        super().__init__()
        self.cfg = cfg
        coeffs = cfg.normalized_weights()
        # store as buffer so it's moved with .to(device)
        self.register_buffer("coeffs", coeffs.view(1, 1, -1))

    @property
    def window(self) -> int:
        return int(self.coeffs.shape[-1])

    def sequence_features(self, seq: torch.Tensor) -> torch.Tensor:
        """Apply causal MA across the full context.

        seq: [B, T] (target history only). Output is [B, T] aligned to seq.
        Uses only past values (zero-padded on the left).
        """
        if seq.dim() != 2:
            raise ValueError(f"Expected 2D tensor [B,T], got {tuple(seq.shape)}")
        padded = f.pad(seq.unsqueeze(1), (self.window - 1, 0))  # left-pad
        # conv1d performs cross-correlation, so flip to align coeff[0] with most recent
        kernel = torch.flip(self.coeffs, dims=[2])
        out = f.conv1d(padded, kernel)  # [B,1,T]
        return out.squeeze(1)

    def baseline(self, seq: torch.Tensor, horizon: int) -> torch.Tensor:
        """Compute a horizon-wide baseline from the most recent context slice."""
        ma_series = self.sequence_features(seq)  # [B,T]
        latest = ma_series[:, -1].unsqueeze(1)  # [B,1]
        return latest.repeat(1, horizon)
