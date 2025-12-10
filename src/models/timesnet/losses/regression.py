from __future__ import annotations

from collections.abc import Callable

import torch

from .composite import range_penalty_loss
from .regression import huber_loss, mae_loss, mape_loss, mse_loss, pinball_loss, smape_loss

# String → callable registry. Each callable returns a tensor loss given (pred, target, **kwargs).
# All losses must support broadcasting over [B, H, T] and reduce to a scalar (mean).
_LOSSES: dict[str, Callable[..., torch.Tensor]] = {
    "mae": mae_loss,
    "mse": mse_loss,
    "huber": huber_loss,
    "mape": mape_loss,
    "smape": smape_loss,
    "pinball": pinball_loss,
    "range_penalty": range_penalty_loss,  # special: needs (pred_high_idx, pred_low_idx)
}


def get_loss(name: str) -> Callable[..., torch.Tensor]:
    key = name.lower()
    if key not in _LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Available: {sorted(_LOSSES.keys())}")
    return _LOSSES[key]
