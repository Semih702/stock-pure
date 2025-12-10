from __future__ import annotations

from . import composite as _comp  # import the module, not functions
from . import regression as _reg  # import the module, not functions
from .types import LossFn

# Optionally help Pylance by being explicit about what's available.
# (Not required, but nice.)
_AVAILABLE = ["mae", "mse", "huber", "mape", "smape", "pinball", "range_penalty"]

# String → callable registry. Each callable returns a tensor loss given (pred, target, **kwargs).
# All losses must broadcast over [B,H,T] (or [B,H] or [B]) and reduce to a scalar.
_LOSSES: dict[str, LossFn] = {
    "mae": _reg.mae_loss,
    "mse": _reg.mse_loss,
    "huber": _reg.huber_loss,
    "mape": _reg.mape_loss,
    "smape": _reg.smape_loss,
    "pinball": _reg.pinball_loss,
    "range_penalty": _comp.range_penalty_loss,
}


def get_loss(name: str) -> LossFn:
    key = name.lower()
    try:
        return _LOSSES[key]
    except KeyError as error:
        raise ValueError(f"Unknown loss '{name}'. Available: {_AVAILABLE}") from error
