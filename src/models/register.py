from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

_MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model class or builder function under a given name.

    Example:
        @register_model("timesnet")
        class TimesNetModel(nn.Module):
            ...
    """

    def deco(cls_or_fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        _MODEL_REGISTRY[name.lower()] = cls_or_fn
        return cls_or_fn

    return deco


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a registered model by name.

    Args:
        name: Model name (case-insensitive).
        **kwargs: Keyword arguments passed to the model constructor.

    Returns:
        An instance of the requested model.

    Raises:
        KeyError: if the model name is unknown.
    """
    key = name.lower()
    if key not in _MODEL_REGISTRY:
        known = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Known models: {known}")
    return _MODEL_REGISTRY[key](**kwargs)
