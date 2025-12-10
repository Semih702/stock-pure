from __future__ import annotations

import torch


def range_penalty_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    pred_high_idx: int,
    pred_low_idx: int,
    target_high_idx: int | None = None,
    target_low_idx: int | None = None,
    reduction: str = "mean",
    **_,
) -> torch.Tensor:
    """
    Penalize (High-Low) range mismatch. Works on multi-target outputs [B,H,T].
    By default assumes the same indices for pred/target; override if needed.

    Args:
      pred, target: [B,H,T]
      pred_high_idx, pred_low_idx: indices in last dim for predicted High/Low
      target_high_idx, target_low_idx: if labels are in different order (optional)
      reduction: 'mean' or 'none'
    """
    if target_high_idx is None:
        target_high_idx = pred_high_idx
    if target_low_idx is None:
        target_low_idx = pred_low_idx

    p_range = pred[..., pred_high_idx] - pred[..., pred_low_idx]
    t_range = target[..., target_high_idx] - target[..., target_low_idx]
    diff = (p_range - t_range) ** 2
    return diff.mean() if reduction == "mean" else diff
