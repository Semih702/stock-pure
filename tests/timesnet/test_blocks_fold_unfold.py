from __future__ import annotations

from typing import cast

import pytest
import torch

from models.timesnet.blocks import TimesBlock
from models.timesnet.types import BTC


@pytest.mark.unit
@torch.no_grad()
@pytest.mark.parametrize(
    "b,t,c,period",
    [
        (2, 96, 4, 12),  # t multiple of period
        (3, 128, 2, 16),  # clean power-of-two case
        (1, 60, 3, 10),  # another exact multiple
    ],
)
def test_fold_unfold_identity_no_padding(b: int, t: int, c: int, period: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(b, t, c, device=device, dtype=torch.float32)
    blk = TimesBlock(d_model=c, d_ff=2 * c, k=1, num_kernels=1)

    y, rows = blk.fold_time(cast(BTC, x), period=period, total_len=t)
    x2 = blk.unfold_time(y, rows=rows, period=period, total_len=t, t_orig=t)

    # Exact structural ops → exact equality (no padding)
    assert x2.shape == x.shape
    assert torch.equal(x2, x)


@pytest.mark.unit
@torch.no_grad()
@pytest.mark.parametrize(
    argnames=["b", "t", "c", "period", "total_len"],
    argvalues=[
        (2, 95, 4, 12, 95),  # not a multiple; total_len == t
        (2, 95, 4, 12, 100),  # not a multiple; total_len > t (forces extra padding)
        (1, 123, 3, 7, 130),  # generic odd sizes
    ],
)
def test_fold_unfold_identity_with_padding(b: int, t: int, c: int, period: int, total_len: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(b, t, c, device=device, dtype=torch.float32)
    blk = TimesBlock(d_model=c, d_ff=2 * c, k=1, num_kernels=1)

    y, rows = blk.fold_time(cast(BTC, x), period=period, total_len=total_len)
    # Unfold back and crop to original t
    x2 = blk.unfold_time(y, rows=rows, period=period, total_len=total_len, t_orig=t)

    assert x2.shape == x.shape
    # Pure reshape/permute with zero-padding & crop → exact equality of original region
    assert torch.equal(x2, x)

    # Sanity on folded shape: [b, c, rows, period]
    assert y.shape[0] == b and y.shape[1] == c and y.shape[3] == max(int(period), 1)
    # rows should be ceil(total_len / period)
    p = max(int(period), 1)
    target_len = max(total_len, t)
    expected_rows = (target_len + p - 1) // p  # ceil division without floats
    assert y.shape[2] == expected_rows == rows


@pytest.mark.unit
@torch.no_grad()
def test_fold_time_preserves_dtype_and_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 50, 3, device=device, dtype=torch.float32)
    blk = TimesBlock(d_model=3, d_ff=6, k=1, num_kernels=1)

    y, rows = blk.fold_time(cast(BTC, x), period=12, total_len=55)
    assert y.device == x.device
    assert y.dtype == x.dtype
    assert rows >= 1
