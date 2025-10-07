from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# optional HF deps (streaming avoids Arrow schema issues)
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None  # will error nicely if used


def set_repro(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_hf_bitcoin(resample: str | None) -> pd.DataFrame:
    if load_dataset is None:
        raise RuntimeError("Please install `datasets` and `pyarrow` for --dataset hf_bitcoin")

    from collections import deque

    ds_iter = load_dataset("Farmaanaa/bitcoin_price_timeseries", split="train", streaming=True)

    max_rows = 200_000  # keep last N rows
    buf = deque(maxlen=max_rows)
    for ex_any in ds_iter:
        ex = cast(dict[str, Any], ex_any)
        err_mode: Literal["raise", "ignore", "coerce"] = "coerce"
        ts = pd.to_datetime(str(ex.get("Timestamp", "")), utc=True, errors=err_mode)
        if pd.isna(ts):
            continue
        try:
            open_price = float(ex.get("Open", "nan"))
            high_price = float(ex.get("High", "nan"))
            low_price = float(ex.get("Low", "nan"))
            close_price = float(ex.get("Close", "nan"))
            volume = float(ex.get("Volume", "nan"))
        except Exception:
            continue
        buf.append((ts, open_price, high_price, low_price, close_price, volume))

    if not buf:
        raise RuntimeError("No rows read from dataset.")
    df = (
        pd.DataFrame(list(buf), columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        .set_index("Timestamp")
        .sort_index()
    )
    if resample:
        df = (
            df.resample(resample)
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna()
        )
    return df.reset_index()


def make_windows(x: np.ndarray, context: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    t, c = x.shape
    n = t - context - horizon + 1
    if n <= 0:
        raise ValueError("Not enough timesteps for given context/horizon")
    x = np.stack([x[i : i + context] for i in range(n)], 0).astype("float32")
    y = np.stack([x[i + context : i + context + horizon, 0] for i in range(n)], 0).astype("float32")
    return x, y


def standardize(train: np.ndarray, test: np.ndarray):
    mu = train.mean(axis=(0, 1), keepdims=True)
    sd = train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (train - mu) / sd, (test - mu) / sd, mu, sd


def build_loaders(
    feats: np.ndarray, context: int, horizon: int, train_ratio: float, batch_size: int
):
    x, y = make_windows(feats, context, horizon)
    n = x.shape[0]
    split = int(n * train_ratio)
    xtr, ytr, xte, yte = x[:split], y[:split], x[split:], y[split:]
    xtr, xte, *_ = standardize(xtr, xte)
    dtr = DataLoader(
        TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
        batch_size=batch_size,
        shuffle=True,
    )
    dte = DataLoader(
        TensorDataset(torch.from_numpy(xte), torch.from_numpy(yte)), batch_size=batch_size
    )
    return dtr, dte, x.shape  # (N, L, C)


@torch.no_grad()
def naive_last(x: torch.Tensor, h: int) -> torch.Tensor:
    return x[:, -1, 0].unsqueeze(1).repeat(1, h)


@torch.no_grad()
def seasonal_naive(x: torch.Tensor, h: int, season: int) -> torch.Tensor:
    seq_len = x.shape[1]
    if seq_len < season:
        return naive_last(x, h)
    base = x[:, -season:, 0]
    reps = (h + season - 1) // season
    return base.repeat(1, reps)[:, :h]


def metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    eps = 1e-8
    e = y_pred - y_true
    return {
        "MAE": e.abs().mean().item(),
        "RMSE": e.pow(2).mean().sqrt().item(),
        "MAPE": (e.abs() / (y_true.abs() + eps)).mean().item(),
    }
