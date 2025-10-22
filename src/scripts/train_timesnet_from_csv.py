"""
Train a tiny TimesNet-based forecaster on a single CSV (e.g., Yahoo Finance OHLCV).

Steps:
1) Split the CSV into train/val/test chronologically (re-usable utility).
2) Build sliding-window loaders from the splits.
3) Train TinyForecastSP and report simple metrics.

Run:
  python src/scripts/train_timesnet_from_csv.py \
      --in-dir src/data/yahoo-finance --name apple.csv \
      --context 96 --horizon 24 --epochs 3 --batch-size 64 --lr 1e-3 \
      --features Close Open High Low Volume --target Close --device cpu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# TimesNet block from your repo (keep as-is if your package is installed; otherwise use `from src.models...`)
from models.timesnet.blocks import TimesBlock

# Use the generic splitter (import via src package path)
from preprocess.split_dataset import split_dataset

# ---------------------------
# Small utilities
# ---------------------------


def set_repro(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    eps = 1e-8
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
    mape = torch.mean(torch.abs((y_true - y_pred) / (y_true + eps))).item()
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def build_sliding_windows(
    arr: np.ndarray, context: int, horizon: int, target_col_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    t, c = arr.shape
    n = t - context - horizon + 1
    if n <= 0:
        raise ValueError(f"Not enough rows ({t}) for context={context} and horizon={horizon}.")
    x = np.zeros((n, context, c), dtype=np.float32)
    y = np.zeros((n, horizon), dtype=np.float32)
    for i in range(n):
        x[i] = arr[i : i + context, :]
        y[i] = arr[i + context : i + context + horizon, target_col_idx]
    return torch.from_numpy(x), torch.from_numpy(y)


def build_loader_from_csv(
    csv_path: Path,
    features: list[str],
    context: int,
    horizon: int,
    target: str,
    batch_size: int,
) -> DataLoader:
    df = pd.read_csv(csv_path, parse_dates=True)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' missing in {csv_path.name}")

    feats = df[features].astype("float32").to_numpy()
    target_idx = features.index(target)

    x, y = build_sliding_windows(feats, context, horizon, target_idx)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


# ---------------------------
# Tiny TimesNet-based model
# ---------------------------


class TinyForecastSP(nn.Module):
    """
    Minimal wrapper: TimesBlock → 1x1 Conv head → horizon vector.
    Input:  [B, L, C]
    Output: [B, H]
    """

    def __init__(self, c: int, horizon: int, k: int = 3, num_kernels: int = 4):
        super().__init__()
        self.block = TimesBlock(d_model=c, d_ff=2 * c, k=k, num_kernels=num_kernels)
        self.head = nn.Conv1d(in_channels=c, out_channels=horizon, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(torch.as_tensor(x), total_len=x.shape[1])  # [B,L,C]
        z = self.head(y.permute(0, 2, 1))  # [B,H,L]
        return z[:, :, -1]  # [B,H]


# ---------------------------
# Main
# ---------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-dir", required=True, help="Folder containing the CSV (e.g., src/data/yahoo-finance)"
    )
    ap.add_argument("--name", required=True, help="CSV file name (e.g., apple.csv)")
    ap.add_argument(
        "--features",
        nargs="+",
        default=["Close", "Open", "High", "Low", "Volume"],
        help="Columns to use as model inputs (order matters).",
    )
    ap.add_argument(
        "--target", default="Close", help="Which column to forecast (must be in --features)."
    )
    ap.add_argument("--context", type=int, default=96, help="Lookback window length.")
    ap.add_argument("--horizon", type=int, default=24, help="Forecast horizon length.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results/quickrun/timesnet_from_csv.json"))
    ap.add_argument(
        "--skip-split",
        action="store_true",
        help="If set, assumes existing split CSVs under <in-dir>/splits/<name-stem>/",
    )
    args = ap.parse_args()

    set_repro(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    in_dir = Path(args.in_dir)
    csv_path = in_dir / args.name
    dataset_name = csv_path.stem  # e.g., 'apple'

    # 1) Split (unless user already split)
    if not args.skip_split if hasattr(args, "skip_splt") else not args.skip_split:  # backward-safe
        print(f"🔧 Splitting {csv_path} into train/val/test...")
        split_dataset(
            in_dir=in_dir,
            name=args.name,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            date_col="Date",
        )
    else:
        print("⏭️  Skipping split; expecting pre-existing split CSVs.")

    # 2) Loaders from split CSVs
    split_root = in_dir / "splits" / dataset_name
    train_csv = split_root / f"{dataset_name}_train.csv"
    val_csv = split_root / f"{dataset_name}_val.csv"
    test_csv = split_root / f"{dataset_name}_test.csv"

    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        raise FileNotFoundError(
            f"Expected split files under {split_root}. Run without --skip-split first."
        )

    dtr = build_loader_from_csv(
        train_csv, args.features, args.context, args.horizon, args.target, args.batch_size
    )
    dval = build_loader_from_csv(
        val_csv, args.features, args.context, args.horizon, args.target, args.batch_size
    )
    dte = build_loader_from_csv(
        test_csv, args.features, args.context, args.horizon, args.target, args.batch_size
    )

    c = len(args.features)
    device = args.device

    # 3) Model, opt, loss
    model = TinyForecastSP(c=c, horizon=args.horizon).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    # 4) Train
    t0 = time.perf_counter()
    model.train()
    for epoch in range(args.epochs):
        tot = 0.0
        steps = 0
        for xb, yb in dtr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tot += loss.item()
            steps += 1
        avg = tot / max(1, steps)
        print(f"epoch {epoch + 1}/{args.epochs} | train L1={avg:.4f}")
    train_time_s = time.perf_counter() - t0

    # 5) Validation
    model.eval()
    with torch.no_grad():
        v_losses = []
        for xb, yb in dval:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            v_losses.append(loss_fn(pred, yb).item())
        val_l1 = float(np.mean(v_losses)) if v_losses else float("nan")
        print(f"val L1={val_l1:.4f}")

    # 6) Test metrics
    yh_list, yt_list = [], []
    with torch.no_grad():
        for xb, yb in dte:
            xb, yb = xb.to(device), yb.to(device)
            yh = model(xb)
            yh_list.append(yh.cpu())
            yt_list.append(yb.cpu())
    yt = torch.cat(yt_list, 0)
    yh = torch.cat(yh_list, 0)
    m = metrics(yt, yh)
    print(f"Test metrics: MAE={m['MAE']:.4f} | RMSE={m['RMSE']:.4f} | MAPE={m['MAPE']:.4f}")

    # 7) Save a tiny report
    res = {
        "dataset": dataset_name,
        "file": str(csv_path),
        "in_dir": str(in_dir),
        "features": args.features,
        "target": args.target,
        "context": args.context,
        "horizon": args.horizon,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": device,
        "train_time_s": float(train_time_s),
        "val_l1": val_l1,
        "metrics": m,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print("Wrote:", args.out)


if __name__ == "__main__":
    main()
