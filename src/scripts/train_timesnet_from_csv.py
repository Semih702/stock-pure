"""
Train a tiny TimesNet-based forecaster on a single CSV (e.g., Yahoo Finance OHLCV) with
15-minute resolution, 64-step input windows, and 16-step forecasts.

Steps:
1) Split the CSV into train/val/test chronologically (re-usable utility).
2) (Optional) Add MA/EMA indicators during split (with warmup to avoid NaNs).
3) Build sliding-window loaders from the splits.
4) Train TinyForecastSP and report simple metrics. Also plots training curves & predictions.

Examples:
  # Plain (15-minute bars, causal MA as residual baseline)
  python src/scripts/train_timesnet_from_csv.py \
    --in-dir src/data/yahoo-finance --name apple.csv \
    --context 64 --horizon 16 --epochs 3 --batch-size 64 --lr 1e-3 \
    --features Close Open High Low Volume --target Close --device cpu

  # With indicators computed by splitter (+ auto-append to features)
  python src/scripts/train_timesnet_from_csv.py \
    --in-dir src/data/yahoo-finance --name apple.csv \
    --context 64 --horizon 16 --epochs 50 --batch-size 128 --lr 1e-3 \
    --features Close Open High Low Volume --target Close \
    --ma 20 50 --ema 12 26 --warmup --append-indicators \
    --device cuda --k 4 --num-kernels 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sized
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure local src/ is on the path when running as a script (no package install required)
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# TimesNet block from your repo (keep as-is if your package is installed; otherwise use `from src.models...`)
from models.timesnet.blocks import TimesBlock
from models.ma import CausalWeightedMA, MAConfig

# Splitter (now supports indicators)
from preprocess.split_dataset import split_dataset

# Plotting utils (generic; no TimesNet dependency)
from vis.plots import (
    inv_scale_horizons,
    save_loss_curves,
    save_multi_horizon_curves,
    save_predictions_vs_actual,
    save_preds_csv,
    save_scatter_pred_vs_true,
)

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


def resample_to_15min(
    csv_path: Path, date_col: str = "Date", rule: str = "15min", save_suffix: str = "_15min"
) -> Path:
    """Resample a high-frequency CSV to 15-minute OHLCV(+VWAP) to satisfy causal 15-minute windows."""
    df_raw = pd.read_csv(csv_path, parse_dates=[date_col])
    df_raw = df_raw.sort_values(by=date_col)

    # Keep only numeric columns for aggregation; drop non-numeric to avoid mean on strings (e.g., symbols)
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError(f"No numeric columns found to resample in {csv_path}")
    dropped = [c for c in df_raw.columns if c not in numeric_cols + [date_col]]
    if dropped:
        print(f"[resample] Dropping non-numeric columns (kept only numeric for resample): {dropped}")

    df = df_raw[[date_col] + numeric_cols].set_index(date_col)

    # Base aggregations
    agg = {}
    for col in numeric_cols:
        lo = col.lower()
        if lo == "open":
            agg[col] = "first"
        elif lo == "high":
            agg[col] = "max"
        elif lo == "low":
            agg[col] = "min"
        elif lo == "close":
            agg[col] = "last"
        elif lo == "volume":
            agg[col] = "sum"
        elif lo == "vwap":
            # handled separately after resample to ensure volume weighting
            continue
        else:
            agg[col] = "mean"

    df_agg = df.resample(rule).agg(agg)

    # VWAP: volume-weighted using original VWAP * Volume if available, else mean
    vwap_col = next((c for c in df.columns if c.lower() == "vwap"), None)
    vol_col = next((c for c in df.columns if c.lower() == "volume"), None)
    if vwap_col is not None:
        if vol_col is not None:
            num = (df[vwap_col] * df[vol_col]).resample(rule).sum()
            den = df[vol_col].resample(rule).sum().replace({0: np.nan})
            df_agg[vwap_col] = num / den
        else:
            df_agg[vwap_col] = df[vwap_col].resample(rule).mean()

    df_agg = df_agg.dropna().reset_index().rename(columns={df_agg.index.name or "index": date_col})
    out_path = csv_path.with_name(f"{csv_path.stem}{save_suffix}.csv")
    df_agg.to_csv(out_path, index=False)
    print(f"[resample] Saved 15-min resampled CSV -> {out_path} (rows={len(df_agg)})")
    return out_path


# ---------------------------
# Config (single object for core assumptions)
# ---------------------------


@dataclass
class RunConfig:
    """Run-wide assumptions (15-minute resolution, causal MA, window sizes)."""

    freq: str = "15T"
    context: int = 64
    horizon: int = 16
    features: list[str] = field(
        default_factory=lambda: ["Close", "Open", "High", "Low", "Volume"]
    )
    target: str = "Close"
    k: int = 3
    num_kernels: int = 4
    epochs: int = 3
    batch_size: int = 64
    lr: float = 1e-3
    device: str = "cpu"
    seed: int = 42
    ma: MAConfig = field(default_factory=MAConfig)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RunConfig":
        ma_cfg = MAConfig(
            mode=args.ma_mode,
            weights=(args.ma_weights or []) if args.ma_mode != "none" else [],
            decay=args.ma_decay,
            decay_rate=args.ma_decay_rate,
        )
        cfg = cls(
            freq=args.freq,
            context=args.context,
            horizon=args.horizon,
            features=list(args.features),
            target=args.target,
            k=args.k,
            num_kernels=args.num_kernels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
            ma=ma_cfg,
        )
        cfg.validate()
        return cfg

    @property
    def target_idx(self) -> int:
        return self.features.index(self.target)

    def validate(self) -> None:
        # enforce 15-minute resolution explicitly
        if pd.to_timedelta(self.freq) != pd.to_timedelta("15min"):
            raise ValueError("This script enforces 15-minute resolution; use freq='15min'.")
        if self.context < 1 or self.horizon < 1:
            raise ValueError("context and horizon must be positive")
        if self.target not in self.features:
            raise ValueError(f"Target '{self.target}' must be included in --features.")
        if self.ma.mode != "none":
            _ = self.ma.normalized_weights()  # validates causal order + normalization
            if self.context < len(self.ma.weights):
                raise ValueError(
                    f"Context ({self.context}) must cover MA window ({len(self.ma.weights)}) "
                    "to avoid zero-padding the causal average."
                )

    def to_dict(self) -> dict:
        return {
            "freq": self.freq,
            "context": self.context,
            "horizon": self.horizon,
            "features": self.features,
            "target": self.target,
            "k": self.k,
            "num_kernels": self.num_kernels,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "device": self.device,
            "seed": self.seed,
            "ma": {
                "mode": self.ma.mode,
                "weights": self.ma.weights,
                "decay": self.ma.decay,
                "decay_rate": self.ma.decay_rate,
            },
        }


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


def _build_clean_windows(
    feats: np.ndarray,  # shape [T, C]
    target: np.ndarray,  # shape [T]
    context: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build ONLY those windows whose entire context and entire horizon are finite.
    Returns:
      x: [N, context, C], y: [N, horizon]
    """
    t, c = feats.shape
    n = t - context - horizon + 1
    if n <= 0:
        raise ValueError(f"Not enough rows ({t}) for context={context} and horizon={horizon}.")

    feats_finite: NDArray[np.bool_] = np.asarray(np.isfinite(feats).all(axis=1))
    targ_finite: NDArray[np.bool_] = np.asarray(np.isfinite(target))  # [T]

    keep_idxs = []
    for i in range(n):
        ctx_ok = feats_finite[i : i + context].all()
        # y spans rows [i+context, i+context+horizon)
        y_ok = targ_finite[i + context : i + context + horizon].all()
        if ctx_ok and y_ok:
            keep_idxs.append(i)

    if len(keep_idxs) == 0:
        raise ValueError("No valid windows after filtering (NaN/Inf in context/horizon).")

    x = np.stack([feats[i : i + context, :] for i in keep_idxs], axis=0).astype(np.float32)
    y = np.stack([target[i + context : i + context + horizon] for i in keep_idxs], axis=0).astype(
        np.float32
    )
    return torch.from_numpy(x), torch.from_numpy(y)


def build_loader_from_csv(
    csv_path: Path,
    features: list[str],
    context: int,
    horizon: int,
    target: str,
    batch_size: int,
    drop_last: bool = True,
    allow_empty: bool = False,
    dropna: bool = True,  # ← we keep this for backward-compat
    scaler_x: StandardScaler | None = None,  # ← add
    scaler_y: StandardScaler | None = None,  # ← add
    shuffle: bool = True,
) -> DataLoader | None:
    # 1) Load
    df = pd.read_csv(csv_path)

    # 2) Column checks
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' missing in {csv_path.name}")

    # 3) Coerce to numeric, ±Inf → NaN
    for col in set(features + [target]):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 4) Optionally drop rows that are bad in ANY feature/target
    if dropna:
        before = len(df)
        df.dropna(subset=list(set(features + [target])), inplace=True)
        dropped = before - len(df)
        if dropped > 0:
            print(f"[loader] {csv_path.name}: dropped {dropped} rows with NaN/Inf in selected cols")

    if len(df) == 0:
        if allow_empty:
            return None
        raise ValueError(f"No usable rows after NaN/Inf removal in {csv_path.name}")

    # 5) Arrays
    feats = df[features].to_numpy(copy=False)
    ycol = df[target].to_numpy(copy=False)

    # 6) Build ONLY clean windows (context + horizon finite)
    try:
        if scaler_x is not None:
            feats = scaler_x.transform(feats)
        if scaler_y is not None:
            ycol = scaler_y.transform(ycol.reshape(-1, 1)).ravel()

        x, y = _build_clean_windows(feats, ycol, context, horizon)
    except ValueError as e:
        if allow_empty:
            print(f"[loader] {csv_path.name}: {e} → returning None")
            return None
        raise

    # 7) Final safety
    if torch.isnan(x).any() or torch.isnan(y).any():
        raise ValueError(f"NaNs detected after windowing for {csv_path.name}")

    ds = TensorDataset(x, y)
    if len(ds) == 0 and allow_empty:
        return None

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def dataset_len(loader) -> int:
    """
    Returns the number of samples in loader.dataset.
    Raises if the dataset is not Sized.
    """
    ds = loader.dataset

    if not hasattr(ds, "__len__"):
        raise TypeError(
            f"Dataset of type {type(ds).__name__} does not implement __len__(). "
            "This is likely an IterableDataset, which is incompatible here."
        )

    # Tell type checker it's safe
    return len(cast(Sized, ds))


# ---------------------------
# Tiny TimesNet-based model
# ---------------------------


class TinyForecastSP(nn.Module):
    """
    Minimal wrapper: TimesBlock + optional causal MA head/baseline.
    Input:  [B, L, C]
    Output: [B, H]
    """

    def __init__(
        self,
        c: int,
        horizon: int,
        k: int = 3,
        num_kernels: int = 4,
        ma_module: CausalWeightedMA | None = None,
        ma_mode: Literal["none", "input", "residual"] = "none",
        target_idx: int = 0,
    ):
        super().__init__()
        if ma_module is None and ma_mode != "none":
            raise ValueError("ma_mode is set but no CausalWeightedMA provided.")

        self.ma_module = ma_module if ma_mode != "none" else None
        self.ma_mode = ma_mode if ma_module is not None else "none"
        self.target_idx = target_idx

        in_channels = c + (1 if self.ma_mode == "input" else 0)
        self.block = TimesBlock(
            d_model=in_channels, d_ff=2 * in_channels, k=k, num_kernels=num_kernels
        )
        self.head = nn.Conv1d(in_channels=in_channels, out_channels=horizon, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = torch.as_tensor(x)
        target_seq = xb[:, :, self.target_idx]

        if self.ma_module is not None and self.ma_mode == "input":
            ma_feat = self.ma_module.sequence_features(target_seq).unsqueeze(-1)  # [B,L,1]
            xb = torch.cat([xb, ma_feat], dim=-1)

        y = self.block(xb, total_len=xb.shape[1])  # [B,L,C(+1)]
        z = self.head(y.permute(0, 2, 1))  # [B,H,L]
        out = z[:, :, -1]  # [B,H]

        if self.ma_module is not None and self.ma_mode == "residual":
            baseline = self.ma_module.baseline(target_seq, horizon=out.shape[1])  # [B,H]
            out = out + baseline

        return out


# ---------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-dir", required=True, help="Folder containing the CSV (e.g., src/data/yahoo-finance)"
    )
    ap.add_argument("--name", required=True, help="CSV file name (e.g., apple.csv)")
    ap.add_argument(
        "--freq", default="15min", help="Data resolution (default: 15min for 15-minute)."
    )
    ap.add_argument("--date-col", default="Date", help="Datetime column name.")
    ap.add_argument(
        "--resample-15m",
        action="store_true",
        help="Resample input CSV to 15-minute OHLCV before splitting (for sub-15m data).",
    )

    # Model / train
    ap.add_argument("--context", type=int, default=64, help="Lookback window length.")
    ap.add_argument("--horizon", type=int, default=16, help="Forecast horizon length.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results/quickrun/timesnet_from_csv.json"))

    # TimesNet knobs
    ap.add_argument("--k", type=int, default=3, help="Top-k Fourier components for TimesNet.")
    ap.add_argument(
        "--num-kernels",
        type=int,
        default=4,
        help="Number of learned frequency kernels per TimesBlock.",
    )

    # Weighted MA control (causal, 15-minute lags)
    ap.add_argument(
        "--ma-mode",
        choices=["none", "input", "residual"],
        default="residual",
        help="Use weighted MA as extra input channel ('input') or additive baseline ('residual').",
    )
    ap.add_argument(
        "--ma-weights",
        type=float,
        nargs="*",
        default=[0.4, 0.25, 0.2, 0.1, 0.05],
        help="Lag weights for causal MA (order: t-1, t-2, ...).",
    )
    ap.add_argument(
        "--ma-decay",
        choices=["none", "exp", "linear"],
        default="exp",
        help="Decay strategy applied to MA weights as lag increases.",
    )
    ap.add_argument(
        "--ma-decay-rate",
        type=float,
        default=0.05,
        help="Decay strength (ignored if --ma-decay=none).",
    )

    # Data columns
    ap.add_argument(
        "--features",
        nargs="+",
        default=["Close", "Open", "High", "Low", "Volume"],
        help="Columns to use as model inputs (order matters).",
    )
    ap.add_argument(
        "--target", default="Close", help="Which column to forecast (must be in --features)."
    )

    # Split control
    ap.add_argument(
        "--skip-split",
        action="store_true",
        help="If set, assumes existing split CSVs under <in-dir>/splits/<name-stem>/",
    )

    # NEW: indicators passed to the splitter
    ap.add_argument(
        "--price-col", default="Close", help="Price column for indicators (default: Close)"
    )
    ap.add_argument(
        "--ma",
        type=int,
        nargs="*",
        default=[],
        help="Simple moving average windows, e.g. --ma 20 50",
    )
    ap.add_argument(
        "--ema",
        type=int,
        nargs="*",
        default=[],
        help="Exponential moving average windows, e.g. --ema 12 26",
    )
    ap.add_argument(
        "--warmup", action="store_true", help="Use previous split tail to warm up indicators"
    )
    ap.add_argument(
        "--append-indicators",
        action="store_true",
        help="Automatically append MA/EMA columns to --features",
    )

    args = ap.parse_args()

    # Sanity checks on args
    if any(col.endswith("_MA_future") for col in args.features):
        raise ValueError(
            "Future-based target columns (e.g., High_MA_future, Low_MA_future) "
            "must NOT be included in --features. "
            "They contain future information and would leak data into the model."
        )

    # If user requested indicators and wants them auto-added to features, extend features now
    indicator_cols: list[str] = []
    indicator_cols += [f"MA{w}" for w in (args.ma or [])]
    indicator_cols += [f"EMA{w}" for w in (args.ema or [])]
    if args.append_indicators and indicator_cols:
        # ensure no duplicates, keep original order then add indicators
        existing = list(dict.fromkeys(args.features))
        to_add = [col for col in indicator_cols if col not in existing]
        args.features = existing + to_add

    run_cfg = RunConfig.from_args(args)

    set_repro(run_cfg.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    in_dir = Path(args.in_dir)
    csv_path = in_dir / args.name
    dataset_name = csv_path.stem

    # Optionally resample to 15-minute OHLCV if the source is higher frequency
    if args.resample_15m:
        print(f"[resample] Resampling {csv_path} to 15-minute bars...")
        csv_path = resample_to_15min(csv_path, date_col=args.date_col, rule="15min")
        dataset_name = csv_path.stem
        in_dir = csv_path.parent

    # 1) Split (unless user already split)
    if not args.skip_split:
        print(
            f"🔧 Splitting {csv_path} into train/val/test..."
            + (" with indicators" if indicator_cols else "")
        )
        split_dataset(
            in_dir=in_dir,
            name=csv_path.name,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            date_col=args.date_col,
            # forward indicator options:
            price_col=args.price_col,
            ma_windows=args.ma,
            ema_windows=args.ema,
            warmup=args.warmup,
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

    train_df = pd.read_csv(train_csv)
    feat_cols = run_cfg.features
    tgt_col = run_cfg.target

    scaler_x = StandardScaler().fit(train_df[feat_cols].to_numpy())
    scaler_y = StandardScaler().fit(train_df[[tgt_col]].to_numpy())

    # If user did NOT warmup indicators, early rows may have NaNs → dropna=True handles it.
    dropna = not args.warmup

    dtr = build_loader_from_csv(
        train_csv,
        run_cfg.features,
        run_cfg.context,
        run_cfg.horizon,
        run_cfg.target,
        run_cfg.batch_size,
        drop_last=True,
        allow_empty=False,
        dropna=dropna,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        shuffle=True,
    )

    if dtr is None:
        raise ValueError("Train DataLoader is None. Did build_loader_from_csv() return empty?")

    dval = build_loader_from_csv(
        val_csv,
        run_cfg.features,
        run_cfg.context,
        run_cfg.horizon,
        run_cfg.target,
        run_cfg.batch_size,
        drop_last=False,
        allow_empty=True,
        dropna=dropna,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        shuffle=False,
    )
    dte = build_loader_from_csv(
        test_csv,
        run_cfg.features,
        run_cfg.context,
        run_cfg.horizon,
        run_cfg.target,
        run_cfg.batch_size,
        drop_last=False,
        allow_empty=True,
        dropna=dropna,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        shuffle=False,
    )

    c = len(run_cfg.features)
    device = run_cfg.device

    #  Debug info DEBUGGGGGGGGGG
    print(f"Using device: {device}")
    print("Train windows:", dataset_len(dtr))
    print("Val windows:", dataset_len(dval) if dval is not None else 0)
    print("Test windows:", dataset_len(dte) if dte is not None else 0)

    # 3) Model, opt, loss
    ma_module = CausalWeightedMA(run_cfg.ma) if run_cfg.ma.mode != "none" else None
    model = TinyForecastSP(
        c=c,
        horizon=run_cfg.horizon,
        k=run_cfg.k,
        num_kernels=run_cfg.num_kernels,
        ma_module=ma_module,
        ma_mode=run_cfg.ma.mode,
        target_idx=run_cfg.target_idx,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=run_cfg.lr)
    loss_fn = nn.L1Loss()

    # 4) Train (with per-epoch val for curves)
    t0 = time.perf_counter()
    model.train()
    train_l1_hist: list[float] = []
    val_l1_hist: list[float] = []

    for epoch in range(run_cfg.epochs):
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
        train_l1_hist.append(float(avg))

        # quick validation
        model.eval()
        with torch.no_grad():
            if dval is None:
                val_avg = float("nan")
            else:
                v_losses = []
                for xb, yb in dval:
                    xb, yb = xb.to(device), yb.to(device)
                    v_losses.append(loss_fn(model(xb), yb).item())
                val_avg = float(np.mean(v_losses)) if v_losses else float("nan")
        val_l1_hist.append(val_avg)
        model.train()

        print(f"epoch {epoch + 1}/{run_cfg.epochs} | train L1={avg:.4f} | val L1={val_avg:.4f}")

    train_time_s = time.perf_counter() - t0

    val_l1 = val_l1_hist[-1] if val_l1_hist else float("nan")

    # Test
    with torch.no_grad():
        model.eval()
        yh_list, yt_list = [], []
        if dte is not None:
            for xb, yb in dte:
                xb, yb = xb.to(device), yb.to(device)
                yh = model(xb)
                yh_list.append(yh.cpu())
                yt_list.append(yb.cpu())

    if len(yt_list) == 0:
        m = {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
        print("Test skipped (no windows).")
        yt_all: torch.Tensor | None = None
        yh_all: torch.Tensor | None = None
    else:
        yt_all = torch.cat(yt_list, 0)  # [N, H]
        yh_all = torch.cat(yh_list, 0)  # [N, H]
        m = metrics(yt_all, yh_all)
        print(f"Test metrics: MAE={m['MAE']:.4f} | RMSE={m['RMSE']:.4f} | MAPE={m['MAPE']:.4f}")

    # 6) Plots & lightweight artifacts
    out_dir = args.out.parent
    loss_png = save_loss_curves(
        train_l1_hist, val_l1_hist if val_l1_hist else None, out_dir=out_dir, title="L1 per epoch"
    )

    preds_png = multi_png = scatter_png = None
    preds_csv = out_dir / "test_predictions.csv"

    if yt_all is not None and yh_all is not None:
        # --- inverse-scale to original units ---
        yt_all_np = inv_scale_horizons(yt_all.numpy(), scaler_y)
        yh_all_np = inv_scale_horizons(yh_all.numpy(), scaler_y)

        # --- metrics on inverse-scaled arrays ---
        m = metrics(torch.from_numpy(yt_all_np), torch.from_numpy(yh_all_np))
        print(
            f"Test metrics (inv-scaled): MAE={m['MAE']:.4f} | RMSE={m['RMSE']:.4f} | MAPE={m['MAPE']:.4f}"
        )

        # --- h=0 arrays for plots ---
        y_true_h0 = yt_all_np[:, 0]
        y_pred_h0 = yh_all_np[:, 0]
        print("true h0 head:", y_true_h0[:10])
        print("pred h0 head:", y_pred_h0[:10])
        print("true mean/std:", y_true_h0.mean(), y_true_h0.std())
        print("pred mean/std:", y_pred_h0.mean(), y_pred_h0.std())

        # --- save CSV (original scale) ---
        save_preds_csv(yt_all_np, yh_all_np, preds_csv)

        # --- line plot ---
        preds_png = save_predictions_vs_actual(
            y_true_h0,
            y_pred_h0,
            out_dir=out_dir,
            title_suffix=f"(H=0, {dataset_name})",
            x_index=None,
        )

        # --- multi-horizon ---
        candidate_h = [0, 1, 3, 7]
        height = yt_all_np.shape[1]
        hs = [h for h in candidate_h if h < height]
        multi_png = save_multi_horizon_curves(
            yt_all_np,
            yh_all_np,
            horizons=hs,
            out_dir=out_dir,
            title_prefix=f"{csv_path.stem} (target={run_cfg.target})",
        )

        # --- scatter ---
        scatter_png = save_scatter_pred_vs_true(y_true_h0, y_pred_h0, out_dir=out_dir)

    # 7) Save a tiny report
    res = {
        "dataset": dataset_name,
        "file": str(csv_path),
        "in_dir": str(in_dir),
        "config": run_cfg.to_dict(),
        "features": run_cfg.features,
        "target": run_cfg.target,
        "context": run_cfg.context,
        "horizon": run_cfg.horizon,
        "epochs": run_cfg.epochs,
        "batch_size": run_cfg.batch_size,
        "lr": run_cfg.lr,
        "device": run_cfg.device,
        "k": run_cfg.k,
        "num_kernels": run_cfg.num_kernels,
        "train_time_s": float(train_time_s),
        "val_l1": float(val_l1),
        "metrics": m,
        "artifacts": {
            "loss_curve": str(loss_png),
            "preds_vs_actual": str(preds_png) if preds_png else "",
            "multi_horizon": str(multi_png) if multi_png else "",
            "scatter": str(scatter_png) if scatter_png else "",
            "test_predictions_csv": str(preds_csv) if preds_csv else "",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print("Wrote:", args.out)


if __name__ == "__main__":
    main()
