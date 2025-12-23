# src/preprocess/split_dataset.py
"""
Split a time-ordered CSV into train / val / test chronologically.
Assumes 15-minute granularity (or multiples thereof) to keep forecasting windows causal.

Usage:
  python src/preprocess/split_dataset.py \
      --in-dir src/data/yahoo-finance --name apple.csv \
      --train 0.7 --val 0.15 --test 0.15 --date-col Date \
      --price-col Close --ma 20 50 --ema 12 26 --warmup
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------
# Indicator helpers
# ---------------------------


def _add_indicators(
    df: pd.DataFrame,
    price_col: str,
    ma_windows: list[int],
    ema_windows: list[int],
) -> pd.DataFrame:
    if price_col not in df.columns:
        raise ValueError(f"price column '{price_col}' not found in CSV (have: {list(df.columns)})")
    out = df.copy()
    for w in ma_windows:
        if w <= 0:
            raise ValueError(f"MA window must be > 0 (got {w})")
        out[f"MA{w}"] = out[price_col].rolling(window=w, min_periods=w).mean()
    for w in ema_windows:
        if w <= 0:
            raise ValueError(f"EMA window must be > 0 (got {w})")
        out[f"EMA{w}"] = out[price_col].ewm(span=w, adjust=False, min_periods=w).mean()
    return out


def _with_warmup(
    main_df: pd.DataFrame,
    prev_tail_df: pd.DataFrame | None,
    price_col: str,
    ma_windows: list[int],
    ema_windows: list[int],
) -> pd.DataFrame:
    """
    If prev_tail_df is provided, vertically stack its tail ahead of main_df to compute indicators,
    then drop the overlap before returning (so saved split has no leakage rows).
    """
    if not ma_windows and not ema_windows:
        return main_df

    max_win = max([*ma_windows, *ema_windows]) if (ma_windows or ema_windows) else 0
    lookback = max(0, max_win - 1)

    if prev_tail_df is None or lookback == 0:
        return _add_indicators(main_df, price_col, ma_windows, ema_windows)

    prev_tail = prev_tail_df.tail(lookback).copy()
    extended = pd.concat([prev_tail, main_df], axis=0, ignore_index=True)
    extended = _add_indicators(extended, price_col, ma_windows, ema_windows)

    # Drop the warm-up rows so the returned frame aligns to main_df only
    return extended.iloc[len(prev_tail) :].reset_index(drop=True)


def _assert_multiple_of_freq(df: pd.DataFrame, date_col: str, freq: str = "15min") -> None:
    """
    Ensure all time deltas are integer multiples of the desired freq.
    Allows gaps (e.g., overnight) as long as they are still multiples.
    """
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found in dataframe")
    deltas = df[date_col].diff().dropna()
    if deltas.empty:
        return
    step = pd.to_timedelta(freq)
    ratios = deltas / step  # float ratios
    ratios_np = np.asarray(ratios, dtype=float)
    bad_mask = np.abs(ratios_np - np.round(ratios_np)) > 1e-6
    if bad_mask.any():
        bad = ratios[bad_mask].head()
        raise ValueError(
            f"Non-{freq} aligned rows detected in {date_col}. "
            f"Examples of bad deltas (in units of {freq}): {bad.tolist()}"
        )

# ---------------------------
# Core split logic
# ---------------------------


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    for name, r in (
        ("train_ratio", train_ratio),
        ("val_ratio", val_ratio),
        ("test_ratio", test_ratio),
    ):
        if r < 0:
            raise ValueError(f"{name} must be >= 0, got {r}")
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {total} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )


def split_dataset(
    in_dir: str | Path,
    name: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    date_col: str = "Date",
    price_col: str = "Close",
    ma_windows: list[int] | None = None,
    ema_windows: list[int] | None = None,
    warmup: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-ordered CSV (in folder/name format) into train/val/test by chronological order.

    Saves to <in_dir>/splits/<csv_stem>/{*_train,val,test}.csv.
    Returns (train_df, val_df, test_df).
    """
    _validate_ratios(train_ratio, val_ratio, test_ratio)

    in_dir = Path(in_dir)
    csv_path = in_dir / name
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Load + parse date if present
    try:
        df = pd.read_csv(csv_path, parse_dates=[date_col])
    except Exception:
        df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(
            f"date_col '{date_col}' not found in {csv_path.name}. "
            f"Available columns: {list(df.columns)}"
        )

    # Sort ascending by time
    df = df.sort_values(by=date_col).reset_index(drop=True)
    _assert_multiple_of_freq(df, date_col, freq="15T")

    n = len(df)
    if n == 0:
        raise ValueError(f"No rows in input CSV: {csv_path}")

    # Compute sizes (ensure coverage of all rows and at least 1 train row if possible)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - (n_train + n_val)

    if n_train < 1:
        n_train = min(1, n)
        remaining = n - n_train
        if (val_ratio + test_ratio) > 0 and remaining > 0:
            n_val = int(round(remaining * (val_ratio / (val_ratio + test_ratio))))
            n_test = remaining - n_val
        else:
            n_val, n_test = 0, remaining

    print(f"[split_dataset] n_train={n_train}, n_test={n_test}")

    # Slice
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()

    # Indicators
    ma_windows = ma_windows or []
    ema_windows = ema_windows or []

    if ma_windows or ema_windows:
        if warmup:
            # compute with warmup (no leakage in returned frames)
            train_df_w = _with_warmup(train_df, None, price_col, ma_windows, ema_windows)
            val_df_w = _with_warmup(val_df, train_df, price_col, ma_windows, ema_windows)
            test_df_w = _with_warmup(test_df, val_df, price_col, ma_windows, ema_windows)
            train_df, val_df, test_df = train_df_w, val_df_w, test_df_w
        else:
            # strict: compute per split independently (initial rows will be NaN until window filled)
            train_df = _add_indicators(train_df, price_col, ma_windows, ema_windows)
            val_df = _add_indicators(val_df, price_col, ma_windows, ema_windows)
            test_df = _add_indicators(test_df, price_col, ma_windows, ema_windows)

    # Save
    dataset_name = csv_path.stem
    out_dir = in_dir / "splits" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / f"{dataset_name}_train.csv"
    val_path = out_dir / f"{dataset_name}_val.csv"
    test_path = out_dir / f"{dataset_name}_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(
        f"[split_dataset] Sizes → "
        f"total rows={n}, "
        f"train={len(train_df)}, "
        f"val={len(val_df)}, "
        f"test={len(test_df)}"
    )

    return train_df, val_df, test_df


# ---------------------------
# CLI
# ---------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Split a time-ordered CSV into train/val/test chronologically."
    )
    ap.add_argument("--in-dir", required=True, help="Input folder (e.g., src/data/yahoo-finance)")
    ap.add_argument("--name", required=True, help="CSV file name (e.g., apple.csv)")
    ap.add_argument("--train", type=float, default=0.7, help="Train ratio (default: 0.7)")
    ap.add_argument("--val", type=float, default=0.15, help="Validation ratio (default: 0.15)")
    ap.add_argument("--test", type=float, default=0.15, help="Test ratio (default: 0.15)")
    ap.add_argument("--date-col", default="Date", help="Datetime column name (default: Date)")

    # Indicators
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
        "--warmup",
        action="store_true",
        help="Use previous split tail to warm up indicators (no leakage in saved splits)",
    )
    return ap


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    split_dataset(
        in_dir=args.in_dir,
        name=args.name,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        date_col=args.date_col,
        price_col=args.price_col,
        ma_windows=args.ma,
        ema_windows=args.ema,
        warmup=args.warmup,
    )
