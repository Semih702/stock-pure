# src/preprocess/split_dataset.py
"""
Split a time-ordered CSV into train / val / test chronologically.

Usage:
  python src/preprocess/split_dataset.py --in-dir src/data/yahoo-finance --name apple.csv \
      --train 0.7 --val 0.15 --test 0.15 --date-col Date
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def split_dataset(
    in_dir: str | Path,
    name: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    date_col: str = "Date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-ordered CSV (in folder/name format) into train/val/test by chronological order.

    Saves to <in_dir>/splits/<dataset_name>/.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    in_dir = Path(in_dir)
    file_path = in_dir / name
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    total = train_ratio + val_ratio + test_ratio
    if not (0.999 <= total <= 1.001):
        raise ValueError(f"train+val+test must sum to 1.0 (got {total:.4f})")

    # Load and parse
    try:
        df = pd.read_csv(file_path, parse_dates=[date_col])
    except ValueError as e:
        df = pd.read_csv(file_path)
        if date_col not in df.columns:
            first_col = df.columns[0]
            try:
                df[first_col] = pd.to_datetime(df[first_col])
                date_col = first_col
            except Exception:
                raise ValueError(
                    f"Could not parse date column ('{date_col}' or '{first_col}')"
                ) from e

    if df[date_col].isna().any():
        df = df.dropna(subset=[date_col])

    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    if n < 3:
        raise ValueError(f"Not enough rows to split (got {n}).")

    # Compute indices
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    # Output paths
    dataset_name = Path(name).stem
    out_dir = in_dir / "splits" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / f"{dataset_name}_train.csv", index=False)
    val_df.to_csv(out_dir / f"{dataset_name}_val.csv", index=False)
    test_df.to_csv(out_dir / f"{dataset_name}_test.csv", index=False)

    print(f"✅ Split '{file_path.name}' ({n} rows) →")
    print(f"  Train: {len(train_df)}  | {out_dir}/{dataset_name}_train.csv")
    print(f"  Val:   {len(val_df)}  | {out_dir}/{dataset_name}_val.csv")
    print(f"  Test:  {len(test_df)}  | {out_dir}/{dataset_name}_test.csv")

    return train_df, val_df, test_df


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
    )
