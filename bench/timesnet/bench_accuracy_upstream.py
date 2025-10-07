from __future__ import annotations

import argparse
import json
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .common import (
    build_loaders,
    load_hf_bitcoin,
    metrics,
    naive_last,
    seasonal_naive,
    set_repro,
    sync,
)


def build_upstream_timesblock(
    upstream_root: Path, dotted: str, c: int, k=3, num_kernels=4
) -> nn.Module:
    """
    Import upstream TimesBlock dynamically: 'module.path:ClassName'
    e.g. 'models.timesnet.blocks:TimesBlock'
    """
    import sys

    sys.path.insert(0, str(upstream_root))
    mod_path, cls_name = dotted.split(":")
    mod = import_module(mod_path)
    times_block = getattr(mod, cls_name)
    return times_block(d_model=c, d_ff=2 * c, k=k, num_kernels=num_kernels)


class TinyForecastUP(nn.Module):
    def __init__(self, block: nn.Module, c: int, horizon: int):
        super().__init__()
        self.block = block
        self.head = nn.Conv1d(in_channels=c, out_channels=horizon, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x, total_len=x.shape[1])  # upstream signature
        z = self.head(y.permute(0, 2, 1))
        return z[:, :, -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--upstream-root",
        type=Path,
        required=True,
        help="Path to the original repo (folder, not pip-installed).",
    )
    ap.add_argument(
        "--upstream-class",
        default="models.timesnet.blocks:TimesBlock",
        help="Dotted path 'module:Class' to upstream TimesBlock.",
    )
    ap.add_argument("--resample", default="1H")
    ap.add_argument("--context", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--out", type=Path, default=Path("results/bench/timesnet_accuracy_upstream.json")
    )
    args = ap.parse_args()

    set_repro(42)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Data (HF bitcoin)
    df = load_hf_bitcoin(args.resample or None)
    feats = df[["Close", "Open", "High", "Low", "Volume"]].astype("float32").to_numpy()

    dtr, dte, shape = build_loaders(
        feats, args.context, args.horizon, args.train_ratio, args.batch_size
    )
    c = shape[2]
    device = args.device

    block = build_upstream_timesblock(args.upstream_root, args.upstream_class, c=c)
    model = TinyForecastUP(block, c=c, horizon=args.horizon).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    # Train timing
    t0 = time.perf_counter()
    model.train()
    for _ in range(args.epochs):
        for xb, yb in dtr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    sync()
    train_time_s = time.perf_counter() - t0

    # Inference timing + metrics
    inf_times = []
    yh_list, yt_list, naive_list, snaive_list = [], [], [], []
    season = 24 if (args.resample and args.resample.upper().endswith("H")) else args.context
    model.eval()
    with torch.no_grad():
        for xb, yb in dte:
            xb, yb = xb.to(device), yb.to(device)
            sync()
            t1 = time.perf_counter()
            yh = model(xb)
            sync()
            inf_times.append((time.perf_counter() - t1) * 1e3)
            yh_list.append(yh.cpu())
            yt_list.append(yb.cpu())
            naive_list.append(naive_last(xb, args.horizon).cpu())
            snaive_list.append(seasonal_naive(xb, args.horizon, season).cpu())

    yt = torch.cat(yt_list, 0)
    yh = torch.cat(yh_list, 0)
    yn = torch.cat(naive_list, 0)
    ys = torch.cat(snaive_list, 0)

    res = {
        "impl": "upstream",
        "dataset": "hf_bitcoin",
        "resample": args.resample,
        "context": args.context,
        "horizon": args.horizon,
        "train_ratio": args.train_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": device,
        "train_time_s": float(train_time_s),
        "infer_time_ms_mean": float(np.mean(inf_times)),
        "infer_time_ms_std": float(np.std(inf_times)),
        "metrics": {
            "model": metrics(yt, yh),
            "naive": metrics(yt, yn),
            "seasonal_naive": metrics(yt, ys),
        },
    }
    args.out.write_text(json.dumps(res, indent=2))

    # Markdown report
    md = args.out.with_suffix(".md")
    md.write_text(
        "\n".join(
            [
                "# TimesNet (upstream) — accuracy & timing",
                f"- dataset: hf_bitcoin (resample={args.resample})",
                f"- context={args.context}, horizon={args.horizon}, epochs={args.epochs}, device={device}",
                f"- train_time_s: {res['train_time_s']:.3f}",
                f"- infer_time_ms: mean={res['infer_time_ms_mean']:.3f}, std={res['infer_time_ms_std']:.3f}",
                "",
                "| model | MAE | RMSE | MAPE |",
                "|---|---:|---:|---:|",
                f"| upstream | {res['metrics']['model']['MAE']:.4f} | {res['metrics']['model']['RMSE']:.4f} | {res['metrics']['model']['MAPE']:.4f} |",
                f"| naive | {res['metrics']['naive']['MAE']:.4f} | {res['metrics']['naive']['RMSE']:.4f} | {res['metrics']['naive']['MAPE']:.4f} |",
                f"| seasonal_naive | {res['metrics']['seasonal_naive']['MAE']:.4f} | {res['metrics']['seasonal_naive']['RMSE']:.4f} |",
                "",
            ]
        )
    )
    print("Wrote:", args.out, md)


if __name__ == "__main__":
    main()
