#!/usr/bin/env python3
"""
Run both accuracy benches (stockpure + upstream), then summarize.

Usage (from repo root):
  python scripts/run_timesnet_bench.py \
      --upstream-root .cache/upstream_timesnet \
      --upstream-class models.timesnet.blocks:TimesBlock \
      --device cpu --epochs 3 --resample 1H

It will write:
  results/bench/timesnet_accuracy_stockpure.json
  results/bench/timesnet_accuracy_upstream.json
  results/bench/timesnet_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "bench"
SP_JSON = RESULTS_DIR / "timesnet_accuracy_stockpure.json"
UP_JSON = RESULTS_DIR / "timesnet_accuracy_upstream.json"
REPORT_MD = RESULTS_DIR / "timesnet_report.md"


def run(cmd: list[str]) -> None:
    print("»", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Expected output missing: {path}")
    return json.loads(path.read_text())


def format_row(d: dict[str, Any]) -> str:
    m = d["metrics"]["model"]
    return (
        f"| {d['impl']} | {d['device']} | {d['epochs']} | "
        f"{d['train_time_s']:.3f} | {d['infer_time_ms_mean']:.3f} ± {d['infer_time_ms_std']:.3f} | "
        f"{m['MAE']:.2f} | {m['RMSE']:.2f} | {m['MAPE']:.4f} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if os.getenv("CUDA_VISIBLE_DEVICES", "") else "cpu")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--context", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--resample", default="1H")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--skip-stockpure", action="store_true")
    ap.add_argument("--skip-upstream", action="store_true")
    ap.add_argument(
        "--upstream-root",
        type=Path,
        required=False,
        help="Path to original repo folder for upstream bench.",
    )
    ap.add_argument(
        "--upstream-class",
        default="models.timesnet.blocks:TimesBlock",
        help="Dotted path 'module:Class' to upstream TimesBlock.",
    )
    ap.add_argument("--write-report", action="store_true", help="Write merged Markdown report.")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Common args for both benches
    common_args = [
        "--resample",
        args.resample,
        "--context",
        str(args.context),
        "--horizon",
        str(args.horizon),
        "--train-ratio",
        str(args.train_ratio),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--device",
        args.device,
    ]

    # 1) Run stockpure bench (unless skipped)
    if not args.skip_stockpure:
        cmd_sp = [
            sys.executable,
            "-m",
            "bench.timesnet.bench_accuracy_stockpure",
            *common_args,
            "--out",
            str(SP_JSON),
        ]
        run(cmd_sp)
    else:
        print("› Skipping stockpure bench")

    # 2) Run upstream bench (unless skipped)
    if not args.skip_upstream:
        if not args.upstream_root:
            ap.error("--upstream-root is required unless --skip-upstream is set")
        cmd_up = [
            sys.executable,
            "-m",
            "bench.timesnet.bench_accuracy_upstream",
            "--upstream-root",
            str(args.upstream_root),
            "--upstream-class",
            args.upstream_class,
            *common_args,
            "--out",
            str(UP_JSON),
        ]
        run(cmd_up)
    else:
        print("› Skipping upstream bench")

    # 3) Read outputs
    rows = []
    if SP_JSON.exists():
        rows.append(read_json(SP_JSON))
    if UP_JSON.exists():
        rows.append(read_json(UP_JSON))

    if not rows:
        raise SystemExit("No bench outputs found.")

    # 4) Print summary to console
    header = (
        "| impl | device | epochs | train_time_s | infer_time_ms (mean±std) | MAE | RMSE | MAPE |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|"
    )
    print("\n=== TimesNet accuracy & timing ===")
    print(header)
    for d in rows:
        print(format_row(d))
    print()

    # 5) Optional merged Markdown report
    if args.write_report:
        REPORT_MD.write_text(
            "\n".join(
                [
                    "# TimesNet accuracy & timing (merged)",
                    f"- dataset: {rows[0]['dataset']} (resample={rows[0]['resample']})",
                    f"- context={rows[0]['context']}, horizon={rows[0]['horizon']}, "
                    f"train_ratio={rows[0]['train_ratio']}",
                    "",
                    header,
                    *[format_row(d) for d in rows],
                    "",
                ]
            )
        )
        print("Wrote", REPORT_MD)


if __name__ == "__main__":
    main()
