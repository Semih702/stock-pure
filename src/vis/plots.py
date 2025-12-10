# src/vis/plots.py
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

# Headless-safe backend (works on servers/HPC)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------
# Helpers
# ---------------------------


def _to_1d_float(x) -> np.ndarray:
    """Coerce to 1-D float64 numpy array (no copies if already fine)."""
    a = np.asarray(x)
    a = np.squeeze(a)
    a = a.astype(np.float64, copy=False)
    return a.ravel()


def _finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filter to indices where both arrays are finite."""
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def _finite(a: np.ndarray) -> np.ndarray:
    """Filter to finite entries only."""
    return a[np.isfinite(a)]


def _debug_stats(name: str, arr: np.ndarray) -> str:
    if arr.size == 0:
        return f"{name}: empty"
    f = _finite(arr)
    if f.size == 0:
        return f"{name}: no finite values (all NaN/Inf)"
    return f"{name}: len={len(arr)}, finite={len(f)}, min={np.min(f):.6g}, max={np.max(f):.6g}"


def _ensure_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Plots
# ---------------------------


def save_loss_curves(
    train_l1: Iterable[float],
    val_l1: Iterable[float] | None,
    out_dir: Path,
    fname: str = "loss_curve.png",
    title: str = "Training / Validation L1",
) -> Path:
    """
    Plot L1 per epoch. Keeps original epoch indices and masks only non-finite values,
    so even if you have a single valid point it will still be visible (markers added).
    Also writes a small .txt breadcrumb next to the image with basic stats.
    """
    _ensure_dir(out_dir)

    tr = _to_1d_float(list(train_l1))
    vr = _to_1d_float(list(val_l1)) if val_l1 is not None else None

    epochs = np.arange(1, len(tr) + 1, dtype=np.int64)

    plt.figure()
    ax = plt.gca()

    # --- Train ---
    if tr.size > 0:
        m_tr = np.isfinite(tr)
        if np.any(m_tr):
            ax.plot(epochs[m_tr], tr[m_tr], label="Train L1", marker="o")
        else:
            ax.plot([], [], label="Train L1 (no finite)")
    else:
        ax.text(0.5, 0.55, "No train epochs", ha="center", va="center", transform=ax.transAxes)

    # --- Val ---
    if vr is not None:
        m_vr = np.isfinite(vr)
        if vr.size > 0 and np.any(m_vr):
            # if val length differs from train (rare), make its own epoch axis
            epochs_v = np.arange(1, len(vr) + 1, dtype=np.int64)
            ax.plot(epochs_v[m_vr], vr[m_vr], label="Val L1", marker="o")
        else:
            ax.plot([], [], label="Val L1 (no finite)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 Loss")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    out_path = out_dir / fname
    plt.savefig(out_path, dpi=160)
    plt.close()

    # tiny debug breadcrumb
    (out_dir / (fname + ".txt")).write_text(
        "\n".join(
            [
                _debug_stats("train_l1", tr),
                _debug_stats("val_l1", vr if vr is not None else np.array([])),
            ]
        )
    )
    return out_path


def inv_scale_horizons(arr_scaled: np.ndarray, scaler) -> np.ndarray:
    no_rows, no_columns = arr_scaled.shape
    out = np.empty_like(arr_scaled, dtype=np.float64)
    for h in range(no_columns):
        out[:, h] = scaler.inverse_transform(arr_scaled[:, h].reshape(-1, 1)).ravel()
    return out


def save_predictions_vs_actual(
    y_true_h0: np.ndarray,
    y_pred_h0: np.ndarray,
    out_dir: Path,
    fname: str = "predictions_vs_actual.png",
    title_suffix: str = "",
    x_index: np.ndarray | None = None,  # <- NEW
) -> Path:
    _ensure_dir(out_dir)
    yt = _to_1d_float(y_true_h0)
    yp = _to_1d_float(y_pred_h0)
    yt, yp = _finite_pair(yt, yp)

    plt.figure()
    if yt.size == 0:
        plt.text(
            0.5,
            0.5,
            "No finite data to plot",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
    else:
        if x_index is not None and len(x_index) >= len(yt):
            x = np.asarray(x_index)[: len(yt)]
        else:
            x = np.arange(len(yt))
        plt.plot(x, yt, label="Actual (h=0)")
        plt.plot(x, yp, label="Predicted (h=0)")
    plt.xlabel("Time" if x_index is not None else "Test window index")
    plt.ylabel("Target value")
    plt.title(f"Predicted vs Actual {title_suffix}".strip())
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / fname
    plt.savefig(out_path, dpi=160)
    plt.close()

    (out_dir / (fname + ".txt")).write_text(
        "\n".join([_debug_stats("y_true_h0", yt), _debug_stats("y_pred_h0", yp)])
    )
    return out_path


def save_multi_horizon_curves(
    y_true: np.ndarray,  # [N, H]
    y_pred: np.ndarray,  # [N, H]
    horizons: Iterable[int],
    out_dir: Path,
    fname: str = "multi_horizon.png",
    title_prefix: str = "Predicted vs Actual",
    max_points: int = 500,
) -> Path:
    _ensure_dir(out_dir)
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    if yt.ndim != 2 or yp.ndim != 2 or yt.shape != yp.shape or yt.size == 0:
        plt.figure()
        plt.text(
            0.5,
            0.5,
            f"Bad shapes: yt{yt.shape}, yp{yp.shape}",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        out_path = out_dir / fname
        plt.savefig(out_path, dpi=160)
        plt.close()
        return out_path

    no_rows, no_columns = yt.shape
    horizons = [h for h in horizons if 0 <= h < no_columns] or [0]

    # Downsample in N dimension
    step = max(1, int(np.ceil(no_rows / max_points)))
    idx = np.arange(0, no_rows, step)

    plt.figure()
    any_plotted = False
    for h in horizons:
        a = yt[:, h]
        b = yp[:, h]
        a, b = _finite_pair(a, b)
        if a.size == 0:
            continue
        any_plotted = True
        plt.plot(idx[: len(a[idx])], a[idx], linestyle="--", label=f"Actual h={h}")
        plt.plot(idx[: len(b[idx])], b[idx], label=f"Pred h={h}")

    if not any_plotted:
        plt.text(
            0.5,
            0.5,
            "No finite horizons to plot",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )

    plt.xlabel("Test window index (downsampled)")
    plt.ylabel("Target value")
    plt.title(f"{title_prefix} [{', '.join(f'h={h}' for h in horizons)}]")
    plt.legend(ncol=2)
    plt.tight_layout()
    out_path = out_dir / fname
    plt.savefig(out_path, dpi=160)
    plt.close()

    lines = [f"N={no_rows}, H={no_columns}, step={step}, horizons={horizons}"]
    for h in horizons:
        a = yt[:, h]
        b = yp[:, h]
        a, b = _finite_pair(a, b)
        lines.append(_debug_stats(f"y_true h={h}", a))
        lines.append(_debug_stats(f"y_pred h={h}", b))
    (out_dir / (fname + ".txt")).write_text("\n".join(lines))
    return out_path


def save_scatter_pred_vs_true(
    y_true_h0: np.ndarray,
    y_pred_h0: np.ndarray,
    out_dir: Path,
    fname: str = "scatter_pred_vs_true.png",
    title: str = "Predicted vs True (h=0) Scatter",
) -> Path:
    _ensure_dir(out_dir)

    yt = _to_1d_float(y_true_h0)
    yp = _to_1d_float(y_pred_h0)
    yt, yp = _finite_pair(yt, yp)

    plt.figure(figsize=(7, 5))

    if yt.size == 0:
        plt.text(
            0.5,
            0.5,
            "No finite data to scatter",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
    else:
        errors = yp - yt

        sc = plt.scatter(
            yt,
            yp,
            c=errors,
            cmap="coolwarm",
            s=35,
            alpha=0.75,
            edgecolors="k",
            linewidths=0.4,
        )

        cbar = plt.colorbar(sc)
        cbar.set_label("Prediction Error (pred - true)")

        mn = float(min(np.min(yt), np.min(yp)))
        mx = float(max(np.max(yt), np.max(yp)))
        plt.plot([mn, mx], [mn, mx], "k--", linewidth=1.2)

    plt.xlabel("True Price")
    plt.ylabel("Predicted Price")
    plt.title(title)
    plt.tight_layout()

    out_path = out_dir / fname
    plt.savefig(out_path, dpi=160)
    plt.close()

    return out_path


def save_preds_csv(
    y_true: np.ndarray,  # [N, H]
    y_pred: np.ndarray,  # [N, H]
    out_csv: Path,
) -> Path:
    """
    Convenience: store a tidy CSV with h=0 only for quick inspection.
    Filters to finite rows so you don't see zeros-from-NaN anywhere.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    assert (
        yt.ndim == 2 and yp.ndim == 2 and yt.shape == yp.shape
    ), "y_true/y_pred must be [N,H] same shape"

    a = yt[:, 0]
    b = yp[:, 0]
    a, b = _finite_pair(a, b)

    df = pd.DataFrame(
        {
            "index": np.arange(len(a)),
            "y_true_h0": a,
            "y_pred_h0": b,
        }
    )
    df.to_csv(out_csv, index=False)

    out_txt = out_csv.with_suffix(out_csv.suffix + ".txt")
    out_txt.write_text("\n".join([_debug_stats("y_true_h0", a), _debug_stats("y_pred_h0", b)]))
    return out_csv
