from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def save_parity_plot(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    out_path: str | Path,
    title: str,
) -> Path:
    """Save a parity plot (predicted vs observed).

    Parameters
    ----------
    y_true:
        Observed values.
    y_pred:
        Predicted mean values.
    y_std:
        Optional predictive standard deviation (plotted as vertical errorbars).
    out_path:
        Output image path.
    title:
        Plot title.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[keep]
    y_pred = y_pred[keep]
    if y_std is not None:
        y_std = np.asarray(y_std, dtype=float)[keep]

    if y_true.size == 0:
        raise ValueError("No finite points to plot")

    lo = float(np.nanmin(np.concatenate([y_true, y_pred])))
    hi = float(np.nanmax(np.concatenate([y_true, y_pred])))
    if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
        pad = 0.05 * (hi - lo)
        lo -= pad
        hi += pad

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    if y_std is not None:
        ax.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ms=4, alpha=0.8)
    else:
        ax.plot(y_true, y_pred, "o", ms=4, alpha=0.8)

    ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_residuals_plot(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    title: str,
) -> Path:
    """Save a residual plot (residual vs predicted)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[keep]
    y_pred = y_pred[keep]

    if y_true.size == 0:
        raise ValueError("No finite points to plot")

    resid = y_true - y_pred

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(y_pred, resid, "o", ms=4, alpha=0.8)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Observed - Predicted)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
