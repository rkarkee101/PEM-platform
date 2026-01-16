from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def save_pareto_effects_plot(
    coef_table: pd.DataFrame,
    *,
    response: str,
    out_path: str | Path,
    top_n: int = 20,
    p_value_threshold: float = 0.05,
    title: Optional[str] = None,
) -> Path:
    """Save a Pareto-style chart of absolute effects.

    Parameters
    ----------
    coef_table:
        DataFrame returned by :func:`pem.doe.analyze.analyze_doe` as
        DoeAnalysisResult.coef_table.
    response:
        Name of the response.
    out_path:
        Output image path (PNG recommended).
    top_n:
        Plot only the top-N effects by magnitude.
    p_value_threshold:
        If p-values are present, annotate significant terms.
    """

    # Local import so the core package does not hard-require matplotlib at import time.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    if coef_table.empty:
        raise ValueError("coef_table is empty")

    df = coef_table.copy()

    if "term" not in df.columns:
        raise ValueError("coef_table must include a 'term' column")

    # Prefer 'effect' (DoE convention for -1/+1 coding), fallback to 'coef'.
    if "effect" in df.columns:
        effect_col = "effect"
    elif "coef" in df.columns:
        effect_col = "coef"
    else:
        raise ValueError("coef_table must include either 'effect' or 'coef'")

    df = df[df["term"] != "Intercept"].copy()
    df["abs_effect"] = pd.to_numeric(df[effect_col], errors="coerce").abs()
    df = df.dropna(subset=["abs_effect"]).sort_values("abs_effect", ascending=False)

    if top_n > 0:
        df = df.head(top_n)

    # Plot (largest at top)
    df = df.sort_values("abs_effect", ascending=True)

    fig_h = max(3.5, 0.35 * len(df) + 1.0)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    ax.barh(df["term"], df["abs_effect"])
    ax.set_xlabel(f"|{effect_col}|")
    ax.set_ylabel("term")

    t = title or f"Pareto of effects: {response}"
    ax.set_title(t)

    # Annotate significance if p-values exist.
    if "p_value" in df.columns:
        pvals = pd.to_numeric(df["p_value"], errors="coerce")
        for i, (term, val, pv) in enumerate(zip(df["term"], df["abs_effect"], pvals)):
            if np.isfinite(pv) and pv <= p_value_threshold:
                ax.text(
                    float(val),
                    i,
                    "  *",
                    va="center",
                )

        ax.text(
            0.99,
            0.01,
            f"* p <= {p_value_threshold:g}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
