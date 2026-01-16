from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import FactorType, ProjectConfig


@dataclass(frozen=True)
class DatasetSplit:
    X: pd.DataFrame
    y: pd.DataFrame


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}. Use .csv or .parquet")


def validate_dataset(df: pd.DataFrame, cfg: ProjectConfig, *, strict: bool = True) -> None:
    """Validate that required factor/response columns exist and basic types are sane.

    This is intentionally pragmatic and lightweight (not a full data quality system).
    """

    missing: List[str] = []
    for col in cfg.factor_names(include_categorical=True) + cfg.response_names():
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Validate factor types
    for factor in cfg.factors:
        series = df[factor.name]
        if factor.type in (FactorType.continuous, FactorType.discrete):
            # Try to coerce to numeric
            coerced = pd.to_numeric(series, errors="coerce")
            if coerced.isna().any() and strict:
                bad_rows = coerced[coerced.isna()].index[:10].tolist()
                raise ValueError(
                    f"Factor '{factor.name}' must be numeric. Example bad rows: {bad_rows}"
                )
        elif factor.type == FactorType.categorical:
            # Ensure values in allowed levels
            allowed = set(map(str, factor.levels or []))
            values = series.astype(str)
            unknown = sorted(set(values.unique()) - allowed)
            if unknown and strict:
                raise ValueError(
                    f"Categorical factor '{factor.name}' contains unknown levels: {unknown}. "
                    f"Allowed: {sorted(allowed)}"
                )

    # Validate responses are numeric
    for response in cfg.responses:
        coerced = pd.to_numeric(df[response.name], errors="coerce")
        if coerced.isna().any() and strict:
            bad_rows = coerced[coerced.isna()].index[:10].tolist()
            raise ValueError(
                f"Response '{response.name}' must be numeric. Example bad rows: {bad_rows}"
            )


def split_X_y(
    df: pd.DataFrame,
    cfg: ProjectConfig,
    responses: Sequence[str],
    *,
    drop_na: bool = True,
) -> DatasetSplit:
    responses = list(responses)
    for r in responses:
        cfg.get_response(r)  # validate

    X = df[cfg.factor_names(include_categorical=True)].copy()
    y = df[list(responses)].copy()

    if drop_na:
        keep = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
        X = X.loc[keep].reset_index(drop=True)
        y = y.loc[keep].reset_index(drop=True)

    return DatasetSplit(X=X, y=y)


def code_2level_numeric(series: pd.Series, low: float, high: float) -> pd.Series:
    """Map a numeric series to coded units in [-1, +1] based on low/high."""

    s = pd.to_numeric(series, errors="coerce")
    mid = (low + high) / 2.0
    half_range = (high - low) / 2.0
    if half_range == 0:
        raise ValueError("Cannot code with zero range.")
    return (s - mid) / half_range


def to_coded_dataframe(
    df: pd.DataFrame,
    cfg: ProjectConfig,
    *,
    factors: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return a coded dataframe for DoE-style regression.

    - Continuous/discrete numeric factors are coded to [-1,+1] based on bounds/levels.
    - Categorical factors are dropped by default (not meaningful for 2-level coding).
    """

    factors = list(factors) if factors is not None else cfg.factor_names(include_categorical=False)

    coded = {}
    for name in factors:
        factor = cfg.get_factor(name)
        if factor.type == FactorType.categorical:
            continue
        lo, hi = factor.low_high()
        coded[name] = code_2level_numeric(df[name], lo, hi)

    return pd.DataFrame(coded)
