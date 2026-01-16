from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pem.core.config import ProjectConfig
from pem.model.surrogate import ModelPattern, train_bundle


@dataclass(frozen=True)
class MetricSummary:
    response: str
    rmse: float
    mae: float
    r2: float


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def cross_validate(
    df: pd.DataFrame,
    cfg: ProjectConfig,
    *,
    responses: Sequence[str],
    pattern: ModelPattern,
    n_splits: int = 5,
    seed: int = 0,
    top_k: int = 5,
    kernel_kind: str = "matern",
    nu: float = 2.5,
    n_restarts_optimizer: int = 2,
    ridge_alpha: float = 1.0,
) -> List[MetricSummary]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    responses = list(responses)
    for r in responses:
        cfg.get_response(r)

    Xcols = cfg.factor_names(include_categorical=True)
    all_idx = np.arange(len(df))

    # Accumulate predictions per response
    preds: Dict[str, List[float]] = {r: [] for r in responses}
    truth: Dict[str, List[float]] = {r: [] for r in responses}

    for train_idx, test_idx in kf.split(all_idx):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        bundle = train_bundle(
            train_df,
            cfg,
            responses=responses,
            pattern=pattern,
            top_k=top_k,
            kernel_kind=kernel_kind,
            nu=nu,
            n_restarts_optimizer=n_restarts_optimizer,
            ridge_alpha=ridge_alpha,
        )

        X_test = test_df[Xcols].copy()
        pred_df = bundle.predict(X_test)

        for r in responses:
            y_true = pd.to_numeric(test_df[r], errors="coerce").to_numpy()
            y_pred = pred_df[f"{r}__mean"].to_numpy()

            # Drop NA pairs
            keep = np.isfinite(y_true) & np.isfinite(y_pred)
            truth[r].extend(y_true[keep].tolist())
            preds[r].extend(y_pred[keep].tolist())

    summaries: List[MetricSummary] = []
    for r in responses:
        y_true = np.asarray(truth[r], dtype=float)
        y_pred = np.asarray(preds[r], dtype=float)
        summaries.append(
            MetricSummary(
                response=r,
                rmse=rmse(y_true, y_pred),
                mae=mae(y_true, y_pred),
                r2=r2(y_true, y_pred),
            )
        )

    return summaries
