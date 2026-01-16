from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from pem.core.config import FactorType, ProjectConfig
from pem.doe.analyze import analyze_doe


class ModelPattern(str, Enum):
    pure_gp = "pure_gp"
    interaction_features_gp = "interaction_features_gp"
    trend_residual_gp = "trend_residual_gp"
    doe_screen_then_gp = "doe_screen_then_gp"


@dataclass
class PredictionResult:
    mean: pd.Series
    std: pd.Series


def _split_feature_columns(cfg: ProjectConfig) -> Tuple[List[str], List[str]]:
    numeric = [
        f.name
        for f in cfg.factors
        if f.type in (FactorType.continuous, FactorType.discrete)
    ]
    categorical = [f.name for f in cfg.factors if f.type == FactorType.categorical]
    return numeric, categorical


def make_preprocessor(
    *,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    interaction_features: bool = False,
) -> ColumnTransformer:
    numeric_steps = [("scaler", StandardScaler())]
    if interaction_features:
        numeric_steps.append(
            (
                "poly",
                PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            )
        )

    numeric_pipe = Pipeline(steps=numeric_steps)

    cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(numeric_cols)),
            ("cat", cat_pipe, list(categorical_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_kernel(
    n_dims: int,
    *,
    kind: str = "matern",
    nu: float = 2.5,
    init_lengthscales: Optional[np.ndarray] = None,
    noise_level: float = 1e-6,
) -> object:
    if init_lengthscales is None:
        init_lengthscales = np.ones(n_dims, dtype=float)
    else:
        init_lengthscales = np.asarray(init_lengthscales, dtype=float)
        if init_lengthscales.shape != (n_dims,):
            raise ValueError("init_lengthscales shape must be (n_dims,)")

    if kind == "matern":
        base = Matern(length_scale=init_lengthscales, nu=nu)
    elif kind == "rbf":
        base = RBF(length_scale=init_lengthscales)
    else:
        raise ValueError(f"Unknown kernel kind: {kind}")

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * base + WhiteKernel(
        noise_level=noise_level, noise_level_bounds=(1e-10, 1e1)
    )
    return kernel


def _init_lengthscales_from_doe_effects(
    df: pd.DataFrame,
    cfg: ProjectConfig,
    *,
    response: str,
    numeric_cols: Sequence[str],
    min_ls: float = 0.2,
    max_ls: float = 5.0,
) -> np.ndarray:
    """Heuristic: strong effect -> shorter lengthscale (more sensitivity)."""

    try:
        res = analyze_doe(df, cfg, response=response, max_interaction_order=1, factors=numeric_cols)
        tbl = res.coef_table
        tbl = tbl[tbl["term"].isin(numeric_cols)].copy()
        if tbl.empty or "effect" not in tbl.columns:
            return np.ones(len(numeric_cols), dtype=float)
        strength = tbl.set_index("term")["effect"].abs()
        # Normalize to 0..1
        s = strength.reindex(list(numeric_cols)).fillna(0.0).to_numpy()
        if np.allclose(s, 0):
            return np.ones(len(numeric_cols), dtype=float)
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        # Map strong (1) -> min_ls, weak (0) -> max_ls
        return max_ls - s * (max_ls - min_ls)
    except Exception:
        # Do not let heuristics break training.
        return np.ones(len(numeric_cols), dtype=float)


class BaseSurrogate:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseSurrogate":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> PredictionResult:
        raise NotImplementedError


class PureGPRSurrogate(BaseSurrogate):
    def __init__(
        self,
        cfg: ProjectConfig,
        *,
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str],
        kernel_kind: str = "matern",
        nu: float = 2.5,
        n_restarts_optimizer: int = 2,
        init_lengthscales: Optional[np.ndarray] = None,
    ):
        self.cfg = cfg
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)

        self.pre = make_preprocessor(
            numeric_cols=self.numeric_cols,
            categorical_cols=self.categorical_cols,
            interaction_features=False,
        )

        # We do not know transformed dimensionality until fit; build GP in fit.
        self.kernel_kind = kernel_kind
        self.nu = nu
        self.n_restarts_optimizer = n_restarts_optimizer
        self.init_lengthscales = init_lengthscales

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PureGPRSurrogate":
        Xt = self.pre.fit_transform(X)
        n_dims = Xt.shape[1]
        init_ls = self.init_lengthscales
        if init_ls is None:
            init_ls = np.ones(n_dims)
        if init_ls.shape != (n_dims,):
            init_ls = np.ones(n_dims)

        kernel = make_kernel(n_dims, kind=self.kernel_kind, nu=self.nu, init_lengthscales=init_ls)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=0,
        )
        self.gp.fit(Xt, y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> PredictionResult:
        Xt = self.pre.transform(X)
        mean, std = self.gp.predict(Xt, return_std=True)
        return PredictionResult(mean=pd.Series(mean), std=pd.Series(std))


class InteractionFeatureGPRSurrogate(PureGPRSurrogate):
    def __init__(
        self,
        cfg: ProjectConfig,
        *,
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str],
        kernel_kind: str = "matern",
        nu: float = 2.5,
        n_restarts_optimizer: int = 2,
    ):
        super().__init__(
            cfg,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            kernel_kind=kernel_kind,
            nu=nu,
            n_restarts_optimizer=n_restarts_optimizer,
        )
        self.pre = make_preprocessor(
            numeric_cols=list(numeric_cols),
            categorical_cols=list(categorical_cols),
            interaction_features=True,
        )


class TrendResidualGPRSurrogate(BaseSurrogate):
    """Linear(+interactions) trend model + GP on residuals."""

    def __init__(
        self,
        cfg: ProjectConfig,
        *,
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str],
        kernel_kind: str = "matern",
        nu: float = 2.5,
        n_restarts_optimizer: int = 2,
        ridge_alpha: float = 1.0,
    ):
        self.cfg = cfg
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)

        # Trend gets interaction features to capture global structure.
        self.pre_trend = make_preprocessor(
            numeric_cols=self.numeric_cols,
            categorical_cols=self.categorical_cols,
            interaction_features=True,
        )
        self.trend = Ridge(alpha=ridge_alpha)

        # GP uses no interaction expansion (it can learn local curvature). That said, you can
        # switch to interaction_features=True if desired.
        self.pre_gp = make_preprocessor(
            numeric_cols=self.numeric_cols,
            categorical_cols=self.categorical_cols,
            interaction_features=False,
        )

        self.kernel_kind = kernel_kind
        self.nu = nu
        self.n_restarts_optimizer = n_restarts_optimizer

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TrendResidualGPRSurrogate":
        Xt_trend = self.pre_trend.fit_transform(X)
        self.trend.fit(Xt_trend, y.to_numpy())
        trend_pred = self.trend.predict(Xt_trend)
        resid = y.to_numpy() - trend_pred

        Xt_gp = self.pre_gp.fit_transform(X)
        n_dims = Xt_gp.shape[1]
        kernel = make_kernel(n_dims, kind=self.kernel_kind, nu=self.nu)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=0,
        )
        self.gp.fit(Xt_gp, resid)
        return self

    def predict(self, X: pd.DataFrame) -> PredictionResult:
        Xt_trend = self.pre_trend.transform(X)
        trend_pred = self.trend.predict(Xt_trend)

        Xt_gp = self.pre_gp.transform(X)
        resid_mean, resid_std = self.gp.predict(Xt_gp, return_std=True)

        mean = trend_pred + resid_mean
        return PredictionResult(mean=pd.Series(mean), std=pd.Series(resid_std))


class DoeScreenThenGPRSurrogate(PureGPRSurrogate):
    def __init__(
        self,
        cfg: ProjectConfig,
        *,
        df_for_screening: pd.DataFrame,
        response: str,
        top_k: int = 5,
        kernel_kind: str = "matern",
        nu: float = 2.5,
        n_restarts_optimizer: int = 2,
    ):
        numeric_cols, categorical_cols = _split_feature_columns(cfg)

        # Use DoE effects on numeric cols only
        doe_res = analyze_doe(
            df_for_screening,
            cfg,
            response=response,
            max_interaction_order=1,
            factors=numeric_cols,
        )
        tbl = doe_res.coef_table
        tbl = tbl[tbl["term"].isin(numeric_cols)].copy()
        tbl["abs_effect"] = tbl.get("effect", tbl["coef"]).abs()
        tbl = tbl.sort_values("abs_effect", ascending=False)
        selected_numeric = tbl["term"].head(top_k).tolist()
        if not selected_numeric:
            selected_numeric = list(numeric_cols)

        # Heuristic init lengthscales based on screened effects.
        init_ls_numeric = _init_lengthscales_from_doe_effects(
            df_for_screening,
            cfg,
            response=response,
            numeric_cols=selected_numeric,
        )

        # We can't know transformed dims until fit; but we can pass numeric init and then
        # fall back to ones if mismatch.
        super().__init__(
            cfg,
            numeric_cols=selected_numeric,
            categorical_cols=categorical_cols,
            kernel_kind=kernel_kind,
            nu=nu,
            n_restarts_optimizer=n_restarts_optimizer,
            init_lengthscales=None,
        )
        self.selected_numeric = selected_numeric
        self._init_ls_numeric = init_ls_numeric

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DoeScreenThenGPRSurrogate":
        Xt = self.pre.fit_transform(X)
        n_dims = Xt.shape[1]

        # Attempt to map numeric init lengthscales into full transformed space.
        # This mapping is approximate; we only apply if no categoricals and no expansion.
        init_ls = np.ones(n_dims)
        if len(self.categorical_cols) == 0 and n_dims == len(self.selected_numeric):
            init_ls = self._init_ls_numeric

        kernel = make_kernel(n_dims, kind=self.kernel_kind, nu=self.nu, init_lengthscales=init_ls)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=0,
        )
        self.gp.fit(Xt, y.to_numpy())
        return self


@dataclass
class TrainedModelBundle:
    """Serializable container with one surrogate per response."""

    cfg: ProjectConfig
    pattern: ModelPattern
    responses: List[str]
    response_models: Dict[str, BaseSurrogate]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out: Dict[str, np.ndarray] = {}
        for r in self.responses:
            pred = self.response_models[r].predict(X)
            out[f"{r}__mean"] = pred.mean.to_numpy()
            out[f"{r}__std"] = pred.std.to_numpy()
        return pd.DataFrame(out)

    def save(self, path: str) -> None:
        dump(self, path)

    @staticmethod
    def load(path: str) -> "TrainedModelBundle":
        obj = load(path)
        if not isinstance(obj, TrainedModelBundle):
            raise TypeError("Loaded object is not a TrainedModelBundle")
        return obj


def train_bundle(
    df: pd.DataFrame,
    cfg: ProjectConfig,
    *,
    responses: Sequence[str],
    pattern: ModelPattern = ModelPattern.pure_gp,
    top_k: int = 5,
    kernel_kind: str = "matern",
    nu: float = 2.5,
    n_restarts_optimizer: int = 2,
    ridge_alpha: float = 1.0,
) -> TrainedModelBundle:
    numeric_cols, categorical_cols = _split_feature_columns(cfg)

    response_models: Dict[str, BaseSurrogate] = {}

    for r in responses:
        cfg.get_response(r)  # validate

        if pattern == ModelPattern.pure_gp:
            init_ls = _init_lengthscales_from_doe_effects(df, cfg, response=r, numeric_cols=numeric_cols)
            use_init_ls = init_ls if len(categorical_cols) == 0 else None
            model = PureGPRSurrogate(
                cfg,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                kernel_kind=kernel_kind,
                nu=nu,
                n_restarts_optimizer=n_restarts_optimizer,
                init_lengthscales=use_init_ls,
            )
        elif pattern == ModelPattern.interaction_features_gp:
            model = InteractionFeatureGPRSurrogate(
                cfg,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                kernel_kind=kernel_kind,
                nu=nu,
                n_restarts_optimizer=n_restarts_optimizer,
            )
        elif pattern == ModelPattern.trend_residual_gp:
            model = TrendResidualGPRSurrogate(
                cfg,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                kernel_kind=kernel_kind,
                nu=nu,
                n_restarts_optimizer=n_restarts_optimizer,
                ridge_alpha=ridge_alpha,
            )
        elif pattern == ModelPattern.doe_screen_then_gp:
            model = DoeScreenThenGPRSurrogate(
                cfg,
                df_for_screening=df,
                response=r,
                top_k=top_k,
                kernel_kind=kernel_kind,
                nu=nu,
                n_restarts_optimizer=n_restarts_optimizer,
            )
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        X = df[cfg.factor_names(include_categorical=True)].copy()
        y = pd.to_numeric(df[r], errors="coerce")
        keep = ~(X.isna().any(axis=1) | y.isna())
        X = X.loc[keep].reset_index(drop=True)
        y = y.loc[keep].reset_index(drop=True)

        response_models[r] = model.fit(X, y)

    return TrainedModelBundle(
        cfg=cfg,
        pattern=pattern,
        responses=list(responses),
        response_models=response_models,
    )
