from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pem.core.config import FactorType, ProjectConfig, ResponseGoal
from pem.doe.design import latin_hypercube
from pem.model.surrogate import TrainedModelBundle


@dataclass(frozen=True)
class RecommendationResult:
    candidates: pd.DataFrame
    ranked: pd.DataFrame


def _random_categorical(levels: Sequence[str], n: int, rng: np.random.Generator) -> List[str]:
    lv = list(map(str, levels))
    idx = rng.integers(0, len(lv), size=n)
    return [lv[i] for i in idx]


def generate_candidates(
    cfg: ProjectConfig,
    *,
    n_candidates: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a mixed-type candidate set.

    - Numeric factors: Latin hypercube in bounds.
    - Categorical factors: uniform random sampling over allowed levels.
    """

    rng = np.random.default_rng(seed)

    numeric_factors = [f for f in cfg.factors if f.type in (FactorType.continuous, FactorType.discrete)]
    cat_factors = [f for f in cfg.factors if f.type == FactorType.categorical]

    if numeric_factors:
        # latin_hypercube uses cfg.include_in_design numeric factors by default, but for
        # optimization we want all numeric factors.
        # We temporarily mark include_in_design=True for numeric factors by building a shallow copy.
        tmp_cfg = cfg.model_copy(deep=True)
        for f in tmp_cfg.factors:
            f.include_in_design = f.type != FactorType.categorical
        X_num = latin_hypercube(tmp_cfg, n_samples=n_candidates, seed=seed)
    else:
        X_num = pd.DataFrame(index=range(n_candidates))

    for f in cat_factors:
        X_num[f.name] = _random_categorical(f.levels or [], n_candidates, rng)

    # Ensure column order matches config order
    ordered_cols = cfg.factor_names(include_categorical=True)
    return X_num[ordered_cols]


def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    best: float,
    *,
    maximize: bool,
) -> np.ndarray:
    """Standard EI for scalar objective."""

    from math import erf, sqrt

    sigma = np.maximum(sigma, 1e-12)

    if maximize:
        imp = mu - best
    else:
        imp = best - mu

    z = imp / sigma

    # Normal PDF and CDF (no scipy dependency)
    pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
    cdf = 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))

    ei = imp * cdf + sigma * pdf
    ei[sigma <= 1e-12] = 0.0
    return ei


def ucb(mu: np.ndarray, sigma: np.ndarray, *, kappa: float = 2.0) -> np.ndarray:
    return mu + kappa * sigma


def _utility_for_response(
    cfg: ProjectConfig,
    response: str,
    mean: np.ndarray,
    std: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Convert a response prediction into a 'utility' that we want to maximize.

    Returns (utility_mean, utility_std, supports_ei)

    - minimize: utility = -mean (EI supported)
    - maximize: utility = +mean (EI supported)
    - target: utility = -abs(mean-target) (EI not recommended; use UCB)
    """

    r = cfg.get_response(response)
    if r.goal == ResponseGoal.minimize:
        return -mean, std, True
    if r.goal == ResponseGoal.maximize:
        return mean, std, True

    # target
    assert r.target is not None
    util = -np.abs(mean - float(r.target))
    return util, std, False


def _scalarize(
    cfg: ProjectConfig,
    pred_df: pd.DataFrame,
    *,
    responses: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Convert multi-response predictions into a scalar utility (mean/std).

    Current method: weighted sum of per-response utilities.
    """

    weights: Dict[str, float] = dict(cfg.scalarization.weights)
    if not weights:
        # default: equal weight
        weights = {r: 1.0 for r in responses}

    util_mean_total = 0.0
    util_var_total = 0.0
    all_support_ei = True

    for r in responses:
        w = float(weights.get(r, 0.0))
        mean = pred_df[f"{r}__mean"].to_numpy(dtype=float)
        std = pred_df[f"{r}__std"].to_numpy(dtype=float)
        u_mean, u_std, supports_ei = _utility_for_response(cfg, r, mean, std)
        util_mean_total = util_mean_total + w * u_mean
        util_var_total = util_var_total + (w * u_std) ** 2
        all_support_ei = all_support_ei and supports_ei

    util_std_total = np.sqrt(util_var_total)
    return np.asarray(util_mean_total, dtype=float), np.asarray(util_std_total, dtype=float), all_support_ei


def recommend_next(
    bundle: TrainedModelBundle,
    cfg: ProjectConfig,
    *,
    n_candidates: int = 2000,
    n_recommendations: int = 8,
    responses: Optional[Sequence[str]] = None,
    acquisition: str = "ucb",
    kappa: float = 2.0,
    best_observed_utility: Optional[float] = None,
    seed: int = 0,
    constraint_fn: Optional[Callable[[Dict[str, object]], bool]] = None,
) -> RecommendationResult:
    """Recommend next experiments.

    Parameters
    ----------
    responses:
        Which response(s) to optimize. Defaults to the bundle's responses.
    acquisition:
        "ucb" (default) or "ei". EI requires best_observed_utility.
    best_observed_utility:
        Best observed utility so far (in the same utility scale). If None and acquisition="ei",
        we fall back to the best predicted utility among candidates.
    constraint_fn:
        Optional function is_feasible(row_dict) -> bool
    """

    responses = list(responses) if responses is not None else list(bundle.responses)

    candidates = generate_candidates(cfg, n_candidates=n_candidates, seed=seed)

    if constraint_fn is None:
        constraint_fn = cfg.load_constraint_function()

    if constraint_fn is not None:
        mask = []
        for _, row in candidates.iterrows():
            ok = bool(constraint_fn(row.to_dict()))
            mask.append(ok)
        candidates = candidates.loc[mask].reset_index(drop=True)

    if len(candidates) == 0:
        raise ValueError("No feasible candidates remain after applying constraints.")

    pred_df = bundle.predict(candidates)

    util_mean, util_std, supports_ei = _scalarize(cfg, pred_df, responses=responses)

    if acquisition.lower() == "ei":
        if not supports_ei:
            # fall back to UCB automatically when target objectives exist
            acquisition = "ucb"
        else:
            if best_observed_utility is None:
                best_observed_utility = float(np.nanmax(util_mean))
            acq = expected_improvement(util_mean, util_std, best_observed_utility, maximize=True)
    if acquisition.lower() == "ucb":
        acq = ucb(util_mean, util_std, kappa=kappa)
    elif acquisition.lower() not in {"ei", "ucb"}:
        raise ValueError(f"Unknown acquisition: {acquisition}")

    ranked = candidates.copy()
    ranked["utility_mean"] = util_mean
    ranked["utility_std"] = util_std
    ranked["acquisition"] = acq

    # Attach per-response predictions
    ranked = pd.concat([ranked, pred_df.reset_index(drop=True)], axis=1)

    ranked = ranked.sort_values("acquisition", ascending=False).reset_index(drop=True)
    top = ranked.head(n_recommendations).reset_index(drop=True)

    return RecommendationResult(candidates=candidates, ranked=top)
