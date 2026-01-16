from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from pem.core.config import FactorType, ProjectConfig, ResponseGoal


@dataclass(frozen=True)
class AxRecommendationResult:
    """Result from Ax-based Bayesian optimization recommendations."""

    # Suggestions in natural units (factor columns) plus an Ax trial index column.
    suggestions: pd.DataFrame

    # Number of historical rows successfully attached to the Ax experiment.
    n_attached: int


def _is_int_like(x: Any) -> bool:
    try:
        xf = float(x)
    except Exception:
        return False
    return bool(np.isfinite(xf) and abs(xf - round(xf)) < 1e-9)


def _coerce_param_value(cfg: ProjectConfig, name: str, value: Any) -> Any:
    """Coerce a dataframe value into a type compatible with Ax search space."""

    f = cfg.get_factor(name)
    if f.type == FactorType.categorical:
        return str(value)
    # numeric
    if _is_int_like(value) and f.type == FactorType.discrete:
        return int(round(float(value)))
    return float(value)


def _build_ax_parameters(cfg: ProjectConfig) -> List[Dict[str, Any]]:
    """Convert :class:`ProjectConfig` factors into Ax parameter specs."""

    params: List[Dict[str, Any]] = []

    for f in cfg.factors:
        if f.type == FactorType.continuous:
            lo, hi = f.low_high()
            params.append(
                {
                    "name": f.name,
                    "type": "range",
                    "bounds": [float(lo), float(hi)],
                    "value_type": "float",
                }
            )

        elif f.type == FactorType.discrete:
            if f.levels is not None and len(f.levels) >= 2:
                vals = list(f.levels)
                # Decide int vs float.
                if all(_is_int_like(v) for v in vals):
                    values = [int(round(float(v))) for v in vals]
                    value_type = "int"
                else:
                    values = [float(v) for v in vals]
                    value_type = "float"
                params.append(
                    {
                        "name": f.name,
                        "type": "choice",
                        "values": values,
                        "value_type": value_type,
                        "is_ordered": True,
                    }
                )
            else:
                lo, hi = f.low_high()
                # Treat as a continuous range if no explicit levels are provided.
                params.append(
                    {
                        "name": f.name,
                        "type": "range",
                        "bounds": [float(lo), float(hi)],
                        "value_type": "float",
                    }
                )

        elif f.type == FactorType.categorical:
            values = list(map(str, f.levels or []))
            params.append(
                {
                    "name": f.name,
                    "type": "choice",
                    "values": values,
                    "value_type": "str",
                    "is_ordered": False,
                }
            )
        else:
            raise ValueError(f"Unsupported factor type: {f.type}")

    return params


def _compute_observed_utility(
    cfg: ProjectConfig,
    df: pd.DataFrame,
    *,
    responses: Sequence[str],
) -> np.ndarray:
    """Compute a scalar utility from observed response columns.

    Utility is always defined as a quantity to **maximize**.

    - minimize:  utility = -y
    - maximize:  utility = +y
    - target:    utility = -abs(y - target)

    For multi-response, PEM uses cfg.scalarization.weights (weighted sum). If no weights are
    provided, equal weights are used.
    """

    responses = list(responses)
    weights = dict(cfg.scalarization.weights)
    if not weights:
        weights = {r: 1.0 for r in responses}

    util_total = np.zeros(len(df), dtype=float)

    for r_name in responses:
        r = cfg.get_response(r_name)
        y = pd.to_numeric(df[r_name], errors="coerce").to_numpy(dtype=float)

        if r.goal == ResponseGoal.minimize:
            u = -y
        elif r.goal == ResponseGoal.maximize:
            u = y
        else:
            # target
            assert r.target is not None
            u = -np.abs(y - float(r.target))

        util_total = util_total + float(weights.get(r_name, 0.0)) * u

    return util_total


def _is_row_valid_for_ax(cfg: ProjectConfig, row: pd.Series) -> bool:
    """Check if a row can be attached to Ax without violating the search space."""

    for f in cfg.factors:
        v = row.get(f.name)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return False

        if f.type == FactorType.categorical:
            allowed = set(map(str, f.levels or []))
            if str(v) not in allowed:
                return False
        else:
            try:
                vf = float(v)
            except Exception:
                return False

            lo, hi = f.low_high()
            if not (float(lo) <= vf <= float(hi)):
                return False

            if f.type == FactorType.discrete and f.levels is not None and len(f.levels) >= 2:
                # Must match a discrete level exactly (within tolerance).
                levels = np.asarray([float(x) for x in f.levels], dtype=float)
                if np.min(np.abs(levels - vf)) > 1e-9:
                    return False

    return True


def recommend_next_ax(
    cfg: ProjectConfig,
    df: pd.DataFrame,
    *,
    responses: Optional[Sequence[str]] = None,
    n_recommendations: int = 8,
    seed: int = 0,
    constraint_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    max_attempts_per_suggestion: int = 50,
) -> AxRecommendationResult:
    """Recommend next experiments using Ax (BoTorch-based Bayesian optimization).

    This command uses Ax in a pragmatic way:

    - It defines a search space from :class:`ProjectConfig`.
    - It computes a single scalar objective called **utility** from the provided response(s),
      according to config response goals + scalarization weights.
    - It attaches the historical rows as completed trials.
    - It asks Ax for the next trials.

    Notes
    -----
    - Multi-response optimization is handled via scalarization into `utility`. If you need true
      Pareto-front multi-objective optimization, you can extend this module to use Ax
      `MultiObjectiveOptimizationConfig`.
    - Arbitrary feasibility constraints are applied by filtering suggestions using the optional
      `constraint_fn` plugin.
    """

    try:
        from ax.service.ax_client import AxClient  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Ax is not installed. Install with: pip install -e \".[ax]\""
        ) from e

    if responses is None:
        responses = cfg.response_names()
    responses = list(responses)
    for r in responses:
        cfg.get_response(r)

    if n_recommendations <= 0:
        raise ValueError("n_recommendations must be > 0")

    if constraint_fn is None:
        constraint_fn = cfg.load_constraint_function()

    # Compute observed utility and drop rows that cannot be used.
    util = _compute_observed_utility(cfg, df, responses=responses)
    work = df.copy()
    work["__utility__"] = util
    work = work[np.isfinite(work["__utility__"].to_numpy(dtype=float))].reset_index(drop=True)

    # Build Ax experiment.
    ax_client = AxClient(random_seed=seed)
    ax_client.create_experiment(
        name=f"{cfg.project_name}_utility",
        parameters=_build_ax_parameters(cfg),
        objective_name="utility",
        minimize=False,
    )

    # Attach historical trials.
    n_attached = 0
    for _, row in work.iterrows():
        if not _is_row_valid_for_ax(cfg, row):
            continue

        params = {f.name: _coerce_param_value(cfg, f.name, row[f.name]) for f in cfg.factors}
        trial_idx = ax_client.attach_trial(params)
        ax_client.complete_trial(
            trial_idx,
            raw_data={"utility": (float(row["__utility__"]), 0.0)},
        )
        n_attached += 1

    # Suggest new trials.
    suggestions: List[Dict[str, Any]] = []

    for _ in range(n_recommendations):
        attempts = 0
        while True:
            attempts += 1
            if attempts > max_attempts_per_suggestion:
                raise RuntimeError(
                    "Unable to find a feasible suggestion from Ax after "
                    f"{max_attempts_per_suggestion} attempts. "
                    "Check constraints or widen bounds/levels."
                )

            params, trial_idx = ax_client.get_next_trial()

            # Apply feasibility constraints.
            if constraint_fn is not None:
                try:
                    ok = bool(constraint_fn(dict(params)))
                except Exception:
                    ok = False
                if not ok:
                    ax_client.abandon_trial(trial_idx, reason="Infeasible by constraint plugin")
                    continue

            suggestions.append({**params, "ax_trial_index": int(trial_idx)})
            break

    sug_df = pd.DataFrame(suggestions)
    # Ensure factor order.
    cols = cfg.factor_names(include_categorical=True)
    sug_df = sug_df[[*cols, "ax_trial_index"]]

    return AxRecommendationResult(suggestions=sug_df, n_attached=n_attached)
