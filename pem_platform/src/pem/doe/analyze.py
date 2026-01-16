from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pem.core.data import to_coded_dataframe
from pem.core.config import ProjectConfig


@dataclass(frozen=True)
class DoeAnalysisResult:
    response: str
    formula: str
    coef_table: pd.DataFrame
    anova_table: pd.DataFrame
    alias_warnings: pd.DataFrame


def _interaction_terms(factor_names: Sequence[str], max_order: int) -> List[str]:
    terms: List[str] = []
    for order in range(2, max_order + 1):
        for combo in combinations(factor_names, order):
            terms.append(":".join(combo))
    return terms


def build_formula(response: str, factor_names: Sequence[str], *, max_interaction_order: int = 1) -> str:
    main = " + ".join(factor_names) if factor_names else "1"
    terms = [main]
    if max_interaction_order >= 2:
        interactions = _interaction_terms(factor_names, max_interaction_order)
        if interactions:
            terms.append(" + ".join(interactions))
    rhs = " + ".join(terms)
    return f"{response} ~ {rhs}"


def detect_aliasing(design_matrix: pd.DataFrame, *, threshold: float = 0.95) -> pd.DataFrame:
    """Approximate alias detection via column correlation.

    This is a pragmatic indicator. True alias structure depends on the generator and resolution.
    """

    if design_matrix.shape[1] < 2:
        return pd.DataFrame(columns=["term_i", "term_j", "corr"])

    corr = np.corrcoef(design_matrix.to_numpy().T)
    names = list(design_matrix.columns)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = corr[i, j]
            if np.isfinite(c) and abs(c) >= threshold:
                rows.append((names[i], names[j], float(c)))
    return pd.DataFrame(rows, columns=["term_i", "term_j", "corr"]).sort_values(
        by="corr", key=lambda s: s.abs(), ascending=False
    )


def analyze_doe(
    df: pd.DataFrame,
    cfg: ProjectConfig,
    *,
    response: str,
    max_interaction_order: int = 2,
    factors: Optional[Sequence[str]] = None,
) -> DoeAnalysisResult:
    """Perform a DoE-style regression + ANOVA on coded factors.

    - Numeric factors are coded to [-1,+1] based on config bounds/levels.
    - Categorical factors are not included in this DoE regression.
    """

    cfg.get_response(response)  # validate
    coded_X = to_coded_dataframe(df, cfg, factors=factors)

    y = pd.to_numeric(df[response], errors="coerce")
    model_df = coded_X.copy()
    model_df[response] = y
    model_df = model_df.dropna(axis=0).reset_index(drop=True)

    factor_names = list(coded_X.columns)
    formula = build_formula(response, factor_names, max_interaction_order=max_interaction_order)

    model = smf.ols(formula, data=model_df).fit()

    coef = model.summary2().tables[1].copy()
    coef.index.name = "term"
    coef = coef.rename(
        columns={
            "Coef.": "coef",
            "Std.Err.": "std_err",
            "P>|t|": "p_value",
        }
    )

    # For -1/+1 coding, effect = 2*coef
    if "coef" in coef.columns:
        coef["effect"] = 2.0 * coef["coef"]

    # Type-II ANOVA is a reasonable default for balanced-ish designs
    anova = sm.stats.anova_lm(model, typ=2)
    anova.index.name = "term"

    # Alias detection on design matrix (including interactions up to max_interaction_order)
    # Build explicit design matrix using patsy via model.model.exog + names
    exog = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    # Drop Intercept for alias scan
    if "Intercept" in exog.columns:
        exog = exog.drop(columns=["Intercept"])
    alias = detect_aliasing(exog)

    return DoeAnalysisResult(
        response=response,
        formula=formula,
        coef_table=coef.reset_index(),
        anova_table=anova.reset_index(),
        alias_warnings=alias,
    )
