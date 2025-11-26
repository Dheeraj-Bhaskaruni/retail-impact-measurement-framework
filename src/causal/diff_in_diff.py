"""
Difference-in-Differences (DiD) estimator.

Compares the change in outcomes over time between treatment and
control groups, controlling for time-invariant confounders.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats as sp_stats
from dataclasses import dataclass


@dataclass
class DiDResult:
    """Results from Difference-in-Differences analysis."""
    att: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    pre_trend_parallel: bool
    pre_trend_p_value: float


def estimate_did(panel: pd.DataFrame,
                 outcome_col: str,
                 treatment_col: str,
                 time_col: str,
                 post_period_start: int,
                 entity_col: str = "store_id",
                 alpha: float = 0.05) -> DiDResult:
    """
    Estimate ATT using DiD with entity and time fixed effects.

    Y_it = alpha_i + gamma_t + delta * (Treated_i * Post_t) + epsilon_it
    """
    df = panel.copy()
    df["post"] = (df[time_col] >= post_period_start).astype(int)
    df["treat_post"] = df[treatment_col] * df["post"]

    entity_dummies = pd.get_dummies(df[entity_col], prefix="store", drop_first=True, dtype=float)
    time_dummies = pd.get_dummies(df[time_col], prefix="week", drop_first=True, dtype=float)

    X = pd.concat([df[["treat_post"]], entity_dummies, time_dummies], axis=1)
    X = sm.add_constant(X)

    # Cluster standard errors at the store level to account for
    # within-store serial correlation across weeks
    model = sm.OLS(df[outcome_col], X).fit(cov_type="cluster",
                                            cov_kwds={"groups": df[entity_col]})

    att = model.params["treat_post"]
    se = model.bse["treat_post"]
    p_value = model.pvalues["treat_post"]

    t_crit = sp_stats.t.ppf(1 - alpha / 2, df=model.df_resid)

    pre_trend_p = _test_parallel_trends(panel, outcome_col, treatment_col,
                                         time_col, post_period_start)

    return DiDResult(
        att=att, se=se, ci_lower=att - t_crit * se, ci_upper=att + t_crit * se,
        p_value=p_value, pre_trend_parallel=pre_trend_p > 0.05,
        pre_trend_p_value=pre_trend_p,
    )


def _test_parallel_trends(panel: pd.DataFrame,
                          outcome_col: str,
                          treatment_col: str,
                          time_col: str,
                          post_period_start: int) -> float:
    """Test whether treatment and control had parallel pre-treatment trends."""
    pre = panel[panel[time_col] < post_period_start].copy()
    pre["treat_time"] = pre[treatment_col] * pre[time_col]
    X = sm.add_constant(pre[["treat_time", treatment_col, time_col]])
    model = sm.OLS(pre[outcome_col], X).fit()
    return model.pvalues["treat_time"]
