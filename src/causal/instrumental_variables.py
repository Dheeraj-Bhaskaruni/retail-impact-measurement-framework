"""
Instrumental Variables (IV) / Two-Stage Least Squares (2SLS).

Used when we suspect unobserved confounders that PSM cannot address.
The instrument must affect treatment but not directly affect outcomes.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class IVResult:
    """Results from IV/2SLS estimation."""
    ate: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    first_stage_f_stat: float
    instrument_relevance: str
    n_obs: int


def two_stage_least_squares(df: pd.DataFrame,
                             outcome_col: str,
                             treatment_col: str,
                             instrument_cols: List[str],
                             covariate_cols: Optional[List[str]] = None,
                             alpha: float = 0.05) -> IVResult:
    """
    Estimate causal effect using 2SLS.

    Stage 1: Regress treatment on instruments + covariates
    Stage 2: Regress outcome on predicted treatment + covariates
    """
    covariates = covariate_cols or []
    exog_cols = instrument_cols + covariates

    # Stage 1
    X_first = sm.add_constant(df[exog_cols])
    first_stage = sm.OLS(df[treatment_col], X_first).fit()

    # F-test for instrument relevance
    r_matrix = np.zeros((len(instrument_cols), X_first.shape[1]))
    for i, inst in enumerate(instrument_cols):
        col_idx = list(X_first.columns).index(inst)
        r_matrix[i, col_idx] = 1
    f_test = first_stage.f_test(r_matrix)
    f_stat = float(f_test.fvalue)

    # Stage 2
    df = df.copy()
    df["treatment_hat"] = first_stage.predict(X_first)

    second_stage_cols = ["treatment_hat"] + covariates
    X_second = sm.add_constant(df[second_stage_cols])
    second_stage = sm.OLS(df[outcome_col], X_second).fit()

    ate = second_stage.params["treatment_hat"]
    se = second_stage.bse["treatment_hat"]
    p_value = second_stage.pvalues["treatment_hat"]

    t_crit = sp_stats.t.ppf(1 - alpha / 2, df=second_stage.df_resid)
    ci_lower = ate - t_crit * se
    ci_upper = ate + t_crit * se

    return IVResult(
        ate=ate, se=se, ci_lower=ci_lower, ci_upper=ci_upper,
        p_value=p_value, first_stage_f_stat=f_stat,
        instrument_relevance="strong" if f_stat > 10 else "weak",
        n_obs=len(df),
    )


def sargan_test(df: pd.DataFrame,
                outcome_col: str,
                treatment_col: str,
                instrument_cols: List[str],
                covariate_cols: Optional[List[str]] = None) -> dict:
    """Sargan overidentification test for instrument validity."""
    if len(instrument_cols) <= 1:
        return {"test": "sargan", "result": "exactly identified - test not applicable"}

    covariates = covariate_cols or []
    exog_cols = instrument_cols + covariates

    X_first = sm.add_constant(df[exog_cols])
    first_stage = sm.OLS(df[treatment_col], X_first).fit()
    df = df.copy()
    df["treatment_hat"] = first_stage.predict(X_first)

    second_stage_cols = ["treatment_hat"] + covariates
    X_second = sm.add_constant(df[second_stage_cols])
    second_stage = sm.OLS(df[outcome_col], X_second).fit()
    residuals = second_stage.resid

    X_sargan = sm.add_constant(df[instrument_cols + covariates])
    sargan_reg = sm.OLS(residuals, X_sargan).fit()

    test_stat = sargan_reg.rsquared * len(df)
    dof = len(instrument_cols) - 1
    p_value = 1 - sp_stats.chi2.cdf(test_stat, dof)

    return {
        "test": "sargan", "statistic": test_stat, "p_value": p_value,
        "dof": dof, "instruments_valid": p_value > 0.05,
    }
