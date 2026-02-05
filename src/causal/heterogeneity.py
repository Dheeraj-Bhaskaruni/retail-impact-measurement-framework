"""
Heterogeneous Treatment Effect (HTE) Analysis.

Estimates how the promotion effect varies across store subgroups —
critical for targeting future campaigns at the stores where
promotions are most effective.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class HTEResult:
    """Heterogeneous treatment effect results."""
    subgroup_col: str
    subgroup_effects: pd.DataFrame
    interaction_p_value: float  # Is heterogeneity statistically significant?
    heterogeneous: bool


def estimate_subgroup_effects(panel: pd.DataFrame,
                              outcome_col: str,
                              treatment_col: str,
                              subgroup_col: str,
                              covariates: Optional[List[str]] = None) -> HTEResult:
    """
    Estimate treatment effects within subgroups.

    Example: effect by store_format (supercenter vs neighborhood vs express)
    or by region (Northeast vs Southeast vs Midwest vs West).
    """
    covariates = covariates or []
    subgroups = panel[subgroup_col].unique()
    results = []

    for sg in sorted(subgroups):
        subset = panel[panel[subgroup_col] == sg]
        treated = subset[subset[treatment_col] == 1][outcome_col]
        control = subset[subset[treatment_col] == 0][outcome_col]

        if len(treated) < 10 or len(control) < 10:
            import warnings
            warnings.warn(f"Skipping subgroup '{sg}' — too few observations "
                          f"(treated={len(treated)}, control={len(control)})")
            continue

        diff = treated.mean() - control.mean()
        se = np.sqrt(treated.var() / len(treated) + control.var() / len(control))
        from scipy import stats
        t_stat = diff / se if se > 0 else 0
        dof = min(len(treated), len(control)) - 1
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=max(dof, 1)))

        results.append({
            "subgroup": sg,
            "n_treated": len(treated),
            "n_control": len(control),
            "treated_mean": treated.mean(),
            "control_mean": control.mean(),
            "effect": diff,
            "se": se,
            "p_value": p_val,
            "significant": p_val < 0.05,
        })

    # Test for heterogeneity via interaction model
    interaction_p = _test_interaction(panel, outcome_col, treatment_col, subgroup_col)

    effects_df = pd.DataFrame(results)
    return HTEResult(
        subgroup_col=subgroup_col,
        subgroup_effects=effects_df,
        interaction_p_value=interaction_p,
        heterogeneous=interaction_p < 0.05,
    )


def _test_interaction(panel: pd.DataFrame,
                      outcome_col: str,
                      treatment_col: str,
                      subgroup_col: str) -> float:
    """Test whether treatment effect varies by subgroup using interaction model."""
    df = panel.copy()
    dummies = pd.get_dummies(df[subgroup_col], drop_first=True, dtype=float)

    interactions = pd.DataFrame()
    for col in dummies.columns:
        interactions[f"treat_x_{col}"] = df[treatment_col] * dummies[col]

    X = pd.concat([df[[treatment_col]], dummies, interactions], axis=1)
    X = sm.add_constant(X)

    model = sm.OLS(df[outcome_col], X).fit()

    # F-test on interaction terms
    interaction_cols = [c for c in model.params.index if c.startswith("treat_x_")]
    if not interaction_cols:
        return 1.0

    r_matrix = np.zeros((len(interaction_cols), X.shape[1]))
    for i, col in enumerate(interaction_cols):
        col_idx = list(X.columns).index(col)
        r_matrix[i, col_idx] = 1

    f_test = model.f_test(r_matrix)
    return float(f_test.pvalue)


def estimate_cate_with_causal_forest(panel: pd.DataFrame,
                                      outcome_col: str,
                                      treatment_col: str,
                                      feature_cols: List[str]) -> Dict:
    """
    Estimate Conditional Average Treatment Effects (CATE) using
    EconML's CausalForestDML.

    Returns individual-level treatment effect estimates and feature
    importance for targeting.
    """
    from econml.dml import CausalForestDML

    Y = panel[outcome_col].values
    T = panel[treatment_col].values
    X = panel[feature_cols].values

    cf = CausalForestDML(
        n_estimators=200,
        min_samples_leaf=20,
        random_state=42,
    )
    cf.fit(Y, T, X=X)

    cate_estimates = cf.effect(X)
    feature_importance = dict(zip(feature_cols, cf.feature_importances_))

    return {
        "cate_mean": float(np.mean(cate_estimates)),
        "cate_std": float(np.std(cate_estimates)),
        "cate_median": float(np.median(cate_estimates)),
        "cate_q25": float(np.percentile(cate_estimates, 25)),
        "cate_q75": float(np.percentile(cate_estimates, 75)),
        "feature_importance": feature_importance,
        "individual_effects": cate_estimates,
    }
