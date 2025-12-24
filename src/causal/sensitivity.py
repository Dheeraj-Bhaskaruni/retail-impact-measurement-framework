"""
Sensitivity Analysis for Causal Estimates.

Assesses how robust our findings are to potential violations of
key assumptions — particularly unobserved confounders.
"""
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RosenbaumResult:
    """Results from Rosenbaum bounds sensitivity analysis."""
    gamma_values: List[float]
    p_upper_bounds: List[float]
    critical_gamma: float  # Gamma at which result becomes insignificant
    robust: bool           # True if critical_gamma > 1.5


def rosenbaum_bounds(treated_outcomes: np.ndarray,
                     control_outcomes: np.ndarray,
                     gamma_range: Tuple[float, float] = (1.0, 3.0),
                     n_steps: int = 20,
                     alpha: float = 0.05) -> RosenbaumResult:
    """
    Rosenbaum sensitivity analysis for matched pair designs.

    Tests how large an unobserved confounder (Gamma) would need to be
    to explain away the observed treatment effect. Higher critical Gamma
    means the result is more robust to hidden bias.

    Gamma = 1.0: no hidden bias (treatment assignment is as-good-as-random)
    Gamma = 2.0: an unobserved confounder could make one unit 2x as likely
                 to be treated as a matched unit

    Parameters
    ----------
    treated_outcomes : Outcomes for treated units in matched pairs
    control_outcomes : Outcomes for matched control units
    gamma_range : Range of Gamma values to test
    """
    diffs = treated_outcomes - control_outcomes
    n = len(diffs)

    if n == 0:
        raise ValueError("No matched pairs provided")

    gammas = np.linspace(gamma_range[0], gamma_range[1], n_steps)
    p_uppers = []
    critical_gamma = gamma_range[1]

    for gamma in gammas:
        # Under worst-case hidden bias at this Gamma level,
        # compute the upper bound on the p-value
        # Using Wilcoxon signed-rank test framework
        ranks = stats.rankdata(np.abs(diffs))
        positive_ranks = ranks[diffs > 0]
        T_obs = np.sum(positive_ranks)

        # Expected value and variance under worst-case bias
        p_plus = gamma / (1 + gamma)
        E_T = p_plus * n * (n + 1) / 2
        V_T = p_plus * (1 - p_plus) * n * (n + 1) * (2 * n + 1) / 6

        if V_T > 0:
            z = (T_obs - E_T) / np.sqrt(V_T)
            p_upper = 1 - stats.norm.cdf(z)
        else:
            p_upper = 0.5

        p_uppers.append(p_upper)

        if p_upper > alpha and critical_gamma == gamma_range[1]:
            critical_gamma = gamma

    return RosenbaumResult(
        gamma_values=gammas.tolist(),
        p_upper_bounds=p_uppers,
        critical_gamma=critical_gamma,
        robust=critical_gamma > 1.5,
    )


def placebo_test(df: pd.DataFrame,
                 placebo_outcome_col: str,
                 treatment_col: str,
                 covariates: List[str],
                 method: str = "psm") -> dict:
    """
    Placebo test: apply the causal method to an outcome that should
    NOT be affected by treatment. If we find a significant effect,
    our method may be invalid.

    Common placebo outcomes: pre-treatment revenue, geographic features.
    """
    from causal.propensity_score import run_psm

    if method == "psm":
        result = run_psm(df, placebo_outcome_col, treatment_col,
                         covariates, caliper=0.1)
        return {
            "test": "placebo",
            "outcome": placebo_outcome_col,
            "method": method,
            "estimated_effect": result.att,
            "p_value": result.p_value,
            "passed": result.p_value > 0.05,  # Should NOT be significant
            "interpretation": (
                "PASS: No spurious effect detected on placebo outcome"
                if result.p_value > 0.05
                else "FAIL: Significant effect on placebo — possible model misspecification"
            ),
        }
    else:
        raise ValueError(f"Unsupported method: {method}")


def effect_stability(df: pd.DataFrame,
                     outcome_col: str,
                     treatment_col: str,
                     covariates: List[str],
                     caliper_values: List[float] = None) -> pd.DataFrame:
    """
    Test stability of PSM estimates across different caliper values.
    Robust results should not change much with caliper choice.
    """
    from causal.propensity_score import run_psm

    if caliper_values is None:
        caliper_values = [0.01, 0.02, 0.05, 0.10, 0.20]

    results = []
    for cal in caliper_values:
        try:
            r = run_psm(df, outcome_col, treatment_col, covariates, caliper=cal)
            results.append({
                "caliper": cal,
                "att": r.att,
                "se": r.se,
                "p_value": r.p_value,
                "n_matched": r.n_matched,
                "significant": r.p_value < 0.05,
            })
        except ValueError:
            results.append({
                "caliper": cal, "att": None, "se": None,
                "p_value": None, "n_matched": 0, "significant": False,
            })

    return pd.DataFrame(results)
