"""
A/B Test Design and Analysis Module.

Provides power analysis for sample size calculation, sequential testing,
and multiple comparison corrections.
"""
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import List


@dataclass
class PowerAnalysis:
    """Sample size and power calculation results."""
    required_sample_per_group: int
    achieved_power: float
    mde: float
    alpha: float


@dataclass
class ABTestResult:
    """Results from A/B test analysis."""
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift: float
    p_value: float
    ci_lower: float
    ci_upper: float
    significant: bool
    power: float


def calculate_sample_size(baseline_mean: float,
                          baseline_std: float,
                          mde: float,
                          alpha: float = 0.05,
                          power: float = 0.80) -> PowerAnalysis:
    """Calculate required sample size per group for a two-sample t-test."""
    effect_size = (mde * baseline_mean) / baseline_std
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = int(np.ceil(2 * ((z_alpha + z_beta) / effect_size) ** 2))

    return PowerAnalysis(
        required_sample_per_group=n, achieved_power=power,
        mde=mde, alpha=alpha,
    )


def analyze_ab_test(control: np.ndarray,
                    treatment: np.ndarray,
                    alpha: float = 0.05) -> ABTestResult:
    """Analyze an A/B test using Welch's t-test."""
    c_mean = np.mean(control)
    t_mean = np.mean(treatment)
    abs_lift = t_mean - c_mean
    rel_lift = abs_lift / c_mean if c_mean != 0 else np.inf

    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

    se = np.sqrt(np.var(treatment, ddof=1) / len(treatment) +
                 np.var(control, ddof=1) / len(control))
    dof = _welch_dof(control, treatment)
    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    ci_lower = abs_lift - t_crit * se
    ci_upper = abs_lift + t_crit * se

    effect_size = abs_lift / np.sqrt((np.var(control, ddof=1) + np.var(treatment, ddof=1)) / 2)
    ncp = effect_size * np.sqrt(len(control) * len(treatment) / (len(control) + len(treatment)))
    power = 1 - stats.t.cdf(t_crit, dof, loc=ncp)

    return ABTestResult(
        control_mean=c_mean, treatment_mean=t_mean,
        absolute_lift=abs_lift, relative_lift=rel_lift,
        p_value=p_value, ci_lower=ci_lower, ci_upper=ci_upper,
        significant=p_value < alpha, power=power,
    )


def _welch_dof(x: np.ndarray, y: np.ndarray) -> float:
    """Welch-Satterthwaite degrees of freedom.

    More accurate than assuming equal df when sample sizes or
    variances differ between groups.
    """
    vx = np.var(x, ddof=1) / len(x)
    vy = np.var(y, ddof=1) / len(y)
    numerator = (vx + vy) ** 2
    denominator = vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1)
    return numerator / denominator if denominator > 0 else min(len(x), len(y)) - 1


def bonferroni_correction(p_values: List[float],
                          alpha: float = 0.05) -> pd.DataFrame:
    """Apply Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    adjusted = [min(p * n, 1.0) for p in p_values]
    return pd.DataFrame({
        "original_p": p_values,
        "adjusted_p": adjusted,
        "significant": [p < alpha for p in adjusted],
    })


def sequential_test(data_stream: List[float],
                    control_mean: float,
                    control_std: float,
                    alpha: float = 0.05,
                    beta: float = 0.20) -> dict:
    """Sequential probability ratio test (SPRT) for early stopping."""
    h0_mean = control_mean
    h1_mean = control_mean * 1.05
    log_a = np.log((1 - beta) / alpha)
    log_b = np.log(beta / (1 - alpha))

    cumulative_lr = 0.0
    for i, obs in enumerate(data_stream):
        lr = (stats.norm.logpdf(obs, h1_mean, control_std) -
              stats.norm.logpdf(obs, h0_mean, control_std))
        cumulative_lr += lr

        if cumulative_lr >= log_a:
            return {"decision": "reject_null", "stopped_at": i + 1, "log_lr": cumulative_lr}
        elif cumulative_lr <= log_b:
            return {"decision": "accept_null", "stopped_at": i + 1, "log_lr": cumulative_lr}

    return {"decision": "inconclusive", "stopped_at": len(data_stream), "log_lr": cumulative_lr}
