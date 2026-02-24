"""
Feature engineering for causal models.

Transforms raw store and transaction data into covariates suitable
for propensity score estimation and causal analysis.
"""
import pandas as pd
import numpy as np
from typing import List


def create_pre_treatment_features(panel: pd.DataFrame,
                                  pre_period_weeks: int = 12) -> pd.DataFrame:
    """Aggregate pre-treatment period metrics as covariates."""
    pre = panel[panel["week"] <= pre_period_weeks]

    agg = pre.groupby("store_id").agg(
        pre_avg_revenue=("revenue", "mean"),
        pre_std_revenue=("revenue", "std"),
        pre_total_units=("units_sold", "sum"),
        pre_avg_basket=("basket_size", "mean"),
        pre_avg_new_customers=("new_customers", "mean"),
    ).reset_index()

    agg["pre_revenue_cv"] = agg["pre_std_revenue"] / agg["pre_avg_revenue"]
    agg["pre_revenue_cv"] = agg["pre_revenue_cv"].fillna(0)
    return agg


def standardize_covariates(df: pd.DataFrame,
                           columns: List[str]) -> pd.DataFrame:
    """Standardize covariates for propensity score estimation."""
    df = df.copy()
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[f"{col}_std"] = (df[col] - mean) / std
        else:
            df[f"{col}_std"] = 0.0
    return df


def compute_growth_rate(panel: pd.DataFrame, baseline_weeks: int = 4) -> pd.DataFrame:
    """Compute revenue growth rate relative to baseline period."""
    baseline = panel[panel["week"] <= baseline_weeks].groupby("store_id")["revenue"].mean()
    current = panel[panel["week"] > baseline_weeks].groupby("store_id")["revenue"].mean()

    growth = pd.DataFrame({
        "store_id": baseline.index,
        "baseline_revenue": baseline.values,
        "post_revenue": current.reindex(baseline.index).values,
    })
    growth["growth_rate"] = (growth["post_revenue"] - growth["baseline_revenue"]) / growth["baseline_revenue"]
    return growth
