"""
Attribution modeling for retail promotions.

Decomposes revenue changes into components: promotion effect,
seasonality, market trends, and store-specific factors.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from dataclasses import dataclass


@dataclass
class AttributionResult:
    """Revenue decomposition results."""
    promotion_effect: float
    seasonality_effect: float
    trend_effect: float
    store_effect: float
    residual: float
    total_change: float
    promotion_share: float


def decompose_revenue(panel: pd.DataFrame,
                      treatment_col: str = "treated",
                      outcome_col: str = "revenue",
                      time_col: str = "week") -> AttributionResult:
    """
    Decompose revenue into causal components using a structural model:
    log(Revenue) = Store_FE + Seasonality + Trend + Treatment_Effect + error
    """
    df = panel.copy()
    df["log_revenue"] = np.log(df[outcome_col])
    df["trend"] = df[time_col]
    df["sin_season"] = np.sin(2 * np.pi * df[time_col] / 52)
    df["cos_season"] = np.cos(2 * np.pi * df[time_col] / 52)

    store_dummies = pd.get_dummies(df["store_id"], prefix="s", drop_first=True, dtype=float)

    X = pd.concat([
        df[[treatment_col, "trend", "sin_season", "cos_season"]],
        store_dummies
    ], axis=1)
    X = sm.add_constant(X)

    model = sm.OLS(df["log_revenue"], X).fit()

    promo = model.params[treatment_col]
    trend = model.params["trend"] * df[time_col].mean()
    season = (model.params["sin_season"] * df["sin_season"].mean() +
              model.params["cos_season"] * df["cos_season"].mean())

    store_cols = [c for c in model.params.index if c.startswith("s_")]
    store_effect = model.params[store_cols].mean() if store_cols else 0

    total = df.groupby(treatment_col)["log_revenue"].mean()
    total_change = total.get(1, 0) - total.get(0, 0)

    # Clamp promotion share to [0, 100] — can exceed 100% if other
    # components partially offset the promotion effect
    promo_share = abs(promo) / abs(total_change) * 100 if total_change != 0 else 0

    result = AttributionResult(
        promotion_effect=promo,
        seasonality_effect=season,
        trend_effect=trend,
        store_effect=store_effect,
        residual=total_change - promo - season - trend,
        total_change=total_change,
        promotion_share=min(promo_share, 100),
    )

    # Sanity check: components should roughly sum to total change
    component_sum = promo + season + trend + result.residual
    if abs(component_sum - total_change) > 0.01:
        import warnings
        warnings.warn(
            f"Attribution components sum to {component_sum:.4f} but total "
            f"change is {total_change:.4f} — check model specification"
        )

    return result
