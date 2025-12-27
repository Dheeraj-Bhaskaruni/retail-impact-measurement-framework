"""
Shared test fixtures and configuration.

Provides reusable test data that mirrors the production schema,
ensuring all tests validate against realistic data shapes.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def rng():
    """Shared random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def store_data(rng):
    """Realistic store-level data matching production schema."""
    n = 50
    return pd.DataFrame({
        "store_id": [f"STR-{i:04d}" for i in range(1, n + 1)],
        "store_name": [f"Store #{i}" for i in range(1, n + 1)],
        "region": rng.choice(["Northeast", "Southeast", "Midwest", "West"], n),
        "state": rng.choice(["NY", "FL", "IL", "CA", "TX"], n),
        "store_format": rng.choice(["supercenter", "neighborhood", "express"], n),
        "store_size": rng.lognormal(8.5, 0.4, n).round(0),
        "avg_weekly_revenue": rng.lognormal(10, 0.5, n).round(2),
        "competitor_density": rng.poisson(3, n),
        "median_household_income": rng.normal(55000, 15000, n).clip(20000).round(0),
        "foot_traffic_index": rng.normal(100, 25, n).clip(10).round(1),
        "warehouse_distance": rng.exponential(30, n).round(1),
        "regional_ad_spend": rng.normal(700000, 50000, n).round(0),
        "treated": np.concatenate([np.ones(25), np.zeros(25)]).astype(int),
    })


@pytest.fixture(scope="session")
def panel_data(store_data, rng):
    """Realistic panel data (store x week) matching production schema."""
    records = []
    for _, store in store_data.iterrows():
        for week in range(1, 26):
            post = 1 if week >= 14 else 0
            treatment_effect = 0.08 * store["treated"] * post
            base = store["avg_weekly_revenue"]
            revenue = base * np.exp(rng.normal(treatment_effect, 0.05))

            records.append({
                "store_id": store["store_id"],
                "week": week,
                "fiscal_week": 202526 + week,
                "post_period": post,
                "revenue": round(revenue, 2),
                "units_sold": int(revenue / rng.uniform(8, 15)),
                "transaction_count": int(revenue / rng.uniform(30, 60)),
                "basket_size": round(rng.normal(45, 12), 2),
                "new_customers": rng.poisson(10 + 3 * store["treated"] * post),
                "return_customer_count": rng.poisson(50),
                "gross_margin": round(revenue * rng.normal(0.32, 0.04), 2),
                "discount_amount": round(revenue * rng.uniform(0.02, 0.08), 2),
                "promo_markdown_cost": round(rng.uniform(200, 800) * store["treated"] * post, 2),
                "treated": store["treated"],
            })
    return pd.DataFrame(records)


@pytest.fixture
def sample_causal_data(rng):
    """Simple data with known treatment effect for unit tests."""
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    propensity = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
    treatment = (rng.uniform(size=n) < propensity).astype(int)
    outcome = 2 + 0.5 * x1 + 0.3 * x2 + 0.8 * treatment + rng.normal(0, 0.5, n)

    return pd.DataFrame({
        "x1": x1, "x2": x2, "treatment": treatment, "outcome": outcome,
    })
