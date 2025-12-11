"""Tests for measurement pipeline components."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.feature_engineering import (
    create_pre_treatment_features,
    standardize_covariates,
    compute_growth_rate,
)


@pytest.fixture
def mock_panel():
    rng = np.random.default_rng(42)
    records = []
    for store in range(1, 11):
        for week in range(1, 53):
            records.append({
                "store_id": store,
                "week": week,
                "revenue": rng.normal(1000, 100),
                "units_sold": int(rng.normal(100, 10)),
                "basket_size": rng.normal(45, 5),
                "new_customers": rng.poisson(10),
                "treated": 1 if store <= 5 else 0,
            })
    return pd.DataFrame(records)


def test_pre_treatment_features(mock_panel):
    features = create_pre_treatment_features(mock_panel, pre_period_weeks=12)
    assert "pre_avg_revenue" in features.columns
    assert "pre_revenue_cv" in features.columns
    assert len(features) == 10


def test_standardize_covariates(mock_panel):
    result = standardize_covariates(mock_panel, ["revenue"])
    assert "revenue_std" in result.columns
    assert abs(result["revenue_std"].mean()) < 0.1


def test_growth_rate(mock_panel):
    growth = compute_growth_rate(mock_panel, baseline_weeks=4)
    assert "growth_rate" in growth.columns
    assert len(growth) == 10
