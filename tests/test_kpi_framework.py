"""Tests for KPI framework."""
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics.kpi_framework import KPICalculator, KPI_REGISTRY


@pytest.fixture
def sample_panel():
    rng = np.random.default_rng(42)
    n_stores = 20
    n_weeks = 10
    records = []
    for store in range(1, n_stores + 1):
        treated = 1 if store <= 10 else 0
        for week in range(1, n_weeks + 1):
            records.append({
                "store_id": store,
                "week": week,
                "revenue": rng.normal(1000 + 80 * treated, 100),
                "units_sold": int(rng.normal(100 + 8 * treated, 10)),
                "basket_size": rng.normal(45 + 2 * treated, 5),
                "new_customers": rng.poisson(10 + 3 * treated),
                "treated": treated,
            })
    return pd.DataFrame(records)


def test_naive_lift(sample_panel):
    calc = KPICalculator(sample_panel)
    result = calc.compute_naive_lift("revenue")
    assert result["naive_lift"] > 0
    assert "treated_mean" in result


def test_incremental_revenue(sample_panel):
    calc = KPICalculator(sample_panel)
    inc_rev = calc.compute_incremental_revenue(att=0.08, n_treated_stores=10, n_weeks=10)
    assert inc_rev > 0


def test_roas(sample_panel):
    calc = KPICalculator(sample_panel)
    assert calc.compute_roas(10000, 5000) == 2.0
    assert calc.compute_roas(10000, 0) == float("inf")


def test_kpi_report(sample_panel):
    calc = KPICalculator(sample_panel)
    report = calc.generate_kpi_report({"att_revenue": 0.08}, promotion_cost=5000)
    assert len(report) > 0
    assert "kpi" in report.columns


def test_kpi_registry():
    assert "incremental_revenue" in KPI_REGISTRY
    assert "roas" in KPI_REGISTRY
