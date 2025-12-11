"""Tests for propensity score matching module."""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from causal.propensity_score import (
    estimate_propensity_scores,
    match_nearest_neighbor,
    assess_balance,
    run_psm,
)


@pytest.fixture
def sample_data():
    """Create sample data with known treatment effect."""
    rng = np.random.default_rng(42)
    n = 200

    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    propensity = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
    treatment = (rng.uniform(size=n) < propensity).astype(int)
    outcome = 2 + 0.5 * x1 + 0.3 * x2 + 0.8 * treatment + rng.normal(0, 0.5, n)

    return pd.DataFrame({
        "x1": x1, "x2": x2, "treatment": treatment, "outcome": outcome
    })


def test_estimate_propensity_scores(sample_data):
    ps = estimate_propensity_scores(sample_data, "treatment", ["x1", "x2"])
    assert len(ps) == len(sample_data)
    assert all(0 <= p <= 1 for p in ps)


def test_match_nearest_neighbor(sample_data):
    ps = estimate_propensity_scores(sample_data, "treatment", ["x1", "x2"])
    matches = match_nearest_neighbor(ps, sample_data["treatment"].values, caliper=0.1)
    assert len(matches) > 0
    for t_idx, c_idx in matches:
        assert sample_data.iloc[t_idx]["treatment"] == 1
        assert sample_data.iloc[c_idx]["treatment"] == 0


def test_assess_balance(sample_data):
    balance = assess_balance(sample_data, ["x1", "x2"], "treatment")
    assert "std_mean_diff" in balance.columns
    assert len(balance) == 2


def test_run_psm_detects_effect(sample_data):
    result = run_psm(sample_data, "outcome", "treatment", ["x1", "x2"], caliper=0.2)
    assert result.att > 0
    assert result.n_matched > 0
    assert result.p_value < 0.1
