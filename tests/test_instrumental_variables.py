"""Tests for instrumental variables module."""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from causal.instrumental_variables import two_stage_least_squares, sargan_test


@pytest.fixture
def iv_data():
    """Create data with a valid instrument."""
    rng = np.random.default_rng(42)
    n = 1000

    z = rng.normal(0, 1, n)
    u = rng.normal(0, 1, n)
    treatment = (0.5 * z + 0.3 * u + rng.normal(0, 0.5, n) > 0).astype(float)
    outcome = 2 + 1.5 * treatment + 0.8 * u + rng.normal(0, 1, n)

    return pd.DataFrame({
        "instrument": z, "treatment": treatment, "outcome": outcome
    })


def test_2sls_recovers_effect(iv_data):
    result = two_stage_least_squares(
        iv_data, outcome_col="outcome", treatment_col="treatment",
        instrument_cols=["instrument"]
    )
    assert 0.5 < result.ate < 3.0
    assert result.n_obs == 1000


def test_first_stage_f_stat(iv_data):
    result = two_stage_least_squares(
        iv_data, outcome_col="outcome", treatment_col="treatment",
        instrument_cols=["instrument"]
    )
    assert result.first_stage_f_stat > 0


def test_sargan_with_one_instrument(iv_data):
    result = sargan_test(
        iv_data, outcome_col="outcome", treatment_col="treatment",
        instrument_cols=["instrument"]
    )
    assert result["result"] == "exactly identified - test not applicable"
