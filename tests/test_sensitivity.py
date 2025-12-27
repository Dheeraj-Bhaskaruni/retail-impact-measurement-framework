"""Tests for sensitivity analysis module."""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from causal.sensitivity import rosenbaum_bounds, effect_stability


def test_rosenbaum_bounds_with_clear_effect():
    """With a large treatment effect, bounds should hold at high Gamma."""
    rng = np.random.default_rng(42)
    n = 100
    treated = rng.normal(10, 2, n)
    control = rng.normal(8, 2, n)  # Clear 2-unit effect

    result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 3.0))
    assert result.critical_gamma > 1.0
    assert len(result.gamma_values) == 20
    assert len(result.p_upper_bounds) == 20


def test_rosenbaum_bounds_fragile_effect():
    """With a tiny effect, bounds should break at low Gamma."""
    rng = np.random.default_rng(42)
    n = 30
    treated = rng.normal(10, 2, n)
    control = rng.normal(9.8, 2, n)  # Very small effect

    result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 2.0))
    # Should be fragile — critical gamma close to 1.0
    assert result.critical_gamma < 2.0


def test_effect_stability(sample_causal_data):
    """PSM estimates should be relatively stable across caliper values."""
    stability = effect_stability(
        sample_causal_data, "outcome", "treatment", ["x1", "x2"],
        caliper_values=[0.05, 0.10, 0.20]
    )
    assert len(stability) == 3
    assert "att" in stability.columns
    assert "n_matched" in stability.columns
    # At least some calipers should produce matches
    assert stability["n_matched"].sum() > 0
