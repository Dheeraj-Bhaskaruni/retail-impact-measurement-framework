"""Tests for A/B testing module."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from causal.ab_testing import (
    calculate_sample_size,
    analyze_ab_test,
    bonferroni_correction,
    sequential_test,
)


def test_sample_size_calculation():
    result = calculate_sample_size(
        baseline_mean=100, baseline_std=20, mde=0.05, alpha=0.05, power=0.80
    )
    assert result.required_sample_per_group > 0
    assert result.achieved_power == 0.80


def test_analyze_ab_test_significant():
    rng = np.random.default_rng(42)
    control = rng.normal(100, 10, 500)
    treatment = rng.normal(105, 10, 500)

    result = analyze_ab_test(control, treatment)
    assert result.significant
    assert result.relative_lift > 0
    assert result.ci_lower > 0


def test_analyze_ab_test_not_significant():
    rng = np.random.default_rng(42)
    control = rng.normal(100, 10, 50)
    treatment = rng.normal(100.5, 10, 50)

    result = analyze_ab_test(control, treatment)
    assert result.p_value > 0.05 or not result.significant


def test_bonferroni_correction():
    p_values = [0.01, 0.03, 0.04, 0.06]
    result = bonferroni_correction(p_values, alpha=0.05)
    assert len(result) == 4
    assert result["adjusted_p"].iloc[0] == 0.04
    assert all(result["adjusted_p"] >= result["original_p"])


def test_sequential_test():
    rng = np.random.default_rng(42)
    data = rng.normal(105, 10, 1000).tolist()
    result = sequential_test(data, control_mean=100, control_std=10)
    assert result["decision"] in ["reject_null", "accept_null", "inconclusive"]
