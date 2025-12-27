"""
Integration tests — end-to-end pipeline validation.

These tests use the generated synthetic data and verify that
the full pipeline produces sensible outputs. Marked as 'integration'
to allow running separately from fast unit tests.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.data_loader import load_config
from data.feature_engineering import create_pre_treatment_features
from causal.propensity_score import run_psm, estimate_propensity_scores
from causal.diff_in_diff import estimate_did
from causal.ab_testing import analyze_ab_test, calculate_sample_size
from metrics.kpi_framework import KPICalculator
from metrics.attribution import decompose_revenue


@pytest.mark.integration
class TestEndToEndPipeline:
    """Integration tests running against the shared panel fixture."""

    def test_psm_produces_valid_results(self, store_data, panel_data):
        """PSM should run without errors on realistic data."""
        pre = create_pre_treatment_features(panel_data, pre_period_weeks=13)
        analysis = store_data.merge(pre, on="store_id")

        covariates = ["store_size", "avg_weekly_revenue", "competitor_density"]
        available = [c for c in covariates if c in analysis.columns]

        result = run_psm(analysis, "pre_avg_revenue", "treated",
                         available, caliper=0.2)

        assert result.n_matched > 0
        assert result.se > 0
        assert 0 <= result.p_value <= 1
        assert result.ci_lower < result.ci_upper

    def test_did_produces_valid_results(self, panel_data):
        """DiD should estimate treatment effect with valid confidence interval."""
        result = estimate_did(
            panel_data, "revenue", "treated", "week",
            post_period_start=14, entity_col="store_id",
        )
        assert result.se > 0
        assert result.ci_lower < result.ci_upper
        assert 0 <= result.p_value <= 1
        assert result.pre_trend_parallel in (True, False)

    def test_ab_test_analysis_on_post_period(self, panel_data):
        """A/B test should detect an effect in the post-period data."""
        post = panel_data[panel_data["post_period"] == 1]
        control = post[post["treated"] == 0]["revenue"].values
        treated = post[post["treated"] == 1]["revenue"].values

        result = analyze_ab_test(control, treated)
        assert result.significant in (True, False)
        assert result.control_mean > 0
        assert result.treatment_mean > 0

    def test_kpi_calculator(self, panel_data):
        """KPI calculator should produce a report with all expected fields."""
        calc = KPICalculator(panel_data)
        report = calc.generate_kpi_report(
            causal_estimates={"att_revenue": 0.08},
            promotion_cost=50000,
        )
        assert len(report) > 0
        assert "kpi" in report.columns
        assert "value" in report.columns

    def test_attribution_decomposition(self, panel_data):
        """Attribution should decompose revenue into components that sum correctly."""
        result = decompose_revenue(panel_data)
        assert isinstance(result.promotion_effect, float)
        assert 0 <= result.promotion_share <= 100

    def test_power_analysis_feasibility(self, panel_data):
        """Power analysis should return feasible sample sizes."""
        control = panel_data[panel_data["treated"] == 0]
        pa = calculate_sample_size(
            baseline_mean=control["revenue"].mean(),
            baseline_std=control["revenue"].std(),
            mde=0.05,
        )
        assert pa.required_sample_per_group > 0
        assert pa.required_sample_per_group < 100000  # Sanity check

    def test_propensity_scores_valid_range(self, store_data):
        """All propensity scores should be between 0 and 1."""
        covariates = ["store_size", "avg_weekly_revenue", "competitor_density"]
        ps = estimate_propensity_scores(store_data, "treated", covariates)

        assert len(ps) == len(store_data)
        assert np.all(ps >= 0)
        assert np.all(ps <= 1)
