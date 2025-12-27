"""Tests for pipeline monitoring and health checks."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.monitoring import run_health_checks, HealthCheck


@pytest.fixture
def healthy_results():
    return {
        "psm": {"att": 2100.0, "se": 500.0, "p_value": 0.001, "n_matched": 150},
        "did": {"att": 2200.0, "se": 400.0, "p_value": 0.0001, "parallel_trends": True},
        "attribution": {"promotion_effect": 0.08, "promotion_share": 74.0},
    }


@pytest.fixture
def degraded_results():
    return {
        "psm": {"att": 500.0, "se": 800.0, "p_value": 0.08, "n_matched": 30},
        "did": {"att": 2200.0, "se": 400.0, "p_value": 0.001, "parallel_trends": False},
        "attribution": {"promotion_effect": 0.02, "promotion_share": 98.0},
    }


def test_healthy_pipeline(healthy_results):
    config = {"campaign": {"id": "PROMO-TEST"}}
    report = run_health_checks(healthy_results, config)
    assert report.overall_status == "healthy"
    assert report.n_failures == 0


def test_degraded_pipeline(degraded_results):
    config = {"campaign": {"id": "PROMO-TEST"}}
    report = run_health_checks(degraded_results, config)
    assert report.overall_status in ("degraded", "critical")
    assert report.n_warnings > 0 or report.n_failures > 0


def test_cross_method_agreement_check(healthy_results):
    config = {"campaign": {"id": "PROMO-TEST"}}
    report = run_health_checks(healthy_results, config)
    agreement_checks = [c for c in report.checks if c.name == "cross_method_agreement"]
    assert len(agreement_checks) == 1
