"""Tests for data validation schemas."""
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.validation import (
    StoreRecord, WeeklyOutcomeRecord, PipelineConfig,
    validate_panel_data, validate_store_data,
)


def test_valid_store_record():
    record = StoreRecord(
        store_id="STR-0001",
        store_name="Store #1",
        region="Northeast",
        state="NY",
        store_format="supercenter",
        store_size=5000.0,
        avg_weekly_revenue=22000.0,
        competitor_density=3,
        median_household_income=55000.0,
        foot_traffic_index=100.0,
        warehouse_distance=25.0,
        treated=1,
    )
    assert record.store_id == "STR-0001"


def test_invalid_store_id():
    with pytest.raises(Exception):
        StoreRecord(
            store_id="INVALID",
            store_name="Bad",
            region="Northeast",
            state="NY",
            store_format="supercenter",
            store_size=5000.0,
            avg_weekly_revenue=22000.0,
            competitor_density=3,
            median_household_income=55000.0,
            foot_traffic_index=100.0,
            warehouse_distance=25.0,
            treated=1,
        )


def test_invalid_revenue():
    with pytest.raises(Exception):
        WeeklyOutcomeRecord(
            store_id="STR-0001",
            week=1,
            revenue=-100.0,  # Negative revenue
            units_sold=50,
            basket_size=45.0,
            new_customers=10,
            treated=1,
            post_period=0,
        )


def test_pipeline_config_defaults():
    config = PipelineConfig(
        campaign_id="PROMO-2025-Q4",
        caliper=0.05,
    )
    assert config.significance_level == 0.05
    assert config.power == 0.80
    assert config.pre_period_weeks == 13


def test_validate_panel_data(panel_data):
    result = validate_panel_data(panel_data, sample_size=50)
    assert "total_sampled" in result
    assert "pass_rate" in result
    assert result["total_sampled"] == 50
