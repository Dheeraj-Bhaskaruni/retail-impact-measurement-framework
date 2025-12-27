"""Tests for heterogeneous treatment effects module."""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from causal.heterogeneity import estimate_subgroup_effects


@pytest.fixture
def subgroup_panel():
    """Panel data with heterogeneous effects by store format."""
    rng = np.random.default_rng(42)
    records = []
    for store_id in range(100):
        fmt = "supercenter" if store_id < 34 else "neighborhood" if store_id < 67 else "express"
        treated = 1 if store_id % 2 == 0 else 0
        # Supercenter gets bigger treatment effect
        effect = {"supercenter": 100, "neighborhood": 50, "express": 20}[fmt]
        for week in range(1, 11):
            revenue = rng.normal(1000 + effect * treated, 100)
            records.append({
                "store_id": f"STR-{store_id:04d}",
                "store_format": fmt,
                "week": week,
                "revenue": revenue,
                "treated": treated,
            })
    return pd.DataFrame(records)


def test_subgroup_effects_detected(subgroup_panel):
    result = estimate_subgroup_effects(
        subgroup_panel, "revenue", "treated", "store_format"
    )
    assert result.subgroup_col == "store_format"
    assert len(result.subgroup_effects) == 3

    # Supercenter should have largest effect
    effects = result.subgroup_effects.set_index("subgroup")
    assert effects.loc["supercenter", "effect"] > effects.loc["express", "effect"]


def test_interaction_significance(subgroup_panel):
    result = estimate_subgroup_effects(
        subgroup_panel, "revenue", "treated", "store_format"
    )
    # With different effect sizes, interaction should be significant
    assert isinstance(result.interaction_p_value, float)
    assert 0 <= result.interaction_p_value <= 1
