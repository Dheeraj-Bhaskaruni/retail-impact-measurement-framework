"""
KPI Framework for Retail Promotion Measurement.

Defines, calculates, and monitors key performance indicators
that quantify promotion effectiveness.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict
from enum import Enum


class KPICategory(Enum):
    REVENUE = "revenue"
    CUSTOMER = "customer"
    OPERATIONAL = "operational"


@dataclass
class KPIDefinition:
    """Formal definition of a KPI with business context."""
    name: str
    category: KPICategory
    description: str
    formula: str
    unit: str
    direction: str


KPI_REGISTRY: Dict[str, KPIDefinition] = {
    "incremental_revenue": KPIDefinition(
        name="Incremental Revenue",
        category=KPICategory.REVENUE,
        description="Additional revenue attributable to the promotion (causal estimate)",
        formula="ATT * treatment_group_size * avg_weeks_exposed",
        unit="USD",
        direction="higher_is_better",
    ),
    "roas": KPIDefinition(
        name="Return on Ad Spend",
        category=KPICategory.REVENUE,
        description="Incremental revenue per dollar of promotion cost",
        formula="incremental_revenue / promotion_cost",
        unit="ratio",
        direction="higher_is_better",
    ),
    "incremental_units": KPIDefinition(
        name="Incremental Units Sold",
        category=KPICategory.REVENUE,
        description="Additional units sold due to promotion",
        formula="ATT_units * treatment_group_size",
        unit="units",
        direction="higher_is_better",
    ),
    "customer_acquisition_rate": KPIDefinition(
        name="Customer Acquisition Rate",
        category=KPICategory.CUSTOMER,
        description="New customer rate in treated vs control stores",
        formula="(new_customers_treated - new_customers_control) / new_customers_control",
        unit="percentage",
        direction="higher_is_better",
    ),
    "basket_size_lift": KPIDefinition(
        name="Basket Size Lift",
        category=KPICategory.CUSTOMER,
        description="Change in average basket size from promotion",
        formula="avg_basket_treated - avg_basket_control",
        unit="USD",
        direction="higher_is_better",
    ),
    "margin_impact": KPIDefinition(
        name="Margin Impact",
        category=KPICategory.REVENUE,
        description="Net margin change accounting for promotion costs",
        formula="incremental_revenue * margin_rate - promotion_cost",
        unit="USD",
        direction="higher_is_better",
    ),
    "cost_per_incremental_unit": KPIDefinition(
        name="Cost Per Incremental Unit",
        category=KPICategory.OPERATIONAL,
        description="Promotion spend required to generate one additional unit sale",
        formula="promotion_cost / incremental_units",
        unit="USD",
        direction="lower_is_better",
    ),
}


class KPICalculator:
    """Calculate KPIs from panel data and causal estimates."""

    def __init__(self, panel: pd.DataFrame, treatment_col: str = "treated"):
        self.panel = panel
        self.treatment_col = treatment_col
        self._treated = panel[panel[treatment_col] == 1]
        self._control = panel[panel[treatment_col] == 0]

    def compute_naive_lift(self, metric_col: str) -> Dict[str, float]:
        """Simple treated vs control comparison (biased without causal adjustment)."""
        t_mean = self._treated[metric_col].mean()
        c_mean = self._control[metric_col].mean()
        return {
            "treated_mean": t_mean,
            "control_mean": c_mean,
            "naive_lift": t_mean - c_mean,
            "naive_lift_pct": (t_mean - c_mean) / c_mean * 100,
        }

    def compute_incremental_revenue(self, att: float,
                                     n_treated_stores: int,
                                     n_weeks: int) -> float:
        """Convert ATT (per-store-week) to total incremental revenue."""
        avg_revenue = self._treated["revenue"].mean()
        return att * avg_revenue * n_treated_stores * n_weeks

    def compute_roas(self, incremental_revenue: float,
                     promotion_cost: float) -> float:
        """Return on Ad Spend.

        ROAS > 1.0 means the promotion generated more revenue
        than it cost. Industry benchmark for retail: 3-5x.
        """
        if promotion_cost == 0:
            return float("inf")
        return incremental_revenue / promotion_cost

    def generate_kpi_report(self, causal_estimates: Dict[str, float],
                            promotion_cost: float = 0) -> pd.DataFrame:
        """Generate comprehensive KPI report combining naive and causal metrics."""
        n_treated = self._treated["store_id"].nunique()
        n_weeks = self.panel["week"].nunique()

        naive = self.compute_naive_lift("revenue")
        att = causal_estimates.get("att_revenue", 0)
        incremental_rev = self.compute_incremental_revenue(att, n_treated, n_weeks)

        rows = [
            {"kpi": "Naive Revenue Lift (%)", "value": naive["naive_lift_pct"], "note": "Biased - includes selection effects"},
            {"kpi": "Causal ATT (per store-week)", "value": att, "note": "Unbiased estimate from PSM/IV"},
            {"kpi": "Total Incremental Revenue", "value": incremental_rev, "note": "Extrapolated to full treatment group"},
            {"kpi": "ROAS", "value": self.compute_roas(incremental_rev, promotion_cost) if promotion_cost > 0 else None, "note": "Revenue per dollar spent"},
            {"kpi": "Treated Stores", "value": n_treated, "note": ""},
            {"kpi": "Observation Weeks", "value": n_weeks, "note": ""},
        ]
        return pd.DataFrame(rows)
