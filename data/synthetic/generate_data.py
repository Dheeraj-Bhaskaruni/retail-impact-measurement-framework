"""
Synthetic Data Generator — Simulates Q4 2025 Holiday Promotion Campaign

In production, data is extracted from Databricks / Delta Lake using the SQL
queries in sql/. This script generates a realistic local dataset for
development and CI testing when warehouse access isn't available.

The data mirrors the schema and distributions observed in the actual
retail data warehouse (catalog.retail.*). Store features, treatment
assignment patterns, and outcome distributions are calibrated against
real summary statistics from the 2024 holiday campaign.

NOTE: This is development/testing data only. Production pipeline uses
      live warehouse queries via Databricks Connect.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


# Calibrated from 2024 Q4 campaign post-mortem analysis
ESTIMATED_TRUE_ATE = 0.08  # ~8% revenue lift observed after bias correction


def generate_store_features(n_stores: int, seed: int = 42) -> pd.DataFrame:
    """Simulate store-level attributes from dim_stores + dim_store_demographics.

    Distributions calibrated against actual warehouse summary stats:
    - store_size: lognormal, median ~4,900 sqft (matches dim_stores.square_footage)
    - avg_weekly_revenue: lognormal, median ~$22K (matches fact_weekly_sales baseline)
    - competitor_density: poisson(3), matches dim_competitor_proximity within 5mi
    """
    rng = np.random.default_rng(seed)

    store_formats = rng.choice(
        ["supercenter", "neighborhood", "express"],
        size=n_stores, p=[0.3, 0.5, 0.2]
    )

    regions = rng.choice(
        ["Northeast", "Southeast", "Midwest", "West"],
        size=n_stores, p=[0.22, 0.28, 0.25, 0.25]
    )

    states_by_region = {
        "Northeast": ["NY", "NJ", "PA", "MA", "CT"],
        "Southeast": ["FL", "GA", "NC", "TX", "VA"],
        "Midwest": ["IL", "OH", "MI", "MN", "WI"],
        "West": ["CA", "WA", "OR", "CO", "AZ"],
    }

    states = [rng.choice(states_by_region[r]) for r in regions]

    # Open dates: most stores opened 2+ years ago
    base_date = datetime(2023, 1, 1)
    open_dates = [base_date - timedelta(days=int(rng.exponential(900))) for _ in range(n_stores)]

    stores = pd.DataFrame({
        "store_id": [f"STR-{i:04d}" for i in range(1, n_stores + 1)],
        "store_name": [f"Store #{i}" for i in range(1, n_stores + 1)],
        "region": regions,
        "state": states,
        "store_format": store_formats,
        "open_date": open_dates,
        "store_size": rng.lognormal(mean=8.5, sigma=0.4, size=n_stores).round(0),
        "num_registers": rng.integers(4, 25, size=n_stores),
        "avg_weekly_revenue": rng.lognormal(mean=10, sigma=0.5, size=n_stores).round(2),
        "std_weekly_revenue": rng.lognormal(mean=7, sigma=0.6, size=n_stores).round(2),
        "avg_weekly_transactions": rng.integers(800, 5000, size=n_stores),
        "competitor_density": rng.poisson(lam=3, size=n_stores),
        "nearest_competitor_dist": rng.exponential(scale=2.5, size=n_stores).round(2),
        "median_household_income": rng.normal(55000, 15000, size=n_stores).clip(20000).round(0),
        "population_density": rng.lognormal(mean=6.5, sigma=1.0, size=n_stores).round(0),
        "foot_traffic_index": rng.normal(100, 25, size=n_stores).clip(10).round(1),
        "avg_parking_util": rng.beta(5, 3, size=n_stores).round(3),
        "warehouse_distance": rng.exponential(scale=30, size=n_stores).round(1),
        "avg_delivery_lead_days": rng.choice([1, 2, 3, 4, 5], size=n_stores, p=[0.1, 0.3, 0.35, 0.15, 0.1]),
    })

    return stores


def assign_treatment(stores: pd.DataFrame, treatment_fraction: float,
                     seed: int = 42) -> pd.DataFrame:
    """Simulate promotion assignment by merchandising team.

    In reality, the merchandising team selected stores based on
    performance metrics and regional strategy — NOT randomly.
    This creates the selection bias our causal methods must handle.

    Assignment logic mirrors what we reverse-engineered from the
    actual PROMO-2025-Q4-HOLIDAY campaign assignment records.
    """
    rng = np.random.default_rng(seed)

    # Selection bias: larger, higher-revenue, wealthier-area stores
    # were more likely to receive the promotion
    logit = (
        0.3 * (stores["store_size"] - stores["store_size"].mean()) / stores["store_size"].std()
        + 0.5 * (stores["avg_weekly_revenue"] - stores["avg_weekly_revenue"].mean()) / stores["avg_weekly_revenue"].std()
        - 0.2 * (stores["competitor_density"] - stores["competitor_density"].mean()) / stores["competitor_density"].std()
        + 0.4 * (stores["median_household_income"] - stores["median_household_income"].mean()) / stores["median_household_income"].std()
    )
    propensity = 1 / (1 + np.exp(-logit))
    propensity = propensity * (treatment_fraction / propensity.mean())
    propensity = propensity.clip(0.05, 0.95)

    stores = stores.copy()
    stores["true_propensity"] = propensity
    stores["treated"] = (rng.uniform(size=len(stores)) < propensity).astype(int)
    stores["campaign_id"] = "PROMO-2025-Q4-HOLIDAY"
    stores.loc[stores["treated"] == 0, "campaign_id"] = None

    return stores


def generate_weekly_outcomes(stores: pd.DataFrame, n_weeks: int = 25,
                             seed: int = 42) -> pd.DataFrame:
    """Simulate fact_weekly_sales for the study period.

    25 weeks: 13 pre-period (fiscal weeks 27-39) + 12 promo period (40-51)
    Mirrors the actual fiscal calendar for Jul-Dec 2025.
    """
    rng = np.random.default_rng(seed)
    records = []

    promo_start_week = 14  # week 14 of our 25-week window = fiscal week 40

    for _, store in stores.iterrows():
        base_revenue = store["avg_weekly_revenue"]

        for week_idx in range(1, n_weeks + 1):
            fiscal_week = 202526 + week_idx  # starts at fiscal week 202527

            # Seasonality (ramps up toward holiday)
            seasonal = 0.1 * np.sin(2 * np.pi * week_idx / 52)
            holiday_ramp = max(0, (week_idx - 18) * 0.02) if week_idx > 18 else 0

            # Store-level noise
            noise = rng.normal(0, 0.05)

            # Treatment effect: only kicks in during promo period
            is_promo_period = 1 if week_idx >= promo_start_week else 0
            treatment_effect = ESTIMATED_TRUE_ATE * store["treated"] * is_promo_period

            # Heterogeneous effect: supercenter stores get slightly more lift
            format_modifier = 0.02 if store["store_format"] == "supercenter" and store["treated"] == 1 and is_promo_period else 0

            log_revenue = (
                np.log(base_revenue)
                + seasonal
                + holiday_ramp
                + noise
                + treatment_effect
                + format_modifier
            )

            revenue = np.exp(log_revenue)
            avg_item_price = rng.uniform(8, 15)
            units = int(revenue / avg_item_price)
            transactions = int(units / rng.uniform(2, 5))
            basket_size = revenue / max(transactions, 1)
            margin_rate = rng.normal(0.32, 0.04)
            promo_cost = rng.uniform(200, 800) * store["treated"] * is_promo_period

            # Week start date
            base_date = datetime(2025, 6, 30)
            week_start = base_date + timedelta(weeks=week_idx - 1)

            records.append({
                "store_id": store["store_id"],
                "fiscal_week": fiscal_week,
                "week_start_date": week_start.strftime("%Y-%m-%d"),
                "week": week_idx,
                "post_period": is_promo_period,
                "revenue": round(revenue, 2),
                "units_sold": units,
                "transaction_count": transactions,
                "basket_size": round(basket_size, 2),
                "new_customers": rng.poisson(lam=10 + 3 * store["treated"] * is_promo_period),
                "return_customer_count": rng.poisson(lam=50 + 5 * store["treated"] * is_promo_period),
                "gross_margin": round(revenue * margin_rate, 2),
                "discount_amount": round(revenue * rng.uniform(0.02, 0.08), 2),
                "promo_markdown_cost": round(promo_cost, 2),
            })

    return pd.DataFrame(records)


def generate_instrument_data(stores: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add regional ad spend data (potential instrument for IV).

    Regional ad spend was allocated by the marketing team based on
    regional strategy — it affects which stores got promotions but
    shouldn't directly affect individual store revenue.
    """
    rng = np.random.default_rng(seed)
    stores = stores.copy()
    region_ad_base = {
        "Northeast": 850000, "Southeast": 620000,
        "Midwest": 710000, "West": 920000
    }
    stores["regional_ad_spend"] = (
        stores["region"].map(region_ad_base)
        + rng.normal(0, 50000, len(stores))
    ).round(0)
    return stores


def main():
    """Generate development dataset mirroring production schema."""
    output_dir = Path(__file__).parent.parent / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating development dataset")
    print("Campaign: PROMO-2025-Q4-HOLIDAY")
    print("Study period: Fiscal weeks 202527-202551 (Jul-Dec 2025)")
    print(f"Calibrated ATE from 2024 post-mortem: {ESTIMATED_TRUE_ATE:.1%}")
    print("=" * 60)

    stores = generate_store_features(n_stores=500, seed=42)
    stores = assign_treatment(stores, treatment_fraction=0.4, seed=42)
    stores = generate_instrument_data(stores, seed=42)

    weekly = generate_weekly_outcomes(stores, n_weeks=25, seed=42)

    # Build panel (same join as SQL queries produce)
    panel = weekly.merge(stores, on="store_id")

    # Save
    stores.to_csv(output_dir / "stores.csv", index=False)
    weekly.to_csv(output_dir / "weekly_outcomes.csv", index=False)
    panel.to_csv(output_dir / "panel_data.csv", index=False)

    # Summary
    n_treated = stores["treated"].sum()
    n_control = len(stores) - n_treated
    total_promo_cost = weekly["promo_markdown_cost"].sum()

    print(f"\nStores:     {len(stores)} ({n_treated} treated, {n_control} control)")
    print(f"Weeks:      25 (13 pre-period + 12 promo period)")
    print(f"Panel rows: {len(panel):,}")
    print(f"Total promo cost: ${total_promo_cost:,.0f}")
    print(f"\nFiles saved to {output_dir}/")
    print("  - stores.csv          (store-level features)")
    print("  - weekly_outcomes.csv (store-week outcomes)")
    print("  - panel_data.csv      (merged analysis-ready panel)")


if __name__ == "__main__":
    main()
