"""
Data Validation Schemas using Pydantic.

Enforces data contracts between the SQL extraction layer and
the analysis pipeline. Catches schema drift and data quality
issues before they corrupt causal estimates.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import date


class StoreRecord(BaseModel):
    """Schema for a single store record from dim_stores."""
    store_id: str = Field(pattern=r"^STR-\d{4}$")
    store_name: str
    region: str = Field(pattern=r"^(Northeast|Southeast|Midwest|West)$")
    state: str = Field(min_length=2, max_length=2)
    store_format: str = Field(pattern=r"^(supercenter|neighborhood|express)$")
    store_size: float = Field(gt=0)
    avg_weekly_revenue: float = Field(gt=0)
    competitor_density: int = Field(ge=0)
    median_household_income: float = Field(gt=0)
    foot_traffic_index: float = Field(gt=0)
    warehouse_distance: float = Field(ge=0)
    treated: int = Field(ge=0, le=1)

    @field_validator("median_household_income")
    @classmethod
    def income_reasonable(cls, v):
        if v > 500000:
            raise ValueError(f"Median income {v} seems unreasonably high")
        return v


class WeeklyOutcomeRecord(BaseModel):
    """Schema for a store-week outcome from fact_weekly_sales."""
    store_id: str
    week: int = Field(ge=1)
    revenue: float = Field(gt=0)
    units_sold: int = Field(ge=0)
    basket_size: float = Field(gt=0)
    new_customers: int = Field(ge=0)
    treated: int = Field(ge=0, le=1)
    post_period: int = Field(ge=0, le=1)

    @field_validator("revenue")
    @classmethod
    def revenue_not_extreme(cls, v):
        """Flag single-week store revenue over $10M as likely data error."""
        if v > 10_000_000:
            raise ValueError(f"Weekly revenue ${v:,.0f} exceeds $10M — likely data error")
        return v


class PipelineConfig(BaseModel):
    """Validated pipeline configuration."""
    campaign_id: str
    caliper: float = Field(gt=0, lt=1)
    significance_level: float = Field(gt=0, lt=1, default=0.05)
    power: float = Field(gt=0, lt=1, default=0.80)
    pre_period_weeks: int = Field(gt=0, default=13)
    total_promo_cost: float = Field(ge=0, default=0)


def validate_panel_data(df, sample_size: int = 100) -> dict:
    """
    Validate a panel DataFrame against the WeeklyOutcomeRecord schema.
    Samples rows for performance on large datasets.

    Returns validation report with pass/fail and error details.
    """
    import pandas as pd

    sample = df.sample(min(sample_size, len(df)), random_state=42)
    errors = []
    validated = 0

    for idx, row in sample.iterrows():
        try:
            WeeklyOutcomeRecord(**row.to_dict())
            validated += 1
        except Exception as e:
            errors.append({"row_index": idx, "error": str(e)})

    return {
        "total_sampled": len(sample),
        "valid": validated,
        "invalid": len(errors),
        "pass_rate": validated / len(sample) if len(sample) > 0 else 0,
        "passed": len(errors) == 0,
        "errors": errors[:10],  # First 10 errors only
    }


def validate_store_data(df, sample_size: int = 100) -> dict:
    """Validate store DataFrame against StoreRecord schema."""
    import pandas as pd

    sample = df.sample(min(sample_size, len(df)), random_state=42)
    errors = []
    validated = 0

    required_cols = list(StoreRecord.model_fields.keys())
    available_cols = [c for c in required_cols if c in df.columns]

    for idx, row in sample.iterrows():
        try:
            record_dict = {k: row[k] for k in available_cols if k in row.index}
            StoreRecord(**record_dict)
            validated += 1
        except Exception as e:
            errors.append({"row_index": idx, "error": str(e)})

    return {
        "total_sampled": len(sample),
        "valid": validated,
        "invalid": len(errors),
        "pass_rate": validated / len(sample) if len(sample) > 0 else 0,
        "passed": len(errors) == 0,
        "missing_columns": list(set(required_cols) - set(df.columns)),
        "errors": errors[:10],
    }
