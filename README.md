# Retail Promotion Impact Measurement Framework

Causal inference framework for measuring the true incremental impact of retail promotions on store revenue — built to answer the question: **"Did this promotion actually work, or would revenue have grown anyway?"**

## Background

In Q4 2025, the merchandising team rolled out a holiday promotion campaign (`PROMO-2025-Q4-HOLIDAY`) across 197 of our 500 stores. Post-campaign, promoted stores showed ~15% higher revenue than non-promoted stores. Leadership asked whether this justified a $1.2M expansion of the program in 2026.

**The problem**: the merchandising team didn't randomly select stores. They chose larger, higher-traffic, wealthier-area stores — the ones *already* outperforming. That 15% "lift" is contaminated by **selection bias**, **seasonality** (holiday ramp), and **market trends**. Naive before/after or treated/control comparisons are unreliable.

**This framework** applies rigorous causal inference to isolate the true promotion effect from confounding factors, providing leadership with defensible numbers for budget decisions.

## What This Framework Does

### Causal Estimation (multiple methods for robustness)
- **Propensity Score Matching (PSM)** — matches each promoted store to a similar non-promoted store based on 5 covariates (store size, baseline revenue, competitor density, demographics, foot traffic). Removes observable selection bias.
- **Difference-in-Differences (DiD)** — compares revenue *changes* (pre vs post campaign) between treatment and control. Controls for time-invariant store-level confounders.
- **Instrumental Variables (2SLS)** — uses warehouse distance and regional ad spend as instruments to handle *unobserved* confounders that PSM can't address.
- **A/B Test Design** — power analysis and sample size calculator for future campaigns where we CAN randomize. Includes sequential testing (SPRT) for early stopping.

### Attribution & Reporting
- **Revenue decomposition**: breaks down revenue changes into promotion effect vs. seasonality vs. trend vs. store characteristics
- **KPI framework**: standardized definitions for incremental revenue, ROAS, customer acquisition lift, basket size change, margin impact
- **Automated pipeline**: end-to-end from data extraction through causal estimation to JSON reports

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data Warehouse | Databricks / Delta Lake (Azure) |
| Data Extraction | SQL (see `sql/` directory) |
| Analysis | Python 3.9+, statsmodels, scipy, scikit-learn |
| Causal Inference | DoWhy, EconML |
| Visualization | matplotlib, seaborn |
| Pipeline | Python orchestration (Databricks Jobs in prod) |
| CI/CD | GitHub Actions |
| Version Control | Git + GitHub |

## Project Structure

```
retail-impact-measurement-framework/
├── sql/                                # Data extraction queries (Databricks SQL)
│   ├── 01_extract_store_features.sql   #   dim_stores + demographics + competition
│   ├── 02_extract_weekly_outcomes.sql  #   fact_weekly_sales + promo assignments
│   └── 03_extract_promotion_costs.sql  #   fact_promo_costs for ROAS
├── config/
│   └── config.yaml                     # Pipeline parameters (covariates, caliper, etc.)
├── data/
│   ├── raw/                            # Extracted data (gitignored)
│   ├── processed/                      # Pipeline outputs (gitignored)
│   └── synthetic/generate_data.py      # Dev/CI data generator (mirrors prod schema)
├── notebooks/
│   ├── 01_eda_and_kpi_definition.ipynb       # Data exploration + KPI definitions
│   ├── 02_propensity_score_matching.ipynb    # PSM analysis + balance diagnostics
│   ├── 03_ab_test_analysis.ipynb             # Power analysis + test design
│   ├── 04_instrumental_variables.ipynb       # IV/2SLS + instrument validity
│   ├── 05_dashboard_and_reporting.ipynb      # Executive summary + attribution
│   └── 06_sensitivity_and_heterogeneity.ipynb # Rosenbaum bounds + HTE analysis
├── src/
│   ├── causal/
│   │   ├── propensity_score.py         # PSM: logistic PS, NN matching, balance checks
│   │   ├── diff_in_diff.py             # DiD: TWFE with clustered SEs, parallel trends test
│   │   ├── instrumental_variables.py   # IV: 2SLS, first-stage F, Sargan test
│   │   └── ab_testing.py               # Welch's t-test, power analysis, SPRT, Bonferroni
│   ├── metrics/
│   │   ├── kpi_framework.py            # KPI registry + calculator
│   │   └── attribution.py              # Revenue decomposition model
│   ├── data/
│   │   ├── data_loader.py              # Load + validate extracted data
│   │   └── feature_engineering.py      # Pre-treatment aggregations, standardization
│   ├── pipeline/
│   │   └── measurement_pipeline.py     # End-to-end orchestration
│   └── utils/
│       ├── statistical_tests.py        # Bootstrap CI, permutation tests, KS, Levene
│       └── visualization.py            # Propensity plots, Love plots, waterfall charts
├── tests/                              # 45+ tests covering all modules
├── docs/
│   └── methodology.md                  # Statistical methodology documentation
├── .github/workflows/ci.yml            # Lint + test on Python 3.9/3.10/3.11
├── Makefile                            # make setup / data / pipeline / test
└── requirements.txt
```

## Setup & Usage

### Local Development
```bash
# One-time setup
make setup

# Generate development data (when you don't have warehouse access)
make data

# Run the full measurement pipeline
make pipeline

# Run tests
make test
```

### Production (Databricks)
In production, the pipeline runs as a Databricks Job:
1. SQL queries in `sql/` extract data from Delta Lake tables
2. `measurement_pipeline.py` runs on a Databricks cluster
3. Results are written to `catalog.analytics.promotion_measurement_results`
4. Dashboards in Databricks SQL pull from the results table

## Key Results (Q4 2025 Holiday Campaign)

| Method | Estimated ATT | p-value | Status |
|--------|--------------|---------|--------|
| Naive comparison | +15.2% | — | **Biased** (includes selection effects) |
| Propensity Score Matching | +7.8% | 0.001 | Significant |
| Difference-in-Differences | +8.0% | <0.001 | Significant, parallel trends validated |
| Instrumental Variables | +8.3% | 0.012 | Significant, strong instruments (F=24.6) |

**Bottom line**: The true promotion effect is ~8%, roughly half of what the naive comparison suggested. After accounting for $1.2M in promotion costs, the ROAS is 3.2x — the campaign is profitable and expansion is justified, but ROI projections should use the 8% figure, not 15%.

### Attribution Breakdown
- Promotion effect: **74%** of observed revenue difference
- Seasonality (holiday ramp): 18%
- Store characteristics (selection): 6%
- Residual: 2%

### Sensitivity & Targeting
- Rosenbaum bounds: results robust up to Gamma ~2.1 (strong)
- Supercenters show highest promotion lift — prioritize for Q1 2026 expansion
- Express stores show minimal lift — may not justify per-store cost

## Methodology

See [docs/methodology.md](docs/methodology.md) for detailed documentation of statistical methods, assumptions, and diagnostic procedures.

## Author

**Dheeraj Bhaskaruni** — Measurement & Causal Inference
