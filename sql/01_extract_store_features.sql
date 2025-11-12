-- ============================================================
-- Store Feature Extraction for Q4 2025 Holiday Promo Analysis
-- Source: Azure Databricks / Delta Lake
-- Owner: Measurement & Analytics Team
-- Last Updated: 2025-12-15
-- ============================================================
-- Pull store-level attributes used as covariates in PSM and
-- confounders in our causal models. Joined from multiple
-- source tables in the retail data warehouse.
-- ============================================================

WITH store_base AS (
    SELECT
        s.store_id,
        s.store_name,
        s.region,
        s.state,
        s.store_format,          -- "supercenter", "neighborhood", "express"
        s.open_date,
        s.square_footage         AS store_size,
        s.num_registers,
        s.num_employees_ft
    FROM catalog.retail.dim_stores s
    WHERE s.is_active = 1
      AND s.open_date < '2025-07-01'   -- exclude stores opened during study period
),

revenue_history AS (
    -- Pre-promotion baseline: 26 weeks before campaign launch (Jul-Sep 2025)
    SELECT
        t.store_id,
        AVG(t.net_revenue)       AS avg_weekly_revenue,
        STDDEV(t.net_revenue)    AS std_weekly_revenue,
        SUM(t.net_revenue)       AS total_baseline_revenue,
        AVG(t.transaction_count) AS avg_weekly_transactions,
        AVG(t.units_sold)        AS avg_weekly_units
    FROM catalog.retail.fact_weekly_sales t
    WHERE t.fiscal_week BETWEEN 202527 AND 202539   -- 13-week pre-period
    GROUP BY t.store_id
),

demographics AS (
    SELECT
        d.store_id,
        d.median_household_income,
        d.population_density,
        d.pct_college_educated,
        d.median_age
    FROM catalog.retail.dim_store_demographics d
),

competition AS (
    SELECT
        c.store_id,
        COUNT(c.competitor_id)   AS competitor_count_5mi,
        MIN(c.distance_miles)    AS nearest_competitor_dist
    FROM catalog.retail.dim_competitor_proximity c
    WHERE c.distance_miles <= 5.0
    GROUP BY c.store_id
),

traffic AS (
    SELECT
        f.store_id,
        AVG(f.daily_foot_traffic)  AS avg_daily_foot_traffic,
        AVG(f.parking_utilization) AS avg_parking_util
    FROM catalog.retail.fact_foot_traffic f
    WHERE f.date_key BETWEEN '2025-07-01' AND '2025-09-30'
    GROUP BY f.store_id
),

supply_chain AS (
    SELECT
        sc.store_id,
        sc.primary_dc_id,
        sc.distance_to_dc_miles  AS warehouse_distance,
        sc.avg_delivery_lead_days
    FROM catalog.retail.dim_supply_chain sc
)

SELECT
    sb.store_id,
    sb.store_name,
    sb.region,
    sb.state,
    sb.store_format,
    sb.store_size,
    sb.num_registers,
    rh.avg_weekly_revenue,
    rh.std_weekly_revenue,
    rh.avg_weekly_transactions,
    rh.avg_weekly_units,
    dm.median_household_income,
    dm.population_density,
    dm.pct_college_educated,
    comp.competitor_count_5mi    AS competitor_density,
    comp.nearest_competitor_dist,
    tr.avg_daily_foot_traffic    AS foot_traffic_index,
    tr.avg_parking_util,
    sc.warehouse_distance,
    sc.avg_delivery_lead_days
FROM store_base sb
LEFT JOIN revenue_history rh ON sb.store_id = rh.store_id
LEFT JOIN demographics dm    ON sb.store_id = dm.store_id
LEFT JOIN competition comp   ON sb.store_id = comp.store_id
LEFT JOIN traffic tr         ON sb.store_id = tr.store_id
LEFT JOIN supply_chain sc    ON sb.store_id = sc.store_id
WHERE rh.avg_weekly_revenue IS NOT NULL   -- drop stores with missing baseline
ORDER BY sb.store_id
