-- ============================================================
-- Data Quality Checks — Run Before Pipeline Execution
-- Validates completeness and consistency of source tables
-- ============================================================

-- Check 1: Store count matches expectations
SELECT 'store_count' AS check_name,
       COUNT(*) AS value,
       CASE WHEN COUNT(*) BETWEEN 450 AND 550 THEN 'PASS' ELSE 'FAIL' END AS status
FROM catalog.retail.dim_stores
WHERE is_active = 1

UNION ALL

-- Check 2: No missing weeks in the study period
SELECT 'week_completeness' AS check_name,
       COUNT(DISTINCT fiscal_week) AS value,
       CASE WHEN COUNT(DISTINCT fiscal_week) = 25 THEN 'PASS' ELSE 'FAIL' END AS status
FROM catalog.retail.fact_weekly_sales
WHERE fiscal_week BETWEEN 202527 AND 202551

UNION ALL

-- Check 3: No null revenue
SELECT 'null_revenue' AS check_name,
       SUM(CASE WHEN net_revenue IS NULL THEN 1 ELSE 0 END) AS value,
       CASE WHEN SUM(CASE WHEN net_revenue IS NULL THEN 1 ELSE 0 END) = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM catalog.retail.fact_weekly_sales
WHERE fiscal_week BETWEEN 202527 AND 202551

UNION ALL

-- Check 4: Treatment assignment is populated
SELECT 'treatment_assigned' AS check_name,
       COUNT(*) AS value,
       CASE WHEN COUNT(*) > 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM catalog.retail.dim_promo_assignments
WHERE campaign_id = 'PROMO-2025-Q4-HOLIDAY'

UNION ALL

-- Check 5: No duplicate store-weeks
SELECT 'duplicate_store_weeks' AS check_name,
       COUNT(*) AS value,
       CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM (
    SELECT store_id, fiscal_week, COUNT(*) AS cnt
    FROM catalog.retail.fact_weekly_sales
    WHERE fiscal_week BETWEEN 202527 AND 202551
    GROUP BY store_id, fiscal_week
    HAVING COUNT(*) > 1
)
