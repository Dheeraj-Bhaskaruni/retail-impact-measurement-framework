-- ============================================================
-- Promotion Cost Extraction for ROAS Calculation
-- Needed to compute: ROAS = Incremental Revenue / Total Cost
-- ============================================================

SELECT
    pc.campaign_id,
    pc.campaign_name,
    pc.start_date,
    pc.end_date,
    pc.num_stores_enrolled,
    SUM(pc.markdown_cost)       AS total_markdown_cost,
    SUM(pc.signage_cost)        AS total_signage_cost,
    SUM(pc.labor_cost)          AS total_incremental_labor,
    SUM(pc.digital_ad_cost)     AS total_digital_cost,
    SUM(pc.markdown_cost + pc.signage_cost
        + pc.labor_cost + pc.digital_ad_cost) AS total_promo_cost
FROM catalog.retail.fact_promo_costs pc
WHERE pc.campaign_id = 'PROMO-2025-Q4-HOLIDAY'
GROUP BY
    pc.campaign_id,
    pc.campaign_name,
    pc.start_date,
    pc.end_date,
    pc.num_stores_enrolled
