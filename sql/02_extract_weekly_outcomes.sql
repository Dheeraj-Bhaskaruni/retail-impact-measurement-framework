-- ============================================================
-- Weekly Outcome Extraction: Q4 2025 Holiday Promo Campaign
-- Covers: 13-week pre-period + 12-week promo period
-- Campaign ID: PROMO-2025-Q4-HOLIDAY
-- ============================================================

WITH promo_assignment AS (
    -- Which stores received the Q4 Holiday promotion
    SELECT
        pa.store_id,
        pa.campaign_id,
        pa.assignment_date,
        1 AS treated
    FROM catalog.retail.dim_promo_assignments pa
    WHERE pa.campaign_id = 'PROMO-2025-Q4-HOLIDAY'
      AND pa.status = 'active'
),

all_stores AS (
    SELECT
        s.store_id,
        COALESCE(pa.treated, 0) AS treated,
        pa.assignment_date
    FROM catalog.retail.dim_stores s
    LEFT JOIN promo_assignment pa ON s.store_id = pa.store_id
    WHERE s.is_active = 1
      AND s.open_date < '2025-07-01'
),

weekly_sales AS (
    SELECT
        ws.store_id,
        ws.fiscal_week,
        ws.week_start_date,
        ws.net_revenue                AS revenue,
        ws.units_sold,
        ws.transaction_count,
        ws.net_revenue / NULLIF(ws.transaction_count, 0) AS avg_basket_size,
        ws.new_customer_count         AS new_customers,
        ws.return_customer_count,
        ws.gross_margin,
        ws.discount_amount,
        ws.promo_markdown_cost
    FROM catalog.retail.fact_weekly_sales ws
    WHERE ws.fiscal_week BETWEEN 202527 AND 202552   -- Jul 2025 - Dec 2025
),

regional_marketing AS (
    -- Regional ad spend as potential instrument (affects promo rollout,
    -- not individual store revenue directly)
    SELECT
        rm.region,
        rm.fiscal_week,
        rm.total_ad_spend           AS regional_ad_spend,
        rm.tv_spend,
        rm.digital_spend
    FROM catalog.marketing.fact_regional_ad_spend rm
    WHERE rm.fiscal_week BETWEEN 202527 AND 202552
)

SELECT
    a.store_id,
    a.treated,
    ws.fiscal_week                  AS week,
    ws.week_start_date,
    CASE
        WHEN ws.fiscal_week >= 202540 THEN 1
        ELSE 0
    END                             AS post_period,      -- promo started week 40
    ws.revenue,
    ws.units_sold,
    ws.transaction_count,
    ws.avg_basket_size              AS basket_size,
    ws.new_customers,
    ws.return_customer_count,
    ws.gross_margin,
    ws.discount_amount,
    ws.promo_markdown_cost,
    rm.regional_ad_spend
FROM all_stores a
INNER JOIN weekly_sales ws ON a.store_id = ws.store_id
LEFT JOIN catalog.retail.dim_stores s ON a.store_id = s.store_id
LEFT JOIN regional_marketing rm
    ON s.region = rm.region
    AND ws.fiscal_week = rm.fiscal_week
ORDER BY a.store_id, ws.fiscal_week
