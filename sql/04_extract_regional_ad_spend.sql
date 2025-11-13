-- ============================================================
-- Regional Ad Spend Extraction (Instrumental Variable)
-- Used as IV in 2SLS — affects promo rollout decisions but
-- should not directly affect individual store revenue
-- ============================================================

SELECT
    rm.region,
    rm.fiscal_week,
    rm.total_ad_spend           AS regional_ad_spend,
    rm.tv_spend,
    rm.digital_spend,
    rm.print_spend,
    rm.total_ad_spend / NULLIF(
        (SELECT COUNT(DISTINCT s.store_id)
         FROM catalog.retail.dim_stores s
         WHERE s.region = rm.region AND s.is_active = 1),
        0
    )                           AS ad_spend_per_store
FROM catalog.marketing.fact_regional_ad_spend rm
WHERE rm.fiscal_week BETWEEN 202527 AND 202552
ORDER BY rm.region, rm.fiscal_week
