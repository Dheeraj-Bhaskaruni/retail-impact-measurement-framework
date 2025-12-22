# Statistical Methodology — Promotion Impact Measurement

**Document Owner**: Measurement & Analytics Team
**Last Updated**: December 2025
**Campaign**: PROMO-2025-Q4-HOLIDAY

## 1. Problem Statement

The merchandising team rolled the Q4 holiday promotion into 197 stores based on internal selection criteria (store size, historical performance, regional strategy). Because stores were **not randomly assigned**, a naive treated-vs-control revenue comparison conflates the true promotion effect with pre-existing differences between store groups.

We need methods that separate the causal promotion effect from:
- **Selection bias**: promoted stores were already higher-performing
- **Seasonality**: holiday shopping ramp affects all stores
- **Market trends**: economic conditions, competitor actions
- **Store heterogeneity**: location quality, management, demographics

## 2. Methods

### 2.1 Propensity Score Matching (PSM)

**Purpose**: Remove observable selection bias by matching promoted stores to similar non-promoted stores.

**Implementation**:
1. Estimate P(Promoted | X) via logistic regression on 5 covariates:
   - `store_size`, `avg_weekly_revenue`, `competitor_density`, `median_household_income`, `foot_traffic_index`
2. 1:1 nearest-neighbor matching with caliper = 0.05
3. Verify covariate balance: Standardized Mean Difference < 0.1 on all covariates
4. Estimate ATT = mean(outcome_treated - outcome_matched_control)
5. Inference via paired t-test on matched differences

**Key Assumption**: Conditional Independence — no unobserved confounders after conditioning on X. Violated if, e.g., store manager quality affects both promotion assignment and revenue.

**Diagnostics**:
- Propensity score overlap (common support)
- Love plot (SMD before/after matching)
- Rosenbaum sensitivity analysis for hidden bias

### 2.2 Difference-in-Differences (DiD)

**Purpose**: Exploit the temporal structure — compare revenue *changes* (pre vs post campaign) between treated and control stores.

**Model**:
```
Revenue_it = alpha_i + gamma_t + delta * (Treated_i x Post_t) + epsilon_it
```
- `alpha_i`: store fixed effects (absorbs all time-invariant store differences)
- `gamma_t`: week fixed effects (absorbs seasonality and macro trends)
- `delta`: **the causal estimand** (ATT)
- Standard errors clustered at store level

**Key Assumption**: Parallel trends — absent the promotion, treated and control stores would have followed the same revenue trajectory.

**Diagnostics**:
- Pre-trend interaction test: regress revenue on `Treatment x Time` in pre-period. Non-significant coefficient (p > 0.05) supports parallel trends.
- Visual inspection of pre-treatment trends

### 2.3 Instrumental Variables / 2SLS

**Purpose**: Address endogeneity from *unobserved* confounders that PSM cannot handle.

**Instruments**:
- `warehouse_distance`: distance to primary distribution center. Affects logistics cost structure and which stores were operationally feasible for the promotion, but shouldn't directly affect consumer demand.
- `regional_ad_spend`: marketing budget allocated at regional level. Affects promotion rollout decisions but is too aggregated to directly impact individual store revenue.

**Stage 1**: Treatment_i = pi_0 + pi_1 * Z_i + pi_2 * X_i + v_i
**Stage 2**: Revenue_i = beta_0 + beta_1 * Treatment_hat_i + beta_2 * X_i + u_i

**Diagnostics**:
- First-stage F-statistic > 10 (Staiger & Stock rule for strong instruments)
- Sargan overidentification test (valid when #instruments > #endogenous variables)
- Comparison with OLS and PSM estimates for consistency

### 2.4 A/B Test Design (Prospective)

**Purpose**: For future campaigns, design properly randomized experiments that don't require observational corrections.

**Components**:
- Power analysis: given baseline revenue and variance, calculate required sample sizes for target MDE
- Welch's t-test for unequal variances
- Sequential Probability Ratio Test (SPRT) for early stopping — reduces cost of experiments
- Bonferroni correction when testing multiple KPIs simultaneously

## 3. Attribution Model

Revenue decomposition via structural regression:

```
log(Revenue_it) = Store_FE + beta_1 * sin(2*pi*t/52) + beta_2 * cos(2*pi*t/52) + beta_3 * trend + delta * Treatment_i + epsilon_it
```

This decomposes the treated-control revenue gap into:
- **Promotion effect** (delta)
- **Seasonality** (Fourier terms)
- **Trend** (linear time)
- **Store characteristics** (fixed effects)

## 4. Robustness & Validation

All estimates validated by:
1. **Cross-method agreement**: PSM, DiD, and IV should yield similar ATT estimates. Large discrepancies indicate assumption violations.
2. **Placebo tests**: run the same analysis on pre-treatment outcomes where we know the true effect is zero.
3. **Sensitivity analysis**: Rosenbaum bounds for PSM; varying caliper width; alternative instrument sets for IV.
4. **Heterogeneity analysis**: estimate treatment effects by store format (supercenter vs neighborhood vs express) and region.

## 5. Limitations

- PSM only controls for observables — if unobserved factors (e.g., local events, manager tenure) drove both selection and outcomes, PSM estimates are biased
- DiD requires parallel trends, which may not hold if the merchandising team selected stores on *growth trajectory*
- IV estimates are only as good as the instruments — if warehouse distance or regional ad spend have direct effects on revenue, IV is invalid
- All methods assume Stable Unit Treatment Value Assumption (SUTVA) — no spillover between stores. This could be violated for geographically proximate stores.
