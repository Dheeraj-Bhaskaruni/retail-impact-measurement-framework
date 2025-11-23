"""
Propensity Score Matching (PSM) for causal inference.

Estimates the Average Treatment Effect on the Treated (ATT) by matching
treated units to similar control units based on estimated propensity scores.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class PSMResult:
    """Results from propensity score matching analysis."""
    att: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_treated: int
    n_matched: int
    balance_table: pd.DataFrame


def estimate_propensity_scores(df: pd.DataFrame,
                                treatment_col: str,
                                covariates: List[str]) -> np.ndarray:
    """Estimate propensity scores using logistic regression."""
    X = df[covariates].values
    y = df[treatment_col].values

    model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    model.fit(X, y)
    return model.predict_proba(X)[:, 1]


def match_nearest_neighbor(propensity_scores: np.ndarray,
                           treatment: np.ndarray,
                           caliper: float = 0.05,
                           n_neighbors: int = 1) -> List[Tuple[int, int]]:
    """Match treated units to nearest control units within caliper."""
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    control_scores = propensity_scores[control_idx].reshape(-1, 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(control_scores)

    matches = []
    for t_idx in treated_idx:
        t_score = propensity_scores[t_idx].reshape(1, -1)
        distances, indices = nn.kneighbors(t_score)

        for dist, c_local_idx in zip(distances[0], indices[0]):
            if dist <= caliper:
                matches.append((t_idx, control_idx[c_local_idx]))

    return matches


def assess_balance(df: pd.DataFrame,
                   covariates: List[str],
                   treatment_col: str,
                   matched_indices=None) -> pd.DataFrame:
    """Compute standardized mean differences to assess covariate balance."""
    if matched_indices is not None:
        treated_idx = [m[0] for m in matched_indices]
        control_idx = [m[1] for m in matched_indices]
        treated = df.iloc[treated_idx]
        control = df.iloc[control_idx]
    else:
        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]

    rows = []
    for cov in covariates:
        t_mean = treated[cov].mean()
        c_mean = control[cov].mean()
        t_std = treated[cov].std()
        c_std = control[cov].std()
        pooled_std = np.sqrt((t_std**2 + c_std**2) / 2)
        smd = (t_mean - c_mean) / pooled_std if pooled_std > 0 else 0

        rows.append({
            "covariate": cov,
            "treated_mean": t_mean,
            "control_mean": c_mean,
            "std_mean_diff": smd,
            "balanced": abs(smd) < 0.1,
        })
    return pd.DataFrame(rows)


def run_psm(df: pd.DataFrame,
            outcome_col: str,
            treatment_col: str,
            covariates: List[str],
            caliper: float = 0.05,
            alpha: float = 0.05) -> PSMResult:
    """Run full propensity score matching pipeline."""
    ps = estimate_propensity_scores(df, treatment_col, covariates)
    df = df.copy()
    df["propensity_score"] = ps

    matches = match_nearest_neighbor(ps, df[treatment_col].values, caliper=caliper)

    if len(matches) == 0:
        raise ValueError(
            f"No matches found within caliper={caliper}. "
            f"Try increasing caliper or check propensity score overlap."
        )

    treated_outcomes = np.array([df.iloc[m[0]][outcome_col] for m in matches])
    control_outcomes = np.array([df.iloc[m[1]][outcome_col] for m in matches])
    diffs = treated_outcomes - control_outcomes

    att = np.mean(diffs)
    se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
    t_stat = att / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(diffs) - 1))
    ci_lower = att - stats.t.ppf(1 - alpha / 2, df=len(diffs) - 1) * se
    ci_upper = att + stats.t.ppf(1 - alpha / 2, df=len(diffs) - 1) * se

    balance = assess_balance(df, covariates, treatment_col, matches)

    return PSMResult(
        att=att, se=se, ci_lower=ci_lower, ci_upper=ci_upper,
        p_value=p_value, n_treated=int(df[treatment_col].sum()),
        n_matched=len(matches), balance_table=balance,
    )
