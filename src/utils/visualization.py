"""Visualization utilities for measurement reports."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional


def plot_propensity_distribution(ps_treated: np.ndarray,
                                 ps_control: np.ndarray,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """Plot propensity score distributions for treated vs control.

    Good overlap between distributions indicates matching is feasible.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(ps_treated, bins=50, alpha=0.5, label="Treated", density=True, color="#2196F3")
    ax.hist(ps_control, bins=50, alpha=0.5, label="Control", density=True, color="#FF9800")
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Density")
    ax.set_title("Propensity Score Distribution: Treated vs Control")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_balance(balance_df: pd.DataFrame,
                 save_path: Optional[str] = None) -> plt.Figure:
    """Love plot showing covariate balance before/after matching."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4CAF50" if b else "#F44336" for b in balance_df["balanced"]]
    ax.barh(balance_df["covariate"], balance_df["std_mean_diff"].abs(), color=colors)
    ax.axvline(x=0.1, color="red", linestyle="--", label="Threshold (0.1)")
    ax.set_xlabel("Absolute Standardized Mean Difference")
    ax.set_title("Covariate Balance After Matching")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_parallel_trends(panel: pd.DataFrame,
                         treatment_col: str,
                         outcome_col: str,
                         time_col: str,
                         intervention_time: int,
                         save_path: Optional[str] = None) -> plt.Figure:
    """Plot parallel trends for DiD assumption validation."""
    trends = panel.groupby([time_col, treatment_col])[outcome_col].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    for label, group in trends.groupby(treatment_col):
        name = "Treated" if label == 1 else "Control"
        color = "#2196F3" if label == 1 else "#FF9800"
        ax.plot(group[time_col], group[outcome_col], label=name, color=color, linewidth=2)

    ax.axvline(x=intervention_time, color="red", linestyle="--", alpha=0.7, label="Intervention")
    ax.set_xlabel("Week")
    ax.set_ylabel(f"Mean {outcome_col.replace('_', ' ').title()}")
    ax.set_title("Parallel Trends: Treatment vs Control")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_attribution_waterfall(attribution_dict: dict,
                                save_path: Optional[str] = None) -> plt.Figure:
    """Waterfall chart showing revenue attribution decomposition.

    Shows how much of the treatment-control revenue gap is explained
    by each component: promotion, seasonality, trend, and residual.
    """
    components = ["Promotion", "Seasonality", "Trend", "Residual"]
    values = [
        attribution_dict["promotion_effect"],
        attribution_dict["seasonality_effect"],
        attribution_dict["trend_effect"],
        attribution_dict.get("residual", 0),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    cumulative = 0
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9E9E9E"]
    for i, (comp, val) in enumerate(zip(components, values)):
        ax.bar(comp, val, bottom=cumulative, color=colors[i], edgecolor="white", linewidth=2)
        cumulative += val

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Effect on Log Revenue")
    ax.set_title("Revenue Attribution Decomposition")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
