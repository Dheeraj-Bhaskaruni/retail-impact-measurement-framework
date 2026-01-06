"""
Export utilities for stakeholder reporting.

Formats pipeline results into tables and summaries suitable
for Confluence pages, Slack messages, and email reports.
"""
import pandas as pd
from typing import Dict, Any
from datetime import datetime


def results_to_markdown(results: Dict[str, Any], campaign_id: str = "") -> str:
    """Format pipeline results as a Markdown summary for Confluence/Slack."""
    lines = [
        f"## Measurement Results — {campaign_id}",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "### Causal Estimates",
        "",
        "| Method | Estimate | p-value | Significant |",
        "|--------|----------|---------|-------------|",
    ]

    if "psm" in results:
        p = results["psm"]
        sig = "Yes" if p["p_value"] < 0.05 else "No"
        lines.append(f"| PSM | {p['att']:,.2f} | {p['p_value']:.4f} | {sig} |")

    if "did" in results:
        d = results["did"]
        sig = "Yes" if d["p_value"] < 0.05 else "No"
        lines.append(f"| DiD | {d['att']:,.2f} | {d['p_value']:.4f} | {sig} |")

    if "attribution" in results:
        a = results["attribution"]
        lines.extend([
            "",
            "### Attribution",
            f"- Promotion share: **{a['promotion_share']:.1f}%**",
            f"- Promotion effect: {a['promotion_effect']:.4f} (log-revenue)",
        ])

    if "health" in results:
        h = results["health"]
        lines.extend([
            "",
            f"### Pipeline Health: **{h['status'].upper()}**",
            f"- Checks passed: {h['passed']}",
            f"- Warnings: {h['warnings']}",
            f"- Failures: {h['failures']}",
        ])

    return "\n".join(lines)


def results_to_csv(results: Dict[str, Any], output_path: str):
    """Export KPI report to CSV for spreadsheet consumers."""
    if "kpi_report" in results:
        df = pd.DataFrame(results["kpi_report"])
        df.to_csv(output_path, index=False)
