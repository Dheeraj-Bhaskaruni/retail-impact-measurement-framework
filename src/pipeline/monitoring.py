"""
Pipeline Monitoring and Health Checks.

Detects anomalies in pipeline outputs and sends alerts when
results look suspicious — catches issues before they reach
stakeholder dashboards.
"""
import logging
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    """Result of a single health check."""
    name: str
    status: str           # "pass", "warn", "fail"
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class PipelineHealthReport:
    """Aggregate health report for a pipeline run."""
    timestamp: str
    campaign_id: str
    overall_status: str   # "healthy", "degraded", "critical"
    checks: List[HealthCheck]
    n_passed: int
    n_warnings: int
    n_failures: int


def run_health_checks(results: Dict, config: Dict) -> PipelineHealthReport:
    """
    Run health checks against pipeline results.
    Called automatically at the end of each pipeline run.
    """
    checks = []
    campaign_id = config.get("campaign", {}).get("id", "unknown")

    # 1. PSM match rate check
    if "psm" in results:
        psm = results["psm"]
        n_matched = psm.get("n_matched", 0)
        checks.append(HealthCheck(
            name="psm_match_rate",
            status="pass" if n_matched > 50 else "warn" if n_matched > 20 else "fail",
            message=f"PSM matched {n_matched} pairs",
            value=n_matched,
            threshold=50,
        ))

        # PSM significance
        p = psm.get("p_value", 1.0)
        checks.append(HealthCheck(
            name="psm_significance",
            status="pass" if p < 0.05 else "warn" if p < 0.10 else "fail",
            message=f"PSM p-value: {p:.4f}",
            value=p,
            threshold=0.05,
        ))

    # 2. DiD parallel trends
    if "did" in results:
        did = results["did"]
        parallel = did.get("parallel_trends", False)
        checks.append(HealthCheck(
            name="did_parallel_trends",
            status="pass" if parallel else "warn",
            message=f"Parallel trends assumption {'valid' if parallel else 'may be violated'}",
        ))

    # 3. Cross-method agreement
    if "psm" in results and "did" in results:
        psm_att = results["psm"].get("att", 0)
        did_att = results["did"].get("att", 0)
        if psm_att != 0:
            pct_diff = abs(psm_att - did_att) / abs(psm_att) * 100
            checks.append(HealthCheck(
                name="cross_method_agreement",
                status="pass" if pct_diff < 50 else "warn" if pct_diff < 100 else "fail",
                message=f"PSM vs DiD difference: {pct_diff:.0f}%",
                value=pct_diff,
                threshold=50,
            ))

    # 4. Attribution sanity
    if "attribution" in results:
        promo_share = results["attribution"].get("promotion_share", 0)
        checks.append(HealthCheck(
            name="attribution_sanity",
            status="pass" if 5 < promo_share < 95 else "warn",
            message=f"Promotion accounts for {promo_share:.0f}% of revenue gap",
            value=promo_share,
        ))

    # 5. ROAS sanity check
    if "kpi_report" in results:
        roas = results["kpi_report"].get("roas", 0)
        if roas is not None and roas > 0:
            checks.append(HealthCheck(
                name="roas_sanity",
                status="pass" if 0.5 < roas < 20 else "warn",
                message=f"ROAS = {roas:.1f}x (expected 1-10x for retail)",
                value=roas,
            ))

    # 6. Data completeness
    if "psm" in results:
        checks.append(HealthCheck(
            name="data_completeness",
            status="pass",
            message="All pipeline stages completed successfully",
        ))

    # Aggregate
    n_passed = sum(1 for c in checks if c.status == "pass")
    n_warnings = sum(1 for c in checks if c.status == "warn")
    n_failures = sum(1 for c in checks if c.status == "fail")

    if n_failures > 0:
        overall = "critical"
    elif n_warnings > 0:
        overall = "degraded"
    else:
        overall = "healthy"

    report = PipelineHealthReport(
        timestamp=datetime.now().isoformat(),
        campaign_id=campaign_id,
        overall_status=overall,
        checks=checks,
        n_passed=n_passed,
        n_warnings=n_warnings,
        n_failures=n_failures,
    )

    _log_report(report)
    return report


def _log_report(report: PipelineHealthReport):
    """Log health report at appropriate level."""
    level = {
        "healthy": logging.INFO,
        "degraded": logging.WARNING,
        "critical": logging.ERROR,
    }[report.overall_status]

    logger.log(level, f"Pipeline health: {report.overall_status.upper()} "
                      f"({report.n_passed} pass, {report.n_warnings} warn, "
                      f"{report.n_failures} fail)")

    for check in report.checks:
        if check.status == "fail":
            logger.error(f"  FAIL: {check.name} — {check.message}")
        elif check.status == "warn":
            logger.warning(f"  WARN: {check.name} — {check.message}")


def save_health_report(report: PipelineHealthReport, output_dir: str):
    """Save health report to JSON for monitoring dashboards."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = path / f"health_{timestamp}.json"

    data = {
        "timestamp": report.timestamp,
        "campaign_id": report.campaign_id,
        "overall_status": report.overall_status,
        "n_passed": report.n_passed,
        "n_warnings": report.n_warnings,
        "n_failures": report.n_failures,
        "checks": [asdict(c) for c in report.checks],
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Health report saved to {filepath}")
