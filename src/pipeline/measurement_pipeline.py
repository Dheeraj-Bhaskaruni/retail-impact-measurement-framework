"""
End-to-end Measurement Pipeline.

Orchestrates data loading, causal estimation, KPI calculation,
and report generation. Designed for reproducible, automated runs.
"""
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from data.data_loader import load_panel_data, load_store_data, load_config
from data.feature_engineering import create_pre_treatment_features
from causal.propensity_score import run_psm
from causal.diff_in_diff import estimate_did
from metrics.kpi_framework import KPICalculator
from metrics.attribution import decompose_revenue

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MeasurementPipeline:
    """Orchestrates the full measurement workflow."""

    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self.results: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """Execute the full pipeline."""
        logger.info("Starting measurement pipeline")

        # Step 1: Load data
        panel = load_panel_data()
        stores = load_store_data()
        logger.info(f"Loaded panel data: {panel.shape[0]} rows, {stores.shape[0]} stores")

        # Step 2: Feature engineering
        store_features = create_pre_treatment_features(panel)
        analysis_df = stores.merge(store_features, on="store_id")

        # Step 3: Propensity Score Matching
        covariates = self.config["propensity_score"]["covariates"]
        available_covariates = [c for c in covariates if c in analysis_df.columns]
        logger.info(f"Running PSM with covariates: {available_covariates}")

        psm_result = run_psm(
            analysis_df,
            outcome_col="pre_avg_revenue",
            treatment_col="treated",
            covariates=available_covariates,
            caliper=self.config["propensity_score"]["caliper"],
        )
        self.results["psm"] = {
            "att": psm_result.att,
            "se": psm_result.se,
            "p_value": psm_result.p_value,
            "ci": [psm_result.ci_lower, psm_result.ci_upper],
            "n_matched": psm_result.n_matched,
        }
        logger.info(f"PSM ATT: {psm_result.att:.4f} (p={psm_result.p_value:.4f})")

        # Step 4: Difference-in-Differences
        did_result = estimate_did(
            panel, outcome_col="revenue", treatment_col="treated",
            time_col="week", post_period_start=13,
        )
        self.results["did"] = {
            "att": did_result.att,
            "se": did_result.se,
            "p_value": did_result.p_value,
            "parallel_trends": did_result.pre_trend_parallel,
        }
        logger.info(f"DiD ATT: {did_result.att:.4f} (p={did_result.p_value:.4f})")

        # Step 5: Attribution decomposition
        attribution = decompose_revenue(panel)
        self.results["attribution"] = {
            "promotion_effect": attribution.promotion_effect,
            "promotion_share": attribution.promotion_share,
            "seasonality_effect": attribution.seasonality_effect,
            "trend_effect": attribution.trend_effect,
        }

        # Step 6: KPI Report
        kpi_calc = KPICalculator(panel)
        kpi_report = kpi_calc.generate_kpi_report(
            causal_estimates={"att_revenue": psm_result.att}
        )
        self.results["kpi_report"] = kpi_report.to_dict(orient="records")

        # Step 7: Save results
        self._save_results()
        logger.info("Pipeline complete")
        return self.results

    def _save_results(self):
        """Save pipeline results to JSON."""
        output_dir = Path(self.config["pipeline"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"results_{timestamp}.json"

        serializable = json.loads(json.dumps(self.results, default=str))
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Results saved to {output_path}")


def main():
    pipeline = MeasurementPipeline()
    results = pipeline.run()
    print("\n=== Pipeline Results Summary ===")
    print(f"PSM ATT: {results['psm']['att']:.4f} (p={results['psm']['p_value']:.4f})")
    print(f"DiD ATT: {results['did']['att']:.4f} (p={results['did']['p_value']:.4f})")
    print(f"Promotion attribution: {results['attribution']['promotion_share']:.1f}%")


if __name__ == "__main__":
    main()
