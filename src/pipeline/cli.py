"""
Command-Line Interface for the Measurement Pipeline.

Usage:
    measure run                    # Full pipeline
    measure run --config path.yaml # Custom config
    measure generate-data          # Generate dev data
    measure validate               # Validate data quality
    measure report                 # Generate KPI report only
"""
import sys
import json
import logging
from pathlib import Path

import click

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("measure")


@click.group()
@click.version_option(version="1.2.0")
def cli():
    """Retail Promotion Impact Measurement Framework."""
    pass


@cli.command()
@click.option("--config", "-c", default=None, help="Path to config YAML")
@click.option("--output", "-o", default=None, help="Output directory for results")
def run(config, output):
    """Run the full measurement pipeline."""
    from pipeline.measurement_pipeline import MeasurementPipeline

    logger.info("Starting measurement pipeline")
    pipeline = MeasurementPipeline(config_path=config)

    if output:
        pipeline.config["pipeline"]["output_dir"] = output

    results = pipeline.run()

    click.echo("\n=== Pipeline Results ===")
    click.echo(f"PSM ATT:  {results['psm']['att']:.4f} (p={results['psm']['p_value']:.4f})")
    click.echo(f"DiD ATT:  {results['did']['att']:.4f} (p={results['did']['p_value']:.4f})")
    click.echo(f"Attribution: {results['attribution']['promotion_share']:.1f}% promotion effect")


@cli.command("generate-data")
@click.option("--n-stores", default=500, help="Number of stores")
@click.option("--seed", default=42, help="Random seed")
def generate_data(n_stores, seed):
    """Generate synthetic development data."""
    # Navigate to project root
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root / "data" / "synthetic"))

    from generate_data import (
        generate_store_features, assign_treatment,
        generate_instrument_data, generate_weekly_outcomes,
    )

    click.echo(f"Generating data for {n_stores} stores (seed={seed})...")

    stores = generate_store_features(n_stores=n_stores, seed=seed)
    stores = assign_treatment(stores, treatment_fraction=0.4, seed=seed)
    stores = generate_instrument_data(stores, seed=seed)
    weekly = generate_weekly_outcomes(stores, n_weeks=25, seed=seed)
    panel = weekly.merge(stores, on="store_id")

    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    stores.to_csv(output_dir / "stores.csv", index=False)
    weekly.to_csv(output_dir / "weekly_outcomes.csv", index=False)
    panel.to_csv(output_dir / "panel_data.csv", index=False)

    click.echo(f"Generated: {len(stores)} stores, {len(panel):,} panel rows")
    click.echo(f"Saved to {output_dir}/")


@cli.command()
@click.option("--data-dir", default=None, help="Directory containing CSV files")
def validate(data_dir):
    """Validate data quality against schemas."""
    import pandas as pd
    from data.validation import validate_panel_data, validate_store_data

    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    else:
        data_dir = Path(data_dir)

    click.echo("=== Data Validation ===\n")

    # Validate stores
    stores_path = data_dir / "stores.csv"
    if stores_path.exists():
        stores = pd.read_csv(stores_path)
        result = validate_store_data(stores)
        status = "PASS" if result["passed"] else "FAIL"
        click.echo(f"Stores:  [{status}] {result['valid']}/{result['total_sampled']} valid")
        if result.get("missing_columns"):
            click.echo(f"  Missing columns: {result['missing_columns']}")
    else:
        click.echo(f"Stores: file not found at {stores_path}")

    # Validate panel
    panel_path = data_dir / "panel_data.csv"
    if panel_path.exists():
        panel = pd.read_csv(panel_path)
        result = validate_panel_data(panel)
        status = "PASS" if result["passed"] else "FAIL"
        click.echo(f"Panel:   [{status}] {result['valid']}/{result['total_sampled']} valid")
        if result["errors"]:
            for err in result["errors"][:3]:
                click.echo(f"  Row {err['row_index']}: {err['error'][:80]}")
    else:
        click.echo(f"Panel: file not found at {panel_path}")


@cli.command()
def report():
    """Generate KPI report from existing pipeline results."""
    from data.data_loader import load_panel_data
    from metrics.kpi_framework import KPICalculator

    panel = load_panel_data()
    calc = KPICalculator(panel)

    naive = calc.compute_naive_lift("revenue")
    click.echo("=== Quick KPI Report ===")
    click.echo(f"Treated mean:   ${naive['treated_mean']:,.2f}")
    click.echo(f"Control mean:   ${naive['control_mean']:,.2f}")
    click.echo(f"Naive lift:     {naive['naive_lift_pct']:.1f}% (BIASED)")
    click.echo("\nRun 'measure run' for unbiased causal estimates.")


if __name__ == "__main__":
    cli()
