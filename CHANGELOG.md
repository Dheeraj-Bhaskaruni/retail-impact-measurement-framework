# Changelog

## [1.2.0] - 2025-12-28
### Added
- CLI interface (`measure run`, `measure validate`, `measure report`)
- Pipeline health monitoring with automated checks
- Pydantic data validation schemas
- Docker and docker-compose support
- Pre-commit hooks configuration

## [1.1.0] - 2025-12-22
### Added
- Rosenbaum bounds sensitivity analysis
- Heterogeneous treatment effects (subgroup analysis + CATE)
- Integration test suite
- conftest.py with shared test fixtures
- Caliper stability analysis for PSM robustness

## [1.0.0] - 2025-12-15
### Added
- Propensity Score Matching (PSM) with balance diagnostics
- Difference-in-Differences (DiD) with parallel trends test
- Instrumental Variables / 2SLS with Sargan test
- A/B test design with power analysis and SPRT
- KPI framework (incremental revenue, ROAS, customer acquisition)
- Revenue attribution decomposition model
- End-to-end measurement pipeline
- SQL extraction queries for Databricks
- Synthetic data generator calibrated to warehouse stats
- 20 unit tests, CI/CD via GitHub Actions
- Statistical methodology documentation

## [0.1.0] - 2025-11-10
### Added
- Initial project scaffolding
- Requirements and configuration
