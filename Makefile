.PHONY: setup data pipeline test test-unit test-integration lint format validate coverage docker clean

# ============================================================
# Setup
# ============================================================

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	. venv/bin/activate && pre-commit install || true
	@echo "Setup complete. Run: source venv/bin/activate"

# ============================================================
# Data
# ============================================================

data:
	. venv/bin/activate && python data/synthetic/generate_data.py

validate:
	. venv/bin/activate && cd src && python -m pipeline.cli validate

# ============================================================
# Pipeline
# ============================================================

pipeline:
	. venv/bin/activate && cd src && python -m pipeline.measurement_pipeline

cli-run:
	. venv/bin/activate && cd src && python -m pipeline.cli run

cli-report:
	. venv/bin/activate && cd src && python -m pipeline.cli report

# ============================================================
# Testing
# ============================================================

test:
	. venv/bin/activate && python -m pytest tests/ -v --tb=short

test-unit:
	. venv/bin/activate && python -m pytest tests/ -v --tb=short -m "not integration"

test-integration:
	. venv/bin/activate && python -m pytest tests/ -v --tb=short -m "integration"

coverage:
	. venv/bin/activate && python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

# ============================================================
# Code Quality
# ============================================================

lint:
	. venv/bin/activate && flake8 src/ tests/ --max-line-length 120 --ignore E501,W503

format:
	. venv/bin/activate && black src/ tests/ --line-length 100
	. venv/bin/activate && isort src/ tests/ --profile black --line-length 100

# ============================================================
# Docker
# ============================================================

docker-build:
	docker build -t measurement-pipeline .

docker-run:
	docker run --rm -v $(PWD)/data:/app/data measurement-pipeline

docker-test:
	docker-compose run --rm test

# ============================================================
# Cleanup
# ============================================================

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
	rm -rf data/processed/*.json
