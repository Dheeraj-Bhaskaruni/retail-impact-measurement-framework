.PHONY: setup data pipeline test lint clean

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

data:
	. venv/bin/activate && python data/synthetic/generate_data.py

pipeline:
	. venv/bin/activate && cd src && python -m pipeline.measurement_pipeline

test:
	. venv/bin/activate && python -m pytest tests/ -v --tb=short

lint:
	. venv/bin/activate && flake8 src/ tests/ --max-line-length 120 --ignore E501,W503

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf data/processed/*.json
