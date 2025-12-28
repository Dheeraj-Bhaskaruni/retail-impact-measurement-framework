# Contributing

## Development Setup

```bash
git clone https://github.com/Dheeraj-Bhaskaruni/retail-impact-measurement-framework.git
cd retail-impact-measurement-framework
make setup
source venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
# All tests
make test

# Unit tests only (fast)
pytest tests/ -m "not integration" -v

# Integration tests only
pytest tests/ -m integration -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Code Standards

- **Formatting**: black (line length 100)
- **Imports**: isort (black profile)
- **Linting**: flake8 (max line 120)
- **Type hints**: Preferred on public function signatures
- **Tests**: Every new module needs corresponding tests in `tests/`

## Adding a New Causal Method

1. Create `src/causal/your_method.py` with a dataclass for results
2. Add tests in `tests/test_your_method.py`
3. Wire it into `src/pipeline/measurement_pipeline.py`
4. Add a notebook in `notebooks/` demonstrating the method
5. Document assumptions in `docs/methodology.md`

## Branch Strategy

- `main`: production-ready code
- `feature/*`: new features
- `fix/*`: bug fixes
- `experiment/*`: exploratory analysis (may not merge)

## Pull Request Checklist

- [ ] All tests pass (`make test`)
- [ ] New code has tests
- [ ] Docstrings on public functions
- [ ] Updated CHANGELOG.md
- [ ] No secrets or data files committed
