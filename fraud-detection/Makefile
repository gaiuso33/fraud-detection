.PHONY: install lint format test cov security data-check clean

install:
	pip install -e ".[api,dev]"

lint:
	ruff check src tests

format:
	ruff format src tests

test:
	pytest

cov:
	pytest --cov=fraud --cov-report=term-missing

security:
	bandit -r src
	pip-audit

data-check:
	python -m fraud.data

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ build *.egg-info
