# Makefile for maya4 development

.PHONY: help install install-dev test test-fast test-verbose lint format clean build publish

help:
	@echo "Available commands:"
	@echo "  install       - Install package in production mode"
	@echo "  install-dev   - Install package with development dependencies"
	@echo "  test          - Run tests with coverage"
	@echo "  test-fast     - Run tests without coverage"
	@echo "  test-verbose  - Run tests with verbose output"
	@echo "  lint          - Run linting checks (flake8, black, isort)"
	@echo "  format        - Auto-format code with black and isort"
	@echo "  clean         - Remove build artifacts and cache files"
	@echo "  build         - Build distribution packages"
	@echo "  publish       - Publish to PyPI (requires credentials)"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest

test-fast:
	pytest --no-cov

test-verbose:
	pytest -vv

test-watch:
	pytest-watch

lint:
	@echo "Running flake8..."
	flake8 maya4 tests
	@echo "Checking black formatting..."
	black --check maya4 tests
	@echo "Checking isort..."
	isort --check-only maya4 tests

format:
	@echo "Formatting with black..."
	black maya4 tests
	@echo "Sorting imports with isort..."
	isort maya4 tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete

build: clean
	python -m build

publish: build
	twine check dist/*
	twine upload dist/*

# Development shortcuts
dev-setup: install-dev
	pre-commit install

test-unit:
	pytest -m unit

test-integration:
	pytest -m integration

test-cov:
	pytest --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"
