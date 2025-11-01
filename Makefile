# Makefile for House Price Prediction System
# Provides convenient commands for development workflow

.PHONY: help install test lint format clean setup dev docs run predict

# Default target
help:
	@echo "🏠 House Price Prediction System - Development Commands"
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup          Set up development environment"
	@echo "  install        Install dependencies"
	@echo "  dev            Install development dependencies"
	@echo ""
	@echo "Development Commands:"
	@echo "  test           Run all tests"
	@echo "  test-cov       Run tests with coverage report"
	@echo "  lint           Run linting checks"
	@echo "  format         Format code with black and isort"
	@echo "  type-check     Run type checking with mypy"
	@echo ""
	@echo "Pipeline Commands:"
	@echo "  run            Run the ML training pipeline"
	@echo "  predict        Run sample prediction"
	@echo "  mlflow         Start MLflow UI"
	@echo ""
	@echo "Cleanup Commands:"
	@echo "  clean          Clean up generated files"
	@echo "  clean-cache    Clean Python cache files"
	@echo "  clean-logs     Clean log files"

# Setup and Installation
setup:
	@echo "🚀 Setting up development environment..."
	./setup_dev.sh

install:
	@echo "📦 Installing dependencies..."
	pip install -r prices-predictor-system/requirements.txt

dev:
	@echo "🛠️ Installing development dependencies..."
	pip install -e .[dev]
	pip install pre-commit
	pre-commit install

# Testing
test:
	@echo "🧪 Running tests..."
	cd prices-predictor-system && python -m pytest tests/ -v

test-cov:
	@echo "🧪 Running tests with coverage..."
	cd prices-predictor-system && python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-specific:
	@echo "🧪 Running specific test..."
	@if [ -z "$(TEST)" ]; then echo "Usage: make test-specific TEST=test_file.py"; exit 1; fi
	cd prices-predictor-system && python -m pytest tests/$(TEST) -v

# Code Quality
lint:
	@echo "🔍 Running linting checks..."
	cd prices-predictor-system && flake8 src/ steps/ pipelines/ tests/
	@echo "✅ Linting complete"

format:
	@echo "🎨 Formatting code..."
	cd prices-predictor-system && black src/ steps/ pipelines/ tests/
	cd prices-predictor-system && isort src/ steps/ pipelines/ tests/
	@echo "✅ Code formatting complete"

type-check:
	@echo "🔍 Running type checks..."
	cd prices-predictor-system && mypy src/ --ignore-missing-imports
	@echo "✅ Type checking complete"

format-check:
	@echo "🔍 Checking code formatting..."
	cd prices-predictor-system && black --check src/ steps/ pipelines/ tests/
	cd prices-predictor-system && isort --check-only src/ steps/ pipelines/ tests/

# Combined quality check
quality: lint type-check format-check
	@echo "✅ All quality checks passed"

# Pipeline Commands
run:
	@echo "🚀 Running ML training pipeline..."
	cd prices-predictor-system && python run_pipeline.py

predict:
	@echo "🔮 Running sample prediction..."
	cd prices-predictor-system && python sample_predict.py

mlflow:
	@echo "📊 Starting MLflow UI..."
	cd prices-predictor-system && mlflow ui

# Documentation
docs:
	@echo "📚 Generating documentation..."
	@echo "Documentation generation not yet implemented"

# Cleanup
clean: clean-cache clean-logs clean-results
	@echo "🧹 Cleanup complete"

clean-cache:
	@echo "🧹 Cleaning Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	find . -name "*~" -delete 2>/dev/null || true

clean-logs:
	@echo "🧹 Cleaning log files..."
	rm -rf logs/*.log 2>/dev/null || true
	rm -rf prices-predictor-system/logs/*.log 2>/dev/null || true

clean-results:
	@echo "🧹 Cleaning result files..."
	rm -rf results/*.png 2>/dev/null || true
	rm -rf results/*.html 2>/dev/null || true
	rm -rf prices-predictor-system/results/*.png 2>/dev/null || true

clean-mlruns:
	@echo "🧹 Cleaning MLflow runs..."
	rm -rf mlruns/ 2>/dev/null || true
	rm -rf prices-predictor-system/mlruns/ 2>/dev/null || true

# Environment management
env-create:
	@echo "🌐 Creating virtual environment..."
	python3 -m venv venv

env-activate:
	@echo "🌐 To activate environment, run: source venv/bin/activate"

env-deactivate:
	@echo "🌐 To deactivate environment, run: deactivate"

# Git hooks
hooks-install:
	@echo "🪝 Installing git hooks..."
	pre-commit install

hooks-update:
	@echo "🪝 Updating git hooks..."
	pre-commit autoupdate

# Continuous Integration simulation
ci: clean format-check lint type-check test
	@echo "✅ CI pipeline completed successfully"

# Development workflow
dev-workflow: format lint type-check test
	@echo "✅ Development workflow completed"

# Security checks
security:
	@echo "🔐 Running security checks..."
	pip-audit || echo "pip-audit not installed, run: pip install pip-audit"

# Dependency management
deps-update:
	@echo "📦 Updating dependencies..."
	pip list --outdated
	@echo "To update, run: pip install --upgrade <package_name>"

deps-check:
	@echo "📦 Checking dependencies..."
	pip check

# Performance profiling
profile:
	@echo "⏱️ Profiling pipeline performance..."
	cd prices-predictor-system && python -m cProfile -o profile_output.prof run_pipeline.py
	@echo "Profile saved to profile_output.prof"

# Docker commands (if Docker support is added)
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t prices-predictor .

docker-run:
	@echo "🐳 Running Docker container..."
	docker run -p 5000:5000 prices-predictor

# Notebook management
notebook:
	@echo "📓 Starting Jupyter notebook..."
	cd prices-predictor-system && jupyter notebook

# Database/data management
data-check:
	@echo "📊 Checking data integrity..."
	cd prices-predictor-system && python -c "import pandas as pd; df = pd.read_csv('extracted_data/AmesHousing.csv'); print(f'Data shape: {df.shape}'); print(f'Missing values: {df.isnull().sum().sum()}')"

# Version management
version:
	@echo "📦 Current version information:"
	@python --version
	@pip --version
	@echo "Project version: $(shell git describe --tags --always --dirty 2>/dev/null || echo 'unknown')"

# Quick start
quick-start: setup install test run
	@echo "🎉 Quick start completed! Your ML pipeline is ready."

# Full setup for new developers
onboard: setup dev hooks-install test docs
	@echo "🎉 Onboarding completed! Welcome to the team."