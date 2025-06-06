# PRSM Development Makefile
# Provides common development tasks

.PHONY: help install install-dev test lint format clean run docker docs

# Default target
help:
	@echo "PRSM Development Commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  clean        Clean up generated files"
	@echo "  run          Run the PRSM development server"
	@echo "  docker       Build and run with Docker"
	@echo "  docs         Build documentation"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

# Testing
test:
	pytest

test-cov:
	pytest --cov=prsm --cov-report=html --cov-report=term

test-integration:
	pytest -m integration

test-unit:
	pytest -m unit

# Code Quality
lint:
	flake8 prsm tests
	mypy prsm
	black --check prsm tests
	isort --check-only prsm tests

format:
	black prsm tests
	isort prsm tests

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Development Server
run:
	uvicorn prsm.api.main:app --reload --host 0.0.0.0 --port 8000

run-worker:
	python -m prsm.workers.main

# Database
db-upgrade:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-revision:
	alembic revision --autogenerate -m "$(msg)"

# Docker
docker-build:
	docker build -t prsm:latest .

docker-run:
	docker-compose up -d

docker-down:
	docker-compose down

# Documentation
docs:
	mkdocs build

docs-serve:
	mkdocs serve

# IPFS
ipfs-start:
	ipfs daemon

ipfs-init:
	ipfs init

# Monitoring
metrics:
	python -m prsm.monitoring.metrics_server

# Development Setup
setup-dev: install-dev
	cp .env.example .env
	@echo "Please edit .env with your configuration"
	@echo "Then run: make db-upgrade"

# CI/CD helpers
ci-test: lint test-cov

# Network setup for P2P
setup-p2p:
	python -m prsm.federation.setup_network