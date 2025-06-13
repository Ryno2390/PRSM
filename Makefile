# PRSM Development Makefile
# Provides common development tasks

.PHONY: help install install-dev test lint format clean run docker docker-optimized docs nwtn-test nwtn-stress load-test

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
	@echo "  docker-optimized  Build optimized Docker images"
	@echo "  docker-performance  Run with performance optimizations"
	@echo "  docs         Build documentation"
	@echo "  k8s-deploy   Deploy to Kubernetes"
	@echo "  k8s-test-autoscaling  Test autoscaling under load"
	@echo "  obs-stack-up  Start observability stack"
	@echo "  obs-dashboards  Show monitoring dashboard URLs"
	@echo "  nwtn-test    Validate NWTN 5-agent pipeline"
	@echo "  nwtn-stress  Stress test NWTN orchestrator (1000 users)"
	@echo "  load-test    Run comprehensive load testing suite"

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

docker-build-optimized:
	./scripts/optimize-docker-build.sh production

docker-build-dev:
	./scripts/optimize-docker-build.sh development

docker-build-all:
	./scripts/optimize-docker-build.sh all

docker-run:
	docker-compose up -d

docker-run-dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

docker-run-performance:
	docker-compose -f docker-compose.yml -f docker-compose.performance.yml up -d

docker-down:
	docker-compose down

docker-clean:
	docker system prune -f
	docker volume prune -f

docker-logs:
	docker-compose logs -f prsm-api

docker-shell:
	docker-compose exec prsm-api bash

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

# Kubernetes autoscaling
k8s-deploy:
	kubectl apply -k deploy/kubernetes/base

k8s-deploy-production:
	kubectl apply -k deploy/kubernetes/overlays/production

k8s-test-autoscaling:
	./scripts/test-autoscaling.sh --duration 300 --concurrent 1000 --rps 100

k8s-scale-test:
	./scripts/test-autoscaling.sh --duration 600 --concurrent 2000 --rps 200

k8s-status:
	kubectl get pods,hpa,vpa -n prsm-system

k8s-logs:
	kubectl logs -f deployment/prsm-api -n prsm-system

# Observability stack
obs-stack-up:
	docker-compose -f docker-compose.yml -f docker-compose.observability.yml up -d

obs-stack-down:
	docker-compose -f docker-compose.yml -f docker-compose.observability.yml down

obs-metrics:
	curl -s http://localhost:9091/metrics | head -20

obs-logs:
	docker-compose -f docker-compose.yml -f docker-compose.observability.yml logs -f grafana-enhanced

obs-dashboards:
	@echo "Grafana: http://localhost:3000 (admin/prsm_admin)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Jaeger: http://localhost:16686"
	@echo "Kibana: http://localhost:5601"

# NWTN Orchestrator Testing (Phase 1 requirements)
nwtn-test:
	@echo "🤖 Running NWTN Agent Pipeline Validation..."
	python scripts/validate-nwtn-agents.py

nwtn-stress:
	@echo "🚀 Running NWTN Orchestrator Stress Test (1000 concurrent users)..."
	python scripts/nwtn-stress-test.py --users 1000 --duration 300 --latency 2000

nwtn-stress-quick:
	@echo "🚀 Running Quick NWTN Stress Test (100 concurrent users)..."
	python scripts/nwtn-stress-test.py --quick

nwtn-stress-extended:
	@echo "🚀 Running Extended NWTN Stress Test (2000 concurrent users)..."
	python scripts/nwtn-stress-test.py --users 2000 --duration 600 --latency 2000

# Load Testing Suite
load-test:
	@echo "📈 Running Comprehensive Load Test Suite..."
	./scripts/load-test-suite.sh --users 1000 --duration 300s

load-test-quick:
	@echo "📈 Running Quick Load Test..."
	./scripts/load-test-suite.sh --quick

load-test-phase1:
	@echo "🎯 Running Phase 1 Load Test Validation..."
	./scripts/validate-phase1-requirements.sh --url http://localhost:8000

# Performance Testing Pipeline
test-phase1: nwtn-test load-test-phase1
	@echo "✅ Phase 1 validation complete!"

test-performance: nwtn-stress load-test
	@echo "📊 Performance testing complete!"

# Development workflow with performance validation
dev-test: install-dev test lint nwtn-test
	@echo "🔧 Development testing complete!"