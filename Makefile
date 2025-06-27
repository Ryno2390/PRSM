# PRSM Development Makefile
# Provides common development tasks

.PHONY: help install install-dev test test-cov lint format clean run docker docker-optimized docs nwtn-test nwtn-stress load-test

# Default target
help:
	@echo "PRSM Development Commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage reporting"
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
	@echo "  ftns-test    Test FTNS microsecond precision and accuracy"
	@echo "  benchmark-quick  Run performance benchmarks (PRSM only)"
	@echo "  benchmark-full   Run comprehensive benchmarks (vs GPT-4/Claude)"
	@echo "  benchmark-load   Run concurrent load test (1000 users)"
	@echo "  validate-compliance  Validate Phase 1 compliance requirements"
	@echo "  test-circuit-breakers  Run circuit breaker failure tests"
	@echo "  validate-resilience  Validate Phase 1 resilience requirements"
	@echo "  bootstrap-network  Deploy 10-node bootstrap test network"
	@echo "  load-test    Run comprehensive load testing suite"
	@echo "  validate-phase1  Complete Phase 1 validation suite"
	@echo "  safeguard-test   Test Recursive Self-Improvement Safeguards"
	@echo "  validate-phase3  Complete Phase 3 validation suite"

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

# FTNS Accounting Ledger Testing
ftns-test:
	@echo "💰 Running FTNS precision and accuracy tests..."
	python scripts/test-ftns-precision.py

ftns-validate:
	@echo "🔍 Validating FTNS microsecond precision..."
	python scripts/test-ftns-precision.py --quick

# Performance Benchmark Testing
test-benchmarks:
	@echo "🎯 Running performance benchmark tests..."
	python scripts/test-performance-benchmarks.py

benchmark-quick:
	@echo "🚀 Running quick benchmark (PRSM only)..."
	python scripts/performance-benchmark-suite.py quick

benchmark-full:
	@echo "🚀 Running comprehensive benchmark (PRSM vs GPT-4/Claude)..."
	python scripts/performance-benchmark-suite.py full

benchmark-load:
	@echo "🔥 Running concurrent load test (1000 users)..."
	python scripts/performance-benchmark-suite.py load

validate-compliance:
	@echo "✅ Validating Phase 1 compliance requirements..."
	python scripts/test-performance-benchmarks.py validate

# Circuit Breaker Testing
test-circuit-breakers:
	@echo "🛡️ Running circuit breaker tests..."
	python scripts/test-circuit-breakers.py

test-circuit-breakers-quick:
	@echo "🔧 Running quick circuit breaker test..."
	python scripts/test-circuit-breakers.py quick

test-circuit-breakers-comprehensive:
	@echo "🧪 Running comprehensive circuit breaker test suite..."
	python scripts/test-circuit-breakers.py comprehensive

test-component-circuits:
	@echo "🔧 Testing component-specific circuit breakers..."
	python scripts/test-circuit-breakers.py components

validate-resilience:
	@echo "🛡️ Validating Phase 1 resilience requirements..."
	python scripts/validate-circuit-breaker-resilience.py

validate-resilience-quick:
	@echo "🔧 Running quick resilience check..."
	python scripts/validate-circuit-breaker-resilience.py quick

# Bootstrap Test Network
bootstrap-network:
	@echo "🚀 Deploying 10-node bootstrap test network..."
	python scripts/bootstrap-test-network.py

bootstrap-network-quick:
	@echo "🔧 Running quick bootstrap test..."
	python scripts/bootstrap-test-network.py quick

# Development workflow with performance validation
dev-test: install-dev test lint nwtn-test ftns-test test-benchmarks test-circuit-breakers
	@echo "🔧 Development testing complete!"

# Phase 2 Economic Model Testing
economic-simulation:
	@echo "🏦 Running economic simulation with agent-based model..."
	python prsm/economics/agent_based_model.py

economic-simulation-comprehensive:
	@echo "📊 Running comprehensive economic validation..."
	python prsm/economics/agent_based_model.py comprehensive

economic-simulation-quick:
	@echo "🔧 Running quick economic simulation test..."
	python3 -c "import asyncio; from prsm.economics.agent_based_model import run_economic_simulation; print('Success:', asyncio.run(run_economic_simulation(steps=24, num_agents=100)))"

jupyter-dashboard:
	@echo "📊 Starting PRSM Economic Dashboard..."
	cd notebooks && jupyter lab economic_dashboard.ipynb

# Distributed Safety Testing
safety-red-team:
	@echo "🔴 Running distributed safety red team exercise..."
	python3 scripts/distributed_safety_red_team.py

safety-red-team-quick:
	@echo "🔧 Running quick safety test..."
	python3 scripts/distributed_safety_red_team.py quick

# Quality Assurance System
quality-assurance:
	@echo "🔍 Running automated model validation pipeline..."
	python3 prsm/quality/automated_validation_pipeline.py

quality-assurance-quick:
	@echo "🔧 Running quick quality assurance test..."
	python3 prsm/quality/automated_validation_pipeline.py quick

# Complete Phase 1 validation
validate-phase1: nwtn-test ftns-test test-benchmarks test-circuit-breakers bootstrap-network-quick load-test-phase1
	@echo "✅ Complete Phase 1 validation finished!"
	python scripts/test-performance-benchmarks.py validate
	python scripts/validate-circuit-breaker-resilience.py

# Phase 2 validation
validate-phase2: economic-simulation-comprehensive safety-red-team-quick quality-assurance-quick
	@echo "✅ Phase 2 validation finished!"
	@echo "  📊 Economic simulation with agent-based modeling: COMPLETED"
	@echo "  🔴 Distributed safety red team exercise: COMPLETED"
	@echo "  🔍 Quality assurance pipeline: COMPLETED"

# Phase 3 Testing
p2p-network-test:
	@echo "🌍 Testing Multi-Region P2P Network..."
	python3 prsm/federation/multi_region_p2p_network.py quick

marketplace-test:
	@echo "🏪 Testing Model Marketplace MVP..."
	python3 prsm/marketplace/model_marketplace.py quick

marketplace-test-full:
	@echo "🏪 Running full Model Marketplace deployment..."
	python3 prsm/marketplace/model_marketplace.py

onboarding-test:
	@echo "👥 Testing Contributor Onboarding System..."
	python3 prsm/onboarding/contributor_onboarding.py quick

onboarding-test-full:
	@echo "👥 Running full Contributor Onboarding deployment..."
	python3 prsm/onboarding/contributor_onboarding.py

data-spine-test:
	@echo "🌐 Testing PRSM Data Spine Proxy..."
	python3 prsm/spine/data_spine_proxy.py quick

data-spine-test-full:
	@echo "🌐 Running full Data Spine Proxy deployment..."
	python3 prsm/spine/data_spine_proxy.py

safeguard-test:
	@echo "🛡️ Testing Recursive Self-Improvement Safeguards..."
	python3 prsm/safety/recursive_improvement_safeguards.py quick

safeguard-test-full:
	@echo "🛡️ Running full Recursive Self-Improvement Safeguards deployment..."
	python3 prsm/safety/recursive_improvement_safeguards.py

# Phase 3 validation
validate-phase3: p2p-network-test marketplace-test onboarding-test data-spine-test safeguard-test
	@echo "✅ Phase 3 validation finished!"
	@echo "  🌍 Multi-Region P2P Network: COMPLETED"
	@echo "  🏪 Model Marketplace MVP: COMPLETED"
	@echo "  👥 Contributor Onboarding System: COMPLETED"
	@echo "  🌐 PRSM Data Spine Proxy: COMPLETED"
	@echo "  🛡️ Recursive Self-Improvement Safeguards: COMPLETED"