# PRSM Development Makefile
# Common development tasks. Targets referencing pre-v1.6-pivot code + scripts
# were removed 2026-04-22 as part of pre-audit hygiene (scope-alignment sprint
# deleted those modules/scripts; retained Makefile entries were broken).

.PHONY: help install install-dev test test-unit test-integration test-chaos test-cov test-hardhat test-all \
        lint format clean run \
        db-upgrade db-downgrade db-revision setup-dev ci-test \
        docker-build docker-run docker-down docker-clean docker-logs docker-shell \
        docker-run-dev docker-run-performance docker-run-observability \
        k8s-deploy k8s-deploy-production k8s-status k8s-logs \
        obs-stack-up obs-stack-down obs-metrics obs-logs obs-dashboards \
        docs docs-serve metrics \
        build publish publish-test \
        publish-pypi publish-pypi-dry-run \
        publish-npm publish-npm-dry-run \
        publish-homebrew publish-homebrew-dry-run \
        publish-all-dry-run \
        docker-push bootstrap-build smoke

# Default target
help:
	@echo "PRSM Development Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install              Install production dependencies"
	@echo "  install-dev          Install development dependencies + pre-commit"
	@echo "  setup-dev            Copy .env.example + run install-dev"
	@echo ""
	@echo "Testing:"
	@echo "  test                 Run full pytest suite"
	@echo "  test-unit            Run unit tests only (tests/unit)"
	@echo "  test-integration     Run integration tests only (tests/integration)"
	@echo "  test-chaos           Run chaos tests only (tests/chaos)"
	@echo "  test-cov             Run tests with coverage reporting"
	@echo "  test-hardhat         Run Solidity contract tests (hardhat)"
	@echo "  test-all             Run pytest + hardhat together"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint                 flake8 + mypy + black check + isort check"
	@echo "  format               black + isort"
	@echo "  clean                Remove generated files"
	@echo ""
	@echo "Run:"
	@echo "  run                  Start API dev server with --reload"
	@echo ""
	@echo "Database (alembic):"
	@echo "  db-upgrade           Apply migrations"
	@echo "  db-downgrade         Revert one migration"
	@echo "  db-revision msg=MSG  Autogenerate a new migration"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build         Build main PRSM image"
	@echo "  docker-run           docker-compose up -d"
	@echo "  docker-run-dev       Run with dev overrides (docker/docker-compose.dev.yml)"
	@echo "  docker-run-performance  Run with performance overrides"
	@echo "  docker-run-observability  Run with observability stack overrides"
	@echo "  docker-down          Stop containers"
	@echo "  docker-clean         Prune unused Docker resources"
	@echo "  docker-logs          Follow prsm-api logs"
	@echo "  docker-shell         Shell into prsm-api"
	@echo "  docker-push          Push image to GHCR"
	@echo "  bootstrap-build      Build bootstrap Docker image"
	@echo ""
	@echo "Kubernetes:"
	@echo "  k8s-deploy           Apply base manifests"
	@echo "  k8s-deploy-production Apply production overlay"
	@echo "  k8s-status           kubectl get pods,hpa,vpa -n prsm-system"
	@echo "  k8s-logs             Follow prsm-api deployment logs"
	@echo ""
	@echo "Observability:"
	@echo "  obs-stack-up         Start observability stack"
	@echo "  obs-stack-down       Stop observability stack"
	@echo "  obs-metrics          curl /metrics (first 20 lines)"
	@echo "  obs-logs             Follow Grafana logs"
	@echo "  obs-dashboards       Print dashboard URLs"
	@echo ""
	@echo "Documentation:"
	@echo "  docs                 Build mkdocs site"
	@echo "  docs-serve           Serve docs locally"
	@echo ""
	@echo "Deployment — Multi-channel publishing:"
	@echo "  build                          Build Python wheel + sdist (dist/)"
	@echo "  publish-test                   Publish to TestPyPI (staging)"
	@echo "  publish-pypi                   Publish to PyPI (production, irreversible)"
	@echo "  publish-pypi-dry-run           Validate PyPI artifacts without uploading"
	@echo "  publish-npm                    Publish prsm-mcp wrapper to npm (irreversible)"
	@echo "  publish-npm-dry-run            Validate npm package without publishing"
	@echo "  publish-homebrew               Publish Homebrew formula (Task 10 pending)"
	@echo "  publish-homebrew-dry-run       Validate Homebrew formula"
	@echo "  publish-all-dry-run            Run all three dry-runs in sequence"
	@echo "  publish                        Back-compat alias for publish-pypi"
	@echo "  smoke                Quick smoke test (bootstrap server)"

# ==============================================================================
# Installation
# ==============================================================================

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

setup-dev: install-dev
	cp .env.example .env
	@echo "Please edit .env with your configuration"
	@echo "Then run: make db-upgrade"

# ==============================================================================
# Testing
# ==============================================================================

test:
	pytest

test-unit:
	pytest tests/unit

test-integration:
	pytest tests/integration

test-chaos:
	pytest tests/chaos

test-cov:
	pytest --cov=prsm --cov-report=html --cov-report=term

test-hardhat:
	cd contracts && npx hardhat test

test-all: test test-hardhat

# ==============================================================================
# Code Quality
# ==============================================================================

lint:
	flake8 prsm tests
	mypy prsm
	black --check prsm tests
	isort --check-only prsm tests

format:
	black prsm tests
	isort prsm tests

# ==============================================================================
# Cleanup
# ==============================================================================

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

# ==============================================================================
# Development Server
# ==============================================================================

run:
	uvicorn prsm.interface.api.main:app --reload --host 0.0.0.0 --port 8000

# ==============================================================================
# Database (alembic)
# ==============================================================================

db-upgrade:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-revision:
	alembic revision --autogenerate -m "$(msg)"

# ==============================================================================
# Docker
# ==============================================================================

# Image registry configuration — override via env or command line.
GHCR_OWNER ?= ryneschultz
IMAGE_NAME ?= prsm
IMAGE_TAG ?= latest

docker-build:
	@echo "🐳 Building Docker image $(IMAGE_NAME):$(IMAGE_TAG)..."
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "✅ Built: $(IMAGE_NAME):$(IMAGE_TAG)"

docker-run:
	docker-compose up -d

docker-run-dev:
	docker-compose -f docker-compose.yml -f docker/docker-compose.dev.yml up -d

docker-run-performance:
	docker-compose -f docker-compose.yml -f docker/docker-compose.performance.yml up -d

docker-run-observability:
	docker-compose -f docker-compose.yml -f docker/docker-compose.observability.yml up -d

docker-down:
	docker-compose down

docker-clean:
	docker system prune -f
	docker volume prune -f

docker-logs:
	docker-compose logs -f prsm-api

docker-shell:
	docker-compose exec prsm-api bash

docker-push:
	@echo "📤 Pushing $(IMAGE_NAME):$(IMAGE_TAG) to GHCR..."
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) ghcr.io/$(GHCR_OWNER)/$(IMAGE_NAME):$(IMAGE_TAG)
	docker push ghcr.io/$(GHCR_OWNER)/$(IMAGE_NAME):$(IMAGE_TAG)
	@echo "✅ Pushed: ghcr.io/$(GHCR_OWNER)/$(IMAGE_NAME):$(IMAGE_TAG)"

bootstrap-build:
	@echo "🚀 Building bootstrap Docker image..."
	cd docker && docker build -f Dockerfile.bootstrap -t prsm-bootstrap:$(IMAGE_TAG) ..
	@echo "✅ Built: prsm-bootstrap:$(IMAGE_TAG)"

# ==============================================================================
# Kubernetes
# ==============================================================================

k8s-deploy:
	kubectl apply -k deploy/kubernetes/base

k8s-deploy-production:
	kubectl apply -k deploy/kubernetes/overlays/production

k8s-status:
	kubectl get pods,hpa,vpa -n prsm-system

k8s-logs:
	kubectl logs -f deployment/prsm-api -n prsm-system

# ==============================================================================
# Observability
# ==============================================================================

obs-stack-up:
	docker-compose -f docker-compose.yml -f docker/docker-compose.observability.yml up -d

obs-stack-down:
	docker-compose -f docker-compose.yml -f docker/docker-compose.observability.yml down

obs-metrics:
	curl -s http://localhost:9091/metrics | head -20

obs-logs:
	docker-compose -f docker-compose.yml -f docker/docker-compose.observability.yml logs -f grafana-enhanced

obs-dashboards:
	@echo "Grafana:    http://localhost:3000 (admin/prsm_admin)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Jaeger:     http://localhost:16686"
	@echo "Kibana:     http://localhost:5601"

metrics:
	@echo "Metrics are available at /metrics when the API server is running (make run)."

# ==============================================================================
# Documentation
# ==============================================================================

docs:
	mkdocs build

docs-serve:
	mkdocs serve

# ==============================================================================
# CI helpers
# ==============================================================================

ci-test: lint test-cov

# ==============================================================================
# Deployment — Multi-channel publishing
# ==============================================================================
# Three channels, canonical naming `publish-{pypi,npm,homebrew}` per Phase 3.x.1
# Task 11 design plan. Each channel has a dry-run variant that validates without
# actually publishing.
#
# `publish` (no suffix) is preserved as a back-compat alias for `publish-pypi`.

# Build Python package (creates .whl and .tar.gz in dist/)
build: clean
	@echo "📦 Building Python package..."
	python -m build
	@echo "✅ Built. Check dist/"

# ---------- PyPI ----------

# Validate PyPI package without uploading
publish-pypi-dry-run: build
	@echo "🧪 Validating PyPI artifacts..."
	@if [ ! -d "dist" ] || [ -z "$$(ls -A dist 2>/dev/null)" ]; then \
		echo "❌ No dist/ directory found. Build failed."; \
		exit 1; \
	fi
	twine check dist/*
	@echo "📦 Would upload these files to PyPI:"
	@ls -1 dist/
	@echo "✅ PyPI dry-run complete. Run 'make publish-pypi' to upload for real."

# Publish to TestPyPI (staging).
# Requires TWINE_USERNAME + TWINE_PASSWORD env vars.
publish-test:
	@echo "🧪 Publishing to TestPyPI..."
	@if [ ! -d "dist" ] || [ -z "$$(ls -A dist 2>/dev/null)" ]; then \
		echo "❌ No dist/ directory found. Run 'make build' first."; \
		exit 1; \
	fi
	twine upload --repository testpypi dist/*
	@echo "✅ Published. Verify: https://test.pypi.org/project/prsm-network/"

# Publish to PyPI (production) — irreversible.
# Requires TWINE_USERNAME + TWINE_PASSWORD env vars.
publish-pypi:
	@echo "🚀 Publishing to PyPI (production)..."
	@if [ ! -d "dist" ] || [ -z "$$(ls -A dist 2>/dev/null)" ]; then \
		echo "❌ No dist/ directory found. Run 'make build' first."; \
		exit 1; \
	fi
	twine upload dist/*
	@echo "✅ Published. Verify: https://pypi.org/project/prsm-network/"

# Back-compat alias.
publish: publish-pypi

# ---------- npm ----------

# Validate npm package without publishing.
publish-npm-dry-run:
	@echo "🧪 Validating npm package..."
	@cd npm && bash test.sh
	@echo "📦 Would publish these files to npm:"
	@cd npm && npm pack --dry-run
	@echo "✅ npm dry-run complete. Run 'make publish-npm' to publish for real."

# Publish to npm registry. Requires `npm login` first.
publish-npm:
	@echo "🚀 Publishing prsm-mcp to npm..."
	@cd npm && bash test.sh
	@cd npm && npm publish --access public
	@echo "✅ Published. Verify: https://www.npmjs.com/package/prsm-mcp"

# ---------- Homebrew ----------

# Validate Homebrew formula without publishing.
# Two-stage validation:
#   1. Ruby syntax check (universal — works without brew installed)
#   2. brew audit (only if brew is available; relaxed flags for compatibility
#      across Homebrew versions). Will report real warnings about placeholder
#      sha256 — that's expected until regenerate-formula.sh runs against a
#      published PyPI artifact.
publish-homebrew-dry-run:
	@echo "🧪 Validating Homebrew formula..."
	@if [ ! -f "ops/homebrew-tap/Formula/prsm.rb" ]; then \
		echo "❌ Formula/prsm.rb not found — Phase 3.x.1 Task 10 scaffold missing."; \
		exit 1; \
	fi
	@echo "  Step 1/2: Ruby syntax check..."
	@ruby -c ops/homebrew-tap/Formula/prsm.rb >/dev/null
	@echo "  ✅ Ruby syntax OK."
	@if command -v brew >/dev/null 2>&1; then \
		echo "  Step 2/2: brew audit (best-effort across versions)..."; \
		brew audit --formula ops/homebrew-tap/Formula/prsm.rb 2>&1 \
			| grep -v "0000000000000000000000000000000000000000000000000000000000000000" \
			| head -20 || true; \
		echo "  ✅ brew audit run (sha256 placeholders expected until regenerate-formula.sh)."; \
	else \
		echo "  Step 2/2: brew not installed; skipping formula audit."; \
	fi
	@echo "✅ Homebrew formula dry-run complete."

# Publish Homebrew formula to the prsm/homebrew-tap repo.
publish-homebrew:
	@echo "🚀 Publishing Homebrew formula..."
	@if [ ! -d "ops/homebrew-tap" ]; then \
		echo "❌ Homebrew tap formula not yet scaffolded (Phase 3.x.1 Task 10 pending)."; \
		echo "   See docs/2026-04-26-phase3.x.1-mcp-server-completion-design-plan.md."; \
		exit 1; \
	fi
	@echo "ℹ️  Push ops/homebrew-tap/Formula/prsm.rb to prsm-network/homebrew-tap repo."
	@echo "   Verify: brew tap prsm-network/tap && brew install prsm"

# Convenience: run all three dry-runs in sequence.
publish-all-dry-run: publish-pypi-dry-run publish-npm-dry-run publish-homebrew-dry-run
	@echo "✅ All dry-runs complete."

# ==============================================================================
# Smoke test
# ==============================================================================

smoke:
	@echo "🔥 Running smoke test..."
	pytest tests/unit/test_bootstrap_server.py -v --tb=short
	@echo "✅ Smoke test passed."
