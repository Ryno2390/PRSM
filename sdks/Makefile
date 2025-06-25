# PRSM SDKs Makefile
# Build, test, and publish all PRSM SDKs

.PHONY: help build test lint clean install publish python-* javascript-* go-*

# Default target
help:
	@echo "PRSM SDKs Build System"
	@echo ""
	@echo "Available commands:"
	@echo "  build          - Build all SDKs"
	@echo "  test           - Run tests for all SDKs"
	@echo "  lint           - Run linting for all SDKs"
	@echo "  clean          - Clean build artifacts"
	@echo "  install        - Install dependencies for all SDKs"
	@echo "  publish        - Publish all SDKs (requires credentials)"
	@echo ""
	@echo "Language-specific commands:"
	@echo "  python-build   - Build Python SDK"
	@echo "  python-test    - Test Python SDK"
	@echo "  python-lint    - Lint Python SDK"
	@echo "  python-publish - Publish Python SDK to PyPI"
	@echo ""
	@echo "  javascript-build   - Build JavaScript SDK"
	@echo "  javascript-test    - Test JavaScript SDK"
	@echo "  javascript-lint    - Lint JavaScript SDK"
	@echo "  javascript-publish - Publish JavaScript SDK to npm"
	@echo ""
	@echo "  go-build       - Build Go SDK"
	@echo "  go-test        - Test Go SDK"
	@echo "  go-lint        - Lint Go SDK"

# Build all SDKs
build: python-build javascript-build go-build

# Test all SDKs
test: python-test javascript-test go-test

# Lint all SDKs
lint: python-lint javascript-lint go-lint

# Clean all build artifacts
clean: python-clean javascript-clean go-clean

# Install dependencies for all SDKs
install: python-install javascript-install go-install

# Publish all SDKs
publish: python-publish javascript-publish

# Python SDK commands
python-install:
	@echo "Installing Python SDK dependencies..."
	cd python && pip install -e ".[dev]"

python-build:
	@echo "Building Python SDK..."
	cd python && python -m build

python-test:
	@echo "Testing Python SDK..."
	cd python && pytest --cov=prsm_sdk --cov-report=term-missing

python-lint:
	@echo "Linting Python SDK..."
	cd python && flake8 prsm_sdk
	cd python && black --check prsm_sdk
	cd python && isort --check-only prsm_sdk
	cd python && mypy prsm_sdk

python-format:
	@echo "Formatting Python SDK..."
	cd python && black prsm_sdk
	cd python && isort prsm_sdk

python-clean:
	@echo "Cleaning Python SDK..."
	cd python && rm -rf build/ dist/ *.egg-info/ .coverage htmlcov/
	cd python && find . -type d -name __pycache__ -delete
	cd python && find . -type f -name "*.pyc" -delete

python-publish:
	@echo "Publishing Python SDK to PyPI..."
	cd python && python -m build
	cd python && twine upload dist/*

# JavaScript SDK commands
javascript-install:
	@echo "Installing JavaScript SDK dependencies..."
	cd javascript && npm install

javascript-build:
	@echo "Building JavaScript SDK..."
	cd javascript && npm run build

javascript-test:
	@echo "Testing JavaScript SDK..."
	cd javascript && npm test

javascript-lint:
	@echo "Linting JavaScript SDK..."
	cd javascript && npm run lint
	cd javascript && npm run typecheck

javascript-format:
	@echo "Formatting JavaScript SDK..."
	cd javascript && npm run lint:fix

javascript-clean:
	@echo "Cleaning JavaScript SDK..."
	cd javascript && rm -rf dist/ node_modules/ coverage/

javascript-publish:
	@echo "Publishing JavaScript SDK to npm..."
	cd javascript && npm run build
	cd javascript && npm publish --access public

# Go SDK commands
go-install:
	@echo "Installing Go SDK dependencies..."
	cd go && go mod download

go-build:
	@echo "Building Go SDK..."
	cd go && go build ./...

go-test:
	@echo "Testing Go SDK..."
	cd go && go test -race -coverprofile=coverage.out ./...

go-lint:
	@echo "Linting Go SDK..."
	cd go && golangci-lint run

go-format:
	@echo "Formatting Go SDK..."
	cd go && go fmt ./...

go-clean:
	@echo "Cleaning Go SDK..."
	cd go && go clean ./...
	cd go && rm -f coverage.out

# Development helpers
dev-setup:
	@echo "Setting up development environment..."
	@which python3 || (echo "Python 3 is required" && exit 1)
	@which node || (echo "Node.js is required" && exit 1)
	@which go || (echo "Go is required" && exit 1)
	$(MAKE) install

# Check all SDKs are working
check: build test lint
	@echo "All SDKs passed checks!"

# Release preparation
release-check:
	@echo "Checking release readiness..."
	$(MAKE) clean
	$(MAKE) build
	$(MAKE) test
	$(MAKE) lint
	@echo "Release checks passed!"