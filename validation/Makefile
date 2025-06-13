# PRSM Validation Pipeline Automation
# Addresses technical reassessment requirement for systematic evidence collection

.PHONY: install setup validate-full validate-benchmarks validate-economic validate-safety validate-network dashboard clean help

# Default target
help:
	@echo "PRSM Validation Pipeline Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install validation dependencies"
	@echo "  setup         Initialize validation infrastructure"
	@echo ""
	@echo "Validation:"
	@echo "  validate-full Run complete validation pipeline"
	@echo "  validate-benchmarks Run performance benchmarks only"
	@echo "  validate-economic   Run economic simulation only"
	@echo "  validate-safety     Run safety tests only"
	@echo "  validate-network    Run network tests only"
	@echo ""
	@echo "Monitoring:"
	@echo "  dashboard     Launch validation dashboard"
	@echo "  evidence      Generate evidence report"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean         Clean validation artifacts"
	@echo "  archive       Archive validation results"

# Installation and setup
install:
	@echo "📦 Installing validation dependencies..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@echo "✅ Dependencies installed"

setup: install
	@echo "🔧 Setting up validation infrastructure..."
	python evidence_collector.py
	mkdir -p logs reports archives
	@echo "✅ Validation infrastructure ready"

# Validation execution
validate-full:
	@echo "🚀 Starting full validation pipeline..."
	python orchestrator.py
	@echo "✅ Full validation completed"

validate-benchmarks:
	@echo "🏆 Running performance benchmarks..."
	python -c "import asyncio; from orchestrator import ValidationOrchestrator, ValidationConfig, EvidenceCollector; \
	config = ValidationConfig(run_benchmarks=True, run_economic_simulation=False, run_safety_tests=False, run_network_tests=False); \
	collector = EvidenceCollector(); \
	orchestrator = ValidationOrchestrator(config, collector); \
	asyncio.run(orchestrator.execute_validation_pipeline())"
	@echo "✅ Benchmarks completed"

validate-economic:
	@echo "💰 Running economic simulation..."
	python -c "import asyncio; from orchestrator import ValidationOrchestrator, ValidationConfig, EvidenceCollector; \
	config = ValidationConfig(run_benchmarks=False, run_economic_simulation=True, run_safety_tests=False, run_network_tests=False); \
	collector = EvidenceCollector(); \
	orchestrator = ValidationOrchestrator(config, collector); \
	asyncio.run(orchestrator.execute_validation_pipeline())"
	@echo "✅ Economic simulation completed"

validate-safety:
	@echo "🛡️ Running safety tests..."
	python -c "import asyncio; from orchestrator import ValidationOrchestrator, ValidationConfig, EvidenceCollector; \
	config = ValidationConfig(run_benchmarks=False, run_economic_simulation=False, run_safety_tests=True, run_network_tests=False); \
	collector = EvidenceCollector(); \
	orchestrator = ValidationOrchestrator(config, collector); \
	asyncio.run(orchestrator.execute_validation_pipeline())"
	@echo "✅ Safety tests completed"

validate-network:
	@echo "🌐 Running network tests..."
	python -c "import asyncio; from orchestrator import ValidationOrchestrator, ValidationConfig, EvidenceCollector; \
	config = ValidationConfig(run_benchmarks=False, run_economic_simulation=False, run_safety_tests=False, run_network_tests=True); \
	collector = EvidenceCollector(); \
	orchestrator = ValidationOrchestrator(config, collector); \
	asyncio.run(orchestrator.execute_validation_pipeline())"
	@echo "✅ Network tests completed"

# Monitoring and reporting
dashboard:
	@echo "📊 Launching validation dashboard..."
	streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
	@echo "🌐 Dashboard available at http://localhost:8501"

evidence:
	@echo "📋 Generating evidence report..."
	python -c "from evidence_collector import EvidenceCollector; \
	import json; \
	collector = EvidenceCollector(); \
	report = collector.generate_evidence_report(); \
	with open('evidence_report.json', 'w') as f: json.dump(report, f, indent=2); \
	print('Evidence report generated: evidence_report.json')"

# Maintenance
clean:
	@echo "🧹 Cleaning validation artifacts..."
	rm -rf __pycache__/ .pytest_cache/ *.pyc
	rm -f *.log validation_results_*.json
	@echo "✅ Cleanup completed"

archive:
	@echo "📦 Archiving validation results..."
	DATE=$$(date +%Y%m%d_%H%M%S); \
	mkdir -p archives/$$DATE; \
	cp -r results/ benchmarks/ economic_simulations/ safety_tests/ network_deployments/ archives/$$DATE/ 2>/dev/null || true; \
	cp *.json archives/$$DATE/ 2>/dev/null || true; \
	echo "✅ Results archived to archives/$$DATE"

# Continuous integration targets
ci-validate: setup validate-full evidence
	@echo "🤖 CI validation pipeline completed"

ci-quick: setup validate-benchmarks evidence
	@echo "🤖 CI quick validation completed"

# Development targets
dev-setup: install
	@echo "👨‍💻 Setting up development environment..."
	pip install black isort mypy pytest-cov
	@echo "✅ Development environment ready"

lint:
	@echo "🔍 Running code quality checks..."
	black --check .
	isort --check-only .
	mypy --ignore-missing-imports .
	@echo "✅ Code quality checks passed"

test:
	@echo "🧪 Running validation framework tests..."
	pytest -v --cov=. --cov-report=term-missing
	@echo "✅ Tests completed"

# Investor demonstration targets
demo-setup:
	@echo "🎭 Setting up investor demonstration..."
	$(MAKE) setup
	$(MAKE) validate-full
	$(MAKE) evidence
	@echo "✅ Investor demo ready - run 'make dashboard' to view"

demo-run: demo-setup dashboard

# Documentation targets
docs:
	@echo "📚 Generating validation documentation..."
	python -c "import json; \
	docs = { \
		'validation_framework': 'Comprehensive evidence collection and validation pipeline', \
		'components': ['evidence_collector', 'orchestrator', 'dashboard'], \
		'usage': 'make validate-full for complete validation', \
		'dashboard': 'make dashboard for real-time monitoring' \
	}; \
	with open('VALIDATION_DOCS.json', 'w') as f: json.dump(docs, f, indent=2)"
	@echo "✅ Documentation generated"