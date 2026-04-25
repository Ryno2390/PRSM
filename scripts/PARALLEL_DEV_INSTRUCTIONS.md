# PRSM Parallel Development - Manual Execution

## Instructions
Open 3 separate terminal windows and run each command below.

---

## Terminal 1 - Core Infrastructure & Security Hardening
```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM && kilo run --auto "# PRSM Core Infrastructure & Security Hardening

## Your Focus
Complete core infrastructure tasks to unblock 21 tests:

### 1. PostgreSQL Test Infrastructure
- Create docker-compose.test.yml with PostgreSQL 16
- Update .github/workflows/ for PostgreSQL service
- Document DATABASE_URL setup in README

### 2. Module Exports (6 tests)
- Export ftns_service singleton and constants from prsm/economy/tokenomics/ftns_service.py
- Create prsm/interface/auth.py for FastAPI dependencies (JWT token extraction)
- Add AgentTask Pydantic model to prsm/core/models.py

### 3. Performance Module (3 tests)
Create prsm/performance/ package with:
- instrumentation.py - Timing decorators, memory profiling context managers
- optimization.py - Caching strategies, batch processing optimization
- benchmark_orchestrator.py - Test orchestration, result aggregation, reporting

### 4. Collaboration Module (2 tests)
Create prsm/collaboration/ package with:
- __init__.py
- collaboration_manager.py - Multi-agent collaboration coordination

Reference: docs/development/TEST_SKIP_RESOLUTION_ROADMAP.md Phase 1, 2, 5"
```

---

## Terminal 2 - NWTN Reasoning Engine
```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM && kilo run --auto "# PRSM NWTN Reasoning Engine Completion

## Your Focus
Complete the NWTN subsystem to unblock 22 tests:

### 1. Orchestrator (Critical - blocks 16 tests)
Create prsm/compute/nwtn/orchestrator.py with NWTNOrchestrator class that:
- Accepts queries with optional context
- Routes through reasoning pipeline
- Returns structured NWTNResponse
- Integrates with existing breakthrough_reasoning_coordinator.py

### 2. Meta-Reasoning Engine (4 tests)
Create prsm/compute/nwtn/meta_reasoning_engine.py:
- System 1 (fast path) - Quick responses for simple queries
- System 2 (deep reasoning) - Multi-step reasoning for complex queries
- Use existing: convergence_analyzer, meta_generation_engine

### 3. Complete System Facade (1 test)
Create prsm/compute/nwtn/complete_system.py:
- Entry point combining orchestrator + provenance + content grounding

### 4. External Storage Config (1 test)
Create prsm/compute/nwtn/external_storage_config.py:
- IPFS configuration, external storage backends

Reference: docs/development/TEST_SKIP_RESOLUTION_ROADMAP.md Phase 3"
```

---

## Terminal 3 - Marketplace & Economy
```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM && kilo run --auto "# PRSM Marketplace & Economy Features

## Your Focus
Complete marketplace functionality to unblock 20 tests:

### 1. RealMarketplaceService CRUD Methods (13 tests)
File: prsm/economy/marketplace/real_marketplace_service.py

Implement these methods:
- create_resource_listing(resource_type, data)
- create_ai_model_listing(model_data)
- create_dataset_listing(dataset_data)
- create_agent_listing(agent_data)
- create_tool_listing(tool_data)
- search_resources(filters)
- get_comprehensive_stats()

### 2. ModelProvider Enum Update
Add PRSM = 'prsm' to ModelProvider enum

### 3. Advanced FTNS Features (4 tests)
Complete advanced tokenomics in prsm/economy/tokenomics/

### 4. Governance Integration (2 tests)
Complete prsm/economy/governance/

Reference: docs/development/TEST_SKIP_RESOLUTION_ROADMAP.md Phase 4"
```

---

## Expected Results
Each instance should complete their assigned tasks autonomously.
When finished, you can verify with:
```bash
python3 -m pytest --co -q 2>&1 | grep -c "skipped"
```
