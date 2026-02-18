# Test Skip Resolution Roadmap

> **Current status:** 920 passed / 31 failed / 77 skipped (as of 2026-02-18)
>
> This document catalogs every remaining skipped test, explains why it skips,
> identifies the root cause, and proposes a concrete fix or implementation task.
> Items are organized into prioritized phases.

---

## Phase 1: Quick Wins (0 code changes required)

### 1A. Database Environment Configuration (10 skips)

**File:** `tests/integration/test_ftns_concurrency_integration.py`

These 10 tests use PostgreSQL-specific features (row-level locking, `SELECT FOR UPDATE`,
`asyncpg`) and skip when `DATABASE_URL` or `TEST_DATABASE_URL` is not set.

| Test | Line | Skip Reason |
|------|------|-------------|
| `test_concurrent_balance_updates` | 124 | No DATABASE_URL |
| `test_concurrent_transaction_processing` | 247 | No DATABASE_URL |
| `test_double_spend_prevention` | 285 | No DATABASE_URL |
| `test_concurrent_reward_distribution` | 332 | No DATABASE_URL |
| `test_atomic_transfer_operations` | 373 | No DATABASE_URL |
| `test_concurrent_marketplace_purchases` | 408 | No DATABASE_URL |
| `test_concurrent_governance_voting` | 466 | No DATABASE_URL |
| `test_high_frequency_micro_transactions` | 517 | No DATABASE_URL |
| `test_concurrent_staking_operations` | 622 | No DATABASE_URL |
| `test_stress_test_transaction_throughput` | 638 | No DATABASE_URL |

**Resolution:** Add a PostgreSQL service to CI (GitHub Actions) and set `DATABASE_URL`.
For local development, add a `docker-compose.test.yml` with a Postgres container:

```yaml
services:
  test-db:
    image: postgres:16
    environment:
      POSTGRES_DB: prsm_test
      POSTGRES_USER: prsm
      POSTGRES_PASSWORD: test
    ports:
      - "5433:5432"
```

Then run: `DATABASE_URL=postgresql+asyncpg://prsm:test@localhost:5433/prsm_test pytest tests/integration/test_ftns_concurrency_integration.py`

**Effort:** Low (infrastructure only, no code changes)

---

### 1B. Optional Third-Party Dependencies (3 skips)

| Test File | Skip Reason | Fix |
|-----------|-------------|-----|
| `tests/integration/test_ui_integration.py` | `selenium` not installed | `pip install selenium` + browser driver |
| `tests/neural/test_semantic_similarity.py` | `sentence-transformers` not installed | `pip install sentence-transformers` (~420 MB) |
| `tests/new_integration_tests/test_integration_demo.py` | `integration_demo_pgvector` not found | Module is a local script, not a package; fix the import path |

**Resolution:**
- Add `selenium` and `sentence-transformers` to an `[optional]` or `[test-full]` extras group in `pyproject.toml`
- For `integration_demo_pgvector`: the test tries to import `from integration_demo_pgvector import ...` as a sibling module. Either add the directory to `sys.path` in the test or convert to a proper relative import.

**Effort:** Low

---

## Phase 2: Export / Import Fixes (6 skips)

These tests skip because existing code doesn't export the symbols the tests expect.

### 2A. FTNS Service Exports (3 skips)

| Test File | Missing Symbol | Source |
|-----------|---------------|--------|
| `tests/environment/persistent_test_environment.py` | `ftns_service` instance | `prsm.economy.tokenomics.ftns_service` |
| `tests/unit/tokenomics/test_ftns_service.py` | Service constants (`INITIAL_BALANCE`, etc.) | `prsm.economy.tokenomics.ftns_service` |
| `tests/test_ftns_budget_manager.py` | `FTNSBudgetManager` class | `prsm.economy.tokenomics.ftns_budget_manager` |

**Resolution:**
1. Export a module-level `ftns_service` singleton from `prsm/economy/tokenomics/ftns_service.py`
2. Export constants (`INITIAL_BALANCE`, `MIN_TRANSACTION_AMOUNT`, etc.) at module level
3. Verify `FTNSBudgetManager` is importable from `prsm.economy.tokenomics.ftns_budget_manager`

**Effort:** Low (add exports to existing modules)

---

### 2B. Specific Missing Symbols (3 skips)

| Test File | Missing Symbol | Fix |
|-----------|---------------|-----|
| `tests/test_budget_api.py` | `prsm.interface.auth` module | Create auth dependency module for FastAPI routes |
| `tests/test_expanded_marketplace.py` | Incomplete SQLAlchemy imports in marketplace models | Fix imports in `prsm/economy/marketplace/` models |
| `tests/test_hybrid_architecture_integration.py` | `AgentTask` class in `prsm.core.models` | Add `AgentTask` Pydantic model to core models |

**Resolution:** Each requires adding a small module or model class. The `prsm.interface.auth` module is a FastAPI dependency that extracts the current user from the JWT token - this is a standard pattern.

**Effort:** Medium

---

## Phase 3: NWTN Subsystem (10 skips + 6 downstream)

The NWTN (Newton) reasoning engine is referenced by many tests but key modules are missing.

### Missing Modules

| Module | Required By | Description |
|--------|-------------|-------------|
| `prsm.compute.nwtn.orchestrator` | 4 test files | Central orchestrator class (`NWTNOrchestrator`) |
| `prsm.compute.nwtn.meta_reasoning_engine` | 4 test files | Meta-reasoning pipeline |
| `prsm.compute.nwtn.complete_system` | 1 test file | End-to-end system entry point |
| `prsm.compute.nwtn.external_storage_config` | 1 test file | External storage configuration |

### Directly Blocked Tests (10)

| Test File | Lines | Blocked By |
|-----------|-------|------------|
| `tests/integration/api/test_endpoint_integration.py` | 28 | `orchestrator` |
| `tests/test_nwtn_direct_prompt_1.py` | 24 | `meta_reasoning_engine` |
| `tests/test_nwtn_final_clean.py` | 23 | `meta_reasoning_engine` |
| `tests/test_nwtn_final_fixed.py` | 23 | `meta_reasoning_engine` |
| `tests/test_nwtn_integration.py` | 36 | `orchestrator` |
| `tests/test_nwtn_prompt_1.py` | 31 | `complete_system` |
| `tests/test_nwtn_provenance_integration.py` | 54 | `meta_reasoning_engine` |
| `tests/test_nwtn_search_fix.py` | 15 | `external_storage_config` |
| `tests/test_nwtn_simple.py` | 14 | `orchestrator` |
| `tests/test_prsm_system_integration.py` | 63 | `orchestrator` |

### Downstream Tests Also Blocked (6)

These user-workflow and integration tests also depend on NWTN:

| Test File | Lines | Additional Dependencies |
|-----------|-------|------------------------|
| `tests/integration/workflows/test_user_workflows.py` | 53 | NWTNOrchestrator |
| `tests/integration/workflows/test_user_workflows.py` | 163 | Marketplace API |
| `tests/integration/workflows/test_user_workflows.py` | 342 | `prsm.collaboration` |
| `tests/integration/workflows/test_user_workflows.py` | 611 | NWTNOrchestrator |
| `tests/integration/workflows/test_user_workflows.py` | 674 | NWTNOrchestrator |
| `tests/test_hierarchical_consensus.py` | 25 | `prsm.performance` |

### Implementation Plan

The NWTN subsystem already has extensive code under `prsm/compute/nwtn/` (reasoning modules,
convergence detection, meta-generation engines, etc.). The missing pieces are the
**orchestrator** (which ties them together) and the **meta-reasoning engine** (which
routes queries through the reasoning pipeline).

**Step 1:** Create `prsm/compute/nwtn/orchestrator.py` with an `NWTNOrchestrator` class that:
- Accepts a query and optional context
- Routes through the existing reasoning pipeline
- Returns a structured response

**Step 2:** Create `prsm/compute/nwtn/meta_reasoning_engine.py` that:
- Implements the meta-reasoning loop (System 1 fast path + System 2 deep reasoning)
- Uses existing modules: `reasoning/`, `convergence_analyzer`, `meta_generation_engine`

**Step 3:** Create `prsm/compute/nwtn/complete_system.py` as a facade that combines
the orchestrator with provenance tracking and content grounding.

**Step 4:** Create `prsm/compute/nwtn/external_storage_config.py` for IPFS/external
storage configuration.

**Effort:** High (core feature implementation)

---

## Phase 4: Marketplace Service Completion (13 skips)

### Missing Methods on `RealMarketplaceService`

| Method | Test Files Using It |
|--------|-------------------|
| `create_resource_listing()` | marketplace_production, marketplace_activation |
| `create_ai_model_listing()` | marketplace_activation |
| `create_dataset_listing()` | marketplace_activation |
| `create_agent_listing()` | marketplace_activation |
| `create_tool_listing()` | marketplace_activation |
| `search_resources()` | marketplace_production, marketplace_activation |
| `get_comprehensive_stats()` | marketplace_production, marketplace_activation |

### Missing Enum Value

`ModelProvider.PRSM` needs to be added to the `ModelProvider` enum.

### Blocked Tests

| Test File | Count |
|-----------|-------|
| `tests/integration/test_marketplace_production_integration.py` | 5 |
| `tests/new_integration_tests/test_marketplace_activation.py` | 8 |

### Implementation Plan

**Step 1:** Add `PRSM = "prsm"` to the `ModelProvider` enum

**Step 2:** Implement CRUD methods on `RealMarketplaceService`:
- `create_resource_listing()` - Generic resource listing creation
- `create_ai_model_listing()` - AI model-specific listing
- `create_dataset_listing()` - Dataset-specific listing
- `create_agent_listing()` - Agent-specific listing
- `create_tool_listing()` - Tool-specific listing

**Step 3:** Implement query methods:
- `search_resources()` - Filtered search across listings
- `get_comprehensive_stats()` - Aggregate marketplace statistics

**Effort:** Medium-High (feature implementation on existing service)

---

## Phase 5: Performance Module (2 skips + 1 benchmark)

### Missing Module: `prsm.performance`

| Test File | What It Tests |
|-----------|--------------|
| `tests/test_performance_instrumentation.py` | Performance metric collection and instrumentation |
| `tests/test_performance_optimization.py` | Automatic performance optimization strategies |
| `tests/test_benchmark_orchestrator.py` | Benchmark orchestration and reporting |

### Implementation Plan

Create `prsm/performance/` package with:
- `instrumentation.py` - Decorators and context managers for timing, memory profiling
- `optimization.py` - Caching strategies, batch processing optimization
- `benchmark_orchestrator.py` - Test orchestration, result aggregation, reporting

**Effort:** Medium

---

## Phase 6: Generic "Not Yet Implemented" Tests (26 skips)

These 26 tests all skip with the generic message "Module dependencies not yet fully
implemented". Each needs individual investigation to determine the actual blocker.

### Grouped by Subsystem

#### Integration Tests (10 files)

| Test File | Likely Dependency |
|-----------|------------------|
| `test_api_integration_comprehensive.py` | Full API stack |
| `test_collaboration_platform_integration.py` | `prsm.collaboration` module |
| `test_complete_collaboration_platform.py` | `prsm.collaboration` module |
| `test_end_to_end_prsm_workflow.py` | NWTN + FTNS + Marketplace |
| `test_full_spectrum_integration.py` | All subsystems |
| `test_integration_suite_runner.py` | Test orchestration infrastructure |
| `test_p2p_integration.py` | P2P networking stack |
| `test_phase7_integration.py` | Phase 7 features (governance?) |
| `test_real_data_integration.py` | Real data pipelines |
| `test_system_resilience_integration.py` | Fault tolerance infrastructure |

#### Tokenomics & Economy (4 files)

| Test File | Likely Dependency |
|-----------|------------------|
| `test_advanced_ftns.py` | Advanced FTNS features |
| `test_advanced_tokenomics_integration.py` | Full tokenomics stack |
| `test_marketplace.py` | Marketplace + FTNS integration |
| `test_full_governance_system.py` | Governance + FTNS |

#### Core System (4 files)

| Test File | Likely Dependency |
|-----------|------------------|
| `test_agent_framework.py` | Agent orchestration framework |
| `test_breakthrough_modes.py` | NWTN breakthrough detection |
| `test_consensus_integration.py` | Consensus mechanism |
| `test_150k_papers_provenance.py` | Provenance at scale |

#### External Integrations (3 files)

| Test File | Likely Dependency |
|-----------|------------------|
| `test_openai_free_tier.py` | OpenAI API integration |
| `test_openai_integration.py` | OpenAI API integration |
| `test_openai_real_integration.py` | OpenAI API integration |

#### Scripts & Performance (3 files)

| Test File | Likely Dependency |
|-----------|------------------|
| `test_governance.py` | Governance scripts |
| `simple_performance_test.py` | Performance testing |
| `test_performance_benchmarks_alt.py` | Benchmark infrastructure |

#### P2P & Federation (2 files)

| Test File | Likely Dependency |
|-----------|------------------|
| `test_p2p_federation.py` | P2P federation protocol |
| `test_production_p2p_federation.py` | Production P2P deployment |

### Resolution Strategy

Many of these will automatically resolve as Phases 3-5 are completed (NWTN, Marketplace,
Performance). The remaining ones likely need:
1. Individual investigation of each file's actual import block
2. Either implementing the missing module or fixing the import path
3. Some may be candidates for deletion if they test features that have been redesigned

**Effort:** High (requires investigation + implementation across many subsystems)

---

## Summary

| Phase | Description | Skips Resolved | Effort |
|-------|-------------|---------------|--------|
| 1A | Database env config for CI | 10 | Low |
| 1B | Optional third-party deps | 3 | Low |
| 2A | FTNS service exports | 3 | Low |
| 2B | Missing symbols/modules | 3 | Medium |
| 3 | NWTN subsystem | 10 (+6 downstream) | High |
| 4 | Marketplace service | 13 | Medium-High |
| 5 | Performance module | 3 | Medium |
| 6 | Generic "not implemented" | 26 | High |
| **Total** | | **77** | |

### Recommended Execution Order

1. **Phase 1** (13 skips) - Zero or minimal code changes, immediate wins
2. **Phase 2** (6 skips) - Small export/import fixes, low risk
3. **Phase 4** (13 skips) - Marketplace is well-defined, existing service to extend
4. **Phase 5** (3 skips) - Performance module is self-contained
5. **Phase 3** (16 skips) - NWTN is the largest feature gap, high impact
6. **Phase 6** (26 skips) - Many will auto-resolve from earlier phases; remainder needs investigation

### Target: Zero Skips

After all phases are complete, the test suite should have **0 skips** (or only
infrastructure-dependent skips that are expected in environments without PostgreSQL,
Selenium, or GPU access). All "not yet implemented" skips should be resolved by either
implementing the feature or removing obsolete test files.
