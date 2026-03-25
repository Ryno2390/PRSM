# Phase 5 — Test Suite Completeness & Stale Skip Audit

## Overview

Phase 4 delivered the external deployment configuration (bootstrap domain fix, multi-region
fallback config, env template, 14 deployment tests). The infrastructure steps (mainnet FTNS,
EU/APAC bootstrap nodes) are pending external access.

Phase 5 targets **test suite completeness**: 58 test files currently skip on module load.
An audit conducted March 24, 2026 found that **~19 of those skips are stale** — the underlying
modules were implemented after the skip was added, but the skip condition was never removed.
Removing stale skips and fixing the remaining ~10 genuine gaps will materially increase the
passing test count and eliminate a significant red flag for investor/developer audits.

**Exit criterion:** All module-level `pytest.skip(..., allow_module_level=True)` calls are
either removed (stale condition) or replaced with a narrowly-scoped `pytest.importorskip()`
or `pytest.mark.skipif`; `pytest` reports 0 module-level skipped test *files*; the total
passing count increases from the current 3,470 baseline.

---

## Background: What Was Found

Running `pytest --collect-only` against all 58 skip files revealed:

| Category | Files | Tests Unlocked |
|----------|-------|----------------|
| Stale skips (module works fine) | ~19 | Unknown — run to find out |
| Genuine gaps (code missing) | ~10 | Requires implementation |
| External dependency (asyncpg) | 1 | Requires `pip install asyncpg` |

Three files were already verified passing after the audit:
- `tests/test_performance_instrumentation.py` — 2 tests pass
- `tests/test_hierarchical_consensus.py` — 2 tests pass
- `tests/test_budget_api.py` — 10 tests pass

These 14 tests are already counted in the 3,470 baseline (the full suite was re-run after
discovering them). The remaining files still carry the stale skip.

---

## Step 1 — Remove Stale Skips (19 Files)

Each of these files has a `try/except ImportError → pytest.skip(allow_module_level=True)`
block. The import now succeeds, so the skip never fires. The right fix is to remove the
entire `try/except` guard and let pytest collect normally. If any test *body* fails for
a real reason, fix the body — do not re-add a module-level skip.

### Group A: Top-level test files (12 files)

| File | Skip Reason (stale) |
|------|---------------------|
| `tests/test_advanced_tokenomics_integration.py` | "Module dependencies not yet fully implemented" |
| `tests/test_agent_framework.py` | "Module dependencies not yet fully implemented" |
| `tests/test_p2p_federation.py` | "P2P network module dependencies not yet fully implemented" |
| `tests/test_150k_papers_provenance.py` | "Module dependencies not yet fully implemented" |
| `tests/test_advanced_ftns.py` | "Module dependencies not yet fully implemented" |
| `tests/test_nwtn_direct_prompt_1.py` | "NWTN meta_reasoning_engine module not yet implemented" |
| `tests/test_nwtn_provenance_integration.py` | "NWTN meta_reasoning_engine module not yet implemented" |

### Group B: scripts_integration (4 files)

| File | Skip Reason (stale) |
|------|---------------------|
| `tests/scripts_integration/test_openai_free_tier.py` | "Module dependencies not yet fully implemented" |
| `tests/scripts_integration/test_governance.py` | "Module dependencies not yet fully implemented" |
| `tests/scripts_integration/test_openai_real_integration.py` | "Module dependencies not yet fully implemented" |
| `tests/scripts_integration/test_openai_integration.py` | "Module dependencies not yet fully implemented" |

### Group C: integration tests (8 files)

| File | Skip Reason (stale) |
|------|---------------------|
| `tests/integration/test_system_resilience_integration.py` | "Module dependencies not yet fully implemented" |
| `tests/integration/test_collaboration_platform_integration.py` | "Module dependencies not yet fully implemented" |
| `tests/integration/test_complete_collaboration_platform.py` | "Module dependencies not yet fully implemented" |
| `tests/integration/test_end_to_end_prsm_workflow.py` | "Module dependencies not yet fully implemented" |
| `tests/integration/test_api_integration_comprehensive.py` | "Module dependencies not yet fully implemented" |
| `tests/integration/api/test_comprehensive_api.py` | "API modules have import errors (pydantic regex issue)" |
| `tests/integration/api/test_endpoint_integration.py` | "API endpoint modules have import errors (pydantic/orchestrator)" |

**Procedure for each file:**
1. Remove the `try/except ImportError` block and the `pytest.skip(allow_module_level=True)` call
2. Run the file: `pytest <file> -v`
3. If any test fails, diagnose and fix; never re-add a module-level skip to paper over a real failure

---

## Step 2 — Fix Genuine Gaps (10 Files)

These files have skips because code genuinely doesn't exist yet. Fix in order of effort.

### 2a. Export FTNS Service Constants (30 min)

**File:** `prsm/economy/tokenomics/ftns_service.py`

**Blocked test:** `tests/unit/tokenomics/test_ftns_service.py` (skip: "FTNS service constants not yet exported")

**What's needed:** The test imports these module-level constants:
```python
BASE_NWTN_FEE
CONTEXT_UNIT_COST
ARCHITECT_DECOMPOSITION_COST
COMPILER_SYNTHESIS_COST
AGENT_COSTS          # dict: agent_type → cost
REWARD_PER_MB
MODEL_CONTRIBUTION_REWARD
SUCCESSFUL_TEACHING_REWARD
```

Search `ftns_service.py` for any existing values that map to these names. If they're
instance attributes on `FTNSService`, extract them as module-level constants. If they don't
exist, add them with reasonable defaults based on the existing pricing logic.

**Verify:** `python -c "from prsm.economy.tokenomics.ftns_service import BASE_NWTN_FEE"`

### 2b. Add Missing Budget Manager Classes (1–2 hr)

**File:** `prsm/economy/tokenomics/ftns_budget_manager.py`

**Blocked test:** `tests/test_ftns_budget_manager.py` (skip: "FTNS budget manager classes not yet fully implemented")

**What's needed:** The test imports:
```python
from prsm.economy.tokenomics.ftns_budget_manager import (
    FTNSBudgetManager, SpendingCategory,    # ✅ exist
    BudgetExpandRequest, BudgetPrediction, BudgetAlert  # ❌ missing
)
```

Add the three missing Pydantic/dataclass models to `ftns_budget_manager.py`.

### 2c. Fix P2P Federation Import Paths (30 min)

**File:** `tests/test_p2p_integration.py`

**Skip reason:** Wrong import path — `collaboration.p2p.*` instead of `prsm.compute.collaboration.p2p.*`

Read the test file and correct the import paths to match the actual module locations in the codebase.

### 2d. Add Breakthrough Mode Exports (30 min)

**File:** `prsm/compute/nwtn/breakthrough_modes.py` (or similar)

**Blocked test:** `tests/test_breakthrough_modes.py`

**What's needed:**
```python
from prsm.compute.nwtn.breakthrough_modes import (
    breakthrough_mode_manager,
    get_breakthrough_mode_config,
    suggest_breakthrough_mode
)
```

Find where breakthrough mode logic lives. If `breakthrough_mode_manager` exists as an
instance, export it. If the functions don't exist, implement them as thin wrappers over
existing breakthrough mode infrastructure.

### 2e. Deferred / Out-of-Scope (Skip These for Phase 5)

These require new modules that are architectural decisions beyond a Phase 5 scope:

| File | Missing | Decision |
|------|---------|----------|
| `test_real_data_integration.py` | `prsm.compute.nwtn.unified_pipeline_controller` | Defer — new module |
| `test_phase7_integration.py` | `prsm.core.enterprise.global_infrastructure.GlobalInfrastructure` | Defer — enterprise-tier feature |
| `test_full_spectrum_integration.py` | `prsm.core.vector_db.VectorDatabase` | Defer — separate vector DB module |
| `test_hybrid_architecture_integration.py` | `prsm.compute.nwtn.hybrid_integration` | Defer — new module |
| `test_integration_suite_runner.py` | Relative test imports | Defer — test infra refactor |
| `test_production_p2p_federation.py` | Federation module exports | Defer — check in Step 1 Group A |
| `test_ftns_concurrency_integration.py` | `asyncpg` package | `pip install asyncpg` if desired |

For each deferred file, update the skip message to be specific:
```python
# Before:
pytest.skip('Module dependencies not yet fully implemented', allow_module_level=True)

# After:
pytest.skip('prsm.compute.nwtn.unified_pipeline_controller not yet implemented (Phase 6)', allow_module_level=True)
```

This signals intentional deferral rather than an unknown bug.

---

## Step 3 — Update `docs/IMPLEMENTATION_STATUS.md`

After completing Steps 1 and 2:
- Update test count badges and table with new passing total
- Add a "Phase 5" section documenting the skip audit
- Remove the outdated "AtomicFTNSService partially broken" note (already fixed in Phase 4 doc update)
- Update skipped count to reflect only intentionally-deferred tests

---

## Step 4 — Write Phase 5 Verification Tests

Create `tests/test_phase5_completeness.py` to verify the state of the test suite itself:

```python
"""
Phase 5 test suite completeness verification.
Ensures no test files have stale module-level skips.
"""
import ast
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
KNOWN_DEFERRED = {
    "test_real_data_integration.py",
    "test_phase7_integration.py",
    "test_full_spectrum_integration.py",
    "test_hybrid_architecture_integration.py",
    "test_integration_suite_runner.py",
    "test_ftns_concurrency_integration.py",
}


def test_no_vague_module_level_skips():
    """
    All remaining module-level skips must reference a specific missing item,
    not a generic 'Module dependencies not yet fully implemented' message.
    """
    vague_message = "Module dependencies not yet fully implemented"
    offenders = []
    for f in REPO_ROOT.glob("tests/**/*.py"):
        if f.name in KNOWN_DEFERRED:
            continue
        text = f.read_text()
        if vague_message in text and "allow_module_level=True" in text:
            offenders.append(str(f.relative_to(REPO_ROOT)))
    assert not offenders, (
        f"These test files still have vague module-level skips:\n"
        + "\n".join(f"  {o}" for o in offenders)
    )
```

---

## Execution Order

| Order | Task | Files Changed | Effort |
|-------|------|--------------|--------|
| 1 | Remove stale skips — Group A (7 files) | 7 test files | 30 min |
| 2 | Remove stale skips — Group B (4 files) | 4 test files | 20 min |
| 3 | Remove stale skips — Group C (7 files) | 7 test files | 30 min |
| 4 | Fix FTNS service constants | `ftns_service.py` | 30 min |
| 5 | Add missing budget manager classes | `ftns_budget_manager.py` | 1–2 hr |
| 6 | Fix P2P integration import paths | `test_p2p_integration.py` | 30 min |
| 7 | Add breakthrough mode exports | `breakthrough_modes.py` | 30 min |
| 8 | Update deferred skip messages (6 files) | 6 test files | 20 min |
| 9 | Update IMPLEMENTATION_STATUS.md | `docs/` | 15 min |
| 10 | Write test_phase5_completeness.py | `tests/` | 30 min |

Steps 1–3 are pure mechanical cleanup and can be done in one pass. Steps 4–7 are small
targeted implementations. The test count improvement from Step 1–3 alone is potentially
significant (19 files × unknown tests per file).

---

## Expected Outcome

**Minimum improvement (stale skips removed, no new failures):**
- 3,470 → 3,470+ passing (actual count depends on how many tests are in the unblocked files)
- Module-level skip files: 58 → ~10 (only genuinely deferred items remain)
- Every remaining skip has a specific, honest message

**Full outcome (Steps 1–10 complete):**
- FTNS service constants exported → `test_ftns_service.py` unblocked
- `BudgetExpandRequest`, `BudgetPrediction`, `BudgetAlert` added → `test_ftns_budget_manager.py` unblocked
- P2P integration import paths fixed → `test_p2p_integration.py` unblocked
- Breakthrough mode exports added → `test_breakthrough_modes.py` unblocked
- All deferred files have specific, honest skip messages

**Investor audit impact:** A reviewer who runs `pytest` sees near-zero module-level skips
and a high passing count, rather than 58 files silently swallowed by stale skip guards.

---

## Completion Summary — [To Be Filled In]

*Fill this section when Phase 5 is complete.*

### Files Changed
<!-- List here -->

### Test Count Delta
<!-- Before: 3,470 | After: ??? -->

### Remaining Deferred Items
<!-- List the items that remain intentionally deferred with justification -->
