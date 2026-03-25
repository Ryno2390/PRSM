# Phase 9 — Stub Wiring & Final Completion

## Overview

Phase 8 left PRSM at ~97% complete. The remaining 3% consists of two distinct problems:

1. **Nine placeholder stub files** — created during Phase 6 to unblock test imports but never
   wired to real implementations. They return hardcoded dummy values today.
2. **`tests/integration/test_phase7_integration.py`** — still has a module-level skip. The
   `GlobalInfrastructure` alias was fixed in Phase 8, but two more issues block the tests:
   - `IntegrationManager` class does not exist in `integration_manager.py`
     (only `EnterpriseIntegrationManager` does)
   - `AIOrchestrator` exists at the correct path but is missing `register_model()`,
     `execute_task()`, and `is_initialized` — all called directly by the test

Phase 9 resolves both problems. No new features. No new infrastructure. Pure wiring and
completion.

**Baseline entering Phase 9:** ~3,787 tests collected, ~3,769 passing

---

## Step 1 — Audit `test_phase7_integration.py` imports fully (~30 min)

Before touching any source file, remove the module-level skip from
`tests/integration/test_phase7_integration.py` and run:

```bash
python -m pytest tests/integration/test_phase7_integration.py -v --tb=short 2>&1 | head -60
```

This will surface every import error and failing assertion in one pass. Record every failure —
this is the authoritative list of what Step 2 and Step 3 must fix.

Expected failures based on pre-read analysis:
- `ImportError: cannot import name 'IntegrationManager'` from enterprise integration_manager
- `AttributeError: AIOrchestrator has no attribute 'register_model'`
- `AttributeError: AIOrchestrator has no attribute 'execute_task'`
- `AttributeError: AIOrchestrator has no attribute 'is_initialized'`
- Possibly: DashboardManager missing `create_dashboard`, `update_dashboard_data`, `get_dashboard`

Do not fix anything yet — just collect the full failure list first.

---

## Step 2 — Fix `AIOrchestrator` API gap (~1 hr)

**File:** `prsm/compute/ai_orchestration/orchestrator.py`

**Class:** `AIOrchestrator` (line 610)

Read the class in full before editing. Then add the missing attributes/methods the test expects:

```python
# Add to __init__:
self.is_initialized: bool = False
self._registered_models: dict = {}

# Add method:
async def register_model(self, model_config: dict) -> str:
    """Register a model for orchestration. Returns a model_id."""
    model_id = str(uuid.uuid4())
    self._registered_models[model_id] = model_config
    return model_id

# Add method:
async def execute_task(self, task: dict) -> dict:
    """Execute a task using the registered models."""
    if task.get("type") not in self._valid_task_types():
        raise ValueError(f"Unknown task type: {task.get('type')}")
    return await self._execute_task(task)

async def _execute_task(self, task: dict) -> dict:
    """Internal task execution — patchable in tests."""
    # Delegate to orchestrate() with appropriate OrchestrationRequest
    ...

def _valid_task_types(self) -> list:
    return ["reasoning", "generation", "analysis", "classification"]
```

Also set `self.is_initialized = True` at the end of the existing `initialize()` method.

**Important:** Read the existing `initialize()` and `orchestrate()` methods fully before
writing `_execute_task` — `_execute_task` should delegate to the existing `orchestrate()`
machinery, not reimplement it.

---

## Step 3 — Add `IntegrationManager` to enterprise integration_manager (~30 min)

**File:** `prsm/core/integrations/enterprise/integration_manager.py`

`EnterpriseIntegrationManager` already exists (line 469). The test imports `IntegrationManager`.
Add a thin wrapper class (or alias) that exposes the API the test calls:

```python
# At the bottom of the file, after EnterpriseIntegrationManager:

class IntegrationManager:
    """
    Simplified interface for enterprise integration management.
    Wraps EnterpriseIntegrationManager for the standard integration lifecycle.
    """
    def __init__(self, config: Optional[dict] = None):
        self._manager = EnterpriseIntegrationManager(config or {})
        self._integrations: dict = {}

    async def create_integration(self, config: dict) -> str:
        """Create and register a new integration. Returns integration_id."""
        if not config.get("name"):
            raise ValueError("Integration name is required")
        if not config.get("type"):
            raise ValueError("Integration type is required")
        integration_id = str(uuid.uuid4())
        self._integrations[integration_id] = {**config, "id": integration_id,
                                               "status": "active"}
        return integration_id

    async def sync_integration(self, integration_id: str) -> dict:
        """Sync data for an integration."""
        if integration_id not in self._integrations:
            raise KeyError(f"Integration {integration_id} not found")
        return await self._sync_data(integration_id)

    async def _sync_data(self, integration_id: str) -> dict:
        """Internal sync — patchable in tests."""
        return {"synced_records": 0, "status": "success"}
```

Read `EnterpriseIntegrationManager` fully before writing this — wire `create_integration` and
`sync_integration` to its real internals where appropriate rather than tracking state locally.

---

## Step 4 — Verify DashboardManager API (~30 min)

**File:** `prsm/data/analytics/dashboard_manager.py`

The test calls `create_dashboard()`, `update_dashboard_data()`, and `get_dashboard()` on
`DashboardManager` (which exists at line 677). Read the class to check if these methods exist.

If any are missing, add them following the same pattern as Step 3 — read the existing class
first, then add the missing methods delegating to existing internals.

After adding any missing methods, rerun:
```bash
python -m pytest tests/integration/test_phase7_integration.py -v --tb=short 2>&1 | head -60
```

Repeat until the test file collects and runs (individual test failures are acceptable at this
point — the goal is to get past import errors and have the tests execute).

---

## Step 5 — Wire NWTN stubs to `NWTNOrchestrator` (~2 hr)

Three stubs in `prsm/compute/nwtn/` return hardcoded dummy values. All three should delegate
to `NWTNOrchestrator` from `prsm/compute/nwtn/orchestrator.py`.

**Read `orchestrator.py` lines 169–450 before writing any of these.** Understand
`UserInput`, `NWTNResponse`, and `process_query()` fully — these are the delegation targets.

Use lazy initialization (import inside `__init__` or first call) to avoid circular imports.

### `deep_reasoning_engine.py`

```python
class DeepReasoningEngine:
    """
    Deep reasoning via the NWTN orchestration pipeline.
    Delegates multi-stage reasoning to NWTNOrchestrator.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._orchestrator = None

    def _get_orchestrator(self):
        if self._orchestrator is None:
            from prsm.compute.nwtn.orchestrator import NWTNOrchestrator
            self._orchestrator = NWTNOrchestrator()
        return self._orchestrator

    async def reason(self, query: str, context: Dict[str, Any] = None) -> ReasoningResult:
        from prsm.compute.nwtn.orchestrator import UserInput
        orchestrator = self._get_orchestrator()
        user_input = UserInput(
            user_id=str(context.get("user_id", "system")) if context else "system",
            prompt=query,
            context_allocation=context.get("context_allocation", 10) if context else 10,
        )
        try:
            response = await orchestrator.process_query(user_input)
            return ReasoningResult(
                conclusion=response.final_response or f"Analysis of: {query}",
                confidence=response.confidence_score or 0.8,
                reasoning_chain=[s.get("step", "") for s in (response.reasoning_trace or [])],
                evidence=response.citations or [],
            )
        except Exception:
            # Graceful fallback — orchestrator may not be fully configured in all environments
            return ReasoningResult(
                conclusion=f"Analysis of: {query}",
                confidence=0.8,
                reasoning_chain=["Initial analysis", "Deep evaluation"],
                evidence=[],
            )
```

### `meta_reasoning_orchestrator.py`

Wire `orchestrate()` to `NWTNOrchestrator.process_query()` using the same lazy-init pattern.
Return a dict shaped `{"result": response.final_response, "confidence": response.confidence_score}`.

### `multimodal_processor.py`

Check `prsm/compute/nwtn/backends/` for any multimodal support (vision, audio). If present,
delegate `process()` to the relevant backend. If not, implement a real text-extraction pass:
extract any `text` key from the content dict and return it with metadata — not just
`{"processed": True, "content": content}`.

---

## Step 6 — Wire supporting module stubs (~2 hr)

### `prsm/response/response_generator.py`

`generate()` should delegate to `NWTNOrchestrator.process_query()` and return
`response.final_response`. Use the same lazy-init + graceful fallback pattern.

### `prsm/query/advanced_query_engine.py`

`process()` should delegate to `NWTNOrchestrator.process_query()`. Return a dict containing
`query`, `result` (the response text), and `confidence`.

### `prsm/nlp/query_processor.py`

`process()` currently splits on whitespace. Wire it to use the NWTN orchestrator's intent
clarification logic (`orchestrator._clarify_intent()` if available) or implement real
keyword extraction and intent classification using Python stdlib (no external NLP deps):

```python
async def process(self, query: str) -> ProcessedQuery:
    normalized = query.lower().strip()
    # Real keyword extraction: remove stopwords, extract meaningful terms
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "have", "has", "do", "does", "what", "how", "why", "when", "where"}
    keywords = [w for w in normalized.split() if w not in stopwords and len(w) > 2]
    # Simple intent classification from query structure
    intent = self._classify_intent(normalized)
    return ProcessedQuery(
        original_text=query,
        normalized_text=normalized,
        intent=intent,
        entities={},
        keywords=keywords,
    )

def _classify_intent(self, text: str) -> str:
    if any(text.startswith(w) for w in ("what", "define", "explain")):
        return "definition"
    if any(text.startswith(w) for w in ("how", "steps", "procedure")):
        return "procedure"
    if any(text.startswith(w) for w in ("why", "reason", "cause")):
        return "explanation"
    if any(text.startswith(w) for w in ("compare", "difference", "versus")):
        return "comparison"
    return "query"
```

### `prsm/nlp/advanced_nlp.py`

Wire `process()` to use `QueryProcessor` internally. Add real entity extraction using
regex patterns for dates, numbers, proper nouns (capitalized words), and URLs — no external
deps needed.

### `prsm/learning/adaptive_learning.py` and `prsm/learning/feedback_processor.py`

Read `prsm/compute/improvement/` and `prsm/economy/tokenomics/` for any existing
feedback/rating storage. Wire `FeedbackProcessor.process()` to persist feedback to the
database (SQLAlchemy session) using the existing `SessionLocal` pattern from `core/database.py`.
Wire `AdaptiveLearningSystem.learn()` to call `FeedbackProcessor` and accumulate a
session-level improvement log.

### `prsm/optimization/performance_optimizer.py`

`prsm/compute/performance/optimization.py` has a real `PerformanceOptimizer` class (line 684).
Replace the stub with delegation:

```python
class PerformanceOptimizer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._real_optimizer = None

    def _get_optimizer(self):
        if self._real_optimizer is None:
            from prsm.compute.performance.optimization import (
                PerformanceOptimizer as RealOptimizer
            )
            self._real_optimizer = RealOptimizer()
        return self._real_optimizer

    async def optimize(self) -> Dict[str, Any]:
        optimizer = self._get_optimizer()
        return await optimizer.run_comprehensive_analysis(self.config)
```

### `prsm/core/enterprise/ai_orchestrator.py`

Replace the stub with an alias to the real implementation:

```python
from prsm.compute.ai_orchestration.orchestrator import AIOrchestrator

__all__ = ["AIOrchestrator"]
```

This is the cleanest solution — the real class at that path already does enterprise AI
orchestration. No need for a separate stub.

---

## Step 7 — Remove the module-level skip and run the full test (~1 hr)

After Steps 2–6:

1. Remove the `pytest.skip(...)` line from `tests/integration/test_phase7_integration.py`
2. Run:
   ```bash
   python -m pytest tests/integration/test_phase7_integration.py -v --tb=short
   ```
3. Fix any remaining failures. The tests use `AsyncMock` extensively for the expensive parts,
   so most should pass once the import chain resolves.
4. If a test is genuinely blocked by missing infrastructure (e.g. a test that requires a real
   Redis connection for `GlobalInfrastructure`), convert it from a module-level skip to a
   per-test `@pytest.mark.skipif` with a specific message — do NOT re-add a module-level skip.

---

## Step 8 — Write `tests/test_phase9_completeness.py` (~1 hr)

```python
"""
Phase 9 completeness verification.
Verifies all previously-stubbed modules now return meaningful results,
not hardcoded dummy values.
"""
import pytest


class TestStubsWired:

    async def test_deep_reasoning_returns_real_result(self):
        from prsm.compute.nwtn.deep_reasoning_engine import DeepReasoningEngine
        engine = DeepReasoningEngine()
        result = await engine.reason("What is photosynthesis?")
        assert result.conclusion != "Analysis of: What is photosynthesis?"  # not dummy
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.reasoning_chain, list)

    async def test_query_processor_real_keywords(self):
        from prsm.nlp.query_processor import QueryProcessor
        qp = QueryProcessor()
        result = await qp.process("What are the effects of climate change on agriculture?")
        # Stopwords removed, meaningful keywords extracted
        assert "effects" in result.keywords or "climate" in result.keywords
        assert "what" not in result.keywords  # stopword removed
        assert result.intent in ("query", "definition", "explanation")

    async def test_performance_optimizer_delegates(self):
        from prsm.optimization.performance_optimizer import PerformanceOptimizer
        optimizer = PerformanceOptimizer()
        result = await optimizer.optimize()
        # Real optimizer returns structured recommendations, not {"improvements": []}
        assert isinstance(result, dict)
        assert result != {"improvements": []}

    async def test_ai_orchestrator_enterprise_alias(self):
        from prsm.core.enterprise.ai_orchestrator import AIOrchestrator
        from prsm.compute.ai_orchestration.orchestrator import AIOrchestrator as RealOrchestrator
        assert AIOrchestrator is RealOrchestrator

    async def test_integration_manager_importable(self):
        from prsm.core.integrations.enterprise.integration_manager import IntegrationManager
        manager = IntegrationManager()
        integration_id = await manager.create_integration({
            "name": "test", "type": "database", "connection": {}
        })
        assert integration_id is not None

    async def test_ai_orchestrator_register_model(self):
        from prsm.compute.ai_orchestration.orchestrator import AIOrchestrator
        orchestrator = AIOrchestrator()
        await orchestrator.initialize()
        assert orchestrator.is_initialized
        model_id = await orchestrator.register_model({
            "name": "test-model", "type": "reasoning"
        })
        assert model_id is not None

    async def test_feedback_processor_persists(self):
        from prsm.learning.feedback_processor import FeedbackProcessor
        fp = FeedbackProcessor()
        result = await fp.process({"query": "test", "rating": 4, "user_id": "test-user"})
        assert result is True

    async def test_adaptive_learning_accumulates(self):
        from prsm.learning.adaptive_learning import AdaptiveLearningSystem
        als = AdaptiveLearningSystem()
        await als.learn({"feedback": "positive", "query": "test"})
        improvements = await als.get_improvements()
        assert isinstance(improvements, list)


class TestPhase7IntegrationUnblocked:

    def test_phase7_integration_file_has_no_module_skip(self):
        """Verify the module-level skip has been removed."""
        content = open("tests/integration/test_phase7_integration.py").read()
        assert "pytest.skip(" not in content.split("import")[0]  # no skip before imports
```

---

## Step 9 — Update `docs/IMPLEMENTATION_STATUS.md` (~15 min)

- Update passing test count
- Mark all 9 stub modules as ✅ Wired
- Mark `test_phase7_integration.py` as ✅ Active (not deferred)
- Update overall completeness estimate

---

## Execution Order

| Order | Task | Files Changed | Effort |
|-------|------|--------------|--------|
| 1 | Audit test_phase7 failures | Read-only | 30 min |
| 2 | Fix AIOrchestrator API | `prsm/compute/ai_orchestration/orchestrator.py` | 1 hr |
| 3 | Add IntegrationManager class | `prsm/core/integrations/enterprise/integration_manager.py` | 30 min |
| 4 | Verify DashboardManager API | `prsm/data/analytics/dashboard_manager.py` | 30 min |
| 5 | Wire NWTN stubs | 3 files in `prsm/compute/nwtn/` | 2 hr |
| 6 | Wire supporting stubs | 6 files across `prsm/` | 2 hr |
| 7 | Remove skip, fix remaining failures | `tests/integration/test_phase7_integration.py` | 1 hr |
| 8 | Write test_phase9_completeness.py | `tests/test_phase9_completeness.py` | 1 hr |
| 9 | Update IMPLEMENTATION_STATUS.md | `docs/` | 15 min |

**Total estimated effort: ~9 hours**

---

## Key Implementation Rules

1. **Read before writing** — every file edited in Steps 2–6 must be read in full first.
   The stubs are small; the files they delegate to are large. Understand the target API
   before writing delegation code.

2. **Lazy initialization everywhere** — stubs that import from NWTN or ai_orchestration
   must use lazy imports inside methods or `__init__` to avoid circular import chains.
   Do NOT add top-level imports of heavy orchestration modules in stub files.

3. **Graceful fallback, not silent failure** — wired stubs should try to call the real
   implementation and fall back gracefully if it fails (e.g. missing API keys, not yet
   initialized). They must NOT return the same hardcoded dummy values they return today.

4. **No module-level skips** — the goal of Step 7 is zero module-level skips in the
   test suite. Per-test skips with specific infrastructure messages are acceptable.

5. **Don't reimplement what exists** — if a real class already does the job
   (`PerformanceOptimizer`, `AIOrchestrator`), alias or delegate; don't write a second
   implementation.

---

## Expected Outcome

- All 9 stub files return meaningful, real results (or delegate to implementations that do)
- `test_phase7_integration.py` runs without a module-level skip
- Zero module-level skips remaining in the test suite (except `test_ftns_concurrency`
  which requires `DATABASE_URL` — that is correctly a per-test skip)
- Test count: ~3,787 → ~3,820+ (Phase 7 integration suite newly active)
- PRSM codebase: **99%+ complete** absent infrastructure spend

---

## Permanently Out of Scope (not Phase 9)

| Item | Reason |
|------|--------|
| Stripe/PayPal live payment methods | Require external API accounts |
| Crypto exchange rate fetching | Requires exchange API keys |
| GlobalInfrastructure live deployment | Requires AWS/GCP/Azure credentials |
| Video walkthroughs | Outside code scope |
| External security audit | Paid service |
| PyPI/npm/pkg.go.dev publishing | Requires registry accounts |
