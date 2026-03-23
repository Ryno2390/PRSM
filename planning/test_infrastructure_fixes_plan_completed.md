# Test Infrastructure Fixes — Implementation Plan

## Overview

With stub elimination essentially complete, this plan fixes the remaining test failures
that are caused by test fixture/infrastructure issues rather than production code bugs.

Current state: **15 failed, 304 passed, 50 skipped, 5 errors** in integration suite.

All 20 failing/erroring tests fall into three distinct root causes:

| Cause | Tests affected | Fix location |
|-------|---------------|--------------|
| DB schema not initialized before `AtomicFTNSService` | 12 (marketplace concurrency) | `test_marketplace_concurrency.py` fixtures |
| Auth mock returns dict, API expects `User` object | 1 (concurrent API requests) | `test_endpoint_integration.py` test |
| WebSocket server not running in CI | 1 (bootstrap connectivity) | skip marker |

After these fixes, the expected result is **~316 passed, 50 skipped, 1 skipped (websocket), 0 failed**.

---

## Phase 1: Marketplace Concurrency — DB Schema Init

**File:** `tests/integration/test_marketplace_concurrency.py`

### Root Cause

`AtomicFTNSService` calls `get_async_session()` → `get_async_engine()` which uses a global
`async_engine` singleton (default: `sqlite:///prsm`). The test environment never calls
`Base.metadata.create_all()` on this engine, so every DB operation hits
`sqlite3.OperationalError: no such table: ftns_balances`.

The conftest.py at `tests/conftest.py` already has a `test_async_engine` session fixture
(line 403) that creates a fresh `sqlite+aiosqlite:///:memory:` engine and runs `create_all`.
The marketplace tests simply never use it.

### Fix

Replace the `ftns_service` class fixture in both `TestMarketplaceConcurrency` and
`TestMarketplaceIntegration` with one that creates and injects a fresh per-test in-memory DB:

```python
@pytest.fixture
async def ftns_service(self):
    """Create AtomicFTNSService backed by a fresh in-memory DB with schema"""
    import prsm.core.database as db_module
    from sqlalchemy import JSON
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.dialects.postgresql import JSONB
    from prsm.core.database import Base

    # Build a fresh isolated in-memory engine for this test
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    # Replace JSONB with JSON for SQLite compatibility
    for table in Base.metadata.tables.values():
        for column in table.columns:
            if isinstance(column.type, JSONB):
                column.type = JSON()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Inject the fresh engine into the global singleton
    original_engine = db_module.async_engine
    db_module.async_engine = engine

    try:
        service = AtomicFTNSService()
        await service.initialize()
        yield service
    finally:
        db_module.async_engine = original_engine
        await engine.dispose()
```

**Important:** This fixture replaces the one in BOTH `TestMarketplaceConcurrency` and
`TestMarketplaceIntegration`. Read the full file to confirm how many classes have their
own `ftns_service` fixture before replacing.

The `funded_user` fixture depends on `ftns_service`, so it does not need changes —
it will automatically use the DB-backed service.

---

## Phase 2: Concurrent API Requests — Auth Mock Fix

**File:** `tests/integration/api/test_endpoint_integration.py`

### Root Cause

`test_concurrent_api_requests` (line 556) patches `AuthManager.get_current_user` to
return a plain dict `{"valid": True, "user_id": "perf_user", "user_role": "researcher"}`.

But `ftns_api.py` imports and `Depends`-injects `prsm.interface.auth.get_current_user`
— a FastAPI dependency, not `AuthManager.get_current_user` directly. The dependency
resolves to a `User` object (from `prsm.core.models.User`), and the endpoint code calls
`current_user.id` — failing because a dict has no `.id` attribute.

### Fix

Use `app.dependency_overrides` to override the `get_current_user` FastAPI dependency
for the duration of the test, returning a proper `User`-like object:

```python
async def test_concurrent_api_requests(self, async_test_client):
    """Test API performance under concurrent load"""
    if async_test_client is None:
        pytest.skip("Async test client not available")

    from prsm.interface.auth import get_current_user
    from unittest.mock import MagicMock

    # Build a mock user object with the .id attribute the endpoint expects
    mock_user = MagicMock()
    mock_user.id = "perf_user"
    mock_user.user_id = "perf_user"
    mock_user.role = "researcher"

    async def override_get_current_user():
        return mock_user

    # Also mock the balance query to avoid DB dependency
    with patch('prsm.core.database.FTNSQueries.get_user_balance') as mock_balance:
        mock_balance.return_value = {
            "total_balance": Decimal("1000.0"),
            "available_balance": Decimal("1000.0"),
            "reserved_balance": Decimal("0.0")
        }

        # Override the FastAPI dependency
        async_test_client.app.dependency_overrides[get_current_user] = override_get_current_user

        try:
            auth_token = "performance_test_token"
            headers = {"Authorization": f"Bearer {auth_token}"}

            async def make_balance_request():
                start_time = asyncio.get_event_loop().time()
                response = await async_test_client.get("/api/v1/ftns/balance", headers=headers)
                end_time = asyncio.get_event_loop().time()
                return {
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "success": response.status_code == 200
                }

            tasks = [make_balance_request() for _ in range(20)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_requests = [r for r in results if isinstance(r, dict) and r["success"]]
            # rest of existing assertions...

        finally:
            async_test_client.app.dependency_overrides.pop(get_current_user, None)
```

**Read the full test method first** (it's around 60 lines). Replace only the mock setup and
teardown at the top/bottom — preserve all the existing assertions about response time,
success counts, etc.

**Note:** `dependency_overrides` is the correct pattern for FastAPI dependency injection
in tests — it bypasses the entire auth middleware cleanly.

---

## Phase 3: WebSocket Bootstrap Test — Skip Marker

**File:** `tests/integration/test_bootstrap_connectivity.py`

### Root Cause

`test_websocket_port_is_open` asserts that port 8765 is open on localhost. This requires
a live WebSocket server to be running — it's an infrastructure check, not a unit test.

### Fix

Add a `pytest.mark.skip` or `pytest.mark.skipif` to this test:

```python
@pytest.mark.skip(reason="Requires live WebSocket server on port 8765 — run manually")
def test_websocket_port_is_open(self):
    ...
```

Read the full test class first to see if there is an existing skip pattern used elsewhere
in the class (some tests may already have `pytest.skip()` calls you should match).

---

## Phase 4: Mesa-Dependent Economics Tests

**File:** `tests/integration/test_economics_and_config.py`

The 10 skipped tests in this file are all `TestAgentBasedModel*` tests that require the
`mesa` package (Mesa agent-based modeling framework). They currently skip silently because
`pytest.importorskip("mesa")` inside the fixtures causes the skip.

There are two options:

**Option A (recommended):** Install Mesa as an optional dev dependency so the tests run:
```
pip install mesa
```
Then verify all 10 tests pass. If they do, add `mesa` to `requirements-dev.txt` or the
test dependency section of `pyproject.toml` (whichever the project uses).

**Option B:** Leave them as skips with a clear explanatory message.

Read `requirements-dev.txt` and `pyproject.toml` to determine which file controls test
dependencies before adding `mesa`.

---

## File Checklist

- [ ] `tests/integration/test_marketplace_concurrency.py` — fix `ftns_service` fixture(s)
- [ ] `tests/integration/api/test_endpoint_integration.py` — fix concurrent test auth mock
- [ ] `tests/integration/test_bootstrap_connectivity.py` — add skip marker
- [ ] `requirements-dev.txt` or `pyproject.toml` — add `mesa` (Phase 4, optional)

**Expected outcome:** ~316 passed (up from 304), 0 failed (down from 15), 5 errors → 0

---

## Notes

- The `ftns_service` fixture in `test_marketplace_concurrency.py` creates a per-test
  in-memory DB. This is intentional — each concurrency test needs an isolated clean slate
  so that concurrent balance operations from test N don't bleed into test N+1.
- Do NOT use the session-scoped `test_async_engine` from conftest for this — shared state
  across concurrency tests would cause non-deterministic race conditions between tests.
- After resetting `db_module.async_engine`, any subsequent DB operations in the test that
  go through `get_async_engine()` will get a new engine. The `finally` block restores the
  original value (which is likely `None` since no prior test has initialized it).
- `dependency_overrides` must be cleaned up in a `finally` block or the override will leak
  into subsequent tests that share the same `async_test_client` (which may be session-scoped).

---

## Implementation Summary

*(To be filled in by you upon completion — rename file to `_completed.md`)*
