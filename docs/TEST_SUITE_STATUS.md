# PRSM Test Suite — Mocking Infrastructure Reference

> **Snapshot date: 2026-02-16. The *numbers* in this doc (test counts, pass/fail rates, specific hanging-file names) are stale — v1.6 scope alignment (April 2026) deleted ~210K LoC and changed the test landscape. Run `pytest` against current code for current metrics.**
>
> **What remains current:** the **mocking infrastructure techniques** documented here (FakeRedis, FakeAsyncPGConnection, HTTP/LLM/time mocking patterns in `tests/conftest.py`, pytest timeout configuration in `pytest.ini`) are still load-bearing in the current codebase. Use this doc as a reference for how the mocking layer works, not as a current health dashboard.
>
> **Historical framing note:** the original title referred to "Phase 3 Completion" using pre-audit-gap-roadmap numbering. That usage does not match current "Phase 3" (Marketplace Matching Engine per [`2026-04-10-audit-gap-roadmap.md`](2026-04-10-audit-gap-roadmap.md)); the February 2026 "Phase 3" referred to a pre-v1.6 internal sprint scheme. Renamed here to avoid collision.
>
> **Related docs:**
> - [`SECURITY_HARDENING_CHECKLIST.md`](SECURITY_HARDENING_CHECKLIST.md) — §10 Security Testing (similar snapshot caveats apply)
> - [`2026-04-10-audit-gap-roadmap.md`](2026-04-10-audit-gap-roadmap.md) — for current test discipline per phase
> - [`GETTING_STARTED.md`](GETTING_STARTED.md) / [`quickstart.md`](quickstart.md) — for running tests locally

## Original snapshot date: 2026-02-16

## Summary

Comprehensive external service mocking has been implemented to allow the PRSM test suite to run in isolation without requiring Redis, PostgreSQL, or external APIs to be running. The mocking patterns below remain the authoritative reference for `tests/conftest.py` structure; specific pass/fail counts are point-in-time and not maintained.

## Changes Implemented

### 1. Global Test Timeout (pytest.ini)
- Set `timeout = 30` seconds (reduced from 60)
- Set `timeout_method = thread` for better cross-platform compatibility
- This causes hanging tests to fail with clear timeout errors

### 2. Comprehensive Mock Fixtures (tests/conftest.py)

#### Redis Mocking
- **FakeRedis class**: Full in-memory implementation supporting:
  - Basic operations: get, set, delete, exists, keys
  - Sorted sets: zadd, zremrangebyscore, zcard
  - Lists: lpush, rpush, lrange
  - Counters: incr, decr
  - Pipelines: Full pipeline support for atomic operations
  - Expiration: expire command support
- **Auto-use fixture**: Automatically mocks redis.asyncio.Redis, redis.Redis, and all connection methods
- **Compatibility**: Works with both sync and async Redis clients

#### PostgreSQL/Database Mocking
- **FakeAsyncPGConnection class**: Mock asyncpg connections
  - execute, fetch, fetchrow, fetchval operations
  - Transaction context managers
- **SQLAlchemy**: Mocked async engines and sessions
- **Auto-use fixture**: Automatically intercepts asyncpg.connect, asyncpg.create_pool, and SQLAlchemy engines

#### HTTP/API Mocking
- **aiohttp**: Mocked ClientSession with standard responses
- **httpx**: Mocked AsyncClient and sync Client
- **requests**: Mocked synchronous HTTP calls
- **Auto-use fixture**: All HTTP libraries return mock 200 OK responses
- **Response structure**: Includes status, json(), text, and content attributes

#### LLM Client Mocking
- **OpenAI**: Mocked AsyncOpenAI and sync OpenAI clients
  - chat.completions.create returns mock completions
  - Includes token usage tracking
- **Anthropic**: Mocked AsyncAnthropic and sync Anthropic clients
  - messages.create returns mock responses
- **Auto-use fixture**: Prevents actual API calls, returns canned responses

#### Time/Sleep Mocking
- **time.sleep**: Patched to not actually sleep (prevents test delays)
- **asyncio.sleep**: Patched to yield control but not wait
- **Auto-use fixture**: Applied globally to speed up tests

#### Early Connection Prevention
- **subprocess**: Mocked subprocess.run and subprocess.Popen
- **socket**: Mocked socket.socket
- **Session-level fixture**: Applied before test collection begins

### 3. Environment Variables
Set automatically in `setup_test_environment` fixture:
- `SKIP_REDIS_TESTS=true`
- `SKIP_POSTGRES_TESTS=true`
- `SKIP_INTEGRATION_TESTS=true`
- `PRSM_ENVIRONMENT=test`
- `PRSM_DATABASE_URL=sqlite:///:memory:`

## Test Results (Feb 2026 snapshot — stale)

> **Numbers below are February 2026 point-in-time measurements. Do not cite as current.** Run `pytest tests/unit -q` against current code for live state.

### Unit Tests (Feb 2026)
Successfully running with mocks:
- **tests/unit/**: 96 passed, 24 failed, 1 skipped in 3.43s
- **tests/test_configuration_management.py**: 31 passed, 4 failed in 3.92s
- **Single test example**: 1 passed in 3.42s

### Known Issues
1. **Full suite collection**: Some tests still appear to hang during collection phase
   - Likely due to module-level imports that connect to services before mocks can intercept
   - Specific test files around test_enhanced_ipfs.py, test_enhanced_models.py area
   - Investigation needed: Check for module-level socket/connection initialization

2. **Benchmark tests**: tests/benchmarks/test_performance_benchmarks.py
   - Contains intentional time.sleep() calls that are now mocked
   - Tests run but may not produce meaningful performance data
   - Consider: Separate real performance tests from unit tests

3. **Integration tests**: Many integration tests properly skip with env vars
   - Some may still try to import modules with external dependencies
   - Solution: Add pytest markers to skip entire files or use lazy imports

## Recommendations

### Immediate Next Steps
1. **Identify hanging imports**: Run pytest with `-vv` and `--collect-only` to see where collection hangs
2. **Add lazy imports**: For test files that import heavy modules, use lazy imports or conditional imports
3. **Separate test categories**: 
   - Unit tests (fully mocked) → fast, always run
   - Integration tests (skip by default) → require services
   - Performance tests (separate suite) → actual benchmarks

### Future Improvements
1. **Module-level mocking**: Consider using pytest_plugins or conftest hooks to mock at import time
2. **Test categories in pytest.ini**: Define clear markers for unit/integration/performance
3. **CI/CD integration**: Run unit tests in PR checks, integration tests on merge
4. **Docker-based integration tests**: For tests that need real services, use docker-compose

## Files Modified
- `pytest.ini`: Updated timeout configuration
- `tests/conftest.py`: Added 400+ lines of comprehensive mocking infrastructure

## Commit
```
commit e9f700b
Author: prsm-coder
Date: 2026-02-16

Add comprehensive external service mocking and test timeouts
```

## Success Metrics
✅ Unit tests run without external dependencies
✅ Tests complete in seconds instead of hanging indefinitely  
✅ Mock infrastructure is comprehensive and reusable
✅ Environment variables properly isolate test execution
⚠️ Full suite collection still has issues (needs investigation)
⚠️ Some integration tests may need additional isolation

## Conclusion (February 2026 snapshot)
The mocking infrastructure work documented here successfully created robust external-service mocks (Redis, PostgreSQL, HTTP, LLM clients, time) allowing most PRSM tests to run in isolation. While some tests had collection/execution issues at the time of writing, the foundation was solid and unit tests were running successfully with proper timeouts and mocking.

**Post-snapshot note (2026-04-16):** the v1.6 scope-alignment sprint (April 2026) deleted ~210K LoC including the NWTN orchestrator, teachers framework, and distillation pipeline — many of which carried their own test surfaces that no longer exist. The mocking infrastructure documented here still governs current `tests/conftest.py`, but the specific test files referenced as problematic (e.g., `test_enhanced_ipfs.py`) are no longer present in the repo. Treat this document as a reference for **how** the test mocking layer works, not as a current health dashboard.
