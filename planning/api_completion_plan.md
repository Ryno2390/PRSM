# API Completion & Stub Elimination — Implementation Plan

## Overview

This plan addresses the 21 remaining stub functions across the PRSM codebase. They fall into five categories:

1. **Tokenomics/Anti-hoarding** — 3 methods returning empty placeholders that break FTNS circulation enforcement
2. **API endpoints** — 6 endpoints returning hardcoded fake data instead of real DB reads
3. **Alert handlers** — 4 `pass` stubs for Slack/email notifications
4. **Load testing infrastructure** — 3 methods returning fake metrics
5. **Workflow rollback** — 1 mock state restoration
6. **Jupyter kernel execution** — 1 simulated code runner

All are production code. None are test-infrastructure stubs.

---

## Phase 1: Anti-Hoarding FTNS Integration

**File:** `prsm/economy/tokenomics/anti_hoarding_engine.py`

The anti-hoarding engine penalizes dormant large holders to encourage FTNS circulation. Currently three data-access methods return empty/None, making the engine a no-op.

### 1a — `_get_user_transactions(user_id: str, since: float) -> List[dict]`

Currently: `return []`

Implementation:
1. Import `get_async_session` from `prsm.core.database`.
2. Query `FTNSTransaction` (the existing ORM model in `database.py`) for rows where `sender_id == user_id OR recipient_id == user_id` AND `created_at >= since`.
3. Return list of dicts with fields: `transaction_id`, `amount`, `timestamp`, `type` (send/receive), `counterparty_id`.
4. Wrap in try/except; on DB error log and return `[]` (graceful degradation).

```python
async def _get_user_transactions(self, user_id: str, since: float) -> List[dict]:
    try:
        async with get_async_session() as session:
            from sqlalchemy import select, or_
            stmt = (
                select(FTNSTransactionModel)
                .where(
                    or_(
                        FTNSTransactionModel.sender_id == user_id,
                        FTNSTransactionModel.recipient_id == user_id,
                    ),
                    FTNSTransactionModel.created_at >= since,
                )
                .order_by(FTNSTransactionModel.created_at.desc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                {
                    "transaction_id": str(r.transaction_id),
                    "amount": float(r.amount),
                    "timestamp": r.created_at,
                    "type": "send" if r.sender_id == user_id else "receive",
                    "counterparty_id": r.recipient_id if r.sender_id == user_id else r.sender_id,
                }
                for r in rows
            ]
    except Exception as e:
        logger.warning("failed_to_get_user_transactions", user_id=user_id, error=str(e))
        return []
```

Check what the `FTNSTransaction` model is actually named in `database.py` and use the correct import. If the table is named differently (e.g. `FTNSTransactionModel`), use that. The key columns to look for are `sender_id`, `recipient_id`, `amount`, and a timestamp column.

### 1b — `_get_users_with_balances(min_balance: Decimal) -> List[str]`

Currently: `return []`

Implementation:
1. Query `FTNSWalletModel` (or equivalent ORM model in `database.py`) for `balance >= min_balance` and `is_active = True`.
2. Return list of `user_id` strings.
3. Apply a reasonable limit (e.g., 10,000) to prevent full-table scan explosions.

```python
async def _get_users_with_balances(self, min_balance: Decimal) -> List[str]:
    try:
        async with get_async_session() as session:
            stmt = (
                select(FTNSWalletModel.user_id)
                .where(FTNSWalletModel.balance >= float(min_balance))
                .limit(10000)
            )
            result = await session.execute(stmt)
            return [row[0] for row in result.all()]
    except Exception as e:
        logger.warning("failed_to_get_users_with_balances", error=str(e))
        return []
```

### 1c — `_get_user_creation_date(user_id: str) -> Optional[float]`

Currently: `return None`

Implementation:
1. Query `UserProfileModel` (or `FTNSWalletModel.created_at`, whichever stores account creation timestamp).
2. Return `created_at` as a float (Unix timestamp).
3. On not-found, return `None` (caller already handles this).

```python
async def _get_user_creation_date(self, user_id: str) -> Optional[float]:
    try:
        async with get_async_session() as session:
            stmt = select(FTNSWalletModel.created_at).where(
                FTNSWalletModel.user_id == user_id
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return float(row) if row is not None else None
    except Exception as e:
        logger.warning("failed_to_get_user_creation_date", user_id=user_id, error=str(e))
        return None
```

**Note:** Before implementing, read the actual ORM model names in `database.py` for FTNS transactions and wallets. Search for `__tablename__ = "ftns"` to confirm.

---

## Phase 2: API Endpoints — Replace Hardcoded Data

### 2a — `reputation_api.py`: `_get_leaderboard_data()` and `get_reputation_analytics()`

**File:** `prsm/interface/api/reputation_api.py`

**`_get_leaderboard_data(limit: int, offset: int) -> List[dict]`** (line ~646)

Currently returns 3 hardcoded sample dicts.

Implementation:
1. Query `ReputationScoreModel` (or equivalent) ordered by `score DESC`, with `limit` and `offset`.
2. JOIN with user profile to get display name.
3. Return list of dicts with `rank`, `user_id`, `username`, `score`, `tier`, `change_30d`.
4. If no reputation table exists yet in the ORM, fall back to the FTNS wallet model sorted by balance as a proxy reputation score.

```python
async def _get_leaderboard_data(limit: int = 50, offset: int = 0) -> List[dict]:
    async with get_async_session() as session:
        # Try reputation scores table first
        try:
            stmt = (
                select(ReputationScoreModel)
                .order_by(ReputationScoreModel.overall_score.desc())
                .offset(offset)
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                {
                    "rank": offset + i + 1,
                    "user_id": r.user_id,
                    "username": r.user_id[:8] + "...",
                    "score": float(r.overall_score),
                    "tier": _score_to_tier(r.overall_score),
                    "change_30d": 0.0,  # Computed separately if historical data exists
                }
                for i, r in enumerate(rows)
            ]
        except Exception:
            return []
```

**`get_reputation_analytics()` → endpoint** (line ~580)

Currently returns hardcoded fake growth, tier distribution, etc.

Implementation:
1. Count total users with reputation records.
2. Group by tier to get distribution.
3. Compute 30-day rolling average for growth rate by counting new reputation records created in last 30 days vs. prior 30 days.
4. Return real values; keep the same response schema so callers aren't broken.

### 2b — `distillation_api.py`: `get_distillation_results()` and `cancel_distillation_job()`

**File:** `prsm/interface/api/distillation_api.py`

**`get_distillation_results(job_id: str)` (line ~398)**

Currently returns hardcoded performance metrics.

Implementation:
1. Query `DistillationResultModel` by `job_id` (the model added in Phase 4 of the previous plan).
2. If no result found, return 404.
3. Serialize the row using real field values: `accuracy_score`, `compression_ratio`, `training_loss`, `validation_loss`, `tokens_used`, `ftns_cost`.

```python
@router.get("/{job_id}/results")
async def get_distillation_results(job_id: str, session=Depends(get_db_session)):
    result = await session.execute(
        select(DistillationResultModel).where(DistillationResultModel.job_id == job_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="No results found for this job")
    return {
        "job_id": job_id,
        "accuracy_score": row.accuracy_score,
        "compression_ratio": row.compression_ratio,
        "training_loss": row.training_loss,
        "validation_loss": row.validation_loss,
        "tokens_used": row.tokens_used,
        "ftns_cost": float(row.ftns_cost),
        "created_at": row.created_at,
    }
```

**`cancel_distillation_job(job_id: str)` (line ~486)**

Currently placeholder with no actual cancellation.

Implementation:
1. Load job from `DistillationJobModel`.
2. If job status is `"completed"` or `"failed"`, return 409 Conflict.
3. Update status to `"cancelled"` in DB.
4. If the `AutomatedDistillationEngine` singleton is accessible (check for `app.state.distillation_engine`), call `engine.cancel_job(job_id)` if that method exists; otherwise just set the DB status.
5. Return `{"job_id": job_id, "status": "cancelled"}`.

### 2c — `monitoring_api.py`: `get_trace_details()`

**File:** `prsm/interface/api/monitoring_api.py`

**`get_trace_details(trace_id: str)` (line ~513)**

Currently returns hardcoded sample trace with fake spans.

Implementation:
1. Check if an OpenTelemetry or structlog trace store is available via `app.state`.
2. If structlog is the only store: query any `monitoring_events` or `audit_log` table for records matching `trace_id` in their metadata JSON.
3. Return actual spans if found, or return 404 if the trace ID doesn't exist.
4. Schema: `{"trace_id": ..., "spans": [...], "duration_ms": ..., "status": ...}`.

If no trace store exists in the DB, implement a minimal in-memory ring buffer: add a `TracingBuffer` singleton (dict of `trace_id → List[span]`) that endpoint handlers can write to when executing, and `get_trace_details` reads from. Store at most 1,000 traces (evict oldest). This is honest behavior — not fake.

### 2d — `recommendation_api.py`: `get_recommendation_analytics()`

**File:** `prsm/interface/api/recommendation_api.py`

**`get_recommendation_analytics()` (line ~497)**

Currently returns hardcoded analytics.

Implementation:
1. Query `RecommendationModel` or equivalent table for:
   - Total recommendations generated (COUNT)
   - Acceptance rate (accepted / total, if acceptance is tracked)
   - Average confidence score (AVG)
   - Distribution by recommendation type
2. If no dedicated recommendations table exists, query the teacher model table for usage counts as a proxy.
3. Return real aggregated values. If no data yet, return zeros (not fake numbers).

### 2e — `auth_api.py`: `list_users()`

**File:** `prsm/interface/api/auth_api.py`

**`list_users()` (line ~373)**

Currently returns only the current user as a single-element example.

Implementation:
1. Add admin check: if current user is not an admin (check `user.role == "admin"` or a flag), return 403.
2. Query `UserProfileModel` (or the model holding user accounts) with pagination.
3. Return list of user summaries: `user_id`, `username`, `email` (masked), `created_at`, `last_login`, `is_active`.
4. Never return passwords or sensitive tokens.

### 2f — `teams_api.py`: `list_team_tasks()` and `deposit_ftns()`

**File:** `prsm/interface/api/teams_api.py`

**`list_team_tasks()` (line ~395)**

Currently: returns placeholder message.

Implementation:
1. Query a tasks or sessions table filtered by `team_id`.
2. Return paginated list of task records.
3. If no tasks table exists for teams, query `ArchitectTaskModel` filtered by `session_id` where sessions belong to the team.

**`deposit_ftns()` (line ~412)**

Currently: comment about wallet retrieval not yet implemented.

Implementation:
1. Look up the `TeamFTNSWalletModel` by `team_id` (this model exists — it was added with the teams tables in an earlier migration).
2. Check the depositing user has sufficient balance in their personal wallet.
3. Deduct from user wallet, credit to team wallet, record a transaction in `FTNSTransactionModel`.
4. Return updated team wallet balance.
5. Use a DB transaction (atomic) — if either update fails, roll back both.

---

## Phase 3: Alert Handlers

**Files:**
- `prsm/compute/performance/db_monitoring.py` (lines ~732–741)
- `prsm/compute/performance/task_monitor.py` (lines ~791–800)

Both files have matching stub pairs: `slack_alert_handler()` and `email_alert_handler()`.

### 3a — Slack Alert Handler

```python
async def slack_alert_handler(alert: AlertEvent) -> None:
    """Send alert to Slack via webhook URL from settings."""
    webhook_url = getattr(settings, 'slack_webhook_url', None) or os.environ.get('SLACK_WEBHOOK_URL')
    if not webhook_url:
        logger.debug("slack_webhook_not_configured", alert_id=alert.alert_id)
        return

    color = {"critical": "#FF0000", "warning": "#FFA500", "info": "#36A64F"}.get(
        alert.severity.lower(), "#808080"
    )
    payload = {
        "attachments": [{
            "color": color,
            "title": f"PRSM Alert: {alert.alert_type}",
            "text": alert.message,
            "fields": [
                {"title": "Severity", "value": alert.severity, "short": True},
                {"title": "Component", "value": getattr(alert, 'component', 'system'), "short": True},
                {"title": "Time", "value": datetime.utcfromtimestamp(alert.timestamp).isoformat(), "short": False},
            ],
            "footer": "PRSM Monitoring",
        }]
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(webhook_url, json=payload)
            resp.raise_for_status()
        logger.info("slack_alert_sent", alert_id=alert.alert_id)
    except Exception as e:
        logger.error("slack_alert_failed", alert_id=alert.alert_id, error=str(e))
```

Import `httpx` at top of file; it's already a project dependency.

Graceful: if `SLACK_WEBHOOK_URL` is not configured, log at DEBUG level and return — don't raise. Operators who want Slack alerts must configure the env var.

### 3b — Email Alert Handler

```python
async def email_alert_handler(alert: AlertEvent) -> None:
    """Send alert email via SMTP settings from environment."""
    smtp_host = os.environ.get('ALERT_SMTP_HOST')
    smtp_port = int(os.environ.get('ALERT_SMTP_PORT', '587'))
    smtp_user = os.environ.get('ALERT_SMTP_USER')
    smtp_pass = os.environ.get('ALERT_SMTP_PASS')
    alert_email = os.environ.get('ALERT_EMAIL_TO')

    if not all([smtp_host, smtp_user, smtp_pass, alert_email]):
        logger.debug("email_alert_not_configured", alert_id=alert.alert_id)
        return

    subject = f"[PRSM {alert.severity.upper()}] {alert.alert_type}"
    body = f"""PRSM Alert Notification

Alert Type: {alert.alert_type}
Severity:   {alert.severity}
Time:       {datetime.utcfromtimestamp(alert.timestamp).isoformat()}
Message:    {alert.message}

This is an automated notification from the PRSM monitoring system.
"""

    import smtplib
    from email.mime.text import MIMEText
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = smtp_user
    msg['To'] = alert_email

    try:
        # Run blocking SMTP in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _send_smtp, smtp_host, smtp_port, smtp_user, smtp_pass, msg)
        logger.info("email_alert_sent", alert_id=alert.alert_id, to=alert_email)
    except Exception as e:
        logger.error("email_alert_failed", alert_id=alert.alert_id, error=str(e))


def _send_smtp(host, port, user, password, msg):
    with smtplib.SMTP(host, port) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)
```

Same pattern in both files. Both handlers must first check `AlertEvent` field names — inspect the dataclass in each file before implementing to confirm field names like `alert_id`, `severity`, `message`, `timestamp`, `alert_type`.

---

## Phase 4: Load Testing Infrastructure

**File:** `prsm/compute/performance/load_testing.py`

### 4a — `_test_database_performance()` (line ~500)

Currently returns placeholder `{"queries_per_second": 0, "latency_p99": 0}`.

Implementation:
1. Run a series of timed benchmark queries against the configured DB:
   - INSERT a temp record (measure write latency)
   - SELECT by primary key (measure read latency)
   - COUNT(*) on a large table (measure scan latency)
   - DELETE the temp record
2. Collect 10 iterations, compute p50/p95/p99 latencies, and queries-per-second.
3. Use `asyncio.gather` to run concurrent queries for the throughput test (default concurrency: 10).
4. Clean up any inserted test records in a `finally` block.

```python
async def _test_database_performance(self) -> dict:
    from prsm.core.database import get_async_session
    latencies = []
    for _ in range(10):
        start = time.perf_counter()
        async with get_async_session() as session:
            await session.execute(text("SELECT 1"))
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    latencies.sort()
    return {
        "queries_per_second": round(1000 / (sum(latencies) / len(latencies)), 1),
        "latency_p50_ms": latencies[4],
        "latency_p95_ms": latencies[9],
        "latency_p99_ms": latencies[9],  # With 10 samples, p99 ≈ max
        "sample_count": len(latencies),
    }
```

### 4b — `_test_ipfs_storage()` (line ~518)

Currently returns placeholder zeros.

Implementation:
1. Import `IPFSClient` from `prsm.compute.spine.ipfs_client` (already exists in codebase).
2. Create a small test payload (1 KB of random bytes).
3. Time `ipfs_client.add(payload)` → get CID.
4. Time `ipfs_client.cat(cid)` → retrieve.
5. Verify retrieved content matches original.
6. Delete the test pin.
7. Return `{"upload_ms": ..., "download_ms": ..., "verified": True/False}`.
8. On any error (IPFS not available), return `{"available": False, "error": str(e)}`.

### 4c — `_test_ml_pipeline()` (line ~534)

Currently returns placeholder zeros.

Implementation:
1. Select the smallest available model from the model registry (or use a stub tokenizer if no models are registered).
2. Run a minimal forward pass: tokenize a short string, run inference, measure wall-clock time.
3. Return `{"inference_ms": ..., "tokens_per_second": ..., "model_id": ...}`.
4. If no model is available, return `{"available": False, "reason": "no_models_registered"}`.

---

## Phase 5: Workflow Rollback Restoration

**File:** `prsm/compute/scheduling/workflow_rollback.py`

**`_restore_workflow_state(workflow_id: str, checkpoint: dict) -> bool`** (line ~554)

Currently logs "Restoring workflow state" but does nothing else; returns True as if successful.

Implementation:
1. Extract `checkpoint["component_states"]` — a dict of `component_name → serialized_state`.
2. For each component in `checkpoint["active_components"]`:
   - Look up the live component instance from `self.workflow_registry[workflow_id]` or `app.state.components`.
   - Call `component.restore_state(serialized_state)` if the method exists.
   - Otherwise, call `setattr` on public attributes matching keys in `serialized_state`.
3. Update `self.workflow_state[workflow_id]` with the checkpoint's `workflow_state` dict.
4. Set `workflow.current_step = checkpoint.get("current_step", 0)`.
5. Log each restored component: `logger.info("component_restored", component=name, workflow_id=workflow_id)`.
6. On any exception during restore, log error and return `False`.
7. Return `True` only if all components restored successfully.

Read the `WorkflowCheckpoint` dataclass fields before implementing to ensure the field names (`component_states`, `active_components`, `workflow_state`, `current_step`) match what's actually stored.

---

## Phase 6: Jupyter Kernel Execution

**File:** `prsm/compute/collaboration/jupyter/jupyter_collaboration.py`

**`_execute_code(session_id: str, code: str) -> dict`** (line ~247)

Currently simulates execution with time.sleep and fake output.

Implementation:
1. Check if `jupyter_client` package is available (`try: import jupyter_client`).
2. If available:
   - Look up the kernel for `session_id` in `self.kernels` dict (this dict likely already exists in the class).
   - If no kernel, start one: `km = jupyter_client.KernelManager(); km.start_kernel(); kc = km.client()`.
   - Store in `self.kernels[session_id] = (km, kc)`.
   - Execute: `msg_id = kc.execute(code)`.
   - Wait for output with timeout (30s): collect `stream`, `execute_result`, and `error` messages.
   - Return `{"output": collected_output, "error": error_text, "execution_count": n, "success": error is None}`.
3. If `jupyter_client` not available:
   - Fall back to `exec()` in a sandboxed local namespace.
   - Capture stdout via `io.StringIO` redirect.
   - Return `{"output": captured_stdout, "error": None, "execution_count": 0, "success": True}`.
4. Always time the execution and include `execution_time_ms` in the response.
5. On timeout: return `{"error": "execution_timeout", "success": False}`.

The fallback (`exec`) is honest behavior for dev/test environments where Jupyter is not installed. The real path uses `jupyter_client` which is an optional dependency. Add `jupyter_client` to the `optional-dependencies` section of `pyproject.toml` under a `[jupyter]` extra.

---

## Phase 7: Cache Analytics

**File:** `prsm/compute/performance/caching.py`

**`get_cache_analytics()` (line ~399)**

Currently returns simulated analytics with `random.random()` values.

Implementation:
1. The `CacheManager` class should already track `hits`, `misses`, and `evictions` as in-memory counters. If not, add `self._hits = 0`, `self._misses = 0`, `self._evictions = 0` to `__init__` and increment them in `get()`, `set()`, and `evict()` respectively.
2. Compute real stats:
   ```python
   total = self._hits + self._misses
   hit_rate = self._hits / total if total > 0 else 0.0
   ```
3. Return `{"hit_rate": hit_rate, "total_requests": total, "hits": self._hits, "misses": self._misses, "evictions": self._evictions, "cache_size": len(self._cache)}`.
4. Remove ALL `random.random()` calls — never use random in analytics.

---

## Phase 8: Integration Tests

**File:** `tests/integration/test_api_completion.py`

Write a new test file covering the newly implemented stubs.

### `TestAntiHoardingFTNS`
- `test_get_user_transactions_returns_empty_for_unknown_user` — unknown user_id → `[]`
- `test_get_user_transactions_filters_by_since` — insert 2 txns; one before `since`, one after → only 1 returned
- `test_get_users_with_balances_filters_by_min` — insert wallets with balances 100, 50, 10; min=60 → only 1 returned
- `test_get_user_creation_date_returns_none_for_unknown` — unknown user → `None`
- `test_get_user_creation_date_returns_timestamp` — insert wallet record → correct float returned

### `TestDistillationAPIEndpoints`
- `test_get_results_returns_404_for_missing_job` — GET `/distillation/{unknown_id}/results` → 404
- `test_get_results_returns_real_data` — insert result row, GET → real field values (not hardcoded)
- `test_cancel_job_updates_status_to_cancelled` — insert pending job, POST cancel → status is `"cancelled"` in DB
- `test_cancel_completed_job_returns_409` — insert completed job, POST cancel → 409

### `TestAlertHandlers`
- `test_slack_alert_skipped_without_webhook` — no env var → no error, debug log only
- `test_slack_alert_sends_correct_payload` — mock httpx, set env var → payload has `attachments[0].title`
- `test_email_alert_skipped_without_smtp_config` — no env vars → no error
- `test_email_alert_sends_via_smtp` — mock `smtplib.SMTP`, set env vars → `send_message` called once

### `TestCacheAnalytics`
- `test_hit_rate_zero_with_no_requests` — fresh cache → `hit_rate == 0.0`
- `test_hit_rate_computed_from_real_counters` — set key, get twice (1 miss on set, 2 hits) → `hit_rate == 0.67 ±0.01`
- `test_no_random_values_in_analytics` — call `get_cache_analytics()` 5 times → results are deterministic (same values each call)

### `TestLoadTesting`
- `test_database_performance_returns_real_latency` — real async DB → `queries_per_second > 0` and `latency_p50_ms > 0`
- `test_ipfs_unavailable_returns_error_dict` — mock IPFS to raise ConnectionError → `{"available": False}` returned

All tests: `@pytest.mark.asyncio`, real SQLite via `tmp_path`, mock patching for external services (Slack, SMTP, IPFS). No hardcoded assertions against specific numeric values — use `> 0`, `>= 0`, `isinstance(x, float)` etc.

---

## File Checklist

| File | Action |
|------|--------|
| `prsm/economy/tokenomics/anti_hoarding_engine.py` | MODIFY — 3 DB-backed methods |
| `prsm/interface/api/reputation_api.py` | MODIFY — `_get_leaderboard_data`, `get_reputation_analytics` |
| `prsm/interface/api/distillation_api.py` | MODIFY — `get_distillation_results`, `cancel_distillation_job` |
| `prsm/interface/api/monitoring_api.py` | MODIFY — `get_trace_details` |
| `prsm/interface/api/recommendation_api.py` | MODIFY — `get_recommendation_analytics` |
| `prsm/interface/api/auth_api.py` | MODIFY — `list_users` |
| `prsm/interface/api/teams_api.py` | MODIFY — `list_team_tasks`, `deposit_ftns` |
| `prsm/compute/performance/db_monitoring.py` | MODIFY — `slack_alert_handler`, `email_alert_handler` |
| `prsm/compute/performance/task_monitor.py` | MODIFY — `slack_alert_handler`, `email_alert_handler` |
| `prsm/compute/performance/load_testing.py` | MODIFY — 3 benchmark methods |
| `prsm/compute/performance/caching.py` | MODIFY — `get_cache_analytics`, add hit/miss counters |
| `prsm/compute/scheduling/workflow_rollback.py` | MODIFY — `_restore_workflow_state` |
| `prsm/compute/collaboration/jupyter/jupyter_collaboration.py` | MODIFY — `_execute_code` |
| `tests/integration/test_api_completion.py` | CREATE |

---

## Implementation Notes

- **Read every file before modifying.** Especially check field names on existing dataclasses and ORM models before assuming names from this plan.
- **No mocks in production paths.** Every implementation must use real DB queries, real HTTP calls, or real exec — not simulations.
- **Graceful degradation for external services.** Slack, SMTP, IPFS, and Jupyter kernel failures must log an error and return cleanly — never propagate to the caller.
- **Atomic DB operations.** The `deposit_ftns` method must deduct and credit in a single transaction.
- **Never use `random` in analytics.** Replace every `random.random()` call in `caching.py` with real counter reads.
