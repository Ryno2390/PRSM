# Federated Evolution & Marketplace Cleanup — Implementation Plan

## Overview

This plan eliminates the next tier of stub implementations. The primary targets are:

1. **`distributed_evolution.py`** — 6 methods using `random.randint/uniform/choice` as fake P2P
   responses. Replace with real HTTP calls to peer nodes and real DB-backed metrics.
2. **`contributor_onboarding.py`** — `random.random() > 0.5` badge assignment. Replace with
   deterministic contribution-based criteria.
3. **`plugin_registry.py`** — File-size-only code complexity check. Extend with real AST-based
   cyclomatic complexity analysis using the `ast` module (already imported).
4. **`knowledge_extractor.py`** — Simulated performance benchmarking. Replace with real timing
   using `time.perf_counter`.

All 4 files are under `prsm/`. None require new external dependencies.

---

## Phase 1: Distributed Evolution — Real P2P Calls via HTTP

**File:** `prsm/compute/federation/distributed_evolution.py`

The `DistributedArchiveSynchronizer` class wraps a `p2p_network` object. The `MockP2PNetwork`
at the top is used as a fallback and `get_connected_peers()` returns `[]` — so currently no real
P2P work ever happens. The 6 stubs below will be implemented using `httpx.AsyncClient` to call
the existing peer REST endpoints exposed by `enhanced_p2p_network.py`, with fallbacks to the
local archive state when peers are unavailable.

### 1a — `_request_archive_metadata(peer_id: str) -> Dict[str, Any]`

Currently: `random.randint(50, 500)` and `random.uniform(0.7, 0.95)`.

Implementation: Query `FederationPeerModel` for the peer's address/port, then attempt an HTTP GET
to the peer's metadata endpoint. Fall back to returning the last cached metadata (stored in
`self.peer_archives`) if the peer is unreachable, or a minimal safe dict if no cache exists:

```python
async def _request_archive_metadata(self, peer_id: str) -> Dict[str, Any]:
    try:
        import httpx
        from prsm.core.database import get_async_session, FederationPeerModel
        from sqlalchemy import select

        async with get_async_session() as session:
            stmt = select(FederationPeerModel).where(
                FederationPeerModel.peer_id == peer_id,
                FederationPeerModel.is_active == True,
            )
            result = await session.execute(stmt)
            peer = result.scalar_one_or_none()

        if peer:
            url = f"http://{peer.address}:{peer.port}/api/v1/federation/archive/metadata"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()

    except Exception as e:
        logger.warning(f"Failed to fetch archive metadata from peer {peer_id}: {e}")

    # Fall back to cached metadata or safe defaults
    if peer_id in self.peer_archives:
        return self.peer_archives[peer_id]

    # Minimal safe dict — no random data
    return {
        "node_id": peer_id,
        "total_solutions": 0,
        "component_types": [],
        "best_performance": 0.0,
        "last_updated": datetime.utcnow().isoformat(),
        "archive_version": "1.0",
        "available": False,
    }
```

### 1b — `_send_sync_request(sync_request: SolutionSyncRequest) -> SolutionSyncResponse`

Currently: delegates to `_simulate_peer_solutions()` which creates random `SolutionNode` objects.

Implementation: HTTP POST to the peer's sync endpoint. On failure, return an unsuccessful
response instead of fake solutions:

```python
async def _send_sync_request(self, sync_request: SolutionSyncRequest) -> SolutionSyncResponse:
    try:
        import httpx
        from prsm.core.database import get_async_session, FederationPeerModel
        from sqlalchemy import select

        async with get_async_session() as session:
            stmt = select(FederationPeerModel).where(
                FederationPeerModel.peer_id == sync_request.target_node_id,
                FederationPeerModel.is_active == True,
            )
            result = await session.execute(stmt)
            peer = result.scalar_one_or_none()

        if peer:
            url = f"http://{peer.address}:{peer.port}/api/v1/federation/archive/sync"
            payload = {
                "request_id": sync_request.request_id,
                "requesting_node_id": sync_request.requesting_node_id,
                "sync_strategy": sync_request.sync_strategy.value,
                "max_solutions": sync_request.max_solutions,
                "newer_than": sync_request.newer_than,
                "component_types": [ct.value for ct in (sync_request.component_types or [])],
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                # Deserialize solutions from JSON response
                solutions = [
                    SolutionNode(**s) for s in data.get("solutions", [])
                    if isinstance(s, dict)
                ]
                return SolutionSyncResponse(
                    request_id=sync_request.request_id,
                    responding_node_id=sync_request.target_node_id,
                    solutions=solutions,
                    total_solutions_available=data.get("total_available", len(solutions)),
                    solutions_sent=len(solutions),
                    data_size_mb=data.get("data_size_mb", len(solutions) * 0.1),
                    success=True,
                )

    except Exception as e:
        logger.warning(f"Sync request to {sync_request.target_node_id} failed: {e}")

    # Return unsuccessful response — no fake data
    return SolutionSyncResponse(
        request_id=sync_request.request_id,
        responding_node_id=sync_request.target_node_id,
        solutions=[],
        total_solutions_available=0,
        solutions_sent=0,
        data_size_mb=0.0,
        success=False,
        error_message="Peer unreachable",
    )
```

Also **delete** the `_simulate_peer_solutions()` method entirely since it is now unused.

### 1c — `_collect_task_results(task, timeout_seconds) -> List[Dict]`

Currently: `asyncio.sleep(random.uniform(0.1, 1.0))` and a fake result dict per node.

Implementation: HTTP GET to each assigned node's task-result endpoint with the real `timeout_seconds`.
Replace the sleep + random generation with real HTTP collection:

```python
async def _collect_task_results(
    self, task: NetworkEvolutionTask, timeout_seconds: float
) -> List[Dict[str, Any]]:
    import httpx
    from prsm.core.database import get_async_session, FederationPeerModel
    from sqlalchemy import select

    results = []

    # Fetch peer address map once
    async with get_async_session() as session:
        stmt = select(FederationPeerModel).where(
            FederationPeerModel.peer_id.in_(task.assigned_nodes),
        )
        db_result = await session.execute(stmt)
        peers = {p.peer_id: p for p in db_result.scalars().all()}

    async def _fetch_node_result(node_id: str) -> Optional[Dict[str, Any]]:
        peer = peers.get(node_id)
        if not peer:
            return None
        url = f"http://{peer.address}:{peer.port}/api/v1/federation/tasks/{task.task_id}/result"
        try:
            async with httpx.AsyncClient(timeout=min(timeout_seconds, 30.0)) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning(f"Failed to collect result from node {node_id}: {e}")
            return None

    fetch_coros = [_fetch_node_result(node_id) for node_id in task.assigned_nodes]
    raw = await asyncio.gather(*fetch_coros, return_exceptions=False)
    results = [r for r in raw if r is not None]

    logger.info(f"Collected {len(results)} task results for task {task.task_id}")
    return results
```

### 1d — `_achieve_consensus_on_improvements(aggregated_result) -> bool`

Currently: `aggregated_result["best_performance_achieved"] > 0.85` with the comment
"Simulate consensus process".

The threshold check itself is fine — but the result currently fed in is computed from random
data (since `_collect_task_results` generates random values). Once phases 1c is implemented,
the aggregated data will be real. Make the consensus threshold configurable and record the
decision:

```python
async def _achieve_consensus_on_improvements(
    self, aggregated_result: Dict[str, Any]
) -> bool:
    # Configurable threshold (default 0.85 if not set on the class)
    threshold = getattr(self, '_consensus_performance_threshold', 0.85)
    best_performance = aggregated_result.get("best_performance_achieved", 0.0)
    consensus_achieved = best_performance > threshold

    logger.info(
        "Consensus evaluation",
        best_performance=best_performance,
        threshold=threshold,
        consensus_achieved=consensus_achieved,
        participating_nodes=aggregated_result.get("participating_nodes", 0),
    )
    return consensus_achieved
```

### 1e — `_deploy_network_improvements(aggregated_result) -> bool`

Currently: `random.random() < 0.9` (90% random success rate).

Implementation: Add top-performing solutions from the aggregated result to the local archive,
which is the actual deployment action for this federated evolution system:

```python
async def _deploy_network_improvements(
    self, aggregated_result: Dict[str, Any]
) -> bool:
    try:
        top_solutions = aggregated_result.get("top_solutions", [])
        if not top_solutions:
            logger.info("No top solutions to deploy")
            return True  # Nothing to do is not a failure

        deployed_count = 0
        for solution_data in top_solutions:
            try:
                if isinstance(solution_data, SolutionNode):
                    solution = solution_data
                else:
                    continue
                await self._process_received_solutions([solution])
                deployed_count += 1
            except Exception as e:
                logger.warning(f"Failed to deploy solution: {e}")

        logger.info(f"Deployed {deployed_count}/{len(top_solutions)} network improvements")
        return deployed_count > 0 or len(top_solutions) == 0

    except Exception as e:
        logger.error(f"Failed to deploy network improvements: {e}")
        return False
```

### 1f — `_measure_network_performance() -> float`

Currently: `base_performance = 0.7` + `random.uniform(-0.1, 0.1)`.

Implementation: average quality scores of active federation peers from DB, falling back to
0.0 (not a fake baseline) when DB is unavailable:

```python
async def _measure_network_performance(self) -> float:
    try:
        from prsm.core.database import get_async_session, FederationPeerModel
        from sqlalchemy import select, func

        async with get_async_session() as session:
            stmt = select(func.avg(FederationPeerModel.quality_score)).where(
                FederationPeerModel.is_active == True,
            )
            result = await session.execute(stmt)
            avg_quality = result.scalar_one_or_none()

        if avg_quality is not None:
            return float(max(0.0, min(1.0, avg_quality)))

    except Exception as e:
        logger.warning(f"Failed to measure network performance from DB: {e}")

    # Fall back to local archive quality if peers unavailable
    if self.local_archive and self.local_archive.solutions:
        scores = [
            s.best_performance_score
            for s in self.local_archive.solutions.values()
            if s.best_performance_score is not None
        ]
        if scores:
            return sum(scores) / len(scores)

    return 0.0
```

---

## Phase 2: Contributor Badge Assignment (Deterministic)

**File:** `prsm/interface/onboarding/contributor_onboarding.py`

### 2a — `_update_contributor_achievements()` (around line 857)

Currently:
```python
if badge not in contributor.badges_earned and random.random() > 0.5:
    contributor.badges_earned.append(badge)
```

Replace with deterministic criteria based on actual contributor state:

```python
elif contributor.level in [ContributorLevel.EXPERT, ContributorLevel.CORE_CONTRIBUTOR]:
    # Deterministic badge criteria based on actual contribution metrics
    badge_criteria = {
        "code_contributor": contributor.contributions_made > 0,
        "mentor": getattr(contributor, 'mentees_count', 0) > 0,
        "reviewer": getattr(contributor, 'reviews_completed', 0) > 0,
    }
    for badge, earned in badge_criteria.items():
        if earned and badge not in contributor.badges_earned:
            contributor.badges_earned.append(badge)
            badges_awarded += 1
```

This removes `random.random()` and replaces it with objective contribution data. If
`mentees_count` or `reviews_completed` fields don't exist on the contributor dataclass,
check the file and either add them or use whatever equivalent field tracks mentoring/reviews.

---

## Phase 3: AST-Based Code Complexity Analysis

**File:** `prsm/economy/marketplace/ecosystem/plugin_registry.py`

### 3a — `_check_code_complexity(package_path, manifest) -> Dict`

Currently: only checks file size (100KB threshold). The comment says "This would integrate
with tools like radon or flake8" but `ast` is already imported at line 21.

Implementation: use `ast.parse()` to count functions, classes, and measure nesting depth
(a proxy for cyclomatic complexity without requiring `radon`):

```python
async def _check_code_complexity(
    self, package_path: Path, manifest: PluginManifest
) -> Dict[str, Any]:
    results = {"penalty": 0, "issues": [], "warnings": []}
    python_files = list(package_path.rglob("*.py"))

    for python_file in python_files:
        try:
            source = python_file.read_text(encoding="utf-8", errors="replace")

            # File size check (keep existing threshold)
            file_size = len(source.encode("utf-8"))
            if file_size > 100_000:
                results["warnings"].append(
                    f"Large file: {python_file.name} ({file_size} bytes)"
                )
                results["penalty"] += 5

            # AST-based complexity analysis
            try:
                tree = ast.parse(source, filename=str(python_file))
            except SyntaxError as e:
                results["issues"].append(f"Syntax error in {python_file.name}: {e}")
                results["penalty"] += 10
                continue

            # Count top-level and nested functions
            func_count = sum(
                1 for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            if func_count > 50:
                results["warnings"].append(
                    f"High function count in {python_file.name}: {func_count} functions"
                )
                results["penalty"] += min(func_count // 10, 20)

            # Measure maximum nesting depth via recursive walk
            max_depth = _max_ast_depth(tree)
            if max_depth > 8:
                results["warnings"].append(
                    f"Deep nesting in {python_file.name}: depth {max_depth}"
                )
                results["penalty"] += min((max_depth - 8) * 2, 15)

            # Count classes
            class_count = sum(
                1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            )
            if class_count > 20:
                results["warnings"].append(
                    f"Many classes in {python_file.name}: {class_count}"
                )
                results["penalty"] += 5

        except Exception as e:
            results["warnings"].append(f"Failed to analyze {python_file.name}: {e}")

    return results
```

Add this helper at module level (NOT inside the class) since `ast.walk` doesn't track depth:

```python
def _max_ast_depth(node: ast.AST, depth: int = 0) -> int:
    """Recursively compute maximum AST nesting depth."""
    children = list(ast.iter_child_nodes(node))
    if not children:
        return depth
    return max(_max_ast_depth(child, depth + 1) for child in children)
```

---

## Phase 4: Real Performance Benchmarking in Knowledge Extractor

**File:** `prsm/compute/distillation/knowledge_extractor.py`

Read the file around line 230–260 to find the exact stub and understand the context (what
model/input is being benchmarked). Then replace the simulated benchmark with real timing:

The typical pattern will be something like:
```python
# BEFORE (simulated):
# performance_score = random.uniform(0.6, 0.9)
# execution_time = random.uniform(0.5, 2.0)

# AFTER (real timing):
import time

start = time.perf_counter()
try:
    # Run the actual model prediction on the benchmark input
    output = model.predict(benchmark_input)  # or model(benchmark_input)
    success = True
except Exception as e:
    logger.warning(f"Benchmark prediction failed: {e}")
    success = False
    output = None
elapsed = time.perf_counter() - start

performance_score = 1.0 / (1.0 + elapsed) if success else 0.0  # latency-based score
execution_time = elapsed
```

Read the file first; adapt to the actual method signature and model API it uses.

---

## Phase 5: Integration Tests

**File:** `tests/integration/test_federated_evolution_cleanup.py` (NEW)

Write 18 tests covering all phases.

### Class: `TestDistributedEvolutionP2P` (8 tests)

- `test_request_archive_metadata_returns_safe_dict_on_db_miss`
  — mock `get_async_session` to return no peer, assert result has `"available": False` and no random values
- `test_request_archive_metadata_attempts_http_when_peer_found`
  — mock DB peer + httpx, assert GET called on correct URL
- `test_request_archive_metadata_falls_back_to_cache_on_http_error`
  — mock httpx to raise `httpx.ConnectError`, pre-set `self.peer_archives[peer_id]`, assert cached value returned
- `test_send_sync_request_returns_failure_when_peer_unreachable`
  — mock httpx ConnectError, assert `response.success is False` and `response.solutions == []`
- `test_send_sync_request_calls_correct_endpoint`
  — mock DB peer + httpx 200 response, assert POST to `/api/v1/federation/archive/sync`
- `test_collect_task_results_returns_empty_when_no_peers_in_db`
  — mock DB returning no peers for assigned nodes, assert returns `[]`
- `test_measure_network_performance_returns_zero_on_empty_db`
  — mock DB avg returning None, assert returns `0.0`
- `test_measure_network_performance_uses_db_quality_scores`
  — mock DB avg returning `0.75`, assert returns `0.75`

### Class: `TestConsensusAndDeployment` (3 tests)

- `test_achieve_consensus_true_when_performance_exceeds_threshold`
  — pass `{"best_performance_achieved": 0.9}`, assert returns `True`
- `test_achieve_consensus_false_when_performance_below_threshold`
  — pass `{"best_performance_achieved": 0.5}`, assert returns `False`
- `test_deploy_network_improvements_returns_true_when_no_solutions`
  — pass `{"top_solutions": []}`, assert returns `True` (no-op is success)

### Class: `TestContributorBadges` (4 tests)

- `test_no_random_badge_assignment`
  — create EXPERT contributor with `contributions_made=0`, `reviews_completed=0`,
    `mentees_count=0`, call `_update_contributor_achievements()` 5 times, assert result is
    identical each call (deterministic)
- `test_code_contributor_badge_when_contributions_positive`
  — contributor with `contributions_made=3` at EXPERT level, assert `"code_contributor"` in badges
- `test_reviewer_badge_when_reviews_positive`
  — contributor with `reviews_completed=2`, assert `"reviewer"` in badges after update
- `test_no_badge_when_no_contributions`
  — contributor at EXPERT with all counts at 0, assert no new badges from potential_badges set

### Class: `TestASTComplexityAnalysis` (3 tests)

- `test_simple_file_returns_zero_penalty`
  — write a simple Python file (`def foo(): pass`) to tmp_path, assert `penalty == 0`
- `test_syntax_error_adds_penalty`
  — write a Python file with intentional syntax error, assert `penalty >= 10` and issue in `issues`
- `test_deeply_nested_code_adds_penalty`
  — write a Python file with 10 levels of `if True:` nesting, assert `penalty > 0` and warning in `warnings`

---

## File Checklist

- [x] `prsm/compute/federation/distributed_evolution.py` — 6 methods (phases 1a–1f); delete `_simulate_peer_solutions()`
- [x] `prsm/interface/onboarding/contributor_onboarding.py` — 1 method (phase 2a)
- [x] `prsm/economy/marketplace/ecosystem/plugin_registry.py` — 1 method + 1 module-level helper (phase 3a)
- [x] `prsm/compute/distillation/knowledge_extractor.py` — 1 stub replaced (phase 4; read file first)
- [x] `tests/integration/test_federated_evolution_cleanup.py` — NEW (18 tests)

**Total stubs eliminated:** ~10 random/fake patterns across 4 files
**New tests added:** 18

---

## Notes

- `distributed_evolution.py` imports `random` at line 20. After this plan is complete, audit
  the file for any remaining `random.*` usages — if none remain, remove the import.
- The `_simulate_peer_solutions()` method is only called from the now-replaced
  `_send_sync_request()`. Delete it rather than leaving dead code.
- `SolutionSyncResponse` may not have a `success` field and `error_message` field — check the
  dataclass definition in the file before implementing 1b. If missing, add them as optional
  fields with defaults `success: bool = True` and `error_message: Optional[str] = None`.
- The knowledge extractor benchmark (phase 4) depends heavily on the exact model API used in
  that file. Read lines 220–270 carefully before implementing; the real fix may be as simple as
  wrapping the existing prediction call in `time.perf_counter()` bookends.
- `httpx` is already in PRSM's dependencies (used in prior plans). No new package additions needed.

---

## Implementation Summary

**Completed: 2026-03-23**

### Phase 1: Distributed Evolution — Real P2P Calls via HTTP

**File:** `prsm/compute/federation/distributed_evolution.py`

1. **`_request_archive_metadata()`** — Replaced `random.randint/uniform` with real HTTP calls to peer nodes via `httpx.AsyncClient`. Falls back to cached metadata or safe defaults when peers are unreachable.

2. **`_send_sync_request()`** — Replaced `_simulate_peer_solutions()` call with HTTP POST to peer's sync endpoint. Returns unsuccessful response with empty solutions on failure.

3. **Deleted `_simulate_peer_solutions()`** — Removed entirely as it was only used by the now-replaced `_send_sync_request()`.

4. **`_collect_task_results()`** — Replaced `asyncio.sleep(random.uniform())` and random result generation with real HTTP collection from peer nodes.

5. **`_achieve_consensus_on_improvements()`** — Made threshold configurable via `_consensus_performance_threshold` attribute. Added proper logging of consensus evaluation.

6. **`_deploy_network_improvements()`** — Replaced `random.random() < 0.9` with actual deployment by processing top solutions through `_process_received_solutions()`.

7. **`_measure_network_performance()`** — Replaced `0.7 + random.uniform(-0.1, 0.1)` with real DB query for average `FederationPeerModel.quality_score`, falling back to local archive quality.

8. **Removed `import random`** — Audited file and removed unused import after all random usages were eliminated.

### Phase 2: Contributor Badge Assignment

**File:** `prsm/interface/onboarding/contributor_onboarding.py`

Replaced `random.random() > 0.5` badge assignment with deterministic criteria:
- `code_contributor`: Awarded when `contributions_made > 0`
- `mentor`: Awarded when `mentees_count > 0`
- `reviewer`: Awarded when `reviews_completed > 0`

### Phase 3: AST-Based Code Complexity Analysis

**File:** `prsm/economy/marketplace/ecosystem/plugin_registry.py`

Extended `_check_code_complexity()` with real AST-based analysis:
- File size check (existing, 100KB threshold)
- Syntax error detection (10 penalty per error)
- Function count analysis (penalty for >50 functions)
- Maximum nesting depth (penalty for depth >8)
- Class count analysis (penalty for >20 classes)

Added module-level helper `_max_ast_depth()` for recursive depth calculation.

### Phase 4: Real Performance Benchmarking

**File:** `prsm/compute/distillation/knowledge_extractor.py`

Replaced simulated performance testing in `_analyze_performance()` with real timing:
- Uses `time.perf_counter()` to measure actual model query latency
- Calculates real tokens/second based on response length and elapsed time
- Falls back to parameter-based estimates only if benchmark fails

### Phase 5: Integration Tests

**File:** `tests/integration/test_federated_evolution_cleanup.py` (NEW)

Created 18 tests covering all phases:
- **TestDistributedEvolutionP2P (8 tests):** P2P HTTP calls, fallback behavior, DB queries
- **TestConsensusAndDeployment (3 tests):** Consensus threshold, deployment logic
- **TestContributorBadges (4 tests):** Deterministic badge assignment
- **TestASTComplexityAnalysis (3 tests):** AST complexity analysis

All tests pass successfully.

### Files Modified

1. `prsm/compute/federation/distributed_evolution.py` — 7 methods updated, 1 deleted, 1 import removed
2. `prsm/interface/onboarding/contributor_onboarding.py` — 1 method updated
3. `prsm/economy/marketplace/ecosystem/plugin_registry.py` — 1 method extended, 1 helper added
4. `prsm/compute/distillation/knowledge_extractor.py` — 1 method updated
5. `tests/integration/test_federated_evolution_cleanup.py` — NEW (18 tests)
