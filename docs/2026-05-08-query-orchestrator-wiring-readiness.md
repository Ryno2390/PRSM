# QueryOrchestrator wiring-readiness assessment

**Date:** 2026-05-08
**Tip at write:** `fb94c480` (`query-orchestrator-core-merge-ready-20260507`)
**Author:** post-sprint cap

> **Status update (historical):** The "What's NOT done" / "What blocks
> production wiring" sections below capture the state at write-time
> (2026-05-08 morning). The production wiring shipped the same day —
> `node.py` now builds the orchestrator (`agent_forge =
> self._build_query_orchestrator_or_none()`, no longer hard-coded
> `None`) and `BROKEN_TOOLS_HIDDEN` is now empty (the MCP tools were
> unhidden). Read the remainder as a point-in-time assessment, not the
> current blocker list.

---

## What's done

QueryOrchestrator core stack — 5 sub-modules + composition class —
landed merge-ready 2026-05-07 across 5 commits:

| Sub-module | Commit | LoC | Tests |
|---|---|---|---|
| `aggregator_selector.py` | `b72feeae` | ~250 | 22 + 5 Monte-Carlo |
| `decomposer.py` | `148c3ae1` | ~190 | 14 |
| `shard_finder.py` | `eaa73ac2` | ~110 | 10 |
| `swarm_runner.py` | `22421118` | ~190 | 9 |
| `QueryOrchestrator` class | `fb94c480` | ~150 | 8 |
| **Total** | | ~890 | **68 + 5 stress = 93** |

Threat-model coverage from
`docs/2026-05-07-aggregator-selector-threat-model.md` is complete in
the modules:

| Adversary | Mitigation lives in | Status |
|---|---|---|
| A1 collusion + per-staker rate limit | `aggregator_selector` | ✅ enforced |
| A2 self-exclusion + fail-closed | `aggregator_selector` | ✅ enforced |
| A3 unbondDelay invariant | StakeBond contract (existing) | ✅ inherited |
| A4 preemption signal | orchestrator retry-loop shell | ⏳ shell deferred |
| A5 DP-noise enforcement | `swarm_runner` | ✅ enforced |
| A6 commit-reveal seed | `aggregator_selector` | ✅ enforced |
| A7 governance denylist | `aggregator_selector` | ✅ enforced |
| A8 pubkey-hash identity | `aggregator_selector` | ✅ enforced |
| A9 commit-verify | `swarm_runner` + `aggregator_selector` | ✅ enforced |
| A10 constant-time selection | `aggregator_selector` | ✅ enforced |

---

## What's NOT done (deliberately separated)

Three deliverables together constitute the
"orchestrator wiring + mainnet ready" task:

1. **Production `SwarmDispatcher` adapter** wrapping
   `prsm/compute/agents/dispatcher.py::AgentDispatcher`
2. **Production `AggregatorClient`** routing partials to the selected
   aggregator and parsing back `(plaintext, AggregationCommit)`
3. **`node.py:1277` rewire** — `agent_forge = QueryOrchestrator(...)`
   replacing `agent_forge = None`
4. **`mcp_server.py` `BROKEN_TOOLS_HIDDEN` unhide** — restore
   `prsm_analyze` / `prsm_dispatch_agent` / `prsm_agent_status` to
   `list_tools()`

These are deliberately separate because they cannot land partially
without silently breaking the canonical workflow. Lifting MCP gating
before production adapters exist would surface 503-ish failures from
a tool that *appears* functional. Keeping the gate on until real
backends ship is the honest path.

---

## What blocks production wiring

### Blocker 1 — Aggregator chain-RPC endpoint design

The `AggregatorClient.aggregate(...)` call is a remote round-trip:
prompter's node sends partials to selected aggregator, aggregator
runs DP-noised combination, returns `(plaintext, AggregationCommit)`.

**No such endpoint exists yet.** Pre-implementation gates:

- **Wire format.** Likely an extension to `prsm/network/transport`
  similar to `RunLayerSliceRequest` / `RunLayerSliceResponse` from
  Phase 3.x.7. `AggregateRequest` carries: query_id, manifest,
  partials (signed by source agents), prompter-pubkey-for-encryption.
  `AggregateResponse` carries: aggregated plaintext (encrypted to
  prompter pubkey), `AggregationCommit` (signed by aggregator).
- **Server-side handler.** Aggregator nodes need a request handler
  that:
  - verifies each partial's signature (provenance — source agents)
  - verifies each partial's `dp_noise_applied` marker
  - applies privacy-budget-bounded combination via the manifest's
    aggregate ops (`COUNT`, `SUM`, `AVERAGE`, etc.)
  - signs `AggregationCommit` with aggregator's identity key
  - returns `(plaintext, commit)`

**Estimated effort:** 1–1.5 weeks for design+impl of one `AggregateRequest`
RPC + 3-node E2E test + receipt-format extension. Pattern lift from
Phase 3.x.1 InferenceReceipt sign/verify is direct.

### Blocker 2 — Six governance parameters await ratification

Per threat-model `§"Open governance questions"`, six parameters need
Foundation-council ratification before mainnet:

1. `p_check` redundancy rate (default proposed: 0.05)
2. `MAX_AGG_FRACTION` per-prompter rate limit (default: 1/N)
3. `MAX_AGG_RETRIES` per query (default: 3)
4. Beacon source: daily Foundation-multisig + every-100th-query
   on-chain anchor at v1; revisit at TVL > $1M
5. Governance denylist mechanism: Foundation Safe 2-of-3 writes to
   deny-list contract; selector polls
6. `challengeWindow` for aggregator misbehavior (default: 24h)

The aggregator_selector module ships with sensible defaults +
env-driven overrides so ratification can run in parallel with
endpoint development. Ratification is NOT a code-blocking item — it's
a policy-blocking item before any production traffic.

### Blocker 3 — Retry-loop shell (A4)

The threat model's A4 mitigations (bounded retries, preemption
signal, escrow refund on no-aggregator) live in a layer that wraps
`QueryOrchestrator.dispatch_query`. This is small (~50–100 LoC) but
needs:

- Escrow integration (existing — `prsm/economy/web3/`)
- `ReputationTracker.record_preemption` callback wiring (existing —
  `prsm/marketplace/reputation.py:133`)
- Bounded retry loop with `MAX_AGG_RETRIES` (gov param above)

**Estimated effort:** 0.5–1 week. Cleanly separable from the chain-RPC
work; can land in parallel.

### Blocker 4 — Top-k method on `_SemanticIndex`

`shard_finder.py` consumes `SemanticIndex.find_top_k(query, k)`. The
existing `ContentUploader._semantic_index.find_nearest()` returns
single-best. A small follow-on adds `find_top_k`:

```python
def find_top_k(self, query: str, k: int) -> list[tuple[str, float, str]]:
    """O(n) sort over the current local index, plus DHT escalation
    on weak local match (existing pattern)."""
```

**Estimated effort:** ~half-day. Bounded by existing test patterns.

### Blocker 5 — Source-agent DP-noise marker setting

The `PartialResult.dp_noise_applied` field is a boolean marker the
source agent sets after applying `dp_noise.py` primitives. The
SwarmDispatcher production adapter must thread this from the
agent's response into the `PartialResult` it constructs. This is
a wire-format detail — straightforward but auditable.

**Estimated effort:** ~half-day, bundled with Blocker 1 work.

---

## Recommended next-sprint scope

Pick **either**:

### Path A — endpoint-first (mainnet-aimed)

1. Design `AggregateRequest`/`AggregateResponse` wire format
2. Build aggregator-side handler + receipt sign/verify
3. Build `AggregatorClient` adapter
4. Build `SwarmDispatcher` adapter around `AgentDispatcher`
5. Add `find_top_k` to `_SemanticIndex`
6. Build retry-loop shell wrapping `QueryOrchestrator`
7. Wire `node.py:1277`
8. Lift `BROKEN_TOOLS_HIDDEN`
9. 3-node E2E test exercising the full canonical workflow
10. Foundation-council ratification packet for the 6 governance params

**Estimated effort:** 2.5–3.5 weeks. **Result:** canonical workflow
operational on mainnet pending council ratification.

### Path B — local-only (validation-aimed)

1. Build `LocalSwarmDispatcher` running WASM agents on the local
   node (single-node operator can dispatch to themselves *if* they
   relax A2, which they shouldn't in production — this is a TEST
   wiring only, gated behind an env var)
2. Build `LocalAggregatorClient` — in-process aggregation
3. Wire single-node test deployment
4. Validate the full stack composes correctly without any chain-RPC
5. Document the gating story so future contributors understand
   this is NOT production wiring

**Estimated effort:** 0.5 week. **Result:** stack validated end-to-end
in test-only mode; no mainnet path; canonical workflow remains gated
on Path A's chain-RPC endpoint.

**Recommendation:** Path A. Path B's "we proved it composes" is
already proved by the 8 composition unit tests in
`test_query_orchestrator_compose.py`. Path A is the work that
unblocks mainnet.

### Parallel-track items (any sprint)

- Six governance parameters ratification (Foundation council action,
  not engineering)
- Auditor engagement update — when production wiring lands, gate #31
  / #40 are fully unblocked from the design side; auditor can
  pre-review the orchestrator core stack now while the wiring sprint
  runs

---

## Risk surface

| Risk | Where it lives | Mitigation |
|---|---|---|
| Wire-format design choices propagate to chain-RPC code | Blocker 1 | Pattern-lift directly from Phase 3.x.1 InferenceReceipt + Phase 3.x.7 LayerStageServer; minimize design freedom |
| Aggregator-side combination correctness (DP budget tracking) | Blocker 1 server-side | Reuse `prsm/security/privacy_budget.py` directly; tight test surface around per-op budget consumption |
| Source-agent DP-noise marker forgery | SwarmDispatcher boundary | The marker is a hint — actual A5 enforcement runs at swarm_runner via verification of the DP-budget receipt the agent signs. The boolean is a fail-fast signal, not the security boundary |
| Top-k semantic-index escalation cost | Blocker 4 | Cap `k` at 32 / `MAX_LIMIT=1024` (already enforced in shard_finder); bound DHT escalation pulls per call (already enforced in `_SemanticIndex._max_remote_pulls_per_query`) |
| Six governance parameters drift | Foundation council | Module ships with proposed defaults + env overrides — ratification can swap defaults without code change |

---

## References

- `docs/2026-05-07-aggregator-selector-threat-model.md` — binding
  design artifact for aggregator_selector
- `docs/2026-05-07-canonical-workflow-gap-list-delta.md` — gap-list
  context the rebuild closes
- Tags: `query-orchestrator-{aggregator-selector,decomposer,shard-finder,swarm-runner}-merge-ready-20260507`,
  `query-orchestrator-core-merge-ready-20260507`
- Sub-modules:
  - `prsm/compute/query_orchestrator/aggregator_selector.py`
  - `prsm/compute/query_orchestrator/decomposer.py`
  - `prsm/compute/query_orchestrator/shard_finder.py`
  - `prsm/compute/query_orchestrator/swarm_runner.py`
  - `prsm/compute/query_orchestrator/orchestrator.py`
- Tests: 6 unit files under `tests/unit/test_aggregator_selector*.py`
  / `test_query_decomposer.py` / `test_shard_finder.py` /
  `test_swarm_runner.py` / `test_query_orchestrator_compose.py`
