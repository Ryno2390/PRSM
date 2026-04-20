# Phase 3: Compute Marketplace — TDD Plan

**Companion to:** `docs/2026-04-20-phase3-marketplace-design.md`.

**Estimated scope:** 8 tasks, ~9 commits, ~1100 LoC production + ~750 LoC tests.

**Pattern:** TDD throughout (failing test → implementation → green). Each task's regression suite includes Phase 2 + Phase 2.1 tests to catch breakage early.

Task ordering reflects sequential dependencies:
- Tasks 1-2 can proceed in parallel after design freeze (listing format + directory both pure data work).
- Tasks 3-4 depend on Task 1 (use `ProviderListing` type).
- Task 5 is the integrator (depends on 1-4 + 6).
- Task 6 (reputation) is independent until Task 5 consumes it.
- Task 7 is the integration test.
- Task 8 is the codex gate.

---

## Task 1: `ProviderListing` + Signing + Gossip Topic

**Why:** The listing dataclass is the wire format every other task consumes. Build it first with full sign/verify roundtrip so downstream tasks can construct listings confidently.

**Files:**
- Create: `prsm/marketplace/__init__.py`
- Create: `prsm/marketplace/listing.py`
- Create: `tests/unit/test_provider_listing.py`
- Modify: `prsm/node/gossip.py` (add `GOSSIP_MARKETPLACE_LISTING`)

**Steps:**

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_provider_listing.py
from __future__ import annotations

import time
import pytest

from prsm.marketplace.listing import (
    ProviderListing,
    build_listing_signing_payload,
    sign_listing,
    verify_listing,
)
from prsm.node.identity import generate_node_identity


def _fresh():
    return generate_node_identity(display_name="test-provider")


def test_listing_signing_roundtrip():
    identity = _fresh()
    listing = sign_listing(
        identity=identity,
        capacity_shards_per_sec=10.0,
        max_shard_bytes=10 * 1024 * 1024,
        supported_dtypes=["float64"],
        price_per_shard_ftns=0.05,
        tee_capable=False,
        stake_tier="standard",
        ttl_seconds=300,
    )
    assert verify_listing(listing) is True


def test_listing_tamper_detection():
    identity = _fresh()
    listing = sign_listing(
        identity=identity,
        capacity_shards_per_sec=10.0,
        max_shard_bytes=10 * 1024 * 1024,
        supported_dtypes=["float64"],
        price_per_shard_ftns=0.05,
        tee_capable=False,
        stake_tier="standard",
        ttl_seconds=300,
    )
    # Mutate a field — frozen dataclass, so reconstruct:
    tampered = ProviderListing(
        **{**listing.to_dict(), "price_per_shard_ftns": 0.01}
    )
    assert verify_listing(tampered) is False


def test_listing_provider_id_must_match_pubkey():
    """Closes the 'claim provider A identity while carrying B pubkey' attack
    at the listing layer (same guard as Phase 2 receipts)."""
    victim = _fresh()
    attacker = _fresh()
    listing = sign_listing(
        identity=attacker,
        capacity_shards_per_sec=10.0,
        max_shard_bytes=1024,
        supported_dtypes=["float64"],
        price_per_shard_ftns=0.05,
        tee_capable=False,
        stake_tier="open",
        ttl_seconds=60,
    )
    # Swap the claimed provider_id to the victim's.
    forged = ProviderListing(
        **{**listing.to_dict(), "provider_id": victim.node_id}
    )
    assert verify_listing(forged) is False


def test_listing_roundtrip_serialization():
    identity = _fresh()
    listing = sign_listing(
        identity=identity, capacity_shards_per_sec=5.0,
        max_shard_bytes=1024, supported_dtypes=["float32"],
        price_per_shard_ftns=0.1, tee_capable=True,
        stake_tier="premium", ttl_seconds=120,
    )
    as_dict = listing.to_dict()
    restored = ProviderListing.from_dict(as_dict)
    assert restored == listing
    assert verify_listing(restored) is True


def test_listing_is_expired():
    identity = _fresh()
    now = int(time.time())
    listing = sign_listing(
        identity=identity, capacity_shards_per_sec=1.0,
        max_shard_bytes=1024, supported_dtypes=["float64"],
        price_per_shard_ftns=0.01, tee_capable=False,
        stake_tier="open", ttl_seconds=1,
    )
    assert listing.is_expired(at_unix=now + 2) is True
    assert listing.is_expired(at_unix=now) is False
```

- [ ] **Step 2: Run tests — verify they fail** (module doesn't exist yet).

- [ ] **Step 3: Implement `prsm/marketplace/listing.py`**

Key shapes (full impl per design §3.1):
- `ProviderListing` — frozen dataclass matching the design.
- `build_listing_signing_payload()` — `keccak256("{listing_id}||...||{ttl_seconds}")`.
- `sign_listing(identity, ...)` — constructs listing, derives provider_id from identity, signs payload.
- `verify_listing(listing)` — four checks: (1) `provider_id == hex(sha256(pubkey))[:32]` binding, (2) Ed25519 signature valid, (3) ttl_seconds ≥ 0, (4) price_per_shard_ftns ≥ 0.
- `ProviderListing.is_expired(at_unix)` → bool.

Uses the Phase 2 pattern: `eth_utils.keccak` + `prsm.node.identity.verify_signature`.

- [ ] **Step 4: Add gossip topic constant**

In `prsm/node/gossip.py`, add:
```python
GOSSIP_MARKETPLACE_LISTING = "marketplace_listing"
```

- [ ] **Step 5: Run tests — expect 5 pass.** Commit:

```
feat(marketplace): add ProviderListing with Ed25519 sign/verify

Phase 3 Task 1. Wire format for provider capacity advertisement over
gossip. Reuses the Phase 2 receipt-verification pattern: keccak256
canonical payload + Ed25519 signature + pubkey→node_id binding.

Five tests: sign/verify roundtrip, tamper detection, forged-provider_id
rejection, dict serialization roundtrip, ttl expiry.

Refs: docs/2026-04-20-phase3-marketplace-design.md §3.1
Refs: docs/2026-04-20-phase3-marketplace-plan.md (Task 1)
```

---

## Task 2: `MarketplaceDirectory`

**Why:** The requester-side aggregator. Subscribes to gossip, maintains active-listing dict, evicts stale. Independent of Task 1's implementation once the `ProviderListing` type exists.

**Files:**
- Create: `prsm/marketplace/directory.py`
- Create: `tests/unit/test_marketplace_directory.py`

**Steps:**

- [ ] **Step 1: Failing test sketch (abbreviated)**

```python
def test_directory_ingests_valid_listing(): ...
def test_directory_rejects_bad_signature(): ...
def test_directory_rejects_mismatched_provider_id(): ...
def test_directory_replaces_older_listing_for_same_provider(): ...
def test_directory_ignores_older_listing_for_same_provider(): ...
def test_directory_evicts_expired_listings(): ...
def test_directory_list_active_providers_filters_expired(): ...
def test_directory_size(): ...
```

- [ ] **Step 2: Implement `MarketplaceDirectory`**

```python
class MarketplaceDirectory:
    def __init__(self, gossip):
        self._listings: Dict[str, ProviderListing] = {}
        gossip.subscribe(GOSSIP_MARKETPLACE_LISTING, self._on_listing)

    async def _on_listing(self, subtype, data, origin):
        listing = ProviderListing.from_dict(data)
        if not verify_listing(listing):
            return
        existing = self._listings.get(listing.provider_id)
        if existing and existing.advertised_at_unix >= listing.advertised_at_unix:
            return
        self._listings[listing.provider_id] = listing

    def list_active_providers(self, at_unix=None) -> List[ProviderListing]:
        now = at_unix or int(time.time())
        return [l for l in self._listings.values() if not l.is_expired(now)]

    def get_listing(self, provider_id): ...
    def size(self, at_unix=None) -> int: ...
    def _evict_expired(self, at_unix): ...
```

- [ ] **Step 3-4: Green + commit.**

---

## Task 3: `MarketplaceAdvertiser`

**Why:** Provider-side counterpart to Task 2. Broadcasts the node's own listing on a timer. Auto-updates when `ComputeProvider._current_jobs` crosses capacity thresholds.

**Files:**
- Create: `prsm/marketplace/advertiser.py`
- Create: `tests/unit/test_marketplace_advertiser.py`

**Steps:**

- [ ] **Step 1: Failing tests**

```python
def test_advertiser_broadcasts_on_start(): ...
def test_advertiser_rebroadcasts_on_interval(): ...
def test_advertiser_reflects_capacity_downgrade_when_at_max_jobs(): ...
def test_advertiser_stops_on_stop(): ...
def test_advertiser_signs_every_broadcast_with_fresh_advertised_at(): ...
```

- [ ] **Step 2: Implement**

```python
class MarketplaceAdvertiser:
    def __init__(
        self, identity, gossip, compute_provider,
        capacity_shards_per_sec: float,
        max_shard_bytes: int, supported_dtypes: List[str],
        price_per_shard_ftns: float, tee_capable: bool,
        stake_tier: str, rebroadcast_interval_sec: float = 90.0,
        ttl_seconds: int = 300,
    ):
        self.identity = identity
        self.gossip = gossip
        self.compute_provider = compute_provider
        self.base_capacity = capacity_shards_per_sec
        ...

    async def start(self): ...
    async def stop(self): ...
    async def _broadcast_loop(self): ...

    def _current_listing(self) -> ProviderListing:
        # If compute_provider._current_jobs >= max_concurrent_jobs,
        # advertise 0 capacity — requesters will filter out.
        effective_capacity = (
            0.0 if self.compute_provider._current_jobs
                    >= self.compute_provider.max_concurrent_jobs
            else self.base_capacity
        )
        return sign_listing(
            identity=self.identity,
            capacity_shards_per_sec=effective_capacity,
            ...
        )
```

- [ ] **Step 3-4: Green + commit.**

---

## Task 4: `EligibilityFilter` + `DispatchPolicy`

**Why:** Pure function. No async, no gossip, no network. Exercises every short-circuit case.

**Files:**
- Create: `prsm/marketplace/policy.py`
- Create: `prsm/marketplace/filter.py`
- Create: `tests/unit/test_eligibility_filter.py`

**Steps:**

- [ ] **Step 1: Failing tests** — one per short-circuit case in design §3.3 (7 cases) + a happy-path.

- [ ] **Step 2: Implement**

```python
@dataclass(frozen=True)
class DispatchPolicy:
    max_price_per_shard_ftns: float = float("inf")
    min_price_per_shard_ftns: float = 0.0    # anti-loss-leader guard
    require_tee: bool = False
    min_stake_tier: str = "open"
    min_reputation_score: float = 0.0
    required_dtype: str = "float64"
    min_capacity_shards_per_sec: float = 0.0
    max_timeout_seconds: float = 30.0
    require_unique_providers: bool = False


class EligibilityFilter:
    _TIER_ORDER = {"open": 0, "standard": 1, "premium": 2, "critical": 3}

    def __init__(self, reputation_tracker=None):
        self._reputation = reputation_tracker

    def filter(
        self, listings: List[ProviderListing],
        policy: DispatchPolicy, at_unix: int = None,
    ) -> List[ProviderListing]:
        now = at_unix or int(time.time())
        out = []
        for l in listings:
            if l.is_expired(now): continue
            if l.price_per_shard_ftns > policy.max_price_per_shard_ftns: continue
            if l.price_per_shard_ftns < policy.min_price_per_shard_ftns: continue
            if policy.require_tee and not l.tee_capable: continue
            if self._TIER_ORDER.get(l.stake_tier, -1) < self._TIER_ORDER[policy.min_stake_tier]: continue
            if policy.required_dtype not in l.supported_dtypes: continue
            if l.capacity_shards_per_sec < policy.min_capacity_shards_per_sec: continue
            if self._reputation is not None:
                score = self._reputation.score_for(l.provider_id)
                if score < policy.min_reputation_score: continue
            out.append(l)
        return out
```

- [ ] **Step 3-4: Green + commit.**

---

## Task 5: Provider-Side `preempt_inflight_shards` + Price-Quote Handler

**Why:** Two ComputeProvider additions. Preemption wiring closes the deferred Phase 2.1 Line A scope. Price-quote handler is the server side of design §3.4.

**Files:**
- Modify: `prsm/node/compute_provider.py`
- Modify: `tests/node/test_compute_provider.py`

**Steps:**

- [ ] **Step 1: Failing tests**

```python
def test_preempt_inflight_shards_sends_preempted_to_all(): ...
def test_preempt_inflight_shards_tracked_as_preempted_not_failed(): ...
def test_price_quote_ack_returns_advertised_price(): ...
def test_price_quote_reject_when_at_capacity(): ...
def test_price_quote_reject_when_shard_too_large(): ...
def test_price_quote_expiry_is_future(): ...
```

- [ ] **Step 2: Implement `preempt_inflight_shards`**

```python
async def preempt_inflight_shards(self, reason: str = "spot_eviction") -> None:
    """Send status=preempted for every in-flight shard request. Called
    from SIGTERM handler or pod-eviction hook. Phase 2.1 Line A."""
    for request_id, (peer_id, shard_index) in list(self._inflight_requests.items()):
        await self._send_shard_response(
            peer_id, request_id,
            status="preempted", shard_index=shard_index, error=reason,
        )
        del self._inflight_requests[request_id]
```

- [ ] **Step 3: Implement price-quote handler**

```python
async def _on_shard_price_quote_request(self, msg, peer):
    payload = msg.payload
    listing_price = self._marketplace_advertiser.current_price_ftns() if self._marketplace_advertiser else None
    if listing_price is None:
        resp = {"subtype": "shard_price_quote_reject", "reason": "no_active_listing", ...}
    elif self._current_jobs >= self.max_concurrent_jobs:
        resp = {"subtype": "shard_price_quote_reject", "reason": "overloaded", ...}
    elif payload.get("shard_size_bytes", 0) > self.MAX_SHARD_BYTES:
        resp = {"subtype": "shard_price_quote_reject", "reason": "shard_too_large", ...}
    elif listing_price > payload.get("max_acceptable_price_ftns", 0):
        resp = {"subtype": "shard_price_quote_reject", "reason": "above_ceiling", ...}
    else:
        resp = {
            "subtype": "shard_price_quote_ack",
            "request_id": payload["request_id"],
            "quoted_price_ftns": listing_price,
            "quote_expires_unix": int(time.time()) + 30,
            ...
        }
    await self.transport.send_to_peer(peer.peer_id, P2PMessage(...payload=resp))
```

Register subtype in `_on_direct_message` router.

- [ ] **Step 4: Green + commit.**

---

## Task 6: `ReputationTracker`

**Why:** Rolling-window counters + score derivation. Pure data, no network. Consumed by Task 4's EligibilityFilter and Task 7's orchestrator.

**Files:**
- Create: `prsm/marketplace/reputation.py`
- Create: `tests/unit/test_reputation_tracker.py`

**Steps:**

- [ ] **Step 1: Failing tests**

```python
def test_reputation_new_provider_is_neutral_0_5(): ...
def test_reputation_after_10_successes_is_1_0(): ...
def test_reputation_5_success_5_fail_is_0_5(): ...
def test_reputation_preempted_does_not_affect_score(): ...
def test_reputation_rolling_window_forgets_oldest(): ...
def test_reputation_latency_percentiles(): ...
def test_reputation_below_10_samples_returns_neutral(): ...
```

- [ ] **Step 2: Implement**

```python
@dataclass
class ProviderReputation:
    provider_id: str
    successful_dispatches: Deque[int] = field(default_factory=lambda: deque(maxlen=1000))
    failed_dispatches: Deque[int] = field(default_factory=lambda: deque(maxlen=1000))
    preempted_dispatches: Deque[int] = field(default_factory=lambda: deque(maxlen=1000))
    latency_samples_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    first_seen_unix: int = 0
    last_seen_unix: int = 0


class ReputationTracker:
    _MIN_SAMPLES_FOR_SCORE = 10
    _NEUTRAL = 0.5

    def record_success(self, provider_id: str, latency_ms: float): ...
    def record_failure(self, provider_id: str): ...
    def record_preemption(self, provider_id: str): ...
    def score_for(self, provider_id: str) -> float:
        rep = self._reputations.get(provider_id)
        if rep is None: return self._NEUTRAL
        total = len(rep.successful_dispatches) + len(rep.failed_dispatches)
        if total < self._MIN_SAMPLES_FOR_SCORE: return self._NEUTRAL
        return len(rep.successful_dispatches) / total

    def latency_p50(self, provider_id) -> float: ...
    def latency_p95(self, provider_id) -> float: ...
```

- [ ] **Step 3-4: Green + commit.**

---

## Task 7: `MarketplaceOrchestrator` + 3-Node E2E

**Why:** Integration layer. Composes directory + filter + randomizer + price handshake + dispatcher + receipt verifier + reputation. One entrypoint `orchestrate_sharded_inference()`. Plus the roadmap-level acceptance test.

**Files:**
- Create: `prsm/marketplace/orchestrator.py`
- Create: `prsm/marketplace/errors.py`
- Create: `prsm/marketplace/price_handshake.py`
- Create: `tests/unit/test_marketplace_orchestrator.py`
- Create: `tests/integration/test_phase3_marketplace_e2e.py`
- Create: `tests/integration/conftest_phase3.py` (loopback 3-node cluster with Marketplace wiring, extending the Phase 2 fixture)

**Steps:**

- [ ] **Step 1: Unit tests (orchestrator)**

```python
def test_orchestrator_dispatches_happy_path(): ...
def test_orchestrator_no_eligible_providers_raises(): ...
def test_orchestrator_preemption_reroutes_to_fresh_pool(): ...
def test_orchestrator_missing_tee_attestation_raises_when_required(): ...
def test_orchestrator_price_quote_rejection_tries_next_provider(): ...
def test_orchestrator_records_reputation_on_success_and_failure(): ...
```

- [ ] **Step 2: Integration test (acceptance criterion, design §11)**

```python
@pytest.mark.asyncio
async def test_phase3_marketplace_end_to_end():
    nodes = await spin_up_three_node_marketplace_cluster()
    requester = nodes[0]
    provider_cheap = nodes[1]  # price 0.03
    provider_expensive = nodes[2]  # price 0.10

    # Both providers broadcast listings.
    await provider_cheap.advertiser.start()
    await provider_expensive.advertiser.start()
    await asyncio.sleep(0.1)  # let gossip propagate

    # Confirm directory sees both.
    assert requester.directory.size() == 2

    # Run a sharded inference with policy that excludes the expensive one.
    shards, input_tensor, expected = _make_sharded_workload()
    policy = DispatchPolicy(max_price_per_shard_ftns=0.05, required_dtype="float64")

    result = await requester.orchestrator.orchestrate_sharded_inference(
        shards=shards, input_tensor=input_tensor,
        job_id="phase3-e2e-1", policy=policy,
    )

    np.testing.assert_array_equal(result, expected)
    # Only the cheap provider got traffic.
    assert requester.reputation.score_for(provider_cheap.identity.node_id) > 0.5
    # Escrow + ledger correctness
    assert await requester.ledger.get_balance(provider_cheap.identity.node_id) > 0
    assert await requester.ledger.get_balance(provider_expensive.identity.node_id) == 0
```

- [ ] **Step 3: Implement**

```python
class MarketplaceOrchestrator:
    def __init__(
        self, identity, directory, advertiser,
        eligibility_filter, reputation, randomizer,
        remote_dispatcher, transport,
    ):
        ...

    async def orchestrate_sharded_inference(
        self, shards, input_tensor, job_id, policy,
    ) -> np.ndarray:
        listings = self.directory.list_active_providers()
        eligible = self.eligibility_filter.filter(listings, policy)
        if not eligible:
            raise NoEligibleProvidersError(...)

        assignments = self.randomizer.assign(
            [l.provider_id for l in eligible], num_shards=len(shards),
        )

        outputs = []
        for shard, assignment in zip(shards, assignments):
            listing = self.directory.get_listing(assignment.node_id)
            quote = await self._negotiate_price(
                listing, shard, policy.max_price_per_shard_ftns,
            )
            if quote is None:
                # Re-roll this shard on a different provider
                ...

            started = time.time()
            try:
                output = await self.remote_dispatcher.dispatch(
                    shard=shard, input_tensor=input_tensor,
                    node_id=assignment.node_id, job_id=job_id,
                    stake_tier=_str_to_tier(listing.stake_tier),
                    escrow_amount_ftns=quote.quoted_price_ftns,
                )
            except ShardPreemptedError:
                self.reputation.record_preemption(assignment.node_id)
                # Re-dispatch this shard to a fresh pool excluding preempted provider
                ...
            except ShardDispatchError as exc:
                self.reputation.record_failure(assignment.node_id)
                raise
            else:
                self.reputation.record_success(
                    assignment.node_id, (time.time() - started) * 1000,
                )
                outputs.append(output)

        return np.concatenate(outputs, axis=0)
```

- [ ] **Step 4: Green + commit.**

---

## Task 8: Codex Review Gate + Phase 3 Merge Tag

**Why:** Same pattern as Phase 2 Task 8. Independent pre-merge verification.

**Steps:**

- [ ] **Step 1: Run codex review** with the same incantation as the Phase 2 gate. Focus prompts:
  - Signature verification on ProviderListing (matches Phase 2 receipt-verification pattern; no regressions).
  - Price-quote ceiling enforcement (requester rejects quotes > listing).
  - Preemption re-routing picks a provider NOT in the preempted set (prevents thrash).
  - Reputation scoring math correctness.
  - Integration test passes with `np.testing.assert_array_equal`.
  - No Phase 2 / Phase 2.1 regressions.

- [ ] **Step 2: Address findings.** P1/P2 → patch + retest + re-run codex until clean.

- [ ] **Step 3: Tag the clean-review commit**

```bash
git tag phase3-merge-ready-$(date +%Y%m%d) -m "Phase 3 marketplace — codex gate passed"
git push origin main
git push origin --tags
```

- [ ] **Step 4: Update PRSM memory with Phase 3 shipped summary.**

Phase 3 done. Natural follow-ons: Phase 3.1 (on-chain batch settlement) once Phase 1.3 Task 8 completes; Phase 3.2 (on-chain provider registry for high-value T3 operators); Phase 4 per roadmap.

---

## Self-Review Checklist

After all 8 tasks complete:

- [ ] All unit tests pass: ~35+ new tests across listing, directory, advertiser, filter, reputation, orchestrator.
- [ ] 3-node marketplace integration test passes with bit-identical local/remote output + price-filtering assertion.
- [ ] Phase 2 + Phase 2.1 regression suite unchanged.
- [ ] ComputeProvider's new `preempt_inflight_shards` + `_on_shard_price_quote_request` handlers have unit tests.
- [ ] `MarketplaceOrchestrator` handles: empty eligible pool, preemption re-route, missing TEE attestation, price-quote rejection.
- [ ] Codex review returns SAFE TO MERGE.
- [ ] No `TODO`, `FIXME`, or `pass # placeholder` markers in any new or modified file.
- [ ] No new files in repo root.
- [ ] Every commit message references `docs/2026-04-20-phase3-marketplace-{design,plan}.md`.

## Estimated Scope

- 9 commits (1 per task + 1 codex-fix)
- 10 new production files in `prsm/marketplace/`
- 2 modified production files (`compute_provider.py`, `gossip.py`, `node.py`)
- 8 new test files (~35 unit tests + 1 integration test)
- Total: ~1100 LoC new production + ~750 LoC tests
- Roughly the same shape as Phase 2 proper.

---

## Execution timing

~5 weeks available before hardware ships (2026-05-29 target). Phase 3 fits comfortably:
- Week 1: Tasks 1-3 (listing, directory, advertiser — mostly pure data).
- Week 2: Tasks 4-5 (filter, provider-side preempt + price-quote handlers).
- Week 3: Task 6-7 implementation (reputation + orchestrator + integration test).
- Week 4: Codex gate + fixes.
- Week 5: Buffer / Phase 3.1 scaffolding / Phase 1.3 Task 9-10 drafting.

---

**Plan authored:** 2026-04-20. Design doc: `docs/2026-04-20-phase3-marketplace-design.md`.
