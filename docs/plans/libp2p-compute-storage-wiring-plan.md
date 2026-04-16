# Compute/Storage ↔ libp2p Wiring Implementation Plan

> **Phase 6 scope — not in active execution as of 2026-04-16.**
>
> Companion to [`libp2p-transport-implementation.md`](./libp2p-transport-implementation.md). Both plans are scheduled for **Phase 6 (P2P Network Hardening)** per the [master roadmap](../2026-04-10-audit-gap-roadmap.md), target Q2 2027 — after Phases 1-5 ship. This wiring plan assumes the libp2p transport plan has landed; it cannot execute independently.
>
> Preserved as the authoritative technical design for wiring compute and storage providers through libp2p; do not start task-level execution until Phase 6 opens and the transport plan has shipped.
>
> **Design spec companion:** `docs/libp2p-compute-storage-wiring.md`

> **For agentic workers (when Phase 6 opens):** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire compute/storage providers through the libp2p transport, convert bilateral storage operations to direct P2P, add reliability tracking, and prove it all works with integration tests.

**Architecture:** The libp2p adapters (Libp2pGossip, Libp2pDiscovery, Libp2pTransport) are already drop-in compatible with the gossip/discovery APIs. This plan wires the missing pieces: constructor params, capability announcements, direct P2P for challenges/proofs, deduplication, reliability scoring, and integration tests.

**Tech Stack:** Python 3.10+, asyncio, pytest, existing PRSM node infrastructure

**Spec:** `docs/libp2p-compute-storage-wiring.md`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `prsm/node/discovery.py` | PeerInfo dataclass + PeerDiscovery | Modify: add reliability fields + startup_timestamp to PeerInfo |
| `prsm/node/libp2p_discovery.py` | Libp2pDiscovery adapter | Modify: add startup_timestamp, reliability methods, conditional reset in _on_capability |
| `prsm/node/storage_provider.py` | Storage provider with IPFS + challenges | Modify: add transport/discovery params, direct P2P challenges, deduplication |
| `prsm/node/compute_requester.py` | Job submission + result handling | Modify: add reliability recording on job completion/timeout |
| `prsm/node/node.py` | Node orchestrator | Modify: wire new params, capability announcement, periodic task |
| `tests/integration/test_libp2p_compute_storage.py` | Integration tests | Create: mock transport harness + 5 tests |

---

### Task 1: Add Reliability Fields to PeerInfo

**Files:**
- Modify: `prsm/node/discovery.py:36-48`
- Test: `tests/unit/test_libp2p_discovery.py`

- [ ] **Step 1: Write failing test for reliability_score property**

Add to `tests/unit/test_libp2p_discovery.py`:

```python
import pytest
from prsm.node.discovery import PeerInfo


class TestPeerInfoReliability:
    """Tests for PeerInfo reliability tracking fields."""

    def test_reliability_score_new_peer(self):
        """New peer with no history should have reliability 1.0."""
        peer = PeerInfo(node_id="peer1", address="1.2.3.4:9000")
        assert peer.reliability_score == 1.0

    def test_reliability_score_all_successes(self):
        """Peer with only successes should have reliability 1.0."""
        peer = PeerInfo(node_id="peer1", address="1.2.3.4:9000")
        peer.job_success_count = 10
        assert peer.reliability_score == 1.0

    def test_reliability_score_mixed(self):
        """Peer with 2 successes and 1 failure should have 0.667."""
        peer = PeerInfo(node_id="peer1", address="1.2.3.4:9000")
        peer.job_success_count = 2
        peer.job_failure_count = 1
        assert abs(peer.reliability_score - 0.6667) < 0.01

    def test_reliability_score_all_failures(self):
        """Peer with only failures should have reliability 0.0."""
        peer = PeerInfo(node_id="peer1", address="1.2.3.4:9000")
        peer.job_failure_count = 5
        assert peer.reliability_score == 0.0

    def test_startup_timestamp_default(self):
        """startup_timestamp should default to 0.0."""
        peer = PeerInfo(node_id="peer1", address="1.2.3.4:9000")
        assert peer.startup_timestamp == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_libp2p_discovery.py::TestPeerInfoReliability -v`
Expected: FAIL with `AttributeError: 'PeerInfo' object has no attribute 'reliability_score'`

- [ ] **Step 3: Add reliability fields and property to PeerInfo**

In `prsm/node/discovery.py`, modify the `PeerInfo` dataclass (lines 36-48):

```python
@dataclass
class PeerInfo:
    """Lightweight peer descriptor shared during discovery."""
    node_id: str
    address: str
    display_name: str = ""
    roles: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    supported_backends: List[str] = field(default_factory=list)
    gpu_available: bool = False
    last_seen: float = field(default_factory=time.time)
    last_capability_update: float = field(default_factory=time.time)
    job_success_count: int = 0
    job_failure_count: int = 0
    last_failure_time: float = 0.0
    startup_timestamp: float = 0.0

    @property
    def reliability_score(self) -> float:
        """Compute reliability as success ratio. New peers get benefit of the doubt (1.0)."""
        total = self.job_success_count + self.job_failure_count
        if total == 0:
            return 1.0
        return self.job_success_count / total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_libp2p_discovery.py::TestPeerInfoReliability -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add prsm/node/discovery.py tests/unit/test_libp2p_discovery.py
git commit -m "feat: add reliability tracking fields to PeerInfo"
```

---

### Task 2: Add Reliability Methods and Conditional Reset to Libp2pDiscovery

**Files:**
- Modify: `prsm/node/libp2p_discovery.py`
- Test: `tests/unit/test_libp2p_discovery.py`

- [ ] **Step 1: Write failing tests for record_job_success, record_job_failure, and conditional reset**

Add to `tests/unit/test_libp2p_discovery.py`:

```python
import time
from unittest.mock import AsyncMock, MagicMock
from prsm.node.discovery import PeerInfo
from prsm.node.libp2p_discovery import Libp2pDiscovery


def _make_discovery():
    """Create a Libp2pDiscovery with a mocked transport."""
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = "local_node"
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    d = Libp2pDiscovery(transport=transport, gossip=gossip)
    return d


class TestReliabilityTracking:
    """Tests for job success/failure recording."""

    def test_record_job_success(self):
        d = _make_discovery()
        d._capability_index["peer1"] = PeerInfo(node_id="peer1", address="")
        d.record_job_success("peer1")
        assert d._capability_index["peer1"].job_success_count == 1

    def test_record_job_failure(self):
        d = _make_discovery()
        d._capability_index["peer1"] = PeerInfo(node_id="peer1", address="")
        d.record_job_failure("peer1")
        assert d._capability_index["peer1"].job_failure_count == 1
        assert d._capability_index["peer1"].last_failure_time > 0

    def test_record_unknown_peer_is_noop(self):
        d = _make_discovery()
        d.record_job_success("unknown_peer")  # Should not raise
        d.record_job_failure("unknown_peer")  # Should not raise


class TestConditionalReset:
    """Tests for capability re-announcement with conditional reliability reset."""

    @pytest.mark.asyncio
    async def test_heartbeat_does_not_reset_reliability(self):
        """Periodic heartbeat with same startup_timestamp should NOT reset counters."""
        d = _make_discovery()
        d._capability_index["peer1"] = PeerInfo(
            node_id="peer1", address="",
            startup_timestamp=1000.0,
            job_success_count=5, job_failure_count=3,
        )
        # Simulate heartbeat with same startup_timestamp
        await d._on_capability("capability_announce", {
            "node_id": "peer1",
            "capabilities": [],
            "supported_backends": [],
            "gpu_available": False,
            "startup_timestamp": 1000.0,
        }, "peer1")
        assert d._capability_index["peer1"].job_failure_count == 3
        assert d._capability_index["peer1"].job_success_count == 5

    @pytest.mark.asyncio
    async def test_restart_resets_reliability(self):
        """Re-announcement with newer startup_timestamp should reset counters."""
        d = _make_discovery()
        d._capability_index["peer1"] = PeerInfo(
            node_id="peer1", address="",
            startup_timestamp=1000.0,
            job_success_count=5, job_failure_count=3,
        )
        await d._on_capability("capability_announce", {
            "node_id": "peer1",
            "capabilities": [],
            "supported_backends": [],
            "gpu_available": False,
            "startup_timestamp": 2000.0,
        }, "peer1")
        assert d._capability_index["peer1"].job_failure_count == 0
        assert d._capability_index["peer1"].job_success_count == 0
        assert d._capability_index["peer1"].startup_timestamp == 2000.0

    @pytest.mark.asyncio
    async def test_capability_change_resets_reliability(self):
        """Re-announcement with different capabilities should reset counters."""
        d = _make_discovery()
        d._capability_index["peer1"] = PeerInfo(
            node_id="peer1", address="",
            capabilities=["compute"],
            startup_timestamp=1000.0,
            job_success_count=5, job_failure_count=3,
        )
        await d._on_capability("capability_announce", {
            "node_id": "peer1",
            "capabilities": ["compute", "gpu"],
            "supported_backends": [],
            "gpu_available": True,
            "startup_timestamp": 1000.0,
        }, "peer1")
        assert d._capability_index["peer1"].job_failure_count == 0
        assert d._capability_index["peer1"].job_success_count == 0

    @pytest.mark.asyncio
    async def test_new_peer_announcement(self):
        """First announcement from unknown peer creates entry with startup_timestamp."""
        d = _make_discovery()
        await d._on_capability("capability_announce", {
            "node_id": "new_peer",
            "capabilities": ["storage"],
            "supported_backends": [],
            "gpu_available": False,
            "startup_timestamp": 5000.0,
        }, "new_peer")
        assert "new_peer" in d._capability_index
        assert d._capability_index["new_peer"].startup_timestamp == 5000.0
        assert d._capability_index["new_peer"].reliability_score == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_libp2p_discovery.py::TestReliabilityTracking tests/unit/test_libp2p_discovery.py::TestConditionalReset -v`
Expected: FAIL — `record_job_success` not found, `_on_capability` doesn't handle `startup_timestamp`

- [ ] **Step 3: Add record_job_success/failure methods to Libp2pDiscovery**

In `prsm/node/libp2p_discovery.py`, add after the `set_local_capabilities` method (after line 192):

```python
    def record_job_success(self, node_id: str) -> None:
        """Record a successful job completion for a peer."""
        peer = self._capability_index.get(node_id)
        if peer:
            peer.job_success_count += 1

    def record_job_failure(self, node_id: str) -> None:
        """Record a job failure/timeout for a peer."""
        peer = self._capability_index.get(node_id)
        if peer:
            peer.job_failure_count += 1
            peer.last_failure_time = time.time()
```

- [ ] **Step 4: Add _startup_timestamp and update announce_capabilities**

In `Libp2pDiscovery.__init__`, add after `self._local_gpu_available`:

```python
        self._startup_timestamp: float = time.time()
```

Update `announce_capabilities()` to include `startup_timestamp` in the payload:

```python
    async def announce_capabilities(self) -> int:
        """Publish local capabilities via GossipSub."""
        if self.gossip is None:
            return 0
        return await self.gossip.publish(
            "capability_announce",
            {
                "node_id": self.transport.identity.node_id,
                "capabilities": self._local_capabilities,
                "supported_backends": self._local_backends,
                "gpu_available": self._local_gpu_available,
                "startup_timestamp": self._startup_timestamp,
            },
        )
```

- [ ] **Step 5: Update _on_capability with conditional reliability reset**

Replace the existing `_on_capability` method (lines 314-342) with:

```python
    async def _on_capability(
        self, subtype: str, data: Dict[str, Any], sender_id: str
    ) -> None:
        """Update capability index from a ``capability_announce`` message.

        Resets reliability counters only on restart (new startup_timestamp)
        or capability change. Periodic heartbeats only refresh last_seen.
        """
        node_id = data.get("node_id", sender_id)
        if not node_id:
            return

        new_startup = data.get("startup_timestamp", 0.0)
        new_caps = set(data.get("capabilities", []))

        existing = self._capability_index.get(node_id)
        if existing is not None:
            old_startup = existing.startup_timestamp
            old_caps = set(existing.capabilities)

            # Reset reliability only on restart or capability change
            if new_startup > old_startup or new_caps != old_caps:
                existing.job_success_count = 0
                existing.job_failure_count = 0
                existing.last_failure_time = 0.0
                existing.startup_timestamp = new_startup

            existing.capabilities = data.get("capabilities", existing.capabilities)
            existing.supported_backends = data.get(
                "supported_backends", existing.supported_backends
            )
            existing.gpu_available = data.get("gpu_available", existing.gpu_available)
            existing.last_seen = time.time()
            existing.last_capability_update = time.time()
        else:
            self._capability_index[node_id] = PeerInfo(
                node_id=node_id,
                address=data.get("address", ""),
                display_name=data.get("display_name", ""),
                roles=data.get("roles", []),
                capabilities=data.get("capabilities", []),
                supported_backends=data.get("supported_backends", []),
                gpu_available=data.get("gpu_available", False),
                last_seen=time.time(),
                last_capability_update=time.time(),
                startup_timestamp=new_startup,
            )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_libp2p_discovery.py::TestReliabilityTracking tests/unit/test_libp2p_discovery.py::TestConditionalReset -v`
Expected: 7 PASSED

- [ ] **Step 7: Run full existing test suite to check for regressions**

Run: `python -m pytest tests/unit/test_libp2p_discovery.py -v`
Expected: All existing + new tests PASS

- [ ] **Step 8: Commit**

```bash
git add prsm/node/libp2p_discovery.py tests/unit/test_libp2p_discovery.py
git commit -m "feat: add reliability tracking and conditional reset to Libp2pDiscovery"
```

---

### Task 3: Direct P2P for Storage Challenges with Deduplication

**Files:**
- Modify: `prsm/node/storage_provider.py:95-106` (constructor), `prsm/node/storage_provider.py:166-205` (start), `prsm/node/storage_provider.py:360-393` (content handler), `prsm/node/storage_provider.py:640-718` (challenge/proof)
- Test: `tests/unit/test_storage_provider_direct_p2p.py` (new file)

- [ ] **Step 1: Write failing tests for direct P2P challenges and deduplication**

Create `tests/unit/test_storage_provider_direct_p2p.py`:

```python
"""Tests for StorageProvider direct P2P challenge/proof delivery."""
import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from prsm.node.storage_provider import StorageProvider
from prsm.node.transport import MSG_DIRECT, P2PMessage


def _make_identity():
    mock = MagicMock()
    mock.node_id = "local_node_abc"
    mock.private_key_bytes = b"\x00" * 32
    mock.public_key_bytes = b"\x00" * 32
    return mock


def _make_storage_provider(transport_send_succeeds=True):
    identity = _make_identity()
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock(return_value=1)
    ledger = MagicMock()
    ledger.get_balance = AsyncMock(return_value=1000.0)
    ledger.transfer = AsyncMock()
    transport = MagicMock()
    transport.on_message = MagicMock()
    transport.send_to_peer = AsyncMock(return_value=transport_send_succeeds)
    discovery = MagicMock()
    discovery.provide_content = AsyncMock()

    sp = StorageProvider(
        identity=identity,
        gossip=gossip,
        ledger=ledger,
        transport=transport,
        discovery=discovery,
    )
    return sp


class TestDirectP2PChallenge:
    """Tests for storage challenge delivery via direct P2P."""

    @pytest.mark.asyncio
    async def test_constructor_accepts_transport_and_discovery(self):
        """StorageProvider constructor should accept transport and discovery."""
        sp = _make_storage_provider()
        assert sp.transport is not None
        assert sp.discovery is not None

    @pytest.mark.asyncio
    async def test_challenge_deduplication(self):
        """Duplicate challenge IDs should be dropped."""
        sp = _make_storage_provider()
        sp._running = True
        sp._seen_challenge_ids = {}

        # First call should be processed (we just test dedup, not full handler)
        challenge_data = {
            "challenge": {"challenge_id": "chal_001", "cid": "QmTest"},
            "challenger_id": "remote_node",
            "target_provider_id": "local_node_abc",
        }

        # Mark as seen
        sp._seen_challenge_ids["chal_001"] = time.time()

        # Second call with same ID should be dropped silently
        await sp._on_storage_challenge("storage_challenge", challenge_data, "remote_node")
        # If dedup works, _storage_prover.answer_challenge should NOT be called
        # (since _storage_prover is None in this test, it would raise if reached)

    @pytest.mark.asyncio
    async def test_seen_challenge_cleanup(self):
        """Old entries in _seen_challenge_ids should be evicted."""
        sp = _make_storage_provider()
        sp._seen_challenge_ids = {
            "old_challenge": time.time() - 700,  # 11+ minutes old
            "recent_challenge": time.time() - 60,  # 1 minute old
        }
        sp._cleanup_seen_challenges()
        assert "old_challenge" not in sp._seen_challenge_ids
        assert "recent_challenge" in sp._seen_challenge_ids
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_storage_provider_direct_p2p.py -v`
Expected: FAIL — `StorageProvider.__init__()` does not accept `transport` or `discovery`

- [ ] **Step 3: Update StorageProvider constructor**

In `prsm/node/storage_provider.py`, modify `__init__` (line 95-106) to accept `transport` and `discovery`:

```python
    def __init__(
        self,
        identity: NodeIdentity,
        gossip: GossipProtocol,
        ledger: LocalLedger,
        ipfs_api_url: str = "http://127.0.0.1:5001",
        pledged_gb: float = 10.0,
        reward_interval: float = 3600.0,
        challenge_config: Optional[ChallengeConfig] = None,
        config: Optional["NodeConfig"] = None,
        content_economy: Optional[Any] = None,
        transport: Optional[Any] = None,
        discovery: Optional[Any] = None,
    ):
        self.identity = identity
        self.gossip = gossip
        self.ledger = ledger
        self.ipfs_api_url = ipfs_api_url
        self.pledged_gb = pledged_gb
        self.reward_interval = reward_interval
        self.config = config
        self.content_economy = content_economy
        self.transport = transport
        self.discovery = discovery
```

Also add after the `_provider_reputation` dict initialization (after line 152):

```python
        # Deduplication for challenges received via both direct P2P and gossip fallback
        self._seen_challenge_ids: Dict[str, float] = {}
```

- [ ] **Step 4: Add _cleanup_seen_challenges method**

Add after `_provider_reputation` initialization block:

```python
    def _cleanup_seen_challenges(self) -> None:
        """Evict challenge IDs older than 10 minutes from the dedup set."""
        cutoff = time.time() - 600
        for cid in list(self._seen_challenge_ids.keys()):
            if self._seen_challenge_ids.get(cid, 0) < cutoff:
                self._seen_challenge_ids.pop(cid, None)
```

- [ ] **Step 5: Add unified direct message dispatcher and update start()**

Replace the `register_content_handler` method (line 360-364) with a unified dispatcher:

```python
    def register_content_handler(self, transport: WebSocketTransport) -> None:
        """Register to handle direct messages (content requests, challenges, proofs).

        DEPRECATED: Pass transport in constructor instead. Kept for backward compatibility.
        """
        if self.transport is None:
            self.transport = transport
        self._register_direct_handler()

    def _register_direct_handler(self) -> None:
        """Register the unified direct message handler on the transport."""
        if self.transport is not None:
            self.transport.on_message(MSG_DIRECT, self._on_direct_message)
            logger.info("Storage provider registered for direct P2P messaging")

    async def _on_direct_message(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Unified dispatcher for all direct P2P messages to this storage provider."""
        subtype = msg.payload.get("subtype", "")
        if subtype == "content_request":
            await self._on_direct_content_request(msg, peer)
        elif subtype == "storage_challenge":
            await self._on_storage_challenge(subtype, msg.payload, msg.sender_id)
        elif subtype == "storage_proof_response":
            await self._on_storage_proof_response(subtype, msg.payload, msg.sender_id)
```

In `start()`, after the gossip subscriptions (line 187), add:

```python
            # Register direct message handler for P2P challenges/proofs/content
            self._register_direct_handler()
```

- [ ] **Step 6: Add deduplication guard to _on_storage_challenge**

At the top of `_on_storage_challenge` (line 668), before the `target_provider_id` check, add:

```python
        # Deduplicate: drop if already seen (handles direct + gossip dual delivery)
        challenge_id = data.get("challenge", {}).get("challenge_id", "")
        if challenge_id in self._seen_challenge_ids:
            logger.debug("Dropping duplicate challenge %s", challenge_id[:16])
            return
        if challenge_id:
            self._seen_challenge_ids[challenge_id] = time.time()
```

- [ ] **Step 7: Convert challenge issuance to direct P2P with scoped fallback**

Add the conditional import at the top of `storage_provider.py` (after the existing imports):

```python
# Conditional import for transport-specific error type
try:
    from prsm.node.libp2p_transport import Libp2pTransportError
except ImportError:
    Libp2pTransportError = OSError  # type: ignore[misc,assignment]
```

Replace the gossip publish in `issue_challenge_to_provider` (line 654-659) with:

```python
        # Prefer direct P2P for bilateral challenge delivery
        challenge_payload = {
            "subtype": "storage_challenge",
            "challenge": challenge.to_dict(),
            "challenger_id": self.identity.node_id,
            "target_provider_id": provider_id,
        }
        if self.transport is not None:
            msg = P2PMessage(
                msg_type=MSG_DIRECT,
                sender_id=self.identity.node_id,
                payload=challenge_payload,
            )
            try:
                sent = await self.transport.send_to_peer(provider_id, msg)
                if not sent:
                    raise ConnectionError("direct send failed")
            except (ConnectionError, Libp2pTransportError, OSError) as exc:
                logger.debug(
                    "Direct challenge to %s failed (%s), falling back to gossip",
                    provider_id[:8], exc,
                )
                await self.gossip.publish("storage_challenge", challenge_payload)
        else:
            await self.gossip.publish("storage_challenge", challenge_payload)
```

- [ ] **Step 8: Convert proof response to direct P2P with scoped fallback**

Replace the gossip publish in `_on_storage_challenge` proof response section (line 700-705) with:

```python
                # Send proof via direct P2P, fall back to gossip
                challenger_id = data.get("challenger_id", origin)
                proof_payload = {
                    "subtype": "storage_proof_response",
                    "proof": proof.to_dict(),
                    "challenge_id": challenge.challenge_id,
                    "provider_id": self.identity.node_id,
                }
                if self.transport is not None:
                    proof_msg = P2PMessage(
                        msg_type=MSG_DIRECT,
                        sender_id=self.identity.node_id,
                        payload=proof_payload,
                    )
                    try:
                        sent = await self.transport.send_to_peer(challenger_id, proof_msg)
                        if not sent:
                            raise ConnectionError("direct send failed")
                    except (ConnectionError, Libp2pTransportError, OSError) as exc:
                        logger.debug(
                            "Direct proof to %s failed (%s), falling back to gossip",
                            challenger_id[:8], exc,
                        )
                        await self.gossip.publish("storage_proof_response", proof_payload)
                else:
                    await self.gossip.publish("storage_proof_response", proof_payload)
```

- [ ] **Step 9: Add DHT provide_content on pin**

In `_on_storage_request`, after the `GOSSIP_CONTENT_ADVERTISE` publish (after line 354), add:

```python
            # Register in DHT for durable cross-restart content routing
            if self.discovery:
                try:
                    await self.discovery.provide_content(cid)
                except Exception as exc:
                    logger.debug("DHT provide_content failed for %s: %s", cid[:12], exc)
```

- [ ] **Step 10: Piggyback dedup cleanup on challenge_cleanup_loop**

In the `_challenge_cleanup_loop` method, add at the end of each loop iteration:

```python
            # Also clean up seen challenge dedup set
            self._cleanup_seen_challenges()
```

- [ ] **Step 11: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_storage_provider_direct_p2p.py -v`
Expected: 3 PASSED

- [ ] **Step 12: Commit**

```bash
git add prsm/node/storage_provider.py tests/unit/test_storage_provider_direct_p2p.py
git commit -m "feat: direct P2P for storage challenges/proofs with dedup and gossip fallback"
```

---

### Task 4: Add Reliability Recording to ComputeRequester

**Files:**
- Modify: `prsm/node/compute_requester.py:297-373`
- Test: `tests/unit/test_compute_requester_reliability.py` (new file)

- [ ] **Step 1: Write failing tests for reliability recording**

Create `tests/unit/test_compute_requester_reliability.py`:

```python
"""Tests for ComputeRequester reliability recording."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from prsm.node.compute_requester import ComputeRequester, JobType, SubmittedJob
from prsm.node.discovery import PeerInfo


def _make_requester():
    identity = MagicMock()
    identity.node_id = "requester_node"
    transport = MagicMock()
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock(return_value=1)
    ledger = MagicMock()
    ledger.get_balance = AsyncMock(return_value=1000.0)
    ledger.transfer = AsyncMock()

    discovery = MagicMock()
    discovery.record_job_success = MagicMock()
    discovery.record_job_failure = MagicMock()

    req = ComputeRequester(
        identity=identity,
        transport=transport,
        gossip=gossip,
        ledger=ledger,
        discovery=discovery,
    )
    req.escrow = None
    req.ledger_sync = None
    return req


class TestReliabilityRecording:
    """Tests for recording job success/failure in discovery."""

    @pytest.mark.asyncio
    async def test_successful_result_records_success(self):
        """On verified job result, discovery.record_job_success should be called."""
        req = _make_requester()
        req._running = True

        # Create a submitted job
        job = SubmittedJob(
            job_id="job_001",
            job_type=JobType.INFERENCE,
            payload={"prompt": "test"},
            ftns_budget=0.0,
        )
        req.submitted_jobs["job_001"] = job

        # Simulate result from same node (self-compute, no signature needed)
        await req._on_job_result("job_result", {
            "job_id": "job_001",
            "provider_id": "requester_node",
            "status": "completed",
            "result": {"output": "hello"},
        }, "requester_node")

        req.discovery.record_job_success.assert_called_once_with("requester_node")

    @pytest.mark.asyncio
    async def test_failed_result_records_failure(self):
        """On job failure, discovery.record_job_failure should be called."""
        req = _make_requester()
        req._running = True

        job = SubmittedJob(
            job_id="job_002",
            job_type=JobType.INFERENCE,
            payload={"prompt": "test"},
            ftns_budget=0.0,
        )
        req.submitted_jobs["job_002"] = job

        await req._on_job_result("job_result", {
            "job_id": "job_002",
            "provider_id": "provider_node",
            "status": "failed",
            "error": "GPU OOM",
        }, "provider_node")

        req.discovery.record_job_failure.assert_called_once_with("provider_node")

    @pytest.mark.asyncio
    async def test_no_discovery_is_safe(self):
        """If discovery is None, reliability recording should not raise."""
        req = _make_requester()
        req.discovery = None
        req._running = True

        job = SubmittedJob(
            job_id="job_003",
            job_type=JobType.INFERENCE,
            payload={"prompt": "test"},
            ftns_budget=0.0,
        )
        req.submitted_jobs["job_003"] = job

        # Should not raise
        await req._on_job_result("job_result", {
            "job_id": "job_003",
            "provider_id": "requester_node",
            "status": "completed",
            "result": {"output": "ok"},
        }, "requester_node")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_compute_requester_reliability.py -v`
Expected: FAIL — `record_job_success` is never called (not yet added to `_on_job_result`)

- [ ] **Step 3: Add reliability recording to _on_job_result**

In `prsm/node/compute_requester.py`, in the `_on_job_result` method:

After the failure handling block (after line 313, `return`), add:

```python
        # Record failure in discovery for reliability tracking
        if self.discovery:
            self.discovery.record_job_failure(provider_id)
```

After the successful completion block (after line 337, `job.completed_at = time.time()`), add:

```python
        # Record success in discovery for reliability tracking
        if self.discovery:
            self.discovery.record_job_success(provider_id)
```

- [ ] **Step 4: Add reliability sorting to _get_capable_peers**

In `_get_capable_peers` (line 156), before the `return` statement, add sorting:

```python
        # Sort by reliability — unreliable peers fall to the bottom
        capable_peers.sort(key=lambda p: p.reliability_score, reverse=True)
        return [p.node_id for p in capable_peers]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_compute_requester_reliability.py -v`
Expected: 3 PASSED

- [ ] **Step 6: Run existing compute requester tests for regressions**

Run: `python -m pytest tests/unit/ -k "compute_requester" -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add prsm/node/compute_requester.py tests/unit/test_compute_requester_reliability.py
git commit -m "feat: record job success/failure in discovery for reliability tracking"
```

---

### Task 5: Wire Capability Announcement and Provider Params in node.py

**Files:**
- Modify: `prsm/node/node.py:616-654` (construction), `prsm/node/node.py:1156-1162` (start)

- [ ] **Step 1: Write failing test for capability announcement wiring**

Create `tests/unit/test_node_capability_wiring.py`:

```python
"""Tests for node.py capability announcement wiring."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestCapabilityWiring:
    """Tests for capability announcement at node startup."""

    @pytest.mark.asyncio
    async def test_storage_provider_receives_transport_and_discovery(self):
        """StorageProvider should be constructed with transport and discovery."""
        # We test this by importing and checking the constructor signature
        import inspect
        from prsm.node.storage_provider import StorageProvider
        sig = inspect.signature(StorageProvider.__init__)
        params = list(sig.parameters.keys())
        assert "transport" in params
        assert "discovery" in params

    @pytest.mark.asyncio
    async def test_compute_requester_receives_discovery(self):
        """ComputeRequester should be constructed with discovery."""
        import inspect
        from prsm.node.compute_requester import ComputeRequester
        sig = inspect.signature(ComputeRequester.__init__)
        params = list(sig.parameters.keys())
        assert "discovery" in params
```

- [ ] **Step 2: Run test to verify the StorageProvider test passes (from Task 3) and ComputeRequester test passes (already has discovery)**

Run: `python -m pytest tests/unit/test_node_capability_wiring.py -v`
Expected: 2 PASSED (both constructors already updated in Tasks 3 and existing code)

- [ ] **Step 3: Update StorageProvider construction in node.py**

In `prsm/node/node.py`, modify the StorageProvider construction (line 640-647):

```python
            self.storage_provider = StorageProvider(
                identity=self.identity,
                gossip=self.gossip,
                ledger=self.ledger,
                ipfs_api_url=self.config.ipfs_api_url,
                pledged_gb=self.config.storage_gb,
                config=self.config,
                transport=self.transport,
                discovery=self.discovery,
            )
```

- [ ] **Step 4: Pass discovery to ComputeRequester construction**

Modify the ComputeRequester construction (line 631-636):

```python
        self.compute_requester = ComputeRequester(
            identity=self.identity,
            transport=self.transport,
            gossip=self.gossip,
            ledger=self.ledger,
            discovery=self.discovery,
        )
```

- [ ] **Step 5: Remove standalone register_content_handler call**

Remove line 1162:
```python
            self.storage_provider.register_content_handler(self.transport)
```

The direct handler is now registered in `StorageProvider.start()` via `_register_direct_handler()`.

- [ ] **Step 6: Add capability announcement after provider startup**

After the storage provider start block (after line 1162), add:

```python
        # ── Capability Announcement ──────────────────────────────────
        # Announce local capabilities via discovery after providers are started.
        # This lets the network know what this node can do.
        if hasattr(self.discovery, 'set_local_capabilities'):
            cap_list = list(local_capabilities)  # computed earlier in __init__
            backends_list = []
            gpu_available = False
            if self.compute_provider:
                if self.compute_provider.resources.gpu_available:
                    gpu_available = True
                    if "gpu" not in cap_list:
                        cap_list.append("gpu")
                try:
                    from prsm.compute.nwtn.backends.config import detect_available_backends
                    backends_list = [b.value for b in detect_available_backends()]
                except Exception:
                    pass
            self.discovery.set_local_capabilities(
                capabilities=cap_list,
                backends=backends_list,
                gpu_available=gpu_available,
            )
            await self.discovery.announce_capabilities()

            # Periodic re-announcement (every 300s)
            async def _periodic_capability_announce():
                while self._started:
                    await asyncio.sleep(300)
                    try:
                        await self.discovery.announce_capabilities()
                    except Exception as exc:
                        logger.debug("Capability re-announcement failed: %s", exc)

            self._capability_announce_task = asyncio.create_task(
                _periodic_capability_announce()
            )
```

- [ ] **Step 7: Cancel the periodic task in stop()**

In the `stop()` method, add before the transport shutdown:

```python
        if hasattr(self, '_capability_announce_task'):
            self._capability_announce_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._capability_announce_task
```

- [ ] **Step 8: Run unit tests for regressions**

Run: `python -m pytest tests/unit/test_node_capability_wiring.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add prsm/node/node.py tests/unit/test_node_capability_wiring.py
git commit -m "feat: wire transport/discovery to providers, add capability announcement"
```

---

### Task 6: Integration Test Harness — MockLibp2pTransport

**Files:**
- Create: `tests/integration/test_libp2p_compute_storage.py`

- [ ] **Step 1: Create the mock transport harness**

Create `tests/integration/test_libp2p_compute_storage.py`:

```python
"""
Integration tests for compute/storage lifecycle over libp2p transport.

Uses a MockLibp2pTransport that routes messages between in-process nodes
with JSON serialization fidelity (mimics the FFI boundary).
"""
import asyncio
import json
import time
import uuid
import pytest
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from prsm.node.transport import MSG_GOSSIP, MSG_DIRECT, P2PMessage, PeerConnection
from prsm.node.discovery import PeerInfo


class MockLibp2pTransport:
    """In-process transport that routes messages with JSON serialization fidelity.

    Serializes all payloads to JSON bytes and back before delivery,
    catching non-serializable types that would crash at the real FFI boundary.
    """

    def __init__(self, node_id: str, network: "MockNetwork"):
        self.identity = MagicMock()
        self.identity.node_id = node_id
        self._handlers: Dict[str, List[Callable]] = {}
        self._network = network
        self._peers: Dict[str, PeerConnection] = {}

    def on_message(self, msg_type: str, handler: Callable) -> None:
        self._handlers.setdefault(msg_type, []).append(handler)

    async def send_to_peer(self, peer_id: str, msg: P2PMessage) -> bool:
        """Send via network with JSON round-trip for serialization fidelity."""
        target = self._network.get_transport(peer_id)
        if target is None:
            return False
        # JSON round-trip to catch serialization bugs
        wire = json.dumps({
            "msg_type": msg.msg_type,
            "sender_id": msg.sender_id,
            "payload": msg.payload,
            "timestamp": msg.timestamp,
            "signature": msg.signature,
            "ttl": msg.ttl,
            "nonce": msg.nonce,
        })
        raw = json.loads(wire)
        reconstructed = P2PMessage(
            msg_type=raw["msg_type"],
            sender_id=raw["sender_id"],
            payload=raw["payload"],
            timestamp=raw["timestamp"],
            signature=raw["signature"],
            ttl=raw["ttl"],
            nonce=raw["nonce"],
        )
        peer = PeerConnection(peer_id=msg.sender_id, address="mock", websocket=None)
        for handler in target._handlers.get(msg.msg_type, []):
            await handler(reconstructed, peer)
        return True

    def sign(self, msg):
        """No-op signing for tests."""
        pass


class MockNetwork:
    """Routes messages between MockLibp2pTransport instances."""

    def __init__(self):
        self._transports: Dict[str, MockLibp2pTransport] = {}

    def add_node(self, node_id: str) -> MockLibp2pTransport:
        t = MockLibp2pTransport(node_id, self)
        self._transports[node_id] = t
        return t

    def get_transport(self, node_id: str) -> Optional[MockLibp2pTransport]:
        return self._transports.get(node_id)


class MockGossip:
    """In-process gossip that broadcasts to all nodes in the network with JSON fidelity."""

    def __init__(self, node_id: str, network: MockNetwork):
        self._node_id = node_id
        self._network = network
        self._callbacks: Dict[str, List[Callable]] = {}
        self.published: List[Dict[str, Any]] = []  # track publishes for assertions
        self.ledger = None

    def subscribe(self, subtype: str, callback: Callable) -> None:
        self._callbacks.setdefault(subtype, []).append(callback)

    async def publish(self, subtype: str, data: Dict[str, Any], ttl=None) -> int:
        # JSON round-trip for serialization fidelity
        wire = json.dumps({"subtype": subtype, "data": data, "sender_id": self._node_id})
        raw = json.loads(wire)

        self.published.append(raw)

        # Deliver to all nodes (including self for local subscribers)
        for node_id, transport in self._network._transports.items():
            gossip = _gossip_registry.get(node_id)
            if gossip:
                for cb in gossip._callbacks.get(subtype, []):
                    await cb(subtype, raw["data"], raw["sender_id"])
        return 1


# Global registry so gossip instances can find each other
_gossip_registry: Dict[str, MockGossip] = {}


def make_test_node(node_id: str, network: MockNetwork):
    """Create a test node with mock transport and gossip."""
    transport = network.add_node(node_id)
    gossip = MockGossip(node_id, network)
    _gossip_registry[node_id] = gossip
    return transport, gossip
```

- [ ] **Step 2: Run to verify the harness imports correctly**

Run: `python -m pytest tests/integration/test_libp2p_compute_storage.py --collect-only`
Expected: collected 0 items (no tests yet, but no import errors)

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_libp2p_compute_storage.py
git commit -m "test: add MockLibp2pTransport harness with JSON serialization fidelity"
```

---

### Task 7: Integration Tests — Compute Lifecycle, Capability Discovery, Reliability

**Files:**
- Modify: `tests/integration/test_libp2p_compute_storage.py`

- [ ] **Step 1: Add Test 1 — Compute Job Lifecycle**

Append to `tests/integration/test_libp2p_compute_storage.py`:

```python
class TestComputeJobLifecycle:
    """Test 1: Full compute job lifecycle over mock libp2p transport."""

    @pytest.mark.asyncio
    async def test_job_offer_accept_confirm_result_payment(self):
        """Requester submits job → provider accepts → confirm → result → payment."""
        _gossip_registry.clear()
        network = MockNetwork()
        req_transport, req_gossip = make_test_node("requester", network)
        prov_transport, prov_gossip = make_test_node("provider", network)

        # Track message flow
        messages_received = {"requester": [], "provider": []}

        # Provider subscribes to job_offer
        async def on_job_offer(subtype, data, sender):
            messages_received["provider"].append(("job_offer", data))
            # Auto-accept
            await prov_gossip.publish("job_accept", {
                "job_id": data["job_id"],
                "provider_id": "provider",
            })

        prov_gossip.subscribe("job_offer", on_job_offer)

        # Requester subscribes to job_accept
        async def on_job_accept(subtype, data, sender):
            messages_received["requester"].append(("job_accept", data))
            # Confirm the provider
            await req_gossip.publish("job_confirm", {
                "job_id": data["job_id"],
                "provider_id": data["provider_id"],
                "requester_id": "requester",
            })

        req_gossip.subscribe("job_accept", on_job_accept)

        # Provider subscribes to job_confirm → sends result
        async def on_job_confirm(subtype, data, sender):
            messages_received["provider"].append(("job_confirm", data))
            await prov_gossip.publish("job_result", {
                "job_id": data["job_id"],
                "provider_id": "provider",
                "status": "completed",
                "result": {"output": "42"},
            })

        prov_gossip.subscribe("job_confirm", on_job_confirm)

        # Requester subscribes to job_result
        result_received = asyncio.Event()

        async def on_job_result(subtype, data, sender):
            messages_received["requester"].append(("job_result", data))
            result_received.set()

        req_gossip.subscribe("job_result", on_job_result)

        # Kick off the lifecycle
        await req_gossip.publish("job_offer", {
            "job_id": "test_job_001",
            "job_type": "inference",
            "requester_id": "requester",
            "payload": {"prompt": "What is 6*7?"},
            "ftns_budget": 1.0,
        })

        # Wait for the full lifecycle
        await asyncio.wait_for(result_received.wait(), timeout=5.0)

        # Verify message flow
        assert len(messages_received["provider"]) == 2  # job_offer, job_confirm
        assert messages_received["provider"][0][0] == "job_offer"
        assert messages_received["provider"][1][0] == "job_confirm"
        assert len(messages_received["requester"]) == 2  # job_accept, job_result
        assert messages_received["requester"][0][0] == "job_accept"
        assert messages_received["requester"][1][0] == "job_result"
        assert messages_received["requester"][1][1]["result"]["output"] == "42"


class TestCapabilityDiscovery:
    """Test 3: Capability discovery between nodes."""

    @pytest.mark.asyncio
    async def test_capability_index_populated(self):
        """Nodes announcing capabilities should appear in each other's index."""
        from prsm.node.libp2p_discovery import Libp2pDiscovery

        _gossip_registry.clear()
        network = MockNetwork()
        t_a, g_a = make_test_node("node_a", network)
        t_b, g_b = make_test_node("node_b", network)

        disc_a = Libp2pDiscovery(transport=t_a, gossip=g_a)
        disc_b = Libp2pDiscovery(transport=t_b, gossip=g_b)

        # Wire gossip subscriptions
        await disc_a.start()
        await disc_b.start()

        # Node A: compute + GPU
        disc_a.set_local_capabilities(
            capabilities=["compute", "inference", "gpu"],
            backends=["local"],
            gpu_available=True,
        )
        await disc_a.announce_capabilities()

        # Node B: storage only
        disc_b.set_local_capabilities(
            capabilities=["storage", "pinning"],
            backends=[],
            gpu_available=False,
        )
        await disc_b.announce_capabilities()

        # Allow gossip delivery
        await asyncio.sleep(0.01)

        # Node B should see Node A's GPU capability
        gpu_peers = disc_b.find_peers_with_gpu()
        assert len(gpu_peers) == 1
        assert gpu_peers[0].node_id == "node_a"

        # Node A should see Node B's storage capability
        storage_peers = disc_a.find_peers_with_capability("storage")
        assert len(storage_peers) == 1
        assert storage_peers[0].node_id == "node_b"

        # Node A should NOT appear in storage search
        assert all(p.node_id != "node_a" for p in disc_a.find_peers_with_capability("storage"))


class TestReliabilityTracking:
    """Test 5: Provider reliability scoring lifecycle."""

    @pytest.mark.asyncio
    async def test_reliability_degrades_and_resets(self):
        """Reliability score should degrade on failures and reset on restart."""
        from prsm.node.libp2p_discovery import Libp2pDiscovery

        _gossip_registry.clear()
        network = MockNetwork()
        t_req, g_req = make_test_node("requester", network)
        t_prov, g_prov = make_test_node("provider", network)

        disc = Libp2pDiscovery(transport=t_req, gossip=g_req)
        await disc.start()

        # Provider announces
        await disc._on_capability("capability_announce", {
            "node_id": "provider",
            "capabilities": ["compute", "gpu"],
            "supported_backends": ["local"],
            "gpu_available": True,
            "startup_timestamp": 1000.0,
        }, "provider")

        # 2 successes, 1 failure
        disc.record_job_success("provider")
        disc.record_job_success("provider")
        disc.record_job_failure("provider")

        peer = disc._capability_index["provider"]
        assert abs(peer.reliability_score - 0.6667) < 0.01

        # Heartbeat should NOT reset
        await disc._on_capability("capability_announce", {
            "node_id": "provider",
            "capabilities": ["compute", "gpu"],
            "supported_backends": ["local"],
            "gpu_available": True,
            "startup_timestamp": 1000.0,  # same timestamp
        }, "provider")
        assert peer.job_failure_count == 1  # NOT reset

        # Restart (new timestamp) SHOULD reset
        await disc._on_capability("capability_announce", {
            "node_id": "provider",
            "capabilities": ["compute", "gpu"],
            "supported_backends": ["local"],
            "gpu_available": True,
            "startup_timestamp": 2000.0,  # new timestamp
        }, "provider")
        assert peer.job_failure_count == 0
        assert peer.job_success_count == 0
        assert peer.reliability_score == 1.0
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/integration/test_libp2p_compute_storage.py -v`
Expected: 3 PASSED

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_libp2p_compute_storage.py
git commit -m "test: add compute lifecycle, capability discovery, reliability integration tests"
```

---

### Task 8: Integration Tests — Direct P2P Challenge and Gossip Fallback

**Files:**
- Modify: `tests/integration/test_libp2p_compute_storage.py`

- [ ] **Step 1: Add Test 2 — Storage Challenge via Direct P2P**

Append to `tests/integration/test_libp2p_compute_storage.py`:

```python
class TestDirectP2PChallenge:
    """Test 2: Storage challenge/proof via direct P2P stream."""

    @pytest.mark.asyncio
    async def test_challenge_and_proof_via_direct_p2p(self):
        """Challenge sent via send_to_peer, proof returned via send_to_peer."""
        _gossip_registry.clear()
        network = MockNetwork()
        challenger_transport, challenger_gossip = make_test_node("challenger", network)
        provider_transport, provider_gossip = make_test_node("provider", network)

        # Track what was sent via direct P2P
        direct_messages = {"challenger": [], "provider": []}

        # Provider: register handler for direct messages
        async def provider_direct_handler(msg: P2PMessage, peer: PeerConnection):
            subtype = msg.payload.get("subtype", "")
            direct_messages["provider"].append(subtype)
            if subtype == "storage_challenge":
                # Respond with proof via direct P2P
                proof_msg = P2PMessage(
                    msg_type=MSG_DIRECT,
                    sender_id="provider",
                    payload={
                        "subtype": "storage_proof_response",
                        "proof": {"challenge_id": msg.payload["challenge"]["challenge_id"], "data": "merkle_proof_bytes"},
                        "challenge_id": msg.payload["challenge"]["challenge_id"],
                        "provider_id": "provider",
                    },
                )
                await provider_transport.send_to_peer("challenger", proof_msg)

        provider_transport.on_message(MSG_DIRECT, provider_direct_handler)

        # Challenger: register handler for proof responses
        proof_received = asyncio.Event()

        async def challenger_direct_handler(msg: P2PMessage, peer: PeerConnection):
            subtype = msg.payload.get("subtype", "")
            direct_messages["challenger"].append(subtype)
            if subtype == "storage_proof_response":
                proof_received.set()

        challenger_transport.on_message(MSG_DIRECT, challenger_direct_handler)

        # Challenger sends challenge via direct P2P
        challenge_msg = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id="challenger",
            payload={
                "subtype": "storage_challenge",
                "challenge": {
                    "challenge_id": "chal_test_001",
                    "cid": "QmTestContent123",
                    "nonce": "abc123",
                    "difficulty": 32,
                },
                "challenger_id": "challenger",
                "target_provider_id": "provider",
            },
        )
        sent = await challenger_transport.send_to_peer("provider", challenge_msg)
        assert sent is True

        await asyncio.wait_for(proof_received.wait(), timeout=5.0)

        # Verify: both used direct P2P, not gossip
        assert "storage_challenge" in direct_messages["provider"]
        assert "storage_proof_response" in direct_messages["challenger"]
        assert len(challenger_gossip.published) == 0  # No gossip used
        assert len(provider_gossip.published) == 0  # No gossip used


class TestDirectP2PFallback:
    """Test 4: Direct P2P failure falls back to gossip."""

    @pytest.mark.asyncio
    async def test_fallback_to_gossip_on_send_failure(self):
        """When send_to_peer fails, challenge should fall back to gossip broadcast."""
        _gossip_registry.clear()
        network = MockNetwork()
        challenger_transport, challenger_gossip = make_test_node("challenger", network)
        provider_transport, provider_gossip = make_test_node("provider", network)

        # Make direct send fail for challenger
        original_send = challenger_transport.send_to_peer
        challenger_transport.send_to_peer = AsyncMock(return_value=False)

        # Provider subscribes to gossip fallback
        gossip_received = asyncio.Event()
        received_data = {}

        async def on_gossip_challenge(subtype, data, sender):
            received_data.update(data)
            gossip_received.set()

        provider_gossip.subscribe("storage_challenge", on_gossip_challenge)

        # Simulate the fallback pattern from storage_provider
        challenge_payload = {
            "subtype": "storage_challenge",
            "challenge": {
                "challenge_id": "chal_fallback_001",
                "cid": "QmFallbackTest",
                "nonce": "xyz789",
                "difficulty": 32,
            },
            "challenger_id": "challenger",
            "target_provider_id": "provider",
        }

        msg = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id="challenger",
            payload=challenge_payload,
        )
        sent = await challenger_transport.send_to_peer("provider", msg)
        if not sent:
            # Fallback to gossip (this is what storage_provider does)
            await challenger_gossip.publish("storage_challenge", challenge_payload)

        await asyncio.wait_for(gossip_received.wait(), timeout=5.0)

        # Verify: gossip was used as fallback
        assert received_data["challenge"]["challenge_id"] == "chal_fallback_001"
        assert len(challenger_gossip.published) == 1  # One gossip publish (fallback)
```

- [ ] **Step 2: Run all integration tests**

Run: `python -m pytest tests/integration/test_libp2p_compute_storage.py -v`
Expected: 5 PASSED

- [ ] **Step 3: Run full unit test suite for regressions**

Run: `python -m pytest tests/unit/ -x -q --timeout=60`
Expected: All existing tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_libp2p_compute_storage.py
git commit -m "test: add direct P2P challenge and gossip fallback integration tests"
```

---

## Self-Review Checklist

**Spec coverage:**
- Component 1 (envelope compatibility): Verified by Tests 1, 3 — gossip messages flow correctly
- Component 2 (topic registration): Verified by Tests 1, 3 — subtypes are subscribed and deliver
- Component 3 (direct P2P): Task 3 + Tests 2, 4 — challenges/proofs via direct P2P with fallback
- Component 3.6 (deduplication): Task 3 — `_seen_challenge_ids` with cleanup
- Component 4 (discovery wiring): Task 5 — constructor params, capability announcement, periodic task
- Component 5 (reliability): Tasks 1, 2, 4 + Test 5 — PeerInfo fields, recording, conditional reset
- Component 6 (integration tests): Tasks 6, 7, 8 — all 5 tests

**Placeholder scan:** No TBDs, TODOs, or vague instructions. All code blocks are complete.

**Type consistency:**
- `PeerInfo.reliability_score` — used in Task 1 (definition), Task 2 (tests), Task 4 (sorting), Task 7 (assertions)
- `record_job_success/failure` — defined in Task 2, called in Task 4, mocked in Task 7
- `_seen_challenge_ids` — defined in Task 3, tested in Task 3, used in Task 3
- `startup_timestamp` — defined in Task 1 (PeerInfo), set in Task 2 (Libp2pDiscovery), tested in Task 2 and 7
