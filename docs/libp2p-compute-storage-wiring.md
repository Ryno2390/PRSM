# Compute/Storage ↔ libp2p Transport Wiring

## Goal

Wire the existing compute provider, storage provider, content economy, and storage proof systems through the new libp2p transport layer. Convert bilateral operations (storage challenges, proof responses) from gossip broadcast to direct P2P messaging. Prove the full lifecycle works with integration tests.

## Architecture

The libp2p transport layer (shipped in v1.2.0) provides drop-in replacements for the WebSocket transport: `Libp2pTransport`, `Libp2pGossip`, and `Libp2pDiscovery`. Higher-level systems (ComputeProvider, StorageProvider, etc.) already call `gossip.subscribe()` and `gossip.publish()` — these APIs are identical across both backends. The wiring work connects the missing pieces: capability announcements, DHT content registration, direct P2P for targeted operations, and integration testing.

## Scope

**In scope:**
- Message envelope compatibility verification (no changes expected)
- GossipSub topic registration verification for all ~20 subtypes
- Convert storage challenges and proof responses to direct P2P via `transport.send_to_peer()`
- Wire `Libp2pDiscovery` capability announcements into node startup
- Wire DHT `provide_content()` into storage provider pin flow
- Wire capability-based peer lookup into compute requester
- Integration tests for full compute and storage lifecycles over libp2p

**Out of scope:**
- Scored provider selection (keeping first-accept-wins)
- New stream protocols beyond `/prsm/direct/1.0.0`
- Changes to the Go shared library
- Changes to the WebSocket fallback path

---

## Component 1: Message Envelope Compatibility

**Status: Already compatible — no changes needed.**

The old `GossipProtocol` and new `Libp2pGossip` expose identical public APIs:
- `subscribe(subtype: str, callback: GossipCallback)` — same callback signature `(str, Dict[str, Any], str)`
- `publish(subtype: str, data: Dict[str, Any], ttl: Optional[int] = None) -> int`
- `start()` / `stop()` lifecycle methods
- `ledger` attribute set post-construction by `node.py`

Both pass the originating node's ID as the third callback argument. The publish return value (int count) has the same semantics.

**Verification:** Integration tests will confirm callbacks receive the expected data shape.

---

## Component 2: GossipSub Topic Registration

**Status: Already handled by lazy subscription — no changes needed.**

`Libp2pGossip.subscribe()` calls `PrsmSubscribe` once per unique subtype, creating the GossipSub topic `prsm/{subtype}`. All compute/storage subtypes are subscribed at startup in their respective `start()` methods, well before any messages arrive.

**Full subtype map:**

| System | Subscribes To | Publishes |
|--------|--------------|-----------|
| ComputeProvider | `job_offer`, `job_confirm`, `job_cancel` | `job_accept`, `job_result` |
| ComputeRequester | `job_accept`, `job_result` | `job_offer`, `job_confirm`, `payment_confirm` |
| StorageProvider | `storage_request` | `storage_confirm`, `content_advertise`, `proof_of_storage` |
| ContentEconomy | `retrieval_bid`, `retrieval_fulfill` | `retrieval_request`, `storage_request` |
| Discovery | `capability_announce`, `shard_available` | `capability_announce`, `shard_available` |

**Note:** `storage_challenge` and `storage_proof_response` move primarily to direct P2P (see Component 3), but gossip subscriptions are retained as fallback receivers in case direct connections fail.

**Verification:** Integration tests will confirm all subtypes are subscribed and messages flow correctly.

---

## Component 3: Direct P2P for Bilateral Operations

**What changes:** Storage challenges and proof responses move from gossip broadcast to direct peer-to-peer messaging via `transport.send_to_peer()`.

**Why:** These are bilateral by nature — a challenge targets one specific provider, and the proof goes back to one specific challenger. Broadcasting them wastes bandwidth and leaks information.

### 3.1 Storage Provider Constructor Change

Add `transport` as a constructor parameter to `StorageProvider`. Currently it receives `transport` only via the post-construction `register_content_handler()` call at `node.py:1162`. The constructor should accept `transport` directly, and the `register_content_handler()` call should be removed (content handler registration moves into `start()`).

### 3.2 Issuing Challenges (Direct P2P)

Current (`storage_provider.py:655`):
```python
await self.gossip.publish("storage_challenge", {
    "challenge": challenge.to_dict(),
    "challenger_id": self.identity.node_id,
    "target_provider_id": provider_id,
})
```

New:
```python
msg = P2PMessage(
    msg_type=MSG_DIRECT,
    sender_id=self.identity.node_id,
    payload={
        "subtype": "storage_challenge",
        "challenge": challenge.to_dict(),
        "challenger_id": self.identity.node_id,
        "target_provider_id": provider_id,
    },
)
try:
    sent = await self.transport.send_to_peer(provider_id, msg)
    if not sent:
        raise ConnectionError("direct send failed")
except Exception:
    # Fallback to gossip if direct connection unavailable
    await self.gossip.publish("storage_challenge", {
        "challenge": challenge.to_dict(),
        "challenger_id": self.identity.node_id,
        "target_provider_id": provider_id,
    })
```

### 3.3 Sending Proof Responses (Direct P2P)

Current (`storage_provider.py:701`):
```python
await self.gossip.publish("storage_proof_response", {
    "proof": proof.to_dict(),
    "challenge_id": challenge.challenge_id,
    "provider_id": self.identity.node_id,
})
```

New:
```python
challenger_id = data.get("challenger_id", origin)
msg = P2PMessage(
    msg_type=MSG_DIRECT,
    sender_id=self.identity.node_id,
    payload={
        "subtype": "storage_proof_response",
        "proof": proof.to_dict(),
        "challenge_id": challenge.challenge_id,
        "provider_id": self.identity.node_id,
    },
)
try:
    sent = await self.transport.send_to_peer(challenger_id, msg)
    if not sent:
        raise ConnectionError("direct send failed")
except Exception:
    await self.gossip.publish("storage_proof_response", {
        "proof": proof.to_dict(),
        "challenge_id": challenge.challenge_id,
        "provider_id": self.identity.node_id,
    })
```

### 3.4 Receiving Challenges and Proofs (Direct Message Handler)

The storage provider already has a direct message handler pattern via `_on_direct_content_request`. We add a unified direct message dispatcher:

```python
async def _on_direct_message(self, msg: P2PMessage, peer: PeerConnection) -> None:
    subtype = msg.payload.get("subtype", "")
    if subtype == "content_request":
        await self._on_direct_content_request(msg, peer)
    elif subtype == "storage_challenge":
        await self._on_storage_challenge(subtype, msg.payload, msg.sender_id)
    elif subtype == "storage_proof_response":
        await self._on_storage_proof_response(subtype, msg.payload, msg.sender_id)
```

This is registered once at startup via `transport.on_message(MSG_DIRECT, self._on_direct_message)`.

### 3.5 Gossip Fallback for Challenges

The gossip subscriptions for `storage_challenge` and `storage_proof_response` are kept as fallback receivers. If a direct send fails and the sender falls back to gossip, the receiver still picks up the message through the gossip subscription. The handler is already idempotent (checks `target_provider_id`), so receiving both direct and gossip copies is safe.

### 3.6 What Stays on Gossip

- `GOSSIP_JOB_CONFIRM` — broadcast so all competing providers know they lost
- `GOSSIP_PAYMENT_CONFIRM` — broadcast for network-wide ledger consensus
- All other job/storage/content subtypes — naturally broadcast (one-to-many)

---

## Component 4: Discovery ↔ Provider Capability Wiring

**What changes:** Three wiring gaps in `node.py` that prevent the libp2p discovery layer from functioning.

### 4.1 Capability Announcement at Startup

After compute and storage providers are initialized (and we know what capabilities the node has), `node.py` should call:

```python
# Build capability list from active providers
capabilities = []
backends = []
gpu_available = False

if self.compute_provider:
    capabilities.append("compute")
    capabilities.append("inference")
    capabilities.append("embedding")
    if self.compute_provider.resources.gpu_available:
        gpu_available = True
        capabilities.append("gpu")
    # Collect supported backends via existing detection
    from prsm.compute.nwtn.backends.config import detect_available_backends
    backends = [b.value for b in detect_available_backends()]

if self.storage_provider and self.storage_provider._running:
    capabilities.append("storage")
    capabilities.append("pinning")

self.discovery.set_local_capabilities(
    capabilities=capabilities,
    backends=backends,
    gpu_available=gpu_available,
)
await self.discovery.announce_capabilities()
```

This runs once at startup. Additionally, a periodic re-announcement task (every 300 seconds) ensures new peers learn about existing providers:

```python
self._capability_announce_task = asyncio.create_task(
    self._periodic_capability_announce()
)
```

### 4.2 DHT Content Provide on Pin

When `StorageProvider` pins content and publishes `GOSSIP_CONTENT_ADVERTISE`, it should also register in the DHT. Add a `discovery` parameter to StorageProvider and call `provide_content()` after successful pin:

In `StorageProvider._on_storage_request()`, after publishing `GOSSIP_CONTENT_ADVERTISE`:
```python
if self.discovery:
    await self.discovery.provide_content(cid)
```

This dual-announces: GossipSub for immediate propagation, DHT for durable cross-restart records.

### 4.3 Compute Requester Capability-Based Peer Lookup

`ComputeRequester._get_capable_peers()` should use discovery to find peers with matching capabilities. Add a `discovery` parameter to ComputeRequester:

```python
async def _get_capable_peers(self, job_type: str) -> List[str]:
    if not self.discovery:
        return []
    
    # Map job type to required capability
    cap_map = {
        "inference": "inference",
        "embedding": "embedding",
        "training": "training",
        "benchmark": "compute",
        "wasm_execute": "compute",
    }
    required_cap = cap_map.get(job_type, "compute")
    peers = self.discovery.find_peers_with_capability(required_cap)
    return [p.node_id for p in peers]
```

### 4.4 Node.py Wiring Changes Summary

- Pass `transport` and `discovery` to `StorageProvider` constructor
- Pass `discovery` to `ComputeRequester` constructor
- Call `discovery.set_local_capabilities()` + `announce_capabilities()` after provider startup
- Add periodic re-announcement background task
- Remove standalone `register_content_handler()` call (moved into StorageProvider.start())

---

## Component 5: Integration Tests

**File:** `tests/integration/test_libp2p_compute_storage.py`

Tests use a `MockLibp2pTransport` that routes messages between two in-process nodes without requiring the Go shared library. The mock implements the same public API as `Libp2pTransport` and delivers messages through the same `_dispatch` / handler mechanism.

### Test 1: Compute Job Lifecycle

Two nodes: requester + provider. Requester submits job offer, provider accepts, requester confirms, provider executes and sends result, requester sends payment confirmation. Asserts all gossip subtypes flowed correctly, job completed, payment recorded.

### Test 2: Storage Pin + Challenge via Direct P2P

Two nodes: challenger + storage provider. Provider pins content, announces via gossip, registers in DHT. Challenger issues challenge via direct P2P. Provider answers via direct P2P. Challenger verifies proof. Asserts: challenge/proof used `send_to_peer`, not `gossip.publish`.

### Test 3: Capability Discovery

Two nodes with different capabilities. Both announce capabilities. Each queries discovery for specific capabilities. Asserts capability index populated correctly, queries return expected peers.

### Test 4: Direct P2P Fallback to Gossip

Challenger issues challenge to provider via direct P2P. `send_to_peer` returns failure (mock simulates connection failure). Challenger falls back to gossip broadcast. Provider receives and answers via gossip fallback. Asserts challenge delivered despite direct connection failure.

---

## Files Modified

| File | Change |
|------|--------|
| `prsm/node/storage_provider.py` | Add `transport` + `discovery` constructor params; direct P2P for challenges/proofs with gossip fallback; unified direct message dispatcher; DHT provide on pin |
| `prsm/node/compute_requester.py` | Add `discovery` constructor param; capability-based peer lookup |
| `prsm/node/node.py` | Pass `transport`/`discovery` to providers; capability announcement at startup; periodic re-announcement task; remove standalone `register_content_handler()` |
| `tests/integration/test_libp2p_compute_storage.py` | New file: 4 integration tests with mock transport harness |

## Files NOT Modified

| File | Reason |
|------|--------|
| `prsm/node/libp2p_gossip.py` | API already compatible |
| `prsm/node/libp2p_discovery.py` | API already has all needed methods |
| `prsm/node/libp2p_transport.py` | No changes needed |
| `prsm/node/compute_provider.py` | Already uses gossip correctly; no new params needed |
| `libp2p/` (Go code) | No changes to shared library |
