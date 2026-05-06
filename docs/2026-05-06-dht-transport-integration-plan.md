# DHT Transport Integration Sprint — Design Plan

**Track ID:** PRSM-DHT-TRANSPORT
**Status:** Plan ratified 2026-05-06; T0 (DHTRequestRouter) shipped.
**Unblocks:** Phase 3.x.5 ManifestDHT T7 (3-node E2E) + PRSM-PROV-1
Item 3 T3.8 (3-node E2E with cross-node embedding gossip).

## Why

Two PRSM DHTs are library-complete but cannot run as real network
services because there's no transport listener wiring at the node
level:

1. **ManifestDHT (Phase 3.x.5)** — `prsm/network/manifest_dht/` —
   `ManifestDHTServer.handle(request_bytes) -> bytes` is ready;
   `ManifestDHTClient` accepts a `RoutingTable` (Protocol) and a
   `SendMessageFn = Callable[[str, bytes], bytes]`. Production needs
   real implementations of both.
2. **EmbeddingDHT (PRSM-PROV-1 Item 3)** — `prsm/network/embedding_dht/`
   — same shape, same gap.

Concrete components already in place:
- `prsm.compute.collaboration.p2p.node_discovery.KademliaDHT` —
  satisfies the `RoutingTable` Protocol via
  `find_closest_peers(target_id, count)`.
- `prsm.node.transport_adapter.TransportAdapter` — gives async
  `open_connection(host, port) -> socket.socket` over Direct or SOCKS.
- `prsm.network.dht_router.DHTRequestRouter` (T0, this commit) —
  multiplexes incoming bytes to the right server.

Concrete components missing:
- A sync `(address, request_bytes) -> response_bytes` adapter that
  wraps `TransportAdapter` so the sync DHT clients can call it from
  the upload-critical path.
- A network listener at the node level that accepts DHT requests and
  invokes `DHTRequestRouter.handle()`.
- node.py wiring that constructs all the above on startup, and adds
  discovered peers to the `KademliaDHT` routing table.

## Scope

### T0 — DHTRequestRouter ✅ shipped

`prsm/network/dht_router.py` + `tests/unit/test_dht_router.py` (21
tests). Multiplexes by JSON `type` field. Catches downstream raises
and non-bytes returns; never raises out. Determinism on protocol
overlap ("error" type → manifest server when both registered).

### T1 — SyncDHTTransport (~150 LOC + ~150 test LOC)

`prsm/network/sync_dht_transport.py`. Wraps a `TransportAdapter` +
asyncio loop in a thread pool to expose a synchronous
`SendMessageFn = Callable[[str, bytes], bytes]`. Implementation
sketch:

```python
class SyncDHTTransport:
    def __init__(
        self, adapter: TransportAdapter, loop: asyncio.AbstractEventLoop,
        *, default_timeout: float = 10.0, max_response_bytes: int = 1 << 20,
    ): ...
    def send(self, address: str, request_bytes: bytes) -> bytes:
        host, port = self._parse_address(address)
        future = asyncio.run_coroutine_threadsafe(
            self._send_async(host, port, request_bytes), self._loop,
        )
        return future.result(timeout=self._timeout)
    async def _send_async(self, host, port, request_bytes) -> bytes:
        sock = await self._adapter.open_connection(host, port, ...)
        await loop.sock_sendall(sock, _length_prefix(request_bytes))
        return await _read_length_prefixed(sock, self._max_response_bytes)
```

Wire framing: 4-byte big-endian length prefix + payload bytes.
Matches existing `prsm.node.transport.py` patterns. Errors from the
adapter (TransportConnectError) and parse failures (truncated frame)
both raise `TransportFailureError` so the DHT clients' existing
`except TransportFailureError` paths in `_pull_remote_embeddings` and
the manifest equivalent fire correctly.

### T2 — DHT TCP listener (~200 LOC + ~200 test LOC)

`prsm/network/dht_listener.py`. Runs an `asyncio.start_server` on a
configurable port; for each connection, reads one length-prefixed
request, calls `router.handle(request_bytes)`, writes the
length-prefixed response, closes the connection. Stateless per
request — keeps the wire model simple and matches the existing one-shot
pattern in ManifestDHTClient.find_providers.

Tests use `asyncio.open_connection` to drive the listener directly and
verify routing through to a stub server.

### T3 — node.py construction + add_peer integration (~100 LOC + ~100 test LOC)

In `prsm/node/node.py` startup:
1. Construct `KademliaDHT(node_id=self.identity.node_id, port=...)`.
2. Construct `LocalManifestIndex` + `ManifestDHTServer` (already exists
   for the registry path; extend to also be passed to the router).
3. Construct `LocalEmbeddingIndex` + `EmbeddingDHTServer`.
4. Construct `DHTRequestRouter(manifest_server=..., embedding_server=...)`.
5. Construct `DHTListener(router, port=...)` and start it.
6. Construct `SyncDHTTransport(adapter=self.transport_adapter, loop=...)`.
7. Construct `ManifestDHTClient` and `EmbeddingDHTClient`, both pointed
   at `KademliaDHT` for routing and `SyncDHTTransport.send` for
   transport.
8. In the existing `discovery.on_peer_discovered` callback, also call
   `kademlia_dht.add_peer(peer_node)` so DHT routing-table state
   tracks live peer set.

Configuration: add a `dht_listen_port` field to `NodeConfig` (default
to a port adjacent to `p2p_port`).

### T4 — Phase 3.x.5 T7 + PRSM-PROV-1 T3.8 E2E re-light (~200 LOC each)

With T3 wired, the existing 3-node E2E test scaffolds in both DHT
packages can target real nodes instead of stubs. `tests/integration/`
spawns three node processes on adjacent ports, has node A upload a
manifest / embedding, has node B request it via the DHT, asserts
content matches. Validates the entire stack end-to-end.

### T5 — Threat model + audit-prep + merge-ready tag

Threat-model addendum §3.19 covers:
- Sybil attacks on the routing table (peer-jurisdiction filter from
  R9 §6.3 already mitigates).
- Eclipse attacks (bootstrap diversity from Phase 6 Task 1).
- Resource exhaustion via large request payloads (MAX_MESSAGE_BYTES
  enforced in DHT protocol parsers; listener also caps at
  max_response_bytes).
- Cross-DHT confusion (mitigated by router's deterministic resolution
  on type-set overlap).

## Non-goals (this sprint)

- Persistent storage of routing table state across restarts. Kademlia
  bootstrap will repopulate on startup; persistence is a separate
  optimization.
- DHT-level peer signing (already exists at the message level via
  PublisherKeyAnchor; routing-table membership is a soft hint, not a
  trust statement).
- Replacing `KademliaDHT` with a libp2p-backed implementation. The
  Protocol-based decoupling means we can swap later without touching
  DHT clients.

## Estimate

| Task | Source LOC | Test LOC | Hours |
|---|---|---|---|
| T0 SyncDHTTransport | ~150 | ~150 | 2-3 |
| T1 DHTListener | ~200 | ~200 | 3-4 |
| T2 node.py wiring | ~100 | ~100 | 2 |
| T3 E2E re-light | — | ~400 | 3-4 |
| T4 Threat model + tag | — | — | 1 |
| **Total** | **~450** | **~850** | **11-14** |

## Risks

- **Mixing sync and asyncio** — the DHT clients are sync because the
  upload critical path is sync. SyncDHTTransport bridges via
  `run_coroutine_threadsafe`, which is the standard pattern but
  requires a long-running asyncio loop in a separate thread. Tests
  must validate no deadlock under concurrent uploads.
- **Per-request connection overhead** — opening a fresh TCP socket
  per DHT lookup costs ~10-50ms RTT on top of message processing. A
  future optimization (T-future) is connection pooling, but for
  initial deploy single-shot is correct (matches existing
  ManifestDHT/EmbeddingDHT one-shot semantics; tracks fewer
  per-peer state machines).
- **Listener concurrency** — `asyncio.start_server` is single-loop;
  if a single DHT request blocks the loop for >100ms (e.g. large
  manifest fetch from disk), other peers see latency spikes. Mitigation:
  put the disk read inside `loop.run_in_executor` for any
  potentially-blocking server-side work.

## Cross-references

- ManifestDHT plan: `docs/2026-04-26-phase-3-x-5-manifest-dht-plan.md`
- EmbeddingDHT plan: `docs/2026-05-06-content-provenance-correctness-plan.md` §2
- TransportAdapter design: R9 Phase 6.2 task scoping
- KademliaDHT: `prsm/compute/collaboration/p2p/node_discovery.py:71`
- DHTRequestRouter (T0): `prsm/network/dht_router.py`
