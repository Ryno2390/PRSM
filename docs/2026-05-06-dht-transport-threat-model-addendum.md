# PRSM-DHT-TRANSPORT Threat Model Addendum — §3.19

**Track:** PRSM-DHT-TRANSPORT (T0 router → T4 3-node E2E).
**Surface:**
- `prsm/network/dht_router.py` (T0 — request multiplexer)
- `prsm/network/sync_dht_transport.py` (T1 — sync↔asyncio bridge,
  `DHTLoopRunner`)
- `prsm/network/dht_listener.py` (T2 — TCP listener)
- `prsm/network/dht_components.py` (T3 — node-level aggregator)
- `tests/unit/test_dht_components.py::TestThreeNodeManifestE2E` and
  `TestTwoNodeEmbeddingE2E` (T4 — live-network E2E with real
  signatures)

**Status:** T0–T4 shipped. Production wiring (T3b — single-callsite
plumb into `prsm.node.node.Node.start()`) deferred to a follow-on
patch. The components module is reachable from tests today and from
node startup once the plumb lands.

This addendum extends the existing PRSM threat-model corpus
(`docs/2026-04-30-phase3.x.11-threat-model-addendum.md` §§3.7–3.8,
`docs/2026-05-06-prsm-prov-1-threat-model-addendum.md` §§3.16–3.17,
`docs/2026-05-06-prsm-prov-1-threat-model-addendum-item-7.md` §3.18)
with the inbound-listener / bridged-transport surface introduced by
the DHT-transport sprint.

---

## §3.19 — Adversaries against the DHT transport layer

### A1. Sybil attacker (routing-table flood)

**Goal:** Saturate B's Kademlia routing table with attacker-controlled
peers so honest peers are evicted and B's `find_closest_peers`
returns only Sybil identities.

**Capability:** Can mint arbitrary `node_id` hex strings, can run
many TCP listeners, can call `add_peer` indirectly via whatever
peer-discovery surface the operator wires (gossip, bootstrap, libp2p
DHT bridge, etc.).

**Mitigation:** `KademliaDHT.add_peer` is bounded per-bucket
(`k_bucket_size = 20`); replaces the *least-recently-seen* peer when
full only if that peer is no longer active. The peer-discovery
*source* — not this transport layer — is the place where Sybil
defense lives. R9 Phase 6.3 peer-jurisdiction filter (already shipped)
plus Phase 6 Task 1 signed-bootstrap filter the candidate set
upstream of `add_peer`. T3 deliberately keeps `DHTNodeComponents.add_peer`
as a thin wrapper without re-validating because layering filters on
the wrong side ("DHT-layer Sybil filter") creates the impression of
defense-in-depth without actually adding any.

**Residual risk:** A node with no peer-discovery filtering at all
(no bootstrap, raw `add_peer` from untrusted gossip) can have its
routing table flooded. Production deployments MUST configure at
minimum the signed-bootstrap path.

### A2. Eclipse attacker

**Goal:** Get B to install a routing table entirely composed of
attacker peers, so all DHT lookups route to attacker storage.

**Capability:** Same as A1 plus the ability to be the *first* peers
B sees on cold start.

**Mitigation:** Bootstrap diversity from Phase 6 Task 1 — the
signed-bootstrap list is anchored to multiple independent operators
and verified at startup. The DHT-transport layer inherits this
property: `add_peer` only accepts already-discovered peers, and
peer-discovery is bootstrap-anchored. Within this layer, no
additional eclipse defense is possible (this is the layer
*receiving* the routing table, not the one populating it).

**Residual risk:** A `dht_components.add_peer(...)` callsite that
bypasses bootstrap (e.g., a libp2p bridge that adds peers
unconditionally) creates an eclipse vector. Code review for new
callsites should flag any path that adds peers without going
through the bootstrap filter.

### A3. Resource-exhaustion via length-prefix attack

**Goal:** Cause the listener to allocate a large request buffer
(or response read-buffer on the client side) by announcing a huge
length and then sending nothing.

**Capability:** TCP-level — any peer can open a connection and
write 4 bytes.

**Mitigation:**
- **Listener side (`DHTListener`):** `max_request_bytes` (default
  1 MiB) is checked against the announced length *before* the
  buffer is allocated. An oversized announcement is logged at
  WARNING and the connection closed without reading the payload.
  The corresponding test `test_oversized_request_rejected_without_read`
  asserts the handler is never invoked.
- **Client side (`SyncDHTTransport`):** symmetrically, the
  `max_response_bytes` cap (also default 1 MiB) rejects oversized
  *response* prefixes with `TransportFailureError` before allocating
  the read buffer. Test
  `test_response_exceeds_max_response_bytes` covers this.
- Both DHT protocols cap `MAX_MESSAGE_BYTES` at 256 KiB at the
  protocol parsers; the transport-layer 1 MiB ceiling is a 4×
  safety margin above protocol-level reasoning.

**Residual risk:** A peer that *fits within* `max_request_bytes`
but never sends the announced payload still ties up a
listener-side handler until `request_timeout` (default 10s) fires.
Mitigated by `max_concurrent_connections` (default 1024) which
caps the simultaneous in-flight handlers.

### A4. Connection-flood (slow-loris)

**Goal:** Exhaust the listener's task budget so legitimate peers
can't get serviced.

**Capability:** Open many TCP connections concurrently, each
sending the request slowly or not at all.

**Mitigation:**
- `max_concurrent_connections` (default 1024) caps simultaneous
  in-flight handlers; over-limit connections are accepted and
  immediately closed with a WARNING log.
- `request_timeout` (default 10s) wraps the read-handle-write
  cycle in `asyncio.wait_for`; a slow handler is cancelled and
  the connection closed.
- The listener does not maintain per-IP connection counts. A
  single attacker IP could fill the 1024 slots; this is an
  acknowledged design choice — production deployments behind a
  reverse proxy (nginx, HAProxy) inherit per-IP rate limiting
  from the proxy. Direct exposure with no proxy MUST be
  configured with conservative `max_concurrent_connections` and
  `request_timeout`.

**Residual risk:** The semaphore is global per-listener, not
per-peer. A single misbehaving operator can starve legitimate
traffic. Mitigation roadmap: add a per-IP token bucket in T-future
when this becomes a measured problem on testnet. (Adding it now,
absent measured abuse, would be premature.)

### A5. Cross-DHT confusion

**Goal:** Send a `find_providers` request (manifest-protocol type)
to a node that runs *only* EmbeddingDHT, hoping for a malformed
response or a crash.

**Capability:** Adversary-controlled wire payloads.

**Mitigation:** `DHTRequestRouter._resolve_target` is deterministic:
it consults `_MANIFEST_TYPES` first, then `_EMBEDDING_TYPES`. A
node running only one of the two replies with an
`UNSUPPORTED_VERSION` error envelope (a structured DHT-protocol
error response, not an exception, not a TCP reset). Tests
`test_works_with_only_manifest_server_registered` and
`test_works_with_only_embedding_server_registered` exercise this.
The shared `"error"` type is resolved to manifest deterministically
on overlap, with a comment explaining why (errors never arrive at
server entry, so the choice is arbitrary but pinned to be
reproducible).

**Residual risk:** None at this layer. The protocol parsers below
the router are individually responsible for rejecting malformed
payloads — both already documented as never-raises.

### A6. Sync↔asyncio bridge starvation

**Goal:** Concurrent uploaders calling
`SyncDHTTransport.send` from many threads produce a deadlock or
unbounded latency, blocking upload-critical-path code.

**Capability:** Many threads, many simultaneous `send()` calls.

**Mitigation:** Each `send()` schedules an *independent* coroutine
on the loop via `asyncio.run_coroutine_threadsafe`. The coroutine
opens its own socket, writes, reads, closes — there is no
per-instance mutable state on the hot path other than the
asyncio loop itself, which is by design designed for concurrent
coroutines. Test
`test_many_concurrent_senders` runs 16 threads × 4 RPCs each (64
total) with a barrier-enforced burst start and asserts no
deadlock and no payload corruption (each thread's response equals
its own request bytes exactly).

**Residual risk:** The asyncio loop runs on a single OS thread.
CPU-bound work in a coroutine (e.g., signature verification) blocks
the loop. The listener already mitigates this via
`loop.run_in_executor` for `router.handle()`. The transport-side
coroutines do only socket I/O, so this risk is bounded.

### A7. Listener-handler exception leak

**Goal:** Crash the listener by triggering an exception in
`router.handle()` or one of the downstream DHT servers.

**Capability:** Any wire-level malformed payload + a regression
that introduces a raise where the protocol contract says
"never-raises."

**Mitigation:** Defense in depth at three layers:
1. The DHT server's `handle()` is documented never-raises and
   converts every parse / dispatch failure into an `ErrorResponse`.
2. The router catches any raise from `target.handle()` and
   converts to an `INTERNAL_ERROR` envelope (test
   `test_router_catches_server_raise_and_returns_error`).
3. The listener's `_handle_connection` catches any raise from the
   router, logs at ERROR with a stack, and closes the connection
   (test `test_router_handle_exception_closes_without_response`).

**Residual risk:** A fault in the listener loop *itself* (e.g.,
buggy `asyncio.start_server` or kernel-level socket accept failure)
would propagate up to the loop's exception handler. The loop is
in a dedicated thread and its termination would render the node's
DHT unavailable, but would not crash the rest of the node. This
is an acceptable degradation — the rest of `node.py` continues to
run and could fail-open to "no DHT" mode. (Existing
`FilesystemModelRegistry` behavior covers this case: when the
DHT is unavailable, lookups fall back to local-only.)

### A8. IPv6 reflection / name confusion

**Goal:** Trick `SyncDHTTransport._parse_address` into routing
to an unintended host by passing a malformed address string.

**Capability:** Control over a peer's announced address (e.g.,
via a malicious DHT response that contains a substituted
`provider.address`).

**Mitigation:** `_parse_address` uses `str.rpartition(":")`,
which correctly extracts the *last* `:port` suffix. For IPv4
hostnames (current PRSM peer model) and DNS names, this is
unambiguous. For IPv6 (containing internal `:` separators), the
parse is ambiguous and v1 deliberately does NOT support IPv6 —
the parser would interpret the trailing IPv6 octet as the port
and fail downstream. Documented in `_parse_address`'s docstring.
Tested via `test_rejects_address_with_no_colon` (and friends).

**Residual risk:** IPv6 deployment is gated on adding bracket
handling to the parser. The PRSM peer model uses IPv4 + DNS, so
this is a planned rather than urgent gap.

---

## Out of scope for this layer

The following are deliberately not addressed at the DHT-transport
layer and have explicit homes elsewhere in the threat model:

- **Manifest / embedding signature forgery** — covered by Phase
  3.x.3 PublisherKeyAnchor + the `verify_signature` callable
  contract on `EmbeddingDHTClient` (`docs/2026-05-06-prsm-prov-1-
  threat-model-addendum.md` §3.16–3.17).
- **Provenance dispute attacks** — covered by §3.18 (on-chain
  ProvenanceRegistryV2 commitment).
- **Timing side channels in inference** — covered by Phase 3.x.10.y
  / 3.x.11.q.x (`docs/2026-04-30-phase3.x.11-threat-model-addendum.md`
  §§3.7–3.8).

---

## Audit-prep cross-reference

This addendum is the §3.19 entry referenced from
`docs/2026-05-06-dht-transport-integration-plan.md` §T5. It
preserves the convention used in prior threat-model addenda:
each adversary block specifies *Goal*, *Capability*, *Mitigation*
(with test pointers when possible), and *Residual risk*.

The named tests are all in `tests/unit/`:
- `test_dht_router.py` (21 tests covering A5 + A7 layers)
- `test_sync_dht_transport.py` (36 tests covering A6 + A8)
- `test_dht_listener.py` (23 tests covering A3 + A4 + A7)
- `test_dht_components.py` (20 tests covering A1 + A2 + E2E)

Total 100 tests across the DHT-transport surface; full suite
runs in <5s on the development machine.
