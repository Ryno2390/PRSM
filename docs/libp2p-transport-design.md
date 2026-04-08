# libp2p Transport Layer Integration

## Goal

Replace PRSM's custom WebSocket transport, epidemic gossip, and bootstrap server with go-libp2p — gaining automatic NAT traversal, battle-tested GossipSub messaging, and Kademlia DHT peer discovery while preserving the exact public API that all consumer modules depend on.

## Architecture

PRSM's networking stack is three layers: transport (WebSocket), gossip (epidemic), and discovery (custom bootstrap). All three get replaced by go-libp2p equivalents, compiled as a C shared library and called from Python via ctypes. Consumer modules (compute_provider, storage_provider, agent_registry, etc.) see no change — they continue calling the same `transport.send_to_peer()`, `gossip.publish()`, and `discovery.find_peers_by_capability()` interfaces.

## Tech Stack

- **go-libp2p** (compiled to C shared library via `go build -buildmode=c-shared`)
- **Python ctypes** (stdlib — no new dependencies)
- **GossipSub v1.1** (replaces custom epidemic gossip)
- **Kademlia DHT** (replaces custom bootstrap server)
- **QUIC + TCP+Noise** (replaces WebSocket transport)
- **AutoNAT + Circuit Relay v2 + Hole Punching** (new — NAT traversal)

---

## Component 1: Go Shared Library (`libp2p/`)

A Go module at the repo root that wraps go-libp2p and exposes a C-compatible FFI API.

### Responsibilities

- Host lifecycle: start/stop a libp2p host with the node's Ed25519 identity
- Transports: QUIC (primary, UDP) + TCP+Noise (fallback)
- Discovery: Kademlia DHT for global peer routing + mDNS for local network
- Messaging: GossipSub for pub/sub topics + direct streams for point-to-point
- NAT traversal: AutoNAT detection, Circuit Relay v2, hole punching — all automatic
- Content routing: DHT provide/find for advertising shard availability

### Exported C API

All functions take an opaque `host_handle C.int` as first argument (returned by `PrsmStart`). This supports multiple libp2p hosts in a single process — essential for multi-node tests without Docker.

| Function | Signature | Purpose |
|----------|-----------|---------|
| `PrsmStart` | `(ed25519_key *C.char, listen_port C.int, bootstrap_addrs *C.char, uds_path *C.char) → C.int` | Start libp2p host, returns opaque host handle |
| `PrsmStop` | `(handle C.int) → C.int` | Graceful shutdown, returns 0 on success |
| `PrsmConnect` | `(handle C.int, multiaddr *C.char) → *C.char` | Dial peer by multiaddr, returns peer ID |
| `PrsmSend` | `(handle C.int, peer_id *C.char, protocol *C.char, data *C.char, data_len C.int) → C.int` | Send bytes to peer via direct stream |
| `PrsmPublish` | `(handle C.int, topic *C.char, data *C.char, data_len C.int) → C.int` | Publish to GossipSub topic |
| `PrsmSubscribe` | `(handle C.int, topic *C.char) → C.int` | Subscribe to GossipSub topic |
| `PrsmUnsubscribe` | `(handle C.int, topic *C.char) → C.int` | Unsubscribe from topic |
| `PrsmPeerCount` | `(handle C.int) → C.int` | Connected peer count |
| `PrsmPeerList` | `(handle C.int) → *C.char` | JSON array of connected peer info |
| `PrsmDHTProvide` | `(handle C.int, key *C.char) → C.int` | Announce content availability on DHT |
| `PrsmDHTFindProviders` | `(handle C.int, key *C.char, limit C.int) → *C.char` | Find content providers, returns JSON |
| `PrsmGetNATStatus` | `(handle C.int) → *C.char` | Returns NAT type: "public", "private", "unknown" |
| `PrsmFree` | `(ptr *C.char) → void` | Free memory allocated by Go for returned strings |

Go internally maps handles to host instances via `sync.Map[int, *Host]`. Handle 0 is reserved as invalid.

### Multi-Instance Support

`PrsmStart` returns an integer handle, not a peer ID string. This allows multiple libp2p hosts in a single process — each with its own identity, listeners, DHT, and GossipSub. Critical for:

- **Multi-node tests:** pytest spins up 2+ nodes in one process without Docker
- **Simulations:** local benchmarking of gossip propagation, DHT convergence, etc.
- **Embedded usage:** a coordinator process managing multiple logical nodes

The Python adapter stores its handle and passes it on every FFI call:

```python
class Libp2pTransport:
    def __init__(self, ...):
        self._handle: int = 0  # Set by start()
    
    async def start(self) -> None:
        self._handle = self._lib.PrsmStart(key, port, addrs, uds_path)
        if self._handle == 0:
            raise Libp2pTransportError("Failed to start libp2p host")
    
    async def send_to_peer(self, peer_id, msg) -> bool:
        return self._lib.PrsmSend(self._handle, peer_id, ...) == 0
```

### FFI Memory Management

Every Go function that returns `*C.char` allocates via `C.CString`, which uses C heap memory invisible to both Go's GC and Python's GC. Without explicit cleanup, these allocations leak.

**Hard requirement:** Every `*C.char` return must be freed by calling `PrsmFree(ptr)` from the Python side. The Python adapter wraps all FFI calls that return strings in a helper that decodes the bytes and immediately frees the pointer:

```python
def _call_string(self, fn, *args) -> str:
    ptr = fn(*args)
    try:
        return ptr.decode("utf-8") if ptr else ""
    finally:
        self._lib.PrsmFree(ptr)
```

### Panic Recovery

Go panics in exported functions would crash the entire Python process with no traceback. Every exported C function is wrapped with panic recovery:

```go
//export PrsmConnect
func PrsmConnect(handle C.int, multiaddr *C.char) *C.char {
    defer func() {
        if r := recover(); r != nil {
            log.Printf("PrsmConnect panic recovered: %v", r)
        }
    }()
    host := hosts.Get(int(handle))
    if host == nil {
        return nil  // Python side raises Libp2pTransportError
    }
    // ... implementation
}
```

On the Python side, null/error returns from Go raise `Libp2pTransportError` — a proper Python exception — instead of propagating silently.

### Control Plane vs Data Plane (Split Architecture)

The FFI boundary is split into two communication channels to avoid GIL contention under high message throughput:

**Control plane (ctypes FFI):** Used for infrequent operations — start, stop, connect, peer queries, DHT provide/find. These cross the CGo/Python boundary directly. Acceptable latency since they're called rarely.

**Data plane (Unix Domain Socket):** Used for high-frequency incoming messages — GossipSub deliveries and direct stream data. Go writes length-prefixed framed messages to a UDS. Python reads them via `asyncio` socket reader — zero FFI involvement on the hot path, no GIL thrashing.

`PrsmStart` accepts a `uds_path` parameter. Go opens a Unix socket at that path and writes incoming messages as framed packets.

**UDS cleanup:** Go's `internal/uds.go` must unlink any existing socket file at `uds_path` before binding (handles crash recovery — if the node was killed via SIGKILL, the stale socket file persists). Python's `Libp2pTransport.stop()` explicitly deletes the socket file after closing the reader. Belt and suspenders — both sides clean up.

Frame format:

```
[4 bytes: payload length (big-endian uint32)]
[payload: JSON object with msg_type, topic_or_peer, data, sender_id]
```

Python's adapter opens an `asyncio` connection to the UDS and reads frames in a background task:

```python
async def _uds_reader(self):
    reader, _ = await asyncio.open_unix_connection(self._uds_path)
    while True:
        length_bytes = await reader.readexactly(4)
        length = int.from_bytes(length_bytes, "big")
        payload = await reader.readexactly(length)
        msg = json.loads(payload)
        await self._dispatch(msg)
```

This eliminates CGo callback overhead entirely from the message delivery path. Python's asyncio event loop handles socket I/O natively and efficiently.

### Backpressure

If Python falls behind processing messages, the kernel UDS buffer fills (~128KB default). Go detects this via write backpressure. Behavior by message type:

- **Gossip messages (best-effort):** Go drops the oldest undelivered message and increments a `messages_dropped` counter exposed via `PrsmPeerList` telemetry. Gossip is inherently tolerant of message loss.
- **Direct stream messages (must-deliver):** Go buffers in a bounded channel per peer (default 256 messages). If the channel fills, Go pauses reading from the libp2p stream, applying backpressure upstream to the sending peer.

### Callback Data Safety

For any remaining ctypes callbacks (e.g., connection lifecycle events on the control plane), the callback must deep-copy all data before returning control to Go:

```python
@ctypes.CFUNCTYPE(None, c_char_p, c_int)
def _on_event(data, data_len):
    # MUST copy before returning — Go may free the buffer after callback returns
    payload = ctypes.string_at(data, data_len)
    loop.call_soon_threadsafe(queue.put_nowait, bytes(payload))
```

This prevents use-after-free when Go reclaims the buffer after the callback returns.

### Identity

The Go host derives its libp2p identity from the same Ed25519 private key that the Python node already uses. The key is passed as raw bytes to `PrsmStart`. This means the node's libp2p peer ID is deterministically derived from its existing PRSM identity — no new key management.

**Key format constraint:** libp2p uses a Protobuf-encoded wrapper around Ed25519 keys internally. The Go side must accept raw 64-byte Ed25519 private keys (as Python's `cryptography` library produces) and convert them to libp2p's `crypto.PrivKey` using `crypto.UnmarshalEd25519PrivateKey(raw_bytes)`. If this conversion is wrong, the libp2p peer ID will differ from PRSM's `node_id` (which is `SHA256(public_key)[:32]`), silently breaking authentication. The implementation must include a startup assertion that verifies the derived peer ID matches the expected PRSM node ID.

### Build

```bash
cd libp2p/
go build -buildmode=c-shared -o build/libprsm_p2p.so ./cmd/libprsm/
# macOS: produces .dylib, Windows: .dll
```

Pre-built binaries for Linux amd64, macOS arm64, and macOS amd64 are committed to `libp2p/build/` for users without Go toolchain.

### Location

```
libp2p/
├── go.mod
├── go.sum
├── cmd/
│   └── libprsm/
│       └── main.go          # CGo exports, C API implementation
├── internal/
│   ├── host.go              # libp2p host setup, transport config
│   ├── gossipsub.go         # GossipSub topic management
│   ├── dht.go               # Kademlia DHT setup, provide/find
│   ├── streams.go           # Direct stream protocol handlers
│   ├── nat.go               # AutoNAT, relay, hole punching config
│   ├── uds.go               # Unix Domain Socket writer (data plane)
│   └── handles.go           # Host handle registry (sync.Map)
├── build/
│   ├── libprsm_p2p.so       # Linux amd64
│   ├── libprsm_p2p.dylib    # macOS arm64
│   └── libprsm_p2p.h        # Auto-generated C header
└── Makefile
```

---

## Component 2: Python Adapter (`prsm/node/libp2p_transport.py`)

A drop-in replacement for `WebSocketTransport` that delegates to the Go shared library via ctypes.

### Interface Contract

Identical to today's `WebSocketTransport`:

```python
class Libp2pTransport:
    def __init__(self, identity: NodeIdentity, host: str, port: int, **kwargs): ...

    # Lifecycle
    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    # Connection
    async def connect_to_peer(self, address: str) -> Optional[PeerConnection]: ...

    # Messaging
    async def send_to_peer(self, peer_id: str, msg: P2PMessage) -> bool: ...
    async def broadcast(self, msg: P2PMessage) -> int: ...
    async def gossip(self, msg: P2PMessage, fanout: int = 3) -> int: ...

    # Handler registration
    def on_message(self, msg_type: str, handler: MessageHandler) -> None: ...

    # Peer info
    @property
    def peer_count(self) -> int: ...
    async def get_peer_count(self) -> int: ...
    @property
    def peer_addresses(self) -> List[str]: ...
    async def get_peer_addresses(self) -> List[str]: ...

    # Identity (used by consumers)
    @property
    def identity(self) -> NodeIdentity: ...

    # Observability
    def get_telemetry_snapshot(self) -> Dict[str, Any]: ...
```

### FFI Bridge Design

The adapter loads the shared library at construction time:

```python
self._lib = ctypes.CDLL("libp2p/build/libprsm_p2p.so")
```

Platform detection selects `.so`, `.dylib`, or `.dll` automatically. If the expected binary is missing or incompatible (wrong architecture, stale version), `start()` raises `Libp2pTransportError` with a clear message directing the user to build locally:

```
Libp2pTransportError: libprsm_p2p shared library not found for darwin/arm64.
Run 'make' in the libp2p/ directory to build from source (requires Go 1.22+).
```

This handles developers on platforms without pre-built binaries (Windows, non-standard Linux distros with different GLIBC versions).

### Message Delivery (UDS Reader)

Incoming messages arrive via Unix Domain Socket (see Control Plane vs Data Plane above). The adapter starts a background `asyncio` task that reads framed messages and dispatches to registered handlers:

```python
async def start(self) -> None:
    self._loop = asyncio.get_running_loop()
    self._uds_path = f"/tmp/prsm_p2p_{self._handle}.sock"
    self._handle = self._lib.PrsmStart(key, port, addrs, self._uds_path.encode())
    self._reader_task = asyncio.create_task(self._uds_reader())

async def _uds_reader(self):
    reader, _ = await asyncio.open_unix_connection(self._uds_path)
    while True:
        length_bytes = await reader.readexactly(4)
        length = int.from_bytes(length_bytes, "big")
        payload = await reader.readexactly(length)
        msg = json.loads(payload)
        for handler in self._handlers.get(msg["msg_type"], []):
            await handler(self._deserialize(msg["data"]), self._make_peer_info(msg["sender_id"]))
```

No GIL contention on the hot path — pure asyncio socket I/O.

### Address Translation

The adapter transparently translates between PRSM address formats and libp2p multiaddrs:

- `wss://host:port` → `/ip4/{host}/tcp/{port}/ws` (legacy compat)
- `host:port` → `/ip4/{host}/udp/{port}/quic-v1`
- `/ip4/...` → passed through as-is (native multiaddr)

### P2PMessage Serialization

P2PMessage is serialized to JSON bytes when crossing the FFI boundary. The Go side treats message content as opaque bytes — it doesn't parse or validate PRSM message structure. Signing and verification remain in Python.

**Serialization optimization path:** JSON is appropriate for v1 since PRSM messages are small control payloads (gossip metadata, job offers, capability announcements). If throughput becomes a bottleneck for large payloads (dense orchestration data, model shard metadata), the FFI serialization can be swapped to Protocol Buffers or FlatBuffers. The adapter isolates serialization to two methods (`_serialize` / `_deserialize`) so this is a single-file change with no consumer impact.

---

## Component 3: GossipSub Wrapper (`prsm/node/libp2p_gossip.py`)

A thin wrapper that preserves the `GossipProtocol` public API while delegating to GossipSub.

### Public API (unchanged)

```python
class Libp2pGossip:
    def __init__(self, transport: Libp2pTransport, **kwargs): ...

    def subscribe(self, subtype: str, callback: GossipCallback) -> None: ...
    async def publish(self, subtype: str, data: Dict[str, Any], ttl: Optional[int] = None) -> int: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def get_catchup_messages(self, since: float, subtypes: Optional[List[str]] = None) -> List[Dict]: ...
    def get_telemetry_snapshot(self) -> Dict[str, Any]: ...
```

### Topic Mapping

Each PRSM gossip subtype maps to a GossipSub topic:

```python
def _topic_name(subtype: str) -> str:
    return f"prsm/{subtype}"
```

Examples:
- `GOSSIP_JOB_OFFER` → `"prsm/job_offer"`
- `GOSSIP_CONTENT_ADVERTISE` → `"prsm/content_advertise"`
- `GOSSIP_AGENT_DISPATCH` → `"prsm/agent_dispatch"`

### What GossipSub Handles Natively

- **Fanout and mesh management** — GossipSub maintains 6 mesh peers per topic, 8 lazy peers for protocol-level redundancy
- **Message deduplication** — by message ID (hash of content), no nonce tracking needed
- **Late-joiner catch-up** — IHAVE/IWANT protocol exchanges message IDs with new peers
- **Heartbeat** — 1-second internal heartbeat for mesh maintenance
- **Flood publishing** — initial publish goes to all mesh peers, then lazy forwarding

### What the Wrapper Still Does

- **Callback registry** — maps topic → list of Python callbacks
- **Lazy subscription** — `PrsmSubscribe` is only called when the first Python callback registers for a subtype, not eagerly for all subtypes. This avoids GossipSub mesh overhead for topics no consumer cares about. With 30+ PRSM gossip subtypes, a node that only does compute (not storage or BitTorrent) should not join meshes for storage/torrent topics.
- **Message envelope** — wraps `data` dict with `sender_id`, `timestamp`, `subtype` before publishing
- **Ledger persistence** — optionally logs messages to the DAG ledger (audit trail)
- **Telemetry** — tracks publish/receive counts per subtype

### Removed (Handled by GossipSub)

- Manual fanout peer selection
- TTL tracking and decrement
- Nonce deduplication
- Digest request/response protocol
- Heartbeat loop for protocol maintenance

---

## Component 4: DHT Discovery Wrapper (`prsm/node/libp2p_discovery.py`)

A wrapper that preserves the `PeerDiscovery` public API while using Kademlia DHT for peer routing.

### Public API (unchanged)

```python
class Libp2pDiscovery:
    def __init__(self, transport: Libp2pTransport, bootstrap_nodes: List[str], **kwargs): ...

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def bootstrap(self) -> int: ...
    def get_known_peers(self) -> List[PeerInfo]: ...
    def find_peers_by_capability(self, required: List[str], match_all: bool = True) -> List[PeerInfo]: ...
    def find_peers_with_capability(self, capability: str) -> List[PeerInfo]: ...
    def find_peers_with_backend(self, backend: str) -> List[PeerInfo]: ...
    def find_peers_with_gpu(self) -> List[PeerInfo]: ...
    def set_local_capabilities(self, capabilities: List[str], backends: List[str], gpu_available: bool = False) -> None: ...
    async def announce_capabilities(self) -> int: ...
    def get_bootstrap_status(self) -> Dict[str, Any]: ...
```

### How It Works

**Bootstrap:** Calls `PrsmConnect(multiaddr)` for each bootstrap node. The DHT then automatically crawls outward, populating the routing table with discovered peers. No custom registration protocol needed.

**Capability discovery:** Subscribes to GossipSub topic `"prsm/capability_announce"`. When peers publish their capabilities, the wrapper updates its local capability index. Lookup methods (`find_peers_with_gpu()`, etc.) query this index.

**Content routing (new — hybrid DHT + GossipSub):** Shard discovery uses two complementary channels:

1. **Immediate notification (GossipSub):** When a node ingests a new shard, it publishes to topic `"prsm/shard_available"` with `{cid, shard_id, dataset_id}`. All subscribed peers learn about the shard within seconds. This handles the "I just uploaded data, who can serve it right now?" case.

2. **Durable routing (DHT):** In parallel, `PrsmDHTProvide(cid)` registers the node as a provider in the Kademlia DHT. This handles the "where is shard X that was uploaded last week?" case. DHT records persist across node restarts and survive gossip message expiry.

Consumers query both:

```python
async def find_content_providers(self, cid: str, limit: int = 20) -> List[PeerInfo]:
    """Find nodes that have a specific content shard.
    
    Checks GossipSub-populated local cache first (instant),
    falls back to DHT lookup (slower but durable).
    """

async def provide_content(self, cid: str) -> None:
    """Announce shard availability via both GossipSub (immediate) and DHT (durable)."""
```

This hybrid approach avoids the Kademlia settlement latency problem — the DHT can take seconds to minutes to propagate provider records to the closest peers, but GossipSub delivers to mesh peers within milliseconds.

**NAT status:** Exposes `PrsmGetNATStatus()` for observability. The node dashboard can show whether the node is publicly reachable or operating behind NAT with relay.

### Removed (Handled by DHT)

- `BootstrapClient` class and JSON registration protocol
- Bootstrap retry/backoff logic
- `maintain_connections()` peer count maintenance loop
- `announce_self()` periodic presence broadcast
- The entire `prsm/bootstrap/` server codebase

---

## Component 5: Node Wiring

### node.py Changes

Three import lines and a config branch:

```python
if self.config.transport_backend == "libp2p":
    from prsm.node.libp2p_transport import Libp2pTransport
    from prsm.node.libp2p_gossip import Libp2pGossip
    from prsm.node.libp2p_discovery import Libp2pDiscovery
    self.transport = Libp2pTransport(identity, host, port, ...)
    self.gossip = Libp2pGossip(self.transport, ...)
    self.discovery = Libp2pDiscovery(self.transport, bootstrap_nodes=..., ...)
else:
    # Legacy WebSocket transport
    self.transport = WebSocketTransport(identity, host, port, ...)
    self.gossip = GossipProtocol(self.transport, ...)
    self.discovery = PeerDiscovery(self.transport, bootstrap_nodes=..., ...)
```

All downstream wiring is identical — consumers receive the transport/gossip/discovery objects through dependency injection and don't know which backend is active.

### config.py Changes

```python
# New fields
transport_backend: str = "libp2p"       # "libp2p" or "websocket"
libp2p_library_path: str = ""           # Auto-detected if empty
enable_relay: bool = True               # Circuit Relay v2
enable_nat_traversal: bool = True       # AutoNAT + hole punching
dht_mode: str = "auto"                  # "server" (public IP), "client" (behind NAT), "auto"

# Bootstrap format changes
bootstrap_nodes: List[str] = field(default_factory=lambda: [
    "/ip4/159.65.x.x/udp/9001/quic-v1/p2p/<bootstrap-peer-id>",
])
```

### Docker Changes

`docker-compose.bootstrap.yml` simplifies — the bootstrap "server" is now just a regular PRSM node configured with `dht_mode: server`:

```yaml
services:
  prsm-bootstrap:
    image: prsm-node:latest    # Same image as regular nodes
    environment:
      PRSM_DHT_MODE: server    # Public DHT server mode
      PRSM_P2P_PORT: 9001
    ports:
      - "9001:9001/udp"        # QUIC
      - "9001:9001/tcp"        # TCP fallback
      - "8000:8000"            # HTTP API
```

---

## Files Changed

| File | Action | Notes |
|------|--------|-------|
| `libp2p/` | **Create** | Go module, shared library source, build scripts |
| `libp2p/build/` | **Create** | Pre-built binaries (Linux, macOS) |
| `prsm/node/libp2p_transport.py` | **Create** | Python adapter, same interface as WebSocketTransport |
| `prsm/node/libp2p_gossip.py` | **Create** | GossipSub wrapper, same interface as GossipProtocol |
| `prsm/node/libp2p_discovery.py` | **Create** | DHT wrapper, same interface as PeerDiscovery |
| `prsm/node/node.py` | **Modify** | Config branch to select transport backend (~10 lines) |
| `prsm/node/config.py` | **Modify** | Add libp2p config fields, multiaddr bootstrap format |
| `prsm/node/transport.py` | **Keep** | Legacy fallback, not deleted |
| `prsm/node/gossip.py` | **Keep** | Legacy fallback, not deleted |
| `prsm/node/discovery.py` | **Keep** | Legacy fallback, not deleted |
| `prsm/bootstrap/` | **Deprecate** | No longer needed for new deployments |
| `docker/docker-compose.bootstrap.yml` | **Modify** | Simplified to regular PRSM node |
| `pyproject.toml` | **No change** | ctypes is stdlib, Go is build-time only |
| `tests/` | **Create** | Unit + integration tests for new transport |

## Consumer Modules — No Changes

These files are NOT modified:
- `prsm/node/compute_provider.py`
- `prsm/node/compute_requester.py`
- `prsm/node/storage_provider.py`
- `prsm/node/content_uploader.py`
- `prsm/node/content_provider.py`
- `prsm/node/agent_registry.py`
- `prsm/node/agent_collaboration.py`
- `prsm/node/ledger_sync.py`
- `prsm/node/bittorrent_provider.py`
- `prsm/node/bittorrent_requester.py`
- `prsm/node/content_economy.py`
- All MCP server tools
- All CLI commands
- All API endpoints

---

## Testing Strategy

**Milestone 0 — "Hello World" FFI bridge:**
- Go library compiles and loads from Python via ctypes
- `PrsmStart` initializes a host and returns a valid handle
- Ed25519 key passed from Python produces the expected libp2p peer ID
- This validates the entire build pipeline and key derivation before building the full data plane

**Unit tests (mock FFI):**
- `Libp2pTransport` adapter: start/stop, send, receive, peer tracking
- `Libp2pGossip` wrapper: publish/subscribe, topic mapping, callback dispatch
- `Libp2pDiscovery` wrapper: bootstrap, capability index, peer lookup

**Integration tests (real Go library):**
- Two nodes discover each other via mDNS on localhost
- Node connects to bootstrap peer via multiaddr
- GossipSub message published by node A arrives at node B
- Direct stream message from A to B and back
- DHT provide/find for content routing

**E2E tests:**
- Full forge pipeline over libp2p transport (query → dispatch → execute → aggregate → settle)
- Existing cross-node E2E test suite runs against libp2p backend

**Backward compat:**
- Existing test suite continues to pass with `transport_backend: websocket`

---

## CI/CD: Shared Library Build

Pre-built binaries in `libp2p/build/` must stay in sync with the Go source. A GitHub Actions workflow triggers on changes to `libp2p/` and cross-compiles for all target platforms:

```yaml
# .github/workflows/libp2p-build.yml
on:
  push:
    paths: ['libp2p/**']
    branches: [main]

jobs:
  build:
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            goos: linux
            goarch: amd64
            ext: .so
          - os: macos-latest
            goos: darwin
            goarch: arm64
            ext: .dylib
          - os: macos-13
            goos: darwin
            goarch: amd64
            ext: .dylib
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'
      - run: |
          cd libp2p
          CGO_ENABLED=1 GOOS=${{ matrix.goos }} GOARCH=${{ matrix.goarch }} \
            go build -buildmode=c-shared \
            -o build/libprsm_p2p_${{ matrix.goos }}_${{ matrix.goarch }}${{ matrix.ext }} \
            ./cmd/libprsm/
      - uses: actions/upload-artifact@v4
        with:
          name: libprsm-${{ matrix.goos }}-${{ matrix.goarch }}
          path: libp2p/build/
```

This prevents version drift where Python expects a new C function signature but the committed binary is stale.

---

## Migration Strategy

Since the network is pre-launch with no existing user base, this is a clean cutover — not a gradual migration. The `transport_backend` config toggle (`"libp2p"` vs `"websocket"`) exists for developer testing only, not for network-level protocol coexistence. Both sides of a connection must speak the same transport protocol. When we ship libp2p as default, all nodes upgrade together.
