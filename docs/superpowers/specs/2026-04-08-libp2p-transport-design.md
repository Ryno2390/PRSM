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

| Function | Signature | Purpose |
|----------|-----------|---------|
| `PrsmStart` | `(ed25519_key *C.char, listen_port C.int, bootstrap_addrs *C.char) → *C.char` | Start libp2p host, returns host peer ID |
| `PrsmStop` | `() → C.int` | Graceful shutdown, returns 0 on success |
| `PrsmConnect` | `(multiaddr *C.char) → *C.char` | Dial peer by multiaddr, returns peer ID |
| `PrsmSend` | `(peer_id *C.char, protocol *C.char, data *C.char, data_len C.int) → C.int` | Send bytes to peer via direct stream |
| `PrsmPublish` | `(topic *C.char, data *C.char, data_len C.int) → C.int` | Publish to GossipSub topic |
| `PrsmSubscribe` | `(topic *C.char) → C.int` | Subscribe to GossipSub topic |
| `PrsmUnsubscribe` | `(topic *C.char) → C.int` | Unsubscribe from topic |
| `PrsmPeerCount` | `() → C.int` | Connected peer count |
| `PrsmPeerList` | `() → *C.char` | JSON array of connected peer info |
| `PrsmDHTProvide` | `(key *C.char) → C.int` | Announce content availability on DHT |
| `PrsmDHTFindProviders` | `(key *C.char, limit C.int) → *C.char` | Find content providers, returns JSON |
| `PrsmSetCallback` | `(fn C.callback_fn) → C.int` | Register Python callback for incoming messages |
| `PrsmGetNATStatus` | `() → *C.char` | Returns NAT type: "public", "private", "unknown" |

### Callback from Go to Python

Go delivers incoming messages (both GossipSub and direct streams) by calling a C function pointer registered via `PrsmSetCallback`. The callback signature:

```c
typedef void (*callback_fn)(const char* msg_type, const char* topic_or_peer, const char* data, int data_len, const char* sender_id);
```

- `msg_type`: `"gossip"` or `"direct"`
- `topic_or_peer`: GossipSub topic name or sender peer ID
- `data`/`data_len`: raw message bytes
- `sender_id`: libp2p peer ID of the sender

### Identity

The Go host derives its libp2p identity from the same Ed25519 private key that the Python node already uses. The key is passed as raw bytes to `PrsmStart`. This means the node's libp2p peer ID is deterministically derived from its existing PRSM identity — no new key management.

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
│   └── callback.go          # C callback bridge
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

Platform detection selects `.so`, `.dylib`, or `.dll` automatically.

### Callback Bridge (Go → Python)

Go calls a C function pointer to deliver incoming messages. The ctypes callback puts messages onto an `asyncio.Queue`. A background coroutine reads the queue and dispatches to registered handlers:

```python
@ctypes.CFUNCTYPE(None, c_char_p, c_char_p, c_char_p, c_int, c_char_p)
def _on_message(msg_type, topic_or_peer, data, data_len, sender_id):
    # Called from Go thread — must not call async code directly
    loop.call_soon_threadsafe(queue.put_nowait, (msg_type, topic_or_peer, data, sender_id))

# Background dispatcher
async def _dispatch_loop():
    while True:
        msg_type, topic, data, sender = await queue.get()
        for handler in handlers[msg_type]:
            await handler(deserialize(data), peer_info)
```

This design avoids GIL contention and keeps all async handler execution on the Python event loop.

### Address Translation

The adapter transparently translates between PRSM address formats and libp2p multiaddrs:

- `wss://host:port` → `/ip4/{host}/tcp/{port}/ws` (legacy compat)
- `host:port` → `/ip4/{host}/udp/{port}/quic-v1`
- `/ip4/...` → passed through as-is (native multiaddr)

### P2PMessage Serialization

P2PMessage is serialized to JSON bytes when crossing the FFI boundary. The Go side treats message content as opaque bytes — it doesn't parse or validate PRSM message structure. Signing and verification remain in Python.

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

- **Fanout and mesh management** — GossipSub maintains 6 mesh peers per topic, 8 lazy peers for protocol-level redundancy protocol protocol
- **Message deduplication** — by message ID (hash of content), no nonce tracking needed
- **Late-joiner catch-up** — IHAVE/IWANT protocol exchanges message IDs with new peers
- **Heartbeat** — 1-second internal heartbeat for mesh maintenance
- **Flood publishing** — initial publish goes to all mesh peers, then lazy forwarding

### What the Wrapper Still Does

- **Callback registry** — maps topic → list of Python callbacks
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

**Content routing (new):** Exposes DHT provide/find to the rest of PRSM:
```python
async def provide_content(self, cid: str) -> None:
    """Announce on DHT that this node has a specific content shard."""

async def find_content_providers(self, cid: str, limit: int = 20) -> List[PeerInfo]:
    """Find nodes that have a specific content shard via DHT."""
```

This directly benefits the semantic sharding system — shard CIDs can be advertised and discovered without gossip flooding.

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
