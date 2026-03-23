# BitTorrent Integration: Implementation Plan

## Overview

BitTorrent will be added as a first-class peer-to-peer transport layer alongside IPFS. It is best
suited for large, high-demand datasets and model weights where swarm-based distribution outperforms
IPFS's content-addressed retrieval. The design mirrors existing IPFS and storage provider patterns
so every new module feels native to the codebase.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│                      PRSM Node                          │
│                                                         │
│  ┌─────────────────┐      ┌─────────────────────────┐  │
│  │  BitTorrent     │      │  BitTorrent             │  │
│  │  Provider       │◄────►│  Requester              │  │
│  │  (seeder)       │      │  (leecher/downloader)   │  │
│  └────────┬────────┘      └──────────┬──────────────┘  │
│           │                          │                  │
│  ┌────────▼──────────────────────────▼──────────────┐  │
│  │            BitTorrentClient (core)               │  │
│  │   libtorrent backend · DHT · piece verification  │  │
│  └────────┬──────────────────────────────────────┬──┘  │
│           │                                      │      │
│  ┌────────▼────────┐             ┌───────────────▼────┐ │
│  │  GossipProtocol │             │  LocalLedger/FTNS  │ │
│  │  (announce/     │             │  (seeding rewards, │ │
│  │   discovery)    │             │   download costs)  │ │
│  └─────────────────┘             └────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1 — Core Infrastructure (Foundation)

### 1.1 New Dependency

Add to `pyproject.toml` and `requirements.txt`:

```toml
# pyproject.toml — add optional group
[project.optional-dependencies]
bittorrent = [
    "libtorrent>=2.0.9",    # C++-backed, production-grade BT library
    "bencode.py>=4.0.0",    # .torrent file encoding/decoding
]
```

`libtorrent-rasterbar` is the only production-quality choice — it powers qBittorrent, Deluge, and
Transmission. The Python bindings are well-maintained and fully async-compatible via thread pool
offloading.

---

### 1.2 `prsm/core/bittorrent_client.py`

Mirrors `ipfs_client.py` in structure. This is the lowest-level abstraction over libtorrent.

**Classes:**

```
BitTorrentConfig          # Dataclass: ports, DHT bootstrap nodes, download dir,
                          # max connections, bandwidth limits, alert poll interval

BitTorrentResult          # Dataclass: success bool, infohash, error, metadata dict

TorrentInfo               # Dataclass: infohash, name, size_bytes, piece_length,
                          # num_pieces, files: List[FileEntry], created_at,
                          # seeders, leechers, download_rate, upload_rate,
                          # progress float (0.0–1.0), state (downloading/seeding/etc.)

BitTorrentClient          # Main class
  async initialize()        # Start libtorrent session, connect DHT bootstrap nodes,
                            # configure port range and bandwidth limits
  async shutdown()          # Graceful teardown, flush state

  async create_torrent(     # Hash content, build .torrent file, return TorrentInfo
      path: Path,
      piece_length: int = 262144,   # 256KB default (configurable)
      comment: str = "",
      private: bool = False,
  ) -> BitTorrentResult

  async add_torrent(        # Add from .torrent bytes or magnet URI, begin seeding
      source: Union[bytes, str],    # .torrent bytes or "magnet:?xt=..."
      save_path: Optional[Path] = None,
      seed_mode: bool = False,      # True = already have all data, just seed
  ) -> BitTorrentResult

  async remove_torrent(     # Stop and optionally delete data
      infohash: str,
      delete_files: bool = False,
  ) -> BitTorrentResult

  async get_status(         # Return TorrentInfo for one or all active torrents
      infohash: Optional[str] = None,
  ) -> Union[TorrentInfo, List[TorrentInfo]]

  async wait_for_completion(  # Block until torrent finishes (with timeout)
      infohash: str,
      timeout: float = 3600.0,
      progress_callback: Optional[Callable] = None,
  ) -> BitTorrentResult

  async get_peers(          # Return list of peers currently connected for torrent
      infohash: str,
  ) -> List[PeerInfo]

  async _poll_alerts()      # Background task: drains libtorrent alert queue,
                            # fires registered callbacks (piece complete, error, etc.)

  def on_alert(             # Register callback for libtorrent alert types
      alert_type: str,
      callback: Callable,
  )
```

**Key design decisions:**
- libtorrent is synchronous C++; wrap all calls in `asyncio.get_event_loop().run_in_executor(None, ...)` to keep everything non-blocking
- Alert polling runs as a background `asyncio.Task` every 100ms
- All progress callbacks receive `(infohash: str, progress: float, stats: dict)`
- Errors return `BitTorrentResult(success=False, error=...)` — never raise exceptions to callers

---

### 1.3 `prsm/core/bittorrent_manifest.py`

Mirrors `ipfs_sharding.py`. Provides a metadata layer so PRSM can track torrents the same way it
tracks IPFS shards.

**Classes:**

```
PieceInfo                 # Dataclass: index, hash (SHA-1 from BT spec),
                          # size, verified bool

FileEntry                 # Dataclass: path, size_bytes, offset_in_torrent

TorrentManifest           # Dataclass: infohash, name, total_size,
                          # piece_length, pieces: List[PieceInfo],
                          # files: List[FileEntry], magnet_uri,
                          # torrent_bytes (raw .torrent),
                          # ipfs_cid (optional: .torrent file pinned to IPFS),
                          # created_at, created_by_node_id,
                          # provenance_id (links to PRSM provenance system)

TorrentManifestIndex      # In-memory index
  add(manifest)             # Index by infohash and name
  get_by_infohash(str)      # O(1) lookup
  search(query)             # Fuzzy name search
  evict_lru(max_size)       # LRU eviction when over limit

TorrentManifestStore      # Persistent store (backed by PostgreSQL or local SQLite)
  async save(manifest)
  async load(infohash) -> Optional[TorrentManifest]
  async list_all() -> List[TorrentManifest]
  async delete(infohash)
```

**Design note:** The `.torrent` file itself should optionally be pinned to IPFS for durability.
This creates a clean bridge: IPFS stores the metadata/manifest, BitTorrent distributes the payload
at scale.

---

## Phase 2 — Node-Level Provider Integration

### 2.1 `prsm/node/bittorrent_provider.py`

Mirrors `storage_provider.py`. This is the seeder side — a node that holds data and offers it to
the swarm.

**Classes:**

```
BitTorrentProviderConfig  # Dataclass: max_upload_mbps, max_torrents,
                          # data_dir, reward_interval_secs, min_seed_time_secs

ActiveTorrent             # Dataclass: infohash, manifest, started_at,
                          # bytes_uploaded, last_reward_at, peer_count

BitTorrentProvider
  __init__(
      identity: NodeIdentity,
      transport: WebSocketTransport,
      gossip: GossipProtocol,
      ledger: LocalLedger,
      bt_client: BitTorrentClient,
      manifest_store: TorrentManifestStore,
      config: BitTorrentProviderConfig,
  )

  async start()             # Register gossip subscriptions, start reward loop,
                            # resume any previously active torrents from DB
  async stop()              # Flush state, stop all torrents gracefully

  async seed_content(       # Create torrent from local path, begin seeding,
      path: Path,           # announce on gossip, store manifest
      name: str,
      provenance_id: Optional[str] = None,
  ) -> TorrentManifest

  async stop_seeding(       # Remove torrent from swarm, announce withdrawal
      infohash: str,
  )

  async _announce(          # Publish GOSSIP_BITTORRENT_ANNOUNCE with TorrentManifest
      manifest: TorrentManifest,
  )

  async _reward_loop()      # Background task: every reward_interval_secs,
                            # calculate uploaded bytes per torrent,
                            # call ledger.credit() for seeding rewards,
                            # update last_reward_at

  async _on_peer_connected( # libtorrent alert callback: log peer event,
      infohash, peer_info   # update ActiveTorrent.peer_count
  )

  def get_active_torrents() -> List[ActiveTorrent]
```

**Gossip messages published:**
- `GOSSIP_BITTORRENT_ANNOUNCE`: `{infohash, name, size_bytes, magnet_uri, piece_length, num_pieces, seeder_node_id, timestamp}`
- `GOSSIP_BITTORRENT_WITHDRAW`: `{infohash, seeder_node_id}` — when stopping

---

### 2.2 `prsm/node/bittorrent_requester.py`

Mirrors `compute_requester.py`. The downloader/consumer side.

**Classes:**

```
DownloadRequest           # Dataclass: request_id, infohash, name,
                          # requester_node_id, save_path, created_at

DownloadResult            # Dataclass: request_id, infohash, success,
                          # path, bytes_downloaded, duration_secs, error

BitTorrentRequester
  __init__(
      identity: NodeIdentity,
      gossip: GossipProtocol,
      bt_client: BitTorrentClient,
      manifest_store: TorrentManifestStore,
      ledger: LocalLedger,
      config: BitTorrentRequesterConfig,
  )

  async start()             # Subscribe to GOSSIP_BITTORRENT_ANNOUNCE,
                            # populate local discovery index

  async request_content(    # Find manifest via gossip index or infohash,
      infohash: str,        # add torrent to BT client, wait for completion,
      save_path: Path,      # deduct FTNS from requester
      timeout: float = 3600.0,
      progress_callback: Optional[Callable] = None,
  ) -> DownloadResult

  async find_torrent(       # Look up TorrentManifest in local gossip index
      infohash: str,        # (populated from announcements)
  ) -> Optional[TorrentManifest]

  async list_available()    # Return all known torrents from gossip index
      -> List[TorrentManifest]

  async _on_announce(       # Gossip subscription callback:
      subtype, data, origin # parse announcement, upsert into discovery index
  )

  async _charge_download(   # Deduct FTNS for downloaded bytes
      infohash: str,        # (optional: implement per-GB pricing)
      bytes_downloaded: int,
  )
```

---

### 2.3 Modifications to `prsm/node/node.py`

Add BitTorrent as a first-class component:

```python
# __init__ additions
self.bt_client = BitTorrentClient(config=bt_config)
self.bt_manifest_store = TorrentManifestStore(db_url=config.database_url)
self.bt_provider = BitTorrentProvider(
    identity=self.identity,
    transport=self.transport,
    gossip=self.gossip,
    ledger=self.ledger,
    bt_client=self.bt_client,
    manifest_store=self.bt_manifest_store,
    config=bt_provider_config,
)
self.bt_requester = BitTorrentRequester(
    identity=self.identity,
    gossip=self.gossip,
    bt_client=self.bt_client,
    manifest_store=self.bt_manifest_store,
    ledger=self.ledger,
    config=bt_requester_config,
)

# startup() additions (in order)
await self.bt_client.initialize()
await self.bt_manifest_store.initialize()
await self.bt_provider.start()
await self.bt_requester.start()

# shutdown() additions
await self.bt_provider.stop()
await self.bt_requester.stop()
await self.bt_client.shutdown()
```

---

### 2.4 Modifications to `prsm/node/gossip.py`

Add new message type constants and retention periods:

```python
# New constants
GOSSIP_BITTORRENT_ANNOUNCE  = "bittorrent_announce"
GOSSIP_BITTORRENT_WITHDRAW  = "bittorrent_withdraw"
GOSSIP_BITTORRENT_STATS     = "bittorrent_stats"

# Retention (add to GOSSIP_RETENTION_SECONDS dict)
GOSSIP_BITTORRENT_ANNOUNCE: 86400,   # 24 hours — same as content
GOSSIP_BITTORRENT_WITHDRAW: 3600,    # 1 hour
GOSSIP_BITTORRENT_STATS:    1800,    # 30 minutes — stats decay quickly
```

---

## Phase 3 — Storage Proofs for BitTorrent Seeders

### 3.1 `prsm/node/bittorrent_proofs.py`

Extends the existing proof-of-storage system to work with torrent pieces.

**The challenge:** BitTorrent pieces are hashed with SHA-1 (spec requirement). The existing PRSM
proof system uses SHA-256 Merkle proofs. This module bridges the two.

**Classes:**

```
TorrentPieceChallenge     # Dataclass: challenge_id, infohash, piece_index,
                          # expected_hash (SHA-1), nonce, deadline,
                          # challenger_node_id

TorrentPieceProof         # Dataclass: challenge_id, infohash, piece_index,
                          # piece_data (bytes), sha1_hash, sha256_hash,
                          # responder_node_id, timestamp

TorrentProofVerifier
  async issue_challenge(    # Send challenge to a known seeder via direct P2P
      seeder_node_id: str,
      infohash: str,
      manifest: TorrentManifest,
  ) -> TorrentPieceChallenge

  async verify_proof(       # Validate piece data matches expected hash
      proof: TorrentPieceProof,
      challenge: TorrentPieceChallenge,
  ) -> bool

  async award_verified_seeder(   # Credit FTNS for honest proof response
      node_id: str,
      amount: Decimal,
      ledger: LocalLedger,
  )

TorrentProofResponder
  async respond_to_challenge(    # Load piece from libtorrent, compute hashes,
      challenge: TorrentPieceChallenge,   # send TorrentPieceProof back
      bt_client: BitTorrentClient,
  ) -> TorrentPieceProof
```

**P2P message types added to transport:**
- `MSG_BT_PIECE_CHALLENGE`: Direct message, challenger → seeder
- `MSG_BT_PIECE_PROOF`: Direct message, seeder → challenger

---

## Phase 4 — Economy Integration

### 4.1 Add FTNS Transaction Types

In `prsm/economy/tokenomics/ftns_service.py`, extend the `FTNSTransactionType` enum:

```python
BITTORRENT_SEEDING_REWARD   = "bittorrent_seeding_reward"
BITTORRENT_DOWNLOAD_FEE     = "bittorrent_download_fee"
BITTORRENT_PROOF_REWARD     = "bittorrent_proof_reward"
BITTORRENT_PROOF_SLASH      = "bittorrent_proof_slash"   # dishonest seeder
```

### 4.2 Pricing Model

Implemented inside `BitTorrentProvider._reward_loop()` and `BitTorrentRequester._charge_download()`:

| Action | FTNS Flow | Rate (configurable) |
|--------|-----------|---------------------|
| Upload 1 GB to swarm | Seeder earns FTNS | `SEEDER_REWARD_PER_GB` |
| Download 1 GB | Requester pays FTNS | `DOWNLOAD_COST_PER_GB` |
| Pass proof challenge | Seeder earns bonus | `PROOF_REWARD_AMOUNT` |
| Fail proof challenge | Seeder loses FTNS | `PROOF_SLASH_AMOUNT` |

These constants live in `prsm/core/config.py` under a new `[bittorrent]` config section.

---

## Phase 5 — API Layer

### 5.1 `prsm/interface/api/routers/bittorrent_router.py`

New FastAPI router. All endpoints require authentication (same JWT middleware as other routers).

```
POST   /api/v1/torrents/create                           # Create torrent from content CID or upload
POST   /api/v1/torrents/add                              # Add existing torrent (magnet or .torrent bytes)
GET    /api/v1/torrents                                  # List all torrents (seeding + available)
GET    /api/v1/torrents/{infohash}                       # Get full TorrentManifest + live status
POST   /api/v1/torrents/{infohash}/seed                  # Start seeding a torrent
DELETE /api/v1/torrents/{infohash}/seed                  # Stop seeding
POST   /api/v1/torrents/{infohash}/download              # Begin download, return request_id
GET    /api/v1/torrents/{infohash}/download/{request_id} # Poll download status
GET    /api/v1/torrents/{infohash}/peers                 # List connected peers
GET    /api/v1/torrents/stats                            # Aggregate seeding/download stats for this node
```

Register in `app_factory.py`:

```python
from prsm.interface.api.routers.bittorrent_router import bittorrent_router
app.include_router(bittorrent_router, prefix="/api/v1", tags=["BitTorrent"])
```

---

## Phase 6 — Configuration

### 6.1 `prsm/core/config.py` additions

New `BitTorrentSettings` block:

```python
class BitTorrentSettings(BaseSettings):
    enabled: bool = True
    port_range_start: int = 6881
    port_range_end: int = 6891
    dht_enabled: bool = True
    dht_bootstrap_nodes: List[str] = [
        "router.bittorrent.com:6881",
        "router.utorrent.com:6881",
        "dht.transmissionbt.com:6881",
    ]
    max_upload_mbps: float = 10.0
    max_download_mbps: float = 50.0
    max_active_torrents: int = 50
    data_dir: str = "~/.prsm/torrents"
    seeder_reward_per_gb: Decimal = Decimal("0.10")   # FTNS
    download_cost_per_gb: Decimal = Decimal("0.05")   # FTNS
    proof_reward_amount: Decimal = Decimal("0.01")    # FTNS
    proof_slash_amount: Decimal = Decimal("0.05")     # FTNS
    reward_interval_secs: int = 3600
    proof_challenge_interval_secs: int = 7200
    announce_interval_secs: int = 1800
```

### 6.2 `.env.example` additions

```
# BitTorrent
PRSM_BT_ENABLED=true
PRSM_BT_PORT_RANGE_START=6881
PRSM_BT_PORT_RANGE_END=6891
PRSM_BT_MAX_UPLOAD_MBPS=10.0
PRSM_BT_MAX_DOWNLOAD_MBPS=50.0
PRSM_BT_DATA_DIR=~/.prsm/torrents
PRSM_BT_SEEDER_REWARD_PER_GB=0.10
PRSM_BT_DOWNLOAD_COST_PER_GB=0.05
```

---

## Phase 7 — Database Schema

### 7.1 New Alembic migration: `alembic/versions/xxx_add_bittorrent_tables.py`

Two new tables:

```sql
CREATE TABLE torrent_manifests (
    infohash         VARCHAR(40) PRIMARY KEY,  -- SHA-1 hex
    name             TEXT NOT NULL,
    total_size_bytes BIGINT NOT NULL,
    piece_length     INTEGER NOT NULL,
    num_pieces       INTEGER NOT NULL,
    magnet_uri       TEXT NOT NULL,
    torrent_bytes    BYTEA,
    ipfs_cid         VARCHAR(64),              -- Optional: .torrent file pinned to IPFS
    created_at       TIMESTAMPTZ NOT NULL,
    created_by       VARCHAR(64) NOT NULL,     -- node_id
    provenance_id    UUID REFERENCES provenance(id),
    metadata         JSONB
);

CREATE TABLE torrent_seeder_log (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    infohash         VARCHAR(40) REFERENCES torrent_manifests(infohash),
    seeder_node_id   VARCHAR(64) NOT NULL,
    bytes_uploaded   BIGINT DEFAULT 0,
    reward_paid      NUMERIC(20,8) DEFAULT 0,
    started_at       TIMESTAMPTZ NOT NULL,
    last_seen_at     TIMESTAMPTZ NOT NULL,
    active           BOOLEAN DEFAULT TRUE
);
```

---

## New Files Summary

| File | Description |
|------|-------------|
| `prsm/core/bittorrent_client.py` | Core libtorrent wrapper (async) |
| `prsm/core/bittorrent_manifest.py` | TorrentManifest + persistence |
| `prsm/node/bittorrent_provider.py` | Seeder node component |
| `prsm/node/bittorrent_requester.py` | Downloader node component |
| `prsm/node/bittorrent_proofs.py` | Challenge-response proof system |
| `prsm/interface/api/routers/bittorrent_router.py` | REST API endpoints |
| `alembic/versions/xxx_add_bittorrent_tables.py` | DB migration |
| `tests/unit/test_bittorrent_client.py` | Unit tests |
| `tests/unit/test_bittorrent_provider.py` | Unit tests |
| `tests/integration/test_bittorrent_swarm.py` | Multi-node integration tests |

**Modified files:**
- `prsm/node/node.py`
- `prsm/node/gossip.py`
- `prsm/core/config.py`
- `prsm/economy/tokenomics/ftns_service.py`
- `prsm/interface/api/app_factory.py`
- `pyproject.toml`
- `requirements.txt`
- `.env.example`

---

## Implementation Order

```
Phase 1: bittorrent_client.py + bittorrent_manifest.py  (pure core, no node wiring)
Phase 2: bittorrent_provider.py + bittorrent_requester.py + node.py wiring
Phase 3: bittorrent_proofs.py + gossip.py additions
Phase 4: ftns_service.py extensions + pricing logic in provider/requester
Phase 5: bittorrent_router.py + app_factory.py registration
Phase 6: config.py + .env.example + pyproject.toml
Phase 7: Alembic migration
         Unit tests throughout; swarm integration tests last
```

Each phase is independently testable before moving to the next.

---

## Implementation Completion Summary

**Completion Date:** 2026-03-23

### Work Completed

#### Phase 1 - Core Infrastructure ✅
- `prsm/core/bittorrent_client.py` - Full async wrapper around libtorrent with all planned methods
- `prsm/core/bittorrent_manifest.py` - TorrentManifest, PieceInfo, FileEntry dataclasses and persistence
- `pyproject.toml` - Added `bittorrent` optional dependencies group with libtorrent and bencodepy

#### Phase 2 - Node-Level Provider Integration ✅
- `prsm/node/bittorrent_provider.py` - Seeder component with reward loop, gossip announcements
- `prsm/node/bittorrent_requester.py` - Downloader component with request tracking
- `prsm/node/node.py` - Full wiring of bt_client, bt_provider, bt_requester in startup/shutdown

#### Phase 3 - Storage Proofs ✅
- `prsm/node/bittorrent_proofs.py` - Challenge-response proof system bridging SHA-1 (BitTorrent) and SHA-256 (PRSM)
- `prsm/node/gossip.py` - Added GOSSIP_BITTORRENT_ANNOUNCE, WITHDRAW, STATS, REQUEST message types and retention periods

#### Phase 4 - Economy Integration ✅
- `prsm/economy/tokenomics/ftns_service.py` - Added BITTORRENT_SEEDING_REWARD, DOWNLOAD_FEE, PROOF_REWARD, PROOF_SLASH transaction types

#### Phase 5 - API Layer ✅
- `prsm/interface/api/routers/bittorrent_router.py` - Full REST API for torrent management
- `prsm/interface/api/router_registry.py` - Router registered in _include_service_routers()

#### Phase 6 - Configuration ✅
- `prsm/core/config.py` - All BitTorrent settings added (bt_enabled, port range, bandwidth limits, FTNS rates)
- `.env.example` - Complete BitTorrent section (Section 18) with all configuration options
- `pyproject.toml` - Optional dependencies group added

#### Phase 7 - Database Schema ✅
- `alembic/versions/011_add_bittorrent_tables.py` - Migration for torrent_manifests and torrent_seeder_log tables

#### Unit Tests ✅
- `tests/unit/test_bittorrent_client.py` - 30 tests covering config, dataclasses, client initialization, and operations
- `tests/unit/test_bittorrent_provider.py` - 17 tests covering provider config, seeding, and reward calculation

### Deviations from Plan

1. **Router registration location:** The plan suggested registering in `app_factory.py`, but the actual implementation uses `router_registry.py` which is the pattern used by other routers in the codebase. This follows the existing convention.

2. **Integration tests:** Multi-node swarm integration tests were not created as part of this implementation. These would require a more complex test setup with actual P2P networking. Unit tests provide coverage for the core functionality.

### Files Created
- `prsm/core/bittorrent_client.py`
- `prsm/core/bittorrent_manifest.py`
- `prsm/node/bittorrent_provider.py`
- `prsm/node/bittorrent_requester.py`
- `prsm/node/bittorrent_proofs.py`
- `prsm/interface/api/routers/bittorrent_router.py`
- `alembic/versions/011_add_bittorrent_tables.py`
- `tests/unit/test_bittorrent_client.py`
- `tests/unit/test_bittorrent_provider.py`

### Files Modified
- `prsm/node/node.py` - Added BitTorrent client wiring
- `prsm/node/gossip.py` - Added BitTorrent message types
- `prsm/core/config.py` - Added BitTorrent settings
- `prsm/economy/tokenomics/ftns_service.py` - Added BitTorrent transaction types
- `prsm/interface/api/router_registry.py` - Registered BitTorrent router
- `pyproject.toml` - Added bittorrent optional dependencies
- `.env.example` - Added BitTorrent configuration section

### Status
All 7 phases complete. The BitTorrent integration is ready for integration testing and deployment.
