# Native Content-Addressed Storage — Design Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## Goal

Replace PRSM's direct IPFS/Kubo dependency with a native content-addressed storage system. IPFS was originally referenced as inspiration for how latent storage could be monetized — not as infrastructure to depend on. The native system brings storage in line with how compute already works: PRSM's own protocol for monetizing latent resources, with no external daemon required.

## Architecture

A unified `prsm/storage/` package replaces all six IPFS library files and their consumers. The package contains focused internal modules (blob store, shard engine, key manager, distribution manager) exposed through a single `ContentStore` interface. Sharding is a core security property — no single node holds complete content. Manifests are encrypted via Shamir's Secret Sharing so no single compromise reveals the reconstruction map. Shard placement enforces geographic diversity and owner exclusion. The entire system is algorithm-agile and post-quantum ready.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Content IDs | SHA-256 with algorithm-agility prefix byte | PQ-ready via upgrade path, zero dependencies |
| Sharding | Kept (security through fragmentation) | Core security property — compromising one node yields useless fragments |
| Manifest storage | Encrypted, replicated across N nodes | Must outlive original uploader (provenance inheritance) |
| Manifest encryption | Shamir's Secret Sharing (threshold) | Decentralized, resilient to node churn, no central authority |
| Storage layout | Flat content-addressed, hash-prefix subdirs | Automatic deduplication, filesystem-friendly |
| Replication | Owner-specified, paid via FTNS | Aligns with economic model, prevents freeloading |
| Shard placement | Geographic/ASN diversity + owner-excluded | Correlated failure protection + security hardening |
| Retrieval model | Pull — requester-driven concurrent requests | Simple, no coordinator, leverages libp2p multiplexing |
| WASM agents | Separate concern, not involved in retrieval | Agents are sandboxed; node layer handles storage |
| Migration | Clean break, no adapter period | Pre-production, no user data to migrate |

---

## 1. Module Structure & Content Addressing

### Package layout

```
prsm/storage/
  __init__.py          # Public API: ContentStore, ContentHash
  blob_store.py        # Local content-addressed file I/O
  shard_engine.py      # Split/reassemble + manifest management
  key_manager.py       # Shamir's Secret Sharing for manifest encryption
  distribution.py      # Replication, placement, P2P shard retrieval
  models.py            # Shared dataclasses
  exceptions.py        # Storage-specific exceptions
```

### Content hash format

```
[1 byte: algorithm ID][32 bytes: hash digest]

Algorithm IDs:
  0x01 = SHA-256 (default)
  0x02 = SHA-3-256 (reserved)
  0x03 = BLAKE3 (reserved)
```

Serialized as hex string for display and storage: `"01a1b2c3d4..."` (66 characters for SHA-256). The algorithm prefix costs 1 byte per hash and enables future migration to stronger algorithms without invalidating existing content.

### Blob store interface

```python
class BlobStore:
    def __init__(self, data_dir: str):
        """Local content-addressed file store."""

    async def store(self, data: bytes) -> ContentHash:
        """Hash content, write to disk, return hash. Deduplicates automatically."""

    async def retrieve(self, content_hash: ContentHash) -> bytes:
        """Read from disk by hash. Raises ContentNotFoundError if missing."""

    async def exists(self, content_hash: ContentHash) -> bool:
        """Check local presence."""

    async def delete(self, content_hash: ContentHash) -> None:
        """Remove from local store."""
```

Storage path: `{data_dir}/{first 2 hex chars of hash}/{remaining hex chars}`. The two-character prefix subdirectory prevents any single directory from growing too large.

---

## 2. Shard Engine

### Sharding behavior

- Content above a configurable threshold (default 1MB) is split into fixed-size chunks (default 256KB). Content below the threshold is stored as a single shard.
- Each shard is individually content-hashed and stored via the blob store.
- A manifest maps the original content hash to an ordered list of shard hashes plus metadata.

### Manifest structure

```python
@dataclass
class ShardManifest:
    content_hash: ContentHash          # Hash of the original complete content
    shard_hashes: List[ContentHash]    # Ordered list of shard hashes
    total_size: int                    # Original content size in bytes
    shard_size: int                    # Size of each shard (last may be smaller)
    algorithm_id: int                  # Hash algorithm used (0x01 = SHA-256)
    created_at: float                  # Unix timestamp
    replication_factor: int            # Owner-specified replica count
    owner_node_id: str                 # Original uploader's node ID
```

### Split flow

1. Hash the complete content -> `content_hash`
2. Split into chunks of `shard_size` bytes
3. Store each chunk via blob store -> ordered list of `shard_hashes`
4. Build manifest with all metadata
5. Serialize manifest to JSON, encrypt it (Section 3), distribute encrypted manifest

### Reassemble flow

1. Retrieve and decrypt manifest (Section 3 key reconstruction)
2. For each shard hash: pull from local blob store, or request via P2P if not local
3. Concatenate shards in manifest order
4. Hash the reassembled content, verify it matches `content_hash`
5. If mismatch -> integrity failure, reject

Individual shards are also verified against their hashes on retrieval, so a corrupted or tampered shard is identified immediately — you know which shard failed, not just that something is wrong.

---

## 3. Key Manager (Shamir's Secret Sharing)

### Manifest encryption

- Each manifest is encrypted with a random AES-256-GCM key (per-content, single-use).
- That key is split using Shamir's Secret Sharing into N shares with a threshold of K.
- Shares are distributed to nodes that hold zero shards of that content — separation of concerns between data fragment holders and key fragment holders.

### Key share structure

```python
@dataclass
class KeyShare:
    content_hash: ContentHash    # Which content this key is for
    share_index: int             # Shamir share index (1-N)
    share_data: bytes            # The share itself
    threshold: int               # K — minimum shares needed
    total_shares: int            # N — total shares created
    algorithm_id: int            # Key algorithm (0x01 = AES-256-GCM)
```

### Threshold defaults

| Content size | Shards | Key threshold |
|-------------|--------|---------------|
| Small (< 10 shards) | < 10 | 3-of-5 |
| Medium (10-100 shards) | 10-100 | 5-of-8 |
| Large (100+ shards) | 100+ | 7-of-12 |

More shards means more valuable content, warranting higher key redundancy.

### Reconstruction flow

1. Requesting node identifies key-share holders via DHT or discovery
2. Requests K shares from available holders via direct P2P
3. Reconstructs AES-256-GCM key via Shamir reconstruction
4. Decrypts manifest
5. Key is used in-memory only, never persisted in reconstructed form

### Provenance inheritance

When ownership transfers (node departure, explicit transfer), the manifest encryption key does NOT need re-encryption or re-distribution:

- Key shares remain on the same nodes
- The governance/provenance layer updates who is authorized to request reconstruction
- Only the access control list changes, not the cryptographic material

If too many key-share holders have left the network (remaining holders drop to threshold K), a **key refresh** is triggered: an authorized party reconstructs the key, re-splits with new shares, and distributes to current healthy nodes.

### Algorithm agility

The `algorithm_id` field on KeyShare allows future migration to post-quantum key encapsulation (e.g., ML-KEM/Kyber) without invalidating existing shares. New content uses the new algorithm; old content retains its existing shares.

---

## 4. Distribution Manager

### Shard placement algorithm

1. Owner uploads content -> shard engine splits and stores locally (temporary)
2. Distribution manager queries discovery for available storage providers
3. Filters out the owner's node (owner-exclusion constraint)
4. Groups remaining providers by ASN (autonomous system number, derived from libp2p peer address) as a proxy for geographic/infrastructure diversity
5. For each shard, selects providers from different ASN groups, weighted by:
   - `PeerInfo.reliability_score` (from existing libp2p discovery layer)
   - Available storage capacity
   - Constraint: no two shards of the same content on the same ASN group
6. Repeats until each shard has `replication_factor` copies across diverse nodes
7. Key shares (Section 3) are placed on nodes that hold zero shards of that content

### Replication policy

```python
@dataclass
class ReplicationPolicy:
    replication_factor: int        # Owner-specified (paid via FTNS)
    min_asn_diversity: int         # Minimum distinct ASN groups across shards
    owner_excluded: bool           # Always True — owner never holds own shards
    key_shard_separation: bool     # Always True — key holders != shard holders
```

### Replication health monitor

- Periodic sweep (configurable, default 5 minutes) checks each content's shard distribution
- If a shard drops below `replication_factor` (node went offline), triggers re-replication to a new provider from a different ASN group
- If a key-share holder goes offline and remaining holders approach threshold K, triggers key refresh (Section 3)
- Leverages existing `PeerInfo.last_seen` and reliability tracking from the libp2p discovery layer (shipped in v1.3.0)

### P2P shard retrieval (pull model)

1. Requester decrypts manifest (Section 3 reconstruction flow)
2. For each shard hash, queries DHT for providers holding that shard
3. Fires concurrent direct P2P requests (`transport.send_to_peer()`) to providers
4. Each provider reads shard from local blob store, returns via direct P2P response
5. Requester verifies each shard hash on receipt
6. After all shards received, reassembles and verifies complete content hash (Section 2)
7. On individual shard failure: retry from alternate provider (replication means multiple sources exist)

### FTNS payment integration

- Storage providers earn FTNS for: holding shards (ongoing), serving shard retrieval requests (per-serve), and holding key shares (ongoing)
- Content owners pay FTNS proportional to: `content_size * replication_factor * storage_duration`
- Integrates with existing content economy payment flows, replacing current IPFS-based payment triggers

---

## 5. Consumer Migration & Cleanup

### Files deleted

**IPFS client libraries (6 files):**
- `prsm/core/ipfs_client.py` — Replaced by `prsm/storage/blob_store.py`
- `prsm/core/ipfs_sharding.py` — Replaced by `prsm/storage/shard_engine.py`
- `prsm/core/infrastructure/ipfs_cdn_bridge.py` — No longer needed
- `prsm/data/ipfs/ipfs_client.py` — Replaced by storage module
- `prsm/data/ipfs/content_verification.py` — Verification built into shard engine
- `prsm/data/data_layer/enhanced_ipfs.py` — Consumers migrate to ContentStore

**Content addressing (absorbed into storage models):**
- `prsm/data/ipfs/content_addressing.py` — ContentHash replaces CID references

**API endpoints:**
- `prsm/interface/api/ipfs_api.py` — Replaced by content API speaking native storage

**Infrastructure:**
- `config/nginx/ipfs-proxy.conf` — Deleted
- IPFS service removed from all docker-compose files (main, quickstart, tutorial, onboarding)
- IPFS Kubernetes manifests removed

**Tests:**
- ~15 IPFS-specific test files deleted

### Consumer migration

All 33+ files that import `IPFSClient` or related IPFS modules migrate to the `ContentStore` public interface:

```python
from prsm.storage import ContentStore

store = ContentStore(data_dir="~/.prsm/storage")

# Store (handles sharding, encryption, distribution internally)
content_hash = await store.store(data, replication_factor=3)

# Retrieve (handles manifest lookup, shard retrieval, reassembly)
data = await store.retrieve(content_hash)

# Check existence
exists = await store.exists(content_hash)

# Delete (owner only)
await store.delete(content_hash)
```

**Consumer groups requiring import updates:**
- **Node layer**: `node.py`, `content_provider.py`, `content_uploader.py`, `storage_provider.py`
- **Compute layer**: `distillation/backends/`, `evolution/archive.py`, `spine/data_spine_proxy.py`, `collaboration/p2p/fallback_storage.py`, `federation/enhanced_p2p_network.py`, `ai_orchestration/model_manager.py`
- **Query/Response**: `advanced_query_engine.py`, `response_generator.py`
- **Reasoning**: `deep_reasoning_engine.py`, `meta_reasoning_orchestrator.py`
- **API layer**: `content_api.py`, `cdn_api.py`
- **User management**: `user_content_manager.py`

### Dependency changes in pyproject.toml

- No new dependencies required: SHA-256 via `hashlib` (stdlib), AES-256-GCM via `cryptography` (already a dependency), Shamir's Secret Sharing implementable with `cryptography` primitives
- Remove `aiofiles` from core dependencies if only used by IPFS client (verify during implementation)
- Remove IPFS-related entries from optional dependency groups if present

---

## 6. Testing Strategy

### Unit tests per component

**Blob store:**
- Store/retrieve round-trip integrity
- Content deduplication (same data -> same hash, single file on disk)
- Hash-prefix directory creation
- Deletion removes file
- Non-existent hash raises `ContentNotFoundError`
- Algorithm-agility prefix byte correctness (0x01 for SHA-256)

**Shard engine:**
- Content below threshold stored as single shard
- Content above threshold splits into correct number of chunks
- Reassembly produces original content byte-for-byte
- Tampered shard caught by hash verification
- Manifest JSON serialization round-trip
- Shard count matches expected `ceil(size / shard_size)`

**Key manager:**
- Shamir split produces correct number of shares (N)
- K shares reconstruct the original key
- K-1 shares fail to reconstruct
- AES-256-GCM encrypt/decrypt round-trip
- Key refresh produces new valid shares for the same key
- Algorithm ID preserved through split/reconstruct cycle

**Distribution manager:**
- Owner excluded from shard placement
- Shards placed across different ASN groups
- Replication factor honored (correct number of copies)
- Shard retrieval from multiple providers
- Fallback to alternate provider on single-provider failure
- Replication health monitor detects under-replicated content and triggers re-replication

### Integration tests

- **Full lifecycle**: Store content on node A -> shards distributed -> retrieve from node B -> verify integrity
- **Node departure**: Store content -> take down a shard-holding node -> replication manager re-distributes -> content still retrievable
- **Key threshold**: Store content -> take down key-share holders until just K remain -> still reconstructable -> take one more down -> key refresh triggered
- **Provenance inheritance**: Store content -> transfer ownership -> new owner retrieves successfully
- **Payment flow**: Store content with replication_factor=3 -> verify FTNS charged proportionally -> providers earn FTNS for serving retrieval

### Mock strategy

- Blob store tests use real file I/O against a temp directory (no mocking)
- Network-level tests use `MockLibp2pTransport` with JSON round-trip fidelity (same pattern from v1.3.0)
- Shamir operations use real crypto (no mocking the math)
- ~15 IPFS-specific test files deleted; integration tests rewritten against native storage interface

---

## Out of Scope

- WASM agent integration with storage retrieval (future: executor pre-fetches required shards)
- S3 or cloud storage backends (PRSM-native only)
- Content search/indexing (existing content economy handles discovery)
- Streaming retrieval for very large content (standard pull model sufficient for current scale)
