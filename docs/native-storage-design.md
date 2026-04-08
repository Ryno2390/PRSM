# Native Content-Addressed Storage — Design Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## Goal

Replace PRSM's direct IPFS/Kubo dependency with a native content-addressed storage system. IPFS was originally referenced as inspiration for how latent storage could be monetized — not as infrastructure to depend on. The native system brings storage in line with how compute already works: PRSM's own protocol for monetizing latent resources, with no external daemon required.

## Architecture

A unified `prsm/storage/` package replaces all six IPFS library files and their consumers. The package contains focused internal modules (blob store, shard engine, key manager, distribution manager) exposed through a single `ContentStore` interface. Sharding is a core security property — no single node holds complete content. Manifests are encrypted via Shamir's Secret Sharing so no single compromise reveals the reconstruction map. Shard placement enforces geographic diversity and owner exclusion. The entire system is algorithm-agile and post-quantum ready.

## Visibility Model

Storage operates in two explicit modes:

**Public artifacts** (Phase 1 — this spec):
- Plaintext shards on provider nodes
- Open retrieval — any node can request content
- Ledger tracks authorship, royalty flows, replication contracts, ownership history
- Integrity guaranteed by content hashing; availability by replication
- Fits PRSM's current content flows: provenance tracking, content serving, model distribution

**Private artifacts** (Phase 2 — deferred):
- Encrypted shards (ciphertext-at-rest)
- Capability-gated key release for retrieval
- Explicit metadata minimization
- Revocation and rekey semantics

Private artifact support is deferred until the network has: a real authorization protocol for key/share release, a ciphertext-at-rest design, and clear revocation and rekey semantics. The Phase 1 architecture is designed to accommodate Phase 2 without structural changes — the shard engine, distribution manager, and key manager interfaces remain the same; private mode adds an encryption layer before content enters the shard engine.

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
| Visibility | Public Phase 1, private deferred | Ship integrity/availability/economics first; add confidentiality with proper auth |

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

**Small object confidentiality note:** Content below the shard threshold is stored as a single shard, meaning every replica holder sees the full plaintext. This is acceptable for public artifacts (Phase 1). For Phase 2 (private artifacts), an "always-encrypt-before-sharding" mode will wrap content in AES-256-GCM before the shard engine processes it, ensuring even single-shard content is ciphertext at rest.

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
    visibility: str                    # "public" or "private" (Phase 2)
```

### Split flow

1. Hash the complete content -> `content_hash`
2. Split into chunks of `shard_size` bytes
3. Store each chunk via blob store -> ordered list of `shard_hashes`
4. Build manifest with all metadata
5. Serialize manifest to JSON, encrypt it (Section 3), distribute encrypted manifest
6. Publish ContentDescriptor to DHT (Section 4)

### Reassemble flow

1. Look up ContentDescriptor from DHT (Section 4) to find manifest replicas and key-share holders
2. Retrieve and decrypt manifest (Section 3 key reconstruction)
3. For each shard hash: pull from local blob store, or request via P2P if not local
4. Concatenate shards in manifest order
5. Hash the reassembled content, verify it matches `content_hash`
6. If mismatch -> integrity failure, reject

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

### Authorization for key-share release

Key-share holders must verify that a requester is authorized before releasing shares. The authorization model uses **signed capability tokens** with the ledger as the source of truth:

**Ledger (source of truth) tracks:**
- Current content owner
- Access policy (public/private, authorized requesters)
- Content epoch (incremented on ownership transfer or policy change)
- Transfer and revocation history

**Capability token (enforcement mechanism):**

```python
@dataclass
class RetrievalTicket:
    content_hash: ContentHash    # What content this grants access to
    requester_node_id: str       # Bound to a specific requester
    epoch: int                   # Must match current content epoch
    issued_at: float             # Unix timestamp
    expires_at: float            # Short-lived (default 5 minutes)
    nonce: str                   # Unique per-ticket, prevents replay
    issuer_signature: bytes      # Signed by content owner or governance
```

**Share release flow:**
1. Requester obtains a `RetrievalTicket` signed by the content owner (or governance for inherited content)
2. Requester presents ticket to key-share holder via direct P2P
3. Holder verifies: signature validity, `requester_node_id` matches sender, `epoch` matches current epoch (from last ledger checkpoint), ticket not expired, nonce not seen before
4. If valid: release share, log the request (holder-side audit trail)
5. If invalid: reject, log the attempt

**For public artifacts (Phase 1):** The content owner's node can auto-issue tickets to any requester, since public content has an open retrieval policy. The ticket mechanism still exists to provide audit logging and replay protection, and it becomes the enforcement gate for private artifacts in Phase 2.

**Ownership transfer flow:**
1. Ledger updates owner and access policy, bumps epoch
2. Old capabilities expire (short TTL) or are immediately invalid because epoch changed
3. New owner obtains signing authority for new tickets
4. Key-share holders reject stale-epoch requests automatically
5. No re-encryption or re-distribution of key shares needed

### Reconstruction flow

1. Requester obtains `RetrievalTicket` from content owner (or governance)
2. Looks up key-share holders from ContentDescriptor (Section 4)
3. Presents ticket to K holders via direct P2P
4. Each holder verifies ticket, releases share if valid
5. Requester reconstructs AES-256-GCM key via Shamir reconstruction
6. Decrypts manifest
7. Key is used in-memory only, never persisted in reconstructed form

### Provenance inheritance

When ownership transfers (node departure, explicit transfer), the manifest encryption key does NOT need re-encryption or re-distribution:

- Key shares remain on the same nodes
- The ledger updates the authorized owner, bumps the epoch
- New owner can issue fresh `RetrievalTicket`s
- Only the access control list changes, not the cryptographic material

If too many key-share holders have left the network (remaining holders drop to threshold K), a **key refresh** is triggered: an authorized party reconstructs the key, re-splits with new shares, and distributes to current healthy nodes.

### Algorithm agility

The `algorithm_id` field on KeyShare allows future migration to post-quantum key encapsulation (e.g., ML-KEM/Kyber) without invalidating existing shares. New content uses the new algorithm; old content retains its existing shares.

---

## 4. Distribution Manager

### ContentDescriptor (bootstrap record)

Every stored content item has a `ContentDescriptor` published to the libp2p DHT. This is the entry point for all retrieval operations — it tells a requester where to find the manifest, key shares, and current policy.

```python
@dataclass
class ContentDescriptor:
    content_hash: ContentHash            # Root content identifier
    manifest_holders: List[str]          # Node IDs holding encrypted manifest replicas
    key_share_holders: List[str]         # Node IDs holding Shamir key shares
    shard_map: Dict[str, List[str]]      # shard_hash -> [node_ids holding that shard]
    replication_policy: ReplicationPolicy # Owner-specified policy
    epoch: int                           # Bumped on ownership transfer or policy change
    owner_node_id: str                   # Current owner
    owner_signature: bytes               # Signs the descriptor for tamper detection
    created_at: float                    # Original creation timestamp
    updated_at: float                    # Last descriptor update
```

**Descriptor lifecycle:**
- Created after initial shard distribution completes
- Updated when replication health monitor re-distributes shards or key shares
- Updated on ownership transfer (new owner, bumped epoch, new signature)
- Signed by the current owner to prevent tampering
- Replicated in the DHT for availability (standard DHT replication)

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

### Degraded mode (sparse network)

The placement constraints (owner exclusion, ASN diversity, key-shard separation) can make uploads impossible when the network is small. The distribution manager applies a **constraint relaxation order** when the ideal placement cannot be achieved:

1. **Relax ASN diversity first** — Allow multiple shards on the same ASN group, but still across different nodes. Log a warning.
2. **Reduce N/K for key shares** — Scale threshold parameters down based on actual network size (e.g., 2-of-3 instead of 3-of-5 when only 4 non-shard nodes exist). Minimum floor: 2-of-3.
3. **Allow owner escrow copy** — As a last resort, the owner retains an encrypted copy of the manifest (not shards). This is a temporary fallback; the replication health monitor promotes to full distribution as nodes join.
4. **Reject before payment** — If even relaxed constraints cannot be met (e.g., fewer than 3 total nodes in the network), reject the upload and do not charge FTNS. Return a clear error with the minimum network size needed.

Each relaxation is logged and the ContentDescriptor records which constraints are degraded. The replication health monitor continuously attempts to upgrade to full constraint satisfaction as the network grows.

### Replication policy

```python
@dataclass
class ReplicationPolicy:
    replication_factor: int        # Owner-specified (paid via FTNS)
    min_asn_diversity: int         # Minimum distinct ASN groups across shards
    owner_excluded: bool           # Always True — owner never holds own shards
    key_shard_separation: bool     # Always True — key holders != shard holders
    degraded_constraints: List[str] # Which constraints are currently relaxed (empty = fully satisfied)
```

### Replication health monitor

- Periodic sweep (configurable, default 5 minutes) checks each content's shard distribution
- If a shard drops below `replication_factor` (node went offline), triggers re-replication to a new provider from a different ASN group
- If a key-share holder goes offline and remaining holders approach threshold K, triggers key refresh (Section 3)
- If constraints are degraded, attempts to upgrade to full satisfaction as new providers join
- Leverages existing `PeerInfo.last_seen` and reliability tracking from the libp2p discovery layer (shipped in v1.3.0)

### P2P shard retrieval (pull model)

1. Requester looks up ContentDescriptor from DHT by `content_hash`
2. Obtains `RetrievalTicket` from content owner (Section 3 authorization)
3. Presents ticket to key-share holders, reconstructs manifest decryption key
4. Decrypts manifest to get ordered shard list
5. For each shard hash, reads `shard_map` from ContentDescriptor to identify providers
6. Fires concurrent direct P2P requests (`transport.send_to_peer()`) to providers
7. Each provider reads shard from local blob store, returns via direct P2P response
8. Requester verifies each shard hash on receipt
9. After all shards received, reassembles and verifies complete content hash (Section 2)
10. On individual shard failure: retry from alternate provider (replication means multiple sources exist)

### Proof of custody (ongoing storage verification)

PRSM's existing storage proof system (`prsm/node/storage_proofs.py`) is carried forward and redefined in terms of `ContentHash`/`shard_hash` instead of IPFS CIDs. This is required for ongoing FTNS rewards — retrieval-time hash checks are not sufficient to prove continuous custody.

**How it works:**
- The `StorageChallenger` periodically issues challenges to shard holders: "prove you hold shard X by returning the hash of bytes at offsets [a, b, c]"
- Challenges reference `shard_hash: ContentHash` (replacing the old `cid: str` field)
- The `StorageProver` reads the shard from local blob store, computes the requested proof, returns it
- The `StorageVerifier` confirms the proof matches expected values
- Successful proofs maintain the provider's eligibility for ongoing storage rewards
- Failed proofs trigger replication to a replacement provider and reduce the failing provider's reliability score

**Changes from current storage_proofs.py:**
- `StorageChallenge.cid` -> `StorageChallenge.shard_hash: ContentHash`
- `StorageProofResponse.cid` -> `StorageProofResponse.shard_hash: ContentHash`
- Remove `ipfs_client` parameter from `StorageChallenger` and `StorageProver`
- Provers read from local `BlobStore` instead of IPFS API
- Challenge/proof messages use the direct P2P transport (already wired in v1.3.0)

### FTNS payment integration

- Storage providers earn FTNS for: holding shards (proven by periodic custody proofs), serving shard retrieval requests (per-serve), and holding key shares (ongoing)
- Content owners pay FTNS proportional to: `content_size * replication_factor * storage_duration`
- Integrates with existing content economy payment flows, replacing current IPFS-based payment triggers

---

## 5. Consumer Migration & Cleanup

### Scope of migration

The migration is deeper than replacing six IPFS client files and updating imports. CID and IPFS references are embedded in data models, permission enums, ledger schemas, and proof structures across the codebase. The migration introduces a neutral `content_id` abstraction (`ContentHash`) and updates all data model references.

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

### Data model migration

The following data models contain `cid`, `ipfs_hash`, or `ipfs_cid` fields that must be renamed to `content_id: str` (holding a serialized `ContentHash`):

**Core models (`prsm/core/models.py`):**
- `ipfs_cid` fields (lines ~481, ~597) -> `content_id`
- `content_cid` (line ~652) -> `content_id`
- `model_cid` (line ~675) -> `model_content_id`

**Information space models (`prsm/data/information_space/models.py`):**
- `ipfs_hash` fields (lines ~66, ~234) -> `content_id`
- IPFS hash references in evidence lists -> content ID references

**Economy models (`prsm/economy/tokenomics/models.py`):**
- `content_cid` column (line ~319) -> `content_id`
- `ipfs_hash` column (line ~787) -> `content_id`
- Database index `idx_provenance_content_creator` updated for new column name

**Auth models (`prsm/core/auth/models.py`):**
- `Permission.IPFS_UPLOAD` -> `Permission.STORAGE_UPLOAD`
- `Permission.IPFS_DOWNLOAD` -> `Permission.STORAGE_DOWNLOAD`
- `Permission.IPFS_PIN` -> `Permission.STORAGE_PIN`
- All role permission sets updated

**Swarm models (`prsm/compute/swarm/models.py`):**
- `shard_cid` -> `shard_content_id`
- `shard_cids` -> `shard_content_ids`

**Distillation models (`prsm/compute/distillation/models.py`):**
- `seed_data_cid` -> `seed_data_content_id`
- `ipfs_cid` -> `content_id`
- `original_cid` / `synthetic_cid` -> `original_content_id` / `synthetic_content_id`

**Agent models (`prsm/compute/agents/models.py`):**
- `required_cids` -> `required_content_ids`

**Agent forge models (`prsm/compute/nwtn/agent_forge/models.py`):**
- `target_shard_cids` -> `target_shard_content_ids`

**Teams models (`prsm/core/teams/models.py`):**
- `output_artifacts` comment "IPFS CIDs" -> "content IDs"

**Chronos models (`prsm/compute/chronos/models.py`):**
- `settlement_hash` comment "IPFS provenance hash" -> "provenance content hash"

**Content uploader (`prsm/node/content_uploader.py`):**
- `cid` field in `UploadedContent` -> `content_id`
- `parent_cids` -> `parent_content_ids`
- `manifest_cid` -> `manifest_content_id`
- Remove IPFS API URL configuration
- `SimilarityIndex` keyed by content_id instead of CID

**Storage proofs (`prsm/node/storage_proofs.py`):**
- `StorageChallenge.cid` -> `StorageChallenge.shard_hash`
- `StorageProofResponse.cid` -> `StorageProofResponse.shard_hash`
- Remove `ipfs_client` parameter
- Read from BlobStore instead of IPFS API

### Consumer migration

All 33+ files that import `IPFSClient` or related IPFS modules migrate to the `ContentStore` public interface:

```python
from prsm.storage import ContentStore

store = ContentStore(data_dir="~/.prsm/storage")

# Store (handles sharding, encryption, distribution internally)
content_hash = await store.store(data, replication_factor=3)

# Retrieve (handles descriptor lookup, key reconstruction, shard retrieval, reassembly)
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

- **Add**: `secret-sharing` (or equivalent audited Shamir's Secret Sharing library) — the `cryptography` package does not expose a secret-sharing module, so this requires either an audited third-party dependency or an in-repo implementation over GF(256). Decision to be made during implementation based on available library quality.
- **Remove**: `aiofiles` from core dependencies if only used by IPFS client (verify during implementation)
- **Remove**: IPFS-related entries from optional dependency groups if present

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

**Authorization:**
- Valid RetrievalTicket accepted by holder
- Expired ticket rejected
- Wrong epoch rejected
- Wrong requester_node_id rejected
- Replayed nonce rejected
- Invalid signature rejected
- Holder logs all attempts (accepted and rejected)

**Distribution manager:**
- Owner excluded from shard placement
- Shards placed across different ASN groups
- Replication factor honored (correct number of copies)
- Degraded mode: ASN relaxation when too few ASN groups
- Degraded mode: reduced N/K when too few non-shard nodes
- Degraded mode: owner escrow when minimal network
- Degraded mode: rejection when network too small
- Shard retrieval from multiple providers
- Fallback to alternate provider on single-provider failure
- Replication health monitor detects under-replicated content and triggers re-replication
- ContentDescriptor published to DHT on store, updated on re-distribution

**Proof of custody:**
- Challenge issued with shard_hash (not CID)
- Prover reads from BlobStore, returns valid proof
- Verifier confirms proof against expected values
- Failed proof triggers re-replication
- Proof messages use direct P2P transport

### Integration tests

- **Full lifecycle**: Store content on node A -> shards distributed -> retrieve from node B -> verify integrity
- **Node departure**: Store content -> take down a shard-holding node -> replication manager re-distributes -> content still retrievable
- **Key threshold**: Store content -> take down key-share holders until just K remain -> still reconstructable -> take one more down -> key refresh triggered
- **Provenance inheritance**: Store content -> transfer ownership -> new owner retrieves successfully (new epoch, fresh tickets)
- **Payment flow**: Store content with replication_factor=3 -> verify FTNS charged proportionally -> providers earn FTNS for serving retrieval
- **Degraded network**: Store content with only 4 nodes -> verify constraint relaxation -> add nodes -> verify upgrade to full constraints
- **Custody proof cycle**: Store content -> issue challenge -> receive proof -> verify -> reward provider

### Mock strategy

- Blob store tests use real file I/O against a temp directory (no mocking)
- Network-level tests use `MockLibp2pTransport` with JSON round-trip fidelity (same pattern from v1.3.0)
- Shamir operations use real crypto (no mocking the math)
- RetrievalTicket signing uses real Ed25519 keys (test-generated)
- ~15 IPFS-specific test files deleted; integration tests rewritten against native storage interface

---

## Out of Scope

- **Private artifacts** (Phase 2): Ciphertext-at-rest, capability-gated key release, revocation/rekey semantics. Deferred until authorization protocol is battle-tested on public artifacts.
- WASM agent integration with storage retrieval (future: executor pre-fetches required shards)
- S3 or cloud storage backends (PRSM-native only)
- Content search/indexing (existing content economy handles discovery)
- Streaming retrieval for very large content (standard pull model sufficient for current scale)
