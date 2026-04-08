# Native Storage Migration — Implementation Plan (Part 2 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove all IPFS/Kubo dependencies from PRSM, rewiring 68+ files to use the native `prsm/storage` module (built in Part 1).

**Architecture:** Delete IPFS library files, add a `get_content_store()` singleton (paralleling the old `get_ipfs_client()`), rename CID/IPFS fields across data models, update all consumer imports, clean up infrastructure configs.

**Tech Stack:** Python 3.10+, existing `prsm/storage` module (ContentStore, ContentHash, BlobStore).

**Spec:** `docs/native-storage-design.md` Section 5 (Consumer Migration & Cleanup)

---

## Task 1: Add ContentStore singleton and delete IPFS library files

**Create singleton pattern** in `prsm/storage/__init__.py`:
```python
_content_store: Optional[ContentStore] = None

def get_content_store() -> Optional[ContentStore]:
    return _content_store

def init_content_store(data_dir: str = "~/.prsm/storage", node_id: str = "") -> ContentStore:
    global _content_store
    if _content_store is None:
        _content_store = ContentStore(data_dir=os.path.expanduser(data_dir), node_id=node_id)
    return _content_store

def close_content_store() -> None:
    global _content_store
    _content_store = None
```

**Delete these files:**
- `prsm/core/ipfs_client.py`
- `prsm/core/ipfs_sharding.py`
- `prsm/core/infrastructure/ipfs_cdn_bridge.py`
- `prsm/data/ipfs/ipfs_client.py`
- `prsm/data/ipfs/content_addressing.py`
- `prsm/data/ipfs/content_verification.py`
- `prsm/data/ipfs/__init__.py`
- `prsm/data/data_layer/enhanced_ipfs.py`
- `prsm/data/data_layer/__init__.py` (if only IPFS re-exports)
- `prsm/interface/api/ipfs_api.py`
- `prsm/interface/infrastructure/ipfs_cdn_bridge.py` (if exists)
- `config/nginx/ipfs-proxy.conf`

**Update `prsm/data/ipfs/` directory** — if empty after deletions, remove the directory.

---

## Task 2: Data model field renames

Rename all CID/IPFS fields to neutral `content_id` terminology:

**`prsm/core/models.py`:**
- `ipfs_cid` -> `content_id` (all occurrences)
- `content_cid` -> `content_id`
- `model_cid` -> `model_content_id`
- Remove IPFS-related comments

**`prsm/core/auth/models.py`:**
- `Permission.IPFS_UPLOAD` -> `Permission.STORAGE_UPLOAD`
- `Permission.IPFS_DOWNLOAD` -> `Permission.STORAGE_DOWNLOAD`
- `Permission.IPFS_PIN` -> `Permission.STORAGE_PIN`
- Update all role permission sets referencing these

**`prsm/data/information_space/models.py`:**
- `ipfs_hash` -> `content_id`
- Update evidence list comments

**`prsm/economy/tokenomics/models.py`:**
- `content_cid` column -> `content_id`
- `ipfs_hash` column -> `content_id`
- Update index names

**`prsm/compute/swarm/models.py`:**
- `shard_cid` -> `shard_content_id`
- `shard_cids` -> `shard_content_ids`

**`prsm/compute/distillation/models.py`:**
- `seed_data_cid` -> `seed_data_content_id`
- `ipfs_cid` -> `content_id`
- `original_cid` / `synthetic_cid` -> `original_content_id` / `synthetic_content_id`
- `custom_training_data` IPFS CID comment -> content ID

**`prsm/compute/agents/models.py`:**
- `required_cids` -> `required_content_ids`

**`prsm/compute/nwtn/agent_forge/models.py`:**
- `target_shard_cids` -> `target_shard_content_ids`

**`prsm/core/teams/models.py`:**
- Update "IPFS CIDs" comment -> "content IDs"

**`prsm/compute/chronos/models.py`:**
- Update "IPFS provenance hash" comment -> "provenance content hash"

**`prsm/node/content_uploader.py`:**
- `cid` field -> `content_id`
- `parent_cids` -> `parent_content_ids`
- `manifest_cid` -> `manifest_content_id`
- All SimilarityIndex keys: `cid` -> `content_id`

**`prsm/node/storage_proofs.py`:**
- `StorageChallenge.cid` -> `StorageChallenge.shard_hash`
- `StorageProofResponse.cid` -> `StorageProofResponse.shard_hash`
- Remove `ipfs_client` constructor params
- Remove IPFS API URL references

**`prsm/node/content_economy.py`:**
- `cid` params/fields -> `content_id`

---

## Task 3: Consumer import migration — API layer

Update all API files to use ContentStore instead of IPFSClient:

- `prsm/interface/api/content_api.py` — replace `get_ipfs_client()` with `get_content_store()`
- `prsm/interface/api/contributor_api.py` — same pattern
- `prsm/interface/api/core_endpoints.py` — same pattern
- `prsm/interface/api/ui_api.py` — same pattern
- `prsm/interface/api/cdn_api.py` — remove IPFS CDN bridge, simplify
- `prsm/interface/api/router_registry.py` — remove ipfs_router registration
- `prsm/interface/api/lifecycle/startup.py` — replace `init_ipfs()` with `init_content_store()`
- `prsm/interface/api/lifecycle/shutdown.py` — replace `close_ipfs()` with `close_content_store()`
- `prsm/interface/api/main_standardized.py` — replace IPFS init/close

**Pattern for each file:**
```python
# Old:
from prsm.core.ipfs_client import get_ipfs_client
ipfs = get_ipfs_client()
result = await ipfs.upload_content(data)
cid = result.cid

# New:
from prsm.storage import get_content_store
store = get_content_store()
content_hash = await store.store_local(data)
content_id = str(content_hash)
```

---

## Task 4: Consumer import migration — Compute layer

Update all compute files:

- `prsm/compute/distillation/orchestrator.py`
- `prsm/compute/distillation/backends/tensorflow_backend.py`
- `prsm/compute/evolution/archive.py`
- `prsm/compute/chronos/api.py`
- `prsm/compute/chronos/clearing_engine.py`
- `prsm/compute/federation/p2p_network.py`
- `prsm/compute/federation/enhanced_p2p_network.py`
- `prsm/compute/federation/distributed_model_registry.py`
- `prsm/compute/ai_orchestration/model_manager.py`
- `prsm/compute/data/synthetic_orchestrator.py`
- `prsm/compute/spine/data_spine_proxy.py`
- `prsm/compute/teachers/real_teacher_implementation.py`
- `prsm/compute/nwtn/orchestrator.py`
- `prsm/compute/nwtn/external_storage_config.py`
- `prsm/compute/collaboration/p2p/fallback_storage.py`

Same pattern: replace `get_ipfs_client()`/`create_ipfs_client()`/`IPFSClient` with `get_content_store()`.

---

## Task 5: Consumer import migration — Node layer and core

- `prsm/node/node.py` — remove IPFS client init, use ContentStore
- `prsm/node/storage_provider.py` — remove `ipfs_api_url`, use BlobStore for pin/verify
- `prsm/node/content_provider.py` — update content serving
- `prsm/node/content_uploader.py` — replace IPFS sharding with native ShardEngine
- `prsm/core/integrations/core/integration_manager.py` — update imports
- `prsm/core/integrations/core/provenance_engine.py` — update imports
- `prsm/core/config.py` — remove IPFS settings, add storage settings

---

## Task 6: Infrastructure cleanup

- Remove IPFS service from `docker-compose.yml` and all variants in `docker/`
- Remove IPFS references from `config/nginx/nginx.conf`
- Remove IPFS manifests from `deploy/kubernetes/`
- Remove IPFS targets from `Makefile`
- Remove `ipfshttpclient` mentions (not in pyproject.toml dependencies)
- Update CLI files (`prsm/cli.py`, `prsm/dev_cli.py`)

---

## Task 7: Test cleanup

- Delete ~13 IPFS-specific test files
- Update integration tests that reference IPFS
- Verify all remaining tests pass
- Run full test suite
