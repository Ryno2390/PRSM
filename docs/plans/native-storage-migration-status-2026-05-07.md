# Native Storage Migration — Execution Status Reconciliation (2026-05-07)

**Re:** `docs/plans/native-storage-migration.md` (the canonical plan, Part 2 of 2)
**Trigger:** Founder direction 2026-05-07 — "shouldn't be ANY references to IPFS or reliance on it in any way"
**Audit method:** Code-survey at tip `origin/main @ 131bdf41`. Cross-references the plan's task list against current file state.

## Top-line: the plan is partially executed

Of the 7 tasks in the canonical plan, 3 appear DONE, 3 are PARTIAL, 1 is unverified.
**Remaining work concentrates in ~14 files** (the canonical plan listed 68+ — most have either been cleaned or deleted via the v1.6.0 sprint).

| Task | Plan file count | Current status | Files still needing work |
|------|----------------|----------------|--------------------------|
| 1. Library deletions + singleton | 12 files + 1 singleton | ✅ **DONE** | All 12 deletion targets gone; `get_content_store()` live at `prsm/storage/__init__.py:53` |
| 2. Data-model field renames | 11 model files | ✅ **DONE** (sampled) | Spot-checks: `core/models.py:0` `ipfs_cid` refs, `auth/models.py:0` `IPFS_*` perms, `information_space/models.py` clean. May have residuals in less-touched models — see verification suite below |
| 3. API-layer migration | 9 files | ⚠️ **PARTIAL (5 remain)** | content_api, contributor_api, core_endpoints, ui_api, lifecycle/startup, lifecycle/shutdown still have IPFS refs. cdn_api deleted; router_registry + main_standardized clean |
| 4. Compute-layer migration | 15 files | ⚠️ **PARTIAL (3 remain)** | Most deleted via v1.6.0. Remaining: `federation/p2p_network.py` (1 trivial ref), `spine/data_spine_proxy.py` (84 refs), `collaboration/p2p/fallback_storage.py` (121 refs) — last two are DELETE candidates |
| 5. Node-layer migration | 6+ files | ❌ **NOT YET** | All 6 still IPFS-laden: `node.py` (7), `storage_provider.py` (32), `content_provider.py` (27), `content_uploader.py` (41), `provenance_engine.py` (4), `config.py` (1) |
| 6. Infrastructure cleanup | 4-5 surfaces | ✅ **DONE** | docker-compose.yml, Makefile, config/nginx, deploy/kubernetes all show 0 IPFS refs |
| 7. Test cleanup | ~13 test files | ❓ **UNVERIFIED** | Survey deferred — runs better as a single sweep after Tasks 3-5 land |

## Total remaining ref count

Verified residual IPFS refs in non-legacy source: **~270** (from the original 513 surveyed).

| File | Refs | Strategy |
|------|------|----------|
| `prsm/compute/collaboration/p2p/fallback_storage.py` | 121 | **Delete.** Header: "IPFS Fallback Storage System for PRSM P2P Collaboration" — the entire file's purpose is IPFS-as-fallback. The native ContentStore + BitTorrent layer subsume both the primary and fallback paths. Verify no callers first. |
| `prsm/compute/spine/data_spine_proxy.py` | 84 | **Delete.** Header: "Seamless HTTPS/IPFS interoperability with intelligent content management" — entire purpose is the IPFS bridge. Verify no callers first. |
| `prsm/node/content_uploader.py` | 41 | **Surgical rewrite.** `_ipfs_add` (line 1599) + `_ipfs_cat` (line 1618) helpers hit `http://127.0.0.1:5001/api/v0/add` and `/api/v0/cat`. Replace with `BitTorrentProvider.seed_content()` + `BitTorrentRequester.request_content()`. Drop `_IPFSClientWrapper` class at line 1898. Rename `cid` field → `content_id` per Task 2 pattern. ~6 callsites of `_ipfs_add` to redirect. |
| `prsm/node/storage_provider.py` | 32 | **Surgical rewrite.** Header: "IPFS pin space contribution... Auto-detects local IPFS daemon, pledges configurable storage, accepts pin requests." Replace IPFS pin model with `BitTorrentProvider.seed_content` (seeding == pinning in BT). Drop `ipfs_api_url` constructor param. |
| `prsm/node/content_provider.py` | 27 | **Surgical rewrite.** Likely similar pattern to `content_uploader.py` — survey + redirect to BitTorrent layer. |
| `prsm/interface/api/content_api.py` + 5 sibling API files | ~25-30 | **Mechanical replace.** Pattern from plan §3: `from prsm.core.ipfs_client import get_ipfs_client` → `from prsm.storage import get_content_store`; `cid = result.cid` → `content_id = str(content_hash)`. |
| `prsm/node/node.py` | 7 | **Mechanical replace.** Init/close calls; remove IPFS client init in favor of `init_content_store()`. |
| `prsm/core/integrations/core/provenance_engine.py` | 4 | **Mechanical replace.** Imports + identifier renames. |
| `prsm/compute/federation/p2p_network.py` | 1 | **Mechanical replace.** Single ref — likely a docstring or import. |
| `prsm/core/config.py` | 1 | **Mechanical replace.** Likely a setting name. |

**Two file deletions account for 205 / 270 = 76% of remaining refs.** Net-of-deletions, ~65 refs need surgical attention across ~10 files.

## Recommended PR sequence

Each PR is small, ships independently, and can land in any order with two exceptions noted.

### PR 1: Delete the two IPFS-purpose files (~120 LoC remove)

- Delete `prsm/compute/collaboration/p2p/fallback_storage.py`
- Delete `prsm/compute/spine/data_spine_proxy.py`
- Pre-delete check: `grep -rn "from prsm.compute.collaboration.p2p.fallback_storage\|import fallback_storage\|from prsm.compute.spine.data_spine_proxy\|import data_spine_proxy" prsm/` — if any caller, redirect or delete the caller too.
- Drops 205 refs in one PR. **Largest leverage per LoC of any task.**

### PR 2: Node-layer surgical rewrites — ContentPublisher + Tier-aware routing (the load-bearing one)

**Architectural decision (2026-05-07):** `ContentStore` and `BitTorrentProvider` are **complementary, not alternatives.** The canonical 3-tier scope memory says "Three content tiers: A public, B encrypted-before-sharding, C erasure-coded + Shamir-split keys" — they map cleanly:

| Tier | ContentStore role | BitTorrent role |
|------|-------------------|-----------------|
| **A (public)** | Skip — public bytes need no encryption/sharding | Seed raw bytes directly via `BitTorrentProvider.seed_content(path)` |
| **B (encrypted)** | Encrypt + shard via `ContentStore.store_local(data)` | Seed the resulting encrypted blob directory |
| **C (erasure + Shamir)** | Full pipeline via `ContentStore.store_local(data)` | Seed the resulting shard directory |

**Concrete surface — new module `prsm/node/content_publisher.py`:**

```python
@dataclass
class PublishedContent:
    content_hash: Optional[ContentHash]   # from ContentStore (None for Tier A)
    torrent_infohash: str                 # from BitTorrentProvider
    manifest: TorrentManifest             # for discovery/network announce

class ContentPublisher:
    """
    Unified upload entry point. Replaces the ~6 _ipfs_add callsites in
    content_uploader.py with one well-typed call.
    """
    def __init__(
        self,
        content_store: ContentStore,
        bt_provider: BitTorrentProvider,
        staging_dir: Path,
    ): ...

    async def publish(
        self,
        data: bytes,
        *,
        tier: ContentTier,        # A | B | C
        provenance_id: str,
    ) -> PublishedContent:
        if tier == ContentTier.A:
            staged = await self._stage_raw(data, provenance_id)
            manifest = await self.bt_provider.seed_content(
                staged, provenance_id=provenance_id
            )
            return PublishedContent(
                content_hash=None,
                torrent_infohash=manifest.infohash,
                manifest=manifest,
            )
        else:
            content_hash = await self.content_store.store_local(data)
            blob_dir = self.content_store.blob_store.directory_for(content_hash)
            manifest = await self.bt_provider.seed_content(
                blob_dir, provenance_id=provenance_id
            )
            return PublishedContent(
                content_hash=content_hash,
                torrent_infohash=manifest.infohash,
                manifest=manifest,
            )
```

**Companion fetch surface (used by `content_provider.py` rewrite):**

```python
class ContentRetriever:
    """Mirror of ContentPublisher for the read path."""
    async def fetch(
        self,
        *,
        torrent_infohash: str,
        content_hash: Optional[ContentHash] = None,  # required for Tier B/C
        key_shares: Optional[List[KeyShare]] = None, # required for Tier B/C
    ) -> bytes:
        # 1. BT request_content → blob bytes
        # 2. If content_hash given: ContentStore.retrieve_local(content_hash)
        #    using the supplied key_shares to decrypt
        ...
```

**Tasks under this PR:**

1. **Add `BlobStore.directory_for(content_hash) -> Path`** (~10 LoC + test). Currently `BlobStore` is internal; this small public method lets `ContentPublisher` resolve a content_hash to its shard directory so BitTorrent can seed it.
2. **Build `ContentPublisher` + `ContentRetriever`** in `prsm/node/content_publisher.py` (~150 source LoC + ~200 test LoC). Unit tests with mocked ContentStore + BitTorrentProvider for each tier.
3. **Rewire `content_uploader.py`:** delete `_ipfs_add` (line 1599), `_ipfs_cat` (line 1618), and `_IPFSClientWrapper` (line 1898). Replace ~6 `_ipfs_add` callsites with `ContentPublisher.publish(data, tier=..., provenance_id=...)`. Drop `ipfs_api_url` constructor param.
4. **Rewire `content_provider.py`:** route reads through `ContentRetriever.fetch`. Likely ~5 callsites.
5. **Rewire `storage_provider.py`:** the existing "IPFS-pin contribution" model collapses naturally onto `BitTorrentProvider.seed_content` (in BT, "seeding" *is* "pinning"). Drop `ipfs_api_url`. Storage providers register a path → seed. Storage proofs (challenge-response) already work against the `BitTorrentProvider.bt_client` surface.
6. **Update `node.py` init:** add `init_content_store(...)` + `init_bt_provider(...)` (likely already exists), wire both into `self.content_publisher = ContentPublisher(...)`. Drop IPFS client init.
7. **Tier defaulting in `prsm_upload_dataset`:** the MCP tool currently doesn't take a tier param. Add it (default `ContentTier.A`); plumb through `/content/upload/shard` → `ContentPublisher.publish`.

**This closes the gap-list "STUB" finding** from `docs/2026-05-07-canonical-workflow-gap-list.md` — once PR 2 lands, `/content/upload/shard` actually distributes content (the placeholder CIDs at `prsm/node/api.py:1392` go away, replaced with real torrent infohashes plus optional content_hashes).

**Why this is the right shape (not just "rewire `_ipfs_add` to BT"):**
1. **Matches the canonical 3-tier architecture** without forcing the wrong abstraction onto either layer.
2. **Single call site** for upload — one well-typed entry point replaces ~6 scattered IPFS HTTP calls.
3. **`replication_factor` semantics become real** — Tier A relies on BT's swarm dynamics; Tier B/C `replication_factor` becomes the BT seed count target.
4. **Preserves the on-chain provenance flow** — both `content_hash` (Tier B/C) and `infohash` (all tiers) map cleanly to `RoyaltyDistributor.distributeRoyalty(contentHash, ...)`. The mainnet contract speaks `bytes32` abstractly; either ID type fits.
5. **Future-proofs the read path** — a separate `ContentRetriever` mirror lets Tier B/C decryption stay symmetric with the publish path; no asymmetric "read uses ContentStore directly, write uses BT" smell.

**Land before PR 3 (API layer)** because the API layer's HTTP endpoints call into these node-layer modules.

### PR 3: API-layer mechanical replacements

- Apply plan §3 pattern to `content_api.py`, `contributor_api.py`, `core_endpoints.py`, `ui_api.py`, `lifecycle/startup.py`, `lifecycle/shutdown.py`.
- Bulk find-replace `get_ipfs_client()` → `get_content_store()` and adjust types.
- Land after PR 2 so the consumers see real content-distribution.

### PR 4: Mechanical residuals + Task 7 test cleanup

- `provenance_engine.py` (4 refs), `federation/p2p_network.py` (1), `core/config.py` (1).
- Survey + delete IPFS-specific test files per plan §7.
- Run full test suite.

### PR 5: Final verification gate

- Repo-wide grep: `grep -rin "ipfs\|IPFS" --include="*.py" prsm/ | wc -l` should return **0** (or only documented historical references in changelogs).
- Update memory entry (`project_v1_6_0_sprint_complete.md` → "post-v1.6.0 native-storage migration COMPLETE").
- Tag: `native-storage-migration-complete-2026-05-XX`.

## Risk + dependencies

- **PR 1 has the biggest delete; verify no callers first.** Two files = 205 refs is a big tug; if any consumer survived, the import will break at startup. The pre-delete grep is non-negotiable.
- **PR 2 must verify BitTorrent layer parity.** `_ipfs_add` returns a CID immediately; `BitTorrentProvider.seed_content` returns an infohash + manifest. Caller signatures need unification — likely a small adapter on top of BitTorrent (`async def publish(content) → ContentHash`) that mirrors the old IPFS API shape so callers don't all churn.
- **Task 2 may have residuals.** I sampled 3 model files cleanly but didn't enumerate all 11. Add a verification step: `grep -rn "ipfs_cid\|ipfs_hash\|content_cid\|model_cid" --include="*.py" prsm/`.
- **No mainnet impact.** All this work is pre-launch internal-system-architecture; the mainnet contracts don't reference IPFS at all (verified — the Solidity contracts use abstract `bytes32 contentHash` and never speak about underlying storage).

## What this unblocks

1. **Closes the gap-list "STUB" finding** for `prsm_upload_dataset` once PR 2 lands — `/content/upload/shard` will actually distribute content via BitTorrent.
2. **Removes 270+ misleading-to-auditor IPFS references** — important pre-condition for the L4 external-auditor pass per PRSM-POL-2 §4.
3. **Half of the canonical 8-step workflow's content-distribution layer.** The other half is the `prsm_upload_dataset` registration→real-publication wiring (PR 2).
4. **Does NOT close the orchestration layer (`/compute/forge` 503).** The QueryOrchestrator rebuild is a separate workstream — both are needed for the canonical loop, but they can land independently.

## Estimated cost

- PR 1: ~30 min (verify no callers, delete, run tests)
- PR 2: **~10-12 hr** (revised up from 6-8 hr after the 2026-05-07 architectural call — `ContentPublisher` + `ContentRetriever` + `BlobStore.directory_for` + 4 file rewrites + tier param plumbing through `prsm_upload_dataset`)
- PR 3: ~3-4 hr (mechanical bulk find-replace + verify)
- PR 4: ~2 hr (residuals + test cleanup)
- PR 5: ~1 hr (verification + tag)

**Total: ~2-2.5 working days** to complete the migration. PR 2 owns ~60% of the work; the rest is mechanical.

---

## Migration closure — 2026-05-07 (later that day)

All five planned PRs landed plus the deferred PR 2b. The repo-wide
`grep -rin "ipfs" prsm/` returns zero matches (only the historical
alembic schema files retain the old column name, which is intentional —
they describe schema state at their migration version, and the
forward-rename lives in alembic 016).

| PR | Commit | Net LoC | Notes |
|----|--------|---------|-------|
| PR 1 | `bef7ee16` | ~−205 | Deleted `fallback_storage.py` + `data_spine_proxy.py` + `public_source_porter.py` + spine module |
| PR 2a | `c0a32fd8` | (created `content_publisher.py`, +159 / −140 in `content_uploader.py`) | ContentPublisher Tier A + content_uploader rewire |
| PR 2c | `cad6cd96` | +44 / −1 | node.py BT layer composition + ContentPublisher attachment |
| PR 3 | `616daedd` | +54 / −133 | API layer comments, identifier renames, `_init_ipfs` → `_init_content_store`, dropped IPFS-daemon prerequisite check from onboarding |
| PR 4 | `2841cf2b` | +104 / −137 | `storage_provider` + `content_provider` rewires; `ipfs_available` → `storage_available`; `_ipfs_cat` → `_fetch_local`; `_fetch_from_gateway` → `_fetch_from_url` (now uses real aiohttp session); `prsm://` URL scheme; deleted dead `_ensure_ipfs_available` and `_ipfs_daemon_proc` |
| PR 5 | `26e83b64` | +216 / −251 | 43-file repo-wide scrub via Agent assist; alembic migration `016_rename_ipfs_cid_to_content_cid.py`; SQLAlchemy column renames in `FTNSTransactionModel` / `TeacherModelModel` / `ModelRegistryModel`; `bittorrent_manifest.ipfs_cid` field deleted entirely; UI mockups + cli_modules + economy/tokenomics + core/integrations cleaned |
| dead-code | `fb1314ba` | −895 | Deleted orphan `semantic_embedding_engine.py` (broken imports, zero callers) |
| **PR 2b** | **`ac0ff969`** | **+619 / −85** | **Deferred ContentStore↔BitTorrent integration for Tier B (encrypted) + Tier C (encrypted + erasure). Multi-file torrent layout: `manifest.bin` + `keyshares.json` + `shard-NNNN.bin`. Added `StorageArtifacts` + `store_local_with_artifacts` / `retrieve_with_artifacts` to ContentStore. ContentPublisher / ContentRetriever auto-route by tier.** |

### Architectural call that held up

The 2026-05-07 architectural decision — *ContentStore and BitTorrent
are complementary, not alternatives* — proved out cleanly. Tier A skips
ContentStore entirely (raw bytes → BT). Tier B/C runs the bytes
through ContentStore's encrypt-then-shard pipeline and packages every
artefact (encrypted manifest, Shamir shares, shards) as a multi-file
torrent. The retriever auto-detects layout and routes through
`ContentStore.retrieve_with_artifacts` for the encrypted case.

### What this unblocks (revised)

1. **Closed:** the gap-list "STUB" finding for `prsm_upload_dataset`.
   Both Tier A and Tier B/C content distribution now works end-to-end.
2. **Closed:** zero-IPFS gate for the L4 external-auditor pass — no
   misleading IPFS references left in `prsm/`.
3. **Closed:** the half of the canonical 8-step workflow's content
   layer that this migration owns. (The other half — the orchestration
   `/compute/forge` 503 — is a separate workstream still open.)

### Known limitations carried forward

- **Shamir key-share colocation in Tier B/C torrents.** PR 2b ships
  with all key shares bundled in `keyshares.json` alongside the
  ciphertext. Distribution-of-shares is the `KeyDistribution.sol`
  follow-on (Phase 7-storage Task 6, already mainnet-deployed) plus a
  Python wiring task. Until that ships, treat Tier B/C as "encrypted at
  rest in a torrent" but not "secret from a node operator who downloads
  the torrent." Documented at the top of `prsm/node/content_publisher.py`.
- **Alembic migration applies on first upgrade.** Operators with
  existing populated DBs need `alembic upgrade head` to apply
  `016_rename_ipfs_cid_to_content_cid`. New deploys are unaffected.

### Final test posture

- `tests/unit/test_content_publisher.py`: 13 tests (4 new for Tier
  B/C). All green.
- Smoke set across `test_content_publisher` +
  `test_cross_node_content` + `test_content_store_integration` +
  `test_content_uploader_onchain_register`: 78 / 78 green.
- Smoke imports verified across `prsm.node.node`, `prsm.storage`,
  `prsm.cli`, `prsm.economy.tokenomics.contributor_manager`,
  `prsm.compute.performance.load_testing`, `prsm.core.bittorrent_manifest`,
  `prsm.data.vector_store`.
