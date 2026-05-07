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

### PR 2: Node-layer surgical rewrites (the load-bearing one)

- Rewire `content_uploader.py` `_ipfs_add` / `_ipfs_cat` → `BitTorrentProvider.seed_content` / `BitTorrentRequester.request_content`. Drop `_IPFSClientWrapper`.
- Rewire `storage_provider.py` IPFS-pin → BitTorrent-seed model.
- Rewire `content_provider.py` content-serve to BitTorrent-source.
- Update `node.py` init path: `init_content_store(...)` instead of IPFS client.
- This is the load-bearing PR: it's where `prsm_upload_dataset` actually starts distributing content (closing the gap-list "STUB" finding from `docs/2026-05-07-canonical-workflow-gap-list.md`).
- **Land before PR 3 (API layer)** because the API layer's HTTP endpoints call into these node-layer modules.

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
- PR 2: ~6-8 hr (the load-bearing rewrite — design the BitTorrent adapter first)
- PR 3: ~3-4 hr (mechanical bulk find-replace + verify)
- PR 4: ~2 hr (residuals + test cleanup)
- PR 5: ~1 hr (verification + tag)

**Total: ~1.5-2 working days** to complete the migration. (Original plan estimated longer because it was sized when Tasks 1+2+6 hadn't yet shipped.)
