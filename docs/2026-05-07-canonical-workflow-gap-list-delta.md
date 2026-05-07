# Canonical 8-Step Workflow Gap-List — Delta vs 2026-05-07 baseline

**Date:** 2026-05-07 (later, post Item-4 productionization sprint)
**Baseline:** `docs/2026-05-07-canonical-workflow-gap-list.md`
(generated against tip `origin/main @ 11ce30a3`)
**Method:** Re-audit of every load-bearing claim in the baseline
gap-list against current tip after 8 commits of native-storage
migration + PRSM-PROV-1 Item 4 productionization landed.

## TL;DR

The baseline gap-list named **three load-bearing layers blocking
"fully operational" status**. Audit at current tip:

| Layer | Baseline status | Current status |
|---|---|---|
| 1. Orchestration (`/compute/forge` / Agent Forge) | ❌ BROKEN | ❌ STILL BROKEN |
| 2. `prsm_upload_dataset` registration-only stub | ❌ STUB | ❌ STILL STUB |
| 3. IPFS-removal incomplete (513 stale refs) | ❌ INCOMPLETE | ✅ **CLOSED** — zero IPFS refs in `prsm/` |

One new finding (same shape as `dependencies.py:102` from earlier
today): `websocket_auth.py:441` had a wrong-submodule-path import
that silent-failed admin permission checks under WebSocket.
**FIXED** in this commit.

## What changed since baseline

### Native-storage migration — fully closed

Baseline claim: "**513 stale IPFS references** in non-legacy source
… `prsm/node/content_uploader.py` (41 refs) actively talks to an
external IPFS daemon at `http://127.0.0.1:5001` and uses `_ipfs_add`
as the real upload path at lines 942 + 1230." (line 244-249)

Current state:
- `grep -rln "ipfs\|IPFS" prsm/ | wc -l` = **0**
- No `_ipfs_add` callsite anywhere in `prsm/`
- No imports of `prsm.compute.spine.ipfs_client`,
  `prsm.data.ipfs.content_addressing`, or
  `prsm.data.ipfs.content_verification` (the three deleted-module
  imports baseline flagged as time-bombs)

This was closed by the native-storage migration sprint shipping
across PR 1 → PR 5 → PR 2b → PR 2c, capped at commit `8d071cc1`
today. Memory: `project_native_storage_migration_complete.md`.

### PRSM-PROV-1 Item 4 — fully closed (out of band of this gap-list)

The baseline didn't enumerate Item 4 because it focused on the
8-step workflow rather than the dedup substrate. Worth flagging
here that the cross-node binary fingerprint dedup lane went LIVE
end-to-end today across 6 commits (T4.9.next → T4.9.next5). Memory:
`project_prov_1_item_4_complete_2026_05_07.md`.

### Single new finding

**`prsm/interface/api/websocket_auth.py:441`** — wrong submodule
path import, same anti-pattern as the `dependencies.py:102` we fixed
earlier today:

```python
# Before
from prsm.core.auth.enhanced_authorization import get_enhanced_auth_manager
# After
from prsm.core.security.enhanced_authorization import get_enhanced_auth_manager
```

The actual module lives at `prsm.core.security.enhanced_authorization`
(per `credential_api.py:19` + `core/security/middleware.py:22`).
Wrapped in try/except, so silent-fails to "no admin permission"
which makes WebSocket admin-only conversation access dead. **FIXED**
in this commit + 2 regression tests pinning the corrected import +
a defense-in-depth source scan that would catch a re-regression.

## What remains (unchanged from baseline)

These three layers are still blockers for "fully operational" status:

### A. `agent_forge = None` — orchestration layer dead
- Confirmed at `prsm/node/node.py:1277` (line moved from baseline's
  `1196` due to the late-bind code we added in T4.9.next4).
- `/compute/forge` returns 503.
- `prsm_analyze`, `prsm_dispatch_agent` MCP tools both depend on
  this path → both still return 503.
- Recommendation from baseline still stands: build a non-AGI
  `QueryOrchestrator`. Estimated 1-2 weeks focused work. Largest
  outstanding gap.

### B. `prsm_upload_dataset` registration-only stub
- Confirmed at `prsm/node/api.py:1392`:
  ```python
  cid=f"Qm{dataset_id}-{i:04d}",  # Placeholder until content upload
  ```
- The native-storage migration rewired `ContentUploader` (the Python
  class) to use `ContentPublisher` + the proprietary BitTorrent
  layer. But the migration **did not rewire `/content/upload/shard`
  in `api.py`** — that endpoint still builds `SemanticShard` objects
  with placeholder CIDs and registers them in `data_listing_manager`
  without any actual content upload.
- Baseline framing was correct: this is the second load-bearing gap
  (the creator-economy pillar — "creators earn 6.4% royalty as
  queries hit their content" — cannot fire without real CIDs).
- Comment was even updated from "Placeholder until IPFS upload" to
  "Placeholder until content upload" as part of the IPFS sweep, but
  the underlying functionality wasn't restored.

### C. `prsm_agent_status` — backing endpoint nonexistent
- Baseline claimed `/compute/status/{job_id}` doesn't exist in
  `prsm/node/api.py`. Re-confirmed: only `/api-info`, `/status`,
  `/rings/status`, `/staking/status`, `/bridge/status` exist on the
  status surface.
- Tool returns 404 in production.

## Recommended next-step ordering (revised)

1. **websocket_auth.py:441 fix** ✅ DONE in this commit.
2. **B1 from baseline — hide `prsm_analyze` / `prsm_dispatch_agent`
   / `prsm_agent_status` from MCP tool list** (~10-30 LoC,
   defensive). Smallest. Should be next.
3. **Wire `/content/upload/shard` to ContentPublisher** —
   surgical fix to remove the placeholder-CID stub. Estimated
   ~100-200 LoC + tests. Can ship before the QueryOrchestrator —
   creators publish, `prsm_search_shards` already works (real per
   baseline), so partial loop closes without orchestrator.
4. **A — design + implement QueryOrchestrator** (1-2 weeks).
   Aggregator-selector threat model first.
5. **Post-A — end-to-end testnet validation** mirroring T10 pattern.

## Files referenced

| File | Lines | Why |
|---|---|---|
| `prsm/node/node.py` | 1277 | `self.agent_forge = None` (still) |
| `prsm/node/api.py` | 1392 | `prsm_upload_dataset` placeholder CID (still) |
| `prsm/interface/api/websocket_auth.py` | 441 | wrong-submodule import (FIXED this commit) |
| `prsm/core/security/enhanced_authorization.py` | 466 | canonical home of `get_enhanced_auth_manager` |
| `docs/2026-05-07-canonical-workflow-gap-list.md` | (whole) | baseline gap-list — IPFS section §3 now obsolete |
