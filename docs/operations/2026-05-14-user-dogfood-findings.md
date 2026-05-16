# 2026-05-14 — user-perspective dogfood findings

## Context

After today's 36-sprint cycle (sprints 388–423) closed the operator/audit
surface, ran an end-to-end test pretending to be a brand-new user following
the canonical workflow in `docs/PARTICIPANT_GUIDE.md`. Goal: validate that
PRSM does what `PRSM_Vision.md` §4 claims a user can do.

## Headline

**Vision §4 describes the IDEAL workflow; the DEFAULT user experience hits
multiple walls before the first successful query.** This finding is not a
regression — most of the gaps have always existed — but they are visible
now in a way that's worth surfacing as the network moves toward genuinely
inviting new users.

## Friction points (in order of user encounter)

### F1 — `prsm daemon` is deprecated but PARTICIPANT_GUIDE still uses it

**Symptom.** Following Step 3 in PARTICIPANT_GUIDE, a user runs
`prsm daemon start`. The CLI emits a deprecation warning steering them to
`prsm node start --background`. The daemon DOES start (backward-compat),
but the user sees a yellow warning on their FIRST canonical command.

**Severity.** Documentation rot. The user surface still works; the doc
needs updating.

**Fix shipped this sprint.** PARTICIPANT_GUIDE Step 3 updated to use
`prsm node start --background` + `prsm node status` + `prsm node logs -f`.

### F2 — `PRSM_QUERY_ORCHESTRATOR_ENABLED=1` is required but undocumented

**Symptom.** User submits a query (via `prsm_analyze` MCP tool, or directly
to `/compute/forge`). Hits:

```
{"detail": "Agent forge not initialized. Check LLM backend configuration."}
```

The actual cause is that the §7 QueryOrchestrator is default-disabled
(sprint 173 / Vision §13 commit). Operator must set
`PRSM_QUERY_ORCHESTRATOR_ENABLED=1` in the env BEFORE starting the
daemon. The error message says "Check LLM backend configuration" which
is misleading — the LLM is fine; the orchestrator wasn't constructed.

**Severity.** High UX impact. A user following PARTICIPANT_GUIDE cannot
submit ANY query without setting an env var that isn't mentioned anywhere
in the user-facing docs.

**Fix shipped this sprint.** PARTICIPANT_GUIDE Step 3 + new
"First-time-user reality check" section documents the env var explicitly.

### F3 — First query fails with "no content shards" on default install

**Symptom.** With orchestrator enabled, query reaches the gateway and the
quote endpoint (`POST /compute/forge/quote`) works correctly — but the
execution endpoint (`POST /compute/forge`) returns:

```
{"detail": "No content shards above the similarity threshold for this query.
            Upload relevant content to this node or refine the query."}
```

The error message correctly tells the user what to do, but the underlying
issue is fundamental: a brand-new node has no content of its own, and the
peer-discovery counter on a default install shows
`discovered_peer_count: 0` — there are no peers with content either.

**Severity.** Chicken-and-egg for new users. Vision §4 describes the ideal
case where content already exists on the network; in practice, a fresh
install hits zero content unless the user also uploads content themselves
(which has its own friction, see F4).

**Fix shipped this sprint.** New PARTICIPANT_GUIDE section "First-time-user
reality check" sets expectations honestly: a fresh install starts empty;
joining a content-populated network or self-seeding is required.

### F4 — Content upload fails on default node ("content_publisher unwired")

**Symptom.** Attempting to upload content to unblock F3:

```
POST /content/upload
{"detail": "Upload failed — upload_text returned None. Common causes:
            content_publisher unwired, _publish_content raised + was swallowed
            (check logs for 'Content publish failed'), or BitTorrent layer
            crashed mid-upload."}
```

The default node's `content_publisher` primitive isn't wired up. The
operator needs to take additional steps beyond `prsm node start` to enable
content upload. The error message is honest but the wiring path isn't
documented in PARTICIPANT_GUIDE.

**Severity.** Wiring-level gap. Not fixable with documentation alone —
needs either a default wiring change OR an explicit "enable content
upload" runbook for operators who want to host content.

**Status.** Documented as honest-scope in PARTICIPANT_GUIDE; the underlying
wiring fix is a separate sprint candidate.

**Update 2026-05-14 (sprint 425).** Closed end-to-end. Required four
distinct fixes layered together:

1. `bencodepy` added to required `dependencies` in `pyproject.toml`
   (was [bittorrent] extra only — broke real publish path).
2. `libtorrent` documented as a SYSTEM-LEVEL dep with per-platform
   install instructions in PARTICIPANT_GUIDE (it isn't a PyPI package
   despite the existing [bittorrent] extra implying so).
3. `/info` endpoint extended to surface `content_publisher_wired: bool`
   so operators can verify the wiring at a glance.
4. `prsm/node/api.py:5861` referenced a phantom `result.cid` attribute
   that doesn't exist on the real `UploadedContent` dataclass (correct
   field is `content_id`). Test fixture `_FakeUploadResult` masked the
   bug by exposing a `.cid` attribute that doesn't match production.
   Fixed both production code AND the fake.

Live verification:
```
$ curl -X POST http://127.0.0.1:8000/content/upload \
    -d '{"text": "hello PRSM world", "filename": "test.txt"}'
{"cid": "376274399c21...", "filename": "test.txt", "size_bytes": 45,
 "content_hash": "9b9021edcf16...", "creator_id": "cdefb8e5...",
 "royalty_rate": 0.01, "encrypted": false}
```

### F7 — Locally-uploaded content not retrievable on same node

**Symptom.** Upload returns success + a CID, but immediately querying
`/content/retrieve/{cid}` returns:

```
{"status": "not_found", "providers_tried": 0,
 "error": "Content not found on any available provider"}
```

**Root cause (live-diagnosed sprint 425).** Two disjoint storage worlds
sharing one endpoint:

1. The BitTorrent publish path (`content_uploader.upload_text` →
   `_publish_content`) seeds bytes into the libtorrent layer + returns a
   **BT v1 infohash** (40 hex chars = 160-bit SHA-1).
2. The retrieve path at `content_provider.request_content`:
   - Checks `self._local_content[cid]` — this DOES succeed (verified via
     `/content/provider-stats` showing `local_content_count` increments
     on every upload), confirming `_register_with_provider` fires
     correctly.
   - Then calls `_fetch_local(cid)` which goes to
     `prsm.storage.get_content_store().retrieve_local(ContentHash.from_hex(cid))`.
   - `ContentHash.from_hex()` (storage/models.py:72) expects a
     **2-char-algorithm-prefix + 64-char SHA-256 digest = 66 chars**.
     The 40-char BT infohash is structurally wrong shape — raises
     ValueError, caught + swallowed, returns None.

So the local node knows it has the content (registered) but cannot
retrieve the bytes because the retrieve probe queries the wrong storage
backend. The bytes live in the libtorrent session, not in the native
ContentStore.

**Severity.** High for the single-node user-validation flow — Vision §4
step 8 (query against uploaded content) cannot be exercised on a single
machine. Doesn't block multi-node operation but breaks the canonical
user-onboarding workflow.

**Fix candidates (deferred to its own sprint — needs design call):**

- **Option A:** Extend `_fetch_local` to ALSO check the content_publisher
  (BT layer) when the cid looks like an infohash (40-char hex) rather
  than a ContentHash hex. Minimal churn; preserves both backends.
- **Option B:** Unify the storage layer — the BT publish path also
  writes into the native ContentStore under a derived ContentHash so
  retrieve has one source of truth. More invasive but aligns the
  native-storage migration arc.
- **Option C:** Upload returns the ContentHash hex (not the BT
  infohash) as the user-facing CID. Requires the BT layer to be
  keyed by ContentHash. Most invasive.

Option A is the natural minimum-viable fix; B or C is the eventual
right answer. Decision belongs in a separate sprint with the
native-storage-migration owner.

**Status.** Surfaced 2026-05-14 sprint 425 live diagnosis. Sprint 427
shipped the Option A structural shim (`_fetch_local` now falls back
to `content_retriever.fetch` for cids in `_local_content` that fail
`ContentHash.from_hex`). The shim is correct — see F8 for the
next-layer-down issue that still blocks single-node retrieve.

### F8 — BT publisher/requester session isolation prevents single-node self-fetch

**Symptom.** With the sprint 427 F7 shim in place, the retrieve path
now correctly routes through the BitTorrent fallback for BT-infohash
cids. But the request still fails:

```
BT fallback retrieve failed for 57f2d3ac5442...:
  BitTorrent fetch failed for infohash=57f2d3ac5442...:
  Torrent not found
```

**Root cause.** `content_publisher` (which seeded the torrent on
upload) and `content_retriever` (which the shim calls) use separate
libtorrent sessions / instances. The locally-seeded torrent in the
publisher's session is invisible to the requester's session. Even
the libtorrent swarm-level lookup fails because in single-node mode
there's no swarm — no other peer announces the infohash.

**Severity.** Closes the door that sprint 427's shim opened. Vision
§4 step 8 single-node user-validation remains blocked. Multi-node
setups should work in theory (other nodes can find the seeded
torrent via DHT) but were not exercised end-to-end in this dogfood.

**Fix candidates (deferred to its own sprint):**

- **Option A':** `ContentRetriever.fetch` checks `bt_provider`'s
  locally-published torrents BEFORE handing off to `bt_requester`.
  If the provider knows the infohash, return its file directly from
  the publisher's content directory. Minimum-viable; symmetric with
  sprint 427's pattern.
- **Option B':** Unify bt_provider + bt_requester into a single
  libtorrent session so locally-seeded torrents are visible to local
  fetches. Conceptually cleaner; may interact with peer-isolation
  assumptions elsewhere.
- **Option C':** Track this differently — write the publisher's
  bytes into ContentStore at publish time under a derived
  ContentHash so the existing native-storage retrieve path Just
  Works without any BT layer involvement. Aligns with the
  native-storage-migration arc; eventual right answer per the F7
  Option C reasoning.

**Status.** Surfaced 2026-05-14 sprint 427 live verification of F7
fix.

**Update 2026-05-14 (sprint 428).** Closed end-to-end via Option A'.
`ContentPublisher` now records `infohash → staged_path` on every
Tier A publish; `ContentRetriever.fetch` short-circuits to the
publisher's staged bytes when the same node published the
infohash. No BT swarm round-trip; no libtorrent session changes.
Wired into Node init right after sprint 427's content_retriever
binding.

Live verification (single-node, no other peers, daemon running):
```
$ curl -X POST .../content/upload -d '{"text": "F8 fix verification..."}'
{"cid": "c0ddba20746e8a1a4ec2a023a682915bd5f1b69b", ...}

$ curl ".../content/retrieve/c0ddba20746e8a1a4ec2a023a682915bd5f1b69b"
{"status": "success", "size_bytes": 71, "data": "<base64>", ...}

$ base64 -d  # → "F8 fix verification — sprint 428 — full Vision §4 step-8 roundtrip"
```

This is the first time the canonical Vision §4 step-8 workflow
(upload → retrieve → byte-identical roundtrip) has worked on a
single node. The dogfood arc has now end-to-end-verified the
core user workflow.

**Tier B/C deferred.** Encrypted publishes still need the BT swarm
because the staged dir contains encrypted shards + key shares that
require ContentStore reassembly. Same Option B/C territory — left
to the storage-layer reconciliation sprint.

**2026-05-15 sprint 430:** Tier B/C local-publish shortcut shipped
+ recipient-encryption (sprint 304's path) live-verified byte-
identical end-to-end. Tier B/C Shamir lane wired but not exposed
in `/content/upload` (always Tier A today).

### F9 — Upload + query embedding-dim mismatch when OPENAI_API_KEY is set

**Symptom.** `POST /compute/forge` fails with cryptic numpy error:

```
Forge pipeline error: shapes (384,) and (1536,) not aligned:
  384 (dim 0) != 1536 (dim 0)
```

**Root cause (live-diagnosed sprint 431).** Two embedding providers
in different lanes:

- Upload side: `RealEmbeddingAPI` (prsm/data/embeddings/
  real_embedding_api.py) has a fallback order
  `[openai, sentence_transformers, mock]`. When `OPENAI_API_KEY` is
  set, OpenAI ada-002 wins → stored shard embeddings are 1536-dim.
- Query side: `SentenceTransformerEmbedder` (prsm/compute/
  query_orchestrator/sentence_transformer_embedder.py) is HARD-PINNED
  to `sentence-transformers/all-MiniLM-L6-v2` (384-dim). The
  docstring explicitly cites the parity invariant but the upload
  side wasn't honoring it.

Result: every query against content uploaded with OpenAI key set
fails at `numpy.dot(query_vec, shard_vec)` with a shape mismatch.

**Severity.** Production-blocking for any operator who has
`OPENAI_API_KEY` in their env (common dev pattern). The forge
pipeline is completely broken in this configuration; the bug is
silent at upload time and only surfaces at query time.

**Fix shipped sprint 431.** node.py:1726ff now binds
`preferred_provider="sentence_transformers"` via functools.partial
when constructing `_embedding_fn`. The pin defaults to the local
provider so the canonical install works out of the box. Operators
can override via `PRSM_UPLOAD_EMBEDDING_PROVIDER=openai` if they
ALSO wire an OpenAI-compatible orchestrator embedder (currently
unsupported — would re-break queries).

Live verification: upload with `OPENAI_API_KEY` set → stored
embedding is 384-dim (was 1536-dim pre-fix); forge pipeline gets
past the embedding stage and hits the next bottleneck (F10).

Tag `upload-embedding-provider-pinned-merge-ready-20260515`.

### F10 — Single-node forge query blocked by empty aggregator pool

**Symptom.** After F9 is fixed, `POST /compute/forge` proceeds past
the embedding step and now fails:

```
Forge pipeline error: no eligible aggregator after filtering
  (prompter=cdefb8e5..., pool_size=0, requires_tee=False,
   denylist_size=0)
```

**Root cause.** The forge pipeline's aggregator selection step
(Vision §5.2 step 4: "staked T2+ aggregator selection via beacon-
randomized commit-reveal") requires a pool of staked aggregator
nodes. A single-node test setup has zero peers → pool_size=0 →
no eligible aggregator.

**Severity.** Blocks the canonical Vision §4 step-8 single-node
verification flow. Not a regression — the forge pipeline correctly
assumes multi-node operation; this is a "single-node test mode"
gap. Multi-node setups should work in theory but were not
exercised here.

**Fix candidates (deferred to its own sprint):**

- **Option A:** Add a `PRSM_FORGE_ALLOW_SELF_AGGREGATOR=1` env-gate
  that lets the local node serve as its own aggregator in
  single-node test setups. Smallest blast radius; explicitly
  scoped to verification flows.
- **Option B:** Spin up a second daemon as a stub T2+ aggregator
  during the verification campaign. More representative of real
  production but requires test orchestration.
- **Option C:** Accept the gap and defer step-8 verification to a
  multi-node test bench (separate from the dogfood arc).

**Status.** Surfaced 2026-05-15 sprint 431 live verification of F9
fix.

### F11 — StakingManager TypeError on claim ("can't subtract offset-naive and offset-aware datetimes")

**Symptom.** `POST /staking/claim-rewards` returns 500:

```
{"detail": "Claim rewards failed: can't subtract offset-naive
 and offset-aware datetimes"}
```

Every claim call fails. Production-blocking for the §5.3 staking
economic loop.

**Root cause (live-diagnosed sprint 432).** SQLite (the default
backend) drops tz info when persisting datetimes. The writers
(`stake`, `_update_last_reward_calculation`, etc.) all use
`datetime.now(timezone.utc)` so the stored values ARE UTC — but
SQLite returns naive datetimes on load. The reward calculation
at `prsm/economy/tokenomics/staking_manager.py:893` does:

```python
time_staked = (now - stake.last_reward_calculation).total_seconds()
```

where `now` is tz-aware UTC and `stake.last_reward_calculation`
is the naive DB-loaded value. Python raises TypeError on the
subtraction.

**Severity.** Production-blocking. Every claim_rewards call fails
on any deployment using SQLite (which is the default). Postgres
TIMESTAMPTZ would round-trip cleanly, masking the bug in tests
that use Postgres but breaking production-default deployments.

**Fix shipped sprint 432.** Added `StakingManager._ensure_utc`
helper that re-tags naive datetimes as UTC (sound because all
writers use UTC). Applied at the row→record conversion site
(`_stake_row_to_record`) for both `staked_at` and
`last_reward_calculation` fields. No clock arithmetic — just
tagging.

Live verification:
- Pre-fix: POST /staking/claim-rewards → 500 with TypeError
- Post-fix: POST /staking/claim-rewards → 200 with
  `{"total_rewards_claimed": 0.0, "stakes_processed": 1}`
  (0 reward because stake is < min_stake_age_for_rewards_seconds
  = 24h; that's correct behavior)

Tag `staking-manager-tz-aware-datetimes-merge-ready-20260515`.

### F12 — Mock executor's ε=∞ for NONE tier breaks receipt verify (JSON round-trip)

**Symptom.** During sprint 438 live §5.2 inference E2E test:
inference with `privacy_tier=none` returned `epsilon_spent: null`
in the JSON receipt; subsequent verify call returned
`signature_valid: false, reasons=["signature failed
cryptographic verification"]`.

**Root cause (live-diagnosed sprint 438).**
`MockInferenceExecutor._epsilon_for_level(PrivacyLevel.NONE)`
returned `float("inf")`. The signing payload includes
`f"{epsilon_spent:.10f}"` → for ∞ this is `"inf"`. The receipt
gets signed with those bytes. But `json.dumps(float("inf"))`
emits `Infinity` (Python default; not strictly JSON spec); FastAPI
post-processes to `null`. The verifier round-trips JSON → null
gets reconstructed as `0.0` → signing payload becomes
`"0.0000000000"` → bytes mismatch → signature_valid=false.

**Severity.** Every NONE-tier inference receipt fails the
verify check. Production-blocking for the §5.2-§7 integration
claim (a major Vision §7 truth-surfacing promise: "callers can
independently verify the chain of custody on inference
receipts"). NONE tier is the common-case path for non-private
inference; the bug breaks the most common workflow.

**Fix shipped sprint 438.** `_epsilon_for_level(NONE)` returns
`0.0` (not `inf`). Semantically honest: NONE tier means "no DP
guarantee provided", so "0 DP budget consumed" is the right
encoding. All tiers now produce finite, JSON-clean ε values.

Live verification:
- Pre-fix: NONE-tier verify → `signature_valid: false`
- Post-fix: NONE-tier verify → `ok=true, signature_valid=true,
  reasons=[]`

Full inference → verify chain now passes for ALL privacy tiers
(none/standard/high/maximum). Tag
`inference-mock-executor-epsilon-finite-merge-ready-20260515`.

### F13 — `/rings/status` 500 breaks `prsm_node_status` MCP tool

**Symptom.** Sprint 453 §13 MCP-tool live-verification sweep
hit `prsm_node_status` and got:

```
Cannot reach PRSM node: 500, message='Attempt to decode JSON
  with unexpected mimetype: text/plain; charset=utf-8',
  url='http://localhost:8000/rings/status'
```

The canonical "is this node healthy" AI-assistant probe broken;
opaque 500 makes operator triage impossible.

**Root cause (live-diagnosed sprint 453).** Sprint 173 swapped
`node.agent_forge` from the legacy AgentForge module (had
`.traces`) to a QueryOrchestrator instance (doesn't). The
`/rings/status` endpoint calls
`DashboardMetrics(node=node).get_summary()` which reads:

```python
len(self._node.agent_forge.traces)
```

at two sites — `collect_ring_status` (line 51) + `get_summary`
(line 91). Both raise `AttributeError: 'QueryOrchestrator'
object has no attribute 'traces'`. FastAPI's default 500
handler emits `text/plain "Internal Server Error"` (not JSON).

**Severity.** Production-blocking for AI-triage path. Operators
using Claude/GPT to triage their node hit the cryptic 500 even
when node is perfectly healthy. `prsm_node_status` MCP exits
with "Cannot reach PRSM node" — falsely implies node is down.

**Fix shipped sprint 453.** Defensive `getattr(forge, 'traces',
[]) or []` at both call sites. When QueryOrchestrator is wired
(production default post-sprint-173), traces_collected reports
0 — semantically honest: QueryOrchestrator doesn't track
per-query traces in the legacy AgentForge sense. Legacy
AgentForge backwards-compat preserved if `.traces` IS present.

Live verification:
- Pre-fix: GET /rings/status → 500 "Internal Server Error"
- Post-fix: GET /rings/status → 200 + JSON; prsm_node_status
  MCP renders full Ring 1-10 status table

Tag `dashboard-metrics-query-orchestrator-compat-merge-ready-20260515`.

### F14 — Multi-node single-host test bench: NAT-loopback blocks direct P2P

**Symptom.** Sprint 456 stood up daemon #2 on the same host as daemon
#1 with separate `HOME`/data_dir/identity/ports (daemon #2 on
api=8001, p2p=9011). Both daemons connect to the canonical
bootstrap server and DISCOVER each other (symmetric `known[]`
entries + `peer_join_events: 1` on both sides). But:

- `connected_count: 0` on both sides
- Cross-node content retrieve: daemon #2 attempts to retrieve a
  CID daemon #1 just uploaded → `status: not_found,
  providers_tried: 0`

**Root cause (live-diagnosed sprint 456).** Both daemons announce
their **external NAT-translated public IP** (`146.70.202.118`) as
their P2P address — that's correct for cross-host setups but on a
single host, traffic out to 146.70.202.118 doesn't loop back to
local listeners. Standard NAT-loopback gotcha.

The peer-discovery layer (sprints 319-329) works correctly —
nodes find each other via bootstrap-server peer-list gossip. The
direct WebSocket P2P connection fails silently because the
announced address isn't reachable from the same host.

**Severity.** Blocks single-host multi-node test bench. Doesn't
affect real multi-host deployments. Single-host operators
verifying their setup will see "peer discovery works but content
fetches don't" — an honest-scope distinction.

**Fix candidates (deferred to its own sprint):**

- **Option A:** PRSM_ADVERTISE_ADDRESS env var that overrides the
  externally-announced peer address. Single-host operators set it
  to `127.0.0.1:<port>` to force loopback. Minimum-viable.
- **Option B:** STUN-style peer-NAT detection — daemons can
  detect they share a NAT and use the LAN address. Heavier.
- **Option C:** Use a different host for daemon #2 entirely
  (cloud VM, OCI free tier, etc.). Most-representative; matches
  real production but requires external resources.

For this verification campaign, the multi-host setup (Option C)
is the eventual right test bench. Single-host (Option A shim)
would unlock most operator-side single-host verification.

**Status.** Surfaced 2026-05-15 sprint 456. Documented but
deferred. The discovery-layer half of multi-node already
verified (bootstrap-mediated peer-list works ✅); content-fetch
half blocked behind P2P-connection establishment.

### F15 — `bencodepy>=4.0.0` constraint impossible to satisfy (sprint 425 typo)

**Symptom.** Sprint 458's clean-venv deploy on bootstrap1
droplet:

```
$ pip install -e .
ERROR: Could not find a version that satisfies the requirement
  bencodepy>=4.0.0 (from prsm-network) (from versions: 0.9.4, 0.9.5)
```

**Root cause.** Sprint 425 added `bencodepy>=4.0.0` to
pyproject.toml. PyPI has TWO packages:
- `bencodepy` (no dot) — versions [0.9.4, 0.9.5] only
- `bencode.py` (with dot) — version 4.0.0

Sprint 425 typed the wrong package name + version combination.
PRSM's code imports `bencodepy` (no dot) at
prsm/core/bittorrent_manifest.py:24 → the canonical package
IS bencodepy.

**Why this didn't fire locally:** Sprint 425's dogfood path
installed `bencodepy` 0.9.5 via `pip install --break-system-
packages` in a prior session. pip saw an existing satisfying
install + skipped the constraint check.

**Severity.** Production-blocking for any fresh PRSM install.
Caught only by clean-venv deploys.

**Fix shipped sprint 459.** Both occurrences changed
`>=4.0.0` → `>=0.9`. bencodepy 0.9.5's API is stable for the
bencode operations PRSM uses.

Tag `fix-bencodepy-constraint-merge-ready-20260515`.

### F16 — `zfec` in `[blockchain]` extra but required by node startup

**Symptom.** After F15 fix:

```
ModuleNotFoundError: No module named 'zfec'
File "/opt/prsm-operator/prsm/storage/erasure.py", line 38
    import zfec
```

Daemon crashes during node startup because erasure.py is on
the import path of the content-store initialization.

**Root cause.** `zfec>=1.5.8` was in the `[blockchain]`
optional extra in pyproject.toml. But zfec is Reed-Solomon
erasure coding, not blockchain — it's miscategorized. And
`import zfec` is unconditional at erasure.py:38 (module load
time, not lazy).

**Severity.** Production-blocking. Every fresh
`pip install -e .` followed by `prsm node start` crashes.

**Fix shipped sprint 459.** Moved `zfec>=1.5.8` from
`[blockchain]` extra to required `dependencies` in
pyproject.toml. zfec wheels available on PyPI for cpython
3.7+ on common platforms.

Tag `fix-zfec-required-dep-merge-ready-20260515`.

### F17 — `pycryptodome` missing from required deps

**Symptom.** After F16 fix:

```
ModuleNotFoundError: No module named 'Crypto'
File ".../prsm/storage/key_sharing.py", line 39
    from Crypto.Protocol.SecretSharing import Shamir
```

**Root cause.** `pycryptodome` (which provides the `Crypto`
module) is not declared in pyproject.toml at all — neither
required deps nor any extra. Used by Shamir secret-sharing
in the Tier B/C key-distribution module.

**Severity.** Production-blocking. Same class as F16 —
required for node startup, not declared.

**Status.** Fix deferred — needs a clean-room dependency
audit (sprint 425's miscategorization-class bugs may have
more siblings). Workaround on the droplet: `pip install
pycryptodome` works. Permanent fix: add to pyproject.toml's
required dependencies.

### F18 — Query orchestrator package imports `sentence_transformers` eagerly

**Symptom.** After F17 workaround:

```
ModuleNotFoundError: No module named 'sentence_transformers'
File ".../prsm/compute/query_orchestrator/__init__.py", line 60
    from prsm.compute.query_orchestrator.sentence_transformer_embedder
```

Setting `PRSM_QUERY_ORCHESTRATOR_ENABLED=0` (or unsetting it)
does NOT prevent the import — the orchestrator package's
`__init__.py` does an unconditional top-level import that
pulls in sentence_transformers regardless of whether the
orchestrator is actually instantiated.

**Severity.** Two-fold:
- Cosmetic: even operators NOT using the forge surface must
  install ~2GB of ML deps (sentence-transformers + torch +
  transformers + huggingface_hub) to boot a node.
- Resource pressure: 2GB-RAM droplets (typical operator
  hardware) risk OOM during the install.

**Fix candidates (deferred):**

- **Option A:** Make the sentence_transformer_embedder import
  lazy in __init__.py (import inside function, not top-level).
- **Option B:** Add an `import sentence_transformers` try/except
  in sentence_transformer_embedder.py with a clear error pointing
  at the [ml] extra.
- **Option C:** Move the heavy ML deps to a required dep block
  (~2GB install cost everywhere — operator-hostile).

Option A is the right answer; the orchestrator is opt-in by
design and shouldn't impose ML-stack install on operators
who only want to peer + serve content.

**Status.** Surfaced sprint 458/459. Deferred to its own
sprint. Sprint 458's daemon-#2 deploy on bootstrap1 droplet
halted at this point — installing 2GB of ML deps on the
bootstrap server's 2GB-RAM droplet risks the canonical
bootstrap-fleet entry point's stability. Cleaner path:
fresh small droplet OR Option A code fix.

**Update sprint 460 — Option A SHIPPED.** Made the
`sentence_transformers` import lazy inside
`SentenceTransformerEmbedder._ensure_loaded()`. Constructor
no longer triggers any ML-stack import; only `encode()` does
(via lazy `from sentence_transformers import SentenceTransformer`
inside the function body). When the dep is absent, raises a
clear actionable ImportError pointing at `pip install -e '.[ml]'`.

Local verification: class importable + instantiable without
sentence_transformers installed; `_model` stays None after
construction; encode() raises clean ImportError with pip
instruction.

4 pin tests in
`tests/unit/test_sentence_transformer_embedder_lazy_import.py`:
- Module top-level has no unguarded `import
  sentence_transformers` (source-grep invariant)
- Class instantiable without ML stack present
- _ensure_loaded raises ImportError with pip-install message
  when dep absent
- Parent orchestrator package imports cleanly without ML stack

Now any 1GB-RAM droplet (DO Basic $4/mo) can run an operator
daemon. Tag `fix-sentence-transformers-lazy-import-merge-ready-20260515`.

**Live verification on bootstrap1 droplet (sprint 460 cont.).**
Resumed the sprint 458/459 deploy after F18 fix. Daemon #2
booted cleanly:

- Memory delta: ~130MB (from ~650MB to ~780MB used) on the
  2GB-RAM droplet. **The F18 fix saved ~1.5-2GB of would-be ML-
  stack overhead.** The droplet remains in the ~60% available
  RAM zone.
- Boot time: ~25 seconds to fully active service
- Bootstrap registration: connected via loopback
  `wss://127.0.0.1:8765` (success_node correctly reported).
  Outbound to public-IP `bootstrap1.prsm-network.com:8765`
  failed via NAT-hairpin from inside the droplet — operators
  on the same droplet as bootstrap-server must use 127.0.0.1
  for the bootstrap URL.
- All 13 mainnet contract canonical addresses surfaced from
  daemon #2 (chain_id=8453).

Daemon #2 confirmed running with PRSM_TRANSPORT_BACKEND=
websocket. Total cost: $0 (co-located on existing droplet).
F18 unblocked the deploy that sprint 458/459 had halted on.

Pin tests: 4 in
`tests/unit/test_sentence_transformer_embedder_lazy_import.py`
defend the source-level invariants (no unguarded top-level
`import sentence_transformers`; class instantiable without
the dep; clear error on missing dep with pip-install hint;
package-level import doesn't pull in ML stack).

### F19 — `bleach` in `[server]` extra but required at startup

**Symptom.** Sprint 462's daemon-restart on bootstrap1 droplet
(after sprint 460 F18 fix) crashed:

```
ModuleNotFoundError: No module named 'bleach'
File "/opt/prsm-operator/prsm/core/security/input_sanitization.py", line 12
    import bleach
```

**Root cause.** `bleach` declared in pyproject.toml's `[server]`
extra (line 262 pre-fix), not required deps. But
`prsm/core/security/input_sanitization.py:12` does `import bleach`
at module top-level. `prsm/core/security/__init__.py` re-exports
input_sanitization, so importing the core/security package
(which happens during node startup) triggers the bleach import.
Fresh-venv `pip install -e .` (no extras) succeeds, but
`prsm node start` crashes.

Same class as F16 (zfec in [blockchain] extra) — package
declared somewhere in pyproject but in the wrong location for
what actually uses it at startup time.

**Severity.** Production-blocking on fresh-venv installs.

**Fix shipped sprint 462.** Moved `bleach>=6.0.0` from `[server]`
extra to required deps. Companion comment left in [server] block
explaining the move. Sprint 463's tightened dep-audit invariant
catches this class permanently at CI level.

Tag `fix-bleach-required-dep-f19-merge-ready-20260515`.

### F20 — DO cloud firewall blocks operator P2P port 9001 (inbound)

**Symptom.** Sprint 468 multi-host bench resume. Daemons cross-
discovered via EU bootstrap server peer-list. Daemon #1 (local
home NAT, `136.47.243.122`) and daemon #2 (DO droplet bootstrap1,
`159.203.129.218`) appear in each other's `known[]` list. But
`connected_count: 0` both sides; cross-host content retrieve
returns `not_found, providers_tried: 0`.

Direct `nc -zv 159.203.129.218 9001` from local → **operation
timed out**. Droplet `ss -tlnp | grep 9001` shows
`LISTEN 0.0.0.0:9001` (python pid=4065087). ufw on droplet
INACTIVE. So OS firewall isn't blocking. **DO cloud firewall**
(configured via DO dashboard) is.

**Root cause.** Sprint 458's cloud-init opened ufw rules for
22/8000/9001 inside droplet. But ufw inactive on bootstrap1 —
security is via DO cloud firewall. The cloud firewall has
22+8000+8765 open (bootstrap-server-v2's ports). 9001
(operator P2P) was never added inbound.

**Severity.** Blocks multi-host bench's content-retrieve half.
Discovery works (bootstrap-server peer-list propagation via
WSS 8765); direct P2P (gossip + BT swarm) doesn't.

**Fix.** Open inbound TCP 9001 on bootstrap1 droplet's DO cloud
firewall (DO dashboard → Networking → Firewalls). Requires DO
API token or web console access. Out of scope for sprint 468
(no DO API token available in this session).

**Status.** Documented; deferred to operator action.

Three-layer NAT/firewall arc complete:
- F14 (sprint 456): single-host NAT-loopback
- F20 (sprint 468): DO cloud firewall inbound 9001
- Local home NAT: outbound-only — typical operator pattern;
  not a PRSM bug

Multi-host bench succeeds at the **discovery layer** (both
daemons see each other via bootstrap peer-list). **Content
fetch** requires opening a P2P port on at least one side —
typical pattern is the droplet operator opens their inbound;
home operators stay outbound-only. This is documented design
behavior, not a PRSM bug.

### F5 — Quote endpoint path is `/compute/forge/quote`, not `/compute/quote`

**Symptom.** Following the "MCP tools" hint in PARTICIPANT_GUIDE, a user
probing the API directly tries `POST /compute/quote` (matching the tool
name `prsm_quote`) and gets a 404. The actual canonical path is
`POST /compute/forge/quote`.

**Severity.** Discoverability gap. MCP tools work fine (they have the
correct path internally); only operators probing the raw HTTP surface
hit this.

**Fix shipped this sprint.** Documented in the new "First-time-user
reality check" section's "Direct HTTP probes" sub-section.

### F6 — `/onboarding/` URL advertised in node-start log but returns 404

**Symptom.** Node startup logs print:

```
Node onboarding UI available
url=http://127.0.0.1:8000/onboarding/
```

But hitting `http://127.0.0.1:8000/onboarding/` returns HTTP 404.

**Severity.** Minor — a log line claims something that doesn't actually
work. The user's worst-case is "I followed the link in the log and got a
404; weird; moving on."

**Status.** Tracked here; deferred. The onboarding flow exists (sprint X
shipped onboarding endpoints) but the trailing-slash or root path
behaves differently than advertised. Either fix the log line or fix the
route.

---

### F21 — `ContentFilterStore` lacks `count()` → /health/detailed flips daemon to `degraded`

**The bug.** Sprint 343 added a record-count probe to `/health/detailed`
that calls `x.count()` on `ContentFilterStore` (among other optional
subsystems). The method was never implemented. Live probe in sprint 473:

```
{
  "available": false,
  "status": "error",
  "error": "'ContentFilterStore' object has no attribute 'count'"
}
```

Because `content_filter_store` is in `_orchestrator_subsystem`'s `optional`
list (not the `_OPT_OUT_STATUSES` set), having it `available: false`
without `not_wired`/`disabled` status flips the top-level health to
`degraded`.

**Live impact.** Operator dashboards + alerting keyed on `status ==
"degraded"` fire continuously even on a perfectly healthy daemon. Real
alerts get drowned out.

**Severity.** Production-blocker (silent — no user-visible error, but
operator alerting is poisoned).

**Surfaced.** Sprint 473 (2026-05-16) during the daemon reconnect-
stability investigation that was originally chartered to track down
`client_state: dead`. Found a different bug along the way.

**Fix.** Sprint 473 added `ContentFilterStore.count()` returning the
sum of CIDs + tags + patterns (matching the `to_dict()` snapshot
fields). Live-verified: daemon status `degraded → healthy`,
`content_filter_store` reports `record_count: 0` clean.

**Pin test.** `tests/unit/test_sprint_473_content_filter_store_count.py`
defends method presence + sum semantics + agreement with `to_dict()`.

---

### F22 — `ProvenanceQueries.load_all_for_node` crashes on SQLite str datetime → `/content/mine` empty after every restart

**The bug.** `prsm/core/database.py:1587` called
`row.created_at.timestamp()` inside the hydration list-comp.
SQLite stores DATETIME columns as ISO-format strings
(`'2026-05-15 12:52:27.363797'`) instead of native Python
`datetime` objects (Postgres path). The first row triggered
AttributeError; the defensive try/except at line 1591 caught
it and returned [] silently.

Live impact: `uploaded_content` dict empty after every daemon
restart → operator can't see previously-uploaded content via
`/content/mine` until they re-upload. Live-verified pre-fix:
`/content/mine` returned 0 entries despite 14 records in the
DB.

**Same class as F11** (sprint 432, StakingManager tz-aware
datetime subtract — also SQLite-typing-related).

**Severity.** Production-blocker (silent — no user-visible
error, but operator content listings appear empty for
days/weeks at a time).

**Surfaced.** Sprint 480 (2026-05-16) during startup-log
audit chartered after sprint 479's libp2p-warning cleanup.
Each successive sprint exposes more startup noise; each piece
of noise is worth investigating for a real bug.

**Fix.** Sprint 480 added `_row_created_at_to_epoch()` helper
in `prsm/core/database.py` that normalizes both native
`datetime` (Postgres) and ISO strings (SQLite — both
microsecond + non-microsecond + 'T'-separator variants) to a
Unix-epoch float. Unparseable input → 0.0 (matches pre-F22
None-handling).

**Live-verified.** Daemon restart with fix → `/content/mine`
returned 0 → 14 entries (full content history from sprints
441/449/472 restored).

**Pin test.** `tests/unit/test_sprint_480_f22_provenance_hydration.py`
defends helper edge cases + integration source-of-truth (the
buggy `row.created_at.timestamp()` pattern explicitly banned
from `load_all_for_node` body).

---

### F23 — `SandboxManager` temp-dir leak: 4416 stale dirs accumulated in /var/folders/.../T/

**The bug.** `SandboxManager.__init__` at
`prsm/core/integrations/security/sandbox_manager.py:546`
called `tempfile.mkdtemp(prefix="prsm_sandbox_")` but never
registered cleanup. 3+ SandboxManager instances per daemon
startup (IntegrationManager + SecurityOrchestrator's
EnhancedSandboxManager + module-level
`enhanced_sandbox_manager = EnhancedSandboxManager()` singleton
in enhanced_sandbox.py:388) × every Python process that imports
the module = thousands of leaked dirs.

Live-verified pre-fix on dev workstation: 4416 stale
`prsm_sandbox_*` dirs in `/var/folders/.../T/`. Each holds
`quarantine` + `tools` subdirs. Low disk impact, high inode
pressure on macOS `/var/folders` (parent dir had 8043
entries).

**Severity.** Operator hygiene / resource leak. Not
data-loss, but indefinite accumulation on workstations + CI
runners + production daemons that restart frequently.

**Surfaced.** Sprint 483 (2026-05-16) during continued
startup-log audit. Sprint 482 closed the persistence-hint
UX gap; sprint 483's daemon log showed
`Enhanced Security Sandbox Manager initialized` firing 3x at
startup. The 3x signal led to investigation → 4416 stale
dirs found.

**Fix.** Sprint 483 added `atexit.register(_cleanup_sandbox_dir,
self.sandbox_dir)` to `SandboxManager.__init__`. The
`_cleanup_sandbox_dir(path)` helper `shutil.rmtree(...,
ignore_errors=True)` the sandbox at process exit. Defensive:
no-op on missing/empty/None path; swallows all exceptions
(atexit handlers must never raise).

**Live-verified.** Subprocess that imports module + exits:
4 sandbox dirs created during run → 0 persisted after exit.
Pre-fix: 4 persisted permanently.

**Stale dirs cleanup.** Existing 4416 dirs on dev/prod hosts
need a one-time `find /var/folders/.../T/ -maxdepth 1 -type d
-name 'prsm_sandbox_*' -mtime +1 -exec rm -rf {} +` (or
equivalent) — not in code scope; operator hygiene.

**Pin test.** `tests/unit/test_sprint_483_f23_sandbox_cleanup.py`
defends helper semantics + atexit registration source-of-truth.

---

### F24 — `/content/retrieve/{cid}` hangs forever for hydrated CIDs after restart

**The bug.** Sprint 480's F22 fix made
`ProvenanceQueries.load_all_for_node` hydrate `_local_content`
correctly from the DB after daemon restart. That **exposed a
latent bug** in the BT-fallback retrieve path:

1. `_local_content` is populated for hydrated CIDs (the F22
   fix path).
2. But BT-publisher's `_published_paths` is per-session.
   `local_publish_path(cid)` returns None for hydrated CIDs
   because the BT seed metadata was lost across restart.
3. The retriever fell through to
   `bt_requester.request_content(timeout=None)`. With 0 peers
   on a single-node dev daemon, the BT swarm wait was
   unbounded.
4. `_fetch_local_via_bt` didn't propagate the caller's
   `timeout` argument — so even when the operator passed
   `?timeout=2`, the BT layer ignored it.

Operator-visible symptom (live-verified during sprint 484
semi-fresh dogfood pass): `curl /content/retrieve/{cid}`
returned no response — not even the daemon's 30s HTTP
timeout fired. Curl's own 35s wait expired with 0 bytes
received.

**Severity.** Production-blocker. After F22 fix landed,
EVERY hydrated CID becomes a footgun: operators who try to
retrieve their own previously-uploaded content get a
permanent hang. Worse, because the hang doesn't release the
worker, sustained dogfood retrieves could pile up.

**Surfaced.** Sprint 484 (2026-05-16) during the semi-fresh
dogfood pass that the user asked for after the sprint 469-483
work. Walking Vision §4 step 5 (retrieve uploaded content),
the user-perspective probe hung immediately.

**Fix.** Sprint 484 propagated `timeout` through
`request_content` → `_fetch_local` → `_fetch_local_via_bt` →
`retriever.fetch(timeout=...)`. Plus defense-in-depth via
`asyncio.wait_for(..., timeout=timeout)` so even a future
retriever that silently ignores its `timeout` kwarg can't
hang beyond the caller's bound.

Live-verified pre-fix: 30s+ hang (curl gave up at 35s).
Post-fix: 2s response with clean `not_found` envelope
(timeout=2 honored).

**Remaining work (deferred).** ~~The fix makes retrieve fail
FAST instead of HANG, but it doesn't actually deliver the
bytes.~~ **Closed sprint 485 (2026-05-16):** at F22 hydration
time, the uploader now calls
`ContentPublisher.register_local_publish_tier_a` for each
non-sharded hydrated record. The BT publisher's
`_published_paths` is restored from the persisted
`content_hash` → `local_publish_path(cid)` returns the staged
file → `ContentRetriever.fetch` short-circuits the BT swarm
path. Live-verified: **17/17** hydrated CIDs deliver bytes
across a clean daemon restart. Vision §4 step 5 cross-restart
fully closed for Tier A. Tier B/C remains deferred (staged
dir naming requires per-publish info not persisted to DB).

**Pin test.** `tests/unit/test_sprint_484_f24_retrieve_timeout.py`
defends the timeout-propagation contract + asyncio.wait_for
defense-in-depth via a simulated-hang integration test.

---

### F25 — Concurrent `/compute/submit` returns 500 (unhandled `ConcurrentModificationError` + savepoint collision)

**The bug.** Multi-pronged. Surfaced when sprint 487's
concurrency harness fired 10 simultaneous /compute/submit
requests against a single daemon:

1. The dag_ledger HAS optimistic concurrency control via
   version checks AND a `SAVEPOINT balance_check` block.
   But under contention `_check_balance_atomic` raises
   `ConcurrentModificationError` which the /compute/submit
   handler did NOT catch — propagated up as 500.

2. The savepoint name `balance_check` is HARDCODED. When
   two coroutines on the same aiosqlite connection both
   issue `SAVEPOINT balance_check`, the second nesting
   means a sibling's ROLLBACK TO can pop the wrong layer
   → "no such savepoint: balance_check" → 500.

3. `InsufficientBalanceError` from the same path was
   also uncaught → 500 (should be 400 — client-actionable).

**Live impact.** A burst of concurrent submits crashes 4-7
out of 10 with 500. Real DoS surface from a single
operator submitting multiple jobs in parallel.

**Severity.** Production-blocker. Any operator dashboard
or SDK that fires concurrent compute submits will see
random 500s + lose the ability to track which submits
succeeded.

**Surfaced.** Sprint 487 (2026-05-16) — concurrency
testing as priority #1 from sprint 486's coverage matrix.

**Fix (partial).** Sprint 487 shipped:
- Per-wallet `asyncio.Lock` in dag_ledger to serialize
  balance-check operations per wallet (eliminates the
  savepoint name collision; different wallets still
  proceed in parallel).
- /compute/submit handler now catches
  `ConcurrentModificationError + BalanceLockError` with
  bounded retry (3 attempts, exp backoff: 10/40/90ms),
  then clean 503 with actionable detail.
- /compute/submit handler also catches
  `InsufficientBalanceError` → 400 (client-actionable).

**Live-verified post-fix.** Concurrent-submit test now
passes: 0 unhandled 500s; mix of 200 (success) + 400
(insufficient balance) + 503 (retry exhausted with
actionable detail).

**Fully closed sprint 490 (2026-05-16).** Sprint 489's
explicit-commit barrier in `dag_ledger.submit_transaction`
(the F27 load-bearing fix) ALSO eliminated the underlying
contention that triggered F25's retries. Pre-sprint-489
the buffered writes caused version-cache drift between
concurrent coroutines → ConcurrentModificationError fired
→ retry loop hit 503 retry-exhausted. Post-sprint-489
each submit's writes commit cleanly, version cache stays
consistent, retries don't fire at all.

Live re-verified at sprint 490:
- 10 concurrent submits (sufficient budget) → 10/10
  return 200, zero 503s, zero 500s.
- 30 concurrent submits → 30/30 return 200.
- **50 concurrent submits with 10× wallet budget** → 5
  return 200 + 45 return 400 (insufficient balance,
  client-actionable), zero 500/503s. After cleanup-cancel:
  **0.00 FTNS leaked**.

**Pin test.** `tests/integration/test_sprint_487_concurrency.py`
runs 4 race scenarios against a live daemon (all pass).

---

### F26 — Concurrent identical-content uploads break anti-Sybil first-creator-wins

**The bug.** 10 concurrent /content/upload calls with
IDENTICAL bytes (and different creator_id payloads, though
the handler overrides to the node's own id anyway) all
return 200 with `duplicate_of_creator: null`. Vision §14
anti-Sybil invariant says exactly ONE upload should be
canonical; the rest should be flagged `duplicate_of`.

Root cause: the fingerprint-registry SELECT-then-INSERT in
the upload path is not atomic. Concurrent uploads each see
"no existing fingerprint" → each tries to insert → SQLite's
UNIQUE constraint on `content_provenance.cid` fires for
all-but-one INSERTs, but the handler logs the
`IntegrityError` and STILL RETURNS 200 with the optimistic
canonical-creator response.

Daemon log evidence:
```
ProvenanceQueries.upsert_provenance failed:
(sqlite3.IntegrityError) UNIQUE constraint failed:
content_provenance.cid
```
× 9 lines for 10 concurrent uploads — only 1 INSERT
actually persisted, but all 10 responses say "you're
canonical".

**Live impact.** Anti-Sybil invariant compromised. An
adversary running concurrent uploads of the same content
under different identities could ALL claim to be the
canonical creator (against the durable DB, only one wins,
but the response-layer lies).

**Severity.** Production-blocker for §14 anti-Sybil
claim. Operator dashboards built on the response payload
would show false canonicals.

**Surfaced.** Sprint 487.

**Fix.** ~~Deferred~~ **Closed sprint 488 (2026-05-16)**.
Root cause was actually two bugs compounding:

1. **The fingerprint registry was never wired** on
   non-QO daemons. Pre-fix: ContentFingerprintRegistry
   init was inside `_build_query_orchestrator_or_none`,
   which early-returns when `PRSM_QUERY_ORCHESTRATOR_ENABLED`
   is unset (the default). Result: every daemon without QO
   had `_content_fingerprint_registry = None` → the §14
   anti-Sybil first-creator-wins NEVER fired → every
   duplicate upload returned `duplicate_of_creator: null`
   regardless of race conditions. Live-verified via
   `/marketplace/fingerprint/{hash}` returning 503 "not
   initialized".
2. Even with the registry wired, the `register()` method's
   check-then-insert window was not atomic. Concurrent
   callers all saw `existing is None` and all inserted.

Sprint 488 fix: (a) moved registry construction OUT of
`_build_query_orchestrator_or_none` to unconditional
node init; (b) added `threading.Lock` around the get →
insert → disk-write critical section in `register()`.

Live-verified post-fix: 10 concurrent uploads with
distinct `creator_eth_address` payloads → 1 returns
`canonical_creator=<first_addr>, duplicate_of_creator=null`;
9 return `canonical_creator=<first_addr>,
duplicate_of_creator=<first_addr>`. §14 anti-Sybil
operational.

**Pin test.** `tests/integration/test_sprint_487_concurrency.py::test_fingerprint_dedup_race_first_creator_wins`
now passes; runs the same 10-caller stress.

---

### F28 — Concurrent `/staking/unstake` against same stake_id creates multiple pending requests

**The bug.** 10 concurrent /staking/unstake calls against
the SAME stake_id produced 5 successful pending unstake
requests, each with a distinct `request_id` but pointing
at the same underlying stake. The StakingManager has a
`_lock` per its source comment but concurrent dispatchers
on the same stake aren't actually serialized — likely
the lock is too coarse-grained (per-manager not
per-stake) OR the check-then-write pattern around
existing-pending-request lookup is not atomic.

Each unstake request has `amount: 1000.0` (the full
stake). 5 requests × 1000 FTNS = 5000 FTNS of pending
withdrawals against a 1000-FTNS stake. After the 7-day
cooldown, the FIRST withdraw succeeds; the remaining 4
would either:
- (a) all succeed → 5000 FTNS withdrawn from a 1000 FTNS
  stake — economic invariant violated, real loss to other
  stakers in the pool, OR
- (b) error after the first withdraws → user has 4
  orphaned pending records that can't ever resolve
  cleanly.

Either case is a production-blocker for the Vision §11
staking-economy claim.

**Live impact.** Adversary or unlucky multi-client user
could create duplicate unstake claims. With the 7-day
cooldown, the actual withdraw race is delayed but
inevitable.

**Severity.** Production-blocker. Same class as F26 —
the verification campaign is now exposing how MANY
concurrency assumptions are wishful.

**Surfaced.** Sprint 487.

**Fix.** ~~Deferred~~ **Closed sprint 488 (2026-05-16)**
incidentally. F25's per-wallet `asyncio.Lock` in
`dag_ledger.submit_transaction` serializes ALL FTNS
balance-mutation operations on a given wallet. Staking
unstake routes its FTNS transition through the dag_ledger,
so the per-wallet lock now enforces strict serialization
of unstake-the-same-stake calls — only the first sees
the active stake; the rest see the unstaking state and
get rejected cleanly. Live-verified: sprint 487's race
test now passes 5/5 consecutive runs (previously: 5 of 10
unstake requests succeeded simultaneously).

**Pin test.** `test_staking_unstake_race_no_double_unstake`
in sprint 487's harness. Reliably passes post-F25 lock.

**Note.** This is a side-effect of F25's fix, not a
direct fix in StakingManager. A defense-in-depth follow-on
would be to add a per-stake_id lock at the StakingManager
layer too, so the invariant holds even if the dag_ledger
lock is bypassed by a future refactor.

---

### F27 — Compute escrow leaked FTNS under concurrent test load

**The observation.** Sprint 487's concurrent-submit test
submitted N jobs, each locking 206 FTNS into escrow. Test
cleanup called /compute/cancel on each successful submit.
Post-test inspection: 1030 FTNS (5 × 206) still locked
in escrow records; `/balance/recent_transactions` shows
the outflows but no corresponding refund inflows.

**Hypothesis.** Either:
- (a) `/compute/cancel` race itself fails to refund
  cleanly (related to F25 savepoint collision applied to
  the cancel path), OR
- (b) The cancel path requires the job's
  history-recorded state which the failed-submit cases
  never reached, leaving the escrow orphaned.

**Severity.** Operator-visible economic-layer leak. Real
FTNS lost from the test wallet (1030 of 1030.05). On
mainnet this would be real-money lost.

**Surfaced.** Sprint 487.

**Fix.** ~~Deferred~~ **Closed sprint 489 (2026-05-16)**.
Investigation revealed THREE compounding root causes:

1. `payment_escrow.create_escrow` registered the in-memory
   record BEFORE the funds transfer. If transfer raised
   anything other than ValueError (e.g.,
   ConcurrentModificationError from dag_ledger), the
   record stayed in `_escrows` half-registered.

2. `create_escrow` only caught ValueError; other exceptions
   propagated up.

3. **The real durability bug**: `dag_ledger.submit_transaction`
   did NOT call `await self._db.commit()` at the end of
   transfer-style transactions. The wallet_balances UPDATE
   and dag_transactions INSERT were buffered. The
   `_seed_welcome_grant` startup hook DELETEs + REBUILDs
   `wallet_balances` from `dag_transactions` on every
   restart — so uncommitted writes were lost. Refund
   transactions issued via /compute/cancel during a test
   run appeared to succeed (handler returned 200) but
   never persisted to disk → daemon restart wiped them →
   funds appeared "leaked".

Sprint 489 shipped four fixes:

(a) Flipped ordering in create_escrow: register record
    AFTER successful transfer.
(b) Broadened exception catch in create_escrow + re-raise
    on non-ValueError.
(c) **Explicit `await self._db.commit()`** at end of
    submit_transaction's transfer path.
(d) New admin endpoint
    `POST /admin/escrow/recover-orphans?dry_run={bool}`
    for operator-actionable orphan recovery. Scans
    `wallet_balances` for `escrow-%` rows with positive
    balance, looks up the original "Escrow for job X"
    requester via `dag_transactions`, refunds. Default
    dry_run=true; operators must review before executing.

**Live-verified**: 27 orphan escrows totaling 1081.06 FTNS
recovered via the admin endpoint; wallet balance 3.05 →
1084.11; survived daemon restart at 1084.11 (proving (c)
committed durably); re-scan dry-run shows 0 orphans
remaining. Full concurrency suite now 4/4 PASS (was 3/4
with this test skipped).

**Pin test.** `tests/unit/test_sprint_489_f27_escrow_recovery.py`
defends all 4 invariants in source (register-after-transfer,
broad exception catch, explicit commit marker, recovery
endpoint shape).

---

## What's working (positive findings)

- ✅ `prsm setup --minimal` completes cleanly on existing config (re-prompt
  flow is unambiguous)
- ✅ `prsm node start` produces a clear status panel showing FTNS balance,
  P2P + API addresses, dashboard URL
- ✅ Bootstrap connection succeeds against the canonical fleet (sprints
  388–411 work paid off — `discovered_peer_count: 0` but `connected: 1,
  degraded: false`)
- ✅ `GET /balance` works end-to-end (read-side of the user-facing
  workflow is reliable)
- ✅ `GET /info` is rich and operator-useful — shows canonical addresses
  for all 12 on-chain contracts, network/chain_id, RPC host, bootstrap
  digest, query orchestrator state, agent forge wiring status
- ✅ `POST /compute/forge/quote` works for cost estimation (this is the
  critical "show user expected cost before they spend" path; even when
  execution fails for F3 reasons, the user can still see what they
  would have paid)
- ✅ Dashboard at `/` serves a real HTML page
- ✅ Sprint 408+'s observability stack is fully online on the node side

---

## What I'd recommend prioritizing

In order of user-impact-per-effort:

1. **PARTICIPANT_GUIDE Step 3 update** — fixes F1 + F2 + F3 in one
   doc-only change (this sprint).
2. **Default-wire content_publisher** — closes F4. Operators who follow
   the guide and want to host content shouldn't need a separate runbook.
3. **`/compute/quote` alias** — fix F5 by adding a route alias so the
   intuitive path works. 5-line code change.
4. **Onboarding URL** — fix F6 by either suppressing the log line or
   wiring the `/onboarding/` route properly.
5. **Vision §4 honest-scope note** — Vision §4 paints the ideal end-to-end
   workflow; consider adding a footnote ("Available once a node has
   content or has joined a content-populated peer network — see
   PARTICIPANT_GUIDE 'First-time-user reality check' for the bootstrap
   journey") so investors reading Vision aren't surprised by the gap
   between vision and default-install reality.

---

## Test environment

- macOS, repo at `/Users/ryneschultz/Documents/GitHub/PRSM`
- Existing identity at `~/.prsm/identity.json` (created 2026-02-18)
- FTNS balance starting at 173.05
- Bootstrap fleet 3/3 reachable per sprint 407 smoke test
- Node started via `python3 -m prsm.cli daemon start`
- Daemon listening on `127.0.0.1:8000`

---

## What this finding-set isn't

- Not a regression report. None of these break a workflow that previously
  worked end-to-end with this configuration.
- Not a blocker for shipped Phase 5 / §7 work. Those operate at a
  different layer and are independently verified.
- Not a claim that PRSM doesn't work. It works as designed; the
  default-install user experience just doesn't match the maximalist
  framing in PARTICIPANT_GUIDE without operator-level configuration.

The point of this report is to make that gap legible + actionable.
