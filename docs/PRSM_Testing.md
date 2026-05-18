# PRSM Functionality Verification Roadmap

> **Authoritative testing roadmap.** Enumerates every shipped piece of
> functionality that should work end-to-end, with current verification
> status. Use this doc to drive the systematic test campaign that
> brings each surface online.

**Last updated:** 2026-05-15 (sprint 429)
**Maintained by:** the autonomous development loop — updated after every
sprint that closes or surfaces a verification gap.

---

## Why this doc exists

The 2026-05-14 user-perspective dogfood arc (sprints 424-428) exposed
the gap between **"tests pass"** and **"feature actually works for a
user"**:

- F4 — `/content/upload` had been "tested" via three fixtures whose
  shape didn't match the real `UploadedContent` dataclass. The
  endpoint shipped with a phantom `result.cid` reference that 500'd
  on the first real upload.
- F7 — `/content/retrieve` returned "not_found" for every locally-
  published cid because the retrieve probe queried a different
  storage backend than the publish path used.
- F8 — Even after the F7 shim, the BT requester couldn't see the
  publisher's seeded torrents because they ran in separate
  libtorrent sessions.

All three bugs hid behind 10k+ green CI tests. This doc is the
inventory that prevents that pattern by enumerating every claim and
gating it on real verification, not just "the test file exists."

## Status legend

| Symbol | Meaning |
|--------|---------|
| ✅ | **Verified end-to-end** — live-tested against a running daemon, byte-identical or behavior-confirmed |
| 🟢 | **Test-pinned** — CI green, but not yet exercised against a real running system |
| ⚠️ | **Partial** — some scenarios verified, others untested or broken |
| 🔬 | **Untested** — claimed live in Vision §11 but no end-to-end verification exists |
| ⏸️ | **Deferred** — intentionally not built yet (gantt-listed for a later sprint) |
| ❌ | **Broken** — known regression with no fix yet |
| 🔗 | **External-gated** — depends on a third-party action (audit, multi-sig ceremony, regulatory filing) |

When a feature is ✅, the **Sprint** column points to the sprint that
shipped or last verified it. When a feature is 🔬 or ⚠️, the
**Notes** column states what needs to be verified.

---

## §4 — End-to-end user workflow (canonical 8-step Vision walkthrough)

The Vision §4 "How It Works" section paints the canonical user
journey. Each step should be live-verifiable on a single node.

| Step | Feature | Surface | Status | Sprint | Notes |
|------|---------|---------|--------|--------|-------|
| 1 | Install PRSM | `pip install prsm` | ✅ | — | Verified during dogfood arc |
| 2 | First-run setup | `prsm setup --minimal` | ✅ | — | Verified |
| 3 | Start node | `prsm node start --background` | ✅ | 424 | Sprint 424 fixed deprecated `prsm daemon` path |
| 4 | Upload content (plain, Tier A) | `POST /content/upload` | ✅ | 425/428 | Live roundtrip green; sprint 425 fixed `result.cid` bug |
| 5 | Verify retrieval (same node, Tier A) | `GET /content/retrieve/{cid}` | ✅ | 428 | First end-to-end single-node roundtrip; byte-identical |
| 6 | Upload encrypted content (recipient encryption) | `POST /content/upload` + recipients | ✅ | 430 | X25519+XChaCha20 encrypt-then-publish; live byte-identical roundtrip |
| 7 | Retrieve encrypted content (same node) | `GET /content/retrieve/{cid}` + decrypt | ✅ | 430 | Live-verified: 431-byte ciphertext → decrypt → byte-identical plaintext |
| 7a | Tier B/C Shamir multi-shard lane (infrastructure) | `ContentPublisher.publish(tier=B)` | 🟢 | 430 | Local-publish shortcut wired (routes staged dir to `_fetch_tier_bc`); not yet exposed in `/content/upload` (always Tier A today) |
| 8 | Query against uploaded content (multi-node only by design) | `POST /compute/forge` | 🟢 | 431 | Embedding stage ✅ verified live (F9 closed); aggregator stage is multi-node-only by A2 design invariant (F10); needs multi-node test bench |
| 8a | Quote query cost | `POST /compute/forge/quote` | ✅ | — | Verified during dogfood arc — works on fresh node |
| 8b | Forge embedding-stage parity | `_embedding_fn` vs `SentenceTransformerEmbedder` | ✅ | 431 | F9 closed: upload-side pinned to sentence_transformers; 384-dim parity verified live with `OPENAI_API_KEY` set |
| 9 | Receive FTNS settlement | `GET /balance` + RoyaltyDistributor | 🔬 | — | Requires cross-node query + on-chain dispatch; not exercised single-node |

---

## §5.1 — Data Layer: BitTorrent + libtorrent + ContentStore

### Content upload + retrieval

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Upload plain text/bytes (Tier A) | `POST /content/upload` | ✅ | 425/428 | E2E |
| Upload encrypted (recipient encryption) | `POST /content/upload` w/ recipients | ✅ | 430 | Live byte-identical roundtrip |
| Retrieve by CID (Tier A, same node) | `GET /content/retrieve/{cid}` | ✅ | 428 | Local-publish shortcut |
| Retrieve + decrypt (recipient-encrypted, same node) | `GET /content/retrieve/{cid}` + `decrypt_for_recipient` | ✅ | 430 | E2E live-tested |
| Bootstrap-mediated peer discovery (multi-node) | bootstrap-server peer-list | ✅ | 456, 468 | Live: 2 daemons on same host (sprint 456); sprint 468 cross-HOST verified — daemon #1 (local home NAT) + daemon #2 (DO droplet) symmetric known[] via EU bootstrap |
| Cross-host content retrieve (direct P2P) | BT swarm + gossip | ⏸️ | 468 | F20: DO cloud firewall blocks inbound TCP 9001 on droplet; direct P2P fails. Discovery works; fetch needs operator firewall config |
| Peer lifecycle: peer_leave propagation | bootstrap-server peer-list | ✅ | 457 | Live: killed daemon #2 → 21s later daemon #1 received peer_leave → `peer_leave_events: 1`, `known_count: 0`, `discovered_peer_count: 0`. Sprint 320-329 hardening operationally verified |
| Direct P2P connection (single-host) | WebSocket transport | ⏸️ | 456 | F14: NAT-loopback blocks single-host direct connection (announced addrs are external IP); multi-host bench is the right test |
| Retrieve by CID (Tier A, cross-node) | `GET /content/retrieve/{cid}` | 🔬 | — | Requires F14 fix or multi-host test bench |
| Retrieve by CID (Tier B/C Shamir, same node) | `ContentRetriever.fetch` | 🟢 | 430 | Routing pinned; infrastructure-only — `/content/upload` doesn't reach this lane today |
| Upload shard | `POST /content/upload/shard` | ⚠️ | 102, 472 | Live schema-pass: missing `dataset_id` → 400 with clean message; full E2E untested (needs sharded-dataset fixture) |
| Recipient manifest | `GET /content/recipient-manifest/{cid}` | ✅ | 304, 472 | Live: schema-defended 422 with `"not an encrypted recipient bundle"` when CID is plaintext Tier A (honest-scope: this is only valid for encrypted Tier B/C bundles) |
| Content metadata | `GET /content/{cid}` | ✅ | 472 | Live: returns 404 `"Content not found in index"` for un-indexed CIDs (Tier A uploads don't auto-index); index-populating uploads addressed by sprint 449's /content/index/stats |
| Content search | `GET /content/search` | ✅ | 287, 472 | Live: returns `{query, results: [], count: 0}` clean envelope; tier filter validates `min_tier` enum (`low/medium/high`) — invalid → 400 with the canonical list |
| My uploaded content | `GET /content/mine` | ✅ | 449, 472 | Live: post-upload entry returns full canonical schema `{content_id, filename, size_bytes, content_hash, creator_id, royalty_rate, access_count, total_royalties, provenance_tx_hash, created_at, is_sharded}` |

### Storage subsystem

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Storage stats | `GET /storage/stats` | ✅ | 449 | Live: returns full schema (storage_available, pledged_gb, used_gb, available_gb, pinned_cids, reward_rate, challenge_config + challenge_stats nested) |
| Pinned content stats | `GET /storage/pinned-stats` | ✅ | 449 | Live: returns `{pinned: [], count: 0}` empty-state |
| Provider reputations | `GET /storage/provider-reputations` | ✅ | 449 | Live: returns `{providers: {}, count: 0}` empty-state |
| Content index stats | `GET /content/index/stats` | ✅ | 449 | Live: returns `{indexed_cids, unique_providers, keyword_entries}` |
| Content mine (own uploads) | `GET /content/mine` | ✅ | 449 | Live: returns sprint 441's fingerprint-test uploads with content_id/filename/size_bytes/content_hash/creator_id/royalty_rate/access_count |
| Provider stats | `GET /content/provider-stats` | ✅ | 428 | Used during F8 diagnosis to verify register_local_content fired |

### BitTorrent layer

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Tier A publish | `ContentPublisher.publish` | ✅ | 428 | Sprint 428 wired infohash → staged_path map |
| Tier B/C publish | `ContentPublisher.publish` | 🟢 | — | Multi-file torrent layout tested in `_legacy/` suite |
| Tier A local-publish shortcut | `ContentRetriever.fetch` | ✅ | 428 | First single-node self-fetch path |
| Tier B/C local-publish shortcut (Shamir dir) | `ContentRetriever.fetch` → `_fetch_tier_bc` | 🟢 | 430 | Routes staged dir to existing Tier B/C reassembly; unit-pinned |
| BT swarm fetch (cross-node) | `bt_requester.request_content` | 🔬 | — | Not exercised in dogfood single-node setup |
| CLI: torrent create/add/list | `prsm torrent ...` | 🟢 | — | Commands exist; operator workflow untested |

### Content provenance + dedup

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| SHA-256 fingerprint registry | `POST /content/upload` hook | ✅ | 291, 425, 441 | Live-verified via §14 chain (sprint 441): upload→ content_hash + canonical_creator recorded; sprint 425 fixed fixture-drift |
| Duplicate detection on re-upload | response `duplicate_of_creator` | ✅ | 291, 441 | Live: re-upload identical text with creator B → response `duplicate_of_creator=A`, canonical preserved (first-creator-wins anti-Sybil invariant) |
| Marketplace fingerprint lookup | `GET /marketplace/fingerprint/{hash}` | ✅ | 291, 441 | Live: returns `duplicate_attempt_count` and canonical-creator linkage |
| EmbeddingDHT cross-node embedding gossip | `prsm.dht.embedding_dht_client` | 🟢 | T3.6 | Vision §11 claims live |
| BinaryFingerprint perceptual hashes | `prsm/marketplace/binary_fingerprint.py` | 🟢 | T4.7 | Calibration deferred to testnet traffic |
| V2 ProvenanceRegistry on-chain embedding commitment | on-chain | ✅ | — | Deployed `0xe0cedDA354...` |
| **On-chain provenance register CLI** | `prsm provenance register <file>` | ✅ | 520 | **First-ever PRSM-daemon-signed ProvenanceRegistry TX on Base mainnet**: tx `0x84b8084b…` block 46165810, success, 50470 gas @ 0.006 Gwei = 0.0000003 ETH. ProvenanceRegistered event emitted. Hash `0xa97f3411…` registered with creator=operator wallet, 800 bps royalty (8%). Vision §11 creator-provenance promise live-attested via CLI |
| **Auto-register provenance on upload** | `POST /content/upload` → `_register_on_chain` | ✅ | 523 | **Vision §11 "creator provenance happens automatically on upload" live-attested**: with `PRSM_PROVENANCE_REGISTRY_ADDRESS + PRSM_ONCHAIN_PROVENANCE=1` set, upload of new content triggers ProvenanceRegistry write transparently. TX `0x82d1776d…` block 46166531 success, 116157 gas |
| **Upload response surfaces provenance_tx_hash** | `POST /content/upload` response field | ✅ | 524 | Threaded through immediate response — single-call confirmation. Live-verified TX `0x2e920d02…`. Defaults to None when no on-chain write happens |
| **Provenance dedup pre-flight** | `prsm provenance register` → `is_registered` check | ✅ | 521 | Re-attempt returns "note: already registered, no-op" WITHOUT spending gas on revert. Saves ~20k gas per double-register attempt |
| **Sustained provenance register + metadata URI** | sustained writes | ✅ | 522 | 2nd TX `0x3c5450f5…` block 46166163, 500 bps royalty + `ipfs://example/sprint522` metadata. Variable royalty_bps + non-empty metadata_uri persist on-chain |
| **V2 ProvenanceRegistry auto-register** | `_build_provenance_client_or_none` → V2 client | ✅ | 526 | **F42 fix**: V1 client incompatible with V2's 5-arg `registerContent`. Added V1-compatible shim to V2 client + V2 detection in build path. **First-ever PRSM-daemon-signed V2 TX `0xdf7cf13b…`** block 46167646 success, to V2 contract `0xe0cedDA354…`, 121276 gas |
| **V2 read-path routing in CLI** | `prsm provenance info` → V2 client | ✅ | 527 | Symmetric with sprint-526 write fix. CLI `_make_client` detects V2 contract + dispatches. Round-trip works for both V1 + V2 hashes |
| **provenance_tx_hash persists across daemon restart** | DB column + 4-layer threading | ✅ | 528 | **F43 fix**: column added, upsert + load + persist + hydrate all updated. Live-verified: tx `0xae097b12…` survived 3 daemon restarts via `/content/mine` |
| **creator_eth_address persists + auto-fills** | DB column + auto-fill on upload | ✅ | 529 | **F44 fix**: same 4-layer persistence pattern as F43. Plus auto-fill: upload defaults to `self.creator_address` from ftns_ledger when caller passes None. Live-verified: upload without explicit param → DB shows operator wallet as creator. Unlocks future multi-wallet bench royalty test |
| **On-chain provenance info CLI** | `prsm provenance info <hash\|file>` | ✅ | 520 | Live-verified roundtrip: info on registered hash returns full record (hash, creator, royalty, registered_at, metadata); info on unregistered hash returns clean "NOT registered" |

---

## §5.2 — Compute Layer: SPRKs via Wasmtime

### Inference

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Inference quote | `POST /compute/inference/quote` | ✅ | 438 | Live-verified with `PRSM_INFERENCE_EXECUTOR=mock` |
| Submit inference (NONE tier) | `POST /compute/inference` | ✅ | 438 | Live E2E + F12 fix: ε=0.0 (not inf), receipt JSON-clean, verify passes |
| Submit inference (standard/high/maximum) | `POST /compute/inference` | ✅ | 438 | Live E2E: signed receipt verifies cleanly via sprint-433 verify path |
| Inference → receipt → verify chain | end-to-end | ✅ | 438 | First time the full §5.2+§7 chain works on single node |
| Streaming inference (endpoint UX) | `POST /compute/inference/stream` | ✅ | 445 | Live: returns 503 + clean refusal "wire a ParallaxScheduledExecutor" when streaming-capable executor not present (UX path verified) |
| Streaming inference (full E2E) | `POST /compute/inference/stream` | 🟢 | 3.x.8 | Audit-prep §7.4-§7.8 unit-pinned; full E2E needs ParallaxScheduledExecutor wiring |
| Privacy budget | `GET /privacy/budget` | ✅ | 445 | Live: returns {max_epsilon, total_spent, remaining, num_operations, spends} |
| Arbitration queue | `GET /content/arbitration/queue` | ✅ | 445 | Live: returns {pending, total} empty-state |
| Tensor parallel sharding | `POST /compute/inference/tensor_parallel/shard` | ✅ | 472 | Live: schema-defended 422 with required-field list `{shard_id, input_activations_b64}` — defense-in-depth against malformed shard dispatch |
| Pipeline stage setup | `POST /compute/inference/pipeline/stage` | ✅ | 472 | Live: schema-defended 422 with full required-field list `{job_id, round_id, stage_id, layer_indices, input_activations_b64}` |

### Forge (query orchestrator)

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Forge quote | `POST /compute/forge/quote` | ✅ | — | Verified live in dogfood arc |
| Submit forge query | `POST /compute/forge` | ⚠️ | — | Default-disabled; needs `PRSM_QUERY_ORCHESTRATOR_ENABLED=1` |
| Single-node forge E2E | `POST /compute/forge` | 🔬 | — | Requires content present; was blocked by F4 before sprint 428 |

### Compute jobs (general)

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Submit job | `POST /compute/submit` | ✅ | 469 | Live: returns `{job_id, status: pending, job_type, ftns_budget}`; locks escrow record. Invalid job_type → 400; payload >`PRSM_MAX_JOB_PAYLOAD_BYTES` (default 100KB) → 413 |
| Job status | `GET /compute/status/{id}` | ✅ | 469 | Live: returns `{job_id, escrow: {escrow_id, requester_id, amount_ftns, status, tx_lock, created_at, ...}}`. Unknown job_id → 404 with actionable detail (LRU-evicted or never-ran) |
| Job status stream | `GET /compute/status/{id}/stream` | ✅ | 469 | Live SSE: `event: status` emitted with escrow snapshot; de-duped by JSON equality (sprint B8 closure); terminal events on history/escrow terminal state or PRSM_STATUS_STREAM_TIMEOUT_SEC |
| Cancel job | `POST /compute/cancel/{id}` | ✅ | 469 | Live: returns `{job_id, history_cancelled, escrow_refunded, refund_amount_ftns}`; pending-job cancel refunds locked FTNS budget in full |
| List jobs | `GET /compute/jobs` | ✅ | 469 | Live: paginated envelope `{jobs, total, offset, limit}`; only surfaces history-recorded jobs (cancelled-pending jobs don't reach history) |
| Stale escrow cleanup | `POST /compute/cleanup-stale` | ✅ | 469 | Live: returns `{cleaned: N}`; empty-state returns 0 |
| Training jobs | `POST /compute/train` | ✅ | 469 | Live: clean 503 with actionable `set PRSM_FEDERATED_WORKER_PRIVKEY env` hint when worker privkey not configured. Full path gated by federated-worker enrollment |
| Compute stats | `GET /compute/stats` | ✅ | 469 | Live: full canonical schema `{resources, allocation, capacity, active_jobs, completed_jobs}` — CPU count/freq, mem, GPU, allocation pcts, concurrent_slots, active_jobs counter |
| Available models | `GET /compute/models` | ✅ | 450 | Live: returns 3 mock models (mock-llama-3-8b / mock-mistral-7b / mock-phi-3) registered by MockInferenceExecutor (sprint 438) |
| Receipts list (persistence across daemon restart) | `GET /compute/receipts` | ✅ | 447 | Live: sprint 438's mock-inference receipts persist; epsilon_spent=0.0 (F12 fix holds); full settler_signature intact |
| Receipt details | `GET /compute/receipt/{job_id}` | ✅ | 469 | Live: 404 with `No receipt for job_id='...'` for jobs without receipts (pending/cancelled). Receipt-bearing jobs covered by sprint 447 persistence test |

### Hardware classification

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Hardware benchmark | `prsm node benchmark` | ✅ | 450 | Live: Apple M4 → Tier T1, 4.60 TFLOPS FP32, thermal=sustained; full CPU/GPU/VRAM/RAM profile rendered |

---

## §5.3 — Economic Layer: FTNS

### Wallet / balance

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Local balance | `GET /balance` | ✅ | — | Verified live in dogfood arc |
| On-chain balance (mainnet) | `GET /balance/onchain` | ✅ | 464 | Live: queries Base mainnet FTNS contract `0x5276a37...` for derived wallet address (0x2Fd48D… in this test); returns balance_wei + balance_ftns + claimable + escrowed |
| On-chain balance (Sepolia testnet) | `GET /balance/onchain` | ✅ | 465 | Live: with `PRSM_NETWORK=testnet` + `BASE_SEPOLIA_RPC_URL` + `PRSM_ONCHAIN_FTNS=1` + `FTNS_TOKEN_ADDRESS` env vars, queries Base Sepolia FTNS contract `0x7F5f00FA…`; canonical_match True; returns 0 for empty wallet |
| Network switching (mainnet ↔ testnet) | `PRSM_NETWORK` env | ✅ | 465 | Live: daemon respects PRSM_NETWORK; chain_id 84532 + network "testnet" + rpc_host sepolia.base.org reported in `/info` |
| On-chain TX (Sepolia self-transfer) | Web3 EIP-1559 | ✅ | 466 | TX `0xa2b06bf59777…` block 41559508 status=1 — first PRSM wallet TX on Base Sepolia; gas 21k @ 6 Gwei = 0.000000126 ETH |
| On-chain TX (Sepolia → testnet foundation_safe) | Web3 EIP-1559 | ✅ | 466 | TX `0xead1f03055…` block 41559540 status=1 — wallet → `0xCCAc7b21…` (config/networks.py:144 testnet deployer); 2nd attribution TX |
| EmissionController halving epoch (constructor-immutable) | Sepolia contract call | ✅ | 466 | Live read: `EPOCH_DURATION_SECONDS() = 3600` (1 hour testnet, vs mainnet 4 years per Vision §11) — Sprint 358's INV-EC-1 claim empirically verified |
| Sepolia FTNS contract identity | Sepolia contract call | ✅ | 466 | Live read: `name()=PRSM Fungible Tokens for Node Support`, `symbol()=FTNS`, `decimals()=18`, `totalSupply()=100,002,060` |
| On-chain TX (Base **mainnet** self-transfer) | Web3 EIP-1559 | ✅ | 467 | TX `0xae65db7370fb…` block 46073119 status=1 — **first PRSM wallet TX on Base mainnet**; 21k gas @ 0.006 Gwei = $0.0004 USD |
| Mainnet FTNS contract identity | mainnet contract call | ✅ | 467 | Live read: name="PRSM Fungible Tokens for Node Support", symbol=FTNS, decimals=18, totalSupply=100,000,000 (vs testnet's 100,002,060 — different deploys) |
| INV-EC-1 mainnet (4-year epoch) | `/admin/formal-verification/check?contract=emission_controller` | ✅ | 467 | Live: EPOCH_DURATION_SECONDS=126,144,000 (4 years exactly); INV-EC-1 status: pass. Companion to sprint 466's testnet 3600 (1 hour) verification |
| Transaction history | `GET /transactions` | ✅ | — | Verified live (sprint 198 bounds-validated `limit`) |
| Wallet spend (30d summary) | `GET /wallet/spend` | ✅ | 464 | Live with FTNS_WALLET_PRIVATE_KEY: returns {address, days, total_spent_ftns, escrows_count} for derived wallet |
| Wallet escrows | `GET /wallet/escrows` | ✅ | 464 | Live with wallet: paginated empty-state for the wallet address |
| `prsm wallet info` CLI | CLI | ✅ | 464 | Live with FTNS_WALLET_PRIVATE_KEY: shows network (testnet vs mainnet), address, explorer URL, balance, foundation-config warnings |
| `prsm node earnings` CLI (with wallet) | CLI | ✅ | 464 | Live: operator address surfaced; Royalty/Heartbeat/Distribution categories ready to wire |
| Transfer (gasless via paymaster) | `POST /wallet/transfer/gasless` | 🟢 | Phase 5 | Endpoint exists; live activation external-gated |
| WaaS provision | `POST /wallet/waas/provision` | 🟢 | Phase 5 | Endpoint exists; requires CDP credentials |
| Paymaster status | `GET /wallet/paymaster/status` | 🟢 | Phase 5 | Endpoint exists |
| **On-chain transfer (real ERC-20)** | `POST /wallet/transfer/onchain` | ✅ | 498 | **F38 fix**: endpoint shipped + 2 TX live on Base mainnet (TX-1 self-transfer 0xa49dd80b…, TX-2 0x1b3b1f5e… 1 FTNS to second wallet). Real ERC-20 transfer signed by `FTNS_WALLET_PRIVATE_KEY` |
| `prsm ftns transfer-onchain` CLI | CLI | ✅ | 499 | Live-verified with 3 mainnet self-transfers via CLI: 0x596ccfdc…, 0x39d1a510…, 0x77c82fce… |
| On-chain TX history | `GET /wallet/transactions/onchain` | ✅ | 500 | Live: persistent SQLite-backed list with full schema (tx_hash, status, block, addrs, amount, created_at, job_id, scope) |
| On-chain TX history CLI | `prsm ftns history --onchain` | ✅ | 500 | Live: Rich table render of persisted TX |
| On-chain TX persistence | SQLite `onchain_transactions` table | ✅ | 501 | **Live-verified roundtrip**: broadcast 2 TX → kill daemon → restart → both TX present with full fidelity |
| On-chain TX stats | `GET /wallet/transactions/onchain/stats` | ✅ | 505 | Live: aggregates {count, confirmed/pending/rejected, total_ftns_sent, first/last_tx_at} |
| On-chain TX stats CLI | `prsm ftns history --onchain --stats` | ✅ | 505 | Live: compact summary with color-coded counts |
| Operator gas status | `GET /wallet/gas-status` | ✅ | 502 | Live: thresholds low<0.0005 / critical<0.0001 ETH, status=ok/low/critical/unavailable. Operator wallet 0.000497 ETH → status=low |
| Operator gas status CLI | `prsm wallet gas-status` | ✅ | 502 | Live: color-coded status + actionable warning by severity |
| Operator gas in /health/detailed | `subsystems.operator_gas` | ✅ | 503 | Live: monitoring tools polling /health/detailed get gas signal for free |
| Daemon-startup gas log | log push signal | ✅ | 504 | Live-verified: "Operator gas low: 0.0004974813 ETH on 0x4acdE458…" in startup log |
| Continuous gas monitor | `GasStatusMonitor` background task | ✅ | 506 | Live: periodic sampler logs ONLY on transitions (ok↔low↔critical) |
| Gas transition webhook | `POST PRSM_WEBHOOK_URL` event=gas.transition | ✅ | 507 | Infrastructure live-verified; transition firing pin-tested |
| `prsm wallet info` shows ETH balance | CLI | ✅ | 508 | Live-verified Base mainnet: shows ETH + gas status [LOW] + warning alongside FTNS + claimable |
| Sepolia parity (FTNS-side surfaces) | all sprint 498-508 endpoints | ✅ | 509 | Live-verified: ftns_ledger canonical_match=True against Sepolia FTNS `0x7F5f00FA…`; operator_gas, /wallet/gas-status, CLI all work identically on testnet |
| Cross-network TX history isolation | `chain_id` column on onchain_transactions | ✅ | 510 | **F39 fix**: Sepolia daemon now correctly hides mainnet TX. Live-verified: mainnet TX 0xaf1b8b35… tagged chain_id=8453, Sepolia view returns count=0 |
| Pending-TX reconciliation | `_reconcile_pending_transactions()` at init | ✅ | 511 | **F40 fix**: pending TX from interrupted broadcasts now get final status from chain receipt on next daemon start |
| Inbound FTNS detection | `GET /wallet/transactions/onchain/inbound` | ✅ | 512 | Live-verified Base mainnet: detected Foundation Safe → 2 FTNS funding at block 46159960 (tx 0xe80410f9…) + 4 self-transfers |
| Inbound CLI | `prsm ftns history --onchain --inbound` | ✅ | 513 | Live: Rich table render of inbound transfers |
| Background inbound poller + webhook | `InboundMonitor` task + event=ftns.inbound | ✅ | 514 | **Live-verified end-to-end**: webhook listener received POST with full payload (event/recipient/from_address/tx_hash/block_number/amount_ftns/timestamp) within 10s of broadcast |
| Inbound stats | `GET /wallet/transactions/onchain/inbound/stats` | ✅ | 515 | Live: 9 inbound, 2.000008 FTNS total, first/last block 46159960/46165077 |
| Inbound monitor in /health/detailed | `subsystems.inbound_monitor` | ✅ | 515 | Live: status=ok, last_scanned_block tracking |
| Inbound stats CLI | `prsm ftns history --onchain --inbound --stats` | ✅ | 516 | Live: symmetric with outbound --stats |

### Staking

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Stake | `POST /staking/stake` | ✅ | 432 | Live-verified: 1000 FTNS staked; total_staked reflects new amount |
| Unstake request | `POST /staking/unstake` | ✅ | 470 | Live: returns `{request_id, stake_id, amount, requested_at, available_at, status: pending}` with 7-day cooldown enforced via `available_at`. Stake status transitions `active → unstaking`; `total_staked` drops; `pending_unstake_requests` populated |
| Stake status | `GET /staking/status` | ✅ | 432 | Live-verified end-to-end with active stake |
| Claim rewards | `POST /staking/claim-rewards` | ✅ | 432 | F11 fixed: tz-aware datetime subtraction now works (was 500 on every claim) |
| Withdraw unstaked | `POST /staking/withdraw/{id}` | ⚠️ | 470 | Live: schema-pass — cancelled request → 400 "Request status invalid: cancelled"; unknown UUID → 404 "Unstake request X not found"; malformed UUID → 400. Happy-path gated by 7-day cooldown (`available_at` invariant, not bypassable without DB clock manipulation) |
| Cancel unstake | `POST /staking/cancel-unstake/{id}` | ✅ | 470 | Live: `{request_id, cancelled: true, reason: null}`. Stake status returns `unstaking → active`; `total_staked` restored; `pending_unstake_requests` cleared. Unknown UUID → 404 |
| Single-user stake → claim E2E | (multi-step) | ✅ | 470 | Live-verified end-to-end via sprint 432 (stake + claim) + sprint 470 (unstake + cancel-unstake lifecycle). Full ledger flow operationally sound |

### Settlement

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Settlement stats | `GET /settlement/stats` | ✅ | 444 | Live: returns canonical schema (empty-state correct) |
| Pending settlements | `GET /settlement/pending` | ✅ | 470 | Live: empty-state `{pending: [], count: 0}` |
| Flush batch | `POST /settlement/flush` | ✅ | 470 | Live: full canonical schema `{settled_count, total_amount, net_transfers, tx_hashes: [], errors: [], duration_seconds}` — clean empty-state on fresh node |
| Settlement history | `GET /settlement/history` | ✅ | 470 | Live: empty-state `{history: [], count: 0}` |
| Settler registry (list-active surface) | `GET /settler/list/active` | ✅ | 444, 470 | Sprint 444 empty-state; sprint 470 with active settler — returns `[{settler_id, address, bond_amount, total_settled}]` |
| Settler registry (register / unbond / sign batch) | `/settler/...` | ✅ | 470 | Live full lifecycle: POST /settler/register (min bond 10000 FTNS enforced — Vision §11 invariant, `bond_amount<min` → 400); GET /settler/{id} (status, can_settle, total_settled, slashed_amount); POST /settler/unbond (30-day cooldown, `unbond_at` set, list/active filters); POST /settler/batch/sign + GET /settler/batch/pending; GET /settler/ledger/export (integrity_hash for chain-of-custody); POST /settler/slash/propose schema-validated (settler_id+slash_amount+reason+proposer_id) |

### Phase 5 fiat surface (commission-ready, external-gated)

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Onramp quote | `POST /wallet/onramp/quote` | ✅ | 451 | Live (schema: usd_amount+destination_address): returns full quote with coinbase-cdp onramp_route + aerodrome swap_route + KYC + tier-limit fields |
| Offramp quote | `POST /wallet/offramp/quote` | ✅ | 451 | Live: clean operator-readable balance breakdown when destination has 0 balance ("requested $X, available $Y...") |
| Pool quote (Aerodrome) | `GET /wallet/pool/quote` | 🟢 | 276-286 | Read-only quoter; live pool external-gated |
| Pool state | `GET /wallet/pool/state` | ✅ | 451 | Live: reports `NOT_CONFIGURED` with operator-actionable note pointing to AERODROME_USDC_FTNS_POOL_ADDRESS env var + seeding ceremony date |
| Fiat compliance audit ring (auto-record) | `GET /admin/fiat-compliance/summary` | ✅ | 451 | Live: my single onramp-quote call auto-recorded as `{onramp_quote: {count: 1, total_usd: 100.0}}` — Vision §11's AUSTRAC/FinCEN/IRS-ready claim attested |
| KYC initiate | `POST /wallet/kyc/initiate` | ✅ | 452 | Live (schema: user_id+email+tier): returns clean PENDING_COMMISSION envelope with vendor=null in dev env (per sprint-285 commissioning pattern) |
| KYC status (lookup) | `GET /wallet/kyc/status` | ✅ | 452 | Live: `{commissioned: false, vendor: null, supported_vendors: ["persona","onfido","plaid"], record_count: N}` |
| KYC webhook | `POST /wallet/kyc/webhook/{vendor}` | 🟢 | Phase 5 | HMAC-SHA256 + replay protection — requires real vendor signed payload to live-verify |
| Fiat-surface health | `GET /admin/fiat-surface/health` | ✅ | 285/422 | `check_fiat_surface_health()` live-verified |
| Fiat-readiness CLI | `prsm node fiat-readiness` | ✅ | 422, 452 | Live: text → "✓ Phase 5 fiat surface ready — OK (no findings)"; JSON → `{overall_status: "ok", findings: []}` |
| Activation runbook | `docs/operations/phase-5-fiat-surface-activation-runbook.md` | ✅ | 421 | Pinned by 11 source-truth-parity tests |

### On-chain royalty distribution

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| RoyaltyDistributor v2 (atomic 70/25/5) | mainnet `0x3E82…D6c2` | ✅ | A-08 | Mainnet ceremony 2026-05-09 |
| On-chain content-access royalty leg | env-gated by `PRSM_ONCHAIN_CONTENT_ROYALTY_ENABLED=1` | 🟢 | 243-261 | 19-sprint arc; live activation requires creator addresses on uploaded content |
| Royalty dispatch summary | `GET /admin/royalty-dispatch-summary` | ✅ | 444 | Live: returns canonical schema (total, status_counts, total_sent_wei, by_allocation_mode, earliest_ts, latest_ts) |
| Royalty dispatch history | `GET /admin/royalty-dispatch-history` | ✅ | 444 | Live: paginated `{entries, total, offset, limit}` envelope |
| Claim royalty | `POST /wallet/royalty/claim` + `prsm node claim-royalty` | 🟢 | — | Multi-step ledger flow untested E2E |

---

## §5.4 — Provenance Layer: On-Chain Royalty Distribution

| Contract | Mainnet Address | Status | Notes |
|----------|-----------------|--------|-------|
| FTNSTokenSimple | `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` | ✅ | 1B max; Foundation Safe sole admin |
| ProvenanceRegistry V1 | `0xdF47…9915` | ✅ | Basescan-verified |
| ProvenanceRegistry V2 | `0xe0cedDA354...` | ✅ | PRSM-CR-2026-05-06-2 ratified |
| RoyaltyDistributor | `0x3E82…D6c2` | ✅ | A-08 ceremony 2026-05-09 |
| BSR + EscrowPool + StakeBond + Ed25519Verifier | (bundle, all verified) | ✅ | Sole-owned by Foundation Safe |
| EmissionController | (Phase 8) | ✅ | EPOCH_DURATION_SECONDS chain-8453 4yr enforcement |
| CompensationDistributor | (Phase 8) | ✅ | |
| StorageSlashing + KeyDistribution | (Phase 7-storage) | ✅ | Sole-owned by Foundation Safe |
| Formal-invariants harness (7 contracts, 20 CRITICAL invariants) | runtime probe | ✅ | Sprint 357-364 |
| Halmos symbolic-execution lane (5 specs, 28 proofs, 16/20 invariants) | `contracts/symbolic-proofs/` | ✅ | Sprint 360-364 |
| `/admin/formal-verification/*` endpoint family | REST | ✅ | Sprint 364 |
| `prsm_formal_verification` MCP | MCP | ✅ | Sprint 364 |

---

## §7 — Private Inference: verifiable-claim infrastructure

### Receipt + verification

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| `InferenceReceipt` schema (with activation_noise_trace + topology_assignment) | dataclass | ✅ | 297 | Pure-additive, byte-identical pre-arc |
| Receipt verification | `POST /compute/receipt/verify` | ✅ | 292/433 | Sprint 433 live-verified: honest signs ok, tampered fails. ed25519 over signing_payload |
| Receipt tamper-detection (signature) | `POST /compute/receipt/verify` | ✅ | 433 | Live tampered `epsilon_spent` → `signature_valid: false`, reason="signature failed cryptographic verification" |
| Receipt verify MCP | `prsm_verify_inference_privacy` | ✅ | 292 | |
| End-to-end topology pathway (RpcChainExecutor → TopologyAwareChainExecutor → ParallaxScheduledExecutor → signed receipt) | composition | ✅ | 415 | 91 cross-suite green |
| End-to-end DP pathway (ActivationDPAwareChainExecutor) | composition | ✅ | 419 | Mirrors topology side |
| `make_rpc_chain_executor` default wraps topology | factory | ✅ | 417 | `wrap_topology_aware=True` default |
| `RpcChainExecutor.execute_chain` post_stage_hook | hook | ✅ | 418 | DP integration point |

### Attestation backends

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Intel ASP (SGX v3 + TDX v4) structural parse | `IntelASPBackend` | ✅ | 293, 448 | Live: parses v3 structural probe → vendor="intel-sgx"; vendor_data populated with version/mrenclave_hex/mrsigner_hex/structural_only |
| AMD KDS (SEV-SNP v2) structural parse | `AMDKDSBackend` | ✅ | 294, 448 | Live: parses v2 structural probe → vendor="amd-sev-snp"; vendor_data populated with version/guest_svn/measurement_hex/report_data_hex/chip_id_hex |
| Attestation registry dispatcher | `verify_attestation()` | ✅ | 448 | Live: routes Intel v3 probe → intel-sgx; AMD v2 probe → amd-sev-snp (correct vendor-detection from quote-version bytes) |
| Real cryptographic signing-chain verification | (deferred) | ⏸️ | — | Structural-only today; `vendor_verified=True` requires real DCAP/KDS keys |
| Apple SEP backend | (deferred) | ⏸️ | — | If iOS-side compute joins the supply tier mix |

### Privacy primitives

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Per-stage Gaussian noise (basic composition) | `ActivationDPInjector` | ✅ | 295 | Defends activation-inversion |
| Topology rotation (uniform/beacon/anti-repeat) | `TopologyRotationPolicy` | ✅ | 296 | `stable_hash()` enables replay verification |
| Privacy budget tracking | `/privacy/budget` + persistent store | ✅ | 445 | Live: returns canonical schema with per-tier budget state (sprint 445); audit-prep §2.4 unit-pinned |

### Enterprise Confidentiality Mode

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Recipient encryption (X25519 + XChaCha20) | `POST /content/upload` w/ recipients | ✅ | 430 | Live byte-identical roundtrip; sprint 430 |
| Recipient manifest read | `GET /content/recipient-manifest/{cid}` | ✅ | 304, 472 | Live (sprint 472): 422 schema-defended `"not an encrypted recipient bundle"` on Tier A CID; full Tier B/C recipient-bundle parsing covered by sprint 430's E2E roundtrip |
| Threshold encryption | (multi-endpoint) | 🟢 | — | Math green |
| `prsm_enterprise_recipient` MCP | MCP | 🟢 | 304 | |

---

## §13 — Operator surfaces (the operator trifecta)

Every operator-facing feature should have REST + CLI + MCP coverage
(the "trifecta"). Status here is the **weakest** of the three.

### Node lifecycle + health

| Feature | REST | CLI | MCP | Status |
|---------|------|-----|-----|--------|
| Node status | `/status` | `prsm node status` | `prsm_node_status` | ✅ |
| Node health (detailed, 14 subsystems) | `/health/detailed` | — | `prsm_node_health` | ✅ Sprint 342-345, 447 (live: ftns_ledger/payment_escrow/job_history/receipt_store all status:ok; canonical_match:true on wired contract address) |
| Node info | `/info` | `prsm node info` | `prsm_info` | ✅ |
| Node peers | `/peers` | `prsm node peers` | `prsm_peers` | ✅ |
| Bootstrap status | `/bootstrap/status` | — | `prsm_bootstrap_status` | ✅ Sprint 447 (live: connected to wss://bootstrap1.prsm-network.com:8765, 28/16 reconnect attempts/successes, multi-fallback US/EU/APAC enabled) |
| Bootstrap test (probe canonical fleet) | — | `prsm node bootstrap-test` | `prsm_bootstrap_test` | ✅ Sprint 385/387 |
| Bootstrap server status | `/admin/bootstrap-server/status` | `prsm bootstrap-server status` | `prsm_bootstrap_server_status` | ✅ Sprint 388-396, 454 (live-probed `bootstrap1.prsm-network.com:8000` → ✓ healthy, 84036s uptime, 1 active conn, 63 total, region nyc3, version 1.0.0) |
| Metrics (Prometheus) | `/metrics` | — | `prsm_metrics_summary` | ✅ Sprint 454 (live: full Prometheus exposition format with 9+ gauges — pending_escrow_count, total_locked_ftns, job_history_size, receipt_store_size, royalty_dispatch_ring_size, escrow_cleanup_task_running=1, build_info{version="1.7.0"}, slash/heartbeat/distribution log counts) |
| Resources (read/write) | `GET/PUT /node/resources` | — | `prsm_node_resources` | ✅ Sprint 450 (GET live: cpu/mem/storage/gpu allocation pcts + bandwidth/active-hours/effective-resources fully reported) |

### Earnings + ledger

| Feature | REST | CLI | MCP | Status |
|---------|------|-----|-----|--------|
| Earnings summary | `/admin/earnings-summary` | `prsm node earnings` | `prsm_earnings_summary` | ✅ Sprint 446 (CLI live: actionable empty-state when PRSM_OPERATOR_ADDRESS unset) |
| Slash history | `/admin/slash-history` | `prsm node slash-history` | `prsm_slash_history` | ✅ Sprint 455 (live: paginated {entries, total, offset, limit} empty-state) |
| Heartbeats | `/admin/heartbeat-history` | `prsm node heartbeats` | `prsm_heartbeat_history` | ✅ Sprint 446, 455 (CLI live: "No entries" empty-state) |
| Distributions | `/admin/distribution-history` | `prsm node distributions` | `prsm_distribution_history` | ✅ Sprint 455 (live: paginated envelope) |
| Webhooks | `/admin/webhook-history` | `prsm node webhooks` | `prsm_webhook_history` | ✅ Sprint 446 (CLI live: "set PRSM_WEBHOOK_URL to enable" actionable empty-state) |
| Trigger heartbeat | `/admin/heartbeat/trigger` | `prsm node trigger-heartbeat` | `prsm_heartbeat_trigger` | ✅ |
| Trigger distribution | `/admin/distribution/trigger` | `prsm node trigger-distribution` | `prsm_distribution_trigger` | ✅ |
| Claim royalty | `/wallet/royalty/claim` | `prsm node claim-royalty` | `prsm_royalty_claim` | ✅ Sprint 519 (live-verified Base mainnet: DRY-RUN returns "Claimable: 0.000000 FTNS (0 wei)" + hint; `--execute` correctly short-circuits with "Nothing to claim — claimable balance is 0" — no wasted gas. Real-claim happy-path with claimable>0 requires multi-wallet consumer bench to accumulate royalties) |
| Audit summary | `/audit/summary` | — | `prsm_audit_summary` | ✅ Sprint 471 (live: 24 sprint-469+470 probe calls auto-recorded with full schema — `{total, status_buckets: {2xx,4xx,5xx}, method_buckets, top_paths}`; auto-record on every request) |
| Audit recent | `/audit/recent` | — | `prsm_audit_recent` | ✅ Sprint 471 (live: paginated `{entries, total, offset, limit}` with full per-call envelope — timestamp, method, path, requester, status_code, request_id) |

### Content + marketplace

| Feature | REST | CLI | MCP | Status |
|---------|------|-----|-----|--------|
| Content filter | `/admin/content-filter` | — | `prsm_content_filter` | ✅ Sprint 269-274, 471 (live full lifecycle: GET → empty; POST /admin/content-filter/tags → `{added,total}`; DELETE /tags/{tag} → `{removed,total}`) |
| Takedown notices | `/admin/takedown-notice` | — | `prsm_takedown_notices` | ✅ Sprint 439 (live chain E2E in §14 table) |
| Notice → filter bridge | `/admin/content-filter/from-notice/{id}` | — | (via `prsm_takedown_notices`) | ✅ Sprint 439 (live E2E) |
| Creator reputation | `/marketplace/creator-reputation/{id}` | — | `prsm_creator_reputation` | ✅ Sprint 287-291, 471 (live: unknown creator → clean default schema `{known:false, score:0.5, tier:new, total_accesses:0, distinct_purchasers:0, repeat_purchaser_count:0}`) |
| Creator stake | `/marketplace/creator-stake/{id}` | — | `prsm_creator_stake` | ✅ Sprint 442 (live in §14 row) |
| Provider reputations | `/marketplace/reputation` | — | `prsm_marketplace_reputation` | ✅ Sprint 471 (live: paginated `{providers, count, limit:100}` empty-state) |
| My content | `/content/mine` | `prsm content mine` | `prsm_my_content` | ✅ Sprint 449, 471 (live: paginated `{entries, total, offset, limit:50}` envelope) |

### Phase 5 fiat operator surfaces

| Feature | REST | CLI | MCP | Status |
|---------|------|-----|-----|--------|
| Fiat surface health | `/admin/fiat-surface/health` | `prsm node fiat-readiness` | `prsm_fiat_surface_health` | ✅ |
| Fiat compliance summary | `/admin/fiat-compliance/summary` | — | `prsm_fiat_compliance` | ✅ Sprint 451, 471 (live: `{by_kind, total_entries}`; sprint-451 attested auto-record-on-onramp-quote) |
| KYC status | `/wallet/kyc/status` | — | `prsm_kyc` | ✅ Sprint 452, 471 (live: `{commissioned:false, vendor:null, supported_vendors:[persona,onfido,plaid], record_count}`) |

### Incident + upgrade + insurance

| Feature | REST | CLI | MCP | Status |
|---------|------|-----|-----|--------|
| Incident open / advance / log event | `/admin/incident/...` | `prsm node incident list/details/playbook` (read-only) | `prsm_incident` | ✅ Sprint 434, 476 (sprint 434: CLI trifecta read-only; sprint 476: **full lifecycle live E2E** — POST /open returns full envelope with timeline; /event appends to timeline preserving phase; /advance transitions detected→triaged (s2 example, timeline grows); /recommendations returns context-specific action items; /comms-template returns ready-to-paste internal-update text; /playbook surfaces full s0-s3 × phase decision tree with Vision §14 PAUSE NOW / Foundation Safe 15min-target / forensics-partner-engage guidance) |
| Insurance fund status | `/admin/insurance-fund/status` | — | `prsm_insurance_fund` | ✅ Sprint 455 (live: treasury_address=Foundation Safe 0x91b0e6F8…, target_bps=500 reserve target, commissioned=false in dev env) |
| Emergency pause status (mainnet contracts) | `/admin/emergency-pause/status` | — | `prsm_emergency_pause` | ✅ Sprint 455 (live: ftns_token + royalty_distributor + BSR + EscrowPool + StakeBond + Ed25519Verifier + StorageSlashing + KeyDistribution + EmissionController all reported with paused state + commissioned flag against chain_id=8453 Base mainnet) |
| Upgrade proposal | `/admin/upgrade/...` | — | `prsm_upgrade` | ✅ Sprint 471, 475 (full lifecycle live: POST /propose → proposed; GET /{id} round-trips; /update advances proposed→reviewed→safe_uploaded with safe_tx_hash; /compose-upgrade returns Safe-uploadable tx with `upgradeToAndCall` calldata, chain_id=8453, explicit storage-layout warning + 4-step instructions; /compose-rollback enforces `status==executed` invariant — Vision §14 "composer produces tx, doesn't execute" verified) |
| TEE policy | `/admin/tee-policy/evaluate` | — | `prsm_tee_policy` | ✅ Sprint 436, 471 (live: `evaluate` → 422 with clear `policy` required-field; `/admin/tee-policy/node-status` → `{effective_tier, vendor, vendor_verified, diagnostic}`) |
| Vulnerability disclosure | `/admin/disclosure/...` | — | `prsm_disclosure` | ✅ Sprint 471, 475 (full lifecycle live: POST /submit → received; GET /{id} round-trips with details_b64 (privacy-preserving); /update advances received→triaged→awarded with triage_notes + payout_ftns; /compose-payout enforces `status==AWARDED` state-machine invariant; on awarded → returns Safe-uploadable bug-bounty payout tx (FTNS ERC-20 `transfer(recipient, 1000e18)` calldata, chain_id=8453, warning + 4-step Safe UI instructions); /record-payout-tx closes audit trail with on-chain tx hash) |

---

## §14 — Risk mitigations (validation, defense, formal verification)

### Content moderation (operator-side enforcement)

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Operator content filter (CID blocklist) | `/admin/content-filter/cids` | ✅ | 439 | Live E2E: POST cids → 451 on retrieve verified |
| Operator content filter (tag blocklist) | `/admin/content-filter/tags` | ✅ | 471 | Live full lifecycle: POST `{tags:[T]}` → `{added:1, total:1}`; GET /admin/content-filter shows tag in blocked_model_tags; DELETE /tags/{T} → `{removed, total:0}` |
| Foundation takedown notice intake (info-only) | `/admin/takedown-notice` | ✅ | 439 | Live-verified: target_cid+sender+jurisdiction+basis required (§14 attribution invariant) |
| Notice → filter bridge | `/admin/content-filter/from-notice/{id}` | ✅ | 439 | Live E2E: notice → bridge → CID auto-added; notice status flips to "acknowledged" |
| Notice lifecycle status transitions | `/admin/takedown-notices/{id}/status` | ✅ | 269-274, 477 | Live full lifecycle: POST /takedown-notice → status=received; POST /{id}/status → transitions received → acknowledged → disputed; GET /{id} round-trips updated state; invalid status value → 422 with canonical list `['acknowledged','disputed','expired','received']` |

### Data quality + Sybil resistance

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Creator reputation tracker (lookup) | `/marketplace/creator-reputation/{id}` | ✅ | 440 | Live: returns clean default (known:false, score:0.5, tier:"new") for unknown creators |
| Creator reputation tracker (auto-record on retrieve) | hook in `/content/retrieve` | 🟢 | 287-291 | Wired correctly; live update gated by operator wallet config (`FTNS_WALLET_PRIVATE_KEY`) — dev env can't trigger |
| Tier classification (new/low/medium/high) | reputation tier auto-records on retrieve | 🟢 | 287-291 | Same wallet-gate as above |
| Search filter by tier | `GET /content/search?min_tier=...&exclude_new=...` | ✅ | 440 | Live: query params accepted cleanly; tier-filter codepath active |
| Creator stake lookup | `GET /marketplace/creator-stake/{id}` | ✅ | 442 | Live: returns clean schema (balance_wei, high_tier_eligible, min_high_tier_stake_wei, commissioned); unknown creator defaults to balance_wei=0 + high_tier_eligible=false |
| Creator stake gate (commissioning) | on-chain `StakeBond` + `commissioned` flag | 🟢 | 287-291 | Lookup surface ready; live stake/slash flow gated by on-chain wallet + StakeBond contract address (dev env: `commissioned: false`) |
| Content fingerprint registry (first-upload registers) | `POST /content/upload` hook | ✅ | 441 | Live E2E: content_hash registered with canonical_creator |
| Content fingerprint registry (duplicate detection) | `POST /content/upload` re-upload | ✅ | 441 | Live E2E: re-upload with different creator surfaces `duplicate_of_creator`; first-creator-wins invariant verified |
| Fingerprint registry (dedup-attempt counter) | `GET /marketplace/fingerprint/{hash}` | ✅ | 441 | `duplicate_attempt_count` increments on each re-upload attempt |

### Formal verification

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Runtime invariants probe (7 contracts, 20 invariants) | `/admin/formal-verification/check?contract=X` | ✅ | 302-359, 443 | Live: INV-RD-3 (Foundation Safe owns RoyaltyDistributor v2) PASS against mainnet; harness fail-soft on missing-selector skips |
| Invariants registry (full list) | `/admin/formal-verification/invariants` | ✅ | 443 | Live: returns all 20 invariants with severity / spec_text / kind / selector / expected |
| Halmos symbolic-execution lane (5 specs, 28 proofs) | `/admin/formal-verification/symbolic` | ✅ | 360-364, 443 | Live: endpoint lists 5 specs with mirrors_runtime_contract + runtime_invariants linkage |
| `@pytest.mark.requires_halmos` CI marker | conftest | ✅ | 366 | |
| Halmos streaming-inference extension | `SpeculationRollbackMathSpec` | ✅ | 367 | First off-chain Python algorithm |
| Halmos H1 bounded iterator | `ChunkStreamingBoundsSpec` | ✅ | 368 | |
| Halmos M2 padding | `M2ResponseSizePaddingSpec` | ✅ | 369 | Wire-observer indistinguishability |

### Validation / DoS hardening

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| JSON Inf/NaN body-guard middleware | api+dashboard | 🟢 | 197-208 | |
| Float-field upper bounds | request models | 🟢 | 197-208 | |
| Payload caps (upload size, replicas, parent_cids, shard) | `/content/upload` | 🟢 | 197-208 | |
| Retrieve timeout bound | `GET /content/retrieve` | 🟢 | 203 | `PRSM_MAX_RETRIEVE_TIMEOUT_SEC` |
| `/transactions` limit bound | query param | ✅ | 198 | b8d70091 |

### Multi-bootstrap fallback

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| `Libp2pDiscovery` accepts fallback nodes | `BootstrapClient` | ✅ | 375 | |
| `BootstrapClient.active_url` records winning candidate | `/bootstrap/status.active_url` | ✅ | 375 | |
| Canonical bootstrap fleet (US + EU + APAC) | DNS | ⚠️ | 384-385 | US (DO) + APAC (AWS Tokyo) live; EU (AWS Frankfurt) cloud-init deploy pending |

---

## Test coverage matrix — depth dimensions

The section-by-section tables above answer **"is this surface
verified end-to-end?"**. They do NOT answer **"verified under
what conditions?"**. Sprints 469-485 surfaced this gap: most ✅
rows mean "happy path returns the right shape", not "survives
concurrent callers" or "delivers correct bytes at scale" or
"degrades gracefully when the disk fills up".

This matrix rates each subject area across the canonical
depth dimensions. A green cell means the dimension was
exercised live (cite the sprint). Red means we know it's
untested. Yellow means partial — some scenarios but not all.

### Depth dimensions

| Symbol | Dimension | Definition |
|--------|-----------|------------|
| **HP** | Happy path | Basic call with valid input returns expected shape + correct value |
| **EP** | Error path | Bad / malformed / out-of-bounds input returns clean error + actionable detail |
| **CC** | Concurrency | N simultaneous callers don't race / deadlock / corrupt shared state |
| **SC** | Scale | Realistic data sizes (Vision-claimed MB/GB for content, M+ records for indices) |
| **AD** | Adversarial | Resists deliberate attack: signature forgery, content-filter evasion, header overflow, replay |
| **FM** | Failure modes | Graceful degradation on disk full / peer crash / partial write / DB unreachable |
| **LR** | Long-running | Stable over 24h+: no memory leak, no FD leak, no slow degradation |
| **DR** | Disaster recovery | Can restore from data loss: DB drop, staging dir deleted, identity file lost |
| **OC** | Real on-chain | Actually executed against mainnet (Base 8453) — not just schema-pin or testnet |
| **XF** | Cross-feature | Interacts correctly with adjacent features (forge + staking, upload + on-chain royalty) |

### Coverage matrix

| Subject | HP | EP | CC | SC | AD | FM | LR | DR | OC | XF |
|---------|----|----|----|----|----|----|----|----|----|----|
| **§4 user workflow (Tier A)** | ✅ 484 | ✅ 484 | ❌ | ❌ | ❌ | ⚠️ 484 | ❌ | ✅ 485 | ❌ | ⚠️ 472 |
| **§4 user workflow (Tier B/C)** | ✅ 430 | ⚠️ 472 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **§5.1 BitTorrent layer** | ✅ 472 | ✅ 472 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ 485 | N/A | ❌ |
| **§5.1 Content index (fingerprint dedup)** | ✅ 441 | ✅ 471 | ❌ | ❌ | ⚠️ 441 | ❌ | ❌ | ❌ | N/A | ❌ |
| **§5.2 Inference (mock executor)** | ✅ 438 | ✅ 469 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | N/A | ❌ |
| **§5.2 Inference (streaming SSE)** | ✅ 469 | ⚠️ 469 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | N/A | ❌ |
| **§5.2 Forge query** | ⚠️ 431 | ✅ 469 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | N/A | ❌ |
| **§5.2 Compute jobs lifecycle** | ✅ 469 | ✅ 469 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | N/A | ❌ |
| **§5.2 Training (compute/train)** | ⚠️ 469 | ✅ 469 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | N/A | ❌ |
| **§5.3 Staking lifecycle** | ✅ 432, 470 | ✅ 470 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **§5.3 Unstake → withdraw (cooldown)** | ⚠️ 470 | ✅ 470 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **§5.3 Settlement (flush + history)** | ✅ 470 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **§5.3 Settler registry lifecycle** | ✅ 470 | ✅ 470 | ❌ | ❌ | ⚠️ 470 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **§5.3 Royalty distribution** | ❌ | ⚠️ 471 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **§5.4 Provenance V2 (on-chain)** | ✅ 443 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ 467 | ❌ |
| **§5.4 Formal verification (INV-*)** | ✅ 443, 466, 467 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | ✅ 443, 467 | N/A |
| **§7 Receipt sign + verify** | ✅ 433 | ✅ 433 | ❌ | ❌ | ✅ 433 | ❌ | ❌ | ⚠️ 447 | N/A | ❌ |
| **§7 Attestation backends (SGX/SEV)** | ✅ 448 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **§7 Privacy primitives (DP + topology)** | ✅ 415, 419 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | N/A | ❌ |
| **§7 Enterprise recipient encryption** | ✅ 430 | ✅ 472 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | N/A | ❌ |
| **§8 Wallet (FTNS local ledger)** | ✅ 432 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | N/A | ❌ |
| **§8 Wallet (on-chain balance)** | ✅ 464, 466, 467 | ❌ | ❌ | N/A | ❌ | ❌ | ❌ | ❌ | ✅ 467 | ❌ |
| **§8 Phase 5 fiat onramp/offramp** | ✅ 451 | ⚠️ 451 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 🔗 | ❌ |
| **§8 KYC handshake** | ✅ 452 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 🔗 | ❌ |
| **§11/12 Governance (proposals/voting)** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **§13 Operator CLI trifecta** | ✅ 434-437, 446 | ⚠️ 446 | N/A | N/A | N/A | ❌ | ❌ | ❌ | N/A | N/A |
| **§13 Audit ring (auto-record)** | ✅ 471 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | N/A | ❌ |
| **§13 Incident response lifecycle** | ✅ 476 | ❌ | ❌ | N/A | ❌ | ❌ | ❌ | ❌ | N/A | ❌ |
| **§13 Upgrade proposal flow** | ✅ 475 | ✅ 475 | ❌ | N/A | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **§13 Disclosure intake + payout** | ✅ 475 | ✅ 475 | ❌ | N/A | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **§13 Emergency pause (9-contract)** | ✅ 455 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | ✅ 455 | N/A |
| **§14 Content moderation chain** | ✅ 439 | ✅ 471 | ❌ | ❌ | ⚠️ 439 | ❌ | ❌ | ❌ | N/A | ❌ |
| **§14 Takedown lifecycle** | ✅ 439, 477 | ✅ 477 | ❌ | N/A | ❌ | ❌ | ❌ | ❌ | N/A | ❌ |
| **§14 Anti-Sybil (creator stake)** | ✅ 442 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **§14 Corp capability (Ed25519)** | ✅ 477 | ✅ 477 | ❌ | ❌ | ⚠️ 477 | ❌ | ❌ | ❌ | N/A | ❌ |
| **§5.3 / §13 Daemon health surface** | ✅ 447, 482 | ✅ 473 | ❌ | ❌ | ❌ | ✅ 473 | ❌ | ❌ | N/A | ❌ |
| **§5.4 P2P discovery (peer join/leave)** | ✅ 456, 457 | N/A | ❌ | ❌ | ❌ | ✅ 457 | ❌ | ❌ | N/A | ❌ |
| **§5.4 P2P retrieve cross-host** | ❌ | N/A | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | N/A | ❌ |
| **Bootstrap reconnect (sentinel recovery)** | ✅ 474 | N/A | ❌ | N/A | ❌ | ✅ 474 | ❌ | ❌ | N/A | ❌ |
| **Dep audit (sprint 461-463 invariants)** | ✅ 461-463 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **Cross-restart persistence** | ✅ 485 | N/A | ❌ | ❌ | N/A | ⚠️ 480 | ❌ | ✅ 485 | N/A | ❌ |

Legend: ✅ live-tested (sprint cited), ⚠️ partial, ❌ untested,
N/A not applicable to this surface, 🔗 external-gated.

### Highest-priority untested cells

Ranked by **risk × likelihood × blast-radius**. These are the
gaps worth a dedicated sprint each rather than one more
endpoint sweep:

1. **CC — Concurrency across the board.** Zero surfaces have
   been tested under simultaneous callers. Locks in
   StakingManager, ContentUploader.uploaded_content,
   SettlerRegistry, payment_escrow are all suspect. Risk:
   a 2-user race condition double-spends FTNS or corrupts
   a stake record. Test plan: 10-100 concurrent
   `httpx.AsyncClient` callers against each lifecycle
   endpoint with conflicting writes; assert post-state
   matches a serial-execution baseline.

2. **SC — Scale.** Largest file uploaded was probably 500B.
   Vision §11 claims multi-GB Tier B/C content. Risk: the
   in-memory hashing path (`hashlib.sha256(data)` in upload
   handler) OOMs on a 1GB upload. Test plan: 1MB / 10MB /
   100MB / 1GB upload + retrieve; measure memory, CPU,
   time; pin acceptable bounds.

3. **AD — Adversarial inputs.** We schema-pin canonical
   fields but never tried to bypass them. Signature
   forgery, replay attacks, content-filter evasion via
   Unicode normalization, JSON-bomb. Risk: a malicious
   client wedges a public node. Test plan: targeted
   negative-cases per §7 receipt verify, §14 content filter,
   §13 audit endpoints; document each defense.

4. **FM — Failure modes / fault injection.** What happens
   when staged file is deleted mid-retrieve? When the DB
   becomes unreachable? When a peer crashes mid-fetch? We
   have *some* coverage (sprint 473 F21, sprint 480 F22,
   sprint 484 F24 all surfaced via accidental fault
   injection) but no systematic suite. Test plan: chaos
   harness that randomly perturbs DB / FS / network during
   the canonical user journey.

5. **LR — Long-running stability.** Daemon has only run
   ~hours in any verification, never 24h+. Risk: memory
   leak (we found F23 sandbox-dir leak almost by accident),
   FD exhaustion, slow-growing dict that never evicts.
   Test plan: 24h soak test with continuous low-rate
   upload/retrieve/stake/claim; pin RSS and FD counts.

6. **OC — Real on-chain mutations (FTNS-side).** Sprint 466
   did 2 testnet ETH-side TX. Sprint 467 did 1 mainnet
   ETH-side TX. We've never actually moved FTNS on chain,
   staked on chain, claimed a royalty on chain. Risk: the
   real on-chain integration has a bug that schema-pin
   testing won't catch. Test plan: $5 of FTNS funding +
   exercise transfer, stake, claim cycle on Sepolia.

7. **XF — Cross-feature interactions.** Each feature
   tested in isolation. What happens when an inference
   receipt's auto-recorded creator access THEN triggers a
   royalty dispatch THEN updates the stake? Test plan:
   end-to-end chain test across §5.2 → §14 reputation →
   §5.3 royalty → §8 wallet for a single content access.

8. **§11/§12 Governance — entirely untested.** Proposals,
   voting, execution, treasury distribution — we have
   admin surfaces probed (sprint 471) but no real proposal
   lifecycle. Risk: governance system can't actually be
   used when a real proposal comes in. Test plan: simulate
   a proposal-lifecycle E2E with multi-signer voting.

The matrix is **NOT** a punch list to clear top-to-bottom —
it's a risk map. The verification campaign should pick
cells by production-risk per unit-effort, not by completeness.

---

## Operator-trifecta gaps (REST-only or MCP-only, missing CLI)

Some operator endpoints exist via REST but lack CLI / MCP wrappers
that would make them invocable from `prsm` or AI assistants. These
are NOT regressions — they're intentional scope decisions, but
worth surfacing here so the verification campaign can choose to
fill them.

| Feature | Has REST | Has CLI | Has MCP | Notes |
|---------|----------|---------|---------|-------|
| Incident response (triage) | ✅ | ✅ Sprint 434 (read-only) | ✅ | Trifecta-complete for read path; mutating commands deferred |
| Insurance fund status + recovery compose | ✅ | ✅ Sprint 435 | 🟢 | Trifecta-complete: `prsm node insurance status/compose-recovery` |
| TEE policy status + evaluate | ✅ | ✅ Sprint 436 | 🟢 | Trifecta-complete: `prsm node tee status/evaluate` |
| Federated learning admin | ✅ | ✅ Sprint 437 (read-only) | 🟢 | Trifecta-complete: `prsm node federated list/details` |
| Pipeline inference admin | ✅ | ✅ Sprint 437 (read-only) | 🟢 | Trifecta-complete: `prsm node pipeline list/details` |
| Bridge deposit/withdraw | ✅ | ✅ | 🟢 | Trifecta-complete |
| `prsm node fiat-readiness` | (uses sprint-285 function) | ✅ | 🟢 (`prsm_fiat_surface_health`) | Trifecta-complete |

---

## Verification campaign priorities

The dogfood arc surfaced that **🟢 (test-pinned, not live-verified)
features are NOT the same as ✅ (live-verified) features**. The
verification campaign promotes 🟢 → ✅ by running real workloads
against real daemons.

Recommended priority order (highest user-impact first):

1. **§4 step 7 — Tier B/C retrieve (encrypted content roundtrip)** —
   Closes the encrypted half of the canonical user workflow. Direct
   continuation of the dogfood arc.
2. **§4 step 8 — Single-node forge E2E** — Now that step 5 works,
   submitting a query against uploaded content closes the inference
   path. Needs `PRSM_QUERY_ORCHESTRATOR_ENABLED=1` + uploaded
   content + orchestrator config.
3. **§5.3 single-user stake → claim flow** — Multi-step economic
   loop. Validates RoyaltyDistributor v2 path end-to-end.
4. **§7 receipt verification on a live inference** — Run an actual
   inference, get a real signed receipt, verify it via
   `/compute/receipt/verify`. Closes the §7 truth-surfacing claim.
5. **Operator-trifecta gaps** — Fill CLI gaps for incident,
   insurance-fund, TEE policy, federated-learning admin. Lowest
   complexity, polishes the operator surface.

Each priority is its own sprint. The autonomous loop picks them up
in order unless an irreversible action surfaces.

---

## Mainnet contracts — external-gated verification

The deployed-fleet's verification status is fixed at deploy time
(immutable; chain-verified) and gated on external ceremonies, not
software sprints. Listed here for completeness.

| Audit gate | Status | Notes |
|------------|--------|-------|
| L3 cryptography external audit | 🔗 | `PRSM-POL-2` substitutes self-audit + Pausable + TVL caps; revisited at trigger events |
| L4 smart-contract external audit | 🔗 | Same substitution |
| L5 ML supply-chain external audit | 🔗 | RFP drafted, not commissioned |
| L6f infra pen-test | 🔗 | RFP drafted, not commissioned |
| L7 economic external audit | 🔗 | RFP drafted, not commissioned |
| L8 securities counsel (Track B, hard gate) | 🔗 | Retained as gate; sprint queue item |

---

## How to update this doc

When a sprint closes a 🟢 → ✅ promotion, edit the row inline and
add a one-line entry to the changelog at the bottom of this file.
When a sprint surfaces a new feature, add a row to the right
§-section table.

Keep the status legend honest. **A row should NOT be ✅ unless
someone has live-tested it.** Use 🟢 freely for "tests exist, not
yet verified live." That distinction is exactly what the dogfood
arc proved we need.

---

## Changelog

- **2026-05-16 sprint 497** — FTNS mainnet TX runbook
  dry-run walkthrough + 3 corrections applied. Walked the
  runbook on a zero-FTNS / zero-ETH test wallet to exercise
  the daemon's signing + RPC + endpoint paths without
  spending. Findings:
  (1) `/wallet/royalty/claim` has a **built-in DRY_RUN
  mode** — returns `{"status":"DRY_RUN","claimable_ftns":0.0}`
  when nothing to claim, not a 400.
  (2) TX-4 stake schema is `creator_id + amount_wei`, NOT
  `creator_eth_address + amount_ftns` as sprint 496 wrote.
  (3) TX-4 is **Foundation-ceremony-gated like TX-5** —
  stages stakes in the PENDING_COMMISSION in-memory mirror
  until `PRSM_CREATOR_STAKE_CONTRACT_ADDRESS` wired.
  
  Sprint 497 patched runbook + added 4 pin tests. Cost
  estimate reduced from "~$0.003 ETH + $5 FTNS" to
  "~$0.001 ETH + 2 FTNS" since TX-4 doesn't move on-chain
  funds today. Pre-broadcast safety also validated: chain
  reverts (e.g., `ERC20InsufficientBalance`) catch operator
  mistakes that slipped past daemon-side checks.

- **2026-05-16 sprint 496** — FTNS-side mainnet TX test
  plan shipped (`docs/operations/ftns-side-mainnet-tx-runbook.md`).
  OC column (real on-chain mutations) is the last untested
  dimension after sprints 487-495 closed concurrency / scale
  / adversarial / fault-injection / cross-feature / long-
  running. The runbook is operator-facing: documents pre-
  conditions (FTNS purchase paths, env vars), staged TX
  sequence (TX-1 self-transfer → TX-2 transfer → TX-3
  royalty claim → TX-4 stake commission; TX-5 settler bond
  deferred), cost estimate ($0.003 ETH + $5 FTNS float),
  risk register (wrong-network / private-key-leak / nonce
  conflict mitigations), and rollback honesty (FTNS-side TX
  are irreversible). 11 pin tests defend canonical contract
  addresses, env vars, staged sequence, cost-estimate
  presence, and a "no leaked private key" sentinel.
  Execution gated on user funding the wallet.

- **2026-05-15 sprint 429** — Initial draft. Inventory: 202 REST
  endpoints, 124 MCP tools, 130+ CLI subcommands, 38 audit-prep
  sections. Status snapshot reflects the post-sprint-428 state of
  the codebase. Vision §4 step-5 (single-node retrieve) marked ✅
  for the first time.
- **2026-05-15 sprint 430** — Tier B/C encrypted-content roundtrip
  closed. Vision §4 steps 6 + 7 (recipient encryption upload +
  retrieve + decrypt) promoted 🟢 → ✅ via live byte-identical
  end-to-end test. ContentRetriever local-publish shortcut extended
  to route Tier B/C staged dirs to `_fetch_tier_bc` (infrastructure
  for the future Shamir-exposing upload endpoint). 6 unit tests
  pinned; 2 new (Tier B/C dir routes locally; malformed dir raises
  not falls through).
- **2026-05-15 sprint 431** — F9 fix: upload-side embedding-provider
  parity. Forge pipeline live-tested + surfaced cryptic
  "shapes (384,) and (1536,) not aligned" error when
  `OPENAI_API_KEY` is set. Root cause: upload-side `_embedding_fn`
  fell through to OpenAI ada-002 (1536-dim) while query-side is
  hard-pinned to MiniLM (384-dim). Fix: node.py binds
  `preferred_provider="sentence_transformers"` via functools.partial.
  Default install works out of the box; operator-overridable via
  `PRSM_UPLOAD_EMBEDDING_PROVIDER`. Live-verified: 384-dim
  embeddings stored even with OpenAI key set; forge pipeline
  passes the embedding stage. Surfaced F10 (single-node empty
  aggregator pool) as the next bottleneck. 4 new tests / 78
  cross-suite green.
- **2026-05-15 sprint 457** — Peer-lifecycle live verification.
  Followed sprint 456's two-daemon bench: killed daemon #2,
  polled daemon #1's /bootstrap/status. **peer_leave propagated
  in 21 seconds** (well under bootstrap heartbeat interval +
  detection grace). Counters reflected the full lifecycle:
  `peer_join_events: 1` (from sprint 456) → `peer_leave_events: 1`,
  `discovered_peer_count: 1 → 0`, `known_count: 1 → 0`. Sprint
  320-329 P2P discovery hardening's peer-lifecycle handling is
  operationally verified end-to-end (join + leave both fire,
  bootstrap-server propagation works). 1 PRSM_Testing.md row
  added. Doc-only sprint.
- **2026-05-15 sprint 456** — Multi-node test bench. Stood up
  daemon #2 on same host with separate `HOME=/tmp/prsm-node-2`,
  api=8001, p2p=9011. Both connected to canonical bootstrap
  server. **Symmetric peer discovery verified**:
  - Daemon #1 (cdefb8e5…) and daemon #2 (147e19a0…) each have
    the other in their `known[]` list with node_id + address
    + last_seen
  - `peer_join_events: 1` on both sides
  - sprint 320-329 P2P discovery hardening operationally
    confirmed working
  But: **F14 surfaced** — both daemons announce their external
  NAT-translated public IP (146.70.202.118), so single-host
  loopback to each other fails. `connected_count: 0` on both
  sides; cross-node content retrieve → `not_found,
  providers_tried: 0`. Three Option A/B/C candidates documented;
  multi-host test bench (Option C, cloud VM) is the eventual
  right answer. 1 row promoted (peer discovery ✅), 1 row
  deferred-with-F14-attribution (direct P2P), 1 row stays 🔬
  (cross-node retrieve).
- **2026-05-15 sprint 455** — §13 admin operator surface sweep.
  Live-verified the remaining §13 admin operator endpoints:
  /admin/slash-history (paginated empty-state),
  /admin/heartbeat-history (idem), /admin/distribution-history
  (idem), /admin/insurance-fund/status (treasury_address=
  Foundation Safe `0x91b0e6F8...`, target_bps=500 reserve
  target, commissioned=false in dev env), /admin/emergency-
  pause/status (live readback of ALL 9 mainnet contracts —
  ftns_token / royalty_distributor / BSR / EscrowPool /
  StakeBond / Ed25519Verifier / StorageSlashing /
  KeyDistribution / EmissionController — with paused state +
  commissioned=true for each on chain_id=8453 Base mainnet),
  /admin/takedown-notices (empty-state), /admin/corp/issuer
  (empty-state). The emergency-pause readback is the
  **safety-critical operator surface** for §14 smart-contract
  exploit response — operators check this BEFORE attempting
  a pause to know which contracts can/can't be paused. 5 §13
  rows promoted ✅. Doc-only.
- **2026-05-15 sprint 454** — §13 /metrics + bootstrap-server live-probe.
  GET /metrics returns full Prometheus exposition format (9+ gauges:
  pending_escrow_count, total_locked_ftns, job_history_size,
  receipt_store_size, royalty_dispatch_ring_size,
  escrow_cleanup_task_running=1, build_info{version="1.7.0"},
  slash/heartbeat/distribution log counts). `prsm bootstrap-server
  status --host bootstrap1.prsm-network.com --port 8000` live-
  probed the **canonical DigitalOcean droplet** running the live
  PRSM bootstrap fleet:
    ✓ healthy
    uptime: 84036 seconds (~23 hours)
    active_connections: 1 (my session)
    total_connections: 63 (cumulative)
    failed_connections: 0
    messages_processed: 1548
    region: nyc3
    version: 1.0.0
  This is **live mainnet bootstrap-fleet attestation**: the public
  PRSM network entry point is operationally healthy. 2 §13 rows
  promoted. Doc-only sprint.
- **2026-05-15 sprint 453** — F13 fix + §13 MCP-tool sweep. The
  MCP-tool live-verification swept 114 registered MCP tools
  (exceeds Vision §11's "73+" claim). Caught **F13 production-
  blocker**: `prsm_node_status` MCP returned cryptic
  "Cannot reach PRSM node: 500" because `/rings/status` 500'd
  on `AttributeError: 'QueryOrchestrator' object has no
  attribute 'traces'`. Root cause: sprint 173's agent_forge
  swap left dashboard_metrics.py reading the legacy
  AgentForge `.traces` attribute. Fixed: defensive
  `getattr(forge, 'traces', []) or []` at both call sites.
  Live-verified post-fix: /rings/status returns full Ring 1-10
  JSON; prsm_node_status renders the full health table.
  4 new pin tests + F1-F13 dogfood-findings count updated.
  Also sample-verified `prsm_info / prsm_peers / prsm_node_health
  / prsm_metrics_summary` MCP tools — all return clean output.
  Tag `dashboard-metrics-query-orchestrator-compat-merge-ready-20260515`.
- **2026-05-15 sprint 452** — KYC surface + fiat-readiness CLI live-
  verified. POST /wallet/kyc/initiate with {user_id, email, tier}
  returns PENDING_COMMISSION envelope (vendor=null in dev env per
  sprint-285's commissioning pattern). GET /wallet/kyc/status
  returns {commissioned: false, vendor: null, supported_vendors:
  ["persona","onfido","plaid"], record_count}. `prsm node
  fiat-readiness` returns "✓ Phase 5 fiat surface ready — OK
  (no findings)" in text mode, `{overall_status: "ok",
  findings: []}` in JSON. Sprint 422's CLI works against live
  env. 4 §8/Phase-5 rows promoted/refreshed. Doc-only.
- **2026-05-15 sprint 451** — §8 + Phase 5 fiat surface live-verified.
  POST /wallet/onramp/quote with {usd_amount: 100,
  destination_address: 0x...} returns full quote: requested_usd,
  ftns_to_receive, usd_rate, kyc_required, kyc_status,
  tier_level, tier_limit_usd, tier_limit_remaining_usd,
  tier_limit_exceeded + quote.{usd_in, usdc_acquired,
  ftns_received, onramp_route: "coinbase-cdp", swap_route:
  "aerodrome", payment_method_alias}. POST /wallet/offramp/quote
  returns operator-readable balance breakdown when
  destination has 0 balance ("requested $X, available $Y…").
  GET /wallet/pool/state reports `NOT_CONFIGURED` with
  actionable env-var + seeding-ceremony-date note. GET
  /admin/fiat-compliance/summary auto-recorded my onramp-quote
  call as `{onramp_quote: {count: 1, total_usd: 100.0}}` —
  the Vision §11 AUSTRAC/FinCEN/IRS-ready auto-record claim
  is **live-attested**. 5 §8/Phase-5 rows promoted ✅. Doc-only.
- **2026-05-15 sprint 450** — Hardware classification + supported-models
  + resources live-verified. `prsm node benchmark` returns full
  hardware profile (Apple M4 / 10 cores / 16GB VRAM/RAM / 4.60 TFLOPS
  FP32 / thermal=sustained) → Compute Tier T1 classification. GET
  /compute/models returns the 3 mock models (mock-llama-3-8b /
  mock-mistral-7b / mock-phi-3) registered by sprint 438's
  MockInferenceExecutor — cross-confirms the executor wiring is
  active. GET /node/resources returns full canonical schema with
  cpu/mem/storage/gpu allocation percentages, bandwidth limits,
  active hours, and computed effective-resources. 3 rows promoted
  🟢 → ✅. Doc-only.
- **2026-05-15 sprint 449** — §5.1 storage + content stat surfaces
  live-verified. /storage/stats returns full canonical schema
  (storage_available, pledged_gb=10.0, used_gb=0.0,
  challenge_config + challenge_stats nested). /storage/pinned-
  stats + /storage/provider-reputations return clean empty-state.
  /content/index/stats returns the 3-key telemetry envelope.
  /content/provider-stats reports `local_content_count: 3` —
  cross-confirms sprint 441's fingerprint-test uploads are
  tracked correctly. /content/mine returns the cross-restart-
  persisted uploads with full field coverage. 5 §5.1 rows
  promoted 🟢 → ✅. Doc-only.
- **2026-05-15 sprint 448** — §7 attestation backends live-verified.
  `IntelASPBackend` parses SGX v3 structural probe → vendor=
  "intel-sgx" with `version/mrenclave_hex/mrsigner_hex/
  structural_only` populated in vendor_data. `AMDKDSBackend`
  parses SEV-SNP v2 structural probe → vendor="amd-sev-snp"
  with `version/guest_svn/measurement_hex/report_data_hex/
  chip_id_hex` populated. `verify_attestation()` registry
  dispatcher correctly routes by quote-version bytes — Intel
  v3 → intel backend; AMD v2 → amd backend.
  `vendor_verified: false` on both — the documented honest-
  scope: real DCAP/KDS signing-chain verification deferred
  until those vendor SDKs are wired. The structural parse IS
  the value-add for sprint-444's live-mainnet-attestation
  pattern: the on-chain probe verifies WHAT is registered;
  the structural parse verifies the registered attestation
  bytes are well-formed under the vendor's published quote
  layout. 3 §7 attestation-backend rows promoted to ✅ with
  live-evidence attribution. Doc-only.
- **2026-05-15 sprint 447** — Bootstrap connectivity + health/detailed +
  receipt-persistence live-verified. /bootstrap/status shows the node
  is actively connected to `wss://bootstrap1.prsm-network.com:8765`
  with 28 reconnect attempts + 16 successes over the session,
  client_state="connected", multi-fallback (US/EU/APAC) enabled
  per sprint 375. /health/detailed reports 14 subsystems including
  ftns_ledger (canonical_match=true on the wired contract address),
  payment_escrow (cleanup_task_running), job_history (count=1),
  receipt_store (count=1). GET /compute/receipts confirms the
  sprint 438 mock-inference receipts PERSIST across daemon restart
  with full fidelity: epsilon_spent=0.0 (F12 fix holds), 64-byte
  settler_signature intact, output_hash intact, tee_attestation
  zero-filled as expected for the mock. **Cross-restart receipt
  persistence is the production reliability guarantee** — operators
  expect signed receipts to survive restarts; this sprint
  verified that operationally. 3 §13 rows attributed to sprint 447.
- **2026-05-16 sprint 486** — Test coverage matrix added.
  Surfaces the DEPTH dimension that the section-by-section
  tables lack. Each subject area rated across 10 dimensions
  (HP/EP/CC/SC/AD/FM/LR/DR/OC/XF). **What's untested becomes
  visible at a glance**: zero concurrency coverage anywhere,
  no scale testing, no adversarial input testing, no
  long-running stability, no FTNS-side on-chain mutations,
  no cross-feature integration tests. Closes the loop on
  the user's challenge ("Have we really thoroughly tested
  ALL of PRSM's functionality? I doubt it") — answer: no,
  and here's the map of what's missing. 8 priority cells
  identified for follow-on dedicated sprints. The matrix is
  a risk map, not a punch list: future sprints should pick
  cells by production-risk per unit-effort, not by
  completeness.

- **2026-05-16 sprint 485** — F24 follow-on closed. F22 +
  F24 + sprint 485 chain end-to-end working: hydrated CIDs
  deliver original bytes across daemon restart. Vision §4
  step 5 cross-restart fully closed for Tier A content.
  17/17 historical hydrated CIDs return success live. Tier
  B/C deferred. Tag
  `sprint-485-f24-followon-hydration-seed-paths-merge-ready-20260516`
  commit `33e200af`.

- **2026-05-16 sprint 484** — Semi-fresh dogfood pass +
  F24 fix shipped. User-authorized re-walk of §4 surfaced
  F24 (retrieve hangs on hydrated CIDs). Fix: propagate
  timeout through request_content → _fetch_local →
  _fetch_local_via_bt → retriever.fetch + asyncio.wait_for
  defense-in-depth. 30s+ hang → 2s clean response.

- **2026-05-16 sprint 478** — cross-table attribution
  alignment. 5 PRSM_Testing.md rows promoted via existing
  sprint coverage they were attributed to elsewhere in the
  doc — bookkeeping consolidation, not new live testing:

  - §4 SHA-256 fingerprint registry ✅ (sprint 291, 425,
    441 — sprint 441's §14 E2E covers).
  - §4 Duplicate detection on re-upload ✅ (sprint 291,
    441 — sprint 441 attested `duplicate_of_creator`).
  - §4 Marketplace fingerprint lookup ✅ (sprint 291,
    441 — sprint 441 attested `duplicate_attempt_count`).
  - §7 Privacy budget tracking ✅ (sprint 445 attested
    /privacy/budget canonical schema).
  - §7 Recipient manifest read ✅ (sprint 472 schema-
    defended + sprint 430 E2E roundtrip).

  Cumulative ✅ rows now 211 (was 206).

- **2026-05-16 sprint 477** — §14 takedown-notice + corp-
  capability lifecycle E2E. 2 rows touched:

  **Takedown lifecycle** (`/admin/takedown-notices/{id}/status`):
  - POST /admin/takedown-notice → status=received with full
    envelope (notice_id, timestamp, target_cid, sender,
    jurisdiction, basis, notice_text, status).
  - POST /{id}/status → transitions received → acknowledged
    → disputed. State changes round-trip via GET /{id}.
  - Invalid status → 422 with canonical list
    `['acknowledged', 'disputed', 'expired', 'received']`.

  **Corp capability** (`/admin/corp/issuer`,
  `/admin/corp/capability/{id}/...`):
  - POST /issuer with 32-byte Ed25519 pubkey → 200 +
    issuer envelope. Non-32-byte pubkey → 422 with clear
    `Ed25519 pubkey must be 32 bytes, got N` message
    (cryptographic-soundness invariant: 44-byte X.509-DER-
    wrapped pubkeys explicitly rejected).
  - GET /issuer → paginated list.
  - GET /capability/{id}/ledger → empty entries envelope.
  - GET /capability/{id}/consumed → `{capability_id,
    consumed: 0}`.

  Cumulative ✅ rows now 206.

- **2026-05-16 sprint 476** — incident-response full lifecycle
  E2E live-verified. Completes the Foundation-Safe-adjacent
  flow trio (upgrade-proposal sprint 475, disclosure-payout
  sprint 475, incident-response sprint 476):
  - POST /admin/incident/open → s2 incident, current_phase
    detected, full envelope with timeline + affected_contracts
    + related_disclosure_id slot.
  - POST /event appends to timeline preserving phase.
  - POST /advance transitions detected → triaged + timeline
    grows.
  - GET /recommendations returns context-specific actions
    keyed on (severity, current_phase).
  - GET /comms-template returns ready-to-paste internal-
    update markdown.
  - GET /playbook surfaces full s0-s3 × phase decision tree
    with explicit **Vision §14 guidance**: "PAUSE NOW:
    invoke prsm_emergency_pause compose_pause" / "Foundation
    Safe signs IMMEDIATELY (target: <15min)" / "Engage
    on-chain forensics partner (Chainalysis / TRM Labs)".

  Sprint 476 closes the operator-side incident-response audit
  surface. The TX-side response (emergency-pause compose) is
  already ✅ via sprint 455's 9-contract mainnet readback.

  Cumulative ✅ rows now 205.

- **2026-05-16 sprint 475** — Foundation-Safe composer flows
  E2E. Two full multi-step lifecycles operationally attested:

  **Upgrade-proposal** (`/admin/upgrade/...`):
  - POST /propose → status=proposed; full proposal envelope
    (proposal_id, target_proxy, new_implementation, severity,
    rationale, init_calldata_hex, reviewer_assignments).
  - GET /{id} round-trips; GET /admin/upgrade list reflects.
  - /update advances proposed → reviewed → safe_uploaded
    with safe_tx_hash recorded.
  - /compose-upgrade returns Safe-uploadable tx with
    `upgradeToAndCall` calldata, chain_id=8453, explicit
    storage-layout warning + 4-step Safe UI instructions.
  - /compose-rollback enforces `status==executed` (correctly
    rejects pre-execution rollback attempts).

  **Disclosure-payout** (`/admin/disclosure/...`):
  - POST /submit → status=received; details stored as
    base64 (privacy-preserving).
  - /update advances received → triaged → awarded with
    triage_notes + payout_ftns.
  - /compose-payout enforces `status==AWARDED` invariant
    (correctly rejects pre-award attempts).
  - On awarded → returns FTNS ERC-20 `transfer(recipient,
    1000e18)` calldata against the mainnet FTNS contract
    (`0x5276a3756C85f2E9e46f6D34386167a209aa16e5`), with
    Safe UI instructions.
  - /record-payout-tx closes audit trail with on-chain
    tx hash.

  **Vision §14 invariant operationally verified for both
  flows: composer PRODUCES the transaction; does NOT
  execute it.** Foundation Safe 2-of-3 hardware multisig
  retains exclusive privilege per the canonical claim.

  Cumulative ✅ rows now 204 (was 202).

- **2026-05-16 sprint 472** — §4 content + §5.2 inference
  paired-surface sweep. 6 PRSM_Testing.md rows promoted:
  - `GET /content/{cid}` → 404 "Content not found in index"
    for un-indexed CIDs (Tier A uploads don't auto-index;
    honest-scope distinction documented).
  - `GET /content/recipient-manifest/{cid}` → 422 schema-
    defended "not an encrypted recipient bundle" on Tier A
    CID (this endpoint is Tier B/C-only by design).
  - `GET /content/search` → clean `{query, results, count}`
    envelope; tier filter validates `min_tier` enum
    (`low/medium/high`) — invalid → 400 with canonical list.
  - `GET /content/mine` post-upload returns full canonical
    schema (content_id, filename, size_bytes, content_hash,
    creator_id, royalty_rate, access_count, total_royalties,
    provenance_tx_hash, created_at, is_sharded).
  - `POST /content/upload/shard` ⚠️ schema-pass: missing
    `dataset_id` → 400 with clean message; full E2E untested
    (needs sharded-dataset fixture).
  - `POST /compute/inference/tensor_parallel/shard` +
    `POST /compute/inference/pipeline/stage` → schema-defended
    422 with full required-field lists. Defense-in-depth
    against malformed shard/stage dispatch on the live
    inference fabric.

  Cumulative ✅ rows now 202 (was 196).

- **2026-05-16 sprint 471** — §13 admin + §14 paired-surface
  sweep. 13 PRSM_Testing.md rows promoted via end-to-end probe
  against running daemon:
  - **Audit ring** live-attested: `/audit/summary` auto-recorded
    all 24 sprint-469+470 probe calls with full schema
    `{total, status_buckets: {2xx, 4xx, 5xx}, method_buckets,
    top_paths}`; `/audit/recent` paginated envelope with full
    per-call detail (timestamp/method/path/requester/status_code).
    Vision §13 audit-ring promise operationally verified.
  - **Content filter lifecycle**: POST /admin/content-filter/tags
    → `{added:1, total:1}`; GET shows tag in `blocked_model_tags`;
    DELETE → `{removed, total:0}`. Tag-blocklist promoted ✅ in
    both §13 + §14 tables.
  - Marketplace surfaces: `/creator-reputation/{id}` clean
    unknown-creator default schema; `/marketplace/reputation`
    paginated providers; `/content/mine` paginated.
  - Fiat compliance + KYC status canonical schemas.
  - Operator admin lists: `/admin/upgrade` + `/admin/disclosure`
    return `{records, count}` empty-state envelopes; sub-routes
    (propose/update/compose-*) schema-pinned for follow-on E2E.
  - TEE policy: `/evaluate` → 422 with clear `policy` required
    field; `/node-status` → `{effective_tier, vendor,
    vendor_verified, diagnostic}` clean honest-scope envelope.
  - Royalty claim (⚠️ schema-pass): 503 with actionable
    `set PRSM_ROYALTY_DISTRIBUTOR_ADDRESS explicitly OR
    PRSM_NETWORK=mainnet`; full flow gated on mainnet wiring.

  No production-blockers surfaced. Cumulative ✅ rows now 196.

- **2026-05-16 sprint 470** — §5.3 staking + settlement +
  settler-registry live sweep. 8 PRSM_Testing.md rows promoted
  via end-to-end probe against running daemon:
  - `POST /staking/unstake` → request_id with `available_at`
    7 days out (Vision §11 cooldown invariant); stake status
    `active → unstaking`, `total_staked` drops, pending list
    populated.
  - `POST /staking/cancel-unstake/{id}` → `{cancelled: true}`;
    stake restored to active; total_staked back; pending
    cleared.
  - `POST /staking/withdraw/{id}` (⚠️ schema-pass) — cancelled
    request → 400 "Request status invalid: cancelled"; unknown
    UUID → 404. Happy path gated by 7-day cooldown.
  - `GET /settlement/pending`, `GET /settlement/history` →
    empty-state envelopes.
  - `POST /settlement/flush` → full canonical schema with
    `tx_hashes: [], errors: [], duration_seconds`.
  - Full settler registry lifecycle: register (min bond 10000
    FTNS enforced — Vision §11 invariant); GET /settler/{id}
    (status, can_settle, total_settled, slashed_amount);
    unbond (30-day cooldown, `unbond_at` set, list/active
    filters); ledger/export with `integrity_hash` (cryptographic
    chain-of-custody surface); slash/propose schema-validated.

  No production-blockers surfaced. Min-bond + cooldown invariants
  operationally attested. Sprint 432's single-user stake →
  claim E2E now complemented by sprint 470's unstake-cycle
  closure — Vision §5.3 economic layer is operationally
  sound.

  Cumulative ✅ rows now 183.

- **2026-05-16 sprint 469** — §5.2 compute job-lifecycle live
  sweep. 9 🟢 rows promoted to ✅ via end-to-end probe against
  running daemon:
  - `POST /compute/submit` → returns `{job_id, status: pending,
    job_type, ftns_budget}`; happy path (embedding) ✅, invalid
    job_type → 400 ✅, oversized payload (`PRSM_MAX_JOB_PAYLOAD_BYTES`
    default 100KB) → 413 ✅.
  - `GET /compute/status/{id}` → escrow snapshot with full schema
    (escrow_id, requester_id, amount_ftns, status, tx_lock,
    created_at); unknown job_id → 404 with actionable detail.
  - `GET /compute/status/{id}/stream` → SSE `event: status` with
    escrow snapshot emitted ✅. De-duped by JSON equality;
    terminal on history/escrow terminal state OR
    `PRSM_STATUS_STREAM_TIMEOUT_SEC`.
  - `POST /compute/cancel/{id}` → `{history_cancelled,
    escrow_refunded, refund_amount_ftns}`; pending-job cancel
    refunds locked FTNS budget in full ✅.
  - `GET /compute/jobs` → paginated envelope `{jobs, total,
    offset, limit}`; cancelled-pending jobs don't reach history.
  - `POST /compute/cleanup-stale` → `{cleaned: 0}` empty-state.
  - `POST /compute/train` → clean 503 with actionable
    `set PRSM_FEDERATED_WORKER_PRIVKEY env` hint when worker
    privkey not configured.
  - `GET /compute/stats` → full canonical schema (resources,
    allocation, capacity, active_jobs, completed_jobs).
  - `GET /compute/receipt/{id}` → 404 `No receipt for job_id='...'`
    for jobs without receipts (pending/cancelled).

  No production-blockers surfaced — the §5.2 compute lifecycle
  is operationally sound. Cumulative ✅ rows now at 176 (was 167).

- **2026-05-16 sprint 468** — Multi-host bench resume. Both
  daemons reconnected to bootstrap fleet via EU (NYC unreachable
  from local network at test time). Cross-host discovery
  verified: daemon #1 (local, `136.47.243.122:9001`) and daemon
  #2 (droplet, `159.203.129.218:9001`) appear in each other's
  `known[]` lists with full capabilities metadata.

  Cross-host **content retrieve** test: daemon #2 uploaded CID
  `f0aadee79…`; daemon #1 retrieve returned `not_found,
  providers_tried: 0`. `connected_count: 0` both sides.

  **F20 surfaced**: direct `nc -zv 159.203.129.218 9001` from
  local times out. ufw on droplet INACTIVE — security is via
  DO cloud firewall, which has 22+8000+8765 open but NOT 9001.
  Operator P2P port not opened on DO cloud firewall (separate
  from droplet OS firewall ufw which sprint 458 cloud-init
  configured but never activated).

  **Three-layer NAT/firewall arc complete** (F14 single-host
  loopback + F20 DO cloud firewall + local home NAT outbound-
  only — typical operator pattern, not a bug).

  Multi-host bench succeeds at **discovery layer** ✅ (sprint
  320-329 P2P discovery hardening cross-host-verified).
  **Content fetch** requires operator to open inbound P2P port
  on at least one side — typical pattern is the droplet
  operator does so; home operators stay outbound-only. F20
  documented + deferred (needs DO API token or web console
  to fix cloud firewall, out of session scope).

  Roadmap row: Cross-host content retrieve ⏸️ pending F20 fix.

- **2026-05-16 sprint 467** — **Level A Base mainnet TX exercise**.
  User funded wallet `0x2Fd48D2d…` with 0.0005 ETH on **Base mainnet**
  (chain_id 8453 — real production network, real funds at stake).
  Switched daemon from Sepolia → mainnet by dropping `PRSM_NETWORK`
  + `BASE_SEPOLIA_RPC_URL` env vars. Confirmed daemon's
  `/health/detailed.ftns_ledger.canonical_match: True` against
  mainnet FTNS contract `0x5276a37…`.

  **First real PRSM TX on Base mainnet**:
    Hash: 0xae65db7370fb50fba5e70572cf0d93511e933b8b698282e089ddbf2d51757d9e
    Block: 46073119 (Base mainnet)
    Status: 1 (success)
    Gas: 21000 @ 0.006 Gwei = 0.000000126 ETH (~$0.0004 USD)
    Basescan: https://basescan.org/tx/0xae65db7370fb50fba5e70572cf0d93511e933b8b698282e089ddbf2d51757d9e

  **Mainnet contract reads** (companion to sprint 466's testnet
  reads — proves cross-network invariant differentiation):

  - Mainnet FTNS `0x5276a37…`:
    - name(): "PRSM Fungible Tokens for Node Support"
    - symbol(): "FTNS", decimals(): 18
    - totalSupply(): **100,000,000.0000 FTNS** (vs testnet's
      100,002,060 — different deploy mints, both immutable post-T1)

  - **Mainnet INV-RD-3 PASS** via daemon's formal-verification
    endpoint:
    - RoyaltyDistributor v2 `0xfEa9aeB9…` reports owner() ==
      `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791` (Foundation
      Safe — 2-of-3 multisig)
    - "Foundation Safe is sole administrator" claim
      empirically verified live.

  - **Mainnet INV-EC-1 PASS** via daemon's formal-verification:
    - EmissionController `0x13A0D76…` reports
      EPOCH_DURATION_SECONDS == **126,144,000** (= 4 years exactly)
    - Cross-network: testnet has 3,600 (1 hour per sprint 466);
      mainnet has 4 years — both immutable per their constructors.
      Vision §11's "constructor-set with chainid-8453 4-year
      enforcement" empirically verified.

  Sprint 467 vs sprint 466 — same wallet, same daemon, two
  networks, paired invariant verifications. The verification-
  campaign methodology now has a complete cross-network
  attestation cycle from one operator's commodity-hardware Mac.

  Final wallet state: 0.0004999874 ETH remaining on mainnet
  (sufficient for ~80,000+ more contract TX before refund).
  Sepolia wallet still has 0.00489975 ETH for further testnet
  exploration.

- **2026-05-15 sprint 466** — Level B Sepolia testnet TX exercise.
  User funded wallet `0x2Fd48D2d…` with 0.005 ETH on Base Sepolia.
  Executed real on-chain TX from the PRSM-configured wallet:

  TX #1: self-transfer 0.0001 ETH
    Hash:  0xa2b06bf59777c728bd89b43fd9687264a72e7eb56b1ab438f46a7ee1923fea08
    Block: 41559508, status: 1
    Gas: 21000 @ 6 Gwei = 0.000000126 ETH
    Basescan: https://sepolia.basescan.org/tx/0xa2b06bf59777…

  TX #2: wallet → testnet foundation_safe `0xCCAc7b21…`
    Hash:  0xead1f03055491e4a76f62a72cda27318926a162678d483ac8fcffe4571bfbf30
    Block: 41559540, status: 1
    Final wallet balance: 0.00489975 ETH

  Plus two contract-level live reads on Sepolia:

  - FTNS contract `0x7F5f00FA…`: name()="PRSM Fungible Tokens
    for Node Support", symbol()="FTNS", decimals()=18,
    totalSupply()=100,002,060 — full ERC-20 identity verified
    on testnet.
  - EmissionController `0x1478F8f5…`: EPOCH_DURATION_SECONDS()
    returns 3600 (1 hour) — matches testnet T10 redeploy
    (2026-05-07) per config/networks.py:164. Empirically verifies
    Sprint 358's INV-EC-1 invariant ("EPOCH_DURATION_SECONDS is
    constructor-set immutable") on the testnet variant.

  Daemon-side `/balance/onchain` returns the SAME state as
  direct Web3 reads (balance_ftns: 0.0) — daemon is correctly
  proxying to live Sepolia.

  Per the testnet config note, FTNS distribution is
  founder-airdropped (not faucet), so we can't programmatically
  acquire FTNS for stake/transfer testing without a Discord
  request. Sprint 466 scope: ETH-only TX + contract reads —
  covers the load-bearing infrastructure verification.

  5 new ✅ rows in PRSM_Testing.md.

- **2026-05-15 sprint 465** — Level B Sepolia testnet wiring (TX
  pending wallet funding). Switched daemon from Base mainnet
  (chain_id 8453) to Base Sepolia (chain_id 84532) via four
  env vars: `PRSM_NETWORK=testnet`, `BASE_SEPOLIA_RPC_URL=
  https://base-sepolia-rpc.publicnode.com`, `PRSM_ONCHAIN_FTNS=1`,
  `FTNS_TOKEN_ADDRESS=0x7F5f00FA…`. Verified:
  - `/info` reports chain_id 84532, network "testnet",
    rpc_host sepolia.base.org, canonical FTNS 0x7F5f00FA…
  - `/health/detailed.ftns_ledger.canonical_match: True` — daemon
    is talking to the **live Base Sepolia FTNS contract**
  - `/balance/onchain` queries Sepolia for the wallet address,
    returns 0 (correct — wallet unfunded)
  - Default public RPC `https://sepolia.base.org` is unreachable
    from this network; `https://base-sepolia-rpc.publicnode.com`
    works reliably (latest block 41558920+ at test time)
  Schema-discovery during the test: env-var precedence for
  testnet RPC is `BASE_SEPOLIA_RPC_URL` (not the mainnet-named
  `BASE_RPC_URL`). The first attempt with `BASE_RPC_URL` got
  ignored — the dual-env-name dispatch is per `prsm.config.
  networks.resolve_endpoints()` lines 270-283.
  TX exercise (transfer, stake, etc.) pending wallet funding via
  Sepolia faucet. The wiring layer is fully verified; the
  funded-TX layer is the natural next sprint.
- **2026-05-15 sprint 464** — On-chain wallet integration (Level C —
  loading + address-derivation, no on-chain TX). Generated fresh
  test wallet (private key + address `0x2Fd48D…`), set
  `FTNS_WALLET_PRIVATE_KEY` env var, restarted daemon. Verified:
  - `/health/detailed.ftns_ledger.connected_address` populated
    with derived address; `canonical_match: True` against
    Base mainnet FTNS contract `0x5276a37...`
  - `/balance/onchain` queries the LIVE mainnet contract for the
    wallet's balance (returns 0 — correct, empty wallet) — full
    {address, balance_wei, balance_ftns, usd_rate,
    claimable_royalties_ftns, escrowed_ftns} envelope
  - `/wallet/spend` + `/wallet/escrows` previously 503'd without
    wallet; now return clean address-scoped data
  - `/marketplace/creator-stake/<addr>` returns full schema for
    the new wallet (balance 0, not high-tier-eligible)
  - `prsm wallet info` CLI shows: chainId 84532 (testnet default
    — discovered PRSM already has Base Sepolia contracts
    deployed), explorer URL, foundation-config warnings,
    halving-curve testnet vs mainnet differences
  - `prsm node earnings` CLI now shows `Operator: 0x2Fd48D…`
    (was "PRSM_OPERATOR_ADDRESS unset")
  Wallet-loading is purely local + side-effect-free (no
  on-chain TX). Sprint 465+ candidate: Level A (fund wallet
  with small Base mainnet ETH for real stake/claim TX) or
  Level B (deploy test contracts to Sepolia + use testnet
  faucet — PRSM already has Sepolia FTNS at 0x7F5f00FA…
  apparently).
- **2026-05-15 sprint 446** — Operator CLI surface live-verified.
  Walked the §13 operator-trifecta CLI lane and confirmed each
  surface returns clean actionable output:
  - `prsm node info` → full Rich table (Node ID, Display Name,
    Public Key, Roles, P2P/API Port, Data Dir, Bootstrap Nodes)
  - `prsm node earnings` → clean "(PRSM_OPERATOR_ADDRESS unset)"
    + "not wired" rows for royalty/heartbeat/distribution
  - `prsm node heartbeats` → "No entries" (correct empty-state)
  - `prsm node webhooks` → "set PRSM_WEBHOOK_URL to enable"
    (actionable)
  - `prsm wallet info` → "no address available — set PRIVATE_KEY
    env var or pass --address" (actionable)
  Three §13 trifecta rows attributed to sprint 446 with "CLI live"
  notes documenting the actionable empty-state messages. This is
  the operator-UX truth-surfacing the dogfood arc was designed
  for: the CLI doesn't crash + doesn't lie about state + tells
  the operator what env var to set. Doc-only.
- **2026-05-15 sprint 445** — Streaming-inference UX path + §7 privacy
  budget + arbitration queue live-verified. /compute/inference/stream
  returns clean 503 "Inference executor does not support streaming.
  Wire a ParallaxScheduledExecutor (Phase 3.x.8.1) to enable
  /compute/inference/stream." — the UX guides operators to the
  correct wiring action. Full streaming E2E remains 🟢 (Parallax
  executor wiring out of scope). /privacy/budget returns canonical
  schema {max_epsilon, total_spent, remaining, num_operations,
  spends}. /content/arbitration/queue returns empty-state cleanly.
  4 rows updated/added in PRSM_Testing.md. Doc-only.
- **2026-05-15 sprint 444** — §5.3 royalty + settlement admin surface
  live-verified. /admin/royalty-dispatch-summary returns canonical
  schema (total, status_counts, total_sent_wei, by_allocation_mode,
  earliest_ts, latest_ts). /admin/royalty-dispatch-history returns
  paginated envelope. /ledger/sync/stats returns 5-field stats
  (txs_broadcast/received/rejected, reconciliations_run,
  discrepancies_found). /settlement/stats + /settler/list/active
  return clean empty-state. 4 PRSM_Testing.md rows promoted 🟢 → ✅.
  Empty-state verification is a real value-add: it confirms the
  endpoints don't 500 / 404 / 503 in absence of data — operators
  integrating against these endpoints get the canonical shape on
  day 1. Doc-only.
- **2026-05-15 sprint 443** — §5.4 formal-verification surface live-
  verified. Hit /admin/formal-verification/invariants — returns the
  full list of 20 critical invariants with severity / spec_text /
  kind / selector / expected_value. Hit /admin/formal-verification/
  check?contract=royalty_distributor — `INV-RD-3` (contract owner ==
  Foundation Safe 0x91b0e6F8...) PASSED against the live Base
  mainnet contract `0xfEa9aeB9...`. Other invariants fail-soft to
  "skipped" with diagnostic "backend returned None" — the harness
  doesn't crash on selectors the dev RPC client doesn't have. Hit
  /admin/formal-verification/symbolic — returns 5 halmos specs with
  mirrors_runtime_contract + runtime_invariants linkage. The
  symbolic→runtime mapping is queryable. 3 PRSM_Testing.md rows
  promoted/added. Doc-only.
- **2026-05-15 sprint 442** — §14 creator-stake lookup live-verified.
  `GET /marketplace/creator-stake/{id}` returns clean schema with
  balance_wei + high_tier_eligible + min_high_tier_stake_wei +
  commissioned fields. Unknown creator → balance_wei=0 +
  high_tier_eligible=false (correct default). `commissioned: false`
  in dev env reflects the honest-scope: live stake/slash flow needs
  on-chain wallet + StakeBond contract address; lookup surface is
  ready regardless. PRSM_Testing.md row split: lookup ✅, full
  commissioning 🟢 with explicit gate-condition note. Doc-only.
- **2026-05-15 sprint 441** — §14 content-fingerprint dedup chain live-
  verified. Full Sybil-resistance evidence chain works end-to-end:
  1. Upload content with creator A → `content_hash` registered;
     `canonical_creator=A`; `duplicate_of_creator=None`.
  2. Re-upload SAME text with creator B → same `content_hash`;
     `canonical_creator=A` (preserved — first-creator-wins);
     `duplicate_of_creator=A` (correctly flagged).
  3. GET `/marketplace/fingerprint/{hash}` → `duplicate_attempt_count: 1`
     (counter incremented on the re-upload).
  This demonstrates the §14 anti-Sybil invariant operationally:
  identity rotation can't claim canonical creator status for
  content someone else already registered. Three PRSM_Testing.md
  rows promoted 🟢 → ✅ (registry / dedup / counter). Doc-only.
- **2026-05-15 sprint 440** — §14 data-quality reputation surface live
  partial-verification. `/marketplace/creator-reputation/{id}` returns
  clean default for unknown creators (known:false, score:0.5,
  tier:"new") — promoted to ✅. `/content/search?min_tier=X&
  exclude_new=true` accepts the §14 tier-filter query params cleanly
  — promoted to ✅. Auto-record on retrieve REMAINS 🟢 — wired
  correctly but live update is gated by operator wallet config
  (`FTNS_WALLET_PRIVATE_KEY`); dev env can't exercise the full
  reputation-accrual path without a real on-chain wallet. Honest-
  scope deferral documented inline. PRSM_Testing.md row updated with
  the dev-vs-live distinction.
- **2026-05-15 sprint 439** — §14 content-moderation chain E2E.
  Promoted §14 content-moderation rows from 🟢 to ✅ via the live
  end-to-end test:
  1. POST /admin/content-filter/cids → blocklist updated (2 CIDs)
  2. GET /content/retrieve/<blocked-cid> → 451 (RFC 7725 Unavailable
     For Legal Reasons — canonical "policy-blocked" code, not 403/404)
  3. POST /admin/takedown-notice with {target_cid, sender,
     jurisdiction, basis, description} → notice received
  4. POST /admin/content-filter/from-notice/<id> → CID auto-added
     to blocklist; notice status flips to "acknowledged"
  5. GET /admin/content-filter → blocked list now includes the
     bridged CID
  Caught a schema discovery during live test: takedown notice
  requires target_cid+sender+jurisdiction+basis (NOT the older
  filer_name+filer_email+claim_basis names from an earlier draft).
  4 new pin tests including: 451 status code is the canonical
  blocked-content code; notice→filter bridge is explicit
  operator-initiated (no auto-bridge — Vision §14 Foundation-
  never-compels invariant); refuse is the default action_on_match.
  Tag `content-moderation-chain-e2e-merge-ready-20260515`.
- **2026-05-15 sprint 438** — §5.2 inference E2E + F12 fix.
  Promoted §5.2 inference rows from 🟢 to ✅. Wired
  MockInferenceExecutor as opt-in via `PRSM_INFERENCE_EXECUTOR=mock`
  in node.py (zero-filled crypto, MUST NOT trust in prod —
  honest-scope). Live-verified full inference → signed receipt →
  /compute/receipt/verify chain. Caught production-blocker F12:
  mock executor's `_epsilon_for_level(NONE)` returned float("inf");
  JSON serialization mapped Infinity → null; verifier reconstructed
  as 0.0 → signing-payload bytes mismatch → signature_valid=false
  for every NONE-tier inference. Fix: NONE tier uses 0.0 (honest
  semantic — NONE means "no DP applied", so "0 budget consumed"
  is the right encoding). Live-verified: all four privacy tiers
  (none/standard/high/maximum) now pass the verify roundtrip
  cleanly. 4 new pin tests including F12 invariant + env-gate
  documentation pin. PRSM_Testing.md §5.2 rows promoted.
  Tag `inference-mock-executor-epsilon-finite-merge-ready-20260515`.
- **2026-05-15 sprint 437** — Federated + pipeline CLI trifecta
  closures (last two from the priority queue). Added
  `prsm node federated list/details` + `prsm node pipeline list/
  details` (read-only triage; mutating endpoints deferred per
  the sprint-434 incident-CLI pattern). Shared
  `_node_admin_list_details` helper between both groups —
  same shape (GET /admin/<group>/job[?status=X] → records list;
  details endpoint takes job_id). Live-verified: federated list
  returns empty (no active jobs); pipeline list returns 503
  (orchestrator not wired in this env, expected); federated
  details on fake-id → 404 + exit 1. 8 pin tests covering both
  groups: registered; status filter vocabulary documented; help
  text mentions read-only scope; required args enforced.
  Operator-trifecta CLI gap status: ALL FIVE gaps closed (incident /
  insurance / TEE / federated / pipeline). PRSM_Testing.md §13
  "Operator-trifecta gaps" table now has zero CLI gaps.
  Tag `cli-node-federated-pipeline-merge-ready-20260515`.
- **2026-05-15 sprint 436** — TEE policy CLI trifecta closure.
  Added `prsm node tee` group with `status` + `evaluate`
  subcommands. `status` shows this node's effective attestation
  tier (operators pre-screen workload eligibility); `evaluate`
  takes a TEEPolicy JSON file + optional attestation_b64 and
  returns evaluation result. Live-verified: status shows
  effective_tier=none / vendor=unknown / no-blob diagnostic;
  evaluate against a permissive policy returns expected
  loud-fail on empty allowed_vendors (by-design). Broken-JSON
  policy → clean error + exit 1. 7 pin tests including: --policy-
  file required (security footgun guard against default-permissive);
  attestation-b64 optional (pre-flight validation); help text
  documents TEEPolicy schema + pre-screen purpose. §13 row
  promoted: REST ✅ + CLI ✅. Tag
  `cli-node-tee-policy-merge-ready-20260515`.
- **2026-05-15 sprint 435** — Insurance-fund CLI trifecta closure.
  Added `prsm node insurance` group with `status` + `compose-recovery`
  subcommands. compose-recovery PRODUCES the multi-sig-uploadable
  recovery tx but does NOT execute (Vision §14: Foundation Safe
  holds the transfer privilege). Default JSON output so operators
  can pipe directly into safe-cli. Live-verified: status returns
  fund_address + treasury_address; compose-recovery surfaces
  clean "insurance fund address not configured" error in dev env.
  6 new pin tests including invariant test that the CLI help text
  states "does not execute" / "multi-sig must sign". §13 row
  promoted: REST ✅ + CLI ✅. Tag
  `cli-node-insurance-recovery-merge-ready-20260515`.
- **2026-05-15 sprint 434** — Priority #5 partial closure: incident-
  response CLI trifecta gap closed (read-only triage commands).
  Added `prsm node incident` group with three subcommands:
  - `list` — wraps `GET /admin/incident` with severity/phase filters
  - `details <incident_id>` — wraps `GET /admin/incident/{id}` with
    404 handling
  - `playbook` — wraps `GET /admin/incident/playbook` (Vision §14:
    response plan published BEFORE any incident)
  Both text + JSON output formats; severity color-coded
  (s0/s1=red, s2=yellow, s3=cyan). Live-verified against running
  daemon: playbook surfaces canonical decision-tree; list returns
  empty (no active incidents); details 404s cleanly on unknown id.
  Mutating commands (open/advance/log-event) deferred — they need
  more careful input-parameter UX; operators use REST or
  `prsm_incident` MCP for those.
  Pin tests (5 new): group registered; help text surfaces canonical
  severity vocabulary (s0/s1/s2/s3, NOT minor/major/critical which
  would silently 422); --format option present everywhere.
  Tag `cli-node-incident-readonly-triage-merge-ready-20260515`.
- **2026-05-15 sprint 433** — Priority #4 closed: §7 receipt
  verification live-tested end-to-end. Generated ed25519 keypair,
  built `InferenceReceipt` (job + request + model + tier + ε +
  TEE attestation + output hash + duration + cost), signed
  `signing_payload()`, POSTed to `/compute/receipt/verify`.
  - Honest receipt with matching ε for tier → `ok=true`,
    `signature_valid=true`, `reasons=[]`, all 6 §7 truth-surfacing
    fields (DP-noise / hardware-attestation / multi-stage /
    activation-noise-trace / topology-structural / topology-
    distinct) green.
  - Tampered receipt (epsilon flipped 0.5→99.9) → `ok=false`,
    `signature_valid=false`, `reasons=["signature failed
    cryptographic verification", ...]`.
  Confirms §7 truth-surfacing claim works as advertised: callers
  can independently verify the chain of custody on inference
  receipts. `attestation_vendor_verified=false` per the documented
  honest-scope deferral (real DCAP/KDS not wired).
  Pin tests (6 new in test_compute_receipt_verify_e2e.py): honest
  passes; tampered epsilon fails; tampered output_hash fails;
  wrong pubkey fails; attestation vendor honest-scope; signing
  payload excludes signature. PRSM_Testing.md §7 rows promoted.
  Tag `compute-receipt-verify-e2e-merge-ready-20260515`.
- **2026-05-15 sprint 432** — F10 design-review closure + F11 fix.
  F10 (single-node forge blocked by empty aggregator pool) marked
  as design limitation — A2 invariant ("prompter never selects
  itself") is load-bearing security; bypassing it would add a
  production backdoor. Multi-node test bench is the eventual
  right answer.
  F11 (production-blocking): StakingManager.claim_rewards raised
  "can't subtract offset-naive and offset-aware datetimes" on
  every call. Root cause: SQLite drops tz info on datetime
  persistence; `now - last_reward_calculation` mixed aware + naive.
  Fix: `_ensure_utc` helper re-tags naive DB values as UTC
  (sound — writers all use `datetime.now(timezone.utc)`).
  Live-verified end-to-end §5.3 stake → claim flow:
  - Faucet to 1042 FTNS
  - Stake 1000 → `total_staked: 1000.0`
  - Claim → `{"total_rewards_claimed": 0.0, "stakes_processed": 1}`
    (0 reward because stake < 24h min_stake_age, correct behavior)
  PRSM_Testing.md §5.3 staking rows promoted to ✅. 5 new tests /
  88 cross-suite green.
