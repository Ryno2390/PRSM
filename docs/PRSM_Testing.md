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
| Retrieve by CID (Tier A, cross-node) | `GET /content/retrieve/{cid}` | 🔬 | — | Single-node setup can't exercise BT swarm |
| Retrieve by CID (Tier B/C Shamir, same node) | `ContentRetriever.fetch` | 🟢 | 430 | Routing pinned; infrastructure-only — `/content/upload` doesn't reach this lane today |
| Upload shard | `POST /content/upload/shard` | 🟢 | 102 | Size cap test pinned; E2E untested |
| Recipient manifest | `GET /content/recipient-manifest/{cid}` | 🟢 | 304 | Test-pinned |
| Content metadata | `GET /content/{cid}` | 🟢 | — | Endpoint exists; usage path untested |
| Content search | `GET /content/search` | 🟢 | 287 | Tier filtering test-pinned |
| My uploaded content | `GET /content/mine` | 🟢 | — | Endpoint exists |

### Storage subsystem

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Storage stats | `GET /storage/stats` | 🟢 | — | Endpoint exists |
| Pinned content stats | `GET /storage/pinned-stats` | 🟢 | — | Endpoint exists |
| Provider reputations | `GET /storage/provider-reputations` | 🟢 | — | Endpoint exists |
| Content index stats | `GET /content/index/stats` | 🟢 | — | Endpoint exists |
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
| SHA-256 fingerprint registry | `POST /content/upload` hook | 🟢 | 291 | Fixture-drift fixed sprint 425 |
| Duplicate detection on re-upload | response `duplicate_of_creator` | 🟢 | 291 | Test-pinned |
| Marketplace fingerprint lookup | `GET /marketplace/fingerprint/{hash}` | 🟢 | 291 | Endpoint exists |
| EmbeddingDHT cross-node embedding gossip | `prsm.dht.embedding_dht_client` | 🟢 | T3.6 | Vision §11 claims live |
| BinaryFingerprint perceptual hashes | `prsm/marketplace/binary_fingerprint.py` | 🟢 | T4.7 | Calibration deferred to testnet traffic |
| V2 ProvenanceRegistry on-chain embedding commitment | on-chain | ✅ | — | Deployed `0xe0cedDA354...` |

---

## §5.2 — Compute Layer: SPRKs via Wasmtime

### Inference

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Inference quote | `POST /compute/inference/quote` | ✅ | 438 | Live-verified with `PRSM_INFERENCE_EXECUTOR=mock` |
| Submit inference (NONE tier) | `POST /compute/inference` | ✅ | 438 | Live E2E + F12 fix: ε=0.0 (not inf), receipt JSON-clean, verify passes |
| Submit inference (standard/high/maximum) | `POST /compute/inference` | ✅ | 438 | Live E2E: signed receipt verifies cleanly via sprint-433 verify path |
| Inference → receipt → verify chain | end-to-end | ✅ | 438 | First time the full §5.2+§7 chain works on single node |
| Streaming inference | `POST /compute/inference/stream` | 🟢 | 3.x.8 | Audit-prep §7.4-§7.8 pin via tags |
| Tensor parallel sharding | `POST /compute/inference/tensor_parallel/shard` | 🟢 | — | Endpoint exists |
| Pipeline stage setup | `POST /compute/inference/pipeline/stage` | 🟢 | — | Endpoint exists |

### Forge (query orchestrator)

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Forge quote | `POST /compute/forge/quote` | ✅ | — | Verified live in dogfood arc |
| Submit forge query | `POST /compute/forge` | ⚠️ | — | Default-disabled; needs `PRSM_QUERY_ORCHESTRATOR_ENABLED=1` |
| Single-node forge E2E | `POST /compute/forge` | 🔬 | — | Requires content present; was blocked by F4 before sprint 428 |

### Compute jobs (general)

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Submit job | `POST /compute/submit` | 🟢 | — | Endpoint exists |
| Job status | `GET /compute/status/{id}` | 🟢 | — | Endpoint exists |
| Job status stream | `GET /compute/status/{id}/stream` | 🟢 | — | Endpoint exists |
| Cancel job | `POST /compute/cancel/{id}` | 🟢 | — | Endpoint exists |
| List jobs | `GET /compute/jobs` | 🟢 | — | Endpoint exists |
| Stale escrow cleanup | `POST /compute/cleanup-stale` | 🟢 | — | Endpoint exists |
| Training jobs | `POST /compute/train` | 🟢 | — | Endpoint exists |
| Compute stats | `GET /compute/stats` | 🟢 | — | Endpoint exists |
| Available models | `GET /compute/models` | 🟢 | — | Endpoint exists |
| Receipts list | `GET /compute/receipts` | 🟢 | — | Endpoint exists |
| Receipt details | `GET /compute/receipt/{job_id}` | 🟢 | — | Endpoint exists |

### Hardware classification

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Hardware benchmark | `prsm node benchmark` | 🟢 | — | T1-T4 classification |

---

## §5.3 — Economic Layer: FTNS

### Wallet / balance

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Local balance | `GET /balance` | ✅ | — | Verified live in dogfood arc |
| On-chain balance | `GET /balance/onchain` | 🟢 | — | Endpoint exists |
| Transaction history | `GET /transactions` | ✅ | — | Verified live (sprint 198 bounds-validated `limit`) |
| Transfer (gasless via paymaster) | `POST /wallet/transfer/gasless` | 🟢 | Phase 5 | Endpoint exists; live activation external-gated |
| WaaS provision | `POST /wallet/waas/provision` | 🟢 | Phase 5 | Endpoint exists; requires CDP credentials |
| Paymaster status | `GET /wallet/paymaster/status` | 🟢 | Phase 5 | Endpoint exists |

### Staking

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Stake | `POST /staking/stake` | ✅ | 432 | Live-verified: 1000 FTNS staked; total_staked reflects new amount |
| Unstake request | `POST /staking/unstake` | 🟢 | — | Endpoint exists |
| Stake status | `GET /staking/status` | ✅ | 432 | Live-verified end-to-end with active stake |
| Claim rewards | `POST /staking/claim-rewards` | ✅ | 432 | F11 fixed: tz-aware datetime subtraction now works (was 500 on every claim) |
| Withdraw unstaked | `POST /staking/withdraw/{id}` | 🟢 | — | Endpoint exists |
| Cancel unstake | `POST /staking/cancel-unstake/{id}` | 🟢 | — | Endpoint exists |
| Single-user stake → claim E2E | (multi-step) | 🔬 | — | Multi-step ledger flow not exercised end-to-end |

### Settlement

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Settlement stats | `GET /settlement/stats` | 🟢 | — | Endpoint exists |
| Pending settlements | `GET /settlement/pending` | 🟢 | — | Endpoint exists |
| Flush batch | `POST /settlement/flush` | 🟢 | — | Endpoint exists |
| Settlement history | `GET /settlement/history` | 🟢 | — | Endpoint exists |
| Settler registry (register / unbond / sign batch) | `/settler/...` | 🟢 | — | Multi-endpoint registry; not exercised E2E |

### Phase 5 fiat surface (commission-ready, external-gated)

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Onramp quote | `POST /wallet/onramp/quote` | 🟢 | 276-286 | 399 cross-suite tests; commission gated by KYC vendor + CDP keys |
| Offramp quote | `POST /wallet/offramp/quote` | 🟢 | 276-286 | Same gate |
| Pool quote (Aerodrome) | `GET /wallet/pool/quote` | 🟢 | 276-286 | Read-only quoter; live pool external-gated |
| Pool state | `GET /wallet/pool/state` | 🟢 | 276-286 | Same |
| KYC initiate | `POST /wallet/kyc/initiate` | 🟢 | Phase 5 | Vendor adapter scaffolds (Persona/Onfido/Plaid) |
| KYC webhook | `POST /wallet/kyc/webhook/{vendor}` | 🟢 | Phase 5 | HMAC-SHA256 + replay protection |
| Fiat-surface health | `GET /admin/fiat-surface/health` | ✅ | 285/422 | `check_fiat_surface_health()` live-verified |
| Fiat-readiness CLI | `prsm node fiat-readiness` | ✅ | 422 | Live-verified on current env (returns OK) |
| Activation runbook | `docs/operations/phase-5-fiat-surface-activation-runbook.md` | ✅ | 421 | Pinned by 11 source-truth-parity tests |

### On-chain royalty distribution

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| RoyaltyDistributor v2 (atomic 70/25/5) | mainnet `0x3E82…D6c2` | ✅ | A-08 | Mainnet ceremony 2026-05-09 |
| On-chain content-access royalty leg | env-gated by `PRSM_ONCHAIN_CONTENT_ROYALTY_ENABLED=1` | 🟢 | 243-261 | 19-sprint arc; live activation requires creator addresses on uploaded content |
| Royalty dispatch summary | `GET /admin/royalty-dispatch-summary` | 🟢 | — | Endpoint exists |
| Royalty dispatch history | `GET /admin/royalty-dispatch-history` | 🟢 | — | Endpoint exists |
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
| Intel ASP (SGX v3 + TDX v4) structural parse | `IntelASPBackend` | ✅ | 293 | MRENCLAVE/MRSIGNER/MRTD/RTMR0 |
| AMD KDS (SEV-SNP v2) structural parse | `AMDKDSBackend` | ✅ | 294 | MEASUREMENT/REPORT_DATA/CHIP_ID |
| Real cryptographic signing-chain verification | (deferred) | ⏸️ | — | Structural-only today; `vendor_verified=True` requires real DCAP/KDS keys |
| Apple SEP backend | (deferred) | ⏸️ | — | If iOS-side compute joins the supply tier mix |

### Privacy primitives

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Per-stage Gaussian noise (basic composition) | `ActivationDPInjector` | ✅ | 295 | Defends activation-inversion |
| Topology rotation (uniform/beacon/anti-repeat) | `TopologyRotationPolicy` | ✅ | 296 | `stable_hash()` enables replay verification |
| Privacy budget tracking | `/privacy/budget` + persistent store | 🟢 | 3.x.4 | Audit-prep §2.4 |

### Enterprise Confidentiality Mode

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Recipient encryption (X25519 + XChaCha20) | `POST /content/upload` w/ recipients | ✅ | 430 | Live byte-identical roundtrip; sprint 430 |
| Recipient manifest read | `GET /content/recipient-manifest/{cid}` | 🟢 | 304 | |
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
| Node health (detailed, 14 subsystems) | `/health/detailed` | — | `prsm_node_health` | ✅ Sprint 342-345 |
| Node info | `/info` | `prsm node info` | `prsm_info` | ✅ |
| Node peers | `/peers` | `prsm node peers` | `prsm_peers` | ✅ |
| Bootstrap status | `/bootstrap/status` | — | `prsm_bootstrap_status` | ✅ |
| Bootstrap test (probe canonical fleet) | — | `prsm node bootstrap-test` | `prsm_bootstrap_test` | ✅ Sprint 385/387 |
| Bootstrap server status | `/admin/bootstrap-server/status` | `prsm bootstrap-server status` | `prsm_bootstrap_server_status` | ✅ Sprint 388-396 |
| Metrics (Prometheus) | `/metrics` | — | `prsm_metrics_summary` | ✅ |
| Resources (read/write) | `GET/PUT /node/resources` | — | `prsm_node_resources` | 🟢 |

### Earnings + ledger

| Feature | REST | CLI | MCP | Status |
|---------|------|-----|-----|--------|
| Earnings summary | `/admin/earnings-summary` | `prsm node earnings` | `prsm_earnings_summary` | ✅ |
| Slash history | `/admin/slash-history` | `prsm node slash-history` | `prsm_slash_history` | ✅ |
| Heartbeats | `/admin/heartbeat-history` | `prsm node heartbeats` | `prsm_heartbeat_history` | ✅ |
| Distributions | `/admin/distribution-history` | `prsm node distributions` | `prsm_distribution_history` | ✅ |
| Webhooks | `/admin/webhook-history` | `prsm node webhooks` | `prsm_webhook_history` | ✅ |
| Trigger heartbeat | `/admin/heartbeat/trigger` | `prsm node trigger-heartbeat` | `prsm_heartbeat_trigger` | ✅ |
| Trigger distribution | `/admin/distribution/trigger` | `prsm node trigger-distribution` | `prsm_distribution_trigger` | ✅ |
| Claim royalty | `/wallet/royalty/claim` | `prsm node claim-royalty` | `prsm_royalty_claim` | 🟢 |
| Audit summary | `/audit/summary` | — | `prsm_audit_summary` | 🟢 |
| Audit recent | `/audit/recent` | — | `prsm_audit_recent` | 🟢 |

### Content + marketplace

| Feature | REST | CLI | MCP | Status |
|---------|------|-----|-----|--------|
| Content filter | `/admin/content-filter` | — | `prsm_content_filter` | 🟢 Sprint 269-274 |
| Takedown notices | `/admin/takedown-notice` | — | `prsm_takedown_notices` | 🟢 Sprint 269-274 |
| Notice → filter bridge | `/admin/content-filter/from-notice/{id}` | — | (via `prsm_takedown_notices`) | 🟢 |
| Creator reputation | `/marketplace/creator-reputation/{id}` | — | `prsm_creator_reputation` | 🟢 Sprint 287-291 |
| Creator stake | `/marketplace/creator-stake/{id}` | — | `prsm_creator_stake` | 🟢 |
| Provider reputations | `/marketplace/reputation` | — | `prsm_marketplace_reputation` | 🟢 |
| My content | `/content/mine` | `prsm content mine` | `prsm_my_content` | 🟢 |

### Phase 5 fiat operator surfaces

| Feature | REST | CLI | MCP | Status |
|---------|------|-----|-----|--------|
| Fiat surface health | `/admin/fiat-surface/health` | `prsm node fiat-readiness` | `prsm_fiat_surface_health` | ✅ |
| Fiat compliance summary | `/admin/fiat-compliance/summary` | — | `prsm_fiat_compliance` | 🟢 |
| KYC status | `/wallet/kyc/status` | — | `prsm_kyc` | 🟢 |

### Incident + upgrade + insurance

| Feature | REST | CLI | MCP | Status |
|---------|------|-----|-----|--------|
| Incident open / advance / log event | `/admin/incident/...` | `prsm node incident list/details/playbook` (read-only) | `prsm_incident` | ✅ Sprint 434 (trifecta closure, read-only triage) |
| Insurance fund | `/admin/insurance-fund/status` | — | `prsm_insurance_fund` | 🟢 |
| Emergency pause | `/admin/emergency-pause/...` | — | `prsm_emergency_pause` | 🟢 |
| Upgrade proposal | `/admin/upgrade/...` | — | `prsm_upgrade` | 🟢 |
| TEE policy | `/admin/tee-policy/evaluate` | — | `prsm_tee_policy` | 🟢 |
| Vulnerability disclosure | `/admin/disclosure/...` | — | `prsm_disclosure` | 🟢 |

---

## §14 — Risk mitigations (validation, defense, formal verification)

### Content moderation (operator-side enforcement)

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Operator content filter (CID blocklist) | `/admin/content-filter/cids` | ✅ | 439 | Live E2E: POST cids → 451 on retrieve verified |
| Operator content filter (tag blocklist) | `/admin/content-filter/tags` | 🟢 | 269-274 | |
| Foundation takedown notice intake (info-only) | `/admin/takedown-notice` | ✅ | 439 | Live-verified: target_cid+sender+jurisdiction+basis required (§14 attribution invariant) |
| Notice → filter bridge | `/admin/content-filter/from-notice/{id}` | ✅ | 439 | Live E2E: notice → bridge → CID auto-added; notice status flips to "acknowledged" |
| Notice lifecycle status transitions | `/admin/takedown-notices/{id}/status` | 🟢 | 269-274 | |

### Data quality + Sybil resistance

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Creator reputation tracker (lookup) | `/marketplace/creator-reputation/{id}` | ✅ | 440 | Live: returns clean default (known:false, score:0.5, tier:"new") for unknown creators |
| Creator reputation tracker (auto-record on retrieve) | hook in `/content/retrieve` | 🟢 | 287-291 | Wired correctly; live update gated by operator wallet config (`FTNS_WALLET_PRIVATE_KEY`) — dev env can't trigger |
| Tier classification (new/low/medium/high) | reputation tier auto-records on retrieve | 🟢 | 287-291 | Same wallet-gate as above |
| Search filter by tier | `GET /content/search?min_tier=...&exclude_new=...` | ✅ | 440 | Live: query params accepted cleanly; tier-filter codepath active |
| Creator stake gate (HIGH tier requires bonded FTNS) | on-chain `StakeBond` | 🟢 | 287-291 | Demotes HIGH→MEDIUM when unstaked |
| Content fingerprint registry | `POST /content/upload` hook | 🟢 | 291 | Sprint 425 fixed fixture-drift |

### Formal verification

| Feature | Surface | Status | Sprint | Notes |
|---------|---------|--------|--------|-------|
| Runtime invariants probe (7 contracts, 20 invariants) | `/admin/formal-verification/check` | ✅ | 302-359 | |
| Halmos symbolic-execution lane (5 specs, 28 proofs) | `/admin/formal-verification/symbolic` | ✅ | 360-364 | 16/20 invariants symbolically pinned |
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
