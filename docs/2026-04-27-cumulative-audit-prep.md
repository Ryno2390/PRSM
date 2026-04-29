# PRSM Mainnet Audit — Cumulative Refresh (2026-04-27)

**Date:** 2026-04-27
**Audit tag:** `cumulative-audit-prep-20260427` (pins commit `107fb150`)
**Supersedes:**
- `phase7.1x-audit-prep-20260422-2` (Phase 7/7.1/7.1x — economic substrate; still valid for that surface, now extended)

This bundle refreshes the audit scope to include **all engineering work that landed between 2026-04-22 and 2026-04-27**: five new sub-phases shipped, one new Solidity contract (`PublisherKeyAnchor.sol`), and a new HTTP API surface (`wallet_api.py`) with a JS SDK helper that drives wallet onboarding. ~161 commits.

**If you're starting the audit engagement, this is the document to begin with.** Prior bundles (`docs/2026-04-22-phase7.1x-audit-prep.md`) remain authoritative for their respective surfaces; this one stacks on top with the additions.

---

## 1. What changed since `phase7.1x-audit-prep-20260422-2`

| Phase | Tag | Surface |
|-------|-----|---------|
| **3.x.2** | `phase3.x.2-merge-ready-20260426` | Persistent model registry — `FilesystemModelRegistry` w/ Ed25519-signed `ModelManifest` |
| **3.x.3** | `phase3.x.3-merge-ready-20260427` | Publisher Key Anchor — **new on-chain contract** + Python client + verifier wrappers |
| **3.x.4** | `phase3.x.4-merge-ready-20260427` | Persistent privacy budget — chained-per-entry signed journal |
| **3.x.5** | `phase3.x.5-merge-ready-20260427` | Manifest DHT — cross-node manifest distribution w/ mandatory anchor verification |
| **Phase 4 Task 3** | (no tag — Task 4/6 deferred) | `wallet_api.py` HTTP surface + JS SDK `WalletAuth` helper |

**Each phase passed two-round independent code review pre-tag.** Round-1 findings (HIGHs + MEDIUMs) were remediated before tagging; round-2 confirmed SAFE-TO-DEPLOY. Findings of note caught at the gate (and resolved):

- **3.x.2** — HIGH path-traversal in `FilesystemModelRegistry._model_dir`; remediated via `_RESERVED_NAMES` rejection + `is_relative_to(root)` defense-in-depth.
- **3.x.3** — HIGH bytes-input bypass in `PublisherKeyAnchorClient.lookup` accepting `bytes` instead of `str`; remediated with explicit `isinstance` check.
- **3.x.4** — HIGH negative-ε credit-back attack on parent class `record_spend` accepting negative budget; remediated with `math.isfinite + epsilon > 0` guard.
- **3.x.5** — HIGH manifest substitution attack: validly-signed manifest under wrong `model_id` accepted by anchor verify; remediated with explicit `manifest.model_id == requested` check before anchor verify in `ManifestDHTClient.get_manifest`. Plus HIGH server-never-raises invariant gap and HIGH `AnchorRPCError` leak.
- **Phase 4 Task 3** — HIGH SIWE no-statement layout violating EIP-4361 (silent login break); HIGH lowercase-address rejection by Python `siwe` library when JS provider returned non-checksummed; both remediated with two-blanks layout + `toChecksumAddress` keccak256 normalization.

All listed findings are CLOSED in the tagged tree; the audit engagement starts from a known-clean baseline.

---

## 2. Scope for this audit (additive)

### 2.1 Solidity (NEW)

**`contracts/contracts/PublisherKeyAnchor.sol`** (124 LoC, Phase 3.x.3 Task 1)

Write-once on-chain registry binding `node_id_bytes16 → 32-byte Ed25519 public key`. Anchors cross-node trust for any artifact-type that includes a `publisher_node_id` field — currently consumed by `ModelManifest` (3.x.2), `PrivacyBudgetEntry` (3.x.4), `InferenceReceipt` (3.x.1), and DHT-fetched manifests (3.x.5). Behavior:

- `register(bytes publicKey)` — compute `node_id = sha256(publicKey)[:16]`, refuse if `_publisherKeys[node_id] != 0` (write-once), store. Emits `PublisherRegistered`.
- `lookup(bytes16 node_id) → bytes` — read-only.
- `adminOverride(bytes16 node_id, bytes newKey)` — multi-sig-only escape hatch for compromised keys. Emits `PublisherKeyOverridden`.

**Hardhat tests:** 20 cases in `contracts/test/PublisherKeyAnchor.t.sol` covering the three operations + admin auth + invalid-length rejection + write-once enforcement.

**Auditor priorities:**
1. Confirm `sha256(publicKey)[:16]` is the right derivation rule. We use sha256 (not keccak256) to match the off-chain Python `NodeIdentity.node_id` derivation in `prsm/node/identity.py:107`. Cross-language byte-equality is the load-bearing invariant. Drift here breaks every artifact-verifier wrapper.
2. `adminOverride` is the only path that writes a non-virgin slot. Confirm the multi-sig admin guard is tight — single-sig admin would let the contract owner silently reattribute an artifact to a substituted key.
3. Greenfield deploy. No existing on-chain state to migrate.
4. Sepolia broadcast already executed operator-side (Phase 3.x.3 Task 2 runbook); Base mainnet bundles into this audit's deploy ceremony.

**Other contracts** (unchanged since their last audit-prep):
- `BatchSettlementRegistry.sol`, `StakeBond.sol`, `EscrowPool.sol`, `Ed25519Verifier.sol`, `EmissionController.sol`, `CompensationDistributor.sol`, `KeyDistribution.sol`, `StorageSlashing.sol`, `BridgeSecurity.sol`, `FTNSBridge.sol`, `FTNSTokenSimple.sol`, `ProvenanceRegistry.sol`, `RoyaltyDistributor.sol` — all already covered by `phase7.1x-audit-prep-20260422-2` or earlier engagements. No diff since.

### 2.2 Python — Phase 3.x.2 Persistent Model Registry

**Files:**
- `prsm/compute/model_registry/models.py` — `ModelManifest` + `ManifestShardEntry` frozen dataclasses; canonical signing payload.
- `prsm/compute/model_registry/signing.py` — Ed25519 sign/verify wrappers.
- `prsm/compute/model_registry/registry.py` — `ModelRegistry` ABC + `InMemoryModelRegistry` + `FilesystemModelRegistry` (path-traversal-defended; `anchor=` and `dht=` kwargs added in 3.x.3 + 3.x.5).

**Trust boundary:** the registry root is a local trust boundary. An attacker with write access to `<root>/<model_id>/` can replace the manifest. Cross-node verification requires `anchor=` (Phase 3.x.3) which delegates pubkey resolution to the on-chain anchor instead of the on-disk sidecar.

**Auditor priorities:**
1. Path-traversal: `_validate_fs_id` regex + `_RESERVED_NAMES` rejection + `is_relative_to` post-resolve check. Confirm no third path exists where a caller-supplied identifier becomes a filesystem path without these gates.
2. Manifest-Ed25519-sign payload is canonical-JSON-of-non-signature-fields with deterministic key ordering. Confirm sign-then-verify roundtrips byte-equal.
3. Atomicity: write order is `shards/*.bin → publisher.pubkey → manifest.json`, all atomic via `.tmp` + `os.replace` + `fsync`. Manifest-last so a crashed write leaves either a complete model or no visible registration.

### 2.3 Python — Phase 3.x.3 Publisher Key Anchor

**Files:**
- `prsm/security/publisher_key_anchor/client.py` — `PublisherKeyAnchorClient` (web3.py wrapper) with negative-cache + retry handling.
- `prsm/security/publisher_key_anchor/exceptions.py` — `PublisherKeyAnchorError` hierarchy.
- `prsm/security/publisher_key_anchor/verifiers.py` — `verify_manifest_with_anchor`, `verify_entry_with_anchor`, `verify_receipt_with_anchor`.
- `scripts/deploy_publisher_key_anchor.py` — Sepolia/Base deploy helper.

**Auditor priorities:**
1. Negative-cache: when `_call_lookup` returns empty bytes (publisher not registered), the client caches the negative for `_NEGATIVE_TTL` seconds. Confirm the TTL is short enough that a publisher registering BETWEEN cache-miss and cache-hit doesn't get a stale "not registered" verdict on a security-critical path.
2. The `lookup` method's input handling: round-1 review caught a bytes-input bypass. Confirm the current `isinstance(node_id, str)` guard at line ~XX is the FIRST operation, not behind any `.lower()` or other coercion that bytes also support.
3. Verifier wrappers raise `AnchorRPCError` on transport-level failures — distinct from `False` (which means signature failed verification). Callers must distinguish "anchor unreachable" (transient, retryable) from "signature invalid" (terminal, drop bytes). The `ManifestDHTClient.get_manifest` per-provider loop catches `AnchorRPCError` and continues to the next provider; confirm equivalent handling in any other consumer.

### 2.4 Python — Phase 3.x.4 Persistent Privacy Budget

**Files:**
- `prsm/security/privacy_budget_persistence/models.py` — `PrivacyBudgetEntry` frozen dataclass.
- `prsm/security/privacy_budget_persistence/signing.py` — chained Ed25519 signatures (each entry's payload includes `prev_hash`).
- `prsm/security/privacy_budget_persistence/store.py` — `PrivacyBudgetStore` ABC + `InMemoryPrivacyBudgetStore` + `FilesystemPrivacyBudgetStore` (atomic append + `anchor=` kwarg).
- `prsm/security/privacy_budget_persistence/tracker.py` — `PersistentPrivacyBudgetTracker` enforcing per-user epsilon budget over the journal.

**Auditor priorities:**
1. Chained-per-entry signing: each entry's `signing_payload` includes a `prev_hash` field that's the sha256 of the previous entry's canonical bytes. A tampered intermediate entry breaks the chain at every subsequent verification. Confirm the chain genesis (`prev_hash = GENESIS_PREV_HASH = 0x00…00`) is strictly enforced for the FIRST entry only.
2. Negative-ε credit-back: round-1 review caught the parent class `record_spend` accepting negative `epsilon` (which would credit budget back). Now guarded with `math.isfinite + epsilon > 0` at the parent ABC level, so all subclasses inherit the protection. Confirm no sibling override accidentally bypasses.
3. RESET entries have no `epsilon_spent` (sentinel `RESET_EPSILON = 0`); they're audit-trail markers, not budget operations. Confirm a malicious RESET cannot zero the cumulative spend mid-journal — the `replay_journal` reconstruction must preserve the running total across RESET boundaries OR start a fresh budget window per RESET (whichever is correct per design).

### 2.5 Python — Phase 3.x.5 Manifest DHT

**Files:**
- `prsm/network/manifest_dht/protocol.py` — JSON wire format, 5 message dataclasses, `MAX_MESSAGE_BYTES = 256 KB`, `MAX_PROVIDERS_PER_RESPONSE = 1024`.
- `prsm/network/manifest_dht/local_index.py` — per-node servable-manifests index w/ rebuild-from-walk + orphan reconciliation.
- `prsm/network/manifest_dht/dht_client.py` — `ManifestDHTClient` w/ mandatory `anchor=` (RuntimeError if None); single-round Kademlia find_providers; substitution-defended get_manifest (model_id assertion before anchor verify).
- `prsm/network/manifest_dht/dht_server.py` — `ManifestDHTServer.handle(bytes) → bytes` that NEVER raises; outer try/except wraps all dispatch; `UNKNOWN_REQUEST_ID` sentinel for unparseable input.

**Auditor priorities:**
1. **Manifest substitution defense** (round-1 HIGH-2): verify the `manifest.model_id != model_id` check at `dht_client.py:377` runs BEFORE `verify_manifest_with_anchor`. A legitimately-signed manifest under a different model_id passes anchor verify but must be rejected at this layer.
2. **Anchor verification mandatory at construction**: client constructor raises `RuntimeError` if `anchor=None` or `anchor` lacks `.lookup`. There is NO trust-the-network mode. Confirm no code path constructs a client without the anchor (grep for `ManifestDHTClient(`).
3. **Server never-raises invariant**: `handle()`'s outer try/except (lines ~143-169) covers the entire dispatch including handler internals. UnicodeDecodeError added to manifest read. Confirm no path through `_handle_*` can escape the outer catch.
4. **Size bounds**: `MAX_MESSAGE_BYTES` gate at top of `parse_message` fires before `json.loads` allocates. `MAX_PROVIDERS_PER_RESPONSE` cap on `ProvidersResponse.from_dict` after JSON parse but before per-element `ProviderInfo.from_dict` allocation. Confirm both fire before any unbounded allocation.
5. **Cache-and-announce**: `FilesystemModelRegistry._fetch_manifest_via_dht` caches the manifest THEN announces on the DHT. If announce fails, `LocalManifestIndex._load_or_rebuild`'s orphan reconciliation auto-recovers on next process restart. Confirm the recovery path is correct AND that the in-process announce-failure logs surface to operators.

### 2.6 Python — Phase 4 Task 3 wallet_api

**File:** `prsm/interface/api/wallet_api.py`

HTTP API surface for the shipped Phase 4 backend modules (SIWE verifier Task 1, wallet binding Task 2, USD display Task 5). Per design plan §5.2; never landed in Tasks 1/2/5. Routes: `POST /siwe/nonce`, `POST /siwe/verify`, `POST /bind`, `GET /binding`, `GET /balance`. Service injection via FastAPI `Depends` + race-safe `set_services` boot hook.

**Auditor priorities:**
1. **Replay protection**: nonce store `consume()` happens AFTER all SIWE invariants verify. Failed verifies leave the nonce live for retry. Confirm the consume call is the COMMIT point, not pre-consumed before signature check.
2. **Binding signature recovery**: `WalletBindingService.bind` recovers the signer from the EIP-191 binding message and asserts equality with the claimed wallet address. Confirm there's no path where a different address is stored than was attested.
3. **Idempotent re-bind vs conflict**: re-binding `(wallet, same_node_id)` returns the existing record unchanged WITHOUT verifying the fresh signature. Re-binding `(wallet, different_node_id)` raises `BindingConflictError`. Confirm the conflict-vs-idempotent dispatch is exhaustive (no path where a different wallet/node combo silently overwrites).
4. **Default in-memory stores**: `WalletApiServices.default_for_dev` ships `InMemoryNonceStore` + `InMemoryWalletBindingStore`. Multi-worker deployments break silently — nonce issued by worker 1 fails consume on worker 2. The docstring is explicit; confirm the operator-runbook escalates this to a deployment-checklist gate (Redis nonce store, durable binding store).

### 2.7 JavaScript — `sdks/javascript/src/wallet-auth.ts`

**Surface:** `WalletAuth` class wrapping the SIWE+binding handshake; provider-agnostic (accepts any EIP-1193 `request({method, params})` provider as a peer dep). `connectCoinbaseWallet()` is the composed flow.

**New runtime dependency:** `js-sha3` (CommonJS-friendly; used only for EIP-55 keccak256 in `toChecksumAddress`).

**Auditor priorities:**
1. **EIP-55 normalization** (round-1 HIGH-2): `toChecksumAddress` at line ~XX runs unconditionally on every EIP-1193-returned address before SIWE message construction. WalletConnect/Privy lowercase addresses no longer break login. Verify against EIP-55 reference vectors (covered in `tests/wallet-auth.test.ts`).
2. **No-statement SIWE layout** (round-1 HIGH-1): TWO blank lines emitted between address and URI even when `statement` is omitted. Confirms backend Python `siwe` library parses the no-statement form. Round-trip pinned in `test_wallet_api.py::TestSiweCrossLanguageRoundTrip`.
3. **No leak of credentials in error paths**: `WalletAuthError.message` may include backend `detail.message` text. Confirm no backend code path returns the user's wallet address, signature, or nonce in error messages (sensitive-logging review section §3 below).
4. **`globalThis.fetch` binding**: `fetch.bind(globalThis)` at line ~189 — required for browser environments where unbound fetch references throw. Harmless on Node 18+. Confirm no breaking interaction with custom `fetchFn` injection.

---

## 3. Cross-cutting threat model

### 3.1 The shared `anchor=` invariant

Phase 3.x.3 / 3.x.4 / 3.x.5 all consume `anchor: PublisherKeyAnchorClient` (or any `.lookup(node_id) → Optional[str]`-shaped duck-typed equivalent). Trust upgrade is uniform: when `anchor=` is set, the on-chain pubkey replaces the on-disk sidecar as the verification anchor.

**The load-bearing invariant:** every consumer that accepts `anchor=` MUST honor it on the read path. Audit each consumer's `get`/`verify`/`fetch_manifest`/`replay_journal` for a code path where `anchor=` is set but the verification falls back to sidecar/embedded-pubkey. Such a fallback is a silent trust-bypass.

### 3.2 Cross-language byte equality

Three derivations have to be byte-equal across Python and Solidity:
1. `node_id = sha256(publicKey)[:16]` — Python at `prsm/node/identity.py:107`; Solidity at `PublisherKeyAnchor.sol`'s `register` function. Round-trip is sanity-tested in `tests/integration/test_publisher_key_anchor_e2e.py::test_register_derives_node_id_from_pubkey`.
2. SIWE message format — Python `siwe` library round-trip vs JS `buildSiweMessage`. Pinned in `tests/unit/test_wallet_api.py::TestSiweCrossLanguageRoundTrip`.
3. Canonical JSON for `ModelManifest.signing_payload` — Python writes, anyone with the public key verifies. Sort order and float-format must be deterministic.

### 3.3 Sensitive-data logging review

A grep across the new modules for `logger.info|logger.debug|logger.warning` containing wallet addresses, signatures, nonces, or private keys. Findings:
- `prsm/interface/api/wallet_api.py` — no logger calls. ✓
- `sdks/javascript/src/wallet-auth.ts` — no console.log/console.warn calls. ✓
- `prsm/network/manifest_dht/dht_server.py` — `logger.warning` includes `manifest_path` (operator-controlled, not user-controlled). ✓
- `prsm/security/publisher_key_anchor/client.py` — RPC calls log `node_id` (public). Does not log private keys or signatures. ✓

Cross-cutting verdict: no sensitive-data leakage in the new surface.

---

## 4. Test coverage at the tag

Cumulative since `phase7.1x-audit-prep-20260422-2`:

| Phase | Unit tests | Integration tests | Notes |
|-------|-----------|-------------------|-------|
| 3.x.2 | ~80 | + e2e | `test_model_registry.py`, `test_model_registry_exports.py` |
| 3.x.3 | ~70 | 25 cross-publisher e2e | `test_publisher_key_anchor*.py`, `test_publisher_key_anchor_e2e.py` |
| 3.x.4 | ~60 | + node integration | `test_privacy_budget_persistence*.py` |
| 3.x.5 | 263 | 12 e2e (3 simulated nodes) | `test_manifest_dht_*.py`, `test_manifest_dht_e2e.py` |
| Phase 4 Task 3 | 26 (Python) + 21 (JS) | — | `test_wallet_api.py`, `wallet-auth.test.ts` |

All tags include the round-2 review confirmation; HIGH/MEDIUM findings are CLOSED in the tagged tree.

---

## 5. Known issues + auditor prompts

### 5.1 Phase 3.x.5 — single-round Kademlia (v1)

The `find_providers` query is single-round (asks K closest peers once; doesn't iterate by asking responders for THEIR closest peers). Sufficient for the small operator set per design plan §1.1 but not cryptographically sound under heavy churn or adversarial routing tables. v2 iterative refinement is deferred.

**Auditor prompt:** is single-round acceptable for the cross-node trust model, given that anchor verification catches any provider's tampered bytes?

### 5.2 Phase 3.x.5 — manifest-only DHT

Shards are NOT distributed via this DHT (per design plan §1.2). A peer that gets a manifest via DHT but doesn't have shards locally cannot fully reconstruct the model. Phase 7-storage handles shard distribution.

**Auditor prompt:** confirm the registry's `get(model_id)` correctly fails closed (raises `ManifestVerificationError` at shard-existence check) when a DHT-fetched manifest is cached but shards are unavailable.

### 5.3 Phase 4 — Task 4 deferred

Task 4 (embedded-wallet vendor — Privy per §8.1 vendor-decision memo) is gated on Foundation operational items (G1-G4 from `docs/2026-04-22-phase4-wallet-vendor-decision.md` §6). Not in scope for this audit; will bundle into a future audit-prep when the vendor contract signs.

### 5.4 Phase 4 — production deployment caveat

`WalletApiServices.default_for_dev` returns in-memory stores. Production deployments MUST build their own services with shared backends (Redis nonce store, SQLite/Postgres binding store). The docstring is explicit; the operator runbook should verify this gate at boot.

---

## 6. Engagement plan (additive)

The 2026-04-22 bundle's engagement plan stands. This refresh ADDS:

1. **One Solidity contract**: `PublisherKeyAnchor.sol` (124 LoC). Estimated audit effort: ~1 day given write-once+lookup+admin pattern is well-understood.
2. **Two new artifact-verifier surfaces**: `verify_manifest_with_anchor` and `verify_entry_with_anchor` paths (Phase 3.x.3 wrappers). Estimated: ~1.5 days for cross-cutting trust-upgrade-invariant verification.
3. **One new HTTP API surface**: `wallet_api.py` (370 LoC) + JS SDK (280 LoC). Estimated: ~1.5 days for SIWE replay/binding/conflict-path verification.
4. **One new P2P protocol surface**: manifest DHT (1.2K LoC across protocol/index/client/server). Estimated: ~3 days for substitution-defense + never-raises + cache-propagation correctness.

**Total additive effort estimate: ~7 audit days.** Combined with the existing 2026-04-22 bundle, total engagement scope is on the order of 3-4 weeks for a single auditor.

---

## 7. Bundle artifacts at the tag

- **Tree:** `cumulative-audit-prep-20260427` (commit `107fb150`, identical tree to `phase4-task3-...` if tagged separately)
- **Per-phase merge-ready tags** (all included):
  - `phase3.x.2-merge-ready-20260426`
  - `phase3.x.3-merge-ready-20260427`
  - `phase3.x.4-merge-ready-20260427`
  - `phase3.x.5-merge-ready-20260427`
- **Design plans** (all under `docs/`):
  - `2026-04-26-phase3.x.2-persistent-model-registry-design-plan.md`
  - `2026-04-27-phase3.x.3-publisher-key-anchor-design-plan.md`
  - `2026-04-27-phase3.x.4-persistent-privacy-budget-design-plan.md`
  - `2026-04-27-phase3.x.5-manifest-dht-design-plan.md`
  - `2026-04-22-phase4-wallet-sdk-design-plan.md`
  - `2026-04-22-phase4-wallet-vendor-decision.md`

---

## 7.1 Third-party-derived components (Phase 3.x.6 only)

Phase 3.x.6 introduces the project's first vendored third-party code
path. Auditors should treat the vendor boundary as a distinct trust
seam.

**Vendor scope.** `prsm/compute/parallax_scheduling/` contains source
copied + modified from [GradientHQ/parallax](https://github.com/GradientHQ/parallax)
at upstream commit `c8c8ebdaaf2924b6d25e2d1caff61e27374cce0b`,
licensed under Apache 2.0. The eight vendored Python modules
(`__init__.py`, `_vendored_utils.py`, `layer_allocation.py`,
`model_info.py`, `node.py`, `node_management.py`, `request_routing.py`,
`scheduler.py`) carry a 6-line attribution header at the top.

**Verbatim vs. modified.** Verbatim: `model_info.py`, `node.py`,
`node_management.py`, `scheduler.py`, `request_routing.py`,
`layer_allocation.py`. Modified: `_vendored_utils.py` is a minimal
port of upstream `parallax_utils` helpers (Apple-Silicon path
stubbed). Import paths in every vendored module rewritten from
`src.scheduling.*` → `prsm.compute.parallax_scheduling.*`. See
`licenses/PARALLAX-NOTICE.txt` modification log for the authoritative
diff.

**PRSM-original delta** (the load-bearing trust contribution; auditor
focus should be here, not the vendored algorithm):
- `prsm_types.py` — PRSM-native data layer (`ParallaxGPU`,
  `AllocationResult`, `RegionPipeline`, `allocate_across_regions`,
  `partition_by_region`, `to_parallax_node`).
- `prsm_request_router.py` — Phase-2 router (`RouteRequest`,
  `GPUChain`, `RequestRouter`, `ProfileSource` Protocol).
- `profile_dht.py` — Ed25519-signed profile DHT mirroring the
  Phase 3.x.5 manifest DHT pattern (~700 LoC); MUST be reviewed
  alongside §3.2 cross-language byte-equality concerns.
- `trust_adapter.py` — the four adapters: `AnchorVerifyAdapter`
  (Phase-1 input filter, anchor=PublisherKeyAnchor.sol),
  `TierGateAdapter` (Phase-2 pre-route filter, hardware-TEE
  attestation), `StakeWeightedTrustAdapter` (rescales profile
  latency by 1/confidence; closes pool-level admission via
  `is_eligible`), `ConsensusMismatchHook` (post-route, sample-
  rate'd redundant chain → Phase 7.1 challenge on output mismatch).
- `prsm/compute/inference/parallax_executor.py` —
  `ParallaxScheduledExecutor` implements the existing
  `InferenceExecutor` Protocol; drop-in replacement for
  `TensorParallelInferenceExecutor`.

**License compliance.** `LICENSE` (root) carries a "Third-party
components" footer pointing to `licenses/PARALLAX-APACHE-2.0.txt`
(verbatim upstream LICENSE) and `licenses/PARALLAX-NOTICE.txt`
(derivative-works notice with pinned commit SHA + modification
log). `README.md` carries a corresponding "Third-Party Components"
section.

**Test coverage at the tag.** 180 tests green:
- `tests/unit/test_parallax_layer_allocation.py` (27)
- `tests/unit/test_parallax_request_routing.py` (16)
- `tests/unit/test_parallax_profile_dht.py` (51)
- `tests/unit/test_parallax_trust_adapter.py` (51)
- `tests/unit/test_parallax_executor.py` (29)
- `tests/integration/test_parallax_e2e.py` (7) — three simulated
  nodes, exercises happy path + region preference + anchor
  enforcement + tier gate + stake weighting + consensus mismatch
  + membership churn

**Auditor prompts:** review `trust_adapter.py` line-by-line; verify
`signing_payload` canonicalization in `profile_dht.SignedProfileEntry`
matches `manifest_dht.ManifestEntry` shape (sort_keys=True; sorted
RTT-to-peers map); verify the "never raises" invariant on
`ProfileDHTServer.handle()`; verify anchor-verify-on-read in
`SignedProfileEntry.verify_with_anchor` (pubkey resolved via
`anchor.lookup(node_id)` at verify time, not from any embedded
field); check that `TrustStack.filter_pool` correctly composes
anchor + stake-eligibility (Phase 3.x.6 Task 7 E2E surfaced and
remediated a hole here pre-tag).

---

## 7.2 Cross-Host ChainExecutor (Phase 3.x.7 only)

Phase 3.x.7 closes the "brain has no hands" gap left by Phase
3.x.6's scheduler: the production `ChainExecutor` implementation
that takes a `GPUChain` (3.x.6 router output) + `InferenceRequest`
and runs inference across the chain stages on real network nodes.

**Module scope.** `prsm/compute/chain_rpc/` — wire protocol
(`protocol.py`), per-node server (`server.py`), client orchestrator
(`client.py`), activation codec (`activation_codec.py`), production-
wiring factories (`factories.py`). Plus
`prsm/compute/inference/multi_stage_attestation.py` for the per-
stage TEE attestation envelope embedded in `InferenceReceipt`.

**No vendored components.** All Phase 3.x.7 code is PRSM-original;
no Apache 2.0 attribution required. The activation codec wraps
the existing Phase 6 Task 6b `ShardChunker` for >10 MiB activations
(codec ready; transport-side wiring deferred to Phase 3.x.7.x).

**Trust seams auditors should focus on:**

  1. `HandoffToken` settler-signing → server anchor-verify
     (`protocol.py:240-310`). Token binds (request_id, settler_node_id,
     chain_stage_index, chain_total_stages, deadline_unix). Forging
     requires forging an Ed25519 signature against an anchor-
     registered identity. Note: chain_stage_index is informational
     server-side; the actual binding to a node identity is enforced
     at the executor's response-verification layer (`client.py`),
     not at the server. Documented limitation; relay-swap detection
     is a Phase 3.x.7.x research item.

  2. `RunLayerSliceResponse.verify_with_anchor` requires
     `expected_stage_node_id` as a keyword-only parameter
     (`protocol.py:607-664`). Lookup uses the EXPECTED identity
     (caller-supplied), not the response's self-claim. This closes
     the substitution hole where any anchor-registered peer could
     impersonate any other anchor-registered peer. See Phase 3.x.7
     Task 8 round-1 H2 finding for the rationale.

  3. `LayerStageServer.handle()` 8-step validation order
     (`server.py:295-460`). Order matters: token verify BEFORE
     deadline check BEFORE registry lookup BEFORE shard coverage
     BEFORE tier gate BEFORE decode/run/encode. Out-of-order would
     leak information (e.g., model existence revealed before token
     check).

  4. `MAX_HANDSHAKE_BYTES = 64 MiB` on `parse_message`
     (`protocol.py:63`). DoS guard fires before `json.loads`
     allocates. 64 MiB chosen to fit typical LLM activations
     (2048-token × 4096-dim float16 ≈ 16 MiB raw → ≈ 33 MiB
     hex+JSON inline). Mitigated by Phase 6 conn-level rate
     limiting + anchor + stake gating per upstream_token. Stages
     can wrap with stricter caps per the docstring contract.

  5. Multi-stage attestation envelope
     (`multi_stage_attestation.py:198-272`). Magic prefix
     `b"PRSM-MS-ATT-V1:"`. `decode_multi_stage_attestation`
     enforces: stage_index 0..N-1 contiguous (no gaps), no
     duplicates, optional `expected_stage_count` for callers that
     know the chain length. Defends against settler omitting a
     SOFTWARE stage to upgrade apparent worst-case TEE type seen
     by a verifier. Receipt's settler signature commits to the
     full envelope bytes via `signing_payload()` hex-encoding.

  6. Worst-case TEE policy (`multi_stage_attestation.py:245-276`).
     SOFTWARE drags hardware down — one software stage in an SGX
     chain → receipt records SOFTWARE. Conservative; auditors
     wanting per-stage trust can iterate `decode_multi_stage_attestation`
     explicitly. Hardware types tied at rank 1 (no SGX-vs-TDX-vs-SEV
     differentiation in v1).

**Test coverage at the tag.** 473 tests green across:
- `tests/unit/test_chain_rpc_protocol.py` (66)
- `tests/unit/test_chain_rpc_server.py` (40)
- `tests/unit/test_chain_rpc_activation_codec.py` (37)
- `tests/unit/test_chain_rpc_client.py` (40)
- `tests/unit/test_chain_rpc_factories.py` (13)
- `tests/unit/test_multi_stage_attestation.py` (53)
- `tests/integration/test_chain_rpc_e2e.py` (12) — three simulated
  PRSM nodes running a real toy 4-layer model end-to-end with output
  bit-identical to single-host reference. Covers all 7 design-plan
  §4 Task 7 acceptance scenarios: happy path, deadline propagation,
  forged token, shard missing, tier gate, signature tampering, mid-
  chain disconnect.

**Round-1 review surface (closed pre-tag):** H1 (cap raise), H2
(verify_with_anchor substitution-rejection), M1 (bool-as-int
hygiene), M2 (envelope contiguity + dedup), L6 (factory tokenizer
warning), L7 (placeholder flag), L8 (init docstring). 12 new
regression tests added covering each remediated invariant.

**Auditor prompts:** start at `client.py` line-by-line — the
RpcChainExecutor orchestrates the whole pipeline; verify the cross-
field consistency checks (request_id echo, expected_stage_node_id)
fire correctly. Then `server.py` for the 8-step validation order +
"never raises" invariant. Then `multi_stage_attestation.py` for the
envelope contiguity logic. The E2E integration test exercises real
adversarial scenarios end-to-end.

---

## 7.3 Chunked Activation Streaming (Phase 3.x.7.1)

Phase 3.x.7.1 layered a v2 chunked-streaming wire path on top of
3.x.7 to unblock production-scale LLM hidden-state activations
(2048-token × 4096-dim float16 already approaches the 64 MiB inline
cap; bigger models exceed it). Activations exceeding the threshold
(default 10 MiB) ride out-of-band as `ActivationChunk` frames over
a streaming transport; smaller activations stay on the inline path
with v1↔v2 byte-equivalent canonical-JSON encoding for rolling-
deploy forward-compat.

**Tag:** `phase3.x.7.1-merge-ready-20260428` at commit `339957ee`.

**Headline guarantee.** 16 MiB float32 activation routed through 16
chunks of 1 MiB each is bit-identical to the single-host reference
computation (`numpy.testing.assert_array_equal`), verified end-to-
end across two simulated stages.

**Trust seams added by 3.x.7.1.**

  1. `_validate_streamed_envelope` (server.py:744-803) fires BEFORE
     any chunk is consumed from the iterator. Rejects manifests
     claiming `payload_bytes ≠ shape × dtype.itemsize`,
     `payload_bytes > max_streamed_payload_bytes` (default 1 GiB),
     or `total_chunks` inconsistent with
     `ceil(payload_bytes/chunk_bytes)`. Defense against network-
     level adversaries inflating the envelope to coerce server
     memory exhaustion during reassembly.

  2. `_reassemble_inbound_chunks` bounded by `expected_total_chunks`
     (server.py:825-844). Excess frames raise
     `ActivationCodecError("excess chunks")` rather than growing
     unbounded — defense bound by the validated envelope, not by
     the iterator's source.

  3. Response signing payload commits to the full envelope
     (`protocol.py:806-816`). The conditional
     `activation_manifest_envelope` dict commits the stage to ALL
     FIVE manifest fields (shard_id, payload_sha256, payload_bytes,
     total_chunks, chunk_bytes), not just the payload hash.
     Tampering any field invalidates the Ed25519 signature. Closes
     the H2 round-1 finding where a relay could inflate
     payload_bytes / total_chunks while leaving payload_sha256
     intact.

  4. Client-side defense-in-depth
     (`client.py:_validate_streamed_response_envelope`). Mirrors
     the server's envelope validation; fires AFTER signature verify
     but BEFORE response chunk consumption + bounded
     `_iter_response_chunks`. Caps configurable via
     `max_streamed_payload_bytes=` constructor arg.

  5. v1↔v2 inline byte equivalence preserved. The conditional
     manifest-envelope encoding omits the key entirely when manifest
     is None — v2 nodes signing inline produce byte-identical
     canonical JSON to v1 nodes, so v1 callers can verify v2
     inline-mode signatures during rolling deploys.
     `SUPPORTED_PROTOCOL_VERSIONS = frozenset({1, 2})`.

  6. `ActivationChunk.request_id` relay-defense binding. Chunks
     splice-defended: a chunk frame must carry the parent request's
     id, blocking a relay from reusing chunks across concurrent
     streams.

**Test coverage at the tag.** 243 chain_rpc-surface tests; 512
across Phase 3.x.6+7+7.1 regression. Includes 16 round-1-remediation
tests (envelope tampering, excess chunks, version-negotiation
mapping) + Response-side M3 test added in round-2 I1.

**Round-1 → round-2 surface (closed pre-tag):**
- H1 (server-side unbounded chunk reassembly) — pre-consumption
  envelope-validation gate + bounded reassembly.
- H2 (response signing payload didn't commit to envelope shape
  fields) — full envelope dict in signing payload + client-side
  mirror validation.
- M3 (inline-XOR-streamed both-empty case) — Request + Response
  `__post_init__` reject the structurally-meaningless empty-blob-
  empty-manifest case.
- L1 (parse_message version handling) — missing/non-int version
  surfaces as `ChainRpcVersionMismatchError` (mapped to
  `UNSUPPORTED_VERSION`), not `MalformedError`.

**Round-2 INFO-level follow-ups (non-blocking):**
- I2: commit-message test count overstated (cosmetic).
- I3: zero-byte payload corner case is dead code under current
  shape validation; document or explicitly reject.
- I4: `1 GiB` literal duplicated client+server; could share a
  named constant.

**Auditor prompts:** the 3.x.7.1 attack surface is small —
streaming is just a wire-format option that defers chunked
delivery. Read `_validate_streamed_envelope` (server) +
`_validate_streamed_response_envelope` (client) side-by-side; both
must enforce the same shape×dtype + cap + total_chunks gates.
Then the conditional encoding in `signing_payload` — verify that
omitting `activation_manifest_envelope` is byte-equivalent to a
v1 inline payload. Then the 16 MiB E2E assertion which is the
golden integration test.

---

## 7.4 Streaming-Token Output (Phase 3.x.8)

Phase 3.x.8 added chat-style incremental output to the cross-host
inference path. Caller iterates the executor's
``execute_chain_streaming(...)`` generator; each yield is a
``StreamToken`` (text_delta + sequence_index + optional token_id +
optional finish_reason); the LAST yield is a
``ChainExecutionResult`` carrying the signed multi-stage receipt.
Closes the "wait-for-full-chain-return" UX gap that blocked
interactive use of ``prsm_inference``.

**Tag:** `phase3.x.8-merge-ready-20260428` at commit `391b92b0`.

**Headline guarantee.** 1/2/3-stage chains produce joined
``StreamToken.text_delta`` bit-identical to the single-host
reference output. Verified end-to-end across 6 E2E scenarios in
``TestStreamingTokenOutput``.

**v1 honest scope (documented in code + commits — auditors should
not assume more).**

  - ``SyntheticStreamingRunner`` is a PLACEHOLDER. It wraps the
    one-shot ``LayerSliceRunner``, runs the full forward pass,
    decodes activation→text, splits into synthetic deltas. Real
    autoregressive decode plugs in later as a runner replacement
    under the same ``StreamingLayerRunner`` Protocol — no
    public-surface change. The wire path is the load-bearing
    surface.
  - **Cancellation = clean upstream cleanup only.** Python
    ``GeneratorExit`` semantics forbid yielding additional values
    after ``.close()``; a partial-output ``ChainExecutionResult``
    cannot be delivered through the wire on caller cancellation.
    Try/finally ``.close()`` propagation at server + executor
    layers ensures resource cleanup; the partial-receipt-on-cancel
    pathway (in-band cancel sentinel via ``.send()`` or
    side-channel inspection API) is a Phase 3.x.8.x follow-up.

**Trust seams added by 3.x.8.**

  1. ``streaming: bool = False`` field on ``RunLayerSliceRequest``
     with conditional encoding (omitted from canonical JSON when
     False). Preserves byte-identity with v2-pre-3.x.8 messages
     so the v1↔v2 forward-compat invariant from 3.x.7.1 still
     holds for non-streamed traffic. M1-style bool-coercion guard
     rejects int 0/1 on both ``__post_init__`` and ``from_dict``.

  2. ``StreamFinalFrame.response.activation_blob`` is the UTF-8
     joined output bytes. The stage's existing
     ``signing_payload`` hex-encodes ``activation_blob`` — so a
     relay tampering ANY ``TokenFrame.text_delta`` causes the
     joined-bytes hash to diverge from what the stage signed,
     invalidating the stream as a whole. No new signing-payload
     field needed. Three layers enforce this: server pre-sign
     (cannot sign inconsistent text), executor post-receive
     (re-checks against signed bytes), MCP adapter (defense-in-
     depth aggregate check).

  3. ``InferenceReceipt.streamed_output: bool`` flag is part of
     the SIGNED payload via conditional trailing line
     ``streamed_output:true`` (omitted entirely when False —
     byte-identical to pre-3.x.8 receipts). Tampering the flag
     in EITHER direction (False→True OR True→False) on a signed
     receipt invalidates the signature. Downgrade-resistant.
     M1-style bool-coercion guard at ``from_dict``.

  4. Server ``handle_token_stream`` validation gates fire BEFORE
     runner dispatch: streaming=True required, no chunked-input
     v1 (manifest must be None), runner configured (else
     INTERNAL_ERROR), stage IS the chain tail per
     ``_is_final_stage``. The existing 8-step gates (token,
     deadline, registry, shard, tier) reused via
     ``_run_validation_gates``. Sequence-index invariant
     (0-indexed strictly increasing) AND joined-text invariant
     (``"".join(text_deltas) == terminal_chunk.full_output_text``)
     enforced server-side BEFORE signing — both fail with
     INTERNAL_ERROR.

  5. Round-1 M1 remediation (auditor-visible "sole error frame on
     failure" invariant): server-side terminal-chunk integrity
     validation reordered to fire BEFORE the terminal
     ``TokenFrame`` is published on the wire. On failure: SOLE
     ``StageError`` frame, no preceding terminal ``TokenFrame``.
     Non-terminal ``TokenFrame``s may already be on the wire
     when a terminal-chunk-side failure occurs — explicitly
     documented in test ``test_runner_joined_text_diverges_from_full_output_text``.

  6. Cancellation cleanup at both server (``_dispatch_token_stream``)
     and executor (``_dispatch_streaming_tail``) layers via
     try/finally that explicitly invokes ``.close()`` on the
     upstream iterator. close-time exceptions swallowed at debug
     log — never propagate through ``GeneratorExit``. Verified
     by tracking-generator tests that record their own
     close_count via ``except GeneratorExit:``.

**Test coverage at the tag.** 386 streaming-relevant tests; 739
across the full Phase 3.x.6+7+7.1+8 + MCP regression. Includes:
- 26 protocol tests (TokenFrame + StreamFinalFrame round-trip +
  streaming-flag conditional encoding + bool coercion).
- 47 server tests (token-stream routing + validation gates +
  runner-error handling + cancellation cleanup).
- 23 client tests (streaming dispatch + frame validation +
  signature verification + cancellation).
- 22 streaming_runner tests.
- 13 MCP-adapter tests (per-token progress + receipt downgrade
  resistance).
- 10 receipt extension tests.
- 6 E2E scenarios.

**Round-1 → round-2 surface:**
- Round-1: APPROVED-FOR-TAG with M1 (terminal-chunk validation
  ordering) + 3 LOWs (cosmetic — unused loop var, defensive
  assertion, receipt-builder dedup).
- M1 remediated pre-tag at commit `391b92b0`.
- Round-2: APPROVED-FOR-TAG.

**Round-2 LOW follow-ups (deferred to Phase 3.x.8.x):**
- L1: unused loop variable in `_dispatch_streaming_tail`
  cleanup path (cosmetic).
- L2: defensive `assert chunk_iter is not None` for human
  readability of finally clause.
- L3: shared `_build_signed_receipt(*, streamed)` helper between
  `ParallaxScheduledExecutor` + `mcp_streaming` to avoid drift.

**Other Phase 3.x.8.x deferred items:**
- Real autoregressive decoder replacement of
  ``SyntheticStreamingRunner``. Future threat-model revision
  needed for timing side-channels (token-arrival timing leaks
  output length).
- Streaming + chunked-input composition (currently rejected at
  server with MALFORMED_REQUEST + at client with
  ACTIVATION_TOO_LARGE).
- Partial-output cancellation receipt via in-band `.send()`
  sentinel or side-channel inspection API.
- HTTP streaming surface — wire ``stream_inference_to_mcp`` to
  ``mcp_server.handle_prsm_inference`` via SSE / chunked-transfer
  endpoint.
- Factory updates (``make_layer_stage_server(streaming_runner=...)``
  + ``make_rpc_chain_executor(token_stream_send_message=...)``).

**Auditor prompts:** the 3.x.8 attack surface is contained — the
streaming-token wire path is additive on top of 3.x.7.1's
chunked-streaming path. Read the conditional-encoding pattern in
both ``RunLayerSliceRequest.streaming`` (wire layer) and
``InferenceReceipt.streamed_output`` (receipt layer); they should
be byte-identical to pre-3.x.8 when False. Read
``_dispatch_token_stream`` lines 911-982 in server.py for the M1
reorder + integrity gates. Read the 6 E2E scenarios in
``TestStreamingTokenOutput`` — they exercise the full surface.
Honest-scope notes (synthetic runner placeholder + cancellation
v1) are documented inline + in commit messages — auditors won't
find a "we said X but did Y" gap.

---

## 7.5 Streaming HTTP Endpoint (Phase 3.x.8.1)

Phase 3.x.8.1 closes the "dormant scaffolding" gap left by Phase
3.x.8 by adding a Server-Sent-Events sibling to the existing
``POST /compute/inference`` endpoint. The streaming wire path +
executor generator API + MCP adapter from 3.x.8 are now wired
end-to-end through the production HTTP surface — ``prsm_inference``
MCP tool's chat-style UX is real, not theoretical.

**Tag:** `phase3.x.8.1-merge-ready-20260428` at commit `67fe8863`.

**Headline guarantees.**

  - ``POST /compute/inference/stream`` emits W3C-compliant SSE
    frames: ``event: token`` (per StreamToken) → ``event: result``
    (signed receipt with ``streamed_output=True``) OR
    ``event: error`` (refund + structured error).
  - End-to-end signed-receipt verification: the wire receipt
    verifies under the settler identity AND tampering
    ``streamed_output`` (downgrade attack) flips signature to
    invalid — Phase 3.x.8 Task 4 invariant proven through the
    HTTP boundary in ``test_signed_receipt_verifies_under_settler_identity``.
  - Design plan §3.4 settle-on-tokens-emitted billing policy
    (round-1 M1 remediation): pre-execute failure (zero tokens)
    → refund; mid-stream failure AFTER tokens hit the wire →
    settle. Closes a real billing griefing vector caught at
    review.
  - Back-compat: existing ``/compute/inference`` unary endpoint
    UNCHANGED on a node also serving ``/stream``.

**v1 honest scope (carried from 3.x.8).**

  - ``SyntheticStreamingRunner`` is still a placeholder. Real
    autoregressive runner is Phase 3.x.10.
  - Cancellation = clean cleanup only (Python ``GeneratorExit``
    semantics). HTTP-layer cancel via ``connection.close()``
    propagates correctly through Starlette → executor → server
    via the Phase 3.x.8 Task 6 cleanup mechanism.
  - Streaming + chunked-input composition still rejected at v1.

**Trust seams added by 3.x.8.1.**

  1. ``_resolve_post_token_billing`` (api.py:228-275) — design
     plan §3.4 enforcement. Branches on ``tokens_emitted``: > 0
     → release escrow at full estimate; == 0 → refund. ALL FOUR
     post-token-loop refund branches in ``_event_generator``
     route through this helper. Closes the griefing vector
     where a malicious node could emit N tokens then crash and
     pay nothing despite forcing real compute on responding
     nodes.

  2. ``_result_to_dict(*, identity)`` (api.py:144-185) — wire-
     side receipt re-sign on ``job_id`` rebind. The executor
     uses an internal ``parallax-stream-job-*`` id; the API is
     authoritative and rebinds to ``infer-stream-*`` for billing
     correlation. Since ``job_id`` is part of the signed
     payload, the rebound receipt MUST be re-signed under
     ``node.identity``. Caught + fixed at Task 5 by the E2E
     test surfaced through ``verify_receipt`` actually
     exercising the cryptographic invariant. Mocked unit tests
     would have missed this — it's why the deep-stack E2E
     coverage is load-bearing.

  3. SSE framing conformance (api.py:101-119, mcp_server.py:570-
     654) — ``_sse_event`` produces W3C-compliant frames;
     ``_parse_sse`` handles chunk-boundary splits, multi-line
     data, comments, CRLF, default ``event: message``,
     forward-compat with unknown event types (silently
     ignored). Defensive flush of unterminated final frame.
     Tested with programmable ``_FakeResponse._chunks`` for
     chunk-boundary control.

  4. Operator-misconfig surfaces structurally as 503 (api.py:632-
     652) — no executor wired: 503 "not initialized"; executor
     lacks ``execute_streaming``: 503 "does not support
     streaming". Doesn't crash with ``AttributeError``.

  5. ``InferenceError`` exception (mcp_server.py:551-566) —
     structured error from a streaming dispatch. ``.code``
     carries server-side machine-readable code; ``.message``
     human-readable. Handler maps to clean "Inference rejected:
     <reason>" surface, no traceback exposed.

  6. Privacy budget gating fires at REQUEST time
     (api.py:1107-1142) via ``can_spend(expected_epsilon)``,
     BEFORE the executor runs. The post-token failure path
     correctly skips ``record_spend`` because nothing was
     actually billed against the budget yet (the executor
     never returned a complete receipt). Documented inline in
     ``_settle_streaming_escrow``.

**Test coverage at the tag.** 794 tests green across Phase
3.x.6+7+7.1+8 + 8.1 surfaces. Includes:
- 10 ParallaxScheduledExecutor streaming tests.
- 18 SSE endpoint unit tests.
- 18 MCP streaming-client tests (10 parser + 8 client behavior).
- 8 E2E FastAPI TestClient scenarios (4 happy path + 1 pre-execute
  refund + 1 §3.4 post-token settle + 2 back-compat).

**Round-1 → round-2 surface:**
- Round-1: NEEDS-REMEDIATION-PRE-TAG with M1 (settle-on-tokens-
  emitted policy violation), M2 (test coverage gap), L1 (dead-
  code redundant re-sign). L2+L3 deferred.
- M1 + M2 + L1 remediated pre-tag at commit `67fe8863`.
- Round-2: APPROVED-FOR-TAG.

**Round-1 LOW follow-ups (deferred to Phase 3.x.8.1.x):**
- L2: ``aiohttp.ClientTimeout(sock_read=120)`` may be too
  generous; consider 30s after production telemetry shows
  inter-token p99 latency.
- L3: defensive ``_parse_sse`` partial-garbage explicit
  JSON-decode test (current behavior correct — raises
  ``InferenceError(MALFORMED_RESPONSE)`` — just lacks the
  test).

**Auditor prompts:** start with ``test_signed_receipt_verifies_under_settler_identity``
in ``test_compute_inference_stream_e2e.py``. It's the cryptographic
invariant the whole point release exists to preserve through the
HTTP boundary. Then read ``_resolve_post_token_billing`` +
``test_mid_stream_failure_after_tokens_settles_not_refunds`` for
the §3.4 billing policy. The SSE framing is vanilla W3C — the
parser handles the edge cases that matter (chunk boundaries,
CRLF, comments, forward-compat). The Task 5 receipt-rebind bug
(invisible to mocks, caught by E2E) is the lesson on why deep-
stack integration tests matter.

---

## 7.6 Real Autoregressive Streaming Runner (Phase 3.x.10)

Phase 3.x.10 closes the SyntheticStreamingRunner placeholder
caveat carried by every streaming phase since 3.x.8. The new
``AutoregressiveStreamingRunner`` drives a real
``transformers.AutoModelForCausalLM.generate(streamer=...)``,
emits real per-token chunks, and is a drop-in for the synthetic
predecessor — same ``StreamingLayerRunner`` Protocol, no
public-surface change. The streaming UX served through
``prsm_inference`` MCP tool now produces genuinely-distinct
tokens from a real model rather than synthetic word splits.

**Headline guarantees.**

  - ``AutoregressiveStreamingRunner`` is structurally a
    ``StreamingLayerRunner`` (Phase 3.x.8 Task 2 Protocol).
    Drop-in replacement for ``SyntheticStreamingRunner`` —
    ``LayerStageServer.handle_token_stream`` consumes either
    runner identically.
  - Tail-only contract enforced. ``is_final_stage=False`` →
    exactly ONE terminal chunk with ``finish_reason="error"``,
    no preceding non-error chunks, no ``model.generate`` call,
    no ``prompt_provider`` call. Sharded autoregressive deferred
    to Phase 3.x.11.
  - UTF-8 multi-byte buffer-and-flush for byte-level BPE
    tokenizers. The ``_HFStreamerAdapter`` cumulative-decode +
    U+FFFD-suffix detection holds the buffer until a whole-
    character boundary emerges, so ``"".join(text_deltas)``
    ALWAYS forms valid UTF-8 even when emoji or CJK codepoints
    span multiple BPE token boundaries. Tested across emoji,
    CJK, ZWJ family sequences (👨‍👩‍👧), and mixed
    interleavings.
  - Sampling-params + stop-condition wiring: ``request.max_tokens``
    + ``request.temperature`` override ``SamplingDefaults``
    (max_tokens=512, temperature=1.0, top_k=50, top_p=0.95).
    Greedy when temperature=0 (``do_sample=False``); else sampled.
    Finish-reason mapping: EOS → ``"stop"``, cap → ``"max_tokens"``,
    exception → ``"error"``.
  - Mid-decode exception path is a single terminal error chunk
    with the partial joined output preserved on
    ``full_output_text`` so the receipt's ``output_hash`` commits
    to whatever did make it to the wire.
  - Tensor-wrap step in-runner: tokenizer.encode returns
    ``List[int]``; HF.generate requires ``[1, seq_len]``
    ``torch.Tensor``. The wrap happens in the runner so
    production callers get HF compat without each operator
    wiring it themselves. Test fakes pass through via duck-typed
    ``.tolist()`` unwrap.

**Trust seams added by 3.x.10.**

  1. ``_HFStreamerAdapter`` (autoregressive_runner.py:88-184) —
     bridges HF's synchronous ``streamer.put(token_ids_tensor)``
     callback contract to the Protocol's ``Iterator[StreamingChunk]``
     pull contract via a buffer-during-generate + yield-after
     pattern. v1 trade-off: real-time delivery happens at the
     SSE / MCP layer above the runner; the runner buffers within
     a single ``.generate()`` call. Async-during-generate is a
     Phase 3.x.10.x perf upgrade.

  2. ``prompt_provider`` callable injected at construction —
     decouples runner from prompt resolution. Production wires
     this to a server-side registry keyed on ``request_id`` (the
     executor stashes the prompt before calling
     ``handle_token_stream``); tests inject deterministic fakes.
     Non-tail dispatch short-circuits BEFORE invoking
     ``prompt_provider`` — important because the prompt resolver
     may be expensive (DB lookup, MCP roundtrip).

  3. Tail-only contract is documented in the runner's class
     docstring; ``test_docstring_documents_phase_3_x_11_deferral``
     turns the "documented in docstring" acceptance gate into an
     introspectable invariant — future doc edits can't silently
     drop the deferral note.

  4. Per-token timing side-channel acknowledged + scoped (Phase
     3.x.10 Task 6 memo). The runner ships Tier-A + Tier-B-eligible
     without padding; Tier C is gated off until Phase 3.x.10.x
     lands constant-time padding (M1 fixed-rate cadence preferred,
     M2 batched-trailing as toggle). See
     ``docs/2026-04-28-phase3.x.10-timing-sidechannel-memo.md`` +
     R3 §3.5 A5 + §10.4.

**v1 honest scope (carried + new).**

  - Sharded autoregressive (each stage running once per token)
    deferred to Phase 3.x.11. The tail must host enough of the
    model to generate locally.
  - Stop sequences not implemented; only EOS + ``max_tokens`` in
    v1.
  - Constant-time padding for Tier C deferred to Phase 3.x.10.x.
    Runner enforces NO content-tier check itself — the dispatch
    layer (Phase 3.x.1 Task 3 content-tier gate) is the
    structural enforcement point.
  - HF generate buffering means the FIRST token reaches the wire
    only after generate() returns — the runner is internally
    blocking-during-generate. SSE-layer streaming UX is preserved
    because the runner emits all tokens in one burst at function
    return; from a user perspective, tokens still arrive
    incrementally as the chain produces them. Async-during-
    generate is Phase 3.x.10.x.

**Test coverage at the tag.**
- 52 unit tests in ``test_autoregressive_runner.py``:
  - 10 helper tests (_coerce_token_ids × 6 + _last_token_id × 4).
  - 17 happy-path runner / shape / construction-validation tests:
    TestRunnerEmitsChunks (3 — N chunks, monotonic indices,
    terminal aggregate fields), TestMaxTokensCap (1),
    TestEosTriggersStop (1), TestMidGenerateException (2 —
    partial-emits-then-error, pre-decode error),
    TestNonTailDispatch (1), TestRunnerIsIterable (1),
    TestEmptyEmission (1), TestRequestOverridesDefaults (2),
    TestProtocolStructural (1), TestConstructorValidation (4).
  - 8 multi-byte UTF-8 tests (ASCII passthrough, emoji 2-token
    + 3-token splits, CJK 2-token, mixed interleaving, end-of-
    stream U+FFFD flush, direct adapter buffer-hold, ZWJ family
    👨‍👩‍👧 across 6 codepoint-misaligned 3-byte chunks).
  - 9 sampling + stop-condition tests.
  - 8 tail-only contract tests (incl. introspectable docstring
    invariant via ``test_docstring_documents_phase_3_x_11_deferral``).
- 4 server-wired integration tests in
  ``tests/integration/test_autoregressive_runner_server_wire.py``
  (round-1 H1+M3 remediation): mocked HF model + real
  ``LayerStageServer.handle_token_stream`` proves the
  joined-text invariant holds end-to-end through the production
  server stack — happy path, mid-decode exception (H1
  regression), sequence-index monotonicity across the partial-
  then-terminal sequence, pre-decode exception falls through to
  StageError.
- 6 slow-marked E2E tests in
  ``tests/integration/test_autoregressive_runner_e2e.py`` (opt-in
  via ``pytest -m slow``). Uses real distilgpt2 via transformers
  + torch:
  - test_real_decode_emits_streaming_chunks
  - test_joined_deltas_form_valid_utf8
  - test_greedy_decode_bit_identical_on_rerun
  - test_finish_reason_correctly_set
  - test_receipt_with_streamed_output_flag_verifies (signs +
    verifies via identity AND via standalone public_key_b64 —
    the prod verification path)
  - test_streamed_flag_tampering_invalidates_signature (Phase
    3.x.8 Task 4 downgrade-resistance proven against the REAL
    output path)

**Round-1 → round-2 surface.**
- Round-1: NEEDS-REMEDIATION-PRE-TAG with H1 (mid-decode
  exception path violated server's joined-text invariant —
  partial output dropped on the wire), M1 (audit-prep line
  numbers off), M2 (test-count math inconsistent), M3 (E2E did
  not route through ``LayerStageServer``, so H1 was undetectable
  by CI), M4 (server doesn't plumb ``request=`` to runner —
  sampling overrides dead-letter through prod path), M5
  (runner exported but no production wiring), L1 (stale
  ``_prompt_id_count`` comment), L3 (redundant ``if piece:``).
  L2/L4 deferred.
- H1 + M1 + M2 + M3 + L1 + L3 remediated pre-tag. M4 + M5
  documented as honest-scope caveats below; deferred to
  Phase 3.x.10.x.
- Round-2: APPROVED-FOR-TAG.

**Production wiring caveats — CLOSED by Phase 3.x.10.x (see §7.7).**

Both caveats below were live deferrals at the
``phase3.x.10-merge-ready-20260428`` tag. Phase 3.x.10.x
(``phase3.x.10.x-merge-ready-20260428`` — see §7.7) closes
both. They are documented here as historical context for
auditors comparing the two tags.

  1. **No production caller constructs the runner yet.** The
     runner is exported at the package surface but no code path
     in ``prsm/`` instantiates ``AutoregressiveStreamingRunner``
     and passes it to ``LayerStageServer(streaming_runner=)``.
     The runtime guarantees in this section are exercised only
     by the test suite. This is analogous to the dormant
     scaffolding pattern Phase 3.x.8 left for Phase 3.x.8.1 to
     close. **Closed by Phase 3.x.10.x §7.7:**
     ``make_autoregressive_streaming_runner(...)`` factory +
     ``make_layer_stage_server(streaming_runner=...)`` extension.

  2. **Sampling overrides via ``request=`` are bypassed in the
     production server path.** The runner accepts an optional
     ``request: Any = None`` for sampling resolution, and when
     consumed directly (as in the unit tests + the slow E2E)
     ``request.max_tokens`` + ``request.temperature`` correctly
     override ``SamplingDefaults``. But
     ``LayerStageServer.handle_token_stream`` (server.py:834-840)
     does not pass ``request=`` through — every server-mediated
     dispatch falls back to the runner's construction-time
     defaults. Plumbing the wire-format extension to carry
     sampling params end-to-end is Phase 3.x.10.x scope.
     **Closed by Phase 3.x.10.x §7.7:** wire-format extension
     adds ``RunLayerSliceRequest.max_tokens`` +
     ``.temperature`` (omit-when-None canonical encoding for
     byte-equivalence); ``LayerStageServer.handle_token_stream``
     constructs ``StreamingSamplingShim`` and forwards as
     ``request=`` to the runner; ``RpcChainExecutor`` populates
     the new wire fields from ``InferenceRequest``.

**Auditor prompts:** start with the runner's class docstring in
``prsm/compute/inference/autoregressive_runner.py:232-272`` for
the contract scope. Then read the
``_HFStreamerAdapter._maybe_flush`` U+FFFD branch
(autoregressive_runner.py:174-189) — that's the multi-byte
correctness invariant. The tail-only contract enforcement at
``autoregressive_runner.py:371-382`` is the Phase 3.x.11
deferral boundary. The mid-decode exception path at
``autoregressive_runner.py:441-471`` (post-H1 remediation) is
the round-1 fix that preserves the server's joined-text
invariant — re-emit buffered pieces as non-terminal TokenFrames
first, then a terminal error chunk. The integration test
``test_mid_decode_exception_partial_emits_then_terminal`` in
``tests/integration/test_autoregressive_runner_server_wire.py``
is the regression test that pins it through the production
server. Receipt verification through the runner's TEE
attestation is exercised by
``test_receipt_with_streamed_output_flag_verifies`` against
real distilgpt2 output. The timing-side-channel memo
(``docs/2026-04-28-phase3.x.10-timing-sidechannel-memo.md``) is
the disclosed residual; Tier C is structurally blocked until
Phase 3.x.10.x.

---

## 7.7 Production Wiring + Sampling-Param Plumbing (Phase 3.x.10.x)

Phase 3.x.10.x is a 6-task point release on Phase 3.x.10 that
closes the round-1 M4 + M5 honest-scope deferrals (see §7.6
"Production wiring caveats — CLOSED by Phase 3.x.10.x"
subsection above). The slice makes the
``AutoregressiveStreamingRunner`` user-visible: operators can
construct a streaming-capable ``LayerStageServer`` with two
factory calls, and end-user
``InferenceRequest.max_tokens`` / ``.temperature`` overrides
now reach the runner instead of dead-lettering at the wire
boundary.

**Tag:** ``phase3.x.10.x-merge-ready-20260428`` at commit
``[applied at tag time]``.

**Headline guarantees.**

  1. **Wire-format extension (M4 closure).** ``RunLayerSliceRequest``
     gains optional ``max_tokens: Optional[int]`` +
     ``temperature: Optional[float]`` fields with
     omit-when-None canonical encoding. Pre-3.x.10.x signed
     bytes remain verifiable (mirrors the Phase 3.x.8
     ``streaming`` flag pattern). No protocol-version bump —
     additive optional fields under v2. Bool-rejection guards
     prevent ``True`` / ``False`` slipping through Python's
     ``bool ⊂ int`` subclass relationship. Range validation:
     ``max_tokens > 0``, ``temperature ∈ [0.0, 2.0]`` with
     ``0.0`` accepted as the runner's greedy-decode signal.

  2. **Server ``request=`` plumbing (M4 closure).**
     ``LayerStageServer.handle_token_stream`` constructs a
     ``StreamingSamplingShim(max_tokens, temperature)`` from
     the parsed wire fields and forwards as ``request=`` to
     the streaming runner. ``StreamingLayerRunner`` Protocol
     formalized with ``request: Any = None`` (was a soft
     extension in 3.x.10). ``SyntheticStreamingRunner``
     accepts + ignores the kwarg.

  3. **Executor populates wire fields (M4 closure).**
     ``RpcChainExecutor._dispatch_streaming_tail`` propagates
     ``InferenceRequest.max_tokens`` + ``.temperature`` into
     the streaming ``RunLayerSliceRequest``. Streaming-only:
     unary ``RunLayerSliceRequest`` construction sites stay
     untouched (non-tail stages have no autoregressive decode
     to override; sampling overrides on those messages would
     be dead metadata).

  4. **Production factories (M5 closure).** New
     ``make_autoregressive_streaming_runner(model, tokenizer,
     tee_attestation, prompt_provider, ...)`` factory in
     ``prsm/compute/inference/factories.py`` builds a
     production-ready runner with operator-friendly RuntimeError
     validation. ``make_layer_stage_server`` gains optional
     ``streaming_runner=`` kwarg; default ``None`` preserves
     back-compat (server rejects token-stream requests with
     INTERNAL_ERROR "not configured for streaming") for
     operators not opting in. SDK callers can wire a streaming
     node in ~5 lines:
     ``from prsm.compute.inference import make_autoregressive_streaming_runner``.

  5. **Byte-equivalence pinned by golden bytes.** Round-1
     review M-TEST-1 remediation: a hardcoded canonical-bytes
     baseline (deterministic across runs) is asserted in
     ``test_golden_canonical_bytes_pre_3_x_10_x_baseline``.
     Any future patch breaking pre-3.x.10.x signature
     verification fails this assertion explicitly.

**Trust seams added by 3.x.10.x.**

  1. ``StreamingSamplingShim`` (streaming_runner.py:51-77) —
     minimal frozen dataclass; runner reads via
     ``getattr(request, "max_tokens", None)``. Decouples the
     runner from the chain_rpc protocol's full envelope.

  2. ``make_autoregressive_streaming_runner`` (factories.py)
     — operator-facing construction with clear RuntimeError
     messages on misconfig (model, tokenizer, tee_attestation,
     prompt_provider). ``tee_attestation`` stays
     operator-sourced (their TEE runtime produces it; the
     factory does not derive from identity material to keep
     TEE platform decisions where they belong).

**v1 honest scope (carries forward).**

  - Tier C constant-time padding still gated until Phase
    3.x.10.y.
  - Sharded autoregressive still gated until Phase 3.x.11.
  - Stop sequences not implemented; only EOS + ``max_tokens``.
  - HF generate buffering: tokens reach the wire only after
    ``.generate()`` returns. Async-during-generate is Phase
    3.x.10.y perf upgrade.
  - Streaming + chunked-input composition still rejected.
  - **NEW for 3.x.10.x — HF prompt-echo behavior.** With
    GPT-2-family byte-level BPE tokenizers, HF's TextStreamer
    emits the prompt back as the first wire chunk before the
    generated tokens. This is observable in the full-stack E2E
    (``test_max_tokens_propagates_end_to_end`` asserts
    cap propagation via ``finish_reason="max_tokens"`` rather
    than chunk count). The ``skip_prompt`` toggle on
    HF TextStreamer would address it; tracked for Phase
    3.x.10.y.

**Test coverage at the tag.**

  - 18 wire-format unit tests in ``test_chain_rpc_protocol.py``
    (TestSamplingOverridesByteEquivalence × 5 incl. golden-bytes
    pin; TestSamplingOverridesValidation × 9;
    TestSamplingOverridesRoundTrip × 5).
  - 6 server-shim plumbing tests in
    ``test_autoregressive_runner_server_wire.py`` — wire
    max_tokens reaches model.generate; temperature=0.0
    triggers greedy; both unset falls back to defaults;
    SyntheticStreamingRunner Protocol back-compat preserved.
  - 4 executor propagation tests in ``test_chain_rpc_client.py``
    — fields propagate to streaming wire request; unset
    InferenceRequest sends None; temperature=0.0 propagates
    as real value; unary path stays untouched.
  - 11 factory tests in ``test_inference_factories.py`` —
    factory builds runner; smoke 1-token decode; rejects each
    misconfig; passes through sampling_defaults + tee_type;
    make_layer_stage_server back-compat preserved on None;
    autoregressive + synthetic runners both wire through.
  - 4 slow-marked HF distilgpt2 full-stack E2E tests in
    ``test_phase3_x_10_x_full_stack_e2e.py`` — max_tokens=4
    cap propagates end-to-end; greedy bit-identical across
    independent server constructions; no-overrides falls back
    to defaults; signed response verifies under stage identity.

**Round-1 → round-2 surface.**

  - Round-1: APPROVED-WITH-PRE-TAG-REMEDIATIONS. 0 HIGH; 2
    MEDIUM (M-DOC-1: §7.7 missing + §7.6 stale; M-TEST-1:
    byte-equivalence test tautological — needed golden-bytes
    pin); LOW findings deferred or no-op verified.
    Implementation technically sound; protocol formalization
    safe; bool-rejection + range validation + None
    distinguishability + lazy-import correctness all verified.
  - Both MEDIUMs remediated pre-tag.
  - Round-2: APPROVED-FOR-TAG.

**Auditor prompts:** start with the byte-equivalence golden
test (``tests/unit/test_chain_rpc_protocol.py``,
``test_golden_canonical_bytes_pre_3_x_10_x_baseline``). It's
the load-bearing pre-3.x.10.x signed-traffic compatibility
guarantee. Then read ``LayerStageServer.handle_token_stream``
shim construction (``prsm/compute/chain_rpc/server.py:829-855``)
for the wire-to-runner forward. The factory pair
(``prsm/compute/inference/factories.py`` +
``prsm/compute/chain_rpc/factories.py``'s ``streaming_runner=``
kwarg) is the operator-facing surface. Full-stack E2E
(``tests/integration/test_phase3_x_10_x_full_stack_e2e.py``)
proves the chain end-to-end against real distilgpt2; the
``finish_reason="max_tokens"`` assertion is the load-bearing
cap-propagation invariant.

---

## 8. Auditor handoff checklist

When the Foundation signs the auditor contract:

- [ ] Send this document + `docs/2026-04-21-audit-bundle-coordinator.md` (refreshed with pointer to this bundle).
- [ ] Send the four merge-ready tags + this cumulative tag for git checkout.
- [ ] Provide read access to `MEMORY.md` summaries for each phase's history (round-1 findings + remediation).
- [ ] Schedule kickoff call covering §3 (cross-cutting threat model) — that's where seam-bugs hide.
- [ ] Set expectations: ≤7 additive audit days for this refresh; full engagement ~3-4 weeks bundled with the 2026-04-22 baseline.

---

## 9. Changelog

- **0.1 (2026-04-27)** — initial cumulative refresh covering 3.x.2/3/4/5 + Phase 4 Task 3. Tag pending at commit `107fb150`.
- **0.2 (2026-04-27)** — added §7.1 "Third-party-derived components" covering Phase 3.x.6 vendor scope (Parallax decentralized inference scheduler, Apache 2.0). Vendor boundary documented; PRSM-original delta (four trust adapters + executor) called out for auditor focus.
- **0.3 (2026-04-28)** — added §7.2 "Cross-Host ChainExecutor" covering Phase 3.x.7 PRSM-original cross-host inference path (RpcChainExecutor + LayerStageServer + multi-stage TEE attestation envelope). 6 trust seams called out for auditor focus including the H2 substitution-rejection invariant remediated pre-tag.
- **0.4 (2026-04-28)** — added §7.3 "Chunked Activation Streaming" covering Phase 3.x.7.1 v2 wire-format extension. 6 trust seams including pre-consumption envelope gate, full-envelope response signing payload, v1↔v2 byte-equivalent inline encoding, and bounded chunk iterators (client + server). Round-1 H1+H2+M3+L1 + round-2 I1 closed pre-tag. Phase 3.x.7.1 tag: phase3.x.7.1-merge-ready-20260428 at 339957ee.
- **0.5 (2026-04-28)** — added §7.4 "Streaming-Token Output" covering Phase 3.x.8 chat-style incremental output path. 6 trust seams including conditional `streaming` flag encoding (byte-identity preservation), three-layer joined-text invariant enforcement, downgrade-resistant `streamed_output` receipt flag, server-side validation gates BEFORE runner dispatch, M1 round-1 sole-error-frame ordering remediation, and cancellation cleanup propagation. v1 honest-scope notes (SyntheticStreamingRunner placeholder + Python GeneratorExit cancellation limit) explicitly documented for auditor focus. Round-1 M1 closed pre-tag; 3 LOWs deferred. Phase 3.x.8 tag: phase3.x.8-merge-ready-20260428 at 391b92b0.
- **0.6 (2026-04-28)** — added §7.5 "Streaming HTTP Endpoint" covering Phase 3.x.8.1 SSE-framed POST /compute/inference/stream. 6 trust seams including design plan §3.4 settle-on-tokens-emitted billing policy (closes a real griefing vector caught at round-1 review), wire-side receipt re-sign on job_id rebind (Task 5 caught + fixed bug invisible to mocked tests), W3C-compliant SSE framing with chunk-boundary buffering, operator-misconfig 503s, InferenceError structured error surface, request-time privacy-budget gating. Round-1 M1+M2+L1 closed pre-tag; 2 LOWs deferred. Phase 3.x.8.1 tag: phase3.x.8.1-merge-ready-20260428 at 67fe8863.
