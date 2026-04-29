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

## 7.8 Tier C Constant-Time Padding + HF Prompt-Echo Fix (Phase 3.x.10.y)

Phase 3.x.10.y is a 6-task slice closing two open issues from
Phase 3.x.10.x:
- Tier C streaming was structurally blocked at the dispatch
  layer per the timing-sidechannel memo §4 — operators serving
  Tier C content fell back to the unary path.
- HF TextStreamer's prompt-echo behavior leaked the prompt
  back as the first wire chunk for byte-level BPE tokenizers
  (observed in 3.x.10.x's full-stack E2E for distilgpt2).

**Tag:** ``phase3.x.10.y-merge-ready-20260429`` at commit
``[applied at tag time]``.

**Headline guarantees.**

  1. **HF prompt-echo fix.** ``_HFStreamerAdapter`` gains
     ``prompt_id_count`` constructor kwarg. While in prompt
     phase (cumulative token count ≤ ``prompt_id_count``),
     ``put()`` accumulates ids without flushing. On the
     boundary-crossing call, ``_print_offset`` is pinned to
     the prompt's decoded length so subsequent
     ``_maybe_flush`` only emits text from generated tokens
     onward. ``end()`` no-ops while still in prompt phase
     (zero-token generation MUST NOT leak prompt text).
     Mirrors HF ``TextStreamer(skip_prompt=True)`` semantics.
     ``AutoregressiveStreamingRunner`` constructs the adapter
     with ``prompt_id_count=len(input_ids)``. Default
     ``prompt_id_count=0`` preserves Phase 3.x.10 back-compat.

  2. **Tier C constant-time padding decorators.** New
     ``prsm/compute/inference/tier_c_decorators.py`` provides
     two ``StreamingLayerRunner``-Protocol-conforming wrappers
     per the timing-sidechannel memo §5:
     - **M2 ``BatchedTrailingStreamingRunner``** — drains the
       inner stream fully and emits ONE terminal chunk with
       joined text. Per-token timing is structurally
       unobservable (one wire frame, no inter-frame deltas).
       Sacrifices streaming UX for paranoid Tier C.
     - **M1 ``FixedRateStreamingRunner``** — daemon producer
       thread iterates the inner runner; consumer-side
       cadence loop sleeps ``cadence_seconds`` between yields,
       drains at most ONE inner chunk per tick. No-op pad
       frames (empty ``text_delta``) fill gaps when no inner
       chunk is ready. Inner terminal stashed for cadence-
       aligned emission after buffer drain. Inter-frame
       wall-clock latency is the cadence (constant),
       independent of inner runner's per-token decode time.

  3. **Tier C dispatch-layer gate (default-deny).**
     ``LayerStageServer.__init__`` accepts
     ``tier_c_streaming_decorator: Optional[Callable[[Any], Any]]``.
     ``handle_token_stream`` branches on
     ``request.content_tier == ContentTier.C``:
     - decorator unset → ``StageError(INTERNAL_ERROR,
       "Tier C streaming requires constant-time padding
       decorator")``;
     - decorator set → wraps streaming_runner per-request,
       validates structural shape (must expose
       ``run_layer_slice_streaming``), invokes wrapped runner.
       Decorator exceptions caught and surfaced as
       INTERNAL_ERROR — NEVER raises through the wire boundary.
     ``make_layer_stage_server`` exposes the kwarg.

  4. **GeneratorExit-driven cleanup.** Round-1 review M1
     remediation: M1's consumer ``finally`` sets a
     ``threading.Event`` that the producer checks between
     inner-iterator steps; producer breaks, calls
     ``inner.close()`` to release accelerator state, exits.
     Daemon-thread semantics handle process-level cleanup
     if the producer is stuck mid-blocking-call in HF's sync
     ``model.generate``. Module docstring at lines 195-204
     documents this contract.

  5. **No partial-state leak through error path.** Round-1
     review H1 remediation: M1's inner-exception path now
     yields NOTHING and lets the server's "exhausted without
     terminal chunk" surface INTERNAL_ERROR cleanly. The
     decorator no longer fabricates ``tee_type`` /
     ``tee_attestation`` values it doesn't legitimately
     hold (which would have failed the server's terminal-
     aggregate-fields gate with a misattributed error).
     Pad frames before error-exit also have empty
     ``text_delta`` and unpopulated aggregate fields — no
     partial inner state observable on the wire.

**Honest scope (carries forward).**

  - **Total stream duration leaks total token count.** Even
    under M1 + M2: stream length still encodes one observation
    of output size. M1 leak ceiling = ``cadence × frame_count``;
    M2 leak ceiling = a single duration value. Operators
    bound the leak by capping ``max_tokens`` for Tier C
    requests.
  - **No-op pad frames inflate wire frame count.** With M1
    cadence=50ms over a 10-second decode, ~200 frames. A
    future optimization could batch consecutive no-op pads;
    deferred to Phase 3.x.10.z.
  - **Async-during-generate** still deferred to Phase 3.x.10.z.
    HF's ``generate()`` is synchronous; tokens reach the wire
    only after the call returns. M1's cadence preserves the
    illusion of streaming UX during decode but the underlying
    timing is "all-at-once after generate returns".
  - **Sharded autoregressive** still gated until Phase 3.x.11.

**Test coverage at the tag.**

  - 6 prompt-echo tests in
    ``test_autoregressive_runner.py::TestHFPromptEchoSkip``
    (skip-prompt semantics, back-compat at
    ``prompt_id_count=0``, runner end-to-end wiring,
    zero-generation edge case, EOS detection compatibility).
  - 30 decorator unit tests in
    ``tests/unit/test_tier_c_decorators.py`` (Protocol
    conformance, happy paths, edge cases, constructor
    validation, cadence timing-mask, error-path no-leak,
    real-distilgpt2 compose smoke for both M1 + M2).
  - 9 dispatch-gate tests in
    ``test_autoregressive_runner_server_wire.py::TestTierCDispatchGate``
    (Tier A/B back-compat, Tier C default-deny, decorator
    apply, decorator misconfig paths, factory forwarding).
  - 6 timing-observer E2E tests in
    ``tests/integration/test_phase3_x_10_y_timing_observer.py``
    (slow-marked) — the load-bearing proof that the timing-
    mask invariant holds: undecorated baseline leaks
    per-token timing (stdev ≥ 15ms with the
    ``[5, 80, 5, 80]`` ms profile); M2 collapses to single
    frame; M1 stdev < 10ms; cross-decorator comparison
    asserts ``baseline_stdev > 3 × m1_stdev``.

  Conftest interaction: two test files override
  ``conftest.py``'s autouse ``time.sleep`` mock with a same-
  named fixture that yields without patching, restoring real
  wall-clock sleep for those files. Without the override, the
  cadence + timing-mask assertions would silently pass with
  every sleep collapsed to instant.

**Round-1 → round-2 surface.**

  - Round-1: APPROVED-WITH-PRE-TAG-REMEDIATIONS. 1 HIGH (M1
    error-path field shape — terminal chunk with
    ``tee_type=None`` failed server's aggregate-fields gate
    and surfaced misattributed StageError). 2 MEDIUM (M1
    docstring/impl mismatch on GeneratorExit cleanup; §7.8
    missing). All three remediated pre-tag.
  - Round-2: APPROVED-FOR-TAG.

**Auditor prompts:** start with the timing-observer E2E
``tests/integration/test_phase3_x_10_y_timing_observer.py::TestTimingMaskComparison``
— that's the load-bearing proof of the mask invariant. Then
read the dispatch-gate code at
``prsm/compute/chain_rpc/server.py:891-924`` for the default-
deny enforcement (Tier C without decorator → INTERNAL_ERROR).
The two decorators in
``prsm/compute/inference/tier_c_decorators.py`` are
StreamingLayerRunner Protocol-conforming wrappers; the
threading model in M1 (lines 153-272) is the only non-trivial
concurrency surface, with GeneratorExit-driven cleanup
documented inline. The HF prompt-echo fix at
``prsm/compute/inference/autoregressive_runner.py:124-188``
is the smaller surface — boundary-crossing semantics in
``put()``, with ``end()`` guarded against zero-generation
prompt-leak.

---

## 7.9 Sharded Autoregressive Decode (Phase 3.x.11)

Phase 3.x.11 is a 9-task slice closing the load-bearing
tail-only contract that's accumulated since Phase 3.x.8: each
chain stage now runs its layers ONCE PER GENERATED TOKEN, with
KV-cache state surviving locally between iterations and
activations crossing the wire at every per-token boundary. The
load-bearing capability for PRSM's distributed-inference value
prop — an operator with a single 24GB GPU can participate in
serving a 70B-parameter inference by hosting one or two layers
of the chain.

**Tag:** ``phase3.x.11-merge-ready-20260430`` at commit
``[applied at tag time]``.

**Headline guarantees.**

  1. **Wire-format extension (Task 1).** ``RunLayerSliceRequest``
     gains ``decode_mode: DecodeMode`` (PREFILL/INCREMENTAL,
     omit-when-PREFILL canonical encoding). ``RunLayerSliceResponse``
     gains ``next_token_id: Optional[int]`` + ``is_terminal: bool``
     (omit-when-default). Both fields preserve byte-equivalence
     with pre-3.x.11 signed bytes — golden-bytes pin in
     ``test_chain_rpc_protocol.py`` enforces this. Validation
     guards bool-rejection on enum + non-negative int range on
     token id.

  2. **Server-side KV-cache manager (Task 2).** New
     ``prsm/compute/chain_rpc/kv_cache.py`` with
     ``KVCacheManager`` + ``KVCacheHandle``. Allocate / get /
     evict / evict_idle / evict_all lifecycle. LRU cap (default
     64) + TTL sweeper (default 300s). Thread-safe via
     ``threading.Lock``; ``OrderedDict`` for O(1) LRU bookkeeping.
     Handle's ``payload`` is opaque to the manager — runner-defined
     (typically per-layer K/V tensors); ``tokens_generated``
     counter (Task 4 addition) tracks per-request token count for
     ``max_tokens`` cap enforcement.

  3. **ShardedAutoregressiveRunner non-tail variant (Task 3).**
     New ``prsm/compute/inference/sharded_runner.py``. Per-stage
     layer-range runner with KV-cache lifecycle. Drives the
     duck-typed ``ShardedLayerForward`` Protocol
     (``forward_prefill`` + ``forward_incremental``). PREFILL
     allocates fresh cache via manager + drives full-prompt
     forward; INCREMENTAL looks up existing handle + drives
     single-position forward with cached KV. Raises
     ``MalformedCacheStateError`` (mapped to MALFORMED_REQUEST
     at wire boundary) when INCREMENTAL arrives with no prior
     PREFILL handle.

  4. **Tail variant (Task 4).** Same runner extended with
     ``apply_lm_head_and_sample`` model method + ``sampling_defaults``
     + ``eos_token_id`` constructor args. Tail dispatch
     (``is_final_stage=True``) projects boundary hidden state
     through LM head, samples per request's sampling params,
     bumps handle's ``tokens_generated``, sets ``is_terminal``
     on EOS detection OR ``tokens_generated >= max_tokens``.
     Non-tail-only construction (no sampling_defaults) preserves
     Task 3's contract: tail dispatch raises
     ``MissingTailCapabilityError``.

  5. **Executor per-token chain loop (Task 5).** New
     ``RpcChainExecutor`` constructor args:
     ``enable_sharded_decode`` + ``tokenizer`` +
     ``cache_evict_send_message`` +
     ``sharded_default_max_tokens``. When enabled,
     ``execute_chain_streaming`` branches to
     ``_execute_chain_streaming_sharded`` —
     tokenizer.encode(prompt) → input_ids → PREFILL chain pass
     → INCREMENTAL decode loop → terminal yields
     ``ChainExecutionResult`` with multi-stage receipt + ``finally``
     block broadcasts ``EvictCacheRequest`` to every stage.
     ``include_sampling_fields=True`` opt-in flag preserves the
     Phase 3.x.10.x non-streaming-tail invariant (existing
     ``test_unary_request_path_unchanged_no_sampling_fields``
     pin).

     **Critical security fix.**
     ``RunLayerSliceResponse.signing_payload`` extended with
     ``next_token_id`` + ``is_terminal`` kwargs (omit-when-default).
     Without this, a malicious downstream relay could swap the
     sampled token without invalidating the stage's signature.
     Pre-3.x.11 signed bytes preserve byte-equivalence (both
     fields default-omitted match pre-3.x.11 canonical JSON).

  6. **EvictCacheRequest wire message + handler (Task 6).** New
     ``EvictCacheRequest`` + ``EvictCacheResponse`` envelopes in
     ``ChainRpcMessageType``. ``LayerStageServer.kv_cache_manager``
     opt-in arg routes ``EvictCacheRequest`` to
     ``_handle_evict_cache`` → ``manager.evict``. Honest scope:
     no signature on the eviction signal — non-load-bearing for
     correctness because the server-side TTL sweeper closes the
     same hole. Idempotent (duplicate broadcasts safe).

  7. **Server-side sharded dispatch (Task 7).**
     ``LayerStageServer.sharded_runner`` opt-in arg routes ALL
     ``RunLayerSliceRequest``s to ``_dispatch_sharded``. When
     NOT wired but request carries ``decode_mode != PREFILL``,
     rejects with MALFORMED_REQUEST (back-compat regular runners
     can't honor sharded semantics). ``is_final_stage`` inferred
     from handoff token's ``chain_stage_index ==
     chain_total_stages - 1``.

  8. **Tier C structural deny (Task 7 honest scope).**
     ``_dispatch_sharded`` rejects ``ContentTier.C`` with
     INTERNAL_ERROR. Sharded decode introduces a NEW timing
     surface (per-token wire dispatch) that Phase 3.x.10.y's
     constant-time padding decorators don't cover. Phase
     3.x.11.q is the placeholder for sharded constant-time
     work; until then, Tier C operators MUST keep
     ``enable_sharded_decode=False``. The dispatch gate is the
     load-bearing enforcement.

  9. **Bit-identical real-distilgpt2 E2E (Task 7).** 4-token
     greedy decode through 2-stage sharded chain (alice: layers
     0-2 + embeddings; bob: layers 3-5 + LM head) produces
     token_ids matching ``model.generate(do_sample=False)``
     bit-identically. Load-bearing correctness proof via real
     HF transformers 5.x ``DynamicCache`` + ``GPT2Block.forward``
     with ``cache_position``.

**Honest scope (carries forward).**

  - **Per-token wire latency tax.** T network round-trips per
    token. WAN deployments: ~150ms minimum per token at 50ms
    RTT × 3 hops. LAN: negligible. Operators serving over WAN
    should keep ``enable_sharded_decode=False``. Pipelining
    (Phase 3.x.11.x) overlaps consecutive-token forward passes
    to amortize.
  - **Tier C incompat.** Sharded decode is Tier-A/B-only at v1
    (per §3.4 of the threat-model addendum). Phase 3.x.11.q.
  - **No memory wipe of evicted KV-cache.** Eviction drops the
    handle reference; Python GC eventually reclaims, but bytes
    may remain in heap until reused. Operator mitigation:
    TEE / user namespaces. Threat-model addendum §3.2.
  - **No KV-cache commitment in receipt.** Signed receipt
    commits to wire activation bytes (per Phase 3.x.7 Task 5
    envelope) but NOT to in-stage cache state. Phase 3.x.11.x
    receipt-format extension (per-token attestation chain).
  - **Cross-stage activation handoff magnification.** Each
    stage observes ``1 + max_tokens`` boundary hidden states
    per request (vs. 1 in the unary/streaming-tail path).
    R3 baseline mitigations (topology randomization, DP noise,
    TEE) carry forward but their per-request privacy budget
    consumes faster. Threat-model addendum §3.3.
  - **No pipelining.** Phase 3.x.11.x.
  - **No speculative decoding.** Phase 3.x.11.y.
  - **No KV-cache compression for cross-host bandwidth.** R7
    benchmark plan §9 covers data-oblivious KV/activation
    compression as research; engineering deferred to Phase
    3.x.11.x.
  - **No cache swap-out / paging.** Long-context workloads that
    exceed per-stage memory caps are rejected. Phase 3.x.11.z.
  - **No mid-stream re-routing.** Once a request's chain is
    committed, all incremental dispatches go to the same stages.
    Phase 3.x.12.

**Threat-model coverage.**

The new threat surfaces introduced by sharded decode are
explicitly characterized in
``docs/2026-04-30-phase3.x.11-threat-model-addendum.md``:

  - §3.1 Per-token wire timing surface — `N × T` timing
    observations per request (vs. 1 in unary, N in streaming-
    tail). Tier C structural deny is the v1 mitigation.
  - §3.2 KV-cache state privacy on stages — temporal coverage
    extends across the full decode (vs. one snapshot per
    request). Eviction + TTL + LRU cap bound the residue
    window. No memory wipe / no encryption-at-rest in v1.
  - §3.3 Cross-stage activation handoff magnification — per-
    request observation surface multiplies by `1 + max_tokens`.
    R3 baseline mitigations carry forward; effective strength
    against scale changes (S2/S3 attacker accumulates
    `(1+max_tokens) × R` activations vs. R).
  - §3.4 Tier C structural incompat — default-deny at dispatch.
    Phase 3.x.11.q deferred.

The R3 threat model
(``docs/2026-04-22-r3-threat-model.md``) cross-references the
addendum in its "Related documents" header — auditors evaluating
sharded-mode operators read both in conjunction.

**Test coverage at the tag.**

  - 28 ``KVCacheManager`` unit tests in
    ``tests/unit/test_kv_cache.py`` (allocate / get / evict /
    evict_idle / LRU / TTL / concurrent allocation).
  - 46 ``ShardedAutoregressiveRunner`` unit tests in
    ``tests/unit/test_sharded_runner.py`` (constructor
    validation, prefill, incremental, layer_range respected,
    cache survival, eviction, tail variant sampling +
    determinism + EOS + max_tokens, tail constructor
    validation).
  - 19 sharded executor tests in
    ``tests/unit/test_chain_rpc_client_sharded.py``
    (construction validation, single-stage smoke + decode_mode
    threading, two-stage autoregressive, max_tokens cap, EOS,
    cancellation eviction broadcast, sampling propagation).
  - 15 ``EvictCacheRequest`` tests in
    ``tests/unit/test_evict_cache.py`` (round-trip, validation,
    server handler happy path / idempotent / unknown id /
    no-manager rejection / cross-request isolation).
  - 5 real-distilgpt2 E2E tests in
    ``tests/integration/test_phase3_x_11_sharded_e2e.py``
    (slow-marked) — bit-identical sharded vs single-host
    greedy + autoregressive non-trivial output + token id is
    real int + cache lifecycle survives + evicted on terminal
    + evicted on caller close.

**Round-1 → round-2 surface.** Pending Task 9 review.

**Auditor prompts:** start with the threat-model addendum
(``docs/2026-04-30-phase3.x.11-threat-model-addendum.md``) §1 +
§3.1 + §3.4 — the new timing + content-tier story. Then read
the bit-identical E2E test
(``tests/integration/test_phase3_x_11_sharded_e2e.py::TestShardedE2EBitIdentical``)
— that's the load-bearing correctness proof. The sharded
dispatch gate at
``prsm/compute/chain_rpc/server.py:_dispatch_sharded`` is the
Tier C default-deny enforcement + role inference from the
handoff token. The per-token executor loop at
``prsm/compute/chain_rpc/client.py:_execute_chain_streaming_sharded``
is the new dispatch surface; the ``finally`` block + eviction
broadcast are the cleanup contract. The signing-payload
extension at
``prsm/compute/chain_rpc/protocol.py:RunLayerSliceResponse.signing_payload``
is the load-bearing security fix (without it, downstream
relays could swap ``next_token_id`` without invalidating the
stage's signature). The cache lifecycle at
``prsm/compute/chain_rpc/kv_cache.py:KVCacheManager`` is
straightforward LRU/TTL/lock; the threading model is one
short-held lock around the OrderedDict.

---

## 7.10 Pipelining + Per-Token Receipt Attestation (Phase 3.x.11.x)

Phase 3.x.11.x is a 6-task slice closing two Phase 3.x.11
honest-scope deferrals: chunked + sharded PREFILL composition
+ per-token receipt attestation envelope. Compute-level
pipelining + speculative decoding remain in Phase 3.x.11.y;
Tier C sharded support remains Phase 3.x.11.q.

**Tag:** ``phase3.x.11.x-merge-ready-20260430``.

**Headline guarantees.**

  1. **IterationAttestation wire-format extension (Task 1).**
     New ``IterationAttestation`` dataclass + ``encode_multi_iteration_attestation``
     + ``decode_multi_iteration_attestation`` under separate
     magic prefix ``PRSM-MI-ATT-V1:`` (vs. legacy
     ``PRSM-MS-ATT-V1:``). Discriminator at the magic-prefix
     level rather than JSON-key-level: old decoders return
     None on the new envelope (back-compat fall-through) +
     new decoder rejects legacy envelope (no cross-confusion).
     Validation: PREFILL/INCREMENTAL coupled to iteration_index
     (0=PREFILL, >0=INCREMENTAL); contiguous iteration_index
     0..N-1; uniform stage counts across iterations; contiguous
     stage_index 0..M-1 within each iteration. Golden-bytes
     pin on the existing multi-stage envelope reconstructs
     canonical JSON from first principles (non-sharded
     receipts byte-equivalent with pre-3.x.11.x).

  2. **Executor per-iteration accumulation (Task 2).**
     ``_execute_chain_streaming_sharded`` replaces flat
     ``cumulative_outcomes: List[StageOutcome]`` (Phase 3.x.11)
     with ``per_iteration_outcomes: List[List[StageOutcome]]``
     + parallel ``per_iteration_decode_modes: List[DecodeMode]``.
     New ``_build_sharded_chain_result`` helper builds one
     IterationAttestation per chain pass, encodes via
     ``encode_multi_iteration_attestation``, aggregates
     duration_seconds + epsilon_spent across ALL iterations,
     uses ``worst_case_tee_type_across_iterations`` for the
     receipt-level tee_type (one SOFTWARE stage in any
     iteration drags the whole receipt to SOFTWARE).
     Closes Phase 3.x.11 threat-model addendum §3.2's "no
     per-iteration cryptographic commitment in receipt" gap.

  3. **Server-side chunked + sharded PREFILL composition (Task 3).**
     New ``_dispatch_streamed_sharded`` lifts the Phase 3.x.11
     Task 9 M1 unary-only guard for PREFILL when sharded_runner
     is wired. INCREMENTAL streamed stays rejected (single-
     position activations don't benefit from chunking).
     Reuses Phase 3.x.7.1 chunked-streaming substrate
     (``_run_validation_gates`` + ``_validate_streamed_envelope``
     for the Phase 3.x.7.1 H1+M1 inflated-payload-bytes
     defence + ``_reassemble_inbound_chunks`` + ``reassemble_chunked``
     + ``chunk_activation``). Tier C structural deny carries
     forward (mirrors ``_dispatch_sharded``). Response signed
     with ``next_token_id`` + ``is_terminal`` (Phase 3.x.11
     Task 5 critical-fix coverage extends to this path).

  4. **Executor lifts chunked + sharded PREFILL guard (Task 4).**
     ``_dispatch_stage`` opens ``should_chunk + enable_sharded_decode``
     branch for PREFILL only. INCREMENTAL chunked still
     raises ACTIVATION_TOO_LARGE with refined "INCREMENTAL is
     unary-only ... single-position" message. Pairs with
     Task 3 server-side opening.

  5. **Real-distilgpt2 chunked-PREFILL E2E (Task 5).**
     ``test_chunked_prefill_path_exercised`` verifies streamed
     transport hit during PREFILL with ~30+ token prompt at
     chunk_threshold=10 KiB (Stage 1 → Stage 2 hidden state
     at distilgpt2's 768 hidden_dim × 4 bytes/fp32 × 30+
     positions ≈ 92+ KiB > threshold). Bit-identical greedy
     output vs single-host (Phase 3.x.11 Task 7 invariant
     carries through chunked path). Receipt's tee_attestation
     decodes as multi-iteration envelope with 1 PREFILL + 2
     INCREMENTALs entries × 2 stage records each.

**Honest scope (carries forward).**

  - **Compute-level pipelining** — overlapping consecutive-
    token decode forward passes requires speculation; ships
    as Phase 3.x.11.y. v1 "pipelining" framing in this slice
    is wire-level only (chunked PREFILL transport overlaps
    Stage K chunk emission with Stage K+1 chunk consumption).
  - **Speculative decoding** — Phase 3.x.11.y.
  - **Per-token KV-cache Merkle commitment in receipt** —
    Phase 3.x.11.x' (apostrophe; not in this slice).
    Receipt commits to per-iteration *attestations* (proves
    the stage was on-watch); does NOT commit to the K/V
    tensors themselves. Adding a Merkle root per iteration is
    significant compute overhead, deferred.
  - **Tier C sharded compat** — Phase 3.x.11.q. Sharded path
    introduces new timing surface that Phase 3.x.10.y M1/M2
    decorators don't cover.
  - **Cache swap-out / paging** — Phase 3.x.11.z.
  - **Mid-stream re-routing** — Phase 3.x.12.

**Critical security carry-over.** Phase 3.x.11 Task 5's
``RunLayerSliceResponse.signing_payload`` extension (committing
``next_token_id`` + ``is_terminal``) covers this path too —
chunked-sharded responses are signed with the same envelope as
inline-sharded responses, so a malicious downstream relay
can't swap the sampled token without invalidating the
signature regardless of which transport path was used.

**Test coverage at the tag.**

  - 33 ``IterationAttestation`` unit tests in
    ``tests/unit/test_multi_iteration_attestation.py``
    (construction validation, round-trip, discriminator,
    structural validation, worst-case TEE, golden-bytes pin
    on legacy envelope).
  - 5 per-iteration receipt tests in
    ``tests/unit/test_chain_rpc_client_sharded.py::TestShardedPerIterationAttestation``
    (1-iteration / 4-iteration shape; duration + epsilon
    aggregation; non-sharded receipt unchanged byte-equivalence;
    cancellation no partial receipt).
  - 4 executor-guard tests in
    ``tests/unit/test_chain_rpc_client_sharded.py::TestShardedChunkedPrefillExecutorGuard``
    (INCREMENTAL still rejected; non-sharded chunked unchanged;
    non-chunked sharded unchanged; PREFILL chunked routes to
    streamed transport).
  - 8 server-side dispatch tests in
    ``tests/unit/test_dispatch_streamed_sharded.py``
    (happy-path PREFILL non-tail + tail; INCREMENTAL/Tier C
    rejections; envelope validation; chunk corruption; runner
    exception mapping; non-sharded back-compat).
  - 3 slow real-distilgpt2 E2E tests in
    ``tests/integration/test_phase3_x_11_sharded_e2e.py::TestShardedE2EChunkedPrefill``
    (chunked path exercised; bit-identical greedy; receipt
    carries iteration envelope).

**Round-1 → round-2 surface.**

  - Round-1: APPROVED-FOR-TAG with 3 LOW findings.
    LOW-2 (defensive ``decode_mode == PREFILL`` assert at
    ``_dispatch_streamed_sharded`` entry — prevents the same
    class of seam-bug as Phase 3.x.11 Task 9 M1) remediated
    pre-tag.
    LOW-1 (golden-bytes pin reconstructs canonical bytes via
    json.dumps rather than hardcoded hex) deferred — current
    pin catches all realistic regression vectors; hardcoded
    hex is belt-and-braces audit-prep.
    LOW-3 (``_build_sharded_chain_result`` empty-iterations
    dead-code path) deferred — non-load-bearing under the
    Phase 3.x.11 Task 5 cancellation-no-receipt contract.

**Auditor prompts.** Start with the bit-identical chunked E2E
test (``tests/integration/test_phase3_x_11_sharded_e2e.py::TestShardedE2EChunkedPrefill::test_chunked_prefill_output_matches_single_host_greedy``)
— that's the load-bearing correctness proof for the
chunked-PREFILL composition. Then read
``prsm/compute/inference/multi_stage_attestation.py`` for the
new envelope shape; the discriminator pattern (separate magic
prefixes) is the load-bearing back-compat enabler. The
``_dispatch_streamed_sharded`` method at
``prsm/compute/chain_rpc/server.py:1038-1217`` is the new
server-side dispatch path; mirrors ``_dispatch_sharded`` for
the inline path with the same Tier C deny + envelope-validation
ordering + runner-exception mapping. The defensive PREFILL
assert at the top of ``_dispatch_streamed_sharded`` (LOW-2
remediation) prevents future-refactor seam-bugs of the
same class as Phase 3.x.11 Task 9 M1. The
``_build_sharded_chain_result`` helper at
``prsm/compute/chain_rpc/client.py:930-1000`` is straightforward
reduction; the empty-iterations branch is dead code under the
current contract (LOW-3 deferral).

---

## 7.11 Speculative Decoding (Phase 3.x.11.y)

Phase 3.x.11.y is a 9-task slice closing the remaining honest-
scope deferral from Phase 3.x.11.x: compute-level pipelining via
speculative decoding. A co-located draft model proposes K=4
candidate tokens per round; the sharded chain verifies them in a
single batched K+1-position forward; the executor accepts the
longest matching prefix and broadcasts ``RollbackCacheRequest``
for the rejected suffix. Under perfect-accept (matched draft +
verifier under greedy), this delivers up to 5× per-round token
amortization.

**Tag:** ``phase3.x.11.y-merge-ready-20260XXX`` at commit
``XXXX`` (pending Task 9 review).

**Headline guarantees.**

  1. **Wire-format extension (Task 1).** ``DecodeMode.VERIFY``
     enum value + ``RunLayerSliceResponse.verified_token_ids``
     (Tuple[int, ...] capped at ``MAX_VERIFY_BATCH_TOKENS=65``)
     + ``accepted_count`` (co-set with verified_token_ids;
     setting one without the other raises malformed at parse
     time) + ``RollbackCacheRequest`` / ``RollbackCacheResponse``
     wire envelopes. ``signing_payload`` extended to commit
     verified_token_ids + accepted_count when set — without
     it, a malicious relay could swap verified tokens without
     invalidating the response signature (mirrors Phase 3.x.11
     Task 5 critical-fix pattern). Pre-3.x.11.y signed bytes
     preserve byte-equivalence (omit-when-default canonical
     encoding pinned by the existing 26-test verify-wire suite).

  2. **KVCacheManager.rollback (Task 2).** New method takes a
     caller-injected ``truncate_fn`` — keeps the manager
     payload-opaque (matches existing allocate/get/evict
     pattern). Lock-held truncation prevents read-stale-cache
     races. Idempotent over-drop semantics (clamp to
     tokens_generated; unknown id / n<=0 / 0-tokens-generated
     all return rolled_back=False without raising).

  3. **DraftModel Protocol + HFDraftModel (Task 3).** New
     ``prsm/compute/inference/draft_model.py``. Reset/propose/
     commit/evict lifecycle. v1 ships greedy-only (raises on
     ``temperature != 0.0``; sampling-correct speculation under
     temperature > 0 requires the Leviathan-2023 rejection-
     sampling correction, deferred to Phase 3.x.11.y.x).
     ``HFDraftModel`` is stateless w.r.t. the model's KV cache —
     each propose re-runs ``model.generate()`` from canonical
     history. Documented as v1 simplification; a stateful-
     KVCache impl is a drop-in Protocol replacement.

  4. **ShardedAutoregressiveRunner VERIFY support (Task 4).**
     Three new optional ``ShardedLayerForward`` Protocol methods:
     ``forward_verify`` (batched K+1-position forward with
     cached KV, payload-opaque to runner),
     ``apply_lm_head_and_sample_batch`` (tail-only, projects
     K+1 hidden states + greedy argmax per position), and
     ``truncate_cache`` (drops last N positions in-place).
     Tail's ``_sample_tail_verify`` computes accepted_count via
     standard speculative-decoding longest-prefix match;
     ``handle.tokens_generated`` bumped by emitted count only
     (speculatively-cached-then-rejected positions don't count
     against ``max_tokens``). 17 unit tests (TestVerifyNonTail
     7 + TestVerifyTail 10) cover all-accepted / zero-accepted /
     partial / EOS-mid-emitted / max_tokens-cap-mid-round /
     proposed-length-mismatch / non-tail-hidden-state-passthrough.

  5. **Executor speculation loop (Task 5).** New
     ``_execute_chain_streaming_sharded_speculative`` branches
     when ``draft_model`` is wired. PREFILL same as Phase
     3.x.11; speculation loop drives draft.propose → chain
     VERIFY → emit verified[:accepted_count + 1] →
     ``RollbackCacheRequest`` broadcast for K+1 - emitted
     positions → draft.commit on actually-emitted prefix →
     continue with last-emitted as next parent. Greedy-only
     gate at entry rejects ``request.temperature > 0`` with
     PROMPT_ENCODE_ERROR. Mid-emit max_tokens truncation:
     when emitting accepted_count+1 would overshoot, truncate
     the emit and bump cached_extra so rollback drops both
     verifier-rejected and cap-truncated tail. Wire-format
     ``RunLayerSliceRequest.proposed_token_ids`` (Optional
     Tuple, omit-when-None) carries K drafts to the tail for
     accepted_count comparison; cross-validated as VERIFY-
     mode-only. ``IterationAttestation`` (Phase 3.x.11.x)
     extended to permit ``decode_mode in {INCREMENTAL, VERIFY}``
     for ``iteration_index > 0`` — speculative iterations
     carry per-stage attestations the same way INCREMENTAL
     iterations do.

  6. **RollbackCacheRequest server handler (Task 6).** New
     ``_handle_rollback_cache`` routes through the runner's
     new ``rollback_cache`` wrapper to
     ``KVCacheManager.rollback``. Runner's wrapper is the
     model-coupling layer — it provides
     ``model.truncate_cache`` as the manager's ``truncate_fn``,
     keeping the server generic.
     ``MissingVerifyCapabilityError`` (raised when model omits
     truncate_cache) maps to MALFORMED_REQUEST so the executor
     can distinguish caller bug from internal crash. Server-
     without-manager / server-without-runner / runner-without-
     rollback_cache paths all return INTERNAL_ERROR.

  7. **Real-distilgpt2 + HFDraftModel E2E (Task 7).**
     ``tests/integration/test_phase3_x_11_y_speculative_e2e.py``
     loads HF distilgpt2, splits 6 layers across 2 stages
     (alice 0-2, bob 3-5), wires HFDraftModel as the draft
     using the SAME distilgpt2 (perfect-accept oracle under
     greedy: every draft proposal matches the verifier's
     argmax). Drives 8-token speculative decode. **Bit-
     identical to single-host greedy proven** — speculation
     is a perf optimization, not a sampling change. Receipt's
     per-iteration envelope confirms ≥1 VERIFY iteration
     fired (vs falling back to single-token decode). Tokens
     emitted > chain iterations (proves amortization).
     Cancellation evicts both server caches AND draft state.

     **Critical adapter remediation during E2E bring-up.**
     The K+1 batched forward in HF GPT2's eager attention path
     defaults to FULL attention across new tokens (the
     auto-mask logic only kicks in for q_len=1 with cached
     past). Without an explicit 4D additive causal mask, each
     new query attends to all K+1 keys (including future
     positions), breaking greedy-equivalence vs single-token
     INCREMENTAL. The fix in the test adapter is an explicit
     mask of shape ``[1, 1, K+1, kv_len]`` with -inf in
     non-attendable positions. **This is GPT2-specific** —
     operators wiring other HF model adapters (Llama, Mistral,
     etc.) need to verify their attention impl handles K+1
     batched cached forward correctly under their attention
     path (sdpa / flash_attention_2 / eager). Documented in
     the adapter docstring.

**Trust seams + auditor focus.**

  1. **VERIFY response signing.** Tail's ``verified_token_ids``
     + ``accepted_count`` are committed in the response signing
     payload (Task 1 wire-format extension). Without this
     commitment, a downstream relay between the tail and the
     executor could swap verified tokens, causing the executor
     to emit wrong content or accept different tokens than the
     tail's actual sampling. The omit-when-default canonical
     encoding ensures pre-3.x.11.y signed bytes are preserved
     bit-equivalent (so legacy receipts continue to verify).
     **Mirrors Phase 3.x.11 Task 5's same-class fix for
     next_token_id + is_terminal.**

  2. **Greedy-only invariant at executor boundary.** v1
     speculation rejects ``temperature > 0`` at executor
     dispatch entry with PROMPT_ENCODE_ERROR. This is BOTH a
     correctness gate (no Leviathan-2023 sampling correction
     yet) AND a threat-model-containment gate (greedy
     speculation produces output bit-identical to non-
     speculative greedy, keeping the threat-surface comparison
     crisp — see threat-model addendum §3.5).

  3. **Rollback as best-effort.** ``RollbackCacheRequest``
     is sent best-effort with no executor-side ack-failure
     remediation; ``KVCacheManager.rollback`` is idempotent
     so duplicate / lost broadcasts don't corrupt cache state;
     server-side TTL sweeper bounds the leak window if a
     broadcast is dropped. **DoS-not-privacy concern**: a
     network adversary dropping rollback broadcasts grows
     stage caches but doesn't leak content. Authenticated
     rollback envelope deferred to Phase 3.x.11.y.x if
     telemetry warrants.

  4. **Per-iteration receipt VERIFY attestation.**
     ``IterationAttestation`` was extended (Task 5) to permit
     ``decode_mode in {INCREMENTAL, VERIFY}`` for
     ``iteration_index > 0``. Speculative iterations are
     committed in the receipt envelope just like INCREMENTAL
     iterations were under Phase 3.x.11.x. The
     iteration_index 0 → PREFILL invariant carries forward
     (pure addition; no breaking change).

  5. **Co-set invariant on response.** ``verified_token_ids``
     and ``accepted_count`` MUST be co-set on
     RunLayerSliceResponse — setting one without the other
     raises ``ChainRpcMalformedError`` at parse time. Defends
     against a malformed peer constructing a partial-VERIFY
     response that confuses the executor.

  6. **K+1 cap defense.**
     ``MAX_VERIFY_BATCH_TOKENS = 65`` caps the K+1 batch size
     at the wire layer (response.verified_token_ids len cap)
     AND at the request layer
     (request.proposed_token_ids len cap = K = 64). Defends
     against a hostile peer claiming a huge speculation depth
     that would explode server-side memory in
     forward_verify's K+1 batched attention computation.

**Honest scope (carries forward).**

  - **Sampling-correct speculation under temperature > 0** —
    Phase 3.x.11.y.x (Leviathan-2023 rejection-sampling
    correction). Greedy-only at v1.

  - **Adaptive K tuning** — auto-adjust speculation depth
    based on observed accept-rate. v1 ships with operator-
    configured K. Phase 3.x.11.y.x.

  - **Constant-time speculation for Tier C** — speculation's
    per-iteration accepted_count adds a NEW timing surface
    (threat-model addendum §3.5) that Phase 3.x.10.y M1/M2
    decorators don't cover. Tier C remains structurally
    denied for sharded decode (Phase 3.x.11.q deferred);
    Tier-C-compatible speculation is Phase 3.x.11.q.y
    bundle.

  - **Authenticated rollback envelope** — current rollback
    is best-effort + unauthenticated. Phase 3.x.11.y.x if
    operator telemetry shows DoS exploitation.

  - **Multi-draft consensus** — propose K candidates per
    draft × multiple drafts → consensus. Research direction;
    Phase 3.x.11.y' (apostrophe).

  - **Cross-request draft caching** — share draft KV across
    same-prefix requests. Phase 3.x.11.y''.

  - **Cache swap-out / paging** — Phase 3.x.11.z.

  - **Mid-stream re-routing** — Phase 3.x.12.

**Critical security carry-overs.**

  - Phase 3.x.11 Task 5's response-signing-payload commitment
    pattern (next_token_id + is_terminal) extends to
    verified_token_ids + accepted_count via the same
    omit-when-default canonical encoding mechanism.
    Speculative responses retain the "downstream relays can't
    swap output without invalidating signature" invariant.

  - Phase 3.x.10.y's Tier C structural deny carries forward
    unchanged; speculation runs entirely on Tier A/B paths.
    The deny gate fires at ``_dispatch_sharded`` BEFORE any
    VERIFY decoding, so the new accept-rate timing surface
    is structurally never exposed on Tier C.

**Test coverage at the tag.**

  - 17 ``ShardedAutoregressiveRunner`` VERIFY unit tests in
    ``tests/unit/test_sharded_runner.py::TestVerifyNonTail`` +
    ``TestVerifyTail``
    (constructor / non-tail dispatch / tail dispatch with
    full-accept / zero-accept / partial-accept / EOS-mid-
    emitted / max_tokens-cap-mid-round / missing
    proposed_token_ids / K+1 length mismatch / bool token
    rejection / no-EOS-when-eos_token_id-None / temperature
    override propagation).

  - 17 executor speculation unit tests in
    ``tests/unit/test_chain_rpc_client_speculative.py::TestSpeculativeConstructionValidation`` + ``TestSpeculationLoop``
    (draft requires sharded / draft missing methods / K bounds /
    rollback callable / temperature > 0 rejected / full-accept
    K+1 emit / zero-accept correction / partial-accept /
    EOS-terminate / max_tokens-cap-truncate / rollback only on
    partial / 2-stage proposed-token threading / cancellation
    eviction / greedy-equivalence / chain-terminal-overrides).

  - 10 ``RollbackCacheRequest`` server-handler tests in
    ``tests/unit/test_rollback_cache.py::TestRollbackCacheHandler`` + ``TestMultiStageBroadcast``
    (happy path / drop-past-clamps / zero-drop idempotent /
    unknown-id / no-manager-rejects / no-runner-rejects /
    no-truncate-method-malformed / request-id propagation /
    no-leak-other-requests / multi-stage broadcast).

  - 32 ``DraftModel`` / ``HFDraftModel`` unit tests in
    ``tests/unit/test_draft_model.py`` (constructor / reset /
    propose / commit / evict / multi-request lifecycle).

  - 26 VERIFY wire-format tests in
    ``tests/unit/test_verify_wire_format.py`` (round-trip /
    omit-when-default byte-equivalence / signature verification /
    co-set invariant / range validation / cap enforcement).

  - 12 ``KVCacheManager.rollback`` tests in
    ``tests/unit/test_kv_cache.py::TestRollback`` (happy path /
    idempotent paths / payload-storage / validation / concurrent
    thread safety).

  - 5 slow real-distilgpt2 E2E tests in
    ``tests/integration/test_phase3_x_11_y_speculative_e2e.py``
    (greedy-equivalence / VERIFY-iteration-in-receipt /
    tokens > iterations amortization / terminal eviction /
    cancellation eviction).

**Auditor reading path.** Start with the bit-identical greedy
E2E test
(``tests/integration/test_phase3_x_11_y_speculative_e2e.py::TestSpeculativeE2EGreedyEquivalence``)
— that's the load-bearing correctness proof for the
speculation. Then read the threat-model addendum §3.5 for the
new accept-rate timing surface, and §3.5's final paragraph for
the rollback-broadcast best-effort scope-honesty point. Then
``prsm/compute/chain_rpc/client.py:_execute_chain_streaming_sharded_speculative``
for the executor's speculation loop with the greedy-only gate
at entry. Then
``prsm/compute/inference/sharded_runner.py:_sample_tail_verify``
for the tail's accepted_count computation — the standard
speculative-decoding longest-prefix-match algorithm with
``handle.tokens_generated += len(emitted)`` (only emitted
counts against max_tokens). Then
``prsm/compute/chain_rpc/server.py:_handle_rollback_cache`` for
the rollback handler — note the
``MissingVerifyCapabilityError → MALFORMED_REQUEST`` mapping
preserves the executor's caller-bug-vs-internal-crash
distinguishability invariant.

---

## 7.12 Sampling-Correct Speculation under Temperature > 0 (Phase 3.x.11.y.x)

Phase 3.x.11.y.x lifts Phase 3.x.11.y's `temperature > 0` raise and ships sampling-correct speculation under positive temperature via the Leviathan-2023 §2.2 rejection-sampling correction. This is the load-bearing closure of Phase 3.x.11.y's most-cited honest-scope deferral; the threat-model addendum gains §3.6 documenting the new wire surfaces and the bit-identical-greedy regression invariant for v1 traffic.

**Tag:** `phase3.x.11.y.x-merge-ready-20260429` (pending Task 9 review).

**Headline guarantees.**

1. **Greedy-equivalence regression preserved.** v1 traffic (request.temperature == 0.0) continues to route through `apply_lm_head_and_sample_batch` and emits K+1 verified entries — bit-identical to Phase 3.x.11.y. All 17 baseline tests in `test_chain_rpc_client_speculative.py::TestSpeculationLoop` (full accept, partial accept, all-reject, max_tokens cap mid-emit, EOS termination, rollback math, cancellation cleanup, greedy-equivalence vs single-host non-speculative) pass unchanged.

2. **Sampling-correctness invariant under temperature > 0.** v2 traffic (T > 0) routes to `apply_lm_head_and_sample_batch_with_rejection`. **Under Option C.1 (q treated as point mass on d_i with mass q(d_i)), the §2.2 marginal-equals-target invariant holds EXACTLY in the degenerate regime (q(d_i) = 1.0 — pure greedy draft) and APPROXIMATELY under stochastic drafts.** The drift between the C.1 marginal and the verifier's true target distribution increases as draft entropy increases. **Proof.** Three layered validations: (a) `test_rejection_sample_speculation.py::test_distribution_convergence_under_many_trials` runs 5000 trials at K=1, q=1.0, target [0.5, 0.3, 0.15, 0.05]; empirical matches target within 0.025 atol. (b) `test_distribution_convergence_at_K4_first_emit` runs 5000 trials at K=4, q=1.0 (multi-position partial-accept paths) with the same target; empirical matches target within 0.025 atol. (c) `test_option_c1_drift_under_stochastic_q_documented` pins the C.1 drift numerically for q=0.6 (5000 trials; empirical matches the analytical-C.1-marginal, NOT target — documents that the drift is real and bounded; pinning catches future helper changes that would break C.1 determinism). The E2E counterpart at `test_phase3_x_11_y_speculative_e2e.py::TestSpeculativeE2EStochastic::test_v2_speculation_first_emit_matches_softmax_marginal` confirms the wire path through 2 stages preserves the marginal at distilgpt2 + T=0.7 + top_k=50 (TV < 0.35 for N=120) — note this E2E samples the FIRST emitted token (PREFILL), validating the executor wires temperature-aware sampling correctly through the chain, not directly the C.1 stochastic-q marginal-correctness claim. **Phase 3.x.11.y.x' (apostrophe) bumps to Option C.3 (full top-M draft distribution wire) if production telemetry shows the C.1 drift matters in real traffic.**

3. **Wire-format extension `proposed_token_probs`.** New `RunLayerSliceRequest.proposed_token_probs: Optional[Tuple[float, ...]]` field. Co-set with `proposed_token_ids` (both set together; both validated against `MAX_VERIFY_BATCH_TOKENS - 1` cap; each prob in [0, 1]). Signing payload extended to commit probs (mirrors Phase 3.x.11 Task 5 + Phase 3.x.11.y critical-fix pattern: prevents man-in-the-middle prob substitution from forging the executor's intent without invalidating signatures). Omit-when-None canonical encoding preserves byte-equivalence with v1 dispatches.

4. **Tail-shape narrowing.** v2 tail responses ship `accepted_count + 1` verified entries (NOT K+1). The runner-side validator at `_sample_tail_verify_stochastic` enforces `accepted_count in [0, K]` and `len(verified) == accepted_count + 1`; the executor-side validator at `_run_chain_iteration_sharded_verify` enforces the same; v1-shaped responses under v2 dispatch surface MALFORMED_RESPONSE (catches a stale-tail backwards-compat hole — `test_v2_tail_v1_shape_at_temp_gt_zero_raises`).

5. **Critical correctness fix to rollback math.** v1 used `cached_extra = len(verified) - len(emitted)` to derive the number of stale K+1-cached positions to roll back. That happened to be right in v1 because `len(verified) == K+1`, but under-counts in v2 partial-accept (where `len(verified) == ac+1`). New math: `cached_extra = (k_round + 1) - len(emitted)`. Stages cache K+1 positions per VERIFY forward regardless of routing mode, so this is correct for both. `test_v2_partial_accept_emits_accepted_plus_one` pins the v2 case explicitly (rollback drops 1, not 0).

6. **Adaptive K state machine.** Per-request rolling-window of last 4 rounds' `(accepted_count, K)` pairs. Once full, every round recomputes smoothed accept-rate `Σ ac / Σ K` and adjusts K for the NEXT round: halve below 25%, double above 75%, hold in [25%, 75%]. Floor 1, ceiling `MAX_VERIFY_BATCH_TOKENS - 1`. Initial K is the constructor's `speculation_depth`. Documented as content-correlated cross-round surface in §3.6.3 of the threat-model addendum (operators wanting flat-K can configure speculation_depth and lose the 10-30% perf win — honest-scope trade).

7. **Server-side backwards-compat.** `LayerStageServer._dispatch_sharded` builds the `run_layer_slice_unary` kwargs dict, conditionally adds `proposed_token_probs=` ONLY when the wire field is set. Pre-3.x.11.y.x runners (no `proposed_token_probs` in their signature) keep working unchanged on v1 traffic; v2 traffic against a stale runner triggers TypeError ("unexpected keyword argument") which the server catches and maps to MALFORMED_REQUEST with a clear "upgrade or set temperature=0.0" message. **No silent fallback** — the executor learns the tail can't honor v2 stochastic dispatch and the operator can fix the deploy.

8. **Executor-side capability check.** When `request.temperature > 0`, the executor's `_execute_chain_streaming_sharded_speculative` validates the draft model exposes `propose_with_probs`. If absent, raises `ChainExecutionError(code=PROMPT_ENCODE_ERROR)` at dispatch entry with a clear "propose_with_probs required" message — no silent T=0 fallback. v1 (greedy) traffic stays on the existing `propose` path even when the draft has `propose_with_probs` (preserves bit-identical regression for T=0).

**Trust seams (auditor focus).**

- **Probability flow on the wire.** `proposed_token_probs` adds K floats per VERIFY round. The signing payload commits these so the executor's intent is non-forgeable; the chain stages forward them faithfully into the runner. No HMAC/signature on the floats themselves beyond the request-level signing payload — adequate for integrity (the request payload is signed) but does increase the wire surface for content-correlation analysis. **Auditor: confirm `RunLayerSliceRequest.signing_payload` covers `proposed_token_probs` bytes; check `tests/unit/test_verify_wire_format.py::TestProposedTokenProbs` for the round-trip / encoding / signing pin.**

- **Server stale-runner backwards-compat.** The TypeError → MALFORMED_REQUEST mapping is the load-bearing seam preserving the v1↔v2 routing contract. **Auditor: confirm `prsm/compute/chain_rpc/server.py:_dispatch_sharded` only catches the specific TypeError signature ("unexpected keyword argument 'proposed_token_probs'" or "unexpected keyword argument") — broader catches would mask runner bugs as MALFORMED_REQUEST. Read `tests/unit/test_layer_stage_server_verify_v2.py` for the 4 routing tests covering: v1 omit, v2 forward, v2-into-stale-runner MALFORMED, v1-into-stale-runner unaffected.**

- **Rejection-sampling helper purity.** `prsm/compute/inference/sharded_runner.py:rejection_sample_speculation` is pure NumPy with caller-injected RNG (no global state, no torch dependency). The model's `apply_lm_head_and_sample_batch_with_rejection` builds the K+1 target distributions and delegates accept/reject/correction to the helper — keeps the load-bearing distribution-correctness math at one inspection point. **Auditor: confirm `tests/unit/test_rejection_sample_speculation.py::test_distribution_convergence_under_many_trials` runs 5000 trials and asserts atol=0.025 against a known target — the load-bearing Leviathan-2023 invariant.**

- **Executor adaptive K boundary.** The rolling-window state lives in `_execute_chain_streaming_sharded_speculative`'s local frame (no per-request persistence beyond the active stream); on cancellation / terminal / exception, the state is dropped naturally. K is bound to `[1, MAX_VERIFY_BATCH_TOKENS - 1]` at every adjustment (not just at construction). **Auditor: confirm `tests/unit/test_chain_rpc_client_speculative.py::TestAdaptiveK` 4-test class covers halve / double / hold-in-band / cap-respected paths.**

- **Tail-shape narrowing v1↔v2 split-validator.** Two separate code paths validate the tail's response — `_sample_tail_verify_greedy` enforces K+1, `_sample_tail_verify_stochastic` enforces ac+1 with `ac in [0, K]`. The executor-side `_run_chain_iteration_sharded_verify` mirrors this split: v1 (probs is None) expects K+1, v2 (probs set) expects ac+1. **Auditor: confirm both validators raise MALFORMED_RESPONSE rather than truncating-to-fit; truncation would silently mask runner bugs.**

- **Greedy-equivalence regression coverage.** All Phase 3.x.11.y v1 tests (`TestSpeculationLoop` 17 tests + `TestSpeculativeE2EGreedyEquivalence::test_speculative_output_matches_single_host_greedy`) pass unchanged — the v1 path is genuinely bit-identical to its Phase 3.x.11.y form. **Auditor: this is the single most important regression to confirm on a fresh checkout. Run `pytest tests/unit/test_chain_rpc_client_speculative.py::TestSpeculationLoop` against the tag and confirm all 17 green.**

**Round-1 review remediations (pre-tag).** TBD on Task 9 review.

**Honest scope.**

- **`proposed_token_probs` content-correlated wire surface.** Each VERIFY round ships K floats — the draft's per-token confidence at d_1..d_K. Threat-model addendum §3.6.2 documents this as a NEW (vs Phase 3.x.11.y) content-correlated wire surface. Tier C structural deny carries forward; Tier A/B operators accept the leak. Constant-time / encrypted variant deferred to Phase 3.x.11.q.y.

- **Adaptive K cross-round content correlation.** K-value depends on previous-rounds' accept-rates. Threat-model addendum §3.6.3 documents the operator-configurable flat-K opt-out. Phase 3.x.11.q.y bundles this with constant-time speculation.

- **Accept-rate channel narrows but doesn't disappear.** Under v2 stochastic, `accepted_count` becomes a noisy random variable with E[accept] = Leviathan-2023 expected-acceptance formula. Operators choosing T > 0 are accepting MORE noise on the accept-rate channel for the same wire-level cost (vs v1's deterministic correlation). §3.6.1 documents the trade.

- **Authenticated rollback envelope still deferred.** Phase 3.x.11.y's best-effort RollbackCacheRequest (no signature, no executor-side ack-failure remediation) carries forward unchanged. Phase 3.x.11.y.x' (note the prime) is the placeholder if telemetry shows the DoS is exploited.

- **`request` parameter only used for sampling params.** The runner's v2 `_sample_tail_verify_stochastic` reads `temperature` / `top_k` / `top_p` from the `request` argument; same plumbing as the existing `_sample_tail_verify_greedy`. No new request-payload semantics.

- **proposed_token_probs not authenticated end-to-end (round-1 review M2 carry-forward).** Phase 3.x.11.y.x extends `RunLayerSliceRequest.signing_payload` to commit `proposed_token_probs` bytes — so the executor's intent is committed via the request-level signing payload. However, a relay between executor and tail can substitute proposed_token_probs in transit before the request is parsed (the same exposure already exists for `proposed_token_ids` from Phase 3.x.11.y; not net-new in this slice). The `HandoffToken` covers `(request_id, settler_node_id, chain_stage_index, chain_total_stages, deadline_unix)` only — not the full request bytes. **Honest scope:** end-to-end request authentication via a `request_signature` field on `RunLayerSliceRequest` is deferred. Until then, threat-model addendum §3.6.2 documents `proposed_token_probs` as a content-correlated wire surface (Tier C structural deny carries forward; Tier A/B operators accept the leak under the same trust model that Phase 3.x.11.y already documented).

- **Adaptive K is v2-only (round-1 review L3 remediated pre-tag).** The executor's adaptive-K state machine fires only when `use_stochastic` (T > 0). v1 (greedy, T=0) traffic preserves Phase 3.x.11.y's flat-K behavior bit-identically. This is intentional: changing v1 K-value across rounds would break the bit-identical regression claim AND introduce a v1 cross-round content-correlation surface that didn't exist before. Auditor-relevant: the v1 SpeculationLoop tests pass without any adaptive K activity.

**Auditor reading path.** Start at this section. Then read the threat-model addendum's §3.6 for the new wire-surface deltas. Then `prsm/compute/inference/sharded_runner.py:rejection_sample_speculation` (the pure-NumPy helper — the load-bearing math sits here, ~80 lines). Then `prsm/compute/chain_rpc/client.py:_execute_chain_streaming_sharded_speculative` for the executor's v1↔v2 routing + adaptive K state machine. Then `prsm/compute/chain_rpc/server.py:_dispatch_sharded` for the server's stale-runner backwards-compat (TypeError → MALFORMED_REQUEST). Then `tests/unit/test_rejection_sample_speculation.py::test_distribution_convergence_under_many_trials` for the empirical 5000-trial proof of the §2.2 invariant.

---

## 7.13 Tier C Constant-Time Sharded Decode (Phase 3.x.11.q)

Phase 3.x.11.q lifts the named "Tier C structurally denied at the sharded dispatch boundary" deferral that has been carrying across Phase 3.x.11 + 3.x.11.x + 3.x.11.y + 3.x.11.y.x. Tier C content can now flow through sharded autoregressive decode at Tier A/B perf characteristics, with the per-token timing surface masked at the chain-executor boundary via one of two operator-wired decorators (mirrors the §7.8 single-host pattern).

**Tag:** `phase3.x.11.q-merge-ready-20260XXX` (pending Task 7 review).

**Headline guarantees.**

1. **`BatchedTrailingShardedExecutor` (M2).** Drains the inner executor's full stream, then emits ONE `StreamToken` (joined text) followed by ONE `ChainExecutionResult`. From a wire observer on the executor → caller path, exactly two events appear regardless of how many tokens the inner chain produced or at what per-token cadence. Verified at the chain-level (unit test) AND end-to-end against real distilgpt2 with `max_tokens=4` (E2E test produces exactly 1 StreamToken). Empty inner stream emits nothing; result-only inner passes ChainExecutionResult through unchanged.

2. **`FixedRateShardedExecutor` (M1).** Each `StreamToken` from the inner executor is held until at least `cadence_seconds` have elapsed since the previous yield. The chain runs at native speed; the decorator's `yield` is what gates emission. Inter-StreamToken intervals on the executor → caller wire are clamped to ≥ cadence regardless of per-token chain compute variance. ChainExecutionResult forwards immediately (terminal isn't part of the per-token timing surface). Clock + sleep are injectable at construction for test determinism. Verified end-to-end at 50ms cadence with real distilgpt2 (every inter-frame interval ≥ cadence - 5ms tolerance).

3. **`make_tier_c_sharded_executor(inner, *, mode, cadence_seconds=None)` factory.** Mode-string selection (`'m2'` / `'m1'`) mirrors Phase 3.x.10.y's pattern. Cadence required for m1, rejected for m2 (M2 has no cadence concept — the misconfig surfaces early). Unknown mode raises ValueError with allowed values.

4. **`ParallaxScheduledExecutor` routing-layer integration.** New constructor kwarg `tier_c_chain_executor: Optional[Any]`; `execute_streaming` branches on `request.content_tier`:
   - Tier A/B → default `chain_executor` (unchanged from sprint-4)
   - Tier C + decorator wired → decorator
   - Tier C + decorator unwired → structured `InferenceResult.failure(...)` with operator-fixable message naming `make_tier_c_sharded_executor`. **No silent fallback to the leaky path** — operators learn the deploy needs the decorator.

5. **Construction-time defense.** `tier_c_chain_executor` without `execute_chain_streaming` is rejected at `__init__` with a clear message naming the factory. Catches operator misconfig at server-start time, not first-Tier-C-request time.

6. **Per-stage `_dispatch_sharded` TIER_GATE deny stays in place** as defense-in-depth. A misconfigured executor that tries to send Tier C dispatches directly to stages still gets rejected at the stage. The chain-level decorator is the trust-policy boundary, not the per-stage server.

7. **Speculation under Tier C remains denied.** Phase 3.x.11.y.x's three new content-correlated wire surfaces (accept-rate, `proposed_token_probs`, adaptive K) are NOT covered by chain-level decorators. Phase 3.x.11.q.y is the bundled placeholder that composes the 3.x.11.q decorators with encrypted/padded probs + masked accept-rate + flat-K mode.

**Trust seams (auditor focus).**

- **Decorator implements `execute_chain_streaming` only.** No synchronous `execute_chain` surface. Tier C non-streaming continues to be denied at the per-stage `_dispatch_sharded` TIER_GATE — the decorator only opens the streaming path. **Auditor: confirm `tests/unit/test_tier_c_sharded_executors.py::TestBatchedTrailingShardedExecutor::test_construction_rejects_non_streaming_inner` and the M1 equivalent — defense-in-depth on the decorator boundary itself.**

- **Routing-layer no-silent-fallback invariant.** When Tier C is requested but no decorator is wired, the failure path is structured (`InferenceResult.failure` with operator-fixable message). The default `chain_executor` is NOT invoked. **Auditor: confirm `tests/unit/test_parallax_executor.py::TestTierCRoutingIntegration::test_tier_c_without_decorator_surfaces_failure` asserts both the failure presence AND that `primary.streaming_calls == []` (default executor not touched).**

- **Per-stage wire still leaks** (load-bearing scope-honesty point). The chain-executor decorator wraps the executor → caller path; the executor → per-stage path continues to dispatch at chain native rate. A network observer on a single stage's transport learns the raw per-token cadence. **This is intentional honest scope** in v1 — Phase 3.x.11.q.x is the bundled placeholder for per-stage cadence wrapping. Documented in threat-model addendum §3.7 + audit-prep §7.13. **Auditor: confirm operators understand they need to compose with per-stage Tier C wrappers (Phase 3.x.10.y pattern) for full-network masking.**

- **Cadence calibration is an operator responsibility.** M1's cadence MUST be ≥ chain native per-token rate; otherwise the decorator yields immediately every tick and provides no masking. Recommended starting point: 2× measured native rate. **Auditor: there's no in-code enforcement that cadence > native — operators set this via config + measure native rate themselves. Phase 3.x.11.q.z (adaptive cadence) deferred.**

- **Total stream duration leaks total token count under M1; total response size leaks total joined-text length under M2.** Cadence × frame count = duration (M1); single trailing frame leaks cumulative byte count (M2). Operators mitigate by capping `max_tokens` per Tier C request; the M2 leak ceiling is text length × bytes/token + the response size cap (operator-configurable padding deferred to Phase 3.x.11.q.x).

- **Routing-layer invariant: only ContentTier.C triggers the decorator.** Unit-test invariants confirm Tier A and Tier B requests do NOT touch the decorator and continue using the default chain_executor. **Auditor: this is the regression boundary — Tier A/B traffic is bit-identical to sprint-4 behavior. Confirm `test_tier_a_uses_default_executor` + `test_tier_b_uses_default_executor` are present.**

**Honest scope (carries forward to follow-up phases).**

- **Per-stage wire still leaks** → Phase 3.x.11.q.x (per-stage cadence wrapper for full-network masking).
- **Speculation under Tier C still denied** → Phase 3.x.11.q.y (bundles 3.x.11.q decorators with encrypted/padded `proposed_token_probs` + masked accept-rate + flat-K mode).
- **Total response size leaks under M2** → Phase 3.x.11.q.x (operator-configurable padding to fixed max-length).
- **Adaptive cadence based on observed network load** → Phase 3.x.11.q.z.
- **Mid-stream cadence change** not supported.
- **Cadence calibration** is operator responsibility (no in-code measurement).

**Round-1 review remediations (pre-tag).** TBD on Task 7 review.

**Auditor reading path.** Start at this section. Then read threat-model addendum §3.7 for the chain-level vs per-stage scope-honesty point. Then `prsm/compute/chain_rpc/tier_c_sharded_executors.py` (~250 lines, two decorators — the load-bearing implementations sit here). Then `prsm/compute/chain_rpc/factories.py:make_tier_c_sharded_executor` for the mode-string factory. Then `prsm/compute/inference/parallax_executor.py:execute_streaming` for the Tier C routing branch (search `Phase 3.x.11.q` to find the routing block). Then `tests/unit/test_tier_c_sharded_executors.py` for the 25 unit tests covering both decorators + factory. Then `tests/unit/test_parallax_executor.py::TestTierCRoutingIntegration` for the 6 routing tests. Then `tests/integration/test_phase3_x_11_sharded_e2e.py::TestTierCShardedDecoratorsE2E` for the 3 real-distilgpt2 E2E tests proving the timing-mask invariants end-to-end.

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
- **0.7 (2026-04-30)** — added §7.9 "Sharded Autoregressive Decode" covering Phase 3.x.11. 9 headline guarantees including wire-format extension (decode_mode + next_token_id + is_terminal with omit-when-default byte-equivalence), KVCacheManager (LRU+TTL+thread-safe lifecycle), ShardedAutoregressiveRunner (non-tail + tail variants), executor per-token chain loop with eviction broadcast on every exit path, EvictCacheRequest wire envelope + handler, server-side sharded dispatch with Tier C structural deny, and bit-identical real-distilgpt2 E2E correctness proof. Critical security fix: ``RunLayerSliceResponse.signing_payload`` extended to commit ``next_token_id`` + ``is_terminal`` (without it, downstream relays could swap sampled tokens without invalidating signatures). Threat-model addendum at ``docs/2026-04-30-phase3.x.11-threat-model-addendum.md`` characterizes the new per-token wire timing surface, KV-cache state privacy on stages, cross-stage activation handoff magnification (`1 + max_tokens` observations per request), and Tier C structural incompat. R3 threat model cross-references the addendum. Phase 3.x.11 tag: phase3.x.11-merge-ready-20260430 (pending Task 9 review).
- **0.8 (2026-04-30)** — added §7.10 "Pipelining + Per-Token Receipt Attestation" covering Phase 3.x.11.x. 5 headline guarantees: IterationAttestation wire-format extension (under separate `PRSM-MI-ATT-V1:` magic prefix; discriminator at magic-prefix level preserves non-sharded-receipt byte-equivalence with golden-bytes pin); executor per-iteration accumulation closing the threat-model addendum §3.2 "no per-iteration cryptographic commitment" gap; server-side `_dispatch_streamed_sharded` lifting the Phase 3.x.11 Task 9 M1 unary-only guard for PREFILL only (INCREMENTAL stays unary-only); executor-side guard mirroring; bit-identical real-distilgpt2 E2E through the chunked path. Pipelining is wire-level only in this slice (chunked PREFILL transport); compute-level pipelining (overlapping consecutive-token decode) requires speculation, deferred to Phase 3.x.11.y. Round-1 LOW-2 remediated pre-tag (defensive `decode_mode == PREFILL` assert at `_dispatch_streamed_sharded` entry, preventing Phase 3.x.11 Task 9 M1-class seam-bugs from a future refactor). Phase 3.x.11.x tag: phase3.x.11.x-merge-ready-20260430.
- **0.9 (2026-04-29)** — added §7.11 "Speculative Decoding" covering Phase 3.x.11.y. 7 headline guarantees: VERIFY wire-format extension (verified_token_ids + accepted_count co-set invariant + RollbackCacheRequest/Response envelopes + signing-payload commitment mirroring Phase 3.x.11 Task 5 critical-fix pattern); KVCacheManager.rollback with caller-injected truncate_fn (payload-opaque); DraftModel Protocol + HFDraftModel reference impl (greedy-only at v1); ShardedAutoregressiveRunner VERIFY support (forward_verify + apply_lm_head_and_sample_batch + truncate_cache); executor speculation loop with greedy-only-at-temperature-0 invariant (PROMPT_ENCODE_ERROR on positive temp; both correctness gate AND threat-model-containment gate); RollbackCacheRequest server handler (MissingVerifyCapabilityError → MALFORMED_REQUEST mapping preserves caller-bug-vs-internal-crash invariant); real-distilgpt2 + HFDraftModel E2E with bit-identical greedy proof. Critical adapter remediation during E2E bring-up: HF GPT2's eager attention defaults to FULL attention across new tokens for K+1 batched cached forward — explicit 4D additive causal mask required. Threat-model addendum §3.5 added covering the new per-iteration accept-rate timing surface (directly observable via accepted_count, indirectly via RollbackCacheRequest.n_positions_to_drop); operator advisory + Tier C structural deny carry-over + greedy-only invariant are the v1 mitigations. Sampling-correct speculation under temperature > 0 deferred to Phase 3.x.11.y.x (Leviathan-2023); constant-time speculation for Tier C deferred to Phase 3.x.11.q.y bundle; authenticated rollback envelope deferred to Phase 3.x.11.y.x if telemetry warrants. Phase 3.x.11.y tag: phase3.x.11.y-merge-ready-20260XXX (pending Task 9 review).

- **1.0 (2026-04-29)** — added §7.12 "Sampling-Correct Speculation under Temperature > 0" covering Phase 3.x.11.y.x. 8 headline guarantees: greedy-equivalence regression preserved (v1 traffic bit-identical to Phase 3.x.11.y); sampling-correctness invariant under T > 0 (Leviathan-2023 §2.2 marginal-output-equals-softmax(logits/T) — proven via 5000-trial unit test + 2-stage E2E with TV<0.35 at distilgpt2+T=0.7+top_k=50); `proposed_token_probs` wire-format extension co-set with proposed_token_ids + signing-payload commitment + omit-when-None canonical encoding; tail-shape narrowing (v2 returns ac+1, NOT K+1; split-validator on both runner and executor sides); critical correctness fix to rollback math (`(k_round + 1) - len(emitted)` instead of `len(verified) - len(emitted)`, which under-counts in v2 partial-accept); adaptive K state machine (rolling 4-round window, halve <25% / double >75%, floor 1 / ceiling MAX_VERIFY_BATCH_TOKENS-1); server-side stale-runner backwards-compat (TypeError → MALFORMED_REQUEST, no silent fallback); executor-side capability check on draft.propose_with_probs. Threat-model addendum §3.6 added covering 3 new content-correlated surfaces: accept-rate channel narrows under stochastic (was deterministic v1, now noisy v2); proposed_token_probs ships K floats per VERIFY round (NEW wire surface, strongest leak); adaptive K cross-round correlation (operator-configurable flat-K opt-out for privacy-vs-perf trade). Phase 3.x.11.q.y bundled placeholder for constant-time speculation (encrypted/padded probs + masked accept-rate). Phase 3.x.11.y.x tag: phase3.x.11.y.x-merge-ready-20260429 (pending Task 9 review).
- **1.1 (2026-04-29)** — added §7.13 "Tier C Constant-Time Sharded Decode" covering Phase 3.x.11.q. 7 headline guarantees: BatchedTrailingShardedExecutor (M2 — single trailing frame), FixedRateShardedExecutor (M1 — cadence-driven yield with injectable clock/sleep), make_tier_c_sharded_executor factory (mode-string selection), ParallaxScheduledExecutor routing-layer integration (Tier A/B unchanged; Tier C with decorator routes correctly; Tier C without decorator surfaces structured failure naming the factory — no silent fallback), construction-time defense (decorator without execute_chain_streaming rejected at __init__), per-stage TIER_GATE deny stays as defense-in-depth, speculation under Tier C remains denied (Phase 3.x.11.q.y bundle). Threat-model addendum §3.7 added (1.3 revision) covering chain-level vs per-stage scope-honesty point — chain-executor decorator masks executor → caller wire only; per-stage cadence wrapping deferred to Phase 3.x.11.q.x. Mitigation table gains 2 new rows. Auditor reading path entries 10 + 11 added. Phase 3.x.11.q tag: phase3.x.11.q-merge-ready-20260XXX (pending Task 7 review).
