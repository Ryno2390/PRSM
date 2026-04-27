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
