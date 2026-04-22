# Phase 7-Storage: Storage Hardening + Content Confidentiality Tiers — Design + TDD Plan

**Document identifier:** `phase7-storage` (referred to as "Master-roadmap Phase 7" in `docs/2026-04-10-audit-gap-roadmap.md` §7)
**Date:** 2026-04-22
**Target execution:** Q3-Q4 2027 (per master roadmap).
**Status:** Combined design + TDD plan drafted ahead of execution. Follows Phase 7 / Phase 8 pattern.
**Depends on:**
- Phase 1.3 mainnet deploy (for on-chain StorageSlashing + KeyDistribution contracts).
- Phase 3 marketplace (provider discovery + price handshake surfaces).
- Phase 7-compute / 7.1 / 7.1x shipped (challenge-pattern precedent — reuses `BatchSettlementRegistry`-style slashing flow, but a separate `StorageSlashing.sol` contract).

---

## ⚠️ Naming disambiguation — essential to read first

There are **three independent "Tier A/B/C" systems** in PRSM. This doc scopes ONE of them and explicitly does NOT re-scope another. Clarification per master roadmap §7:

| System | Status | Doc |
|---|---|---|
| **Compute verification Tier A/B/C** (receipt-only / redundant-execution / stake-slash) | ✅ SHIPPED as Phase 7 + 7.1 + 7.1x | `docs/2026-04-21-phase7-staking-slashing-design.md`, etc. |
| **Content confidentiality Tier A/B/C** (plaintext / encrypted-before-sharding / K-of-N key-shares) | **SCOPED IN THIS DOC (§3.2-§3.3)** | this file |
| **Hardware supply tiers T1-T4** (consumer / prosumer / cloud / meganode) | Operational classification | `PRSM_Vision.md` §6 |

The master-roadmap Phase 7 bundled three workstreams when originally scoped (storage hardening §7.1 + content confidentiality §7.2 + verification tiers §7.3). §7.3 was pulled out and shipped on its own timeline as "Phase 7 / 7.1 / 7.1x" during 2026-04. §7.1 and §7.2 remain — this document scopes both, shipped together because they share cryptographic substrate (Reed-Solomon erasure coding is the common primitive).

When this document says "Tier B" / "Tier C" without qualifier, it means **content confidentiality tier**, not compute verification tier.

---

## 1. Context & Goals

Today PRSM stores content (datasets, model weights, documents) via `prsm/storage/shard_engine.py` with replication but **no erasure coding and no confidentiality** beyond transport-level TLS. Tier A plaintext storage is acceptable for dev/staging workloads; at production scale two specific gaps block the network:

1. **Durability.** Simple replication (`replication_factor=3` in current shard_engine) is vulnerable to correlated failure. Loss of 3 of 3 replicas means permanent data loss. Reed-Solomon erasure coding at `(k=6, n=10)` tolerates up to 4 simultaneous node losses with byte-perfect reconstruction and lower storage overhead (~1.67x vs 3x for replication).

2. **Confidentiality.** No mechanism today prevents a storage provider from reading stored content. Healthcare, legal, financial-services workloads — the addressable market `PRSM_Vision.md` §2 targets and that Prismatica's commissioned-dataset economics depend on — cannot land on PRSM until content can be stored encrypted-at-rest such that the provider cannot decrypt.

Phase 7-Storage ships both together because their cryptographic foundations (keyed encryption, Reed-Solomon erasure, cryptographic challenge-response for proofs) compose better than they would if delivered separately.

### 1.1 Non-goals for Phase 7-Storage

- **Not a new storage substrate.** `prsm/storage/blob_store.py` stays; Phase 7-Storage adds layers atop.
- **Not redistributed content indexing.** Content discovery (Kademlia DHT in `prsm/node/`) unchanged.
- **Not access-control lists.** Confidentiality tiers are all-or-nothing per-content. Fine-grained ACLs deferred.
- **Not homomorphic encryption.** Tier C is shared-secret + threshold, not HE. (HE is R1 research scope.)
- **Not revocation of already-distributed keys.** If a decryption key has been released, it cannot be un-released. Publisher controls pre-release distribution.
- **Not retroactive encryption.** Already-published Tier A content stays Tier A unless republished.

### 1.2 Backwards compatibility

- **Tier A** (existing plaintext storage) remains the default. Storage callers who don't opt into B or C see no change.
- **Existing ShardManifest** format gains optional confidentiality-tier fields; absent fields mean Tier A.
- **StorageChallenge protocol** (from `prsm/node/storage_proofs.py`) is extended with an on-chain-settleable flavor. Off-chain-only challenges continue to work for dev/staging.

---

## 2. Scope

### 2.1 In scope — §7.1 Storage hardening

**Python:**
- `prsm/storage/erasure.py` (new) — Reed-Solomon `(k=6, n=10)` encoder + decoder. Replaces `replication_factor` path for tier B/C content; optional for tier A.
- `prsm/storage/shard_engine.py` extensions — accept `ShardingMode.REPLICATION | ShardingMode.ERASURE`; route accordingly.
- `prsm/storage/proof.py` (extracted from `prsm/node/storage_proofs.py`) — storage-proof challenge/response with on-chain verification path.
- `prsm/storage/heartbeat.py` (new) — periodic liveness reporting from storage providers; failure to heartbeat triggers challenge.

**Solidity:**
- `StorageSlashing.sol` — burn staked collateral on failed storage challenge. Follows Phase 7 `StakeBond` pattern but with storage-specific reasons (e.g., `PROOF_FAILED`, `HEARTBEAT_MISSING`).
- Reuses `StakeBond` for staking; `StorageSlashing` is a new authorized slasher.

### 2.2 In scope — §7.2 Content confidentiality tiers

**Python:**
- `prsm/storage/encryption.py` (new) — AES-256-GCM encrypt-before-shard pipeline.
- `prsm/storage/key_distribution.py` (new) — client-side key split (Shamir `(m=3, n=5)` default for Tier C) + authenticated decryption.
- `prsm/storage/content_tier.py` (new) — tier selection + metadata serialization in `ShardManifest`.

**Solidity:**
- `KeyDistribution.sol` — on-chain key-distribution contract. Publisher deposits encrypted key shares; royalty payment by consumer triggers release of enough shares for reconstruction. Events emit the released shares; consumers pull via events.

**Client-side:**
- Encryption pipeline invoked by `ContentUploader`; decryption pipeline invoked by `ContentProvider` on authorized download.
- Key-management UI (Phase 4 dashboard extension) for publisher to set tier + pricing + revocation policy.

### 2.3 Out of scope (deferred)

- **Tier D information-theoretic confidentiality** — research scope R5, not shippable in this phase.
- **Per-user granular ACLs** — content is tier-B-encrypted once; every paying consumer gets the same key. Audience-specific encryption is a follow-up.
- **Multi-publisher collaborative storage** — one publisher per content object in v1.
- **Content migration tier-A → tier-B** — republishing required; no in-place upgrade.
- **Erasure coding parameter variants** beyond `(6, 10)` — single parameterization at v1; multi-parameter support in a Phase 7-Storage.x.
- **Tier C (K-of-N key shares) UX for consumers** — Tier C is primarily for publisher-side storage assurance (e.g., regulated-archive use cases). Consumer decryption involves coordinating multiple providers' key-share releases; UX complexity deferred.

---

## 3. Protocol

### 3.1 §7.1 Storage hardening — Reed-Solomon + PoR + slashing

**Encoding (publisher-side, one-time at upload):**

```
Content C (arbitrary bytes)
    │
    ▼
Erasure code: RS(k=6, n=10) produces 10 shards
    where any 6 reconstruct C byte-perfect
    │
    ▼
Each shard gets: merkle_root(shard_content)
    │
    ▼
Shards distributed to 10 storage providers
    (chosen via MarketplaceOrchestrator with geographic diversity per SUPPLY-1)
    │
    ▼
ShardManifest (on IPFS / content-index) records:
    - 10 provider IDs
    - 10 shard merkle_roots
    - content_hash (original C)
    - reed_solomon_params (k, n, generator_polynomial_id)
```

**Reconstruction (consumer-side, on download):**

```
Fetch ShardManifest by content_hash
    │
    ▼
Request shards from 10 providers (parallel; 6-of-10 sufficient for reconstruction)
    │
    ▼
First 6 valid shards → Reed-Solomon decode → reconstructed C
    │
    ▼
Verify content_hash(reconstructed C) == ShardManifest.content_hash
```

**Challenge/response (anyone can challenge any provider at any time):**

```
Challenger picks (provider, shard_merkle_root) pair
    │
    ▼
Challenger requests a random chunk_range within the shard
    │
    ▼
Provider returns (chunk, merkle_proof)
    │
    ▼
Challenger verifies merkle_proof against shard_merkle_root
    │
    ▼
If verification fails → submit on-chain to StorageSlashing.sol
If verification passes → no action
```

**Heartbeat (lightweight liveness):**

```
Every 1 hour: provider signs a heartbeat (provider_id, all_hosted_shard_ids, timestamp)
    │
    ▼
Publishes to a Foundation-operated or decentralized aggregator
    │
    ▼
Any heartbeat gap >6 hours for an on-record provider → escalates to challenge
```

### 3.2 §7.2 Tier B content confidentiality (AES-256-GCM + on-chain key release)

**Publisher-side (upload time):**

```
Content C
    │
    ▼
Generate random 256-bit encryption key K
    │
    ▼
AES-256-GCM encrypt: C' = E(K, C)
    │
    ▼
Erasure code C' into 10 shards (§3.1 flow)
    │
    ▼
Distribute shards
    │
    ▼
Publish ShardManifest with tier=TIER_B, content_hash=hash(C')
    │
    ▼
Publisher encrypts K under each authorized consumer's pubkey
    (or deposits K into KeyDistribution.sol with a payment-trigger spec)
```

**Consumer-side (download time, Tier B):**

```
Consumer pays royalty via RoyaltyDistributor (Phase 1.3)
    │
    ▼
RoyaltyDistributor event triggers KeyDistribution.release(content_hash, consumer)
    │
    ▼
KeyDistribution emits Released event with encrypted K (encrypted under consumer pubkey)
    │
    ▼
Consumer fetches shards from 10 providers
    │
    ▼
Reconstructs C' from any 6 shards
    │
    ▼
Decrypts K with private key
    │
    ▼
Decrypts C = AES-256-GCM decrypt(K, C')
    │
    ▼
Verifies C matches published plaintext-content-hash (if publisher registered one)
```

### 3.3 §7.2 Tier C content confidentiality (K-of-N Shamir key split + erasure)

Tier C layers Shamir Secret Sharing on top of Tier B. Rationale: in Tier B, if the `KeyDistribution` contract is compromised (or governance decides to release keys against publisher wishes), ALL keys for ALL Tier B content become exposed. Tier C requires cooperation of M-of-N independent key-share holders; no single point of compromise.

**Publisher-side (Tier C):**

```
Content C → encrypt under K → shard C' via Reed-Solomon (as Tier B)
    │
    ▼
Shamir-split K: K = k_1, k_2, ..., k_5 (m=3, n=5) — any 3 shares reconstruct K
    │
    ▼
Distribute k_1..k_5 to 5 independent key-share holders (who are NOT storage providers for the same content)
    │
    ▼
ShardManifest records: tier=TIER_C, key_share_holder_ids=[h_1..h_5], shamir_params=(3,5)
```

**Consumer-side (Tier C, authorized):**

```
Consumer pays royalty
    │
    ▼
RoyaltyDistributor events trigger each of 3+ key-share holders to release their k_i
    │
    ▼
Consumer collects 3+ k_i values, Shamir-reconstructs K
    │
    ▼
Consumer fetches shards, reconstructs C', decrypts to C
```

**Threshold guarantees:**

- **Data loss resistance:** any 4 of 10 storage providers can fail without data loss (erasure coding).
- **Confidentiality resistance:** any 2 of 5 key-share holders can collude without exposing content. Compromise requires ≥3 of 5.
- **Reconstruction threshold:** consumer needs ≥6 of 10 shards AND ≥3 of 5 key shares.

### 3.4 On-chain contracts summary

- `StorageSlashing.sol` — slashes provider stake on proven failed PoR challenge. Challenger gets 70% bounty (matches Phase 7 compute-slashing); 30% to Foundation.
- `KeyDistribution.sol` — stores encrypted key material; releases on royalty payment.

Both contracts are authorized minor slashers / distributors; Foundation owns both.

---

## 4. Data model

### 4.1 `ShardManifest` extensions

```python
@dataclass(frozen=True)
class ShardManifest:
    # existing Phase 3 fields:
    content_hash: ContentHash
    algorithm: AlgorithmID
    owner_node_id: str
    shards: List[ShardEntry]
    # new Phase 7-Storage fields (all optional for Tier A compat):
    sharding_mode: ShardingMode = ShardingMode.REPLICATION
    erasure_params: Optional[ErasureParams] = None  # (k, n, polynomial)
    confidentiality_tier: ContentTier = ContentTier.A
    encryption_params: Optional[EncryptionParams] = None  # {algorithm, key_id_hex, iv}
    key_distribution: Optional[KeyDistributionInfo] = None  # {contract_addr, key_hash, share_holders}
```

### 4.2 `StorageChallenge` on-chain surface

```solidity
interface IStorageSlashing {
    function submitProofFailure(
        address provider,
        bytes32 shardId,
        bytes32 merkleRoot,
        bytes32 challengeNonce,
        bytes calldata actualResponse,
        bytes calldata expectedProof
    ) external;
    
    function submitHeartbeatMissing(address provider, bytes32 shardId) external;
}
```

### 4.3 `KeyDistribution` contract surface

```solidity
interface IKeyDistribution {
    function depositKey(
        bytes32 contentHash,
        bytes calldata encryptedKey,
        address royaltyDistributor,
        uint256 releaseFeeFtnsWei
    ) external;
    
    function release(bytes32 contentHash, address recipient) external;
    // Requires royaltyDistributor.verifyPayment(recipient, contentHash, releaseFeeFtnsWei)
    
    event KeyReleased(bytes32 indexed contentHash, address indexed recipient, bytes encryptedKey);
}
```

Client-side listens for `KeyReleased` events matching their address.

---

## 5. Integration points

### 5.1 Existing `prsm/storage/shard_engine.py`

Extended with `ShardingMode` enum + erasure-coding branch. Existing `replicate` path retained.

### 5.2 Existing `prsm/node/storage_proofs.py`

Refactored: off-chain challenge/response logic extracted into `prsm/storage/proof.py`; on-chain settlement hook added. Existing off-chain flow retained for dev/staging.

### 5.3 `ContentUploader` / `ContentProvider`

Upload: accepts `tier` + `release_fee_ftns` parameters. Tier A unchanged. Tier B/C trigger encryption pipeline.

Download: detects tier from ShardManifest; Tier B/C triggers key-fetch + decrypt pipeline post-payment.

### 5.4 Phase 1.3 `RoyaltyDistributor`

Phase 7-Storage adds a `verifyPayment` view function that `KeyDistribution.sol` calls before releasing keys. Minimal contract change.

### 5.5 Phase 7 `StakeBond`

Storage providers stake via existing `StakeBond` (no separate stake bond). `StorageSlashing.sol` becomes an authorized slasher. Bond cap applies across compute + storage providers; operators running both services need bond ≥ max(compute_tier_required, storage_tier_required).

### 5.6 Phase 3 `MarketplaceOrchestrator`

Storage-provider selection: tier-required matches, geographic diversity (SUPPLY-1 metrics), and stake-tier verification match existing compute-provider selection logic. Minor extensions to select storage-specific providers and enforce Reed-Solomon distribution constraints (e.g., don't place 10 shards in the same region).

---

## 6. TDD plan

**9 tasks**.

### Task 1: Reed-Solomon erasure-coding library

- `prsm/storage/erasure.py` with `(k=6, n=10)` encode + decode.
- Third-party library (`zfec` or `erasure-codes` crate bindings) for the Reed-Solomon math; no roll-your-own.
- Tests: encode+decode round-trip on known test vectors; reconstruct from any 6-of-10 subset; corruption detection on any single shard; handling of content ≤ shard_size edge case.
- Expected ~20 tests.

### Task 2: ShardEngine erasure path

- Extend `shard_engine.py` with `ShardingMode.ERASURE` branch.
- Update `ShardManifest` with erasure_params.
- Distribution logic: select 10 providers with SUPPLY-1-compatible geographic diversity.
- Tests: upload content via erasure mode; kill 4 of 10 shard providers; verify reconstruction succeeds; verify reconstruction fails at 5 killed.
- Expected ~15 tests.

### Task 3: StorageSlashing.sol + on-chain proof verification

- Contract implementing IStorageSlashing.
- Slasher role on `StakeBond`.
- 70/30 bounty split matching Phase 7 compute-slashing convention.
- Tests: valid proof-failure slashes provider; invalid proof rejected; heartbeat-missing slash after grace period; double-slash prevention; challenger self-slash 100%-to-Foundation (same §3.4 pattern as Phase 7).
- Expected ~15 tests.

### Task 4: Python storage-proof challenge/response flow

- `prsm/storage/proof.py` refactor + on-chain submission path.
- Challenger selects random chunk-range; verifies Merkle proof; escalates to on-chain on failure.
- Heartbeat aggregator (Foundation-operated or decentralized).
- Tests: valid proof accepted; invalid proof triggers slash; heartbeat gap detection; challenge batching; integration with `MarketplaceOrchestrator` provider reputation.
- Expected ~15 tests.

### Task 5: AES-256-GCM encryption pipeline

- `prsm/storage/encryption.py`.
- Key generation + IV management + authenticated decryption.
- Integration with `ContentUploader` (Tier B path).
- Tests: encrypt+decrypt round-trip; IV uniqueness; authentication failure on tampered ciphertext; large-file streaming encryption.
- Expected ~15 tests.

### Task 6: KeyDistribution.sol contract

- On-chain key storage + payment-gated release.
- Integration with Phase 1.3 `RoyaltyDistributor` via `verifyPayment` hook.
- Tests: key deposit + release happy path; release fails without payment; release emits correct events; multiple consumers for same content; key revocation (if ever allowed — see §8.5).
- Expected ~15 tests.

### Task 7: Shamir key-share distribution (Tier C)

- `prsm/storage/key_distribution.py` with Shamir `(m=3, n=5)`.
- Key-share holder selection (distinct from storage providers).
- Multi-holder release coordination on consumer payment.
- Tests: Shamir split+reconstruct; reconstruct fails at m=2; all 5 holders independently release; quorum reconstruction; collusion resistance at m-1 holders.
- Expected ~15 tests.

### Task 8: End-to-end integration tests

- Full Tier A / Tier B / Tier C upload+download scenarios.
- Provider churn during active download.
- Challenge-then-slash scenarios.
- Tests land in `tests/integration/test_phase7_storage_*.py`. Target ≥3 E2E scenarios (one per tier).

### Task 9: Review gate + merge-ready tag + audit

- Independent code review on cumulative diff.
- External audit on `StorageSlashing.sol` + `KeyDistribution.sol` (bundled with Phase 7 compute-slashing audit if feasible; separate otherwise).
- `phase7-storage-merge-ready-YYYYMMDD` tag.
- Audit-prep bundle analogous to Phase 7 audit prep.

---

## 7. Acceptance criterion

Three concrete criteria:

1. **Durability under 40% provider loss.** Upload a 100 MB file via Tier A with erasure `(6, 10)`. Shut down 4 of the 10 storage providers. Verify the file downloads byte-identical from the remaining 6 providers. Reproducible across 100 trials.

2. **Tier B confidentiality with payment-gated release.** Publisher uploads Tier B content with `release_fee_ftns = 1.0`. Unpaid consumer fetches shards but cannot decrypt. Consumer pays 1.0 FTNS via `RoyaltyDistributor`; `KeyDistribution` emits `KeyReleased`; consumer decrypts successfully. Byte-for-byte match to original plaintext.

3. **Tier C key-share threshold enforcement.** Tier C upload with Shamir `(3, 5)`. Attacker compromises 2 of 5 key-share holders; reconstruction fails. Consumer pays, obtains 3 shares, reconstructs successfully.

---

## 8. Open issues

### 8.1 Reed-Solomon parameter selection (k=6, n=10)

Initial parameters match the master roadmap. Alternatives:
- `(k=10, n=16)` — higher overhead but tolerates 6 simultaneous failures.
- `(k=4, n=7)` — lower overhead but tolerates only 3 failures.
- `(k=6, n=10)` — 1.67x overhead, tolerates 4 failures. Reasonable default.

Calibrate after §10 chaos-test results show observed provider-failure rates.

### 8.2 Heartbeat aggregation architecture

Foundation-operated aggregator is simplest but introduces a single point of trust/failure. Decentralized aggregator (e.g., DHT-replicated heartbeat log) is more robust but engineering-heavy. MVP: Foundation-operated with signed heartbeat roots published to a public feed; decentralization is Phase 7-Storage.x.

### 8.3 Key-share holder selection (Tier C)

Holders should be independent from storage providers for the same content (§3.3 collusion resistance). Selection: random from the pool of operators who've opted into key-share-holding stake. Requires a separate stake tier in `StakeBond` or a new key-share-specific stake mechanism. Design decision for Task 7.

### 8.4 Re-encryption for key rotation

If a publisher needs to rotate encryption keys (e.g., suspected key leak), they must re-encrypt and re-shard the content. In-place key rotation without re-sharding is infeasible under current design. Document as a limitation.

### 8.5 Key revocation semantics

If a consumer already received their decryption key via `KeyDistribution.release`, the publisher cannot revoke it retroactively. Publisher can stop releasing NEW keys. Document this as a publisher-controlled pre-release policy, not a post-release revocation.

### 8.6 Storage provider incentives for erasure-coded content

A storage provider hosting 1 of 10 erasure shards is equally valuable as a provider hosting 1 of 3 replicas. Per-shard pricing should reflect this (i.e., erasure shards should price lower per-byte than replica copies because the total storage burden network-wide is lower). Task 2 design review sets initial price ratios.

### 8.7 Tier migration (A → B) requires re-upload

Currently no mechanism exists to upgrade Tier A content to Tier B in place. Publisher must republish. Document as known limitation; Phase 7-Storage.x can add a migration helper if adoption warrants.

### 8.8 Proof-challenge economics

Challengers spend gas; successful challenges earn 70% of slashed stake. If no one challenges, providers face no real enforcement. Foundation may need to operate a "watchdog" that periodically issues challenges funded from Foundation reserve. Budget line in Foundation treasury.

### 8.9 Bonding amount for storage providers

Currently `StakeBond` uses tier thresholds for compute providers. Storage providers likely need different thresholds — smaller providers can host 1-shard commitments with small stake. Design Task 3 for a tier-scaled stake requirement.

---

## 9. Dependencies + risk register

### R1 — Erasure-coding library production stability

`zfec` is well-tested but has had security issues in historical releases. Mitigation: pin a specific version; track advisories; have fallback lib identified.

### R2 — `KeyDistribution.sol` as a censorship vector

If Foundation governance loses autonomy, KeyDistribution could be co-opted to withhold keys. Mitigation: Tier C explicitly mitigates (requires m-of-n independent holders). Document Foundation-centric Tier B as explicitly trust-dependent on Foundation.

### R3 — Provider collusion against erasure-coded content

If one operator quietly controls 5 of 10 "different" providers hosting shards, they have `≥k` shards and can reconstruct content without needing the remaining 5. Mitigation: SUPPLY-1 diversity-aware provider selection; operator-identity binding via stake; challenge economics make sustained spoofing expensive. Confirm at Task 2.

### R4 — Metadata leakage

Even Tier C content has observable metadata: content-hash, shard sizes, access patterns. Correlation-attack resistance is out of scope; document as a known limitation. For strict metadata confidentiality, the publisher layers onion-routing above PRSM.

### R5 — Audit capacity

Phase 7-Storage adds two new contracts + significant crypto surface. Audit should target both contracts + encryption pipeline. Bundled with Phase 7 / 7.1 / 7.1x audit if timing aligns; separate if not.

### R6 — AES-256-GCM IV handling

Improper IV reuse is catastrophic for GCM. Mitigation: IV derivation bound to content_hash + chunk_index; test explicitly for IV uniqueness across chunks + across re-uploads.

### R7 — Shamir implementation choice

Multiple Python libraries exist (`pyshamir`, `secretsharing`, etc.). Audit each for side-channel resistance; pick one with explicit constant-time guarantees on reconstruction.

---

## 10. Estimated scope

- **9 tasks.**
- **Expected LOC:** ~2500 Python + ~500 Solidity.
- **Test footprint target:** +~100 tests unit + 3 E2E scenarios.
- **Calendar duration:** 6-8 weeks engineering + 3-4 weeks audit + remediation. Bundled with Phase 7 audit engagement saves ~2 weeks.
- **Budget:** audit ~$75k-$150k; incremental infrastructure (heartbeat aggregator, challenger watchdog) ~$500-$2000/month.

---

## 11. Relationship to other phases + standards

- **Phase 1.3** — `RoyaltyDistributor` is the payment-trigger for Tier B/C key release. Minimal contract change.
- **Phase 3** — marketplace provider selection extends with storage-specific constraints.
- **Phase 7 compute** — shared `StakeBond`; `StorageSlashing` is a second authorized slasher.
- **Phase 7.1 compute consensus** — CONSENSUS_MISMATCH pattern could inspire a "storage-consensus-mismatch" reason code if multiple challengers disagree on proof validity. Deferred; current StorageSlashing is single-challenger-determinative.
- **PRSM-SUPPLY-1** — shard distribution must respect diversity thresholds; erasure coding amplifies the geographic-diversity requirement (losing one geographic region shouldn't lose k shards).
- **PRSM-CIS-1** — CIS-compliant storage operators could provide stronger confidentiality guarantees on plaintext-content via hardware confidentiality. Tier B over CIS = defense-in-depth.
- **R5 Tier C content hardening (research)** — information-theoretic Tier D sits on top of this phase's Tier C; research-partnership arc.
- **R2 MPC research** — MPC-based storage (instead of encryption-based Tier B) is an alternative design path; R2's research informs whether Phase 7-Storage.x should adopt.

---

## 12. Ratification path

Per PRSM-GOV-1 §9.2:

1. Public comment on this plan (30 days).
2. Revisions.
3. Foundation Board standards vote (simple majority; no supermajority required since this doesn't change any prohibited-amendment item).
4. Phase 7-Storage Task 1 kickoff.

Target ratification: **Q2 2027** (ahead of Q3-Q4 target execution).

---

## 13. Changelog

- **0.1 (2026-04-22):** initial design + TDD plan. Promotes master-roadmap Phase 7 (§7.1 storage hardening + §7.2 content confidentiality tiers) from stub to partner-handoff-ready scoping. Explicitly documents §7.3 (verification tiers) as already-shipped under different numbering (Phase 7 / 7.1 / 7.1x).
