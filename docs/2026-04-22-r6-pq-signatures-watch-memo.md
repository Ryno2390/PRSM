# R6 Post-Quantum Signatures: Trigger-Watch Memo

**Document identifier:** R6-WATCH-1
**Version:** 0.1 Draft
**Status:** Watch memo, not research plan. PRSM is not migrating today. This document specifies what we watch, what the migration would look like when triggered, and what low-cost preparations keep the option open.
**Date:** 2026-04-22
**Drafting authority:** PRSM founder
**Promotes:** `docs/2026-04-14-phase4plus-research-track.md` §R6 from stub to watch memo.
**Related documents:**
- `docs/2026-04-14-phase4plus-research-track.md` §R6 — original research stub.
- `prsm/node/identity.py` — Ed25519 keypair generation + signing for node identity.
- `prsm/compute/shard_receipt.py` — Ed25519 signature verification in `ReceiptOnlyVerification`.
- `contracts/contracts/BatchSettlementRegistry.sol` — `ISignatureVerifier` interface + `INVALID_SIGNATURE` challenge path.
- Ethereum L1/L2 roadmap external signals.

---

## 1. Purpose

PRSM's signature surface today is Ed25519 (node identity, shard receipts, `INVALID_SIGNATURE` challenges) plus secp256k1 implicitly through Ethereum transactions. The R6 stub's question is: *when* should this migrate to post-quantum primitives?

**Short answer:** not now, but with post-quantum signature standards finalized in 2024 the watch-state has shifted. This memo records:

1. The exact current signature surface in PRSM.
2. Why the threat model for signatures is weaker than for encryption (no store-now-forge-later).
3. What triggers would move R6 to engineering work.
4. What low-cost preparations we make *now* to keep the migration option open — specifically receipt-format headroom for dual-sign transition.
5. Candidate post-quantum primitives worth tracking.

This is a watch memo. It does not propose engineering work. It does propose one small doc-level change (tracking the already-fired NIST trigger) and one small receipt-format recommendation that protects future optionality.

---

## 2. Current signature surface in PRSM

Three distinct layers use signatures. Each has different migration implications.

### 2.1 Node identity + shard receipts (Ed25519)

- **Where:** `prsm/node/identity.py` (keypair generation + signing) and `prsm/compute/shard_receipt.py` (receipt construction + `ReceiptOnlyVerification`).
- **What it signs:** the canonical receipt payload (job_id, shard_index, output_hash, executed_at_unix) per Phase 2 §Task 2.
- **Signature lifetime:** each receipt's signature must be verifiable during the batch challenge window (default 3 days per `MIN_CHALLENGE_WINDOW_SECONDS`). After challenge window closes, the signature is historical.
- **Migration scope:** receipt format, node-identity format, signing library, verifier library.

### 2.2 On-chain settlement via `INVALID_SIGNATURE` challenges

- **Where:** `contracts/contracts/BatchSettlementRegistry.sol` `_handleInvalidSignature` + `ISignatureVerifier` interface.
- **What it verifies:** the Ed25519 signature on a committed receipt during an `INVALID_SIGNATURE` challenge.
- **Production verifier:** pluggable — Phase 3.1 ships `MockSignatureVerifier` for tests; production substitution of an audited Ed25519 library is a pre-mainnet-deploy step (tracked in Phase 3.1 audit-prep §1.3 and Phase 7.1x audit-prep §8.6).
- **Migration scope:** contract-level verifier swap — pluggable architecture already accommodates a PQ verifier drop-in as long as the interface stays stable.

### 2.3 Ethereum transaction signatures (secp256k1)

- **Where:** every Ethereum tx PRSM contracts receive — `commitBatch`, `challengeReceipt`, `slash`, `claimBounty`, etc. Signed by whatever key the calling wallet controls.
- **Scope:** PRSM does not own this layer. When Ethereum / Base migrate to PQ signatures, PRSM migrates transparently.
- **Migration scope:** zero PRSM-side code. PRSM inherits L2-level PQ migration for free when it happens.

### 2.4 Summary

| Layer | Scheme | PRSM owns migration? | Urgency driver |
|---|---|---|---|
| Node identity | Ed25519 | ✅ | PRSM's own threat model |
| Receipt sig | Ed25519 | ✅ | PRSM's own threat model |
| On-chain verifier | Ed25519 via `ISignatureVerifier` | ✅ (pluggable) | Matches receipt sig layer |
| L1/L2 tx sig | secp256k1 | ❌ | Ethereum roadmap |

§2.1-§2.2 are the actual R6 scope. §2.3 is out of scope.

---

## 3. Threat model

### 3.1 Why signatures are weaker threat than encryption

Classical cryptography lives in two buckets under the quantum threat:

- **Encryption:** harvest-now-decrypt-later. An adversary recording encrypted traffic today can decrypt it whenever a fault-tolerant quantum computer exists. Urgency is RIGHT NOW for long-lived secrets.
- **Signatures:** no harvest-now-forge-later equivalent. An adversary cannot forge a signature on a message they haven't seen the private key for — quantum computers break key recovery (Shor's on elliptic curves), but only once they actually exist. When a signature is used for *authentication at time T*, a future quantum adversary at time T+N cannot retroactively forge a signature the original signer never made.

PRSM's Ed25519 signatures fall in the second bucket. The threat is: *once a fault-tolerant quantum computer exists, a quantum-capable adversary could forge new receipts that claim to be from a PRSM provider they didn't actually bribe or compromise.* That is a threat at time T+N for signatures generated after the attacker-has-quantum date, not for signatures generated before it.

### 3.2 Quantum availability timelines

Consensus expert estimates for cryptographically-relevant fault-tolerant quantum computers (Shor's algorithm at ECDSA-breaking scale):

- **Optimistic:** 2030-2035.
- **Consensus:** 2035-2040.
- **Pessimistic:** 2045+ or never at scale.

PRSM's signatures are deliberately short-lived (3-day challenge window for receipts; node-identity keys rotate with operator churn). This matters because signatures generated *before* attacker-has-quantum are not threatened; the question is when PRSM's *active* signing infrastructure needs migration.

### 3.3 Signature-specific threat chain

A quantum adversary at time T can, assuming they've achieved cryptographically-relevant quantum:

1. **Recover private keys from observed public keys.** Given any Ed25519 public key exposed on-chain or in a committed receipt, recover the corresponding private key.
2. **Forge new receipts under the recovered identity.** Produce receipts that verify under the original identity's public key but weren't signed by the legitimate operator.
3. **Earn FTNS on forged receipts.** Submit forged receipts through the settlement pipeline; if not challenged, collect payment.

Mitigation against a detected quantum attack: rotate all node-identity keys to PQ algorithms, accept no pre-migration signatures after a cut-off date, and publish the cut-off broadly.

This is doable but expensive (every operator must rotate). The watch question: how much lead time do we need to execute this rotation smoothly?

---

## 4. Triggers

The R6 stub named two triggers. One has fired; the other hasn't. This memo also adds two that weren't in the stub.

### 4.1 Trigger A — NIST finalization of post-quantum signature standards ✅ FIRED

- **Stub trigger:** "NIST finalization of post-quantum signature standards."
- **Actual status:** NIST finalized **ML-DSA** (Dilithium, FIPS 204) and **SLH-DSA** (SPHINCS+, FIPS 205) in August 2024. **FN-DSA** (Falcon) standardization is still in progress as of early 2026.
- **Implication:** the first of the stub's two named triggers has already fired. PRSM does not auto-trigger engineering on this alone — the stub required BOTH triggers — but this does elevate the watch-state.

### 4.2 Trigger B — Ethereum L2 migration plans ⏳ WATCHING

- **Stub trigger:** "Ethereum-level migration plans."
- **Current status:** no active EIP for L1 PQ signature migration as of early 2026. Research-level work exists (EF cryptography team tracking ML-DSA integration). Base (our L2) follows L1.
- **What we watch:** (a) any EIP progressing to "Last Call" status on PQ signatures, (b) Base-specific roadmap statements, (c) Vitalik or EF leadership public comments on timeline.
- **Trigger condition:** a published Ethereum roadmap with a PQ-migration target year commits PRSM to align within 6 months of L1 rollout.

### 4.3 Trigger C (new) — Credible quantum-advantage announcement

- **Not in the stub** but important to track separately.
- **Condition:** a credible announcement (not a vendor press release — peer-reviewed or government-validated) that fault-tolerant quantum computers at 1000+ logical qubits exist or are imminent.
- **Implication:** even absent NIST + Ethereum moving, a credible quantum-advantage announcement moves R6 to urgent engineering. PRSM cannot wait for Ethereum's migration schedule if the underlying threat materializes.

### 4.4 Trigger D (new) — Hybrid-sign adoption elsewhere in PRSM's stack

- **Not in the stub** but a natural timing signal.
- **Condition:** any of PRSM's dependencies (cryptography library, hardware vendor, TEE attestation CA) ship dual-sign (classical + PQ) as a standard mode.
- **Implication:** PRSM's cost-to-migrate drops substantially when the ecosystem is already doing hybrid. Moving ahead of the ecosystem is expensive; moving with the ecosystem is routine.

### 4.5 Trigger action mapping

| Trigger | Status | Action if fires |
|---|---|---|
| A NIST standards | ✅ FIRED | Watch-state elevated; no engineering yet |
| B Ethereum roadmap | ⏳ | Scope PRSM migration as a companion to Ethereum's |
| C Quantum announcement | ⏳ | Urgent engineering convocation, bypassing B |
| D Ecosystem hybrid-sign | ⏳ | Opportunistic migration — lowest cost window |

Any two of B/C/D firing, or C alone, promotes R6 from watch to engineering.

---

## 5. Readiness posture (what we do NOW)

No engineering work yet. Three low-cost preparations keep the migration option open:

### 5.1 Receipt-format headroom

PRSM's `ShardExecutionReceipt` today carries an Ed25519 signature as a base64 string. Current signature size: 64 bytes raw, ~88 bytes base64. Candidate PQ signatures are LARGER:

| Scheme | Raw sig size | b64 size |
|---|---|---|
| Ed25519 (current) | 64 bytes | ~88 bytes |
| ML-DSA-44 (PQ, smallest) | ~2,420 bytes | ~3,232 bytes |
| ML-DSA-65 (PQ, balanced) | ~3,293 bytes | ~4,393 bytes |
| SLH-DSA-128s | ~7,856 bytes | ~10,475 bytes |

A 3-4KB signature in every shard receipt is a real bandwidth cost. Receipt format should be ready to carry an optional PQ signature alongside the classical one without requiring a wire-format break.

**Recommendation (deferred, not a today-action):** when Phase 2's `ShardExecutionReceipt` next sees a breaking format change for any reason, add an optional `pq_signature: Optional[str]` field. Defaults to `None` during Ed25519-only mode. Callers verify classical OR PQ depending on which is present.

**Why not add it now:** every field added to the receipt format adds wire bytes to every PRSM inference regardless of use. The field stays zero-cost if we add it during an already-planned format revision; it's a real cost if we add it just-in-case. Triggered by B/C/D above, or by a batched receipt-format revision.

### 5.2 Pluggable signature verifier at the contract layer

Already in place. `contracts/contracts/BatchSettlementRegistry.sol`'s `ISignatureVerifier` interface is pluggable — the production deployment specifies a verifier address, and the verifier can be any contract implementing the interface. Swapping Ed25519 for ML-DSA at the contract layer is an `owner.setSignatureVerifier()` call, not a full contract redeployment.

**Confirming note:** this is already Phase 3.1's design intent. No change needed.

### 5.3 Monitor `cryptography` library PQ support

PRSM's Python stack uses `cryptography` (PyCA) for Ed25519. `cryptography` doesn't currently ship ML-DSA/SLH-DSA as of early 2026. Candidate alternate libraries:

- **liboqs / oqs-python** (Open Quantum Safe project) — shipping ML-DSA / SLH-DSA / Falcon bindings. Under active development.
- **PyCryptodome** — may add PQ primitives ahead of `cryptography`.
- **cloudflare/circl** (Go) — if we ever add Go-side signing, reference PQ impl.

**Watch-state action:** mark a PyCA issue subscription for when ML-DSA lands. Consider oqs-python for prototyping if B/C/D fire before PyCA ships.

---

## 6. Migration playbook (when triggers fire)

If R6 promotes to engineering:

### 6.1 Phase 1: Dual-sign receipts (3-6 months)

- Receipt format adds optional `pq_signature` field.
- Nodes begin signing with BOTH Ed25519 and ML-DSA (or SLH-DSA depending on selection).
- Verifier library accepts either signature as valid during this phase.
- No operator is required to migrate yet; hybrid is opt-in.

### 6.2 Phase 2: PQ-required for new receipts (6-12 months)

- Protocol version bump announces a cut-off date.
- After cut-off, receipts without PQ signatures are rejected at commit time.
- Existing committed batches from before cut-off still settle with Ed25519 (grandfather period).

### 6.3 Phase 3: Ed25519 deprecation (12-18 months)

- `ISignatureVerifier` implementation swapped (via `setSignatureVerifier()`) to PQ-only.
- Any straggler Ed25519 receipts in the accumulation window fail to commit.
- PRSM is fully PQ for signatures.

Total: 12-18 months end-to-end. This is the lead time we need; §4's triggers set when we start the clock.

### 6.4 Non-migration paths

Ed25519 is not abandoned outside the receipt path. Classical signatures remain valid for:

- Historical audit of pre-migration batches (no threat retroactively).
- Non-settlement-critical signing (e.g., log attestations, ephemeral comms).
- Systems where the quantum threat is not in-scope (internal tooling).

Only the settlement-critical signatures migrate.

---

## 7. Candidate primitive selection

If R6 engineering triggers, primitive selection becomes a decision. Pre-decision bias:

- **ML-DSA-65** (Dilithium balanced) is the default choice. Smallest PQ sig among standardized schemes (~3.3KB), fast enough for per-receipt signing, strong security margin.
- **SLH-DSA** (SPHINCS+) is a fallback for conservative deployments — hash-based (no lattice assumptions), but 2-3× larger signatures. Reserve for environments where lattice-cryptanalysis breakthroughs are the primary risk.
- **FN-DSA** (Falcon) is tempting (small signatures ~1KB) but has sidechannel-risk in its floating-point sampling. Not recommended until implementations mature.

**Final selection should happen at trigger time, not now.** The field is moving fast enough that a 2026 recommendation may be superseded by 2028 analysis.

---

## 8. Budget implications

No current spend. If R6 promotes to engineering:

- **Phase 1-3 migration:** estimated 3-4 engineer-months (library integration, receipt format, verifier swap, operator comms, grandfather-period handling).
- **Testing:** a test-migration on Sepolia preceding mainnet.
- **Operator transition support:** documentation + migration-assistance during the hybrid window.
- **Audit:** PQ verifier contract + receipt-format change is a breaking change requiring a dedicated audit engagement (~$50k-$100k depending on scope).

Total: roughly one-quarter of an engineer + $50k-$100k audit. Small compared to the underlying protocol-audit costs. Justified on-trigger.

---

## 9. Open issues

### 9.1 Cross-trigger correlation

Triggers B and D are likely correlated — an Ethereum-level migration plan will probably reference specific libraries (D). When B fires, D likely fires with or shortly after it. This means the §4.5 "any two of B/C/D" condition may simplify to "B OR C."

### 9.2 Operator-key rotation cost

Every existing PRSM operator has an Ed25519 identity today. A migration requires operators to generate a new PQ identity and re-register. Rotation mechanics not specified — do we keep the Ed25519 identity alive with a PQ "co-signer" permission? Do we migrate identity cleanly to a new key? Design decision deferred to trigger time.

### 9.3 Hardware-wallet PQ support

The Foundation multi-sig is secp256k1 today (via Safe). PQ multi-sig on hardware wallets is still immature. When Ethereum/Base migrate, the multi-sig stack needs PQ-capable hardware wallets — a dependency outside PRSM's control.

### 9.4 StakeBond / registry contracts

PQ migration at the contract layer requires either (a) swapping `ISignatureVerifier` (cheap, already supported) or (b) re-deploying contracts with PQ-native verification paths (expensive). Option (a) is the default.

---

## 10. Cross-references

- **R1 (FHE for private inference)** — overlaps in underlying cryptographic infrastructure. If R1 partners with a specialized cryptography group, that group may also be the right partner for R6 execution.
- **R8 (anti-exfiltration)** — the PRSM-CIS-1 silicon standard assumes a specific attestation-signature algorithm. PQ migration in R6 affects attestation-chain roots in R8.
- **PRSM-GOV-1 §9.4 emergency amendments** — a Trigger C (credible quantum announcement) would likely use the emergency amendment path to accelerate migration timeline.
- **Phase 2 Line Item C (TEE attestation)** — TEE vendor attestation signatures are typically separate from PRSM's identity signatures. PQ in one doesn't require PQ in the other, though ecosystem pressure correlates them.

---

## 11. Ratification

This document does not require governance ratification; it is a watch memo. Ratifies implicitly on publication. Next review trigger: any of §4's B/C/D firing, OR 2027-04-22 scheduled review (one year from now), whichever first.

---

## 12. Changelog

- **0.1 (2026-04-22):** initial draft, founder-authored. Promoted from R6 stub. NIST Trigger A documented as already-fired; B/C/D added as new triggers with status.
