# Team C — Signature/Crypto Self-Audit Findings (2026-05-06)

## Summary

Reviewed C-INT-01 remediation (`signingMessageHash` binding in `ReceiptLeaf`) plus full crypto surface of the four in-scope contracts. The fix itself is **correctly implemented** for the narrow vulnerability it addresses: challenger can no longer choose the verification message, so the original "any holder of (pubkey, signature) bytes can slash any provider" attack is fully closed. End-to-end Python↔Solidity parity is consistent.

| ID | Severity | Title |
|----|----------|-------|
| C-01 | Medium | `signing_message_hash` is provider-chosen — no on-chain consistency check vs. other leaf fields |
| C-02 | Low | Receipt signing payload lacks domain-prefix string (cross-protocol replay primitive) |
| C-03 | Informational | `signing_message_hash` field is unconstrained 32 bytes |
| C-04 | Informational | Stale comment in PoC test (`signingMessageHash: ZeroHash` default in factory) |

CRIT-1 fix verification: **complete**. All challenge code paths consume `leaf.signingMessageHash` directly; auxData layout no longer carries challenger-supplied message; regression test asserts the original attack reverts. No code path constructs receipts without the bound message hash.

## Findings

### C-01 — `signing_message_hash` is provider-chosen with no on-chain consistency to other leaf fields (Severity: MEDIUM)

**Location:**
- `BatchSettlementRegistry.sol:55-70` (`ReceiptLeaf` struct)
- `BatchSettlementRegistry.sol:636-661` (`_handleInvalidSignature`)
- `prsm/settlement/merkle.py:144-147` (off-chain canonical computation)

**Description:**
The C-INT-01 fix binds `signingMessageHash` into the leaf, closing the challenger-controlled-message attack. But the contract performs no check that `leaf.signingMessageHash == keccak256(canonical(leaf.jobIdHash, leaf.shardIndex, leaf.outputHash, leaf.executedAtUnix))`. It cannot — the canonical signing payload uses the original `job_id` STRING, not its keccak256 hash. So the on-chain layer trusts the provider's claim that `signingMessageHash` corresponds to the rest of the leaf.

The leaf docstring at line 64 implies binding-to-the-leaf-content. The implementation only binds the signature to the message hash, not the message hash to the leaf content.

**Attack scenario:**
A malicious provider Pn with valid Ed25519 keypair generates one signature `σ = Sign(kn, m)` over an arbitrary 32-byte digest `m`. They commit N batches, each with one leaf:
- `providerPubkeyHash = keccak256(Kn)` (real)
- `signatureHash = keccak256(σ)` (real)
- `signingMessageHash = m` (real)
- `jobIdHash`, `shardIndex`, `outputHash`, `executedAtUnix`, `valueFtns` = arbitrary fabricated values

INVALID_SIGNATURE challenges fail (signature is valid over m). DOUBLE_SPEND challenges fail (each leaf's encoding is distinct). Only on-chain defense remaining is requester's NO_ESCROW veto — requires monitoring every batch they're named in.

**Impact:**
INVALID_SIGNATURE challenge surface is silent on receipt-content forgery. Requester is on the hook for paying fabricated value at finalize unless they NO_ESCROW within window.

**Recommended fix:**
**Option A (recommended for this pass):** Update leaf docstring to clarify `signingMessageHash` is provider-supplied and only proves "the (pubkey, signature) form a valid Ed25519 signature over SOME 32-byte preimage", not "over the canonical receipt content." Document NO_ESCROW + CONSENSUS_MISMATCH as the actual primitives that catch receipt-content forgery.

**Option B:** Add Python parity test that asserts every committed leaf has `signing_message_hash == build_receipt_signing_payload(...)`. Off-chain dispatcher / challenger watchdog refuses to honor leaves that violate.

**Option C:** Extend leaf encoding to store `jobId` string. ~3-5 days + parity rebuild. Not recommended.

---

### C-02 — Receipt signing payload has no domain-prefix string (Severity: LOW)

**Location:** `prsm/compute/shard_receipt.py:26-42`

**Description:** Payload is `keccak256("{job_id}||{shard_index}||{output_hash}||{executed_at_unix}")` with no PRSM-domain prefix. Cross-protocol replay primitive if the same Ed25519 provider key is reused across PRSM subsystems. Identical pattern to C-INT-07 HandoffToken finding.

**Recommended fix:** Prefix with `b"PRSM-Receipt-v1\n"` before keccak256. Apply same prefix at leaf-encoding side. Coordinate with C-INT-07 HandoffToken fix in deferred-bundle PR.

---

### C-03 — `signing_message_hash` leaf field has no on-chain shape constraint (Severity: INFO)

`signingMessageHash` is `bytes32`; contract accepts any 32-byte value. Future schema changes adding structure aren't enforced at on-chain level. Document in caller-assumptions.md.

---

### C-04 — Stale `signingMessageHash: ZeroHash` default in regression test factory (Severity: INFO)

`makeLeaf` factory defaults `signingMessageHash` to `ZeroHash`. Future test author copying could land on degenerate input. Either change default to a non-trivial hash or remove the default entirely.

## Notes / Out-of-scope

1. **Ed25519Lib RFC 8032 conformance** — port diff unchanged; deferred to L3 specialist audit per `audits/findings/L3-crypto/upstream-port-diff.md`.

2. **Ed25519Verifier wrapper input handling** — `signature.length != 64` and `publicKey.length != 32` checks return false (not revert) per `ISignatureVerifier`. `bytes32(slice[0:32])` casts safe; `mstore(add(m, 32), messageHash)` correctly populates 32-byte buffer.

3. **D-03 cross-wire snapshotting (for Ed25519Verifier)** — `b.signatureVerifierAtCommit` captured at commit time, consulted by `_handleInvalidSignature`. Owner rotation only affects FUTURE commits.

4. **`StakeBond.slasher` immutable + `EscrowPool.settlementRegistry` immutable** — confirmed; closes B-CROSS-1/B-CROSS-3.

5. **Cross-chain replay** — N/A; BSR is single-chain by design.

6. **`ISignatureVerifier.verify` is `pure`** — C-INT-02 fix confirmed live, closes silent-state-reading vector.

7. **Out of scope reminder** — `BridgeSecurity.sol` not in this audit's scope; C-INT-03/04/05/06 against that file not re-audited. C-INT-07 (HandoffToken) — C-02 above is the parallel finding.
