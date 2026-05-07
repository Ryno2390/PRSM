# Team C — Signature/Crypto Self-Audit Findings (2026-05-07, re-run)

**Scope:** Crypto + integrity surfaces of the in-scope contracts at post-fix tip
`2c93c5bb` on `main`.
**Mandate:** PRSM-POL-2 §4.1 — confirm prior L4-self-2026-05-06 findings are
correctly remediated and no new HIGH/CRITICAL surfaces opened by the fixes.

## Summary

**Re-audit verdict: 0 NEW CRITICAL · 0 NEW HIGH · 0 NEW MEDIUM · 0 NEW LOW · 0 NEW INFO.**

All four prior Team C findings (C-01 / MED-5, C-02 / LOW-2, C-03 / INFO-2,
C-04 / INFO-3) have been addressed at the disposition recorded in the
2026-05-06 consolidated:
- **C-01 (MED-5)** — documentation-only fix shipped; NatSpec is accurate and
  complete (see verification below).
- **C-02 (LOW-2)** — domain-prefix still pending; confirmed pending (no
  regression, no escalation).
- **C-03 (INFO-2)** — folded into the MED-5 NatSpec block; verified.
- **C-04 (INFO-3)** — non-degenerate factory default shipped; PoC tests still
  exercise the binding correctly.

The HIGH-1 / HIGH-2 / MED-3 / MED-4 / MED-6 / MED-7 / LOW-3 remediations from
the other teams' tracks all touched code that crypto-side correctness depends
on (commit-time snapshots, slash gas floor, verifier wiring). None of those
fixes regressed any signature, Merkle, Ed25519, or canonical-payload
invariant. The C-INT-01 fix from L2 remains live at every relevant call site.

## Verification of prior-round dispositions

### C-01 (MED-5) — NatSpec accuracy and completeness

**File:** `contracts/contracts/BatchSettlementRegistry.sol:64-100`
**Disposition recorded:** documentation-only (DOCUMENTED 2026-05-07).

The new NatSpec block on `ReceiptLeaf.signingMessageHash` is **accurate**:

1. The claim "INVALID_SIGNATURE challenge proves only that the provider's
   pubkey did NOT sign the 32-byte preimage `signingMessageHash`" is
   **consistent** with the on-chain code path. `_handleInvalidSignature`
   (`:757-782`) computes:

   ```
   bool valid = ISignatureVerifier(verifierAtCommit).verify(
       leaf.signingMessageHash, signature, publicKey
   );
   return !valid;
   ```

   The verifier checks Ed25519(`signature`, `leaf.signingMessageHash`,
   `publicKey`). It says nothing about whether `signingMessageHash` matches
   anything else in the leaf. Challenge-success therefore proves only the
   non-verification of the leaf's own message-hash field — exactly what the
   NatSpec states.

2. The claim that the Python-side canonical formula
   `keccak256("{job_id}||{shard_index}||{output_hash}||{executed_at_unix}")`
   is **convention, not on-chain invariant** is consistent with the actual
   off-chain code. `prsm/compute/shard_receipt.py:26-42`
   (`build_receipt_signing_payload`) and `prsm/settlement/merkle.py:142-148`
   compute the same payload, and `batched_receipt_to_leaf` populates
   `signing_message_hash` from it — but the contract performs no such check.

3. The NatSpec correctly identifies **NO_ESCROW + CONSENSUS_MISMATCH +
   DOUBLE_SPEND** as the receipt-content-forgery primitives. Verified against
   `_handleNoEscrow` (`:791-796`), `_handleConsensusMismatch` (`:869+`), and
   the DOUBLE_SPEND path that consumes the leaf's `_hashLeaf` digest
   (`:707-735`). Forgery of any of (`jobIdHash`, `shardIndex`, `outputHash`,
   `executedAtUnix`) produces a different leaf hash → DOUBLE_SPEND distinguishes
   it from a real prior receipt; CONSENSUS_MISMATCH catches it under k≥2
   redundant dispatch; NO_ESCROW lets the named requester deny authorization
   per-receipt.

4. The NatSpec correctly notes that full on-chain binding would require
   storing the variable-length `job_id` string, and explains why that is
   gas-prohibitive at batch scale.

**Python parity:** confirmed — `merkle.py:142-148` reproduces exactly the same
formula as `shard_receipt.py:39-42` (both `keccak("{job_id}||{shard_index}||
{output_hash}||{executed_at_unix}")` with utf-8 encoding), and
`batched_receipt_to_leaf` writes that hash into the leaf. No drift.

**Verdict: complete and accurate.**

---

### C-02 (LOW-2) — domain-prefix still pending

**File:** `prsm/compute/shard_receipt.py:26-42`
**Disposition recorded:** pending (LOW priority defensive cleanup; coordinated
with C-INT-07 HandoffToken in deferred-bundle PR).

Confirmed: payload still lacks `b"PRSM-Receipt-v1\n"` prefix. No code change
since 2026-05-06. No new Ed25519-keyed PRSM subsystems have been introduced
that would convert this LOW into a higher-severity primitive. **Status:
unchanged, still LOW.**

---

### C-03 (INFO-2) — folded into MED-5 NatSpec

**File:** `BatchSettlementRegistry.sol:97-100` (within the C-01 block).
**Disposition recorded:** DOCUMENTED 2026-05-07.

The MED-5 block explicitly carries the INFO-2 caller note: `bytes32` accepts
any 32-byte value; key-revocation awareness is not enforced here but at the
publisher-key anchor layer. **Verified.**

---

### C-04 (INFO-3) — non-degenerate factory default

**File:** `contracts/test/audit-team-c/C-INT-01-invalid-signature-forgery.test.js:53-59`
**Disposition recorded:** REMEDIATED 2026-05-07.

`makeLeaf` now defaults to:

```js
signingMessageHash: ethers.keccak256(ethers.toUtf8Bytes("default-signing-message")),
```

instead of `ZeroHash`. **Critically, both PoC tests in this file explicitly
override the default** (`signingMessageHash: ethers.hexlify(realMessageBytes)`
on `:155`, `signingMessageHash: ethers.hexlify(claimedMessageBytes)` on
`:218`), so the default change does NOT cause any test that probes the
C-INT-01 binding to silently pass when it should fail. The factory default is
now used only by callers that don't care about the field, which is the
intended hygiene improvement. **Verified.**

## Verification of OTHER teams' fixes — crypto-side regression scan

### INFO-5 (D-06) — `commitBatch` storage-pointer ordering

**File:** `BatchSettlementRegistry.sol:454-502`

Snapshot writes (`challengeWindowSecondsAtCommit`, `totalPausedAtBatchOrigin`,
`lookbackWindowSecondsAtCommit`, `escrowPoolAtCommit`, `stakeBondAtCommit`,
`signatureVerifierAtCommit`, `metadataURI`) all happen BEFORE
`b.status = BatchStatus.PENDING` on `:492`. The only writes AFTER status flip
are `lastPendingBatchExpiry[msg.sender]` (a separate mapping) and the
`BatchCommitted` event emit. **No external call between snapshot writes and
status flip, no observability of a half-initialised PENDING batch.**
Crypto-relevant snapshot integrity (`signatureVerifierAtCommit` —
consumed by `_handleInvalidSignature` via `verifierAtCommit` parameter) is
preserved across the reorder. **Verified.**

### Ed25519Verifier wrapper — `pure` interface (C-INT-02 from L2)

**File:** `contracts/contracts/Ed25519Verifier.sol:40-58`

Confirmed `pure override`. Length checks (`signature.length != 64`,
`publicKey.length != 32`) return false (not revert), matching
`ISignatureVerifier` contract. `bytes32(slice[0:32])` casts on calldata are
length-checked and safe. The `mstore(add(m, 32), messageHash)` correctly
populates the 32-byte buffer for `Ed25519Lib.verify`. **No change since L2;
still clean.**

### `signatureVerifierAtCommit` cross-wire (D-03 snapshot)

**File:** `BatchSettlementRegistry.sol:483, :766-779`

Cross-wire snapshot at commit (`:483`) is consumed in
`_handleInvalidSignature` via the `verifierAtCommit` parameter (`:766-779`).
Owner rotation via `setSignatureVerifier` only affects FUTURE commits.
Combined with MED-7's new `code.length > 0` check on the setter, the
verifier-rotation primitive is now bounded both temporally (per-batch
snapshot) and shape-wise (must be a contract). **No regression.**

### `MIN_SLASH_GAS` floor

**File:** `BatchSettlementRegistry.sol:328` and `:684-685`

`MIN_SLASH_GAS = 150_000`, checked via `gasleft() < MIN_SLASH_GAS` immediately
before the `try IStakeBond(bondAtCommit).slash(...)` call (`:684, :694`). The
NatSpec at `:312-327` correctly explains the rationale (under-budgeted
`eth_estimateGas` could otherwise let the inner slash silently revert OOG
inside the try/catch). Real slash cost remains in the ~80K-200K range so 150K
is conservative-but-not-pathological. **No regression.**

## Findings

### None.

The re-audit found zero new findings of any severity in the crypto/integrity
surface. The 2026-05-06 finding tally for Team C
(C-01 MED · C-02 LOW · C-03 INFO · C-04 INFO) is the controlling ledger;
dispositions recorded in `consolidated.md §6` are accurately reflected in the
post-fix code at commit `2c93c5bb`.

## Notes / Out-of-scope

1. **HandoffToken / PublisherKeyAnchor** — out of scope for this re-run (no
   crypto-relevant code change since L4-self-2026-05-06 in
   `prsm/security/publisher_key_anchor/` or `prsm/network/handoff/`). Status:
   unchanged.

2. **Ed25519Lib RFC 8032 conformance** — port diff still unchanged; deferred
   to L3 specialist audit per `audits/findings/L3-crypto/upstream-port-diff.md`.

3. **Cross-chain replay** — N/A; BSR remains single-chain by design.

4. **`StakeBond.slasher` immutable + `EscrowPool.settlementRegistry`
   immutable** — re-confirmed; closes B-CROSS-1/B-CROSS-3 still.

5. **`ISignatureVerifier.verify` is `pure`** — re-confirmed; closes C-INT-02
   silent-state-reading vector still.

6. **Convergence check** — Teams A/B/D fixes (per-batch snapshots, pause
   accumulator, gas floor, immutable cross-wires) all hold the
   `signatureVerifierAtCommit` discipline correctly. No new way for a
   compromised owner or governance change to swap the verifier mid-flight or
   retroactively flip an already-committed batch's signature-validation
   surface.

— Generated by Team C re-audit, 2026-05-07. Pinned commit `2c93c5bb`.
