# L3 Pre-Engagement — Caller Assumptions for Ed25519Verifier

**Date:** 2026-05-05
**Purpose:** Document what `BatchSettlementRegistry` (the sole on-chain
caller of Ed25519 verification) assumes about the verifier's behavior. The
specialist auditor should verify each assumption holds.

## Background

The on-chain Ed25519 verifier is invoked from exactly one production callsite:

```solidity
// contracts/contracts/BatchSettlementRegistry.sol:555
bool valid = signatureVerifier.verify(messageHash, signature, publicKey);
```

The verifier is wired through the `ISignatureVerifier` interface:

```solidity
interface ISignatureVerifier {
    function verify(
        bytes32 messageHash,
        bytes calldata signature,
        bytes calldata publicKey
    ) external pure returns (bool);
}
```

`Ed25519Verifier.sol` implements this interface and delegates to
`Ed25519Lib.verify(k, r, s, m)` after splitting the 64-byte signature into
`(r, s)`.

## Stated assumptions

The Registry treats the return value of `signatureVerifier.verify` as
authoritative for the INVALID_SIGNATURE challenge path. The following
assumptions must hold for that authority to be sound. **Each assumption is a
target for the L3 specialist auditor.**

### A1. Pure function

**Assumption:** `verify(messageHash, signature, publicKey)` is a pure
function. Same inputs → same output, regardless of:
- caller (msg.sender)
- block conditions (number, timestamp, hash, basefee)
- contract state
- prior call history

**Why it matters:** If verification depends on context, an attacker can
manipulate the context to alter the verdict (replay attack, deferred-state
attack).

**Verifier confirms via:** declaring `pure` in the function signature.
Solidity enforces this at compile time. Any state-mutating operation would
fail to compile.

**Auditor checks:** No SLOAD, no SSTORE, no CALL, no CREATE, no
block-context opcodes (TIMESTAMP, NUMBER, COINBASE, BLOCKHASH, etc.). Pure
declaration is enforced by the compiler but auditor should confirm at
assembly level no inline-assembly bypass.

### A2. Deterministic — same input → same output

**Assumption:** Repeat invocations with identical (messageHash, signature,
publicKey) byte sequences return identical bool.

**Why it matters:** Determinism is required for chain consensus.

**Verifier confirms via:** pure declaration + RFC 8032 verification is
deterministic by construction.

**Auditor checks:** No use of randomness sources (RANDOM, BLOCKHASH,
PREVRANDAO).

### A3. Malformed input → false, not revert

**Assumption:** When signature length ≠ 64 OR publicKey length ≠ 32, the
function returns `false`, not `revert`.

**Why it matters:** The Registry's challenge-path logic interprets `false`
as "signature was invalid." If the verifier reverts, the entire challenge
transaction reverts, which can be exploited by an attacker to grief honest
challengers (submit a malformed-signature challenge that passes the
challenger's gas check but reverts inside the verifier, costing the
challenger their gas).

**Verifier confirms via:**

```solidity
// Ed25519Verifier.sol:45-46
if (signature.length != SIGNATURE_LEN) return false;
if (publicKey.length != PUBKEY_LEN) return false;
```

**Auditor checks:** Confirm there are no other reachable reverts. Key
sub-cases:
- Does Ed25519Lib.verify revert on point decompression of a non-canonical
  encoding? (RFC 8032 says "reject" — but should it return false or revert?
  PRSM expects false. Verify by inspection of decompression code.)
- Does Sha512.hash revert on any input? (KATs all pass; should be safe.)
- Does the assembly `mstore(add(m, 32), messageHash)` ever revert? (No —
  this is a pure memory write, not a panic-class operation.)
- Does the verifier handle gas-exhaustion gracefully? (No mitigation
  needed; revert-on-OOG is normal EVM behavior and the Registry's
  MIN_SLASH_GAS floor handles this externally.)

### A4. Verdict is exclusively a function of (messageHash, signature, publicKey)

**Assumption:** The verifier never accepts a signature for a different key,
never rejects a valid (canonical) signature for the correct key.

**Why it matters:** If the verifier accepts forgeries, malicious challenges
slash honest stakers. If it rejects valid signatures, honest providers can
be slashed by malicious challenges.

**Verifier confirms via:** RFC 8032 §5.1.7 verification algorithm.

**Auditor checks:** This is the core of the engagement. Edge cases:
- Low-order public keys (RFC 8032 §5.1.7 step 3).
- Non-canonical encoding of S (must reject S ≥ L).
- Signature malleability — does `verify(m, sig, pk)` and
  `verify(m, sig', pk)` both return true for `sig'` derived from `sig` by
  any transformation other than message replay?
- Point at infinity for R or A.

### A5. messageHash is treated as the message verbatim

**Assumption:** When the verifier receives `messageHash` (32 bytes), it
treats those 32 bytes as the FULL message that was signed. The off-chain
signer must have signed exactly those 32 bytes (typically the keccak256 of
their canonical receipt).

**Why it matters:** If the verifier's interpretation of `messageHash`
differs from the signer's, valid signatures will fail verification (or
worse, invalid signatures will succeed because of a mismatch).

**Verifier confirms via:**

```solidity
// Ed25519Verifier.sol:52-55
bytes memory m = new bytes(32);
assembly {
    mstore(add(m, 32), messageHash)
}
return Ed25519Lib.verify(k, r, s, m);
```

The 32-byte `messageHash` is converted to a 32-byte `bytes memory m` and
passed verbatim to the lib.

**Auditor checks:** Confirm the assembly correctly writes the 32-byte
messageHash to the `bytes`'s data area without overrun or off-by-one.
Specifically, `bytes memory m = new bytes(32)` allocates 32 bytes of length
prefix + 32 bytes of data; `mstore(add(m, 32), messageHash)` writes to the
correct offset (data starts at `m + 32`).

### A6. No side channels usable from on-chain context

**Assumption:** The verifier's gas usage does not leak information about
the validity (or cause-of-rejection) of inputs in a way that an on-chain
caller can exploit.

**Why it matters:** On-chain side channels are weaker than off-chain (no
microsecond timing) but gas-usage difference between "rejected because
malformed length" vs "rejected because curve math says invalid" might allow
an adversary to learn something about a signature without on-chain
verification cost.

**Verifier confirms via:** all paths through the verifier perform full work
when input is well-formed.

**Auditor checks:** The malformed-length early returns ARE a side channel
(constant-time vs full-cost). PRSM accepts this — distinguishing
malformed from invalid is not a confidentiality concern. Auditor should
confirm there are no FURTHER side channels within the cryptographic path
(e.g., early return on low-order point detection vs full math leading to
final reject).

### A7. Bounded gas usage

**Assumption:** The verifier's gas usage is bounded above by some value
≤ MIN_SLASH_GAS budget allowed in the challenge path.

**Why it matters:** If gas usage is unbounded (e.g., loops over the message
that an attacker can size), an adversary can submit challenges that cost
infinite gas to verify, locking the challenge path.

**Verifier confirms via:**
- `messageHash` is fixed 32 bytes (no attacker-controlled length).
- Existing test (`Ed25519Verifier.test.js:112-141`) measures gas at
  ~456,686. Loose bound: 400K–3M.

**Auditor checks:** Confirm the SHA-512 hash step is bounded by message
length and message length is fixed at 32 bytes (in production via wrapper).
Confirm the curve math has constant gas cost regardless of which scalar/point
is used (this is a side-channel concern but also a gas-DoS concern —
auditor should explore worst-case inputs).

### A8. Constant verdict for canonical signatures

**Assumption:** If a signature was produced by a compliant Ed25519 signer
over a known public key and known message, `verify(messageHash, signature,
publicKey)` returns `true` — always, deterministically, on every chain.

**Why it matters:** This is the core safety invariant. RFC 8032 §7.1 vectors
empirically confirm this for the canonical corpus.

**Verifier confirms via:** RFC 8032 §7.1 KATs pass (see `test-vectors-ed25519.md`).

**Auditor checks:** Random differential testing against `tweetnacl` and/or
Node `crypto` over thousands of random (key, message) pairs to confirm
no false-rejection.

## Threat model summary for the auditor

The Registry's INVALID_SIGNATURE challenge path is a slashing primitive. It
is invoked rarely (only on dispute) but when invoked it can result in:

- **Honest slash:** challenger correctly identifies a forged provider
  signature. Provider's stake is slashed.
- **Adversarial slash:** challenger forges a "challenge" that the verifier
  incorrectly accepts as valid (i.e., the verifier returns `true` for an
  invalid signature). Honest provider's stake is slashed unfairly.
- **Adversarial dismissal:** an honest challenge is verifier-rejected. The
  attacker keeps slashable stake.

Categories #2 and #3 are the high-severity attack surface for L3.

## Hand-off package

This memo + the test-vector files + the upstream port diff = the entire
context the auditor needs to begin their engagement. A pre-engagement
budget conversation should reference these artifacts directly.
