# Team C — Signature & Cryptographic Surface — Findings

**Audit pin:** commit `589c14d2` (HEAD of `main`) — equivalent to
freeze tag `cumulative-audit-prep-20260504-h`.
**Date:** 2026-05-04.
**Scope:** Integration-layer signature/replay/payload-binding analysis
across the on-chain contracts. RFC 8032 / FIPS 180-4 spec-conformance is
deferred to the L3 specialist crypto audit per the pre-engagement
artifacts in `audits/findings/L3-crypto/`.

---

## Summary by severity

| Severity      | Count |
|---------------|-------|
| **Critical**  |   1   |
| High          |   0   |
| Medium        |   2   |
| Low           |   2   |
| Informational |   3   |

Headline concern is **C-INT-01 (Critical)**: the INVALID_SIGNATURE
challenge path in `BatchSettlementRegistry` does NOT bind the
canonical signing message to the on-chain receipt leaf. As a
consequence, **any party in possession of a published off-chain
receipt can adversarially slash the receipt's provider with
100% success rate**, regardless of the receipt's actual validity.
The attack uses the real (pubkey, signature) bytes (which are
mandatorily public so third parties can verify off-chain) and an
arbitrary attacker-chosen `signingMessage`. PoC is in
`contracts/test/audit-team-c/C-INT-01-invalid-signature-forgery.test.js`
and verifies under the real `Ed25519Verifier` (not a mock).

This finding **must block production deployment of the
INVALID_SIGNATURE challenge surface and the StakeBond slasher
wiring** until either (a) `ReceiptLeaf` is extended with a
`signingMessageHash` field, or (b) the contract reconstructs the
canonical signing message on-chain from existing leaf fields.

---

## C-INT-01 — Adversarial slashing via unbound `signingMessage` (CRITICAL)

**Affected file:** `contracts/contracts/BatchSettlementRegistry.sol`
(`_handleInvalidSignature`, lines 540-557).

**Attack vector category:** C10 (off-chain field omission) +
C12-adjacent (asymmetric forgery primitive), with
realized impact via the slasher integration on the same call.

### Description

`_handleInvalidSignature` decodes the challenger-supplied `auxData`
into `(signingMessage, publicKey, signature)`, then enforces:

```solidity
if (keccak256(publicKey) != leaf.providerPubkeyHash) return false;
if (keccak256(signature) != leaf.signatureHash) return false;

bytes32 messageHash = keccak256(signingMessage);
bool valid = signatureVerifier.verify(messageHash, signature, publicKey);
return !valid; // challenge succeeds iff verification fails
```

Bindings present:

- `publicKey` is bound to `leaf.providerPubkeyHash`.
- `signature` is bound to `leaf.signatureHash`.

Bindings **absent**:

- `signingMessage` is supplied verbatim by the challenger and is
  **not** committed in the `ReceiptLeaf` struct. The struct contains
  `outputHash` and other fields that *could* be the signing input,
  but the contract never reconstructs or hashes them. The
  challenger has unrestricted choice of `signingMessage`.

### Why this is exploitable in practice

The PRSM design publishes off-chain receipts so any party can
independently verify them. A receipt necessarily includes the
provider's `pubkey` and `signature` in plaintext — otherwise no third
party could verify it off-chain. Therefore **`pubkey` and `signature`
bytes are public knowledge** for any receipt that ever traverses the
network; only the leaf-committed *hashes* of those bytes are stored
on-chain.

An attacker who has seen ANY published receipt:

1. Has the real `(pubkey, signature)` bytes whose `keccak256` matches
   `leaf.providerPubkeyHash` and `leaf.signatureHash`.
2. Picks an arbitrary `signingMessage` the provider never signed
   (e.g., 32 random bytes, or the literal string
   `"attacker-picked-arbitrary-bytes"`).
3. Submits `challengeReceipt(batchId, leaf, [], INVALID_SIGNATURE,
   abi.encode(signingMessage, pubkey, signature))`.
4. `verifier.verify(keccak256(signingMessage), signature, pubkey)`
   returns `false` (because `signature` was produced over the real
   message, not the attacker's choice).
5. `_handleInvalidSignature` returns `!valid == true` — challenge
   proven.
6. `challengeReceipt` invalidates the receipt (provider loses the
   payment for honest work) AND, if `stakeBond` is configured and the
   batch was committed at a non-zero `tier_slash_rate_bps`, the
   provider's stake is slashed via `stakeBond.slash(provider,
   challenger, batchId)`. With `tier_slash_rate_bps == 10000` (Tier C
   "critical"), 100% of the provider's bond is slashed; 70% goes to
   the attacker as bounty (line 361 of `StakeBond.sol`); 30% goes to
   the Foundation reserve.

The attacker's gain: 70% of the provider's full stake. The
provider's loss: 100% of the stake + the receipt's value.

### Impact

- **Scope.** Affects every provider who has ever published a
  receipt, with the live `Ed25519Verifier` wired (production
  configuration). Pre-Phase-7 deployments without `stakeBond`
  configured suffer only receipt invalidation; post-Phase-7
  deployments suffer full stake slashing on top.
- **Cost to attacker.** ≈ one challenge-tx of gas (the gas-floor
  guard requires ≥ 150K gas for the slash + ~450K for the verifier =
  ~600K-1M gas).
- **Cost to victim provider.** 100% of bonded stake (when
  `tier_slash_rate_bps == 10000`) plus the value of one receipt's
  worth of unpaid work.
- **Probability of attacker detection.** Zero on-chain — the
  challenge appears identical to a legitimate INVALID_SIGNATURE
  challenge (which was supposed to fire only when the provider
  posted a forged signature). Off-chain investigation could
  reconstruct the attack only by re-running Ed25519 verification
  against the *intended* signing message — which the contract
  doesn't know.
- **Sybil amplification.** Every published receipt is one slashable
  surface. A high-volume provider has hundreds of receipts in any
  given window; an attacker can pick whichever receipt corresponds
  to the largest stake or the most-funded batch.

### PoC

`contracts/test/audit-team-c/C-INT-01-invalid-signature-forgery.test.js`
deploys the real (non-mock) `Ed25519Verifier`, the real
`StakeBond`, the real `EscrowPool`, and the real
`BatchSettlementRegistry`. The test:

1. Generates a real PyNaCl-equivalent (tweetnacl) Ed25519 keypair
   for the provider and signs a valid 32-byte message — the real
   signature does verify (sanity check passes).
2. Provider commits a leaf binding the real `(pubkey, signature)`
   hashes and bonds 50 000 FTNS at the critical (100% slash) tier.
3. Attacker (a wholly separate signer with no relationship to the
   provider) submits an INVALID_SIGNATURE challenge with the real
   pubkey + real signature but a literally-string `"attacker-picked-
   arbitrary-bytes"` for `signingMessage`.
4. The challenge succeeds. Provider's stake goes from 50 000 FTNS
   → 0. Attacker's `slashedBountyPayable` goes from 0 → 35 000 FTNS
   (70% of 50 000). The receipt is marked invalidated.

Run:

```
cd contracts && npx hardhat test test/audit-team-c/C-INT-01-invalid-signature-forgery.test.js
```

Output:

```
  Audit Team C — C-INT-01: INVALID_SIGNATURE adversarial slashing
    ✔ CRITICAL: any holder of (pubkey, signature) bytes can slash the provider with arbitrary signingMessage

  1 passing (332ms)
```

### Recommended remediation

Either (any one of these closes the vulnerability):

**Option A — extend the leaf.** Add a `signingMessageHash` field to
`ReceiptLeaf`. Require:

```solidity
if (keccak256(signingMessage) != leaf.signingMessageHash) return false;
```

before the verifier call. This forces the challenger to supply the
exact bytes the provider committed to as the canonical signing
input. Updates required: leaf-encoding spec (off-chain Python),
`_hashLeaf` continues to work (struct change auto-reflows in
`abi.encode`), `Phase 3.1` design §6 needs the new field documented
as part of the canonical receipt → leaf encoder.

**Option B — reconstruct on-chain.** Define the canonical signing
message as a deterministic function of leaf fields the provider
already commits to. For example:

```solidity
bytes32 reconstructedMessageHash = keccak256(abi.encode(
    leaf.jobIdHash,
    leaf.shardIndex,
    leaf.providerIdHash,
    leaf.providerPubkeyHash,
    leaf.outputHash,
    leaf.executedAtUnix,
    leaf.valueFtns
));
bool valid = signatureVerifier.verify(reconstructedMessageHash, signature, publicKey);
return !valid;
```

This requires the off-chain signer to sign EXACTLY this canonical
form; any divergence must be caught by parity tests.
`SolidityPythonParity.test.js` already exists for the leaf-hash side
— add a parallel `parity-signing-message` corpus.

**Option A is recommended** — it's strictly more flexible (the
canonical signing message can include fields not present in the
leaf, e.g., session keys, deadline, audit metadata) and reduces the
risk of a future leaf-field addition silently changing the signing
preimage and breaking off-chain signers.

Either way, also add a regression test asserting that an attacker-
picked `signingMessage` ≠ the canonical one is rejected with
`ChallengeNotProven`.

### Severity reasoning

CRITICAL (CVSS-equivalent ≈ 9.5):

- **Confidentiality:** N/A.
- **Integrity:** Total — provider's stake is fully forfeitable on
  attacker's choice of timing, with no defense.
- **Availability:** High — provider can be permanently locked out
  of the network (stake gone → tier drops to "open" → no future
  dispatches at staked tiers).
- **Authentication bypass:** Yes — the contract's stated
  authentication (Ed25519 signature over the canonical receipt) is
  bypassed because the contract doesn't know what "canonical" means.
- **Exploit complexity:** Trivial — single-tx attack, no race
  condition, no preimage search.

---

## C-INT-02 — `ISignatureVerifier.verify` declared `view`, allowing future state-reading verifier (MEDIUM)

**Affected file:**
`contracts/contracts/BatchSettlementRegistry.sol:31-36`.

The interface declares:

```solidity
function verify(
    bytes32 messageHash,
    bytes calldata signature,
    bytes calldata publicKey
) external view returns (bool);
```

`view` permits the implementation to read storage / call-frame
state. The current production `Ed25519Verifier` is `pure` — strictly
stronger and matches caller-assumption A1 (pure function) +
A2 (deterministic) in `audits/findings/L3-crypto/caller-
assumptions.md`. However, `setSignatureVerifier(address)` is owner-
adjustable at any time. A future migration that swaps in a `view`
implementation — even an honest one that reads e.g. a key-revocation
registry — would silently weaken A1+A2 because:

- A2 (determinism) is broken if the verifier reads any mutable
  storage (the same `(messageHash, signature, publicKey)` could
  return different verdicts at different blocks).
- A4 (verdict is a function of input only) is broken for the same
  reason.

The PRSM challenge surface depends critically on the verifier being
pure: if the verifier becomes context-dependent, an attacker can
manipulate the off-chain mempool / block-ordering to schedule
challenges when verification differs (replay across state).

**Recommendation:** change the interface to `pure`. The interface
is internal-PRSM only (no third-party integrators), so the change
is safe. If `view` is genuinely needed for some future verifier
(e.g., one that reads PublisherKeyAnchor on-chain), introduce a
SECOND interface `IStatefulSignatureVerifier` and clearly document
which contracts may use which.

**Severity:** Medium. Not directly exploitable today (current
verifier is pure), but creates a low-friction path to silently
weaken security via owner-controlled `setSignatureVerifier`.

---

## C-INT-03 — `BridgeSecurity` keeps a typo'd legacy `BRIDGE_MESSAGE_TYPEHASH` constant in storage (LOW / Info)

**Affected file:**
`contracts/contracts/BridgeSecurity.sol:37-47`.

Lines 37-41 define `BRIDGE_MESSAGE_TYPEHASH` as a hard-coded
`0xd479...e0e` literal whose comment claims it represents
`keccak256("BridgeMessage(address recipient,uint256 amount,uint256 sourceChain,bytes32 sourceTransactionId,uint256 nonce)")`.

Inspection: the literal `0xd479a2d9a9ff3b1db67c6b3d7c8a0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e`
is repetitive `0e0e...` — clearly a placeholder, not a real keccak256.

Lines 44-47 then define `BRIDGE_MESSAGE_TYPEHASH_V2` as the actual
`keccak256(...)` with the corrected field-name set
(`sourceChainId`/`sourceTxId` instead of `sourceChain`/
`sourceTransactionId`). `hashBridgeMessage` (line 452) uses
`_V2`. So the V1 constant is **dead code**.

Risk: a future caller (or a future code-author who doesn't read the
git history) might use the V1 constant, producing a structHash that
doesn't match what validators sign — causing all bridge-in calls
to fail. Cleanliness, not security.

**Recommendation:** delete `BRIDGE_MESSAGE_TYPEHASH` (the V1 line)
and rename `BRIDGE_MESSAGE_TYPEHASH_V2` → `BRIDGE_MESSAGE_TYPEHASH`.

**Severity:** Low (operational cleanliness; not a live bug).

---

## C-INT-04 — `BridgeMessage` does not include `destinationChainId` (LOW)

**Affected file:**
`contracts/contracts/BridgeSecurity.sol:59-65`.

The struct contains `sourceChainId` but not `destinationChainId`.
Cross-chain replay across destination chains is mitigated by the
EIP-712 domain separator (which DOES bind `block.chainid` and
`address(this)` — see lines 528-534), so a signature targeted at
e.g. Base mainnet cannot replay against Ethereum mainnet via the
same `verifyingContract` address (chainId differs in the domain
separator).

However:

- If the bridge is ever deployed using CREATE2 with the same salt
  on multiple chains, the contract addresses will collide — and the
  chainId binding becomes the sole defense. Defense-in-depth would
  add `destinationChainId` to the struct so even a single-domain-
  separator misconfiguration cannot replay.
- The current scheme requires every validator to also know the
  destination chain implicitly (via the verifying contract's
  chainId). If validators ever sign on an off-chain relay that
  doesn't know the destination chainId, they could be tricked into
  signing a message that lands on the wrong chain. Adding
  `destinationChainId` to the struct makes the destination explicit
  in the signed payload.

**Recommendation:** add `destinationChainId` to `BridgeMessage` and
require it equals `block.chainid` in `verifyBridgeSignatures`.
Belt-and-suspenders; the EIP-712 domain separator already covers
the on-chain replay vector.

**Severity:** Low (defense-in-depth; current single-line defense
is adequate for in-scope deployments).

---

## C-INT-05 — Minor: `BridgeSecurity._verifySignature` uses `abi.encodePacked` for fixed-width sig assembly (Informational)

**Affected file:**
`contracts/contracts/BridgeSecurity.sol:514`.

```solidity
bytes memory signatureBytes = abi.encodePacked(sig.r, sig.s, sig.v);
```

This is `abi.encodePacked` of three fixed-size types: `bytes32 +
bytes32 + uint8` = exactly 65 bytes. **No collision is possible**
because none of the operands is dynamic-length. Pattern is OZ-
canonical `(r, s, v)` packing.

The auditor flag this only because `abi.encodePacked` IS a known
collision source when used with consecutive dynamic-length operands
— it is NOT one here. Safe as written.

**Recommendation:** add a comment confirming the safety so a future
contributor doesn't wrongly conclude a refactor is needed. (Or
switch to `abi.encode(sig.r, sig.s, sig.v)` — slightly more gas,
identical correctness, no encoder ambiguity for the reader.)

**Severity:** Informational.

---

## C-INT-06 — Minor: `BridgeSecurity.removeValidator` can leave threshold > totalValidators when totalValidators reaches 0 (LOW / Info)

**Affected file:**
`contracts/contracts/BridgeSecurity.sol:386-401`.

```solidity
if (signatureThreshold > totalValidators && totalValidators > 0) {
    signatureThreshold = totalValidators;
    ...
}
```

If the admin removes the last validator (totalValidators → 0),
the second clause fires and the threshold is left unchanged
(e.g., 1). All subsequent `verifyBridgeSignatures` calls will
revert with `InsufficientSignatures(1, 0)` — this BRICKS the
bridge until the admin adds a new validator. Not a security
break (the bricked state cannot process invalid messages either),
but it's a footgun: the admin who removes the last validator may
not realize the bridge is locked.

**Recommendation:** either (a) revert when removing the last
validator unless threshold is also reduced explicitly, or (b)
leave-as-is but add a NatSpec warning.

**Severity:** Low (operational, not security).

---

## C-INT-07 — Off-chain HandoffToken signing payload uses JSON canonical encoding without explicit domain prefix (Informational)

**Affected file (off-chain reference, OUT OF AUDIT SCOPE per team
prompt — flagged for completeness):**
`prsm/compute/chain_rpc/protocol.py:HandoffToken.signing_payload`.

The `signing_payload` is `json.dumps(payload, sort_keys=True)` of a
5-or-6-key dict. There is no "PRSM-HandoffToken-v1" magic prefix or
versioning byte. As a result, if the Ed25519 settler key is EVER
reused for a different signing context that happens to use the same
JSON shape (unlikely but not structurally prevented), signatures
could cross-context replay.

The `deadline_unix` field bounds replay to within the deadline
window. The phase 3.x.11 addendum §3.8(7) explicitly documents the
"replay attack window inside `deadline_unix`" as honest-scope:
per-stage nonce cache deferred.

Flagged for completeness. Off-chain payload construction is the
team prompt's "out of scope but worth noting" category; the L3
specialist crypto audit may pick this up.

**Recommendation:** prefix `signing_payload` with a domain string
e.g., `b"PRSM-HandoffToken-v1\n" + json.dumps(...)`. Trivial change;
removes one class of cross-protocol replay risk.

**Severity:** Informational (off-chain, deferred per scope).

---

## C-INT-08 — Minor: `Ed25519Verifier` declares `verify` as `pure` while interface says `view`; future-compatibility note (Informational)

**Affected files:**
`contracts/contracts/Ed25519Verifier.sol:44` (`pure`) +
`contracts/contracts/BatchSettlementRegistry.sol:35` (`view`).

`pure` is strictly stronger than `view`, so the implementation is
compliant with the interface (Solidity allows narrowing the
state-mutability of an override). Today this is a non-issue.

See **C-INT-02** above for the related — and live — concern that
the `view` declaration is too permissive. Listed separately here
only because the implementation today actually IS `pure`; the
finding above addresses the interface-side weakness.

**Severity:** Informational (consistency note).

---

# Vectors evaluated and cleared

For each of the 12 vectors C1–C12 in the team prompt, an explicit
verdict is given below.

| ID  | Vector                                              | Verdict                                                                                                                                                                                                                                                                                                                          |
|-----|-----------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| C1  | Cross-chain replay (Sepolia → Mainnet)              | **CLEARED for on-chain ECDSA paths.** `BridgeSecurity` EIP-712 domain separator binds `block.chainid` and `address(this)` (lines 528-534). The Ed25519 challenge path has no chainId binding by design — the leaf-committed message is supposed to bind it. This is contingent on **C-INT-01** being remediated; see also C2.    |
| C2  | Cross-contract replay                               | **CLEARED for `BridgeSecurity`** (domain separator binds `verifyingContract`). For Ed25519 challenge path: contingent on **C-INT-01** remediation — once `signingMessage` is bound, the off-chain canonical receipt encoding is responsible for binding the verifying contract address into the message.                          |
| C3  | Ed25519 malleability                                | **Deferred to L3 specialist audit.** Baseline RFC 8032 §7.1 KATs all pass; bit-flip rejection works (`audits/findings/L3-crypto/test-vectors-ed25519.md`).                                                                                                                                                                        |
| C4  | Ed25519 low-order point rejection                   | **Deferred to L3 specialist audit.**                                                                                                                                                                                                                                                                                             |
| C5  | Ed25519 non-canonical S encoding (S ≥ L)            | **Deferred to L3 specialist audit.**                                                                                                                                                                                                                                                                                             |
| C6  | SHA-512 implementation bugs                         | **Deferred to L3 specialist audit.** All 11 FIPS 180-4 KATs pass (`audits/findings/L3-crypto/test-vectors-sha512.md`).                                                                                                                                                                                                           |
| C7  | ECDSA s-malleability                                | **CLEARED.** `BridgeSecurity._verifySignature` uses OpenZeppelin Contracts v5.3.0 `ECDSA.recover`, which enforces `s ≤ secp256k1n / 2` (low-s) per the v4.7.3+ default. Inspected `node_modules/@openzeppelin/contracts/utils/cryptography/ECDSA.sol` — low-s check is present in `tryRecover`. No bypass via short-sig / vs-form. |
| C8  | Signature stripping / multi-sig downgrade           | **CLEARED.** `BridgeSecurity.verifyBridgeSignatures` enforces `signatures.length >= signatureThreshold` (line 254), counts only `validSignatureCount` and re-checks `>= signatureThreshold` after the loop (line 295), and rejects duplicate validators by an O(n²) prior-scan. Cannot be downgraded to single-sig.               |
| C9  | Replay within the challenge window                  | **CLEARED.** `BatchSettlementRegistry.invalidatedReceipts[batchId][leafHash]` mapping prevents the same receipt being challenged twice (line 423). The same receipt across DIFFERENT batches is the DOUBLE_SPEND path, separately handled.                                                                                       |
| C10 | Off-chain field omission                            | **CONFIRMED — see C-INT-01 (CRITICAL).** `signingMessage` is supplied by the challenger and never bound to the leaf.                                                                                                                                                                                                             |
| C11 | `abi.encodePacked` collision in signing payloads    | **CLEARED.** Audited every `abi.encodePacked` usage:<br>• `BridgeSecurity._computeDomainSeparator` uses `abi.encode` (safe).<br>• `BridgeSecurity.hashBridgeMessage` packs `"\x19\x01" + ds + structHash` — all fixed-size, EIP-712 canonical.<br>• `BridgeSecurity._verifySignature` packs r+s+v — all fixed-size, OZ canonical. |
| C12 | Public-key registration spoofing                    | **CLEARED for `PublisherKeyAnchor`.** `register(publicKey)` derives `nodeId = bytes16(sha256(publicKey))` on-chain, so the caller can only register pubkeys whose sha256 they actually compute — they cannot anchor a victim's nodeId to an attacker-controlled key. Length-check enforces 32-byte keys.                          |

---

# Out-of-scope notes

These items were noticed during the audit but are out of the team
prompt's scope. Documented for the audit-bundle coordinator's
dispatch:

1. **Off-chain HandoffToken replay-window** — `deadline_unix` is the
   sole replay defense; documented honest-scope per phase 3.x.11
   addendum §3.8(7). See **C-INT-07** above.

2. **Off-chain HandoffToken JSON-canonical signing payload** — no
   domain prefix string. See **C-INT-07** above.

3. **`RunLayerSliceRequest` end-to-end authentication** — explicitly
   honest-scope deferred per addendum line 203. Not a contract-layer
   concern.

---

# Reproduction

```sh
cd contracts
npx hardhat test test/audit-team-c/
```

Expected: 1 passing test (the C-INT-01 PoC). The PoC asserts the
exploit; once **C-INT-01** is remediated by either Option A or
Option B above, this test should be inverted (assert that the
attempted forgery REVERTS with `ChallengeNotProven`) and the L3
specialist crypto audit can proceed.
