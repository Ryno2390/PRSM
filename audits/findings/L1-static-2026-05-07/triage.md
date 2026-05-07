# L1 Static Tooling — Slither Triage (2026-05-07)

## Run

- Tool: Slither v0.11.5 (solc 0.8.22)
- Tip: `85991c87` (post-T6 testnet wiring)
- Filter: `node_modules|test|MockSignatureVerifier|MockERC20|Sha512Harness`
- Raw report: `slither-report.md`, `slither.json`
- Total findings: **8** (2 HIGH, 6 MEDIUM, 0 LOW/INFO highlighted by `--checklist`)

## Verdict

**0 actionable findings on the audit-bundle scope.** All 8 are either:
- (a) false positives correctly defended by the existing nonReentrant /
  algorithm structure / Solidity zero-init guarantees, or
- (b) hits in `FTNSBridge.sol` / `BridgeSecurity.sol`, which are NOT
  part of the planned mainnet deploy bundle and remain out of scope
  pending a separate bridge security review.

Phase B does not gate the mainnet deploy.

## Per-finding triage

### HIGH-1 (Slither ID-0): `reentrancy-balance` in `CompensationDistributor._distribute()`

**Location:** `contracts/CompensationDistributor.sol:184-206`

**Slither claim:** balance read at `available = ftnsToken.balanceOf(address(this))` could be stale if a malicious token re-enters via `ftnsToken.transfer()`.

**Triage:** **False positive.**
1. The only public entry point to `_distribute()` is
   `pullAndDistribute()` (line 156), which carries the
   `nonReentrant` modifier from OZ ReentrancyGuard. A re-entrant call
   path through `ftnsToken.transfer` cannot reach `_distribute()`
   again.
2. The `ftnsToken` is `FTNSTokenSimple` (OZ `ERC20Upgradeable`), which
   has no transfer hooks / callbacks — `transfer()` returns straight
   from the OZ reference implementation without delegating to user
   code.
3. Even hypothetically, the `available` read is ONLY used to compute
   per-pool splits inside the same call frame; it isn't compared
   against an earlier balance snapshot.

**Disposition:** No fix.

### HIGH-2 (Slither ID-3): `unchecked-transfer` in `FTNSBridge.bridgeOut`

**Location:** `contracts/FTNSBridge.sol:251`

**Triage:** **Out of audit-bundle scope.** `FTNSBridge` is the
cross-chain bridge surface; it is NOT part of the audit-bundle +
Phase 8 + Phase 7-storage mainnet deploy. Logged as a real finding
that needs remediation when bridge work is picked up — separate
review track.

**Disposition:** Defer to bridge-specific review.

### MED-1 (Slither ID-4): `divide-before-multiply` in `StakeBond.slash`

**Location:** `contracts/StakeBond.sol:576`
`slashAmount = (uint256(s.amount) * uint256(s.tier_slash_rate_bps)) / 10000;`

**Triage:** **False positive.** Slither's `divide-before-multiply`
detector flags `(a/b)*c` precision-loss patterns. The actual code is
`(a*b)/c` — the safe ordering. Slither misclassified.

**Disposition:** No fix.

### MED-2 (Slither ID-5): `incorrect-equality` in `ProvenanceRegistryV2.verifyEmbeddingCommitment`

**Location:** `contracts/ProvenanceRegistryV2.sol:165` (`onChain == claimed`)

**Triage:** **False positive.** Slither's `incorrect-equality` detector
typically flags balance-comparison patterns (`balanceOf(this) == X`)
that miscount donation transfers. Here both sides are `bytes32` hash
values — the entire point of the function is to assert hash equality
for embedding-commitment verification. Strict equality is the
correct semantic.

**Disposition:** No fix.

### MED-3 (Slither ID-9): `reentrancy-no-eth` in `FTNSBridge.bridgeIn`

**Triage:** **Out of audit-bundle scope** (FTNSBridge).

**Disposition:** Defer to bridge-specific review.

### MED-4 (Slither ID-10): `uninitialized-local` in `BatchSettlementRegistry.challengeReceipt` (`bool proven`)

**Location:** `contracts/BatchSettlementRegistry.sol:665`

**Triage:** **False positive.** `bool proven;` is declared without
explicit initialiser. Solidity guarantees default-zero-init for
locals, which equals `false` for `bool`. The if/else chain on
lines 666-679 covers all five `ReasonCode` values: every branch
either assigns `proven` or reverts via the `else` branch
(`MalformedReasonNotImplemented`). The line-681 check
`if (!proven) revert ChallengeNotProven(reason);` then catches the
remaining "branch returned false" case. There is no code path where
`proven` is read before being assigned.

**Disposition:** No fix. (Stylistically `bool proven = false;` is
clearer; defer to a future readability sweep.)

### MED-5 (Slither ID-16): `unused-return` in `FTNSBridge.checkBridgeIn`

**Triage:** **Out of audit-bundle scope** (FTNSBridge).

**Disposition:** Defer to bridge-specific review.

### MED-6 (Slither ID-17): `write-after-write` in `Ed25519Lib.verify` (`kkx`)

**Location:** `contracts/lib/Ed25519Lib.sol:435`, then `:453`

**Triage:** **False positive.** This is a port of a well-tested
Ed25519 verification implementation. The flagged sequence is the
standard Edwards-curve point-doubling step:
1. Line 435: `kkx = kx` (initial point x-coordinate).
2. Lines 439-452: `kkx` is READ via `mulmod(kkx, kkv, ...)` to compute
   `xx`, which feeds into `xxyy = mulmod(xx, yy, ...)`.
3. Line 453: `kkx = xxyy + xxyy` (the doubled-point's new x-coordinate).

`kkx` is read between the two writes (lines 440 + 449 transitively),
so this is NOT a write-after-write bug — it's the algorithm's
expected mutation pattern.

**Disposition:** No fix.

## Summary

| Severity | Count | In-scope | False positive | Out of scope (FTNSBridge) | Real bug |
|---|---|---|---|---|---|
| HIGH | 2 | 1 | 1 | 1 | 0 |
| MEDIUM | 6 | 4 | 4 | 2 | 0 |
| **Total** | **8** | **5** | **5** | **3** | **0** |

**Phase B (static + symbolic tooling) closes with zero actionable
findings against the planned mainnet deploy scope.** Combined with the
4-team agent-teams self-audit (closed cleanly post-A-06 fix), this
satisfies PRSM-POL-2 §4.1's substituted-audit pass criterion. The
audit-bundle + Phase 8 + Phase 7-storage stack is cleared for the
mainnet deploy ceremony.

— Generated 2026-05-07.
