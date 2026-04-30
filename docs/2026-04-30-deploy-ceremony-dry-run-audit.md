# Deploy-Ceremony Dry-Run Audit — 2026-04-30

**Context:** all three multi-sig hardware devices arrive 2026-05-01. This
audit walks the existing mainnet deploy infrastructure with fresh eyes
under the assumption hardware lands in <12 hours, surfaces gaps, and
addresses the load-bearing ones in code.

## Scope

- `scripts/rehearse-deploy.sh` — orchestrator
- `contracts/scripts/deploy-audit-bundle.js` — Phase 3.1 + 7 + 7.1 bundle
- `contracts/scripts/deploy-phase8-emission.js` — emission + comp distributor
- `contracts/scripts/deploy-phase7-storage.js` — storage slashing + key dist
- `contracts/scripts/deploy.js` — Phase 1.3 FTNSToken (UUPS proxy)
- `contracts/hardhat.config.js` — network configs
- `contracts/scripts/transfer-ownership.js` — **NEW** in this audit

## Method

Walked the orchestrator end-to-end against a fresh hardhat-local node
twice: once before extending it, once after. Both passed all 15+
post-deploy invariants. Hardware-day blockers identified by asking "if
the operator runs this exact script tomorrow with real signers, what
breaks or silently misconfigures?"

## Gaps Surfaced

### G1 — Ownership-transfer script missing **(HIGH — addressed)**

The runbook documents a two-phase deploy model (hot-deployer cross-wires
under invariants → `transferOwnership(MULTISIG)` at end), but no script
existed to perform step 2. Without it, the operator either re-implements
the transfer ad-hoc on mainnet day (ceremony-time bug surface) or skips
the transfer entirely (deployer hot key retains owner() rights forever).

**Fix:** added `contracts/scripts/transfer-ownership.js`. Reads three
manifests (audit-bundle, phase8, phase7-storage), iterates 7 Ownable
contracts (EscrowPool, BatchSettlementRegistry, StakeBond,
EmissionController, CompensationDistributor, StorageSlashing,
KeyDistribution), uses minimal Ownable ABI to avoid artifact dependency,
verifies post-transfer `owner()`, writes a transfer manifest. Idempotent:
re-running after a successful transfer skips cleanly. Mainnet hardening:
rejects EOA multi-sig (must be a contract on `base`/`mainnet`), rejects
deployer == multi-sig.

**Verified end-to-end** by extending `rehearse-deploy.sh` with a
post-deploy "[5/6] Rehearsing ownership transfer to stub multi-sig" step
that uses hardhat default account #1 as `STUB_MULTISIG`. The rehearsal
now exercises:
1. Initial transfer of all 7 Ownable contracts to the stub.
2. Idempotency re-run that must skip all 7 — failure aborts the script.

### G2 — `AUTHORIZED_VERIFIER` silent fallback **(MEDIUM — addressed)**

Original orchestrator default:
```bash
: "${AUTHORIZED_VERIFIER:=${FOUNDATION_RESERVE_WALLET}}"
```
On testnet/mainnet this would silently set
`StorageSlashing.authorizedVerifier` to whatever
`FOUNDATION_RESERVE_WALLET` is — they are semantically distinct (the
verifier is the off-chain prover EOA, NOT the foundation treasury).

**Fix:** network-conditional guard. On `hardhat-local`, fall back to
`FOUNDATION_RESERVE_WALLET` and log it. On any other network, hard-fail
with an explanation rather than silently misconfiguring.

### G3 — `PRIVATE_KEY` placeholder breaks all networks **(MEDIUM — addressed)**

Pre-existing `contracts/.env` contains `PRIVATE_KEY=your_private_key_here`.
The hardhat config guard `process.env.PRIVATE_KEY ? [...] : []` only
filtered empty/undefined; the placeholder string slipped through and
tripped hardhat's "private key too short, expected 32 bytes" config
validator. This blocked **all** network targets including
`hardhat-local` (i.e., the rehearsal couldn't even run).

**Fix:** added `pkAccounts()` helper validating `0x[0-9a-fA-F]{64}`. If
`PRIVATE_KEY` is set but malformed, return `[]` rather than passing it
through. All 8 network entries now use the helper.

### G4 — Pre-deployed verifier not interface-checked on mainnet **(MEDIUM — documented)**

`StorageSlashing.authorizedVerifier` is declared as `address` and the
contract cannot enforce that what's at that address actually implements
the verifier protocol. On mainnet this is normally a Foundation-owned
EOA running the off-chain prover, which is fine, but if/when migrated to
a verifier contract (PROOF_VERIFIER_CONTRACT in the runbook), the
deploy-time check is just `extcodesize > 0` — not interface compliance.

**Status:** not blocking for v1 mainnet (verifier stays an EOA per
current runbook). Flag for the verifier-contract migration milestone.

### G5 — Phase 1.3 FTNSToken deploy not in orchestrator **(LOW — documented)**

`deploy.js` deploys FTNSToken as a UUPS proxy. It is **not** wired into
`rehearse-deploy.sh`. On mainnet day, FTNSToken is the FIRST contract
deployed (every downstream contract takes its address as a constructor
arg). Mainnet operator currently has to remember to run `deploy.js`
first, capture the proxy address, export it as `FTNS_TOKEN_ADDRESS`,
then run `rehearse-deploy.sh`.

**Status:** not addressed in this dry-run because Phase 1.3 has its own
audit clock (separate ceremony documented in the Phase 1.3 deploy
runbook). Bundling into `rehearse-deploy.sh` is appropriate when both
run on the same day, which is the current plan. Flagged for the
mainnet-day master checklist.

### G6 — `FOUNDATION_RESERVE_WALLET` defaults to `0x...dEaD` **(LOW — documented)**

Orchestrator default:
```bash
: "${FOUNDATION_RESERVE_WALLET:=0x000000000000000000000000000000000000dEaD}"
```
On hardhat-local this is fine. On mainnet, if the operator forgets to
set `FOUNDATION_RESERVE_WALLET`, every contract that takes it as
constructor arg gets `0x...dEaD` (a burn address) and EmissionController
mints would be irretrievable. Currently relies on operator vigilance.

**Status:** would be a HIGH gap if the operator could run mainnet
without an explicit env var. Hardening proposal: extend the existing
mainnet check (lines 81-84 of `rehearse-deploy.sh` for FTNS) to
require `FOUNDATION_RESERVE_WALLET` on `base`/`mainnet`. Not implemented
in this dry-run because the runbook calls it out, but worth wiring as a
belt-and-braces guard before mainnet day.

## Test Evidence

Two full rehearsals run end-to-end against a fresh hardhat-local node:

**Run 1** (before transfer-ownership extension):
- ✅ MockERC20 deploy
- ✅ Audit bundle (EscrowPool + BatchSettlementRegistry + StakeBond +
      MockSignatureVerifier + cross-wires; 15/15 invariants)
- ✅ Phase 8 emission (EmissionController + CompensationDistributor +
      cross-wire; invariants)
- ✅ Phase 7-storage (StorageSlashing + KeyDistribution; invariants)

**Run 2** (with transfer-ownership extension):
- ✅ All four deploy steps green (re-run from fresh)
- ✅ 7/7 Ownable contracts transferred to stub multi-sig (account #1)
- ✅ Idempotency: 7/7 skipped on immediate re-run (no spurious txs)

## Files Changed

- `contracts/scripts/transfer-ownership.js` (new, ~290 lines)
- `scripts/rehearse-deploy.sh` (network-conditional `AUTHORIZED_VERIFIER`
  guard + transfer-ownership rehearsal step + idempotency check)
- `contracts/hardhat.config.js` (`pkAccounts()` placeholder-rejecting
  helper + 8 network-config call-sites updated)

## Hardware-Day Readiness — Honest Status

| Item | Status |
|---|---|
| Audit-bundle deploy | ✅ rehearsed |
| Phase 8 emission deploy | ✅ rehearsed |
| Phase 7-storage deploy | ✅ rehearsed |
| Cross-wire txs | ✅ rehearsed (post-deploy invariants pass) |
| `transferOwnership` to multi-sig | ✅ rehearsed (this audit) |
| Idempotency under partial-ceremony re-run | ✅ rehearsed |
| Phase 1.3 FTNSToken deploy | ⚠️ separate script, not orchestrator-bundled |
| `FOUNDATION_RESERVE_WALLET` mainnet guard | ⚠️ relies on operator vigilance |
| Verifier-contract migration | ⚠️ deferred (EOA prover for v1) |
| Hardware signer-set documentation | ⚠️ operator-side (not engineering) |
| Real-multisig (not stub) ceremony rehearsal | ⏳ blocked until 2026-05-01 |

The G1+G2+G3 fixes are the load-bearing ones — they were "ceremony-time
bug surface" before this audit and are now verified-green via end-to-end
rehearsal. G4+G5+G6 are documented operator-side flags.

## Next Steps (operator-side, not engineering)

1. **2026-05-01 morning:** initialize hardware devices, generate signers,
   deploy Foundation 2-of-3 Safe on Base.
2. **Run real-multisig rehearsal:** point `FOUNDATION_MULTISIG=` at the
   Safe address on Base Sepolia and re-run `rehearse-deploy.sh
   NETWORK=base-sepolia` end-to-end.
3. **Mainnet ceremony:** repeat against `NETWORK=base` with operator
   inputs verified, hardware signers present, and the transfer manifest
   archived for the Foundation.
