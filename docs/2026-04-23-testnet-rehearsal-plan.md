# Mainnet Deploy Rehearsal

**Date:** 2026-04-23
**Owner:** engineering lead
**Status:** rehearsal infra landed; local hardhat dry-run green

This document turns mainnet deploy day from a live-fire ceremony into a
rehearsed, mechanical sequence. Every contract, every constructor arg,
every cross-wire call, and every post-deploy invariant has already been
exercised end-to-end against a local hardhat node. The hardware multi-sig
ceremony is the *only* remaining variable.

---

## 1. What this replaces

Before this sprint, mainnet deploy would have been a live-fire ceremony:

- Operator hand-types constructor args into `--args` flags under time pressure.
- Cross-wire calls happen one-by-one with no transactional rollback.
- Invariant verification is manual (eyeballing Basescan reads).
- A typo on an immutable constructor field → permanent loss, re-deploy only fix.

Now the three deploy scripts — `deploy-audit-bundle.js`,
`deploy-phase8-emission.js`, `deploy-phase7-storage.js` — each do:

1. **Preflight validation** — checksum, contract-at-address, balance non-zero,
   chain-id logged, mainnet-only guards (no mock verifier on mainnet,
   foundation-wallet ≠ deployer on mainnet).
2. **Ordered deploy** — every contract in the right sequence, reading deployed
   addresses into the next constructor.
3. **Cross-wire** — every setter call required by the audit-bundle coordinator
   §7, with the tx hash captured.
4. **Invariant check** — reads every cross-wired field back from chain and
   diffs against the expected address. Script exits non-zero on mismatch.
5. **Manifest emission** — every deployed address + every tx hash +
   every parameter saved to `contracts/deployments/<bundle>-<network>-<ts>.json`
   for post-deploy verification, audit-firm handoff, and Basescan URL emission.

---

## 2. Contracts covered

| Deploy script | Bundle | Contracts |
|---|---|---|
| `deploy-audit-bundle.js` | Phase 3.1 + 7 + 7.1 audit scope | EscrowPool, BatchSettlementRegistry, production Ed25519Verifier (MockSignatureVerifier only if USE_MOCK_VERIFIER=1 on non-mainnet), StakeBond |
| `deploy-phase8-emission.js` | Phase 8 emission layer | EmissionController, CompensationDistributor |
| `deploy-phase7-storage.js` | Phase 7-storage | StorageSlashing, KeyDistribution |

Plus the existing `deploy-provenance.js` (Phase 1.3) and
`deploy-mock-ftns.js` (test FTNS on non-mainnet).

**Total on mainnet day:** 9 contract deploys + 7 cross-wire txs + ~4 follow-up
governance txs (MINTER_ROLE grant, royalty-verifier wiring, etc.).

---

## 3. Cross-wire matrix (from audit-bundle coordinator §7)

The six critical setter calls the `deploy-audit-bundle.js` script makes in order:

| # | Caller | Target | Reason |
|---|---|---|---|
| 1 | EscrowPool.setSettlementRegistry | registry | requester settlement path |
| 2 | Registry.setEscrowPool | escrow | challenge→escrow refund path |
| 3 | Registry.setSignatureVerifier | verifier | batch-commit signature check |
| 4 | Registry.setStakeBond | stakeBond | slash hook on valid challenge |
| 5 | StakeBond.setSlasher | registry | authorizes registry as slasher |
| 6 | StakeBond.setFoundationReserveWallet | foundation | 30% slash-share destination |

Plus `EmissionController.setAuthorizedDistributor` from Phase 8.

Ordering matters: setting the slasher on StakeBond before the registry exists
would require a second tx to rotate; the script's sequence avoids all retries.

---

## 4. Rehearsal result (2026-04-23)

Full end-to-end rehearsal against local hardhat passed:

```
[0/5] hardhat node up at http://127.0.0.1:8545
[1/5] MockERC20 MFTNS                         ✅
[2/5] audit bundle (4 contracts + 6 wires)    ✅ (15/15 invariants)
[3/5] Phase 8 emission (2 contracts + 1 wire) ✅
[4/5] Phase 7-storage (2 contracts)           ✅
[5/5] Rehearsal complete.
```

Typo caught + fixed during rehearsal: pool-address EIP-55 checksum in
`rehearse-deploy.sh` defaults. This is exactly the class of typo the
rehearsal exists to catch before mainnet.

No signatures required from a multi-sig, no mainnet gas burned, no
external-audit clock consumed. Full green on a laptop in ~20 seconds.

---

## 5. Hardware-day runbook

Once the Foundation 2-of-3 multi-sig is provisioned (Ledger + Trezor +
Keystone — see Multi-Sig Action Plan):

### Step 1 — Base Sepolia hot rehearsal
```bash
NETWORK=base-sepolia \
FTNS_TOKEN_ADDRESS=0xd979c096BE297F4C3a85175774Bc38C22b95E6a4 \
FOUNDATION_RESERVE_WALLET=<multi-sig-address> \
PRIVATE_KEY=<deployer-hot-key-with-testnet-eth> \
./scripts/rehearse-deploy.sh
```

This is the last dress rehearsal before mainnet. Run it at least once with
the actual multi-sig address configured as `FOUNDATION_RESERVE_WALLET`.

### Step 2 — Audit-firm review against Base Sepolia manifests
The auditor's review tree references the `audit-bundle-base-sepolia-*.json`
manifest as "this is the exact bytecode+wiring we deployed to mainnet."

### Step 3 — Base mainnet deploy (multi-sig-controlled)
```bash
NETWORK=base \
FTNS_TOKEN_ADDRESS=0x5276a3756C85f2E9e46f6D34386167a209aa16e5 \
FOUNDATION_RESERVE_WALLET=<multi-sig-address> \
SIGNATURE_VERIFIER_ADDRESS=<production-ed25519-verifier> \
AUTO_VERIFY=1 \
./scripts/rehearse-deploy.sh
```

Every contract deploy + cross-wire here is signed by the multi-sig; each
needs 2-of-3 signatures. Expect 9 + 7 + 1 = 17 signatures over the ceremony.

### Step 4 — Freeze tag
```bash
git tag mainnet-20260XXX && git push origin mainnet-20260XXX
```

---

## 6. What remains hardware-gated

These items can only execute once the 2-of-3 Foundation multi-sig exists
and `FOUNDATION_RESERVE_WALLET` / `PRIVATE_KEY` point to real hardware:

- Phase 1.3 Task 8 (FTNSToken + Provenance + Royalty mainnet deploy).
- Audit-bundle mainnet deploy (the scripts in this doc).
- Phase 8 mainnet deploy + MINTER_ROLE grant to EmissionController.
- Phase 7-storage mainnet deploy.
- Phase 6 Task 2 (signed bootstrap list — requires Foundation signing key).

Everything else is unblocked and mechanical given `./scripts/rehearse-deploy.sh`
returns 0 on Base Sepolia.

---

## 7. Follow-ups not yet automated

These are deliberately out-of-scope for the rehearsal to keep the script
idempotent; each is a single governance tx that runs once post-deploy:

- **FTNSToken.grantRole(MINTER_ROLE, EmissionController)** — requires Foundation
  multi-sig that owns FTNSToken. Single tx; not retry-safe in a script
  (role grants are cumulative, so running twice has no effect, but running
  against the wrong controller address is permanent).
- **KeyDistribution.setRoyaltyVerifier(RoyaltyDistributor)** — single tx.
- **Registry.setSettlementLookbackWindow / setChallengeWindowSeconds** —
  only needed if post-deploy governance wants to tune from defaults.

Document these in the Multi-Sig Action Plan as "step 5: post-deploy governance"
once the ceremony skeleton is decided.

---

## 8. File inventory

New this sprint (`phase-rehearsal-20260423`):

- `contracts/scripts/deploy-audit-bundle.js` — Phase 3.1 + 7 + 7.1 bundled deploy.
- `contracts/scripts/deploy-phase8-emission.js` — EmissionController + CompensationDistributor.
- `contracts/scripts/deploy-phase7-storage.js` — StorageSlashing + KeyDistribution.
- `scripts/rehearse-deploy.sh` — orchestrator; spins up hardhat node, runs all four scripts in order, captures manifests.
- `contracts/deployments/.gitignore` — excludes `*localhost*.json` rehearsal artifacts.
- `docs/2026-04-23-testnet-rehearsal-plan.md` — this document.

Unchanged but referenced:

- `contracts/scripts/deploy-provenance.js` — Phase 1.3 (Sepolia bake-in already complete).
- `contracts/scripts/deploy-mock-ftns.js` — testnet FTNS stand-in.
- `docs/2026-04-21-audit-bundle-coordinator.md` — §7 deploy ceremony steps.
