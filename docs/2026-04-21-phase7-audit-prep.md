# Phase 7 Audit-Prep Bundle

**Date:** 2026-04-21
**Audit tag:** `phase7-audit-prep-20260421`
**Merge-ready tag:** `phase7-merge-ready-20260421` (identical tree; audit-prep = the state we want auditors to review)
**Prior merge-ready tag this builds on:** `phase3.1-merge-ready-20260421`

This document is the first artifact an external auditor should read. It frames the scope, lists what changed vs the already-audited baseline, surfaces the two known issues we want the auditor to opine on, and points to the bundled-engagement plan with Phase 3.1.

---

## 1. Scope

### 1.1 In scope

Phase 7 introduces on-chain stake-and-slash for providers in the PRSM compute marketplace (Tier C verification — single-provider execution with slashable stake). The audit surface:

**Solidity (new):**
- `contracts/contracts/StakeBond.sol` — provider stake lifecycle (bond / requestUnbond / withdraw) + slash / claimBounty / drainFoundationReserve.

**Solidity (modified):**
- `contracts/contracts/BatchSettlementRegistry.sol` — extended `commitBatch` signature with `uint16 tierSlashRateBps`; added `setStakeBond` governance entry; added slash-hook in `challengeReceipt` success path for `DOUBLE_SPEND` and `INVALID_SIGNATURE` reason codes (try/catch best-effort).

**Python (new):**
- `prsm/economy/web3/stake_manager.py` — sync Web3 wrapper around StakeBond.

**Python (modified):**
- `prsm/marketplace/orchestrator.py` — on-chain tier gate before price handshake.
- `prsm/marketplace/reputation.py` — SlashEvent stream + `record_slash` + SLASH_WEIGHT=100 into score_for.

### 1.2 Out of scope (unchanged since prior audits)

- `contracts/contracts/FTNSToken.sol` (Phase 1.1).
- `contracts/contracts/ProvenanceRegistry.sol`, `RoyaltyDistributor.sol` (Phase 1.1).
- `contracts/contracts/EscrowPool.sol` (Phase 3.1).
- `contracts/contracts/BatchSettlementRegistry.sol` functions OTHER than `commitBatch` / `challengeReceipt` / `setStakeBond` — those are the Phase 7 touch-points; `finalizeBatch`, `voidBatch`, `setEscrowPool`, `setSignatureVerifier`, `setChallengeWindow` are unchanged from `phase3.1-merge-ready-20260421`.
- ISignatureVerifier interface + `MockSignatureVerifier.sol` (Phase 3.1).
- Phase 7.1 redundant-execution verification (Tier B) — **not yet built.**

### 1.3 Excluded by design

- **`slash-rate governance`** — the 50% / 100% per-tier slash rates are hard-coded in the StakeBond contract. PRSM-GOV-1 §13.2 designates these as supermajority-governance-amendable; an on-chain governance interface for them is Phase 7.x work (see §8.4 of the Phase 7 design doc). Auditor: treat current rates as the spec.
- **Tier thresholds** (5K / 25K / 50K FTNS) — same disposition. Hard-coded, governance-amendable via Phase 7.x.
- **Ed25519 production verifier** — the `INVALID_SIGNATURE` slash path depends on `signatureVerifier.verify()`. Phase 3.1 ships a `MockSignatureVerifier` for tests; production substitution of an audited Ed25519 library is a mainnet-deploy step, NOT an amendment to the BatchSettlementRegistry. Covered in §8.6 of the Phase 7 design doc.

---

## 2. Design documents the auditor must read first

In order:

1. `docs/2026-04-21-phase7-staking-slashing-design.md` — Phase 7 design + TDD plan. **Essential.** §3 (Protocol) + §4 (Data model) + §8 (Open issues) are load-bearing.
2. `docs/2026-04-20-phase3-marketplace-design.md` — Phase 3 marketplace context (DispatchPolicy, EligibilityFilter, ReputationTracker pre-Phase-7).
3. `docs/2026-04-21-phase3.1-batch-settlement-design.md` — Phase 3.1 batched settlement (the slashing hook lives in Phase 3.1's challenge path).

Supporting references:

- **PRSM-GOV-1** (governance white paper) — §13.2 flags slash rates as supermajority-amendable.
- **PRSM-TOK-1** (tokenomics white paper) — stake sizing rationale (tier thresholds vs expected receipt value).

---

## 3. Commit range

The audit should cover exactly this range:

```
git log phase3.1-merge-ready-20260421..phase7-audit-prep-20260421 -- \
  contracts/contracts/StakeBond.sol \
  contracts/contracts/BatchSettlementRegistry.sol \
  prsm/economy/web3/stake_manager.py \
  prsm/marketplace/orchestrator.py \
  prsm/marketplace/reputation.py \
  tests/integration/test_phase7_stake_slash_e2e.py \
  contracts/test/StakeBond.test.js \
  contracts/test/BatchSettlementSlashing.test.js
```

Task-by-task commit mapping (chronological):

| Task | Commit | Content |
|------|--------|---------|
| 1 | `c1886c4` | StakeBond.sol — bond / requestUnbond / withdraw lifecycle |
| 2 | `7855c9b` | StakeBond.sol — slash + bounty accrual + claimBounty + drainFoundationReserve |
| 3 | `eb50153` | BatchSettlementRegistry.sol — tier_slash_rate_bps + setStakeBond + slash hook |
| 4 | `693b99c` | prsm/economy/web3/stake_manager.py (new) |
| 5 | `5918fd4` | prsm/marketplace/orchestrator.py — on-chain tier gate |
| 6 | `3e2e19e` | prsm/marketplace/reputation.py — SlashEvent + record_slash |
| 7 | `3ea59c3` | tests/integration/test_phase7_stake_slash_e2e.py — E2E hardhat |
| 8 | `ce1122e` | docs/2026-04-21-phase7-staking-slashing-design.md — status + review findings |

Test footprint at the audit tag: **231 green** (122 Solidity + 109 Python).

---

## 4. Threat model (auditor priorities)

Listed in priority order. Item 4.1 is the highest-stakes review question.

### 4.1 Economic: bounty-split integrity

`StakeBond.slash()` splits the slashed FTNS between the challenger (70%) and a Foundation reserve pool (30%), with a self-slash override to 100% Foundation. The reviewer flagged §8.3 in the design doc as the primary collusion vector.

**Specific questions:**
- Can a challenger who is NOT the same EOA as the provider but IS controlled by the same principal (via a separate wallet) net-profit? Expected answer: no — 30% skim to Foundation makes the attack strictly unprofitable. Please confirm.
- Can the challenger repeatedly slash the same provider? `slash()` relies on `stakes[provider].status == BONDED || UNBONDING` and `s.amount > 0`. A fully-drained stake cannot be slashed again. Please confirm the state-machine invariants hold under adversarial re-bond timing.
- Lazy-transfer pattern: FTNS does not move during slash — it accrues to `slashedBountyPayable[challenger]` and `foundationReserveBalance`. Claim happens via `claimBounty()` (pull, zero-before-transfer). Please verify this is safe against re-entrancy (constructor sets `nonReentrant`; OZ ReentrancyGuard).

### 4.2 Challenge path correctness

The registry's `challengeReceipt` invokes `stakeBond.slash(...)` only on `DOUBLE_SPEND` or `INVALID_SIGNATURE`. `NO_ESCROW` and `EXPIRED` deliberately do not slash (rationale: NO_ESCROW is requester-attested and vulnerable to griefing; EXPIRED is protocol hygiene not misbehavior).

**Specific questions:**
- Is the `try/catch` around `stakeBond.slash` the right posture, or should it revert? See §5.1 below — the reviewer has flagged this as a known issue.
- Can a challenger submit `DOUBLE_SPEND` against a provider who never bonded? Expected answer: yes, the challenge still succeeds (invalidates receipt) but the slash silently skips via the `address(stakeBond) != address(0)` guard and `stakes[provider].amount > 0` check. Please confirm this is the intended surface.
- Provider tier dodge: the slash rate is snapshotted at `bond()` time into `Stake.tier_slash_rate_bps`. A provider who later commits batches at a lower `tierSlashRateBps` cannot reduce their actual slash exposure. `BatchSettlementSlashing.test.js:153` covers this explicitly. Please verify the snapshot invariant under unbond-and-rebond sequences.

### 4.3 Access control

- `StakeBond.slasher` is set via `setSlasher(address)` (onlyOwner). In production, the slasher is the BatchSettlementRegistry's address. `slash()` reverts with `CallerNotSlasher` if called from any other address. Please verify no bypass via `delegatecall`, proxy, or impersonation.
- `StakeBond.owner` holds the keys to `setSlasher`, `setUnbondDelay`, `setFoundationReserveWallet`, `drainFoundationReserve`. At mainnet, this is a 2-of-3 Foundation multi-sig (PRSM-GOV-1 §8). Please verify `transferOwnership` is the standard OpenZeppelin Ownable pattern with no Phase 7 customizations.

### 4.4 Arithmetic

- `slashAmount = (s.amount * s.tier_slash_rate_bps) / 10000`. `s.amount` is `uint128`, rate is `uint16` capped at 10000. Intermediate multiplication widens to `uint256`. Cast back to `uint128` via `s.amount -= uint128(slashAmount)` after the defensive clamp `if (slashAmount > s.amount) slashAmount = s.amount`. Please confirm no overflow paths.
- `challengerShare = (slashAmount * 7000) / 10000`; `foundationShare = slashAmount - challengerShare`. Both fit in `uint256`; subtraction can't underflow since `challengerShare <= slashAmount` by construction. Please verify.

### 4.5 Storage layout

Phase 7 only adds one field to the `Batch` struct in BatchSettlementRegistry: `uint16 tier_slash_rate_bps`. Placement is after the existing `BatchStatus status` (uint8) to preserve slot-packing (both fit in the same 32-byte slot as `status` + `tier_slash_rate_bps`, with padding). **Pre-existing mainnet deployments of BatchSettlementRegistry DO NOT exist** (Phase 3.1 is merge-ready but not yet mainnet-deployed), so storage-upgrade concerns do not apply — the auditor may treat this as a greenfield contract on deploy.

---

## 5. Known issues (carried from the Phase 7 Task 8 review gate)

The independent code-reviewer agent ran on `phase7-merge-ready-20260421` and returned **SAFE TO MERGE** with two non-blocker follow-ups. Both are disclosed here for auditor context; both will be addressed pre-mainnet.

### 5.1 Challenge-tx gas floor (design doc §8.7)

**Issue:** `challengeReceipt` wraps `stakeBond.slash` in `try/catch`. When the challenger under-pays gas, the outer tx succeeds (receipt invalidated, event emitted) but the nested slash silently OOGs via the catch. `eth_estimateGas` produces a too-small budget because it simulates the tx as succeeding without the slash.

**Impact:** In competitive-challenger races, a challenger who under-pays gas can burn a receipt (winning the first-to-invalidate race) while inadvertently shielding the provider from slash. This is adverse selection.

**Current mitigation:** Task 7's E2E test hard-codes a 1_000_000-gas budget. Phase 7 wallet/orchestrator integrations must do the same on production challenges.

**Preferred hardening (auditor opinion wanted):**
- Option A: `require(gasleft() >= MIN_SLASH_GAS, "InsufficientGasForSlash")` immediately before the `try stakeBond.slash(...)` block. Suggested floor: 150_000 gas (below ~200K actual cost, leaves headroom for catch path).
- Option B: drop the `try/catch` under the branch `address(stakeBond) != address(0) && tier_slash_rate_bps > 0`. Rationale: the only realistic reverts from `slash()` (`NotSlashable`, `NothingToSlash`) are legitimate failure modes the challenger should learn about, not silently succeed through.

**Auditor: which option do you recommend, or is there a better one?**

### 5.2 Cross-process nonce-race on shared provider keys (design doc §8.8)

**Issue:** `StakeManagerClient._tx_lock` serializes builds-and-sends within a single Python process. A provider running `ProvenanceRegistryClient` + `StakeManagerClient` + other Web3 wrappers from multiple processes (or multiple machines) against the same keypair can hit a nonce collision between `get_transaction_count(..., "pending")` and `send_raw_transaction`.

**Impact:** Two txs share a nonce; one reverts, the other lands. Annoying, not catastrophic. No slashable behavior.

**This is not a regression** — the pattern has shipped since Phase 1.1 (`provenance_registry.py` / `royalty_distributor.py`). Phase 7 adds a second client per keypair so the exposure factor grows by ~1.

**Current mitigation:** document "one keypair per process" as an operator invariant. See `docs/OPERATOR_GUIDE.md` §On-chain Keypairs for the runbook entry.

**Phase 7.x refinement (not gating mainnet):** shared in-process lock registry keyed on `(rpc_url, address)`.

**Auditor: no action required** unless you see a production-grade concern that the operator-invariant doc doesn't address.

---

## 6. Test inventory the auditor can re-run

All tests run locally with no mainnet dependencies.

### 6.1 Solidity (hardhat)

```bash
cd contracts
npx hardhat test \
  test/StakeBond.test.js \
  test/BatchSettlementRegistry.test.js \
  test/BatchSettlementChallenge.test.js \
  test/BatchSettlementSlashing.test.js
```

Expected: 122 passing in ~2 seconds. Gas report emitted (useful cross-check for the §5.1 MIN_SLASH_GAS estimate).

### 6.2 Python unit

```bash
.venv/bin/python -m pytest \
  tests/contracts/test_stake_manager_client.py \
  tests/unit/test_reputation_tracker.py \
  tests/unit/test_marketplace_orchestrator.py \
  -q
```

Expected: 88 passing.

### 6.3 Python E2E (live hardhat)

```bash
.venv/bin/python -m pytest \
  tests/integration/test_phase7_stake_slash_e2e.py \
  -v
```

Expected: 1 passing in ~5 seconds (boots hardhat node, compiles, deploys, drives bond → commit → challenge → slash → claim → reputation).

Requires: `npx hardhat` on PATH, hardhat `node_modules` installed. Auto-skips otherwise.

---

## 7. Engagement plan

### 7.1 Bundled engagement with Phase 3.1 Task 10 + Phase 7.1 Task 9

Per the Phase 7 design doc §6 Task 9, Phase 3.1 design doc §Task 10, and Phase 7.1 design doc §6 Task 9, the audit is **bundled across three merge-ready trees** in a single engagement. The auditor reviews:

- `phase3.1-merge-ready-20260421` (Tier A receipt-only + batched settlement)
- `phase7-merge-ready-20260421` (Tier C stake + slashing — this bundle)
- `phase7.1-merge-ready-20260421` (Tier B redundant execution)

Top-level entry point for the auditor: `docs/2026-04-21-audit-bundle-coordinator.md`. Individual per-phase bundles: this doc + `docs/2026-04-21-phase7.1-audit-prep.md`.

Rationale:
- Phase 7's slash path lives inside Phase 3.1's `challengeReceipt`; Phase 7.1 adds one more branch to that same handler. The three seams are tight — splitting the audits risks seam-crossing issues falling between engagements.
- Shared context reduces auditor ramp time → lower cost.
- One finding backlog, one remediation cycle, one deploy.

### 7.2 Hardware-gated prerequisites (before engagement kicks off)

- [ ] Foundation 2-of-3 multi-sig hardware wallets configured (Ledger + YubiHSM quorum per PRSM-GOV-1 §8).
- [ ] Treasury funded on Base mainnet with FTNS + USDC for auditor retainer.
- [ ] Auditor selected and engagement letter signed.

### 7.3 Deploy ceremony (post-audit, hardware-gated)

Follows the Phase 1.3 Task 8 / PRSM-ECON-WP-1 deploy-plan pattern. Key ceremony inputs:

1. Audit report with all findings remediated (or explicitly waived with documented rationale).
2. Multi-sig quorum physically assembled.
3. Base mainnet RPC + Etherscan verification key.
4. Deploy script: deploy MockERC20 replacement (real FTNSToken address from Phase 1.1 prod), EscrowPool, BatchSettlementRegistry, MockSignatureVerifier replacement (audited Ed25519 library), StakeBond. Cross-wire: `setStakeBond`, `setSlasher`, `setSignatureVerifier`, `setEscrowPool`, `setFoundationReserveWallet`.
5. Hardhat-verify on Etherscan for each contract.
6. Freeze-tag the deployed-artifacts commit: `phase7-mainnet-20260501` (or whatever the deploy date ends up being).

---

## 8. Contact

- **Engineering lead:** Ryne Schultz (schultzryne@gmail.com).
- **Audit coordination:** TBD post-multi-sig setup.
- **Security disclosure:** security@prsm.ai (per OPERATOR_GUIDE.md).

For audit questions on specific design decisions, the Phase 7 design doc's §8 (Open issues) captures the rationale trail. For questions on deviations from the TDD plan, the Phase 7 design doc's §6 per-task "Design deviation" callouts explain each one.
