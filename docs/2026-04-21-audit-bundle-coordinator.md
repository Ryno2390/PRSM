# PRSM Mainnet Audit — Bundle Coordinator

**Date:** 2026-04-27 (refreshed for cumulative tag)
**Bundle tag reference:** `cumulative-audit-prep-20260427` — extends `phase7.1x-audit-prep-20260422-2` with five additional phases (3.x.2 / 3.x.3 / 3.x.4 / 3.x.5 + Phase 4 Task 3).
**Engagement model:** single auditor, one remediation cycle, one Base mainnet deploy ceremony.

This is the first document an external auditor should read. It frames the full audit tree as one coherent engagement, points to the per-phase scope bundles that drill into each surface, and surfaces the cross-phase seams that an auditor should focus on first.

> **📌 For auditors starting here:** the most recent audit-prep bundle is **`docs/2026-04-27-cumulative-audit-prep.md`** — it stacks on top of the 2026-04-22 baseline with the new in-scope surface (one new Solidity contract, four new Python phases, one new HTTP API + JS SDK helper). All round-1 review HIGH/MEDIUM findings across the new phases are **RESOLVED pre-audit**. The 2026-04-22 baseline (`docs/2026-04-22-phase7.1x-audit-prep.md`) remains authoritative for the 7/7.1/7.1x economic substrate; this refresh ADDS to it rather than replacing.

---

## 1. What's being audited

Three phases ship the core PRSM economic layer and must be reviewed together because their challenge / slash paths are tightly coupled:

| Phase | Tier | Merge-ready tag | Surface |
|-------|------|-----------------|---------|
| **Phase 3.1** | A (receipt-only) | `phase3.1-merge-ready-20260421` | `BatchSettlementRegistry` + `EscrowPool` + challenge handlers (DOUBLE_SPEND / INVALID_SIGNATURE / NO_ESCROW / EXPIRED) |
| **Phase 7** | C (stake + slash) | `phase7-merge-ready-20260421` | `StakeBond.sol` (new) + registry slash-hook extension + Python StakeManager + marketplace on-chain tier gate |
| **Phase 7.1** | B (redundant execution) | `phase7.1-merge-ready-20260421` | `CONSENSUS_MISMATCH` reason code + cross-batch handler + Python MultiDispatcher / ShardConsensus + orchestrator consensus routing |

**Why bundled:**
- Phase 7's `stakeBond.slash(...)` call sits inside Phase 3.1's `challengeReceipt` flow.
- Phase 7.1 adds one more `ReasonCode` branch to that same `challengeReceipt` dispatch chain.
- All three phases share the same `ReceiptLeaf` struct, the same batched-settlement Merkle-tree machinery, and the same slash economics (70/30 challenger/Foundation split with self-slash 100%-to-Foundation).

Splitting the audits would risk seam-crossing findings falling between engagements. One auditor, one remediation cycle, one deploy.

---

## 2. Per-phase scope bundles (read in order)

1. **`docs/2026-04-21-phase7.1-audit-prep.md`** — most recent; covers the CONSENSUS_MISMATCH extension.
2. **`docs/2026-04-21-phase7-audit-prep.md`** — covers StakeBond + slash hook into Phase 3.1.
3. **Phase 3.1 surface** — covered by the Phase 7 bundle's §1.2 "out of scope" (Phase 3.1 was already merge-ready and forms the unchanged substrate Phase 7 and 7.1 build on).

Each per-phase bundle contains: in/out-of-scope, commit range, threat model, known issues with auditor prompts, engagement plan.

---

## 3. Cross-phase seams (auditor priority zones)

### 3.1 `challengeReceipt` dispatch chain

**Location:** `contracts/contracts/BatchSettlementRegistry.sol` lines ~369-445.

```
Phase 3.1: dispatch to 4 handlers + slash skipped
Phase 7:   extend slash-hook allowlist: DOUBLE_SPEND + INVALID_SIGNATURE
Phase 7.1: append CONSENSUS_MISMATCH to enum (value=5), add cross-batch
           handler, extend slash-hook allowlist
```

An auditor should read this function end-to-end once and verify:
- Enum ordering is stable across the three phases (no re-ordering).
- The slash-hook's reason allowlist exactly matches the set of handlers that represent provable misbehavior (not griefing, not hygiene).
- The `try/catch` around `stakeBond.slash` behaves correctly under every handler's return path.

### 3.2 `ReceiptLeaf` canonical form

The struct is defined in Phase 3.1 and consumed by all three phases:
- Phase 3.1's Merkle tree commits leaves.
- Phase 7's slash hook passes the batchId (not the leaf) into StakeBond; no leaf manipulation.
- Phase 7.1's handler decodes a second `ReceiptLeaf` from auxData and verifies its hash matches a proof against a different batch's root.

Please confirm `abi.encode(ReceiptLeaf)` encoding is deterministic across calldata and memory locations, and that the Python-side `_leaf_hash` helper in `tests/integration/test_phase7_1_consensus_e2e.py` produces byte-identical output.

### 3.3 Slash economics under three reason codes

`StakeBond.slash(provider, challenger, reasonId)` is unchanged since Phase 7 Task 2. Three reason codes now route through it:

| Reason | Slash target | Challenger | Self-slash guard |
|--------|--------------|------------|------------------|
| DOUBLE_SPEND | `b.provider` | caller of `challengeReceipt` | triggers if `caller == b.provider` → 100% Foundation |
| INVALID_SIGNATURE | `b.provider` | caller of `challengeReceipt` | triggers if `caller == b.provider` → 100% Foundation |
| CONSENSUS_MISMATCH | `b.provider` (= minority's committer) | `b.requester` (MVP auth) | triggers if `b.requester == b.provider` → 100% Foundation |

The sybil-requester griefing vector (Phase 7.1 audit-prep §5.2) bypasses the self-slash guard because the attacker uses TWO distinct EOAs. The design argument is that the 30% Foundation skim makes it negative-EV; please verify the math holds across realistic gas + stake + FTNS-price parameter ranges per PRSM-ECON-WP-1.

### 3.4 `tier_slash_rate_bps` snapshot invariant

Phase 7 introduced a two-field snapshot chain:
1. `Stake.tier_slash_rate_bps` — snapshotted at `bond()` time into the provider's stake record.
2. `Batch.tier_slash_rate_bps` — snapshotted at `commitBatch()` time into each batch.

The slash amount is computed from **Stake.tier_slash_rate_bps**, not Batch.tier_slash_rate_bps. The batch field is used only as the slash-hook's "is this batch slashable at all?" predicate (`> 0`). Phase 7.1 preserves this exactly — all three reason codes that slash (DOUBLE_SPEND / INVALID_SIGNATURE / CONSENSUS_MISMATCH) reach the same hook and the same `StakeBond.slash` that reads from `stakes[provider]`.

Please confirm a provider cannot dodge slash by:
- Re-bonding at a lower tier before a challenge lands (blocked by `AlreadyBonded` check in `bond()`).
- Unbonding mid-challenge (blocked by the UNBONDING status still being slashable per Phase 7 design §3.1).
- Withdrawing mid-challenge (blocked by unbond-delay timer + WITHDRAWN status not slashable, so race is provider-loses).

---

## 4. Consolidated test inventory

```bash
# Solidity (hardhat)
cd contracts && npx hardhat test \
  test/StakeBond.test.js \
  test/BatchSettlementRegistry.test.js \
  test/BatchSettlementChallenge.test.js \
  test/BatchSettlementSlashing.test.js \
  test/BatchSettlementConsensus.test.js \
  test/BatchSettlementGasFloor.test.js

# Python unit (excludes conftest-brittle legacy tests)
.venv/bin/python -m pytest \
  tests/contracts/ \
  tests/unit/test_reputation_tracker.py \
  tests/unit/test_marketplace_orchestrator.py \
  tests/unit/test_orchestrator_consensus.py \
  tests/unit/test_shard_consensus.py \
  tests/unit/test_multi_dispatcher.py \
  tests/unit/test_phase7_1_reason_code.py \
  tests/unit/test_phase7_1_reputation_observability.py

# Python E2E (live hardhat)
.venv/bin/python -m pytest \
  tests/integration/test_phase7_stake_slash_e2e.py \
  tests/integration/test_phase7_1_consensus_e2e.py -v
```

Expected at the current tip: **427 passing total** (142 Solidity + 283 Python unit + 2 Python E2E).

---

## 5. Consolidated known-issues list (across all three bundles)

From the three per-phase review gates. All are non-blockers for tag but tracked through the audit:

### From Phase 7 (Task 8 review)

- **§8.7 — Challenge-tx gas floor.** ✅ RESOLVED PRE-AUDIT. Landed contract-level `MIN_SLASH_GAS = 150_000` with `require(gasleft() >= MIN_SLASH_GAS)` before the `stakeBond.slash` try/catch. Preserves best-effort semantics for legitimate slash-ineligibility while excluding the OOG path. 7 new tests in `BatchSettlementGasFloor.test.js` pin the behavior.
- **§8.8 — Cross-process nonce-race on shared provider keys.** Not a regression from Phase 1.1; Phase 7 widens exposure. Operator runbook invariant now in `OPERATOR_GUIDE.md` §On-chain Keypairs.

### From Phase 7.1 (Task 8 review)

- **§8.6 — `consensus_minority_queue` persistence.** ✅ RESOLVED pre-audit via three shipped artifacts: (1) `ConsensusChallengeSubmitter` service (`prsm/marketplace/consensus_submitter.py`), (2) SQLite-backed `ConsensusChallengeQueue` (`prsm/marketplace/consensus_queue.py`) with PENDING → SUBMITTABLE → SUBMITTED/FAILED lifecycle + crash-safety test, (3) `process_submittable_queue` runner with outcome-based retry classification. Remaining follow-ups (exponential backoff, multi-process claim leases, `BatchCommitted` event watcher) are Phase 7.1x.next+ operational polish, not contract-security.
- **§8.7 — Sybil-requester griefing vector.** ✅ RESOLVED PRE-AUDIT via `consensus_group_id` binding in `Batch` struct + `_handleConsensusMismatch`. Both batches must share a non-zero `consensus_group_id` AND be committed by different providers, multiplying the attacker cost from `1×` stake burn to `k×`. As a bonus, the requester-only auth is replaced by open third-party challenger (closes §8.5 too). 3 new tests pin the behavior.
- **§8.8 — asyncio timeout/cancel propagation.** ✅ VERIFIED + FIXED PRE-AUDIT. Full grep pass completed; `asyncio.TimeoutError` is wrapped by `_dispatch_once`'s retry loop into `ShardDispatchError`; `asyncio.CancelledError` correctly propagates (outer-loop cancellation). One real gap found and fixed: `PeerNotConnectedError` now classified as partial-response in `MultiShardDispatcher` rather than aborting the gather.

---

## 6. Hardware-gated prerequisites

Shared across all three phases — these all fire once together, not per-phase:

- [ ] Foundation 2-of-3 multi-sig hardware wallets configured (Ledger + Trezor + Keystone per PRSM-GOV-1 §8).
- [ ] Treasury funded on Base mainnet with FTNS + USDC for auditor retainer.
- [ ] Auditor selected and engagement letter signed.

---

## 7. Deploy ceremony (post-audit)

Single ceremony covering all three phases. Follows the Phase 1.3 Task 8 multi-sig action plan (at `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/My Vault/Agent-Shared/Multi-Sig_Action_Plan.md`).

Key deploy steps in order:
1. Deploy FTNSToken (Phase 1.1 — already live on mainnet if Phase 1.3 Task 8 shipped; skip if so).
2. Deploy EscrowPool (Phase 3.1).
3. Deploy BatchSettlementRegistry (Phase 3.1 — includes Phase 7 + 7.1 extensions baked in; single deploy).
4. Deploy production Ed25519 signature verifier (replaces `MockSignatureVerifier`).
5. Deploy StakeBond (Phase 7).
6. Cross-wire: `setEscrowPool`, `setSignatureVerifier`, `setStakeBond`, `setSlasher`, `setFoundationReserveWallet`, `setSettlementRegistry`.
7. Verify all contracts on Basescan.
8. Freeze-tag the deployed-artifacts commit: `mainnet-20260XXX`.

---

## 8. Contact

- **Engineering lead:** Ryne Schultz (schultzryne@gmail.com).
- **Audit coordination:** TBD post-multi-sig setup.
- **Security disclosure:** security@prsm.ai.

For audit questions on specific design decisions, each per-phase bundle's §8 (Open issues) captures the rationale trail. For cross-phase questions, this coordinator is the integration map.
