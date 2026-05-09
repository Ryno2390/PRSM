# A-08 RoyaltyDistributor v2 Redeploy Ceremony Plan

**Document identifier:** A-08-CEREMONY-PLAN-1
**Version:** 0.1 Draft
**Status:** Pre-ceremony planning. To be ratified by founder council resolution before execution. **NOT** a council resolution itself.
**Date drafted:** 2026-05-09
**Target ceremony date:** 2026-05-15 (per `PRSM_Vision.md` §13 forward-roadmap gantt; movable subject to founder availability + Sepolia rehearsal completion)
**Drafting authority:** PRSM founder

> **Companion documents:**
> - `docs/governance/A-08-recoverStranded-design.md` — the ADR this ceremony executes against. Source-truth for design rationale.
> - `contracts/contracts/RoyaltyDistributor.sol` — v2 source already on disk; this ceremony deploys the existing source to mainnet.
> - `contracts/scripts/deploy-provenance.js` — already updated for the 4-arg constructor; same script ships v2.
> - `contracts/scripts/verify-royalty-distributor-v2-deployment.js` — post-deploy assertion runner (shipped alongside this doc).

---

## 1. Purpose

The mainnet-deployed `RoyaltyDistributor` at `0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2` is the **v1** contract from the 2026-05-04 Phase 1.3 Task 8 deploy. It has no `Ownable*` surface and no `recoverStranded` recovery path for stranded donations (per A-08 design doc Problem section).

The repo's current `contracts/contracts/RoyaltyDistributor.sol` source is **v2** with three improvements bundled per `A-08-recoverStranded-design.md`:

1. `Ownable2Step` — owner = Foundation Safe; 2-step transfer flow defends against typo-on-transfer
2. `totalClaimable` accumulator — kept in lockstep with `claimable[]` writes; enables `recoverStranded` math
3. `recoverStranded(address to)` — owner-gated sweep of donations not credited to any creator/operator/network slot

This ceremony deploys v2 to mainnet, transfers ownership to Foundation Safe via Ownable2Step, and updates `prsm/config/networks.py` MAINNET to point at the v2 address.

**The ceremony does NOT migrate balances from v1 to v2.** v1 stays in place; existing claimable balances remain claimable on v1 indefinitely. New on-chain royalty distributions route through the new v2 address (operators update env / pinned address). Section §5 covers the operator-side migration.

**What this ceremony is NOT:**

- Not a Pausable (HIGH-3 / D-02) integration. That's separately scoped.
- Not a v1 distributor decommission. v1 remains live for legacy-claim purposes until balances drain naturally.
- Not a balance migration. Existing creators with claimable balances on v1 must call v1's `claim()` against the v1 address; the Foundation will not auto-migrate.
- Not the final RoyaltyDistributor architecture — future Pausable/upgradeability work may produce a v3.

---

## 2. Pre-flight checklist

Before founder begins ceremony execution:

### 2.1 Repo state

- [ ] `git status` clean on `main` at the commit being deployed.
- [ ] `contracts/contracts/RoyaltyDistributor.sol` matches the audit-prep-§7.x referenced commit hash.
- [ ] `npx hardhat compile` succeeds without warnings.
- [ ] `npx hardhat test` passes the full RoyaltyDistributor test suite (current count: 24 + 9 A-08-specific = ≥33 tests green per design doc Implementation checklist).

### 2.2 Network + RPC

- [ ] Base mainnet RPC endpoint reachable (`PRSM_BASE_RPC_URL` or default `https://mainnet.base.org`).
- [ ] Block explorer (`https://basescan.org`) reachable for source verification post-deploy.
- [ ] Etherscan-API key present in env (`BASESCAN_API_KEY`) for `hardhat verify:verify` step.

### 2.3 Hardware + signer set

- [ ] Deployer hot key (whichever address is used for deploy txs) has ≥ 0.005 ETH on Base for gas + L1 data fee buffer (per `2026-05-04-task8-deploy-ceremony-lessons.md`).
- [ ] Foundation Safe `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791` has 3-of-3 hardware-wallet signers (Ledger + Trezor + OneKey) all reachable. The `acceptOwnership` step is a Safe-multisig transaction; signers must be online.
- [ ] Multi-sig signer addresses verified per `~/.prsm-foundation-private/Multi-Sig_Addresses.txt`.

### 2.4 Sepolia rehearsal

- [ ] **Sepolia rehearsal complete** within last 7 days. Same `deploy-provenance.js` invocation against Base Sepolia, same Ownable2Step transferOwnership + acceptOwnership flow, same `verify-royalty-distributor-v2-deployment.js` assertions. Rehearsal manifest archived under `contracts/deployments/`.
- [ ] Sepolia rehearsal txs included in pre-ceremony briefing for council awareness.

### 2.5 Manifest preparation

- [ ] Canonical mainnet addresses confirmed against `prsm/config/networks.py` MAINNET block:
  - FTNS Token: `0x5276a3756C85f2E9e46f6D34386167a209aa16e5`
  - ProvenanceRegistry V2: `0xe0cedDA354f99526c7fbb9b9651e12aDB2180dbf`
  - Foundation Safe (network treasury + post-handoff owner): `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791`
- [ ] Constructor args sanity-checked one final time against `RoyaltyDistributor.sol` constructor signature: `(_ftns, _registry, _networkTreasury, _initialOwner)`.
- [ ] **`_initialOwner` set to deployer hot key** (NOT directly to Safe). Ownable2Step requires the immediate owner to call `transferOwnership(safe)`; the Safe then calls `acceptOwnership()`. Going through this 2-step is what defends against typo-on-transfer.

---

## 3. Constructor argument table

The 4 immutable args from `RoyaltyDistributor.sol` constructor:

| Arg | Type | Mainnet value | Source of truth |
|---|---|---|---|
| `_ftns` | `address` | `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` | `prsm/config/networks.py` MAINNET.ftns_token; matches `FTNS_TOKEN_ADDRESS` env var if set |
| `_registry` | `address` | `0xe0cedDA354f99526c7fbb9b9651e12aDB2180dbf` | `prsm/config/networks.py` MAINNET.provenance_registry_v2 (V2; NOT V1) |
| `_networkTreasury` | `address` | `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791` | Foundation Safe; same as final owner post-handoff |
| `_initialOwner` | `address` | *deployer hot key* | The wallet executing the deploy tx; later transferred to Safe via Ownable2Step |

**Validation guards in deploy-provenance.js** (already shipped per design doc Implementation checklist):

- chainId pin (rejects deploy if connected to non-mainnet)
- Canonical FTNS pin (rejects if FTNS_TOKEN_ADDRESS doesn't match the canonical Base mainnet address)
- treasury-is-contract pin (rejects if `_networkTreasury` resolves to an EOA — Foundation Safe is a contract)

---

## 4. Step-by-step ceremony execution

### Step 4.1 — Deploy v2 contracts

```bash
# Run from repo root with deployer hot key in env
export DEPLOYER_PRIVATE_KEY="0x..."
export FTNS_TOKEN_ADDRESS="0x5276a3756C85f2E9e46f6D34386167a209aa16e5"
export PROVENANCE_REGISTRY_ADDRESS="0xe0cedDA354f99526c7fbb9b9651e12aDB2180dbf"
export NETWORK_TREASURY="0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791"
# ROYALTY_DISTRIBUTOR_OWNER unset → defaults to deployer hot key

cd contracts
npx hardhat run scripts/deploy-provenance.js --network base
```

Output: a deploy manifest at `contracts/deployments/provenance-base-<timestamp>.json` containing:
- New v2 RoyaltyDistributor address
- Existing ProvenanceRegistry address (re-listed, not re-deployed)
- Constructor args used
- Initial owner = deployer hot key
- Verification status (Basescan source-verify result)

**Note:** the existing `deploy-provenance.js` script also re-deploys ProvenanceRegistry by default. For an A-08-only redeploy, **modify the script invocation** (or add a `--skip-registry` flag — TBD at execution time) to deploy only the RoyaltyDistributor against the existing ProvenanceRegistry. If the script doesn't currently support skipping registry, the founder must edit the deploy-provenance.js invocation manually before run-time.

### Step 4.2 — Source-verify on Basescan

The `deploy-provenance.js` script auto-runs `hardhat verify:verify` post-deploy; if that fails (rate limit / API hiccup), retry manually:

```bash
npx hardhat verify --network base \
  <NEW_DISTRIBUTOR_ADDRESS> \
  "0x5276a3756C85f2E9e46f6D34386167a209aa16e5" \
  "0xe0cedDA354f99526c7fbb9b9651e12aDB2180dbf" \
  "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791" \
  "<DEPLOYER_HOT_KEY_ADDRESS>"
```

Expected: Basescan shows source-verified green checkmark + readable contract surface.

### Step 4.3 — Run post-deploy verification

```bash
PROVENANCE_MANIFEST=contracts/deployments/provenance-base-<timestamp>.json \
  npx hardhat run scripts/verify-royalty-distributor-v2-deployment.js \
  --network base
```

Expected exit code 0. The script asserts:

- Code at v2 address is non-empty (i.e., contract actually deployed)
- `ftns()` returns canonical FTNS Token address
- `registry()` returns canonical ProvenanceRegistry V2 address
- `networkTreasury()` returns Foundation Safe address
- `owner()` returns deployer hot key (BEFORE transferOwnership)
- `pendingOwner()` is zero address
- `totalClaimable()` == 0 (fresh deploy with no royalty payments yet)

**If any assertion fails: STOP. Do not proceed to Step 4.4.** Either (a) investigate the deploy state, (b) abandon this ceremony attempt, or (c) the contract bytecode is not what was expected (compile mismatch).

### Step 4.4 — Initiate Ownable2Step transfer

From the deployer hot key, call `transferOwnership(<Safe address>)` on the new v2 distributor:

```bash
# Pseudo: actual invocation depends on chosen tooling (cast / ethers script / etc.)
cast send <NEW_DISTRIBUTOR_ADDRESS> \
  "transferOwnership(address)" \
  "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791" \
  --rpc-url https://mainnet.base.org \
  --private-key $DEPLOYER_PRIVATE_KEY
```

Verify: re-run `verify-royalty-distributor-v2-deployment.js` with the `EXPECT_PENDING_OWNER` env var set; assertion passes if `pendingOwner() == Safe address` AND `owner() == deployer hot key` (transfer initiated, not yet accepted).

### Step 4.5 — Foundation Safe acceptOwnership

This is the multi-sig transaction. Three hardware-wallet signers cosign on the Safe UI (or via Safe SDK):

1. Connect Safe at `https://app.safe.global/transactions/queue?safe=base:0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791`
2. Compose new transaction → "Contract interaction" → target = `<NEW_DISTRIBUTOR_ADDRESS>` → method = `acceptOwnership()` → no args
3. Each of 3 signers (Ledger / Trezor / OneKey) signs in sequence; threshold reached → tx executes
4. Verify on Basescan: tx receipt `OwnershipTransferred(deployer → Safe)` event emitted

### Step 4.6 — Final post-acceptOwnership verification

```bash
PROVENANCE_MANIFEST=contracts/deployments/provenance-base-<timestamp>.json \
EXPECT_FINAL_OWNER=0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791 \
  npx hardhat run scripts/verify-royalty-distributor-v2-deployment.js \
  --network base
```

Expected: all assertions pass + final owner check passes (`owner() == Foundation Safe` AND `pendingOwner() == 0x0`).

**If this final verification fails: the v2 distributor exists but is not properly owned. DO NOT update networks.py until investigated — operators pointing at a non-Safe-owned distributor would route royalties through a contract no one trusts.**

### Step 4.7 — networks.py manifest update

```python
# prsm/config/networks.py — MAINNET block
royalty_distributor="<NEW_V2_ADDRESS>",  # was: 0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2 (v1)
```

Commit + push as a separate commit on `main` after the on-chain ceremony completes. Commit message references this plan doc + the deploy manifest path.

### Step 4.8 — Audit-prep + MEMORY refresh

- Add §7.x entry to `docs/2026-04-27-cumulative-audit-prep.md` capturing the ceremony — new address, transfer txs, post-deploy verification proof, deferred-handoff items resolved
- Update memory entry `project_t10_a08_2026_05_07.md` description to mark A-08 as ceremony-complete (was: "mainnet-untouched; A-08 ships at the planned v2 redeploy ceremony")
- Tag the ceremony-completion commit `a-08-v2-redeploy-ceremony-complete-<date>`

---

## 5. Operator-side migration plan

### 5.1 Default operator path (recommended)

Operators who haven't pinned `PRSM_ROYALTY_DISTRIBUTOR_ADDRESS` use the canonical pinned address from `prsm/config/networks.py`. After Step 4.7 commit lands + operator upgrades to the new release, their node automatically uses v2. **No env-var change needed.**

### 5.2 Operators with pinned address

Operators who pinned `PRSM_ROYALTY_DISTRIBUTOR_ADDRESS` to the v1 address must update:

```bash
# Old (v1):
# export PRSM_ROYALTY_DISTRIBUTOR_ADDRESS="0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2"

# New (v2):
export PRSM_ROYALTY_DISTRIBUTOR_ADDRESS="<NEW_V2_ADDRESS>"
```

Restart the node. Verify via `prsm_balance_check` MCP tool that the new RoyaltyDistributor is wired (the source field should still report `onchain` — no behavior change at the user-facing level).

### 5.3 Existing v1 claimable balances

Creators / operators with non-zero claimable balances on v1 retain the right to claim them. The migration plan does NOT call `recoverStranded` against v1 (v1 has no such surface). Each holder calls v1's `claim()` directly against the v1 address. This is operator-driven; the Foundation does not initiate.

To check v1 claimable balance from the new release:

```python
from prsm.economy.web3.royalty_distributor import RoyaltyDistributorClient
v1_client = RoyaltyDistributorClient(
    rpc_url="https://mainnet.base.org",
    contract_address="0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2",  # v1
)
# Then call v1_client.claimable(my_address) per the v1 surface
```

### 5.4 Stranded donation recovery (post-acceptOwnership)

The point of A-08 is `recoverStranded`. If anyone has donated FTNS directly to v1 in the past, those donations are NOT recoverable on v1 (no recovery surface). They remain stranded permanently. Going forward on v2: any direct donation can be recovered by Foundation Safe via `recoverStranded(<destination>)`.

---

## 6. Rollback / abort considerations

### 6.1 If Step 4.1 fails (deploy reverts)

- Failure modes: insufficient gas, RPC hiccup, constructor revert (chainId / FTNS / treasury guard fired).
- Recovery: investigate root cause; attempt deploy again later (deploy is idempotent in the sense that a failed-deploy doesn't change on-chain state).
- No rollback needed — v1 stays the canonical pinned address.

### 6.2 If Step 4.5 fails (Safe doesn't acceptOwnership)

- Failure modes: signer unavailable, Safe queue stuck, signers reject the tx.
- Effect: v2 distributor exists with `pendingOwner == Safe`, but `owner == deployer hot key`.
- Risk: deployer hot key has full ownership rights including `recoverStranded`. **Until Safe accepts, the v2 distributor is functionally owned by the hot key.** Operators MUST NOT update env to v2 until Step 4.5 completes.
- Recovery: re-attempt acceptOwnership tx via Safe UI / SDK once signers are online. Deployer hot key can ALSO call `transferOwnership(<different address>)` if Safe is permanently unable to accept (which would be an unusual emergency).

### 6.3 If Step 4.7 (networks.py) is committed before Step 4.5 completes

- This would cause operator nodes upgrading to point at a v2 distributor not yet owned by the Safe.
- **Discipline check at execution time:** Step 4.7 ONLY runs after Step 4.6 final verification passes (which includes `EXPECT_FINAL_OWNER` check).

### 6.4 Mid-ceremony abandonment

If the ceremony is abandoned mid-flight (e.g., signers unreachable for 24h+ between Steps 4.1 and 4.5):

- v2 distributor exists on-chain, owned by deployer hot key.
- networks.py NOT updated; operators continue using v1.
- v2 has zero balance, no on-chain royalty flow → no economic impact.
- Recovery: complete Steps 4.4-4.5 when signers available; or transfer hot-key ownership to a different signer (e.g., a single-sig hot key Foundation control) and treat that as the canonical owner pending future cleanup.

### 6.5 Post-ceremony rollback

If a critical bug is discovered post-ceremony (Steps 4.1-4.8 all complete), the rollback is:

1. Revert the networks.py commit on `main`.
2. Push a release reverting the v2 pin.
3. Operators upgrading auto-revert to v1.
4. v2 distributor still exists; Foundation Safe owns it. `recoverStranded` can sweep any balance accrued during the v2 window.
5. v2 is operationally orphaned; mark deprecated in audit-prep.

This is operationally graceful — no funds at risk.

---

## 7. Honest scope (what this ceremony does NOT do)

- **HIGH-3 / D-02 Pausable integration.** Separately scoped per design doc. If HIGH-3 disposition lands before ceremony day, founder may bundle (modify the deploy script + this plan); otherwise A-08-only.
- **v1 distributor decommission.** v1 stays live indefinitely. Sufficiently old / drained v1 may eventually be marked deprecated in OPERATOR_GUIDE but no on-chain action.
- **Auto-migration of claimable balances v1 → v2.** Holders self-migrate via v1's `claim()`. Foundation does not initiate.
- **Force operator env-var update.** Operators using pinned v1 env vars continue routing to v1 until they update. Pinned-address operators see this in their logs (the wired-in v1 address); they're expected to update at their own pace post-ceremony.
- **Burn of v1 distributor / set v1 to address(0) network treasury.** v1 has no Ownable surface, so even the Foundation cannot disable it. v1 retires only by operator behavior (no one points new flows at it).
- **Smart-contract audit of v2 source.** Per `2026-05-06-resource-constrained-audit-strategy.md` (PRSM-POL-2), agent-teams self-audit + 14-day public review window applies. External audit deferred per the policy.

---

## 8. Sepolia rehearsal commitment

Before mainnet ceremony, founder commits to running the full 4.1-4.8 sequence on Base Sepolia within the 7 days preceding ceremony date. Specifically:

1. Deploy v2 distributor against a Sepolia-deployed FTNS test token + Sepolia ProvenanceRegistry test instance.
2. Initiate transferOwnership to a Sepolia test multi-sig (or single-sig test wallet acting as Safe-equivalent).
3. Accept ownership.
4. Run post-deploy verification.
5. Test `recoverStranded` end-to-end: send FTNS test tokens directly to distributor, confirm `recoverStranded` sweeps the donation.
6. Archive Sepolia rehearsal manifest under `contracts/deployments/sepolia-a08-rehearsal-<timestamp>.json` for ceremony-day reference.

If the Sepolia rehearsal surfaces any issue (script bug, deploy gotcha, ownership flow weirdness), the mainnet ceremony slips by ≥ 7 days for re-rehearsal.

---

## 9. Council ratification

This plan is **NOT** a council resolution. Before execution, founder will draft a brief PRSM-CR-2026-05-15 (or whatever ceremony date) ratifying:

1. Adoption of this plan as the canonical ceremony reference.
2. Authorization to spend deploy gas (operational; under PRSM-POL-1 §5 disbursement tier).
3. Pre-commitment to the rollback plan in §6 if the ceremony surfaces issues.
4. Post-ceremony commitment to update audit-prep + MEMORY per Step 4.8.

The CR will reference this plan by ID + commit hash.

---

## 10. Open questions

These are deliberately unresolved at planning time; founder will close them at execution time.

- **Q1.** Does `deploy-provenance.js` support skipping registry redeploy via flag? If not, founder edits the script body for the A-08-only run. Manual edit of a deploy script under time pressure is a P-something risk; the better path is to add `--royalty-only` flag to the script ahead of ceremony day. **Action item: add the flag in a pre-ceremony PR.**
- **Q2.** Should Foundation pre-fund the v2 distributor with a small FTNS amount to test `recoverStranded` immediately post-ceremony, or wait for an organic donation? Default proposal: **don't pre-fund** — natural rollout exercise the path organically.
- **Q3.** Should the ceremony day include a Pausable integration check (HIGH-3), or strictly A-08-only? Default proposal: A-08-only this ceremony; HIGH-3 in a future v3 deploy.

---

**End of plan.**
