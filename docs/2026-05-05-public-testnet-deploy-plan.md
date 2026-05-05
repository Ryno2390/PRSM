# Public Testnet Deploy Plan (Base Sepolia)

**Date:** 2026-05-05
**Author:** Founder
**Status:** Plan — pending execution
**Goal:** Make PRSM end-to-end-functional for any user **right now**, on
public testnet, without waiting for the external audit clear that gates
mainnet.

**Why now:** the audit pipeline gates mainnet deploy of the economic
substrate (EscrowPool, StakeBond, BatchSettlementRegistry, EmissionController,
CompensationDistributor, StorageSlashing, KeyDistribution). It does NOT
gate testnet. Today, a user who downloads PRSM cannot earn FTNS in any
real on-chain sense — the user-facing CLI's "earnings" are Python attribute
mutations (`prsm/user_content_manager.py:481`). Testnet deploy converts
"PRSM has merge-ready code" into "any user can run PRSM and see their
on-chain testnet-FTNS balance grow as they contribute."

**Non-goals:**
- Not a mainnet deploy. Mainnet still waits for L4 audit clear.
- Not a security claim. Testnet FTNS has no monetary value.
- Not a parameter-final claim. Halving / compensation-rate constants may
  re-tune after L7 economic audit.

---

## 1. Current state — what exists vs. what's missing

### 1.1 What's already built

| Asset | Status |
|---|---|
| Hardhat config has `base-sepolia` network (chainId 84532) | ✅ `contracts/hardhat.config.js` |
| `scripts/rehearse-deploy.sh` already supports `NETWORK=base-sepolia` mode | ✅ orchestrates MockFTNS + audit-bundle + Phase 8 emission + Phase 7-storage in one shot |
| Audit-bundle deploy script | ✅ `contracts/scripts/deploy-audit-bundle.js` |
| Phase 8 emission deploy script | ✅ `contracts/scripts/deploy-phase8-emission-stack.js` |
| Phase 7-storage deploy script | ✅ `contracts/scripts/deploy-phase7-storage.js` |
| Provenance + RoyaltyDistributor deploy script (v2 with HIGH-1 burn fix + Pausable + D-04 pull-payment) | ✅ `contracts/scripts/deploy-provenance.js` |
| MockFTNS already deployed on Base Sepolia | ✅ `contracts/deployments/mock-ftns-base-sepolia-1776008606203.json` |
| ProvenanceRegistry already deployed on Base Sepolia | ✅ `contracts/deployments/provenance-base-sepolia-1776008618090.json` |
| Verify-deployment script | ✅ `contracts/scripts/verify-audit-bundle-deployment.js` |
| `OnChainFTNSLedger` Web3 client | ✅ `prsm/economy/ftns_onchain.py` (instantiated in `prsm/node/node.py:453`) |
| `RoyaltyDistributorClient` Web3 client | ✅ `prsm/economy/web3/royalty_distributor.py` |
| Bootstrap discovery substrate | ✅ Live at `wss://bootstrap1.prsm-network.com:8765` |

### 1.2 What's missing

| Gap | Impact | Effort |
|---|---|---|
| **Audit-bundle not deployed to Sepolia** | No EscrowPool / StakeBond / Registry on testnet → no real economic settlement on testnet | 30 min (run rehearse-deploy.sh against Sepolia) |
| **Phase 8 emission stack not on Sepolia** | No halving emission active on testnet; no testnet-FTNS minting curve | 30 min (same script) |
| **Phase 7-storage not on Sepolia** | No storage-slashing economy on testnet | 30 min (same script) |
| **RoyaltyDistributor v2 not on Sepolia** | The Sepolia provenance deploy used the older code. Need to re-deploy provenance + RoyaltyDistributor with current source (post-D-04 pull-payment + Pausable). | 30 min |
| **Basescan verification of Sepolia contracts** | Without verification, users can't read the contract source on Basescan | 1 hour (auto via hardhat-verify, but per-contract manual fallback if API hiccups) |
| **Bootstrap node doesn't advertise testnet contract addresses** | Users have no way to discover which contracts to point at on testnet | 2-3 hours — bootstrap protocol extension |
| **No `prsm join-testnet` CLI command** | Users have no easy on-ramp to testnet | 2-4 hours — CLI flag + env-var preset |
| **`OnChainFTNSLedger` not wired into the user-facing earnings path** | `user_content_manager.py:481` mutates `user_profile.ftns_balance` in Python. Should call the on-chain ledger or RoyaltyDistributor instead. | 4-8 hours — refactor the earnings path |
| **No testnet-FTNS faucet** | Users can't get testnet-FTNS to test spending | 1-2 hours — simple faucet contract OR hand out from deployer |
| **No E2E smoke test against deployed Sepolia contracts** | Even if everything deploys, integration bugs invisible until users hit them | 4-6 hours — 3-node chaos harness pointed at Sepolia |
| **Public docs explaining "join testnet now"** | Without instructions, users won't try it | 2 hours — README section + quickstart |

**Total effort:** ~20-30 engineering hours over 2-4 days. Most of it is
integration / testing / docs, NOT new infrastructure.

---

## 2. Task breakdown

### Task T1 — Deploy stack to Base Sepolia (~2 hours)

**What:** Run the existing rehearsal in real-network mode.

```bash
# Pre-reqs:
# - Funded deployer account on Base Sepolia (free from sepoliafaucet.com
#   or coinbase faucet at https://www.coinbase.com/faucets/base-sepolia)
# - PRIVATE_KEY env var set to deployer key
# - BASE_SEPOLIA_RPC_URL env var set (or default to https://sepolia.base.org)
# - ETHERSCAN_API_KEY env var set for verification

# Deploy fresh MockFTNS (or reuse existing 0x...606203)
NETWORK=base-sepolia FTNS_TOKEN_ADDRESS=<existing-mock-ftns-or-skip> \
    ./scripts/rehearse-deploy.sh
```

**Outputs:**
- `contracts/deployments/audit-bundle-base-sepolia-<ts>.json`
- `contracts/deployments/phase8-emission-base-sepolia-<ts>.json`
- `contracts/deployments/phase7-storage-base-sepolia-<ts>.json`
- Re-deployed RoyaltyDistributor (current source = v2 with D-04 pull-payment)

**Acceptance:** all post-deploy invariant checks pass; manifest committed
to repo.

---

### Task T2 — Basescan verification (~1 hour)

**What:** Verify all deployed contracts on basescan.org/sepolia.

```bash
cd contracts
# Auto-verification is built into deploy-audit-bundle.js per its
# "Optional Basescan verification" branch. If anything fails, manual:
npx hardhat verify --network base-sepolia <address> <constructor-args...>
```

**Acceptance:** every Sepolia-deployed contract shows green-checkmark
"Source Code" tab on basescan.org/sepolia.

---

### Task T3 — Testnet contract registry (~2 hours)

**What:** Single source of truth file for testnet contract addresses,
loadable by the node and CLI.

**Deliverable:** `prsm/config/networks.py`:

```python
# Lookup by network name. New networks added by appending here.
NETWORK_CONFIGS = {
    "mainnet": {
        "chain_id": 8453,
        "rpc_url_default": "https://mainnet.base.org",
        "ftns_token": "0x5276a3756C85f2E9e46f6D34386167a209aa16e5",
        "provenance_registry": "0xdF470BFa9eF310B196801D5105468515d0069915",
        "royalty_distributor": "0x3E82...",   # current v1 — to be updated post-redeploy
        "foundation_safe": "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791",
        # audit-bundle contracts deployed at L4 clear:
        "escrow_pool": None,
        "stake_bond": None,
        "settlement_registry": None,
        "emission_controller": None,
        "compensation_distributor": None,
        "storage_slashing": None,
        "key_distribution": None,
    },
    "testnet": {
        "chain_id": 84532,
        "rpc_url_default": "https://sepolia.base.org",
        "ftns_token": "<from mock-ftns-base-sepolia-1776008606203.json>",
        "provenance_registry": "<from T1 redeploy>",
        "royalty_distributor": "<from T1 redeploy>",
        # Foundation Safe on testnet — separate testnet Safe OR deployer EOA:
        "foundation_safe": "<deployer EOA, or a testnet Safe>",
        "escrow_pool": "<from T1>",
        "stake_bond": "<from T1>",
        "settlement_registry": "<from T1>",
        "emission_controller": "<from T1>",
        "compensation_distributor": "<from T1>",
        "storage_slashing": "<from T1>",
        "key_distribution": "<from T1>",
    },
}
```

**Acceptance:** node startup with `--network testnet` loads testnet
addresses; mainnet is the default.

---

### Task T4 — Bootstrap node advertises testnet config (~2-3 hours)

**What:** Bootstrap discovery currently advertises mainnet contract
addresses implicitly. Extend the protocol to advertise per-network
contract bundles.

**Deliverable:** Bootstrap protocol extension — either:
- (a) Bootstrap returns BOTH mainnet + testnet contract bundles, client
  picks based on its `--network` flag, OR
- (b) Bootstrap host serves two endpoints (mainnet WSS + testnet WSS),
  client connects based on `--network`.

**Recommendation:** **(a)** — single bootstrap endpoint, network-tagged
contract bundles in the response. Simpler infrastructure, less DNS work.

**Acceptance:** `prsm node start --network testnet` connects to bootstrap,
gets testnet contract bundle, instantiates `OnChainFTNSLedger` /
`RoyaltyDistributorClient` against testnet RPC.

---

### Task T5 — `prsm join-testnet` CLI command (~2-4 hours)

**What:** One-command on-ramp for new users.

**Behavior:**
```
$ prsm join-testnet
==> Configuring PRSM for Base Sepolia testnet...
==> Generating local wallet (saved to ~/.prsm/testnet-wallet.json)...
==> Wallet address: 0xabc...
==> Requesting testnet ETH from Coinbase faucet...
==> Requesting testnet FTNS from PRSM faucet...
==> Configuration saved to ~/.prsm/testnet.env
==> Run `prsm node start --network testnet` to begin contributing.
```

**Implementation:** new subcommand in `prsm/cli.py` that:
1. Generates a fresh local wallet (eth-account)
2. Persists keystore to `~/.prsm/testnet-wallet.json`
3. Optionally calls Coinbase Sepolia faucet API for ETH
4. Optionally calls PRSM testnet-FTNS faucet endpoint
5. Writes env file: `PRSM_NETWORK=testnet`, `FTNS_WALLET_PRIVATE_KEY=...`

**Acceptance:** fresh laptop → `pip install prsm` → `prsm join-testnet`
→ `prsm node start --network testnet` → node connects, has FTNS,
visible on Basescan.

---

### Task T6 — Wire `OnChainFTNSLedger` into the earnings path (~4-8 hours)

**What:** The "I uploaded content and earned FTNS" code path in
`prsm/user_content_manager.py:481` mutates a Python attribute. Should
call the on-chain ledger.

**Current bug:**
```python
# prsm/user_content_manager.py:481
user_profile.ftns_balance += ftns_earned   # ← in-memory float, not on-chain
```

**Target:**
```python
# Pseudocode — actual flow depends on whether earnings come from
# RoyaltyDistributor (when someone pays for the content) or from
# CompensationDistributor (provider-side rewards).

if self.onchain_ledger:
    # Reads real on-chain balance instead of the in-memory float
    user_profile.ftns_balance = await self.onchain_ledger.get_balance()

# Earnings credit:
# - For data-uploads: RoyaltyDistributor credits `claimable[creator]`
#   when a payer settles for that content (via the existing pull-payment
#   path post-D-04). Display logic should read claimable + balance.
# - For compute/storage providers: CompensationDistributor credits
#   `claimable[provider]` per CompensationDistributor.distribute().
#   Display logic same.

# Display "earned today" = sum of (claimable + actual transfers in last 24h)
# rather than the locally-tracked Python float.
```

**Acceptance:** the CLI's `prsm status` and the user dashboard show the
real on-chain balance + claimable from RoyaltyDistributor and
CompensationDistributor. The Python `ftns_balance` attribute is removed
or marked deprecated for display.

**Note:** this task is the bridge between "we deployed contracts" and
"users see real economic activity." Most of the value of T1-T5 depends
on T6 actually showing earnings.

---

### Task T7 — Testnet-FTNS faucet (~1-2 hours)

**What:** Simple way for users to get testnet-FTNS once they have a
testnet wallet.

**Options:**
- **Option A (simplest):** Manual airdrop. Founder runs a one-line
  script that transfers 1000 testnet-FTNS to any address that requests
  it via a Google Form / Discord channel. Workable for first 50 users.
- **Option B (auto):** Faucet endpoint at faucet.prsm-network.com that
  accepts an address + simple captcha, sends 1000 testnet-FTNS via
  ERC20 transfer from a faucet wallet. ~1 day of dev.
- **Option C (per-account):** Make the testnet `EmissionController`
  emit a small pre-allocation per first-seen address. Most "true to
  prod" but requires contract change.

**Recommendation:** **Option A** for first 2 weeks (validates demand).
Promote to **Option B** if user count exceeds 20.

**Acceptance:** users have a documented way to get testnet-FTNS within
~1 hour of asking.

---

### Task T8 — E2E smoke test (~4-6 hours)

**What:** A script that spins up 3 testnet nodes (compute provider +
storage provider + data uploader), submits a query that pays FTNS,
and verifies all 3 providers receive testnet-FTNS via the
CompensationDistributor pull-payment flow.

**Deliverable:** `tests/integration/testnet-smoke.py` — runs against
the actual deployed Sepolia contracts.

**Acceptance:**
1. 3 nodes start, register their respective roles, post stake (where
   required).
2. Uploader submits a content item; ProvenanceRegistry record visible
   on Basescan.
3. Payer submits a query; EscrowPool deposit visible on Basescan.
4. After settlement, claimable mappings non-zero on RoyaltyDistributor
   + CompensationDistributor for the right addresses.
5. Each provider calls `claim()`; FTNS balance increases on Basescan.

This is the **ground-truth check** that PRSM works end-to-end with real
on-chain settlement. Until it passes, "PRSM is functional" is unproven.

---

### Task T9 — User-facing docs (~2 hours)

**What:** A README section + quickstart that says: "want to try PRSM?
Run these 3 commands."

**Deliverable:** `docs/QUICKSTART_TESTNET.md` covering:
1. `pip install prsm`
2. `prsm join-testnet`
3. `prsm node start --network testnet`
4. How to confirm it's working (Basescan link to your address)
5. How to upload content (`prsm upload <file>`)
6. How to submit a query (`prsm query "..."`)
7. How to claim earnings (`prsm claim`)
8. Known limitations (testnet has no monetary value; halving rates may
   change post-L7 audit; etc.)

**Cross-links:**
- README.md gets a "Join the testnet" callout box near the top.
- prsm-network.com landing page (out of repo scope) gets a CTA.

---

## 3. Sequencing

**Recommended execution order** (with dependencies):

```
Day 1 morning:    T1 (deploy)  →  T2 (verify)
Day 1 afternoon:  T3 (network config)
Day 2 morning:    T4 (bootstrap)  +  T7 (faucet) in parallel
Day 2 afternoon:  T5 (CLI command)
Day 3 morning:    T6 (earnings wiring) — biggest task, can spill to Day 4
Day 3 afternoon:  T8 (E2E smoke test) — gates user-launch
Day 4:            T9 (docs) + T8 fixes if needed
```

**Critical path:** T1 → T3 → T4 → T8. Without T8 passing, no public
launch.

**Parallelizable:** T7 (faucet) + T9 (docs) can be done by anyone
in parallel with the technical work.

---

## 4. Open decisions

These need a founder yes/no before T1 can run:

1. **Use existing Sepolia MockFTNS or redeploy?**
   - Existing: `mock-ftns-base-sepolia-1776008606203.json` from 2026-04
   - Redeploy: cleaner, gets latest source, but breaks any addresses
     already pointing at it.
   - **Recommendation:** redeploy. Nobody is using the old one yet.

2. **Foundation "Safe" address on testnet — what to use?**
   - Options: (a) deployer EOA acts as Safe; (b) deploy a testnet Safe
     using the same hardware setup as mainnet; (c) a 1-of-1 testnet
     Safe for simplicity.
   - **Recommendation:** (a) deployer EOA. Testnet has no real money;
     full Safe ceremony is overhead. Document that mainnet uses the
     real 2-of-3.

3. **Halving emission rate on testnet — same as mainnet plan, or
   accelerated for testing?**
   - Mainnet plan: halving every ~12 months.
   - Testnet accelerated: halving every ~7 days lets users see the
     curve in action.
   - **Recommendation:** accelerated (e.g., 1-day epoch, halving every
     14 days). Document clearly that mainnet uses different parameters.

4. **Public bootstrap node — same droplet handles both mainnet and
   testnet, or separate?**
   - Same: simpler, single TLS cert, single instance.
   - Separate: cleaner failure isolation.
   - **Recommendation:** same droplet, single endpoint, network-tagged
     responses (per T4 option (a)).

5. **Faucet — Option A (manual) or B (web endpoint)?**
   - A: founder time, OK for first 20 users.
   - B: ~1 day of dev, scales further.
   - **Recommendation:** A for first 2 weeks; promote to B if user
     count exceeds 20.

---

## 5. Validation plan

**Before declaring "PRSM testnet is live":**

1. T8 E2E smoke test passes against deployed Sepolia contracts.
2. Founder personally runs through `prsm join-testnet` flow on a fresh
   laptop, gets testnet-FTNS, runs a node, sees balance grow.
3. Foundation council member (or trusted external dev) repeats the
   above flow independently — catches "works on my machine" bugs.
4. Basescan verification green-checkmark on every deployed contract.
5. Forta monitoring (L10a) wired to also watch testnet contracts so
   any unexpected on-chain activity surfaces.

**Success metric for the first 2 weeks:**
- ≥ 3 external users have run `prsm join-testnet` and earned non-zero
  testnet-FTNS via real protocol activity.
- Smoke test passes once per day on cron.

---

## 6. Costs

| Item | Cost |
|---|---|
| Sepolia ETH for deployer (initial deploys + ongoing tx) | ~$0 (faucet) |
| Sepolia ETH for testnet-FTNS faucet operator | ~$0 (faucet) |
| DigitalOcean droplet for bootstrap (already running) | $0 incremental |
| TLS cert (already issued) | $0 |
| Founder time (this plan) | ~20-30 hours |
| External tester time | ~2-4 hours |

**Total cash cost: ~$0.** This is purely an engineering-time investment.

---

## 7. What this enables

**Once T1-T9 are done:**

1. The claim "PRSM is real and functional" becomes verifiable by anyone
   with a laptop and an internet connection.
2. Audit firms responding to the L3-L8 RFPs see a live testnet they can
   probe — much stronger evidence than "merge-ready code in CI."
3. Investor due diligence (Reg D 506(c) Prismatica raise) benefits from
   "demo me the protocol working" → just hand them the testnet
   quickstart.
4. Bug discovery: real users find integration bugs that hardhat-localhost
   never surfaces. Most of these are caught BEFORE mainnet, which
   means cheaper fixes.
5. Community building: discord / forum / GitHub issues populate with
   real users sharing testnet experiences. This is how decentralized
   protocols actually grow.
6. Mainnet deploy day (post-L4 clear) becomes trivially safer: same
   deploy script, same validation, just with hardware-multisig deployer
   and real funds.

---

## 8. What this does NOT enable

- **Real economic value** for users. Testnet-FTNS has zero monetary
  value. This is intentional and must be loudly stated.
- **Real security guarantees.** External audit isn't done. Tokens are
  test tokens. Don't connect mainnet wallets.
- **Final tokenomics parameters.** Halving rate, compensation rate,
  slashing penalties may change post-L7 economic audit.
- **Production-grade availability.** Single bootstrap droplet is fine
  for testnet; production needs L6f infrastructure pen-test before
  scaling.

---

## 9. Decision needed from founder

To proceed with T1, please confirm:

- [ ] Yes, deploy fresh MockFTNS on Sepolia (vs. reuse existing).
- [ ] Yes, deployer EOA acts as Foundation "Safe" on testnet.
- [ ] Yes, accelerated halving (1-day epoch, 14-day halving) on testnet.
- [ ] Yes, same droplet handles both mainnet + testnet bootstrap.
- [ ] Yes, manual faucet (Option A) for first 2 weeks.

Once all five are confirmed, T1 can run today.

---

*Companion: this plan implements the "deploy gap, not code gap"
observation from the 2026-05-05 dev-state review. References
`audits/AUDIT_PLAN.md` v1.1 for the audit pipeline this is designed
to NOT block on.*
