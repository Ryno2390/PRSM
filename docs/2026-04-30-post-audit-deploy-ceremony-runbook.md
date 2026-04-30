# PRSM Post-Audit Deploy-Ceremony Runbook

**SCOPE — read first.** This runbook covers the **post-external-audit
ceremony** that deploys the audit-bundle (Phase 3.1 + 7 + 7.1: EscrowPool,
BatchSettlementRegistry, StakeBond) + Phase 8 emission
(EmissionController, CompensationDistributor) + Phase 7-storage
(StorageSlashing, KeyDistribution) — 9 contracts, plus the 7-Ownable
`transferOwnership` handoff to the Foundation Safe and (optional) FTNS
AccessControl role handoff.

**This is NOT the runbook for tomorrow's ceremony.** Tomorrow (assuming
hardware-arrival 2026-05-01) the operator runs **Phase 1.3 Task 8**
(`deploy-provenance.js` for ProvenanceRegistry + RoyaltyDistributor with
the Foundation Safe as `NETWORK_TREASURY`). For Task 8 see
`docs/2026-04-30-phase1.3-task8-engineering-runbook.md` (engineering
companion) and the operator-side `Multi-Sig_Action_Plan.md` in the
Foundation vault. The post-audit ceremony documented HERE runs
**weeks-to-months later** after external auditors sign off on the
streaming-inference + emission + storage stack (Phase 7 Task 9 +
Phase 7.1 Task 9, both currently `in_progress`).

**Audience:** the operator running the post-audit ceremony with the
Foundation 2-of-3 hardware multi-sig signers. Chronological master
runbook stitching the 5 deploy scripts + 2 transfer scripts + multi-sig
governance follow-ups into one sequence with checkpoint pauses.

**Pre-reqs:**
- All three hardware signing devices initialized + tested.
- Foundation 2-of-3 Safe deployed on Base mainnet at known address.
- Base Sepolia full-ceremony rehearsal completed end-to-end (this
  runbook, run with `NETWORK=base-sepolia`, must produce a green
  manifest before T-0).
- **External audit sign-off** on the audit-bundle + emission + storage
  contracts. This is the load-bearing gate — the runbook below cannot
  legitimately run before audit completion.
- **Phase 1.3 Task 8 already executed.** ProvenanceRegistry +
  RoyaltyDistributor live on Base mainnet; the Foundation Safe is
  battle-tested as `NETWORK_TREASURY`.
- Funded deployer hot wallet on Base mainnet (~0.5 ETH; see §1.5 for
  gas math).

**Honest-scope notes:**
- FTNS already exists at `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` on
  Base mainnet (verified 2026-04-30: symbol=FTNS, totalSupply=100M,
  matches FTNSTokenSimple bytecode). Use `FTNS_DEPLOY_MODE=existing` +
  provide that address. The `=real` and `=mock` modes exist for
  testnet/hardhat-local rehearsal only.
- This runbook does NOT cover Provenance / Royalty / NWTN-side contracts
  — those are Phase 1.3 Task 8 (separate, earlier ceremony).
- `MINTER_ROLE → EmissionController` is intentionally a **post-handoff
  multi-sig governance tx** (not a deployer action). It cannot be
  scripted as part of this runbook because it requires the multi-sig
  signature, not the deployer's. See §6.

---

## 1. Pre-flight (T-3d → T-1d)

### 1.1 Hardware signer initialization
For each of the three hardware devices (Ledger / Trezor / Keystone or
equivalent):
1. Initialize from factory state with a fresh seed phrase.
2. Record the seed phrase on the offline backup card (no photos, no cloud).
3. Verify the receive address by typing it on the device screen, NOT
   copy-pasting from screen.
4. Store the device + backup card in three physically separated
   locations (2-of-3 quorum survives loss of any one location).

### 1.2 Foundation Safe deployment (Base mainnet)
1. Deploy a fresh Gnosis Safe at https://app.safe.global with all three
   hardware addresses as signers, threshold = 2.
2. Test with a 2-of-3 signed dust-value tx to verify the signer chain
   works end-to-end. (Skipping this step is a leading cause of
   ceremony-day "we can't sign anything" panics.)
3. Record the Safe address; this is `FOUNDATION_MULTISIG` for every
   downstream step.

### 1.3 Foundation pool-sink addresses
The CompensationDistributor takes three pool sinks as constructor args.
Decide whether each is:
- A separate Safe (recommended — finer-grained governance).
- The Foundation main Safe (simpler — one signer set governs all flows).
- An off-chain custodied EOA the Foundation controls.

Document the decision per pool. Set `CREATOR_POOL` / `OPERATOR_POOL` /
`GRANT_POOL` env vars accordingly. (Orchestrator hard-fails if any are
unset on `NETWORK=base`.)

### 1.4 Verifier EOA
`AUTHORIZED_VERIFIER` is the off-chain prover EOA that signs storage
proofs for `StorageSlashing`. This is NOT the Foundation Safe; it's
the operational hot wallet running the prover service. Generate a
dedicated key for this purpose, not a re-use of the deployer key.

### 1.5 Deployer hot wallet
- Generate a fresh deployer private key. Do NOT re-use any other key.
- Fund with ~0.5 ETH on Base mainnet (typical mainnet ceremony costs
  ~0.15-0.25 ETH at typical gas; ~0.5 ETH gives 2x headroom for retries).
- Record the deployer address. The deployer holds 7 Ownable contracts
  + 4 FTNS roles between cross-wire and handoff steps. Treat as
  load-bearing for ~30-90 minutes.
- After §5 + §6 complete, the deployer key has zero on-chain authority.
  It can be destroyed (or kept as audit-trail evidence).

### 1.6 Env-var matrix (mainnet)
Set these in a `.env.mainnet` file (do NOT commit):

```bash
NETWORK=base
PRIVATE_KEY=0x...                          # deployer hot wallet
BASE_RPC_URL=<archival RPC, see §1.7>

# Foundation governance addresses
FOUNDATION_MULTISIG=0x...                  # Safe address from §1.2
FOUNDATION_RESERVE_WALLET=0x...            # treasury (often = multi-sig)
TREASURY_ADDRESS=0x...                     # FTNS initial-supply recipient

# Pool sinks
CREATOR_POOL=0x...
OPERATOR_POOL=0x...
GRANT_POOL=0x...

# Operational
AUTHORIZED_VERIFIER=0x...                  # off-chain prover EOA from §1.4

# FTNS routing
FTNS_DEPLOY_MODE=existing                  # OR "real" if minting fresh
FTNS_TOKEN_ADDRESS=0x...                   # only if mode=existing
```

### 1.7 RPC selection
Default `https://mainnet.base.org` is fine for one-shot deploys but
rate-limited and may drop intermediate txs under burst load. Recommend
an archival provider (Alchemy / Infura / QuickNode) with the URL pasted
into `BASE_RPC_URL`. Test BEFORE T-0 with:

```bash
curl -s -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
  "${BASE_RPC_URL}"
```

### 1.8 Etherscan v2 API key
For post-deploy contract verification. Set `ETHERSCAN_API_KEY` (one key
covers Base mainnet via the v2 unified multichain API per
`hardhat.config.js`).

---

## 2. T-1d: Base Sepolia full-ceremony rehearsal

This rehearsal **must** produce a green manifest before T-0. It exercises
the same scripts with a Sepolia stub multi-sig (NOT the production
hardware Safe — running the production Safe on Sepolia for rehearsal is
overkill and risks mis-binding signers to a testnet artifact).

### 2.1 Run the orchestrator

```bash
NETWORK=base-sepolia \
  FTNS_DEPLOY_MODE=real \
  FOUNDATION_RESERVE_WALLET=0x... \
  TREASURY_ADDRESS=0x... \
  CREATOR_POOL=0x... \
  OPERATOR_POOL=0x... \
  GRANT_POOL=0x... \
  AUTHORIZED_VERIFIER=0x... \
  PRIVATE_KEY=0x... \
  BASE_SEPOLIA_RPC_URL=<rpc> \
  ./scripts/rehearse-deploy.sh
```

### 2.2 Acceptance criteria
- All 5 deploy steps green.
- 7/7 Ownable transfers succeed; 7/7 skip cleanly on idempotent re-run.
- 5/5 FTNS role-transfer actions succeed; 5/5 skip cleanly on
  idempotent re-run.
- All 5 manifest JSONs written to `contracts/deployments/`.
- Etherscan-verify all 9 contracts (audit-bundle 4 + Phase 8 2 +
  Phase 7-storage 2 + FTNS proxy 1).

### 2.3 Failure handling
If any deploy step fails:
1. **DO NOT** retry on the same hardhat-internal state. Read the error.
2. If gas exhaustion: bump `gas` / `gasPrice` in `hardhat.config.js`
   `base-sepolia` entry, then re-run. The deploy scripts are NOT
   idempotent on partial failure (a half-deployed contract leaves
   stranded gas + a manifest gap).
3. If RPC drop: re-run from the failed step; earlier manifests are
   preserved.
4. If invariant check fires (e.g., `slashing.stakeBond != stakeBondAddr`):
   STOP. Investigate before redeploying — this signals a contract bug
   or env-var mistake.

---

## 3. T-0: Mainnet ceremony

**Conservative timing estimate:** 60-120 minutes wall-clock, dominated
by hardware-multi-sig signing latency (~30s per Safe tx between
"propose" → "collect signatures" → "execute"). Of that, only the post-
handoff multi-sig governance txs require the hardware. The deployer
phase is hot-key-fast (~5 minutes for all deploys + transfers).

### 3.1 Pre-ceremony briefing (T-0 minus 30min)

All three signers + the operator on a video call. Walk through:
1. Confirm everyone has their hardware device + backup card.
2. Confirm Safe address + signers via app.safe.global.
3. Confirm `.env.mainnet` values (read aloud, sign by sign).
4. Confirm the deployer hot wallet is funded (`eth_getBalance`).
5. Confirm RPC reachable (run §1.7 test).
6. **Read the abort criteria (§7) aloud.** Everyone must agree on what
   triggers an abort.

### 3.2 Phase 1: Deployer hot-key deployments

Run the orchestrator with mainnet flags. **Do NOT use
`SKIP_TRANSFER=1`** — we want the deployer to do the cross-wires under
hot-key authority and hand off in §3.3.

```bash
source .env.mainnet
./scripts/rehearse-deploy.sh
```

The script will:
1. **[1/5]** Deploy FTNSTokenSimple UUPS proxy (skipped if `FTNS_DEPLOY_MODE=existing`).
2. **[2/5]** Deploy audit bundle: EscrowPool, BatchSettlementRegistry,
   StakeBond, Ed25519Verifier; cross-wire EscrowPool↔Registry,
   StakeBond.setSlasher, etc. Run 15 invariants.
3. **[3/5]** Deploy Phase 8 emission: EmissionController,
   CompensationDistributor; cross-wire setAuthorizedDistributor. Run
   invariants.
4. **[4/5]** Deploy Phase 7-storage: StorageSlashing, KeyDistribution.
   Run invariants.
5. **[5/6]** Run transfer-ownership.js (hands all 7 Ownable contracts
   to `FOUNDATION_MULTISIG`) + transfer-ftns-roles.js if FTNS was
   deployed fresh.

**Checkpoint A** — pause here even if the script ran clean. Read the
manifests aloud:

```bash
ls -1t contracts/deployments/*-base-*.json | head -7
```

Verify each manifest has:
- `network: "base"` (not "base-sepolia")
- `chainId: "8453"` (not 84532)
- `deployer:` matching the expected hot-key address
- All cross-wires recorded

Etherscan-eyeball the deployer's tx log: 9 contract creations + ~6
cross-wire setters + 7 transferOwnership txs (+ 5 FTNS role txs if
fresh). Roughly 20-30 txs total.

### 3.3 Verify ownership/role handoff stuck

Before proceeding to §3.4, prove the handoff is ON-CHAIN:

```bash
# For each of the 7 Ownable addresses, confirm owner() == FOUNDATION_MULTISIG:
cast call <addr> "owner()(address)" --rpc-url $BASE_RPC_URL

# For FTNS, confirm:
cast call <ftns> "hasRole(bytes32,address)(bool)" \
  0x0000000000000000000000000000000000000000000000000000000000000000 \
  $FOUNDATION_MULTISIG --rpc-url $BASE_RPC_URL
# expect true

cast call <ftns> "hasRole(bytes32,address)(bool)" \
  0x0000000000000000000000000000000000000000000000000000000000000000 \
  $DEPLOYER_ADDRESS --rpc-url $BASE_RPC_URL
# expect false (deployer renounced)
```

If any check fails: ABORT (§7). Do NOT proceed to §3.4 — multi-sig
governance txs against contracts the multi-sig doesn't yet own will
revert.

---

## 4. T-0: Multi-sig governance follow-ups

These are 2-of-3 hardware-signed Safe transactions. Each takes ~30-60
seconds wall-clock for the signing flow.

### 4.1 Grant `MINTER_ROLE` on FTNSToken to EmissionController

This is the load-bearing post-handoff tx. Without it, EmissionController
cannot mint and the entire emission economy is dead.

In Safe Transaction Builder:
- **To:** FTNS token address
- **ABI fragment:** `grantRole(bytes32 role, address account)`
- **role:** `keccak256("MINTER_ROLE")` (`0x9f2df0fed2c77648de5860a4cc508cd0818c85b8b8a1ab4ceeef8d981c8956a6`)
- **account:** EmissionController address (from `phase8-emission-base-*.json`)

Collect 2 signatures, execute. Verify post-tx:

```bash
cast call <ftns> "hasRole(bytes32,address)(bool)" \
  0x9f2df0fed2c77648de5860a4cc508cd0818c85b8b8a1ab4ceeef8d981c8956a6 \
  <emissionController> --rpc-url $BASE_RPC_URL
# expect true
```

### 4.2 (Optional) Configure POL parameters
If the tokenomics ratification packet calls for non-default POL
parameters, configure now. Each is a multi-sig tx against
EmissionController.

### 4.3 (Optional) Configure StakeBond parameters
Similarly, any non-default minimum-stake / unbonding-period parameters.

### 4.4 (Optional) Pause-by-default posture
Some ops decisions favor deploying-paused, then unpausing in a separate
tx after monitoring is confirmed live. If chosen, the pause txs go here.

---

## 5. Post-ceremony (T+0 → T+3d)

### 5.1 Etherscan verify (T+0 immediately)
For each contract:

```bash
cd contracts
npx hardhat verify --network base <address> <constructor-args...>
```

Manifests record constructor args; the audit-bundle deploy script
prints the verify command line for each contract at deploy time.

Verification has to happen quickly because Etherscan caches the
"unverified" state and external auditors / users start seeing
unverified bytecode.

### 5.2 Activate monitoring (T+0 within 1h)
- Forta detection bots: re-target to mainnet contract addresses (was
  on testnet for rehearsal).
- Alert routing: PagerDuty + Telegram channels confirmed live.
- On-chain monitoring dashboard: contract addresses pinned.

### 5.3 Update audit-prep bundle (T+1)
The cumulative audit-prep doc references `0x...` addresses that were
testnet placeholders. Replace with mainnet addresses and re-tag:

```bash
# Update docs/2026-04-27-cumulative-audit-prep.md §-deploy-addresses
git tag mainnet-deploy-<YYYYMMDD>
git push origin mainnet-deploy-<YYYYMMDD>
```

### 5.4 Public announcement (T+1 to T+3)
Coordinate with comms; do not announce before §5.1 + §5.2 are green.
Post-announce, the auditor shortlist + ecosystem partners get the
contract-address packet.

---

## 6. Multi-sig governance txs to NOT defer

The following are explicitly part of the ceremony, not "later":
- **§4.1 MINTER_ROLE grant** — without this emission is dead.

The following CAN be deferred to later governance sessions if needed:
- §4.2 POL parameter tuning (defaults are sensible; tune from
  observed mainnet behavior).
- §4.3 StakeBond parameter tuning (same logic).
- §4.4 pause-by-default if not chosen at §3.1 briefing.

---

## 7. Abort criteria

**Pause the ceremony immediately if any of these fire:**

1. Any deploy script throws an invariant failure (`slashing.stakeBond
   != ...` etc). This signals a contract bug or env-var mistake.
2. Manifest network/chainId doesn't match expected (`"base"` / `8453`).
3. Deployer hot-wallet balance drops below 0.05 ETH mid-ceremony
   (gas-exhaustion risk on subsequent txs).
4. RPC stops responding (multiple consecutive timeouts on tx
   submission). Wait for recovery; do NOT swap RPCs mid-ceremony if
   ANY tx has been broadcast on the original.
5. Hardware signer reports a tx-data mismatch when displaying for
   confirmation. **DO NOT sign past a mismatch.** Investigate the
   Safe Transaction Builder payload before retrying.
6. Verification in §3.3 fails: ownership/roles did NOT transfer.
7. Multi-sig signer indicates suspected device compromise or signer-
   identity uncertainty.

### Abort recovery
- If aborted in §3.2 (deployer phase): partial deploys may have
  succeeded. Manifests are written per-step. Decide whether to:
  (a) Continue from the failed step on the SAME deployer key (the
      audit-bundle script reads earlier manifests), OR
  (b) Restart from scratch with a fresh deployer key — the partially-
      deployed contracts on mainnet stay there, owner = old deployer.
      They're effectively orphaned but harmless (no `migrate` path).
- If aborted in §3.3 verification: ownership stuck on deployer. Re-
  running `transfer-ownership.js` against the same manifests is
  idempotent and safe.
- If aborted in §4 (multi-sig phase): the multi-sig owns everything.
  Re-running multi-sig governance txs is a separate Safe tx flow —
  no engineering action required.

---

## 8. Appendix: command quick-reference

### A. Run the full orchestrator (mainnet)
```bash
source .env.mainnet
./scripts/rehearse-deploy.sh
```

### B. Run only the ownership transfer (idempotent)
```bash
FOUNDATION_MULTISIG=0x... \
  AUDIT_BUNDLE_MANIFEST=contracts/deployments/audit-bundle-base-XXX.json \
  PHASE8_MANIFEST=contracts/deployments/phase8-emission-base-XXX.json \
  PHASE7_STORAGE_MANIFEST=contracts/deployments/phase7-storage-base-XXX.json \
  npx hardhat run contracts/scripts/transfer-ownership.js --network base
```

### C. Run only the FTNS role handoff (idempotent)
```bash
FOUNDATION_MULTISIG=0x... \
  PHASE1_FTNS_MANIFEST=contracts/deployments/phase1-ftns-base-XXX.json \
  npx hardhat run contracts/scripts/transfer-ftns-roles.js --network base
```

### D. Verify ownership of all 7 Ownable contracts
```bash
for addr in $ESCROW_POOL $REGISTRY $STAKE_BOND $EMISSION_CONTROLLER \
            $COMP_DISTRIBUTOR $STORAGE_SLASHING $KEY_DISTRIBUTION; do
  echo -n "$addr owner: "
  cast call $addr "owner()(address)" --rpc-url $BASE_RPC_URL
done
```

### E. Multi-sig MINTER_ROLE grant payload (Safe Transaction Builder)
- **To:** FTNS token
- **Function:** `grantRole(bytes32,address)`
- **role:** `0x9f2df0fed2c77648de5860a4cc508cd0818c85b8b8a1ab4ceeef8d981c8956a6`
- **account:** EmissionController address

---

## 9. Document provenance

- Authored 2026-04-30 ahead of multi-sig hardware arrival 2026-05-01.
- Walks the orchestrator commits at HEAD (G1 + G2 + G3 + G5 + G6 wired).
- Covers both `FTNS_DEPLOY_MODE=existing` (token already deployed on
  Base mainnet) and `FTNS_DEPLOY_MODE=real` (fresh Phase 1.3 deploy)
  paths.
- Companion to `2026-04-30-deploy-ceremony-dry-run-audit.md`, which
  enumerates the 6 gaps surfaced by the dry-run + their resolutions.
