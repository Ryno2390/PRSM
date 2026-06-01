# PRSM Canonical Workflow — Base Mainnet Bring-Up Runbook

**Status:** Stage 1 (preparation) drafted 2026-05-06 immediately
following the Sepolia trace pass. Stage 2 (live mainnet execution) is
the next-session deliverable, gated on the operational decisions in
§3 below.

**Scope:** execute the same Vision §4 step 6 royalty-distribution
trace on Base mainnet that
`docs/2026-05-06-canonical-workflow-base-sepolia-trace.md` documents
on Base Sepolia. Same script, same shape, real money instead of
testnet money.

---

## 1. What is already true vs. what is not

**Already deployed on Base mainnet (per Phase 1.3 Task 8, 2026-05-04):**

| | Address | Source |
|---|---|---|
| FTNS Token | `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` | `prsm/config/networks.py:MAINNET` |
| ProvenanceRegistry v1 | `0xdF470BFa9eF310B196801D5105468515d0069915` | `prsm/config/networks.py:MAINNET` |
| RoyaltyDistributor | `0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2` | Deployment manifest at `contracts/deployments/provenance-base-1777917793612.json` |
| Foundation Safe (2-of-3 multisig, Ledger + Trezor + OneKey) | `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791` | Memory: Phase 1.3 Task 8 deploy complete |

The contracts have been mainnet-live for 2 days as of this runbook
draft. They've not been exercised in a single end-to-end
distribute_royalty call.

**Already true in this commit:**

- `MAINNET.royalty_distributor` is now pinned at the deployed address
  (was `None` with a TODO; resolved in this commit).
- The exercise script
  `scripts/exercise_canonical_workflow_base_sepolia.py` is
  network-agnostic at runtime — the contract addresses are
  compile-time constants, BUT they're easy to template as env
  vars for a mainnet run. (See §4 for the mainnet variant.)

**Not yet true:**

- The deployer EOA `0xBbEB1cb42F1D5ad05B46eE023D6e4871D813C9a0` (the
  Sepolia deployer) holds **zero FTNS on mainnet** and **zero ETH on
  Base mainnet**. It cannot fund itself.
- No mainnet wallet other than the Foundation Safe holds material
  FTNS — the entire 100M genesis supply lives in the Safe.
- No "demo payer" wallet exists yet on mainnet.

## 2. The bring-up question, stated honestly

The Sepolia trace already proves the canonical-workflow step 6 flow
**works against the same contract code that's deployed on mainnet**.
A mainnet trace is not new technical evidence — the bytecode is
identical. A mainnet trace is **demo theater for a real chain**: it
proves the flow fires when there is a real reason for it to fire.

That distinction matters because mainnet has costs:

1. **Real ETH for gas** — the deployer EOA needs ~0.005 Base ETH
   funded from somewhere (~$0.20 at current Base prices, but a real
   external transfer).
2. **Real FTNS in motion** — the demo payer needs FTNS. The only
   place FTNS exists today is the Foundation Safe; getting any to
   the demo payer requires a 2-of-3 hardware-multisig disbursement.
3. **Real on-chain history** — the trace tx becomes part of the
   permanent Base mainnet chain. There is no "delete trace" button.

Three operational paths are credible. Pick one before §4 execution:

### Path A: Run the trace now with a Foundation-Safe-funded demo payer

**Mechanics:**
1. Foundation Safe (2-of-3 multisig) disburses 100 FTNS to the demo
   payer wallet in one Safe transaction (real ceremony with two of
   the three hardware wallets).
2. Demo payer wallet is funded with ~0.005 Base ETH from any source
   (Coinbase fiat → Base, or the founder's personal Base wallet).
3. The exercise script runs against mainnet env config.
4. The trace tx hash is published as a screenshot in the next-day
   transparency post.

**Cost:** ~$0.20 in real ETH gas + 1 hour of multisig ceremony time.
**Risk:** real chain history with the trace forever; trivial financial
exposure but the optics matter.

### Path B: Wait for an actual creator + actual content + actual user

**Mechanics:** treat the Sepolia trace as the technical proof and
defer the mainnet trace until a real creator uploads real content and
a real user pays for access. The first mainnet RoyaltyPaid event is
then organic, not synthetic.

**Cost:** 0. **Risk:** unknowable timeline — could be weeks to months
before organic usage drives the first tx, depending on the rest of
the launch sequence.

### Path C: Hybrid — run a single self-payer trace today, organic trace later

**Mechanics:** the Foundation Safe ALSO acts as payer + creator
(self-funded, self-attributed). FTNS moves within the Foundation's
own custody chain and shows up on Basescan as a single tx with the
RoyaltyPaid event populated for the same address as creator and
serving_node.

**Cost:** ~$0.20 gas + same multisig ceremony, but the FTNS lands
back in the Foundation's claimable[] balances. **Risk:** the trace
shows the same address everywhere, which is honest but visually
weaker than Path A. Investors/auditors may read it as "self-dealing
test" rather than "system works."

**Recommendation: Path A.** The cost is trivial; the demo evidence is
clean (three distinct addresses on Basescan); and the multisig
ceremony itself is operational practice that the team will need to
get fluent with anyway. Path C is acceptable if the multisig
ceremony has scheduling friction; Path B is the right answer only if
the launch timeline genuinely defers all production usage by months.

## 3. Pre-flight checklist (Path A)

Before running the script:

- [ ] **Foundation Safe disbursement** of 100 FTNS to the demo payer
      wallet. Disbursement amount is well below the PRSM-POL-1 §5
      tier-1 threshold of < $10K (FTNS market price is $0; even at
      bull-case $5/FTNS this is $500). Founder + 1 signer
      authorization sufficient. **Document this disbursement in the
      next quarterly transparency report per PRSM-POL-1 §7.1.**
- [ ] **Demo payer wallet** identified. Two viable options:
      - The existing Sepolia deployer `0xBbEB...C9a0` — same address,
        same private key in the burner env file. Reuses operational
        familiarity but mixes testnet + mainnet signing on one key
        (acceptable for burner; avoid for production-critical keys).
      - A fresh deterministic ephemeral wallet derived from a new
        seed label. Cleaner separation but requires a new env file.
      Recommendation: reuse the Sepolia deployer for this single
      demo trace; rotate before any production-critical signing.
- [ ] **Base mainnet ETH funding** for the demo payer (~0.005 ETH).
      Source: founder's personal Base wallet (one external transfer)
      OR Coinbase fiat → Base bridge. Document the funding source in
      the quarterly transparency report.
- [ ] **Mainnet RPC URL** — Alchemy or Infura Base mainnet endpoint.
      Set as `BASE_MAINNET_RPC_URL` in the env file (separate from
      `BASE_SEPOLIA_RPC_URL`).
- [ ] **Council acknowledgement** — record in this runbook the
      acknowledgement that the trace is authorized. The Foundation
      Safe disbursement bullet above is the primary control. No
      separate council resolution required (this is operational,
      not policy-amending).
- [ ] **Public-relations read-through** — confirm the trace tx +
      basescan link will be published as a transparency artifact.
      Pre-write the announcement so post-trace publication is
      mechanical (avoids "we did it but didn't talk about it
      publicly for 3 weeks" optics).

## 4. Execution plan (Path A)

The exercise script is currently named `*_base_sepolia.py` but is
network-agnostic in implementation — the contract addresses are
constants near the top of the file. Two paths to mainnet execution:

### 4.1 Lightweight: env-driven addresses (recommended)

Patch the script to read addresses from env vars (with the existing
Sepolia values as defaults), then run with mainnet values exported.
This is a small, reviewable change. Sketch:

```python
FTNS_TOKEN = os.environ.get(
    "PRSM_FTNS_TOKEN", "0xF8d0c1AE75441d3C3Dd2A2420C0789043916412a",
)
PROVENANCE_REGISTRY = os.environ.get(
    "PRSM_PROVENANCE_REGISTRY",
    "0x2911f9a0a02896486CdF59d6d369764841DC0eA4",
)
ROYALTY_DISTRIBUTOR = os.environ.get(
    "PRSM_ROYALTY_DISTRIBUTOR",
    "0xB790045ff826C76fe02DBc54a6ef0021951Fd892",
)
EXPLORER = os.environ.get(
    "PRSM_EXPLORER", "https://sepolia.basescan.org",
)
CHAIN_ID = int(os.environ.get("PRSM_CHAIN_ID", "84532"))
```

Then for the mainnet run:

```bash
export BASE_MAINNET_RPC_URL=https://mainnet.base.org   # or Alchemy
export PRIVATE_KEY=<demo payer private key>
export PRSM_FTNS_TOKEN=0x5276a3756C85f2E9e46f6D34386167a209aa16e5
export PRSM_PROVENANCE_REGISTRY=0xdF470BFa9eF310B196801D5105468515d0069915
export PRSM_ROYALTY_DISTRIBUTOR=0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2
export PRSM_EXPLORER=https://basescan.org
export PRSM_CHAIN_ID=8453
export BASE_SEPOLIA_RPC_URL="$BASE_MAINNET_RPC_URL"  # script reads this var
PYTHONPATH=. .venv/bin/python3.14 \
    scripts/exercise_canonical_workflow_base_sepolia.py
```

### 4.2 Alternative: rename + symlink

Rename the script to `exercise_canonical_workflow.py` (network-
neutral) and add a thin wrapper at
`exercise_canonical_workflow_base_mainnet.py` that exports the
mainnet env vars and calls the network-neutral script. This is a
larger refactor but keeps the per-network entry points discoverable.

**Recommendation: 4.1.** Smaller change, single file diff,
backwards-compatible.

## 5. Expected mainnet trace shape

If everything is wired right, the script's terminal output is
identical in shape to the Sepolia trace except for:
- `Chain id: 8453` instead of `84532`
- Tx URLs at `https://basescan.org/tx/...` instead of `sepolia.basescan.org`
- Three new mainnet tx hashes (approve + register + distribute)
- Three new mainnet `claimable[]` balance reads

Expected tx count: 3 if no prior registration exists for the
content_hash (register + approve + distribute), or 2 if reusing a
prior registration (approve + distribute).

The math at the contract level is identical: 10 FTNS gross at 10%
royalty rate splits 1.0 / 0.2 / 8.8.

## 6. Failure modes specific to mainnet

The Sepolia trace identified three robustness fixes already shipped
in the script (RPC propagation polling on register / ownership /
allowance). Mainnet adds two more risk categories:

1. **Higher gas prices.** Base mainnet gas is cheaper than Ethereum
   mainnet but still 5-10× Sepolia. Each tx may cost up to ~$0.10.
   Pre-funding 0.01 ETH ($0.40-$0.80) instead of 0.005 gives margin.
2. **Real confirmation latency.** Sepolia blocks at ~2s; Base
   mainnet at ~2s also. The polling loops in the script (15s budget
   per check) cover real-chain latency without modification.

If the distribute_royalty tx reverts on mainnet (which would be
news, since Sepolia passed), the diagnostic order is:
- Decode the revert reason from the receipt.
- Confirm `ProvenanceRegistry.contents(content_hash).creator` is
  actually populated on mainnet (the `transferContentOwnership`
  step or the initial `register_content` may not have landed).
- Confirm `FTNS.allowance(payer, distributor) >= gross`.
- Confirm `FTNS.balanceOf(payer) >= gross`.

The exercise script's pre-flight blocks all four of those before
attempting distribute, so a revert at the distribute step would
indicate either a contract-state regression on mainnet or a
contract-difference between mainnet and Sepolia (which would itself
be news).

## 7. After the trace

- Update `prsm/deployments/contract_addresses.json` to record any
  mainnet addresses not already present (RoyaltyDistributor was
  already in the deployment manifest; ProvenanceRegistry v1 too).
- Publish the trace as
  `docs/2026-05-XX-canonical-workflow-base-mainnet-trace.md` with
  the same shape as the Sepolia trace doc.
- Tag: `canonical-workflow-base-mainnet-trace-passes-YYYYMMDD`.
- Quarterly transparency report (per PRSM-POL-1 §7.1): include the
  Foundation Safe disbursement of 100 FTNS, the gas-funding source,
  and the trace tx URL.

## 8. What this runbook does NOT cover

- **V2 contract on mainnet.** ProvenanceRegistryV2 mainnet deploy is
  authorized by `PRSM-CR-2026-05-06-2` (separate council resolution)
  and orthogonal to this trace. V2 is the embedding-commitment
  dispute layer; the canonical-workflow trace exercises v1 royalty
  distribution.
- **Multi-node SPRK dispatch on mainnet.** Already proven on Sepolia
  AND in the unit tests. Re-running cross-node on mainnet adds no
  evidence not already in the test surface.
- **Mainnet contract upgrades** (UUPS proxy admin operations). All
  Phase 1.3 contracts are non-upgradeable on the audit-bundle
  branch; no upgrade ceremony in this runbook.
- **Bug bounty program activation** (L9 per `audits/AUDIT_PLAN.md`).
  Activation gated on L4 first-pass completion for fund-custodying
  contracts; that gate remains in force per `PRSM-CR-2026-05-06-2` §5.

## 9. Stop conditions

If any of these conditions are observed during execution, ABORT and
investigate before retry:

- Pre-flight detects FTNS balance lower than expected (suggests
  prior unrecorded movement of the disbursement).
- Pre-flight detects ETH balance lower than expected (suggests
  unrecorded gas spend).
- ProvenanceRegistry returns a creator address other than the
  expected ephemeral creator after `transferContentOwnership`
  (suggests a different actor has manipulated the registry slot for
  the content_hash; investigate payload-label collision).
- Distribute tx receipt status = 0 (revert).
- RoyaltyPaid event amounts diverge from preview() amounts
  (suggests contract bytecode difference between Sepolia and
  mainnet — would itself be news, file under "different beast"
  before any further work).

In all of the above, the safe response is: stop, take a screenshot,
post in the Foundation Slack, return to investigation.
