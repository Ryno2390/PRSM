# Session Summary — Deploy-Ceremony Prep, 2026-04-30

**Audience:** the operator picking up tomorrow morning (2026-05-01) for
hardware-arrival day. Reads cold without scrolling commit history.

**Scope:** what got built between commits `34b59c11..7ddf87b3` (11
commits total) over the evening of 2026-04-30, after re-reading the
PRSM Foundation vault docs (`Multi-Sig_Action_Plan.md` + `PRSM_Tokenomics.md`)
and pivoting from the post-audit ceremony focus to Phase 1.3 Task 8
prep + closing 6+10 audit gaps with executable infrastructure.

---

## TL;DR

Hardware lands 2026-05-01. **Phase 1.3 Task 8** (`deploy-provenance.js`
for ProvenanceRegistry + RoyaltyDistributor with the Foundation Safe
as `NETWORK_TREASURY`) is the immediate next mainnet ceremony — narrow
scope, single deploy ceremony, disposable deployer key per
`Multi-Sig_Action_Plan.md` Phase 5-6.

The audit-bundle stack ceremony (audit-bundle + Phase 8 emission +
Phase 7-storage; 9 contracts + transferOwnership) is **post-external-
audit**, weeks-to-months out, gated on Phase 7 Task 9 + Phase 7.1
Task 9 audit completion.

Both ceremonies now have:
- Hardened deploy scripts with mainnet-only safety guards
- Hardhat-local rehearsal infrastructure proven green end-to-end
- Engineering runbooks + audit gap-closure docs

---

## What got built tonight, by category

### A. Post-audit ceremony hardening (the bigger ceremony, later)
- `scripts/rehearse-deploy.sh` — six audit gaps closed (G1+G2+G3+G5+G6).
  G4 (verifier-contract migration) deliberately deferred since EOA
  prover is the v1 design.
- `contracts/scripts/transfer-ownership.js` — 7-Ownable handoff
  ceremony script (audit-bundle EscrowPool/Registry/StakeBond + Phase 8
  EmissionController/CompensationDistributor + Phase 7-storage
  StorageSlashing/KeyDistribution).
- `contracts/scripts/transfer-ftns-roles.js` — FTNS AccessControl role
  handoff (DEFAULT_ADMIN_ROLE grant to multi-sig + 4 renounce txs);
  belt-and-braces against stranding the contract by renouncing admin
  before multi-sig has it.
- `contracts/scripts/deploy-phase1-ftns.js` — fresh FTNSTokenSimple
  UUPS proxy deploy for testnet rehearsal (production already at
  `0x5276a3756C85f2E9e46f6D34386167a209aa16e5`).
- `contracts/hardhat.config.js` — `pkAccounts()` helper rejecting the
  `your_private_key_here` placeholder that previously blocked all
  network configs including hardhat-local.
- `docs/2026-04-30-deploy-ceremony-dry-run-audit.md` — full gap-by-gap
  audit doc (G1-G6).
- `docs/2026-04-30-post-audit-deploy-ceremony-runbook.md` — chronological
  T-3d → T+3d operator runbook (renamed from the original "mainnet
  deploy-ceremony" name to disambiguate scope).

### B. Phase 1.3 Task 8 ceremony prep (tomorrow's ceremony)
- `contracts/scripts/deploy-provenance.js` — three new mainnet-only
  guards added: chainId pin (refuse `--network=base` if RPC reports
  != 8453); canonical FTNS pin (refuse `FTNS_TOKEN_ADDRESS != 0x5276...`
  unless `FORCE_NONCANONICAL_FTNS=1`); treasury-is-contract on mainnet
  (Safe must have bytecode, not be an EOA).
- `contracts/scripts/verify-provenance-deployment.js` — post-deploy
  on-chain state verifier (manifest → call all 4 immutable getters
  via ABI fragments → assert equality + bytecode + symbol + treasury
  sanity).
- `scripts/rehearse-task8.sh` — hardhat-local rehearsal orchestrator:
  spins up node, deploys MockERC20, deploys provenance with stub
  treasury, runs verifier. Also supports `NETWORK=base-sepolia` for
  testnet rehearsal.
- `scripts/pre-task8-checklist.sh` — 10-check executable mainnet gate
  (env vars → key format → derive deployer → RPC chainId → Etherscan
  validity → canonical FTNS → on-chain symbol → treasury != deployer
  → treasury bytecode → balance ≥ 0.003 ETH). Wires F1+F5+F9 from the
  Multi-Sig Action Plan engineering audit.
- `scripts/post-task8-handoff-checklist.sh` — post-deploy ops integration
  helper. Greps repo for OLD-address references, surfaces known
  integration touchpoints (Forta bots, pause-tx templates, audit-prep
  doc, MEMORY.md, .env templates), generates ready-to-use PR body +
  project-memory entry stub at `/tmp/`. Network-aware (won't lie about
  "Base mainnet" if run against a non-mainnet manifest). Wires F3.
- `docs/2026-04-30-phase1.3-task8-engineering-runbook.md` — engineering
  companion to operator-side `Multi-Sig_Action_Plan.md`. Pre-flight,
  hardhat-local dry-run, T-0 mainnet flow, post-deploy verification,
  rollback semantics.
- `docs/2026-04-30-multisig-action-plan-engineering-audit.md` — 10-finding
  audit (3 HIGH/MEDIUM, rest LOW) of the operator-side plan. Provides
  minimum-viable patch list the operator can apply to their vault copy.

### C. Cleanup
- Deleted `contracts/scripts/deploy.js` + `contracts/scripts/verify-deployment.js`
  (both stale: ethers v5; referenced non-existent contracts FTNSToken/
  FTNSMarketplace/FTNSGovernance/TimelockController; targeted
  polygon-mumbai which Polygon shut down 2024-04). Reconciled with
  the new deploy-provenance.js + verify-provenance-deployment.js per
  current architecture.
- `contracts/package.json` — replaced stale `deploy:testnet` /
  `deploy:mainnet` / `verify` npm scripts with `deploy:provenance:sepolia`
  / `deploy:provenance:base` / `verify:provenance`. Updated description
  (Base, not Polygon).
- `contracts/README.md` — replaced "Deployment" section with the
  ceremony layering table + the canonical Task 8 command sequence.
- `docs/FTNS_TESTNET_DEPLOYMENT.md` — added stale-content notice
  flagging the polygon-mumbai/deploy.js sections as obsolete.

### D. Memory writes
- `~/.claude/projects/.../memory/project_production_ftns_state_2026_04_30.md`
  — verified 2026-04-30 via direct Base RPC: FTNSTokenSimple at
  `0x5276a3756C85f2E9e46f6D34386167a209aa16e5`, symbol=FTNS,
  totalSupply=100M intact, name="PRSM Fungible Tokens for Node Support",
  EIP-1967 implementation `0xd979c096be297f4c3a85175774bc38c22b95e6a4`.
- (Existing) `feedback_repo_scope_prsm_vs_prismatica.md` —
  re-confirmed: engineering docs in PRSM repo, investor materials NOT.
- (Existing) `project_multisig_hardware_arriving_2026_05_01.md` —
  hardware unblocks Phase 7 Task 9 + Phase 7.1 Task 9 + the testnet
  rehearsal → mainnet ceremony chain.

---

## Hardware-day readiness — one table

| Ceremony | Status | When | Run sequence |
|---|---|---|---|
| **Phase 1.3 Task 8** (Provenance + Royalty) | ✅ ready, end-to-end rehearsal proven green | Tomorrow (2026-05-01 evening per Multi-Sig Action Plan timeline) | §1 below |
| **Audit-bundle ceremony** (audit-bundle + Phase 8 emission + Phase 7-storage) | ✅ rehearsal proven green; ⚠️ gated on external audit completion | Weeks-to-months out | §2 below |

Both ceremonies on hardhat-local, dry-run twice tonight, both green.

---

## §1 — Phase 1.3 Task 8 run sequence (tomorrow)

**Operator-side prep first** (per `Multi-Sig_Action_Plan.md` in the
Foundation vault):
- Phase 1: hardware device init × 3 (Ledger / Trezor / OneKey;
  ~90-120 min)
- Phase 2: fund Ledger from MetaMask
- Phase 3: create Foundation 2-of-3 Safe on Base
- Phase 4: round-trip signing test (verify all three devices can sign;
  per F6 audit finding, also test threshold-rejection — refuse to
  execute with only 1 sig)
- Phase 5: generate disposable deployer EOA + fund $10 worth of ETH

**Engineering-side execution** (this is what the operator runs after
hardware day completes Phase 1-5):

```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM

# (a) Pre-deploy gate — 10 checks, exits 1 on any fail
export PRIVATE_KEY=<from operator's deployer terminal>
export FTNS_TOKEN_ADDRESS=0x5276a3756C85f2E9e46f6D34386167a209aa16e5
export NETWORK_TREASURY=<SAFE_ADDR from Multi-Sig Action Plan §3.3>
export BASE_RPC_URL=<archival RPC, e.g. Alchemy/Infura/QuickNode>
export ETHERSCAN_API_KEY=<Etherscan v2 key>
export AUTO_VERIFY=1
./scripts/pre-task8-checklist.sh
# Must exit 0 with all 10 ✓ before proceeding.

# (b) Deploy
cd contracts
npx hardhat run scripts/deploy-provenance.js --network base 2>&1 \
  | tee /tmp/mainnet-deploy-$(date +%s).log
# Captures output to log per F4 audit finding.

# (c) Post-deploy on-chain state check
PROVENANCE_MANIFEST=deployments/provenance-base-<ts>.json \
  npx hardhat run scripts/verify-provenance-deployment.js --network base
# Must report: ✅ All on-chain state matches manifest.

# (d) Post-deploy ops handoff (find OLD-address refs + generate PR + memory)
cd ..
MAINNET_MANIFEST=contracts/deployments/provenance-base-<ts>.json \
  ./scripts/post-task8-handoff-checklist.sh
# Generates /tmp/task8-mainnet-handoff-pr-body-<ts>.md and
# /tmp/task8-mainnet-deploy-memory-stub-<ts>.md ready to commit/post.
```

Each step gates the next. If `(a)` fails, do not proceed. If `(c)`
fails (which would be a deploy-time bug), stop and investigate before
trusting the deploy.

---

## §2 — Post-audit ceremony run sequence (later)

This is the bigger ceremony — 9 contracts + 7 transferOwnership. Gated
on external audit sign-off (Phase 7 Task 9 + Phase 7.1 Task 9, both
currently `in_progress`).

The operator runbook is `docs/2026-04-30-post-audit-deploy-ceremony-runbook.md`.
Headline command:

```bash
# After auditor sign-off + Foundation Safe is battle-tested:
source .env.mainnet  # all the FOUNDATION_RESERVE_WALLET, CREATOR_POOL,
                      # OPERATOR_POOL, GRANT_POOL, AUTHORIZED_VERIFIER,
                      # PRIVATE_KEY, BASE_RPC_URL, FOUNDATION_MULTISIG,
                      # FTNS_TOKEN_ADDRESS, FTNS_DEPLOY_MODE=existing
./scripts/rehearse-deploy.sh
```

Internally chains: deploy-mock-ftns OR phase1-ftns → deploy-audit-bundle
→ deploy-phase8-emission → deploy-phase7-storage → transfer-ownership
(7 contracts) → transfer-ftns-roles (5 actions) → idempotency checks
on both transfers.

Hardhat-local rehearsal in both `FTNS_DEPLOY_MODE=mock` (default) and
`FTNS_DEPLOY_MODE=real` is proven green tonight. Base Sepolia rehearsal
should be scheduled once external audit reports come back.

---

## Open follow-ups (deliberately not closed tonight)

These are flagged for future sessions, not gaps in tomorrow's prep:

1. **G4 verifier-contract migration** (`StorageSlashing.authorizedVerifier`
   accepts EOA prover for v1; eventual migration to a verifier contract
   when feasible).
2. **F10 BASE_RPC_URL archival recommendation** (audit-doc note;
   operator-side choice, not a code gap).
3. **Phase 1.3 Task 9 + Task 10** (canary rollout + closeout — pick up
   after Task 8 lands per `Multi-Sig_Action_Plan.md` Phase 6).
4. **FTNSToken DEFAULT_ADMIN_ROLE handoff** (separate decision on its
   own timeline; the operator note in user-confirmed memory says the
   admin EOA is being loaded onto a hardware device — handoff when the
   Foundation governance decides).
5. **`docs/FTNS_TESTNET_DEPLOYMENT.md` aggressive cleanup** (currently
   has a deprecation banner; full rewrite or content trim is a future
   pickup).
6. **Forta bot mainnet re-targeting** (post-Task-8 step #2a in the
   handoff checklist; mechanical when triggered by the script's
   discovery output).
7. **Audit-prep retag post-Task-8** (`cumulative-audit-prep-mainnet-task8-<YYYYMMDD>`
   per Task 8 runbook §6 step 3).

---

## Session commit reference

```
34b59c11 deploy-ceremony dry-run: transferOwnership script + rehearsal extension + audit doc
a03497d3 deploy-ceremony G6: hard-fail on missing reserve-wallet + pool sinks on testnet/mainnet
76bdc1db deploy-ceremony G5: bundle Phase 1.3 FTNSToken into orchestrator + role-handoff ceremony
4ff4c9bb mainnet deploy-ceremony master runbook
85988825 phase 1.3 task 8 prep: deploy-provenance hardening + engineering runbook + post-audit runbook reframe
92b41123 phase 1.3 task 8 rehearsal: orchestrator script + on-chain state verifier
802561b3 multi-sig action plan engineering audit — 10 findings (3 HIGH/MEDIUM, rest LOW)
01582d34 phase 1.3 task 8 pre-flight checklist — wires F1+F5+F9 audit findings into executable infrastructure
b60cad78 phase 1.3 task 8 post-deploy handoff checklist — wires F3 audit finding
8a073b50 purge stale deploy.js + verify-deployment.js + reconcile downstream refs
7ddf87b3 post-task8 handoff: network-aware artifact templates (chain-test seam fix)
```

11 commits, ~3000 LoC delta (mostly +; one purge cleanup with -552/+53).

---

## What this doc explicitly is NOT

- Not a full ceremony runbook — see `docs/2026-04-30-phase1.3-task8-engineering-runbook.md`
  (Task 8) and `docs/2026-04-30-post-audit-deploy-ceremony-runbook.md`
  (post-audit) for those.
- Not a list of remaining engineering work — those are tracked in
  TaskList tasks #31 and #40 (both `in_progress`, hardware-gated, now
  unblocked tomorrow).
- Not a status report on the streaming-inference subsystem — that
  hit roadmap cap at Phase 3.x.11.q.x earlier in the day, separate
  from tonight's deploy-ceremony work.

---

## One-line summary for tomorrow's stand-up

> Phase 1.3 Task 8 mainnet deploy infrastructure ready: 4-script T-0
> sequence (`pre-task8-checklist` → `deploy-provenance` → `verify` →
> `post-task8-handoff-checklist`) proven green on hardhat-local, with
> deploy-script mainnet-hardened (chainId pin + canonical FTNS pin +
> treasury-is-contract guard) and Multi-Sig_Action_Plan.md gaps
> captured in a 10-finding audit doc with executable F1+F3+F5+F9
> closures. Post-audit ceremony rehearsal also re-verified green.
