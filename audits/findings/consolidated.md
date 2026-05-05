# L2 AI Multi-Team Audit — Consolidated Findings

**Date:** 2026-05-05
**Pinned commit:** `589c14d2` (HEAD of `main`, identical to
`cumulative-audit-prep-20260504-h`)
**Teams:** A (economic) · B (access control) · C (signature/crypto) ·
D (state composition). All 4 teams ran independently in parallel.
**Source documents:** `team-a-findings.md` · `team-b-findings.md` ·
`team-c-findings.md` · `team-d-findings.md`
**PoC tests:** `contracts/test/audit-team-{a,b,c,d}/` — every Confirmed
finding has a passing PoC against the real production contracts (not mocks).

## 1. Severity rollup (deduplicated, post-cross-team merge)

| Severity | Count | Production-blocking? |
|----------|-------|----------------------|
| **Critical** | **2** | YES — both block Gate B; one affects LIVE mainnet posture |
| **High** | **7** | YES — all 7 should remediate before Gate B |
| **Medium** | **8** | Should remediate before L4 contest |
| **Low** | **6** | Best-practice cleanup, batchable |
| **Informational** | **10** | Code/doc quality, no exploit |
| **TOTAL** | **33** | |

Dedupe notes: A-02 ≡ D-01 (slash-evasion race, found from 2 angles); D-03
is the umbrella of B-CROSS-1/2/3 cross-wire findings; D-06/D-07 are LOW
restatements of MEDIUM findings; A-03 INFO ⊂ B-CROSS-2 MEDIUM.

## 2. CRITICAL — must remediate immediately

### CRIT-1: C-INT-01 — Adversarial slashing via unbound `signingMessage`

**Source:** Team C (`team-c-findings.md` §C-INT-01)
**Location:** `BatchSettlementRegistry._handleInvalidSignature` (lines 540–557)
**PoC:** `contracts/test/audit-team-c/C-INT-01-invalid-signature-forgery.test.js` — passes against real Ed25519Verifier
**Gate impact:** **PRODUCTION-BLOCKING for Gate B.** Audit-bundle stack (#31, #40) cannot deploy with this primitive in place.

Anyone with a published off-chain receipt can submit an INVALID_SIGNATURE
challenge using the real `(pubkey, signature)` bytes plus an
attacker-chosen `signingMessage`. The verifier correctly returns false
(signature wasn't over the chosen message), the contract treats false as
"challenge proven," and 100% of the provider's stake is slashed with 70%
paid to the attacker as bounty. Single-tx attack, no preconditions beyond
having seen one published receipt.

**Recommended fix (Team C Option A):** add `signingMessageHash` field to
`ReceiptLeaf`. Require `keccak256(signingMessage) == leaf.signingMessageHash`
before verifier call. Updates: leaf encoding spec, off-chain Python parity,
parity tests. ~1–2 days engineering.

### CRIT-2: B-FTNS-1 — FTNS `DEFAULT_ADMIN_ROLE` held by out-of-Safe hot key on LIVE mainnet

**Source:** Team B (`team-b-findings.md` §B-FTNS-1)
**Location:** `FTNSTokenSimple` at `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` (Base mainnet)
**Verification:** Direct RPC against `mainnet.base.org` and `base.llamarpc.com`
**Gate impact:** **LIVE MAINNET.** Stated Invariant #1 (DEFAULT_ADMIN_ROLE on Foundation Safe) is currently NOT held.

Direct on-chain verification confirms:
- `hasRole(DEFAULT_ADMIN_ROLE, deployer 0x55d2...)` = `false`
- `hasRole(DEFAULT_ADMIN_ROLE, Foundation Safe 0x91b0...)` = `false`

An out-of-Safe hot key (the original FTNS deployer, distinct from the
audit-bundle deployer) holds admin authority on a UUPS-upgradeable contract
holding 100M FTNS. If this key is compromised before being either (a)
handed to the Safe via `transfer-ftns-roles.js` or (b) renounced via the
same script, the attacker can `grantRole(MINTER_ROLE, attacker)` and mint
up to 900M FTNS, push a malicious UUPS upgrade, pause the entire token, or
burn arbitrary balances.

**Note:** This is consistent with the previously-deferred state per
`docs/2026-04-27-cumulative-audit-prep.md §G4` and PRSM project memory
(`admin = hot key being loaded to hardware`). The audit's contribution is
formal confirmation that this is now a live CRITICAL exposure, not a
benign pending-task item.

**Recommended fix:** run `scripts/transfer-ftns-roles.js` immediately using
the hardware-loaded key as deployer signer. Transfer DEFAULT_ADMIN_ROLE to
the Safe and renounce all four deployer roles. ~1 day operational, gated on
hardware-wallet availability of the FTNS deployer key.

## 3. HIGH — must remediate before respective gates

### HIGH-1: A-01 — RoyaltyDistributor split diverges from PRSM-TOK-1 §8.1

**Source:** Team A (`team-a-findings.md` §A-01)
**Location:** `RoyaltyDistributor.distributeRoyalty` (lines 73–76). Live at `0x3E82...D6c2` Base mainnet.
**PoC:** `contracts/test/audit-team-a/SplitInvariant.test.js` — 3 passing
**Gate impact:** LIVE MAINNET; doc-vs-code mismatch on non-Ownable + non-upgradeable contract.

Spec: 20% burn + 80%-of-remainder split → effective 6.4% creator / 1.6% treasury / 72% node / 20% burn.
Contract: 8% creator / 2% treasury / 90% node / **0% burn**.

The 20% burn that anchors §8.2's deflationary tokenomics narrative is
absent from the deployed contract.

**Recommended fix (Path 1, preferred):** re-deploy v2 RoyaltyDistributor
implementing `IFTNSToken.burn(burnAmt)` in `distributeRoyalty`. Migrate
off-chain integrations + Forta bot registry to new address. Abandon v1.
~3–5 days engineering + ceremony + communications.

Path 2 (amend §8.1 to match contract) is inferior because the deflationary
narrative collapses; flag for completeness.

### HIGH-2: A-02 ≡ D-01 — Slash-evasion race when `unbondDelay < challengeWindow`

**Source:** Team A (§A-02) AND Team D (§D-01) — same finding from two angles
**Location:** `StakeBond.unbondDelaySeconds` + `BatchSettlementRegistry.challengeWindowSeconds`
**PoC:** `contracts/test/audit-team-a/UnbondRace.test.js` + `contracts/test/audit-team-d/UnbondVsChallengeRace.test.js`
**Gate impact:** Production-blocking for Gate B. Slasher is decorative without this fix.

Provider double-spends + requestUnbond same block → withdraw after 1 day
(MIN unbondDelay) → challenger lands valid challenge on day 5 (within MAX
30-day window). Receipt invalidated successfully but `stakeBond.slash`
reverts `NotSlashable(WITHDRAWN)`; revert silently swallowed by
`try/catch` in BSR. Provider keeps full stake; 70/30 bounty/Foundation
slash split entirely bypassed.

**Recommended fix:** in `StakeBond.requestUnbond`, set
`unbond_eligible_at = max(now + unbondDelaySeconds, now + registry.challengeWindowSeconds())`.
Requires StakeBond ↔ BSR cross-reference. ~1 day engineering. Plus
defense-in-depth: emit `SlashSwallowed` event when the silent swallow fires.

### HIGH-3: D-02 — Treasury contracts have no pause surface

**Source:** Team D (`team-d-findings.md` §D-02)
**Location:** None of `BatchSettlementRegistry.sol`, `EscrowPool.sol`, `StakeBond.sol`, `RoyaltyDistributor.sol` import OZ `Pausable` or use `whenNotPaused`.
**PoC:** `contracts/test/audit-team-d/PauseCoverageGap.test.js` — asserts absence of `pause()`/`paused()` selectors
**Gate impact:** Production-blocking for Gate B; live RoyaltyDistributor inherits gap (re-deploy required).

No surgical incident response is possible. Only kill-switch is
`FTNSTokenSimple.pause()` which globally halts all FTNS transfers
including legitimate user activity. Administratively too heavy for
incident response.

**Recommended fix:** add OZ `Pausable` to all four audit-bundle contracts.
PAUSER_ROLE assigned to Foundation Safe (or separate fast-response signer
set with lower threshold). Specify pause semantics for unbond timer
(continue/stop/reset; recommend continue, withdraw gated). ~2–3 days.

### HIGH-4: B-VERIF-1 — Audit-bundle deployment verifier calls non-existent getters

**Source:** Team B (`team-b-findings.md` §B-VERIF-1)
**Location:** `contracts/scripts/verify-audit-bundle-deployment.js` lines 145–160 + 197–210
**PoC:** `contracts/test/audit-team-b/B-AccessControl-PoC.test.js` "VERIFIER-ABI" — passes
**Gate impact:** Operational tool; the post-handoff verifier silently fails on routine cross-wire checks.

Verifier ABI: `challengeWindow()` / `unbondDelay()` / `ftnsToken()`.
Actual contract storage: `challengeWindowSeconds` / `unbondDelaySeconds` / `ftns`.
Three calls revert; `try/catch` swallows; verifier exits 1 with confusing
"getter reverted" messages that look like deploy corruption signals but
are actually verifier bugs.

**Note:** This is a script I wrote and committed in the previous session
(`46167ac2`). Self-introduced bug. Trivial fix.

**Recommended fix:** rename the three ABI declarations + getter callsites
to match actual contract storage. ~5 min, no on-chain change. Should ship
**TODAY** since the script gates the next deploy ceremony.

### HIGH-5: B-RENOUNCE-1 — Sole-admin renounce permanently bricks FTNSTokenSimple

**Source:** Team B (`team-b-findings.md` §B-RENOUNCE-1)
**Location:** `FTNSTokenSimple.sol` (uses OZ `AccessControlUpgradeable.renounceRole`)
**PoC:** `contracts/test/audit-team-b/B-AccessControl-PoC.test.js` "B4" — passes
**Gate impact:** Post-handoff foot-gun. One bad multi-sig tx = permanent UUPS-upgrade brick.

A single multi-sig signature renouncing DEFAULT_ADMIN_ROLE permanently disables:
- `_authorizeUpgrade` (no future UUPS upgrades)
- `grantRole(MINTER_ROLE, EmissionController)` for Phase 8
- All future role rotations

The script-level guard in `transfer-ftns-roles.js` only protects the
ceremony, not post-handoff Safe operation.

**Recommended fix:** override `renounceRole` to revert when role ==
DEFAULT_ADMIN_ROLE. **Caveat:** requires UUPS upgrade — chicken-and-egg
problem if admin is ever renounced. Apply BEFORE the Foundation Safe
holds admin (which means alongside the CRIT-2 handoff), OR via an
immediate post-handoff upgrade. ~1 day + UUPS upgrade ceremony.

### HIGH-6: B-CROSS-1 — `EscrowPool.setSettlementRegistry` re-pointable post-handoff drains escrow

**Source:** Team B (`team-b-findings.md` §B-CROSS-1)
**Location:** `EscrowPool.sol:162–166`
**PoC:** `contracts/test/audit-team-b/B-AccessControl-PoC.test.js` "B5 — cross-wire mutability" — drains in 3 txs
**Gate impact:** Compromised-Safe blast radius; Stated Invariant #6 NOT held.

Stated Invariant #6 (cross-wires immutable post-deploy) is not held.
`setSettlementRegistry` accepts ANY address with no contract-bytecode or
interface check. Compromised multi-sig → set registry to attacker EOA →
attacker calls `settleFromRequester(victim, attacker, victim.balance)`
for every depositor. All escrow drained, single block.

**Recommended fix (priority 1):** make `settlementRegistry` immutable
(constructor-only); remove the setter. Forces redeploy if rotation is
ever needed (acceptable — registry is a permanent trust anchor).
**OR (priority 2):** 14-day timelock on the setter. ~1–2 days.

### HIGH-7: B-CROSS-3 — `StakeBond.setSlasher` accepts arbitrary address; bounty-farming attack

**Source:** Team B (`team-b-findings.md` §B-CROSS-3)
**Location:** `StakeBond.sol:290–294`
**PoC:** `contracts/test/audit-team-b/B-AccessControl-PoC.test.js` "owner can re-point StakeBond.slasher" — passes
**Gate impact:** Compromised-Safe blast radius; Stated Invariant #7 NOT held.

Stated Invariant #7 (slasher = BSR address only) enforced ONLY at slash time, not at setter. Compromised Safe → setSlasher(attacker_eoa) → attacker.slash(victim, attacker, fabricated_batch_id) for every staked provider → claimBounty → drains all slashed FTNS at 70% rate.

**Recommended fix:** make `slasher` immutable. Forces StakeBond redeploy
if Registry is ever upgraded — acceptable for an immutable cross-wire.
~0.5 days.

### Cross-team finding (note): B-RENOUNCE-1 + B-PAUSE-1 share remediation

If the renounceRole override fix (HIGH-5) is applied, it should also
guard PAUSER_ROLE renounce when paused → addresses B-PAUSE-1 (MEDIUM)
in the same patch.

## 4. MEDIUM — should remediate before L4 contest

| ID | Source | Title | Remediation | Days |
|----|--------|-------|-------------|------|
| C-INT-02 | C | `ISignatureVerifier.verify` declared `view` not `pure` | Change interface to `pure`; introduce separate `IStatefulSignatureVerifier` if ever needed | 0.5 |
| D-03 | D | Owner cross-wire mutation instant; in-flight batches soft-bricked | Subsumed by HIGH-6 (B-CROSS-1) and HIGH-7 (B-CROSS-3) immutable fixes; remaining setters get 7-day timelock | 1–2 |
| D-04 | D | Royalty push-payment with no recovery; share-stranding via `transferContentOwnership` | Switch to pull-payment (mirror `StakeBond.slashedBountyPayable + claimBounty`); OR per-transfer `try/catch` with stranded routing | 2 |
| D-05 | D | `setChallengeWindowSeconds` retroactively shortens window for PENDING batches | Snapshot `challengeWindowSecondsAtCommit` per batch | 1 |
| B-CROSS-2 | B | `EscrowPool.setFtnsToken` strands real-token balances | Add on-chain `totalEscrowedBalance` accumulator; reject `setFtnsToken` while >0 | 1 |
| B-OWNABLE-1 | B | Single-step `Ownable` everywhere — handoff typo bricks contract | Replace `Ownable` with `Ownable2Step` on all 7 contracts | 1 |
| B-PAUSE-1 | B | All-PAUSER + admin renounce while paused → permanent freeze | Bundled with HIGH-5 renounceRole override | 0 (bundled) |
| B-TREASURY-1 | B | `RoyaltyDistributor.networkTreasury` only checks "is contract" not "is canonical Safe" | Add canonical-Safe pin in `deploy-provenance.js` (mirror existing `CANONICAL_FTNS_BASE_MAINNET`) | 0.25 |

## 5. LOW + INFO — best-practice cleanup

Batchable into a single cleanup PR:

- **C-INT-03** delete dead/typo'd `BRIDGE_MESSAGE_TYPEHASH` legacy constant
- **C-INT-04** add `destinationChainId` to `BridgeMessage` (defense-in-depth; EIP-712 already covers it)
- **C-INT-05** comment that `BridgeSecurity._verifySignature`'s `abi.encodePacked(r, s, v)` is safe (fixed-width)
- **C-INT-06** `removeValidator` last-validator brick state — guard or document
- **C-INT-07** off-chain HandoffToken JSON signing payload missing domain prefix (deferred, addendum §3.8(7))
- **C-INT-08** `Ed25519Verifier.verify` is `pure` while interface says `view` — consistency note (resolved by C-INT-02)
- **A-03** `EscrowPool.setFtnsToken` escape hatch (subsumed by B-CROSS-2)
- **A-04** Bond-tier vs batch-tier decoupling — off-chain trust assumption
- **A-05** Wash-trade economics 10× cheaper than §8.1 intends (resolves with HIGH-1 burn fix)
- **D-06** Same as B-CROSS-1/2/3 cluster
- **D-07** `EscrowPool.setFtnsToken` strand (= B-CROSS-2)
- **D-08** `providerBatchSequence` increments before collision check (informational; revert rolls back)

## 6. Live mainnet posture (Gate A)

**Phase 1.3 contracts deployed 2026-05-04 are affected by:**

| Finding | Severity | Affects | Remediation path |
|---------|----------|---------|------------------|
| **CRIT-2 (B-FTNS-1)** | Critical | FTNSTokenSimple admin role | Run `transfer-ftns-roles.js` from hardware-loaded FTNS deployer key |
| **HIGH-1 (A-01)** | High | RoyaltyDistributor split | Re-deploy v2 with burn (non-Ownable + non-upgradeable means no in-place fix) |
| **HIGH-3 (D-02)** | High (deferred) | RoyaltyDistributor pause surface | Bundle with HIGH-1 v2 re-deploy |
| **MEDIUM B-TREASURY-1** | Medium | Future RoyaltyDistributor deploys | Add canonical-Safe pin in script; no live impact |

**Phase 1.3 contracts NOT affected by audit-bundle-stack findings**
(C-INT-01, A-02/D-01 race, B-RENOUNCE-1, B-CROSS-*, B-OWNABLE-1) — these
all live in pre-deploy code.

## 7. Audit-bundle pre-deploy posture (Gate B)

The audit-bundle stack (`#31`, `#40`) **CANNOT DEPLOY** until at minimum:

- **CRIT-1 (C-INT-01)** patched — adversarial slashing primitive eliminated
- **HIGH-2 (A-02/D-01)** patched — slash race closed via cross-contract invariant
- **HIGH-3 (D-02)** patched — pause surface added before deploy (post-deploy add requires re-deploy of non-upgradeable contracts)
- **HIGH-4 (B-VERIF-1)** patched — verifier script now actually works
- **HIGH-6 (B-CROSS-1)** patched — settlementRegistry made immutable or timelocked
- **HIGH-7 (B-CROSS-3)** patched — slasher made immutable

HIGH-5 (renounceRole override) only applies to FTNSTokenSimple; not strictly Gate-B-blocking but should bundle.

## 8. Recommended remediation sequencing

### Today (2026-05-05) — Operational

- ☐ **CRIT-2 (FTNS admin handoff):** founder-side decision on hardware-loaded FTNS deployer key. If hardware available, run `transfer-ftns-roles.js` immediately. If not, document mitigation timeline + accept residual risk explicitly in Foundation council record.
- ☐ **HIGH-4 (B-VERIF-1):** ship script fix today. Trivial; gates next deploy.

### Week 1 — Engineering remediation sprint (CRIT + HIGH)

- ☐ CRIT-1 C-INT-01 — `signingMessageHash` field on `ReceiptLeaf` + parity tests (1–2 days)
- ☐ HIGH-2 A-02/D-01 — `requestUnbond` cross-contract invariant + `SlashSwallowed` event (1 day)
- ☐ HIGH-3 D-02 — OZ `Pausable` on 4 audit-bundle contracts (2–3 days)
- ☐ HIGH-5 B-RENOUNCE-1 + bundled B-PAUSE-1 — `renounceRole` override for DEFAULT_ADMIN_ROLE + PAUSER_ROLE-while-paused (1 day)
- ☐ HIGH-6 B-CROSS-1 — `settlementRegistry` immutable (0.5 day)
- ☐ HIGH-7 B-CROSS-3 — `slasher` immutable (0.5 day)
- ☐ Re-run all team PoC tests inverted (each Confirmed-exploit test should now FAIL the exploit). ~1 day.

### Week 2 — A-01 RoyaltyDistributor v2 re-deploy

- ☐ Implement burn + add Pausable in distribute path (1 day)
- ☐ Hardhat tests + parity verification (1 day)
- ☐ Deploy ceremony Base mainnet (0.5 day)
- ☐ Update Forta bot registry (0.25 day)
- ☐ Communications (0.25 day)

### Week 3 — MEDIUM cleanup pass

- ☐ C-INT-02 interface fix
- ☐ B-OWNABLE-1 → `Ownable2Step` migration (7 contracts)
- ☐ D-04 push-payment → pull-payment refactor
- ☐ D-05 per-batch challengeWindow snapshot
- ☐ B-CROSS-2 totalEscrowedBalance accumulator
- ☐ B-TREASURY-1 deploy script canonical-Safe pin
- ☐ LOW + INFO batch (C-INT-03, 04, 06; A-04; D-08)

### Week 4 — Verification + L1 + audit re-run

- ☐ L1 static-analysis tooling pass on remediated code (Slither, Aderyn, Mythril)
- ☐ Optional: L2 multi-team re-run on remediated code (verify no regressions, ~$30–80 in API)
- ☐ Re-tag audit-prep at remediated tip

### Week 5+ — L3 + L4 engagement

- L3 Ed25519 specialist audit (Trail of Bits / NCC) — already shovel-ready
- L4 Code4rena contest scoping with consolidated findings as starting brief

## 9. Cost / time summary

| Phase | Engineering days | Wall-clock | Cost |
|-------|------------------|------------|------|
| Day 0 ops fixes (CRIT-2, HIGH-4) | 0.5 | hours | $0 |
| Week 1 — CRIT + HIGH sprint | ~6 | 1 week | $0 |
| Week 2 — A-01 v2 re-deploy | ~3 | 1 week | ~$50 gas + ceremony |
| Week 3 — MEDIUM cleanup | ~5 | 1 week | $0 |
| Week 4 — L1 + verification | ~2 | partial week | $30–80 (optional L2 re-run) |
| **L2-driven remediation total** | **~16 days** | **~4 weeks** | **<$200** |

This audit pass surfaced **2 CRITICAL + 7 HIGH findings**. Each would have
cost roughly $30K to surface through L4 contest. The L2 audit cost <$100 in
API + ~11 minutes wall-clock per team. **Cost-reduction relative to
contest-only path: ≥600×.**

## 10. What this audit DID NOT cover

- **L3 (Ed25519 internal cryptography):** RFC 8032 line-by-line review of `Ed25519Lib.sol` curve math. Deferred to specialist (Trail of Bits / NCC). Pre-engagement test corpus passes (RFC 8032 §7.1 + FIPS 180-4 KATs all green).
- **L5 (Off-chain Python ML supply chain):** `prsm/inference/`, `prsm/chain_rpc/`, `prsm/streaming/`. Separate audit, separate vendor profile.
- **L6f (Network infrastructure):** Bootstrap node DDoS resilience, transport-layer adapters, DHT poisoning resistance. Pen-test layer.
- **L7 (Economic / game-theory at scale):** Tokenomics modeling, MEV exposure, agent-based simulation. Specialist firm (Gauntlet / Chaos Labs).
- **L8 (Legal / regulatory):** Counsel review.
- **L4 (paid human contest):** Code4rena / Sherlock / solo firm against post-remediation code.

These remain pending per the AUDIT_PLAN gating sequence (`audits/AUDIT_PLAN.md` §6.6).

## 11. Bottom-line

**This is the most productive single hour of pre-mainnet hardening the
project has had.**

- 2 CRITICAL findings, both with validated PoCs.
- 7 HIGH findings, all with validated PoCs.
- 1 of 2 CRITICALs is on **live mainnet** (FTNS admin role) — was a known
  pending item but is now formally documented as a current exposure that
  needs Foundation council attention this week, not next month.
- 1 of 7 HIGHs is **a script bug I introduced last session** — exactly the
  kind of self-blind-spot AI audit teams are uniquely positioned to catch.
- The audit-bundle stack (`#31`, `#40`) **must not deploy** until at least
  CRIT-1 + HIGH-2 + HIGH-3 + HIGH-4 + HIGH-6 + HIGH-7 are remediated.
- All fixes together fit inside ~4 weeks of engineering, well under the L4
  contest's wall-clock and at <$200 cost.

**Recommended next steps (in order):**

1. Foundation council emergency review of CRIT-2 (FTNS admin).
2. Same-day fix of HIGH-4 (verifier script bug).
3. Week-1 engineering sprint per §8 above.
4. Week-2 RoyaltyDistributor v2 re-deploy.
5. Cleanup + L1 + L4-contest scope freeze.

Auditors who want to reproduce: every PoC test is in
`contracts/test/audit-team-{a,b,c,d}/` and runnable via `npx hardhat test`
from the `contracts/` directory. All pass against the production
contracts (not mocks). Pinned commit `589c14d2`.
