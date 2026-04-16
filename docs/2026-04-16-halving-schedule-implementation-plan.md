# Halving Schedule Implementation Plan

**Status:** Scope document. Implementation deferred to Phase 8 (new phase). Operational policy guidance effective immediately.

**Context:** PRSM_Tokenomics.md §4 introduces a Bitcoin-style halving schedule as one of four mechanisms producing FTNS's value trajectory. The halving governs the rate at which the foundation distributes FTNS as compensation for network work — with rates halving every 4 years per Epoch. No existing contract implements the halving schedule today. This document specifies how it will be delivered, starting with foundation operational policy and evolving to on-chain strict enforcement.

**Cross-references:**
- Tokenomics: `PRSM_Vision.md` (external) §4 Halving schedule specification
- Regulatory posture: PRSM_Tokenomics §3 (equity-investment architecture) and §9 (Howey analysis)
- Risk Register: C5 (operational compliance), C6 (rate calibration), C7 (governance capture)

## 1. Decision: Hybrid Implementation (Option C)

Three implementation options were considered:

- **Option A: Foundation operational policy only.** Foundation applies halving rates off-chain when distributing from its 100M initial allocation. No contract changes. Simplest but depends on foundation discipline; no trustless enforcement.
- **Option B: On-chain emission contract with strict enforcement.** A new minting contract authorized to mint up to the 1B cap, with halving rates encoded as protocol parameters. Strong trust guarantees but requires meaningful engineering effort and contract audit cycles before deployment.
- **Option C: Hybrid (selected).** Start with Option A immediately; migrate to Option B as a named deliverable in Phase 8. Preserves optionality, minimizes immediate engineering load, matches evolution of successful protocols (Ethereum started with discretionary parameters and migrated to stricter on-chain enforcement over years).

**Decision ratified 2026-04-16** by founder. Hybrid approach is the target.

## 2. Operational Policy (Effective Immediately)

Effective immediately upon foundation formation, the halving schedule is applied via operational discipline rather than contract enforcement:

### 2.1 Policy statement

1. **Epoch tracking.** The foundation publishes an Epoch 1 start date (aligned with PRSM mainnet launch) and a transparent 4-year calendar for subsequent epoch transitions. Epoch transitions happen on known future dates; participants can compute their expected compensation rate for any future date.

2. **Baseline rate publication.** Before mainnet, the foundation board ratifies Epoch 1 baseline compensation rates for:
   - Creator royalty bonuses (FTNS paid per content access, on top of user-paid royalty split)
   - Node operator compensation (FTNS paid per PCU served, on top of user-paid job fees)
   - Contributor grants (FTNS paid per completed work item in foundation-defined categories)
   - Any other foundation-paid compensation streams created post-launch

3. **Halving application.** At each epoch transition (every 4 years), all foundation-controlled compensation rates halve. This is a single administrative action executed by the foundation treasury role via existing token transfer mechanics — no contract upgrade required.

4. **Public logging.** Each epoch transition is publicly logged via on-chain event (or at minimum a foundation-signed attestation published on IPFS and referenced from the foundation website) documenting the pre- and post-halving rates for each compensation stream.

5. **Governance adjustability.** The halving schedule can be modified by foundation governance vote, but any modification must be (a) published at least 90 days before taking effect, (b) justified in writing, (c) subject to stakeholder comment period. Modifications that increase compensation rates are disallowed per the monotone-decreasing invariant in Tokenomics §10 invariant #7 (except in the specific case of pausing rather than increasing; see §2.6).

### 2.2 Out of scope for operational policy

The operational policy does *not* apply to:

- **User-to-user payments.** When a user pays a creator for content access, or pays a node operator for compute, those payments flow per the on-chain royalty/compensation contracts at rates not subject to the halving. The halving applies only to the foundation's own compensation distributions.
- **Protocol fees and burn.** The 2% network fee and 20% burn operate per existing contract logic, unaffected by halving.
- **Staking yield.** Staking rewards derive from the 2% network fee pool per §5.3 and are unaffected by halving.

### 2.3 Trust implications

Under operational-policy-only enforcement, participants must trust the foundation to apply the halving as documented. This is a weaker trust model than strict on-chain enforcement, but comparable to how many established protocols operate (Ethereum's EIP-1559 parameters, Solana's inflation schedule, etc., are all governance-adjustable rather than strictly fixed).

Mitigations that strengthen the operational-only model:

- Public logging (§2.1 #4) creates accountability — any deviation from the schedule is observable.
- Governance 90-day advance notice (§2.1 #5) prevents surprise changes.
- Monotone-decreasing invariant (§2.1 #5) prevents the foundation from using the mechanism inflationarily.
- Risk Register tracking (C5, C6, C7) ensures ongoing operational review.

These mitigations are not equivalent to strict on-chain enforcement but make the operational policy substantially more trustworthy than pure foundation discretion.

### 2.4 Epoch 1 baseline rate calibration

The foundation must set Epoch 1 baseline compensation rates before mainnet. Calibration methodology:

1. **Forward-simulation scenarios.** Run bull/base/bear adoption simulations using Tokenomics §6 scenarios as input. For each scenario, compute:
   - Total FTNS distributed from foundation allocation through Epoch 1 (4 years)
   - Network participation levels (nodes, creators, contributors)
   - Incentive strength (FTNS earned per unit of work × USD value at time of earning)

2. **Acceptance criteria for Epoch 1 rates:**
   - Foundation allocation depletion in Epoch 1 is less than 40% under base case (leaves 60%+ for Epochs 2+)
   - Incentive strength exceeds operator opportunity cost by 2×+ under base case (otherwise participation fails)
   - Epoch 1 distribution under bear case still fits within 60% of allocation (otherwise bear depletion is too fast)

3. **Rate ratification.** Foundation board reviews simulation results and ratifies Epoch 1 rates. Rates published in a foundation governance document 90+ days before mainnet.

**Specific rate values are not set in this document.** They require simulation work with modeled adoption data that is not yet available. Target completion: pre-mainnet foundation governance ratification.

### 2.5 Distribution mechanics (operational)

During Epoch 1, foundation-paid compensation flows via:

1. **Creator royalty bonuses.** When a content access event occurs on-chain, the existing `RoyaltyDistributor` handles the user-to-creator royalty split. In parallel (off-chain or via a separate settlement contract), the foundation computes a bonus amount = `Epoch 1 rate × access count × content access weight` and periodically (monthly) distributes bonuses to creators via bulk transfer.

2. **Node operator compensation.** Similar pattern — the existing compute dispatch contracts handle user-to-node payments for PCUs served; the foundation computes bonuses based on cumulative PCUs served by each node and distributes monthly.

3. **Contributor grants.** Discrete grant decisions per governance-approved programs, paid from foundation treasury in FTNS. Grant sizes are denominated in USD-equivalent but paid in FTNS at the time of grant (analogous to how venture funds pay in their native currency).

### 2.6 Pause provisions

Governance may temporarily pause the halving schedule in specific network-health-emergency scenarios. Pause rationale and duration must be published in advance per §2.1 #5. Pause is *not* the same as rate increase — during pause, rates remain at current level rather than halving to lower level. No pause may result in rates higher than the current epoch level.

## 3. Phase 8: On-Chain Strict Enforcement

Phase 8 delivers on-chain enforcement of the halving schedule. This transitions the trust model from foundation-operational to code-enforced.

### 3.1 Phase 8 scope

Phase 8 introduces new contracts:

1. **`EmissionController.sol`** — a new contract that mints FTNS from a governance-authorized allocation (up to the 1B cap) at rates controlled by the halving schedule. The contract has:
   - Immutable halving schedule (4-year epochs, 0.5 halving factor) — changeable only via upgrade with 90-day advance notice
   - Per-epoch rate limits (cannot exceed current-epoch baseline × epoch_factor)
   - Public, deterministic epoch-rate calculation readable by any caller

2. **`CompensationDistributor.sol`** — a contract that pulls from `EmissionController` and distributes to compensation pools (creator royalty bonus pool, node operator pool, contributor grant pool) according to governance-set weights.

3. **Migration from operational-only.** Foundation operational compensation during Epochs 1-2 (years 0-8) continues from the 100M genesis allocation. Phase 8 contracts authorize new minting beyond 100M up to the 1B cap at halving-controlled rates. The operational policy for the remaining foundation genesis allocation continues in parallel.

### 3.2 Phase 8 timing

Target: **Q4 2028** (end of Epoch 1). This timing provides:

- Adequate time for Phase 1-7 delivery (mainnet, compute dispatch, marketplace, wallet SDK, fiat on-ramp, P2P hardening, content tiering)
- Observational data from Epoch 1 operational policy to inform contract design
- Pressure alignment: Phase 8 ships before Epoch 2's rate halving takes effect, ensuring the first on-chain-enforced halving is a real mechanism rather than a no-op

Earliest viable: Q2 2028 (gives 6-9 months pre-Epoch 2 for audit and integration testing).
Latest acceptable: Q2 2029 (significant slippage past the Epoch 2 transition weakens trust).

### 3.3 Phase 8 scope boundaries

Phase 8 specifically does *not* include:

- Changes to the halving schedule itself (that remains governance-adjustable per §2.1 #5)
- Changes to burn, staking, or royalty distribution mechanics (those are separate contracts)
- Changes to foundation genesis allocation distribution (operational policy continues for that allocation)
- UX changes (separate phase)

Phase 8 is narrowly scoped to new-emission control and distribution.

### 3.4 Phase 8 dependencies

Requires before start:
- Phases 1-7 shipped and stable
- Epoch 1 operational policy has >= 2 years of observational data
- Foundation governance structure finalized (enables authorizing the emission contract)
- Security-audit firms relationship established (Phase 8 is safety-critical; minimum 2 audits)

Requires during execution:
- Formal verification on `EmissionController.sol` mint paths
- Multi-stakeholder testnet exercise before mainnet deployment
- Emergency pause mechanism for Phase 8 contracts tied to multisig governance

## 4. Governance and Safeguards

### 4.1 Governance principles

1. **Monotone-decreasing invariant.** Compensation rates can halve or be paused but never increase. Enforced in operational policy (§2.1 #5) and in contract code (§3.1) once Phase 8 ships.

2. **Advance notice for changes.** All halving schedule changes require 90-day advance notice with public rationale. Applies to both operational-period changes and Phase 8 contract upgrades.

3. **Stakeholder comment period.** Changes subject to a minimum 30-day stakeholder comment period during the 90-day notice window.

4. **Transparent logging.** Every epoch transition, every rate change, every governance decision on halving schedule is logged publicly (on-chain event post-Phase 8, foundation-signed attestation pre-Phase 8).

### 4.2 Safeguards against governance capture

Risk: a faction of governance participants could vote to modify the halving schedule in ways that benefit them at the expense of early adopters.

Mitigations:

- **Monotone-decreasing invariant** prevents inflationary changes (can only halve, never increase).
- **Supermajority threshold** for halving schedule changes (proposed: 75% of governance voting weight, higher than ordinary 60% threshold).
- **Multi-class voting** — if governance structure separates foundation board, user class, creator class, and operator class, halving changes require supermajority in each class.
- **Legitimate kill-switch:** if governance is captured and halving is modified in bad faith, the forkability invariant of the broader protocol (`PRSM_Vision.md` §1 and §2) applies — stakeholders can fork to a version that preserves the original schedule. This is the ultimate safeguard.

### 4.3 Safeguards against operational non-compliance

Risk: during the operational-only period (pre-Phase 8), foundation treasury could fail to apply halving correctly.

Mitigations:

- Public logging creates external accountability.
- Risk Register entry C5 tracks this explicitly; foundation board reviews quarterly.
- Observable discrepancies between published rates and actual distributions trigger escalation per C5 escalation criterion.
- Phase 8 delivery closes this trust gap by moving enforcement on-chain.

## 5. Open Questions (resolved before mainnet)

1. **Epoch 1 rate values.** Requires simulation work; see §2.4.
2. **Compensation pool allocation.** What percentage of foundation's 100M allocation goes to creator royalty bonuses vs. node operator compensation vs. contributor grants vs. operational reserve? Current Tokenomics §3.4 provides ranges (30-40% / 25-35% / 15-20% / 10-15% / 5-10% for Prismatica). Precise percentages require board ratification.
3. **Grant program structure.** Specific mechanics of contributor grants (application process, evaluation criteria, vesting) need foundation governance definition pre-mainnet.
4. **Emergency-pause criteria.** Specific conditions under which governance can pause the halving schedule per §2.6. Needs codification.

## 6. Timeline

| Milestone | Target | Notes |
|---|---|---|
| Document ratified by founder | 2026-04-16 | Done |
| Risk Register entries C5/C6/C7 added | 2026-04-16 | Coordinated with this document |
| Tokenomics §4.2 clarification | 2026-04-16 | Coordinated with this document |
| Epoch 1 rate simulation work begins | Post-mainnet, pre-foundation-formation | Requires modeled adoption data |
| Epoch 1 rates ratified by foundation board | Pre-mainnet + 90 days notice | Published for stakeholder review |
| Mainnet launch + Epoch 1 start | Targeted per PRSM_Vision roadmap | Existing Phase 1 milestone |
| Epoch 1 → Epoch 2 halving | Mainnet + 4 years | Applied via operational policy |
| Phase 8 design and scoping | Q2-Q3 2028 | Precedes implementation |
| Phase 8 implementation | Q4 2028 target | On-chain enforcement live |
| Epoch 2 → Epoch 3 halving | Mainnet + 8 years | First halving under strict on-chain enforcement |

## 7. Changelog

- **2026-04-16:** Document created; hybrid approach (Option C) ratified; coordinated with Risk Register C5/C6/C7 additions and Tokenomics §4.2 clarification.
