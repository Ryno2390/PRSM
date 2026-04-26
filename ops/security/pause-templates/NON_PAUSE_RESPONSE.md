# Non-Pause Response Runbook

**Purpose:** Containment options when an incident affects an **immutable** PRSM contract that has no pause function. Pre-built menu of alternatives so on-call engineer doesn't waste time during P0 looking for a pause that doesn't exist.

---

## Why most PRSM contracts can't be paused

Tokenomics §10 invariants (load-bearing for the entire creator-protection thesis):
- "Total supply cap is hard."
- "Creator royalties are enforced on-chain. No off-chain payment path can bypass the `RoyaltyDistributor`."
- "Burn is irreversible."
- "Foundation cannot mint above cap or re-mint burned tokens."
- "FTNS is distributed only as compensation, never sold by the foundation."

Pause is a foundation-side override mechanism. If `RoyaltyDistributor` had `pause()`, the second invariant would qualify to "creator royalties enforced on-chain unless we pause." That weakens the unconditional guarantee that distinguishes PRSM from custodial platforms.

The architectural commitment is: **for invariant-load-bearing contracts, no pause means no pause-shaped backdoor.** During an incident, the cost is reduced response options. The benefit is the unconditional guarantee remains true even during the incident.

---

## Contracts in this category (9)

| Contract | What it does | Why no pause |
|---|---|---|
| `RoyaltyDistributor` | Atomic split of payments to creators / nodes / treasury | Creator-royalty invariant |
| `EscrowPool` | Pre-paid FTNS held until job settles | Prompter-payment invariant; pause = locked funds |
| `BatchSettlementRegistry` | Receipt commitments + challenge resolution | Challenger-rights invariant |
| `ProvenanceRegistry` | Content-creator binding (immutable on-chain truth) | Provenance invariant |
| `CompensationDistributor` | Phase 8 — pull-based weighted split for compensation | Compensation invariant |
| `Ed25519Verifier` | Signature verification primitive | No state to pause |
| `KeyDistribution` | Phase 7 — Shamir-split decryption keys | Decryption-availability invariant |
| `StakeBond` | Stake locking + unbonding | Staker-rights invariant |
| `StorageSlashing` | Phase 7 — slashing for failed storage proofs | Slashing-fairness invariant |

---

## Response menu by incident type

Choose the response category most aligned with the incident type. Multiple categories may apply simultaneously.

### Category A — Public communication (always do this first)

If pause isn't possible, the FIRST containment lever is **rapid, accurate public disclosure** so users can self-protect.

**Action:** Per Exploit Response Playbook §3 communication protocol, post initial holding statement within 30 minutes:
- Identify affected contract by name and address
- State pause is not available + why (architectural invariant)
- Specify any user actions that reduce exposure
- Commit to next-update ETA

**Pre-drafted template** in private repo `2026-04-26-exploit-response-operational-annex.md` §6.3.

**Effectiveness:** Limits damage by reducing user activity during exploit window. Cannot stop the exploit itself.

### Category B — Pull oracle / data feed support

If exploit depends on a price oracle, semantic-shard CID, or external data feed PRSM controls:

**Action:** Update the gateway / scheduler to stop routing queries that depend on the affected feed.

**Specific levers:**
- Foundation gateway: stop publishing CIDs that route through compromised node-set
- Pricing oracle: rotate to alternate feed (if multi-source)
- Node operator allowlist: temporarily restrict to high-reputation nodes only

**Owner:** Protocol Engineering on-call + foundation council coordination.

**Effectiveness:** Stops new attack flow; existing in-flight transactions complete normally.

### Category C — Bridge / chain-level intervention

If exploit involves the bridge or sequencer-level state:

**Action:** Coordinate with Coinbase / Base team for sequencer-level intervention.

**Reach via:**
- Coinbase compliance contact (private annex)
- Base team incident contact (private annex)

**Effectiveness:** Highest impact but slowest path. Reserve for catastrophic incidents.

### Category D — FTNS-token-level pause (cascade containment)

If incident on immutable contract is severe enough that halting all FTNS transfers is justified:

**Action:** Use `safe-tx/ftns-token-pause.json` template (cascade pause).

**Side effects:**
- Halts ALL FTNS transfers across the entire network
- Stops legitimate flows: creator royalties, escrow operations, staking
- Significant business-disruption signal

**When justified:**
- Active drain on a contract that holds FTNS reserves
- Compromise that allows unauthorized minting via downstream contract
- Existential threat to FTNS supply integrity

**When NOT justified:**
- Localized exploit that doesn't threaten supply integrity
- Recovery can be effected via other means
- Contract is upgradeable and patch is available faster than pause coordination

### Category E — Insurance fund deployment

If exploit succeeds and user funds are lost:

**Action:** Per Exploit Playbook §7, deploy insurance fund to reimburse affected users.

**Threshold:** 4-of-5 governance multi-sig per Q4 ratification.

**Effectiveness:** Doesn't stop exploit; restores affected users financially.

### Category F — Coordinated counterparty freeze

If drained funds are sent to a known venue (CEX, mixer, bridge):

**Action:** Engage on-chain analytics partners (Chainalysis, TRM Labs) and CEX compliance teams for fund tracing + freeze.

**Reach via:** private annex §4.4.

**Effectiveness:** Recovery-oriented; depends on counterparty cooperation.

### Category G — Governance escalation (worst case)

If exploit is severe and immutable contracts can't be remediated in place, the migration paths in `PRSM_Vision.md` §14 contingency layer apply:

- Path 3: Fork to FTNS2 with patched contracts; legacy holders migrate
- Path 5: Full decentralization (foundation dissolves, on-chain DAO continues)

**Threshold:** Foundation council unanimous + community-vote ratification (per Q6 + future DAO activation).

**Effectiveness:** Existential-risk only. Multi-month timeline. Last resort.

---

## Decision matrix

| Incident type | Affected contract category | Recommended response |
|---|---|---|
| Active drain on RoyaltyDistributor | Immutable | Category A (comms) + Category D (token pause) + Category F (counterparty freeze) |
| EscrowPool drain | Immutable | Category A + Category D + Category F + Category E (insurance) |
| Sybil attack on ProvenanceRegistry | Immutable | Category A + Category B (oracle pull — restrict node-set) |
| StorageSlashing exploit (false slash) | Immutable | Category A + Category G (severe — possible fork-to-v2) |
| Bridge double-spend | Mixed (Bridge has pause) | `safe-tx/ftns-bridge-pause.json` + Category C (Coinbase/Base coord) |
| FTNSToken supply manipulation | Has pause | `safe-tx/ftns-token-pause.json` + Category E |
| EmissionController halving bug | Has pause | `safe-tx/emission-controller-pause.json` |

---

## What this runbook does NOT solve

- **Exploits that succeed in seconds before any human can respond.** Forta monitoring (`ops/monitoring/forta-bots/`) detects; this runbook contains. Neither prevents.
- **Compromised cryptographic primitives** (BridgeSecurity / Ed25519Verifier flaws). Those require coordinated upgrade per `safe-tx/bridge-security-upgrade.json` + Phase 7.x re-deploy if applicable.
- **Loss of council multi-sig quorum** during incident. See Exploit Playbook §6 failure-mode escalation.

---

## Pre-incident commitments

- Each immutable contract's incident response category is reviewed during quarterly tabletop exercises
- Public communications templates pre-drafted in private annex
- Counterparty contact list (Coinbase, Base team, on-chain analytics) maintained quarterly
- Insurance fund balance ≥5% of foundation treasury per Vision §14

---

## Versioning

- **0.1 (2026-04-26):** Initial runbook covering 9 immutable contracts + 7 response categories. Tabletop validation pending post-board.
