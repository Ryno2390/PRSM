# On-Chain Monitoring Vendor Decision: Forta vs. Tenderly

**Date:** 2026-04-26
**Status:** Decision recorded; subject to revisit at Phase 2 (mainnet operational data)
**Decision authority:** Founder (pre-board); Foundation council post-formation
**Related docs:**
- `docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md` §11 — pre-mainnet readiness checklist
- `2026-04-26-exploit-response-operational-annex.md` (private repo) §3.2 — channel readiness
- Risk Register A1-A8 — smart-contract risks this monitoring detects

---

## 1. Decision

**Adopt Forta as primary on-chain monitoring vendor.** Tenderly is deferred to Phase 2 add-on consideration based on operational data from Forta usage during initial mainnet operation.

## 2. Why Forta primary

### 2.1 Decentralization alignment

Forta is itself a decentralized network: detection bots are public packages run by independent scanner-node operators on the Forta network. This aligns with PRSM's Vision §1 thesis ("AI infrastructure that is fundamentally aligned with the democratic process"). PRSM monitoring its own contracts via a centralized SaaS would be a small but visible inconsistency with the protocol's positioning.

### 2.2 Cost structure

- **Forta:** Free for public bots. PRSM detection bots will be public (subscriber-network funding via FORT token). Marginal cost per additional monitored contract: $0.
- **Tenderly:** Tiered SaaS — free tier limited to 5 alerts / 100 actions per month; Pro starts at $200/month, Custom enterprise pricing for full feature set (Web3 Actions, real-time alerting, advanced filtering).
- At PRSM's pre-mainnet scale, free Forta covers full scope. Tenderly's free tier is insufficient for production monitoring of 5+ contracts.

### 2.3 Web3-native developer ergonomics

- Forta bots are TypeScript/JavaScript packages with `handleTransaction` + `handleBlock` handlers. Codebase lives in version control alongside the contracts they monitor — version-pin and review parity with smart-contract code.
- Tenderly alerts are configured via web UI or Tenderly CLI. Configuration drift risk: alert rules can diverge from contract changes if not carefully managed; harder to code-review.
- For PRSM, where every contract change goes through audit + public diff, Forta's "monitoring-as-code" pattern is structurally better.

### 2.4 Open-source extensibility

Forta detection bots are open-source by default. The community can:
- Audit our detection rules for completeness/effectiveness
- Contribute new detectors via pull request
- Run bots independently for redundant coverage
- Fork and customize for their own deployments

This converts monitoring from a centralized vendor relationship into a public good that strengthens with adoption.

### 2.5 Coverage on Base L2

Both Forta and Tenderly support Base. Forta has had Base mainnet support since 2024-Q1 (verified via Forta network documentation and Base ecosystem registry). No coverage gap.

### 2.6 Audit trail and forensic value

Forta findings are public + persistent on the Forta network. Post-incident, forensic analysis can reference all historical findings without depending on a vendor's data retention policy. This is operationally important per Exploit Response Playbook §8 (forensics + evidence preservation).

## 3. Why not Tenderly primary

### 3.1 SaaS lock-in risk

Tenderly operates as a centralized service. If Tenderly suffers an outage, PRSM monitoring goes dark. If Tenderly modifies pricing or terms unfavorably, migration cost is real. Risk Register would need a new entry: "Tenderly service interruption disables monitoring." Forta as a network is structurally more resilient (multiple scanner-node operators).

### 3.2 Cost asymmetry

Tenderly's value-add over Forta is largely in faster alerting (push notification SLA), polished UI, and integrated debugging tools. None of these justify a $200+/month recurring cost during pre-mainnet phase. Post-mainnet, if alert latency becomes operationally critical (e.g., active drain detected at T+30s vs. T+90s), Tenderly add-on consideration becomes valid.

### 3.3 Integration complexity

Tenderly Web3 Actions require API key management + per-action quota tracking. Forta scanner-node infrastructure handles dispatch automatically.

## 4. Why Tenderly is on the table for Phase 2

If/when these conditions hold post-mainnet:
- **Alert latency becomes operationally critical** (Forta's 30-90s typical latency proves insufficient for real-world incident response)
- **Forta scanner-node availability becomes unreliable** in PRSM's deployment regions
- **PRSM scale exceeds free-tier coverage** (>10 monitored contracts, >100 detection rules)
- **Treasury budget supports premium SaaS** (post-Phase-1.3 treasury fee accumulation)

Then Tenderly becomes a candidate for a complementary deployment: Forta as primary with Tenderly as secondary fast-alert layer for the most-critical contracts only (FTNSToken, RoyaltyDistributor, EscrowPool).

## 5. Implementation plan

### 5.1 Phase 1 (this engagement)
- Forta detection bot scaffold at `ops/monitoring/forta-bots/`
- Detectors for top-5 critical contracts (FTNSToken, ProvenanceRegistry, RoyaltyDistributor, BatchSettlementRegistry, EscrowPool)
- Alert routing to war-room Discord channel via webhook
- Documentation for extending detection rules
- Deploy to Forta network as public bot

### 5.2 Phase 2 (post-mainnet, conditional)
- Evaluate Forta latency metrics over first 30 days of mainnet operation
- If latency adequate: continue Forta-only
- If latency inadequate: scope Tenderly add-on for critical-path contracts only

### 5.3 Phase 3 (post-board)
- Foundation council reviews monitoring effectiveness annually
- Per-incident reviews when KPI exceeds threshold per Risk Register §5

## 6. Decision record

**Decision:** Forta primary, Tenderly conditional Phase 2.

**Rationale:** Decentralization alignment + cost structure + monitoring-as-code ergonomics + open-source extensibility outweigh Tenderly's premium-tier alert latency and UI advantages at PRSM's current scale.

**Reversibility:** High. If Forta proves operationally insufficient, migration to Tenderly or hybrid Forta+Tenderly is a 1-2 week engineering effort.

**Approver:** Ryne Schultz (founder), 2026-04-26.

**Reviewer at promotion:** Foundation council post-formation; first review at quarterly post-mainnet checkpoint.

---

## 7. Versioning

- **0.1 (2026-04-26):** Initial decision pre-mainnet. Forta selected; Tenderly tracked for Phase 2 reconsideration.
