# PRSM Documentation Index

**Last updated:** 2026-04-22
**Purpose:** Navigable map of the 89 docs in this directory for auditors, Foundation officers, technical partners, research partners, investors, and engineers joining the project.

If you don't know where to start:

- **Investors / partners** — [`2026-04-22-prsm-investor-executive-summary.md`](./2026-04-22-prsm-investor-executive-summary.md) (executive summary) + [`2026-04-22-session-summary-engineering-sprint.md`](./2026-04-22-session-summary-engineering-sprint.md) (recent engineering changelog).
- **Auditors** — [`2026-04-10-audit-gap-roadmap.md`](./2026-04-10-audit-gap-roadmap.md) (master roadmap) + per-phase audit-prep docs in §2 below.
- **Foundation officers** — governance charters in §4 + ops runbooks in §2.
- **Research partners** — R-track scoping docs in §3.
- **New engineers** — [`GETTING_STARTED.md`](./GETTING_STARTED.md) + [`architecture.md`](./architecture.md) + [`DEVELOPMENT_GUIDE.md`](./DEVELOPMENT_GUIDE.md) in §5.

---

## 1. Master roadmap

| Doc | Purpose |
|---|---|
| [`2026-04-10-audit-gap-roadmap.md`](./2026-04-10-audit-gap-roadmap.md) | **Master roadmap.** Single source of truth for phase status, scope, and planning-artifact pointers. Updated on every phase milestone. |
| [`glossary.md`](./glossary.md) | Terminology + disambiguation reference (tier systems, phase numbering, Prismatica vs Foundation). |

---

## 2. Phase plans + execution artifacts

Phases ordered by number. Each row: design/plan doc + status + any follow-on runbooks / audit-prep docs.

### Phase 1 / 1.3 — on-chain provenance + royalty distribution

| Doc | Purpose | Status |
|---|---|---|
| [`2026-04-10-phase1-onchain-provenance-plan.md`](./2026-04-10-phase1-onchain-provenance-plan.md) | Phase 1 design + TDD plan | Shipped |
| [`2026-04-11-phase1.3-completion-plan.md`](./2026-04-11-phase1.3-completion-plan.md) | Phase 1.3 completion plan | Engineering complete; mainnet deploy hardware-gated |
| [`2026-04-11-phase1.3-task0-findings.md`](./2026-04-11-phase1.3-task0-findings.md) | Pre-execution audit findings | Historical |
| [`2026-04-11-phase1.3-sepolia-bakein-log.md`](./2026-04-11-phase1.3-sepolia-bakein-log.md) | 7-day Sepolia bake-in log | Passed; mainnet deploy pending |
| [`ONCHAIN_PROVENANCE.md`](./ONCHAIN_PROVENANCE.md) | Operator-facing reference | Reference |
| [`FTNS_TESTNET_DEPLOYMENT.md`](./FTNS_TESTNET_DEPLOYMENT.md) | Testnet deploy procedure | Reference |

### Phase 2 / 2.1 — remote compute dispatch + confidential compute

| Doc | Purpose | Status |
|---|---|---|
| [`2026-04-12-phase2-remote-compute-design.md`](./2026-04-12-phase2-remote-compute-design.md) | Phase 2 design | Shipped |
| [`2026-04-12-phase2-remote-compute-plan.md`](./2026-04-12-phase2-remote-compute-plan.md) | Phase 2 TDD plan | Shipped |
| [`CONFIDENTIAL_COMPUTE_SPEC.md`](./CONFIDENTIAL_COMPUTE_SPEC.md) | Confidential compute spec (Rings 7-10) | Reference |

### Phase 3 / 3.1 — marketplace matching engine + batch settlement

| Doc | Purpose | Status |
|---|---|---|
| [`2026-04-14-phase3-preplanning.md`](./2026-04-14-phase3-preplanning.md) | Original 2026-04-14 preplanning stub | **HISTORICAL** — see status doc below |
| [`2026-04-20-phase3-marketplace-design.md`](./2026-04-20-phase3-marketplace-design.md) | Phase 3 marketplace design | Shipped |
| [`2026-04-20-phase3-marketplace-plan.md`](./2026-04-20-phase3-marketplace-plan.md) | Phase 3 TDD plan | Shipped |
| [`2026-04-21-phase3.1-batch-settlement-design.md`](./2026-04-21-phase3.1-batch-settlement-design.md) | Phase 3.1 batch settlement design | Shipped |
| [`2026-04-22-phase3-status-and-forward-plan.md`](./2026-04-22-phase3-status-and-forward-plan.md) (**PRSM-PHASE3-STATUS-1**) | Current Phase 3 delivery status + Phase 3.x follow-on scope (MCP completion / liquidity / operator toolkit) | Supersedes preplanning stub |

### Phase 4 — wallet SDK + consumer onboarding

| Doc | Purpose | Status |
|---|---|---|
| [`2026-04-22-phase4-wallet-sdk-design-plan.md`](./2026-04-22-phase4-wallet-sdk-design-plan.md) | Phase 4 design + TDD plan | Backend Tasks 1, 2, 5 + E2E shipped; frontend 3, 4 gated |
| [`2026-04-22-phase4-wallet-vendor-decision.md`](./2026-04-22-phase4-wallet-vendor-decision.md) (**PHASE4-WALLET-VENDOR-1**) | §8.1 vendor decision — Privy conditional + G1-G6 green-light checklist | Partner-handoff ready |

### Phase 5 — fiat on-ramp + KYC

| Doc | Purpose | Status |
|---|---|---|
| [`2026-04-22-phase5-fiat-onramp-design-plan.md`](./2026-04-22-phase5-fiat-onramp-design-plan.md) | Phase 5 design + TDD plan | Backend Task 5 + Tasks 2-4 adapter scaffolds shipped |

### Phase 6 — P2P network hardening

| Doc | Purpose | Status |
|---|---|---|
| [`2026-04-22-phase6-p2p-hardening-design-plan.md`](./2026-04-22-phase6-p2p-hardening-design-plan.md) | Phase 6 design + TDD plan | `phase6-merge-ready-20260422` |
| [`2026-04-22-phase6-task2-bootstrap-ops-runbook.md`](./2026-04-22-phase6-task2-bootstrap-ops-runbook.md) (**PHASE6-TASK2-BOOTSTRAP-OPS-1**) | Foundation bootstrap-node ops runbook | Partner-handoff ready |
| [`2026-04-22-phase6-task4-dht-tuning-plan.md`](./2026-04-22-phase6-task4-dht-tuning-plan.md) (**PHASE6-TASK4-DHT-TUNING-1**) | DHT parameter-sweep measurement plan | Partner-handoff ready |
| [`BOOTSTRAP_DEPLOYMENT_GUIDE.md`](./BOOTSTRAP_DEPLOYMENT_GUIDE.md) | Operator-facing bootstrap reference | Reference |

### Phase 7 — verification triad (compute-side)

| Doc | Purpose | Status |
|---|---|---|
| [`2026-04-21-phase7-staking-slashing-design.md`](./2026-04-21-phase7-staking-slashing-design.md) | Phase 7 Tier C stake + slash design | `phase7-merge-ready-20260421` |
| [`2026-04-21-phase7-audit-prep.md`](./2026-04-21-phase7-audit-prep.md) | Phase 7 external-audit prep | Ready |
| [`2026-04-21-phase7.1-redundant-execution-design.md`](./2026-04-21-phase7.1-redundant-execution-design.md) | Phase 7.1 Tier B k-of-n redundant execution design | `phase7.1-merge-ready-20260421` |
| [`2026-04-21-phase7.1-audit-prep.md`](./2026-04-21-phase7.1-audit-prep.md) | Phase 7.1 audit prep | Ready |
| [`2026-04-22-phase7.1x-audit-prep.md`](./2026-04-22-phase7.1x-audit-prep.md) | Phase 7.1x pre-audit hardening summary | `phase7.1x-merge-ready-20260422-2` |
| [`2026-04-21-audit-bundle-coordinator.md`](./2026-04-21-audit-bundle-coordinator.md) | Bundled audit engagement pattern | Reference |
| [`2026-04-23-testnet-rehearsal-plan.md`](./2026-04-23-testnet-rehearsal-plan.md) | Mainnet deploy rehearsal — bundled deploy scripts + local hardhat dry-run | Rehearsal infra live; hardware day is mechanical |

### Phase 7-storage — content durability + confidentiality

| Doc | Purpose | Status |
|---|---|---|
| [`2026-04-22-phase7-storage-design-plan.md`](./2026-04-22-phase7-storage-design-plan.md) | Combined §7.1 + §7.2 design + TDD plan | `phase7-storage-merge-ready-20260422` |

### Phase 8 — on-chain halving enforcement

| Doc | Purpose | Status |
|---|---|---|
| [`2026-04-16-halving-schedule-implementation-plan.md`](./2026-04-16-halving-schedule-implementation-plan.md) | Original halving scope doc | Superseded by design plan |
| [`2026-04-22-phase8-design-plan.md`](./2026-04-22-phase8-design-plan.md) | Phase 8 design + TDD plan | `phase8-engineering-complete-20260422` (Tasks 5-9 non-engineering) |

---

## 3. Research track (R1-R8)

Every item partner-handoff-ready as of 2026-04-22.

| Doc | Identifier | Scope |
|---|---|---|
| [`2026-04-14-phase4plus-research-track.md`](./2026-04-14-phase4plus-research-track.md) | — | R-track index; stubs superseded by scoping docs below |
| [`2026-04-22-r1-fhe-inference-scoping-doc.md`](./2026-04-22-r1-fhe-inference-scoping-doc.md) | **R1-SCOPING-1** | FHE for private inference — scheme selection + selective-layer + composition |
| [`2026-04-22-r2-mpc-scoping-doc.md`](./2026-04-22-r2-mpc-scoping-doc.md) | **R2-SCOPING-1** | MPC for sharded inference |
| [`2026-04-22-r3-threat-model.md`](./2026-04-22-r3-threat-model.md) | **R3-TM-1** | Activation-inversion threat model + red-team methodology |
| [`2026-04-22-prsm-supply-1-supply-diversity-standard.md`](./2026-04-22-prsm-supply-1-supply-diversity-standard.md) | **PRSM-SUPPLY-1** | R4 promoted to governance standard (supply-cap + diversity bonuses) |
| [`2026-04-22-r5-tier-c-hardening-scoping-doc.md`](./2026-04-22-r5-tier-c-hardening-scoping-doc.md) | **R5-SCOPING-1** | Tier C content-confidentiality hardening beyond Shamir + AES |
| [`2026-04-22-r6-pq-signatures-watch-memo.md`](./2026-04-22-r6-pq-signatures-watch-memo.md) | **R6-WATCH-1** | Post-quantum signatures trigger-watch memo |
| [`2026-04-22-r7-benchmark-plan.md`](./2026-04-22-r7-benchmark-plan.md) | **R7-BENCH-1** | KV / activation compression benchmark plan |
| [`2026-04-22-r8-defense-stack-composition.md`](./2026-04-22-r8-defense-stack-composition.md) | **R8-COMP-1** | Defense-stack composition analysis (5-layer × 14-threat matrix) |

---

## 4. Governance + standards

| Doc | Identifier | Scope |
|---|---|---|
| [`2026-04-21-prsm-gov-1-foundation-governance-charter.md`](./2026-04-21-prsm-gov-1-foundation-governance-charter.md) | **PRSM-GOV-1** | Foundation governance charter |
| [`2026-04-21-prsm-tok-1-ftns-tokenomics.md`](./2026-04-21-prsm-tok-1-ftns-tokenomics.md) | **PRSM-TOK-1** | FTNS tokenomics specification |
| [`2026-04-22-prsm-supply-1-supply-diversity-standard.md`](./2026-04-22-prsm-supply-1-supply-diversity-standard.md) | **PRSM-SUPPLY-1** | Operator-supply diversity standard |
| [`2026-04-21-prsm-cis-1-confidential-inference-silicon.md`](./2026-04-21-prsm-cis-1-confidential-inference-silicon.md) | **PRSM-CIS-1** | Confidential inference silicon standard |
| [`2026-04-19-confidential-inference-silicon-standard.md`](./2026-04-19-confidential-inference-silicon-standard.md) | — | Predecessor CIS draft |
| [`2026-04-14-hybrid-tokenomics-legal-tracking.md`](./2026-04-14-hybrid-tokenomics-legal-tracking.md) | — | Hybrid tokenomics legal/governance workstream tracker |
| [`2026-04-09-v1.6-scope-alignment-design.md`](./2026-04-09-v1.6-scope-alignment-design.md) | — | v1.6 scope-alignment design doc (historical; pre-pivot) |

---

## 5. Investor / partner communications

| Doc | Purpose |
|---|---|
| [`2026-04-21-prsm-investor-executive-summary.md`](./2026-04-21-prsm-investor-executive-summary.md) | Prior executive summary (Phase 3 as most-recent milestone) |
| [`2026-04-22-prsm-investor-executive-summary.md`](./2026-04-22-prsm-investor-executive-summary.md) | Current executive summary (through Phase 7 + 7.1 + 7.1x) |
| [`2026-04-22-session-summary-engineering-sprint.md`](./2026-04-22-session-summary-engineering-sprint.md) | 2026-04-22 engineering-sprint changelog (Phase 4/5/6/7-storage/8 sprint) |
| [`2026-04-21-prsm-economic-model-white-paper.md`](./2026-04-21-prsm-economic-model-white-paper.md) | Economic-model white paper (FTNS pricing, settlement, supply) |
| [`SCIENCE_FIRST_MEDIA_KIT.md`](./SCIENCE_FIRST_MEDIA_KIT.md) | Science-first framing media kit |

---

## 6. Engineering reference

### 6.1 Getting started + onboarding

| Doc | Purpose |
|---|---|
| [`GETTING_STARTED.md`](./GETTING_STARTED.md) | Top-of-funnel setup |
| [`quickstart.md`](./quickstart.md) | Quickstart guide |
| [`MACOS_SETUP.md`](./MACOS_SETUP.md) | macOS-specific setup |
| [`SECURE_SETUP.md`](./SECURE_SETUP.md) | Secure setup for production-adjacent dev |
| [`CONTRIBUTOR_ONBOARDING.md`](./CONTRIBUTOR_ONBOARDING.md) | Contributor journey |
| [`CONTRIBUTOR_SYSTEM_SUMMARY.md`](./CONTRIBUTOR_SYSTEM_SUMMARY.md) | Contribution-system overview |
| [`CURATED_GOOD_FIRST_ISSUES.md`](./CURATED_GOOD_FIRST_ISSUES.md) | Starter issues |
| [`PARTICIPANT_GUIDE.md`](./PARTICIPANT_GUIDE.md) | End-user participant guide |

### 6.2 Architecture + design

| Doc | Purpose |
|---|---|
| [`architecture.md`](./architecture.md) | System architecture |
| [`TECH_CHOICES.md`](./TECH_CHOICES.md) | Tech-stack decisions + rationale |
| [`libp2p-transport-design.md`](./libp2p-transport-design.md) | libp2p transport design |
| [`libp2p-compute-storage-wiring.md`](./libp2p-compute-storage-wiring.md) | libp2p-to-compute/storage wiring |
| [`native-storage-design.md`](./native-storage-design.md) | Native storage design |
| [`SOVEREIGN_EDGE_AI_SPEC.md`](./SOVEREIGN_EDGE_AI_SPEC.md) | Sovereign-edge AI architectural spec |
| [`ai-integration.md`](./ai-integration.md) | AI integration patterns |

### 6.3 API / CLI / SDK

| Doc | Purpose |
|---|---|
| [`API_REFERENCE.md`](./API_REFERENCE.md) | API reference |
| [`API_VERSIONING_GUIDE.md`](./API_VERSIONING_GUIDE.md) | API versioning policy |
| [`CLI_REFERENCE.md`](./CLI_REFERENCE.md) | CLI reference |
| [`SDK_DEVELOPER_GUIDE.md`](./SDK_DEVELOPER_GUIDE.md) | SDK developer guide |
| [`SDK_DOCUMENTATION_ENHANCEMENTS.md`](./SDK_DOCUMENTATION_ENHANCEMENTS.md) | SDK docs backlog |
| [`FTNS_API_DOCUMENTATION.md`](./FTNS_API_DOCUMENTATION.md) | FTNS API reference |

### 6.4 Development + operations

| Doc | Purpose |
|---|---|
| [`DEVELOPMENT_GUIDE.md`](./DEVELOPMENT_GUIDE.md) | Local dev workflow |
| [`configuration.md`](./configuration.md) | Configuration reference |
| [`DEPENDENCY_COMPATIBILITY.md`](./DEPENDENCY_COMPATIBILITY.md) | Dependency compatibility matrix |
| [`OPERATOR_GUIDE.md`](./OPERATOR_GUIDE.md) | Operator-facing guide |
| [`PRODUCTION_OPERATIONS_MANUAL.md`](./PRODUCTION_OPERATIONS_MANUAL.md) | Production ops manual |
| [`EXCEPTION_HANDLING_GUIDELINES.md`](./EXCEPTION_HANDLING_GUIDELINES.md) | Exception-handling patterns |
| [`TROUBLESHOOTING.md`](./TROUBLESHOOTING.md) + [`TROUBLESHOOTING_GUIDE.md`](./TROUBLESHOOTING_GUIDE.md) | Troubleshooting references |
| [`IMPLEMENTATION_STATUS.md`](./IMPLEMENTATION_STATUS.md) | Implementation status (see also master roadmap in §1) |
| [`TEST_SUITE_STATUS.md`](./TEST_SUITE_STATUS.md) | Test suite status |
| [`PUBLISHING_TESTPYPI.md`](./PUBLISHING_TESTPYPI.md) | TestPyPI publish procedure |

### 6.5 Security

| Doc | Purpose |
|---|---|
| [`SECURITY_HARDENING.md`](./SECURITY_HARDENING.md) | Security-hardening reference |
| [`SECURITY_HARDENING_CHECKLIST.md`](./SECURITY_HARDENING_CHECKLIST.md) | Checklist companion |
| [`SECURITY_CONFIGURATION_AUDIT.md`](./SECURITY_CONFIGURATION_AUDIT.md) | Configuration audit |
| [`PENETRATION_TESTING_GUIDE.md`](./PENETRATION_TESTING_GUIDE.md) | Penetration-testing guide |
| [`REMEDIATION_HARDENING_MASTER_PLAN.md`](./REMEDIATION_HARDENING_MASTER_PLAN.md) | Remediation-hardening master plan |

---

## 7. Conventions

### 7.1 Doc identifier format

Dated planning artifacts use `YYYY-MM-DD-slug.md`. Load-bearing docs also carry a stable identifier (e.g., `PRSM-GOV-1`, `PRSM-PHASE3-STATUS-1`, `PHASE4-WALLET-VENDOR-1`) listed in the table above. Identifiers are stable across doc revisions; date + filename may change, identifier does not.

### 7.2 Status markers

Status columns use:
- **Shipped** — engineering complete; merge-ready tag placed.
- **Merge-ready** — engineering complete; awaiting audit / ops / governance for ship.
- **Partner-handoff ready** — scoping doc complete; partner engagement can begin.
- **Reference** — ongoing-reference doc, no status lifecycle.
- **Historical** — superseded by a named successor doc.

### 7.3 Adding to this index

New planning artifacts go in §2 (by phase) or §3 (research track). New governance standards go in §4. New reference docs go in §6. Update the row + the INDEX's `Last updated` header.

---

## 8. Changelog

- **2026-04-22:** initial index covering 89 docs. Adds navigation structure for partner / auditor / Foundation-officer / investor / engineer audiences. Covers master roadmap, phase plans, research track, governance standards, investor comms, engineering reference.
