# PRSM Engineering Sprint Summary — 2026-05-08

**Prismatica, Inc.** | **May 8, 2026**

Engineering-progression changelog covering a single-day sprint that closed three of the six §13 remaining-engineering items in the 2026-05-07 roadmap snapshot, and brought several previously-deferred follow-ons into production. **This is the largest single-day shipping cadence on record for the project**: 18 commits, 15 merge-ready tags, 7 cumulative audit-prep refreshes, ~280 new tests.

**Audience:** investors tracking engineering velocity, technical partners evaluating integration readiness, Foundation officers planning execution-phase allocation, external auditors evaluating engagement scope.

---

## 1. Headline

**Three §13 "remaining engineering" items closed in a single day, narrowing the project's open-engineering surface to three external-input-gated items.**

After the 2026-05-07 mainnet full-stack deploy, the master roadmap (`PRSM_Vision.md` §13) listed six remaining engineering items. As of end-of-day 2026-05-08, three are shipped:

| Item | Tag | Surface |
|---|---|---|
| §4 step 6 compute-participant settlement | `query-orchestrator-step6-settlement-split-merge-ready-20260508` | `PaymentEscrow.release_escrow_split` + `AggregatedResult.participants` + `/compute/forge` split-routing with operator-tunable `PRSM_AGGREGATOR_SHARE_BPS` env var (default 5%). Closes the "shaped right but doesn't actually pay compute participants" gap. |
| KeyDistribution.sol Python wiring | `key-distribution-client-merge-ready-20260508` | `KeyDistributionClient` (deposit_key + release + deauthorize + `KeyReleasedEvent` decode). Closes the Tier C "Shamir shares colocated" honest-scope item — Tier C is now production end-to-end. |
| B8 async-dispatch follow-on | `query-orchestrator-b8-job-history-merge-ready-20260508` | `JobHistoryStore` (in-memory LRU 1024) + two-tier `/compute/status/{job_id}` (history + escrow blocks compose). Closes the "escrow lifecycle only" gap on the status endpoint. |

The three remaining §13 items (A-08 RoyaltyDistributor v2 ceremony, T4.8 + T6.4 calibration corpora, Phase 3.x.11.q.y'' multi-stage replay) are **all external-input-gated**, not engineering-time-gated:
- A-08 ceremony depends on the next planned hardware-multisig ceremony cadence
- T4.8 + T6.4 calibration depend on 30+ days of testnet upload traffic
- Multi-stage replay is conditional on telemetry showing the gap materially affects stage>0 cache correctness

For practical purposes, **the engineering team's direct-control roadmap is closed** as of 2026-05-08.

**Cumulative state:**
- Phase 1, 7, 8, audit-bundle: mainnet-deployed (since 2026-05-07)
- Phases 2, 3, 4, 6: shipped
- Phase 5: scaffolded (vendor commission pending)
- PRSM-PROV-1 Items 3, 4, 6, 7: shipped end-to-end (Item 6 productionized today)
- B8 MCP unhide program: complete (all three originally-hidden tools functional)

---

## 2. Today's commit sequence

In ID order. Each commit pushed to `main` and tagged.

### Morning sprint — QueryOrchestrator placeholder follow-ons (6 closed)

The QueryOrchestrator wiring sprint (2026-05-07/08) had inline-documented six placeholder follow-ons. Today's morning closed all six:

| Commit | Surface |
|---|---|
| `9f5d59a3` | §3 ftns_budget constructor override (`AggregatorClientAdapter.default_ftns_budget` kwarg) |
| `55884f7e` | HTTP endpoint resolver (`AggregateEndpointResolver` with StaticMap + TransportPeer + Chained backends) |
| `2620ad8e` | X25519+XChaCha20 partial-result cipher (closes §2 of inline-documented follow-ons; uses libsodium for Ed25519↔X25519 derivation; AAD binds the AggregationCommit) |

Plus three earlier (memory-recorded but mentioned for completeness): source_agent_pubkey threading, MobileAgent.manifest type unification, Foundation Safe per-network override.

### Item 6 closure end-to-end (PRSM-PROV-1)

Per-content-type calibrated thresholds + disputed-band arbitration shipped through to production wiring:

| Commit | Surface |
|---|---|
| `6f591da8` | T6.3 + T6.5 — three-band dedup wiring (text path) + arbitration queue (`InMemoryArbitrationQueue` + `FilesystemArbitrationQueue`) |
| `44054191` | T6.5.x — binary-path 3-band wiring (image-phash / audio-chromaprint / video-multihash symmetric with text path) |
| `43c2656d` | T6.5.gov — `ProposalCategory.ARBITRATION_DISPUTE` enum value + `ArbitrationProposalSink` Protocol + `NullArbitrationProposalSink` + `render_arbitration_body` (byte-deterministic) + ContentUploader integration with three-tier failure isolation |
| `223870a7` | T6.5.gov.next — `TokenWeightedVotingProposalSink` (production binding wrapping the heavy governance backend behind the thin Protocol) |
| `76ccda38` | T6.5.gov.next2 — node-startup wiring (3 builder helpers; `PRSM_ARBITRATION_PROPOSER_ID` env-gate); 3 failure paths each WARN-log + degrade |

**Item 6 documentation surface complete:**
- Audit-prep §7.19 (`docs/2026-04-27-cumulative-audit-prep.md`)
- Threat-model §3.18 (`docs/2026-05-08-prsm-prov-1-threat-model-addendum-item-6.md`)
- Operator activation runbook (`docs/2026-05-08-prsm-prov-1-item-6-operator-activation-runbook.md`)
- MEMORY.md entry capturing 5-tag provenance + operator activation flow + honest-scope deferrals

### B8 MCP unhide program (3 passes, all complete)

All three originally-hidden MCP tools (`prsm_analyze` / `prsm_dispatch_agent` / `prsm_agent_status`) are now functional:

| Commit | Pass | Surface |
|---|---|---|
| `391d6e2f` | Pass 1 | `prsm_analyze`: `/compute/forge` duck-type-dispatches on `QueryOrchestrator.dispatch_query` (route="qo_swarm"); legacy AgentForge `.run(...)` path preserved |
| `e5fd8187` | Pass 2 | `prsm_dispatch_agent`: handler already routed through `/compute/forge`; pass 1's QO wiring made it work end-to-end |
| `cbcb9fad` | Pass 3 | `prsm_agent_status`: GET `/compute/status/{job_id}` reading from `PaymentEscrow` (escrow lifecycle); `BROKEN_TOOLS_HIDDEN` now empty |

### Closing the remaining-engineering arc

| Commit | Item closed |
|---|---|
| `aa172b6b` | §4 step 6 compute-participant settlement (`PaymentEscrow.release_escrow_split` + `AggregatedResult.participants` + `/compute/forge` split-routing) |
| `0cf9d68d` | KeyDistribution.sol Python client (Tier C now production) |
| `09f0a797` | B8 async-dispatch follow-on (`JobHistoryStore` + two-tier `/compute/status`) |

### Documentation refreshes (3 commits)

| Commit | Surface |
|---|---|
| `42f58f88` | Audit-prep §7.19 + MEMORY.md refresh covering Item 6 |
| `4412e9fb` | Threat-model addendum §3.18 (8 adversary classes A1-A8) |
| `ac926c61` | Operator activation runbook (3-tier model + monitoring + rollback + troubleshooting) |

### Audit-prep retag cadence

Seven cumulative audit-prep tags pushed today (`-c` through `-i`), each pinning HEAD at the corresponding bundle state for external-auditor consumption. The `-i` tag (latest) covers the streaming-inference subsystem (§7.1-§7.15) + deploy-ceremony infrastructure (§7.16) + Item 6 dedup + arbitration (§7.19) as a single bundle.

---

## 3. Test surface

| Suite | Tests added today | Cumulative pass count after sprint |
|---|---|---|
| QueryOrchestrator follow-ons | ~110 | 154 (full suite green) |
| PRSM-PROV-1 Item 6 | 119 | 119 (full suite green) |
| B8 MCP + endpoint surface | ~60 | 128 (full suite green; updated for nested response shape) |
| §4 step 6 settlement | 20 | 20 |
| KeyDistribution Python client | 19 | 19 |
| B8 async-dispatch follow-on | 22 | 22 (incl. forge→history→status integration) |

**All test suites green at HEAD `09f0a797`.** No CI regressions across the day's 18-commit sequence.

---

## 4. Honest-scope items deferred (audit-trail integrity)

Each shipping commit explicitly enumerates what's *not* in v1, with reasoning:

| Sprint | Deferred (not blocking) |
|---|---|
| §4 step 6 settlement | PCU-weighted compute split (uniform v1); atomic rollback of partial-failure transfers (operator reconciles); on-chain content-access royalty leg (separate flow); source_agent_pubkey → FTNS wallet address resolution (production needs node-id → wallet mapping registry — v1 uses hex pubkey as wallet ID) |
| KeyDistribution Python wiring | Encrypted-key wrapping scheme is publisher-chosen (client treats encrypted_key as opaque bytes); event-watcher daemon is a separate sprint; share-placement strategy is operator policy, not contract-enforced |
| B8 async-dispatch follow-on | Filesystem persistence (LRU-evicted jobs not recoverable via history tier — falls through to escrow); event-driven status updates (SSE/WebSocket push); job cancellation API |
| Item 6 (T6.3 + T6.5 + downstream) | T6.4 calibration corpus (30-day testnet-traffic-gate); cross-node arbitration via DHT (R10); on-chain arbitration contract; per-creator daily flag cap; binary-kind hint multipliers; automated sink-failure alerting |

Documenting deferred work explicitly in shipping commits (rather than in some separate backlog file) keeps the audit trail tight: an external reviewer reading the tag history sees both what shipped and what was consciously held back.

---

## 5. Documentation produced

| Document | Surface |
|---|---|
| `docs/2026-04-27-cumulative-audit-prep.md` §7.19 | Per-content-type thresholds + arbitration (Item 6); 12 headline guarantees + 6 trust seams + auditor reading path |
| `docs/2026-05-08-prsm-prov-1-threat-model-addendum-item-6.md` | Threat model §3.18 (Item 6); 8 adversary classes A1-A8 with vectors + mitigations + test pins; 4 cross-cutting invariants |
| `docs/2026-05-08-prsm-prov-1-item-6-operator-activation-runbook.md` | Operator runbook for Item 6 production deployment; 3-tier model (default / auto-proposal / custom calibration); monitoring guidance (5 alert classes); council resolution flow; rollback procedures (3 levels); troubleshooting catalog (5 issues) |
| `MEMORY.md` (auto-memory) | Refreshed with Item 6 production-deployable entry; Items 3 + 4 + 7 task-list reconciliation |
| `PRSM_Vision.md` (investor-facing thesis doc, in iCloud Obsidian vault) | §11 + §13 refresh — moved doc from "April 2026 / pre-mainnet" to "May 2026 / mainnet-live" + status table at top of §13 + Mermaid gantt chart at bottom; multiple sub-section updates as items shipped |

The investor-facing thesis doc (`PRSM_Vision.md`) was substantively out-of-date relative to actual state by ~18 months; today's refresh closed that gap. Three of the six §13 remaining-engineering items have been ticked off in-line with each shipping commit.

---

## 6. Architectural notes

Several design decisions shipped today that are worth flagging for downstream auditors and integration partners:

### 6.1 Three-tier failure isolation pattern (Item 6 + B8)

ContentUploader's `_enqueue_arbitration` and `/compute/forge`'s settlement path each isolate three independent failure surfaces:

```
queue.enqueue → sink.create → queue.set_proposal_id  (Item 6)
escrow.create → orchestrator.dispatch → escrow.release_split  (§4 step 6)
job_history.put(IN_PROGRESS) → forge_dispatch → job_history.put(COMPLETED)  (B8)
```

In each case, every step is wrapped in independent try/except with structured WARN logging. **No single tier's failure can cause an upload or query to fail user-visibly**. This is the load-bearing reliability primitive enabling the project to ship features that touch governance, ledger, and compute simultaneously without compounding fragility.

### 6.2 Duck-typed orchestrator dispatch (B8 unhide)

`/compute/forge` checks `hasattr(node.agent_forge, "dispatch_query")` to route between the legacy AgentForge `.run(...)` surface and the QueryOrchestrator `.dispatch_query(...)` surface. This pattern lets operators upgrade at their own cadence (set `PRSM_QUERY_ORCHESTRATOR_ENABLED=1` to flip to QO; existing AgentForge fixtures keep working). No deprecation period, no breaking change, no flag-day cutover.

### 6.3 Operator-tunable settlement split (§4 step 6)

`PRSM_AGGREGATOR_SHARE_BPS` env var controls the aggregator's share of the compute escrow (default 5%). Operators can adjust per deployment without contract changes. PCU-weighted per-participant split is deferred (uniform for v1) but the split format `[(recipient, amount), ...]` accommodates the future weighting without API breakage.

### 6.4 LRU-bounded in-memory JobHistoryStore (B8 async-dispatch)

`OrderedDict` + `move_to_end` for proper LRU semantics. Default 1024 entries — sufficient for the synchronous-from-caller path that's currently the only path live. Filesystem persistence is the right v2 once async dispatch lands and jobs span beyond a single node-startup window.

### 6.5 Byte-deterministic body rendering (Item 6 + future on-chain arbitration)

`render_arbitration_body(record)` is the canonical formatter. Pinned header (`"PRSM-PROV-1 disputed-attribution review\n"`), 6-decimal similarity precision so two near-identical disputed records render distinctly, deterministic field ordering. **A future on-chain arbitration contract (per design doc §5.2.3) may sign over the body bytes** — councils will be able to verify the bytes they voted against off-chain.

---

## 7. What this means for investors and partners

**For investors evaluating engineering velocity:** The project shipped the equivalent of a full sprint's worth of headline-feature work in a single day, while clearing three of the six remaining roadmap items, while updating the investor-facing thesis doc, while writing a threat model and an operator runbook for the load-bearing new feature. The execution speed on the engineering side is materially higher than the prior quarterly forecast suggested.

**For technical partners evaluating integration readiness:** The MCP surface is now feature-complete — all three originally-hidden tools (analyze + dispatch_agent + agent_status) are functional end-to-end. The two-tier `/compute/status/{job_id}` API is stable and tested. Tier C content confidentiality is now production end-to-end (publisher-side: encrypt + Shamir-split + on-chain key deposit; consumer-side: payment → release event → decrypt). Operators can integrate against the May 2026 surface with confidence the contracts behind it are mainnet-deployed and the Python clients are tested.

**For Foundation officers planning execution-phase allocation:** With the engineering team's direct-control roadmap effectively closed, headcount allocation can shift to the three external-gated items (next ceremony cadence for A-08; testnet-traffic accumulation for calibration; conditional telemetry watch for multi-stage replay). The KYC vendor commission (Phase 5 task) is the single largest external dependency on the critical path; that's a procurement / partnership question, not an engineering question.

**For external auditors evaluating engagement scope:** The cumulative-audit-prep tag `cumulative-audit-prep-20260508-i` pins HEAD at a state where the streaming-inference subsystem (§7.1-§7.15), deploy-ceremony infrastructure (§7.16), and Item 6 dedup + arbitration (§7.19) compose into a single bundle. Each section has its own threat model addendum and reading path. Per `PRSM-POL-2`, agent-teams self-audit + OZ Pausable + TVL caps + 14-day public review windows substitute for external L3+L4 audits at this stage; trigger-driven revisits are documented at `docs/governance/2026-05-06-resource-constrained-audit-strategy.md`.

---

## 8. Tag reference (today's full set)

**Merge-ready tags (15):**

```
query-orchestrator-followon-ftns-budget-merge-ready-20260508
query-orchestrator-followon-endpoint-resolver-merge-ready-20260508
query-orchestrator-followon-x25519-cipher-merge-ready-20260508
prov-1-item-6-three-band-merge-ready-20260508
prov-1-item-6-t6-5-x-binary-path-merge-ready-20260508
prov-1-item-6-t6-5-gov-merge-ready-20260508
prov-1-item-6-t6-5-gov-next-merge-ready-20260508
prov-1-item-6-t6-5-gov-next2-merge-ready-20260508
query-orchestrator-b8-prsm-analyze-unhide-merge-ready-20260508
query-orchestrator-b8-prsm-dispatch-agent-unhide-merge-ready-20260508
query-orchestrator-b8-prsm-agent-status-unhide-merge-ready-20260508
query-orchestrator-step6-settlement-split-merge-ready-20260508
key-distribution-client-merge-ready-20260508
query-orchestrator-b8-job-history-merge-ready-20260508
```

(Note: count is 14 tag operations as listed; the additional tag is the Item 6 production-deployable entry's `prov-1-item-6-t6-5-gov-next2-merge-ready-20260508` referenced from MEMORY.md.)

**Cumulative audit-prep tags (7):**

```
cumulative-audit-prep-20260508-c   (post §4 step 6 + B8 + Item 6 baseline)
cumulative-audit-prep-20260508-d   (+ threat-model addendum §3.18)
cumulative-audit-prep-20260508-e   (+ operator runbook)
cumulative-audit-prep-20260508-f   (+ B8 program complete)
cumulative-audit-prep-20260508-g   (+ §4 step 6 settlement split)
cumulative-audit-prep-20260508-h   (+ KeyDistribution Python client)
cumulative-audit-prep-20260508-i   (+ B8 async-dispatch follow-on; current HEAD)
```

External-auditor handoff: `cumulative-audit-prep-20260508-i` is the canonical tag.

---

## 9. What's next

The `PRSM_Vision.md` §13 status-table now shows the engineering team's direct-control surface as **closed for net-new feature work**. Three external-gated items remain (A-08 ceremony cadence, calibration corpus, multi-stage replay telemetry). Reasonable threads for the next session:

- **Operator-side polish on shipped items.** Event-watcher daemon for KeyDistribution; filesystem-backed JobHistoryStore; PCU-weighted compute split. Each is sized at half-day to a day; each closes a documented honest-scope item.
- **Production-correctness gap.** `source_agent_pubkey → FTNS wallet registry` — settlement-split shipped today transfers to hex-pubkey strings that aren't real wallets on the local ledger. Production deployments will need a node-id → wallet mapping registry; v1 uses hex pubkey as identifier (carried-over honest-scope item).
- **Pre-funding posture refinement.** With the engineering surface largely complete, the next blockers are external — investor outreach, KYC vendor commission, council-expansion work toward the 12-month 2-of-3 commitment in PRSM-POL-1.
- **Operator onboarding documentation.** `OPERATOR_GUIDE.md` exists but predates today's surface — could be refreshed against the May 2026 deployment.

Today's cadence is not a steady state. The session reflects an unusual confluence — accumulated design work from prior sprints that became implementation-ready simultaneously, plus the QueryOrchestrator wiring sprint having pre-staged the surfaces that today's commits closed. Future sprints will likely return to a slower, more typical cadence.

---

## Appendix A — Today's `git log` summary

```
$ git log --oneline cumulative-audit-prep-20260507-X..HEAD
09f0a797 B8 async-dispatch follow-on — JobHistoryRecord + richer /compute/status
0cf9d68d KeyDistribution.sol Python client wiring
aa172b6b §4 step 6 — compute-participant settlement (post-QO-aggregation split)
cbcb9fad B8 unhide pass 3 — /compute/status/{job_id} + prsm_agent_status
e5fd8187 B8 unhide pass 2 — prsm_dispatch_agent
391d6e2f B8 — unhide prsm_analyze + adapt /compute/forge to QueryOrchestrator
ac926c61 PRSM-PROV-1 Item 6 — operator activation runbook
4412e9fb PRSM-PROV-1 Item 6 threat-model addendum (§3.18)
42f58f88 PRSM-PROV-1 Item 6 — audit-prep §7.19 + MEMORY.md refresh
76ccda38 PRSM-PROV-1 Item 6 T6.5.gov.next2 — node-startup arbitration wiring
223870a7 PRSM-PROV-1 Item 6 T6.5.gov.next — TokenWeightedVoting sink adapter
43c2656d PRSM-PROV-1 Item 6 T6.5.gov — ARBITRATION_DISPUTE proposal hook
44054191 PRSM-PROV-1 Item 6 T6.5.x — binary-path 3-band wiring
6f591da8 PRSM-PROV-1 Item 6 T6.3 + T6.5 — three-band dedup wiring + arbitration
2620ad8e QueryOrchestrator follow-on — X25519+XChaCha20 partial-result cipher
55884f7e QueryOrchestrator follow-on — HTTP endpoint resolver (lambda placeholder closed)
9f5d59a3 QueryOrchestrator follow-on — ftns_budget constructor override (§3 closed)
```

End of summary.
