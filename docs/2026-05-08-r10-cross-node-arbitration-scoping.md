# R10 Cross-Node Arbitration Scoping — Network-Wide Disputed-Content Resolution

**Document identifier:** R10-SCOPING-1
**Version:** 0.1 Draft
**Status:** Architectural scoping. Specifies the design space for promoting today's per-node arbitration queue to a network-wide cross-node mechanism. Captures threat-model deltas, three candidate architectures, named promotion triggers, and Foundation boundaries. **NOT an execution plan.** **NOT a commitment to ship.** **NOT a replacement for the per-node mechanism shipped in PRSM-PROV-1 Item 6.**
**Date:** 2026-05-08
**Drafting authority:** PRSM founder
**Promotes:** new research-track item (R10) raised after Item 6 closure on 2026-05-08, during the audit-prep refresh that documented per-node arbitration's per-node-only scope as a known limitation.

**Related documents:**
- `docs/2026-05-08-prsm-prov-1-threat-model-addendum-item-6.md` — Item 6 threat model (§3.18, A1–A8). R10 generalizes A6 ("cross-node duplicate dispute") and A7 ("stale-state poisoning") from acknowledged-but-unaddressed risks to first-class scoping work.
- `docs/2026-05-08-prsm-prov-1-item-6-operator-activation-runbook.md` — Item 6 operator activation runbook. R10 promotion would alter §2 (activation tiers) and §4 (council resolution flow).
- `docs/2026-04-27-cumulative-audit-prep.md` §7.19 — audit-prep entry for Item 6, lists per-node scope as a deliberate honesty caveat.
- `prsm/data/dedup/arbitration.py` — `FilesystemArbitrationQueue` (per-node). The R10 work would either replace this or layer a coordination plane above it.
- `prsm/economy/governance/arbitration_sink.py` — `TokenWeightedVotingProposalSink`. R10 may push the dedup decision _below_ this layer (so duplicate proposals never reach governance) or _above_ it (so governance arbitrates the dedup itself).
- `docs/2026-05-06-dht-transport-sprint.md` (PRSM-DHT-TRANSPORT) — existing DHT transport substrate. Candidate Architecture A reuses this directly.
- `contracts/v2/ProvenanceRegistryV2.sol` — existing on-chain provenance surface. Candidate Architecture B layers a new `ArbitrationRegistry.sol` adjacent to this.

---

## 1. Purpose

The question raised: **what changes when arbitration of disputed-band attribution records must coordinate across nodes, and at what point in PRSM's lifecycle should that promotion happen?**

Today's per-node mechanism (Item 6, shipped 2026-05-08):

- Each node maintains its own `FilesystemArbitrationQueue` at `~/.prsm/arbitration_queue/`.
- When a content upload routes into the disputed band (between `arbitration_floor` and the derivative threshold), a local `DisputedAttributionRecord` is enqueued.
- If `PRSM_ARBITRATION_PROPOSER_ID` is set, the record optionally creates a `ProposalCategory.ARBITRATION_DISPUTE` token-weighted governance proposal via `TokenWeightedVotingProposalSink`.
- Three-tier failure isolation keeps uploads green even when governance is unreachable.

This is correct and shippable for early bake-in. **It is not what a healthy mature network looks like.** Specifically:

- Two nodes encountering the same disputed content, in the same upload window, will independently raise duplicate governance proposals.
- A malicious uploader operating their own node may simply suppress local enqueuing — there is no global "this content was disputed" lookup that an honest peer can consult.
- Once a proposal resolves on-chain, the resolution does not propagate back to peers' local queues — peers continue to treat the content as disputed indefinitely.

R10 scopes the design space for a network-wide arbitration plane that addresses these gaps **without sacrificing the per-node mechanism's correctness, low latency, or operator simplicity**.

**R10 is a scoping-only document.** It answers seven framing questions:

1. What is today's per-node mechanism actually doing, and what is it not doing? (§2-§3)
2. What additional adversaries appear when arbitration is network-wide rather than per-node? (§4)
3. What architectural principle should a cross-node mechanism adopt? (§5)
4. What are the candidate architectures, and what are their tradeoffs? (§6)
5. How do existing PRSM substrates (DHT transport, on-chain provenance, governance backend) compose into each candidate? (§6)
6. What does the Foundation specifically NOT do, even after R10 promotion? (§7)
7. Under what named conditions should R10 be promoted from research-track to execution? (§8)

**R10 is NOT:**
- A replacement for the per-node mechanism. The per-node queue continues to be load-bearing for per-node honesty even after R10 ships.
- A commitment to any specific architecture. Three are scoped; a fourth may emerge.
- A roadmap for governance reform. R10 specifies mechanism for cross-node coordination of arbitration; it does not redesign how disputes are resolved (that remains token-weighted voting under the existing governance backend).
- A precondition for any current PRSM milestone. None of the existing audit-pipeline gates depend on R10 shipping.

---

## 2. What per-node arbitration does today

### 2.1 Routing pipeline

```
ContentUploader.upload()
  ↓
ThresholdResolver.resolve(content_type)
  → EffectiveThresholds{derivative, arbitration_floor}
  ↓
similarity_score = compute(payload, parent_record)
  ↓
3-band routing:
  if similarity ≥ derivative:        clear-uphold (record parent attribution)
  if similarity ≥ arbitration_floor: disputed-band → arbitration queue + sink
  if similarity <  arbitration_floor: clear-reject (no parent attribution)
```

### 2.2 Disputed-band processing (per-node)

```
DisputedAttributionRecord {
  content_cid:        bytes,
  parent_record_cid:  bytes,
  similarity_score:   float,
  arbitration_floor:  float,
  derivative:         float,
  uploader_pubkey:    bytes,
  parent_creator_id:  str,
  enqueued_at:        int (unix ms),
  proposal_id:        str | None  # set later via sink integration
}
  ↓
arbitration_queue.enqueue(record)         # filesystem persist
  ↓ (if PRSM_ARBITRATION_PROPOSER_ID set)
proposal_id = arbitration_proposal_sink.create_proposal(record)
  ↓
arbitration_queue.set_proposal_id(record.content_cid, proposal_id)
```

### 2.3 What three-tier failure isolation means

Each of the three calls (`enqueue`, `create_proposal`, `set_proposal_id`) is independently wrapped in try/except. A failure in any tier logs and proceeds; uploads stay green. This is load-bearing for operator UX — but it means that a *successful* enqueue followed by a *failed* `set_proposal_id` leaves the local queue with a record that has no proposal-id, which will be reconciled by operator action (or, post-R10, by network-wide reconciliation).

### 2.4 What this accomplishes

- Per-node honesty: an operator who runs Item 6 will surface their own disputed-band records.
- Operator-controlled escalation: governance proposals are gated on `PRSM_ARBITRATION_PROPOSER_ID`, so the network does not get spammed with proposals from operators who haven't opted in.
- Failure isolation: governance brokenness does not break uploads.
- Audit trail: filesystem queue persists across restarts.

### 2.5 What this does NOT accomplish

- Cross-node visibility: Node B does not know Node A already enqueued the same content-CID.
- Cross-node deduplication: governance may receive duplicate proposals minutes apart from different nodes.
- Adversarial-silence resistance: a malicious uploader running their own node simply does not enqueue.
- Resolution propagation: when a proposal resolves, peers' local queues do not update.
- Quorum on routing: nodes with different threshold calibrations may route the same content into different bands; per-node arbitration takes whichever band the local node decided.

---

## 3. Threat model — per-node baseline (already addressed)

For completeness, the threat model addendum at §3.18 (A1–A8) covers per-node adversaries already. R10 inherits these and adds new ones below. Briefly:

- A1 (per-node): malicious threshold tampering — addressed via signed-config + threshold-resolver tests.
- A2 (per-node): queue corruption — addressed via filesystem JSON round-trip + idempotent-on-equal/raise-on-conflict invariant.
- A3 (per-node): proposal-sink injection — addressed via Protocol typing + `NullArbitrationProposalSink` default.
- A4 (per-node): governance-side DoS — addressed via three-tier failure isolation.
- A5 (per-node): proposal-id collision — addressed via byte-deterministic `render_arbitration_body`.

A6, A7, A8 are the gateway threats R10 elevates from caveats to scoping work — see §4 below.

---

## 4. Threat model — cross-node adversaries

R10 introduces network-scale adversaries that per-node arbitration explicitly cannot address.

### 4.1 R10-A1: Sybil dispute flooding

**Adversary:** an operator-adversary creates N pseudo-nodes, each of which routes some content into the disputed band and creates a governance proposal.

**Goal:** drown legitimate disputes in noise; force token-holders to vote on hundreds of garbage proposals; or trigger a per-content-CID denial-of-attribution (legitimate parent never gets attributed because the proposal is buried in spam).

**Why it works against per-node:** there is no cross-node count of "how many distinct nodes raised this dispute." Each pseudo-node looks like a unique honest signal.

**R10 mitigation candidates:**
- Stake-weighted dispute submission cost (FTNS posted at proposal creation, refunded if dispute upheld).
- Cross-node dedup before proposal submission — a single canonical proposal per content-CID, regardless of how many nodes flag it.
- Time-window throttling — at most one proposal per content-CID per 7-day window, regardless of how many nodes route it.

### 4.2 R10-A2: Eclipse attack on dispute-lookup DHT

**Adversary:** controls the DHT view of a target node, returning "no existing dispute" for content that the rest of the network has already disputed.

**Goal:** target node creates a duplicate proposal (which legitimate nodes recognize as duplicate and reject) — wasting target's gas / governance attention. Or, target node believes content is dispute-free and routes it to clear-uphold when peers know it's disputed.

**Why it works:** Kademlia α-of-k lookups can be eclipsed if all α responses come from adversary-controlled peers.

**R10 mitigation candidates:**
- Diversity-aware peer selection (Kademlia bucketing + IP/AS-prefix deduplication, already in `prsm/p2p/`).
- Cross-validation: every dispute-lookup result must be confirmed by ≥ 2 peers from disjoint /16 prefixes.
- On-chain anchor of the canonical dispute set as ground truth (Architecture B/C below).

### 4.3 R10-A3: Replay across content-creator boundaries

**Adversary:** lifts a signed `DisputedAttributionRecord` from an old, resolved dispute and re-broadcasts it against a new (innocent) content-CID with the same signature.

**Goal:** forge a dispute against innocent content using a real-but-stale signature.

**Why it works:** today's `DisputedAttributionRecord` body includes content-CID and parent-record-CID, but if the body's signature scope or domain-separation tag is weak, replay across CIDs is possible.

**R10 mitigation candidates:**
- Domain-separation prefix per dispute-event (`b"PRSM-ARBITRATION-V1\x00"` already partial in `render_arbitration_body`; promote to full DST per RFC 9380).
- Network-wide dispute-event nonce or epoch (anchored to block height or DHT epoch).
- Strict body-hash binding inside the signature's authenticated-data scope.

### 4.4 R10-A4: Front-running on-chain resolution

**Adversary:** sees a pending dispute proposal, deduces voting outcome from public token-holder positions, front-runs resolution by uploading "fixed" content or buying votes.

**Goal:** capture economic upside of being first to resolve.

**Why it works against on-chain resolution:** mempool visibility of pending governance txs is unavoidable on Base.

**R10 mitigation candidates:**
- Commit-reveal voting (commitment hash on-chain first, plaintext vote in a later tx).
- Sealed-bid arbitration with timed reveal.
- Off-chain vote aggregation + single batched on-chain settlement (composable with existing `TokenWeightedVoting` backend).

### 4.5 R10-A5: Foundation-Safe censorship of resolution

**Adversary:** the Foundation Safe (or its 2-of-3 multisig signers, in collusion).

**Goal:** revert dispute resolution outcomes that token-holders voted on, replace them with Foundation-preferred outcomes.

**Why it's interesting at network scale:** at per-node scope this attack only affects one node's view; at cross-node scope a single Foundation tx can rewrite the canonical state for everyone.

**R10 mitigation candidates:**
- Time-lock + 14-day public-review window before binding (parallel to PRSM-POL-2's TVL-cap + public-review pattern).
- Token-weighted veto: any sufficiently-staked token-holder block can extend the review window or trigger re-vote.
- Sole-Owner posture self-limitation: Foundation explicitly disclaims authority to revert disputes that pass quorum (governance charter amendment).

### 4.6 R10-A6: Cross-node duplicate proposal creation

**Adversary:** none required — this is a benign-environment failure mode. R10's primary motivating threat.

**Goal:** N/A — the failure is honest nodes both raising the same dispute.

**Why it happens:** per-node mechanism has no global "is this disputed already?" check. Two nodes routing the same content within minutes will both call `create_proposal`.

**R10 mitigation candidates:**
- DHT lookup before proposal creation: "any existing proposal for this content-CID?"
- On-chain registry lookup before proposal creation: same question, on-chain answer.
- Sticky proposer: first node to dispute holds the lock for a 7-day window; subsequent nodes append votes rather than creating new proposals.

### 4.7 R10-A7: Stale-state poisoning

**Adversary:** none required — second benign-environment failure mode.

**Goal:** N/A — failure is local queues drifting from canonical state over time.

**Why it happens:** when a proposal resolves on-chain (or via off-chain governance), the resolution is not pushed back to nodes' local queues. A node restarted weeks later still has open queue entries for disputes the network resolved long ago.

**R10 mitigation candidates:**
- Periodic reconciliation: each node, on startup and every N hours, reconciles its local queue against the canonical dispute set.
- Push notification: on resolution, governance publishes a `ResolutionEvent` that all listening nodes apply locally.
- Pull-on-read: when a queue entry is read, the node first checks canonical state for an updated outcome.

### 4.8 R10-A8: Mixed-band poisoning

**Adversary:** an operator running deliberately-miscalibrated thresholds.

**Goal:** route content that the network considers clear-uphold into the local node's disputed band (or vice versa), generating proposals the network doesn't consider valid (or suppressing proposals the network does consider valid).

**Why it works:** thresholds are operator-configurable. Per-node arbitration treats whatever local thresholds say as ground truth.

**R10 mitigation candidates:**
- Quorum-on-routing: a dispute is canonical only when ≥ K nodes independently route the same content into the disputed band.
- Threshold attestation: nodes broadcast their `EffectiveThresholds` config + signed cert; outlier configurations trigger reputation penalties.
- Outlier filtering on-chain: on-chain registry rejects disputes whose claimed thresholds deviate from the network's published median by more than D.

---

## 5. Architectural principle

R10 commits to four principles. These constrain the candidate architectures (§6).

### 5.1 Per-node mechanism remains load-bearing

Cross-node coordination layers _above_ the per-node queue, not in place of it. Operators may continue to run with `PRSM_ARBITRATION_PROPOSER_ID` unset (Tier 1 — local queue only) and benefit only from local visibility. R10 does not force every operator into the cross-node plane.

### 5.2 No new governance authority

R10 does not create a new dispute-arbitration authority. The token-weighted voting backend (`TokenWeightedVoting` + `ProposalCategory.ARBITRATION_DISPUTE`) remains the sole resolution path. R10 changes _how proposals reach the backend_ and _how outcomes propagate back_ — not who decides outcomes.

### 5.3 Coordination is best-effort under partition

Cross-node coordination must degrade to per-node operation cleanly under network partition. A node disconnected from the DHT or from the on-chain registry must continue to operate (uploads green, local queue functional) — it just loses cross-node deduplication and resolution propagation until reconnected.

### 5.4 Operator opt-in remains visible

R10 introduces new env knobs. The activation surface stays explicit (per Item 6's three-tier model). No silent defaults that change operator behavior on upgrade.

---

## 6. Candidate architectures

Three candidates. Each is internally coherent; each has tradeoffs. R10 does not pick a winner — that decision belongs to the execution-phase work that closes the named promotion triggers in §8.

### 6.1 Architecture A — DHT-anchored

Reuse the existing `SyncDHTTransport` + `EmbeddingDHTServer` substrate. Disputed records become DHT-keyed values: `arbitration:<content_cid>` → `signed(DisputedAttributionRecord + status)`.

**Lookup flow (before creating proposal):**
```
existing = dht.get(f"arbitration:{content_cid}")
if existing and existing.status in {"open", "resolved-uphold", "resolved-reject"}:
    # dedup against existing; either join votes or honor resolution
    skip_proposal_creation()
else:
    proposal_id = sink.create_proposal(record)
    dht.put(f"arbitration:{content_cid}", sign(record + "open"))
```

**Resolution propagation:** on governance resolution, the proposer broadcasts a signed update to the DHT.

**Pros:**
- Reuses already-deployed DHT transport (`docs/2026-05-06-dht-transport-sprint.md`).
- No on-chain cost — gas-free dedup.
- Sub-second cross-node lookup latency.
- Composable with existing `PublisherKeyAnchorClient` for sig-verify.

**Cons:**
- Eclipse-attack surface (R10-A2) requires diversity-aware lookup confirmation.
- DHT churn means stale records may persist past their TTL.
- No binding settlement — disputes can be re-litigated indefinitely if a peer disagrees with a DHT-stored resolution.
- Requires careful nonce/epoch handling to defeat replay (R10-A3).

**Effort estimate:** 2-3 weeks. Builds on shipped DHT primitives; main work is signed-record schema, eclipse-resistant lookup, and resolution-propagation listener.

### 6.2 Architecture B — On-chain `ArbitrationRegistry.sol`

New Solidity contract adjacent to `ProvenanceRegistryV2.sol`. Each disputed content gets an on-chain record:

```solidity
struct DisputeRecord {
    bytes32 contentCid;
    bytes32 parentRecordCid;
    address proposer;
    uint256 proposalId;        // governance backend ID
    uint8   status;            // 0=Open, 1=UpheldParent, 2=RejectedParent
    uint64  openedAt;
    uint64  resolvedAt;
}
mapping(bytes32 => DisputeRecord) public disputes;
```

**Lookup flow (before creating proposal):**
```
existing = registry.disputes(content_cid)
if existing.status != 0 || existing.openedAt > 0:
    # dedup; query on-chain
    skip_proposal_creation()
else:
    proposal_id = sink.create_proposal(record)
    registry.openDispute(content_cid, parent_cid, proposal_id)  # sole-owner Foundation Safe gates
```

**Resolution propagation:** governance backend writes resolution outcome to the contract; nodes either listen for `DisputeResolved` events or pull-on-read.

**Pros:**
- Single canonical source of truth — no eclipse, no stale state.
- Composes with existing on-chain provenance + token-weighted voting infrastructure.
- Auditable: every dispute opening + resolution leaves a Basescan trail.
- Front-running mitigations (R10-A4) reuse existing patterns from Phase 8 + governance contracts.

**Cons:**
- Gas cost per dispute opening (modest on Base, but non-zero).
- Foundation Safe is sole owner — concentrates censorship risk (R10-A5) unless mitigated by §5.2 + time-lock.
- Slower lookup latency (chain RPC) — matters less for the dedup use case (cold-path) but matters for rapid-fire upload flows.
- Requires new contract deploy ceremony + audit.

**Effort estimate:** 4-6 weeks. Contract draft + Hardhat tests + audit + deploy ceremony + Python client + Python integration. Audit is the long pole.

### 6.3 Architecture C — Hybrid (DHT for fast lookup, on-chain for binding settlement)

DHT handles "is this disputed?" (read-side). On-chain handles "what was the outcome?" (write-side, binding).

**Flow:**
```
# Read path (cold):
fast_check = dht.get(f"arbitration:{content_cid}")
if fast_check is None:
    # DHT miss — fall back to canonical on-chain check
    canonical = registry.disputes(content_cid)
    if canonical.openedAt > 0:
        # populate DHT; we just learned something
        dht.put(f"arbitration:{content_cid}", canonical)
    else:
        # genuinely undisputed
        proposal_id = sink.create_proposal(record)
        registry.openDispute(...)
        dht.put(f"arbitration:{content_cid}", "open")
else:
    # DHT hit; honor it (verify against canonical only on write-suspicion)
    skip_proposal_creation()

# Resolution path: governance writes to registry → emits event → nodes
# update DHT entries via background watcher.
```

**Pros:**
- Fast path is sub-second (DHT).
- Slow path is canonical (on-chain).
- DHT stale-state corrected by on-chain reconciliation.
- Eclipse attacks become detectable: an eclipsed node will have its DHT lookup miss and fall through to chain — losing only latency, not correctness.

**Cons:**
- Most complex: two layers to reason about, more code paths, more failure modes during DHT/chain divergence.
- Requires both DHT and on-chain components to be deployed and operated.
- Effective only when cache-hit ratio on DHT is high; cold-cache deployments degrade to Architecture B's latency profile.

**Effort estimate:** 6-9 weeks. Architecture B's contract work + Architecture A's DHT work + a non-trivial reconciliation layer between them. Architecture C should be considered only if the named promotion triggers (§8) make it clear that BOTH sub-second lookup AND binding settlement are network-critical.

### 6.4 Architecture comparison summary

| Property | A (DHT) | B (on-chain) | C (Hybrid) |
|---|---|---|---|
| Lookup latency | Sub-second | Chain-RPC | Sub-second (cache hit) / Chain-RPC (miss) |
| Binding settlement | No | Yes | Yes |
| Gas cost per dispute | $0 | Modest on Base | Modest on Base |
| Eclipse-attack surface | Yes | No | No (chain ground truth) |
| Stale-state risk | Yes | No | Mitigated by reconciliation |
| Foundation-censorship surface | Low | High (mitigated by §5.2 + time-lock) | High (same mitigations) |
| New audit needed | DHT schema review | Solidity audit (long pole) | Both |
| Effort | 2-3 weeks | 4-6 weeks | 6-9 weeks |
| Recommended when | Need fast dedup but do not yet need binding settlement | Need binding settlement and gas cost is acceptable | Need both |

---

## 7. Foundation boundaries

The Foundation, even after R10 promotion, explicitly does NOT:

- **Decide individual disputes.** Token-weighted voting decides; the Foundation Safe only executes the on-chain settlement of the outcome. (Carries forward from PRSM-GOV-1 §4.3.)
- **Maintain a curated dispute registry.** No allowlist of "sanctioned disputes" or denylist of "suppressed disputes." (Same neutrality posture as transport-layer in R9.)
- **Adjudicate cross-jurisdictional disputes.** Disputes are decided by token-weighted vote, full stop. R10 does not introduce jurisdictional gating, regional registries, or content-class allowlists.
- **Fast-track high-stakes disputes.** No discretionary acceleration. The 14-day public-review window applies uniformly.
- **Pre-empt governance.** The Foundation Safe cannot open or close a dispute except by executing a passing proposal.

These boundaries are **stronger than mere policy** — they should be encoded as contract-level revocation of Foundation authority over dispute-resolution writes (e.g., `ArbitrationRegistry.sol` opens/closes disputes only when called by `TokenWeightedVoting` proposal-execution path; Foundation Safe is _not_ a permitted caller).

---

## 8. Promotion triggers

R10 stays in research-track until **at least one** of the following is observed. These are intended as objective, operator-visible signals rather than judgment calls.

### 8.1 T1 — Cross-node duplicate observed in production

Operator telemetry shows ≥ 2 distinct nodes independently raising governance proposals for the same content-CID within a 24-hour window. This is the gateway evidence that the per-node mechanism has crossed the cost/benefit boundary.

**Detection:** governance-backend logs include proposer_id; an off-chain observer aggregates by content-CID + 24h bucket.

### 8.2 T2 — Council resolution SLA breach

The 7-day SLA on council resolution (per Item 6 operator runbook §4) is breached because a duplicate proposal was opened mid-resolution and council had to redo deliberation. Even one occurrence is a strong signal.

### 8.3 T3 — Sybil-flood attempt

Operator telemetry detects ≥ 10 disputes opened from ≥ 5 distinct proposer-IDs within a 24-hour window targeting the same content-creator-id. This pattern is unambiguously adversarial; per-node arbitration cannot defend against it.

### 8.4 T4 — TVL crosses $50,000

Per PRSM-POL-2's audit-revisit trigger schedule, TVL > $50K is an automatic full-stack security re-evaluation. R10 promotion is one of the items that should be on that re-evaluation's docket.

### 8.5 T5 — VC term sheet specifying mainnet network growth

If a Series A or seed-extension term sheet conditions investment on N+ active disputed-content resolutions per month, the per-node mechanism's capacity is the binding constraint and R10 promotion is on the critical path.

### 8.6 T6 — External auditor flag

External security review (post-PRSM-POL-2 §4 trigger schedule) flags per-node arbitration as Sybil-vulnerable or eclipse-vulnerable. The auditor's specific finding determines which of A/B/C is appropriate.

### 8.7 T7 — Calibration corpus surfaces persistent disputed band

Item 6 T6.4 (calibration corpus, gated on 30+ days testnet upload traffic) identifies a content-type for which the disputed band reliably contains > 5% of uploads. At that volume, per-node duplicate-proposal noise dominates governance bandwidth.

---

## 9. Integration phases (conditional on promotion)

If R10 is promoted (any trigger fires), execution proceeds in phases. Each phase is independently shippable and provides value alone — partial promotion is supported.

### 9.1 Phase R10.1 — Reconciliation read-only (1-2 weeks)

Read-only integration with a chosen substrate (DHT or on-chain). On startup and every N hours, reconcile local `FilesystemArbitrationQueue` entries against the canonical set. Log mismatches; do not write.

This is a no-risk diagnostic that produces operator visibility into the cross-node gap before any binding writes happen. Phase R10.1 alone closes R10-A7 (stale-state poisoning).

### 9.2 Phase R10.2 — Dedup-on-write (3-4 weeks)

Before `sink.create_proposal()` is called, perform a cross-node lookup. If an open proposal exists, append a vote rather than create a duplicate. If a resolved record exists, honor the resolution (no new proposal).

Phase R10.2 closes R10-A6 (duplicate proposals) and R10-A1 (Sybil flood mitigation).

### 9.3 Phase R10.3 — Binding settlement (4-6 weeks, if Architecture B/C chosen)

Deploy `ArbitrationRegistry.sol` (or hybrid). Push governance resolution outcomes on-chain. Nodes pull canonical state.

Phase R10.3 closes R10-A2 (eclipse), R10-A3 (replay), R10-A4 (front-running, with appropriate commit-reveal), and R10-A8 (mixed-band poisoning, via outlier filtering).

### 9.4 Phase R10.4 — Full deployment + audit (4-8 weeks)

External Solidity audit (if Architecture B/C). Operator runbook update. Audit-prep refresh. Threat-model promotion of A6/A7/A8 from "mitigated by R10" to "addressed."

---

## 10. NOT in scope (explicit non-goals)

- **Replacing token-weighted voting.** R10 changes how proposals reach voting and how outcomes propagate; the voting mechanism is unchanged.
- **Cross-jurisdictional dispute coordination.** No regional registries, no jurisdictional gating, no content-class allowlists. (Maintains R9 neutrality posture.)
- **Real-time dispute streaming.** R10 targets eventual-consistency cross-node coordination, not push-streaming. A high-throughput stream is a separate research item.
- **Off-chain reputation outside FTNS-token weight.** The dispute-resolution authority remains FTNS-token-weighted voting; R10 does not introduce a separate reputation score.
- **AI-assisted dispute pre-screening.** No "the AI suggests this is/isn't a real dispute." Disputes are routed by mechanical similarity threshold (per Item 6) and resolved by humans (token-holders).
- **Backwards-incompatible per-node breakage.** Operators running today's per-node setup MUST continue to work post-R10. R10 layers above; it does not replace.

---

## 11. Open questions

These are deliberately left unresolved at scoping time. Each will be addressed by the corresponding phase if R10 is promoted:

- **Q1.** What is the canonical DHT key schema for disputes? `arbitration:<cid>` is suggestive; binding choice happens at Phase R10.2.
- **Q2.** What domain-separation tag should signed dispute bodies use? Today's `render_arbitration_body` prefixes with `b"PRSM-ARBITRATION-V1\x00"`. R10-A3 mitigation may require a stronger DST.
- **Q3.** How does sticky-proposer (one-proposal-per-CID-per-7-days) interact with the existing `set_proposal_id` flow? Probably: append-vote on the existing proposal-id, but the wire format is undecided.
- **Q4.** Where does outlier-threshold filtering (R10-A8) live — on-chain (rejecting proposals with deviant claimed-thresholds) or in governance backend (filtering at vote-tally time)?
- **Q5.** Does Architecture C's DHT cache-hit ratio in production justify the additional complexity, or does Architecture B (on-chain only) win on operational simplicity? Probably impossible to answer without bake-in data.

---

## 12. Conclusion

R10 specifies the design space for promoting per-node arbitration to a network-wide mechanism. It is intentionally conservative: today's per-node mechanism is correct and shippable, and the cost of premature promotion (extra moving parts, audit cycles, governance complexity) outweighs its benefits at current scale.

The named promotion triggers in §8 are designed to be **objective signals from operator telemetry**, not judgment calls. When a trigger fires, R10 has a clear answer for "what should we build?" — three architectures with explicit tradeoffs, four phases of incremental rollout, and Foundation boundaries that prevent scope creep.

Until then, the per-node mechanism (Item 6) remains the canonical disputed-content resolution path, and R10 sits in `docs/Phase4+ research track` as the pre-built escalation route.

---

**End of document.**
