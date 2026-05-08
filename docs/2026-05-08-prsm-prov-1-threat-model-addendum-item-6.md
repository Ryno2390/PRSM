# PRSM-PROV-1 Threat Model Addendum — Item 6 (§3.18)

**Date:** 2026-05-08
**Track:** PRSM-PROV-1 Content Provenance Correctness (Items 3–7 from the
2026-05-05 embedding audit)
**Cross-references:**
- Plan: [`docs/2026-05-06-content-provenance-correctness-plan.md`](2026-05-06-content-provenance-correctness-plan.md) §5
- T6.5 design: [`docs/2026-05-07-PRSM-PROV-1-T6.5-arbitration-queue-design.md`](2026-05-07-PRSM-PROV-1-T6.5-arbitration-queue-design.md)
- Companion §3.18 in the master threat model surface
- Audit-prep §7.19 (paired with this doc)
- Item 3 addendum: [`docs/2026-05-06-prsm-prov-1-threat-model-addendum.md`](2026-05-06-prsm-prov-1-threat-model-addendum.md) §3.16
- Item 7 addendum: [`docs/2026-05-06-prsm-prov-1-threat-model-addendum-item-7.md`](2026-05-06-prsm-prov-1-threat-model-addendum-item-7.md) §3.17

This addendum captures the threat model for Item 6 — per-content-type
calibrated thresholds and the disputed-band arbitration queue. Pre-Item-6,
every dedup hit at `sim ≥ DERIVATIVE_THRESHOLD` (a single class constant
of 0.92) auto-prepended the matching CID as a parent of the new upload,
splitting royalties under the 70/25/5 schedule. Item 6 routes the
borderline middle band to council review rather than auto-attributing.
This addendum names the new adversary classes that emerge when there is
a queue + a council in the loop.

---

## §3.18.1 Trust boundaries

Item 6 introduces three new trust boundaries:

- **Uploader vs. dedup engine.** The uploader supplies content +
  optional `content_type_hint` metadata (e.g. `"scientific_abstract"`,
  `"code"`, `"prose"`). The dedup engine consults
  `ThresholdResolver` to compute effective `(derivative, duplicate,
  arbitration_floor)` thresholds for the kind+hint combination. The
  uploader's hint is *advisory* — bounded below by per-kind floors
  in `prsm/data/dedup_thresholds.yaml::floors` so a malicious hint
  cannot push thresholds arbitrarily low to evade dedup.

- **Dedup engine vs. arbitration queue.** When `sim ∈
  [arbitration_floor, derivative)`, the dedup engine stops short of
  auto-attribute and emits a `DisputedAttributionRecord` to the
  `ArbitrationQueue`. The queue is node-local in v1 (no cross-node
  consensus on flag-set membership); this is honest scope.

- **Arbitration queue vs. governance council.** Disputed records may
  be surfaced as `ProposalCategory.ARBITRATION_DISPUTE` proposals via
  an `ArbitrationProposalSink`. The council votes; the queue's
  `resolve` method records the verdict. Idempotent on equal decision,
  raises on conflicting decision.

The existing trust seams from Items 3 + 7 (publisher signature → on-chain
anchor → consumer-side verifier) are unchanged; Item 6 layers on top.

## §3.18.2 Adversaries in scope

| ID | Adversary | Capability |
|---|---|---|
| A1 | **Threshold-evading uploader** | Submits content + a malicious `content_type_hint` aiming to push the effective threshold low enough to slip past dedup as "unique original" content. |
| A2 | **Queue-grief uploader** | Submits a high volume of content engineered to land in the disputed band, swamping the council with arbitration proposals to either bury legitimate disputes or burn down the system proposer's FTNS balance. |
| A3 | **Council collusion** | A subset of council members coordinate to UPHELD a disputed claim against an unrelated content's derivative parent (effectively confiscating royalties to a colluding party). |
| A4 | **Conflict-of-interest proposer** | An actor who is the uploader OR candidate parent's creator submits the arbitration proposal themselves, biasing the framing or short-circuiting the system-level rationale. |
| A5 | **Webhook double-delivery** | Governance backend's resolve webhook delivers the same verdict twice (network retry, replay log) → must be idempotent or the queue diverges. |
| A6 | **Conflicting-verdict double-resolve** | Backend bug or operator mistake delivers a `REJECTED_PARENT` resolve, then later a different `UPHELD_PARENT` for the same record → must raise rather than silently flip. |
| A7 | **Body-rendering tampering** | A relay between the queue and the council UI mutates the proposal body. A future on-chain arbitration contract (per design doc §5.2.3) signs over body bytes; a tampered body would cause the contract verifier to reject. |
| A8 | **Anchor-less proposal** | Sink failure (insufficient FTNS, eligibility rejection) causes `create_arbitration_proposal` to return `None` → the queue still has the record but no council surface for review. Auditor concern: silent-degradation could mask a systemic governance outage. |

Out of scope (handled by other tracks or honest scope):
- Cross-node arbitration consistency (R10 deferred — v1 is node-local).
- On-chain arbitration contract design (long-term per design doc §5.2.3).
- Calibration correctness of YAML threshold values themselves (T6.4
  corpus-gated; v1 ships with conservative defaults).
- FTNS bond economic security (Phase 7 territory).

## §3.18.3 Attack mitigations

### A1 — Threshold-evading uploader

**Vector:** Uploader submits PNG with metadata
`content_type_hint: "scientific_abstract"` (a hint with a tightening
multiplier of 1.05 on the derivative threshold) to force their image
upload through a wrong-kind threshold path that would never trigger
dedup.

**Mitigations:**

1. **Hints are kind-scoped in YAML.** The
   `content_type_multipliers` section in
   `prsm/data/dedup_thresholds.yaml` keys multipliers by `<hint>:
   <kind>` — `scientific_abstract:text-vector` only applies to
   text-vector dedup. An image upload's binary-fingerprint path
   consults a different kind in the resolver and the
   `content_type_hint` is ignored (silently — auditor: confirm
   `test_unknown_hint_silently_ignored` in
   `test_dedup_thresholds.py` pins the no-cross-kind-hint
   property).

2. **Per-kind floors.** Even within the correct kind, the YAML
   `floors` section caps how far the multiplier can push the
   threshold. `floors.text-vector.derivative = 0.85` means a
   loosening multiplier cannot push the effective derivative
   threshold below 0.85 regardless of the hint. **Pin:**
   `test_floor_enforced_against_loosen_multiplier`.

3. **Construction-time invariant.** `EffectiveThresholds.__post_init__`
   raises `ValueError` if `arbitration_floor > derivative` or
   `duplicate < derivative`, so a pathological YAML can't produce
   thresholds that would silently break the band logic. **Pin:**
   `test_arbitration_floor_above_derivative_rejected_at_construction`.

**Residual:** A correctly-keyed tightening hint (e.g.
`scientific_abstract:text-vector` with 1.05× multiplier on derivative)
is *meant* to tighten and is not evading dedup. The mitigations target
the loosening-attack direction.

### A2 — Queue-grief uploader

**Vector:** Adversarial uploader posts hundreds of borderline-similar
documents per day, each landing in the disputed band, to overwhelm the
council and force expensive proposal-creation costs on the system
proposer (which pays FTNS submission fees).

**Mitigations (v1):**

1. **Existing per-creator FTNS bond requirement** (Phase 7 staking).
   Adversaries pay to participate in the network; spam costs them too.

2. **Sink failures degrade gracefully.** When the system proposer's
   FTNS balance is depleted, `TokenWeightedVotingProposalSink` returns
   `None` rather than raising. The queue still has the records;
   councils can author proposals manually. **Pin:**
   `test_value_error_returns_none` in
   `test_arbitration_sink_adapter.py`.

3. **Arbitration is non-blocking.** A queue or sink failure at any
   level WARN-logs and degrades; the upload itself completes
   regardless of governance state. This means the *attacker can't
   deny service to honest uploaders* by spamming the queue. **Pin:**
   `test_enqueue_failure_does_not_raise` +
   `test_sink_raising_does_not_break_upload` +
   `test_set_proposal_id_failure_swallowed`.

**v2 hardening (deferred):** Per-creator daily flag cap. Triggered on
first observed griefing pattern in testnet logs.

### A3 — Council collusion

**Vector:** A subset of council members vote `UPHELD_PARENT` on a
disputed-attribution record where the candidate parent is in fact
unrelated, redirecting royalties to a colluding party.

**Mitigations:**

1. **Out of Item 6 scope.** Council membership integrity is governance
   territory (PRSM-POL-1 governance council expansion commitments).
   Item 6 provides the proposal surface; the voting layer determines
   whether collusion crosses the quorum threshold.

2. **Disputed-attribution proposals carry full provenance.** The
   proposal `metadata` field includes `arbitration_record_id`,
   `fingerprint_kind`, `similarity`, `new_cid`, `candidate_parent_cid`
   so independent council members can verify the underlying claim
   off-chain.

3. **Resolution is auditable.** The `_ResolutionRecord` includes
   `by_council` (list of voter addresses) so post-hoc analysis can
   identify pattern-collusion.

**Residual:** Same as any council-vote system: minority-collusion
within quorum is by design the point of voting. Slow-rotation council
membership (PRSM-POL-1 §6.2 12-month expansion target) reduces the
window. Auditor focus is on the proposal surface integrity, not the
voting outcome correctness.

### A4 — Conflict-of-interest proposer

**Vector:** The uploader (who would profit from a `REJECTED_PARENT`
verdict) or the candidate parent's creator (who would profit from
`UPHELD_PARENT`) authors the proposal themselves, framing the body
text in a misleading way.

**Mitigations:**

1. **System-level proposer enforced at the sink.** The
   `TokenWeightedVotingProposalSink` constructor takes
   `proposer_id` as configuration, NOT per-record input. The
   operator-supplied `proposer_id` is the same for every
   disputed-attribution proposal regardless of who uploaded.

2. **Proposer-id documented as non-uploader, non-candidate-parent.**
   Module docstring at
   `prsm/economy/governance/arbitration_sink.py` explicitly calls
   out that `proposer_id` should be a system-level address (the
   Foundation Safe or a delegate), not an interested party. Operator
   guidance is the load-bearing mitigation here.

3. **Body is byte-deterministic.** `render_arbitration_body` is
   the canonical text the council sees. A council member who
   suspects misframing can re-render the body locally from the
   queue record + compare bytes. **Pin:**
   `test_deterministic_for_equal_records`.

**Residual:** A compromised system proposer (FTNS-wallet-key
compromise) could submit proposals with mutated bodies. The Foundation
Safe's hardware-multisig requirement (Phase 1.3 Task 8) is the line of
defense for the funds; the body-rendering determinism is the line of
defense for council verifiability.

### A5 — Webhook double-delivery (idempotency)

**Vector:** Governance backend's webhook fires twice for the same
proposal close (network retry, exactly-once-failure replay). Without
idempotency, the second `resolve` call could either crash, silently
overwrite state, or double-count something.

**Mitigation:** `ArbitrationQueue.resolve` is **idempotent on equal
decision**. A second call with the same `(record_id, decision)`
silently returns. **Pin:** `test_resolve_twice_idempotent` in
`test_arbitration_queue.py`.

### A6 — Conflicting-verdict double-resolve

**Vector:** Backend bug delivers `REJECTED_PARENT` to webhook,
operator mistake later delivers `UPHELD_PARENT` for the same record
(or the inverse).

**Mitigation:** `ArbitrationQueue.resolve` **raises `ValueError` on
conflicting decision**. This protects against silent state-flip but
does NOT auto-resolve the conflict — operator intervention required.
**Pin:** `test_resolve_with_conflicting_decision_raises`.

**Operator runbook:** On conflict, the queue retains the *first*
verdict; the conflicting later call is rejected. If the later
verdict is the correct one, operator must add an explicit
`reopen` operation (out of v1 scope) or correct via a new record.

### A7 — Body-rendering tampering

**Vector:** A relay between `_render_arbitration_body` and the
council UI mutates the body bytes. A future on-chain arbitration
contract (design doc §5.2.3) signing over body bytes would reject
the tampered version, but in v1 (off-chain governance), tampered
bodies could mislead voters.

**Mitigations:**

1. **Determinism is pinned.** Every council member can re-derive
   the canonical body from the queue record + `render_arbitration_body`.
   Mismatch detection is local + cheap.

2. **Pinned header + 6-decimal similarity precision.** The
   `"PRSM-PROV-1 disputed-attribution review\n"` header lets a
   verifier reject any body that lacks the prefix. The 6-decimal
   precision means two near-identical disputed records render
   distinctly so a relay can't substitute one record's body for
   another's.

3. **Forward-looking design.** When the on-chain arbitration
   contract ships (long-term per design doc §5.2.3), the contract
   will sign over the body bytes — the determinism is what makes
   that future signing payload computable.

**Residual:** A council UI that *itself* renders the body from the
queue record (rather than receiving a pre-rendered body from a relay)
trivially closes A7. Operator guidance: council UIs should fetch the
record and render locally.

### A8 — Anchor-less proposal (silent-degradation)

**Vector:** The system proposer's FTNS balance depletes; every
`create_arbitration_proposal` call returns `None`. Disputed records
accumulate in the queue but no council surface receives them. From a
black-box perspective, the system looks fine — uploads complete,
queue records exist — but council review is silently absent.

**Mitigations:**

1. **WARN-log on every sink failure** with structured fields
   (`record_id`, `proposer_id`, exception). Operators monitoring
   logs see the pattern develop. **Pin:** logger.warning calls in
   both `_enqueue_arbitration` and the sink's
   `create_arbitration_proposal` exception handlers.

2. **Queue records are still retrievable** via
   `ArbitrationQueue.list_pending` even when no proposal_id was
   linked. Councils with direct queue read-access (operator
   provides) can author proposals manually.

3. **Operator runbook** (T6.5.gov.next2 module docstring):
   "Monitor logs for repeated failures (suggests Foundation
   address out of FTNS) and refill."

**Residual (acknowledged honest scope):** No automatic alerting
on sink-failure patterns. Production deployments should add a
logs-pipeline alert on the `TokenWeightedVotingProposalSink:
backend rejected arbitration proposal` log line. Documented but
not in-tree.

## §3.18.4 Cross-cutting invariants

These hold across all eight adversaries above:

1. **Default-None preserves legacy.** A node with `threshold_resolver=None`
   AND `arbitration_queue=None` reproduces pre-Item-6 binary
   auto-attribute behavior bit-identically. Operators upgrade by
   configuring env vars; pre-existing deployments keep working unchanged.
   **Pin:** `test_no_resolver_uses_legacy_class_constants` +
   `test_no_arbitration_queue_disables_disputed_band` (both lanes).

2. **Three-tier failure isolation in `_enqueue_arbitration`.** Each of
   `(queue.enqueue)` / `(sink.create_arbitration_proposal)` /
   `(queue.set_proposal_id)` is wrapped in independent try/except;
   any failure WARN-logs and degrades gracefully without blocking the
   upload. This composes with A2 + A8 mitigations.

3. **Three-band routing is symmetric across text + binary lanes.**
   T6.5.x mirrors the embedding-path 3-band branch into the
   binary-fingerprint path so the disputed-band concept applies
   consistently. **Pin:** `TestUploadBinaryThreeBandRouting`
   (5 tests in `test_content_uploader_arbitration.py`).

4. **YAML schema validation is enforced at construction.** Missing
   defaults section, non-mapping defaults, missing tier fields,
   non-numeric tier values all raise `ThresholdResolverError`
   at `ThresholdResolver.__init__` rather than failing late at
   resolve-time.

## §3.18.5 Audit-prep §7.19 (paired)

This addendum is paired with audit-prep §7.19 in
`docs/2026-04-27-cumulative-audit-prep.md`. The audit-prep entry
covers what landed (12 headline guarantees + 6 trust seams + auditor
reading path); this addendum covers who could attack it (8 adversary
classes) and what defends each one. Together they form the Item 6
bundle for external review.

Tag pinning the surface: `prov-1-item-6-t6-5-gov-next2-merge-ready-20260508`.

## §3.18.6 Status

**Threat model:** complete for v1 surface.

**Honest scope deferred:**
- T6.4 calibration corpus (30+ days testnet traffic gate)
- Cross-node arbitration via DHT (R10)
- On-chain arbitration contract (long-term)
- Per-creator daily flag cap (anti-griefing v2)
- Binary-kind hint multipliers (T6.4 territory)
- Automated sink-failure alerting (operator-runbook territory)

**Auditor reading path:**
1. This addendum (start here).
2. Audit-prep §7.19 in
   `docs/2026-04-27-cumulative-audit-prep.md` (paired surface).
3. T6.5 design doc:
   `docs/2026-05-07-PRSM-PROV-1-T6.5-arbitration-queue-design.md`.
4. Source: `prsm/data/dedup/thresholds.py` →
   `prsm/data/dedup/arbitration.py` →
   `prsm/economy/governance/arbitration_sink.py` →
   `prsm/node/content_uploader.py` (search `T6.5`) →
   `prsm/node/node.py` (search `_build_threshold_resolver_or_none`).
5. Tests: `test_dedup_thresholds.py` (32) +
   `test_arbitration_queue.py` (32) +
   `test_content_uploader_arbitration.py` (32) +
   `test_arbitration_sink_adapter.py` (14) +
   `test_node_arbitration_wiring.py` (9) — **119 tests total**.
