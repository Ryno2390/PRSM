# L5 Off-Chain ML Supply-Chain Audit — Request for Proposal

**Engagement:** PRSM off-chain inference + streaming + receipt-attestation
review
**Issuing organization:** PRSM Foundation (Cayman Islands nonprofit)
**Issued:** 2026-05-05
**Response deadline:** 2026-05-26 (3 weeks — broader vendor pool than L3)
**Engagement window:** 2026-06 to 2026-07
**Budget envelope:** $40,000 – $80,000 USD

**Primary contact:** schultzryne@gmail.com / security@prsm.network
**PGP:** see SECURITY.md
**Repository:** https://github.com/Ryno2390/PRSM

---

## 1. Engagement summary (TL;DR)

PRSM is a decentralized inference protocol on Base mainnet. Beyond the
on-chain settlement contracts (covered by L3 / L4 audits running
concurrently), PRSM ships ~100K LoC of Python that performs the actual
ML inference work: tensor-parallel sharded execution across untrusted
relay nodes, signed receipts, KV-cache management, and Tier C
constant-time guarantees that enable confidential inference against
adversarial hosts.

**Standard smart-contract audit firms cannot review this.** The right
vendor profile is an **ML systems security firm or academic lab** with
prior published work on ML side-channels, TEE attestation, and
adversarial relay-host scenarios.

We are seeking a focused review of:

- `prsm/compute/inference/` — TensorParallelInferenceExecutor,
  ParallaxScheduledExecutor, sampling, streaming runners
- `prsm/compute/chain_rpc/` — RPC protocol, handoff tokens, layer stage
  servers, activation codecs
- `prsm/compute/streaming/` — autoregressive/sharded runners,
  KV-cache manager, draft-model speculation

**Total: ~100K LoC across ~180 Python files.** Subset for review (~30K
LoC of security-load-bearing code) is identified in §4.

---

## 2. Why this is L5-specific

The on-chain layer (L3 + L4) handles:
- Receipt validation against committed Merkle root
- Stake / slash economics
- Settlement against escrow

What it does NOT handle, and what L5 covers:
- Whether the OFF-CHAIN signed receipts are forgeable
- Whether a malicious relay host can extract the user's prompt or
  activations from intermediate-layer compute
- Whether the KV-cache leaks across concurrent requests on the same
  relay node
- Whether Tier C's constant-time padding (`FixedRateStreamingRunner`,
  `BatchedTrailingShardedExecutor`) actually achieves the
  threat-model-stated indistinguishability under realistic observers
- Whether per-iteration attestation chains are tamper-evident under
  adversarial sequence manipulation
- Whether the manifest DHT (Phase 3.x.5) is sybil-resistant under
  Byzantine majority

These are **ML systems security** questions, distinct from smart-
contract security or generic application security.

---

## 3. Vendor preferences

**Top picks:**

1. **Trail of Bits — ML Assurance practice.** Cross-pollination with
   our L3 + L4 firm engagement. Prior published work on ML
   side-channels, model extraction, TEE auditing.
2. **NCC Group — Cryptography Services + ML.** Has reviewed
   confidential-computing platforms; fits our Tier C scope.
3. **Galois — Formal Methods + ML Safety.** Prior DARPA-funded ML
   safety work; well-suited for the constant-time padding
   indistinguishability proofs.

**Academic candidates** (lower cost, slower wall-clock, often higher
methodological rigor):

4. **Berkeley RDI** (Center for Responsible Decentralized
   Intelligence) — published work on decentralized ML threat models.
5. **CMU CyLab — ML Systems Security** — published work on inference
   attack surfaces.
6. **Stanford CRFM** — published work on foundation-model security.
7. **MIT CSAIL — Adversarial ML group**.

**Selection criteria:**

- Published prior work on ML side-channels OR TEE attestation OR
  adversarial relay scenarios (last 24 months).
- Apache-2.0 / MIT-friendly engagement terms (we want to publish the
  report).
- Availability in the W2-W14 window (target start: 2026-06).
- Quoted price within the $40K-$80K envelope.

This RFP is being sent to **#1 (Trail of Bits ML)** and **#2 (NCC ML)**
in commercial track plus **#4 (Berkeley RDI)** and **#5 (CMU CyLab)**
in academic track. Proposals from #3 / #6 / #7 considered if scope fit
or availability is better.

---

## 4. Scope of work

### 4.1 In scope

**Tier 1 (must review — security-load-bearing):**

| Path | LoC est. | Purpose |
|------|----------|---------|
| `prsm/compute/inference/executor.py` + adjacent | ~5K | TensorParallel + ParallaxScheduled executor, signing, receipt issuance |
| `prsm/compute/inference/streaming/` | ~6K | Autoregressive + sharded + speculation runners |
| `prsm/compute/inference/streaming/kv_cache.py` + manager | ~3K | KV-cache lifecycle, eviction, rollback |
| `prsm/compute/chain_rpc/server.py` + client | ~4K | RPC protocol, handoff tokens, layer stage servers |
| `prsm/compute/chain_rpc/codec.py` + activation chunker | ~3K | Activation tensor encoding + chunking |
| `prsm/compute/inference/streaming/tier_c/` | ~3K | Constant-time padding decorators (M1/M2 — FixedRate / BatchedTrailing) |
| `prsm/compute/receipts.py` (sign/verify) | ~2K | Ed25519 receipt signing/verification |
| `prsm/compute/chain_rpc/encrypted_handoff.py` | ~2K | Per-request ephemeral X25519 + encrypted activation handoff |

**Subtotal Tier 1: ~28K LoC.**

**Tier 2 (review if budget allows):**

| Path | LoC est. | Purpose |
|------|----------|---------|
| `prsm/compute/inference/scheduling/` | ~3K | Adaptive K speculation, draft-model integration |
| `prsm/compute/manifest_dht/` | ~3K | Manifest DHT (Phase 3.x.5 — sybil resistance) |
| `prsm/compute/inference/timing/` (if present) | ~1K | Timing-side-channel mitigations |

**Subtotal Tier 2: ~7K LoC.**

**Total in-scope (Tier 1 + 2): ~35K LoC of ~100K total Python.** The
remaining ~65K is application logic / wallet / fiat / infra and is out
of scope (covered by L1/L4 SAST + general code review).

### 4.2 Audit dimensions

The auditor's report should cover at minimum:

1. **Receipt forgery surface.** Can a malicious relay node forge a
   signed receipt that the on-chain Registry would accept as valid?
   This is the inverse problem of L3's on-chain Ed25519 verifier
   review — L3 verifies the verifier; L5 verifies the SIGNER.

2. **Activation extraction by relay hosts.** Each layer-stage server
   in a tensor-parallel pipeline sees intermediate activations. Can a
   malicious relay reconstruct the user's prompt or output from those
   intermediates? Especially under speculation (where draft tokens
   propagate), under chunked dispatch, and across concurrent
   requests on the same node.

3. **KV-cache cross-contamination.** Phase 3.x.11 introduced sharded
   KV-cache management with per-request keys. Validate isolation
   under: (a) concurrent requests from different users on same
   relay, (b) request-cancellation + cache-eviction races, (c)
   speculation rollback paths (RollbackCacheRequest, replay-prefix
   forward path).

4. **Tier C constant-time guarantee.** `FixedRateStreamingRunner` (M1
   cadence-driven) and `BatchedTrailingStreamingRunner` (M2) are
   designed to make a network observer unable to distinguish prompt
   length / output length. Validate the indistinguishability claim
   under realistic observation models (timing + size + count).
   Threat-model docs at
   `docs/2026-04-22-r3-threat-model.md` + addenda.

5. **Per-iteration attestation tamper-evidence.** Phase 3.x.11.x
   added `IterationAttestation` per-token / per-iteration. Validate
   that an attacker who controls a subset of iterations cannot stitch
   a valid-looking attestation chain that omits or replaces
   adversarial iterations.

6. **Manifest DHT sybil resistance.** Phase 3.x.5 manifest DHT —
   validate the sybil + Byzantine-majority threat model against
   realistic adversary capabilities.

7. **Per-request ephemeral key handling.** Phase 3.x.11.q.y' added
   per-request DH key negotiation + X25519AnchoredCipher factory.
   Validate that per-request keys are: (a) not reused across
   requests, (b) not predictable from observable state, (c) cleanly
   rotated on rollback.

8. **Off-chain Ed25519 implementations.** Off-chain signing/verification
   uses cryptography library or PyNaCl. Validate library version pins,
   absence of low-order point acceptance, and consistency with the
   on-chain verifier (so honest signers don't get challenges they
   shouldn't).

### 4.3 Out of scope

| Path / Domain | Why out of scope |
|---------------|------------------|
| `contracts/contracts/` | L3 (crypto) + L4 (composition) coverage |
| `prsm/wallet/`, `prsm/fiat/` | Covered by Phase 4-5 vendor decisions |
| `prsm/p2p/` | Covered by L6f network pen-test |
| Front-end / SDK | Not deployed adversarially |
| Generic Python supply chain (pip-audit) | Covered by L1 dependency-vuln scanning |

---

## 5. Deliverables

We expect the engagement to produce:

1. **Audit report (primary deliverable).**
   - Severity-classified findings (CRITICAL / HIGH / MEDIUM / LOW / INFO).
   - For each finding: description, exploit primitive, recommended fix,
     reproducibility (test case if possible).
   - Executive summary suitable for non-ML-experts (1-2 pages).
   - Detailed technical body with file/line references.

2. **Adversarial-relay test harness (if applicable).** If new test
   vectors or exploit-prototype code are produced during the review,
   hand them off in a form we can include in
   `audits/findings/L5-ml-supply-chain/`.

3. **Constant-time formal proof or indistinguishability bound.** If
   the auditor has formal-methods capability, we'd value an
   indistinguishability bound on the Tier C cadence/padding
   decorators. Optional.

4. **Optional: delta review on fixes.** If the report contains
   findings, we'll prepare fixes within 2-4 weeks. We'd value a brief
   delta review (≤ 1 week) confirming the fixes resolve the findings.
   Quote separately.

5. **License of deliverables.** Apache-2.0 or equivalent — we want
   the right to publish on GitHub and link from
   `audits/findings/L5-ml-supply-chain/`.

---

## 6. Pre-engagement artifacts

To minimize ramp-up cost, the auditor will receive:

| Artifact | Location | What it gives |
|----------|----------|---------------|
| **R3 threat model** | `docs/2026-04-22-r3-threat-model.md` | Full threat-model from sprint planning |
| **Phase 3.x.11 threat addendum** | `docs/2026-04-30-phase3.x.11-threat-model-addendum.md` | Sharded autoregressive + speculation threat addendum |
| **Audit-prep §7.x notes** | `docs/2026-04-30-cumulative-audit-prep-bundle-h.md` §7.x | Per-sprint security commentary on each Phase 3.x.X subsystem |
| **R7 KV-compression benchmark** | `docs/2026-04-22-r7-benchmark-plan.md` | Adjacent threat-model context |
| **R6 post-quantum trigger memo** | `docs/2026-04-22-r6-post-quantum-trigger-watch.md` | Post-quantum migration trigger watch (informs Ed25519 lifetime) |
| **Master audit plan** | `audits/AUDIT_PLAN.md` v1.1 | Layer L5 = this engagement |
| **L3 RFP (related)** | `audits/rfp/L3-ed25519-crypto-rfp.md` | The on-chain crypto specialist engagement (concurrent) |
| **L4 RFP (related)** | `audits/rfp/L4-code4rena-contest-scope.md` + `L4-firm-rfp-addendum-20260505.md` | The on-chain protocol audit (concurrent) |

The auditor will additionally have access to:

- Full repository: https://github.com/Ryno2390/PRSM
- Python test suite (~283 unit + 2 E2E passing pre-L5; counts may
  change during engagement)
- Live testnet: bootstrap1.prsm-network.com:8765 (DigitalOcean
  droplet) for adversarial-relay scenarios

---

## 7. Engagement details

### 7.1 Timeline

- **2026-05-26:** Vendor proposals due
- **2026-06-02:** Vendor selection
- **2026-06-09:** Engagement starts (target — overlap with L3 + L4)
- **2026-06-09 to 2026-07-07:** Audit window (4 weeks for Tier 1; 6
  for Tier 1 + 2)
- **2026-07-14:** Final report delivered
- **2026-07-28:** Optional delta review on fixes (if findings)

### 7.2 Communication

- Primary contact: founder (founder@prsm.network)
- Backup contact: deputy-founder (per
  `docs/security/D7_DEPUTY_FOUNDER_SUCCESSION.md`)
- Engagement Slack channel / Signal group can be set up on request
- Findings reported under coordinated-disclosure terms — public
  disclosure after fixes deploy (no embargo > 90 days)

### 7.3 Engagement terms

- **Compensation:** lump sum or T&M, per vendor proposal. Budget
  envelope $40K-$80K USD.
- **Payment:** 40% on engagement start, 40% on draft report, 20% on
  final. Wire / USDC / USDT acceptable.
- **NDA:** mutual NDA covering pre-disclosure findings; superseded by
  public-report publication on disclosure date.
- **Liability:** auditor's standard MSA, capped at engagement fee.
- **Right to publish:** Foundation retains right to publish the report
  (Apache-2.0 or equivalent). Auditor may co-publish.
- **Right to disclose vendor name:** Foundation may name the auditor.
- **Academic engagements:** flexibility on payment terms, IP licensing,
  and student-author co-authorship per institution norms.

---

## 8. Proposal request

Vendor proposals should include:

1. **ML systems security qualifications.** Prior published work on
   ML side-channels / TEE attestation / adversarial-relay scenarios
   (last 24 months preferred).
2. **Lead reviewer CV.** Specifically the person who would lead the
   engagement.
3. **Methodology.** How you approach ML systems security audits
   (threat modeling? differential testing? formal methods?
   adversarial probing of live systems?).
4. **Scope quote.** Days of effort, total fee, breakdown by Tier 1
   vs Tier 2 (so we can scope-down if budget is tight).
5. **Timeline.** Earliest start, expected completion, delta-review
   terms.
6. **Sample report.** Public sample of comparable ML systems
   security audit work (model extraction, side-channels,
   confidential inference, etc.).
7. **Engagement terms.** License, NDA, liability per §7.3.
8. **References.** 2-3 prior clients we can contact.

---

## 9. Submission

Send proposals to: **schultzryne@gmail.com** (founder) and CC
**security@prsm.network**

Subject: `PRSM L5 ML Supply-Chain Audit RFP — [Your Firm]`

Attachments: PDF preferred. Sample reports as separate attachments.

We will acknowledge receipt within 24 hours and schedule a 30-minute
clarification call within 1 week.

---

## 10. Defense-in-depth context

L5 is one of 4 reviewers covering different surfaces:

| Layer | Surface | Vendor track |
|-------|---------|--------------|
| L3 | On-chain crypto (Ed25519Lib + Sha512) | Crypto specialist (ToB Crypto / NCC Crypto) |
| L4 | On-chain protocol composition | Code4rena public + firm pair-review |
| **L5 (this RFP)** | **Off-chain ML supply chain** | **ML systems security firm or academic lab** |
| L6f | Network infrastructure | Network pen-test firm |

A finding that escapes all four is the residual risk we're explicitly
accepting. Auditor should know: this is one engagement of four, not
the only line of defense.

---

## 11. Signoff

**Issuing party:** PRSM Foundation (Cayman Islands)
**Authorized signatory:** Founder
**Date issued:** 2026-05-05

This RFP is non-binding until a mutually-executed engagement agreement
is in place. Foundation reserves the right to decline all proposals
or adjust scope based on responses received.

---

*See `audits/AUDIT_PLAN.md` §5 L5 for the strategic rationale behind
treating ML supply-chain as a distinct audit layer.*
