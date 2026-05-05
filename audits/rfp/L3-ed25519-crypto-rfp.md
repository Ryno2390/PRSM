# L3 Cryptographic Specialist Audit — Request for Proposal

**Engagement:** PRSM Ed25519 + SHA-512 on-chain verifier review
**Issuing organization:** Foundation (Cayman Islands nonprofit)
**Issued:** 2026-05-05
**Response deadline:** 2026-05-19 (2 weeks)
**Engagement window:** W2-W4 of 2026-05 (target start: 2026-05-26)
**Budget envelope:** $15,000 – $30,000 USD

**Primary contact:** schultzryne@gmail.com / security@prsm.network
**PGP:** see SECURITY.md
**Repository:** https://github.com/Ryno2390/PRSM

---

## 1. Engagement summary (TL;DR)

PRSM is a decentralized inference / royalty / settlement protocol on Base
mainnet. Our Phase 3.1 batch settlement system uses on-chain Ed25519
signature verification for the INVALID_SIGNATURE challenge path. The
verifier is a hand-rolled Solidity port of `chengwenxi/Ed25519`
(Apache-2.0).

We are seeking a **cryptography-specialist firm** (NOT a generic smart-
contract audit firm) to perform a focused review of:

- `contracts/contracts/lib/Ed25519Lib.sol` (887 LoC)
- `contracts/contracts/lib/Sha512.sol`     (328 LoC)
- `contracts/contracts/Ed25519Verifier.sol` (59 LoC integration wrapper)

**Total:** ~1,274 LoC of curve-arithmetic + hash code under audit.

We have completed pre-engagement work (RFC 8032 + FIPS 180-4 test vectors,
upstream port diff, caller-assumptions memo) so the auditor can start with
a clean baseline. See §6.

---

## 2. Why us / why this scope

This is a focused crypto-primitive audit, not a sprawling protocol review.
The scope is narrow and self-contained:

- **Inputs:** raw bytes (publicKey, signature, messageHash) — no protocol
  context required.
- **Outputs:** `bool` (valid / invalid) — no state mutation, no callbacks.
- **Trust boundary:** the caller (BatchSettlementRegistry) treats the
  return value as ground truth on whether a signature is valid.

The deliverable is a report on whether `Ed25519Lib` faithfully implements
RFC 8032 §5.1.7 and whether `Sha512` faithfully implements FIPS 180-4 — at
the gas-cost and edge-case level that matters in adversarial contexts.

**Why your firm specifically:** we prefer cryptography practices with prior
published work on Ed25519, Curve25519, or comparable elliptic-curve
implementation review. See §3.

---

## 3. Vendor preferences

**Top picks (in approximate order):**

1. **Trail of Bits — Cryptography practice.** Prior work on ZCash, Zoom
   (Ed25519 identity flows), multiple curve implementations.
2. **NCC Group — Cryptography Services.** Prior work on Signal,
   libsodium-adjacent code, multiple Ed25519 implementations.
3. **Cure53.** Prior work on Signal, Tutanota, multiple crypto messengers.
4. **Kudelski Security.** Specialized crypto firm.
5. **Least Authority.** ZCash + Tezos crypto auditing.

This RFP is being sent to **#1 and #2**. We'd consider proposals from
others on the list if availability or scope fit is better.

**Selection criteria:**

- Cryptography practice (not generic smart-contract audit)
- Curve-arithmetic / hand-rolled-crypto experience
- Apache-2.0 / MIT-friendly engagement terms (we want to publish the
  report)
- Availability in the W2-W4 window (target start: 2026-05-26)
- Quoted price within the $15K–$30K envelope

---

## 4. Scope of work

### 4.1 In scope

| File | LoC | Description |
|------|-----|-------------|
| `contracts/contracts/lib/Ed25519Lib.sol` | 887 | Curve25519 + Ed25519 verify implementation |
| `contracts/contracts/lib/Sha512.sol`     | 328 | SHA-512 hash function |
| `contracts/contracts/Ed25519Verifier.sol` | 59  | Integration wrapper exposing `ISignatureVerifier` |

### 4.2 Audit dimensions

The auditor's report should cover at minimum:

1. **RFC 8032 §5.1.7 line-by-line conformance.** Does the verify path
   match the spec, including:
   - R, A, S decoding (canonical / non-canonical handling)
   - hash-then-reduce of `R || A || M` (the SHA-512 of the catenation,
     reduced mod L)
   - subgroup check: is `[8]A` rejected for low-order points?
   - canonicality of S: rejection of `S ≥ L`?
   - point at infinity handling
   - the 8R = 8(SB - hA) batch verification equation (per RFC 8032 §5.1.7
     step 3)

2. **FIPS 180-4 SHA-512 conformance.** Does `Sha512.sol` produce identical
   output to the reference implementation on:
   - empty input
   - single-byte input
   - block-aligned multi-block input
   - input crossing the 64-bit length encoding boundary
   - inputs near the 2^128 length limit

3. **Upstream port faithfulness.** Compare `Ed25519Lib.sol` against
   `chengwenxi/Ed25519` upstream. Identify every algorithmic change made
   during the Solidity 0.8.22 port.
   See `audits/findings/L3-crypto/upstream-port-diff.md`. Is the diff
   complete? Did anything slip in beyond the documented mechanical changes?

4. **Edge cases / known attack surfaces.**
   - Low-order-point inputs (the 8 low-order points of Curve25519)
   - Non-canonical scalar / point encodings (S ≥ L, A ≥ p)
   - Malleability under (R, S) → (R, S + L) mutation — does verify reject
     non-canonical S?
   - Batch-vs-single verification semantics (RFC 8032 §5.1.7 mandates
     8h(R, A, M) = 8(SB - hA), not h(R, A, M) = SB - hA)
   - Cofactor handling

5. **Side-channel review.**
   - Gas-side-channel: an attacker observing the gas cost of a verify
     could distinguish branches (e.g., short-circuit on invalid length).
     Is this exploitable?
   - Variable-time arithmetic: any operations that branch on secret data?
     (Note: this is a verifier — the public key + signature + message are
     all PUBLIC inputs by definition, so secret-dependent timing is
     largely a non-issue. We want this confirmed, not assumed.)

6. **Integration assumptions.** Verify that the assumptions
   `BatchSettlementRegistry` makes about the verifier (A1–A8 in
   `audits/findings/L3-crypto/caller-assumptions.md`) hold. In
   particular:
   - **A1:** `verify(...)` returns `false` rather than reverts on
     malformed input
   - **A2:** function is deterministic and `pure` (post-MEDIUM C-INT-02)
   - **A3:** no state mutation
   - **A4:** the function does not call back into the caller
   - **A5:** invalid length inputs return `false` cleanly (no OOG)
   - …(see memo)

### 4.3 Out of scope

The auditor should NOT spend time on:

- BatchSettlementRegistry integration logic (covered by L4 contest)
- Off-chain Python Ed25519 implementations (covered by L5 ML supply-
  chain audit)
- Other contracts in the bundle (EscrowPool, StakeBond,
  RoyaltyDistributor, etc.)
- General Solidity best-practices linting (Slither, Mythril, etc. are
  L1 tooling)

If during the review the auditor identifies issues outside scope that
have direct cryptographic impact (e.g., an integration error that
neutralizes the verifier), please note them in an "out-of-scope
observations" appendix — they're useful to us even if not the focus.

---

## 5. Deliverables

We expect the engagement to produce:

1. **Audit report (primary deliverable).**
   - Severity-classified findings (CRITICAL / HIGH / MEDIUM / LOW / INFO)
   - For each finding: description, exploit primitive, recommended fix,
     reproducibility (test case if possible)
   - Executive summary suitable for non-cryptographers (1-2 pages)
   - Detailed technical body with line-level references

2. **Reproducible test harness (if applicable).** If new test vectors or
   PoC contracts are produced during the review, hand them off in a form
   we can include in our `audits/findings/L3-crypto/` folder.

3. **Optional: delta review on fixes.** If the report contains
   findings, we'll prepare fixes within 1-2 weeks. We'd value a brief
   delta review (≤ 1 week) confirming the fixes resolve the findings.
   Quote separately if not included in base scope.

4. **License of deliverables.** Apache-2.0 or equivalent — we want the
   right to publish the report on GitHub and link from
   `audits/findings/L3-crypto/`. Confirm in your proposal.

---

## 6. Pre-engagement artifacts (already prepared)

To minimize ramp-up cost, we have prepared the following materials. They
are included as context, NOT as a substitute for the auditor's
independent review.

| Artifact | Location | Status |
|----------|----------|--------|
| **L3 decision memo** | `audits/decisions/L3-ed25519-decision.md` | Founder-ratified 2026-05-05 |
| **RFC 8032 test vectors (Ed25519)** | `audits/findings/L3-crypto/test-vectors-ed25519.md` | 11/11 passing |
| **FIPS 180-4 test vectors (SHA-512)** | `audits/findings/L3-crypto/test-vectors-sha512.md` | 11/11 passing + avalanche |
| **Upstream port diff** | `audits/findings/L3-crypto/upstream-port-diff.md` | Mechanical changes only (pragma bump, library rename, named imports, `unchecked` blocks for 0.8.x) |
| **Caller-assumptions memo (A1–A8)** | `audits/findings/L3-crypto/caller-assumptions.md` | Documents what BatchSettlementRegistry assumes about the verifier |
| **Master audit plan** | `audits/AUDIT_PLAN.md` v1.1 | Layer L3 = this engagement |

The auditor will additionally have access to:

- Full repository: https://github.com/Ryno2390/PRSM
- Hardhat test suite (488 tests passing as of 2026-05-05)
- Existing 29 unit tests across `Ed25519Verifier`, `Ed25519Lib`,
  `Sha512` (7 baseline + 22 added during pre-engagement work)

---

## 7. Engagement details

### 7.1 Timeline

- **2026-05-19:** Vendor proposals due
- **2026-05-22:** Vendor selection
- **2026-05-26:** Engagement starts (target)
- **2026-06-09 to 2026-06-23:** Audit window (2-4 weeks per vendor scope)
- **2026-06-30:** Final report delivered
- **2026-07-14:** Optional delta review on fixes (if findings)

### 7.2 Communication

- Primary contact: founder (founder@prsm.network)
- Backup contact: deputy-founder (per
  `docs/security/D7_DEPUTY_FOUNDER_SUCCESSION.md`)
- Engagement Slack channel / Signal group can be set up on request
- Findings reported under coordinated-disclosure terms — public
  disclosure after fixes deploy (no embargo > 90 days)

### 7.3 Engagement terms

- **Compensation:** lump sum or T&M, per vendor proposal. Budget
  envelope $15K–$30K USD.
- **Payment:** 50% on engagement start, 50% on final-report delivery.
  Wire / USDC / USDT acceptable.
- **NDA:** mutual NDA covering pre-disclosure findings; superseded by
  public-report publication on disclosure date.
- **Liability:** auditor's standard MSA, capped at engagement fee.
- **Right to publish:** Foundation retains right to publish the report
  (Apache-2.0 or equivalent). Auditor may co-publish on their own site.
- **Right to disclose vendor name:** Foundation may name the auditor in
  the published report (this is the norm; confirm in your proposal).

### 7.4 Mitigating risk context

The auditor should know:

- **The verifier is hot-swappable.**
  `BatchSettlementRegistry.setSignatureVerifier` allows the Foundation
  Safe to upgrade the verifier without protocol migration. Per L2 audit
  MEDIUM D-03, in-flight batches use the verifier address snapshotted at
  commit time, so swaps affect FUTURE batches only. This means findings
  can be remediated by deploying a new verifier and pointing Registry at
  it — not a redeploy of the entire protocol.
- **The verifier is on the challenge-only path.** Happy-path commit +
  finalize do NOT call `verify`. Only the INVALID_SIGNATURE challenge
  reason invokes the verifier. This means a verifier bug primarily
  affects the dispute path, not normal traffic.
- **Defense-in-depth.** This audit is one of 4 reviewers (L2 multi-team
  AI audit completed; L4 public contest; L11 annual re-audit). A bug
  that escapes all four is the residual risk we're explicitly accepting.

---

## 8. Proposal request

Vendor proposals should include:

1. **Cryptography practice qualifications.** Prior Ed25519 / Curve25519 /
   curve-arithmetic engagements (last 24 months preferred).
2. **Lead auditor CV.** Cryptographer who would lead the engagement.
3. **Methodology.** How you approach hand-rolled-crypto audits in EVM
   contexts (analytical proof? fuzzing? differential testing against
   reference implementations? formal verification?).
4. **Scope quote.** Days of effort, total fee, breakdown by phase.
5. **Timeline.** Earliest start, expected completion, delta-review terms.
6. **Sample report.** Public sample of comparable curve-implementation
   audit work (Ed25519 / secp256k1 / BLS / similar).
7. **Engagement terms.** License, NDA, liability per §7.3.
8. **References.** 2-3 prior clients we can contact.

---

## 9. Submission

Send proposals to: **schultzryne@gmail.com** (founder) and CC
**security@prsm.network**

Subject: `PRSM L3 Ed25519 Audit RFP — [Your Firm]`

Attachments: PDF preferred for the proposal. Please include any sample
reports as separate attachments.

We will acknowledge receipt within 24 hours and schedule a 30-minute
clarification call within 1 week.

---

## 10. Signoff

**Issuing party:** PRSM Foundation (Cayman Islands)
**Authorized signatory:** Founder
**Date issued:** 2026-05-05

This RFP is non-binding until a mutually-executed engagement agreement
is in place. Foundation reserves the right to decline all proposals or
adjust scope based on responses received.

---

*See `audits/decisions/L3-ed25519-decision.md` for the strategic
rationale behind engaging a specialist firm rather than replacing the
hand-rolled library or rearchitecting the protocol's signature surface.*
