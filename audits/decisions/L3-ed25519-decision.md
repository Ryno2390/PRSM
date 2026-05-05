# L3 Decision Memo — Ed25519 On-Chain Verifier

**Decision:** Engage specialist crypto audit on the **existing**
`Ed25519Lib.sol` + `Sha512.sol`. Do **NOT** replace the library or
re-architect the protocol's signature surface.

**Date:** 2026-05-05
**Author:** Founder
**Decision lifecycle:** Draft → Foundation council ratification → action
**Supersedes:** Open question in `AUDIT_PLAN.md` §5 L3
**Status:** Draft (this memo). Pending foundation review.

---

## 1. Question

The PRSM audit-bundle stack uses a hand-rolled Solidity Ed25519 verifier
(`Ed25519Lib.sol` 887 LoC + `Sha512.sol` 328 LoC) ported from
`chengwenxi/Ed25519` (Apache-2.0). Hand-rolled cryptography in EVM is the
single highest-risk file class in the bundle. `AUDIT_PLAN.md` §5 L3 marked
this as a decision blocker on layer L3:

> Either (a) replace with a battle-tested alternative ($0, requires
> evaluation), or (b) engage specialist firm review ($15–30K, weeks).

This memo runs the evaluation and reaches a decision.

## 2. Investigation summary

### 2.1 Native EVM precompile?

**No.** Neither Ethereum mainnet nor Base ships an Ed25519 precompile as of
2026-05.

The closest precompile is `ecrecover` (address 0x01) for ECDSA-secp256k1, plus
the recently-adopted **RIP-7212 P256VERIFY precompile** (address 0x100, 3450
gas) for secp256r1 — adopted by Base, Optimism, Arbitrum, zkSync. **RIP-7212
is the wrong curve** (P-256, not Curve25519); cannot be used for Ed25519
verification.

Ed25519-specific precompile proposals exist (e.g., discussion threads, never
finalized) but no live precompile on Base. **Conclusion: not currently
available, and not on a near-term roadmap I can verify.**

### 2.2 Battle-tested alternative Solidity library?

**No.** I evaluated every viable candidate found via search:

| Candidate | Status | Verdict |
|-----------|--------|---------|
| `chengwenxi/Ed25519` (current upstream) | 2 commits, 11 stars, 3 forks. **No published audit.** Apache-2.0. | Already in use; not "battle-tested" by industry standard. |
| `javgh/ed25519-solidity` | Partial (only "some" elliptic curve operations). 1.25M gas per scalar mult. | **Incomplete** — does not implement full RFC 8032 verify. Reject. |
| Witnet `vrf-solidity` | Verifiable Random Function for **secp256k1**, not Ed25519. | Wrong curve. Reject. |
| Witnet `bls-solidity` | BLS signatures over BN256, not Ed25519. | Wrong primitive. Reject. |
| Daimo Ed25519 | Daimo uses **passkeys (secp256r1)**, not Ed25519. | Wrong curve. Reject. |
| OpenZeppelin Ed25519 | Does not exist as of OZ v5.x. | N/A. |
| Solady | Gas-optimized utilities; no Ed25519 verifier. | N/A. |
| Hedera HIP-632 system contract | Hedera-only, not portable to Base. | Wrong network. Reject. |

**Conclusion:** **There is no well-audited, battle-tested, drop-in
replacement for the hand-rolled Ed25519 verifier on Base.** The realistic
on-chain Ed25519 verifier landscape consists of (a) `chengwenxi/Ed25519` and
its derivatives, and (b) bespoke audits of those derivatives. The PRSM port
is one of those derivatives.

### 2.3 Architectural-rework alternative — switch to secp256k1?

Considered: **switch the entire protocol's signature surface from Ed25519 to
secp256k1** (use `ecrecover` on-chain, ~3000 gas instead of ~500K–2M gas).

**Pros:**
- ~150–600× cheaper on-chain.
- Native EVM tooling, no hand-rolled crypto required.
- ecrecover has been in production since Ethereum genesis (2015); its
  cryptographic correctness is established.

**Cons:**
- Requires rewrite of off-chain Ed25519 across ~10 Python files
  (`prsm/mcp_server.py`, `prsm/marketplace/listing.py`, `prsm/storage/distribution.py`,
  `prsm/compute/shard_receipt.py`, `prsm/node/bootstrap.py`,
  `prsm/node/settler_registry.py`, `prsm/node/transport.py`,
  `prsm/node/dag_ledger.py`, `prsm/node/storage_proofs.py`,
  `prsm/node/agent_identity.py`).
- Breaks compatibility with all Phase 3.x receipts already produced.
- Phase 3.x.3 PublisherKeyAnchor on-chain key registry would need migration.
- Non-EVM ecosystems (Solana, NaCl, Signal, modern non-blockchain crypto)
  use Ed25519. PRSM's Ed25519 design is consistent with non-EVM crypto best
  practice; switching to secp256k1 ties PRSM to an EVM-specific curve
  forever.
- Estimated rework: 4–8 weeks engineering + 2–4 weeks testing + re-deploy
  PublisherKeyAnchor + re-issue all in-flight receipts.

**Critical gating fact** (discovered in source review): the on-chain
Ed25519Verifier is invoked **only on the INVALID_SIGNATURE challenge path**
of `BatchSettlementRegistry` (per contract docstring + line 555 of
`BatchSettlementRegistry.sol`). Happy-path commit + finalize **incur zero
Ed25519 cost.** Therefore the 500K–2M gas hit only fires on rare disputes,
not normal traffic. The economic argument for switching curves loses much of
its weight under this constraint.

**Conclusion:** Architectural rework would be a major multi-month
engineering project for marginal benefit (cheaper-but-rarely-fired challenge
path). **Reject as scope creep.**

### 2.4 Critical mitigating fact — verifier is hot-swappable

`BatchSettlementRegistry.sol` exposes the Ed25519 verifier through an
`ISignatureVerifier` interface and includes a setter:

```solidity
function setSignatureVerifier(address newVerifier) external onlyOwner {
    address old = address(signatureVerifier);
    signatureVerifier = ISignatureVerifier(newVerifier);
    // ...
}
```
*(`BatchSettlementRegistry.sol:713-714`)*

This means **the verifier can be swapped post-deploy** by Foundation Safe
governance. If a better-audited Ed25519 verifier emerges later (or if a Base
Ed25519 precompile ships, or if our specialist audit finds a fixable bug),
we deploy a replacement and call `setSignatureVerifier`. No migration of
in-flight receipts required because off-chain Ed25519 keys are unchanged.

**This drastically reduces the lock-in risk of the current decision.** The
Ed25519Lib choice is reversible.

## 3. Decision

**Engage specialist crypto audit on the existing `Ed25519Lib.sol` +
`Sha512.sol`.**

**Rationale:**
1. No battle-tested replacement exists; the "replace" path is not viable.
2. Architectural rework is disproportionate scope for a challenge-path-only
   verifier.
3. The verifier is hot-swappable, so any specialist findings can be
   remediated by deploying a new verifier and pointing Registry at it.
4. A specialist audit produces an artifact (audit report) that downstream
   reviewers (L4 contest, L11 annual re-audit, investors, regulators) can
   reference — strictly more valuable than a self-assertion that the lib is
   probably fine.

**Cost:** $15–30K.

**Wall-clock:** 2–4 weeks (from engagement to delivered report).

**Vendor selection criteria:**
- Cryptography practice with curve-arithmetic experience (NOT a generic
  smart-contract audit firm).
- Prior published work on Ed25519, Curve25519, or comparable
  curve-implementation review.
- Apache-2.0/MIT-friendly engagement terms (so we can publish the report).

**Recommended candidates** (in approximate order of preference for this
specific scope):

1. **Trail of Bits — Cryptography practice.** Audited ZCash, Zoom (Ed25519
   identity flows), multiple curve implementations. Strong reputational
   weight.
2. **NCC Group — Cryptography Services.** Audited Signal, libsodium-adjacent
   work, multiple Ed25519 implementations.
3. **Cure53.** Audited Signal, Tutanota, multiple crypto messengers; strong
   curve-implementation track record.
4. **Kudelski Security.** Specialized crypto firm; smaller engagements
   typical.
5. **Least Authority.** ZCash + Tezos crypto auditing; strong but
   ZK-protocol-leaning.

**My ranked pick:** Trail of Bits or NCC Group. Either is appropriate; pick
based on (a) availability in the W2–W4 audit window and (b) quoted scope
fit.

## 4. Scope of the engagement

The auditor's scope of work should be:

**In scope:**
- `contracts/contracts/lib/Ed25519Lib.sol` (887 LoC)
- `contracts/contracts/lib/Sha512.sol` (328 LoC)
- `contracts/contracts/Ed25519Verifier.sol` (59 LoC, integration wrapper)
- Comparison of `Ed25519Lib` against `chengwenxi/Ed25519` upstream — verify
  the port is faithful and didn't introduce regressions.
- Comparison against RFC 8032 §5.1.7 line-by-line.
- Test-vector validation against RFC 8032 Appendix A.
- Test-vector validation of Sha512 against FIPS 180-4 known vectors.
- Side-channel review (timing-based attacker observing gas usage).
- Edge cases: low-order points, non-canonical encodings (S ≥ L), point at
  infinity, malleability under (R, S) mutation.

**Out of scope:**
- BatchSettlementRegistry integration logic (covered by L4 contest).
- Off-chain Python Ed25519 (covered by L5 ML supply-chain audit).
- Other contracts in the bundle.

**Deliverable:**
- Audit report with severity-classified findings.
- Test vectors + reproducible test harness if not already present.
- Public-facing summary suitable for inclusion in `audits/findings/L3-crypto/`.

## 5. Pre-engagement actions (engineering, before vendor RFP)

Before engaging a vendor, do the following internally to reduce auditor
ramp-up time and cost:

1. **Run RFC 8032 Appendix A test vectors against `Ed25519Lib`.** If any
   fail, the lib is broken at spec-conformance level and we surface that
   ourselves (free). If they pass, we hand the auditor a passing baseline.
2. **Run FIPS 180-4 Sha512 known-answer tests.** Same logic.
3. **Diff `Ed25519Lib.sol` against the upstream `chengwenxi/Ed25519`.**
   Identify every change made during the port to Solidity 0.8.22. Hand the
   auditor a diff document so they can focus review on changed lines.
4. **Document caller assumptions.** Write a one-page memo on what
   `BatchSettlementRegistry` assumes about the verifier's behavior
   (deterministic, pure, malformed-input → false not revert, no state, etc.)
   so the auditor can verify those assumptions hold.

These four items are ~2 days engineering work and reduce auditor cost by
giving them a clean starting point.

## 6. Decision criteria (preserved for future reference)

If, in the future (e.g., after a Base Ed25519 precompile ships, or a new
audited library emerges), this decision needs to be revisited, the
re-evaluation criteria from `AUDIT_PLAN.md` §5 L3 are:

| Criterion | Replace if… | Audit if… |
|-----------|-------------|-----------|
| Audit history | Candidate has ≥1 published independent audit | No suitable candidate exists |
| Gas cost | Candidate ≤ 110% of current Ed25519Lib gas usage | Current is already gas-optimal |
| License | MIT / Apache 2.0 compatible | License blockers on all candidates |
| Maintenance | Active commits in last 12 months | All candidates abandoned |
| EVM/Base compat | Compiles with our Solidity version, runs on Base | Compatibility blockers |

As of 2026-05-05, the "Audit if…" column wins on every criterion.

## 7. Open issues for ratification

The Foundation council should ratify or push back on:

- **OK to spend $15–30K on this audit?** Falls within the L3 budget envelope
  in `AUDIT_PLAN.md` §8.
- **Trail of Bits vs NCC Group selection?** Both are appropriate; pick on
  availability + scope quote.
- **Public-or-private audit report?** Recommend **public** — it's a
  reputation-building artifact and contains no exploitable details once the
  fixes (if any) are deployed.
- **Pre-engagement actions cleared to start?** §5 above is ~2 days of
  engineering on the founder's plate; OK to proceed in parallel with vendor
  outreach.

## 8. Next actions (sequenced)

1. **Pre-engagement work** ✅ **COMPLETED 2026-05-05** (founder, ~3 hours):
   - ✅ RFC 8032 §7.1 test vectors all 4 verify, with bit-flip / wrong-key /
     wrong-message rejections — 11 passing — `audits/findings/L3-crypto/test-vectors-ed25519.md`
   - ✅ FIPS 180-4 Sha512 KATs all 11 pass + avalanche property —
     `audits/findings/L3-crypto/test-vectors-sha512.md`
   - ✅ Upstream port diff complete; only mechanical changes (pragma bump,
     library rename, named import, `unchecked` blocks for 0.8.x) — no
     algorithmic changes — `audits/findings/L3-crypto/upstream-port-diff.md`
   - ✅ Caller-assumptions memo (A1–A8) covering BatchSettlementRegistry's
     trust model on the verifier — `audits/findings/L3-crypto/caller-assumptions.md`
   - **Test summary:** 29 tests passing across Ed25519Verifier, Ed25519Lib,
     and Sha512 (existing 7 + new 22).
2. **Vendor outreach** (founder, parallel with above):
   - RFP to Trail of Bits Cryptography practice
   - RFP to NCC Group Cryptography Services
   - Quotes back within ~1 week
3. **Engagement** (vendor, 2–4 weeks): contracted scope of §4 above.
4. **Triage + remediation** (founder + vendor, 1–2 weeks): fix any
   findings; re-run vendor delta-review on fixes.
5. **Hot-swap deployment** (only if breaking findings): deploy new verifier,
   call `setSignatureVerifier`. Documented in
   `EXPLOIT_RESPONSE_PLAYBOOK.md` flow.

**Total wall-clock from this memo to L3 cleared: 4–7 weeks.**
This fits inside the W2–W8 window of the master plan timeline (§6.6).

## 9. Risk acceptance

If the council ratifies this decision, the residual risk acknowledged is:

> A subtle implementation bug in `Ed25519Lib.sol` that escapes both our
> RFC-8032 test vectors AND the specialist audit AND the L2 multi-team
> review AND the L4 contest. Probability: low (defense-in-depth across four
> independent reviewers). Impact: forged signatures on the
> INVALID_SIGNATURE challenge path could allow either (a) malicious
> challenges that incorrectly slash honest stakers, or (b) honest
> challenges to be dismissed when they shouldn't be. Mitigation:
> hot-swappable verifier (§2.4) means any post-discovery bug can be patched
> without protocol upgrade.

This is a typical residual-risk profile for hand-rolled cryptography behind
a swappable interface, and is acceptable given the defense-in-depth posture.

---

**Status: draft pending Foundation council ratification.**

When ratified, this memo becomes the L3 decision-of-record and task #330
(`L3 decision`) updates from "decision required" to "audit engagement
in progress."
