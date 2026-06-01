# L4 Firm RFP Addendum — Post-Audit-Sprint State (2026-05-05)

**Type:** Addendum to `docs/2026-04-23-auditor-shortlist-and-rfp.md`
**Issued:** 2026-05-05
**Purpose:** Refresh the audit-bundle's evidence layer for firm RFP recipients
with what's changed since 2026-04-23, so they can scope their proposal
against the current state rather than the 2026-04-23 snapshot.

**Recipients:** Top-3 firms from the existing shortlist:

1. **Trail of Bits** — pair-review track (concurrent with L3 crypto engagement)
2. **Spearbit / Cantina (elite)** — Base-mainnet experience strongest
3. **OpenZeppelin Security** — OZ-pattern fluency

(Firms #4-#6 from the original shortlist remain on standby per original
doc §2.)

---

## 1. What's changed since 2026-04-23

### 1.1 L2 multi-team adversarial review CLOSED (2026-04-30 → 2026-05-05)

We ran 4 parallel AI agent teams (economic / access-control /
signature-crypto / state-composition) against the audit-bundle
contracts. They produced 33 findings: 1 CRITICAL + 7 HIGH + 8 MEDIUM +
LOW/INFO (B-PAUSE-1 closed transitively by HIGH-5 fix, so 7 actionable
MEDIUMs).

**As of 2026-05-05, ALL 1 CRIT + 7 HIGH + 7 actionable MEDIUM are
remediated and committed:**

| ID | Severity | Title | Remediation commit |
|----|----------|-------|--------------------|
| CRIT-1 (C-INT-01) | Critical | Adversarial slashing via unbound `signingMessage` | bind `signingMessageHash` in `ReceiptLeaf` |
| HIGH-2 (A-02/D-01) | High | Slash-evasion race when `unbondDelay < challengeWindow` | cross-contract invariant in `requestUnbond` |
| HIGH-3 (D-02) | High | No pause surface | OZ Pausable on 4 audit-bundle contracts |
| HIGH-5 (B-RENOUNCE-1) | High | Sole-admin renounce bricks FTNSTokenSimple | renounceRole override |
| HIGH-6 (B-CROSS-1) | High | `EscrowPool.settlementRegistry` re-pointable | made immutable |
| HIGH-7 (B-CROSS-3) | High | `StakeBond.slasher` re-pointable | made immutable |
| B-TREASURY-1 | Medium | Treasury-pin missing | canonical Foundation Safe pinned in deploy-provenance.js |
| C-INT-02 | Medium | `ISignatureVerifier.verify` declared `view` not `pure` | interface tightened |
| D-05 | Medium | `setChallengeWindowSeconds` retroactively shortens window | per-batch snapshot |
| B-OWNABLE-1 | Medium | Single-step Ownable everywhere | migrated to Ownable2Step on 7 contracts |
| B-CROSS-2 | Medium | `setFtnsToken` strands real-token balances | totalEscrowedBalance accumulator |
| D-03 | Medium | Owner cross-wire mutation soft-bricks in-flight batches | per-batch escrowPool/stakeBond/verifier snapshots |
| D-04 | Medium | Royalty push-payment with no recovery | RoyaltyDistributor pull-payment refactor |

**Two CRIT-classed findings remain operational, NOT code:**

- **HIGH-1 (A-01) RoyaltyDistributor split** — needs v2 redeploy in
  Week 2 (the deployed v1 has a fixable on-chain bug; non-Ownable +
  non-upgradeable means redeploy is the only fix). v2 source is
  ready and includes the corrected split (HIGH-1) + Pausable (HIGH-3-deferred) +
  pull-payment (D-04). (Note: the originally-planned burn-on-use was
  subsequently DROPPED; the deployed v2 has no burn-on-use.)
- **CRIT-2 (B-FTNS-1) FTNS admin = hot key** — operational handoff
  to Foundation Safe pending; not a code change. Addressed by a
  ceremony in Week 2 alongside the v2 redeploy.

**Source PoCs inverted into REGRESSION tests:** Each L2 finding's PoC
test was inverted post-fix to assert the fix holds. See
`contracts/test/audit-team-{a,b,c,d}/*.test.js`.

### 1.2 L3 cryptographic specialist engagement OPENED

`audits/decisions/L3-ed25519-decision.md` ratified by founder
2026-05-05. RFP packet at `audits/rfp/L3-ed25519-crypto-rfp.md` going
to Trail of Bits Cryptography practice + NCC Group Cryptography
Services. **Ed25519Lib + Sha512 + Ed25519Verifier are out-of-scope
for the L4 firm engagement** — those are L3 territory.

### 1.3 L1 static + symbolic tooling WIRED INTO CI

`.github/workflows/solidity-static-analysis.yml`:

- **Per-PR:** Slither + Aderyn (must pass; SARIF uploaded to GitHub
  code-scanning).
- **Weekly:** Mythril (targeted at 9 audit-bundle contracts) + Halmos +
  Echidna (Halmos/Echidna scaffolded but no-op until property contracts
  exist).

Slither config: `contracts/slither.config.json`.
Documentation: `audits/L1-static-tooling/README.md`.

**Net for the firm RFP:** firms can verify locally that L1 is clean
before pricing. Findings already caught by SAST should NOT be billed.

### 1.4 Test count + coverage

- **2026-04-23 RFP:** 142 Solidity + 283 Python = 427 tests passing
- **2026-05-05:** 488 Solidity tests passing (+345 net new from L2 PoC
  inversions + MEDIUM regressions + Halmos discovery probes)
- Python unit tests unchanged in scope; off-chain components are L5
  scope.

### 1.5 Contracts added to scope since 2026-04-23

None. Scope is unchanged at 9 in-scope contracts (~2,300 SLOC). What
DID change is the IMPLEMENTATION of those contracts — see §1.1
remediation list.

### 1.6 Solidity-config update

`contracts/hardhat.config.js` now sets `viaIR: true` (per MEDIUM D-03
+ D-05 stack-too-deep mitigation when the Batch struct grew with
per-batch snapshot fields). Firms should compile with `viaIR: true`
to match production behavior.

---

## 2. Updated suggested-engagement model

The original RFP's "two-stage Code4rena + firm" recommendation stands.
**With L2 + L3 + L1 + MEDIUM cleanup all closed, the firm's scope can
narrow to:**

1. **Cross-contract composition** between EscrowPool ↔
   BatchSettlementRegistry ↔ StakeBond ↔ RoyaltyDistributor under
   adversarial sequencing.
2. **Phase 7-storage layer** (StorageSlashing + KeyDistribution) — has
   not yet been adversarially reviewed beyond L1 + L2.
3. **Phase 8 emission stack** (EmissionController +
   CompensationDistributor) — same.
4. **UUPS upgrade surface** on FTNSTokenSimple — initialization
   reentry, slot collisions, role-based authorization.
5. **Integration with Foundation Safe** (signer collusion, key
   compromise scenarios) — see `docs/security/INSIDER_THREAT_AND_COLLUSION_POLICY.md`
   for the threat model already in place.

**Out of scope for firm engagement (covered elsewhere):**
- Ed25519Lib / Sha512 / Ed25519Verifier → L3
- Per-contract pattern smells → L1
- Off-chain Python (`prsm/`) → L5
- Front-end / SDK → not deployed on-chain

**Recommended scope quote:** ~$60-80K for ~3 weeks of focused
cross-composition + layered review.

---

## 3. New companion documents firms should review

Beyond the original RFP body, recipients should pull these for context:

| Doc | Purpose |
|-----|---------|
| `audits/AUDIT_PLAN.md` v1.1 | Master 11-layer plan |
| `audits/findings/consolidated.md` | L2 multi-team review summary |
| `contracts/test/audit-team-{a,b,c,d}/` | Inverted L2 PoCs (now regressions) |
| `audits/decisions/L3-ed25519-decision.md` | Why crypto is L3-scope |
| `audits/L1-static-tooling/README.md` | L1 CI tooling status |
| `audits/rfp/L4-code4rena-contest-scope.md` | Concurrent C4 contest scope |
| `SECURITY.md` | Vulnerability disclosure policy |
| `docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md` | Incident response |
| `docs/security/INSIDER_THREAT_AND_COLLUSION_POLICY.md` | Insider threat model |
| `docs/security/KEY_ROTATION_RUNBOOK.md` | Key rotation procedures |

---

## 4. Mainnet-deployed evidence (verifiable)

Firms can verify Foundation operational readiness on Basescan:

- **Foundation Safe (2-of-3):** [`0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791`](https://basescan.org/address/0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791)
- **ProvenanceRegistry:** [`0xdF470BFa9eF310B196801D5105468515d0069915`](https://basescan.org/address/0xdF470BFa9eF310B196801D5105468515d0069915)
- **RoyaltyDistributor v1 (legacy, retained for legacy claimable only):** [`0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2`](https://basescan.org/address/0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2) — the canonical live RoyaltyDistributor is now v2 [`0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e`](https://basescan.org/address/0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e)
- **FTNSTokenSimple v1:** [`0x5276a3756C85f2E9e46f6D34386167a209aa16e5`](https://basescan.org/address/0x5276a3756C85f2E9e46f6D34386167a209aa16e5)

The audit-bundle stack itself (Phase 3.1 + 7 + 7.1 + 7-storage) is
**not yet deployed** — this engagement is the gating layer before
deploy. That's the standard pre-mainnet pattern.

---

## 5. Updated submission timeline

- **2026-05-12:** Send refreshed RFPs (this addendum + original doc) to
  top-3 firms
- **2026-05-19 to 2026-05-26:** Initial proposals back; intro calls
- **2026-06-02:** Vendor selection
- **2026-06-09:** Engagement kickoff (target — overlaps with C4 contest
  start)
- **2026-06-09 to 2026-06-30:** Audit window (3 weeks)
- **2026-07-07:** Final report delivered
- **2026-07-21:** Foundation remediation sprint complete; delta review
- **2026-08-01:** Public report publication

**Wall-clock from this addendum to L4 cleared: 12-13 weeks** (matches
the W2-W14 envelope of `AUDIT_PLAN.md` §6.6).

---

## 6. What firms should NOT bid on

To save firms time and keep proposals focused:

- DO NOT propose to audit Ed25519Lib + Sha512 (L3 separate engagement).
- DO NOT propose to audit Phase 1.3 already-deployed contracts in
  isolation — they're either (a) on the v2-redeploy path (HIGH-1) or
  (b) in operational handoff (CRIT-2). Audit them as part of the
  cross-composition surface.
- DO NOT propose to redo SAST that's already in CI; we'll provide the
  Slither + Aderyn output.
- DO NOT propose to audit the off-chain Python — that's L5 (separate
  ML-supply-chain firm).

---

## 7. Standby firms

If the top-3 are unavailable in the W2-W14 window, the original
shortlist's #4-#6 firms (ConsenSys Diligence, Sherlock, Cantina
Reviews) remain on standby. Reach out via the original RFP template
(`docs/2026-04-23-auditor-shortlist-and-rfp.md` §4).

---

## 8. Engagement contact

Same as L3 + original RFP:

- **Primary:** schultzryne@gmail.com
- **Security mailbox:** security@prsm.network
- **PGP:** see SECURITY.md

---

*This addendum is non-binding until a mutually-executed engagement
agreement is in place. Foundation reserves the right to adjust scope
based on responses received.*
