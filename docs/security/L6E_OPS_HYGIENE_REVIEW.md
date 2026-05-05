# L6e — Continuous Ops Hygiene Review

**Layer:** L6e of the 11-layer defense-in-depth audit plan
(`audits/AUDIT_PLAN.md` §5).
**Type:** Quarterly self-review template + drill schedule.
**Cadence:** Quarterly, executed within 14 days of quarter close.
**Owner:** Founder (until Foundation Director of Engineering hire), then
DoE.
**Companions:**
- [`KEY_ROTATION_RUNBOOK.md`](KEY_ROTATION_RUNBOOK.md) — what to do when a key
  is lost or suspected compromised.
- [`INSIDER_THREAT_AND_COLLUSION_POLICY.md`](INSIDER_THREAT_AND_COLLUSION_POLICY.md) —
  governance constraints on signers + collusion mitigations.
- [`EXPLOIT_RESPONSE_PLAYBOOK.md`](EXPLOIT_RESPONSE_PLAYBOOK.md) — incident
  response posture.
- [`docs/governance/2026-05-05-treasury-reserve-and-risk-transfer-policy.md`](../governance/2026-05-05-treasury-reserve-and-risk-transfer-policy.md) —
  PRSM-POL-1 reserve floor + disbursement authority.
- [`docs/PRODUCTION_OPERATIONS_MANUAL.md`](../PRODUCTION_OPERATIONS_MANUAL.md) —
  infrastructure / deploy operational surface.

---

## 1. Purpose

L6a-L6d shipped point-in-time policies (Foundation Safe deploy, hardware
diversity, key-rotation runbook, insider-threat policy). **L6e is the
continuous half:** policies do not stay correct on their own. Hardware
firmware ships updates. Signer life circumstances change. PGP keys get
revoked. Bootstrap-node TLS certificates expire. Reserve floors drift
with treasury inflows / outflows.

This document is the **template**. Each quarter the founder produces an
instance at `docs/security/L6E-reviews/YYYY-Qx.md` with the checklist
filled in, archived for auditor traceability and future-DoE handoff.

---

## 2. Schedule

| Quarter | Window | First instance |
|---------|--------|----------------|
| Q2 2026 (calendar) | 2026-07-01 → 2026-07-14 | first review |
| Q3 2026 | 2026-10-01 → 2026-10-14 | |
| Q4 2026 | 2027-01-01 → 2027-01-14 | |
| Q1 2027 | 2027-04-01 → 2027-04-14 | |

**Annual rotation drill** (from `KEY_ROTATION_RUNBOOK.md` §299) folds
into the Q4 review each year — full key-rotation rehearsal under
controlled conditions, no actual rotation unless drill surfaces a real
issue.

---

## 3. Pre-review preparation

### 3.1 Commands to run before opening the checklist

```bash
# Foundation Safe health snapshot
python3 scripts/foundation-safe-health-check.py \
  --network base --safe 0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791

# Audit-bundle deployment verifier
node contracts/scripts/verify-audit-bundle-deployment.js --network base

# Forta alert-routing smoke test (does not page anyone — synthetic only)
cd ops/monitoring/forta-bots && npm run smoke-test

# Treasury balance vs reserve floor (PRSM-POL-1 §3.1)
# Cross-reference scripts/foundation-safe-health-check.py output
# against the $2M floor + $10M target thresholds
```

### 3.2 Artifacts to gather

- Most recent `cumulative-audit-prep-*` tag SHA + commit log since the
  prior quarterly review.
- `audits/findings/` open-finding count by severity.
- `.github/workflows/solidity-static-analysis.yml` last-week run status
  (Slither + Aderyn fast tier; Mythril slow tier).
- Forta bot uptime + alert-volume statistics for the quarter.
- Bootstrap-node uptime + TLS-cert expiry for
  `bootstrap1.prsm-network.com:8765`.

---

## 4. The checklist

Each item carries a status: `✅ verified`, `⚠️ drift detected`, or
`🔴 issue — open subtask`. Drift / issue items must each spawn a
follow-up commit or task before the review is signed off.

### 4.1 Signer roster (Foundation Safe 2-of-3)

- [ ] Each of the 3 signing addresses still resolves to the intended
      hardware wallet — re-verify via the wallet UI's address-display
      flow (do not just trust a stored copy).
      Cross-reference: `~/.prsm-foundation-private/Multi-Sig_Addresses.txt`
      (chmod 600, NEVER committed to repo).
- [ ] No signer has had a life event that materially affects the
      threat model: jurisdiction change, employment change at a
      conflicting party, health event reducing availability,
      disclosed coercion risk.
- [ ] D7 deputy-founder protocol (task #134) — succession contact
      list current; deputy-founder seed-share envelope unsealed-test
      status verified (envelope intact OR drill performed within last
      12 months).
- [ ] Insider-threat policy (`INSIDER_THREAT_AND_COLLUSION_POLICY.md`)
      §3 collusion-resistance constraints still hold:
      - 3 distinct jurisdictions
      - No 2 signers under same employer or familial relationship
      - At least 1 Foundation Director (post-charter)

### 4.2 Hardware wallet status

- [ ] Ledger firmware up to date — within 1 minor version of latest
      stable. Document version + update date.
- [ ] Trezor firmware up to date — same standard.
- [ ] OneKey firmware up to date — same standard.
- [ ] All 3 devices physically accounted for (each signer
      attests in writing).
- [ ] PIN / passphrase recovery still works — at-most-once-per-year
      verification by signer (test with a sub-account, never the
      Safe-signing account).

### 4.3 Seed storage

- [ ] Each signer's seed-storage location unchanged OR change
      ratified by council (per `KEY_ROTATION_RUNBOOK.md`).
- [ ] No seed material ever transcribed to:
      - cloud-synced storage (iCloud, Drive, Dropbox, OneDrive)
      - mobile keyboards / clipboards
      - photographs (including printed-and-photographed)
      - password managers (FIDO/WebAuthn for accounts is fine; raw
        seed mnemonic is not)
- [ ] Backup-share material (if Shamir split or duplicate steel
      backup) physically separated by ≥ 1 jurisdiction or ≥ 100km.

### 4.4 Emergency contacts + escalation paths

- [ ] Founder has each signer's primary + secondary contact (phone +
      Signal).
- [ ] Each signer has founder's contact + the other 2 signers' Signal.
- [ ] Forta P0 / P1 alert routing: Discord + Slack + PagerDuty +
      email targets all answered by humans within agreed SLA. Last
      smoke-test date: ___________
- [ ] PagerDuty escalation policy points to a person, not a stale
      account.
- [ ] `security@prsm.network` PGP key not within 90 days of expiry;
      published in `SECURITY.md` matches local key fingerprint.

### 4.5 On-chain treasury layer

- [ ] Foundation Safe balance ≥ floor reserve ($2M USD-equivalent
      per PRSM-POL-1 §3.1). If below, trigger §8 floor-breach
      protocol and document.
- [ ] Treasury composition: % FTNS vs USDC vs ETH vs other. Trend
      since prior quarter.
- [ ] 2% network-fee accumulation matches expected rate from
      RoyaltyDistributor v2 settlement volume.
- [ ] No unexpected outflows since prior review (every Safe tx since
      the prior review accounted for in `audits/findings/L6e-reviews/`
      ledger).
- [ ] `EmissionController` next-halving date + halving math sanity
      check.
- [ ] `RoyaltyDistributor.claimable` total reconciles with FTNS
      balance held by the contract (D-04 pull-payment invariant —
      this is the same invariant L1 §5 lists as a Halmos target).

### 4.6 Smart contract operational state

- [ ] `Pausable` state of audit-bundle contracts: each unpaused
      unless explicitly pause-active for a documented reason.
- [ ] `Ownable2Step` pendingOwner status: should be `address(0)` on
      all 7 audit-bundle contracts. Anything else means a handoff is
      mid-flight and requires explanation.
- [ ] No `DEFAULT_ADMIN_ROLE` holders other than expected (Foundation
      Safe + emergency admin if applicable).
- [ ] FTNSTokenSimple `MINTER_ROLE` holders match expected list
      (EmissionController only, post-Phase-8).
- [ ] Per-batch challenge-window snapshots (D-03) — recent batches
      sample shows `challengeWindowSecondsAtCommit` matching the
      registry's value at commit time.

### 4.7 Static / symbolic tooling (L1 wiring health)

- [ ] Slither + Aderyn fast tier passing on `main` for the entire
      quarter.
- [ ] Mythril weekly slow tier produced reports; no `high` severity
      not already triaged.
- [ ] Halmos / Echidna no-op probes still no-op (i.e., nobody added
      `.t.sol` files outside the planned property-suite sprint).
- [ ] CI runner secrets / tokens not within 90 days of rotation
      cadence.

### 4.8 Findings register status

- [ ] `audits/findings/consolidated.md` — all closed L2 findings
      remain remediated (no regression PRs reverting fixes).
- [ ] L1 in-flight findings: count by severity; aging.
- [ ] L3-L8 vendor engagements status pulled from
      `audits/rfp/README.md` — proposal-receipt dates current.

### 4.9 Bootstrap + infrastructure

- [ ] `bootstrap1.prsm-network.com:8765` TLS cert ≥ 30 days from
      expiry. Renewal automation status verified.
- [ ] DigitalOcean droplet patches current. SSH-key inventory matches
      authorized list.
- [ ] No new bootstrap nodes added without RFP-driven L6f review of
      the operator.

---

## 5. Annual key-rotation drill (Q4 each year)

Layered on top of the quarterly checklist for the Q4 review. Per
`KEY_ROTATION_RUNBOOK.md` §299:

- [ ] Tabletop walkthrough with all 3 signers present (video call OK)
      of the lost-key scenario.
- [ ] Each signer narrates their seed-storage location + recovery
      procedure without retrieving the actual seed material (the goal
      is to verify they remember the procedure, not to produce the
      seed).
- [ ] Deputy-founder envelope status: physical check that the seed-
      share envelope tamper-evident seal is intact.
- [ ] One signer simulates a rotation-request transaction on a
      testnet Safe (Sepolia or Base Sepolia) to confirm the
      transaction-construction flow + their personal hardware
      device's signing flow still work.
- [ ] Drill outcome documented: items that took longer than expected
      OR procedural friction discovered get filed as runbook
      improvements.

---

## 6. Sign-off

Each instance review concludes with:

```
Reviewed by:    __________________________
Date:           __________________________
Drift items:    __________________________ (pointers to follow-up commits / tasks)
Floor reserve:  __________________________ (USD-equivalent at quarter close)
Open severities: CRIT __ HIGH __ MED __ LOW __
Next review:    __________________________ (quarterly cadence)
Council notify: __________________________ (date posted to council channel)
```

The signed instance is committed to
`docs/security/L6E-reviews/YYYY-Qx.md`. If any 🔴 issue surfaces, the
review is **not** signed off until the issue has either:
- been fixed in a referenced commit, OR
- been triaged into a tracked follow-up task with target completion
  date, OR
- been escalated to council under PRSM-POL-1 §8 or
  `EXPLOIT_RESPONSE_PLAYBOOK.md` if it is severity-material.

---

## 7. Audit traceability

For external auditors / investors performing due diligence:

- **Where to look:** `docs/security/L6E-reviews/` directory contains
  one signed file per quarter from 2026-Q2 onward.
- **What's verifiable:** each instance references commit SHAs, Forta
  smoke-test logs, treasury-balance snapshots — all third-party
  re-derivable.
- **What's not in this directory:** seed-storage location specifics
  (signer-private), individual signer life-event details (need-to-know
  council only), specific Forta alert payloads (operationally
  sensitive).

---

## 8. Amendment process

This template (not the per-quarter instances) is amended by:

1. Founder proposes edit on a feature branch with rationale in commit
   message.
2. At least 1 council signer reviews + acks before merge to `main`.
3. Material changes (adding / removing checklist sections, changing
   cadence, changing sign-off authority) require ratification under
   PRSM-POL-1 §10 amendment process.

---

*L6e is the layer that catches what the policy layers wrote down but
forgot to keep current. If this checklist ever feels routine and
boring, the layer is doing its job.*
