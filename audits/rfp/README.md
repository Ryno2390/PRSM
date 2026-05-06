# PRSM Audit Pipeline RFP Index

**Status:** All vendor-track RFPs drafted as of 2026-05-05. Awaiting
founder execution of vendor outreach.

**Outreach packet:** [`outreach-emails.md`](outreach-emails.md) — one
email per recipient (~22 firms across 8 RFPs), ready to copy/paste.

**Master plan:** `audits/AUDIT_PLAN.md` v1.1
**Disclosure policy:** `SECURITY.md`

---

## RFP packets in this directory

| Layer | Packet | Recipients | Budget | Wall-clock | Status |
|-------|--------|------------|--------|-----------|--------|
| **L3** | [L3-ed25519-crypto-rfp.md](L3-ed25519-crypto-rfp.md) | Trail of Bits Crypto + NCC Group Crypto | $15K-$30K | 2-4 weeks | RFP drafted; pre-engagement artifacts complete |
| **L4** | [L4-code4rena-contest-scope.md](L4-code4rena-contest-scope.md) | Code4rena (public contest) | ~$40K | 14-day contest | Scoping packet drafted |
| **L4** | [L4-firm-rfp-addendum-20260505.md](L4-firm-rfp-addendum-20260505.md) | Trail of Bits + Spearbit-Cantina + OpenZeppelin (firm pair-review) | $60K-$80K | 3 weeks | Refresh addendum on top of `docs/2026-04-23-auditor-shortlist-and-rfp.md` |
| **L5** | [L5-ml-supply-chain-audit-rfp.md](L5-ml-supply-chain-audit-rfp.md) | Trail of Bits ML + NCC Group ML + Berkeley RDI + CMU CyLab | $40K-$80K | 4-6 weeks | RFP drafted |
| **L6f** | [L6f-infrastructure-pentest-rfp.md](L6f-infrastructure-pentest-rfp.md) | NCC Group + Bishop Fox + Doyensec | $20K-$50K | 2-4 weeks | RFP drafted |
| **L7** | [L7-economic-game-theory-audit-rfp.md](L7-economic-game-theory-audit-rfp.md) | Gauntlet + Chaos Labs + BlockScience + MIT DCI | $50K-$100K | 4-8 weeks | RFP drafted |
| **L8** | [L8-legal-counsel-rfp.md](L8-legal-counsel-rfp.md) | Track A: Mourant + Walkers + Ogier (Cayman). Track B: Cooley + Lowenstein + DLx Law (US securities) | $30K-$80K (A) + $50K-$150K (B) | 8 weeks initial + ongoing retainer | RFP drafted (two tracks) |
| **L9** | [L9-bug-bounty-program-design.md](L9-bug-bounty-program-design.md) | Immunefi (Channel A) + self-hosted security@prsm.network (Channel B) | $50K + $25K initial pool | Continuous post-launch | Design spec drafted; activation gated POST-L4 |

**Total combined budget envelope: ~$280K - $645K** across all engagements.

---

## Sequencing

The RFP packets are designed for **parallel execution** where possible:

```
W0 (now)             W2-W4              W6-W12             W14+
─────                ─────              ─────              ────
L3 outreach   →      L3 audit (2-4w)    L3 fixes
L4 outreach   →      L4 contest +       L4 fixes
                     firm review
L5 outreach   →      L5 audit (4-6w)    L5 fixes
L6f outreach  →      L6f pen-test (2-4w) L6f fixes
L7 outreach   →      L7 sim (4-8w)      L7 param tuning
L8 outreach   →      L8 opinions (8w+)  L8 ongoing retainer
                                                           L9 activate
                                                           L11 council ratify
```

Outreach order priority (for founder bandwidth):

1. **L4 firm RFP** (gates Phase 7 / 7.1 mainnet deploy — task #31, #40)
2. **L4 Code4rena intake** (parallel)
3. **L3 crypto specialist** (gates audit-bundle full-clear)
4. **L8 securities counsel** (gates Reg D 506(c) Prismatica raise)
5. **L8 Cayman counsel** (parallel with US counsel)
6. **L5 ML supply chain** (gates Tier C public availability)
7. **L7 economic** (gates DEX listing)
8. **L6f infrastructure** (gates production-grade bootstrap node)
9. **L9 bug bounty** (post-L4 only)
10. **L11 treasury policy ratification** (council deliberation, no
    vendor)

---

## What's been done internally (no RFP needed)

| Layer | Status | Artifact |
|-------|--------|----------|
| L0 | ✅ Continuous internal review | (no artifact) |
| L1 | ✅ Wired in CI | `.github/workflows/solidity-static-analysis.yml` + `audits/L1-static-tooling/README.md` |
| L2 | ✅ AI multi-team review closed | `audits/findings/consolidated.md` (1 CRIT + 7 HIGH + 7 MEDIUM all remediated) |
| L6a | ✅ Foundation Safe deployed | Phase 1.3 Task 8 (2026-05-04) |
| L6b | ✅ Hardware wallet diversity | Ledger + Trezor + OneKey verified |
| L6c | ✅ Key rotation runbook | `docs/security/KEY_ROTATION_RUNBOOK.md` |
| L6d | ✅ Insider threat policy | `docs/security/INSIDER_THREAT_AND_COLLUSION_POLICY.md` |
| L6e | ✅ Continuous ops hygiene template + archive dir | `docs/security/L6E_OPS_HYGIENE_REVIEW.md` + `docs/security/L6E-reviews/` (first review window 2026-07-01 → 2026-07-14) |
| L10a | ✅ Forta routing wired | Discord + Slack + PagerDuty + email; smoke-test CLI |
| L10d | ✅ SECURITY.md + PGP | Published 2026-05-05 |
| L11 (policy) | ✅ Treasury policy ratified 2026-05-06 | `docs/governance/2026-05-05-treasury-reserve-and-risk-transfer-policy.md` (PRSM-POL-1 v1.0) + Council Resolution `docs/governance/PRSM-CR-2026-05-06-1.md` |

---

## Cross-references

- **Master plan:** `audits/AUDIT_PLAN.md` v1.1
- **Decisions:** `audits/decisions/` (L3 ratified)
- **Findings:** `audits/findings/` (L2 consolidated, L3 pre-engagement,
  L1 ongoing CI artifacts)
- **Tooling:** `audits/L1-static-tooling/README.md`
- **Disclosure:** `SECURITY.md` + `docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md`
- **Governance:** `docs/governance/2026-05-05-treasury-reserve-and-risk-transfer-policy.md`

---

## Status update protocol

When a vendor proposal arrives:

1. Update the relevant `Lx-*-rfp.md` with proposal-receipt date + vendor.
2. Add to `audits/AUDIT_PLAN.md` Lx status block.
3. After engagement letter signed: cross-link engagement letter from
   the RFP doc.
4. After report received: file under
   `audits/findings/Lx-<vendor>-<date>/` and update master plan.

When a vendor declines or is unavailable:

1. Note decline reason in the RFP doc (one line).
2. Move to next firm in the recipient list.
3. If all listed firms decline, escalate to council for next-step
   decision.

---

*This index is maintained by the Foundation founder until DoE hire.*
