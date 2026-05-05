# PRSM Insider-Threat & Signer-Collusion Policy

**Version:** 1.0 (initial draft)
**Date:** 2026-05-05
**Status:** Pre-mainnet — published before Foundation council ratification
**Audience:** Foundation council, signing-authority holders, founder/deputy-founder, external auditors, sophisticated investors performing due diligence
**Owner:** Foundation council (post-formation); founder until provisioned
**Related:** [`KEY_ROTATION_RUNBOOK.md`](KEY_ROTATION_RUNBOOK.md) · [`EXPLOIT_RESPONSE_PLAYBOOK.md`](EXPLOIT_RESPONSE_PLAYBOOK.md) · [`audits/AUDIT_PLAN.md`](../../audits/AUDIT_PLAN.md) §5 L6d

## 1. Purpose

A 2-of-3 multi-sig is secure against any single signer compromise but **NOT
against collusion of two signers.** Two collaborating signers form a quorum
under the threshold and can move treasury funds, mutate Ownable contract
state, or revoke MINTER/PAUSER roles without intervention from the third
signer.

This document defines the policy that constrains *who* can be a signer, *how*
relationships between signers are constrained to make collusion costly, and
*what* governance + technical mechanisms detect or contain collusion if it
occurs anyway.

## 2. The collusion threat is inherent, not solvable by cryptography

No threshold scheme prevents threshold collusion. A 3-of-5 raises the
collusion bar (need 3 instead of 2) but introduces operational complexity
(more signers means more loss/turnover events). Increasing the threshold
trades single-key compromise risk for collusion risk; the right threshold
is a function of the population of trustworthy signers available.

The mitigation is not cryptographic. It is **structural** (who can be a
signer) and **procedural** (how easy or hard it is for any 2 signers to
coordinate without detection).

## 3. Threat model

### 3.1 Adversary profiles

We consider four collusion profiles:

| # | Profile | Likelihood | Impact |
|---|---------|------------|--------|
| 1 | **Two signers conspire to drain treasury** | Low | Catastrophic |
| 2 | **One signer is coerced (kidnapping, blackmail) by an adversary already controlling a second key** | Very low | Catastrophic |
| 3 | **One signer's machine is compromised; attacker leverages it + a phishing attack on a second signer** | Low-medium | Catastrophic |
| 4 | **Council member is coerced into approving a malicious-but-plausible action that 2 signers then sign in good faith** | Medium | Material loss; depends on action |

Profiles 1 and 2 are the "true insider threat." Profile 3 collapses to a
hybrid of insider + external attacker. Profile 4 is the most likely real-world
manifestation: well-intentioned signers approve a bad transaction because
the social or governance layer was attacked, not the cryptographic layer.

### 3.2 What an attacker gains

If two signers collude, they can execute any Foundation Safe transaction:

- Transfer arbitrary FTNS or ETH from the Safe to attacker-controlled addresses.
- Add a fourth Safe owner (an attacker-controlled address) and increase the
  threshold to 3-of-4 — pinning the third honest signer out.
- Call `transferOwnership` on Ownable contracts (RoyaltyDistributor, etc.)
  to point at attacker-controlled contracts.
- Grant MINTER_ROLE or DEFAULT_ADMIN_ROLE on FTNSToken to attacker-controlled
  addresses.

Phase 1.3 contracts (ProvenanceRegistry, RoyaltyDistributor) are
non-Ownable post-handoff for *most* methods. The 2% network fee path is
hardcoded immutable. **But:** the Safe still holds material balance, can
mint FTNS via FTNSToken's MINTER_ROLE chain, and controls the still-Ownable
audit-bundle contracts (EscrowPool, BatchSettlementRegistry, StakeBond)
once they deploy.

## 4. Diversity requirements (the structural mitigation)

The Foundation Safe shall maintain owner-set diversity across **four**
independent axes. No two signers should match on more than two axes.

### 4.1 Geographic diversity

**Constraint:** No two signers reside in the same country at the time of
their appointment. If a signer subsequently relocates to another signer's
country, the Foundation council shall be notified within 30 days; if no
remediation is taken within 12 months (typical: rotate the more-recently-
relocated signer), the council shall force-rotate.

**Rationale:** Coercion (Profile 2) is a function of physical reach. State
actors and organized adversaries can coerce signers within their
jurisdictional reach but face significant friction to coerce across
jurisdictions simultaneously.

### 4.2 Jurisdictional diversity

**Constraint:** No two signers operate under the same primary legal
jurisdiction. A signer's "primary jurisdiction" is the country whose laws
substantively govern their employment, residency, and tax obligations.

This is a stricter standard than geographic diversity (you can be
geographically in country X but jurisdictionally tied elsewhere).

**Rationale:** Legal coercion (subpoena, court order, regulatory action)
operates jurisdictionally, not geographically. A signer compelled by court
order in jurisdiction X cannot be compelled by a court in jurisdiction Y
without slow, cooperative international procedures.

### 4.3 Organizational diversity

**Constraint:** No two signers are employed by the same organization, hold
equity in the same operating company at material levels (>5%), or report to
the same governance body outside the Foundation council itself.

**Rationale:** A single organization can apply pressure (employment, equity,
business relationship) across multiple of its members simultaneously. The
Foundation Safe should not be vulnerable to any single organization's
internal politics or external regulatory action.

### 4.4 Role / function diversity

**Constraint:** Of the 3 signers, at least one shall be:
- A Foundation employee or board member (operational continuity).
- An external party not employed by the Foundation (independent oversight).
- A technical security expert capable of independently verifying
  Safe-transaction payloads (defense against Profile 4).

**Rationale:** Function-based diversity defends against the Profile 4
governance attack — a malicious-but-plausible transaction that an
operationally-busy signer might rubber-stamp. The technical-security signer
is positioned to catch these.

## 5. Foundation Safe (Phase 1.3 mainnet) — current state

The Foundation Safe at `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791` was
deployed 2026-05-04 with three signers (addresses recorded out-of-repo per
operational policy). At deployment time:

- Threshold: 2 of 3.
- Hardware wallet diversity: Ledger Nano S Plus, Trezor Safe 3, OneKey
  Classic 1S Pure (three different vendors).
- **Pending:** Geographic + jurisdictional + organizational + role
  diversity is currently held by founder oversight pre-council formation.
  Once council is formed, the council shall ratify each signer against the
  4-axis diversity constraint above.

This is acceptable for the current treasury-balance level (modest fee
accumulation pre-Phase-8 emission). It is **NOT** acceptable for
post-Gate-B treasury balance (post-audit-bundle deploy + post-bug-bounty +
post-DEX-listing). The Foundation council shall finalize signer-set
ratification BEFORE Gate B clears.

## 6. Threshold sizing — when to expand beyond 3-of-3

The 2-of-3 structure is appropriate when treasury at risk ≤ $5M USD-equiv.

When treasury at risk crosses thresholds, signer-set expansion is mandatory:

| Treasury at risk | Signer set | Threshold |
|------------------|------------|-----------|
| ≤ $5M | 3 signers | 2-of-3 |
| $5M – $50M | 5 signers | 3-of-5 |
| > $50M | 7 signers | 4-of-7 |

These thresholds are policy proposals pending council ratification. The
expansion event is a Safe transaction that adds owners + raises threshold —
a 2-of-3 (or current threshold) operation.

## 7. Detection mechanisms

### 7.1 Continuous on-chain monitoring (L10a)

Forta detection bots subscribed to the Foundation Safe address surface
EVERY successful Safe transaction as a real-time alert. Any 2-of-3 tx not
expected by all 3 signers triggers immediate investigation.

**Status:** Forta bots populated with Phase 1.3 contract addresses; alert
routing pending (#334).

### 7.2 Out-of-band attestation (proposed)

Every Safe transaction shall be **publicly announced** by the proposing
signer in a designated Foundation channel BEFORE signing. The transaction
hash + intent description shall be posted ≥ 24 hours pre-execution (except
for emergency pauses, which use the EXPLOIT_RESPONSE_PLAYBOOK fast-path).

A surviving honest signer who sees an unannounced transaction landing
on-chain has unambiguous evidence of collusion and triggers the §8
mitigation flow.

**Status:** Policy proposal; pending council ratification.

### 7.3 Quarterly attestation

Each quarter, every signer shall sign an attestation message confirming:

- Their device + seed are still under their sole control.
- They have not been approached by an adversary attempting to influence
  their signing decisions.
- They have followed the announced-transaction policy for all signing
  actions in the quarter.

Failure to attest, OR an attestation that flags any concern, triggers
investigation per the §8 mitigation flow.

**Status:** Policy proposal; pending council ratification.

## 8. Mitigation — what happens if collusion is detected

Collusion is a higher-severity event than Scenario C of the
KEY_ROTATION_RUNBOOK because the attack is already authorized, not just
threatened. The detection happens *after* the malicious transaction has
landed on-chain.

### 8.1 Honest signer's first-hour actions

The third (uncolluding) signer becomes the incident commander.

1. ☐ Document the unauthorized transaction (block number, tx hash, recipient,
   amount, time of detection).
2. ☐ Notify Foundation council. If council not yet provisioned, notify
   deputy-founder via D7 protocol.
3. ☐ DO NOT attempt unilateral action against the colluding signers. They
   hold quorum; any rotation tx requires their cooperation.
4. ☐ Activate EXPLOIT_RESPONSE_PLAYBOOK §11 — pause all pausable contracts
   the colluding signers can still influence (FTNSToken pause, audit-bundle
   pause when deployed). The third honest signer + the playbook's pre-staged
   pause transactions can execute pause unilaterally on contracts where
   PAUSER_ROLE is granted broadly.

### 8.2 Recovery path

The treasury once moved cannot be unmoved on-chain. Recovery is:

- **On-chain:** redeploy fresh contracts owned by a fresh, council-ratified
  Safe with 4-axis-diverse signers. Migrate any recoverable balance.
- **Off-chain:** legal action against the colluding parties (defamation,
  fraud, breach of fiduciary duty depending on jurisdiction).
- **Reputational:** transparent post-mortem published within 7 days.
  Concealment is itself a terminal trust event.

### 8.3 Why time-locks help (proposed enhancement)

A timelock module on the Foundation Safe (e.g., OpenZeppelin
TimelockController wrapping the Safe) introduces a delay between transaction
proposal and execution. For non-emergency actions:

- Signer proposes a tx → it enters a queue.
- Anyone can inspect the queue.
- After delay (proposed: 48 hours), the tx becomes executable.
- During the delay, an honest minority signer (or external observer) can
  trigger pause, alert the community, take legal preventive action.

**Trade-off:** Timelock delays legitimate actions too. For genuine
emergencies (active exploit), the playbook's fast-path bypasses the
timelock. The 48-hour delay applies only to "treasury moves" not
"emergency pauses."

**Status:** Proposed; not yet implemented. Implementation requires modifying
the Safe ownership architecture, which is itself a major operation.

## 9. Drill cadence

- **Quarterly:** Attestation cycle (§7.3) + Foundation council review of
  any anomalies surfaced by Forta or out-of-band attestation.
- **Annually:** Tabletop exercise — simulated collusion scenario walked
  through by the full council. Document timing of each response step.

Document each exercise outcome in `docs/security/drills/`.

## 10. Hardening roadmap

These are proposed improvements not yet implemented:

| # | Proposal | Effort | When |
|---|----------|--------|------|
| H1 | Timelock module wrapping Safe (48h queue for non-emergency actions) | High — architectural | Pre-Gate B |
| H2 | Out-of-band attestation policy + tooling | Low — process + form | Pre-Gate B |
| H3 | Quarterly attestation cycle | Low — calendar reminder + form | Post-council formation |
| H4 | 3-of-5 expansion (when treasury crosses $5M) | High — recruit 2 additional signers | Triggered by treasury growth |
| H5 | Geographic + jurisdictional + organizational diversity ratification | Medium — council decision + verification | Pre-Gate B |
| H6 | Role-diversity formalization (engineer / external / business) | Medium — council decision | Pre-Gate B |
| H7 | Annual tabletop drill | Low — annual | First drill: 2027-Q2 |

## 11. Operationalization status

Engineering / on-chain side:
- ✅ Foundation Safe live with 3 hardware-wallet-vendor diversity.
- ✅ Forta detection bots subscribed to Safe address (alert routing #334
  pending).
- ✅ EXPLOIT_RESPONSE_PLAYBOOK with pre-staged pause transactions.
- ✅ Out-of-repo signer-address registry with chmod 600.

Governance / process side (PENDING council ratification):
- ☐ Council ratification of this policy as the binding insider-threat
  framework.
- ☐ 4-axis diversity verification against current 3 signers.
- ☐ §7.2 out-of-band attestation tooling + policy.
- ☐ §7.3 quarterly attestation cycle initiation.
- ☐ §6 threshold-sizing schedule confirmed as treasury-growth trigger.
- ☐ §10 H1 timelock module evaluation + scoping (pre-Gate-B).
- ☐ Annual drill scheduled.
- ☐ D7 deputy-founder briefing on collusion detection + response.

## 12. Public commitment

Per the PRSM Vision §14 commitment to publish defenses BEFORE incidents,
this policy is published in a public repo. Adversaries gain modest
information from this (they learn we have monitoring + attestation) but
the deterrent value of public commitment + community oversight outweighs
the obscurity loss.

The Foundation council shall not weaken this policy without simultaneously
publishing the change and its rationale.
