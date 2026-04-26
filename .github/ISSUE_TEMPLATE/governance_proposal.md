---
name: 🏛️ Governance Proposal
about: Propose a parameter change or policy change for foundation council review
title: '[Gov] '
labels: ['governance', 'needs triage']
assignees: ''
---

## 🏛️ Governance Proposal

This template formalizes a governance proposal per [PRSM-GOV-1 Foundation Governance Charter](../../docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md). For informal discussion, use [GitHub Discussions](https://github.com/prsm-network/PRSM/discussions) or Discord `#governance` channel first.

### 📋 Proposal Summary

<!-- One paragraph: what are you proposing and why? -->

### 🎯 Type

- [ ] **Operational parameter** (POL deployment threshold, monitoring rule, etc.)
  → 3-of-5 council vote with 30-day timelock
- [ ] **Protocol parameter** (fee rate, halving config, supply cap)
  → 4-of-5 governance vote; DAO vote post-activation milestone
- [ ] **Policy** (PRSM-POLICY-* governance subsidiary policy)
  → 4-of-5 governance vote
- [ ] **Smart contract upgrade** (UUPS upgrade target)
  → 4-of-5 governance + audit sign-off
- [ ] **Foundation board composition or process change**
  → Full board vote per charter
- [ ] **Other:** _______

### 📜 Authority and rationale

**Which decision-authority does this require?**
<!-- Check the matching threshold from PRSM-GOV-1 Foundation Governance Charter -->

- [ ] Council operational (3-of-5)
- [ ] Council governance (4-of-5)
- [ ] Council unanimous
- [ ] DAO vote (post-activation)
- [ ] Founder unilateral (pre-board only, with public-rationale commit)

**Existing references:**
- [ ] References ratified parameter from `docs/2026-04-21-prsm-tok-1-ftns-tokenomics.md`
- [ ] References Risk Register entry: _______
- [ ] References Vision §_______
- [ ] References specific contract: _______

### 🔍 Specific change proposed

**Current state:**
<!-- What is the current parameter / policy? Include exact values, addresses, references. -->

**Proposed state:**
<!-- What should it become? Include exact values. -->

**Implementation:**
<!-- How is this enacted? Smart contract call? Documentation update? Foundation operational policy? -->

### 🧮 Impact analysis

**Who is affected:**
<!-- Creators / node operators / prompters / Foundation / Prismatica / external partners -->

**Quantitative impact** (where applicable):
<!-- Pool depletion implications, fee revenue change, etc. -->

**Risk register impact:**
<!-- Does this raise or lower the severity of any Risk Register entry? -->

**Reversibility:**
<!-- Can this be reversed via governance later? Or is it irreversible (e.g., supply cap reduction)? -->

### 📊 Alternatives considered

<!-- What other options exist? Why is this proposal preferred? -->

### ⚖️ Tokenomics §10 invariant check

PRSM has 8 inviolable invariants in Tokenomics §10. Confirm this proposal does NOT violate any:

- [ ] Total supply cap is hard (1B FTNS)
- [ ] Creator royalties enforced on-chain (no pause / off-chain bypass)
- [ ] Burn is irreversible
- [ ] POL is elastic (no hard price commitments)
- [ ] Staking rewards from real network fees only
- [ ] Foundation cannot mint above cap or re-mint burned
- [ ] Halving schedule is monotone-decreasing
- [ ] FTNS distributed only as compensation (never sold by foundation)

If ANY of these would be violated, the proposal cannot proceed via governance — it requires a fork (per Vision §14 Path 3) and is out of charter scope.

### 🧑‍⚖️ Counsel review

- [ ] Proposal does not require legal counsel review
- [ ] Proposal requires legal counsel review (will be added to Gate 3 counsel briefing scope)
- [ ] Already reviewed by counsel — date + firm: _______

### 📅 Proposed timeline

- **Discussion period:** <!-- 7-30 days typical -->
- **Decision target:** <!-- when council votes -->
- **Implementation target:** <!-- when changes go live -->
- **Timelock if applicable:** 30 days for council operational, longer for governance

### 🤝 Discussion before formal proposal

- [ ] I have already discussed this in Discord `#governance` for at least 7 days
- [ ] I have already opened a GitHub Discussion (link: _______)
- [ ] This is the first formal write-up; I'm requesting initial feedback

---

**Note:** Pre-foundation-board (currently), governance proposals are reviewed by founder with public rationale + 14-day delay before enactment per the Q6 ratification "founder unilateral with public commit" pattern. Post-board (Q3 2026 target), full council process applies.
