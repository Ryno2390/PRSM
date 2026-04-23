# PRSM Foundation Jurisdiction Scoping

**Date:** 2026-04-23
**Owner:** engineering lead (hand-off to incoming legal counsel)
**Status:** Scoping complete; recommendation ready for counsel review
**Depends on:** nothing (fully unblocked)
**Unblocks:** legal counsel shortlist, Foundation formation, audit engagement, PRSM-GOV-1 §9 governance votes

---

## 0. Purpose of this doc

Hand an incoming legal-counsel firm a thought-out jurisdiction analysis so
they spend billable hours confirming or challenging the analysis — not
explaining jurisdiction basics. Research is current as of 2026-04-23 from
public sources; every factual claim is cited. Flagged items are explicit
about what couldn't be verified without counsel's help.

**This is not legal advice.** Every formation path requires
jurisdictional-counsel sign-off before entity filing.

---

## 1. Constraints shaping the choice

From the repo state, these are the non-negotiable inputs:

| Constraint | Value | Implication |
|-----------|-------|-------------|
| Founder domicile | US (Ohio) | Any offshore jurisdiction needs a founder-recusal pattern to avoid US tax nexus / ECI exposure |
| Co-founders at formation | 0 (planned later) | Need jurisdiction that permits solo director/founder with easy co-founder addition |
| Token (FTNS) profile | Capped supply, halving emission, no pre-sale, no VC allocation, utility-token (access to P2P compute) | Fits most non-US utility-token carve-outs; worst-case exposure is US securities |
| Governance model | PRSM-GOV-1: token-weighted votes + Foundation-reserved veto for emergency + legal-compliance | Needs flexible statutory framework, not rigid statutory board |
| Treasury size (post Series A) | ~$5-20M USDC-equivalent + on-chain FTNS | Treasury must be held through a Safe owned by the entity; banking access required for fiat ops |
| Audit engagement | Pending (see `docs/2026-04-23-auditor-shortlist-and-rfp.md`) | Auditor signs with the Foundation as counterparty; Foundation must exist first |
| Hardware multi-sig | 2-of-3 Safe on Base, per-device storage plan | Entity must legally own the Safe; signer residency varies |

---

## 2. Candidate jurisdictions

Five jurisdictions, in approximate order of fit for PRSM's profile.

### 2.1 Cayman Foundation Company — **recommended primary**

**Tax treatment.** Zero direct tax: no income, no capital gains, no corporate. Token issuance is not a taxable event. Treasury held at cost with no mark-to-market obligation. The Cayman Monetary Authority (CIMA) Aug 2025 tokenised-funds consultation clarified tokenisation alone doesn't trigger VASP "issuance of virtual assets" regulated-activity classification.

**Token-issuance legal risk.** Under the VASP Act (2023, amended 2025), utility tokens used purely for access to services are outside VASP licensing when no exchange / custody / transfer service is provided. FTNS — P2P compute access, no pre-sale, no VC allocation — fits the carve-out cleanly.

**Multi-sig compatibility.** Foundation Companies routinely hold Gnosis Safes. No jurisdictional KYC on signers (only at the banking layer).

**Governance flexibility.** **Explicitly designed for DAOs.** Foundation Companies can operate without members or shareholders, governed by charter + supervisor. PRSM-GOV-1's token-weighted voting + reserved Foundation veto maps directly onto this model.

**Formation cost + timeline.** ~$15-40K all-in, 1-4 weeks. Bell Rock Group offers 24-hour formation; standard crypto-legal-firm setup (Walkers, Maples, Ogier, Campbells, Carey Olsen) is 2-4 weeks.

**Ongoing compliance.** ~$15-35K/yr: registered office, supervisor fee, annual government filings. **Foundation Companies are explicitly excluded from the Economic Substance Act** — no substance headache at PRSM's pre-revenue stage. ES notification remains annual. No director residency. No statutory audit below fund thresholds.

**Banking.** Sygnum (Swiss) is the standard tier-1 crypto-custody partner for Cayman entities; USD banking routes through Cayman National, Butterfield, or Swiss (Sygnum / AMINA). Native crypto treasury lives in the Safe; banks serve fiat operations only.

**Comparable protocols.** Arbitrum Foundation, Optimism Foundation, Polygon Labs UI (Cayman) Ltd, Balancer, SushiSwap, early dYdX. CCN reported >1,700 crypto foundations registered by end-2025.

**Solo-founder compatibility.** Single director permitted. Adding co-founders is a supervisor-signed resolution — fast and cheap.

**US tax nexus.** The Foundation Company is typically a non-shareholder entity, reducing CFC exposure. But a US founder acting as director with decision-making authority can create ECI / US-trade-or-business risk. **Standard mitigant:** appoint a professional Cayman director (Bell Rock, Walkers, Maples) as the operational director; founder takes an advisory-only role for treasury decisions. Budget: +$10-25K/yr for the professional director service.

### 2.2 Swiss Verein — **recommended alternate**

**Tax treatment.** Switzerland does not treat utility-token launches as taxable events at the Verein level. FINMA's 2018 ICO guidelines (refined through 2026) classify tokens into payment / utility / asset; utility tokens with access rights at issuance are not securities. Treasury held at cost; wealth tax applies but private-holding capital-gains are exempt.

**Token-issuance legal risk.** Lowest-risk non-US jurisdiction. Strongest legal precedent — the Ethereum Foundation + Solana Foundation + Cardano + Polkadot + Tezos + NEAR + Cosmos + dYdX + Safe + LUKSO + TON blueprint.

**Multi-sig compatibility.** Verein can own a Safe; no signer KYC at the jurisdictional level.

**Governance flexibility.** Verein (association) is **highly flexible** — can adopt token-vote-driven governance directly. Strong DAO-wrapper fit. Stronger than Stiftung (Foundation), which is statute-rigid and subject to federal Foundation Supervisory Authority.

**Formation cost + timeline.** 2 founders (minimum), no capital requirement, 1 day to legally form, ~$15-40K with crypto-native legal (MME, Bär & Karrer). **Biggest catch: needs 2 founders on day one** — PRSM is currently solo-founder. A trusted ally (co-counsel, advisor, family member) can serve as second founder temporarily.

**Ongoing compliance.** ~$5-20K/yr; no federal supervisor oversight (unlike Stiftung).

**Banking.** AMINA (formerly SEBA), Sygnum, PostFinance all serve crypto-entities as of 2025-2026 with no major closures reported. New FinIA Payment / Crypto Institutions categories in consultation Oct 2025, expected live late 2026 — further banking-access tailwind.

**Comparable protocols.** Swiss Stiftung is the industry standard (Ethereum, Solana, etc.); Verein is less common but well-understood — Switzerland has the deepest crypto-legal bench.

**Solo-founder compatibility.** 2 members minimum; adding members later is trivial.

**US tax nexus.** Verein is typically outside CFC scope (no shareholders). US founder's role needs structuring to avoid ECI — similar pattern to Cayman (advisory seat + professional Swiss-resident director on the Vorstand). Budget: +$15-30K/yr for Swiss-resident director.

### 2.3 Swiss Stiftung (Foundation)

**Rigorous but slower and costlier than Verein.** Federal supervisory authority oversight, minimum CHF 50K capital, 3-person board, annual audit required. **~$100-150K all-in, 2-4 months**; annual compliance $30-80K. Used by Ethereum, Solana, Polkadot, Tezos, Cardano. **Structural catch:** a Stiftung must not have founders with indirect financial interest in the tokens. Ethereum Foundation works because founders retained zero team tokens — PRSM's no-pre-sale profile aligns, but a future-founder-token grant would be incompatible with Stiftung rules.

**When to consider:** if PRSM later raises a Series A and banking preferences shift to the Swiss corridor, a Verein-to-Stiftung upgrade path exists but costs $100K+.

### 2.4 BVI Foundation / Crypto-Catamaran

**BVI Foundation standalone is newer (introduced 2023) and less battle-tested** than Cayman. BVI BC (Business Company): ~$3-8K, 1-2 weeks — this is the workhorse for **issuer SPVs**, not governance entities. The "crypto catamaran" model pairs a BVI BC (token issuer) with a Cayman Foundation (governance) under a VISTA Trust orphan structure: ~$40-70K all-in.

**VISTA Trust is a different tool** — an orphan SPV structure where a purpose trust holds shares in an SPV so the SPV has no beneficial owner. This is for the token-issuer's legal structure, not the governing Foundation itself. **Conflating the two is a common mistake.**

**Banking.** Harder than Cayman in 2025-2026. Most BVI crypto entities bank in Switzerland or Cayman-adjacent providers. **Flagged — we could not verify a 2026-current BVI-domiciled crypto-friendly local bank list from public sources.**

**Comparable protocols.** Tether (BVI-incorporated historically), Bitfinex, many issuer SPVs. No major protocol uses BVI *Foundation* standalone for governance.

**When to consider:** if PRSM later needs a separate token-issuance SPV layered under the governance Foundation. Add in a later phase, not at formation.

### 2.5 Wyoming DAO LLC / DUNA

**Highest US-nexus risk.** Full US SEC/CFTC jurisdiction. Post-*Ripple* (July 2023) + post-*Ooki* (CFTC default judgment against a DAO, with token-holders held personally liable as an unincorporated association) shape the risk landscape. 2025-2026 SEC posture under Chair Atkins has dismissed or closed 12+ crypto cases (Coinbase, Kraken, Ripple, Robinhood, Crypto.com) — a major climate shift, but enforcement risk remains non-zero. CLARITY Act passed House July 2025, cleared Senate Agriculture Committee Jan 2026, expected 2026 signing — would codify commodity-vs-security split.

**DUNA (Decentralized Unincorporated Nonprofit Association, effective July 2024) requires 100+ members to form.** This is a hard blocker for PRSM today — there is no token-holder community yet.

**DAO LLC defaults to pass-through (partnership) taxation** — members pay tax on distributive share. Token issuance by an LLC can be a taxable event absent capital-contribution characterization. DUNA can elect corporate tax or pursue 501(c) exemption — more flexible than DAO LLC for non-profit token emission, but the 100-member threshold gates it.

**Formation.** DAO LLC: ~$500 state + $15-50K legal, 2-6 weeks. DUNA: similar cost + 3-8 weeks, but member-threshold-blocked.

**Banking.** US-domestic (Mercury, Lead Bank, Customers, Axos). **US banks will KYC every signer** — the hard constraint. Most don't custody native crypto treasury; fiat-only. On-chain treasury lives in the Safe.

**Comparable protocols.** Uniswap adopted DUNI (Wyoming DUNA) in Sept 2025 — only 3 DUNAs existed as of early 2026. Still new.

**When to consider:** revisit post-TGE once PRSM has a token-holder community large enough to clear the 100-member DUNA threshold. Today: blocked.

### 2.6 Delaware Nonprofit Corporation

**Full US SEC/CFTC exposure.** Delaware is well-understood by US regulators, which cuts both ways: predictable but directly in jurisdiction. Corporate income tax unless 501(c)(3/4/6) exemption approved by IRS (6-18 month application adding $5-15K). Token issuance by a nonprofit is likely a taxable event absent specific 501(c) characterization; unrelated business income (UBI) treatment possible.

**Statutory board requirements** (director, officers, annual meeting, bylaws) conflict with pure token-weighted governance without carefully drafted Foundation-veto bylaws carve-outs. Uniswap uses Delaware Foundation + **separate offshore SPV for the token** — the norm is to never issue the token from a Delaware nonprofit.

**Formation.** ~$20-55K, 2 weeks to exist + 6-18 months to get 501(c) approval and comfortable banking.

**When to consider:** as a **secondary grants / operating arm** paired with an offshore token-issuer (Cayman Foundation). Not as the primary Foundation.

---

## 3. Synthesis matrix

| Dimension | Cayman Foundation | Swiss Verein | Swiss Stiftung | BVI Catamaran | Wyoming DUNA | Delaware Nonprofit |
|-----------|-------------------|--------------|----------------|---------------|--------------|---------------------|
| Formation cost | $15-40K | $15-40K | $100-150K | $40-70K | $15-50K | $20-55K + 501c |
| Timeline to entity | 1-4 wk | 1-2 wk | 2-4 mo | 2-4 wk | 3-8 wk + blocked | 2 wk + 12mo 501c |
| Annual compliance | $15-35K | $5-20K | $30-80K | $23-45K | $22-60K | $15-50K |
| Token-issuance risk | Lowest | Low | Lowest | Lowest | High (US) | High (US) |
| Solo-founder OK | ✅ 1 dir | ⚠️ 2 members | ❌ 3 board | ✅ 1 council | ❌ 100 members | ✅ 1 inc |
| US tax nexus shield | ✅ w/ pro dir | ✅ w/ CH dir | ✅ w/ CH dir | ✅ w/ pro dir | ❌ direct US | ❌ direct US |
| Gov flexibility | High (DAO-designed) | High | Rigid | High | High | Medium |
| 2026 banking | Sygnum via CH | AMINA / Sygnum | AMINA / Sygnum | Harder | US banks + KYC | US banks + KYC |
| DAO precedent | Arbitrum, Optimism, Polygon | (less common) | Ethereum, Solana | Tether, Bitfinex | Uniswap DUNI | Uniswap Fdn |
| Recommendation | **Primary** | **Alternate** | Future upgrade | Later phase SPV | Post-TGE revisit | Secondary arm |

---

## 4. Recommendation

**Primary: Cayman Foundation Company. Alternate: Swiss Verein.**

### Why Cayman wins for PRSM specifically

1. **Solo-founder compatible** — single director at formation, co-founders added via simple supervisor resolution. Wyoming DUNA is blocked by the 100-member threshold; Swiss Stiftung needs a 3-person board; Verein needs 2 members (manageable but an extra coordination step).
2. **Fastest time to entity + banking-ready**: 1-4 weeks vs. Swiss Stiftung's 2-4 months, Delaware nonprofit's 12+ months for 501(c) clearance.
3. **Explicit ES Act exclusion** — Foundation Companies are carved out of the Economic Substance regime, avoiding substance headache at PRSM's pre-revenue stage.
4. **Proven for exactly PRSM's profile** — Arbitrum, Optimism, Polygon all chose it for the same reason: L2 / protocol with Foundation-governed token treasury + Safe multi-sig.
5. **FTNS profile fits VASP Act carve-outs cleanly** — capped supply, halving emission, no pre-sale, no VC allocation, utility-token semantics.
6. **US-founder nexus manageable** via professional Cayman director model (Bell Rock / Walkers / Maples). Standard mitigant; well-understood by auditors and counsel.
7. **Cost-to-value** is the best of the 5: ~$25-35K setup, ~$20-35K/yr ongoing.

### When to choose Swiss Verein instead

- If PRSM anticipates a **Swiss-corridor banking relationship** (Sygnum / AMINA prefer Swiss-domiciled counterparties for certain services), starting as a Verein preserves that optionality.
- If the incoming legal counsel's benchwork is Swiss-heavy and they'd onboard faster with Swiss-native structures.
- If PRSM's positioning leans on the Ethereum / Solana jurisdictional lineage for credibility.

**Trade-offs against Cayman:** Verein needs 2 founders on day one (vs. 1 for Cayman); DAO-specific case law is thinner than Cayman; Stiftung upgrade path (if Series A triggers a jurisdiction-upgrade conversation) costs an additional $100K+.

### Why NOT the others

- **Wyoming DUNA** — blocked by the 100-member threshold until PRSM has a token-holder community. Revisit post-TGE.
- **Delaware nonprofit** — full US SEC/CFTC exposure; only makes sense as a secondary grants entity paired with an offshore token-issuer. Not a primary Foundation choice for PRSM's stage.
- **BVI catamaran** — duplicative cost vs. Cayman standalone at PRSM's scale. Primary use (orphan SPV via VISTA Trust) is token-issuer infrastructure, not governance. Add later if PRSM needs a token-issuer separate from governance.
- **Swiss Stiftung** — right answer at $100M+ treasury scale but overkill today. The Verein-to-Stiftung upgrade path is well-trodden if scale demands it.

---

## 5. Decision memo template for incoming counsel

Copy/paste-ready for the first counsel kickoff call. Fill the 3 placeholders.

```markdown
# PRSM Foundation Formation — Decision Memo

**Draft by:** Ryne Schultz (engineering lead, PRSM)
**For:** {{COUNSEL_NAME}} / {{COUNSEL_FIRM}}
**Date:** {{DATE}}

## Tentative decision
Cayman Foundation Company, with Swiss Verein as alternate if you
identify reasons Cayman is inappropriate for PRSM's profile.

## Why Cayman (summary)
1. Solo-founder compatible (1 director; co-founders added later via
   supervisor resolution).
2. Entity existing + banking-ready in 1-4 weeks.
3. Foundation Companies excluded from the ES Act — no substance
   operations needed at pre-revenue stage.
4. Proven for Arbitrum / Optimism / Polygon — same protocol profile.
5. FTNS utility-token (capped supply, halving, no pre-sale, no VC)
   fits VASP Act carve-outs.
6. US-founder nexus managed via professional Cayman director model.

## Confirmed constraints (non-negotiable)
- Founder is US-domiciled (Ohio). Need ECI / CFC / PFIC protection.
- Solo founder today; 1-3 co-founders post-formation.
- Treasury: 2-of-3 Safe on Base, FTNS + USDC, ~$5-20M post-Series-A.
- Governance: PRSM-GOV-1 token-weighted + Foundation veto.
- Auditor engagement pending; Foundation must be counterparty.

## Questions for counsel (priority order)

1. Does Cayman Foundation Company fit PRSM's profile better than your
   preferred structure? If not, which structure and why?
2. Recommended professional-director firm — Bell Rock, Walkers,
   Maples, or another?
3. Does the VASP Act carve-out apply to FTNS as described?
4. Can the auditor (Trail of Bits / Cantina / OpenZeppelin) sign with
   a Cayman Foundation Company before it opens its bank account, or
   do we need bank-ready first?
5. What's the ECI-risk mitigation structure your firm typically uses
   for US-founder + Cayman Foundation combinations?
6. Is the founder-token-allocation consideration (relevant for Swiss
   Stiftung) a concern for Cayman Foundation Company formation?
7. All-in formation quote (foundation + first-year compliance +
   professional director) — ballpark, to budget.
8. Earliest realistic start date for formation engagement.

## Attached (from PRSM repo)
- docs/2026-04-23-foundation-jurisdiction-scoping.md (this analysis)
- docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md (governance charter)
- docs/2026-04-22-prsm-supply-1-supply-diversity-standard.md (emission schedule)
- docs/2026-04-21-prsm-tok-1-ftns-tokenomics.md (token design)
```

---

## 6. What this doc does NOT do

- **Does not commit PRSM to any jurisdiction.** Counsel can override.
- **Does not substitute for counsel advice.** Every claim above needs
  jurisdictional counsel confirmation before action.
- **Does not cover the professional-director / trustee engagement** —
  scope that separately once the primary jurisdiction is locked.
- **Does not address operating-subsidiary or grants-entity design** —
  e.g., a Delaware nonprofit for US grants + public-facing dev relations,
  paired with a Cayman token-issuer. That's a post-formation question.

---

## 7. Flagged items (could not verify from public 2025-2026 sources)

Items the scoping research couldn't lock down from public sources —
incoming counsel should refresh these before any formation decision:

- **Specific BVI local crypto-friendly bank names in 2026** — the market
  default is Cayman/Swiss routing, but a direct BVI option may exist.
- **Whether Mercury / Customers Bank / Lead Bank custody native crypto
  treasury for Cayman Foundations in 2026** — they serve fiat operations;
  native crypto treasury assumed to be self-custody via Safe.
- **BVI removal from EU Annex I "blacklist"** — should be reconfirmed
  against the current EU Council list before relying on it.
- **Exact 2026 all-in legal fee quotes** — ranges above come from
  2024-2025 public rate cards; live RFPs required for budgeting.

---

## 8. Dependency chain (what this unblocks)

```
This doc (2026-04-23)
  ↓
Legal counsel shortlist (next non-code doc)
  ↓
Counsel engagement + jurisdiction confirmation
  ↓
Foundation entity formation
  ↓
┌─────────────────────────────┬─────────────────────────────┐
│                             │                             │
Safe ownership transfer    Auditor engagement        PRSM-GOV-1
to Foundation               (Foundation as          §9 governance votes
                            counterparty)
  ↓                              ↓                             ↓
Hardware ceremony          Audit clock starts        Full DAO operation
```

---

## 9. Related documentation

- `docs/2026-04-23-auditor-shortlist-and-rfp.md` — audit firm shortlist (Foundation is counterparty).
- `docs/2026-04-23-testnet-rehearsal-plan.md` — mainnet deploy rehearsal (Safe is Foundation-owned).
- `docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md` — the governance charter this jurisdiction must accommodate.
- `docs/2026-04-21-prsm-tok-1-ftns-tokenomics.md` — token design (relevant to token-issuance legal risk).
- `docs/2026-04-22-prsm-supply-1-supply-diversity-standard.md` — emission / supply profile.
- `docs/2026-04-14-hybrid-tokenomics-legal-tracking.md` — legal-track tracker.

---

## 10. Research provenance

Research conducted 2026-04-23 from public sources. Every factual claim
in §2 cites a URL. No firm or jurisdiction was contacted during
research — this doc prepares for outreach, not substitutes for it.

Sources consulted (non-exhaustive):
- Cayman: Ogier Q4 2025 fund update, Walkers ES overview (July 2025),
  Bell Rock, Mourant, Carey Olsen, Collas Crill, CCN Web3 hub reporting.
- Swiss: FINMA guidelines, Global Legal Insights 2026, MME 2025, Legal
  Nodes, Chambers Blockchain 2025, Chambers Fintech 2025, My-Swiss-Company,
  Spheriq, Goldblum, Koinly Swiss Guide 2026.
- BVI: Carey Olsen 2026 guide, Law.asia token-offering piece, Conyers /
  Mourant VISTA Trust guides, daospv crypto-catamaran blog, LegalBison.
- Wyoming: Toku DUNA guide, Falcon Rappaport, a16z DUNA analysis,
  Proskauer Ooki coverage, Blockworks DUNI reporting, Uniswap
  governance forum, TechBuzz, USLLCGlobal.
- Delaware: a16z DAO framework pt 1, Uniswap Foundation public filings.
- Regulatory: The Block 2026 crypto-regulation outlook,
  democrats-financialservices Jan 2026 SEC letter, Chair Atkins SEC
  agenda reporting, CLARITY Act congressional status tracking.

Pricing and timeline figures reflect 2024-2025 public rate cards;
firm-specific 2026 quotes require RFP.
