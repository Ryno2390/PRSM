# PRSM-GOV-1: PRSM Foundation Governance Charter

**Document identifier:** PRSM-GOV-1
**Version:** 0.1 Draft
**Status:** Pre-legal-formation draft. Principles + options for attorney consultation before entity formation. Specific jurisdictional choices flagged throughout as *decisions required*.
**Date:** 2026-04-21
**Drafting authority:** PRSM founder, pending Foundation convocation
**Companions:**
- `docs/2026-04-21-prsm-cis-1-confidential-inference-silicon.md` — the Confidential Inference Silicon standard this charter governs.
- `docs/TECH_CHOICES.md` — PRSM technical-choice rationale for investor context.

**This document is not legal advice.** Before acting on any specific provision, the founder MUST consult qualified corporate counsel in each intended jurisdiction (US, Switzerland, Cayman Islands, or other candidates identified in §5.1).

---

## 1. Purpose

This charter specifies the governance relationship between two legal entities that together operate the PRSM ecosystem:

1. **The PRSM Foundation** — a non-profit entity owning and governing the protocol, its standards, and the trust-anchor infrastructure.
2. **Prismatica** — a for-profit commercial entity operating first-implementer hardware, first-deployed meganodes, and other revenue-earning services built on top of PRSM.

The charter establishes the structural separation between these entities, the mechanisms that prevent the founder or any successor from collapsing them into a single commercial interest, and the processes by which standards are ratified, amended, and enforced.

It is a deliberate choice to put these commitments in writing **before** any of the potential conflicts-of-interest become economically meaningful. A governance charter ratified after the founder has already taken material Foundation-side decisions would be compromised at ratification; one ratified in advance becomes a binding constraint on the founder's future decisions.

---

## 2. Governance principles

The structural decisions below implement four principles, in order of priority.

**P1. No single entity, including the founder, can unilaterally rent-seek from the PRSM ecosystem.**
Any economic return earned by Prismatica through its ecosystem role is subject to Foundation-set standards. Any Foundation action that would favor Prismatica over competitors requires documented justification auditable by every FTNS holder.

**P2. The founder's influence is bounded and time-limited.**
Founder holds concrete formal authority during the bootstrapping phase (roughly Years 1-3, see §8). After that phase, the founder's authority devolves to community governance under rules ratified in advance. The founder cannot postpone this devolution.

**P3. The Foundation is authoritative for standards; Prismatica is authoritative for its commercial operations. Neither can override the other.**
The Foundation cannot tell Prismatica how to price its products. Prismatica cannot tell the Foundation what the silicon standard requires. The boundary is maintained by entity separation + conflict recusal + public audit.

**P4. Transparency is the default; opacity requires justification.**
Voting records, financial statements, committee correspondence (with routing delay for operational security), and certification decisions are all public by default. Exceptions require a documented public rationale.

These principles are the tests applied when any ambiguous governance question arises. Rules below implement them; if a rule fails a principle test in an edge case, the principle wins and the rule is amended (§13).

---

## 3. Entity summary

| | PRSM Foundation | Prismatica |
|---|---|---|
| **Legal form** | Non-profit (specific vehicle TBD, §5.1) | For-profit corporation (Delaware C-corp recommended) |
| **Tax status** | Tax-exempt where jurisdiction permits | Taxable |
| **Purpose** | Protocol stewardship, standards, public infrastructure | Commercial products: T4 meganodes, first-implementer CIS hardware, managed services |
| **Board** | 5-7 members, diverse, staggered terms | 3-7 members, standard corporate form |
| **Revenue** | Grants, FTNS allocation, certification fees (cost-recovery) | Product sales, services, Prismatica-owned IP licensing |
| **Assets** | Protocol contracts, trademarks, registries, standards docs, treasury FTNS | Hardware designs (circuit-level), proprietary performance optimizations, customer relationships, equity |
| **Can modify standards?** | Yes, via formal ratification process (§9) | No. Must petition as any other implementer. |
| **Can modify protocol contracts?** | Yes, via on-chain governance + 2-of-3 multi-sig | No |
| **Can modify each other's policies?** | No | No |
| **Revenue flow from the other** | None directly (Prismatica pays certification fees at cost, on-chain FTNS fees as any user would) | None from Foundation (Foundation does not subsidize Prismatica) |

---

## 4. Foundation: purpose, authority, limits

### 4.1 Mission statement

The PRSM Foundation's mission is to steward the PRSM protocol and its open standards such that any qualified party can participate as a node operator, model publisher, chip manufacturer, or governance participant, without any single entity (including the Foundation itself) holding rent-extracting leverage.

The Foundation's measure of success is NOT the size of its treasury, the number of Foundation-employed staff, or the market capitalization of FTNS. It is the diversity of the PRSM ecosystem — the count of independent implementers at each protocol layer, the breadth of model publishers, the geographic and institutional diversity of node operators — measured against targets published in its annual report (§11).

### 4.2 Formal powers

The Foundation has exclusive authority over:

1. **Protocol standards.** PRSM-CIS-1 (silicon), successor standards for future primitives, and amendments thereto. Standards are ratified by the process in §9.
2. **Contract governance.** The Foundation holds upgrade authority for Base-deployed contracts: ProvenanceRegistry, RoyaltyDistributor, AttestationRegistry, and successor contracts. Upgrade execution requires 2-of-3 hardware multi-sig (§10.3). No upgrade proceeds without published justification and a 30-day community review period.
3. **Certification authority.** The Foundation grants and revokes PRSM-CIS-1 certifications per §15 of that standard. The certification committee is a Foundation-elected body with the composition constraints specified in PRSM-CIS-1 §15.4.
4. **Trademark and domain ownership.** "PRSM," the PRSM logo (when registered), and the `prsm.network` / `prsm.org` domain assets (or successor domains) are Foundation-owned.
5. **Treasury management.** Foundation-held FTNS treasury allocation, fiat reserves, and grants are managed by the Foundation board within limits set by its charter. Any single transaction over $250,000 equivalent requires the full board's approval and 48-hour notice period.
6. **Research grants.** Foundation sponsors academic and independent-research work on PRSM-adjacent topics (R1-R8 from the research track). Grant decisions are made by a research committee reporting to the Foundation board; no grant may flow to any entity in which the founder or any board member has material financial interest without supermajority approval AND full recusal of conflicted members.

### 4.3 Express limits

The Foundation MUST NOT:

1. **Develop or sell commercial products competing with implementers.** No Foundation-branded chip. No Foundation-operated meganodes. No Foundation-managed customer relationships.
2. **Subsidize or favor Prismatica (or any single implementer) in its economic operations.** Certification fees are cost-recovery. Grants are open-bid. On-chain contract operations are policy-blind to who operates them.
3. **Make the founder employed by the Foundation if the founder simultaneously holds executive roles at Prismatica.** During the bootstrapping phase (§8), the founder may serve as an *uncompensated* Foundation board director while holding an executive Prismatica role; after the bootstrapping phase, the founder MUST choose one affiliation (§8.3).
4. **Hold more than 20% of circulating FTNS supply after Year 5.** Foundation treasury allocation is large during bootstrap to fund grants and operations, but its long-run stake is capped.
5. **Operate confidentially.** All voting records, all standards drafts, all committee correspondence older than 90 days, and all financial statements MUST be public on a fixed cadence (§11).
6. **Accept donations or grants subject to strategic influence.** Donors may fund specific research tracks, but donors MAY NOT dictate standard outcomes or certification decisions. Any donor attempting to condition support on a specific certification outcome results in immediate refusal and public disclosure of the attempt.

### 4.4 Board composition

The Foundation board has **5-7 voting members** (odd-count preferred; 7 is the recommended starting size after bootstrap). Seats are categorized:

| Seat category | Count | Selection |
|---|---|---|
| Academic / research | 1-2 | Nominated by a committee of existing academic members + confirmed by supermajority board vote |
| PRSM-staked community | 2 | Elected by token-weighted staker vote (Phase 7+); before staking contracts ship, appointed by founder + confirmed by existing board |
| Independent technical | 1-2 | External security/cryptography expertise; no commercial relationship with any PRSM ecosystem participant |
| Ecosystem operators | 1 | Meganode operator, chip designer, model publisher, etc. — rotating category |
| Founder (bootstrap only) | 1 | Founder holds seat during bootstrapping phase only (§8) |

Board member terms are **3 years**, staggered (one-third of seats turn over each year after full ramp). Members may serve **at most 2 terms** (6 years consecutive). After a mandatory 1-term off-board period, former members may stand again.

No more than **one member** may be affiliated with Prismatica at any given time. No more than **one member** may be affiliated with any other single first-implementer entity.

Board members who develop disqualifying conflicts mid-term must resign or be removed by 2/3 vote of remaining members. "Disqualifying conflicts" include: taking material equity in any PRSM implementer after appointment; taking a paid executive role at any implementer; being named in material litigation involving PRSM.

### 4.5 Executive function

The Foundation employs an **Executive Director** reporting to the board. The ED:

- Is compensated at market rate for a non-profit executive (benchmarked against Ethereum Foundation, Web3 Foundation, etc.).
- May not hold material equity or executive role in Prismatica or any other PRSM implementer.
- Serves at the board's pleasure (standard at-will with severance terms set by charter).
- Handles day-to-day operations: staff management, budget execution within board-approved envelope, external representation.

The ED has no vote on: standards ratification, certification decisions, contract upgrades, or grants above $50,000. These are board-reserved.

The Foundation MAY employ other staff (technical writer, community manager, finance officer, counsel). All staff must be disclosed publicly (name + role + compensation range) in the annual report.

### 4.6 Funding

Foundation revenue comes from, in order of expected magnitude over time:

1. **Initial FTNS treasury allocation.** The protocol's genesis token distribution reserves a Foundation tranche (specific percentage TBD in PRSM-TOK-1 — see §7.2). This funds operations during bootstrap.
2. **Annual FTNS protocol fees.** A small percentage of on-chain transaction fees flows to the Foundation treasury for ongoing operations. Percentage TBD; target is enough to cover operations without making the Foundation dependent on protocol-level fee extraction.
3. **Certification fees** at cost-recovery for PRSM-CIS-1 certification work.
4. **Research-track donations.** Earmarked funding from industry partners, philanthropic organizations, grant programs.
5. **Never: revenue-sharing arrangements with Prismatica or any implementer.**

---

## 5. Foundation: legal form

### 5.1 Candidate jurisdictions — *DECISION REQUIRED*

The Foundation requires a legal form that:
- Is recognized as non-profit / non-distributing in its home jurisdiction.
- Permits the holding of digital assets (tokens, treasury crypto) without punitive tax treatment on paper gains.
- Provides permanence: the founder cannot dissolve the entity and take the assets.
- Has established precedent for protocol foundations with comparable scope.

**Candidate A: Swiss Association (Verein) / Foundation (Stiftung).**

- Precedent: Ethereum Foundation (Stiftung Ethereum, Zug), Web3 Foundation (Verein, Zug), Interledger Foundation, others.
- Pros: mature regulatory framework for protocol foundations; strong founder-can't-capture protections under Swiss foundation law; respected globally.
- Cons: requires Swiss resident or representative; annual regulatory filings; higher setup cost (~CHF 50,000-150,000 for Stiftung).
- Recommended for: this charter if the founder is willing to work through a Zug-based formation process.

**Candidate B: Cayman Islands Foundation Company.**

- Precedent: many Web3 projects (Uniswap Foundation, etc.) formed post-2022.
- Pros: specifically designed for non-profit with no members and no shareholders; founder can establish with less regulatory overhead than Swiss; permits treasury flexibility.
- Cons: perception concerns (some institutional partners dislike Cayman entities); newer precedent; less proven in edge cases.

**Candidate C: Delaware Non-Stock Public Benefit Corporation with LLC wrapper.**

- Precedent: some US-based protocols.
- Pros: familiar US jurisdiction; may help with US investor conversations; moderate setup cost.
- Cons: less proven for pure-protocol governance (typically used for commercial public benefit corps); IRS 501(c)(3) status is hard to obtain for protocol stewardship vs. "charitable/educational" purposes.

**Candidate D: US 501(c)(3) Public Charity.**

- Pros: best US tax treatment; institutional legitimacy.
- Cons: IRS determination is slow and uncertain for protocol-stewardship purposes; annual reporting burden (Form 990); restricts political/governance activities; not historically used by protocol foundations.

**Recommendation (pending counsel):** Swiss Stiftung in Zug if cost + geographic proximity allow; otherwise Cayman Foundation Company. Avoid Candidates C and D for PRSM's specific scope.

### 5.2 Founding documents checklist

Whichever jurisdiction is chosen, the founding documents MUST explicitly:

1. Name the Foundation's charitable / non-distributing purpose in terms sufficient for the jurisdiction's non-profit framework.
2. Prohibit distribution of assets to the founder, board, or any insider except as reasonable compensation for services rendered.
3. Require that, upon dissolution, remaining assets flow to another organization with compatible purposes (NOT to Prismatica; specifically name categories like "other protocol foundations," "academic institutions," or "public-interest charities").
4. Specify amendment procedures requiring supermajority vote AND public comment period AND regulatory review where applicable.
5. Include the conflict-of-interest policy (§12) as a binding part of the founding documents, not merely an internal policy the board can repeal.
6. Incorporate this charter (PRSM-GOV-1) by reference, with a pointer to its canonical on-chain hash recorded at Foundation registration time.

---

## 6. Prismatica: purpose, authority, limits

### 6.1 Mission statement

Prismatica is a for-profit commercial entity that builds and operates PRSM-compatible commercial products. Its first focus is the T4 meganode — a large-scale inference operation running on CIS-compliant hardware — and the design + operation of first-implementer silicon conforming to PRSM-CIS-1.

Prismatica's commercial success is expected and healthy for the ecosystem; PRSM's open-standards approach only works if implementers can earn sustainable returns. **What Prismatica CANNOT do is become the de facto controller of PRSM standards or infrastructure.** The charter enforces this separation.

### 6.2 Prismatica's formal scope

Prismatica MAY:

1. **Design and manufacture CIS-compliant chips** through contracted fabs. Prismatica owns its specific circuit designs, its fab relationships, its performance optimizations.
2. **Build and operate T4 meganodes** at its own cost, capture revenue from the services they provide, subject to PRSM protocol fees that apply to all operators.
3. **Offer managed services** to model publishers who want Prismatica to handle the deployment of their weights to Prismatica-operated meganodes. These commercial relationships are outside PRSM's protocol scope.
4. **License its non-PRSM-essential IP** to third parties on commercial terms.
5. **Employ the founder** and pay market-rate compensation + equity.
6. **Raise venture capital** and operate as a conventional commercial entity with a standard corporate form.

Prismatica MUST NOT:

1. **Claim authority over PRSM standards.** All Prismatica public communications MUST represent Prismatica as "a first implementer of PRSM-CIS-1" rather than "the standard-bearer" or equivalent wording. The distinction is strict; marketing teams sometimes get this wrong and the Foundation reserves the right to publicly correct.
2. **Misrepresent certification status.** Only chips certified by the Foundation can be marketed as "PRSM-CIS-1 certified." Provisional or in-progress certifications must be represented as such.
3. **Use Foundation trademarks** beyond the Foundation's standard licensing terms (which are symmetric to all implementers).
4. **Hold on-chain governance power disproportionate to its operational stake.** Prismatica may hold FTNS for treasury reasons but MUST NOT make Foundation-unfavorable votes as a strategy. Prismatica's on-chain voting behavior is subject to disclosure in its investor reports.

### 6.3 Prismatica's legal form — *RECOMMENDATION*

**Delaware C-Corporation**, standard startup corporate form.

Rationale:
- Investors expect this form for venture financing.
- Clean separation between shareholders (founders, employees, investors) and operational control.
- Established corporate case law for conflicts, fiduciary duty, derivative actions.
- Tax treatment is predictable.
- Employment law (stock options, equity compensation) is well-established.

Alternatives (Nevada, Wyoming, non-US) offer marginal advantages in specific tax or privacy dimensions but generally don't outweigh Delaware's investor familiarity for a venture-backed company.

### 6.4 Prismatica board composition

Standard venture-financed Delaware corporation:

- **Common-stock seats** (founder + senior employees): 2 seats at seed; adjusts at subsequent rounds.
- **Preferred-stock / investor seats**: 1-2 seats from Series A onward.
- **Independent seats**: 1-2 seats, mutually agreed by common + preferred. MUST include at least one director with security/cryptography expertise; MAY include the Foundation's ED or a designated Foundation observer (non-voting).

Standard fiduciary duty applies: directors act in the best interest of Prismatica and its shareholders. Directors MUST recuse from any decision where they hold a conflicting interest in the Foundation or any PRSM ecosystem participant.

### 6.5 Foundation observer on Prismatica board

The Foundation MAY, at its option, appoint a non-voting observer to Prismatica's board. The observer:
- Has no voting rights and no formal duty to Prismatica or its shareholders.
- Attends board meetings (or specified subset thereof, per negotiated terms).
- Is bound by standard NDA for board materials not affecting the public-policy surface.
- Reports to the Foundation board — NOT to the Foundation ED — on governance-relevant observations.

The observer provision gives the Foundation visibility into Prismatica's planning without granting governance authority over Prismatica's commercial decisions. Whether Prismatica agrees to this arrangement is a matter of negotiation; the Foundation cannot compel it. The expectation: Prismatica as first implementer agrees voluntarily as a credibility signal that it is not trying to escape governance scrutiny.

---

## 7. FTNS token

### 7.1 Token issuance authority

FTNS is issued by the Foundation. The Foundation holds the contract that mints the token, governs the emission schedule, and controls monetary policy within the parameters ratified in PRSM-TOK-1 (pending separate document).

Neither Prismatica nor any other implementer MAY issue FTNS. Neither may control the minting contract.

### 7.2 Token allocation — *DEFERRED to PRSM-TOK-1*

The specific genesis allocation percentages (Foundation treasury %, founder allocation %, investor allocation %, ecosystem incentives %, public sale %) are out of scope for this charter; they belong in PRSM-TOK-1. This document only constrains:

- Foundation's long-run holdings ≤20% of circulating supply after Year 5 (§4.3 item 4).
- Founder's personal FTNS holdings are disclosed publicly; any vesting schedules are public; any secondary-market sales are disclosed with 24-hour lag for market-integrity reasons.
- Prismatica's FTNS treasury holdings are disclosed in its investor reports and subject to public summary.

### 7.3 Token-holder rights

PRSM-TOK-1 will specify the rights attaching to FTNS holding. For this charter's purposes, it is sufficient that:
- FTNS-holder votes on matters reserved to token holders (e.g., Phase 7+ staking parameters, certain economic-policy votes) are ratified by on-chain governance.
- Foundation board composition is partially elected by staked token holders (§4.4).
- No vote-buying: Prismatica cannot accumulate FTNS specifically to influence a vote unfavorable to the Foundation. This is enforceable via the structural mechanisms (term limits, conflict recusal, public disclosure) rather than a rule that would be unenforceable at the token-contract level.

---

## 8. Founder's dual role during bootstrapping

### 8.1 Bootstrap phase definition

The **bootstrapping phase** lasts from Foundation formation until the earlier of:
(a) **Year 3** from Foundation formation.
(b) The point at which Foundation staking is live and a successor board has been elected by staked-token holders completing one full election cycle.

During bootstrap, the founder may simultaneously:

- Serve as an **uncompensated** director on the Foundation board, holding one of the category seats per §4.4.
- Serve as **CEO of Prismatica** with standard executive compensation (salary + equity + bonus per Prismatica's compensation committee).
- Hold personal FTNS tokens under a standard vesting schedule ratified by both the Foundation board and Prismatica's board of directors.

### 8.2 Mandatory recusals during bootstrap

Even during bootstrapping, the founder MUST recuse from Foundation board decisions on:

1. Any certification vote involving Prismatica or a Prismatica-affiliated implementer.
2. Any standards vote that materially affects Prismatica's competitive position vis-à-vis other implementers (as determined by the Foundation's independent audit committee).
3. Any research-grant vote where Prismatica or a Prismatica-affiliated entity is a candidate recipient.
4. Any Foundation vote that would modify Prismatica's permissions, trademark rights, or certification status.
5. Any Foundation treasury transaction over $10,000 equivalent that could confer a benefit on Prismatica (service contract, supplier relationship, etc.).

Recusal means: the founder leaves the room / video call, the vote proceeds without their participation, and the recusal is recorded in the minutes published in the annual report.

### 8.3 End-of-bootstrap choice

At the end of the bootstrapping phase, the founder MUST choose one of:

1. **Remain on Foundation board, leave Prismatica.** Founder may retain Prismatica equity subject to the standard vesting schedule but MUST step down from any executive or board role at Prismatica. Foundation board service may continue.
2. **Remain at Prismatica, leave Foundation board.** Founder may retain Foundation FTNS holdings subject to their vesting but MUST step down from Foundation board. The founder may continue to participate in Foundation processes as any other community member: submitting public comments, appearing before working groups, etc.
3. **Leave both.** Retire from active governance and commercial role entirely. Retain passive equity / token holdings under standard vesting.

Option 1 or 2 MUST be made in writing to the Foundation board at least 180 days before end-of-bootstrap. Option 3 is implicit if neither 1 nor 2 is invoked.

This is the mechanism that prevents the founder from indefinitely occupying both sides of the Foundation/Prismatica boundary. After Year 3 (or the successor-board trigger, whichever is first), the founder is structurally constrained to one side.

### 8.4 Succession planning during bootstrap

During Year 1, the founder MUST:
- Identify and recruit at least 4 additional Foundation board members across the seat categories in §4.4.
- Recruit a Foundation Executive Director unaffiliated with Prismatica.
- Publish the names, qualifications, and affiliations of all prospective board members to the community for comment before formal appointment.

By Year 2 end, the Foundation MUST have:
- Full-size board (minimum 5, target 7) in place and conducting regular meetings.
- ED hired.
- First annual report published.

By Year 3 end (end of bootstrap):
- Staking live (Phase 7+).
- At least one staking-elected board member seated.
- Founder's end-of-bootstrap choice executed.

---

## 9. Standards ratification process

### 9.1 Standards scope

PRSM standards include:
- PRSM-CIS-1 (silicon, this charter's companion).
- PRSM-CIS-n for n > 1 (successor silicon standards).
- PRSM-ATTEST-n (attestation protocols beyond silicon — e.g., for TEE attestation in the interim before CIS silicon ships).
- PRSM-ECON-n (tokenomics and economic parameters).
- PRSM-CONTRACT-n (on-chain contract architecture standards).
- Other standards the Foundation ratifies from time to time.

### 9.2 Ratification stages

A new standard (or substantive amendment to an existing one) proceeds through:

1. **Working-group formation** (§9.3). Working groups are constituted for each standard; membership is set by the Foundation board on open nomination.
2. **Draft development.** Working group produces drafts. Drafts are numbered v0.1, v0.2, etc. Major version bumps indicate breaking changes; minor version bumps indicate clarifications or additive features.
3. **Public comment period.** Once the working group believes a draft is ready for external review, it is published with an explicit comment period of at least 60 days for v0.x drafts and at least 120 days for v1.0 ratification.
4. **Committee review.** The relevant Foundation committee (certification committee for CIS standards, economic committee for ECON standards, etc.) produces a formal review incorporating public comments.
5. **Board vote.** The Foundation board votes on ratification. Ratification of a v1.0 (first normative release) requires supermajority (2/3). Amendments to ratified standards require simple majority unless they would affect existing implementers' certification status, in which case supermajority is required.
6. **On-chain anchoring.** A cryptographic hash of the ratified document is recorded in the Foundation's StandardsRegistry contract on Base, with the Foundation's multi-sig signature (§10.3) as the authority attestation.
7. **Effective date.** Standards become effective on a date specified in their ratification resolution. First-release standards typically allow 6-12 months before certification testing against them begins.

### 9.3 Working groups

Each working group:
- Has a Foundation-appointed Chair, term 2 years, maximum 2 terms.
- Has 5-9 voting members selected by the Chair + Foundation board with public nomination.
- Is subject to the same conflict-of-interest rules as the Foundation board (§4.4, §12).
- Meets regularly; minutes are published after a 90-day operational-security delay.
- Has the authority to commission research, engage external experts, and spend from a Foundation-allocated budget within envelope.
- Reports to the Foundation board on a quarterly cadence.

### 9.4 Emergency amendments

The Foundation board MAY, by unanimous vote + published rationale, adopt emergency amendments to a standard without the full public comment period. Emergency amendments are valid for 90 days, during which the standard public comment + committee review process MUST complete. If the process does not conclude in favor of the emergency amendment by day 90, the amendment lapses automatically and the prior version is restored.

Emergency amendments are intended for vulnerabilities or critical bugs that demand immediate action (e.g., a newly-discovered side-channel attack that requires a certification threshold change). They are NOT a mechanism for routine changes.

---

## 10. On-chain governance execution

### 10.1 Foundation multi-sig

The Foundation operates a **2-of-3 Safe multi-sig** on Base (initially; successor chains may be added by standards amendment). The three signers are Foundation-elected officers holding keys on hardware wallets with public identities disclosed in the annual report.

All Foundation on-chain actions (contract upgrades, treasury movements, standards registry writes, certification registrations) execute through this multi-sig. Single-signer or zero-signer actions are not permitted for Foundation-authoritative operations.

### 10.2 Signer rotation

Signers are rotated at least every 2 years, staggered so no more than one key changes in any 12-month period. Rotation requires:
- Nomination of a successor by the outgoing signer + concurrence of the other two signers.
- Board supermajority approval.
- On-chain execution of a Safe transaction adding the new signer + removing the old signer.
- Public disclosure of the new signer's identity and key material commitment (key custody location, hardware device model).

A signer who loses their key (hardware failure, physical loss) triggers immediate emergency rotation. Foundation reserves the right to escrow signer hardware in Foundation-operated secure locations as a redundancy.

### 10.3 Transaction transparency

Every Foundation multi-sig transaction is:
- Publicly announced at least 14 days in advance, unless it's a routine operation listed in the pre-approved-operations schedule.
- Published with full transaction details on the Foundation's transparency page.
- Subject to community comment during the 14-day notice period; substantive objections require Foundation board response before execution.

### 10.4 Emergency transactions

The Foundation MAY execute an emergency transaction without the 14-day notice if:
- Time sensitivity is explicitly justified in a published emergency-rationale document.
- Emergency is unanimously declared by all three multi-sig signers.
- Post-execution, the transaction is subject to a mandatory 30-day review period during which community objections can trigger rollback (via a subsequent governed transaction) if technically feasible.

Emergency transactions are rare by design — the 2-of-3 structure is intentionally slow for policy-critical actions.

---

## 11. Transparency and reporting

### 11.1 Annual report

The Foundation publishes an **annual report** on its anniversary of formation, covering:

1. Mission-progress metrics (ecosystem diversity indices, certification counts by level, grant recipients + outcomes, protocol usage metrics).
2. Governance activity (board votes, committee actions, standards ratified, certifications granted/revoked).
3. Financial statements (audited where applicable): revenue, expenses, treasury positions (fiat + FTNS + other assets).
4. Board member roster with affiliations and compensation.
5. Staff roster with roles and compensation ranges.
6. Known conflicts-of-interest and how they were handled.
7. Year-ahead plans.

The annual report is mailed to every registered stakeholder and posted on the Foundation's public site + anchored on-chain via StandardsRegistry.

### 11.2 Quarterly disclosures

Quarterly, the Foundation publishes:

1. Treasury position snapshot.
2. Material changes to certifications, standards, or governance.
3. Upcoming agenda items for the following quarter.

### 11.3 Continuous disclosures

Published in real time (on a dedicated transparency page):

1. Every multi-sig transaction with its rationale.
2. Every certification grant / revocation with redacted test summary.
3. Every standards-registry update.
4. All board meeting minutes 90 days after the meeting.
5. All committee meeting minutes 90 days after the meeting.
6. Donations and grants above $25,000 at the point of receipt / issuance.

### 11.4 Independent audit

The Foundation commissions an **annual independent audit** of its financial statements by a qualified auditor. The audit report is published verbatim in the annual report, including any findings of control weaknesses or qualified opinions.

At least every 3 years, the Foundation commissions an **independent governance audit** examining whether the conflict-of-interest rules, recusal requirements, and separation-of-entity commitments are being followed. This audit is conducted by a third party with no commercial relationship to any ecosystem participant; findings are published with Foundation response.

---

## 12. Conflict-of-interest policy

### 12.1 Scope

This policy binds: Foundation board members, Foundation officers, Foundation working-group members, Foundation committee members, and Foundation employees. Policy-equivalent language is required (by charter) in the Prismatica-side governance for any individual serving in a role that interfaces with Foundation decisions.

### 12.2 Disclosure requirements

Every covered individual MUST disclose, at appointment and annually thereafter:

- Equity, token holdings, or financial interest in any PRSM ecosystem participant.
- Employment, consulting, or advisory relationships with any PRSM ecosystem participant.
- Family or close-personal relationships with individuals in such positions.
- Material transactions (gifts, paid travel, speaking fees) from ecosystem participants above $2,500 per year.

Disclosures are public (redacted where privacy law requires). Failure to disclose a material conflict is grounds for immediate removal.

### 12.3 Recusal triggers

Covered individuals recuse from any decision where:

- They have a direct financial interest in the outcome.
- They have an immediate family member with such interest.
- They hold an employment / consulting / advisory role at an affected entity.
- A reasonable person would perceive a material conflict.

Recusal means: no participation in discussions, no vote, physical or video absence from the decision-making session. Recusal is recorded in minutes.

### 12.4 Per-meeting affirmations

At the start of each Foundation board meeting and each certification committee meeting, attendees affirm (verbally, for the minutes) that their disclosures are current. Material changes since last affirmation must be stated at this point.

### 12.5 Policy enforcement

The Foundation appoints an **Ethics Liaison** (either a board member without conflicts or an external counsel) to:
- Receive conflict disclosures and maintain the register.
- Rule on close-call recusal questions in real time (with appeal to the full board).
- Investigate alleged violations.
- Publish annual conflict-register summary in the annual report.

The Ethics Liaison may be the same person as the Foundation's General Counsel if the appointment does not itself create conflicts; otherwise the role is separate.

---

## 13. Charter amendment

### 13.1 Amendment authority

This charter (PRSM-GOV-1) may be amended only by:

1. Foundation board supermajority (2/3) approval.
2. A public comment period of at least **120 days** for material amendments, **60 days** for clarifications.
3. On-chain recording of the amended charter hash in the StandardsRegistry.
4. Ratification by, where applicable, the Foundation's jurisdictional regulator (e.g., Swiss foundation supervisory authority).

Amendments affecting Sections 2 (Principles), 4.3 (Foundation Express Limits), 6.2 (Prismatica MUST NOTs), or 8 (Founder Dual Role) require a supermajority of the FULL board — not merely a 2/3 of those present. This makes core structural commitments harder to relax.

### 13.2 Prohibited amendments

Certain provisions are **not amendable** by the standard process. Changes to these require Foundation dissolution and reformation (i.e., the community forms a new foundation with different rules):

- The principle that the Foundation does not develop or operate commercial products competing with implementers.
- The multi-manufacturer requirement at C2+ in PRSM-CIS-1.
- The prohibition on Foundation-to-Prismatica revenue-sharing arrangements.

The rationale for prohibited amendments: these are the load-bearing commitments that make the governance split credible. An amendment process that could unravel them would undermine the credibility from day one.

### 13.3 Sunset for legacy provisions

As this is v0.1, some provisions may prove unworkable in practice. The charter EXPECTS revision through the v0.2, v1.0 ratification cycle. Amendments are a normal governance activity, not a crisis. The only provisions to guard against are those that would favor a narrow interest over the ecosystem — those are the ones the supermajority + public-comment + prohibited-amendments structure is designed to defeat.

---

## 14. Open issues

Items requiring resolution before v1.0 ratification:

### 14.1 Jurisdiction selection — *HIGHEST PRIORITY*

§5.1 presents options without final recommendation. This choice must be made before Foundation formation, and it shapes every other governance provision's enforceability. **Action:** founder consults qualified corporate counsel in at least two candidate jurisdictions; publishes comparison memo for community review; selects jurisdiction with Foundation board-in-formation concurrence.

### 14.2 FTNS allocation specifics — *DEFERRED to PRSM-TOK-1*

§7 intentionally does not set specific percentages. PRSM-TOK-1 (pending) will specify:
- Foundation treasury allocation at genesis.
- Founder allocation with vesting.
- Ecosystem incentives.
- Investor allocations (if any).
- Public sale design (if any).
- Emission schedule.

These numbers must survive a separate ratification process but are not governance-charter territory.

### 14.3 Board member initial selection

The founder cannot simultaneously nominate a majority of initial board members AND credibly claim the founder is not capturing governance. The charter specifies recruitment timelines (§8.4) but not the specific selection mechanism for the first cohort. **Options under consideration:**
- Founder nominates initial 4 members; each subsequent new member requires confirmation of the existing board (including the confirmed members).
- Founder nominates initial 2 members; 2-member board nominates the next 2 by unanimous vote; 4-member board nominates remainder by supermajority.
- Advisory panel of 5 external figures (published identities) nominates the initial board with founder's concurrence required.

**Recommendation pending; target resolution in v0.2.**

### 14.4 Protocol fee recipient specifics

§4.6 item 2 references annual FTNS protocol fees flowing to the Foundation. The specific mechanism (percentage of per-transaction fees, percentage of FTNS emission, fixed annual grant from Foundation treasury at bootstrap) is a PRSM-TOK-1 question but has governance implications. **Target resolution:** PRSM-TOK-1 v0.1 with cross-reference back to this charter.

### 14.5 Prismatica-specific commitments template

Prismatica's own corporate governance must include certain provisions (conflict recusal for directors who serve on Foundation bodies, disclosure of FTNS holdings, etc.) for the charter's inter-entity boundary to be enforceable. A **Prismatica Governance Template** companion document should specify the minimum provisions Prismatica's corporate documents must include. **Target:** 2026-Q3 draft.

### 14.6 Non-profit status filing strategy

If the jurisdiction chosen is US-based, whether to pursue 501(c)(3) determination is a major decision with 18-36 month timeline. Non-US jurisdictions have their own non-profit filing processes. **Action:** incorporated into §14.1 jurisdiction-selection counsel engagement.

### 14.7 Emergency governance continuity

If the full Foundation board becomes unable to function (catastrophic loss of signers, simultaneous resignations, regulatory action), the charter needs a continuity procedure. Current draft relies on "the remaining members appoint successors," which fails if zero members remain. **Target resolution:** v0.2 emergency-continuity appendix.

### 14.8 Policy on Foundation employment after board service

Former Foundation board members — particularly the Executive Director — accumulate specialized knowledge. A cooling-off period before they can take commercial roles at ecosystem participants is common in government ethics contexts. Should PRSM adopt a similar policy? **Options:** 12-month, 24-month, indefinite, or none. **Target:** v0.2 discussion.

---

## 15. Acknowledgements

Drafting of this charter draws on governance patterns established by:

- **Ethereum Foundation** (Zug) — the template for protocol-foundation governance at scale.
- **Apache Software Foundation** — meritocratic community governance model.
- **IETF / RFC process** — public, documented standards ratification.
- **Linux Foundation** — multi-stakeholder open-source governance.
- **Confidential Computing Consortium** — domain-specific standards-setting body.
- **Trusted Computing Group** — hardware-security standards with implementer diversity.

Specific rhyming decisions with these precedents are noted inline. Where this charter diverges, it is typically toward stricter founder-containment than the precedents require — PRSM's specific threat model (single-founder + first-implementer affiliation from day one) justifies the additional rigor.

---

## 16. Change log

**v0.1 (2026-04-21):** Initial draft for founder/counsel review. Authored by Claude Opus 4.7 on behalf of the PRSM founder. Covers principles, entity scopes, founder dual-role containment, standards ratification process, on-chain execution, transparency, conflicts, and amendment. Ten open issues flagged with target resolution milestones. Not legal advice; counsel consultation required before any provision is acted upon.

---

**End of PRSM-GOV-1 v0.1 Draft.**
