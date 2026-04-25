# PRSM-POLICY-JURISDICTION-1: Foundation Boundary Policy for Jurisdictional Censorship Resistance

**Document identifier:** PRSM-POLICY-JURISDICTION-1
**Version:** 0.1 Draft
**Status:** Draft ready for legal-counsel review + board ratification once Foundation forms
**Subsidiary to:** `docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md` §4.3 (Foundation express limits)
**Date:** 2026-04-23
**Drafting authority:** PRSM founder (pre-Foundation)
**Ratification authority:** First full Foundation board, per PRSM-GOV-1 §9
**Related documents:**
- `docs/2026-04-23-r9-transport-censorship-resistance-scoping.md` — R9-SCOPING-1, the design doc this policy codifies as ratified governance commitment.
- `docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md` §4.3 — Foundation express limits (this policy's anchor).
- Foundation jurisdiction-scoping analysis (Cayman Foundation Company) — this policy operates within that jurisdiction choice (document in private repo).
- Foundation officer independence rubric — this policy cross-references §3 of that document (in private repo).
- `docs/2026-04-22-r3-threat-model.md` — operator-level threat model this policy's user-facing risk disclaimer references.
- `docs/2026-04-22-prsm-supply-1-supply-diversity-standard.md` — geographic-concentration metrics this policy's peer-filter primitive consumes.
- `prsm/node/transport_adapter.py` + `prsm/node/jurisdiction_filter.py` + `prsm/node/content_self_filter.py` — the protocol-layer mechanisms this policy governs.

---

## 1. Purpose

PRSM's protocol layer ships transport-pluggability, peer-jurisdiction filtering, and content self-filter mechanisms (R9 Phase 6.2-6.4) so operators in censoring jurisdictions can make informed, local decisions about their own compliance posture. Shipping those mechanisms does NOT make the Foundation a circumvention-advocacy organization.

This policy draws the line. It codifies what the Foundation commits to doing, what the Foundation commits to NEVER doing, and what operators and users must understand about the limits of protocol-layer protection.

**This policy is the governance-ratified version of R9-SCOPING-1 §8.** The scoping doc is engineering intent; this doc is the ratified Foundation commitment once it exists.

The policy binds:
- The Foundation board + Executive Director + any Foundation employees.
- Foundation-operated infrastructure and communications channels.
- Any Foundation-authored or Foundation-approved PRSM documentation, tooling, or outreach.

The policy does NOT bind:
- Individual operators — they configure their own node behavior within the protocol's pluggable interfaces.
- Third-party mission-specific non-profits (Tor Project, Access Now, OONI, etc.) that may operate infrastructure compatible with PRSM protocols.
- Users who choose to run the protocol regardless of their jurisdictional context.

---

## 2. Foundation operational commitments

The Foundation commits to the following on an ongoing basis. These are affirmative obligations of the governance body.

### 2.1 Ship mechanisms neutrally

The Foundation commits to maintaining the protocol-layer mechanisms for transport pluggability, peer filtering, and content self-filter such that:

- No jurisdiction is named or targeted in default configuration.
- No transport, GeoIP source, or content-classification service is endorsed over another by Foundation publications.
- Every operator-facing configuration is documented without presuming the operator's threat model.

### 2.2 Document risk honestly

The Foundation commits to user-facing documentation that:

- States plainly, without obfuscation, the jurisdictions where PRSM operation is high-risk.
- Explains the specific legal, technical, and enforcement risks in each such jurisdiction (to the extent verifiable from public sources).
- Includes the disclaimer language from §4 on every document intended for operators in high-risk jurisdictions.
- Is updated at minimum annually to track regulatory changes (PRC 2023 Interim Measures, EU AI Act provisions, similar).

### 2.3 Support mechanism audits

The Foundation commits to making the transport pluggability, peer filter, and content self-filter mechanisms available for independent security audit on the same terms as any other PRSM protocol component (per PRSM-GOV-1 §4.2 authority on standards review + audit scope).

### 2.4 Not impede delegated operation

If a third-party mission-aligned non-profit (The Tor Project, OONI, Access Now, similar) operates circumvention infrastructure compatible with PRSM's pluggable-transport interface, the Foundation commits not to impede that operation through protocol changes, reputation sanctions, or governance actions. The Foundation remains separate from such operations; it does not participate.

### 2.5 Uphold protocol-layer privacy properties

The Foundation commits that any telemetry, logging, or observability the Foundation operates directly (e.g., on Foundation-hosted bootstrap list endpoints) will:

- Not retain operator IP addresses beyond the TTL required for legitimate DDoS mitigation.
- Not correlate FTNS on-chain activity with operator network telemetry.
- Be subject to the annual independent audit per PRSM-GOV-1 §6.

---

## 3. Foundation operational non-commitments

The Foundation commits to NEVER doing the following. These are negative obligations — irrevocable absent dissolution (see §5.2).

### 3.1 No Foundation-operated circumvention infrastructure

The Foundation SHALL NOT:

- Operate Tor bridges, Tor exit relays, Snowflake proxies, V2Ray servers, Trojan servers, Shadowsocks servers, or equivalent circumvention-transport infrastructure under Foundation operational control.
- Fund the operation of such infrastructure by third parties.
- Maintain configuration presets that wire operators to specific known-to-work circumvention endpoints.

Rationale: operating such infrastructure is the mission of purpose-built non-profits (The Tor Project, Wireguard-adjacent orgs, Access Now's bridge program). The Foundation does not pursue that mission. Conflating the two missions would (a) dilute the Foundation's technical standards-body mission, (b) make the Foundation a target of adversarial jurisdictions in ways the technical mission should not be, (c) create funding + legal exposure unsuited to the Foundation's charter.

### 3.2 No adversary-specific preset configurations

The Foundation SHALL NOT ship documentation, templates, configuration files, or presets that:

- Are keyed to specific censoring jurisdictions (e.g., "PRC mode", "Russia mode", "Iran mode").
- Advertise themselves as having been tested-and-verified safe in any such jurisdiction.
- Imply through default configuration that the combination of transport + resolver + filter has been validated for any specific threat model.

Rationale: Such presets necessarily lie. Adversarial jurisdictions evolve their detection faster than the Foundation could update presets. Publishing a "PRC mode" that is detected and blocked within weeks creates false confidence that puts operators at greater risk than running with honest uncertainty. Per R9-SCOPING-1 §3: transport-layer protection is not a promise of safety.

### 3.3 No curated bridge / proxy distribution

The Foundation SHALL NOT:

- Maintain lists of "currently working" Tor bridges for specific adversaries.
- Distribute bridge lines, proxy credentials, or similar out-of-band operational data.
- Act as an intermediary between Tor Project's bridge-distribution system (BridgeDB, Moat, similar) and PRSM operators.

Rationale: Active bridge distribution is time-sensitive operational work requiring dedicated infrastructure, threat-model expertise, and jurisdiction-specific legal counsel. It is The Tor Project's core competence, not the Foundation's.

### 3.4 No operational-circumvention advocacy

The Foundation SHALL NOT:

- Sign open letters, petitions, or coalition statements advocating for specific legal reforms in specific countries.
- Join advocacy coalitions whose primary mission is circumvention of state censorship.
- Publicly name-and-shame specific state actors for censorship actions against PRSM or its users.
- Represent itself in international venues (UN, ITU, IGF, similar) in an advocacy capacity on censorship topics.

Rationale: Technical standards bodies that become advocacy organizations lose their technical credibility in one direction and their advocacy effectiveness in the other. PRSM's mission is neutral compute infrastructure. Advocacy is legitimate work but is done by purpose-specific organizations (Access Now, EFF, Article 19) who have the dedicated staff + legal infrastructure to pursue it safely.

### 3.5 No commercial endorsements of circumvention tools

The Foundation SHALL NOT:

- Publish "recommended VPN provider" lists.
- Enter into marketing, affiliate, or partnership arrangements with commercial circumvention-tool providers.
- Include commercial tool recommendations in Foundation-authored documentation, beyond neutral statement that such tools exist.

Rationale: Commercial VPN + circumvention providers vary in trust posture, funding sources, operational security, and jurisdictional exposure. The Foundation is not positioned to vet them on users' behalf and should not appear to.

### 3.6 No legal defense fund or individual legal support

The Foundation SHALL NOT:

- Promise, represent, or imply legal defense support to operators prosecuted for PRSM operation in any jurisdiction.
- Operate an "operator legal support" program.
- Reimburse operator legal fees from Foundation treasury absent formal board action under specific, case-by-case criteria to be established in a separate policy.

Rationale: Legal defense is specialized non-profit work (EFF, Digital Defense Fund). The Foundation's treasury and governance are not structured for this. Promising what the Foundation cannot reliably deliver is worse than not promising.

---

## 4. User-facing risk disclaimer template

The Foundation commits (§2.2) to including the following disclaimer in any documentation, tool, or configuration file intended for operators in high-risk jurisdictions. The specific text below is the baseline; Foundation publications may elaborate but shall not weaken.

### 4.1 Baseline disclaimer text

```
PRSM is neutral infrastructure protocol. PRSM is not safe to operate
in any jurisdiction that criminalizes unlicensed crypto infrastructure,
unlicensed generative AI services, or the content types your node may
serve. Specifically: the People's Republic of China, the Russian
Federation, the Islamic Republic of Iran, and other jurisdictions with
similar regulatory frameworks prosecute operators of infrastructure
like PRSM. If you operate PRSM in such a jurisdiction, you accept:

(1) That protocol-layer mechanisms (transport pluggability, peer
    filtering, content self-filter) REDUCE but DO NOT ELIMINATE your
    personal legal risk. State-level adversaries with ML-equipped
    traffic analysis can detect PRSM-protocol traffic even through
    circumvention transports, on average.

(2) That PRSM Foundation does not provide legal defense support,
    legal advice, or operational-circumvention advice. You are
    responsible for consulting local counsel and complying with
    local law.

(3) That your on-chain FTNS activity is publicly traceable. Exchange
    KYC, blockchain analytics services, and compelled disclosure from
    VASPs can re-identify you even if network-layer anonymity holds.

(4) That PRSM Foundation makes no promise that the protocol is, will
    be, or can be made safe to operate in any specific censoring
    jurisdiction.

Your operation is your own decision, at your own risk, under your
own legal responsibility.
```

### 4.2 Disclaimer placement

Required placement of this text:
- In every Foundation-authored "operator guide" or equivalent documentation.
- In the `OPERATOR_GUIDE.md` repository documentation.
- In any configuration-template file the Foundation publishes for the transport adapter + jurisdiction filter + content self-filter.
- In the README for any Foundation-published PRSM SDK (Python, JavaScript, Go) in the section introducing node operation.
- As a one-click acknowledgment on any Foundation-operated onboarding surface (dashboard, CLI `prsm setup` wizard interactive prompt).

### 4.3 Non-English translations

Translations shall preserve the specific jurisdictional enumeration (PRC, Russia, Iran) and the four numbered disclaimers. Weakening the text in translation — for example, dropping the jurisdictional enumeration to reduce diplomatic friction — is prohibited.

---

## 5. Amendment process + prohibited amendments

### 5.1 Amendable sections

Sections §1, §2, §4, §6, §7 may be amended by 2/3 supermajority of the Foundation board, with no fewer than 30 days public notice + 14 days public comment period per PRSM-GOV-1 §9.2.

### 5.2 Prohibited amendments

The following sections are prohibited from amendment short of Foundation dissolution per PRSM-GOV-1 §4.3:

- §3.1 (no Foundation-operated circumvention infrastructure)
- §3.2 (no adversary-specific preset configurations)
- §3.6 (no legal defense fund absent separate policy)

Amendment of these sections requires amendment of the Foundation's core charter (PRSM-GOV-1 §4.3), which itself requires dissolution-and-reconstitution of the Foundation.

Rationale: The prohibited amendments codify commitments that, if reversed, fundamentally change the Foundation's mission from neutral standards-body to advocacy organization. That transition should not happen quietly by board vote; it requires Foundation dissolution + transparent reconstitution under a new charter.

### 5.3 Clarifying amendments

Amendments to §2, §4, §6, §7 that clarify without weakening Foundation commitments (e.g., extending the §2.2 documentation commitment to newly-recognized jurisdictions) may be adopted by simple majority board vote with 7 days public notice. Clarifying amendments cannot relax any §3 non-commitment.

---

## 6. Review cadence

### 6.1 Annual review

The Foundation board shall review this policy annually. Reviews shall assess:

- Whether jurisdictional context (PRC / Russia / Iran / EU / US regulatory posture) has shifted sufficiently to warrant §4.1 text updates.
- Whether any §3 non-commitment has been incidentally violated through Foundation-staff action, partnership, or tooling; if so, whether remediation is required.
- Whether the §4.1 disclaimer remains sufficient given observed operator-prosecution cases (public, verified).

Reviews shall be documented in the annual transparency report per PRSM-GOV-1 §6.

### 6.2 Triggered review

Any of the following triggers a within-30-day board review outside the annual cadence:

- Documented prosecution of a PRSM operator in any jurisdiction.
- Credible threat analysis (from a Foundation security partner, independent security researcher, or R3 threat-model track) identifying a new class of operator risk not addressed by existing protocol mechanisms.
- A jurisdictional regulatory change materially affecting PRSM operator risk in any jurisdiction.
- A Foundation-staff action that a reasonable observer would construe as violating §3.

### 6.3 Policy sunset

This policy does not sunset. §6 reviews may adopt clarifying amendments but cannot terminate the policy absent the prohibited-amendment process in §5.2.

---

## 7. Related protocol mechanisms

This policy governs the Foundation's conduct around the following protocol-layer mechanisms. Each mechanism's technical specification is authoritative in its own document; this policy governs only Foundation operational + communications conduct toward the mechanism.

- **Transport pluggability** — `prsm/node/transport_adapter.py`. Users select their own transport (Tor, V2Ray, Shadowsocks, etc. via local SOCKS5). Foundation ships the pluggable interface neutrally per §2.1.
- **Bootstrap over transport** — `prsm/node/bootstrap_transport.py`. HTTPS + DNS-over-HTTPS bootstrap discovery routed through configured transport. Foundation operates the primary bootstrap endpoint but does not operate adversary-specific alternate endpoints per §3.1.
- **RPC over transport** — `prsm/node/rpc_transport.py`. FTNS settlement RPC routed through configured transport. Foundation does not operate "safe RPC endpoints" for specific adversaries per §3.1.
- **Peer-jurisdiction filter** — `prsm/node/jurisdiction_filter.py`. Operator-configured GeoIP-based peer-selection policy. Foundation ships no default blocklist per §3.2.
- **Content self-filter** — `prsm/node/content_self_filter.py`. Operator-configured defensive content-refusal primitive. Foundation ships no default rule sets per §3.2.

---

## 8. Cross-references to external organizations

This policy recognizes that mission-specific organizations pursue work PRSM Foundation explicitly does not. The following are illustrative; no endorsement is implied and no partnership is assumed.

- **The Tor Project** (US 501(c)(3)) — operates Tor network infrastructure + pluggable transport ecosystem.
- **Access Now** (non-profit) — digital rights advocacy + operator legal defense in high-risk jurisdictions.
- **Electronic Frontier Foundation** — digital civil liberties + some operator legal defense.
- **Article 19** (UK charity) — international freedom-of-expression advocacy.
- **Open Observatory of Network Interference (OONI)** — censorship measurement + reporting.

Users seeking operational, legal, or advocacy support that PRSM Foundation does not provide should engage these organizations directly per their own policies + jurisdictional applicability.

---

## 9. Ratification + effective date

### 9.1 Draft status

This document is drafted pre-Foundation-formation by the PRSM founder acting as draft authority. It has no ratified governance status until adopted by the Foundation board under PRSM-GOV-1 §9.

### 9.2 Ratification process

On Foundation formation, the first full board shall:

1. Review this draft against the actual governance context of the Foundation's formation jurisdiction (Cayman Foundation Company per current plan).
2. Obtain legal-counsel review specifically on whether §3 non-commitments are consistent with Foundation's charter + jurisdictional rules.
3. Ratify, amend-and-ratify, or reject by 2/3 supermajority vote.

### 9.3 Pre-ratification interim status

Until ratification, this document shall be treated as binding guidance on Foundation-associated action by the pre-formation founder + draft-authority team. All §3 non-commitments shall be honored in interim operations even without ratified status.

Specifically: between the date of this draft (2026-04-23) and Foundation ratification, the PRSM founder commits to:

- Not personally operating Tor bridges, V2Ray servers, or equivalent infrastructure while acting in PRSM-representative capacity.
- Not signing circumvention-advocacy coalition statements in PRSM's name.
- Not publishing adversary-specific PRSM configuration presets.
- Including the §4.1 disclaimer in any PRSM-operator documentation published before Foundation formation.

### 9.4 Effective date

If ratified, the policy becomes effective on the date of the board resolution adopting it. Prior interim-status action remains valid.

---

## 10. What this policy does NOT do

- **Does NOT bind individual operators.** Operators configure their own node behavior within the protocol's pluggable interfaces. This policy constrains only the Foundation.
- **Does NOT predetermine Foundation amendments.** The §5.3 clarifying-amendment process exists to allow evolution without requiring dissolution.
- **Does NOT provide legal advice.** Every jurisdictional claim here is drawn from publicly-available regulatory text; operators consulting this document must confirm applicability with their own counsel.
- **Does NOT close operator outreach channels.** Operators in any jurisdiction remain free to interact with the Foundation, receive documentation, participate in governance, etc. This policy governs only what the Foundation does, not whom the Foundation speaks to.
- **Does NOT amend PRSM-GOV-1.** This is a subsidiary policy; PRSM-GOV-1 is authoritative in case of conflict.

---

## 11. Flagged items for legal-counsel review

Legal counsel engaged for Foundation formation should specifically review:

- **Q1:** Whether §3's prohibited amendments are enforceable under Cayman Foundation Company governance structure. Prohibited amendments are explicit in PRSM-GOV-1 §4.3; counsel should confirm this policy's §5.2 reference to that mechanism is operationally sound.
- **Q2:** Whether §4.1 disclaimer language creates implicit legal assurance the Foundation cannot back up. If so, the language should be tightened (not weakened) to avoid implied warranty concerns.
- **Q3:** Whether §2.5 observability commitment (no operator-IP retention, no on-chain-to-network correlation) is operationally feasible given any Foundation-operated infrastructure's actual telemetry needs. If a narrower commitment is needed, draft at the technical-feasibility boundary.
- **Q4:** Whether §3.4 advocacy non-participation conflicts with any legitimate Foundation charitable-purpose obligations under Cayman Foundation Company charitable-status rules.
- **Q5:** Whether §6.2 triggered-review thresholds (operator prosecution, credible threat analysis, regulatory change, staff-action observable as §3 violation) are workable given realistic Foundation staffing. If not, amend the triggers without weakening the underlying commitment.

---

## 12. Document provenance + research citations

This policy draws on:

- **R9-SCOPING-1** (2026-04-23-r9-transport-censorship-resistance-scoping.md) — the engineering-level rationale for every commitment here.
- **PRSM-GOV-1** §4.3 — Foundation express limits, the charter anchor for §5.2 prohibited amendments.
- **Ethereum Foundation + Web3 Foundation + Tor Project public policy statements** — comparable boundary frameworks.
- **Access Now + EFF + Article 19 public policy frameworks** — references for what the Foundation specifically DOES NOT do (because other orgs do it).
- **PRC CAC Interim Measures for Generative AI Services (2023)** — jurisdictional context for §4.1 enumeration.
- **EU AI Act operator provisions** — jurisdictional context for §4.1.

No external consultation was conducted during drafting; the policy is internally sourced from PRSM's own technical + governance documents. Legal counsel review during ratification (§9.2) is where external authority enters.
