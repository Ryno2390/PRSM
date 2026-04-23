# R9 Transport-Layer Censorship Resistance + Jurisdictional Isolation — Scoping

**Document identifier:** R9-SCOPING-1
**Version:** 0.1 Draft
**Status:** Architectural scoping. Specifies mechanism design, integration points with existing Phase 6 P2P hardening, and the explicit boundaries between "protocol-level neutrality" and "operational-circumvention advice" — NOT an execution plan, NOT legal advice, NOT a promise that PRSM is safe to run in any specific jurisdiction.
**Date:** 2026-04-23
**Drafting authority:** PRSM founder
**Promotes:** new research track item (R9) raised after developer-community feedback on the US/China AI-bifurcation and Great Firewall risk for Chinese PRSM participants.
**Related documents:**
- `docs/2026-04-22-r3-threat-model.md` — R3 threat model. R9 generalizes R3's operator-level threat model to state-level and jurisdictional adversaries.
- `docs/2026-04-22-r8-defense-stack-composition.md` — R8 composition analysis. R9 extends R8 from "weight exfiltration" to "operator prosecution" as distinct threat surfaces.
- `docs/2026-04-22-phase6-p2p-hardening-design-plan.md` + `docs/2026-04-22-phase6-task4-dht-tuning-plan.md` — Phase 6 P2P hardening. R9's transport-pluggability extends these primitives.
- `docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md` §4.3 (Foundation limits) — R9 respects the Foundation's constitutional non-involvement in operational censorship circumvention.
- `docs/2026-04-22-prsm-supply-1-supply-diversity-standard.md` — supply-diversity standard. R9's jurisdictional peer-selection interacts with SUPPLY-1's geographic concentration metrics.
- `PRSM_Vision.md` §6-7 "Honest caveats" — vision-doc framing the user-risk posture R9 specifies protocol-level support for.

---

## 1. Purpose

The question raised: **how does PRSM protect users in jurisdictions with adversarial state censorship (notably but not exclusively PRC) without either (a) becoming a circumvention tool that exceeds PRSM's neutral-infrastructure mandate, or (b) abandoning those users to run protocol-level risk alone?**

This doc specifies the architectural answer. It is NOT an implementation plan — execution spans multiple Phase 6 follow-on tasks plus a new governance-policy document (PRSM-POLICY-JURISDICTION-1, draft referenced in §9).

**R9 is a scoping-only document.** It answers six framing questions:

1. What is the realistic threat model for a user operating a PRSM node inside a hostile-state-actor jurisdiction? (§2)
2. What can transport-layer circumvention (Tor, obfs4, meek, Snowflake, V2Ray, etc.) actually accomplish, and what can it not? (§3)
3. What architectural principle should PRSM adopt for censorship resistance? (§4)
4. What defensive mechanisms are available at the transport, identity, and content layers? (§5-§7)
5. What does PRSM's Foundation specifically NOT do, and why? (§8)
6. What integration work would be required to execute R9's proposal? (§9)

**R9 is NOT:** a China-specific policy document, an implementation roadmap, legal advice, or a promise about what running PRSM in any jurisdiction entails. Every user in every jurisdiction retains full responsibility for their own legal compliance. R9 specifies mechanism; it does not sell safety.

---

## 2. Threat model — jurisdictional adversary

### 2.1 Who the adversary is

A state-level actor with:

- Full traffic visibility at the ISP level (national firewall, DPI, flow-level analysis).
- Legal authority to compel disclosure from domestic crypto exchanges, KYC'd on-ramps, and banking counterparties.
- Legal authority to prosecute users for operating unlicensed crypto infrastructure, running unlicensed generative AI services, or serving politically-sensitive content.
- Technical capability to detect crypto-protocol traffic patterns even when wrapped in circumvention transports (obfs4, meek, etc. are detectable at scale given sufficient ML + telemetry budget).
- Intermittent capability to deploy active traffic-injection attacks (probe-and-block on suspected Tor bridges).

### 2.2 What the adversary can do

| Capability | Realism |
|---|---|
| Identify a PRSM node operator by network traffic pattern analysis | **High** — crypto-protocol traffic is distinguishable from normal encrypted traffic even through Tor |
| Correlate on-chain FTNS activity with a KYC'd exchange account | **High** — trivial on-chain graph analysis post-KYC-breach |
| Compel disclosure from domestic VASPs | **High** — formal legal process |
| Block Tor relays + known pluggable-transport bridges | **High** — cat-and-mouse; adversary wins on average |
| Block meek domain-fronting as CDNs cooperate or de-allow | **High** — most major CDNs (Google, Cloudflare, AWS) have already removed or restricted domain-fronting capability as of 2025-2026 |
| Prosecute a user for content served by their node | **Jurisdiction-specific but increasingly high** — e.g., PRC 2023 Interim Measures for Generative AI Services; EU AI Act provisions for operator responsibility for specific content types; UAE + Russia AI Safety frameworks |
| Prosecute a user for receiving/sending FTNS payments | **Jurisdiction-specific** — depends on local crypto legal regime |
| Active MITM on the user's crypto on-ramps | **High for state-controlled ISPs** — TLS interception via state-issued root CAs |
| Break end-to-end cryptographic primitives (AES-256-GCM, Ed25519, Merkle proofs) | **Low** — these are the primitives protecting PRSM's core protocol surface and remain secure against state-level attackers under current cryptographic assumptions |

### 2.3 What the adversary CANNOT do

- Break the cryptographic primitives securing PRSM's on-chain layer (FTNS balance confidentiality against honest majorities, Merkle-proof integrity of batched receipts).
- Access model weights that the user hasn't downloaded locally (they must run the model themselves to serve it; PRSM doesn't store models centrally).
- Forge FTNS balance state (public ledger).
- Suppress the PRSM network's operation globally — only locally within their jurisdictional reach.

### 2.4 User-archetype risk matrix

R9 distinguishes between user-archetype risk profiles:

| Archetype | Primary risk | R9-addressable |
|---|---|---|
| **Passive FTNS holder** (buys FTNS, stakes, earns) | On-chain doxxing via KYC'd exchange | **Partial** — peer-selection helps hide network activity; exchange KYC is out-of-scope |
| **Active provider** (runs a compute node, serves inference) | Traffic-pattern detection + content liability | **Partial** — transport-pluggability helps with traffic pattern, content self-filter helps with liability, but prosecution risk depends on jurisdiction |
| **Content contributor** (uploads models, datasets) | Content liability based on what they upload | **Minimal** — protocol-level mechanism cannot address what content a user chooses to contribute |
| **Governance participant** (votes on proposals, sits on Foundation board) | Identity exposure through on-chain governance actions | **Partial** — anonymous governance votes are feasible; board service is necessarily public |

**R9 optimizes most strongly for the "active provider" archetype** — the user whose primary risk is traffic-pattern detection and content-serving liability. Passive holders and content contributors have additional risk layers that R9 partially addresses but cannot eliminate.

---

## 3. What transport-layer circumvention can and cannot do

### 3.1 What it CAN do

- **Hide which endpoint you're communicating with** (within reasonable approximation).
- **Defeat naive DPI signatures** (obfs4, meek, Snowflake shape traffic to look like TLS-to-a-CDN).
- **Provide geographic diversity** at the network layer — a user in PRC appears to be connecting from the Tor exit's country.
- **Interoperate with existing protocol operation** — Tor + obfs4 has been carrying real crypto-protocol traffic for years.

### 3.2 What it CANNOT do

- **Reliably defeat the Great Firewall.** PRC has identified, blocked, and continues to block most Tor bridges as they're deployed. obfs4 requires out-of-band bridge distribution which is increasingly monitored. Snowflake relies on volunteer proxies in uncensored jurisdictions but has non-trivial detection surface. **Users in PRC typically use VPN (paid commercial + free-tier) + V2Ray + Trojan + Shadowsocks combinations, not Tor.**
- **Hide crypto-traffic patterns from ML-equipped adversaries.** State-level adversaries can build classifiers against Tor+obfs4-carried protocol traffic. This works most of the time.
- **Address latency-sensitive protocols.** Tor adds 200-800ms RTT. PRSM Tier A/B/C verification models include latency assumptions that degrade under Tor routing.
- **Protect against post-hoc identification.** If a user is identified through a KYC'd exchange breach or social-media deanonymization, transport-layer protection provides zero retroactive defense.
- **Provide content-level protection.** A user serving politically-sensitive AI-generated content through Tor is still prosecutable for the content; Tor only obscures *where they sent it from*, not *what they sent*.
- **Substitute for self-custody operational security.** A user who loses a seed phrase to a phishing attack had transport-level anonymity that didn't matter.

### 3.3 Transport-primitive catalog (R9 as "transport-pluggable" means supporting all of these, not picking one)

| Transport | Description | China-effective as of 2026 | Latency overhead |
|-----------|-------------|------------------------------|-------------------|
| **Tor + obfs4** | Classical Tor with obfs4 pluggable transport | Low (heavily blocked) | 200-800ms |
| **Tor + Snowflake** | WebRTC-proxied Tor bridges from volunteer browser-running relays | Moderate (intermittent) | 300-1000ms |
| **Tor + meek** | Domain-fronted Tor over CDN TLS | Low (CDN cooperation withdrawn) | 300-500ms |
| **V2Ray / VMess / VLESS** | Commercial circumvention protocols with regular signature updates | Moderate-to-high (active cat-and-mouse) | 100-300ms |
| **Trojan** | HTTPS-like traffic obfuscation via shared-key TLS handshake | Moderate (active blocking periodically) | 50-200ms |
| **Shadowsocks / SSR** | Simple SOCKS over custom crypto; ubiquitous in China | High near-term, trending lower as detection improves | 50-150ms |
| **XRay** | Fork of V2Ray; faster signature updates | Moderate-to-high | 50-200ms |
| **Hysteria / Tuic** | QUIC-based protocols for high-latency/lossy links | High-but-narrow (niche but works) | 50-150ms |
| **Snowflake standalone** | WebRTC ICE-based NAT-traversal (not Tor-specific) | Moderate (future roadmap) | 100-400ms |
| **Direct** | No circumvention; PRSM default | Zero in-jurisdiction; full outside | 0 |

**Important**: R9 does not pick winners. The architectural principle is that users select their transport based on their jurisdictional threat model. The Foundation ships a neutral mechanism.

---

## 4. Architectural principle: pluggable transports, not hard dependencies

### 4.1 Why Tor as a fundamental dependency would be wrong

If PRSM required Tor routing for all operation:

- **95% of users (in uncensored jurisdictions) take a 200-800ms latency hit for zero benefit.** Tier A receipt-only latency targets would be violated.
- **PRSM becomes dependent on a single external foundation (The Tor Project) for protocol operation.** Upstream governance changes, funding disruption, or regional blocking disrupts PRSM.
- **Tor itself is illegal or restricted in several jurisdictions** (PRC, Russia, Iran, UAE depending on specific interpretation). Making Tor mandatory reduces PRSM's reachable user base rather than expanding it.
- **It signals that PRSM is primarily a circumvention tool**, not neutral infrastructure. This crosses the Foundation's mandate boundary (§8).

### 4.2 Why transport-pluggability is right

- **Users in uncensored jurisdictions** get native low-latency operation.
- **Users in censored jurisdictions** can configure their nodes to route through whichever transport works best for their specific environment (which may be obfs4 today, V2Ray tomorrow, meek when CDN cooperation returns, etc.).
- **The Foundation ships the mechanism** without endorsing any particular circumvention tool, preserving neutrality.
- **Users bear responsibility for their own transport choice** and its legal implications in their jurisdiction.
- **Integration with existing Phase 6 primitives** (signed bootstrap lists, NAT traversal policy) is clean — the transport layer sits below these.

### 4.3 Where the line sits

The Foundation ships:

1. A **pluggable-transport interface** at the libp2p layer.
2. **Reference implementations** for widely-used circumvention transports (Tor with pluggable-transport support; V2Ray/Trojan protocol adapters).
3. **Jurisdictional peer-selection filters** (§6).
4. **Client-side content self-filters** (§7).
5. **Non-legal-advice framework** for documentation of jurisdictional risk (§8).

The Foundation does NOT ship:

1. Active bridge distribution for Tor (leave this to The Tor Project).
2. Curated "good VPN provider" lists (operational-level endorsement).
3. Any statement that PRSM is "safe to run" in any specific censoring jurisdiction.
4. Any Circumvention-as-a-Service offering on behalf of users.
5. Configuration presets for specific regimes (i.e., no "China Mode" that implies endorsement).

---

## 5. Transport-layer: pluggable transport interface

### 5.1 Extending Phase 6 primitives

Phase 6 Task 1 (signed bootstrap list) already supports HTTPS + DNS TXT fallback discovery. Phase 6 Task 3 (NAT traversal policy) already has a decision-table architecture.

R9 proposes a **third axis**: transport-type selection. Just as Phase 6 Task 3 lets a user say "I'm behind CGNAT, use a TURN relay," Phase 6 + R9 would let a user say "I'm in a censoring jurisdiction, use Tor+obfs4 for bootstrap and V2Ray for peer-to-peer."

Operationally, this means extending `prsm/node/transport.py` (libp2p connection factory) with an orderable list of transport adapters:

```python
TRANSPORT_PREFERENCES = [
    TransportType.TCP_DIRECT,          # default
    TransportType.TOR_OBFS4,           # user configured
    TransportType.V2RAY_VMESS,         # user configured
    TransportType.SHADOWSOCKS,         # user configured
    TransportType.FALLBACK_FAIL_CLOSED,  # refuse to connect without censorship-resistant transport
]
```

Each transport is a separate adapter that implements:

- `connect(peer_multiaddr) -> Connection`
- `listen() -> Server`
- Transport-specific configuration (bridge lines for Tor, server addresses for V2Ray, etc.)

The node operator's configuration file selects which transports to enable; the protocol itself is transport-agnostic.

### 5.2 Failure modes and fallback behavior

Key decisions:

- **Fail-closed mode** — if a user's configured transport fails, should the node stop operating or fall back to the next transport in the preferences list? This is a user decision; default should be fail-closed for users who explicitly enabled censorship-resistance.
- **Transport handshake timing** — some transports (Tor) have significant connection setup cost. Connection pooling + reuse becomes critical.
- **Peer-discovery transport** — the signed bootstrap list (Phase 6 Task 1) must itself be accessible over the user's configured transport. This means the DNS TXT fallback needs to support Tor + V2Ray resolution too, or the HTTPS primary must be fronted through the same transport.
- **FTNS payment flows** — if the FTNS settlement layer runs over web3 RPC (which it does), the RPC endpoints must also be reachable through the user's configured transport. Operator-configured "RPC over Tor" or "RPC through circumvention-aware proxy" is a first-class concern.

### 5.3 Reference integrations (not comprehensive — user can add adapters)

- **Tor** via `stem` (Python Tor controller library) + system `tor` daemon + user-provided bridge lines.
- **V2Ray / Trojan** via the user's local V2Ray client as SOCKS proxy; PRSM node connects through SOCKS.
- **Shadowsocks** via the user's local `ss-local` as SOCKS proxy.
- **Snowflake** (standalone, not Tor-specific) via `snowflake-client` in proxy mode.

Reference implementations are non-exhaustive. The pluggable transport interface is the contract; operators can ship adapters for any transport they care about.

---

## 6. Jurisdictional peer-selection filters

### 6.1 Motivation

A user in PRC may prefer their PRSM traffic to not transit any PRSM node identified as within PRC — both to avoid legal-process compelled-disclosure risk on the domestic infrastructure and because traffic staying domestic signals a lower threat model than the user is actually operating under.

### 6.2 Mechanism

Extend Phase 6 Task 3 (NAT traversal policy) with a **peer-jurisdiction filter**:

```python
class PeerJurisdictionFilter:
    excluded_jurisdictions: Set[str]  # ISO 3166-1 alpha-2 codes
    required_jurisdictions: Optional[Set[str]]  # if set, ONLY connect to these
    policy: Literal["strict", "soft-preference"]  # fail-closed vs best-effort
```

**Data source**: GeoIP lookup on peer IP addresses. Not perfect — users on VPNs appear at the VPN exit country, which may or may not be what the filter wants. The filter is a heuristic, not a guarantee.

### 6.3 Integration with PRSM-SUPPLY-1

The supply-diversity standard (PRSM-SUPPLY-1) already publishes per-jurisdiction node concentration metrics. R9's peer-selection filter can consume SUPPLY-1's metrics to:

- Warn users configuring overly-restrictive jurisdiction filters that the filter may result in severely limited peer availability.
- Surface jurisdictional distribution as a factor in peer-selection decisions.

### 6.4 Limits

- **Does not protect against traffic analysis at the adversary's own infrastructure.** If you're in PRC and all your traffic exits to Tor bridges in Europe, the PRC adversary sees "you use Tor a lot" even if they can't decrypt it.
- **Does not protect against legal-process compelled disclosure from non-domestic peers.** If your traffic routes through a peer in a jurisdiction that has an MLAT with PRC, that peer is still subject to compelled disclosure.
- **Does not provide plausible deniability** that you're running a PRSM node — it only shapes which peers you connect to.

---

## 7. Content-liability self-filter

### 7.1 Motivation

In jurisdictions where operators are legally responsible for content served by their node, the operator needs a way to ensure their node does NOT serve specific content categories. A simple defensive self-filter:

- "I will not serve any model flagged as 'politically-sensitive' under category tag X."
- "I will not run inference requests containing substrings matching pattern Y."
- "I will not cache any content with CID in my local blacklist."

### 7.2 Mechanism

Client-side. The node's own configuration; no network-layer coordination. The node's inference-orchestrator (Phase 2 remote-compute) consults the filter before accepting a dispatch request:

```python
class ContentSelfFilter:
    blocked_content_ids: Set[str]  # explicit CID blocklist
    blocked_model_tags: Set[str]    # tags from ModelCatalog
    blocked_input_patterns: List[re.Pattern]  # regex on prompt content
    action_on_match: Literal["refuse", "log_and_refuse", "silent_refuse"]
```

### 7.3 Why this is defensive, not censorship

**The filter protects the operator, not the content.** Other operators elsewhere can still serve what this operator refuses. The filter is a one-way decision by the individual node operator, not a network-level action.

The filter is also **not a substitute for legal compliance** — an operator who runs PRSM in a jurisdiction with strict liability rules still bears full legal responsibility for what their node serves. The filter is a tool for *implementing* compliance, not a substitute for it.

### 7.4 What PRSM does NOT do

- **Does not ship preset filters tied to specific jurisdictions.** "Preset for PRC" would imply endorsement of PRC's content rules, which is not the Foundation's role.
- **Does not maintain a central content-classification service.** Operators define their own filters based on their own compliance analysis.
- **Does not have a Foundation-level takedown mechanism.** Operators can refuse to serve content; the content itself stays on the network as long as any operator serves it.

---

## 8. Foundation policy boundaries (PRSM-GOV-1 §4.3 integration)

### 8.1 What the Foundation commits to

- **Ship mechanism neutrally.** The pluggable transport interface, jurisdictional peer filters, and content self-filter are published as protocol features with no jurisdiction-specific preset bias.
- **Document risk honestly.** Documentation identifies what the mechanisms can and cannot protect against. Explicit "PRSM is not safe to operate in PRC / Russia / Iran / similar jurisdictions" language where relevant, stated plainly rather than obscured.
- **Not take operational circumvention advocacy positions.** The Foundation speaks to mechanism design, not to any particular adversary's suppression tactics.

### 8.2 What the Foundation specifically does NOT do

Enumerated explicitly to avoid future governance drift:

1. **No Foundation-operated Tor bridges.** The Foundation does not run censorship-circumvention infrastructure on behalf of users.
2. **No circulation of specific-adversary bridge lists.** The Foundation does not maintain "current working bridges for PRC" documentation or similar.
3. **No "safe mode" presets keyed to specific regimes.** No "Safe Mode: China" configuration template that implies the combination is vetted as actually safe.
4. **No legal defense fund or similar.** The Foundation does not promise legal support to users prosecuted in hostile jurisdictions. This is beyond its mandate and funding level.
5. **No endorsement of specific VPN / proxy providers.** The Foundation does not recommend or warn-against commercial circumvention services.
6. **No participation in operational circumvention-advocacy campaigns.** The Foundation does not sign petitions, join coalitions, or publicly advocate for specific legal reforms in specific countries.

### 8.3 Why these limits matter

The Foundation's constitutional mission per PRSM-GOV-1 is neutral technical standard-setting + infrastructure governance. Operational activism is a different mission; the Foundation cannot pursue both simultaneously without either (a) diluting the technical mission or (b) becoming a target of the jurisdictions it purports to help users evade.

Organizations that have successfully pursued operational censorship-circumvention missions (The Tor Project, OONI, Access Now) have done so as purpose-specific non-profits with dedicated funding, legal defense capacity, and explicit advocacy positioning. PRSM Foundation is none of those.

**Delegation path**: users, advocacy groups, and mission-specific non-profits are welcome to operate circumvention infrastructure that uses PRSM's protocol features. The Foundation does not impede; it simply does not lead.

---

## 9. Implementation path (if R9 is prioritized for execution)

R9 as drafted is scoping-only. Execution would require these concrete work items, approximately ordered:

### 9.1 Phase 6.2 — Transport pluggability

- **Task 1.** Define `TransportAdapter` interface at `prsm/node/transport.py`.
- **Task 2.** Implement reference adapters: SOCKS (for Tor/V2Ray/Shadowsocks proxy-through) and direct-TCP (default).
- **Task 3.** Bootstrap-list discovery over alternate transports — extend Phase 6 Task 1's HTTPS+DNS to route through configured transport.
- **Task 4.** FTNS RPC endpoint configurability — allow operator to route web3 RPC through configured transport.
- **Effort estimate:** 3-4 engineering weeks.

### 9.2 Phase 6.3 — Peer-jurisdiction filter

- **Task 1.** `PeerJurisdictionFilter` implementation in `prsm/node/` or similar.
- **Task 2.** GeoIP database integration (MaxMind GeoLite2 as reference; updateable).
- **Task 3.** Integration with SUPPLY-1's jurisdiction-concentration metrics.
- **Effort estimate:** 1-2 engineering weeks.

### 9.3 Phase 6.4 — Content self-filter

- **Task 1.** `ContentSelfFilter` in the inference-orchestrator path.
- **Task 2.** Integration with Model Catalog tagging + CID blocklist.
- **Task 3.** Operator-config documentation (without jurisdiction-specific presets).
- **Effort estimate:** 2 engineering weeks.

### 9.4 PRSM-POLICY-JURISDICTION-1

New governance-policy document:

- Codifies §8.1-§8.3 (Foundation commits + does-not-do) as formal PRSM-GOV-1 subsidiary policy.
- Referenced from PRSM-GOV-1 §4.3 (express limits).
- Subject to board ratification when Foundation forms.
- **Effort estimate:** docs-only, ~1-2 weeks of drafting + legal review.

### 9.5 User documentation

- Non-legal-advice framework for jurisdictional risk.
- Configuration examples for common threat models (explicitly without preset endorsement).
- Cross-reference to `OPERATOR_GUIDE.md` expansion on operator legal responsibility.
- **Effort estimate:** 1 week docs.

**Total execution effort: ~8-10 engineering weeks + 2-3 docs weeks.**

---

## 10. Composition with existing research tracks

### 10.1 R3 (threat model)

R3's operator-level activation-inversion threat model is a strict subset of R9's jurisdictional adversary model. R9 extends R3 from "malicious node co-operator" to "state-level adversary with legal+technical capabilities that R3 did not scope." Both documents should coexist; R9 references R3 rather than replacing it.

### 10.2 R8 (defense-stack composition)

R8 scoped the weight-exfiltration defense stack. R9 addresses a separate sub-threat in the same overall threat catalog: operator prosecution. R8's composition matrix could be extended with a "jurisdictional threats" row, cross-referencing R9. This is a low-cost addition post-R8 merge-ready.

### 10.3 PRSM-GOV-1

R9 does not modify the charter directly but proposes PRSM-POLICY-JURISDICTION-1 as a subsidiary policy document referenced from GOV-1 §4.3. This should be drafted in parallel with R9's execution (per §9.4).

### 10.4 PRSM-SUPPLY-1

SUPPLY-1's geographic-concentration metrics feed R9's peer-jurisdiction filter (§6.3). SUPPLY-1's diversity-bonus mechanism may interact with jurisdictional filtering in ways that deserve separate analysis — e.g., does a node that refuses to connect to PRC-located peers still qualify for diversity bonuses? The conservative answer is yes (node is still contributing to supply diversity globally), but this needs explicit statement in SUPPLY-1 v0.2.

---

## 11. Open questions

These are NOT solved in R9-SCOPING-1 and should be addressed during execution:

- **Q1: Bootstrap-list transport fallback.** If a user in PRC boots a fresh PRSM node, how do they reach the bootstrap list in the first place before they know which transport works? Chicken-and-egg. Possible answers: pre-shipped hardcoded bridges (fragile); user-provided seed bootstrap file (requires out-of-band delivery); fallback to a well-known CDN with short TTL. No clean answer.
- **Q2: Transport-attack signature propagation.** If a transport's signature is deployment-blocked in one jurisdiction this week, how does the rest of the network know? Do we build a blocklist of known-compromised transports, or leave it to users? Leaning toward user discretion + community documentation.
- **Q3: Audit integration for circumvention-mode operation.** When an audit firm reviews PRSM, do they audit transport-pluggability? Does the Foundation need to certify that specific transport adapters don't exfiltrate user data? Probably yes but mechanism unclear.
- **Q4: FTNS flows through circumvention.** If a user's FTNS transactions route through a third-party circumvention service, they're trusting that service not to correlate their on-chain identity with their IP. What's the right operator advice here?
- **Q5: Fail-closed semantics when user's configured transport is down.** If Tor is blocked today and the user configured Tor-only, does the node (a) stop serving and lose income, (b) fall back to direct and blow their threat model, or (c) queue up work for Tor to come back? Defaults vary by user's threat model; needs explicit configuration choice.
- **Q6: Cross-protocol composition.** If R2 MPC is later deployed on PRSM, and the MPC requires higher bandwidth than any sufficient circumvention transport can deliver, MPC-tier use becomes infeasible in censoring jurisdictions. Hierarchy of tier-availability by jurisdictional risk needs explicit characterization.
- **Q7: Foundation legal jurisdiction.** Per the Foundation jurisdiction scoping doc, Cayman is primary. Does Cayman have any specific legal exposure to a hostile jurisdiction pursuing PRSM users? Counsel should review this as part of formation engagement.

---

## 12. Promotion triggers

R9 moves from scoping to funded execution when at least one fires:

- **T1 — User demand signal.** ≥10 distinct PRSM node-operator requests from censored-jurisdiction users for transport-pluggability, with ≥3 of those from active Phase 2 remote-compute participants.
- **T2 — Legal counsel directive.** Foundation-formation counsel identifies transport-pluggability as necessary for Foundation liability reduction (this is plausible if counsel interprets "no operational support for censorship" as including "you should have mechanism to not serve certain jurisdictions").
- **T3 — Partnership signal.** A mission-aligned non-profit (Tor Project, OpenObservatory, Access Now) approaches the Foundation about a technical partnership around transport integration.
- **T4 — Jurisdictional enforcement event.** A user in PRC, Russia, Iran, or similar jurisdiction is prosecuted for PRSM participation. Even if nobody has been yet, this trigger would escalate R9 to critical priority.
- **T5 — Phase 6 Task 3 (NAT traversal) natural completion.** Phase 6.3/6.4 naturally extend the NAT traversal policy work — if Phase 6 Task 3 is being extended anyway for NAT, adding transport-pluggability at the same time is cheaper than retrofitting later.

**Earliest realistic execution:** Q4 2026 (post-Foundation-formation + post-Phase-6-Task-3). Latest reasonable: Q2 2027 (before T4 becomes a real risk with scaled adoption).

---

## 13. Budget estimate

Per §9, total execution is ~8-10 engineering weeks + 2-3 docs weeks.

Engineering-only equivalent cost: ~$30-50K if Foundation staff, or $80-120K if external contractors at senior-protocol rates. No partner-research-group engagement required — this is applied integration, not primary research.

---

## 14. What this doc does NOT do

- **Does NOT commit PRSM to execution of any proposed work item.** This is scoping; prioritization is a Foundation board decision once formed.
- **Does NOT provide legal advice to users** in any jurisdiction. Every proposed mechanism has legal implications that vary by user location and role; users must consult local counsel.
- **Does NOT promise circumvention capability.** The Great Firewall is an active adversary that wins against protocol-level defenses on average. R9 reduces attack surface; it does not eliminate it.
- **Does NOT speak for the Foundation.** PRSM Foundation did not exist at drafting time. Post-formation, the board may amend or reject this scoping entirely.
- **Does NOT coordinate with The Tor Project, Access Now, or similar.** Those are independent organizations; R9 references their work but does not speak on their behalf.

---

## 15. Related documentation

- `docs/2026-04-22-r3-threat-model.md` — R3 (operator-level threat model; R9 generalizes).
- `docs/2026-04-22-r8-defense-stack-composition.md` — R8 (weight-exfiltration defense composition; R9 parallel sub-threat).
- `docs/2026-04-22-phase6-p2p-hardening-design-plan.md` — Phase 6 P2P hardening (integration target).
- `docs/2026-04-22-phase6-task4-dht-tuning-plan.md` — Phase 6 Task 4 tuning plan (adjacent work).
- `docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md` §4.3 — Foundation express limits (§8 constraint source).
- `docs/2026-04-22-prsm-supply-1-supply-diversity-standard.md` — SUPPLY-1 geographic diversity (R9 peer-filter integration).
- `docs/2026-04-14-phase4plus-research-track.md` — R-track master index (add R9 entry).
- `PRSM_Vision.md` §6-7 — Honest caveats framing R9's user-risk posture.

---

## 16. Research provenance + limitations

Research underlying R9 was conducted 2026-04-23. Primary source categories:

- **Current Great-Firewall effectiveness against Tor + pluggable transports** — OONI Probe reports, Tor Project blog posts on bridge-distribution effectiveness, academic publications (NDSS / USENIX Security) on transport-signature detection.
- **PRC legal regime for generative AI** — PRC Cyberspace Administration (CAC) Interim Measures for Generative AI Services (2023, amended 2025); State Council AI Regulation framework (draft 2024, effective 2025).
- **Comparable protocol approaches to transport-pluggability** — Filecoin + IPFS libp2p transport layer, Ethereum P2P stack, Tor's own pluggable-transport specification (PT 3.0).
- **Existing foundation-boundary practice** — Ethereum Foundation, Filecoin Foundation, Web3 Foundation public policy statements on jurisdictional risk.

**Flagged limitations:**

- Great Firewall effectiveness is a moving target; any specific claim about transport viability in this doc should be reconfirmed at execution time.
- PRC AI regulation is evolving rapidly; the 2023 Interim Measures have already been supplemented by multiple subsequent rulings.
- This doc does not address EU AI Act provisions for protocol operators — a parallel jurisdictional analysis for EU is out of R9 scope but would be a natural follow-on (R9.1 "Extended jurisdictional analysis").
- Russian and Iranian jurisdictional regimes have their own evolutions that mirror PRC only partially; R9 treats them as "similar class of threat" without individual characterization.

**No authoritative legal claim is made in any section of this document.** All legal context is drawn from publicly-available regulatory text and secondary reporting. Any user acting on R9's mechanism or guidance consults their own counsel.
