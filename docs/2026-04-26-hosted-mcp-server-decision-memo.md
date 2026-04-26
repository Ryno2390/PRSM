# Hosted MCP Server Decision Memo

**Document identifier:** PRSM-HOSTED-MCP-DECISION-1
**Date:** 2026-04-26
**Status:** Decision recorded; **defer 12+ months post-mainnet** with named re-evaluation triggers
**Decision authority:** Founder (pre-board); Foundation council post-formation
**Closes:** Phase 3.x.1 Task 12 per `docs/2026-04-26-phase3.x.1-mcp-server-completion-design-plan.md`
**Companion docs:**
- `docs/2026-04-22-phase3-status-and-forward-plan.md` §3.1 — original framing of the question
- `docs/2026-04-22-phase5-fiat-onramp-design-plan.md` — Phase 5 KYC framework this depends on
- `2026-04-26-gate3-counsel-briefing-memo.md` (private repo) Q-OPS-3 — MSB analysis

---

## 1. Executive Summary

**The question:** Should the PRSM Foundation operate a hosted "PRSM-as-a-service" MCP server — a public HTTPS endpoint that MCP clients (Claude Desktop, Cursor, etc.) can connect to without users running their own PRSM node?

**Recommendation: Defer indefinitely with explicit re-evaluation triggers.**

The hosted-MCP path is not blocked by engineering. It is blocked by:
- **Regulatory inversion** — Foundation custody of user FTNS makes Foundation a Money Services Business candidate, requires Phase 5 KYC framework to be operational first, and triggers state-by-state money transmitter analysis
- **Strategic incoherence** — a Foundation-run hosted service is itself a centralized AI provider, undermining the structural-alternative thesis from Vision §1 that distinguishes PRSM from Anthropic / OpenAI / Google
- **Operational unreadiness** — infrastructure, 24/7 ops, customer support, and SLA commitments require a budget the Foundation does not yet have
- **Demand absence** — no MCP-client-ecosystem partner has requested it; current signals indicate self-hosted is acceptable for the Python-comfortable developer audience

**Re-evaluation triggers (any one):**
1. **Self-hosted MCP usage** reaches ≥1,000 monthly active users AND ≥30% report (in survey or support-channel signals) that they would prefer hosted
2. **MCP-client-ecosystem partner** (Anthropic Claude Desktop team, Cursor, Continue, etc.) explicitly requests hosted PRSM as integration prerequisite
3. **Phase 5 KYC framework** is operational and proven, with retained counsel attesting that Foundation custody of user FTNS for hosted MCP billing falls within the existing framework rather than requiring separate analysis
4. **Foundation operational budget** has ≥$1M annually allocated specifically for hosted-service operations, distinct from protocol-stewardship operations

**24-month sunset:** if none of the four triggers hit by 2028-04-26, the question is closed permanently — hosted MCP is not part of PRSM's product.

The rest of this memo is the substantive analysis behind that recommendation.

---

## 2. The Question, Made Concrete

### 2.1 What "self-hosted MCP" looks like (current state)

```
User installs Python package locally:
  pip install prsm-network

User starts a PRSM node:
  prsm node start

User starts the MCP server:
  prsm mcp-server

MCP client connects via stdio to the local MCP server.
User's own node holds their FTNS, signs transactions, makes settlements.
Foundation provides the bootstrap node + protocol; never custodies user funds.
```

Friction points:
- Python installation requirement (excludes users who don't already have Python)
- Local-machine-on requirement (laptop must be running for the MCP tools to be available)
- One-time identity setup (`prsm node start` generates Ed25519 keypair, FTNS welcome grant)
- Self-managed key custody (lost laptop = lost node identity)

### 2.2 What "hosted MCP" would look like

```
User signs up at prsm-network.com:
  Email + password → account
  Optional: KYC for fiat on-ramp
  Foundation creates a custodial wallet with FTNS balance

User configures MCP client:
  Endpoint: https://api.prsm-network.com/mcp/v1
  API key: their account token

MCP client connects to Foundation-hosted MCP server.
Foundation runs PRSM nodes that serve as the user's "node."
Foundation custodies the user's FTNS for billing.
User pays in fiat (Stripe / Coinbase Commerce); Foundation converts to FTNS.
```

Ergonomic improvement: Python install, local node, and key custody all become Foundation problems.

### 2.3 What gets traded for that ergonomic improvement

The trade is structural, not just operational. The next four sections quantify it.

---

## 3. Regulatory Considerations

### 3.1 Money Services Business classification (FinCEN)

Self-hosted MCP: Foundation never touches user FTNS. Each user's FTNS lives in their own non-custodial wallet. Foundation's role is protocol stewardship + bootstrap node operation. **Not an MSB candidate.**

Hosted MCP: Foundation custodies user FTNS in pooled or per-user wallets. User deposits fiat (Stripe), Foundation converts to FTNS, holds it for the user, spends it on the user's behalf for inference.

This is a money-transmission pattern. Per Tokenomics §9.3 + Q-OPS-3 (Gate 3 counsel briefing), MSB classification requires:
- Federal: FinCEN registration, AML/KYC program, BSA compliance, suspicious activity reporting
- State: 50-state mosaic of money transmitter licenses (~$500K legal cost, 12-24 month timeline)
- Foundation jurisdiction matters but does not eliminate US-side exposure for US users

The Tokenomics §9.3 mitigation strategy (third-party fiat handlers, decentralized swaps, foundation entity offshore) was specifically designed around the **non-custodial** architecture. Hosted MCP undoes that mitigation by making the Foundation directly custodial.

### 3.2 KYC / AML obligations

Self-hosted MCP: KYC is voluntary, controlled by the user's choice of fiat on-ramp (Coinbase Commerce, etc. — third party owns the KYC).

Hosted MCP: Foundation's account-creation flow becomes the KYC gate. Foundation must implement:
- Customer identification (CIP)
- Customer due diligence (CDD)
- Enhanced due diligence (EDD) for high-risk accounts
- Sanctions screening (OFAC, EU lists, etc.)
- Transaction monitoring + suspicious activity reporting (SAR)
- Record retention for 5+ years

Operational cost of full KYC/AML: $500K-$2M/year at moderate scale, scaling with account count.

### 3.3 Phase 5 dependency

`docs/2026-04-22-phase5-fiat-onramp-design-plan.md` (Phase 5: Fiat On-Ramp & KYC) is targeted Q1 2027. Phase 5 lands the KYC framework that hosted MCP would need.

**Hosted MCP launched before Phase 5 ships forces a parallel Phase 5 analysis:** without Phase 5's vetted KYC framework, Foundation must build hosted-MCP KYC from scratch under time pressure. This is exactly the failure mode Phase 5's design-first approach exists to prevent.

**Hosted MCP launched after Phase 5 ships still requires** a counsel determination (Q-NPF-2 / Q-OPS-3) that Phase 5's framework covers MCP billing custody, not just the standard fiat-on-ramp flow. That determination is non-trivial.

### 3.4 Verdict on regulatory

Hosted MCP without Phase 5 operational = high-risk, untested regulatory posture.
Hosted MCP with Phase 5 operational = adds 6-12 months of counsel work on top of a Phase 5 launch that already consumed counsel bandwidth.

**Earliest realistic launch window for hosted MCP, conditional on Phase 5 success: Q3 2027 minimum.** That is 12+ months from this memo regardless of any other consideration.

---

## 4. Operational Considerations

### 4.1 Infrastructure requirements

Self-hosted MCP: 1 bootstrap node ($20/month DigitalOcean droplet). Each user's MCP server runs on the user's own hardware.

Hosted MCP: PRSM node fleet operating as users' nodes:
- ≥3 high-availability application servers behind load balancer
- Multi-region deployment (NA + EU + APAC)
- Database (account-state, billing, audit logs)
- RPC infrastructure (Base mainnet RPC at scale — likely Alchemy/QuickNode enterprise tier)
- Object storage (audit logs, attestation artifacts)
- Monitoring + alerting (Datadog / Grafana)
- DDoS protection (Cloudflare enterprise)
- Customer-facing dashboard
- Support ticketing system

**Estimated infrastructure cost:** $5K-$30K/month at modest scale (1K-10K monthly active users), scaling with usage. At 100K MAU: $50K-$200K/month.

### 4.2 Staffing requirements

- **Site Reliability** — 24/7 on-call rotation. Minimum 2 SRE-class engineers; realistic for 24/7 coverage = 4 SRE-class engineers
- **Customer Support** — billing questions, account recovery, dispute resolution. Minimum 1 FTE per ~500 daily active users
- **Compliance / KYC operations** — 1+ FTE handling SAR filings, sanctions screening, CDD/EDD reviews
- **Engineering** — separate from protocol engineering; hosted-service maintenance, feature development, security response

Total estimated FTE: 7-12 people. At foundation salary norms (~$200K loaded cost), that is **$1.4M-$2.4M/year in personnel alone.**

### 4.3 SLA commitments

Self-hosted MCP: no SLA. Users run on their own hardware, accept their own uptime.

Hosted MCP: customer-facing SLA expectations (especially if charging for the service):
- Uptime: 99.9% typical (8.7 hours/year max downtime)
- Latency: p95 inference latency budget (<2s for `prsm_quote`, <30s for `prsm_analyze`)
- Support response time: <24h for paid users, faster for higher tiers
- Incident communication: status page, post-mortems within 7 days

Failing SLAs creates legal exposure (refunds, contractual claims) and reputational exposure (publicly-tracked downtime).

### 4.4 Support burden

Per industry benchmarks (Stripe / Coinbase / equivalent): hosted SaaS services field 0.5-2 support tickets per active user per year. At 10K MAU: 5K-20K tickets/year. At 100K MAU: 50K-200K.

PRSM is currently solo-maintained. The leap from "solo" to "supports a hosted service" is not incremental — it requires building the customer-support stack from scratch.

### 4.5 Verdict on operational

Hosted MCP requires a budget and team structure the Foundation does not have and will not have within 12 months.

---

## 5. Security Considerations

### 5.1 Honeypot creation

Self-hosted MCP: each user's FTNS is dispersed across thousands of independent non-custodial wallets. An attacker compromising one user gets one user's funds.

Hosted MCP: Foundation custody pools user FTNS in foundation-controlled wallets. An attacker compromising the Foundation's hot wallet gets every user's funds simultaneously. This is the **MtGox / Coinbase / FTX failure mode** — pooled custody is a structural target.

PRSM's existing hot-wallet infrastructure (3-of-5 multi-sig council) is appropriate for **operational treasury** scale. Scaling it to handle thousands of users' active operating balances introduces new failure modes (per-user balance accounting, internal transfers, withdrawal latency, account-recovery flows).

### 5.2 Hosted server as attack surface

Self-hosted MCP: each MCP server is a process on a user's local machine. Compromise = compromise of one user's machine.

Hosted MCP: the public HTTPS endpoint is a single attack surface for thousands of users. This is the highest-value target on the entire PRSM network, including the smart contracts.

Compare to Risk Register entries:
- A1 RoyaltyDistributor exploit (Critical, atomic split, contract-enforced — limited blast radius)
- A4 FTNSToken contract exploit (Critical, supply integrity)
- **Hosted MCP exploit would be a new entry, severity Critical, category D operational rather than A on-chain** — a new failure mode added to the system

### 5.3 User-key vs. user-account compromise

Self-hosted MCP: user lost their hardware wallet → user lost their FTNS. Foundation has no role.

Hosted MCP: user account compromised (phishing, weak password, credential reuse) → Foundation has lost user's FTNS, even though the protocol is intact. **Foundation gets blamed for a compromise that happened outside PRSM's protocol layer.**

This pattern has played out repeatedly in centralized exchanges. Coinbase's reputation suffers from individual account compromises despite Coinbase itself being uncompromised. PRSM does not need that pattern.

### 5.4 Verdict on security

Hosted MCP creates failure modes that don't exist in self-hosted PRSM. Some mitigations exist (hardware HSM custody, withdrawal limits, etc.) but each one adds operational complexity without restoring the structural simplicity of non-custodial architecture.

---

## 6. Strategic Considerations

### 6.1 Alignment with Vision §1 thesis

PRSM_Vision.md §1 frames the project's existence as a structural alternative to centralized AI providers:

> "PRSM is a structural alternative — a way to build, deploy, and benefit from AI infrastructure that is fundamentally aligned with the democratic process rather than fundamentally hostile to it... no single entity can decide who gets to participate."

Foundation running a hosted MCP service makes the Foundation itself a centralized provider of MCP access. The Foundation can then:
- Decide who gets accounts (KYC + ToS gating)
- Decide which users to deplatform (regulatory pressure, compliance demands)
- Decide which queries to allow (content policy)
- Hold user funds (custodial)

These are the exact decision-points PRSM was built to NOT have. A hosted service inverts the architecture.

### 6.2 Forkability and decentralization

Vision §1 explicitly: *"The protocol is open and forkable. If PRSM itself ever becomes captured or misaligned, the network can fork, the data persists, and the participants retain agency."*

Self-hosted MCP preserves forkability — anyone can run the protocol; the Foundation is one node operator among many.

Hosted MCP centralizes a service tier that users come to depend on. If the Foundation goes down (regulatory action, financial difficulty, mission drift), users on the hosted tier lose their access to their funds and their inference capability **even though the underlying protocol is intact**.

This is the substance of the "single point of failure" critique applied to PRSM itself.

### 6.3 Tokenomics §10 invariants

Tokenomics §10 lists 8 inviolable invariants. Hosted MCP raises questions about:

> "FTNS is distributed only as compensation, never sold by the foundation."

Hosted MCP requires the Foundation to either:
- (a) Buy FTNS on the secondary market to credit user accounts after fiat deposit — this is Foundation FTNS purchasing, which sits adjacent to "never sold by the foundation"
- (b) Hold user-deposited FTNS in foundation custody — Foundation becomes a holder of user-owned FTNS at scale, blurring the line between operations and custody

Counsel review (Q-OPS-3 in Gate 3 briefing) would need to determine which path is compatible with the invariant. Both interpretations are defensible; neither is clean.

### 6.4 What MCP-client partners actually want

Vision §1 + §13 emphasize MCP integration as the canonical UX surface. Partner asks (T3 trigger from Phase 3 status doc) drive MCP server scope.

What partners (Anthropic Claude Desktop team, Cursor, Continue, Cody) actually want:
- **An npm package or one-line install** so users can add PRSM to MCP config without leaving their terminal — addressed by Phase 3.x.1 Task 9 (npm wrapper)
- **A documented `prsm_inference` tool** so the LLM can route confidential workloads — addressed by Phase 3.x.1 Tasks 6-7
- **Streaming responses** so the LLM can show progress — addressed by Phase 3.x.1 Task 8

None of those requires hosted MCP. All are addressable by improving the self-hosted experience.

### 6.5 Verdict on strategic

Hosted MCP would compromise the structural-independence claim that makes PRSM different from centralized AI providers, in exchange for an ergonomic improvement that is mostly addressable by other means.

---

## 7. Demand Signal

Per Phase 3 status doc §3.1 promotion triggers:

| Trigger | Status as of 2026-04-26 |
|---|---|
| T1 — Phase 3 marketplace ≥20 active providers + ≥50 daily dispatches | NOT MET — pre-mainnet |
| T3 — At least one MCP-client ecosystem partner expresses concrete interest | NOT MET — outreach not started |

**T3 is the load-bearing signal.** If Anthropic Claude Desktop, Cursor, or Continue specifically asks "we'd integrate PRSM if it had a hosted endpoint," the analysis above gets a counter-pressure that may justify partial concessions.

Without T3 signal, the project is building infrastructure for a use case nobody has requested. That is the standard recipe for wasted engineering effort.

**Discord signal as of 2026-04-26:** 1 member. Insufficient sample for adoption-curve inference.

---

## 8. Counter-Arguments Considered

### 8.1 "Self-hosted is too friction-heavy for mainstream users"

True for non-Python-comfortable users. But:
- npm package wrapper (Task 9) reduces friction to `npx prsm-mcp` — single command
- Homebrew formula (Task 10) reduces to `brew install prsm/tap/prsm`
- Default-install includes auto-bootstrap and auto-FTNS-welcome-grant
- Mainstream-user adoption is a Phase 4-5 problem (consumer onboarding), not a Phase 3.x.1 problem

The right response to friction is to reduce it, not to centralize.

### 8.2 "Hosted service generates Foundation revenue via fees"

Hosted MCP fee schedule: maybe 5-10% margin on inference cost per user. At 10K MAU × $50/month average spend = $500K/month gross, $25-50K/month margin.

That margin would be consumed by:
- Infrastructure cost ($5-30K/month)
- Personnel cost ($120-200K/month for 7-12 FTE)
- Compliance cost ($50-200K/year)
- Customer support cost ($50-100K/month at this scale)

**At plausible early-MAU levels, hosted MCP is a net cost center, not a revenue source.** Even at high MAU, the margin doesn't justify the strategic compromise.

### 8.3 "Other protocols (e.g. Filecoin) run hosted services"

Filecoin operates hosted infrastructure (Web3.Storage, NFT.Storage) explicitly as Foundation-run services. Fair counterexample.

But:
- Filecoin's value proposition is storage, where centralized hosted services are common precedent (S3, etc.) — there is no structural-alternative thesis comparable to PRSM's
- Filecoin's hosted services are content-storage front-ends, not custodial-FTIL-billing services — the regulatory profile is different
- Web3.Storage launched 4+ years post-Filecoin mainnet, when operations capacity existed; not pre-mainnet

The comparison doesn't transfer cleanly.

### 8.4 "We can run a thin hosted service that doesn't custody FTNS"

Considered. A "thin" hosted MCP server that proxies user requests to user-owned wallets without custody could exist:
- User signs FTNS transactions on their device
- Hosted server provides MCP endpoint + bootstrap + connection management
- Hosted server never holds user keys or FTNS

This is the "non-custodial proxy" pattern. It's structurally sound but operationally still requires:
- Per-user authentication (account = login + their wallet address)
- Session management
- Rate limiting / abuse prevention
- 24/7 ops + monitoring

Engineering scope ~50% of full hosted MCP; strategic compromise reduced (not eliminated — Foundation still runs the SPOF endpoint); regulatory exposure reduced substantially (no MSB classification since no custody).

This option warrants a separate evaluation if the original "Foundation custody" hosted MCP is rejected. **Recommendation: include "non-custodial proxy" as Option C in the re-evaluation when triggers fire.**

---

## 9. What to Do Instead

The user-friction problem hosted MCP is meant to solve has a series of cheaper, less-strategic-cost answers:

### 9.1 Distribution polish (Tasks 9-11 in Phase 3.x.1)

`npx prsm-mcp` is 1 command. `brew install prsm/tap/prsm` is 1 command. Either eliminates the "Python install" friction for mainstream users.

**Estimated effort:** ~10-20 hours of engineering, all already designed in Phase 3.x.1.

### 9.2 First-time-experience polish (Phase 4)

Phase 4 Wallet SDK introduces email-based onboarding via embedded wallets (Privy/Web3Auth/Magic.link). This eliminates seed-phrase friction for self-hosted users.

**Estimated effort:** Already in scope as Phase 4; budget already allocated.

### 9.3 Bootstrap node geographic expansion

Single-region bootstrap (NYC3) creates first-connection latency for non-NA users. Adding Europe + APAC bootstrap nodes addresses some of the friction hosted-MCP would solve, without strategic compromise.

**Estimated effort:** ~3-4 hours engineering, ~$40/month operational cost.

### 9.4 Better error messages + onboarding wizard

Most "self-hosted is too hard" complaints reduce to specific friction points (port forwarding, RPC config, wallet creation). Each can be addressed by better in-CLI guidance.

**Estimated effort:** Ongoing, distributed across many small PRs.

---

## 10. Decision Record

**Decision (2026-04-26): defer hosted MCP server indefinitely.**

**Re-evaluation triggers (any one):**
1. Self-hosted MCP usage ≥1,000 MAU AND ≥30% surveyed prefer hosted
2. MCP-client ecosystem partner explicitly requests hosted PRSM as integration prerequisite
3. Phase 5 KYC framework operational + counsel-attested as covering hosted-MCP custody
4. Foundation operational budget ≥$1M/year specifically allocated for hosted-service operations

**Sunset:** if no trigger fires by 2028-04-26, hosted MCP is permanently out of scope.

**Approver:** Ryne Schultz (founder), 2026-04-26.

**Reviewer at promotion:** Foundation council post-formation; first review at quarterly post-mainnet checkpoint.

**Reversibility:** High. This decision changes nothing about the existing protocol or self-hosted MCP server. If a trigger fires, the project gains a clear scope handoff to begin hosted-MCP work without retroactive design changes.

---

## 11. Phase 3.x.1 Implications

This decision closes Phase 3.x.1 Task 12 and resolves Gap #4 from `docs/2026-04-22-phase3-status-and-forward-plan.md` §2.3:

> ⚠️ **Gap #4** — Hosted "PRSM-as-a-service" MCP server decision: deferred.

**Status update for Phase 3.x.1 Task 12:** ✅ Complete via this memo. No engineering work scoped under Phase 3.x.1 for hosted MCP server.

The remaining 4 gaps (privacy tier enforcement, FTNS billing, distribution, streaming) all proceed without dependency on hosted MCP.

---

## 12. Versioning

- **0.1 (2026-04-26):** Initial decision. Defer 12+ months with named re-evaluation triggers and 24-month sunset. To be revisited only if a trigger fires; otherwise this decision stands.
