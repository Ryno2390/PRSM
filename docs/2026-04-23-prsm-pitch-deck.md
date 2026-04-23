# PRSM — Pitch Deck

**Date:** 2026-04-23
**Draft status:** v1 for founder review → data-room upload
**Companion to:** `docs/2026-04-22-prsm-investor-executive-summary.md` (the written short-form)

---

## How to use this doc

This is a 12-slide pitch deck rendered in markdown for draft-and-iterate
before building the final visual deck. Each slide has:

- **Headline** — the one-line claim on the slide
- **Visual / content** — bullet list OR a description of the chart/diagram
- **Speaker notes** — what the founder says out loud for 30-60 seconds

Target length: 10-12 minutes spoken + 15-20 minutes Q&A. Shorter than
standard VC decks (18-20 slides) because PRSM's "why now" + "what's
shipped" are the story — not the product mechanics.

For visual execution: prefer Figma / Pitch / Keynote native over PDF
export. Each slide should be consumable in 10-15 seconds visually while
the speaker elaborates.

---

## SLIDE 1 — Title

**Headline:** PRSM — Neutral infrastructure for frontier AI.

**Visual:**
- PRSM wordmark (top-center)
- Subtitle: "A P2P protocol turning underutilized compute into a single priced mesh for AI inference, storage, and data."
- Footer: "Prismatica, Inc. — Reg D Rule 506(c) — April 2026"
- Contact: founder email + data-room link

**Speaker notes:** "PRSM is what you'd build if you'd watched the Web3 stack mature through two cycles and asked: what's the piece nobody shipped yet? It's neutral infrastructure for frontier AI. Let me show you why that matters."

---

## SLIDE 2 — The opportunity

**Headline:** Frontier AI is centralizing faster than any prior tech wave.

**Visual:** Three-column chart showing concentration:

| Layer | Incumbents | Concentration |
|-------|-----------|---------------|
| GPU supply | NVIDIA + AMD + Intel | 3 firms |
| Attestation root | NVIDIA CCA | 1 firm |
| Frontier labs publishing open weights | (effectively zero at SOTA) | 0 firms |

Below: *"The industry needs neutral infrastructure that is cryptographically credible, economically aligned, and permissionless — not another vertically-integrated stack."*

**Speaker notes:** "The centralization pattern in AI is worse than anything we saw in Web2. Three companies control GPU supply, one company controls the attestation root that verifies confidential compute, and zero open-weights labs can safely publish SOTA models because they'll be exfiltrated. This isn't a market failure that fixes itself — it's a coordination problem, and coordination problems are what protocols solve."

---

## SLIDE 3 — The solution

**Headline:** One protocol, three resource markets, one token.

**Visual:** Simplified PRSM architecture diagram

```
┌───────────────────────────────────────────────────┐
│                   LLM (any — OAuth / local / API) │
└────────┬──────────────┬──────────────┬────────────┘
         │              │              │
         ▼              ▼              ▼
    ┌─────────┐   ┌──────────┐   ┌──────────┐
    │Inference│   │ Storage  │   │   Data   │
    │  Mesh   │   │   Mesh   │   │   Mesh   │
    └────┬────┘   └─────┬────┘   └─────┬────┘
         │              │              │
         └──────────────┴──────────────┘
                        │
                ┌───────▼────────┐
                │  FTNS token    │
                │  (capped,      │
                │   halving,     │
                │   utility)     │
                └────────────────┘
```

- **Contributors earn FTNS** for sharing latent resources (consumer + T3 cloud + T4 dedicated)
- **LLMs consume PRSM via MCP tools** — reasoning happens in the LLM, execution on PRSM nodes
- **No ICO. No pre-sale. No VC token allocation.** Compensation only.

**Speaker notes:** "PRSM ties together three resource markets — compute, storage, data — under one token. The token only enters circulation through earning. That's important for regulatory reasons, but more importantly it's what makes the incentives clean. The people who get rich here are the people who contributed resources."

---

## SLIDE 4 — Why now

**Headline:** Five independent clocks all ringing in 2026.

**Visual:** Five horizontal progress bars, all at ~90%:

1. **Verification triad engineering-complete** (Tier A + B + C, all three merge-ready 2026-04)
2. **H100 Confidential Compute shipping** — baseline trust floor is here
3. **Frontier labs publicly asking** "what would let us publish weights?" — the question is out
4. **Regulatory posture matured** — MiCA + FINMA + Cayman all workable (unlike 2019)
5. **Audit engagement is the last engineering gate** — no blockers; hardware-assembly timing

**Speaker notes:** "The 'why now' for PRSM isn't one trend — it's five independent clocks all ringing at the same time. Confidential compute shipping at the silicon level, regulators giving us workable paths, frontier labs asking the question, and — on our side — the verification triad finishing. All of these had to align. They did."

---

## SLIDE 5 — How it works (the verification triad)

**Headline:** Three trust tiers. One protocol. Engineering-complete.

**Visual:** Three-column comparison:

| | **Tier A** | **Tier B** | **Tier C** |
|---|---|---|---|
| **Name** | Receipt-only | Redundant execution | Stake + slash |
| **Use case** | Open / low-value jobs | High-value jobs | Premium / critical jobs |
| **Mechanism** | Cryptographic receipts batched on-chain; challenge with slashable proofs | k-of-n providers must agree; CONSENSUS_MISMATCH slashes minority | Provider posts stake; forged receipts burn stake (70% challenger / 30% Foundation) |
| **Shipped in** | Phase 3.1 | Phase 7.1 | Phase 7 |
| **Status (2026-04)** | `phase3.1-merge-ready` | `phase7.1-merge-ready` | `phase7-merge-ready` |

Footer: *"First protocol in the space with all three verification tiers shipped in engineering."*

**Speaker notes:** "Centralized AI gives you one trust model: trust us. PRSM gives users three — pick the one that matches your job's risk tolerance. Tier A for open jobs is cheap and fast. Tier C for premium jobs has provider stake on the line. Tier B for high-value jobs has k-of-n redundancy so a majority of execution nodes would have to collude. And I want to emphasize: all three are merge-ready. Not a roadmap. Code, tests, audit-prep bundles, rehearsal scripts."

---

## SLIDE 6 — Market

**Headline:** Three markets PRSM unlocks simultaneously.

**Visual:** Three-tier TAM stack (approximate 2026 figures; cite sources in data room):

| Market | 2026 size | PRSM's angle |
|--------|----------|--------------|
| AI inference compute | ~$45B/yr | Arbitrage + T3 underutilization capture |
| Confidential-compute-verified workloads | ~$3B/yr, growing 90% YoY | First cryptographically-credible neutral mesh |
| Frontier-model weight-security (custody-equivalent) | *emerging* | CIS-1 silicon standard defines the category |

Below: *"Launch tactic: capture T3 arbitrage margin (consumer + prosumer GPU), use that to seed T4 dedicated nodes, use those to underwrite the CIS-1 weight-security market."*

**Speaker notes:** "The market question I always get is 'isn't AI compute already commoditized?' The answer is yes — for raw inference, for trusted-party jobs. But confidential-compute-verified jobs are a separate TAM that's growing 90% year-over-year and the lion's share of it ends up on AWS Nitro Enclaves, Google Confidential, Azure Confidential — because there's no neutral alternative. PRSM is the neutral alternative. And the weight-security TAM doesn't even exist yet as a market — CIS-1 defines it."

---

## SLIDE 7 — What's shipped

**Headline:** This is not a roadmap pitch.

**Visual:** Shipped-phases table (pulled from exec summary):

| Phase | Coverage | Status |
|---|---|---|
| **1.3** | On-chain provenance + royalty distribution | 7-day Sepolia bake-in passed; mainnet-ready |
| **2** | Remote compute dispatch + Ed25519 receipts + escrow + topology | v0.35.0 on PyPI |
| **2 Rings 7-10** | TEE runtime + DP noise + tensor-parallel sharding | v0.35.0 on PyPI |
| **3** | FTNS-priced marketplace + price discovery + reputation | `phase3-merge-ready` |
| **3.1** | Batched on-chain settlement (Tier A) | `phase3.1-merge-ready` |
| **7** | Tier C stake-slash | `phase7-merge-ready` |
| **7.1** | Tier B redundant execution | `phase7.1-merge-ready` |
| **7.1x** | Pre-audit hardening (6 findings resolved pre-audit) | `phase7.1x-audit-prep-20260422-2` |

Footer: **427 tests green** (142 Solidity + 283 Python unit + 2 E2E) — all on `main` at `github.com/Ryno2390/PRSM`

**Speaker notes:** "Every row in this table is a merge-ready tag in the public repo. Every commit is signed. 427 tests pass. This isn't a deck pitch that turns into a hiring drive — the hiring already happened, the work already shipped. What's left is mainnet deployment, and that's gated on hardware assembly and Foundation formation, not engineering."

---

## SLIDE 8 — Technical moat

**Headline:** Four compounding moats. None of them are "the team will execute."

**Visual:**

1. **CIS-1 confidential-silicon standard** — RFC-style spec, 3 conformance levels (C1 / C2 / C3), quantified weight-exfiltration lower bounds 2²⁰ → 2⁴⁰ inferences. *Nobody else has specified a frontier-grade confidentiality target.*
2. **Vendor-independent attestation** — On-chain registry on Base binds chip identity to protocol governance, not to NVIDIA / single CA. Multi-manufacturer requirement at C2+ is a **prohibited amendment** — can't be removed without dissolving the Foundation.
3. **Complete economic backstop** — 3 slashable reason codes, 70/30 bounty split, consensus_group_id sybil mitigation requiring k-distinct-provider keys to attack.
4. **8-layer research track with 6 partner-handoff docs** — defense-stack composition, MPC, activation-inversion red-team, PQ-signatures watch, benchmark plan, supply-diversity standard.

**Speaker notes:** "When a competitor tries to catch up, they have to match four things that compound: a confidentiality standard, an on-chain attestation registry constitutionally bound to multi-manufacturer requirements, an economic slashing architecture that actually ships, and research track depth that has partners lined up on specific arcs. Each one individually would be a year of work. Together they're structural."

---

## SLIDE 9 — Governance + regulatory moat

**Headline:** What Web3 history taught us: write the constraints into the constitution, not the roadmap.

**Visual:** Two-column framing:

**PRSM-GOV-1 prohibited amendments (can't be changed without dissolving Foundation):**
- Foundation cannot build chips
- Foundation cannot revenue-share with Prismatica
- Founder must pick one entity after Year 3
- Foundation certifies Prismatica's chips on same terms as any other implementer
- Transparency default — every tx, vote, cert decision, financial statement public

**Regulatory posture:**
- FTNS distributed only as compensation for services rendered
- No ICO / no retail sale / no bonding curve
- Bootstrap capital via Prismatica equity (Reg D 506(c)) — a cleanly-classified security
- FTNS and equity are **separate instruments with separate treatment**

**Speaker notes:** "The real pattern-match Web3 investors do is 'who captures the rents?' I don't answer that with a promise. I answer it structurally. The Foundation cannot become Prismatica. The founder has to pick an entity after three years. Prismatica's chips go through the same certification as anyone else's. And for the regulatory lane — the token and the equity are fully separate instruments. You're buying equity today. FTNS is earned, not sold."

---

## SLIDE 10 — Business model + returns

**Headline:** Investor returns via Prismatica enterprise-value growth.

**Visual:** Three-stream EV breakdown:

| Revenue stream | Description |
|----------------|-------------|
| **Managed T4 meganodes** | Prismatica operates dedicated high-performance PRSM nodes; predictable operating revenue |
| **CIS chip licensing** | First-implementer royalties as CIS-1 standard gets adopted by additional fabs |
| **FTNS treasury appreciation** | Prismatica earns FTNS organically for resources contributed; appreciation from network usage, not speculation |

**Deal structure:**
- **Instrument:** Prismatica Inc. equity, Delaware C-corp
- **Exemption:** Reg D Rule 506(c), accredited-only
- **Use of proceeds:** CIS chip design + fab, T4 meganode deployment, ML research team, commissioned data pipeline, external audit retainer, 24-36 month runway
- **Round size / valuation / terms:** *see term sheet in data room*

**Speaker notes:** "You're not buying the token. The token is a separate protocol instrument that gets earned by whoever contributes resources. You're buying equity in Prismatica, which grows enterprise value three ways: operating revenue from managed nodes, licensing royalties as CIS-1 adopts, and organic FTNS appreciation from the treasury. Clean Delaware C-corp, Reg D 506(c). Term sheet in the data room."

---

## SLIDE 11 — Team + governance

**Headline:** Solo founder today. Foundation formation + external board recruitment active.

**Visual:**

**Founder:** [name, role, background] — *bio in data room*

**Foundation board** (PRSM-GOV-1 §8.4):
- Target: first 2 external members by Q3 2026
- Seed: domain expertise across silicon, ML, cryptography
- Filing cadence: public board meeting minutes, public financials quarterly

**Advisory pipeline** (if applicable): *[redacted names]*

**Partners we've sought outreach with**: *[data room for specifics; mentioned in speaker notes only]*

**Speaker notes:** "I want to be direct about this: I'm solo today. The Foundation formation is the next step after this raise — jurisdiction is scoped, counsel is shortlisted, audit firms are shortlisted. First external board members target Q3 2026. And I'll say what you're thinking: solo-founder risk is real. The answer is that the repo state is public and reproducible, 427 tests pass, and the Series A capital specifically funds co-founder recruitment. I'd rather raise the money to hire the right people than hire the wrong people before raising."

---

## SLIDE 12 — The ask

**Headline:** [Round size] at [valuation] to deploy the verification triad to mainnet.

**Visual:**

**Round:** Series [A/seed — fill] at [valuation] — *term sheet in data room*

**Use of proceeds (24-36 month runway):**

| Allocation | % | Purpose |
|-----------|---|---------|
| CIS silicon design + first fab run | ~35% | First-implementer chip execution |
| T4 meganode deployment | ~20% | Seed supply-side of network |
| ML research team | ~15% | CIS-2 spec development + R1/R5 research arcs |
| Commissioned data pipeline | ~10% | High-value data vertical |
| External audit retainer | ~5% | Phase 3.1 + 7 + 7.1 bundled audit |
| G&A + legal + Foundation formation | ~15% | Runway |

**What closes this round unlocks (the next 12 months):**

- Mainnet deploy of verification triad
- Foundation formation complete + first Foundation board members seated
- CIS chip design phase complete, first fab engagement signed
- First T4 meganode online earning FTNS
- First frontier-lab partnership on CIS-1 adoption

**Speaker notes:** "The ask is [round size] at [valuation]. That funds 24-36 months of runway covering the first CIS chip design and fab, T4 meganode deployment, ML research team, external audit retainer for the bundled mainnet audit, and Foundation formation. What closing this round unlocks in the next 12 months is concrete: mainnet deploy, Foundation formation, CIS chip design complete, first T4 meganode online, first frontier-lab partnership signed. I'm asking for accredited investors only under Reg D 506(c). Term sheet in the data room. Questions?"

---

## Appendix slides (optional, for Q&A)

These are supplementary slides to have ready for common investor questions. Don't include in the main deck; cue them up if asked.

### A1 — Deep-dive on CIS-1 conformance levels

Diagram of C1 / C2 / C3 confidentiality tiers with exfiltration-lower-bound metric. Source: `docs/2026-04-21-prsm-cis-1-confidential-inference-silicon.md`.

### A2 — Economic model (bear/base/bull 5-year)

FTNS issuance curve + halving schedule. Source: `docs/2026-04-21-prsm-economic-model-white-paper.md`.

### A3 — Competitive landscape

Comparison vs. Akash, io.net, Bittensor, Gensyn, Render, Filecoin Compute. Differentiators: CIS-1 silicon standard, three-tier verification, governance constraints, Prismatica licensing revenue stream.

### A4 — Regulatory position

Howey analysis summary for FTNS + equity-separation rationale + jurisdiction scoping (Cayman primary). Sources: `docs/2026-04-21-prsm-tok-1-ftns-tokenomics.md`, `docs/2026-04-23-foundation-jurisdiction-scoping.md`.

### A5 — Research track

8 R-items with status (6 partner-handoff-ready docs, 2 pending partner identification). Source: `docs/2026-04-14-phase4plus-research-track.md`.

### A6 — Supply-side diversity standard

PRSM-SUPPLY-1: soft-cap mechanisms + diversity bonus triggers at >30% single provider or >40% single country. Source: `docs/2026-04-22-prsm-supply-1-supply-diversity-standard.md`.

### A7 — Risks + mitigations

Honest-brokerage slide covering: solo-founder risk, Foundation-formation execution risk, regulatory-posture shifts, audit-finding risk, CIS chip fabrication risk, competitive closure. One mitigation bullet per risk.

---

## Execution notes for the deck build-out

### What to add for the visual version
- **Slide 1:** Professional wordmark treatment — the current repo uses plain text; a designed mark is worth commissioning before first investor meetings
- **Slide 3:** The architecture diagram needs a clean illustrator/Figma treatment. The ASCII version above is a placeholder
- **Slide 5:** The verification triad table should become a visual — three columns with iconography for each tier
- **Slide 7:** "Shipped" slide benefits from a screenshot of the repo or CI badge
- **Slide 9:** The prohibited-amendments list can become a single-side visual with a "locked" iconography treatment
- **Slide 12:** Use-of-proceeds pie chart

### What to remove or tighten
- Appendix slides (A1-A7) stay in the Keynote/Pitch file but are *never* clicked through in the main pitch — they're Q&A ammunition
- The current v1 is ~12 slides of main deck + 7 appendix. Cut to 10 + 5 if a specific investor's meeting format demands
- Speaker notes should be memorized, not read

### Deck variants to build later
- **30-second elevator** — slides 1, 3, 7 only
- **5-minute exec summary** — slides 1, 2, 3, 5, 7, 12
- **45-minute deep dive** — all 12 + every appendix
- **Foundation-board-candidate pitch** — swap slide 12 (the ask) for a board-service invitation slide

### Things NOT in this deck (deliberately)
- **Team slides with stock photos of the single founder** — feels thin. Better to be direct about solo-founder status in speaker notes (see slide 11).
- **Customer logos / traction slides** — pre-revenue; no real traction to show. Attempting to fake it is worse than admitting the pre-revenue stage.
- **Competitive-landscape matrix in main deck** — A3 appendix only. Investors who ask will see it; investors who don't ask don't need it.
- **Product screenshots of consumer UI** — PRSM is a protocol, not an app. A screenshot of the MCP tool interface would undersell the real product (infrastructure).

---

## What this doc is + isn't

**Is:** a draft pitch-deck content outline for iteration with the founder before building the visual deck. Intended to be the source-of-truth for wording; visual deck built on top.

**Isn't:** the visual deck. The visual deck is built in Figma/Pitch/Keynote on top of this content. Words here are final; layout is not.

**Iteration cycle:** Founder reviews → edits inline → rebuild visual → dry-run with an advisor → adjust → production deck.

---

## Related documentation

- `docs/2026-04-22-prsm-investor-executive-summary.md` — the written short-form this deck expands from
- `docs/2026-04-23-foundation-jurisdiction-scoping.md` — Foundation formation context (for A4 appendix)
- `docs/2026-04-23-auditor-shortlist-and-rfp.md` — audit readiness context (for slide 7 + A7)
- `docs/2026-04-23-legal-counsel-shortlist-and-rfp.md` — legal path context (for slide 9)
- `docs/2026-04-21-prsm-economic-model-white-paper.md` — economic model (for A2 appendix)
- `docs/2026-04-21-prsm-cis-1-confidential-inference-silicon.md` — CIS-1 standard (for A1 appendix)
- `docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md` — governance (for slide 9)
- `docs/2026-04-21-prsm-tok-1-ftns-tokenomics.md` — tokenomics (for slide 9 + A4)
