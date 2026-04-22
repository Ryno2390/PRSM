# PRSM — Investor Executive Summary

> ⚠️ **SUPERSEDED 2026-04-22.** This version reflects state through Phase 3 (as of 2026-04-21) and predates the Phase 3.1 + 7 + 7.1 + 7.1x shipment, PRSM-SUPPLY-1, the pre-audit hardening arc, and the Phase 4+ research-track expansion. **Use `docs/2026-04-22-prsm-investor-executive-summary.md` instead.** This file retained only for historical reference.

**Prismatica, Inc.** | **Reg D Rule 506(c)** | **April 2026**
**Contact:** *[founder contact]* | **Data room:** *[access link]*

---

## The opportunity

Frontier AI inference is centralizing faster than any prior technology wave. Three companies control the GPU supply, one company controls the attestation root, and zero open-weights labs can safely publish state-of-the-art models without expecting them to be exfiltrated and relisted under different branding. The industry needs neutral infrastructure that is cryptographically credible, economically aligned, and permissionless — not another vertically-integrated stack.

**PRSM is a P2P protocol that turns underutilized consumer electronics, T3 cloud arbitrage, and dedicated T4 meganodes into a single priced mesh for AI inference, storage, and data.** Contributors earn FTNS tokens for sharing latent resources. Third-party LLMs (local, API, or OAuth) consume PRSM services via MCP tools. Reasoning happens in the LLM; execution happens on PRSM nodes.

## What's shipped

As of 2026-04-21, on `main` at `github.com/Ryno2390/PRSM`:

| | Status |
|---|---|
| **Phase 1.3** — On-chain provenance + royalty distribution on Base mainnet | Complete. Contracts deployed, 7-day Sepolia bake-in passed. Mainnet deploy pending multi-sig hardware (May 2026). |
| **Phase 2** — Remote compute dispatch with signed receipts, FTNS escrow, Ed25519 attestation | Shipped. 4-pass codex gate, `phase2-merge-ready-20260420` tag. |
| **Phase 2.1** — TopologyRandomizer, ShardPreemptedError, TEE attestation schema | Shipped. `phase2.1-merge-ready-20260420` tag. |
| **Phase 3** — FTNS-priced compute marketplace with price discovery, reputation, end-to-end orchestration | Shipped. 3-node acceptance test passes bit-identical output. `phase3-merge-ready-20260420` tag. |

**Test suite:** 111 tests green across unit + integration + end-to-end. Bit-identical output guaranteed between local and remote execution paths via shared `execute_shard_locally` helper.

## The technical moat

1. **CIS-1 confidential-silicon standard.** 784-line RFC-style spec defining three conformance levels (C1 / C2 / C3) with quantified weight-exfiltration lower bounds: 2²⁰ inferences at baseline to 2⁴⁰ at SOTA tier. Governance-separated: Foundation owns the standard; any fab can implement. **No one else in this space has specified a frontier-grade confidentiality target.**
2. **Vendor-independent attestation.** On-chain AttestationRegistry on Base binds chip identity to protocol governance, not to NVIDIA or any single CA. Multi-manufacturer requirement at C2+ is a **prohibited amendment** (cannot be removed without dissolving the Foundation).
3. **Phase-7-ready economic backstop.** The protocol reserves a `stake_tier` field in every listing and a signing-payload format compatible with on-chain slashing; the slashing contract ships at Phase 7.

## The governance-credibility moat

Investors who understand Web3 history know the real question isn't "what's the tech?" — it's "who captures the rents?" PRSM answers structurally, not promissorily:

- **Foundation cannot build chips.** Prohibited amendment.
- **Foundation cannot revenue-share with Prismatica.** Prohibited amendment.
- **Founder must pick one entity after Year 3.** Cannot serve both Foundation board and Prismatica executive indefinitely.
- **Foundation certifies Prismatica's chips on the same terms as any other implementer.** C2+ certification explicitly requires ≥2 independent manufacturers.
- **Transparency default.** Every on-chain transaction, committee vote, certification decision, and financial statement is public on a fixed cadence.

Details: `docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md` (597 lines).

## The regulatory clarity

- **FTNS is distributed only as compensation for services rendered.** No retail sale. No bonding curve. No ICO.
- **Howey analysis passes cleanly** for FTNS recipients: no investment of money, no expectation of profit from others' efforts. Mirrors Bitcoin's regulatory posture.
- **Bootstrap capital raises through Prismatica equity** (Reg D 506(c)) — a cleanly-classified security — NOT through FTNS distribution. Investors buy equity; FTNS is a separate instrument with separate treatment.

Details: `docs/2026-04-21-prsm-tok-1-ftns-tokenomics.md` (605 lines).

## The deal

- **Instrument:** Prismatica Inc. equity, Delaware C-corp, Reg D Rule 506(c) accredited-only.
- **Use of proceeds:** First-implementer CIS chip design and fabrication, T4 meganode deployment, ML research team, commissioned-data pipeline, 24-36 month operating runway.
- **Investor returns:** Prismatica's enterprise-value growth (operating revenue from managed meganodes + CIS chip licensing + FTNS treasury appreciation from organic earning, not speculation).
- **Round size, valuation, terms:** *[specified in pitch deck and data room]*.

## Why now

1. **H100 Confidential Compute is shipping** — raising the baseline trust floor and making the market ready for a protocol that treats attestation as a first-class primitive rather than a marketing bullet.
2. **Frontier labs are publicly asking "what would it take for us to publish weights to a decentralized substrate?"** — PRSM's CIS-1 roadmap is the concrete technical answer.
3. **Regulatory posture has matured.** MiCA, FINMA guidance, Cayman Foundation Company framework all provide workable non-profit-foundation paths that did not exist in 2019.
4. **Phase 3 marketplace is live on our test infrastructure and bit-identical to local baseline.** The primitives are not speculative — they are running in 3-node clusters today.

## Team

**Founder:** [name, role, background summary — bio in data room]

Foundation board and Prismatica advisory roles currently in recruitment per `PRSM-GOV-1 §8.4` (Year 1 milestones). First external board members targeted for Q3 2026.

---

**Read next:**
- `docs/TECH_CHOICES.md` — why Ethereum / Base / ERC-20, with investor-pushback responses.
- `docs/2026-04-21-prsm-cis-1-confidential-inference-silicon.md` — the silicon standard in full.
- `docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md` — governance charter.
- `docs/2026-04-21-prsm-tok-1-ftns-tokenomics.md` — tokenomics standard.
- `README.md` — protocol architecture and shipped rings.

*This document is a summary and is not an offer to sell securities. Any offer will be made only through formal offering materials to accredited investors under Regulation D Rule 506(c).*
