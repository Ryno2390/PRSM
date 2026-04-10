# PRSM Investment Materials

## ⚠️ INVESTMENT STATUS DISCLAIMER

**PRSM ships working infrastructure today (v1.6.2)** — all 10 Sovereign-Edge AI rings are live, the Python/JavaScript/Go SDKs are published, and the FTNS token is deployed on Base mainnet. Series A funding accelerates network growth: geographic bootstrap expansion, fiat on-ramp credentials, external security audit, and marketing to node operators and data providers. Full network self-sustainability target: 18-24 months post-funding.

**Investment Opportunity**: Series A funding to scale a shipped P2P infrastructure protocol into a self-sustaining mesh network of consumer-class nodes serving any third-party LLM via MCP tools.

## 🎯 Executive Summary

PRSM is a **P2P infrastructure protocol for open-source collaboration**. Consumer-class nodes share latent storage, compute, and data; contributors earn FTNS tokens; any third-party LLM (Claude, GPT, local Llama) reaches the network through MCP (Model Context Protocol) tools. Reasoning happens inside the caller's LLM — PRSM supplies the infrastructure that LLM uses to reach distributed resources without routing through a centralized provider.

**PRSM is not an AGI framework.** It does not host models. It does not claim to build superintelligence. It is infrastructure — the plumbing that makes open, private, contributor-owned AI compute possible.

## 🏛 Corporate Structure

PRSM operates as a **two-entity structure** designed to align open-protocol integrity with commercial sustainability and investor returns.

### Structure Overview

```
┌──────────────────────────────────────────────────────────────┐
│                 PRSM Foundation (Non-Profit)                 │
│              501(c)(3) or equivalent  ·  [Planned]           │
│                                                              │
│  Governs PRSM protocol  ·  Holds IP  ·  Ensures openness    │
│  Shareholder in Prismatica  ·  Self-funds via dividends      │
└───────────────────────────┬──────────────────────────────────┘
                            │  Holds equity · Receives dividends
                            ▼
┌──────────────────────────────────────────────────────────────┐
│               Prismatica Inc. (For-Profit)                   │
│           Delaware C-Corp  ·  [Pending Formation]            │
│                                                              │
│  Mega-node operator  ·  Base-load compute/data/storage       │
│  Earns FTNS royalties  ·  Investor-facing (Series A)         │
└──────────────────────────────────────────────────────────────┘
```

### PRSM Foundation — Neutral Protocol Layer

The PRSM Foundation is the non-profit steward of the PRSM protocol — analogous to the Mozilla Foundation or Ethereum Foundation. It:

- Owns and governs the PRSM protocol and associated IP
- Ensures the network remains open, decentralized, and not capturable by any single commercial entity
- Holds an equity stake in Prismatica Inc., receiving dividends that fund ongoing protocol operations

**Formation status**: Planned — to be established after Prismatica Inc. is formed.

### Prismatica Inc. — Commercial Mega-Node Operator

Prismatica Inc. is the for-profit entity and the **investor-facing vehicle for Series A**. It operates as the PRSM network's primary mega-node in the early growth phase, providing base-load compute, data, and storage capacity. Prismatica earns **FTNS provenance royalties** for every resource unit it contributes to the network.

**Formation status**: Pending — Delaware C-Corp via Stripe Atlas.

### Why This Structure Benefits Investors

| Feature | Investor Benefit |
|---------|-----------------|
| **Royalty model** | Prismatica earns FTNS on every network transaction involving its resources — a passive royalty position that grows with adoption |
| **Network flywheel** | More nodes join → more compute/data/storage needed → Prismatica earns more FTNS → FTNS appreciates with supply/demand dynamics |
| **Treasury growth** | Prismatica's FTNS holdings grow without requiring token sales at a loss |
| **Foundation alignment** | Foundation's equity in Prismatica means protocol stewardship and commercial success are structurally aligned |
| **Comparable models** | Early AWS (infrastructure provider for a platform ecosystem), ARM Holdings (royalties on a chip architecture), Berkshire Hathaway (float / capital compounding model) |

Investing in Prismatica is equivalent to **owning a royalty position on the PRSM network** — the more the network grows, the more Prismatica earns, and the more value accrues to shareholders.

See [`docs/business/PRISMATICA_STRATEGY.md`](./PRISMATICA_STRATEGY.md) and [`docs/business/legal/LEGAL_STRUCTURE.md`](./legal/LEGAL_STRUCTURE.md) for full detail.

---

## 📊 Investment Opportunity

**Series A: $18M**
- **Valuation**: $72M pre-money
- **Use of funds**: Network growth (bootstrap regional expansion), fiat on-ramp completion, external security audit, marketing to node operators and data providers
- **Timeline**: 18-24 months to network self-sustainability, 36 months to positive cash flow at Prismatica
- **Exit strategy**: Strategic acquisition or IPO within 5-7 years

## 🎮 Market Positioning

### The Infrastructure Thesis
- **Problem**: Frontier AI labs hoard data, compute, and models behind API walls. Billions of consumer devices sit idle with latent storage, compute, and sometimes proprietary data that never leaves the device.
- **Solution**: PRSM is a P2P infrastructure protocol that aggregates those latent consumer resources into a mesh network any third-party LLM can reach via MCP tools. Contributors earn FTNS; users leverage PRSM through their LLM of choice.
- **Market**: Every organization running LLM inference over proprietary data — every research lab, every enterprise with compliance constraints, every developer who doesn't want to send their queries to a centralized API.
- **Timing**: MCP adoption is making LLMs natively tool-aware, and consumer GPU capacity has far outpaced centralized datacenter growth.

### Competitive Advantages
1. **First-mover advantage** in MCP-native P2P compute infrastructure
2. **Network effects** — value increases with every node that joins, every dataset published, and every LLM that adds PRSM to its tool belt
3. **Privacy by construction** — WASM zero-persistence + semantic sharding + differential privacy + model sharding
4. **Contribution-minted token** — FTNS has no pre-mine, no ICO, no foundation reserve to dump. Supply enters circulation only when real work is done on the network.

## 📈 Financial Projections

### Revenue Model
- **Prismatica mega-node FTNS royalties** — primary income stream, growing with network adoption
- **Foundation dividends from Prismatica equity** — funds ongoing protocol development
- **Chronos fiat bridge fees** — FTNS ↔ USD/USDT conversion
- **Treasury 5% cut** — funds protocol upkeep under on-chain governance

### 5-Year Financial Forecast
| Year | Revenue | Users | Agents | ARR Growth |
|------|---------|--------|---------|-----------|
| 2026 | $0.5M | 5K | 25K | N/A (Launch) |
| 2027 | $2.5M | 25K | 125K | 400% |
| 2028 | $10M | 100K | 500K | 300% |
| 2029 | $35M | 350K | 1.75M | 250% |
| 2030 | $100M | 1M | 5M | 185% |

*Note: Projections assume successful Series A funding and 18-month production deployment timeline.*

## 🛠 Technical Implementation

### Development Progress (all 10 Sovereign-Edge AI Rings shipped as of v1.6.2)
- **Ring 1**: Wasmtime WASM sandbox + hardware profiler
- **Ring 2**: Mobile agent dispatch + gossip bidding
- **Ring 3**: Semantic sharding + swarm map-reduce
- **Ring 4**: Hybrid FTNS pricing + prosumer staking
- **Ring 5**: WASM mobile agent runtime + MCP tool surface (16 tools)
- **Ring 6**: Production hardening (dynamic gas, RPC failover, CLI UX)
- **Ring 7**: TEE runtime abstraction + differential privacy
- **Ring 8**: Tensor-parallel model sharding + collusion detection
- **Ring 9**: NWTN training pipeline (fine-tuned NWTN LLM itself is future work)
- **Ring 10**: Integrity verification + privacy budget + hash-chained audit log

### Technology Stack
- **Backend**: Python/FastAPI, PostgreSQL, Redis (Upstash), Wasmtime WASM runtime
- **Compute**: Consumer-class nodes (gaming PCs, consoles, laptops, phones)
- **Storage**: Native ContentStore with semantic shard manifests
- **Interop**: Model Context Protocol (MCP) — any MCP-compatible LLM gains PRSM tools
- **Blockchain**: FTNS token live on **Base mainnet** — [`0x5276a3756C85f2E9e46f6D34386167a209aa16e5`](https://basescan.org/address/0x5276a3756C85f2E9e46f6D34386167a209aa16e5)
- **Fiat bridge**: Chronos — FTNS ↔ USD/USDT for node operators cashing out

### SDK Availability (Live as of 2026-03-28)
- **Python SDK**: `pip install prsm-python-sdk` — [pypi.org/project/prsm-python-sdk](https://pypi.org/project/prsm-python-sdk)
- **JavaScript SDK**: `npm install prsm-sdk` — [npmjs.com/package/prsm-sdk](https://npmjs.com/package/prsm-sdk)
- **Go SDK**: `go get github.com/Ryno2390/PRSM/sdks/go@v0.2.0` — [pkg.go.dev/github.com/Ryno2390/PRSM/sdks/go](https://pkg.go.dev/github.com/Ryno2390/PRSM/sdks/go)

## 👥 Team & Expertise

### Leadership Team
- **CEO**: AI coordination vision and execution
- **CTO**: Distributed systems and blockchain architecture
- **VP Engineering**: Scalable infrastructure and DevOps
- **VP Product**: Enterprise AI and user experience

### Advisory Board
- Enterprise AI executives from Fortune 500 companies
- Blockchain and distributed systems experts
- Former executives from successful SaaS exits

## 🎯 Go-to-Market Strategy

### Target Participants
1. **Data providers** — Anyone with proprietary data (research labs, domain experts, hobbyist archivists) who can earn 80% of every query against their content
2. **Compute providers** — Consumer hardware owners (gaming PCs, consoles, laptops, phones) who can earn FTNS for sharing idle cycles
3. **LLM users** — Developers and researchers who want their LLM (Claude, GPT, local) to reach distributed data without routing through centralized APIs
4. **MCP-compatible clients** — Claude Desktop, OpenClaw, and any MCP-aware coding or research assistant

### Growth Strategy
- **Open-source grassroots** — drive adoption through the PyPI package, SDK ecosystem, and MCP tool exposure
- **Data provider incentive** — 100 FTNS welcome grant + 80% royalty share creates strong on-ramp for data-rich contributors
- **Bootstrap regional expansion** — EU + APAC bootstrap nodes once the NYC3 node saturates
- **Developer content** — technical blog posts, conference talks, and direct engagement with research labs

## 🚀 Strategic Partnership Opportunities

### Target Integration Partners (Planned)

PRSM's architecture is designed for compatibility with major AI and cloud providers:

- **Cloud Providers**: AWS, GCP, Azure deployment-ready architecture
- **AI Model Providers**: API compatibility with OpenAI, Anthropic, Google models
- **Enterprise Software**: Standard integration patterns for enterprise adoption

**Status**: No formal partnerships established. Partnership development is a key post-funding activity.

### Apple Integration Opportunity

PRSM's architecture aligns with Apple's privacy-focused approach to AI:
- Device-local processing capabilities
- Privacy-preserving computation
- iOS/macOS integration potential

**Status**: Conceptual opportunity - formal outreach planned post-funding

## 📊 Market Analysis

### Total Addressable Market (TAM)
- **Global AI inference spend**: hundreds of billions annually, growing fast, almost entirely captured by a handful of centralized providers today
- **Idle consumer compute**: billions of devices with GPUs that could participate if the incentive and infrastructure existed
- **Proprietary data sitting on-prem**: regulated industries (healthcare, finance, research) that cannot ship data to centralized AI providers but could run agents against it locally

### Competitive Landscape
- **vs. centralized AI providers (OpenAI, Anthropic)**: complementary, not competitive — PRSM gives those LLMs hands and a wallet via MCP
- **vs. decentralized AI token projects**: PRSM mints at contribution time (no pre-mine), ships working infrastructure (WASM sandbox, semantic sharding, TEE integration), and has a live token on Base mainnet rather than a roadmap
- **Competitive moats**: network effects on the data provider side, MCP native integration, contribution-minted FTNS, privacy by construction

## 💡 Investment Thesis

### Why Now?
1. **MCP adoption** is accelerating — every major LLM is adding native tool use. PRSM is MCP-native.
2. **Centralization backlash** — regulated industries, privacy-sensitive users, and sovereignty-concerned governments are actively looking for alternatives to centralized AI APIs.
3. **Idle consumer compute capacity** vastly exceeds centralized datacenter growth, and the incentive layer (FTNS) is finally live.
4. **Infrastructure is shipped** — not a promise. All 10 rings are live in v1.6.2.

### Risk Mitigation
- **Multiple forcing functions**: Economic inevitability reduces adoption risk
- **Diversified revenue**: Multiple monetization channels
- **Strong team**: Proven execution in distributed systems
- **Strategic partnerships**: Reduced go-to-market risk

## 📋 Due Diligence Materials

### Technical Documentation
- [Complete Architecture Guide](../architecture.md)
- [API Reference Documentation](../API_REFERENCE.md)
- [Security and Compliance Guide](../SECURITY_HARDENING.md)
- [Production Operations Manual](../PRODUCTION_OPERATIONS_MANUAL.md)

### Business Documentation
- [Game-Theoretic Investment Thesis](../GAME_THEORETIC_INVESTOR_THESIS.md)
- [Risk Assessment and Mitigation](../../archive/completed_roadmaps/RISK_MITIGATION_ROADMAP.md)
- [Phase Implementation Summary](../PHASE_IMPLEMENTATION_SUMMARY.md)
- [Token Economics Model](../tokenomics.md)

### Financial Materials
- Financial projections and modeling
- Current metrics and KPIs
- Competitive analysis report
- Market sizing and opportunity assessment

## 🎯 Next Steps

### For Potential Investors
1. **Schedule deep-dive presentation** with technical and business teams
2. **Review due diligence materials** and technical documentation
3. **Connect with existing customers** and strategic partners
4. **Participate in Series A funding round** - Limited spots available

### Investment Timeline
- **Week 1-2**: Initial investor meetings and presentations
- **Week 3-4**: Due diligence and technical review
- **Week 5-6**: Final terms and documentation
- **Week 7-8**: Funding close and onboarding

## 📞 Contact Information

**Investment Team**
- Email: funding@prsm.ai
- Phone: +1 (555) 123-PRSM
- Investment Deck: [Download PDF](./PRSM_Investment_Deck.pdf)

**Leadership Team**
- CEO: ceo@prsm.ai
- CTO: cto@prsm.ai
- CFO: cfo@prsm.ai

---

*This document contains forward-looking statements and projections. Past performance does not guarantee future results. Please review all risk factors and due diligence materials before making investment decisions.*

**Ready to invest in the infrastructure of AI coordination?**
**[Schedule a meeting with our investment team →](mailto:funding@prsm.ai?subject=Series%20A%20Investment%20Interest)**