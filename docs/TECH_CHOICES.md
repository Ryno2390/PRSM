# Technology Choices — Why Ethereum + Base for PRSM

**Audience:** investors, technical advisors, and developers evaluating PRSM's chain architecture. This document exists to give a complete, defensible answer to "why did you pick this chain?" without hand-waving.

**Short answer:** PRSM is built on **Base**, an Ethereum Layer 2 (L2) operated by Coinbase. We chose Ethereum because it is the only smart-contract chain that simultaneously delivers credible decentralization, a mature developer ecosystem, clear US regulatory posture, and the broadest audit/tooling coverage. We chose Base specifically because it gives us ~$0.01 transactions and 2-second blocks while inheriting Ethereum's security, plus Coinbase's involvement provides a legitimacy narrative and a natural fiat on-ramp for Phase 5.

The rest of this document expands that reasoning into an argument we can stand behind in any technical-depth investor conversation.

---

## Evaluating chains: the six-criteria framework

Not every cryptocurrency is safe to build on. Some are outright scams; others are technically sound but misaligned with PRSM's use case. Our evaluation criteria, in rough order of importance:

1. **Credibly decentralized consensus.** No single party (company, foundation, or individual) can halt the chain, censor transactions, or unilaterally change the rules. This is the property that makes the chain *different from a regular database entry at a fintech company*. Without it, PRSM's "royalty flow cannot be stopped" promise collapses.

2. **Open-source, audited protocol code.** Anyone can inspect what the chain does. Legitimate chains welcome audits; scams resist them.

3. **Transparent token supply distribution.** How much went to founders, VCs, the public? A 70% insider allocation with no cliff is a scam template. A transparent premine with disclosed unlock schedules is fine.

4. **Real usage, not just speculation.** Is the chain actually used for something — payments, smart contracts, applications — or is the only activity trading the token itself?

5. **Ecosystem maturity.** Independent developers, tooling, infrastructure providers, wallet integrations, exchange listings. A chain that only one company builds on is a company, not a chain.

6. **Track record under adversarial conditions.** Has the chain survived attacks, bugs, market crashes, regulatory scrutiny? How did it respond?

**By these criteria, the number of truly "safe" chains to build on is small — probably fewer than 10 out of the ~20,000 tokens in circulation meet all six.**

---

## The major chains, honestly

| Chain | Launched | Consensus | Core strength | Honest weakness |
|---|---|---|---|---|
| **Bitcoin** | 2009 | Proof of Work | Longest track record, most battle-tested, unambiguous store-of-value narrative | Limited programmability — not useful for applications like PRSM |
| **Ethereum** | 2015 | Proof of Stake (since 2022) | Largest developer ecosystem, most tooling, most auditor coverage, most liquid smart-contract chain | Expensive on L1 (solved by L2s) |
| **Solana** | 2020 | PoH + PoS hybrid | Very fast, low fees, consumer-app UX closer to Web2 | Multiple chain halts (2021-2024); validators expensive → more centralized than Ethereum |
| **Cosmos / Polkadot** | 2019-2020 | PoS + app-specific chains | App sovereignty (each app is its own chain) | Smaller ecosystem; fragmented liquidity |
| **Avalanche** | 2020 | PoS + subnets | Fast finality, EVM-compatible | Moderate ecosystem; value accrual to AVAX token unclear |
| **Aptos / Sui** | 2022-2023 | PoS + Move VM | New VM design, high throughput | Too new to evaluate long-term survival |

Everything not in that table — and many things in it — is weaker on at least one of the six criteria.

---

## Why Ethereum specifically

Ethereum is the only place where building PRSM is *sane* for a small team. Here are the five reasons that decision is defensible in any room:

### 1. Developer ecosystem (the load-bearing reason)

Ethereum has an order-of-magnitude lead in developer mindshare over any other smart-contract chain. For PRSM in practice:

- **Tooling:** Hardhat, Foundry, OpenZeppelin, Ethers.js — the daily toolkit is Ethereum-native
- **ERC-20 token standard:** Every wallet, exchange, and indexer speaks this. FTNS inherits all of that compatibility automatically
- **Auditor pool:** Dozens of reputable Solidity audit firms (Trail of Bits, OpenZeppelin, Certora, Spearbit, Sigma Prime, Code4rena, ConsenSys Diligence). Solana has perhaps 3-4 firms at equivalent tier
- **Educational resources:** Every tutorial, blog post, and Stack Overflow answer defaults to Ethereum. For a small team, "10× more people have already solved problems you're about to encounter" is load-bearing

### 2. Credible decentralization

Ethereum has ~1 million validators globally distributed across dozens of jurisdictions. **Zero chain halts since Proof of Stake launched in September 2022.** No single entity — not even the Ethereum Foundation — can censor transactions, confiscate assets, or change the protocol unilaterally.

Solana has had **7+ chain halts** since launch, including multi-hour outages in 2022-2024. Its validator hardware requirements (~$5K/month operating cost) cap the validator set at ~2,000 nodes, concentrated in fewer datacenters.

For PRSM's promise that **royalty flow cannot be stopped by any single actor**, only Ethereum-family chains currently deliver this with full confidence.

### 3. US regulatory clarity

- The SEC has implicitly classified ETH-the-asset as not-a-security (allowed ETH futures ETFs in 2023, then spot ETFs in 2024)
- The CFTC has explicitly classified ETH as a commodity
- No major US-regulator enforcement action has targeted Ethereum-the-protocol

This is about as clear as the current US crypto regulatory environment gets.

SOL's regulatory status is genuinely unclear — the SEC named SOL in securities actions against Binance and Coinbase. A Solana spot ETF was delayed while Ethereum's was approved. For a US-based foundation raising from US-based investors and onboarding US-based creators, Ethereum has materially less legal surface area.

### 4. The L2 scaling answer

Ethereum L1 is slow (~15 TPS) and expensive ($5-50 per transaction). This is the standard critique from competitors like Solana.

**Ethereum's answer is Layer 2 chains** — Base, Optimism, Arbitrum, zkSync, others — that inherit Ethereum's security while delivering:

- ~$0.01 per transaction
- 2-second block times
- Thousands of TPS

This gives PRSM Ethereum's decentralization + tooling + regulatory posture + ecosystem, without paying Ethereum's cost/speed penalty.

When an investor asks *"why not Solana?"*, the one-sentence answer is:

> **"We need Ethereum's decentralization and tooling, we don't need Solana's peak speed for our use case, and we get near-Solana performance via Base L2 without giving up any of Ethereum's other advantages."**

### 5. Composability with existing infrastructure

PRSM does not exist in isolation. It needs to integrate with:

- **Stablecoins:** USDC is native on Base (critical for Phase 5 fiat onramp)
- **DEXs:** Uniswap V3 is on Base (for FTNS/USDC liquidity when needed)
- **Oracles:** Chainlink is on Base (for price feeds in future phases)
- **Cross-chain:** LayerZero, Wormhole all support Base
- **Wallets:** Every major wallet (MetaMask, Rainbow, Coinbase Wallet, Frame) supports Base natively
- **Block explorers:** Basescan is the gold standard for on-chain transparency — PRSM's "anyone can audit the royalty flow on-chain" promise is delivered by a mature, public, free tool

Building on Solana would mean rebuilding most of this against a different VM (SVM, not EVM), a different wallet ecosystem (Phantom-centric), a different explorer standard (Solscan), different audit tooling, and a different bridge landscape. For PRSM's stage, that's months of avoidable friction with no offsetting benefit.

---

## Why Base specifically (vs other Ethereum L2s)

Among Ethereum L2s, we chose **Base** over Optimism, Arbitrum, zkSync, and others for five specific reasons:

### 1. Coinbase as operator — legitimacy narrative

Base is built and operated by Coinbase, a regulated US public company (NASDAQ: COIN). For investor conversations, "PRSM deploys on Base, which is operated by Coinbase" is a sentence that immediately establishes legitimacy. No other L2 has this clarity.

### 2. Native USDC

Circle (the USDC issuer) issues **native USDC directly on Base** — not a bridged or wrapped version. This matters for Phase 5's fiat on-ramp: users can move USD → USDC on Coinbase → send to their PRSM wallet with zero bridging friction, zero bridge risk, zero custody handoff.

### 3. OP Stack foundation

Base uses the **OP Stack** — the same open-source rollup technology as Optimism. It's the most battle-tested L2 codebase, with:

- Public, auditable code
- Security inherited from Ethereum L1 via fraud proofs
- A clear decentralization roadmap (trust-minimized withdrawals, multiple sequencers)

If Coinbase-as-entity ever fails or faces regulatory trouble, PRSM's contracts can be redeployed on any OP Stack L2 (Optimism, Zora, Mode, etc.) with minimal code changes.

### 4. Coinbase user base = natural fiat onramp

Coinbase has ~100 million verified users. Any Coinbase user can buy ETH or USDC and withdraw directly to Base in seconds. For PRSM's creator/user acquisition strategy, "install PRSM → use your existing Coinbase account to fund your wallet" is a dramatically lower-friction path than anything involving third-party exchanges, bridges, or custody setups.

### 5. Growing ecosystem, growing network effect

As of 2026, Base is the fastest-growing Ethereum L2 by transaction volume, active addresses, and new protocol deployments. That matters for PRSM's long-term story: the infrastructure keeps improving, the user base keeps growing, and more integrations keep showing up on Base natively.

---

## Addressing investor pushback

A well-prepared investor will push back on chain choice. Here are the most common objections and how to respond:

### "But Solana is faster and cheaper."

True on L1 comparison. False on L2 comparison: Base delivers Solana-competitive speed (2-second blocks, ~$0.01 fees) while inheriting Ethereum's security. The real question is whether PRSM's use case — royalty payments, content provenance, creator payouts — benefits more from raw speed than from decentralization and regulatory clarity. **It doesn't.** PRSM is not a high-frequency trading platform; it's infrastructure for long-term creator compensation.

### "What if Ethereum gets disrupted by the next-generation chain?"

Possible on a 5-10 year horizon; highly unlikely on a 1-3 year horizon. Ethereum's moat is its ecosystem: tens of thousands of developers, hundreds of billions in TVL, thousands of applications. Displacing that requires both order-of-magnitude better technology *and* the time to build an equivalent ecosystem. No competitor is close. Meanwhile, PRSM's contracts are EVM-compatible, so if a competitor ever did win, migrating is tractable — we deploy the same Solidity on the new EVM chain.

### "Why not be early on a brand-new chain and capture value faster?"

Because being early on a weak chain is worse than being average on a strong chain. PRSM's value comes from **users and creators trusting the network**, which requires the network to be obviously safe. A brand-new chain introduces trust uncertainty that compounds PRSM's own early-stage trust uncertainty. The incremental reach from being first on a smaller chain is dwarfed by the credibility tax of being on an unproven one.

### "What if Coinbase gets regulatory trouble and Base goes down?"

Fair concern. Three mitigations:

1. Base is on a decentralization roadmap — more sequencers over time, trust-minimized withdrawals to Ethereum L1
2. Even if Coinbase-as-entity fails, the bridge to Ethereum L1 ensures no funds are permanently lost — users can withdraw to L1 via the trust-minimized path
3. PRSM's contracts are portable: if we ever needed to migrate, any OP Stack L2 (Optimism, Zora, Mode, future entrants) could host the same code

This is a meaningful risk to track, not a disqualifying one.

### "Why don't you run your own chain?"

Building our own L1 would cost $20-100M+ in engineering and security audits before a single user transacted, and would take 2-4 years. It would also require bootstrapping validator economics, wallet support, exchange listings, and block-explorer coverage from zero — none of which is PRSM's core competence or value-add. Every dollar spent building our own chain is a dollar not spent making PRSM itself better.

---

## The three distinct questions

Crypto conversations often conflate three questions that should stay separate:

1. **"Will the asset price go up?"** — speculation question, not PRSM's concern or promise
2. **"Is the chain technically sound?"** — Ethereum: yes. Base: yes
3. **"Does this chain's properties match what the project needs?"** — this is the architecture question

PRSM's architecture question is: *"Where can we deploy contracts that (a) cannot be censored, (b) have low enough transaction costs for micropayment royalty flow, (c) are auditable by anyone for free, and (d) have a clean US regulatory story?"*

The answer is **Ethereum L2, specifically Base.** There is no close second for this combination of requirements.

---

## Cross-references

- **Repository README** (`README.md`) — top-level PRSM overview with a concise version of this argument
- **Master roadmap** (`docs/2026-04-10-audit-gap-roadmap.md`) — phase-by-phase build plan, all phases on Base
- **FTNS token on Base mainnet** — [`0x5276a3756C85f2E9e46f6D34386167a209aa16e5`](https://basescan.org/address/0x5276a3756C85f2E9e46f6D34386167a209aa16e5)
- **Phase 1 on-chain provenance plan** (`docs/2026-04-10-phase1-onchain-provenance-plan.md`) — the first production contracts landing on Base mainnet
- **Phase 2 remote-compute plan** (`docs/2026-04-12-phase2-remote-compute-plan.md`) — the launch UX thesis that builds on Base's properties

---

*Last updated 2026-04-20. Document owner: PRSM Foundation. Open a PR if any claim here goes stale relative to current chain state.*
