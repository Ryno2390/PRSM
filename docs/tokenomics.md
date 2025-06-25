# PRSM Tokenomics

PRSM operates on a decentralized, merit-based token economy that rewards scientific collaboration, computational contributions, and data sharing. The native token—**FTNS** (Fungible Tokens for Node Support), pronounced “photons”—is designed to fuel user access to the network while creating transparent incentives for maintaining and expanding the PRSM ecosystem.

---

## 🌟 What Are FTNS?

**FTNS (photons)** are the core utility and incentive token within PRSM. They serve as:

- Fuel to access PRSM’s AGI systems (especially NWTN)
- Royalties for users whose data, models, or compute are used
- Incentives for participating in governance, hosting, and research
- Dividends distributed based on token holdings
- Optional currency for external trading

---

## 💡 FTNS Use Cases

| Usage | Description |
|-------|-------------|
| 🔍 Query NWTN | Primary use: “purchase” context access to the Core AGI (NWTN) |
| 📦 Upload/Host Data | Token rewards based on access frequency via IPFS provenance |
| 🤖 Train Models | Earn FTNS based on downstream model usage |
| 🧠 Teach Models | Distilled teachers earn tokens when curricula are reused |
| 🧾 Provide Compute or Storage | Passive rewards for contributing processing or disk capacity |
| 🎓 Submit Research | Earn tokens based on frequency and utility of reuse |

---

## 🧠 Query-as-Fuel Model

FTNS is designed to be spent to access intelligence, not just held.

- New users receive an initial FTNS balance upon signup
- Each query to NWTN consumes FTNS, proportional to complexity and context required
- To continue querying, users must contribute: data, compute, storage, or models

This architecture ensures sustainable usage and aligns incentives with participation.

---

## 🔄 Earning FTNS

FTNS is earned dynamically based on verified system contribution.

| Contribution Type | FTNS Earned |
|-------------------|-------------|
| Submitting Novel Research | Based on reuse and downstream influence |
| Hosting High-Value Data | Tracked via IPFS hash access frequency |
| Providing Compute | Based on verified task execution |
| Distilling Sub-Models | Based on performance and task selection frequency |
| Training Other Models | RLVR-based earnings (e.g., Absolute Zero) |
| Running Model Shards | Passive rewards for maintaining torrent-like distributed models |

---

## 📦 Data Provenance & IPFS

PRSM uses IPFS for secure content-based addressing. Each uploaded item receives a unique cryptographic hash.

This allows:

- Transparent attribution of first contributors
- Access frequency tracking
- Royalties paid to original hosts

> Every time a model, agent, or researcher uses your content, you earn FTNS. Think of it as Spotify for science—tokens paid per use.

---

## 🧮 Smart Contract Example (Pseudocode)

```
OnDataAccess(data_id):
    contributor = resolveOriginalUploader(data_id)
    frequency += 1
    reward = base_rate * frequency_weight(data_id)
    sendTokens(to=contributor, amount=reward)
```

All activity is publicly visible on the DAG ledger (e.g., IOTA Wasp).

---

## 🏦 FTNS and Dividend Distribution

In addition to usage-based rewards, FTNS serves as a claim on network-wide value:

- Every quarter, a dividend pool is calculated
- Users may select a payout ratio (tokens vs. cash) in their profile
- The more FTNS held, the greater the share of dividends

This encourages long-term alignment and platform reinvestment.

---

## 🌍 Trading and Exchange

FTNS can optionally be traded outside PRSM:

- On crypto exchanges (e.g., against BTC, ETH, stablecoins)
- As a mechanism for institutions to acquire access without earning
- As a speculative or governance asset

Note: External ownership of FTNS does not grant voting power or usage rights unless tokens are brought back on-chain.

---

## 🔐 Abuse Prevention

- FTNS issuance tied to verifiable system activity (no inflation)
- Governance votes can flag abusers or revoke earnings
- Circuit-breaker architecture lets nodes suspend malicious behavior

---

## 📈 Long-Term Vision

- FTNS becomes the “gas” for querying AI models in PRSM
- Value of FTNS increases as PRSM becomes more useful and trusted
- Contributors benefit passively from adoption of their uploads and models
- A self-sustaining open-source economy emerges

---

## 📌 Summary

FTNS powers PRSM’s ecosystem by rewarding contribution and enabling access. Whether you’re contributing research, compute, data, or training models, FTNS ensures your effort is traceable and rewarded.

FTNS is not a speculative token—it’s programmable fuel for scientific progress.
