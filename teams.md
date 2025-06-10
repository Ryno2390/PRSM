## 🧑‍🤝‍🧑 Teams in PRSM

In a decentralized research ecosystem like PRSM, collaboration must scale across disciplines, geographies, and institutions. The `Teams` module formalizes collaborative units within the PRSM protocol, enabling coordinated task execution, shared token accumulation, and distributed model development—all without sacrificing the open, permissionless ethos of the network.

---

## 🔧 Overview

A **Team** is a smart contract-governed collective of user profiles that:

- Shares access to compute, storage, and model assets
- Pools FTNS token earnings and distributes them according to predefined rules
- Submits collaborative tasks (e.g., multi-agent training jobs)
- Hosts team-specific dashboards, reputation scores, and governance rights

---

## 🛠️ Core Functionality

### 🧱 Team Formation

- Teams can be created by any verified user (must stake a minimal FTNS bond)
- Requires:
  - Team Name (namespace must be unique)
  - Description / research goals
  - Team Avatar / Logo
  - Member invitation list (via PRSM usernames or DID handles)
- Once created, a smart wallet is deployed for the team’s FTNS and metadata registry.

---

### 🔐 Team Wallets

- Each team has a **multisig smart wallet** that:
  - Accepts pooled FTNS contributions
  - Receives royalties from data/model usage (based on provenance IDs)
  - Pays for API costs (e.g., querying NWTN or deploying sub-models)
- Roles:
  - `Owner(s)`: Can update governance policy
  - `Operator(s)`: Can submit jobs / query APIs on behalf of the team
  - `Treasurer(s)`: Authorized to distribute FTNS to team members

---

### 📈 Token Sharing Policies

Token rewards earned by the team (from compute, storage, provenance, etc.) can be split using programmable logic:

```json
{
  "reward_policy": "proportional",
  "metrics": ["task_submissions", "model_contributions", "query_accuracy"],
  "weights": [0.4, 0.4, 0.2]
}
```

> Teams can choose fixed shares, performance-weighted, stake-weighted, or reputation-weighted distributions.

---

### 🧠 Collaborative AI Contributions

- Teams can:
  - Co-develop distilled sub-models
  - Run parallel evaluation and benchmarking tasks
  - Create `Team-Only` provenance-flagged datasets
  - Maintain long-running experiments (e.g. RL loops, bayesian optimization, etc.)

> All collaborative outputs are cryptographically signed by the team wallet and version-tracked in the PRSM IPFS registry.

---

## 🔎 Discovery & Indexing

- Teams will appear in the public **PRSM Team Directory**, filterable by:
  - Field (e.g. "genomics", "quantum materials")
  - Impact score (via FTNS earned + dataset/model usage stats)
  - Organizational affiliation (optional)
- Popular teams may attract contributors, investors, or institutional partners

---

## 🧩 Integration with Profiles & Governance

- Individual contributors can:
  - Belong to multiple teams
  - Opt in to automatic royalty split per team agreement
  - Display team badges and roles in their public profile
- Team creation and membership changes are:
  - On-chain events
  - Verifiable via PRSM’s DAG ledger

---

## 📜 Governance Layer

Teams can choose from modular governance templates:

| Model        | Description                             |
|--------------|-----------------------------------------|
| Autocratic   | One founder has full control            |
| Meritocratic | FTNS-weighted votes + code contribution |
| Democratic   | One user = one vote                     |
| DAO Hybrid   | Custom smart contract constitution      |

All proposals (e.g., change weights, add/remove members, rotate roles) are handled through on-chain voting using the team’s smart wallet.

---

## 🧪 Advanced Features (V2)

- **Research DAO Launchpads**: Teams can upgrade into mini-DAOs with NFT or FTNS staking systems
- **Cross-Team Alliances**: Smart contracts for shared infrastructure or datasets
- **Team-Level Tokenomics**: Custom token creation for large multi-stakeholder initiatives

---

## 📍 Suggested API Endpoints

| Endpoint | Function |
|----------|----------|
| `POST /teams/create` | Register a new team with metadata |
| `GET /teams/:id` | View team profile and public outputs |
| `POST /teams/:id/propose` | Submit governance proposal |
| `POST /teams/:id/reward` | Trigger reward distribution function |
| `POST /teams/:id/members/add` | Add new member via invite |
| `GET /teams/leaderboard` | Ranked list of top teams |

---

## 🖥️ Suggested Frontend Features

- Team dashboard with:
  - Total FTNS earned
  - Top contributors
  - Current proposals
  - Active training jobs
- Badge system for:
  - Verified Team
  - High Impact Contributions
  - Multidisciplinary
- Live feed of team activity (public by default, private optional)

---

## 💡 Example Team Use Cases

- **NanoForge**: A 5-person team running experiments on APM pathway discovery
- **SolarCollective**: Climate modelers pooling sub-models on photovoltaics
- **OpenFold DAO**: Community replicating and evolving AlphaFold on PRSM infra
- **Prismatica Core Research Group**: Internal team within Prismatica submitting foundational experiments and distilling NWTN refinements

---

## 🔚 Summary

Teams are a vital layer of the PRSM protocol, providing the social and economic glue necessary for collaborative scientific progress in a decentralized, scalable ecosystem. Through customizable governance, programmable incentives, and verifiable outputs, PRSM Teams can rival traditional labs—without their institutional constraints.

```bash
Ready to form a team? Run:
> prsm teams create
```

---

> “Collaboration is not the opposite of competition; it is the scaffolding on which progress is built.”