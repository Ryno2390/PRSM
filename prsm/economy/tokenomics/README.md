# FTNS Tokenomics

FTNS (Fungible Tokens for Node Support) is PRSM's economic primitive — the token that compensates contributors for sharing latent storage, compute, and data. This module implements the Ring 4 tokenomics layer: ledger, pricing, staking, and settlement.

## Core principle

FTNS is **minted at contribution time**. There is no pre-mine, no ICO, and no foundation reserve to dump. New nodes receive a 100 FTNS welcome grant; all other supply enters circulation as settlement payouts when queries are served.

## Token facts

| | |
|---|---|
| Symbol | FTNS |
| Decimals | 18 |
| Live contract | `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` (Base mainnet) |
| Testnet | `0xd979c096BE297F4C3a85175774Bc38C22b95E6a4` (Ethereum Sepolia) |
| Fiat bridge | Chronos (`prsm/compute/chronos/`) — FTNS ↔ USD/USDT |

## Revenue split (80 / 15 / 5)

Every query that consumes FTNS is settled through the Ring 4 pipeline with a deterministic split:

| Recipient | Share | Why |
|-----------|-------|-----|
| **Data owner(s)** | 80% | Publishing proprietary data to a node's ContentStore is the rarest resource on the network |
| **Compute providers** | 15% | Running the WASM agents, honoring quota, paying for power |
| **Treasury** | 5% | Protocol upkeep, funded under on-chain governance |

Settlement happens atomically on the DAG ledger (`atomic_ftns_service.py`) with Ed25519 signatures. Multi-party royalty distribution is batched through `prsm/node/multi_party_escrow.py` for gas efficiency.

## Pricing

PRSM's pricing is **not** a token-per-token LLM model. It uses a hybrid menu:

- **PCU (Priced Compute Unit) menu** — deterministic rates for compute tiers (T1 → T4), memory, and WASM fuel
- **Data market** — per-shard and per-access fees chosen by the data owner on upload
- **Network fee** — flat 5% treasury cut

Use `prsm compute quote "query"` to get a free cost estimate before committing FTNS. The quote path runs the Ring 3/4 planner without executing anything.

## Staking tiers

Providers can stake FTNS to boost their yield rate:

| Tier | Stake | Yield boost |
|------|-------|-------------|
| Casual | 0 FTNS | 1.0× |
| Pledged | 100 FTNS | 1.25× |
| Dedicated | 1,000 FTNS | 1.5× |
| Sentinel | 10,000 FTNS | 2.0× + aggregator fees |

`staking_manager.py` enforces stake locks and slashing for misbehavior.

## Key modules

| File | Purpose |
|------|---------|
| `atomic_ftns_service.py` | Atomic ledger operations with DAG signatures |
| `database_ftns_service.py` | Persistent ledger backed by SQLAlchemy |
| `ftns_service.py` | High-level FTNS API used by the rest of the codebase |
| `contributor_manager.py` | Tracks contributor status and yield multipliers |
| `dynamic_supply_controller.py` | Algorithmic supply adjustments under governance |
| `anti_hoarding_engine.py` | Circulation health checks |
| `emergency_protocols.py` | Governance-gated emergency halts |
| `ftns_budget_manager.py` | Per-query budget envelopes |
| `staking_manager.py` | Staking tiers + slashing |
| `liquidity_provenance.py` | Provenance tracking for derivative content |
| `strategic_provenance.py` | Parent-chain royalty distribution |

## CLI

```bash
prsm ftns yield-estimate --hours 8 --stake 1000   # Monthly earnings estimate
prsm ftns balance                                 # Check your balance
prsm node benchmark                               # See your hardware tier
```

See [`docs/FTNS_API_DOCUMENTATION.md`](../../../docs/FTNS_API_DOCUMENTATION.md) for the full FTNS API reference.
