# FTNS Tokenomics

FTNS (Fungible Tokens for Node Support) is PRSM's economic primitive — the token that compensates contributors for sharing latent storage, compute, and data. This module implements the Ring 4 tokenomics layer: ledger, pricing, staking, and settlement.

## Core principle

FTNS supply follows the deployed v1 **HALVING-TO-CAP** schedule. Emission opens at ~31.5M FTNS/yr (1 FTNS/sec) and halves every 4 years toward zero, capped at a 900M emission ceiling. Combined with the 100M genesis allocation, that is a **1B FTNS MAX_SUPPLY hard cap** (~352M asymptotic circulating supply). There is no ICO. Emission is distributed to contributors via the on-chain `CompensationDistributor` (`0xa9551F5a3AeAB39cc8315AcD8caC2886Bd04f244`) on a **50/30/20 split** (creator / operator / grant). There is **no burn-on-use** — the `burnFrom` entrypoint exists for the fiat bridge only.

## Token facts

| | |
|---|---|
| Symbol | FTNS |
| Decimals | 18 |
| Live contract | `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` (Base mainnet, chain 8453 — LIVE) |
| Max supply | 1B FTNS hard cap (900M emission ceiling + 100M genesis); ~352M asymptotic |
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

## Staking (utility-only)

Staking is **utility-only**: a lock-based stake confers service discounts and dispatch priority — there is **no token yield, APY, or yield multiplier**. The deployed v1 emission split has no staker-yield pool to fund a return from; staking buys priority access, not income. Benefits scale with the lock commitment:

| Lock | Service discount | Dispatch priority |
|------|------------------|-------------------|
| None / 0 days | — | — |
| 30 days | 2% | +0.10 |
| 90 days | 5% | +0.25 |
| 365 days | 10% | +0.50 |

`staking_manager.py` enforces the utility-benefit lock and slashing for misbehavior.

## Key modules

| File | Purpose |
|------|---------|
| `atomic_ftns_service.py` | Atomic ledger operations with DAG signatures |
| `database_ftns_service.py` | Persistent ledger backed by SQLAlchemy |
| `ftns_service.py` | High-level FTNS API used by the rest of the codebase |
| `contributor_manager.py` | Tracks contributor status and emission-share eligibility |
| `dynamic_supply_controller.py` | Algorithmic supply adjustments under governance |
| `anti_hoarding_engine.py` | Circulation health checks |
| `emergency_protocols.py` | Governance-gated emergency halts |
| `ftns_budget_manager.py` | Per-query budget envelopes |
| `staking_manager.py` | Staking tiers + slashing |
| `liquidity_provenance.py` | Provenance tracking for derivative content |
| `strategic_provenance.py` | Parent-chain royalty distribution |

## CLI

```bash
prsm node earnings              # This node's emission/royalty earnings dashboard
prsm node wallet-balance        # Live Base mainnet FTNS + USDC + ETH balances
prsm node benchmark             # See your hardware tier
```

See [`docs/FTNS_API_DOCUMENTATION.md`](../../../docs/FTNS_API_DOCUMENTATION.md) for the full FTNS API reference.
