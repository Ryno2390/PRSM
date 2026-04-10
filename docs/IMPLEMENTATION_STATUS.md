# PRSM Implementation Status

[![Version](https://img.shields.io/badge/version-1.7.0-blue.svg)](https://pypi.org/project/prsm-network/)
[![Rings](https://img.shields.io/badge/rings-10%2F10%20shipped-brightgreen.svg)](#sovereign-edge-ai-rings)

This document tracks which subsystems ship with PRSM and how they map to the 10-ring Sovereign-Edge AI architecture.

> **Scope context (v1.6.0+):** PRSM is a P2P infrastructure protocol for open-source collaboration — not an AGI framework. The v1.6.0 release removed ~210K LoC of legacy AGI scaffolding (old NWTN orchestrator, teacher/distillation framework, multi-agent reasoning pipeline, voicebox, etc.). The entries below describe the current product, not historical scope.

---

## What you can do with a fresh `pip install prsm-network`

- `prsm node start` — join the live bootstrap network and run a node
- `prsm compute run --query "..." --budget 1.0` — submit queries through the Ring 1-10 pipeline
- `prsm compute quote "..."` — cost estimate before committing FTNS
- `prsm storage upload ./file.parquet --royalty-rate 0.05` — publish data through the ContentStore
- `prsm node benchmark` / `prsm ftns yield-estimate` — hardware tier + provider earnings estimate
- `prsm mcp-server` — expose 16 PRSM tools to any MCP-compatible LLM
- Python / JavaScript / Go SDKs — all three published

---

## Sovereign-Edge AI Rings

| Ring | Name | Status | Key Delivery | Primary Source Path |
|------|------|--------|--------------|--------------------|
| 1 | The Sandbox | Shipped | Wasmtime WASM runtime, hardware profiler, compute tiers T1-T4 | `prsm/compute/wasm/` |
| 2 | The Courier | Shipped | Mobile agent dispatch, gossip-based bidding, escrow settlement | `prsm/compute/agents/` |
| 3 | The Swarm | Shipped | Semantic vector sharding, parallel map-reduce, quorum aggregation | `prsm/compute/swarm/` |
| 4 | The Economy | Shipped | PCU pricing menu, prosumer staking tiers, yield estimation, ContentStore royalty splits | `prsm/economy/tokenomics/` |
| 5 | Agent Forge | Shipped (re-scoped) | WASM mobile agent runtime + MCP tools — reasoning happens in the caller's third-party LLM | `prsm/compute/agents/` |
| 6 | The Polish | Shipped | Dynamic gas pricing, RPC failover, settler signature verification, CLI UX | — |
| 7 | The Vault | Shipped | TEE runtime abstraction, differential privacy noise injection | `prsm/compute/tee/` |
| 8 | The Shield | Shipped | Tensor-parallel model sharding, randomized pipelines, collision detection | `prsm/compute/model_sharding/` |
| 9 | The Mind | Training pipeline only | `AgentTrace` collection, quality evaluation, JSONL export — the fine-tuned NWTN LLM itself is future work | `prsm/compute/nwtn/training/` |
| 10 | The Fortress | Shipped | Integrity verification, privacy budget tracking, hash-chained audit log | `prsm/compute/security/` |

### End-to-End Flow

```
Third-party LLM (Claude / GPT / local) calls prsm_analyze via MCP

  → LLM:     Decomposes the query into WASM mobile-agent instructions
  → Ring 3:  Finds relevant semantic shards by embedding similarity
  → Ring 4:  Quotes cost (compute + data + network fee)
  → Ring 3:  Fans out parallel agents to shard-holding nodes
  → Ring 2:  Each agent dispatched via gossip bidding
  → Ring 1:  Executed in WASM sandbox on provider hardware
  → Ring 7:  Differential privacy noise applied to intermediate activations
  → Ring 3:  Results aggregated when quorum met
  → Ring 4:  FTNS settled (80% data owner / 15% compute / 5% treasury)
  → Ring 9:  AgentTrace saved for future NWTN training corpus

  ← Result returned to the LLM for final synthesis
```

---

## Subsystem Status

### Core Protocol

| Subsystem | Status | Notes |
|-----------|--------|-------|
| P2P Node Infrastructure | Shipped | Identity, transport, discovery, gossip — `prsm/compute/federation/` |
| FTNS DAG Ledger | Shipped | SQLite, atomic ops, Ed25519 signatures |
| Compute Job Pipeline | Shipped | Submit → accept → execute → pay |
| Native ContentStore | Shipped | Post-v1.5.0 native storage replaces IPFS dependency at the app layer (kept as optional backend) |
| Storage proofs | Shipped | Challenge/response proofs with replication tracking |
| Content Economy | Shipped | Royalty distribution (80/15/5 Phase 4 model), multi-party escrow, replication enforcement |
| Web Onboarding UI | Shipped | 6-step wizard at `/onboarding/` |
| Rate Limiting | Shipped | Per-IP + per-user + per-endpoint; Redis-backed when `REDIS_URL` is set |
| Circuit Breakers | Library only | `prsm/core/circuit_breaker.py` is a fully implemented adaptive state machine (CLOSED/OPEN/HALF_OPEN, sliding-window failure rate, configurable thresholds). It is **not currently wired** into any live HTTP client — there is no first-party LLM client to protect since `node.agent_forge` was removed in v1.6.0. Available for downstream integrations via `from prsm.core.circuit_breaker import get_breaker`. |
| OpenTelemetry Tracing | Shipped | Console / Jaeger / OTLP via `OTEL_EXPORTER` env var |
| Secrets Management | Shipped | Centralized `SecretsManager` with required-variable validation |
| Alembic Migrations | Shipped | Covers all ORM tables |

### Sovereign-Edge AI (Rings 1-10)

| Subsystem | Status | Notes |
|-----------|--------|-------|
| WASM Sandbox | Shipped | Wasmtime runtime, fuel-limited, memory-capped |
| Hardware Profiler | Shipped | TFLOPS, GPU, thermal, TEE detection |
| Mobile Agent Dispatch | Shipped | Gossip bidding, WebSocket binary transfer, escrow |
| Semantic Sharding | Shipped | Centroid-based clustering, cosine similarity search |
| Swarm Map-Reduce | Shipped | Parallel fan-out, quorum-based aggregation |
| Hybrid Pricing | Shipped | PCU menu, data market, spot arbitrage, 80/15/5 splits |
| Prosumer Staking | Shipped | 4 tiers (Casual → Sentinel), yield estimation |
| Confidential Compute | Shipped | TEE abstraction, DP noise (configurable ε) |
| Model Sharding | Shipped | Tensor parallelism, randomized pipelines |
| Collusion Detection | Shipped | DP-noise-aware output comparison |
| NWTN Training Pipeline | Shipped | AgentTrace → JSONL export → model card — no fine-tuned model yet |
| Security Audit Tools | Shipped | Integrity verification, privacy budget, audit chain |
| MCP Server | Shipped | 16 tools exposed via `prsm mcp-server` / `prsm/mcp_server.py` |
| Python SDK | Published | `pip install prsm-network` (v1.7.0) |
| JavaScript/TypeScript SDK | Published | `npm install prsm-sdk` |
| Go SDK | Published | `go get github.com/Ryno2390/PRSM/sdks/go` |
| Payment Gateway (Stripe/PayPal) | Code complete — needs keys | See `prsm/economy/payments/fiat_gateway.py` |
| FTNS Token (Base mainnet) | Deployed | `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` |
| Chronos FTNS↔USD/USDT bridge | Shipped (mock exchanges) | `prsm/compute/chronos/` — routing/clearing/cashout API real; Coinbase/Binance/Kraken integrations are sandbox mocks pending API keys |

---

## Known Live Infrastructure

| Resource | Value |
|----------|-------|
| Bootstrap node | `wss://bootstrap1.prsm-network.com:8765` |
| FTNS live token (Base mainnet) | `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` |
| Docker compose (bootstrap) | `docker/docker-compose.bootstrap.yml` |
| Deployment guide | [`docs/BOOTSTRAP_DEPLOYMENT_GUIDE.md`](BOOTSTRAP_DEPLOYMENT_GUIDE.md) |
| Operator guide | [`docs/OPERATOR_GUIDE.md`](OPERATOR_GUIDE.md) |

---

## Intentionally Deferred

| Item | Reason |
|------|--------|
| Fine-tuned NWTN LLM | Requires a large corpus of production `AgentTrace` records. Ring 9 ships only the training pipeline today. |
| KYC/AML for fiat gateway | Legal/regulatory — requires counsel |
| Multi-region bootstrap auto-scaling | Operational — requires live AWS/GCP/Azure accounts |
| External security audit | Pre-Series-A milestone |

---

## What Was Removed in v1.6.0

The v1.6.0 "scope alignment" release removed roughly 210K LoC of legacy AGI-era scaffolding that no longer fit the P2P infrastructure thesis. Specifically:

- Old NWTN orchestrator and meta-reasoning engine (`prsm/compute/nwtn/orchestrator.py`, `complete_nwtn_pipeline_v4.py`, etc.)
- NWTN voicebox / stochastic-parrot detector / potemkin analyzer
- Phase 10 NWTN Agent Team (BSC / Active Whiteboard / Live Scribe / OpenClaw session manager)
- Teacher-model framework (`prsm/compute/teachers/`, including SEAL / RLVR / ReSTEM)
- Automated distillation platform (`prsm/compute/distillation/`)
- Recursive self-improvement / evolution framework (`prsm/compute/evolution/`, `prsm/compute/improvement/`)
- Legacy AGI-era Agent Forge (5-layer Architect → Prompter → Router → Compiler → Executor pipeline)
- Marketplace for AI models (replaced by the Ring 4 ContentStore + royalty pipeline)
- AGI safety circuit breakers and output gating

See [`CHANGELOG.md`](../CHANGELOG.md) v1.6.0 entry for the full removal list. The Ring 5 name "Agent Forge" was retained, but the current implementation is a WASM mobile-agent runtime — reasoning is performed by the caller's third-party LLM.
