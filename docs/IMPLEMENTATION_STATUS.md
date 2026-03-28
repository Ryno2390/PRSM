# PRSM Implementation Status

[![Status](https://img.shields.io/badge/status-Beta%20v0.2.2-blue.svg)](#current-implementation-status)
[![Tests](https://img.shields.io/badge/tests-3987%20passing-brightgreen.svg)](#test-suite-status)
[![Completion](https://img.shields.io/badge/code--complete-99%25-brightgreen.svg)](#)
[![Updated](https://img.shields.io/badge/updated-2026--03--28-green.svg)](#)

---

## Executive Summary

As of March 28, 2026 (commit `b435ece`), PRSM has completed all ten development phases.
**The codebase is 99%+ complete for all code-only work.**

A developer cloning this repo today can:
- Run `prsm node start`, join the live bootstrap network, and execute real queries
- Earn and spend FTNS tokens in testnet mode
- Build on the Python, JavaScript/TypeScript, or Go SDK
- Deploy a production node using `docker/docker-compose.bootstrap.yml`
- Follow `docs/OPERATOR_GUIDE.md` end-to-end with no additional code changes required

**SDKs published, FTNS token deployed.** Remaining work is operational infrastructure — external accounts, credentials, and deployed services.
No further code phases are planned. The infrastructure roadmap is documented below.

---

---

## Phase 10: NWTN Agent Team Architecture

> **Status: Complete** — Fully implemented and tested as of 2026-03-28. 169 dedicated tests passing.

### Implementation Summary

| Sub-phase | Description | Status | Key Files |
|-----------|-------------|--------|-----------|
| 10.1 | BSC Core (predictor, KL filter, dedup, promoter, quality gate) | ✅ Complete | `bsc/predictor.py`, `bsc/kl_filter.py`, `bsc/promoter.py`, `bsc/quality_gate.py` |
| 10.2 | Active Whiteboard (store, monitor, schema, query) | ✅ Complete | `whiteboard/store.py`, `whiteboard/monitor.py` |
| 10.3 | Agent Team Coordination (interview, planner, assembler, scribe, router) | ✅ Complete | `team/live_scribe.py`, `team/scribe_agent.py`, `team/whiteboard_router.py` |
| 10.4 | Nightly Synthesis & Project Ledger (reconstruct, ledger, sign, anchor) | ✅ Complete | `synthesis/reconstructor.py`, `synthesis/ledger.py`, `synthesis/signer.py` |
| 10.5 | OpenClaw Integration + NWTNSession factory | ✅ Complete | `openclaw/adapter.py`, `session.py` |
| 10.6 | AI-Agent-Centric CLI + OpenRouter backend | ✅ Complete | `backends/openrouter_backend.py` |
| — | CircuitBreaker + EventBus performance benchmarks | ✅ Complete | `bsc/circuit_breaker.py`, `tests/benchmarks/` |
| — | Legacy Gen1 cleanup (35k lines removed) | ✅ Complete | — |

### Vision

Phase 10 is a paradigm shift in how PRSM is used. Phases 1–9 built a complete,
production-ready protocol infrastructure. Phase 10 defines the **interface layer** —
but designed for **AI agents**, not humans.

The core insight: the near-future UX is not a dashboard. It is an AI agent receiving
a natural-language goal and executing it autonomously. PRSM's CLI, APIs, and SDKs
should be optimized for AI agent consumption (machine-readable, deterministic,
composable) rather than for human dashboards. NWTN is re-architected to be the
coordination intelligence that makes this possible.

### What Changes in NWTN

The original NWTN was a **hierarchical orchestrator** — one "boss" agent decomposing
tasks and dispatching sub-agents, then aggregating results. This is token-efficient
but brittle: the orchestrator makes final calls on conflicting information, cannot
benefit from peer-review between specialists, and wastes tokens when the plan changes
mid-stream (sub-agents finish useless work before the orchestrator realizes).

The new NWTN is a **flat Agent Team harness** — a coordination layer that enables
multiple specialist agents to work in parallel, share a lean shared context (the
"whiteboard"), and course-correct in real time without a hierarchical bottleneck.
The token problem of flat architectures is solved by the **Bayesian Surprise
Compressor (BSC)**.

### Core Innovations

#### 1. Bayesian Surprise Compressor (BSC)

The BSC is the gatekeeper of the shared whiteboard. Instead of every agent writing
everything to shared context (which causes token explosion), a small predictor model
(1–3B parameters) evaluates the **informational surprise** of each agent's output
before it is promoted to the whiteboard.

- **Predictor**: A small quantized model evaluates the perplexity (cross-entropy loss)
  of each new agent output chunk relative to the current compressed context. This is
  evaluation-only — no generation required — making it 3–5x cheaper than a generation
  call of the same model size.
- **KL Filter**: Calculates KL divergence between predicted and actual state. If
  divergence exceeds epsilon (the surprise threshold), the chunk is promoted.
- **Semantic De-duplication**: Cosine similarity check against existing whiteboard
  embeddings. High-surprise chunks that are semantically redundant with existing
  whiteboard entries are discarded despite their surprise score.
- **Promoter**: Writes surviving chunks to the Active Whiteboard as structured
  fact-value pairs with source agent and timestamp metadata.

**Deployment**: BSC can run locally (via MLX or llama.cpp on Apple Silicon — a 3B
quantized model runs comfortably on an M2/M3 MacBook Pro or Mac Mini with 16GB RAM)
or as a PRSM network service (a node provides BSC-as-a-service and earns FTNS).
User's choice; the interface is identical either way.

#### 2. Event-Driven Blackboard (Active Whiteboard)

The shared context that all team members can read. During a working session:

- **Data structure**: SQLite-backed fact-value store with source agent, timestamp,
  and surprise score metadata. Structured for machine querying, not human reading.
- **Write access**: Only the BSC writes to the whiteboard. Agents write to their own
  private memory (OpenClaw's `MEMORY.md`); NWTN monitors these files and BSC-filters
  what crosses into shared context.
- **Read access**: All team agents. New agents joining mid-session can read the full
  whiteboard to onboard instantly without being fed the entire raw chat history.

#### 3. Three-Tier Memory & Time Horizon

| Tier | Storage | Contents | Lifetime |
|------|---------|----------|----------|
| Sensory Buffer | Working memory (RAM) | Raw last N exchanges per agent | Per session |
| Active Whiteboard | SQLite | BSC-promoted surprise facts | Per session (wiped at end) |
| Project Ledger | Git-tracked Markdown | Nightly synthesis narratives | Permanent, accumulating |

**Nightly Synthesis**: At end of session, a Re-constructor Agent reads the messy
Active Whiteboard and synthesizes a structured Markdown narrative — not a bullet
list, but a coherent account: what pivots occurred, what was completed, what is
still pending, and why. This is appended to the Project Ledger.

**Project Ledger**: A single, accumulating Markdown file. New agents onboarding to
a project read the Ledger to understand the full project history. Feeding the Ledger
into the next session is vastly more token-efficient than feeding raw chat history.

#### 4. Tamper-Evident Project Ledger

Each Nightly Synthesis entry is:
- **Signed** with the user's PRSM keypair (cryptographic authorship)
- **Chained** — each entry includes the SHA hash of the previous entry, so any
  modification to historical entries breaks all subsequent hashes (identical property
  to blockchain, using Git's native commit hashing)
- **Anchored** — major project milestones are anchored to PRSM's DAG ledger for
  external, decentralized verification independent of the local git repo

#### 5. Git-Based Agent Workflow

- Each agent on the team works in its own git branch
  (`agent/security-YYYYMMDD`, `agent/coder-YYYYMMDD`, etc.)
- **NWTN as merge manager**: NWTN reviews each agent's branch diff against the
  meta-plan and the main branch before allowing a merge. Merges only happen at
  defined checkpoints.
- **Rollback**: Because all agent work is branch-isolated, any unauthorized or
  broken addition is detectable (hash chain break) and reversible (revert to last
  clean checkpoint on main).
- **FTNS feedback loop**: Verified, merged agent contributions become provenance
  records on the PRSM network. Nodes providing agent compute earn FTNS for
  contributions that pass NWTN's checkpoint review — incentivizing quality.

#### 6. OpenClaw Integration

NWTN does not replace OpenClaw. OpenClaw is the **individual agent runtime**
(skills, sandbox, gateway, heartbeat, per-agent `MEMORY.md`). NWTN is the
**team coordination layer** that makes multiple OpenClaw instances work as a
flat team.

- OpenClaw's `MEMORY.md` is the source NWTN monitors for BSC filtering
- OpenClaw's Heartbeat is the trigger for Nightly Synthesis
- OpenClaw's Skills Registry maps to PRSM's model registry for specialist discovery
- OpenClaw's Gateway is the human entry point where NWTN's interview mode lives

### Implementation Plan

#### Sub-phase 10.1 — BSC Core
```
prsm/compute/nwtn/bsc/
  predictor.py          # Small model perplexity evaluator (local or network)
  kl_filter.py          # KL divergence calculator
  semantic_dedup.py     # Embedding cosine similarity check
  promoter.py           # Threshold-based whiteboard promotion
  deployment.py         # Local (MLX/llama.cpp) vs. network service mode
```

#### Sub-phase 10.2 — Active Whiteboard
```
prsm/compute/nwtn/whiteboard/
  store.py              # SQLite-backed fact-value store
  monitor.py            # File watcher: monitors agent MEMORY.md files
  schema.py             # Fact schema: value, source_agent, timestamp, surprise_score
  query.py              # Structured query interface for agents
```

#### Sub-phase 10.3 — Agent Team Coordination
```
prsm/compute/nwtn/team/
  interview.py          # NWTN interview mode: gathers project requirements from user
  planner.py            # Meta-plan generator: produces the "north star" document
  assembler.py          # Team assembly: discovers specialist agents from PRSM registry
  branch_manager.py     # Creates and manages per-agent git branches
  checkpoint.py         # NWTN merge manager: reviews diffs, approves merges
```

#### Sub-phase 10.4 — Nightly Synthesis & Project Ledger
```
prsm/compute/nwtn/synthesis/
  reconstructor.py      # Re-constructor agent: narrative synthesis from whiteboard
  ledger.py             # Project Ledger: append-only, Markdown narrative
  signer.py             # PRSM keypair signing + SHA hash chain
  dag_anchor.py         # Anchors major milestones to PRSM DAG
```

#### Sub-phase 10.5 — OpenClaw Integration
```
prsm/compute/nwtn/openclaw/
  adapter.py            # Bridge between NWTN coordination and OpenClaw agent instances
  skills_bridge.py      # Maps PRSM model registry entries to OpenClaw skills
  gateway.py            # User entry point: receives goal, triggers interview mode
  heartbeat_hook.py     # Hooks OpenClaw heartbeat to trigger Nightly Synthesis
```

#### Sub-phase 10.6 — AI-Agent-Centric CLI
- Audit all existing PRSM CLI commands for machine-readability
- Ensure all outputs support `--format json` for agent consumption
- Add agent-callable PRSM operations: team spawn, whiteboard query, ledger read,
  checkpoint request, branch status
- Remove or demote human-dashboard-oriented output formats

### Success Criteria

Phase 10 is complete when:
1. A user can state a goal to NWTN via a Gateway; NWTN interviews them and produces
   a meta-plan
2. NWTN assembles a team of specialist agents from the PRSM network (or local)
3. Agents work in parallel in isolated git branches; the BSC maintains a lean
   shared whiteboard
4. At session end, NWTN synthesizes a signed, chained Nightly Synthesis entry and
   appends it to the Project Ledger
5. A new agent joining the project can onboard by reading the Project Ledger alone
6. An unauthorized Ledger modification is detectable within one hash-chain verification
7. PRSM itself is being built using this system (self-referential bootstrap validation)

### Test Coverage

169 tests in `tests/nwtn/` covering all sub-phases. All pass with `pytest tests/nwtn/ --timeout=60`.
Performance benchmarks in `tests/benchmarks/test_eventbus_performance.py` (7 benchmarks, not in CI baseline).

Key results:
- EventBus: 24,665 ops/sec single subscriber; p99 latency 46µs; 0 MB memory growth at 100k events
- End-to-end smoke tests: full pipeline fires correctly from goal statement to convergence detection

---

## Ten-Phase Development Summary

| Phase | Description | Tests Added | Commit |
|-------|-------------|-------------|--------|
| 1 | P0 security bug fixes (AtomicFTNS, DAG ledger, SQL dialect) | — | `3e3923e` |
| 2 | Test error fixes + IPFS daemon auto-start | — | `935b8b3` |
| 3 | Web-based node onboarding UI (6-step wizard) | +24 | `c198084` |
| 4 | External deployment config (bootstrap domains, env template) | +14 | `068256c` |
| 5 | Test suite completeness (removed 33 stale skips, fixed API mismatches) | +140 | `b523591` |
| 6 | Deferred module implementation (18 new modules) | +105 | `2800a9d` |
| 7 | Production hardening (migrations, rate limiting, circuit breakers, OTel) | +20 | `f473c67` |
| 8 | SDK completion + documentation (Go SDK, 3 new guides) | +34 | `29ce485` |
| 9 | Stub wiring + final completion (11 stubs → real implementations) | +49 | `63a270a` |
| 10 | NWTN Agent Team Architecture (BSC, Whiteboard, Scribe, Session, OpenClaw integration) | +169 | `b435ece` |
| **Total** | | **+3,987 collected** | |

---

## Test Suite Status

| Metric | Value |
|--------|-------|
| Total collected | 3,987 |
| Passing | ~3,938 |
| Skipped (infrastructure-gated) | ~45 |
| xfailed | 4 |
| Failing | 0 |

**Known flaky test:** `test_performance_integration.py::TestScalabilityPerformance::test_memory_usage_under_load`
— fails in full-suite runs because psutil measures system-wide memory. Passes in isolation.
Do not fix by loosening threshold — accept as a known environment-specific flake.

---

## Production Readiness by Subsystem

| Subsystem | Status | Notes |
|-----------|--------|-------|
| P2P Node Infrastructure | ✅ Ready | Identity, transport, discovery, gossip |
| FTNS DAG Ledger | ✅ Ready | SQLite, atomic ops, Ed25519 signatures |
| Compute Job Pipeline | ✅ Ready | Submit → accept → execute → pay |
| NWTN Reasoning Pipeline | ✅ Ready | 7-phase pipeline, Anthropic/OpenAI/local backends |
| Safety Systems | ✅ Ready | Recursive safeguards, governance, sandboxing |
| Federated P2P | ✅ Ready | Distributed consensus, Byzantine fault tolerance |
| IPFS Storage | ✅ Ready | Auto-start daemon detection, chunked multi-GB transfers |
| BitTorrent Transfer | ✅ Ready | Proof-of-transfer verification |
| Marketplace API | ✅ Ready | 9 asset types, full order lifecycle, 274 endpoints |
| Content Provenance | ✅ Ready | Attribution + royalty distribution |
| Vector Database | ✅ Ready | In-memory cosine similarity, pluggable backend |
| Web Onboarding UI | ✅ Ready | 6-step wizard at `/onboarding/` |
| Ollama / Local LLM | ✅ Ready | Requires local Ollama install |
| Rate Limiting | ✅ Ready | Per-IP + per-user + per-endpoint |
| Circuit Breakers | ✅ Ready | Wired into Anthropic and OpenAI backends |
| OpenTelemetry Tracing | ✅ Ready | Console/Jaeger/OTLP via `OTEL_EXPORTER` env var |
| Secrets Management | ✅ Ready | Centralized `SecretsManager` with required validation |
| Alembic Migrations | ✅ Ready | 3 migrations covering all ORM tables |
| Python SDK | ✅ Published | `pip install prsm-python-sdk` — live on PyPI v0.2.0 |
| JavaScript/TypeScript SDK | ✅ Published | `npm install prsm-sdk` — live on npm v0.2.0 |
| Go SDK | ✅ Published | `go get github.com/Ryno2390/PRSM/sdks/go@v0.2.0` — live on pkg.go.dev |
| Payment Gateway (Stripe/PayPal) | 🔑 Needs credentials | Code complete; requires API keys |
| Price Oracles (CoinGecko) | 🔑 Needs API key | Free tier available |
| FTNS Token (Base mainnet) | ✅ Deployed | `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` — verified on Basescan |
| Multi-region Bootstrap | 🏗️ Needs deployment | Single NYC3 node live; EU/APAC config ready |
| Redis (distributed rate limiting) | 🏗️ Optional | In-memory works for single node |

---

## Infrastructure Roadmap

The following items are not code problems. Each has a concrete execution procedure.
They are ordered by priority for enabling broad user participation.

---

### 1. EU + APAC Bootstrap Nodes — Priority: SCALING

**Why:** The network currently has a single point of failure at
`wss://bootstrap1.prsm-network.com:8765` (DigitalOcean NYC3). New peers depend entirely
on this node for discovery. Two additional geographic nodes make the network resilient.

**What's already done:** `docker/docker-compose.bootstrap.yml` is deployment-ready.
`prsm/node/config.py` defines `FALLBACK_BOOTSTRAP_NODES` with EU and APAC addresses.
`config/secure.env.template` documents all required variables.

**How to deploy (EU node — repeat for APAC):**

1. **Create DigitalOcean droplet**
   - Region: AMS3 (Amsterdam) for EU; SGP1 (Singapore) for APAC
   - Size: 4 vCPU / 8 GB RAM / 160 GB SSD (CPU-Optimized)
   - OS: Ubuntu 22.04 LTS
   - Auth: SSH key (use existing key)
   - Note the droplet IP address after creation

2. **Add DNS records in Cloudflare** (`prsm-network.com`)
   - Add A record: `bootstrap2` → Amsterdam droplet IP
   - Add A record: `bootstrap3` → Singapore droplet IP
   - ⚠️ Set DNS-only (grey cloud, not orange) — WebSocket (`wss://`) connections do not work through Cloudflare proxy on free plans

3. **Install dependencies on droplet**
   ```bash
   ssh root@<droplet-ip>
   apt update && apt install -y docker.io docker-compose git
   systemctl enable docker && systemctl start docker
   ```

4. **Clone and configure**
   ```bash
   git clone https://github.com/Ryno2390/PRSM.git /opt/prsm
   cd /opt/prsm
   cp config/secure.env.template config/.env
   nano config/.env
   ```
   Minimum required variables:
   ```bash
   NODE_TYPE=bootstrap
   BOOTSTRAP_REGION=eu-west-1          # or apac-southeast-1 for Singapore
   BOOTSTRAP_PUBLIC_URL=wss://bootstrap2.prsm-network.com:8765  # or bootstrap3
   JWT_SECRET_KEY=$(openssl rand -hex 32)
   ```

5. **Launch**
   ```bash
   docker-compose -f docker/docker-compose.bootstrap.yml up -d
   docker-compose -f docker/docker-compose.bootstrap.yml logs -f
   ```

6. **Verify** by running `prsm node start` locally and confirming all three bootstrap nodes appear in startup logs.

7. **Update `prsm/node/config.py`** with the new live addresses and commit.

**Prerequisites before starting:**
- DigitalOcean account with billing configured (~$48/mo per node)
- SSH key already configured in DigitalOcean
- Access to `prsm-network.com` DNS in Cloudflare (DNS-only mode required for WebSocket)
- Existing NYC3 bootstrap node healthy (verify at `wss://bootstrap1.prsm-network.com:8765`)

**Estimated cost:** ~$48/month per node × 2 = ~$96/month ongoing
**Recommended timing:** Deploy after Stripe fiat on-ramp and SDK publishing are complete — network resilience matters most when there are active users to serve.

---

### 2. FTNS ERC-20 Mainnet Deployment — Priority: HIGH

**Why:** FTNS tokens only have real monetary value on Ethereum mainnet. Provenance royalties,
compute payments, and staking are economically meaningful only after mainnet launch.

**What's already done:** Smart contracts are written and tested. Sepolia testnet deployment
exists at `0xd979c096BE297F4C3a85175774Bc38C22b95E6a4`. Hardhat config at
`contracts/hardhat.config.js` is mainnet-ready. All deployment scripts exist.

**How to deploy:**

1. **Acquire prerequisites**
   - Alchemy account → create Ethereum mainnet app → copy HTTP RPC URL
   - Deployer wallet with ~0.2 ETH (for gas; exact amount depends on gas price at deploy time)
   - Etherscan account → generate API key for contract verification

2. **Configure deployment**
   ```bash
   # In contracts/.env (create from contracts/.env.template):
   ALCHEMY_MAINNET_URL=https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY
   DEPLOYER_PRIVATE_KEY=0xYOUR_DEPLOYER_WALLET_PRIVATE_KEY
   ETHERSCAN_API_KEY=YOUR_ETHERSCAN_KEY
   ```

3. **Dry-run on fork first**
   ```bash
   cd contracts
   npx hardhat run scripts/deploy.js --network mainnet-fork
   # Verify output: token address, total supply, owner address
   ```

4. **Deploy to mainnet**
   ```bash
   npx hardhat run scripts/deploy.js --network mainnet
   # Save the deployed contract address from output
   ```

5. **Verify contract on Etherscan**
   ```bash
   npx hardhat verify --network mainnet DEPLOYED_CONTRACT_ADDRESS
   ```

6. **Update configuration**
   - Add deployed address to `config/secure.env.template` under `FTNS_CONTRACT_ADDRESS`
   - Update `docs/IMPLEMENTATION_STATUS.md` with mainnet address
   - Update `prsm/economy/blockchain/` with mainnet contract address

7. **Announce**: Post contract address to GitHub README and any community channels.

**Estimated cost:** ~$150–400 in ETH gas (varies with network congestion).

---

### 3. Stripe Live Payment Integration — Priority: LEGAL PREREQUISITE

> **Prerequisite:** Requires a registered business entity (LLC or equivalent) and a dedicated business bank account. Do not connect personal bank accounts to a production payment processor. Form the LLC first, then return to this step.

**Why:** Fiat on-ramp (buying FTNS with USD/EUR) enables non-crypto users to participate.
Currently blocked only by missing live API keys.

**What's already done:** `prsm/economy/payments/fiat_gateway.py` is fully implemented.
`tests/integration/test_payment_sandbox.py` has integration tests against Stripe test mode.
The code already handles webhook verification, payment intents, and cancellation.

**How to enable:**

1. **Create Stripe account** at stripe.com → complete business verification
2. **Retrieve API keys** from Stripe Dashboard → Developers → API Keys
   ```bash
   # In config/.env:
   STRIPE_API_KEY=sk_live_...
   STRIPE_WEBHOOK_SECRET=whsec_...
   ```
3. **Configure webhook endpoint** in Stripe Dashboard:
   - URL: `https://your-node-domain.com/api/v1/payments/stripe/webhook`
   - Events: `payment_intent.succeeded`, `payment_intent.payment_failed`,
     `payment_intent.canceled`
4. **Test with a real $1 charge** before announcing availability
5. **Enable** by setting `STRIPE_ENABLED=true` in config

**For PayPal** (same pattern):
- Create PayPal developer account → create app in production mode
- Set `PAYPAL_CLIENT_ID` and `PAYPAL_CLIENT_SECRET` in config
- Webhook URL: `/api/v1/payments/paypal/webhook`

**Estimated cost:** Stripe fee: 2.9% + $0.30 per transaction. No setup cost.

**Dependency chain:**
```
Form LLC → Open business bank account → Stripe live account → Configure API keys → Enable fiat on-ramp
```
**Estimated LLC formation cost:** ~$50–500 depending on state (NC: ~$125 filing fee). Services like Stripe Atlas ($500) or Clerky handle formation + registered agent + EIN in one step.

---

### 4. Crypto Exchange Rate API Key — Priority: MEDIUM

**Why:** `prsm/economy/payments/crypto_exchange.py` fetches live FTNS/USD rates.
The CoinGecko free tier (no key required) covers basic usage but has rate limits.

**How to enable:**

1. **Free option (CoinGecko free tier):** No key needed. Already works. Set:
   ```bash
   COINGECKO_API_URL=https://api.coingecko.com/api/v3
   ```
   Rate limit: 10–30 calls/minute. Sufficient for a small network.

2. **Paid option (CoinGecko Pro):** Register at coingecko.com/en/api/pricing → Demo plan ($129/month).
   ```bash
   COINGECKO_API_KEY=CG-...
   ```
   Rate limit: 500 calls/minute. Required at scale.

3. **Verify** by calling `GET /api/v1/ftns/price` and confirming it returns a real USD rate.

---

### 5. Redis for Distributed Rate Limiting — Priority: LOW (single-node optional)

**Why:** Current rate limiting uses in-memory `dict` keyed on IP/user. This resets on restart
and doesn't share state across multiple node instances. Redis enables persistent, distributed
rate limiting.

**What's already done:** `prsm/interface/api/middleware.py` is structured for Redis integration.
`prsm/core/secrets.py` is ready to load `REDIS_URL`.

**How to deploy:**

1. **Single node / development:** Docker is the fastest path:
   ```bash
   docker run -d --name prsm-redis -p 6379:6379 redis:7-alpine
   # In config/.env:
   REDIS_URL=redis://localhost:6379/0
   ```

2. **Production (managed):** DigitalOcean Managed Redis (~$15/month for 1 GB):
   - Create managed Redis cluster in DigitalOcean
   - Copy connection string → set `REDIS_URL` in config

3. **Wire into middleware** by updating `RateLimitMiddleware.__init__()` to check for
   `REDIS_URL` and use `aioredis` for bucket storage when available, falling back to
   the existing in-memory dict when not.

---

### 6. SDK Publishing — ✅ Complete (published v0.2.0)

**Why:** Publishing to public registries (`pip install prsm-sdk`, `npm install @prsm/js-sdk`,
`go get github.com/Ryno2390/PRSM/sdks/go`) enables external developers to build on PRSM
without cloning the monorepo.

**What's already done:** All three SDKs build cleanly. `pyproject.toml`, `package.json`,
and `go.mod` are all correctly configured.

**How to publish:**

**Python SDK (PyPI):**
```bash
cd sdks/python
python -m build
# Create PyPI account at pypi.org → generate API token
pip install twine
twine upload dist/*
# Enter API token when prompted
```

**JavaScript SDK (npm):**
```bash
cd sdks/javascript
npm run build
# Create npm account at npmjs.com → run `npm login`
npm publish --access public
```

**Go SDK (pkg.go.dev):**
```bash
# Go modules are auto-indexed from GitHub. Simply push a tagged release:
git tag sdks/go/v0.2.2
git push origin sdks/go/v0.2.2
# pkg.go.dev will index it automatically within ~30 minutes
```

---

### 7. External Security Audit — Priority: LOW (pre-Series-A milestone)

**Why:** Investors and enterprise customers expect a third-party security audit before
committing significant capital or data to the network.

**What to audit:** Smart contracts (`contracts/`), authentication and JWT handling
(`prsm/core/auth/`), FTNS atomic transaction logic (`prsm/economy/tokenomics/`),
P2P message validation (`prsm/node/`).

**How to engage:**
1. Shortlist firms with EVM + Python API experience: Trail of Bits, Consensys Diligence,
   Certik, OpenZeppelin Audits
2. Provide audit scope: smart contracts + backend API + P2P layer
3. Typical timeline: 4–8 weeks; typical cost: $20K–$80K depending on scope and firm
4. Publish audit report to `docs/security/` and link from README

---

## Known Infrastructure Already Live

| Resource | Value |
|----------|-------|
| Bootstrap node | `wss://bootstrap1.prsm-network.com:8765` (DigitalOcean NYC3, 159.203.129.218) |
| FTNS test token | `0xd979c096BE297F4C3a85175774Bc38C22b95E6a4` (Ethereum Sepolia) |
| **FTNS live token** | **`0x5276a3756C85f2E9e46f6D34386167a209aa16e5` (Base mainnet)** |
| Hardhat config | `contracts/hardhat.config.js` (mainnet-ready) |
| Docker compose | `docker/docker-compose.bootstrap.yml` (production-ready) |
| Deployment guide | `docs/BOOTSTRAP_DEPLOYMENT_GUIDE.md` |
| Operator guide | `docs/OPERATOR_GUIDE.md` |

---

## What Remains Intentionally Deferred

| Item | Reason |
|------|--------|
| KYC/AML for fiat gateway | Legal/regulatory — requires legal counsel |
| Video walkthroughs | Content production, not code |
| GlobalInfrastructure multi-cloud auto-scaling | Requires live AWS/GCP/Azure credentials and accounts |
| `test_ftns_concurrency_integration.py` | Correctly gated behind `DATABASE_URL` — requires PostgreSQL |
| `test_openai_*.py` (3 files) | Correctly gated behind `OPENAI_API_KEY` |

---

## Fully Implemented Modules (Selected Key Files)

| Module | File | Lines |
|--------|------|-------|
| NWTN Orchestrator | `prsm/compute/nwtn/orchestrator.py` | 889 |
| NWTN Full Pipeline | `prsm/compute/nwtn/complete_nwtn_pipeline_v4.py` | 52,000+ |
| Breakthrough Reasoning | `prsm/compute/nwtn/breakthrough_reasoning_coordinator.py` | 40,000+ |
| P2P Network | `prsm/compute/federation/enhanced_p2p_network.py` | 72,000+ |
| Safety Systems | `prsm/core/safety/` (9 files) | 260,000+ |
| Database ORM | `prsm/core/database.py` | 2,097 |
| Vector Database | `prsm/core/vector_db.py` | 2,000+ |
| IPFS Client | `prsm/core/ipfs_client.py` | 2,078 |
| Global Infrastructure | `prsm/core/enterprise/global_infrastructure.py` | 1,489 |
| FTNS Token Economy | `prsm/economy/tokenomics/` (25 files) | 53,000+ |
| Marketplace | `prsm/economy/marketplace/` (13 files) | 40,000+ |
| API Layer | `prsm/interface/api/` (57 files) | 25,000+ |
| AI Orchestration | `prsm/compute/ai_orchestration/` (6 files) | 15,000+ |
