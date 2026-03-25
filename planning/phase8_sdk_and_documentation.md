# Phase 8 — SDK Completion & Documentation

## Overview

Phase 7 hardens the running node. Phase 8 makes PRSM *buildable on* — it completes the
developer SDK layer and produces the documentation that lets non-Ryne humans actually
understand and participate in the network.

**Baseline entering Phase 8:** ~3,730 passing (estimated after Phase 7), 0 failing

**Key discovery before writing this plan:**
- Python SDK (`sdks/python/prsm_sdk/`): complete — 9 modules, 12 unit tests (all mocked)
- JavaScript/TypeScript SDK (`sdks/javascript/src/`): complete — 11 modules, ~12,000 lines
- Go SDK (`sdks/go/`): partial — has client, nwtn, governance; missing ftns, marketplace, storage
- `docs/`: comprehensive technical docs exist; missing non-technical user guide and operator guide
- SDK tests are **unit tests with mocks only** — no integration tests against a live PRSM server

Phase 8 focuses on the genuine gaps: Go SDK completion, SDK integration tests, and the
two missing documentation audiences (non-technical users and node operators).

---

## Step 1 — Python SDK Integration Tests (~3 hr)

### What exists
`sdks/python/tests/test_client.py` — 12 unit tests, all using `AsyncMock` to mock HTTP.
The tests verify client initialization and request construction, but never touch a real
PRSM server.

### Why it matters
An external developer who clones the Python SDK and runs `pytest` should see tests that
actually call the PRSM API — proving the SDK works against real endpoints, not just
against mocked JSON.

### Fix

Create `sdks/python/tests/test_integration.py`:

```python
"""
Python SDK integration tests — runs against a live PRSM server.

Prerequisites:
    - PRSM node running at http://localhost:8000 (or PRSM_TEST_URL env var)
    - PRSM_TEST_API_KEY set, or test user created via /auth/register

Skip gracefully if server is not reachable.
"""
import pytest
import asyncio
import os
import httpx
from prsm_sdk import PRSMClient
from prsm_sdk.exceptions import AuthenticationError

PRSM_URL = os.getenv("PRSM_TEST_URL", "http://localhost:8000")
PRSM_KEY = os.getenv("PRSM_TEST_API_KEY", "")


def prsm_server_available() -> bool:
    try:
        import httpx
        resp = httpx.get(f"{PRSM_URL}/health", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(
    not prsm_server_available(),
    reason="PRSM server not running at localhost:8000 — start node with `prsm node start`"
)
class TestPRSMClientIntegration:

    @pytest.fixture
    async def client(self):
        async with PRSMClient(api_key=PRSM_KEY, base_url=PRSM_URL) as c:
            yield c

    async def test_health_check(self, client):
        health = await client.health()
        assert health["status"] in ("healthy", "degraded")

    async def test_get_ftns_balance(self, client):
        balance = await client.ftns.get_balance()
        assert balance.user_id is not None
        assert float(balance.balance) >= 0

    async def test_list_marketplace_listings(self, client):
        listings = await client.marketplace.list_listings(limit=5)
        assert isinstance(listings, list)

    async def test_submit_and_track_compute_job(self, client):
        job = await client.compute.submit_job(
            job_type="benchmark",
            parameters={"iterations": 1},
        )
        assert job.job_id is not None
        assert job.status in ("submitted", "queued", "running")

    async def test_invalid_api_key_raises(self):
        async with PRSMClient(api_key="invalid_key", base_url=PRSM_URL) as bad_client:
            with pytest.raises(AuthenticationError):
                await bad_client.ftns.get_balance()
```

Also update `sdks/python/README.md` to document how to run integration tests:
```markdown
## Running Integration Tests

Unit tests (no server needed):
    pytest tests/test_client.py tests/test_models.py

Integration tests (requires running node):
    prsm node start &
    PRSM_TEST_API_KEY=your_key pytest tests/test_integration.py
```

---

## Step 2 — Go SDK Completion (~4 hr)

### What exists
| Module | Status |
|--------|--------|
| `client/client.go` | ✅ Complete — Query, EstimateCost, HealthCheck, makeRequest |
| `nwtn/nwtn.go` | ✅ Complete — NWTN pipeline queries |
| `governance/governance.go` | ✅ Complete — governance proposals |
| `types/types.go` | ✅ Complete — shared types |
| `websocket/websocket.go` | ✅ Complete — WebSocket client |
| `ftns/` | ❌ Missing |
| `marketplace/` | ❌ Missing |
| `storage/` | ❌ Missing |

### Fix

**Create `sdks/go/ftns/ftns.go`:**

```go
package ftns

import (
    "context"
    "fmt"
    "github.com/Ryno2390/PRSM/sdks/go/client"
    "github.com/Ryno2390/PRSM/sdks/go/types"
)

type FTNSClient struct {
    client *client.Client
}

func New(c *client.Client) *FTNSClient {
    return &FTNSClient{client: c}
}

func (f *FTNSClient) GetBalance(ctx context.Context) (*types.FTNSBalance, error) {
    var balance types.FTNSBalance
    err := f.client.MakeRequest(ctx, "GET", "/api/v1/ftns/balance", nil, &balance)
    return &balance, err
}

func (f *FTNSClient) Transfer(ctx context.Context, req *types.FTNSTransferRequest) (*types.FTNSTransaction, error) {
    var tx types.FTNSTransaction
    err := f.client.MakeRequest(ctx, "POST", "/api/v1/ftns/transfer", req, &tx)
    return &tx, err
}

func (f *FTNSClient) GetTransactionHistory(ctx context.Context, limit int) ([]*types.FTNSTransaction, error) {
    var txs []*types.FTNSTransaction
    endpoint := fmt.Sprintf("/api/v1/ftns/transactions?limit=%d", limit)
    err := f.client.MakeRequest(ctx, "GET", endpoint, nil, &txs)
    return txs, err
}

func (f *FTNSClient) Stake(ctx context.Context, amount float64, durationDays int) (*types.StakePosition, error) {
    var pos types.StakePosition
    req := map[string]interface{}{"amount": amount, "duration_days": durationDays}
    err := f.client.MakeRequest(ctx, "POST", "/api/v1/ftns/stake", req, &pos)
    return &pos, err
}
```

**Create `sdks/go/marketplace/marketplace.go`:**

```go
package marketplace

// List, search, create, purchase listings
// Mirror the Python SDK's marketplace.py methods
```

**Create `sdks/go/storage/storage.go`:**

```go
package storage

// Upload, download, pin operations
// Mirror the Python SDK's storage.py methods
```

**Add missing types to `sdks/go/types/types.go`:**

```go
type FTNSBalance struct {
    UserID  string  `json:"user_id"`
    Balance float64 `json:"balance"`
    // ...
}

type FTNSTransferRequest struct {
    ToUserID string  `json:"to_user_id"`
    Amount   float64 `json:"amount"`
    Memo     string  `json:"memo,omitempty"`
}

// Add: MarketplaceListing, ComputeJob, StorageObject, StakePosition
```

**Write tests for new modules:**

Add to `sdks/go/types/types_test.go` (already has 376 lines of tests) — add test cases for
new types. Add `sdks/go/ftns/ftns_test.go`, `sdks/go/marketplace/marketplace_test.go`.

**Update `sdks/go/README.md`:**
```markdown
## Modules

| Module | Import | Purpose |
|--------|--------|---------|
| client | `.../client` | Core HTTP client |
| nwtn | `.../nwtn` | AI query pipeline |
| ftns | `.../ftns` | Token balance and transfers |
| marketplace | `.../marketplace` | Asset listings |
| storage | `.../storage` | IPFS-based file storage |
| governance | `.../governance` | On-chain proposals and voting |
| websocket | `.../websocket` | Real-time event streams |
```

---

## Step 3 — SDK Publishing Preparation (~2 hr)

### Python SDK — PyPI

`sdks/python/pyproject.toml` should exist and be correct. Verify it has:

```toml
[project]
name = "prsm-sdk"
version = "0.2.1"   # Match node version
description = "Python SDK for the PRSM Protocol"
requires-python = ">=3.11"
dependencies = [
    "aiohttp>=3.9",
    "pydantic>=2.0",
]

[project.urls]
Homepage = "https://github.com/Ryno2390/PRSM"
Documentation = "https://github.com/Ryno2390/PRSM/tree/main/sdks/python"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Add a `sdks/python/CHANGELOG.md` documenting what's in v0.2.1.

Verify the package builds: `cd sdks/python && python -m build` — should produce
`dist/prsm_sdk-0.2.1.tar.gz` and `dist/prsm_sdk-0.2.1-py3-none-any.whl`.

### JavaScript SDK — npm

Verify `sdks/javascript/package.json` has:
```json
{
  "name": "@prsm/js-sdk",
  "version": "0.2.1",
  "description": "JavaScript/TypeScript SDK for PRSM Protocol",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "build": "tsc",
    "test": "jest"
  }
}
```

Add `sdks/javascript/CHANGELOG.md`.

Verify builds: `cd sdks/javascript && npm run build` — should produce `dist/` without errors.

### Go SDK — go.mod

`sdks/go/go.mod` should have the correct module path matching how external Go developers
will import it:

```go
module github.com/Ryno2390/PRSM/sdks/go

go 1.21
```

Verify: `cd sdks/go && go build ./...` — all packages build cleanly.

### Note: Publishing is infrastructure
Actually publishing to PyPI, npm registry, and pkg.go.dev requires accounts and CI/CD setup.
That's infrastructure, not code. **The Phase 8 deliverable is that the packages build cleanly
and are structured correctly for publishing** — not that they're published.

---

## Step 4 — Non-Technical User Guide (~3 hr)

### What exists
`docs/QUICKSTART_GUIDE.md` — technical quickstart aimed at developers (CLI commands, pip
install, etc.). `docs/INVESTOR_QUICKSTART.md` — economic/business framing.

### What's missing
A guide for someone who: knows nothing about Python, wants to participate in PRSM by
contributing compute or data, and wants to earn FTNS. This is the guide for the eventual
"broad public participation" audience.

### Create `docs/PARTICIPANT_GUIDE.md`

Structure:
```markdown
# PRSM Participant Guide — Earn FTNS by Contributing to Science

## What is PRSM?
[3 paragraph plain-English explanation: decentralized AI, FTNS token, why it matters]

## What is FTNS?
[Explain the token: what it's worth, how you earn it, how you spend it]

## How to Participate

### Option A: Contribute Compute Power
[Step-by-step: install, run node, watch FTNS accumulate — no coding required]

### Option B: Share Your Data
[Step-by-step: upload a dataset via the marketplace UI — no coding required]

### Option C: Use PRSM for AI Queries
[Step-by-step: make an AI query, spend FTNS, receive quality-verified results]

## Frequently Asked Questions
- "Do I need an API key?" (No, but Ollama local inference works without one)
- "How much can I earn?" (Depends on hardware + uptime)
- "Is my data safe?" (IPFS encryption + provenance tracking explained)
- "What is FTNS worth?" (Testnet only until mainnet — governance sets real price)
- "How do I get started without any FTNS?" (Welcome grant of 100 FTNS on first registration)

## Safety and Privacy
[Brief: what PRSM sees, what it doesn't, how data stays yours]

## Community and Support
[GitHub issues, Discord/Slack if it exists, roadmap link]
```

---

## Step 5 — Node Operator Guide (~3 hr)

### What exists
`docs/BOOTSTRAP_DEPLOYMENT_GUIDE.md` — covers bootstrap node deployment specifically.
`docs/PRODUCTION_OPERATIONS_MANUAL.md` — exists (check length before writing to avoid duplication).

### Create or expand `docs/OPERATOR_GUIDE.md`

Structure:
```markdown
# PRSM Node Operator Guide

## Who This Guide Is For
[DevOps/sysadmin running a persistent PRSM node — not the developer audience]

## System Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 8 cores |
| RAM | 4 GB | 32 GB |
| Storage | 50 GB SSD | 500 GB NVMe |
| Network | 10 Mbps | 100 Mbps |
| OS | Ubuntu 22.04 LTS | Same |

## Installation

### Docker Compose (Recommended for Production)
[Step-by-step: docker pull, docker-compose.yml, env file setup, first run]

### systemd Service (Bare Metal)
[Step-by-step: clone, pip install, write prsm.service unit file, enable, start]

## Configuration Reference
[Table of every env var in config/secure.env.template with description + example]

## Monitoring
[How to read /health, /health/metrics; example Grafana dashboard JSON if applicable]

## Upgrading
[git pull → alembic upgrade head → restart — safe upgrade procedure]

## Backup and Recovery
[What to back up: database file, node_identity.json, .env file]
[How to restore from backup]

## Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Node won't connect to bootstrap | Firewall blocking port 8765 | Open outbound TCP 8765 |
| IPFS not starting | ipfs binary not on PATH | `brew install ipfs` or `apt install ipfs` |
| "JWT secret too short" | SECRET_KEY env var missing or weak | Generate: `openssl rand -hex 32` |
| FTNS balance stuck | Database locked | Restart node; check for orphaned SQLite WAL |
```

---

## Step 6 — SDK Developer Guide (~2 hr)

### What exists
`docs/SDK_DOCUMENTATION_ENHANCEMENTS.md` — describes improvements made, not how to use the SDK.

### Create `docs/SDK_DEVELOPER_GUIDE.md`

A practical guide for an external developer who wants to build a product on PRSM:

```markdown
# Building on PRSM — SDK Developer Guide

## Choosing Your SDK
| Language | Package | Status |
|----------|---------|--------|
| Python | `pip install prsm-sdk` | ✅ Complete |
| JavaScript/TypeScript | `npm install @prsm/js-sdk` | ✅ Complete |
| Go | `go get github.com/Ryno2390/PRSM/sdks/go` | ✅ Complete |

## Python — 5-Minute Quickstart
[Working code example: install → authenticate → submit query → get result]

## Python — Common Use Cases
### AI Query with Budget Control
[Code example using ftns.budget + compute.submit_job]

### Upload and Share a Dataset
[Code example using storage.upload + marketplace.create_listing]

### Stream Real-Time Results
[Code example using websocket module]

## JavaScript — 5-Minute Quickstart
[Same structure as Python section]

## Go — 5-Minute Quickstart
[Same structure]

## Authentication
[How API keys work, where to get one, how to rotate]

## Error Handling
[Common errors, what they mean, how to retry correctly]

## Rate Limits
[Per-user limits, per-endpoint limits, how to handle 429]

## Examples Repository
See `sdks/python/examples/` for production-ready patterns:
- FastAPI integration with PRSM backend
- Batch dataset processing
- Scientific research paper analysis
```

---

## Step 7 — Examples Audit and Completion (~2 hr)

### What exists
`sdks/python/examples/` — check what's there. The SDK documentation mentions:
- `production/fastapi_integration.py`
- `production/docker_deployment.py`
- `scientific/research_paper_analysis.py`

### Fix

For each example file that exists:
1. Run it (or at minimum `python -c "import X"`) to verify it imports cleanly
2. Verify it matches the current SDK API (method names, parameter names)
3. Add a 3-line comment at the top: what it does, what PRSM services it uses, what FTNS it consumes

For missing examples (check what the SDK_DOCUMENTATION_ENHANCEMENTS.md promised):
- Write them if < 2 hr each
- Or add a "Coming in Phase 9" comment in the SDK guide if too large

---

## Step 8 — Write `tests/test_phase8_sdk.py` (~1 hr)

Consolidate SDK-level verification tests:

```python
"""
Phase 8 SDK completeness verification.
Verifies that all three SDKs build and their core APIs exist.
"""
import subprocess
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def test_python_sdk_imports():
    from prsm_sdk import PRSMClient
    from prsm_sdk.ftns import FTNSClient
    from prsm_sdk.compute import ComputeClient
    from prsm_sdk.marketplace import MarketplaceClient
    from prsm_sdk.storage import StorageClient
    assert PRSMClient is not None


def test_python_sdk_builds():
    result = subprocess.run(
        ["python", "-m", "build", "--wheel", "--no-isolation"],
        cwd=REPO_ROOT / "sdks/python",
        capture_output=True, timeout=60
    )
    assert result.returncode == 0, result.stderr.decode()


def test_go_sdk_builds():
    result = subprocess.run(
        ["go", "build", "./..."],
        cwd=REPO_ROOT / "sdks/go",
        capture_output=True, timeout=60
    )
    assert result.returncode == 0, result.stderr.decode()


def test_go_sdk_has_ftns_module():
    assert (REPO_ROOT / "sdks/go/ftns/ftns.go").exists()


def test_go_sdk_has_marketplace_module():
    assert (REPO_ROOT / "sdks/go/marketplace/marketplace.go").exists()


def test_go_sdk_has_storage_module():
    assert (REPO_ROOT / "sdks/go/storage/storage.go").exists()


def test_participant_guide_exists():
    guide = REPO_ROOT / "docs/PARTICIPANT_GUIDE.md"
    assert guide.exists()
    assert guide.stat().st_size > 3000, "Participant guide seems too short"


def test_operator_guide_exists():
    guide = REPO_ROOT / "docs/OPERATOR_GUIDE.md"
    assert guide.exists()
    assert guide.stat().st_size > 3000, "Operator guide seems too short"


def test_sdk_developer_guide_exists():
    guide = REPO_ROOT / "docs/SDK_DEVELOPER_GUIDE.md"
    assert guide.exists()
```

---

## Execution Order

| Order | Task | Files Changed | Effort |
|-------|------|--------------|--------|
| 1 | Python SDK integration tests | `sdks/python/tests/test_integration.py` | 3 hr |
| 2 | Go SDK: ftns module | `sdks/go/ftns/ftns.go` + types | 2 hr |
| 3 | Go SDK: marketplace module | `sdks/go/marketplace/marketplace.go` | 1 hr |
| 4 | Go SDK: storage module | `sdks/go/storage/storage.go` | 1 hr |
| 5 | SDK publishing prep (build verification) | `sdks/*/pyproject.toml`, `package.json`, `go.mod` | 2 hr |
| 6 | Non-technical participant guide | `docs/PARTICIPANT_GUIDE.md` | 3 hr |
| 7 | Node operator guide | `docs/OPERATOR_GUIDE.md` | 3 hr |
| 8 | SDK developer guide | `docs/SDK_DEVELOPER_GUIDE.md` | 2 hr |
| 9 | Examples audit and completion | `sdks/python/examples/` | 2 hr |
| 10 | Write test_phase8_sdk.py | `tests/test_phase8_sdk.py` | 1 hr |

**Total estimated effort: ~20 hours**

---

## Expected Outcome

| Audience | What They Can Do After Phase 8 |
|----------|-------------------------------|
| External developer (Python) | `pip install prsm-sdk` → functional in 10 minutes with working examples |
| External developer (Go) | `go get` → complete SDK with ftns, marketplace, storage, governance |
| Non-technical participant | Follow `PARTICIPANT_GUIDE.md` end-to-end without touching code |
| Node operator (DevOps) | Follow `OPERATOR_GUIDE.md` to deploy a persistent production node |
| Investor doing technical due diligence | Run `pytest sdks/python/tests/` and see SDK tests pass; run `go build ./...` and see Go SDK build |

---

## What Remains After Phase 8 (Phase 9 scope)

- **GlobalInfrastructure enterprise tier** (`test_phase7_integration.py`) — the final deferred test
- **SDK actual publishing** (PyPI, npm, pkg.go.dev) — infrastructure, not code
- **Video walkthroughs** — outside scope of code-only work
- **KYC/AML for fiat gateway** — legal/regulatory, not code
- **External security audit** — paid service, not code

After Phase 8, PRSM is at **~95% of everything achievable without infrastructure investment**.

---

## Completion Summary — [To Be Filled In]

*Fill this section when Phase 8 is complete.*

### Files Changed
<!-- List here -->

### Test Count Delta
<!-- Before: ~3,730 | After: ??? -->

### Documentation Coverage
<!-- What audiences are now served -->
