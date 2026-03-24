# Phase 4 — External Deployment

## Overview

Phase 3 delivered the browser-based node onboarding UI. This phase completes the
remaining **Infrastructure Usability** and **Network Reliability** items needed
before PRSM can support broad public participation:

1. **Fix bootstrap default domain** — the hardcoded default in `prsm/node/config.py`
   points to `bootstrap.prsm.io` which does not resolve; the live server is
   `bootstrap1.prsm-network.com:8765`.
2. **Deploy FTNS to Ethereum mainnet** — FTNS currently lives on Sepolia testnet only;
   all smart contracts and the Hardhat pipeline are complete.
3. **Deploy EU and APAC bootstrap nodes** — a single US-East bootstrap is a single
   point of failure; Docker, config, and monitoring all exist; only infrastructure
   provisioning is missing.
4. **Update `config/secure.env.template`** — add all deployment-relevant env vars that
   are currently undocumented.

**Exit criterion:** Ethereum Etherscan shows a verified FTNS contract at a mainnet address;
`prsm node start` can connect to a bootstrap node with sub-500 ms latency from the US,
EU, and APAC; running `pytest tests/test_phase4_deployment.py -v` reports 0 failed.

---

## Step 1 — Fix Bootstrap Default Domain

The live bootstrap server is `bootstrap1.prsm-network.com:8765`. The default in code is
`bootstrap.prsm.io:9001` (wrong host, wrong port). Fix is two lines.

**File:** `prsm/node/config.py` lines 20–28

```python
DEFAULT_BOOTSTRAP_NODES = [
    os.getenv("BOOTSTRAP_PRIMARY", "wss://bootstrap1.prsm-network.com:8765"),
]

FALLBACK_BOOTSTRAP_NODES = [
    os.getenv("BOOTSTRAP_FALLBACK_EU",   "wss://bootstrap-eu.prsm-network.com:8765"),
    os.getenv("BOOTSTRAP_FALLBACK_APAC", "wss://bootstrap-apac.prsm-network.com:8765"),
]
```

Also update `config/node_config.json.template` (the onboarding wizard default):

```json
"bootstrap_fallback_nodes": [
  "wss://bootstrap-eu.prsm-network.com:8765",
  "wss://bootstrap-apac.prsm-network.com:8765"
]
```

**Verify:** `prsm node start --dry-run` should print the corrected URLs.

---

## Step 2 — Deploy FTNS to Ethereum Mainnet

### 2a. Environment Setup

Add to `.env` (never committed — same pattern as `config/secure.env.template`):

```bash
MAINNET_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/<your-alchemy-key>
PRIVATE_KEY=<deployer-account-private-key>       # must hold ~0.1 ETH for gas
ETHERSCAN_API_KEY=<your-etherscan-api-key>
```

These go in `config/secure.env.template` as documented placeholders (no real values):

```bash
# Ethereum Mainnet Deployment (Phase 4)
MAINNET_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_KEY
ETHERSCAN_API_KEY=YOUR_ETHERSCAN_API_KEY
DEPLOYER_PRIVATE_KEY=YOUR_DEPLOYER_PRIVATE_KEY_WITH_ETH
```

### 2b. Hardhat Deployment

The Hardhat pipeline is fully configured in `contracts/hardhat.config.js`. Mainnet
network config already exists in `prsm/economy/blockchain/networks.py` at key `"mainnet"`.

Run:

```bash
cd contracts
npx hardhat compile
npx hardhat run scripts/deploy.js --network mainnet
```

After deployment, the script prints the contract address. Record it.

### 2c. Contract Verification

```bash
npx hardhat verify --network mainnet <DEPLOYED_CONTRACT_ADDRESS>
```

If the token constructor takes arguments, pass them:

```bash
npx hardhat verify --network mainnet <ADDRESS> <arg1> <arg2>
```

Verify success via: `https://etherscan.io/address/<ADDRESS>#code`

### 2d. Update Python Layer

In `prsm/economy/blockchain/networks.py`, the `"mainnet"` entry already uses
`MAINNET_RPC_URL` from the environment. After deployment, record the contract address
in a new `config/contracts_mainnet.json` file (gitignored if it contains sensitive
addresses, but safe to commit just addresses):

```json
{
  "network": "mainnet",
  "chain_id": 1,
  "ftns_token": "<DEPLOYED_ADDRESS>",
  "deployed_at": "2026-03-25T00:00:00Z",
  "etherscan_url": "https://etherscan.io/address/<DEPLOYED_ADDRESS>"
}
```

Update `prsm/economy/blockchain/smart_contracts.py` to load this file when
`WEB3_NETWORK=mainnet`.

### 2e. Update secure.env.template

```bash
# Active mainnet deployment (set after Phase 4)
WEB3_NETWORK=mainnet
FTNS_TOKEN_ADDRESS_MAINNET=<DEPLOYED_ADDRESS>
```

---

## Step 3 — Deploy EU Bootstrap Node

### 3a. Provision Server

Minimum spec: **DigitalOcean 2vCPU / 4GB RAM / 80GB SSD** in **AMS3 (Amsterdam)** or
**FRA1 (Frankfurt)**. The production Docker Compose in `docker/docker-compose.bootstrap.yml`
allocates 2 CPU / 2GB RAM — this spec gives headroom.

```bash
doctl compute droplet create bootstrap-eu \
  --region ams3 \
  --size s-2vcpu-4gb \
  --image ubuntu-24-04-x64 \
  --ssh-keys <your-ssh-key-id>
```

### 3b. DNS

Create an A record pointing `bootstrap-eu.prsm-network.com` → EU droplet IP.

Using DigitalOcean DNS:

```bash
doctl compute domain records create prsm-network.com \
  --record-type A \
  --record-name bootstrap-eu \
  --record-data <EU_DROPLET_IP> \
  --record-ttl 300
```

### 3c. TLS Certificate

On the EU server, install Certbot and issue a cert for `bootstrap-eu.prsm-network.com`:

```bash
apt-get install -y certbot
certbot certonly --standalone -d bootstrap-eu.prsm-network.com \
  --non-interactive --agree-tos -m admin@prsm.ai
```

The cert is written to `/etc/letsencrypt/live/bootstrap-eu.prsm-network.com/`.

### 3d. Deploy via Docker Compose

```bash
# Copy docker/ directory to the EU server
scp -r docker/ root@<EU_IP>:/opt/prsm-bootstrap/

# Copy prsm source (or use the Docker Hub image if published)
# Then on the EU server:
cd /opt/prsm-bootstrap
docker compose -f docker-compose.bootstrap.yml up -d
```

The compose file already handles:
- Bootstrap server + HA replica (profiles: ha)
- PostgreSQL peer state persistence
- Redis cache
- Prometheus + Grafana (profiles: monitoring)
- Nginx reverse proxy with SSL (profiles: loadbalancer)

For production, start all profiles:
```bash
docker compose -f docker-compose.bootstrap.yml \
  --profile ha --profile monitoring --profile loadbalancer up -d
```

Mount the Let's Encrypt certs:
```yaml
# docker-compose.bootstrap.yml — nginx service volumes
volumes:
  - /etc/letsencrypt/live/bootstrap-eu.prsm-network.com:/etc/ssl/prsm:ro
```

### 3e. Verify EU Bootstrap

```bash
curl -s http://bootstrap-eu.prsm-network.com:8000/health | python -m json.tool
# Should return {"status": "healthy", "peer_count": N, ...}

wscat -c wss://bootstrap-eu.prsm-network.com:8765
# Should open WebSocket connection
```

---

## Step 4 — Deploy APAC Bootstrap Node

Same procedure as Step 3 with:

| Setting | Value |
|---------|-------|
| Region | `sgp1` (Singapore) or `blr1` (Bangalore) |
| Droplet name | `bootstrap-apac` |
| DNS | `bootstrap-apac.prsm-network.com` |
| Certbot domain | `bootstrap-apac.prsm-network.com` |

```bash
doctl compute droplet create bootstrap-apac \
  --region sgp1 \
  --size s-2vcpu-4gb \
  --image ubuntu-24-04-x64 \
  --ssh-keys <your-ssh-key-id>

doctl compute domain records create prsm-network.com \
  --record-type A \
  --record-name bootstrap-apac \
  --record-data <APAC_DROPLET_IP> \
  --record-ttl 300
```

---

## Step 5 — Update `config/secure.env.template`

The template currently only documents Polygon Mumbai. Expand it to cover all Phase 4
deployment variables:

```bash
# ============================================================
# PRSM Secure Configuration Template
# ============================================================
# Do NOT put real credentials here. Register via the PRSM
# credential management API or set as environment variables.

# JWT Security (REQUIRED)
SECRET_KEY=GENERATE_SECURE_RANDOM_STRING_64_CHARS_MINIMUM

# Database
DATABASE_URL=postgresql://username:password@localhost:5432/prsm
REDIS_URL=redis://localhost:6379/0

# AI Backend Keys (register via credential API; set here for dev only)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# ============================================================
# Blockchain — P2P Bootstrap Nodes
# ============================================================
BOOTSTRAP_PRIMARY=wss://bootstrap1.prsm-network.com:8765
BOOTSTRAP_FALLBACK_EU=wss://bootstrap-eu.prsm-network.com:8765
BOOTSTRAP_FALLBACK_APAC=wss://bootstrap-apac.prsm-network.com:8765

# ============================================================
# Blockchain — Ethereum / Web3
# ============================================================
WEB3_NETWORK=mainnet           # Options: mainnet, sepolia, polygon, localhost

# Sepolia Testnet (development)
SEPOLIA_RPC_URL=https://rpc.sepolia.org

# Ethereum Mainnet (production — requires paid RPC provider)
MAINNET_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_KEY

# Contract Addresses (populated after deployment)
FTNS_TOKEN_ADDRESS_SEPOLIA=0xd979c096BE297F4C3a85175774Bc38C22b95E6a4
FTNS_TOKEN_ADDRESS_MAINNET=<FILL_AFTER_DEPLOYMENT>

# Deployment Account (NEVER commit a real private key)
DEPLOYER_PRIVATE_KEY=YOUR_DEPLOYER_PRIVATE_KEY_WITH_ETH_FOR_GAS

# Contract Verification
ETHERSCAN_API_KEY=YOUR_ETHERSCAN_API_KEY
POLYGONSCAN_API_KEY=YOUR_POLYGONSCAN_API_KEY
```

---

## Step 6 — Write Deployment Verification Tests

Create `tests/test_phase4_deployment.py`:

```python
"""
Phase 4 deployment verification tests.

These tests verify the external deployment configuration is correct.
They do NOT hit live infrastructure — they validate config values,
file structure, and contract address formats.

For live integration testing, set PRSM_LIVE_TESTS=1 in the environment.
"""
import os
import json
import re
import socket
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Bootstrap config tests
# ---------------------------------------------------------------------------

def test_bootstrap_default_is_live_domain():
    """Default BOOTSTRAP_PRIMARY env var default matches live server domain."""
    from prsm.node.config import DEFAULT_BOOTSTRAP_NODES
    assert len(DEFAULT_BOOTSTRAP_NODES) >= 1
    primary = DEFAULT_BOOTSTRAP_NODES[0]
    assert "prsm-network.com" in primary, (
        f"Default bootstrap should use prsm-network.com, got: {primary}"
    )
    assert "8765" in primary, (
        f"Default bootstrap should use port 8765, got: {primary}"
    )


def test_fallback_bootstrap_nodes_defined():
    """Fallback bootstrap nodes reference EU and APAC regions."""
    from prsm.node.config import FALLBACK_BOOTSTRAP_NODES
    assert len(FALLBACK_BOOTSTRAP_NODES) >= 2
    combined = " ".join(FALLBACK_BOOTSTRAP_NODES)
    assert "eu" in combined.lower() or "ams" in combined.lower(), (
        "Expected an EU fallback bootstrap node"
    )
    assert "apac" in combined.lower() or "sgp" in combined.lower() or "asia" in combined.lower(), (
        "Expected an APAC fallback bootstrap node"
    )


def test_node_config_template_has_correct_bootstrap():
    """config/node_config.json.template uses the live bootstrap domain."""
    template_path = REPO_ROOT / "config" / "node_config.json.template"
    assert template_path.exists(), "config/node_config.json.template must exist"
    content = template_path.read_text()
    data = json.loads(content)
    bootstrap = data.get("bootstrap_nodes", [])
    assert len(bootstrap) >= 1
    assert any("prsm-network.com" in b for b in bootstrap), (
        f"Template bootstrap_nodes should reference prsm-network.com, got: {bootstrap}"
    )


# ---------------------------------------------------------------------------
# Mainnet contract address tests
# ---------------------------------------------------------------------------

ETH_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


def test_sepolia_contract_address_format():
    """Known Sepolia FTNS contract address is a valid Ethereum address."""
    address = "0xd979c096BE297F4C3a85175774Bc38C22b95E6a4"
    assert ETH_ADDRESS_RE.match(address), f"Invalid address format: {address}"


def test_mainnet_contract_address_when_set():
    """If FTNS_TOKEN_ADDRESS_MAINNET env var is set, it must be a valid address."""
    mainnet_addr = os.getenv("FTNS_TOKEN_ADDRESS_MAINNET", "")
    if not mainnet_addr or mainnet_addr.startswith("<"):
        pytest.skip("Mainnet contract not yet deployed")
    assert ETH_ADDRESS_RE.match(mainnet_addr), (
        f"FTNS_TOKEN_ADDRESS_MAINNET is not a valid Ethereum address: {mainnet_addr}"
    )


def test_contracts_mainnet_json_when_exists():
    """If config/contracts_mainnet.json exists, it has required fields."""
    contracts_path = REPO_ROOT / "config" / "contracts_mainnet.json"
    if not contracts_path.exists():
        pytest.skip("contracts_mainnet.json not yet created")
    data = json.loads(contracts_path.read_text())
    assert "ftns_token" in data, "contracts_mainnet.json must have ftns_token field"
    assert "etherscan_url" in data, "contracts_mainnet.json must have etherscan_url"
    assert ETH_ADDRESS_RE.match(data["ftns_token"]), (
        f"ftns_token is not a valid Ethereum address: {data['ftns_token']}"
    )


# ---------------------------------------------------------------------------
# secure.env.template coverage tests
# ---------------------------------------------------------------------------

def test_secure_env_template_documents_bootstrap_vars():
    """secure.env.template documents bootstrap environment variables."""
    template_path = REPO_ROOT / "config" / "secure.env.template"
    content = template_path.read_text()
    assert "BOOTSTRAP_PRIMARY" in content, "Template must document BOOTSTRAP_PRIMARY"
    assert "BOOTSTRAP_FALLBACK_EU" in content, "Template must document BOOTSTRAP_FALLBACK_EU"
    assert "BOOTSTRAP_FALLBACK_APAC" in content, "Template must document BOOTSTRAP_FALLBACK_APAC"


def test_secure_env_template_documents_mainnet_vars():
    """secure.env.template documents mainnet deployment variables."""
    template_path = REPO_ROOT / "config" / "secure.env.template"
    content = template_path.read_text()
    assert "MAINNET_RPC_URL" in content, "Template must document MAINNET_RPC_URL"
    assert "ETHERSCAN_API_KEY" in content, "Template must document ETHERSCAN_API_KEY"
    assert "WEB3_NETWORK" in content, "Template must document WEB3_NETWORK"


# ---------------------------------------------------------------------------
# Live connectivity tests (require PRSM_LIVE_TESTS=1)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("PRSM_LIVE_TESTS"),
    reason="Set PRSM_LIVE_TESTS=1 to run live connectivity tests"
)
def test_bootstrap1_resolves():
    """bootstrap1.prsm-network.com resolves to a valid IP."""
    ip = socket.gethostbyname("bootstrap1.prsm-network.com")
    assert ip and ip != "0.0.0.0", f"Expected valid IP, got: {ip}"


@pytest.mark.skipif(
    not os.getenv("PRSM_LIVE_TESTS"),
    reason="Set PRSM_LIVE_TESTS=1 to run live connectivity tests"
)
def test_bootstrap1_port_open():
    """bootstrap1.prsm-network.com port 8765 is accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(5.0)
        result = s.connect_ex(("bootstrap1.prsm-network.com", 8765))
    assert result == 0, f"Port 8765 is not open on bootstrap1 (connect_ex returned {result})"


@pytest.mark.skipif(
    not os.getenv("PRSM_LIVE_TESTS"),
    reason="Set PRSM_LIVE_TESTS=1 to run live connectivity tests"
)
def test_bootstrap_eu_resolves():
    """bootstrap-eu.prsm-network.com resolves once deployed."""
    try:
        ip = socket.gethostbyname("bootstrap-eu.prsm-network.com")
        assert ip and ip != "0.0.0.0"
    except socket.gaierror:
        pytest.fail("bootstrap-eu.prsm-network.com does not resolve — deploy EU node first")


@pytest.mark.skipif(
    not os.getenv("PRSM_LIVE_TESTS"),
    reason="Set PRSM_LIVE_TESTS=1 to run live connectivity tests"
)
def test_bootstrap_apac_resolves():
    """bootstrap-apac.prsm-network.com resolves once deployed."""
    try:
        ip = socket.gethostbyname("bootstrap-apac.prsm-network.com")
        assert ip and ip != "0.0.0.0"
    except socket.gaierror:
        pytest.fail("bootstrap-apac.prsm-network.com does not resolve — deploy APAC node first")
```

---

## Step 7 — Update `docs/IMPLEMENTATION_STATUS.md`

After completing each deliverable, update the relevant section:

- **Bootstrap nodes**: change Sepolia-only → Multi-region
- **FTNS Token**: change "testnet only" → "deployed to mainnet at `<ADDRESS>`"
- **Phase 4 status**: add new section documenting what was deployed and when

---

## Execution Order

| Order | Task | Code change? | Infrastructure? | Time |
|-------|------|-------------|-----------------|------|
| 1 | Fix bootstrap default domain | ✅ Yes | No | 5 min |
| 2 | Update secure.env.template | ✅ Yes | No | 15 min |
| 3 | Write + run Phase 4 tests | ✅ Yes | No | 30 min |
| 4 | Deploy FTNS to mainnet | Hardhat CLI | No new infra | 1-2 hr |
| 5 | Deploy EU bootstrap | No | DigitalOcean | 2-3 hr |
| 6 | Deploy APAC bootstrap | No | DigitalOcean | 2-3 hr |
| 7 | Update IMPLEMENTATION_STATUS.md | ✅ Yes | No | 15 min |

Steps 1-3 are pure code and can be done immediately. Steps 4-6 require external
accounts and infra access (DigitalOcean, Alchemy/Infura, domain DNS control).

---

## Implementation Notes

- **bootstrap.prsm.io vs bootstrap1.prsm-network.com**: The original docs reference
  `bootstrap.prsm.io` but the live server is `bootstrap1.prsm-network.com`. This plan
  standardizes everything on `*.prsm-network.com`. If `bootstrap.prsm.io` needs to be
  kept alive for backwards compatibility, add a CNAME record pointing it to
  `bootstrap1.prsm-network.com`.

- **Mainnet gas costs**: Deploying the FTNS token contract costs roughly 0.05–0.15 ETH
  depending on gas prices. The bridge and staking contracts cost more. Estimate
  0.2–0.5 ETH total for the full contract suite.

- **FTNS monetary value**: Once deployed to mainnet, FTNS has real monetary value.
  The bridge contract enables moving tokens between the local PRSM ledger and the
  on-chain ERC20. This is when the FTNS economy becomes production-grade.

- **Bootstrap node costs**: Each 2vCPU/4GB DigitalOcean droplet costs ~$24/month.
  The monitoring + HA profiles roughly double RAM usage — the 4GB spec handles
  the default profile; upgrade to 8GB if running monitoring.

- **SSL auto-renewal**: Set up a cron job on each bootstrap server:
  ```cron
  0 0 1 * * certbot renew --quiet && docker restart nginx
  ```

---

## Verification Checklist

```bash
# Step 1 — bootstrap default fix
pytest tests/test_phase4_deployment.py::test_bootstrap_default_is_live_domain -v

# Step 2 — env template coverage
pytest tests/test_phase4_deployment.py::test_secure_env_template_documents_bootstrap_vars \
       tests/test_phase4_deployment.py::test_secure_env_template_documents_mainnet_vars -v

# Step 3 — full Phase 4 test suite (skips live connectivity by default)
pytest tests/test_phase4_deployment.py -v

# Step 4 — after mainnet deployment
FTNS_TOKEN_ADDRESS_MAINNET=<ADDRESS> pytest tests/test_phase4_deployment.py::test_mainnet_contract_address_when_set -v

# Step 5+6 — after bootstrap nodes deployed
PRSM_LIVE_TESTS=1 pytest tests/test_phase4_deployment.py -v -k "resolves or port_open"

# Full suite still passes (Phase 4 adds ~12 new tests)
pytest --ignore=tests/benchmarks --ignore=tests/test_seal.py -q --timeout=60
```

Expected: All `test_phase4_deployment.py` tests pass (live tests skipped until infrastructure
is up); full suite adds ~12 tests to the passing count; 0 regressions.
