# PRSM Node Operator Guide

A practical guide for DevOps engineers and system administrators running persistent PRSM nodes.

## Who This Guide Is For

This guide is for you if you:
- Are a DevOps engineer deploying PRSM infrastructure
- Run servers and want to contribute to the PRSM network
- Need to deploy a PRSM node for your organization
- Want to participate in the network as a compute provider

If you're looking to use PRSM as an end-user, see the [Participant Guide](PARTICIPANT_GUIDE.md) instead.

---

## Quick Start: Docker Compose (Recommended)

The fastest way to run a production PRSM node.

### Prerequisites

- Docker 20.10+ and Docker Compose v2
- 4GB+ RAM (8GB+ recommended)
- 50GB+ disk space (SSD recommended)
- Open ports: 8000 (API), 8765 (P2P)

### Step 1: Get the Code

```bash
git clone https://github.com/prsm-network/PRSM.git
cd PRSM
```

### Step 2: Configure Environment

```bash
# Copy the example environment file
cp config/secure.env.template .env

# Edit the configuration
nano .env
```

**Required settings:**

```bash
# Security (REQUIRED - generate your own!)
SECRET_KEY=<generate with: openssl rand -hex 32>

# Database
DATABASE_URL=sqlite:///./prsm_node.db

# Initial Admin
INITIAL_ADMIN_EMAIL=admin@example.com
INITIAL_ADMIN_PASSWORD=<secure-password>

# P2P Network
P2P_BOOTSTRAP_NODES=bootstrap1.prsm.io:8765,bootstrap2.prsm.io:8765
```

### Step 3: Launch

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f prsm-api
```

### Step 4: Verify

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "version": "1.0.0", ...}
```

---

## System Requirements

### Minimum Configuration

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| CPU | 2 cores | 8 cores | More cores = more concurrent queries |
| RAM | 4 GB | 32 GB | AI workloads benefit from more RAM |
| Storage | 50 GB SSD | 500 GB NVMe | Fast storage improves IPFS performance |
| Network | 10 Mbps | 100 Mbps | Low latency is critical for P2P |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS | Also supports macOS, Windows |

### Hardware Recommendations by Use Case

| Use Case | CPU | RAM | GPU | Storage |
|----------|-----|-----|-----|---------|
| Light Node | 4 cores | 8 GB | None | 100 GB SSD |
| Standard Node | 8 cores | 16 GB | Optional | 250 GB NVMe |
| Compute Provider | 16+ cores | 32+ GB | NVIDIA 8GB+ VRAM | 500 GB NVMe |
| Bootstrap Node | 8 cores | 16 GB | None | 100 GB SSD |

---

## Installation Methods

### Option 1: Docker Compose (Recommended for Production)

Best for: Most production deployments

```yaml
# docker-compose.yml
version: '3.8'

services:
  prsm-api:
    image: prsm/api:latest
    ports:
      - "8000:8000"
      - "8765:8765"
    environment:
      - SECRET_KEY=${SECRET_KEY}
      - DATABASE_URL=postgresql://prsm:password@db:5432/prsm
      - REDIS_URL=redis://redis:6379
    volumes:
      - prsm-data:/data
      - ./config:/app/config
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=prsm
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=prsm
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  prsm-data:
  postgres-data:
  redis-data:
```

```bash
# Launch
docker-compose up -d
```

### Option 2: Systemd (Bare Metal)

Best for: Servers where you want direct control

```bash
# Install dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv ipfs

# Clone and setup
git clone https://github.com/prsm-network/PRSM.git /opt/prsm
cd /opt/prsm
python3.11 -m venv venv
source venv/bin/activate
pip install -e .

# Create systemd service
sudo nano /etc/systemd/system/prsm.service
```

**prsm.service:**
```ini
[Unit]
Description=PRSM Node
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=prsm
Group=prsm
WorkingDirectory=/opt/prsm
Environment="PATH=/opt/prsm/venv/bin"
ExecStart=/opt/prsm/venv/bin/prsm node start
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable prsm
sudo systemctl start prsm

# Check status
sudo systemctl status prsm
```

### Option 3: Kubernetes

Best for: Cloud-native deployments with auto-scaling

See `k8s/` directory for Helm charts and deployment manifests.

```bash
# Deploy with Helm
helm install prsm ./k8s/helm/prsm \
  --set secrets.secretKey=$(openssl rand -hex 32) \
  --set persistence.enabled=true
```

---

## Configuration Reference

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SECRET_KEY` | JWT signing key (32+ chars) | `openssl rand -hex 32` |
| `DATABASE_URL` | Database connection | `postgresql://user:pass@host/db` |

### Network Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `P2P_LISTEN_PORT` | 8765 | P2P network port |
| `P2P_BOOTSTRAP_NODES` | *none* | Comma-separated bootstrap addresses |
| `API_HOST` | 0.0.0.0 | API bind address |
| `API_PORT` | 8000 | API port |

### Performance Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT_QUERIES` | 10 | Max parallel AI queries |
| `QUERY_TIMEOUT` | 300 | Query timeout (seconds) |
| `CACHE_TTL` | 3600 | Cache TTL (seconds) |
| `WORKER_THREADS` | 4 | Background worker threads |

### Security Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_REQUESTS` | 100 | Requests per minute per user |
| `RATE_LIMIT_WINDOW` | 60 | Window in seconds |
| `MAX_QUERY_COST` | 100 | Max FTNS per query |
| `CIRCUIT_BREAKER_THRESHOLD` | 5 | Failures before circuit opens |

---

## Monitoring

### Health Endpoints

```bash
# Basic health check
GET /health
# Response: {"status": "healthy", "version": "1.0.0"}

# Detailed metrics
GET /health/metrics
# Response: Prometheus-format metrics

# Readiness check (for load balancers)
GET /health/ready
# Response: 200 if ready, 503 if not
```

### Key Metrics to Monitor

| Metric | Healthy Range | Alert Threshold |
|--------|---------------|-----------------|
| `api_latency_p99` | < 2000ms | > 5000ms |
| `error_rate` | < 1% | > 5% |
| `active_connections` | Varies | > 80% max |
| `ftns_queue_depth` | < 100 | > 500 |
| `ipfs_repo_size` | < 80% disk | > 90% |
| `p2p_peers_connected` | > 3 | < 2 |

### Grafana Dashboard

Import the provided dashboard: `monitoring/grafana-dashboard.json`

Key panels:
- Request rate and latency
- Error rate by endpoint
- FTNS transaction volume
- P2P network health
- Resource utilization

---

## Upgrading

### Safe Upgrade Procedure

```bash
# 1. Backup database
./scripts/backup.sh

# 2. Pull latest code
git pull origin main

# 3. Run database migrations
alembic upgrade head

# 4. Restart services
docker-compose restart

# 5. Verify
curl http://localhost:8000/health
```

### Zero-Downtime Upgrade (Kubernetes)

```bash
# Rolling update
kubectl set image deployment/prsm-api \
  prsm-api=prsm/api:v1.2.0 \
  --record

# Monitor rollout
kubectl rollout status deployment/prsm-api

# Rollback if needed
kubectl rollout undo deployment/prsm-api
```

---

## Backup and Recovery

### What to Back Up

| Component | Frequency | Retention |
|-----------|-----------|-----------|
| Database | Daily | 30 days |
| `node_identity.json` | Once | Forever |
| `.env` file | On change | Keep last 3 |
| IPFS repo (optional) | Weekly | 4 weeks |

### Backup Commands

```bash
# SQLite backup
cp prsm_node.db prsm_node.db.backup

# PostgreSQL backup
pg_dump prsm > backup_$(date +%Y%m%d).sql

# Full backup script
./scripts/backup-system.sh backup --output /backups/
```

### Recovery Procedure

```bash
# 1. Stop services
docker-compose down

# 2. Restore database
cp prsm_node.db.backup prsm_node.db
# OR
psql prsm < backup_20250101.sql

# 3. Restart services
docker-compose up -d

# 4. Verify
curl http://localhost:8000/health
```

---

## Troubleshooting

### Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Node won't connect to bootstrap | Firewall blocking port 8765 | `sudo ufw allow 8765/tcp` |
| IPFS not starting | ipfs binary not on PATH | `sudo apt install ipfs` or add to PATH |
| "JWT secret too short" | SECRET_KEY missing or weak | Generate: `openssl rand -hex 32` |
| FTNS balance stuck | Database locked | Restart node; check for orphaned SQLite WAL |
| High memory usage | Too many concurrent queries | Reduce `MAX_CONCURRENT_QUERIES` |
| Slow queries | Model provider API issues | Check external API status |
| P2P not connecting | NAT traversal failed | Enable UPnP or configure port forwarding |

### Diagnostic Commands

```bash
# Check service status
docker-compose ps
systemctl status prsm

# View recent logs
docker-compose logs --tail=100 prsm-api
journalctl -u prsm -n 100

# Check port bindings
netstat -tlnp | grep -E '8000|8765'

# Test database connection
python -c "from prsm.core.database import get_db; print('OK')"

# Test IPFS
ipfs id

# Check P2P peers
curl http://localhost:8000/api/v1/p2p/peers
```

### Log Analysis

```bash
# Find errors
docker-compose logs prsm-api | grep -i error

# Find slow queries
docker-compose logs prsm-api | grep "duration_ms" | awk '$NF > 5000'

# Monitor in real-time
docker-compose logs -f prsm-api | grep -E "ERROR|WARN"
```

---

## On-chain Keypairs

Nodes that participate in on-chain economic activity (provenance, royalty distribution, stake bonding, batched settlement) hold at least one Ethereum-compatible keypair. The invariants below apply to ALL such keypairs.

### Invariant: one keypair per OS process

**Do not share a provider keypair across multiple processes or multiple machines.**

Every PRSM Web3 client (`ProvenanceRegistryClient`, `RoyaltyDistributorClient`, `StakeManagerClient`, etc.) serializes tx build-and-send via an in-process lock. That lock does NOT coordinate across processes. If two processes on the same host (or two hosts) sign transactions for the same keypair concurrently, they can read the same `pending` nonce from the RPC and both broadcast against it. One transaction will revert; the other lands.

This is a correctness-preserving failure (no double-spend, no slashable misbehavior), but it wastes gas and generates confusing operator telemetry.

**Practical rules:**

- One Python process = one provider keypair. If you run a compute node AND a content-hosting node, give them separate keypairs.
- If you run multiple replicas (Kubernetes deployment with `replicas > 1`), ensure each replica has its own keypair — do NOT mount the same keyfile across pods.
- If you operate both a Phase 1.1 content node and a Phase 7 compute-provider node, use a distinct keypair per role. They may share an owner (for accounting), but not a signing key.
- Rotation: when rotating keys, finalize the old key's pending txs BEFORE the new key begins signing.

### Invariant: key material stays local

Never upload keyfile contents to external services (pastebins, shared notes, CI logs, screenshot tools). Key material belongs on the host that uses it; if you need to back it up, encrypt first with a passphrase only you hold.

### Invariant: hardware wallets for Foundation / multi-sig roles

Treasury, governance, and slasher-admin roles (owner of `StakeBond`, owner of `BatchSettlementRegistry`) MUST run behind a 2-of-3 hardware-wallet multi-sig. See PRSM-GOV-1 §8 for the quorum spec. Operator nodes do NOT hold these keys — they belong exclusively to the Foundation.

---

## Redundant-Execution Dispatch (Tier B)

Jobs submitted with `DispatchPolicy.consensus_mode="majority"` run on k providers in parallel (default k=3). The orchestrator compares output hashes, returns the majority's result, and records the minority for later on-chain challenge.

### What operators need to know

**1. Minority receipts accumulate in-process.**

`MarketplaceOrchestrator.consensus_minority_queue` is a per-process list. Each dispatch that finds a disagreeing provider appends a record there. Nothing auto-submits these as on-chain `CONSENSUS_MISMATCH` challenges — that's the job of a separate **consensus-submitter service**.

If you run the orchestrator without a consensus-submitter:
- The job still returns the correct (majority) output to the caller.
- The minority provider gets a reputation penalty via `ReputationTracker`.
- **But their stake is NOT slashed on-chain.** The minority walks away unpunished economically.

This is acceptable for early bake-in (Phase 7.1 MVP) but unsafe at scale. If you're running consensus dispatch in production, either wire a consensus-submitter service or document that your deployment does not enforce the economic layer of Tier B.

**2. Crash recovery drops pending challenges.**

The queue is in-memory. A process restart between dispatch and submission drops whatever minority receipts haven't been drained. Phase 7.1x's consensus-submitter is the planned solution (SQLite/Redis-backed drain). Until then, treat the queue as best-effort — critical-tier jobs with consensus enabled should also pair with on-chain settlement audit (cross-check batch emissions vs orchestrator logs).

**3. Cost model.**

k=3 consensus means 3× the compute cost per shard. Requester pays all k providers from a single escrow. Partial responses (e.g., 2-of-3 with one preempted) still consense if the threshold is met; the non-respondent forfeits their escrow share per Phase 2's existing timeout semantics.

**4. Price discovery.**

Phase 7.1 MVP splits `policy.max_price_per_shard_ftns` evenly across k providers rather than running k parallel price handshakes. Per-provider price discovery in the consensus path is Phase 7.1x work. Set `max_price_per_shard_ftns` generously relative to the per-provider ceiling you'd accept; the effective per-provider budget is `max_price / k`.

### Operator checklist for Tier B

- [ ] `policy.consensus_mode` set to `"majority"` (or `"unanimous"` if you require full k-agreement)
- [ ] `policy.consensus_k` set appropriately (default 3; higher for critical)
- [ ] `multi_dispatcher` wired on the orchestrator (else RuntimeError at dispatch)
- [ ] `stake_manager_client` wired (else the on-chain tier gate skips, per Phase 7)
- [ ] `provider_address_resolver` wired (else the on-chain tier gate skips, per Phase 7)
- [ ] Consensus-submitter service deployed and draining `consensus_minority_queue`, OR you have a documented policy that your deployment doesn't enforce the economic layer
- [ ] Eligible pool sized ≥ consensus_k; smaller pools fail-fast with `NoEligibleProvidersError`

---

## Security Checklist

- [ ] Generate unique `SECRET_KEY` (never use default)
- [ ] Change default admin password
- [ ] Enable TLS/HTTPS (use nginx or certbot)
- [ ] Configure firewall (only expose 8000, 8765)
- [ ] Set up rate limiting
- [ ] Enable audit logging
- [ ] Configure log rotation
- [ ] Set up monitoring alerts
- [ ] Create backup schedule
- [ ] Document recovery procedures

---

## Getting Help

- **Documentation**: [docs.prsm.ai](https://docs.prsm.ai)
- **GitHub Issues**: [github.com/prsm-network/PRSM/issues](https://github.com/prsm-network/PRSM/issues)
- **Discord**: [discord.gg/prsm](https://discord.gg/prsm)
- **Email**: ops@prsm.ai

For production incidents, email security@prsm.ai for security issues.

---

## Appendix: Example Configurations

### Minimal Development Config

```bash
# .env (development)
SECRET_KEY=dev-secret-key-do-not-use-in-production
DATABASE_URL=sqlite:///./prsm_dev.db
DEBUG=true
LOG_LEVEL=DEBUG
```

### Production Config

```bash
# .env (production)
SECRET_KEY=<32+ character random string>
DATABASE_URL=postgresql://prsm:password@db:5432/prsm
REDIS_URL=redis://redis:6379
P2P_BOOTSTRAP_NODES=bootstrap1.prsm.io:8765,bootstrap2.prsm.io:8765
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
LOG_LEVEL=INFO
DEBUG=false
```

### High-Performance Compute Node

```bash
# .env (compute provider)
SECRET_KEY=<random>
DATABASE_URL=postgresql://prsm:password@db:5432/prsm
MAX_CONCURRENT_QUERIES=50
WORKER_THREADS=16
QUERY_TIMEOUT=600
GPU_ENABLED=true
GPU_MEMORY_FRACTION=0.9
```
