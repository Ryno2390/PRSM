# PRSM Production Deployment Guide

## Prerequisites

- Ubuntu 22.04+ or Debian 12+
- Python 3.11+
- 8GB+ RAM, GPU recommended
- Public IP with ports 8000 and 9001 open
- Domain pointing to this server (for TLS)

## Quick Deploy

### 1. Install

```bash
sudo useradd -m -s /bin/bash prsm
sudo mkdir -p /opt/prsm/{data,logs,venv}
sudo chown -R prsm:prsm /opt/prsm

sudo -u prsm python3 -m venv /opt/prsm/venv
sudo -u prsm /opt/prsm/venv/bin/pip install prsm-network[wasm]
```

### 2. Configure

```bash
sudo cp deploy/production/prsm.env.template /opt/prsm/.env
sudo nano /opt/prsm/.env  # Fill in your values
```

### 3. Start

**As systemd service (recommended):**
```bash
sudo cp deploy/production/prsm-node.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable prsm-node
sudo systemctl start prsm-node
sudo journalctl -u prsm-node -f  # Watch logs
```

**Or with Docker:**
```bash
docker-compose -f docker/docker-compose.demo.yml up -d
```

### 4. Verify

```bash
curl http://localhost:8000/status
curl http://localhost:8000/rings/status
```

### 5. Seed Data (Prismatica)

```bash
prsm marketplace list-dataset \
  --title "NADA NC Vehicle Registrations 2025" \
  --dataset-id nada-nc-2025 \
  --base-fee 5.0 \
  --per-shard 0.5 \
  --shards 12 \
  --require-stake 100
```

## TLS Setup (recommended)

Use Caddy for automatic TLS:

```bash
sudo apt install caddy
cat > /etc/caddy/Caddyfile << 'EOF'
api.prsm-network.com {
    reverse_proxy localhost:8000
}
EOF
sudo systemctl restart caddy
```

## Monitoring

```bash
# Check node health
curl http://localhost:8000/rings/status

# Check settlement queue
curl http://localhost:8000/settlement/stats

# Watch logs
sudo journalctl -u prsm-node -f
```

## Prismatica Mega-Node Setup

For Prismatica's seed node, additionally:

1. Set `PRSM_STAKE_AMOUNT=10000` (Sentinel tier)
2. Upload seed datasets via API or CLI
3. Ensure bootstrap domain resolves to this server
4. Monitor with `/rings/status` endpoint
