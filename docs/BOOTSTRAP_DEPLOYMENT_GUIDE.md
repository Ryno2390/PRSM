# PRSM Bootstrap Server Deployment Guide

This guide provides step-by-step instructions for deploying the PRSM Bootstrap Server to enable P2P network discovery.

## Overview

The bootstrap server is the entry point for new nodes joining the PRSM network. It provides:
- **Peer Discovery**: Helps new nodes find existing peers
- **Connection Management**: Tracks active peers and their status
- **Network Bootstrapping**: Enables decentralized network formation

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        External Access                           │
│  wss://bootstrap.prsm-network.com:8765 (WebSocket Secure)       │
│  https://bootstrap.prsm-network.com:8000 (Health/Metrics API)   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Load Balancer (nginx)                         │
│  - SSL Termination                                               │
│  - Rate Limiting                                                 │
│  - Request Routing                                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│   Bootstrap Primary       │   │   Bootstrap Replica       │
│   :8765 (WebSocket)       │   │   :8765 (WebSocket)       │
│   :8000 (HTTP API)        │   │   :8000 (HTTP API)        │
└───────────────────────────┘   └───────────────────────────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Data Layer                                │
│  ┌─────────────────┐  ┌─────────────────┐                       │
│  │   PostgreSQL    │  │     Redis       │                       │
│  │   (Peer Data)   │  │   (Sessions)    │                       │
│  └─────────────────┘  └─────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required
- **Domain Name**: `prsm-network.com` (or your domain)
- **DNS Management**: Ability to create A records
- **Server**: VPS or cloud VM with:
  - 2+ CPU cores
  - 4GB+ RAM
  - 20GB+ storage
  - Ubuntu 22.04 LTS or similar

### Optional (for production)
- **SSL Certificate**: Let's Encrypt (free) or commercial
- **Cloud Provider**: AWS, GCP, DigitalOcean, etc.
- **Monitoring**: Prometheus + Grafana stack

---

## Task 1: DNS Configuration

Create these DNS A records pointing to your server's IP address:

| Subdomain | Record Type | TTL | Purpose |
|-----------|-------------|-----|---------|
| `bootstrap.prsm-network.com` | A | 300 | Primary bootstrap node |
| `fallback1.prsm-network.com` | A | 300 | Fallback bootstrap #1 |
| `fallback2.prsm-network.com` | A | 300 | Fallback bootstrap #2 |

### Example (using Cloudflare)
```bash
# Get your server's public IP
curl -s ifconfig.me

# Create DNS records via Cloudflare dashboard or API
# For initial deployment, all three can point to the same IP
```

### Example (using AWS Route 53)
```bash
# Create hosted zone if not exists
aws route53 create-hosted-zone \
  --name prsm-network.com \
  --caller-reference "$(date +%s)"

# Create A record
aws route53 change-resource-record-sets \
  --hosted-zone-id YOUR_ZONE_ID \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "bootstrap.prsm-network.com",
        "Type": "A",
        "TTL": 300,
        "ResourceRecords": [{"Value": "YOUR_SERVER_IP"}]
      }
    }]
  }'
```

---

## Task 2: Server Deployment

### Option A: Docker on VPS (Recommended for simplicity)

#### Step 1: Prepare Server
```bash
# SSH into your server
ssh root@your-server-ip

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
usermod -aG docker $USER

# Install Docker Compose (if not included)
apt install docker-compose-plugin -y

# Clone repository
git clone https://github.com/Ryno2390/PRSM.git
cd PRSM
```

#### Step 2: Configure Environment
```bash
# Create environment file
cat > .env << 'EOF'
# Server Configuration
PRSM_ENV=production
PRSM_LOG_LEVEL=INFO
PRSM_DOMAIN=prsm-network.com
PRSM_REGION=us-east-1

# Network Configuration
PRSM_BOOTSTRAP_PORT=8765
PRSM_API_PORT=8000
PRSM_MAX_PEERS=1000
PRSM_PEER_TIMEOUT=300
PRSM_HEARTBEAT_INTERVAL=30

# SSL Configuration
PRSM_SSL_ENABLED=true
PRSM_SSL_CERT_PATH=/etc/ssl/certs/prsm.crt
PRSM_SSL_KEY_PATH=/etc/ssl/private/prsm.key

# Database
POSTGRES_PASSWORD=your_secure_password_here

# Auth Secret (generate with: openssl rand -hex 32)
PRSM_AUTH_SECRET=your_auth_secret_here

# External IP (your server's public IP)
PRSM_EXTERNAL_IP=YOUR_SERVER_IP
EOF
```

#### Step 3: SSL Certificate Setup

**Using Let's Encrypt (Recommended):**
```bash
# Install certbot
apt install certbot -y

# Create SSL directory
mkdir -p /etc/ssl/prsm/certs /etc/ssl/prsm/private

# Obtain certificate
certbot certonly --standalone \
  -d bootstrap.prsm-network.com \
  --non-interactive --agree-tos \
  --email your-email@example.com

# Copy certificates
cp /etc/letsencrypt/live/bootstrap.prsm-network.com/fullchain.pem /etc/ssl/prsm/certs/prsm.crt
cp /etc/letsencrypt/live/bootstrap.prsm-network.com/privkey.pem /etc/ssl/prsm/private/prsm.key
chmod 600 /etc/ssl/prsm/private/prsm.key

# Set up auto-renewal
crontab -e
# Add: 0 0 1 * * certbot renew --quiet && cp /etc/letsencrypt/live/bootstrap.prsm-network.com/*.pem /etc/ssl/prsm/
```

**Self-signed (Development only):**
```bash
# Generate self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/prsm/private/prsm.key \
  -out /etc/ssl/prsm/certs/prsm.crt \
  -subj "/CN=bootstrap.prsm-network.com"
```

#### Step 4: Start Services
```bash
# Start bootstrap server with all dependencies
cd docker
docker compose -f docker-compose.bootstrap.yml up -d

# Check status
docker compose -f docker-compose.bootstrap.yml ps

# View logs
docker compose -f docker-compose.bootstrap.yml logs -f bootstrap
```

### Option B: AWS Deployment

```bash
# From your local machine with AWS CLI configured
./scripts/deploy_bootstrap_aws.sh -r us-east-1 -e production

# The script handles:
# - VPC and network setup
# - EC2 instance provisioning
# - Security group configuration
# - SSL certificate via ACM
# - CloudWatch logging
```

### Option C: GCP Deployment

```bash
# From your local machine with gcloud CLI configured
./scripts/deploy_bootstrap_gcp.sh -p your-project -r us-east1 -e production

# The script handles:
# - VPC network creation
# - Compute Engine instance
# - Cloud DNS configuration
# - Firewall rules
```

---

## Task 3: Firewall Configuration

### Required Ports

| Port | Protocol | Purpose | Source |
|------|----------|---------|--------|
| 22 | TCP | SSH | Your IP only |
| 80 | TCP | HTTP (certbot) | Anywhere |
| 443 | TCP | HTTPS | Anywhere |
| 8765 | TCP | WebSocket P2P | Anywhere |
| 8000 | TCP | Health/Metrics API | Anywhere |

### UFW (Ubuntu)
```bash
# Enable firewall
ufw enable

# Allow required ports
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 8765/tcp
ufw allow 8000/tcp

# Check status
ufw status
```

### AWS Security Group
```bash
# Create security group
aws ec2 create-security-group \
  --group-name prsm-bootstrap \
  --description "PRSM Bootstrap Server"

# Add rules
aws ec2 authorize-security-group-ingress \
  --group-name prsm-bootstrap \
  --protocol tcp --port 22 --cidr YOUR_IP/32
aws ec2 authorize-security-group-ingress \
  --group-name prsm-bootstrap \
  --protocol tcp --port 8765 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress \
  --group-name prsm-bootstrap \
  --protocol tcp --port 8000 --cidr 0.0.0.0/0
```

---

## Task 4: Verification

### Test WebSocket Connectivity

```python
# test_bootstrap.py
import asyncio
import websockets

async def test_bootstrap():
    uri = "wss://bootstrap.prsm-network.com:8765"
    try:
        async with websockets.connect(uri) as ws:
            print(f"✓ Connected to {uri}")
            # Send a ping
            await ws.send('{"type": "ping"}')
            response = await ws.recv()
            print(f"✓ Received response: {response}")
            return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

asyncio.run(test_bootstrap())
```

### Test Health Endpoint

```bash
# HTTP health check
curl -s https://bootstrap.prsm-network.com:8000/health | jq

# Expected response:
# {
#   "status": "healthy",
#   "peers": 0,
#   "uptime_seconds": 1234,
#   "version": "1.0.0"
# }
```

### Test from Local Node

```bash
# Install PRSM
pip install prsm

# Start node with bootstrap
prsm node start --no-dashboard

# Expected output:
# ✓ Bootstrap success via wss://bootstrap.prsm-network.com
# ✓ Discovered 0 peers
# ✓ Node started successfully
```

---

## Task 5: Monitoring (Optional)

### Enable Prometheus + Grafana

```bash
# Start with monitoring profile
docker compose -f docker-compose.bootstrap.yml --profile monitoring up -d

# Access Grafana at http://your-server:3000
# Default credentials: admin/admin
```

### Key Metrics to Monitor

| Metric | Alert Threshold |
|--------|-----------------|
| `prsm_bootstrap_active_connections` | > 800 (warning) |
| `prsm_bootstrap_total_peers` | < 10 for 10m (warning) |
| `prsm_bootstrap_error_rate` | > 10/sec (warning) |
| `prsm_bootstrap_uptime` | < 60s (critical) |

---

## Local Development Testing

If you don't have cloud access, test locally:

```bash
# Start local bootstrap server
docker compose -f docker/docker-compose.bootstrap-local.yml up -d

# Wait for services to be healthy
docker compose -f docker/docker-compose.bootstrap-local.yml ps

# Test WebSocket (non-SSL for local)
python -c "
import asyncio
import websockets

async def test():
    async with websockets.connect('ws://localhost:8765') as ws:
        await ws.send('{\"type\": \"ping\"}')
        print('Response:', await ws.recv())

asyncio.run(test())
"

# Test health endpoint
curl http://localhost:8000/health

# Stop services
docker compose -f docker/docker-compose.bootstrap-local.yml down
```

---

## Troubleshooting

### Common Issues

#### 1. WebSocket Connection Refused
```bash
# Check if service is running
docker compose -f docker-compose.bootstrap.yml ps

# Check logs
docker compose -f docker-compose.bootstrap.yml logs bootstrap

# Verify port is listening
netstat -tlnp | grep 8765
```

#### 2. SSL Certificate Errors
```bash
# Verify certificate exists
ls -la /etc/ssl/prsm/certs/prsm.crt
ls -la /etc/ssl/prsm/private/prsm.key

# Check certificate validity
openssl x509 -in /etc/ssl/prsm/certs/prsm.crt -text -noout

# Test SSL connection
openssl s_client -connect bootstrap.prsm-network.com:8765
```

#### 3. Database Connection Issues
```bash
# Check PostgreSQL is running
docker compose -f docker-compose.bootstrap.yml ps postgres

# Test database connection
docker compose -f docker-compose.bootstrap.yml exec postgres \
  psql -U prsm -d prsm -c "SELECT 1"
```

#### 4. DNS Not Resolving
```bash
# Check DNS resolution
dig bootstrap.prsm-network.com
nslookup bootstrap.prsm-network.com

# Flush DNS cache
sudo systemd-resolve --flush-caches
```

---

## Production Checklist

Before going live, verify:

- [ ] DNS records created and propagating
- [ ] SSL certificate installed and valid
- [ ] Firewall rules configured correctly
- [ ] Health endpoint returns `{"status": "healthy"}`
- [ ] WebSocket connection succeeds from external network
- [ ] Local `prsm node start` bootstraps successfully
- [ ] Monitoring and alerting configured
- [ ] Log aggregation set up
- [ ] Backup strategy for PostgreSQL data
- [ ] Auto-renewal for SSL certificates

---

## Cost Estimates

### VPS (DigitalOcean/Linode/Vultr)
- **Basic**: $12-20/month (2 CPU, 4GB RAM)
- **Recommended**: $24-40/month (4 CPU, 8GB RAM)

### AWS
- **EC2 t3.medium**: ~$30/month
- **RDS PostgreSQL**: ~$15/month (or use containerized)
- **ElastiCache Redis**: ~$15/month (or use containerized)
- **Data transfer**: ~$0.09/GB
- **Total**: ~$60-100/month

### GCP
- **e2-medium**: ~$25/month
- **Cloud SQL**: ~$15/month (or use containerized)
- **Memorystore**: ~$15/month (or use containerized)
- **Total**: ~$55-90/month

---

## Next Steps

After successful deployment:

1. **Add Fallback Servers**: Deploy additional bootstrap servers in different regions
2. **Configure Federation**: Link bootstrap servers for peer sharing
3. **Set Up Monitoring**: Integrate with your existing monitoring stack
4. **Document Runbooks**: Create operational procedures for common issues
5. **Load Testing**: Verify the server handles expected peer counts

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/Ryno2390/PRSM/issues
- Documentation: `docs/BOOTSTRAP_DEPLOYMENT.md`
