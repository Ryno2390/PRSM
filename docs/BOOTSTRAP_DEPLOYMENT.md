# PRSM Bootstrap Server Deployment Guide

This guide covers the deployment and operation of the PRSM Bootstrap Server infrastructure.

## Overview

The Bootstrap Server is a critical component of the PRSM P2P network, serving as an entry point for new peers joining the network. It provides:

- **Peer Discovery**: New peers connect to discover other network participants
- **Connection Management**: Tracks active peers and their capabilities
- **Network Bootstrapping**: Enables decentralized network formation
- **Health Monitoring**: Provides health checks and metrics for observability

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Load Balancer (Optional)    │
                    │         (nginx/HAProxy)             │
                    └──────────────┬──────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
    ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
    │  Bootstrap   │       │  Bootstrap   │       │  Bootstrap   │
    │  Server 1    │       │  Server 2    │       │  Server N    │
    │  (Primary)   │       │  (Replica)   │       │  (Replica)   │
    └──────┬───────┘       └──────┬───────┘       └──────┬───────┘
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
            ┌──────────────┐             ┌──────────────┐
            │  PostgreSQL  │             │    Redis     │
            │  (Persist)   │             │   (Cache)    │
            └──────────────┘             └──────────────┘
```

## Quick Start

### Local Development

```bash
# Start bootstrap server with dependencies
docker-compose -f docker/docker-compose.bootstrap.yml up -d

# Check server health
curl http://localhost:8000/health

# View logs
docker logs prsm-bootstrap
```

### Production Deployment

```bash
# Deploy to AWS
./scripts/deploy_bootstrap_aws.sh --region us-east-1 --environment production

# Deploy to GCP
./scripts/deploy_bootstrap_gcp.sh --project my-project --region us-east1
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PRSM_DOMAIN` | Domain name for the server | `prsm-network.com` |
| `PRSM_BOOTSTRAP_PORT` | WebSocket port | `8765` |
| `PRSM_API_PORT` | HTTP API port | `8000` |
| `PRSM_MAX_PEERS` | Maximum tracked peers | `1000` |
| `PRSM_PEER_TIMEOUT` | Peer timeout (seconds) | `300` |
| `PRSM_HEARTBEAT_INTERVAL` | Heartbeat interval (seconds) | `30` |
| `PRSM_SSL_ENABLED` | Enable SSL/TLS | `true` |
| `PRSM_SSL_CERT_PATH` | SSL certificate path | `/etc/ssl/certs/prsm.crt` |
| `PRSM_SSL_KEY_PATH` | SSL private key path | `/etc/ssl/private/prsm.key` |
| `PRSM_REGION` | Server region identifier | `us-east-1` |
| `PRSM_AUTH_SECRET` | Authentication secret | (generated) |
| `POSTGRES_PASSWORD` | PostgreSQL password | (required) |

### Configuration File

Configuration can also be loaded from a file:

```python
from prsm.bootstrap.config import BootstrapConfig

config = BootstrapConfig(
    domain="bootstrap.prsm-network.com",
    port=8765,
    ssl_enabled=True,
    max_peers=2000,
    peer_timeout=600,
)
```

## SSL/TLS Configuration

### Certificate Requirements

1. **Domain Certificate**: For the bootstrap server domain
2. **Private Key**: Corresponding private key
3. **CA Certificate** (optional): For client certificate verification

### Let's Encrypt (Recommended)

```bash
# Install certbot
sudo apt-get install certbot

# Obtain certificate
sudo certbot certonly --standalone -d bootstrap.prsm-network.com

# Certificates will be at:
# /etc/letsencrypt/live/bootstrap.prsm-network.com/fullchain.pem
# /etc/letsencrypt/live/bootstrap.prsm-network.com/privkey.pem
```

### Self-Signed (Development Only)

```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \
  -days 365 -nodes -subj "/CN=localhost"
```

## Monitoring

### Prometheus Metrics

The bootstrap server exposes metrics at `http://localhost:8000/metrics`:

- `prsm_bootstrap_active_connections` - Current active connections
- `prsm_bootstrap_total_peers` - Total known peers
- `prsm_bootstrap_messages_processed_total` - Total messages processed
- `prsm_bootstrap_bytes_sent_total` - Total bytes sent
- `prsm_bootstrap_bytes_received_total` - Total bytes received
- `prsm_bootstrap_errors_total` - Total errors

### Grafana Dashboard

Import the dashboard from `docker/monitoring/grafana/dashboards/bootstrap-dashboard.json`:

```bash
# Access Grafana
open http://localhost:3000

# Default credentials
Username: admin
Password: ${GRAFANA_ADMIN_PASSWORD}
```

### Alerting Rules

Alert rules are defined in `docker/monitoring/alert_rules.yml`:

| Alert | Severity | Condition |
|-------|----------|-----------|
| `BootstrapServerDown` | Critical | Server unreachable for 1m |
| `TooManyActiveConnections` | Warning | > 800 connections |
| `ConnectionLimitReached` | Critical | >= 1000 connections |
| `HighErrorRate` | Warning | > 10 errors/sec |
| `LowPeerCount` | Warning | < 10 peers for 10m |

## Health Checks

### HTTP Health Check

```bash
# Basic health check
curl http://localhost:8000/health

# Response
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "active_connections": 50,
  "total_peers": 150,
  "region": "us-east-1"
}
```

### Docker Health Check

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' prsm-bootstrap
```

### Kubernetes Health Check

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

## Scaling

### Horizontal Scaling

Deploy multiple bootstrap servers behind a load balancer:

```yaml
# docker-compose.bootstrap.yml with replica
services:
  bootstrap:
    deploy:
      replicas: 3
```

### Federation

Configure federation between bootstrap servers:

```bash
PRSM_FEDERATION_PEERS=bootstrap-us-west.prsm-network.com:8765,bootstrap-eu.prsm-network.com:8765
```

## Security

### Authentication

Enable peer authentication:

```bash
PRSM_AUTH_REQUIRED=true
PRSM_AUTH_SECRET=your-secure-secret
```

### Rate Limiting

Configure rate limits:

```bash
PRSM_RATE_LIMIT_REQUESTS=100  # requests per minute
PRSM_RATE_LIMIT_WINDOW=60     # seconds
```

### IP/Peer Banning

Ban malicious peers:

```bash
PRSM_BANNED_IPS=192.168.1.100,10.0.0.50
PRSM_BANNED_PEERS=malicious-peer-id
```

### Firewall Rules

Required open ports:

| Port | Protocol | Purpose |
|------|----------|---------|
| 8765 | TCP | WebSocket P2P |
| 8000 | TCP | HTTP API |
| 9090 | TCP | Prometheus metrics (internal) |
| 22 | TCP | SSH (management) |
| 443 | TCP | HTTPS (if using load balancer) |

## Troubleshooting

### Common Issues

#### Server Won't Start

```bash
# Check logs
docker logs prsm-bootstrap

# Common causes:
# - Port already in use
# - SSL certificate not found
# - Database connection failed
```

#### Peers Can't Connect

```bash
# Check firewall
sudo ufw status

# Check if port is open
netstat -tlnp | grep 8765

# Check SSL certificate
openssl s_client -connect bootstrap.prsm-network.com:8765
```

#### High Memory Usage

```bash
# Check peer count
curl http://localhost:8000/metrics | grep prsm_bootstrap_total_peers

# Reduce max peers
PRSM_MAX_PEERS=500
```

### Logs

```bash
# Docker logs
docker logs -f prsm-bootstrap

# Application logs
tail -f /app/logs/bootstrap.log

# With journalctl (systemd)
journalctl -u prsm-bootstrap -f
```

## Rollback Procedures

### Docker Rollback

```bash
# Stop current version
docker-compose -f docker/docker-compose.bootstrap.yml down

# Checkout previous version
git checkout v1.0.0

# Restart
docker-compose -f docker/docker-compose.bootstrap.yml up -d
```

### Kubernetes Rollback

```bash
# Rollback to previous deployment
kubectl rollout undo deployment/prsm-bootstrap -n prsm

# Rollback to specific revision
kubectl rollout undo deployment/prsm-bootstrap -n prsm --to-revision=2
```

## Maintenance

### Database Backup

```bash
# PostgreSQL backup
docker exec prsm-bootstrap-postgres pg_dump -U prsm prsm > backup.sql

# Restore
docker exec -i prsm-bootstrap-postgres psql -U prsm prsm < backup.sql
```

### Certificate Renewal

```bash
# Let's Encrypt renewal
sudo certbot renew

# Restart server to load new certificate
docker restart prsm-bootstrap
```

### Log Rotation

Logs are automatically rotated via Docker:

```json
{
  "log-opts": {
    "max-size": "100m",
    "max-file": "5"
  }
}
```

## Support

For issues and support:

1. Check the [troubleshooting guide](#troubleshooting)
2. Review [GitHub Issues](https://github.com/prsm-network/prsm/issues)
3. Join the [Discord community](https://discord.gg/prsm)
