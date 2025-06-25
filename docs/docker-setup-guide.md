# PRSM Docker Setup Guide

Choose the right Docker configuration for your needs and experience level.

## 🚀 Quick Selection Guide

| Scenario | Configuration | Startup Time | Use When |
|----------|--------------|--------------|----------|
| **First Time User** | `quickstart` | 30 seconds | Testing PRSM for first time |
| **Tutorial Learner** | `tutorial` | 2 minutes | Following documentation tutorials |
| **New Developer** | `onboarding` | 1 minute | Starting PRSM development |
| **Full Development** | `dev` | 3 minutes | Active development work |
| **Production** | `production` | 5 minutes | Deploying to production |

## ⚡ Quickstart (30 seconds)

**Best for**: First-time users, testing, quick demos

```bash
# Start minimal services (Redis + IPFS)
docker-compose -f docker-compose.quickstart.yml up -d

# Verify services are running
docker-compose -f docker-compose.quickstart.yml ps

# Test PRSM
python examples/tutorials/hello_world_complete.py

# Stop when done
docker-compose -f docker-compose.quickstart.yml down
```

**What you get:**
- ✅ Redis (in-memory, no persistence)
- ✅ IPFS (minimal test profile)
- ✅ Works with all tutorials
- ⚠️ No data persistence
- ⚠️ No monitoring tools

## 📚 Tutorial Environment (2 minutes)

**Best for**: Following tutorials, learning PRSM

```bash
# Start tutorial environment
docker-compose -f docker-compose.tutorial.yml up -d

# Access tutorial dashboard
open http://localhost:3000

# Access Redis monitoring
open http://localhost:8081
# Login: tutorial / prsm_tutorial

# Optional: Enable Jupyter notebooks
docker-compose -f docker-compose.tutorial.yml --profile jupyter up -d
open http://localhost:8888
# Token: prsm_tutorial_jupyter

# Optional: Enable web-based VS Code
docker-compose -f docker-compose.tutorial.yml --profile web-dev up -d
open http://localhost:8443
# Password: prsm_tutorial_code
```

**What you get:**
- ✅ Persistent Redis + IPFS
- ✅ Redis monitoring dashboard
- ✅ Tutorial documentation server
- ✅ Optional Jupyter notebooks
- ✅ Optional web-based VS Code
- ✅ Perfect for learning

## 🛠️ Developer Onboarding (1 minute)

**Best for**: New developers getting started

```bash
# Start onboarding environment
docker-compose -f docker-compose.onboarding.yml up -d

# Optional: Add development tools
docker-compose -f docker-compose.onboarding.yml --profile tools up -d

# Access tools:
# Redis Commander: http://localhost:8082
# SQLite Browser: http://localhost:3001
# IPFS Web UI: http://localhost:5002
```

**What you get:**
- ✅ Minimal, fast startup
- ✅ Essential services only
- ✅ Optional development tools
- ✅ Memory-optimized for development
- ✅ Good balance of features vs speed

## 🔧 Full Development (3 minutes)

**Best for**: Active development, debugging

```bash
# Start full development environment
docker-compose -f docker-compose.dev.yml up -d

# Access development tools:
# PgAdmin: http://localhost:5050 (admin@prsm.dev / dev_password)
# Redis Commander: http://localhost:8081
# Jupyter: http://localhost:8888 (token: prsm_dev_token)
```

**What you get:**
- ✅ PostgreSQL database
- ✅ PgAdmin for database management
- ✅ Redis Commander
- ✅ Jupyter notebooks
- ✅ Live code reloading
- ✅ Development logging
- ✅ Full feature set

## 🏢 Production (5 minutes)

**Best for**: Production deployment

```bash
# Create production environment file
cp .env.example .env
# Edit .env with production values

# Start production stack
docker-compose up -d

# Access monitoring:
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
# Alertmanager: http://localhost:9093
```

**What you get:**
- ✅ Full production stack
- ✅ Monitoring and alerting
- ✅ Load balancing
- ✅ Health checks
- ✅ Security hardening
- ✅ Scalable architecture

## 🎯 Quick Commands Reference

### Common Operations

```bash
# Check service status
docker-compose -f [compose-file] ps

# View logs
docker-compose -f [compose-file] logs [service-name]

# Stop all services
docker-compose -f [compose-file] down

# Stop and remove data
docker-compose -f [compose-file] down -v

# Update images
docker-compose -f [compose-file] pull
docker-compose -f [compose-file] up -d
```

### Troubleshooting

```bash
# Check service health
docker-compose -f [compose-file] ps --format table

# View specific service logs
docker-compose -f [compose-file] logs redis
docker-compose -f [compose-file] logs ipfs

# Restart a service
docker-compose -f [compose-file] restart redis

# Clean up everything
docker-compose -f [compose-file] down -v --remove-orphans
docker system prune -f
```

## 🔧 Environment Configuration

### Required Environment Variables

Create `.env` file in project root:

```bash
# Required for production
POSTGRES_PASSWORD=your_secure_password
GRAFANA_PASSWORD=your_grafana_password

# API Keys (required for AI functionality)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
HUGGINGFACE_API_KEY=your_huggingface_key

# Optional: Notification integrations
SLACK_WEBHOOK_URL=your_slack_webhook
SMTP_PASSWORD=your_smtp_password
```

### Service-Specific Configuration

All configurations are in the `config/` directory:
- `config/api_keys.env.example` - API key template
- `config/redis.conf` - Redis configuration
- `config/prometheus.yml` - Monitoring setup
- `config/grafana/` - Dashboard configurations

## 🚨 Common Issues & Solutions

### Port Conflicts

```bash
# If ports are already in use, modify docker-compose files:
ports:
  - "6380:6379"  # Use different host port

# Or stop conflicting services:
sudo systemctl stop redis
sudo systemctl stop postgresql
```

### Permission Issues

```bash
# Fix Docker permissions (Linux/Mac):
sudo usermod -aG docker $USER
# Logout and login again

# Fix file permissions:
sudo chown -R $USER:$USER ./dev-data/
```

### Service Won't Start

```bash
# Check logs for specific error:
docker-compose -f [compose-file] logs [service-name]

# Most common issues:
# 1. Port already in use
# 2. Insufficient disk space
# 3. Missing environment variables
# 4. Docker daemon not running
```

### Reset Everything

```bash
# Nuclear option - removes all PRSM Docker data:
docker-compose -f docker-compose.quickstart.yml down -v
docker-compose -f docker-compose.tutorial.yml down -v
docker-compose -f docker-compose.onboarding.yml down -v
docker-compose -f docker-compose.dev.yml down -v
docker-compose down -v

# Remove all PRSM volumes:
docker volume ls | grep prsm | awk '{print $2}' | xargs docker volume rm

# Remove all unused Docker resources:
docker system prune -af --volumes
```

## 📊 Performance Tuning

### Memory Usage by Configuration

| Configuration | Memory Usage | Suitable For |
|--------------|--------------|--------------|
| Quickstart | ~100MB | Testing, tutorials |
| Tutorial | ~300MB | Learning, examples |
| Onboarding | ~200MB | Initial development |
| Development | ~800MB | Active development |
| Production | ~2GB+ | Production workloads |

### Optimization Tips

```bash
# Allocate more memory to Docker (Docker Desktop):
# Settings → Resources → Memory: 4GB+

# For low-memory systems, use quickstart configuration
# For development, use onboarding configuration
# For full features, use development configuration
```

## 🎓 Next Steps

After setting up Docker:

1. **Verify Setup**: `prsm-dev status`
2. **Run Tutorial**: `python examples/tutorials/hello_world_complete.py`
3. **Learn Concepts**: Follow [Tutorial Guide](./tutorials/README.md)
4. **Build Something**: Start with [API Fundamentals](./tutorials/02-foundation/api-fundamentals.md)

---

**Need Help?** Check the [Troubleshooting Guide](./TROUBLESHOOTING_GUIDE.md) or join our [Discord community](https://discord.gg/prsm).