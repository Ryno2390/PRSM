# PRSM Deployment Guide

This guide covers deploying PRSM (Protocol for Recursive Scientific Modeling) in various environments from development to production.

## 🏗️ Architecture Overview

PRSM is a distributed system with the following components:

### Core Services
- **PRSM API** - FastAPI application (Python 3.11+)
- **PostgreSQL** - Primary database with Alembic migrations
- **Redis** - Caching and task queues
- **IPFS** - Distributed storage for models and data

### Vector Databases
- **Weaviate** - Primary vector database
- **ChromaDB** - Alternative vector database (optional)

### Monitoring & Observability
- **Prometheus** - Metrics collection
- **Grafana** - Dashboards and visualization
- **Nginx** - Reverse proxy and load balancing (optional)

## 🚀 Quick Start

### Prerequisites

- Docker 24.0+ and Docker Compose v2.0+
- 8GB+ RAM recommended
- 50GB+ storage for production

### Development Deployment

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ryno2390/PRSM.git
   cd PRSM
   ```

2. **Setup environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Deploy development stack**
   ```bash
   ./scripts/deploy.sh dev
   ```

4. **Access services**
   - PRSM API: http://localhost:8000
   - Grafana: http://localhost:3000
   - PostgreSQL: localhost:5432
   - Redis: localhost:6379

### Production Deployment

1. **Prepare production environment**
   ```bash
   cp .env.example .env
   # Configure production values in .env
   ```

2. **Deploy production stack**
   ```bash
   ./scripts/deploy.sh prod
   ```

3. **Verify deployment**
   ```bash
   ./scripts/deploy.sh health
   ```

## 📋 Environment Configuration

### Required Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@host:port/database
POSTGRES_PASSWORD=your_secure_password

# Redis
REDIS_URL=redis://host:port/db

# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
HUGGINGFACE_API_KEY=your_huggingface_key

# Vector Databases
PINECONE_API_KEY=your_pinecone_key
WEAVIATE_URL=http://weaviate:8080

# Security
JWT_SECRET_KEY=your_jwt_secret
ENCRYPTION_KEY=your_encryption_key

# Monitoring
GRAFANA_PASSWORD=your_grafana_password
```

### Production-Specific Variables

```bash
# Application
PRSM_ENV=production
PRSM_LOG_LEVEL=INFO
PRSM_WORKERS=8

# Security
FTNS_ENABLED=true
FTNS_INITIAL_GRANT=1000

# SSL/TLS (if using)
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem
```

## 🐳 Docker Deployment

### Service Architecture

```yaml
services:
  prsm-api:        # Main application
  postgres:        # Database
  redis:           # Cache/Queue
  ipfs:           # Distributed storage
  weaviate:       # Vector database
  prometheus:     # Metrics
  grafana:        # Dashboards
  nginx:          # Proxy (optional)
```

### Docker Compose Commands

```bash
# Start all services
docker-compose up -d

# Start with specific profiles
docker-compose --profile proxy up -d

# Scale API instances
docker-compose up -d --scale prsm-api=3

# View logs
docker-compose logs -f prsm-api

# Stop services
docker-compose down

# Clean up (⚠️ removes data)
docker-compose down -v
```

### Custom Docker Builds

```bash
# Build production image
docker build -t prsm:latest .

# Build development image
docker build --target development -t prsm:dev .

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 -t prsm:latest .
```

## 🗄️ Database Management

### Initial Setup

```bash
# Run database migrations
docker-compose exec prsm-api alembic upgrade head

# Check migration status
docker-compose exec prsm-api alembic current

# Create new migration
docker-compose exec prsm-api alembic revision --autogenerate -m "Description"
```

### Backup and Restore

```bash
# Create backup
./scripts/deploy.sh backup

# Manual database backup
docker-compose exec postgres pg_dump -U prsm prsm > backup.sql

# Restore from backup
docker-compose exec -T postgres psql -U prsm prsm < backup.sql
```

## 📊 Monitoring and Observability

### Default Monitoring Stack

- **Grafana**: http://localhost:3000
  - Username: admin
  - Password: (set in GRAFANA_PASSWORD)

- **Prometheus**: http://localhost:9090
  - Metrics collection and alerting

### Key Metrics

- API response times and throughput
- Database connection pools and query performance
- Redis cache hit rates and memory usage
- IPFS storage and network statistics
- Container resource utilization

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Database connectivity
docker-compose exec prsm-api python -c "
from prsm.core.database_service import get_database_service
import asyncio
async def test():
    db = get_database_service()
    health = await db.get_health_status()
    print(health)
asyncio.run(test())
"

# Redis connectivity
docker-compose exec redis redis-cli ping
```

## 🔒 Security Considerations

### Production Security Checklist

- [ ] **Environment Variables**: Store secrets in secure environment files
- [ ] **Database Security**: Use strong passwords and connection encryption
- [ ] **API Keys**: Rotate keys regularly and use least-privilege access
- [ ] **Network Security**: Configure firewall rules and VPC settings
- [ ] **SSL/TLS**: Enable HTTPS for all external communications
- [ ] **Container Security**: Run containers as non-root users
- [ ] **Dependency Scanning**: Regularly scan for vulnerabilities
- [ ] **Access Control**: Implement proper authentication and authorization

### Security Scanning

```bash
# Vulnerability scanning
docker run --rm -v $(pwd):/app aquasec/trivy fs /app

# Dependency audit
pip-audit -r requirements.txt

# Container scanning
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image prsm:latest
```

## 📈 Scaling and Performance

### Horizontal Scaling

```yaml
# docker-compose.override.yml
services:
  prsm-api:
    deploy:
      replicas: 3
    
  nginx:
    # Enable load balancer
    profiles: ["proxy"]
```

### Performance Tuning

1. **Database Optimization**
   ```sql
   -- PostgreSQL settings
   shared_buffers = 256MB
   effective_cache_size = 1GB
   max_connections = 200
   ```

2. **Redis Configuration**
   ```bash
   # Memory optimization
   maxmemory 512mb
   maxmemory-policy allkeys-lru
   ```

3. **Application Tuning**
   ```bash
   # Environment variables
   PRSM_WORKERS=8  # CPU cores * 2
   UVICORN_BACKLOG=2048
   ```

### Load Testing

```bash
# Simple load test
for i in {1..100}; do
  curl -f http://localhost:8000/health &
done
wait

# Apache Bench
ab -n 1000 -c 10 http://localhost:8000/health

# Locust (install first)
locust -f tests/load_test.py --host=http://localhost:8000
```

## 🚨 Troubleshooting

### Common Issues

1. **Container Won't Start**
   ```bash
   # Check logs
   docker-compose logs prsm-api
   
   # Check resource usage
   docker stats
   
   # Verify environment
   docker-compose config
   ```

2. **Database Connection Issues**
   ```bash
   # Test connectivity
   docker-compose exec prsm-api python -c "
   import asyncio
   from prsm.core.database import test_connection
   asyncio.run(test_connection())
   "
   
   # Check database logs
   docker-compose logs postgres
   ```

3. **Memory Issues**
   ```bash
   # Check memory usage
   docker stats --no-stream
   
   # Adjust memory limits
   # Add to docker-compose.yml:
   deploy:
     resources:
       limits:
         memory: 2G
   ```

### Debug Mode

```bash
# Enable debug logging
PRSM_LOG_LEVEL=DEBUG docker-compose up

# Access container shell
docker-compose exec prsm-api bash

# Check Python dependencies
docker-compose exec prsm-api pip list
```

## 🔄 CI/CD Pipeline

### GitHub Actions

The repository includes automated CI/CD pipelines:

- **CI Pipeline** (`.github/workflows/ci.yml`)
  - Code quality checks (linting, type checking)
  - Unit and integration tests
  - Security scanning
  - Docker image building

- **CD Pipeline** (`.github/workflows/cd.yml`)
  - Automated staging deployment
  - Production deployment on releases
  - Health checks and monitoring
  - Rollback capabilities

### Manual Deployment

```bash
# Build and push images
docker build -t ghcr.io/ryno2390/prsm:latest .
docker push ghcr.io/ryno2390/prsm:latest

# Deploy to staging
kubectl apply -f k8s/staging/

# Deploy to production
kubectl apply -f k8s/production/
```

## 📝 Maintenance

### Regular Tasks

1. **Database Maintenance**
   ```bash
   # Vacuum and analyze
   docker-compose exec postgres psql -U prsm prsm -c "VACUUM ANALYZE;"
   
   # Check database size
   docker-compose exec postgres psql -U prsm prsm -c "
   SELECT pg_size_pretty(pg_database_size('prsm'));
   "
   ```

2. **Log Rotation**
   ```bash
   # Clean old logs
   docker system prune -f
   
   # Rotate application logs
   find logs/ -name "*.log" -mtime +30 -delete
   ```

3. **Security Updates**
   ```bash
   # Update base images
   docker-compose pull
   docker-compose up -d
   
   # Update Python dependencies
   pip install -r requirements.txt --upgrade
   ```

### Backup Strategy

- **Daily**: Database snapshots
- **Weekly**: Full system backup including volumes
- **Monthly**: Archive old logs and data
- **Quarterly**: Disaster recovery testing

## 📞 Support

### Getting Help

- **Documentation**: Check `/docs` directory
- **Logs**: Use `docker-compose logs` for debugging
- **Health Checks**: Monitor `/health` endpoints
- **Metrics**: Review Grafana dashboards

### Emergency Procedures

1. **Service Down**
   ```bash
   # Quick restart
   docker-compose restart prsm-api
   
   # Full recovery
   ./scripts/deploy.sh stop
   ./scripts/deploy.sh prod
   ```

2. **Database Issues**
   ```bash
   # Rollback migration
   docker-compose exec prsm-api alembic downgrade -1
   
   # Restore from backup
   ./scripts/restore-backup.sh latest
   ```

3. **Performance Issues**
   ```bash
   # Scale up
   docker-compose up -d --scale prsm-api=5
   
   # Check resource usage
   docker stats
   ```

---

For additional support or questions, please refer to the project documentation or contact the PRSM development team.