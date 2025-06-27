# PRSM Administrator Guide

Comprehensive guide for PRSM system administrators, covering installation, configuration, monitoring, and maintenance of PRSM production deployments.

## 🚀 System Requirements

### Minimum Requirements
- **CPU**: 4 cores, 2.5GHz
- **Memory**: 16GB RAM
- **Storage**: 100GB SSD
- **Network**: 1Gbps connection
- **OS**: Linux (Ubuntu 20.04+, RHEL 8+, CentOS 8+)

### Recommended Production Requirements
- **CPU**: 16+ cores, 3.0GHz
- **Memory**: 64GB+ RAM
- **Storage**: 1TB+ NVMe SSD
- **Network**: 10Gbps connection
- **Load Balancer**: HAProxy, NGINX, or cloud LB

## 🔧 Installation & Setup

### Quick Installation
```bash
# Download PRSM installer
curl -sSL https://install.prsm.ai | bash

# Or using Docker
docker run -d --name prsm-server \
  -p 8000:8000 \
  -e PRSM_API_KEY=your-key \
  prsm/server:latest
```

### Production Installation
```bash
# Clone repository
git clone https://github.com/Ryno2390/PRSM.git
cd PRSM

# Install dependencies
./scripts/install-dependencies.sh

# Configure environment
cp config/production.env.example .env
# Edit .env with your configuration

# Database setup
alembic upgrade head

# Start services
docker-compose -f docker-compose.production.yml up -d
```

## ⚙️ Configuration Management

### Core Configuration Files
- **`.env`** - Environment variables and secrets
- **`config/api_keys.env`** - API key configurations
- **`config/production.yml`** - Production settings
- **`config/nginx/`** - Web server configuration
- **`config/prometheus.yml`** - Monitoring configuration

### Database Configuration
```yaml
# config/database.yml
database:
  host: localhost
  port: 5432
  name: prsm_production
  user: prsm_admin
  password: ${DATABASE_PASSWORD}
  pool_size: 20
  max_overflow: 50
```

### Redis Configuration
```yaml
# config/redis.yml
redis:
  host: localhost
  port: 6379
  db: 0
  password: ${REDIS_PASSWORD}
  max_connections: 100
```

## 🔒 Security Administration

### Authentication Setup
```bash
# Generate JWT secrets
openssl rand -hex 32 > config/jwt_secret.key

# Set up API key authentication
python scripts/generate_api_keys.py --admin --count 10

# Configure TLS certificates
certbot certonly --nginx -d api.yourprsm.com
```

### Firewall Configuration
```bash
# UFW example
ufw allow 22/tcp    # SSH
ufw allow 443/tcp   # HTTPS
ufw allow 80/tcp    # HTTP (redirect to HTTPS)
ufw deny 8000/tcp   # Block direct API access
ufw enable
```

### Security Hardening
- Enable fail2ban for SSH protection
- Configure rate limiting in NGINX
- Implement IP whitelisting for admin endpoints
- Regular security updates and patching
- Enable audit logging

## 📊 Monitoring & Observability

### Health Monitoring
```bash
# Check service health
curl http://localhost:8000/health

# Database health
python scripts/check_db_health.py

# Redis health
redis-cli ping
```

### Performance Monitoring
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards  
- **AlertManager**: Alert routing
- **Jaeger**: Distributed tracing

### Key Metrics to Monitor
- API response times and throughput
- Database connection pool usage
- Redis memory usage
- Token generation rate
- Error rates by endpoint
- FTNS token transaction volume

## 🔄 Backup & Recovery

### Database Backups
```bash
# Daily automated backup
pg_dump prsm_production | gzip > backup_$(date +%Y%m%d).sql.gz

# Point-in-time recovery setup
# Configure PostgreSQL WAL archiving
```

### Configuration Backups
```bash
# Backup configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz config/ .env

# Backup to S3
aws s3 cp config_backup_$(date +%Y%m%d).tar.gz s3://prsm-backups/
```

### Disaster Recovery Plan
1. **Recovery Time Objective (RTO)**: 30 minutes
2. **Recovery Point Objective (RPO)**: 1 hour
3. **Backup retention**: 30 days
4. **Testing frequency**: Monthly

## 🚀 Scaling & Performance

### Horizontal Scaling
```yaml
# docker-compose.scale.yml
services:
  prsm-api:
    replicas: 3
  prsm-worker:
    replicas: 5
```

### Database Scaling
- Read replicas for query distribution
- Connection pooling with PgBouncer
- Query optimization and indexing
- Partitioning for large tables

### Caching Strategy
- Redis for session data
- Application-level caching
- CDN for static assets
- Database query result caching

## 🔧 Maintenance Tasks

### Daily Tasks
- Review system logs for errors
- Check backup completion
- Monitor resource usage
- Review security alerts

### Weekly Tasks
- Update security patches
- Review performance metrics
- Clean up old logs and backups
- Test disaster recovery procedures

### Monthly Tasks
- Security audit and review
- Capacity planning assessment
- Update documentation
- Review and rotate API keys

## 📋 Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
free -h
# Check process memory
ps aux --sort=-%mem | head

# Solutions:
# - Increase memory limits in config
# - Optimize query patterns
# - Scale horizontally
```

#### Database Connection Issues
```bash
# Check connection pool status
python scripts/check_db_connections.py

# Solutions:
# - Increase pool size
# - Check for connection leaks
# - Restart database service
```

#### API Rate Limiting
```bash
# Check rate limit logs
grep "rate limit" /var/log/nginx/access.log

# Solutions:
# - Adjust rate limits in NGINX config
# - Implement user-specific limits
# - Scale API servers
```

### Log Analysis
```bash
# Application logs
tail -f logs/prsm.log

# Database logs
tail -f /var/log/postgresql/postgresql.log

# NGINX logs
tail -f /var/log/nginx/access.log
```

## 📞 Support & Escalation

### Internal Support
1. **Level 1**: System monitoring alerts
2. **Level 2**: Performance degradation
3. **Level 3**: Security incidents or data corruption

### External Support
- **Email**: admin-support@prsm.ai
- **Emergency**: +1-555-PRSM-911
- **Documentation**: [Operations Manual](./PRODUCTION_OPERATIONS_MANUAL.md)

### Escalation Matrix
| Severity | Response Time | Escalation |
|----------|---------------|------------|
| Critical | 15 minutes | Immediate |
| High | 1 hour | Within 2 hours |
| Medium | 4 hours | Next business day |
| Low | 24 hours | Weekly review |

## 📚 Additional Resources

### Documentation
- [Production Operations Manual](./PRODUCTION_OPERATIONS_MANUAL.md)
- [Security Hardening Guide](./SECURITY_HARDENING.md)
- [API Reference](./API_REFERENCE.md)
- [Troubleshooting Guide](./TROUBLESHOOTING_GUIDE.md)

### Training Materials
- Administrator certification program
- Video tutorials and webinars
- Best practices documentation
- Case studies and examples