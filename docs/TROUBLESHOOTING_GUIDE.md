# PRSM Troubleshooting Guide

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Common Issues](#common-issues)
3. [API & Authentication](#api--authentication)
4. [Database Issues](#database-issues)
5. [IPFS & Storage](#ipfs--storage)
6. [Model Training](#model-training)
7. [Web3 & FTNS](#web3--ftns)
8. [Performance Issues](#performance-issues)
9. [Security & Access](#security--access)
10. [Deployment Issues](#deployment-issues)
11. [Monitoring & Debugging](#monitoring--debugging)
12. [Getting Support](#getting-support)

## Quick Diagnostics

### System Health Check
```bash
# Check overall system health
curl -s http://localhost:8000/health | jq '.'

# Check component status
curl -s http://localhost:8000/health/detailed \
  -H "Authorization: Bearer YOUR_TOKEN" | jq '.'
```

### Service Status
```bash
# Check all PRSM services
docker-compose ps

# Check Kubernetes pods
kubectl get pods -n prsm-production

# Check service logs
docker-compose logs prsm-api --tail=50
kubectl logs -n prsm-production deployment/prsm-api --tail=50
```

### Database Connection Test
```bash
# Test PostgreSQL connection
docker-compose exec postgres psql -U postgres -d prsm_production -c "\dt"

# Check active connections
docker-compose exec postgres psql -U postgres -c "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"
```

### IPFS Status Check
```bash
# Check IPFS node status
docker-compose exec ipfs ipfs id

# Check IPFS swarm peers
docker-compose exec ipfs ipfs swarm peers | wc -l

# Test IPFS connectivity
docker-compose exec ipfs ipfs cat QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG/readme
```

## Common Issues

### 1. Service Won't Start

**Symptoms**: 
- Service fails to start
- Container exits immediately
- "Connection refused" errors

**Diagnosis**:
```bash
# Check Docker logs
docker-compose logs service-name

# Check port conflicts
netstat -tulpn | grep :8000

# Verify environment variables
docker-compose config
```

**Solutions**:
- Check for port conflicts and change ports if needed
- Verify all required environment variables are set
- Ensure database is running before starting API service
- Check file permissions and ownership

### 2. Database Migration Issues

**Symptoms**:
- "Table doesn't exist" errors
- Database schema mismatches
- Migration failures

**Diagnosis**:
```bash
# Check migration status
python -m alembic current

# View migration history
python -m alembic history

# Check database tables
docker-compose exec postgres psql -U postgres -d prsm_production -c "\dt"
```

**Solutions**:
```bash
# Run pending migrations
python -m alembic upgrade head

# Reset database (CAUTION - destroys data)
python scripts/migrate.py --reset

# Create new migration
python -m alembic revision --autogenerate -m "description"
```

### 3. Import/Module Errors

**Symptoms**:
- "ModuleNotFoundError" errors
- Import failures in tests or runtime

**Diagnosis**:
```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Test specific imports
python -c "from prsm.core.models import User"

# Check installed packages
pip list | grep prsm
```

**Solutions**:
```bash
# Install package in development mode
pip install -e .

# Reinstall dependencies
pip install -r requirements.txt

# Check for circular imports
python -m prsm.api.main --help
```

## API & Authentication

### Authentication Failures

**Symptoms**:
- "401 Unauthorized" responses
- "Invalid token" errors
- Login failures

**Diagnosis**:
```bash
# Test login endpoint
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test"}'

# Validate JWT token
python -c "
import jwt
token = 'your_token_here'
print(jwt.decode(token, options={'verify_signature': False}))
"
```

**Solutions**:
- Verify username/password combinations
- Check JWT secret key configuration
- Ensure system clock is synchronized
- Regenerate tokens if expired

### API Rate Limiting

**Symptoms**:
- "429 Too Many Requests" errors
- Requests being blocked

**Diagnosis**:
```bash
# Check rate limit headers
curl -I http://localhost:8000/api/v1/health

# Monitor Redis rate limit keys
docker-compose exec redis redis-cli keys "*rate_limit*"
```

**Solutions**:
- Wait for rate limit window to reset
- Implement request queuing/retry logic
- Contact admin for rate limit increases
- Use API keys instead of user tokens

### API Response Issues

**Symptoms**:
- Slow API responses
- Timeout errors
- Empty responses

**Diagnosis**:
```bash
# Test API response time
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/api/v1/health

# Check API logs
docker-compose logs prsm-api | grep ERROR

# Monitor active requests
curl http://localhost:8000/health/detailed -H "Authorization: Bearer TOKEN"
```

**Solutions**:
- Scale API instances horizontally
- Optimize database queries
- Enable caching for frequently accessed data
- Check for memory leaks or blocking operations

## Database Issues

### Connection Pool Exhausted

**Symptoms**:
- "too many connections" errors
- Long database connection waits
- Application hangs

**Diagnosis**:
```sql
-- Check active connections
SELECT count(*), state FROM pg_stat_activity GROUP BY state;

-- Check max connections
SHOW max_connections;

-- Check connection pool settings
SELECT * FROM pg_settings WHERE name LIKE '%connection%';
```

**Solutions**:
- Increase PostgreSQL max_connections
- Optimize connection pool settings
- Fix connection leaks in application code
- Scale database or use connection pooler

### Slow Database Queries

**Symptoms**:
- Slow API responses
- High database CPU usage
- Query timeouts

**Diagnosis**:
```sql
-- Find slow queries
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;

-- Check locks
SELECT * FROM pg_locks WHERE NOT granted;

-- Analyze table statistics
ANALYZE;
```

**Solutions**:
- Add appropriate database indexes
- Optimize slow queries
- Update table statistics with ANALYZE
- Consider query result caching

### Database Disk Space

**Symptoms**:
- Database writes failing
- "No space left on device" errors

**Diagnosis**:
```bash
# Check disk space
df -h

# Check database size
docker-compose exec postgres psql -U postgres -c "
SELECT pg_size_pretty(pg_database_size('prsm_production'));
"

# Check largest tables
docker-compose exec postgres psql -U postgres -d prsm_production -c "
SELECT table_name, pg_size_pretty(pg_total_relation_size(table_name::regclass)) 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY pg_total_relation_size(table_name::regclass) DESC LIMIT 10;
"
```

**Solutions**:
- Clean up old data and logs
- Vacuum and reindex database tables
- Extend disk space
- Implement data retention policies

## IPFS & Storage

### IPFS Node Connection Issues

**Symptoms**:
- "Connection refused" to IPFS
- Content not found errors
- Slow file retrievals

**Diagnosis**:
```bash
# Check IPFS daemon status
docker-compose exec ipfs ipfs id

# Test IPFS connectivity
docker-compose exec ipfs ipfs swarm peers

# Check IPFS configuration
docker-compose exec ipfs ipfs config show
```

**Solutions**:
- Restart IPFS daemon
- Check firewall settings for IPFS ports
- Add bootstrap nodes
- Clear IPFS cache and re-sync

### File Upload/Download Issues

**Symptoms**:
- Upload timeouts
- File corruption
- Missing files

**Diagnosis**:
```bash
# Test file upload
curl -X POST http://localhost:8000/api/v1/files/upload \
  -H "Authorization: Bearer TOKEN" \
  -F "file=@test.txt"

# Check IPFS storage
docker-compose exec ipfs ipfs stats repo

# Verify file integrity
docker-compose exec ipfs ipfs cat QmHASH
```

**Solutions**:
- Increase upload timeout limits
- Check available storage space
- Verify file permissions
- Use IPFS pin commands to ensure persistence

### IPFS Performance Issues

**Symptoms**:
- Slow file access
- High latency for retrievals
- Frequent timeouts

**Diagnosis**:
```bash
# Monitor IPFS performance
docker-compose exec ipfs ipfs stats bw

# Check peer connections
docker-compose exec ipfs ipfs swarm peers | wc -l

# Test retrieval performance
time docker-compose exec ipfs ipfs cat QmHASH > /dev/null
```

**Solutions**:
- Add more IPFS gateway nodes
- Enable IPFS clustering
- Implement content caching
- Use IPFS accelerated DHT

## Model Training

### Training Job Failures

**Symptoms**:
- Training jobs stuck or failing
- Out of memory errors during training
- Model artifacts not saved

**Diagnosis**:
```bash
# Check training job status
curl http://localhost:8000/api/v1/training/jobs \
  -H "Authorization: Bearer TOKEN"

# Monitor GPU/CPU usage
nvidia-smi  # for GPU
htop        # for CPU

# Check training logs
docker-compose logs prsm-worker | grep training
```

**Solutions**:
- Reduce batch size or model size
- Increase available memory/GPU memory
- Check for data loader issues
- Verify training data format

### Model Loading Issues

**Symptoms**:
- "Model not found" errors
- Model loading timeouts
- Inference failures

**Diagnosis**:
```bash
# List available models
curl http://localhost:8000/api/v1/models \
  -H "Authorization: Bearer TOKEN"

# Test model loading
python -c "
from prsm.agents.executors.model_executor import ModelExecutor
executor = ModelExecutor()
print(executor.list_available_models())
"
```

**Solutions**:
- Verify model files exist in IPFS
- Check model format compatibility
- Update model registry
- Clear model cache and reload

### Training Resource Issues

**Symptoms**:
- Training jobs queued indefinitely
- Resource allocation failures
- Out of memory errors

**Diagnosis**:
```bash
# Check available resources
docker stats

# Monitor job queue
curl http://localhost:8000/api/v1/training/queue \
  -H "Authorization: Bearer TOKEN"

# Check resource limits
kubectl describe node  # for Kubernetes
```

**Solutions**:
- Scale training infrastructure
- Optimize resource requests
- Implement job prioritization
- Use distributed training

## Web3 & FTNS

### Wallet Connection Issues

**Symptoms**:
- Cannot connect to wallet
- Transaction failures
- Invalid addresses

**Diagnosis**:
```bash
# Test Web3 connection
curl http://localhost:8000/api/v1/web3/status \
  -H "Authorization: Bearer TOKEN"

# Check wallet balance
python scripts/test_web3_integration.py

# Verify network connectivity
curl https://polygon-rpc.com
```

**Solutions**:
- Check network configuration (mainnet vs testnet)
- Verify wallet has sufficient gas
- Update Web3 provider endpoints
- Check wallet permissions

### Transaction Failures

**Symptoms**:
- Transactions fail or hang
- "Insufficient gas" errors
- Network timeout errors

**Diagnosis**:
```bash
# Check transaction status
curl http://localhost:8000/api/v1/web3/transactions/TX_HASH \
  -H "Authorization: Bearer TOKEN"

# Monitor gas prices
curl https://gasstation-mainnet.matic.network/
```

**Solutions**:
- Increase gas limit for transactions
- Check gas price recommendations
- Verify network is not congested
- Retry failed transactions

### FTNS Token Issues

**Symptoms**:
- Incorrect token balances
- Transfer failures
- Staking issues

**Diagnosis**:
```bash
# Check FTNS balance
curl http://localhost:8000/api/v1/web3/balance \
  -H "Authorization: Bearer TOKEN"

# Verify contract deployment
python scripts/deploy_contracts.py --verify
```

**Solutions**:
- Refresh balance from blockchain
- Check contract interactions
- Verify token contract addresses
- Re-sync with blockchain state

## Performance Issues

### High Memory Usage

**Symptoms**:
- Application crashes with OOM
- Slow response times
- System becomes unresponsive

**Diagnosis**:
```bash
# Monitor memory usage
docker stats

# Check memory leaks
python -c "
import psutil
import gc
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Objects: {len(gc.get_objects())}')
"
```

**Solutions**:
- Increase available memory
- Implement garbage collection
- Optimize data structures
- Use memory profiling tools

### High CPU Usage

**Symptoms**:
- System sluggishness
- High load averages
- Request timeouts

**Diagnosis**:
```bash
# Monitor CPU usage
top -p $(pgrep python)

# Profile application
python -m cProfile -o profile.stats script.py
```

**Solutions**:
- Scale horizontally with more instances
- Optimize CPU-intensive operations
- Implement caching
- Use async/await for I/O operations

### Network Issues

**Symptoms**:
- Slow API responses
- Connection timeouts
- High latency

**Diagnosis**:
```bash
# Test network connectivity
ping api.prsm.org

# Check network latency
traceroute api.prsm.org

# Monitor network usage
iftop
```

**Solutions**:
- Check network configuration
- Use CDN for static content
- Optimize API payload sizes
- Implement connection pooling

## Security & Access

### Permission Denied Errors

**Symptoms**:
- "403 Forbidden" responses
- Access denied to resources
- Role-based access failures

**Diagnosis**:
```bash
# Check user permissions
curl http://localhost:8000/api/v1/users/me \
  -H "Authorization: Bearer TOKEN"

# Verify role assignments
python -c "
from prsm.auth.models import User
user = User.get_by_username('username')
print(user.permissions)
"
```

**Solutions**:
- Verify user roles and permissions
- Check access control configuration
- Update user permissions as needed
- Review security policies

### Security Audit Failures

**Symptoms**:
- Security warnings in logs
- Failed authentication attempts
- Suspicious activity alerts

**Diagnosis**:
```bash
# Check security logs
docker-compose logs prsm-api | grep SECURITY

# Monitor authentication attempts
grep "authentication" /var/log/prsm/security.log

# Check for malicious patterns
tail -f /var/log/prsm/access.log | grep -E "(SQL|XSS|injection)"
```

**Solutions**:
- Review and update security policies
- Implement additional monitoring
- Block suspicious IP addresses
- Update authentication mechanisms

## Deployment Issues

### Container Build Failures

**Symptoms**:
- Docker build errors
- Missing dependencies
- Build timeouts

**Diagnosis**:
```bash
# Build with verbose output
docker build --no-cache --progress=plain .

# Check build logs
docker-compose build --no-cache

# Test individual steps
docker run -it python:3.12 /bin/bash
```

**Solutions**:
- Update base images
- Fix dependency conflicts
- Increase build timeout
- Use multi-stage builds

### Kubernetes Deployment Issues

**Symptoms**:
- Pods stuck in pending state
- CrashLoopBackOff errors
- Service discovery failures

**Diagnosis**:
```bash
# Check pod status
kubectl describe pod POD_NAME -n prsm-production

# Check events
kubectl get events -n prsm-production --sort-by='.lastTimestamp'

# Check resource quotas
kubectl describe quota -n prsm-production
```

**Solutions**:
- Check resource requests and limits
- Verify node capacity
- Fix configuration issues
- Update deployment manifests

### Environment Configuration

**Symptoms**:
- Wrong environment variables
- Configuration mismatches
- Service discovery failures

**Diagnosis**:
```bash
# Check environment variables
docker-compose config

# Verify configuration
kubectl get configmap -n prsm-production -o yaml

# Test configuration
python -c "from prsm.core.config import settings; print(settings.database_url)"
```

**Solutions**:
- Update environment files
- Sync configuration across environments
- Validate configuration format
- Use configuration management tools

## Monitoring & Debugging

### Log Analysis

**Viewing Application Logs**:
```bash
# Docker Compose
docker-compose logs -f prsm-api

# Kubernetes
kubectl logs -f deployment/prsm-api -n prsm-production

# System logs
journalctl -u prsm-api -f
```

**Log Patterns to Look For**:
- ERROR: Application errors
- CRITICAL: System failures
- WARNING: Performance issues
- SECURITY: Security events

### Performance Monitoring

**Database Performance**:
```sql
-- Slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;

-- Connection usage
SELECT count(*), state FROM pg_stat_activity GROUP BY state;
```

**API Performance**:
```bash
# Response time testing
curl -w "@curl-format.txt" -o /dev/null -s API_ENDPOINT

# Load testing
ab -n 1000 -c 10 http://localhost:8000/health
```

### Debug Mode

**Enable Debug Logging**:
```bash
export PRSM_LOG_LEVEL=DEBUG
export PRSM_DEBUG=true
```

**Python Debugging**:
```python
import pdb; pdb.set_trace()  # Add breakpoint
python -m pdb script.py      # Debug script
```

## Getting Support

### Information to Collect

Before contacting support, gather:

1. **System Information**:
   ```bash
   # System details
   uname -a
   docker --version
   docker-compose --version
   
   # Service status
   docker-compose ps
   
   # Recent logs
   docker-compose logs --tail=100 > logs.txt
   ```

2. **Error Details**:
   - Exact error messages
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots if applicable

3. **Environment Details**:
   - Operating system
   - Hardware specifications
   - Network configuration
   - Deployment method (Docker/Kubernetes)

### Support Channels

1. **GitHub Issues**: Report bugs and feature requests
   - Repository: https://github.com/PRSM-AI/PRSM/issues
   - Include system info and logs

2. **Community Forum**: General questions and discussions
   - Discord: https://discord.gg/prsm
   - Reddit: r/PRSM

3. **Email Support**: Critical issues and enterprise support
   - Technical: support@prsm.org
   - Security: security@prsm.org
   - Enterprise: enterprise@prsm.org

4. **Documentation**: Comprehensive guides and references
   - Main docs: https://docs.prsm.org
   - API reference: https://api-docs.prsm.org
   - Tutorials: https://tutorials.prsm.org

### Emergency Contacts

For critical production issues:
- **Emergency Hotline**: +1-555-PRSM-911
- **Security Incidents**: security@prsm.org
- **Status Page**: https://status.prsm.org

---

## Document Information

**Version**: 1.0  
**Last Updated**: June 11, 2025  
**Next Review**: July 11, 2025  
**Owner**: PRSM Support Team  
**Contact**: support@prsm.org  

**Related Documentation**:
- [Production Operations Manual](PRODUCTION_OPERATIONS_MANUAL.md)
- [API Reference](API_REFERENCE.md)
- [Security Hardening Guide](SECURITY_HARDENING.md)
- [Development Setup Guide](../CONTRIBUTING.md)

---

*This troubleshooting guide is maintained by the PRSM community. If you find issues or have suggestions, please contribute via GitHub or contact our support team.*