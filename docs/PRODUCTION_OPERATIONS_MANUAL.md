# PRSM Production Operations Manual

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Deployment Procedures](#deployment-procedures)
4. [Monitoring & Alerting](#monitoring--alerting)
5. [Backup & Disaster Recovery](#backup--disaster-recovery)
6. [Performance Management](#performance-management)
7. [Security Operations](#security-operations)
8. [Incident Response](#incident-response)
9. [Maintenance Procedures](#maintenance-procedures)
10. [Capacity Planning](#capacity-planning)
11. [Troubleshooting](#troubleshooting)
12. [Emergency Procedures](#emergency-procedures)

## Overview

This manual provides comprehensive operational procedures for running PRSM (Protocol for Recursive Scientific Modeling) in production environments. It covers deployment, monitoring, maintenance, and incident response procedures for system administrators and DevOps teams.

### System Requirements

#### Minimum Production Requirements
- **CPU**: 8 cores (16 recommended)
- **Memory**: 32GB RAM (64GB recommended)
- **Storage**: 500GB SSD (1TB recommended)
- **Network**: 1Gbps connection
- **OS**: Ubuntu 20.04+ or CentOS 8+

#### Recommended Production Architecture
- **Load Balancer**: Nginx or HAProxy
- **Application Servers**: 3+ PRSM API instances
- **Database**: PostgreSQL 15+ with streaming replication
- **Cache**: Redis Cluster (3+ nodes)
- **Storage**: IPFS cluster with multiple gateways
- **Monitoring**: Prometheus + Grafana + Alertmanager
- **Container Orchestration**: Kubernetes 1.25+

## System Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Kubernetes    │    │   Monitoring    │
│   (Nginx/HAProxy)│    │    Cluster      │    │ (Prometheus)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PRSM API      │    │   PostgreSQL    │    │    Grafana      │
│  (3+ instances) │    │   (Primary +    │    │  (Dashboards)   │
│                 │    │   Replicas)     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Redis Cluster  │    │  IPFS Cluster   │    │  Alertmanager   │
│  (Cache/Queue)  │    │ (Distributed    │    │ (Notifications) │
│                 │    │  Storage)       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Service Dependencies

1. **PostgreSQL** - Primary database (critical)
2. **Redis** - Caching and queuing (critical)
3. **IPFS** - Distributed storage (critical)
4. **Prometheus** - Metrics collection (important)
5. **External APIs** - Model providers (degraded operation possible)

## Deployment Procedures

### Production Deployment Checklist

#### Pre-Deployment
- [ ] Backup current database
- [ ] Verify all environment variables
- [ ] Test deployment in staging environment
- [ ] Notify stakeholders of deployment window
- [ ] Ensure monitoring systems are operational
- [ ] Verify rollback procedures are ready

#### Deployment Steps

```bash
# 1. Deploy to staging and validate
./scripts/deploy-k8s.sh --environment staging --tag v1.x.x
./scripts/validate-deployment.sh --environment staging

# 2. Create production backup
./scripts/backup-system.sh backup --environment production --type full

# 3. Deploy to production
./scripts/deploy-k8s.sh --environment production --tag v1.x.x

# 4. Validate deployment
./scripts/validate-deployment.sh --environment production

# 5. Run smoke tests
./scripts/test-monitoring.sh integration
```

#### Post-Deployment
- [ ] Verify all services are healthy
- [ ] Check monitoring dashboards
- [ ] Validate critical user workflows
- [ ] Monitor error rates for 2 hours
- [ ] Send deployment notification
- [ ] Update deployment documentation

### Rollback Procedures

#### Automated Rollback
```bash
# Quick rollback to previous version
kubectl rollout undo deployment/prsm-api -n prsm-production
kubectl rollout undo deployment/prsm-worker -n prsm-production

# Verify rollback status
kubectl rollout status deployment/prsm-api -n prsm-production
```

#### Manual Rollback
```bash
# Restore from backup if needed
./scripts/backup-system.sh restore --environment production --backup-id BACKUP_ID

# Deploy previous known-good version
./scripts/deploy-k8s.sh --environment production --tag v1.x.x-previous
```

## Monitoring & Alerting

### Key Metrics to Monitor

#### Application Metrics
- **API Response Time**: < 2s (95th percentile)
- **Error Rate**: < 1% of total requests
- **Request Throughput**: Monitor trends
- **Active Sessions**: Track user activity
- **FTNS Transaction Rate**: Monitor token economy

#### Infrastructure Metrics
- **CPU Usage**: < 80% average
- **Memory Usage**: < 85% average
- **Disk Usage**: < 90% capacity
- **Network Latency**: < 100ms between services
- **Database Connections**: < 80% of max_connections

#### Business Metrics
- **Model Training Success Rate**: > 95%
- **Marketplace Transaction Volume**: Track trends
- **Governance Participation**: Monitor voter engagement
- **Security Incidents**: Zero tolerance for breaches

### Monitoring Dashboard URLs

- **System Overview**: `https://grafana.prsm.org/d/prsm-overview`
- **Security Dashboard**: `https://grafana.prsm.org/d/prsm-security`
- **Prometheus**: `https://prometheus.prsm.org:9090`
- **Alertmanager**: `https://alertmanager.prsm.org:9093`

### Alert Severity Levels

#### Critical (Immediate Response Required)
- Service completely down
- Database connection failures
- Security breaches detected
- Circuit breaker triggered
- Data corruption detected

#### Warning (Response within 1 hour)
- High error rates (> 5%)
- Performance degradation
- Resource usage approaching limits
- Failed health checks
- Authentication failures

#### Info (Awareness notifications)
- Deployment completed
- Scheduled maintenance
- Configuration changes
- Capacity planning alerts

## Backup & Disaster Recovery

### Backup Schedule

#### Daily Backups (Automated)
- **Database**: Full backup at 2 AM UTC
- **Redis**: Snapshot at 3 AM UTC
- **IPFS**: Incremental backup at 4 AM UTC
- **Configuration**: Daily configuration export

#### Weekly Backups
- **Full system backup**: Sundays at 1 AM UTC
- **Kubernetes manifests**: All deployments and configs
- **Monitoring data**: Prometheus long-term storage

#### Monthly Backups
- **Archive to cold storage**: AWS Glacier/Azure Archive
- **Test restore procedures**: Verify backup integrity
- **Update disaster recovery documentation**

### Backup Commands

```bash
# Create full system backup
./scripts/backup-system.sh backup \
  --environment production \
  --type full \
  --storage s3://prsm-backups/production

# Create database-only backup
./scripts/backup-system.sh backup \
  --environment production \
  --database-only

# List available backups
./scripts/backup-system.sh list

# Restore from backup
./scripts/backup-system.sh restore \
  --environment production \
  --backup-id backup_production_20250611_020000
```

### Disaster Recovery Procedures

#### Complete System Failure
1. **Assess Impact**: Determine extent of failure
2. **Activate DR Plan**: Notify team and stakeholders
3. **Switch to Backup Infrastructure**: If available
4. **Restore from Backups**: Latest known-good state
5. **Validate System**: Full system validation
6. **Resume Operations**: Gradual traffic restoration
7. **Post-Incident Review**: Document lessons learned

#### Recovery Time Objectives (RTO)
- **Critical Services**: 4 hours maximum
- **Full System**: 8 hours maximum
- **Data Loss (RPO)**: Maximum 1 hour

## Performance Management

### Performance Optimization

#### Database Optimization
```sql
-- Monitor slow queries
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;

-- Check connection usage
SELECT count(*), state FROM pg_stat_activity GROUP BY state;

-- Monitor cache hit ratio
SELECT 
  sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) * 100 
  AS cache_hit_ratio
FROM pg_statio_user_tables;
```

#### Redis Optimization
```bash
# Monitor Redis performance
redis-cli --latency-history -i 1

# Check memory usage
redis-cli info memory

# Monitor hit rate
redis-cli info stats | grep keyspace
```

#### API Performance Tuning
- **Worker Processes**: Scale based on CPU cores
- **Connection Pooling**: Optimize database connections
- **Caching Strategy**: Implement L1/L2/L3 caching
- **Load Balancing**: Distribute requests efficiently

### Scaling Procedures

#### Horizontal Scaling
```bash
# Scale API instances
kubectl scale deployment prsm-api --replicas=5 -n prsm-production

# Scale worker instances
kubectl scale deployment prsm-worker --replicas=3 -n prsm-production

# Verify scaling
kubectl get pods -n prsm-production
```

#### Vertical Scaling
```bash
# Update resource limits
kubectl patch deployment prsm-api -n prsm-production -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "prsm-api",
          "resources": {
            "requests": {"memory": "2Gi", "cpu": "1000m"},
            "limits": {"memory": "4Gi", "cpu": "2000m"}
          }
        }]
      }
    }
  }
}'
```

## Security Operations

### Security Monitoring

#### Daily Security Checks
- Review security logs for anomalies
- Check authentication failure rates
- Monitor rate limiting triggers
- Verify SSL certificate status
- Review access patterns for suspicious activity

#### Weekly Security Reviews
- Audit user access permissions
- Review API key usage patterns
- Check for software vulnerabilities
- Validate backup encryption
- Test security alert procedures

#### Monthly Security Tasks
- Security patch updates
- Access control audit
- Penetration testing review
- Security policy updates
- Incident response drill

### Security Incident Response

#### Immediate Actions (0-15 minutes)
1. **Isolate Affected Systems**: Prevent spread
2. **Assess Impact**: Determine scope of breach
3. **Activate Security Team**: Notify key personnel
4. **Preserve Evidence**: Maintain forensic integrity
5. **Implement Containment**: Stop ongoing attack

#### Short-term Actions (15 minutes - 4 hours)
1. **Detailed Investigation**: Analyze attack vectors
2. **System Restoration**: Restore from clean backups
3. **Security Hardening**: Patch vulnerabilities
4. **Communication**: Notify stakeholders as required
5. **Monitor for Reoccurrence**: Enhanced monitoring

#### Long-term Actions (4+ hours)
1. **Root Cause Analysis**: Complete investigation
2. **Process Improvements**: Update security procedures
3. **Documentation**: Update incident response plan
4. **Training**: Security awareness updates
5. **Compliance Reporting**: Regulatory notifications

## Maintenance Procedures

### Scheduled Maintenance Windows

#### Weekly Maintenance (Sundays 2-4 AM UTC)
- **System Updates**: Security patches and updates
- **Database Maintenance**: VACUUM, REINDEX, statistics update
- **Log Rotation**: Archive and compress old logs
- **Performance Review**: Analyze weekly performance metrics

#### Monthly Maintenance (First Sunday 1-6 AM UTC)
- **Major Updates**: Application and infrastructure updates
- **Capacity Review**: Resource usage analysis
- **Backup Testing**: Restore test validation
- **Security Audit**: Access control and vulnerability scan
- **Documentation Update**: Procedure and runbook updates

### Maintenance Procedures

#### Database Maintenance
```sql
-- Daily maintenance
VACUUM ANALYZE;

-- Weekly maintenance
REINDEX DATABASE prsm_production;
UPDATE pg_stat_statements_reset();

-- Check for bloated tables
SELECT 
  table_name,
  pg_size_pretty(pg_total_relation_size(table_name::regclass)) as size,
  pg_size_pretty(
    pg_total_relation_size(table_name::regclass) - 
    pg_relation_size(table_name::regclass)
  ) as index_size
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY pg_total_relation_size(table_name::regclass) DESC;
```

#### System Updates
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Docker containers
docker-compose pull
docker-compose up -d

# Update Kubernetes deployments
kubectl set image deployment/prsm-api prsm-api=prsm:latest -n prsm-production
```

## Capacity Planning

### Resource Monitoring

#### CPU and Memory Trends
- Monitor 95th percentile usage over 30 days
- Plan capacity increases when consistently > 70%
- Consider seasonal patterns in research workflows

#### Storage Growth
- Track database growth rate (typical: 10-20% monthly)
- Monitor IPFS storage consumption
- Plan for model training data requirements

#### Network Utilization
- Monitor inter-service communication patterns
- Plan for P2P federation bandwidth requirements
- Consider CDN usage for global distribution

### Scaling Thresholds

#### Scale Up Triggers
- **CPU**: > 80% for 4+ hours
- **Memory**: > 85% for 2+ hours
- **Response Time**: > 3s (95th percentile)
- **Error Rate**: > 2% for 30+ minutes
- **Queue Depth**: > 1000 pending jobs

#### Scale Down Considerations
- **Resource Usage**: < 50% for 24+ hours
- **Cost Optimization**: During low-usage periods
- **Maintenance Windows**: Planned capacity reduction

## Troubleshooting

### Common Issues and Solutions

#### API Performance Issues

**Symptoms**: Slow response times, timeouts
**Diagnosis**:
```bash
# Check API response times
curl -w "@curl-format.txt" -o /dev/null -s https://api.prsm.org/health

# Monitor database connections
kubectl exec -n prsm-production deployment/postgres -- \
  psql -U postgres -c "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"

# Check Redis performance
kubectl exec -n prsm-production deployment/redis -- redis-cli --latency
```

**Solutions**:
- Scale API instances horizontally
- Optimize database queries
- Increase connection pool size
- Enable/optimize caching

#### Database Connection Issues

**Symptoms**: Connection refused, too many connections
**Diagnosis**:
```bash
# Check connection count
kubectl exec -n prsm-production deployment/postgres -- \
  psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"

# Check max connections
kubectl exec -n prsm-production deployment/postgres -- \
  psql -U postgres -c "SHOW max_connections;"
```

**Solutions**:
- Increase max_connections in PostgreSQL
- Optimize connection pooling
- Identify and kill long-running queries
- Scale database resources

#### IPFS Synchronization Issues

**Symptoms**: Content not found, slow retrieval
**Diagnosis**:
```bash
# Check IPFS peers
kubectl exec -n prsm-production deployment/ipfs -- ipfs swarm peers | wc -l

# Check IPFS connectivity
kubectl exec -n prsm-production deployment/ipfs -- ipfs id

# Monitor IPFS performance
kubectl exec -n prsm-production deployment/ipfs -- \
  ipfs stats repo
```

**Solutions**:
- Restart IPFS node
- Clear IPFS cache
- Check network connectivity
- Add more IPFS gateway nodes

### Log Analysis Procedures

#### Application Logs
```bash
# View recent API logs
kubectl logs -n prsm-production deployment/prsm-api --tail=100

# Follow logs in real-time
kubectl logs -n prsm-production deployment/prsm-api -f

# Search for errors
kubectl logs -n prsm-production deployment/prsm-api | grep ERROR

# Export logs for analysis
kubectl logs -n prsm-production deployment/prsm-api --since=1h > api-logs.txt
```

#### System Logs
```bash
# Check system events
kubectl get events -n prsm-production --sort-by='.lastTimestamp'

# Monitor resource usage
kubectl top pods -n prsm-production
kubectl top nodes

# Check pod status
kubectl describe pod -n prsm-production <pod-name>
```

## Emergency Procedures

### Emergency Contact Information

#### Primary Contacts
- **System Administrator**: admin@prsm.org
- **DevOps Engineer**: devops@prsm.org
- **Security Team**: security@prsm.org
- **Database Administrator**: dba@prsm.org

#### Escalation Matrix
1. **Level 1**: System Administrator (0-15 minutes)
2. **Level 2**: DevOps Engineer (15-30 minutes)
3. **Level 3**: Security Team (30-60 minutes)
4. **Level 4**: Engineering Leadership (1+ hours)

### Emergency Response Procedures

#### System-Wide Outage
1. **Immediate Assessment** (0-5 minutes)
   - Check monitoring dashboards
   - Verify infrastructure status
   - Determine scope of outage

2. **Initial Response** (5-15 minutes)
   - Activate incident response team
   - Notify stakeholders
   - Begin diagnostic procedures

3. **Resolution Efforts** (15+ minutes)
   - Implement fix or rollback
   - Monitor system recovery
   - Validate all services

4. **Communication** (Throughout)
   - Update status page
   - Notify users of progress
   - Document actions taken

#### Security Incident
1. **Immediate Isolation** (0-5 minutes)
   - Isolate affected systems
   - Preserve evidence
   - Activate security team

2. **Containment** (5-30 minutes)
   - Stop ongoing attack
   - Assess damage scope
   - Implement additional controls

3. **Recovery** (30+ minutes)
   - Restore from clean backups
   - Patch vulnerabilities
   - Monitor for reoccurrence

### Emergency Commands

#### Immediate System Shutdown
```bash
# Emergency stop all services
kubectl scale deployment --all --replicas=0 -n prsm-production

# Stop specific service
kubectl scale deployment prsm-api --replicas=0 -n prsm-production
```

#### Emergency Rollback
```bash
# Quick rollback to previous version
kubectl rollout undo deployment/prsm-api -n prsm-production
kubectl rollout undo deployment/prsm-worker -n prsm-production
```

#### Emergency Database Access
```bash
# Direct database access
kubectl exec -it -n prsm-production deployment/postgres -- \
  psql -U postgres -d prsm_production

# Emergency database dump
kubectl exec -n prsm-production deployment/postgres -- \
  pg_dump -U postgres prsm_production > emergency_backup.sql
```

---

## Document Information

**Version**: 1.0  
**Last Updated**: June 11, 2025  
**Next Review**: July 11, 2025  
**Owner**: PRSM DevOps Team  
**Contact**: operations@prsm.org  

**Change Log**:
- v1.0 (2025-06-11): Initial production operations manual

---

*This document is part of the PRSM production documentation suite. For additional information, see:*
- [Administrator Guide](ADMINISTRATOR_GUIDE.md)
- [API Documentation](API_REFERENCE.md)
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)
- [Security Procedures](SECURITY_HARDENING.md)