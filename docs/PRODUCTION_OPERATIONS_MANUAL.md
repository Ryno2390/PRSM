# PRSM Production Operations Manual

> **Scope note (2026-04-16):** This manual was authored June 2025 for a centralized, Kubernetes-orchestrated deployment model (load balancer + k8s API replicas + PostgreSQL primary/replica + Redis cluster + IPFS cluster + Prometheus). That remains a valid operating mode for **Prismatica's T4 meganode infrastructure** and for **Foundation-operated Phase 1 on-chain indexer / API gateway** components.
>
> **It is not the operating mode for the bulk of PRSM supply.** The PRSM protocol is peer-to-peer: T1 consumer-edge and T2 prosumer nodes (the numerical majority) run single-process `prsm node start` on residential / prosumer hardware, not k8s pods. T3 cloud-arbitrage operators run on RunPod / Lambda / CoreWeave, not hyperscaler-k8s. Per the four-tier supply architecture in `PRSM_Vision.md` §6, only T4 meganodes map cleanly to the k8s operating model documented below.
>
> **Reading guidance:**
> - If you operate a **T4 meganode** (Prismatica or strategic partner), this manual is directly applicable.
> - If you operate the **Foundation's on-chain API gateway / indexer / monitoring infrastructure**, this manual's k8s patterns are applicable.
> - If you operate a **T1/T2/T3 node**, use [`GETTING_STARTED.md`](GETTING_STARTED.md), [`quickstart.md`](quickstart.md), and [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for local-dev guidance. This manual is orientational only.
>
> **Related docs:**
> - [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — local development troubleshooting
> - [`TROUBLESHOOTING_GUIDE.md`](TROUBLESHOOTING_GUIDE.md) — production / containerized-deployment troubleshooting (companion to this manual)
> - [`SECURITY_HARDENING.md`](SECURITY_HARDENING.md) + [`SECURITY_HARDENING_CHECKLIST.md`](SECURITY_HARDENING_CHECKLIST.md) + [`SECURITY_CONFIGURATION_AUDIT.md`](SECURITY_CONFIGURATION_AUDIT.md) — security suite
> - [`BOOTSTRAP_DEPLOYMENT_GUIDE.md`](BOOTSTRAP_DEPLOYMENT_GUIDE.md) — bootstrap-node deploy
> - [`2026-04-10-audit-gap-roadmap.md`](2026-04-10-audit-gap-roadmap.md) — master roadmap; Phase 1 on-chain ops + Phase 6 P2P hardening change operational surface
> - [`2026-04-11-phase1.3-sepolia-bakein-log.md`](2026-04-11-phase1.3-sepolia-bakein-log.md) — live bake-in operations log as of April 2026
>
> **Terminology:** "PRSM" expands to "Protocol for Research, Storage, and Modeling." Earlier drafts of this doc used "Recursive Scientific Modeling" — that is legacy and should be disregarded. Metrics sections below mention "Model Training Success Rate," "Marketplace Transaction Volume," and "Governance Participation" — the first two were NWTN-era framings (NWTN orchestrator deleted in v1.6 scope alignment, April 2026); the third depends on governance structure not yet finalized. These metric names are retained in the manual as illustrative examples of the *kind* of operational metric worth tracking, but the specific metric names may not map to current code.

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

This manual provides comprehensive operational procedures for running PRSM (**Protocol for Research, Storage, and Modeling**) infrastructure in production environments — primarily T4 meganode operators and Foundation-operated on-chain indexer / gateway components (see scope note at top). It covers deployment, monitoring, maintenance, and incident response procedures for system administrators and DevOps teams.

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

### Collaboration Telemetry & Alert Handling Runbook (P3)

This runbook operationalizes collaboration telemetry exported by [`prsm/core/monitoring/metrics.py`](prsm/core/monitoring/metrics.py) and evaluated by collaboration alert rules in [`prsm/core/monitoring/alerts.py`](prsm/core/monitoring/alerts.py).

#### Key Metrics and Expected Healthy Ranges

| Metric | Healthy Range | Investigation Trigger |
|---|---|---|
| `collab_transport_handshake_failure_rate` | `< 0.10` sustained | `> 0.15` for 5m |
| `collab_transport_handshake_replay_nonce_delta` | `0–1` per 2m window | `>= 5` in 2m |
| `collab_transport_dispatch_failure_rate` | `< 0.05` sustained | `> 0.10` for 5m |
| `collab_manager_dispatch_failure_rate` | `< 0.05` sustained | `> 0.08` for 5m |
| `collab_protocol_stalled_total` | `0–1` | `> 2` for 5m |
| `collab_protocol_stalled_ratio` | `< 0.15` | `> 0.25` for 5m |

For dashboards and smoothing, use recording rules from [`config/prometheus/recording_rules.yml`](config/prometheus/recording_rules.yml):
- `prsm:collab_handshake_failure_rate_5m`
- `prsm:collab_replay_nonce_delta_max_5m`
- `prsm:collab_transport_dispatch_failure_rate_5m`
- `prsm:collab_manager_dispatch_failure_rate_5m`
- `prsm:collab_protocol_stalled_ratio_5m`

#### Alert Meanings

Configured by [`AlertManager.setup_collaboration_rules()`](prsm/core/monitoring/alerts.py:694):

- `collab_handshake_failure_rate_high` (warning): elevated handshake rejects; often trust/auth mismatch or malformed peer handshake behavior.
- `collab_replay_nonce_spike` (critical): likely replay attempt burst or nonce state desynchronization.
- `collab_dispatch_failure_rate_high` (warning): transport-level handler dispatch path instability.
- `collab_manager_dispatch_failure_rate_high` (warning): collaboration manager dispatch path degraded.
- `collab_stalled_protocols_detected` (warning): collaboration transitions not reaching terminal states.
- `collab_stalled_protocol_ratio_high` (warning): broad degradation where many protocols stall.

#### Triage Steps (15-minute first pass)

1. **Validate exporter visibility**
   - Confirm collaboration metrics are enabled in [`config/metrics-exporter.yml`](config/metrics-exporter.yml) (`collaboration: true`).
   - Confirm recording rules are loaded from [`config/prometheus/recording_rules.yml`](config/prometheus/recording_rules.yml).
2. **Classify alert type**
   - Trust-path: handshake/replay metrics.
   - Reliability-path: dispatch/stalled protocol metrics.
3. **Correlate reason taxonomies**
   - Handshake reason series: `collab_transport_handshake_failures_by_reason_total{reason=...}`
   - Dispatch reason series: `collab_transport_dispatch_failures_by_reason_total{reason=...}`
   - Gossip drop reasons: `collab_gossip_drop_by_reason_total{reason=...}`
4. **Confirm impact on protocol flow**
   - Check `collab_protocol_transition_total` vs `collab_protocol_terminal_outcome_total` divergence.
   - Verify `collab_protocol_stalled_total` and `collab_protocol_stalled_ratio` trend direction over 15m.
5. **Run focused canary suite**
   - Execute targeted collaboration/trust regression tests listed in the P3 canary procedure before and after mitigation.

#### Rollback / Safing Steps

1. **Fail closed on suspicious trust traffic**
   - Temporarily block or isolate peers with dominant replay/timestamp/auth failures while investigation proceeds.
2. **Safing on collaboration protocol instability**
   - Pause new collaboration session dispatch while allowing in-flight sessions to complete/cancel.
3. **Rollback path (code/config regression suspected)**
   - Roll back to last known-good deployment using standard rollback procedure in this document.
   - Restore previous alert/recording rule config if a recent tuning change increased false positives.
4. **Recovery gate**
   - Keep safing in place until canary suite passes and key collaboration metrics return to healthy ranges for at least one full 5-minute rule window.

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

**Version**: 1.1
**Originally authored**: June 11, 2025
**Last updated**: 2026-04-16 (scope banner, name-expansion fix, cross-references — procedural bodies unchanged)
**Review cadence**: quarterly; next review due 2026-07-16
**Owner**: Foundation DevOps Lead (role to be filled; see `PRSM_Vision.md` §12 Team) + Prismatica infrastructure team for T4 meganode operations
**Contact**: operations contact to be published on foundation website at launch

**Change Log**:
- v1.0 (2025-06-11): Initial production operations manual
- v1.1 (2026-04-16): Added scope banner clarifying T4-meganode vs T1/T2/T3 operating mode applicability; fixed protocol name expansion; added cross-references to companion suite and master roadmap; flagged NWTN-era metric names as illustrative rather than current

---

*This document is part of the PRSM production documentation suite. For additional information, see:*
- [Administrator Guide](ADMINISTRATOR_GUIDE.md) (verify existence — may be historical)
- [API Documentation](API_REFERENCE.md)
- [Troubleshooting Guide (Production)](TROUBLESHOOTING_GUIDE.md) — direct companion to this manual
- [Troubleshooting (Local Dev)](TROUBLESHOOTING.md) — for T1/T2/T3 node operators
- [Security Hardening](SECURITY_HARDENING.md) + [Checklist](SECURITY_HARDENING_CHECKLIST.md) + [Config Audit](SECURITY_CONFIGURATION_AUDIT.md) + [Pen-Test Guide](PENETRATION_TESTING_GUIDE.md) — security suite
- [Master Roadmap](2026-04-10-audit-gap-roadmap.md) — phase plan driving operational surface changes
- [Phase 1 Sepolia Bake-in Log](2026-04-11-phase1.3-sepolia-bakein-log.md) — current live-ops artifact
