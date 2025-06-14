# PRSM Enterprise Multi-Region Deployment Guide

## Overview

This guide covers the deployment and management of PRSM at enterprise scale across multiple geographic regions, ensuring 99.9% uptime SLA and global performance optimization.

## Architecture Overview

### Global Infrastructure Topology

```
    🌍 Global Load Balancer (CloudFlare/AWS Global Accelerator)
                    │
        ┌───────────┼───────────┐
        │           │           │
    🇺🇸 US-WEST    🇺🇸 US-EAST   🇪🇺 EU-WEST
   (Primary)      (Secondary)   (Secondary)
        │           │           │
    ┌───┼───┐   ┌───┼───┐   ┌───┼───┐
    │   │   │   │   │   │   │   │   │
   API DB IPFS API DB IPFS API DB IPFS
        │           │           │
        └───────────┼───────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
   🇯🇵 AP-NE-1   🇸🇬 AP-SE-1   🇧🇷 SA-EAST
   (Secondary)   (Secondary)   (Edge)
```

### Regional Distribution Strategy

#### Tier 1 Regions (Full Stack)
- **US-West-2** (Primary): Oregon, USA
- **US-East-1** (DR Primary): Virginia, USA
- **EU-West-1** (EMEA Primary): Ireland

#### Tier 2 Regions (API + Cache)
- **AP-Northeast-1**: Tokyo, Japan
- **AP-Southeast-1**: Singapore

#### Edge Locations (CDN + Cache)
- **SA-East-1**: São Paulo, Brazil
- **AF-South-1**: Cape Town, South Africa
- **AP-South-1**: Mumbai, India

## Deployment Architecture

### Infrastructure Components

#### Per Region (Tier 1)
```yaml
Kubernetes Cluster:
  - 3-100 nodes (auto-scaling)
  - Multi-AZ deployment
  - Dedicated node pools for different workloads

API Layer:
  - 5-50 replicas (HPA)
  - 500ms p95 latency target
  - 10,000 RPS capacity per region

Worker Layer:
  - 3-30 replicas (HPA)
  - GPU-enabled nodes for model execution
  - Queue-based task distribution

Database Layer:
  - PostgreSQL cluster (3 replicas)
  - Cross-region read replicas
  - Automated failover

Cache Layer:
  - Redis cluster (6 nodes)
  - Multi-AZ replication
  - Cross-region backup

Storage Layer:
  - IPFS cluster (3+ nodes)
  - Content replication factor: 3
  - Geographic content distribution

Monitoring:
  - Prometheus + Grafana
  - Distributed tracing (Jaeger)
  - Log aggregation (ELK/Loki)
```

#### Per Region (Tier 2)
```yaml
Kubernetes Cluster:
  - 2-20 nodes (auto-scaling)
  - Single-AZ deployment acceptable

API Layer:
  - 3-15 replicas (HPA)
  - 1000ms p95 latency target
  - 5,000 RPS capacity per region

Cache Layer:
  - Redis cluster (3 nodes)
  - Local caching only
  - Backup to Tier 1 region

Edge Services:
  - CDN integration
  - Static content caching
  - API response caching
```

## Deployment Process

### 1. Infrastructure Provisioning

#### Prerequisites
```bash
# Install required tools
brew install terraform kubectl helm aws-cli google-cloud-sdk azure-cli

# Authenticate with cloud providers
aws configure
gcloud auth login
az login

# Clone PRSM repository
git clone https://github.com/PRSM-Network/PRSM.git
cd PRSM
```

#### Deploy Infrastructure
```bash
# Navigate to enterprise deployment
cd deploy/enterprise

# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var="environment=production" \
  -var="regions=us-west-2,us-east-1,eu-west-1,ap-northeast-1,ap-southeast-1"

# Apply infrastructure
terraform apply -auto-approve

# Save outputs for Kubernetes configuration
terraform output -json > terraform-outputs.json
```

### 2. Multi-Region Kubernetes Deployment

#### Deploy to All Regions
```bash
# Run enterprise deployment script
./scripts/deploy-enterprise.sh \
  --environment production \
  --regions us-west-2,us-east-1,eu-west-1,ap-northeast-1,ap-southeast-1 \
  --providers aws,gcp,azure \
  --strategy blue-green
```

#### Manual Region Deployment
```bash
# Deploy to specific region
./scripts/deploy-enterprise.sh \
  --environment production \
  --regions us-west-2 \
  --providers aws

# Verify deployment
kubectl --context prsm-us-west-2 get pods -n prsm-system
kubectl --context prsm-us-west-2 get services -n prsm-system
```

### 3. Service Mesh Configuration

#### Deploy Istio Service Mesh
```bash
# Install Istio
istioctl install --set values.global.meshID=prsm-mesh \
  --set values.global.multiCluster.clusterName=prsm-us-west-2 \
  --set values.global.network=prsm-network

# Apply PRSM service mesh configuration
kubectl apply -f deploy/enterprise/istio/
```

#### Cross-Region Service Discovery
```bash
# Create cross-region service entries
kubectl apply -f - <<EOF
apiVersion: networking.istio.io/v1beta1
kind: ServiceEntry
metadata:
  name: prsm-api-us-east-1
  namespace: prsm-system
spec:
  hosts:
  - prsm-api.us-east-1.local
  location: MESH_EXTERNAL
  ports:
  - number: 443
    name: https
    protocol: HTTPS
  resolution: DNS
  addresses:
  - 10.1.0.0/16  # CIDR of us-east-1 region
EOF
```

### 4. Global Load Balancing

#### AWS Global Accelerator Setup
```bash
# Create Global Accelerator
aws globalaccelerator create-accelerator \
  --name prsm-global-accelerator \
  --ip-address-type IPV4 \
  --enabled

# Add listeners and endpoint groups for each region
aws globalaccelerator create-listener \
  --accelerator-arn arn:aws:globalaccelerator::123456789012:accelerator/abcd1234 \
  --protocol TCP \
  --port-ranges FromPort=443,ToPort=443
```

#### CloudFlare Load Balancing
```bash
# Configure CloudFlare load balancer
curl -X POST "https://api.cloudflare.com/client/v4/zones/{zone_id}/load_balancers" \
  -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
  -H "Content-Type: application/json" \
  --data '{
    "name": "api.prsm.network",
    "fallback_pool": "us-west-2-pool",
    "default_pools": ["us-west-2-pool", "us-east-1-pool", "eu-west-1-pool"],
    "description": "PRSM API Global Load Balancer",
    "enabled": true,
    "steering_policy": "geo"
  }'
```

## Monitoring and Observability

### 1. Global Monitoring Dashboard

#### Deploy Monitoring Stack
```bash
# Deploy Prometheus federation
helm install prometheus-federation prometheus-community/kube-prometheus-stack \
  -f deploy/enterprise/monitoring/prometheus-values.yaml \
  -n monitoring

# Deploy Grafana with global dashboards
kubectl apply -f deploy/enterprise/monitoring/grafana-dashboards.yaml
```

#### Key Metrics to Monitor

##### Regional Health Metrics
```promql
# API availability per region
up{job="prsm-api",region=~".*"} * 100

# Request latency p95 per region
histogram_quantile(0.95, 
  rate(http_request_duration_seconds_bucket{job="prsm-api"}[5m])
) by (region)

# Error rate per region
rate(http_requests_total{job="prsm-api",status=~"5.."}[5m]) / 
rate(http_requests_total{job="prsm-api"}[5m]) * 100 by (region)

# Cross-region request distribution
sum(rate(http_requests_total{job="prsm-api"}[5m])) by (region)
```

##### Federation Network Metrics
```promql
# Connected federation nodes per region
prsm_federation_nodes_connected by (region)

# Cross-region network latency
prsm_federation_network_latency_seconds by (source_region, target_region)

# IPFS content replication status
prsm_ipfs_content_replication_factor by (region, content_hash)

# Database replication lag
pg_stat_replication_replay_lag by (region, replica)
```

### 2. Alerting Configuration

#### Critical Alerts
```yaml
groups:
- name: prsm-regional-health
  rules:
  - alert: PRSMRegionDown
    expr: up{job="prsm-api",region=~".*"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "PRSM region {{ $labels.region }} is down"
      description: "All PRSM API instances in {{ $labels.region }} are unreachable"

  - alert: PRSMCrossRegionLatencyHigh
    expr: prsm_federation_network_latency_seconds > 0.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High cross-region latency detected"
      description: "Latency between {{ $labels.source_region }} and {{ $labels.target_region }} is {{ $value }}s"

  - alert: PRSMGlobalErrorRateHigh
    expr: sum(rate(http_requests_total{job="prsm-api",status=~"5.."}[5m])) / sum(rate(http_requests_total{job="prsm-api"}[5m])) > 0.05
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Global PRSM error rate is high"
      description: "Global error rate is {{ $value | humanizePercentage }}"
```

### 3. Distributed Tracing

#### Jaeger Configuration
```bash
# Deploy Jaeger operator
kubectl apply -f https://github.com/jaegertracing/jaeger-operator/releases/download/v1.35.0/jaeger-operator.yaml

# Create Jaeger instance for PRSM
kubectl apply -f - <<EOF
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: prsm-jaeger
  namespace: prsm-system
spec:
  strategy: production
  storage:
    type: elasticsearch
    options:
      es:
        server-urls: http://elasticsearch:9200
  query:
    replicas: 2
  collector:
    replicas: 3
    resources:
      requests:
        cpu: 100m
        memory: 128Mi
      limits:
        cpu: 500m
        memory: 512Mi
EOF
```

## Disaster Recovery and Failover

### 1. Automated Failover Configuration

#### Regional Failover Strategy
```yaml
# Primary region failure detection
- alert: PRSMPrimaryRegionDown
  expr: up{job="prsm-api",region="us-west-2"} == 0
  for: 2m
  labels:
    severity: critical
    action: failover
  annotations:
    summary: "Primary region us-west-2 is down"
    description: "Initiating failover to us-east-1"
    runbook_url: "https://docs.prsm.network/runbooks/regional-failover"
```

#### Failover Process
```bash
#!/bin/bash
# Regional failover script

PRIMARY_REGION="us-west-2"
FAILOVER_REGION="us-east-1"

# 1. Detect primary region failure
if ! kubectl --context "prsm-${PRIMARY_REGION}" get pods -n prsm-system &>/dev/null; then
  echo "Primary region ${PRIMARY_REGION} is unreachable"
  
  # 2. Promote secondary region
  kubectl --context "prsm-${FAILOVER_REGION}" scale deployment prsm-api -n prsm-system --replicas=20
  kubectl --context "prsm-${FAILOVER_REGION}" scale deployment prsm-worker -n prsm-system --replicas=10
  
  # 3. Update global load balancer
  aws globalaccelerator update-endpoint-group \
    --endpoint-group-arn "${FAILOVER_ENDPOINT_GROUP_ARN}" \
    --traffic-dial-percentage 100
  
  # 4. Notify operations team
  curl -X POST "$SLACK_WEBHOOK_URL" \
    -H 'Content-type: application/json' \
    --data "{\"text\":\"🚨 PRSM failover completed: ${PRIMARY_REGION} → ${FAILOVER_REGION}\"}"
fi
```

### 2. Database Replication and Backup

#### Cross-Region Database Setup
```sql
-- Configure PostgreSQL streaming replication
-- On primary (us-west-2)
CREATE USER replicator REPLICATION LOGIN CONNECTION LIMIT 3 ENCRYPTED PASSWORD 'replication_password';

-- Configure pg_hba.conf
-- host replication replicator 10.0.0.0/8 md5

-- On replica (us-east-1)
-- Recovery configuration in postgresql.conf
standby_mode = 'on'
primary_conninfo = 'host=prsm-postgres.us-west-2.internal port=5432 user=replicator'
restore_command = 'aws s3 cp s3://prsm-wal-archive/%f %p'
```

#### Automated Backup Strategy
```bash
#!/bin/bash
# Database backup script

REGIONS=("us-west-2" "us-east-1" "eu-west-1")
BACKUP_BUCKET="prsm-database-backups"

for region in "${REGIONS[@]}"; do
  echo "Creating backup for region: $region"
  
  # Create logical backup
  kubectl --context "prsm-${region}" exec -n prsm-system deployment/postgres -- \
    pg_dumpall -U postgres | \
    gzip | \
    aws s3 cp - "s3://${BACKUP_BUCKET}/${region}/$(date +%Y%m%d_%H%M%S)_full_backup.sql.gz"
  
  # Create point-in-time recovery backup
  kubectl --context "prsm-${region}" exec -n prsm-system deployment/postgres -- \
    pg_basebackup -U postgres -D - -Ft -z | \
    aws s3 cp - "s3://${BACKUP_BUCKET}/${region}/$(date +%Y%m%d_%H%M%S)_basebackup.tar.gz"
done
```

## Performance Optimization

### 1. Regional Content Distribution

#### IPFS Content Pinning Strategy
```javascript
// Intelligent content pinning based on usage patterns
const pinningStrategy = {
  // Pin all content in primary regions
  tierOne: ['us-west-2', 'us-east-1', 'eu-west-1'],
  
  // Pin popular content in secondary regions
  tierTwo: ['ap-northeast-1', 'ap-southeast-1'],
  
  // Cache frequently accessed content at edge
  edge: ['sa-east-1', 'af-south-1', 'ap-south-1']
};

async function optimizeContentDistribution() {
  const contentStats = await getContentAccessStats();
  
  for (const [contentHash, stats] of Object.entries(contentStats)) {
    const popularity = stats.accessCount / stats.ageInDays;
    
    if (popularity > 100) {
      // Pin to all regions
      await pinToRegions(contentHash, [...tierOne, ...tierTwo]);
    } else if (popularity > 10) {
      // Pin to tier one only
      await pinToRegions(contentHash, tierOne);
    }
    
    // Intelligent edge caching based on geographic access
    const topRegions = getTopAccessRegions(stats.accessByRegion);
    for (const region of topRegions) {
      const nearestEdge = findNearestEdgeLocation(region);
      await cacheAtEdge(contentHash, nearestEdge);
    }
  }
}
```

### 2. Auto-Scaling Optimization

#### Predictive Scaling Based on Global Patterns
```yaml
# Custom metrics for predictive scaling
apiVersion: v2
kind: HorizontalPodAutoscaler
metadata:
  name: prsm-api-predictive-hpa
  namespace: prsm-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: prsm-api
  minReplicas: 5
  maxReplicas: 50
  metrics:
  # Standard CPU/Memory metrics
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  
  # Custom business metrics
  - type: Pods
    pods:
      metric:
        name: prsm_requests_per_second_predicted
      target:
        type: AverageValue
        averageValue: "50"
  
  # Cross-region load balancing metric
  - type: External
    external:
      metric:
        name: prsm_global_queue_depth
      target:
        type: Value
        value: "100"
  
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
      - type: Pods
        value: 4
        periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
      selectPolicy: Min
```

### 3. Network Optimization

#### Inter-Region Network Mesh
```bash
# Setup VPC peering for optimal network paths
aws ec2 create-vpc-peering-connection \
  --vpc-id vpc-12345678 \
  --peer-vpc-id vpc-87654321 \
  --peer-region us-east-1

# Configure routing for direct inter-region communication
aws ec2 create-route \
  --route-table-id rtb-12345678 \
  --destination-cidr-block 10.1.0.0/16 \
  --vpc-peering-connection-id pcx-1234567890abcdef0
```

## Security Configuration

### 1. Cross-Region Security Policies

#### Network Security
```yaml
# Global network policy for cross-region communication
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: prsm-cross-region-policy
  namespace: prsm-system
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: prsm-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  # Allow from other PRSM regions
  - from:
    - namespaceSelector:
        matchLabels:
          name: prsm-system
    - podSelector:
        matchLabels:
          app.kubernetes.io/component: federation
    ports:
    - protocol: TCP
      port: 8000
  
  # Allow from load balancers
  - from:
    - ipBlock:
        cidr: 0.0.0.0/0
        except:
        - 169.254.169.254/32  # Block metadata service
    ports:
    - protocol: TCP
      port: 8000
  
  egress:
  # Allow to other PRSM regions
  - to:
    - namespaceSelector:
        matchLabels:
          name: prsm-system
  
  # Allow to external APIs
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: UDP
      port: 53
```

#### Identity and Access Management
```yaml
# Cross-region service account federation
apiVersion: v1
kind: ServiceAccount
metadata:
  name: prsm-cross-region-sa
  namespace: prsm-system
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/prsm-cross-region-role
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: prsm-cross-region-role
rules:
- apiGroups: [""]
  resources: ["services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["networking.istio.io"]
  resources: ["serviceentries", "virtualservices"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
```

## Cost Optimization

### 1. Regional Cost Analysis

#### Cost Monitoring Dashboard
```promql
# Cost per region per hour
sum(
  label_replace(
    kube_node_info{node=~".*"} * on(node) group_left(instance_type) 
    kube_node_labels{label_node_kubernetes_io_instance_type=~".*"},
    "instance_type", "$1", "label_node_kubernetes_io_instance_type", "(.*)"
  ) * on(instance_type) group_left(cost_per_hour)
  aws_instance_cost_per_hour
) by (region)

# Request cost efficiency (cost per 1M requests)
(
  sum(aws_instance_cost_per_hour) by (region) * 24 * 30  # Monthly cost
) / (
  sum(rate(http_requests_total{job="prsm-api"}[30d])) by (region) * 30 * 24 * 3600 / 1000000  # Monthly requests in millions
)
```

#### Spot Instance Integration
```yaml
# Spot instance node group for non-critical workloads
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: prsm-production-cluster
  region: us-west-2

nodeGroups:
- name: spot-workers
  instanceTypes: ["c5.large", "c5.xlarge", "c4.large", "c4.xlarge"]
  spot: true
  minSize: 1
  maxSize: 20
  desiredCapacity: 3
  volumeSize: 100
  volumeType: gp3
  
  labels:
    workload-type: "spot"
    node-lifecycle: "spot"
  
  taints:
  - key: "spot-instance"
    value: "true"
    effect: NoSchedule
  
  tags:
    "k8s.io/cluster-autoscaler/enabled": "true"
    "k8s.io/cluster-autoscaler/prsm-production-cluster": "owned"
```

### 2. Intelligent Workload Placement

#### Cost-Aware Scheduling
```yaml
# Priority class for cost optimization
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: prsm-cost-optimized
value: 100
globalDefault: false
description: "Priority class for cost-optimized PRSM workloads"
---
# Deployment with cost-aware scheduling
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prsm-worker-batch
  namespace: prsm-system
spec:
  replicas: 5
  template:
    spec:
      priorityClassName: prsm-cost-optimized
      tolerations:
      - key: "spot-instance"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      nodeSelector:
        workload-type: "spot"
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            preference:
              matchExpressions:
              - key: node-lifecycle
                operator: In
                values: ["spot"]
```

## Maintenance and Updates

### 1. Rolling Updates Across Regions

#### Blue-Green Deployment Strategy
```bash
#!/bin/bash
# Blue-green deployment across regions

REGIONS=("us-west-2" "us-east-1" "eu-west-1")
NEW_IMAGE_TAG="v1.2.0"

for region in "${REGIONS[@]}"; do
  echo "Deploying to region: $region"
  
  # Deploy green environment
  kubectl --context "prsm-${region}" set image deployment/prsm-api \
    prsm-api="prsm-api:${NEW_IMAGE_TAG}" -n prsm-system
  
  # Wait for rollout
  kubectl --context "prsm-${region}" rollout status deployment/prsm-api -n prsm-system
  
  # Run health checks
  if ! curl -f "https://api-${region}.prsm.network/health"; then
    echo "Health check failed for region: $region"
    kubectl --context "prsm-${region}" rollout undo deployment/prsm-api -n prsm-system
    exit 1
  fi
  
  echo "Deployment successful for region: $region"
done
```

### 2. Automated Maintenance Windows

#### Maintenance Scheduling
```yaml
# CronJob for automated maintenance
apiVersion: batch/v1
kind: CronJob
metadata:
  name: prsm-maintenance
  namespace: prsm-system
spec:
  schedule: "0 2 * * 0"  # Sunday 2 AM UTC
  timeZone: "UTC"
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: maintenance
            image: prsm-maintenance:latest
            command:
            - /bin/bash
            - -c
            - |
              # Regional maintenance rotation
              REGIONS=("us-west-2" "us-east-1" "eu-west-1")
              CURRENT_WEEK=$(date +%U)
              REGION_INDEX=$((CURRENT_WEEK % 3))
              MAINTENANCE_REGION=${REGIONS[$REGION_INDEX]}
              
              echo "Performing maintenance on region: $MAINTENANCE_REGION"
              
              # Scale down non-critical services
              kubectl --context "prsm-${MAINTENANCE_REGION}" scale deployment prsm-worker \
                -n prsm-system --replicas=1
              
              # Run maintenance tasks
              kubectl --context "prsm-${MAINTENANCE_REGION}" exec deployment/postgres \
                -n prsm-system -- pg_dump prsm > /tmp/backup.sql
              
              # Update system packages
              kubectl --context "prsm-${MAINTENANCE_REGION}" patch daemonset node-exporter \
                -n monitoring --patch '{"spec":{"updateStrategy":{"type":"RollingUpdate"}}}'
              
              # Scale back up
              kubectl --context "prsm-${MAINTENANCE_REGION}" scale deployment prsm-worker \
                -n prsm-system --replicas=5
          
          restartPolicy: OnFailure
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Cross-Region Connectivity Issues
```bash
# Test cross-region connectivity
kubectl --context prsm-us-west-2 run test-pod --image=nicolaka/netshoot -it --rm -- \
  curl -v https://api-us-east-1.prsm.network/health

# Check VPC peering status
aws ec2 describe-vpc-peering-connections \
  --filters "Name=status-code,Values=active" \
  --query 'VpcPeeringConnections[].{ID:VpcPeeringConnectionId,Status:Status.Code,Requester:RequesterVpcInfo.CidrBlock,Accepter:AccepterVpcInfo.CidrBlock}'

# Verify security group rules
aws ec2 describe-security-groups \
  --group-ids sg-12345678 \
  --query 'SecurityGroups[].IpPermissions[?IpProtocol==`tcp` && FromPort==`443`]'
```

#### 2. Database Replication Lag
```sql
-- Check replication status on primary
SELECT client_addr, state, sent_lsn, write_lsn, flush_lsn, replay_lsn, 
       write_lag, flush_lag, replay_lag
FROM pg_stat_replication;

-- Check replica lag
SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) AS lag_seconds;

-- Force checkpoint if lag is high
CHECKPOINT;
```

#### 3. Auto-scaling Issues
```bash
# Check HPA status
kubectl get hpa -n prsm-system -o wide

# Check metrics server
kubectl top nodes
kubectl top pods -n prsm-system

# Check custom metrics
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/prsm-system/pods/*/prsm_requests_per_second"

# Force scale if needed
kubectl scale deployment prsm-api -n prsm-system --replicas=10
```

#### 4. Service Mesh Issues
```bash
# Check Istio configuration
istioctl proxy-config cluster prsm-api-deployment-12345-abcde.prsm-system

# Verify mTLS configuration
istioctl authn tls-check prsm-api-service.prsm-system.svc.cluster.local

# Check traffic distribution
istioctl proxy-config endpoints prsm-api-deployment-12345-abcde.prsm-system

# Debug traffic routing
kubectl logs -f deployment/istio-proxy -c istio-proxy -n prsm-system
```

## Performance Benchmarks

### Expected Performance Metrics

#### Regional Performance Targets
```yaml
US-West-2 (Primary):
  - API Latency P95: < 200ms
  - Throughput: > 15,000 RPS
  - Availability: > 99.95%
  - Error Rate: < 0.1%

US-East-1 (Secondary):
  - API Latency P95: < 300ms
  - Throughput: > 10,000 RPS
  - Availability: > 99.9%
  - Error Rate: < 0.2%

EU-West-1 (Secondary):
  - API Latency P95: < 400ms
  - Throughput: > 8,000 RPS
  - Availability: > 99.9%
  - Error Rate: < 0.2%

Cross-Region:
  - Federation Sync: < 500ms
  - Content Replication: < 2s
  - Failover Time: < 30s
  - Recovery Time: < 5m
```

#### Load Testing
```bash
# Run distributed load test
k6 run --vus 1000 --duration 10m \
  --env API_BASE_URL=https://api.prsm.network \
  --env REGIONS="us-west-2,us-east-1,eu-west-1" \
  scripts/load-test-enterprise.js

# Regional performance test
./scripts/test_real_benchmarking.py --comprehensive --regions all
```

## Conclusion

This multi-region deployment provides PRSM with:

- **Global scale**: 5+ regions with automatic failover
- **High availability**: 99.9% uptime SLA
- **Performance**: Sub-500ms global latency
- **Cost efficiency**: Spot instances and intelligent scaling
- **Security**: End-to-end encryption and zero-trust architecture
- **Observability**: Comprehensive monitoring and alerting

The infrastructure is designed to handle enterprise-scale traffic while maintaining cost efficiency and providing a foundation for continued growth and expansion.