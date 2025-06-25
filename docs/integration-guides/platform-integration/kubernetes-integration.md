# Kubernetes Integration Guide

Deploy and scale PRSM on Kubernetes for production-grade container orchestration and management.

## 🎯 Overview

This guide covers deploying PRSM on Kubernetes, including multi-environment setups, auto-scaling, monitoring, and best practices for production deployments.

## 📋 Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Helm 3.x installed
- Docker registry access
- Basic knowledge of Kubernetes concepts

## 🚀 Quick Start

### 1. Basic Deployment

```yaml
# k8s/basic/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prsm-api
  namespace: prsm
  labels:
    app: prsm-api
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: prsm-api
  template:
    metadata:
      labels:
        app: prsm-api
        version: v1
    spec:
      containers:
      - name: prsm-api
        image: prsm/api:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: prsm-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: prsm-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: prsm-api-service
  namespace: prsm
spec:
  selector:
    app: prsm-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP
```

### 2. Quick Deploy Script

```bash
#!/bin/bash
# scripts/k8s-quick-deploy.sh

set -e

# Create namespace
kubectl create namespace prsm --dry-run=client -o yaml | kubectl apply -f -

# Create secrets
kubectl create secret generic prsm-secrets \
  --from-literal=database-url="$DATABASE_URL" \
  --from-literal=redis-url="$REDIS_URL" \
  --from-literal=api-key="$PRSM_API_KEY" \
  --namespace=prsm \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy application
kubectl apply -f k8s/basic/ -n prsm

# Wait for deployment
kubectl rollout status deployment/prsm-api -n prsm

echo "PRSM deployed successfully!"
kubectl get pods -n prsm
```

## 🔧 Production Configuration

### Namespace and RBAC

```yaml
# k8s/production/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: prsm
  labels:
    name: prsm
    environment: production

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: prsm-service-account
  namespace: prsm

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: prsm
  name: prsm-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: prsm-role-binding
  namespace: prsm
subjects:
- kind: ServiceAccount
  name: prsm-service-account
  namespace: prsm
roleRef:
  kind: Role
  name: prsm-role
  apiGroup: rbac.authorization.k8s.io
```

### Configuration Management

```yaml
# k8s/production/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prsm-config
  namespace: prsm
data:
  # Application configuration
  LOG_LEVEL: "INFO"
  MAX_WORKERS: "4"
  REQUEST_TIMEOUT: "30"
  ENABLE_METRICS: "true"
  ENABLE_TRACING: "true"
  
  # Database configuration
  DB_POOL_SIZE: "20"
  DB_MAX_OVERFLOW: "30"
  DB_POOL_TIMEOUT: "30"
  
  # Redis configuration
  REDIS_POOL_SIZE: "10"
  REDIS_TIMEOUT: "5"
  
  # AI configuration
  AI_RESPONSE_CACHE_TTL: "3600"
  AI_MAX_TOKENS: "4096"
  AI_TEMPERATURE: "0.7"

---
apiVersion: v1
kind: Secret
metadata:
  name: prsm-secrets
  namespace: prsm
type: Opaque
stringData:
  database-url: "postgresql://user:pass@postgres:5432/prsm"
  redis-url: "redis://redis:6379/0"
  api-key: "your-secure-api-key"
  jwt-secret: "your-jwt-secret"
  encryption-key: "your-encryption-key"
```

### Production Deployment

```yaml
# k8s/production/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prsm-api
  namespace: prsm
  labels:
    app: prsm-api
    tier: backend
    environment: production
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: prsm-api
  template:
    metadata:
      labels:
        app: prsm-api
        tier: backend
        environment: production
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: prsm-service-account
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: prsm-api
        image: prsm/api:1.0.0
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        envFrom:
        - configMapRef:
            name: prsm-config
        - secretRef:
            name: prsm-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
            ephemeral-storage: "1Gi"
          limits:
            memory: "4Gi"
            cpu: "2000m"
            ephemeral-storage: "5Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
            scheme: HTTP
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
            scheme: HTTP
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: /startup
            port: 8000
            scheme: HTTP
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 30
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /app/cache
      volumes:
      - name: tmp
        emptyDir: {}
      - name: cache
        emptyDir:
          sizeLimit: 1Gi
      nodeSelector:
        kubernetes.io/os: linux
      tolerations:
      - key: "app"
        operator: "Equal"
        value: "prsm"
        effect: "NoSchedule"
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - prsm-api
              topologyKey: kubernetes.io/hostname
```

## 📊 Auto-scaling Configuration

### Horizontal Pod Autoscaler

```yaml
# k8s/autoscaling/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: prsm-api-hpa
  namespace: prsm
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: prsm-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: prsm_active_requests
      target:
        type: AverageValue
        averageValue: "10"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max

---
apiVersion: autoscaling/v2
kind: VerticalPodAutoscaler
metadata:
  name: prsm-api-vpa
  namespace: prsm
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: prsm-api
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: prsm-api
      minAllowed:
        cpu: 250m
        memory: 512Mi
      maxAllowed:
        cpu: 4000m
        memory: 8Gi
      controlledResources: ["cpu", "memory"]
      controlledValues: RequestsAndLimits
```

### Cluster Autoscaler

```yaml
# k8s/autoscaling/cluster-autoscaler.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
  labels:
    app: cluster-autoscaler
spec:
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      serviceAccountName: cluster-autoscaler
      containers:
      - image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.25.0
        name: cluster-autoscaler
        resources:
          limits:
            cpu: 100m
            memory: 300Mi
          requests:
            cpu: 100m
            memory: 300Mi
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/prsm-cluster
        - --balance-similar-node-groups
        - --skip-nodes-with-system-pods=false
        env:
        - name: AWS_REGION
          value: us-west-2
        volumeMounts:
        - name: ssl-certs
          mountPath: /etc/ssl/certs/ca-certificates.crt
          readOnly: true
      volumes:
      - name: ssl-certs
        hostPath:
          path: "/etc/ssl/certs/ca-bundle.crt"
```

## 🌐 Service Mesh Integration

### Istio Configuration

```yaml
# k8s/istio/gateway.yaml
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: prsm-gateway
  namespace: prsm
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - api.prsm.ai
    tls:
      httpsRedirect: true
  - port:
      number: 443
      name: https
      protocol: HTTPS
    tls:
      mode: SIMPLE
      credentialName: prsm-tls-secret
    hosts:
    - api.prsm.ai

---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: prsm-vs
  namespace: prsm
spec:
  hosts:
  - api.prsm.ai
  gateways:
  - prsm-gateway
  http:
  - match:
    - uri:
        prefix: /api/v1
    route:
    - destination:
        host: prsm-api-service
        port:
          number: 80
        subset: stable
      weight: 90
    - destination:
        host: prsm-api-service
        port:
          number: 80
        subset: canary
      weight: 10
    fault:
      delay:
        percentage:
          value: 0.1
        fixedDelay: 5s
    retries:
      attempts: 3
      perTryTimeout: 30s
    timeout: 60s

---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: prsm-dr
  namespace: prsm
spec:
  host: prsm-api-service
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        maxRequestsPerConnection: 10
    loadBalancer:
      simple: LEAST_CONN
    outlierDetection:
      consecutiveErrors: 3
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
  subsets:
  - name: stable
    labels:
      version: v1
  - name: canary
    labels:
      version: v2
```

### Service Monitor for Prometheus

```yaml
# k8s/monitoring/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: prsm-api-monitor
  namespace: prsm
  labels:
    app: prsm-api
    prometheus: kube-prometheus
spec:
  selector:
    matchLabels:
      app: prsm-api
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
    scrapeTimeout: 10s
    metricRelabelings:
    - sourceLabels: [__name__]
      regex: 'prsm_.*'
      action: keep

---
apiVersion: v1
kind: Service
metadata:
  name: prsm-api-metrics
  namespace: prsm
  labels:
    app: prsm-api
    prometheus: kube-prometheus
spec:
  selector:
    app: prsm-api
  ports:
  - name: http
    port: 8000
    targetPort: 8000
    protocol: TCP
```

## 🗄️ Database Integration

### PostgreSQL Deployment

```yaml
# k8s/database/postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: prsm
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:14
        env:
        - name: POSTGRES_DB
          value: prsm
        - name: POSTGRES_USER
          value: prsm
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
        ports:
        - containerPort: 5432
          name: postgres
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - prsm
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - prsm
          initialDelaySeconds: 5
          periodSeconds: 5
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi

---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: prsm
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  clusterIP: None

---
apiVersion: v1
kind: Secret
metadata:
  name: postgres-secret
  namespace: prsm
type: Opaque
stringData:
  password: "secure-postgres-password"
```

### Redis Deployment

```yaml
# k8s/database/redis.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: prsm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
          name: redis
        command:
        - redis-server
        - --appendonly
        - "yes"
        - --maxmemory
        - "1gb"
        - --maxmemory-policy
        - "allkeys-lru"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "500m"
        livenessProbe:
          tcpSocket:
            port: 6379
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: prsm
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: prsm
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 50Gi
```

## 📊 Monitoring and Observability

### Prometheus Configuration

```yaml
# k8s/monitoring/prometheus.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
      - "/etc/prometheus/rules/*.yml"
    
    scrape_configs:
      - job_name: 'kubernetes-apiservers'
        kubernetes_sd_configs:
        - role: endpoints
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
        - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
          action: keep
          regex: default;kubernetes;https
      
      - job_name: 'prsm-api'
        kubernetes_sd_configs:
        - role: endpoints
          namespaces:
            names:
            - prsm
        relabel_configs:
        - source_labels: [__meta_kubernetes_service_label_app]
          action: keep
          regex: prsm-api
        - source_labels: [__meta_kubernetes_endpoint_port_name]
          action: keep
          regex: http

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-rules
  namespace: monitoring
data:
  prsm-rules.yml: |
    groups:
    - name: prsm-alerts
      rules:
      - alert: PRSMHighErrorRate
        expr: rate(prsm_requests_total{status=~"5.."}[5m]) / rate(prsm_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "PRSM API high error rate"
          description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"
      
      - alert: PRSMHighLatency
        expr: histogram_quantile(0.95, rate(prsm_request_duration_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "PRSM API high latency"
          description: "95th percentile latency is {{ $value }}s"
      
      - alert: PRSMPodCrashLooping
        expr: rate(kube_pod_container_status_restarts_total{namespace="prsm"}[5m]) > 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "PRSM pod crash looping"
          description: "Pod {{ $labels.pod }} is crash looping"
```

### Grafana Dashboard

```yaml
# k8s/monitoring/grafana-dashboard.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prsm-dashboard
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  prsm-dashboard.json: |
    {
      "dashboard": {
        "id": null,
        "title": "PRSM API Dashboard",
        "tags": ["prsm", "api"],
        "timezone": "browser",
        "panels": [
          {
            "id": 1,
            "title": "Request Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(prsm_requests_total[5m])",
                "legendFormat": "{{ method }} {{ status }}"
              }
            ],
            "yAxes": [
              {
                "label": "Requests/sec"
              }
            ]
          },
          {
            "id": 2,
            "title": "Response Time",
            "type": "graph",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, rate(prsm_request_duration_seconds_bucket[5m]))",
                "legendFormat": "95th percentile"
              },
              {
                "expr": "histogram_quantile(0.50, rate(prsm_request_duration_seconds_bucket[5m]))",
                "legendFormat": "50th percentile"
              }
            ]
          },
          {
            "id": 3,
            "title": "Error Rate",
            "type": "singlestat",
            "targets": [
              {
                "expr": "rate(prsm_requests_total{status=~\"5..\"}[5m]) / rate(prsm_requests_total[5m])",
                "legendFormat": "Error Rate"
              }
            ]
          }
        ],
        "time": {
          "from": "now-1h",
          "to": "now"
        },
        "refresh": "10s"
      }
    }
```

## 🔧 CI/CD Integration

### GitOps with ArgoCD

```yaml
# k8s/gitops/application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: prsm-api
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/prsm-k8s-manifests
    targetRevision: main
    path: overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: prsm
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
      - Validate=true
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
      - PruneLast=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m

---
apiVersion: argoproj.io/v1alpha1
kind: AppProject
metadata:
  name: prsm
  namespace: argocd
spec:
  description: PRSM Application Project
  sourceRepos:
  - 'https://github.com/your-org/prsm-k8s-manifests'
  destinations:
  - namespace: prsm
    server: https://kubernetes.default.svc
  - namespace: prsm-staging
    server: https://kubernetes.default.svc
  clusterResourceWhitelist:
  - group: ''
    kind: Namespace
  namespaceResourceWhitelist:
  - group: ''
    kind: Service
  - group: apps
    kind: Deployment
  - group: networking.k8s.io
    kind: Ingress
  roles:
  - name: prsm-admins
    description: Full access to PRSM project
    policies:
    - p, proj:prsm:prsm-admins, applications, *, prsm/*, allow
    groups:
    - prsm-team
```

### GitHub Actions Integration

```yaml
# .github/workflows/k8s-deploy.yml
name: Deploy to Kubernetes

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          ghcr.io/${{ github.repository }}/prsm-api:${{ github.sha }}
          ghcr.io/${{ github.repository }}/prsm-api:latest
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'latest'
    
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBECONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Deploy to staging
      if: github.event_name == 'pull_request'
      run: |
        export KUBECONFIG=kubeconfig
        kubectl set image deployment/prsm-api prsm-api=ghcr.io/${{ github.repository }}/prsm-api:${{ github.sha }} -n prsm-staging
        kubectl rollout status deployment/prsm-api -n prsm-staging
    
    - name: Deploy to production
      if: github.ref == 'refs/heads/main'
      run: |
        export KUBECONFIG=kubeconfig
        kubectl set image deployment/prsm-api prsm-api=ghcr.io/${{ github.repository }}/prsm-api:${{ github.sha }} -n prsm
        kubectl rollout status deployment/prsm-api -n prsm
    
    - name: Run health check
      run: |
        export KUBECONFIG=kubeconfig
        kubectl wait --for=condition=available --timeout=300s deployment/prsm-api -n ${{ github.ref == 'refs/heads/main' && 'prsm' || 'prsm-staging' }}
```

## 🔄 Backup and Disaster Recovery

### Velero Backup Configuration

```yaml
# k8s/backup/velero-backup.yaml
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: prsm-daily-backup
  namespace: velero
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  template:
    includedNamespaces:
    - prsm
    excludedResources:
    - events
    - events.events.k8s.io
    storageLocation: default
    volumeSnapshotLocations:
    - default
    ttl: 720h  # 30 days
    hooks:
      resources:
      - name: postgres-backup-hook
        includedNamespaces:
        - prsm
        labelSelector:
          matchLabels:
            app: postgres
        pre:
        - exec:
            container: postgres
            command:
            - /bin/bash
            - -c
            - 'pg_dump -U $POSTGRES_USER $POSTGRES_DB > /tmp/backup.sql'
        post:
        - exec:
            container: postgres
            command:
            - /bin/bash
            - -c
            - 'rm -f /tmp/backup.sql'

---
apiVersion: velero.io/v1
kind: BackupStorageLocation
metadata:
  name: default
  namespace: velero
spec:
  provider: aws
  objectStorage:
    bucket: prsm-k8s-backups
    prefix: velero
  config:
    region: us-west-2
    s3ForcePathStyle: "false"

---
apiVersion: velero.io/v1
kind: VolumeSnapshotLocation
metadata:
  name: default
  namespace: velero
spec:
  provider: aws
  config:
    region: us-west-2
```

## 🛡️ Security Configuration

### Pod Security Policy

```yaml
# k8s/security/pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: prsm-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
  readOnlyRootFilesystem: true

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: prsm-psp-role
  namespace: prsm
rules:
- apiGroups: ['policy']
  resources: ['podsecuritypolicies']
  verbs: ['use']
  resourceNames: ['prsm-psp']

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: prsm-psp-binding
  namespace: prsm
roleRef:
  kind: Role
  name: prsm-psp-role
  apiGroup: rbac.authorization.k8s.io
subjects:
- kind: ServiceAccount
  name: prsm-service-account
  namespace: prsm
```

### Network Policies

```yaml
# k8s/security/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: prsm-network-policy
  namespace: prsm
spec:
  podSelector:
    matchLabels:
      app: prsm-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: istio-system
    - podSelector:
        matchLabels:
          app: istio-proxy
    ports:
    - protocol: TCP
      port: 8000
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
  - to: []
    ports:
    - protocol: TCP
      port: 443

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-default
  namespace: prsm
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

## 📋 Best Practices

### Resource Management

```yaml
# k8s/best-practices/resource-quotas.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: prsm-quota
  namespace: prsm
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    persistentvolumeclaims: "10"
    services: "10"
    secrets: "20"
    configmaps: "20"

---
apiVersion: v1
kind: LimitRange
metadata:
  name: prsm-limits
  namespace: prsm
spec:
  limits:
  - default:
      cpu: 1000m
      memory: 2Gi
    defaultRequest:
      cpu: 100m
      memory: 256Mi
    type: Container
  - max:
      storage: 100Gi
    type: PersistentVolumeClaim
```

### Deployment Script

```bash
#!/bin/bash
# scripts/k8s-production-deploy.sh

set -e

NAMESPACE="prsm"
IMAGE_TAG="${1:-latest}"
ENVIRONMENT="${2:-production}"

echo "Deploying PRSM to Kubernetes..."
echo "Environment: $ENVIRONMENT"
echo "Image Tag: $IMAGE_TAG"

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply security policies
kubectl apply -f k8s/security/ -n $NAMESPACE

# Apply configurations
kubectl apply -f k8s/production/configmap.yaml -n $NAMESPACE

# Create secrets (if not exist)
if ! kubectl get secret prsm-secrets -n $NAMESPACE >/dev/null 2>&1; then
    echo "Creating secrets..."
    kubectl create secret generic prsm-secrets \
        --from-literal=database-url="$DATABASE_URL" \
        --from-literal=redis-url="$REDIS_URL" \
        --from-literal=api-key="$PRSM_API_KEY" \
        --namespace=$NAMESPACE
fi

# Apply database deployments
kubectl apply -f k8s/database/ -n $NAMESPACE

# Wait for database to be ready
kubectl wait --for=condition=available --timeout=300s deployment/postgres -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/redis -n $NAMESPACE

# Update image tag in deployment
sed "s|image: prsm/api:latest|image: prsm/api:$IMAGE_TAG|g" k8s/production/deployment.yaml | kubectl apply -f - -n $NAMESPACE

# Apply autoscaling
kubectl apply -f k8s/autoscaling/ -n $NAMESPACE

# Apply monitoring
kubectl apply -f k8s/monitoring/ -n $NAMESPACE

# Apply Istio configuration (if Istio is installed)
if kubectl get crd gateways.networking.istio.io >/dev/null 2>&1; then
    kubectl apply -f k8s/istio/ -n $NAMESPACE
fi

# Wait for deployment
kubectl rollout status deployment/prsm-api -n $NAMESPACE --timeout=600s

# Verify deployment
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE

echo "PRSM deployed successfully!"

# Run health check
EXTERNAL_IP=$(kubectl get service istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ ! -z "$EXTERNAL_IP" ]; then
    echo "Health check: http://$EXTERNAL_IP/health"
    curl -f "http://$EXTERNAL_IP/health" || echo "Health check failed"
fi
```

---

**Need help with Kubernetes integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).