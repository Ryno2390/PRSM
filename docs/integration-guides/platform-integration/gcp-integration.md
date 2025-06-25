# Google Cloud Platform Integration Guide

Deploy and integrate PRSM with Google Cloud Platform services for enterprise-scale AI applications.

## 🎯 Overview

This guide covers deploying PRSM on Google Cloud Platform (GCP), integrating with GCP services, and leveraging cloud-native features for scalability and reliability.

## 📋 Prerequisites

- Google Cloud Platform account with billing enabled
- `gcloud` CLI installed and configured
- Docker installed locally
- PRSM source code or Docker image
- Basic knowledge of GCP services

## 🚀 Quick Setup

### 1. GCP Project Setup

```bash
# Create new project
gcloud projects create prsm-production --name="PRSM Production"

# Set current project
gcloud config set project prsm-production

# Enable required APIs
gcloud services enable \
  container.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  sql.googleapis.com \
  redis.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com \
  secretmanager.googleapis.com \
  cloudresourcemanager.googleapis.com
```

### 2. Basic Cloud Run Deployment

```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/prsm-production/prsm-api

gcloud run deploy prsm-api \
  --image gcr.io/prsm-production/prsm-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="ENVIRONMENT=production"
```

## ☁️ Google Cloud Run Deployment

### Dockerfile for Cloud Run

```dockerfile
# Dockerfile.cloudrun
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Cloud Run expects the service to listen on port 8080
ENV PORT=8080
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 prsm.api.main:app
```

### Cloud Run Service Configuration

```yaml
# cloudrun-service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: prsm-api
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/cpu-throttling: "false"
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "100"
        run.googleapis.com/execution-environment: gen2
        run.googleapis.com/memory: "4Gi"
        run.googleapis.com/cpu: "2"
    spec:
      containerConcurrency: 10
      timeoutSeconds: 300
      containers:
      - image: gcr.io/prsm-production/prsm-api:latest
        ports:
        - containerPort: 8080
        env:
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
        - name: PRSM_API_KEY
          valueFrom:
            secretKeyRef:
              name: prsm-secrets
              key: api-key
        resources:
          limits:
            cpu: "2000m"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Deployment Script

```bash
#!/bin/bash
# deploy-cloudrun.sh

set -e

PROJECT_ID="prsm-production"
SERVICE_NAME="prsm-api"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "Building Docker image..."
docker build -f Dockerfile.cloudrun -t ${IMAGE_NAME}:latest .

echo "Pushing image to Container Registry..."
docker push ${IMAGE_NAME}:latest

echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME}:latest \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --concurrency 10 \
  --max-instances 100 \
  --min-instances 1 \
  --timeout 300 \
  --set-env-vars="ENVIRONMENT=production,GCP_PROJECT=${PROJECT_ID}" \
  --set-secrets="DATABASE_URL=prsm-secrets:database-url,REDIS_URL=prsm-secrets:redis-url,PRSM_API_KEY=prsm-secrets:api-key"

echo "Getting service URL..."
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format "value(status.url)")
echo "Service deployed at: ${SERVICE_URL}"
```

## 🗃️ Google Cloud SQL Integration

### PostgreSQL Setup

```bash
# Create Cloud SQL PostgreSQL instance
gcloud sql instances create prsm-postgres \
  --database-version=POSTGRES_14 \
  --tier=db-g1-small \
  --region=us-central1 \
  --storage-type=SSD \
  --storage-size=100GB \
  --storage-auto-increase \
  --backup-start-time=03:00 \
  --maintenance-window-day=SUN \
  --maintenance-window-hour=04 \
  --deletion-protection

# Create database
gcloud sql databases create prsm_production --instance=prsm-postgres

# Create user
gcloud sql users create prsm_user \
  --instance=prsm-postgres \
  --password=secure_password_here
```

### Database Connection Configuration

```python
# prsm/core/database_gcp.py
import os
import sqlalchemy
from google.cloud.sql.connector import Connector
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class GCPDatabaseManager:
    def __init__(self):
        self.instance_connection_name = os.environ.get('INSTANCE_CONNECTION_NAME')
        self.db_user = os.environ.get('DB_USER', 'prsm_user')
        self.db_pass = os.environ.get('DB_PASS')
        self.db_name = os.environ.get('DB_NAME', 'prsm_production')
        self.connector = None
        self.engine = None

    def init_connection_pool(self):
        """Initialize connection pool using Cloud SQL Connector."""
        def getconn():
            if not self.connector:
                self.connector = Connector()
            
            conn = self.connector.connect(
                self.instance_connection_name,
                "pg8000",
                user=self.db_user,
                password=self.db_pass,
                db=self.db_name,
            )
            return conn

        self.engine = create_engine(
            "postgresql+pg8000://",
            creator=getconn,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        
        return self.engine

    def get_session(self):
        """Get database session."""
        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        return SessionLocal()

    def close_connections(self):
        """Close all connections."""
        if self.connector:
            self.connector.close()

# Usage
db_manager = GCPDatabaseManager()
engine = db_manager.init_connection_pool()
```

### Migration with Cloud SQL

```python
# migrations/gcp_migrate.py
import os
from alembic import command
from alembic.config import Config
from prsm.core.database_gcp import GCPDatabaseManager

def run_migrations():
    """Run database migrations on Cloud SQL."""
    # Initialize database connection
    db_manager = GCPDatabaseManager()
    engine = db_manager.init_connection_pool()
    
    # Configure Alembic
    alembic_cfg = Config("alembic.ini")
    
    # Override database URL for Cloud SQL
    database_url = f"postgresql+pg8000://{db_manager.db_user}:{db_manager.db_pass}@/{db_manager.db_name}?unix_sock=/cloudsql/{db_manager.instance_connection_name}/.s.PGSQL.5432"
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    
    # Run migrations
    command.upgrade(alembic_cfg, "head")
    
    print("Migrations completed successfully")

if __name__ == "__main__":
    run_migrations()
```

## 🔄 Google Cloud Memorystore (Redis)

### Redis Instance Setup

```bash
# Create Redis instance
gcloud redis instances create prsm-redis \
  --size=1 \
  --region=us-central1 \
  --redis-version=redis_6_x \
  --tier=basic

# Get Redis connection info
gcloud redis instances describe prsm-redis --region=us-central1
```

### Redis Client Configuration

```python
# prsm/core/redis_gcp.py
import os
import redis
from google.cloud import redis as cloud_redis

class GCPRedisClient:
    def __init__(self):
        self.redis_host = os.environ.get('REDIS_HOST')
        self.redis_port = int(os.environ.get('REDIS_PORT', 6379))
        self.redis_password = os.environ.get('REDIS_PASSWORD')
        self.client = None

    def connect(self):
        """Connect to Cloud Memorystore Redis."""
        self.client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # Test connection
        self.client.ping()
        return self.client

    def get_cache_key(self, prefix, *args):
        """Generate cache key with namespace."""
        return f"prsm:{prefix}:{':'.join(map(str, args))}"

    def cache_query_result(self, query_hash, result, ttl=3600):
        """Cache PRSM query result."""
        key = self.get_cache_key("query", query_hash)
        self.client.setex(key, ttl, json.dumps(result))

    def get_cached_query(self, query_hash):
        """Get cached query result."""
        key = self.get_cache_key("query", query_hash)
        cached = self.client.get(key)
        return json.loads(cached) if cached else None

# Initialize Redis client
redis_client = GCPRedisClient()
redis_client.connect()
```

## 🔒 Google Secret Manager

### Secrets Management

```bash
# Create secrets
gcloud secrets create prsm-api-key --data-file=api_key.txt
gcloud secrets create database-url --data-file=db_url.txt
gcloud secrets create redis-url --data-file=redis_url.txt

# Grant Cloud Run access to secrets
gcloud projects add-iam-policy-binding prsm-production \
  --member="serviceAccount:prsm-service@prsm-production.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Secret Manager Client

```python
# prsm/core/secrets_gcp.py
from google.cloud import secretmanager
import os

class GCPSecretManager:
    def __init__(self, project_id=None):
        self.project_id = project_id or os.environ.get('GCP_PROJECT')
        self.client = secretmanager.SecretManagerServiceClient()

    def get_secret(self, secret_id, version_id="latest"):
        """Retrieve secret from Secret Manager."""
        name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version_id}"
        
        try:
            response = self.client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            raise Exception(f"Failed to retrieve secret {secret_id}: {str(e)}")

    def create_secret(self, secret_id, secret_value):
        """Create a new secret."""
        parent = f"projects/{self.project_id}"
        
        # Create secret
        secret = {"replication": {"automatic": {}}}
        response = self.client.create_secret(
            request={
                "parent": parent,
                "secret_id": secret_id,
                "secret": secret
            }
        )
        
        # Add secret version
        self.client.add_secret_version(
            request={
                "parent": response.name,
                "payload": {"data": secret_value.encode("UTF-8")}
            }
        )

# Initialize secret manager
secret_manager = GCPSecretManager()

# Usage in application
def get_config_from_secrets():
    return {
        'database_url': secret_manager.get_secret('database-url'),
        'redis_url': secret_manager.get_secret('redis-url'),
        'api_key': secret_manager.get_secret('prsm-api-key')
    }
```

## 📊 Google Cloud Monitoring

### Monitoring Configuration

```python
# prsm/monitoring/gcp_monitoring.py
from google.cloud import monitoring_v3
from google.cloud.monitoring_dashboard import v1
import time
import json

class GCPMonitoring:
    def __init__(self, project_id=None):
        self.project_id = project_id or os.environ.get('GCP_PROJECT')
        self.client = monitoring_v3.MetricServiceClient()
        self.dashboard_client = v1.DashboardsServiceClient()

    def create_custom_metric(self, metric_type, metric_kind="GAUGE", value_type="DOUBLE"):
        """Create custom metric descriptor."""
        descriptor = monitoring_v3.MetricDescriptor()
        descriptor.type = f"custom.googleapis.com/{metric_type}"
        descriptor.metric_kind = getattr(monitoring_v3.MetricDescriptor.MetricKind, metric_kind)
        descriptor.value_type = getattr(monitoring_v3.MetricDescriptor.ValueType, value_type)
        descriptor.description = f"PRSM {metric_type} metric"

        project_name = f"projects/{self.project_id}"
        self.client.create_metric_descriptor(
            name=project_name, 
            metric_descriptor=descriptor
        )

    def send_metric(self, metric_type, value, labels=None):
        """Send metric data point."""
        project_name = f"projects/{self.project_id}"
        
        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/{metric_type}"
        
        if labels:
            series.metric.labels.update(labels)

        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 10 ** 9)
        interval = monitoring_v3.TimeInterval(
            {"end_time": {"seconds": seconds, "nanos": nanos}}
        )
        point = monitoring_v3.Point(
            {"interval": interval, "value": {"double_value": value}}
        )
        series.points = [point]

        self.client.create_time_series(
            name=project_name, 
            time_series=[series]
        )

    def create_prsm_dashboard(self):
        """Create monitoring dashboard for PRSM."""
        dashboard_json = {
            "displayName": "PRSM Production Dashboard",
            "mosaicLayout": {
                "tiles": [
                    {
                        "width": 6,
                        "height": 4,
                        "widget": {
                            "title": "Query Response Time",
                            "xyChart": {
                                "dataSets": [
                                    {
                                        "timeSeriesQuery": {
                                            "timeSeriesFilter": {
                                                "filter": 'metric.type="custom.googleapis.com/prsm/query_duration"',
                                                "aggregation": {
                                                    "alignmentPeriod": "60s",
                                                    "perSeriesAligner": "ALIGN_MEAN"
                                                }
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    },
                    {
                        "width": 6,
                        "height": 4,
                        "xPos": 6,
                        "widget": {
                            "title": "Query Success Rate",
                            "xyChart": {
                                "dataSets": [
                                    {
                                        "timeSeriesQuery": {
                                            "timeSeriesFilter": {
                                                "filter": 'metric.type="custom.googleapis.com/prsm/query_success_rate"',
                                                "aggregation": {
                                                    "alignmentPeriod": "300s",
                                                    "perSeriesAligner": "ALIGN_MEAN"
                                                }
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                ]
            }
        }

        project_name = f"projects/{self.project_id}"
        dashboard = monitoring_dashboard_v1.Dashboard()
        dashboard.display_name = dashboard_json["displayName"]
        dashboard.mosaic_layout = dashboard_json["mosaicLayout"]

        self.dashboard_client.create_dashboard(
            parent=project_name,
            dashboard=dashboard
        )

# Initialize monitoring
monitoring = GCPMonitoring()

# Usage in PRSM application
def track_query_metrics(start_time, success=True, user_tier="standard"):
    duration = time.time() - start_time
    
    # Send response time metric
    monitoring.send_metric(
        "prsm/query_duration", 
        duration,
        {"user_tier": user_tier, "success": str(success)}
    )
    
    # Send success rate metric
    monitoring.send_metric(
        "prsm/query_success_rate",
        1.0 if success else 0.0,
        {"user_tier": user_tier}
    )
```

### Alerting Configuration

```python
# prsm/monitoring/alerting.py
from google.cloud import monitoring_v3

def create_alert_policies(project_id):
    """Create alert policies for PRSM."""
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"

    # High error rate alert
    error_rate_policy = monitoring_v3.AlertPolicy(
        display_name="PRSM High Error Rate",
        conditions=[
            monitoring_v3.AlertPolicy.Condition(
                display_name="Error rate > 5%",
                condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                    filter='metric.type="custom.googleapis.com/prsm/query_success_rate"',
                    comparison=monitoring_v3.ComparisonType.COMPARISON_LESS_THAN,
                    threshold_value=0.95,
                    duration={"seconds": 300},
                    aggregations=[
                        monitoring_v3.Aggregation(
                            alignment_period={"seconds": 60},
                            per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN,
                        )
                    ],
                ),
            )
        ],
        notification_channels=[],  # Add notification channel IDs
        alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
            auto_close={"seconds": 1800}  # 30 minutes
        ),
    )

    # High latency alert
    latency_policy = monitoring_v3.AlertPolicy(
        display_name="PRSM High Latency",
        conditions=[
            monitoring_v3.AlertPolicy.Condition(
                display_name="Response time > 10s",
                condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                    filter='metric.type="custom.googleapis.com/prsm/query_duration"',
                    comparison=monitoring_v3.ComparisonType.COMPARISON_GREATER_THAN,
                    threshold_value=10.0,
                    duration={"seconds": 180},
                    aggregations=[
                        monitoring_v3.Aggregation(
                            alignment_period={"seconds": 60},
                            per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_PERCENTILE_95,
                        )
                    ],
                ),
            )
        ],
    )

    # Create alert policies
    client.create_alert_policy(name=project_name, alert_policy=error_rate_policy)
    client.create_alert_policy(name=project_name, alert_policy=latency_policy)
```

## 🗄️ Google Cloud Storage

### Storage Integration

```python
# prsm/storage/gcs.py
from google.cloud import storage
import os
import json
from datetime import datetime, timedelta

class GCSManager:
    def __init__(self, bucket_name=None):
        self.bucket_name = bucket_name or os.environ.get('GCS_BUCKET')
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)

    def upload_conversation_log(self, user_id, conversation_data):
        """Upload conversation log to GCS."""
        timestamp = datetime.utcnow().strftime("%Y/%m/%d")
        blob_name = f"conversations/{timestamp}/{user_id}/{datetime.utcnow().isoformat()}.json"
        
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(
            json.dumps(conversation_data, indent=2),
            content_type='application/json'
        )
        
        return blob.public_url

    def upload_model_artifacts(self, model_id, artifacts_path):
        """Upload model artifacts to GCS."""
        blob_name = f"models/{model_id}/artifacts.tar.gz"
        blob = self.bucket.blob(blob_name)
        
        blob.upload_from_filename(artifacts_path)
        
        # Set lifecycle policy
        blob.metadata = {
            'model_id': model_id,
            'upload_time': datetime.utcnow().isoformat()
        }
        blob.patch()
        
        return blob.public_url

    def get_signed_url(self, blob_name, expiration_hours=1):
        """Generate signed URL for secure access."""
        blob = self.bucket.blob(blob_name)
        
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.utcnow() + timedelta(hours=expiration_hours),
            method="GET"
        )
        
        return url

    def create_bucket_with_lifecycle(self):
        """Create bucket with lifecycle management."""
        bucket = self.client.create_bucket(self.bucket_name)
        
        # Set lifecycle rules
        lifecycle_rule = {
            "action": {"type": "Delete"},
            "condition": {
                "age": 365,  # Delete after 1 year
                "matchesPrefix": ["conversations/"]
            }
        }
        
        bucket.lifecycle_rules = [lifecycle_rule]
        bucket.patch()

# Initialize GCS manager
gcs_manager = GCSManager()
```

## 🚀 Google Kubernetes Engine (GKE)

### GKE Cluster Setup

```bash
# Create GKE cluster
gcloud container clusters create prsm-cluster \
  --zone=us-central1-a \
  --machine-type=e2-standard-4 \
  --num-nodes=3 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10 \
  --enable-autorepair \
  --enable-autoupgrade \
  --enable-ip-alias \
  --enable-network-policy

# Get cluster credentials
gcloud container clusters get-credentials prsm-cluster --zone=us-central1-a
```

### Kubernetes Deployment

```yaml
# k8s/gke-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prsm-api
  labels:
    app: prsm-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: prsm-api
  template:
    metadata:
      labels:
        app: prsm-api
    spec:
      serviceAccountName: prsm-workload-identity
      containers:
      - name: prsm-api
        image: gcr.io/prsm-production/prsm-api:latest
        ports:
        - containerPort: 8000
        env:
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
        - name: GCP_PROJECT
          value: "prsm-production"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
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
spec:
  selector:
    app: prsm-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer

---
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: prsm-ssl-cert
spec:
  domains:
    - api.prsm.ai

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: prsm-ingress
  annotations:
    kubernetes.io/ingress.global-static-ip-name: prsm-ip
    networking.gke.io/managed-certificates: prsm-ssl-cert
    kubernetes.io/ingress.class: "gce"
spec:
  rules:
  - host: api.prsm.ai
    http:
      paths:
      - path: /*
        pathType: ImplementationSpecific
        backend:
          service:
            name: prsm-api-service
            port:
              number: 80
```

### Workload Identity Setup

```bash
# Create service account
gcloud iam service-accounts create prsm-workload-identity

# Bind to Kubernetes service account
gcloud iam service-accounts add-iam-policy-binding \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:prsm-production.svc.id.goog[default/prsm-workload-identity]" \
  prsm-workload-identity@prsm-production.iam.gserviceaccount.com

# Grant necessary permissions
gcloud projects add-iam-policy-binding prsm-production \
  --member="serviceAccount:prsm-workload-identity@prsm-production.iam.gserviceaccount.com" \
  --role="roles/cloudsql.client"

gcloud projects add-iam-policy-binding prsm-production \
  --member="serviceAccount:prsm-workload-identity@prsm-production.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## 📋 CI/CD with Cloud Build

### Cloud Build Configuration

```yaml
# cloudbuild.yaml
steps:
  # Build Docker image
  - name: 'gcr.io/cloud-builders/docker'
    args: [
      'build',
      '-f', 'Dockerfile.cloudrun',
      '-t', 'gcr.io/$PROJECT_ID/prsm-api:$COMMIT_SHA',
      '-t', 'gcr.io/$PROJECT_ID/prsm-api:latest',
      '.'
    ]

  # Push to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/prsm-api:$COMMIT_SHA']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/prsm-api:latest']

  # Run tests
  - name: 'gcr.io/$PROJECT_ID/prsm-api:$COMMIT_SHA'
    entrypoint: 'python'
    args: ['-m', 'pytest', 'tests/', '-v']
    env:
      - 'ENVIRONMENT=test'

  # Deploy to staging
  - name: 'gcr.io/cloud-builders/gcloud'
    args: [
      'run', 'deploy', 'prsm-api-staging',
      '--image', 'gcr.io/$PROJECT_ID/prsm-api:$COMMIT_SHA',
      '--region', 'us-central1',
      '--platform', 'managed',
      '--tag', 'staging'
    ]

  # Run integration tests
  - name: 'gcr.io/cloud-builders/curl'
    args: ['--fail', '${_STAGING_URL}/health']

  # Deploy to production (on main branch)
  - name: 'gcr.io/cloud-builders/gcloud'
    args: [
      'run', 'deploy', 'prsm-api',
      '--image', 'gcr.io/$PROJECT_ID/prsm-api:$COMMIT_SHA',
      '--region', 'us-central1',
      '--platform', 'managed'
    ]

substitutions:
  _STAGING_URL: 'https://prsm-api-staging-xyz.a.run.app'

options:
  machineType: 'E2_HIGHCPU_8'
  timeout: '1200s'
```

### Trigger Setup

```bash
# Create build trigger
gcloud builds triggers create github \
  --repo-name=prsm \
  --repo-owner=your-org \
  --branch-pattern="^main$" \
  --build-config=cloudbuild.yaml \
  --include-logs-with-status
```

## 📊 Logging and Observability

### Structured Logging

```python
# prsm/logging/gcp_logging.py
import logging
import json
from google.cloud import logging as cloud_logging
from google.cloud.logging.handlers import CloudLoggingHandler

class GCPLogger:
    def __init__(self, service_name="prsm-api"):
        self.service_name = service_name
        
        # Initialize Cloud Logging
        client = cloud_logging.Client()
        client.setup_logging()
        
        # Create logger
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(logging.INFO)
        
        # Add Cloud Logging handler
        handler = CloudLoggingHandler(client)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_query(self, user_id, prompt, response_time, success=True, error=None):
        """Log PRSM query with structured data."""
        log_data = {
            "service": self.service_name,
            "event_type": "prsm_query",
            "user_id": user_id,
            "prompt_length": len(prompt),
            "response_time": response_time,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if error:
            log_data["error"] = str(error)
            self.logger.error(f"Query failed: {json.dumps(log_data)}")
        else:
            self.logger.info(f"Query completed: {json.dumps(log_data)}")

    def log_system_event(self, event_type, data):
        """Log system events."""
        log_data = {
            "service": self.service_name,
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.info(f"System event: {json.dumps(log_data)}")

# Initialize logger
gcp_logger = GCPLogger()
```

## 💰 Cost Optimization

### Resource Management

```python
# prsm/optimization/gcp_costs.py
from google.cloud import monitoring_v3
from google.cloud.billing import budgets_v1
import os

class GCPCostOptimizer:
    def __init__(self, project_id=None):
        self.project_id = project_id or os.environ.get('GCP_PROJECT')
        self.monitoring_client = monitoring_v3.MetricServiceClient()

    def create_budget_alert(self, budget_amount=1000):
        """Create budget alert for cost control."""
        client = budgets_v1.BudgetServiceClient()
        
        budget = budgets_v1.Budget(
            display_name="PRSM Monthly Budget",
            budget_filter=budgets_v1.Filter(
                projects=[f"projects/{self.project_id}"],
                services=["services/6F81-5844-456A"]  # Compute Engine
            ),
            amount=budgets_v1.BudgetAmount(
                specified_amount={"currency_code": "USD", "units": budget_amount}
            ),
            threshold_rules=[
                budgets_v1.ThresholdRule(
                    threshold_percent=0.8,
                    spend_basis=budgets_v1.ThresholdRule.Basis.CURRENT_SPEND
                ),
                budgets_v1.ThresholdRule(
                    threshold_percent=1.0,
                    spend_basis=budgets_v1.ThresholdRule.Basis.CURRENT_SPEND
                )
            ]
        )

        parent = f"billingAccounts/{os.environ.get('BILLING_ACCOUNT_ID')}"
        return client.create_budget(parent=parent, budget=budget)

    def optimize_cloud_run_config(self):
        """Recommend Cloud Run optimizations."""
        recommendations = []
        
        # Check current metrics
        # This would analyze actual usage patterns
        
        recommendations.append({
            "service": "prsm-api",
            "current_memory": "4Gi",
            "recommended_memory": "2Gi",
            "potential_savings": "50%"
        })
        
        return recommendations

# Usage
cost_optimizer = GCPCostOptimizer()
cost_optimizer.create_budget_alert(1000)  # $1000 monthly budget
```

## 🔧 Best Practices

### Security Hardening

```python
# prsm/security/gcp_security.py
from google.cloud import security_center
import os

class GCPSecurity:
    def __init__(self):
        self.org_id = os.environ.get('GCP_ORG_ID')
        self.project_id = os.environ.get('GCP_PROJECT')

    def configure_vpc_security(self):
        """Configure VPC security settings."""
        # This would configure firewall rules, private IPs, etc.
        security_config = {
            "firewall_rules": [
                {
                    "name": "prsm-api-allow",
                    "direction": "INGRESS",
                    "allowed": [{"IPProtocol": "tcp", "ports": ["8080"]}],
                    "source_ranges": ["130.211.0.0/22", "35.191.0.0/16"]  # Load balancer IPs
                }
            ],
            "private_ip": True,
            "authorized_networks": []
        }
        return security_config

    def setup_iam_policies(self):
        """Setup least-privilege IAM policies."""
        policies = {
            "cloud_run_service": [
                "roles/cloudsql.client",
                "roles/secretmanager.secretAccessor"
            ],
            "cicd_service": [
                "roles/run.admin",
                "roles/storage.admin"
            ]
        }
        return policies

# Security configuration
security = GCPSecurity()
```

### Terraform Configuration

```hcl
# terraform/main.tf
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Cloud SQL
resource "google_sql_database_instance" "prsm_postgres" {
  name             = "prsm-postgres"
  database_version = "POSTGRES_14"
  region           = var.region
  deletion_protection = true

  settings {
    tier = "db-g1-small"
    
    backup_configuration {
      enabled = true
      start_time = "03:00"
    }
    
    maintenance_window {
      day  = 7
      hour = 4
    }
  }
}

# Cloud Run
resource "google_cloud_run_service" "prsm_api" {
  name     = "prsm-api"
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/prsm-api:latest"
        
        resources {
          limits = {
            cpu    = "2000m"
            memory = "4Gi"
          }
        }
        
        env {
          name = "DATABASE_URL"
          value_from {
            secret_key_ref {
              name = google_secret_manager_secret.database_url.secret_id
              key  = "latest"
            }
          }
        }
      }
    }
  }
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}
```

---

**Need help with GCP integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).