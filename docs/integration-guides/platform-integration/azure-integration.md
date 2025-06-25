# Microsoft Azure Integration Guide

Deploy and integrate PRSM with Microsoft Azure services for enterprise-scale AI applications.

## 🎯 Overview

This guide covers deploying PRSM on Microsoft Azure, integrating with Azure services, and leveraging cloud-native features for scalability, security, and enterprise integration.

## 📋 Prerequisites

- Microsoft Azure subscription
- Azure CLI installed and configured
- Docker installed locally
- PRSM source code or Docker image
- Basic knowledge of Azure services

## 🚀 Quick Setup

### 1. Azure Resource Setup

```bash
# Login to Azure
az login

# Create resource group
az group create \
  --name prsm-production \
  --location eastus

# Create Azure Container Registry
az acr create \
  --resource-group prsm-production \
  --name prsmregistry \
  --sku Standard \
  --admin-enabled true

# Get ACR login server
az acr show \
  --name prsmregistry \
  --resource-group prsm-production \
  --query loginServer \
  --output tsv
```

### 2. Basic Container Instances Deployment

```bash
# Build and push image to ACR
az acr build \
  --registry prsmregistry \
  --image prsm-api:latest \
  .

# Create container instance
az container create \
  --resource-group prsm-production \
  --name prsm-api \
  --image prsmregistry.azurecr.io/prsm-api:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server prsmregistry.azurecr.io \
  --registry-username prsmregistry \
  --registry-password $(az acr credential show --name prsmregistry --query "passwords[0].value" -o tsv) \
  --dns-name-label prsm-api-demo \
  --ports 8000 \
  --environment-variables ENVIRONMENT=production
```

## ☁️ Azure Container Apps Deployment

### Container App Configuration

```yaml
# azure-container-app.yaml
properties:
  managedEnvironmentId: /subscriptions/{subscription}/resourceGroups/prsm-production/providers/Microsoft.App/managedEnvironments/prsm-env
  configuration:
    ingress:
      external: true
      targetPort: 8000
      traffic:
        - weight: 100
          latestRevision: true
    registries:
      - server: prsmregistry.azurecr.io
        username: prsmregistry
        passwordSecretRef: registry-password
    secrets:
      - name: registry-password
        value: # ACR password
      - name: database-connection
        keyVaultUrl: https://prsm-keyvault.vault.azure.net/secrets/database-url
      - name: redis-connection
        keyVaultUrl: https://prsm-keyvault.vault.azure.net/secrets/redis-url
  template:
    containers:
      - image: prsmregistry.azurecr.io/prsm-api:latest
        name: prsm-api
        env:
          - name: DATABASE_URL
            secretRef: database-connection
          - name: REDIS_URL
            secretRef: redis-connection
          - name: ENVIRONMENT
            value: production
        resources:
          cpu: 2.0
          memory: 4Gi
        probes:
          - type: Liveness
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          - type: Readiness
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
    scale:
      minReplicas: 1
      maxReplicas: 10
      rules:
        - name: http-scale-rule
          http:
            metadata:
              concurrentRequests: "10"
```

### Deployment Script

```bash
#!/bin/bash
# deploy-container-app.sh

set -e

RESOURCE_GROUP="prsm-production"
LOCATION="eastus"
ACR_NAME="prsmregistry"
APP_NAME="prsm-api"
ENV_NAME="prsm-env"

echo "Creating Container App Environment..."
az containerapp env create \
  --name $ENV_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

echo "Building and pushing image..."
az acr build \
  --registry $ACR_NAME \
  --image prsm-api:latest \
  .

echo "Creating Container App..."
az containerapp create \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $ENV_NAME \
  --image ${ACR_NAME}.azurecr.io/prsm-api:latest \
  --target-port 8000 \
  --ingress external \
  --cpu 2.0 \
  --memory 4Gi \
  --min-replicas 1 \
  --max-replicas 10 \
  --registry-server ${ACR_NAME}.azurecr.io \
  --registry-username $ACR_NAME \
  --registry-password $(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv) \
  --secrets database-url="$DATABASE_URL" redis-url="$REDIS_URL" \
  --env-vars ENVIRONMENT=production DATABASE_URL=secretref:database-url REDIS_URL=secretref:redis-url

echo "Getting app URL..."
APP_URL=$(az containerapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query "properties.configuration.ingress.fqdn" -o tsv)
echo "App deployed at: https://$APP_URL"
```

## 🗃️ Azure Database for PostgreSQL

### Database Setup

```bash
# Create PostgreSQL server
az postgres flexible-server create \
  --resource-group prsm-production \
  --name prsm-postgres \
  --location eastus \
  --admin-user prsmadmin \
  --admin-password SecurePassword123! \
  --sku-name Standard_D2s_v3 \
  --tier GeneralPurpose \
  --public-access 0.0.0.0 \
  --storage-size 128 \
  --version 14

# Create database
az postgres flexible-server db create \
  --resource-group prsm-production \
  --server-name prsm-postgres \
  --database-name prsm_production

# Configure firewall rule for Azure services
az postgres flexible-server firewall-rule create \
  --resource-group prsm-production \
  --name prsm-postgres \
  --rule-name AllowAzureServices \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0
```

### Database Connection Configuration

```python
# prsm/core/database_azure.py
import os
import asyncpg
import asyncio
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import sqlalchemy
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

class AzureDatabaseManager:
    def __init__(self):
        self.keyvault_url = os.environ.get('AZURE_KEYVAULT_URL')
        self.credential = DefaultAzureCredential()
        self.secret_client = SecretClient(
            vault_url=self.keyvault_url,
            credential=self.credential
        ) if self.keyvault_url else None
        self.engine = None

    async def get_database_url(self):
        """Get database URL from Key Vault or environment."""
        if self.secret_client:
            secret = self.secret_client.get_secret("database-url")
            return secret.value
        return os.environ.get('DATABASE_URL')

    async def init_connection_pool(self):
        """Initialize async connection pool."""
        database_url = await self.get_database_url()
        
        # Replace postgres:// with postgresql+asyncpg://
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql+asyncpg://', 1)
        
        self.engine = create_async_engine(
            database_url,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
        
        return self.engine

    async def get_session(self):
        """Get async database session."""
        if not self.engine:
            await self.init_connection_pool()
        
        AsyncSessionLocal = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        return AsyncSessionLocal()

    async def health_check(self):
        """Check database connectivity."""
        try:
            async with self.get_session() as session:
                result = await session.execute("SELECT 1")
                return {"status": "healthy", "database": "connected"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

# Initialize database manager
db_manager = AzureDatabaseManager()
```

### Database Migration with Azure

```python
# migrations/azure_migrate.py
import asyncio
import os
from alembic import command
from alembic.config import Config
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

async def run_azure_migrations():
    """Run database migrations on Azure PostgreSQL."""
    
    # Get database URL from Key Vault
    keyvault_url = os.environ.get('AZURE_KEYVAULT_URL')
    if keyvault_url:
        credential = DefaultAzureCredential()
        secret_client = SecretClient(vault_url=keyvault_url, credential=credential)
        database_url = secret_client.get_secret("database-url").value
    else:
        database_url = os.environ.get('DATABASE_URL')
    
    # Configure Alembic
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    
    # Run migrations
    command.upgrade(alembic_cfg, "head")
    print("Azure PostgreSQL migrations completed successfully")

if __name__ == "__main__":
    asyncio.run(run_azure_migrations())
```

## 🔄 Azure Cache for Redis

### Redis Setup

```bash
# Create Redis cache
az redis create \
  --resource-group prsm-production \
  --name prsm-redis \
  --location eastus \
  --sku Standard \
  --vm-size c1 \
  --enable-non-ssl-port true

# Get Redis connection string
az redis show-connection-string \
  --resource-group prsm-production \
  --name prsm-redis \
  --query primaryConnectionString \
  --output tsv
```

### Redis Client Configuration

```python
# prsm/core/redis_azure.py
import os
import redis
import json
import asyncio
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import aioredis

class AzureRedisClient:
    def __init__(self):
        self.keyvault_url = os.environ.get('AZURE_KEYVAULT_URL')
        self.credential = DefaultAzureCredential()
        self.secret_client = SecretClient(
            vault_url=self.keyvault_url,
            credential=self.credential
        ) if self.keyvault_url else None
        self.client = None
        self.async_client = None

    async def get_redis_url(self):
        """Get Redis URL from Key Vault or environment."""
        if self.secret_client:
            secret = self.secret_client.get_secret("redis-url")
            return secret.value
        return os.environ.get('REDIS_URL')

    async def connect(self):
        """Connect to Azure Cache for Redis."""
        redis_url = await self.get_redis_url()
        
        # Async Redis client
        self.async_client = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # Sync Redis client
        self.client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # Test connection
        await self.async_client.ping()
        return self.async_client

    async def cache_query_result(self, query_hash, result, ttl=3600):
        """Cache PRSM query result."""
        key = f"prsm:query:{query_hash}"
        await self.async_client.setex(key, ttl, json.dumps(result))

    async def get_cached_query(self, query_hash):
        """Get cached query result."""
        key = f"prsm:query:{query_hash}"
        cached = await self.async_client.get(key)
        return json.loads(cached) if cached else None

    async def store_session_data(self, session_id, data, ttl=86400):
        """Store session data."""
        key = f"prsm:session:{session_id}"
        await self.async_client.setex(key, ttl, json.dumps(data))

    async def get_session_data(self, session_id):
        """Get session data."""
        key = f"prsm:session:{session_id}"
        data = await self.async_client.get(key)
        return json.loads(data) if data else None

    async def health_check(self):
        """Check Redis connectivity."""
        try:
            await self.async_client.ping()
            return {"status": "healthy", "redis": "connected"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

# Initialize Redis client
redis_client = AzureRedisClient()
```

## 🔒 Azure Key Vault Integration

### Key Vault Setup

```bash
# Create Key Vault
az keyvault create \
  --resource-group prsm-production \
  --name prsm-keyvault \
  --location eastus \
  --enable-rbac-authorization true

# Create secrets
az keyvault secret set \
  --vault-name prsm-keyvault \
  --name database-url \
  --value "postgresql://prsmadmin:SecurePassword123!@prsm-postgres.postgres.database.azure.com/prsm_production"

az keyvault secret set \
  --vault-name prsm-keyvault \
  --name redis-url \
  --value "redis://prsm-redis.redis.cache.windows.net:6380"

az keyvault secret set \
  --vault-name prsm-keyvault \
  --name prsm-api-key \
  --value "your-secure-api-key"
```

### Key Vault Client

```python
# prsm/core/secrets_azure.py
import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import asyncio

class AzureKeyVaultManager:
    def __init__(self, vault_url=None):
        self.vault_url = vault_url or os.environ.get('AZURE_KEYVAULT_URL')
        self.credential = DefaultAzureCredential()
        self.client = SecretClient(
            vault_url=self.vault_url,
            credential=self.credential
        )

    def get_secret(self, secret_name):
        """Retrieve secret from Key Vault."""
        try:
            secret = self.client.get_secret(secret_name)
            return secret.value
        except Exception as e:
            raise Exception(f"Failed to retrieve secret {secret_name}: {str(e)}")

    def set_secret(self, secret_name, secret_value):
        """Create or update secret in Key Vault."""
        try:
            return self.client.set_secret(secret_name, secret_value)
        except Exception as e:
            raise Exception(f"Failed to set secret {secret_name}: {str(e)}")

    def list_secrets(self):
        """List all secrets in the vault."""
        try:
            secrets = []
            for secret in self.client.list_properties_of_secrets():
                secrets.append({
                    'name': secret.name,
                    'enabled': secret.enabled,
                    'created': secret.created_on,
                    'updated': secret.updated_on
                })
            return secrets
        except Exception as e:
            raise Exception(f"Failed to list secrets: {str(e)}")

    def delete_secret(self, secret_name):
        """Delete secret from Key Vault."""
        try:
            return self.client.begin_delete_secret(secret_name)
        except Exception as e:
            raise Exception(f"Failed to delete secret {secret_name}: {str(e)}")

# Initialize Key Vault manager
key_vault = AzureKeyVaultManager()

# Usage in application
def get_config_from_key_vault():
    return {
        'database_url': key_vault.get_secret('database-url'),
        'redis_url': key_vault.get_secret('redis-url'),
        'api_key': key_vault.get_secret('prsm-api-key')
    }
```

## 📊 Azure Monitor Integration

### Application Insights Setup

```bash
# Create Application Insights
az monitor app-insights component create \
  --app prsm-insights \
  --location eastus \
  --resource-group prsm-production \
  --kind web

# Get instrumentation key
az monitor app-insights component show \
  --app prsm-insights \
  --resource-group prsm-production \
  --query "instrumentationKey" \
  --output tsv
```

### Monitoring Configuration

```python
# prsm/monitoring/azure_monitoring.py
import os
import logging
import time
from datetime import datetime
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter, AzureMonitorLogExporter, AzureMonitorMetricExporter
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from applicationinsights import TelemetryClient

class AzureMonitoring:
    def __init__(self):
        self.instrumentation_key = os.environ.get('APPLICATIONINSIGHTS_CONNECTION_STRING')
        self.telemetry_client = TelemetryClient(self.instrumentation_key)
        self.setup_telemetry()

    def setup_telemetry(self):
        """Setup OpenTelemetry with Azure Monitor."""
        # Trace provider
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)

        # Azure Monitor exporters
        trace_exporter = AzureMonitorTraceExporter(
            connection_string=self.instrumentation_key
        )
        metric_exporter = AzureMonitorMetricExporter(
            connection_string=self.instrumentation_key
        )
        log_exporter = AzureMonitorLogExporter(
            connection_string=self.instrumentation_key
        )

        # Instrument libraries
        RequestsInstrumentor().instrument()
        SQLAlchemyInstrumentor().instrument()

    def track_prsm_query(self, user_id, prompt_length, response_time, success=True, confidence=None):
        """Track PRSM query metrics."""
        # Custom event
        self.telemetry_client.track_event(
            'prsm_query',
            {
                'user_id': user_id,
                'prompt_length': prompt_length,
                'success': success,
                'confidence': confidence or 0
            },
            {
                'response_time': response_time
            }
        )

        # Custom metric
        self.telemetry_client.track_metric(
            'prsm_response_time',
            response_time,
            properties={'user_id': user_id, 'success': str(success)}
        )

        if confidence:
            self.telemetry_client.track_metric(
                'prsm_confidence',
                confidence,
                properties={'user_id': user_id}
            )

    def track_error(self, error, context=None):
        """Track errors and exceptions."""
        self.telemetry_client.track_exception(
            type(error),
            error,
            error.__traceback__,
            properties=context or {}
        )

    def track_dependency(self, name, data, duration, success=True, result_code=None):
        """Track external dependencies."""
        self.telemetry_client.track_dependency(
            name=name,
            data=data,
            duration=duration,
            success=success,
            result_code=result_code
        )

    def flush(self):
        """Flush telemetry data."""
        self.telemetry_client.flush()

# Initialize monitoring
azure_monitoring = AzureMonitoring()

# Usage decorator
def monitor_prsm_query(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            azure_monitoring.track_prsm_query(
                user_id=kwargs.get('user_id', 'unknown'),
                prompt_length=len(kwargs.get('prompt', '')),
                response_time=duration,
                success=True,
                confidence=getattr(result, 'confidence', None)
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            azure_monitoring.track_prsm_query(
                user_id=kwargs.get('user_id', 'unknown'),
                prompt_length=len(kwargs.get('prompt', '')),
                response_time=duration,
                success=False
            )
            azure_monitoring.track_error(e, context={'function': func.__name__})
            raise
    return wrapper
```

### Custom Dashboards

```python
# prsm/monitoring/azure_dashboards.py
from azure.mgmt.dashboard import DashboardManagementClient
from azure.identity import DefaultAzureCredential
import json

class AzureDashboardManager:
    def __init__(self, subscription_id, resource_group):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.credential = DefaultAzureCredential()
        self.client = DashboardManagementClient(
            credential=self.credential,
            subscription_id=subscription_id
        )

    def create_prsm_dashboard(self):
        """Create Azure dashboard for PRSM monitoring."""
        dashboard_definition = {
            "properties": {
                "lenses": {
                    "0": {
                        "order": 0,
                        "parts": {
                            "0": {
                                "position": {"x": 0, "y": 0, "rowSpan": 4, "colSpan": 6},
                                "metadata": {
                                    "inputs": [
                                        {
                                            "name": "resourceTypeMode",
                                            "isOptional": True
                                        },
                                        {
                                            "name": "ComponentId",
                                            "value": {
                                                "SubscriptionId": self.subscription_id,
                                                "ResourceGroup": self.resource_group,
                                                "Name": "prsm-insights"
                                            }
                                        }
                                    ],
                                    "type": "Extension/AppInsightsExtension/PartType/AppMapGalPt"
                                }
                            },
                            "1": {
                                "position": {"x": 6, "y": 0, "rowSpan": 4, "colSpan": 6},
                                "metadata": {
                                    "inputs": [
                                        {
                                            "name": "query",
                                            "value": """
                                            customEvents
                                            | where name == "prsm_query"
                                            | summarize count() by bin(timestamp, 1h)
                                            | render timechart
                                            """
                                        }
                                    ],
                                    "type": "Extension/AppInsightsExtension/PartType/AnalyticsLineChartPart"
                                }
                            }
                        }
                    }
                },
                "metadata": {
                    "model": {
                        "timeRange": {
                            "value": {
                                "relative": {
                                    "duration": 24,
                                    "timeUnit": "hours"
                                }
                            }
                        }
                    }
                }
            },
            "name": "PRSM Production Dashboard",
            "type": "Microsoft.Portal/dashboards",
            "location": "global",
            "tags": {
                "hidden-title": "PRSM Production Dashboard"
            }
        }

        return self.client.dashboards.create_or_update(
            resource_group_name=self.resource_group,
            dashboard_name="prsm-dashboard",
            dashboard=dashboard_definition
        )

# Usage
dashboard_manager = AzureDashboardManager("your-subscription-id", "prsm-production")
dashboard_manager.create_prsm_dashboard()
```

## 🗄️ Azure Blob Storage

### Storage Configuration

```bash
# Create storage account
az storage account create \
  --name prsmstorageaccount \
  --resource-group prsm-production \
  --location eastus \
  --sku Standard_LRS \
  --kind StorageV2

# Create container
az storage container create \
  --name prsm-data \
  --account-name prsmstorageaccount \
  --public-access off
```

### Blob Storage Client

```python
# prsm/storage/azure_storage.py
import os
import json
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.identity import DefaultAzureCredential

class AzureBlobManager:
    def __init__(self):
        self.account_name = os.environ.get('AZURE_STORAGE_ACCOUNT')
        self.container_name = os.environ.get('AZURE_STORAGE_CONTAINER', 'prsm-data')
        
        # Use Managed Identity in Azure
        self.credential = DefaultAzureCredential()
        account_url = f"https://{self.account_name}.blob.core.windows.net"
        
        self.blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=self.credential
        )
        self.container_client = self.blob_service_client.get_container_client(
            self.container_name
        )

    def upload_conversation_log(self, user_id, conversation_data):
        """Upload conversation log to blob storage."""
        timestamp = datetime.utcnow().strftime("%Y/%m/%d")
        blob_name = f"conversations/{timestamp}/{user_id}/{datetime.utcnow().isoformat()}.json"
        
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        conversation_json = json.dumps(conversation_data, indent=2)
        blob_client.upload_blob(
            conversation_json,
            overwrite=True,
            metadata={
                'user_id': user_id,
                'content_type': 'conversation_log',
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        
        return blob_client.url

    def upload_model_artifacts(self, model_id, file_path):
        """Upload model artifacts to blob storage."""
        blob_name = f"models/{model_id}/artifacts.tar.gz"
        
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        with open(file_path, "rb") as data:
            blob_client.upload_blob(
                data,
                overwrite=True,
                metadata={
                    'model_id': model_id,
                    'content_type': 'model_artifacts',
                    'upload_time': datetime.utcnow().isoformat()
                }
            )
        
        return blob_client.url

    def generate_sas_url(self, blob_name, expiry_hours=1):
        """Generate SAS URL for secure blob access."""
        from azure.storage.blob import generate_blob_sas, BlobSasPermissions
        
        sas_token = generate_blob_sas(
            account_name=self.account_name,
            container_name=self.container_name,
            blob_name=blob_name,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=expiry_hours),
            credential=self.credential
        )
        
        return f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}?{sas_token}"

    def set_blob_lifecycle_policy(self):
        """Set lifecycle management policy."""
        lifecycle_policy = {
            "rules": [
                {
                    "enabled": True,
                    "name": "delete-old-conversations",
                    "type": "Lifecycle",
                    "definition": {
                        "filters": {
                            "blobTypes": ["blockBlob"],
                            "prefixMatch": ["conversations/"]
                        },
                        "actions": {
                            "baseBlob": {
                                "delete": {
                                    "daysAfterModificationGreaterThan": 365
                                }
                            }
                        }
                    }
                }
            ]
        }
        
        return lifecycle_policy

# Initialize blob manager
blob_manager = AzureBlobManager()
```

## 🚀 Azure Kubernetes Service (AKS)

### AKS Cluster Setup

```bash
# Create AKS cluster
az aks create \
  --resource-group prsm-production \
  --name prsm-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-addons monitoring \
  --enable-managed-identity \
  --enable-auto-scaler \
  --min-count 1 \
  --max-count 10 \
  --generate-ssh-keys

# Get AKS credentials
az aks get-credentials \
  --resource-group prsm-production \
  --name prsm-cluster
```

### Kubernetes Deployment

```yaml
# k8s/azure-deployment.yaml
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
        image: prsmregistry.azurecr.io/prsm-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: AZURE_CLIENT_ID
          value: "your-managed-identity-client-id"
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
        - name: AZURE_KEYVAULT_URL
          value: "https://prsm-keyvault.vault.azure.net/"
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
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: prsm-ingress
  annotations:
    kubernetes.io/ingress.class: azure/application-gateway
    appgw.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.prsm.ai
    secretName: prsm-tls-secret
  rules:
  - host: api.prsm.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: prsm-api-service
            port:
              number: 80
```

### Workload Identity Setup

```bash
# Create user-assigned managed identity
az identity create \
  --name prsm-workload-identity \
  --resource-group prsm-production

# Get identity client ID
IDENTITY_CLIENT_ID=$(az identity show \
  --name prsm-workload-identity \
  --resource-group prsm-production \
  --query clientId \
  --output tsv)

# Assign Key Vault permissions
az keyvault set-policy \
  --name prsm-keyvault \
  --object-id $(az identity show --name prsm-workload-identity --resource-group prsm-production --query principalId --output tsv) \
  --secret-permissions get list

# Create Kubernetes service account
kubectl create serviceaccount prsm-workload-identity

# Annotate service account
kubectl annotate serviceaccount prsm-workload-identity \
  azure.workload.identity/client-id=$IDENTITY_CLIENT_ID
```

## 📋 CI/CD with Azure DevOps

### Azure Pipeline Configuration

```yaml
# azure-pipelines.yml
trigger:
  branches:
    include:
    - main

variables:
  azureServiceConnection: 'azure-service-connection'
  containerRegistry: 'prsmregistry.azurecr.io'
  imageRepository: 'prsm-api'
  dockerfilePath: 'Dockerfile'
  tag: '$(Build.BuildId)'

stages:
- stage: Build
  displayName: Build and Push
  jobs:
  - job: Build
    displayName: Build
    pool:
      vmImage: 'ubuntu-latest'
    steps:
    - task: Docker@2
      displayName: Build and push image
      inputs:
        command: buildAndPush
        repository: $(imageRepository)
        dockerfile: $(dockerfilePath)
        containerRegistry: $(azureServiceConnection)
        tags: |
          $(tag)
          latest

- stage: Test
  displayName: Run Tests
  dependsOn: Build
  jobs:
  - job: Test
    displayName: Test
    pool:
      vmImage: 'ubuntu-latest'
    steps:
    - script: |
        docker run --rm $(containerRegistry)/$(imageRepository):$(tag) python -m pytest tests/ -v
      displayName: 'Run unit tests'

- stage: Deploy
  displayName: Deploy to Production
  dependsOn: Test
  condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
  jobs:
  - deployment: Deploy
    displayName: Deploy
    pool:
      vmImage: 'ubuntu-latest'
    environment: 'production'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: AzureContainerApps@1
            displayName: Deploy to Container Apps
            inputs:
              azureSubscription: $(azureServiceConnection)
              containerAppName: 'prsm-api'
              resourceGroup: 'prsm-production'
              imageToDeploy: '$(containerRegistry)/$(imageRepository):$(tag)'
```

### Build Pipeline Script

```bash
#!/bin/bash
# scripts/azure-deploy.sh

set -e

RESOURCE_GROUP="prsm-production"
ACR_NAME="prsmregistry"
APP_NAME="prsm-api"
IMAGE_TAG="${BUILD_BUILDID:-latest}"

echo "Building image..."
az acr build \
  --registry $ACR_NAME \
  --image prsm-api:$IMAGE_TAG \
  --image prsm-api:latest \
  .

echo "Running tests..."
docker run --rm ${ACR_NAME}.azurecr.io/prsm-api:$IMAGE_TAG python -m pytest tests/ -v

echo "Deploying to Container Apps..."
az containerapp update \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --image ${ACR_NAME}.azurecr.io/prsm-api:$IMAGE_TAG

echo "Checking deployment health..."
sleep 30
APP_URL=$(az containerapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query "properties.configuration.ingress.fqdn" -o tsv)
curl -f "https://$APP_URL/health" || exit 1

echo "Deployment successful!"
```

## 💰 Cost Management

### Azure Cost Optimization

```python
# prsm/optimization/azure_costs.py
from azure.mgmt.consumption import ConsumptionManagementClient
from azure.mgmt.billing import BillingManagementClient
from azure.identity import DefaultAzureCredential
import os

class AzureCostManager:
    def __init__(self, subscription_id=None):
        self.subscription_id = subscription_id or os.environ.get('AZURE_SUBSCRIPTION_ID')
        self.credential = DefaultAzureCredential()
        self.consumption_client = ConsumptionManagementClient(
            credential=self.credential,
            subscription_id=self.subscription_id
        )

    def get_current_month_costs(self):
        """Get current month costs by resource group."""
        scope = f"/subscriptions/{self.subscription_id}/resourceGroups/prsm-production"
        
        try:
            usage_details = self.consumption_client.usage_details.list(
                scope=scope,
                expand="properties/meterDetails"
            )
            
            costs = {}
            for usage in usage_details:
                service = usage.meter_details.meter_category
                cost = usage.cost
                costs[service] = costs.get(service, 0) + cost
            
            return costs
        except Exception as e:
            return {"error": str(e)}

    def create_budget_alert(self, budget_amount=500):
        """Create budget alert for cost control."""
        # This would use Azure Management APIs to create budgets
        budget_config = {
            "amount": budget_amount,
            "timeGrain": "Monthly",
            "timePeriod": {
                "startDate": "2024-01-01",
                "endDate": "2024-12-31"
            },
            "filters": {
                "resourceGroups": ["prsm-production"]
            },
            "notifications": {
                "notification1": {
                    "enabled": True,
                    "operator": "GreaterThan",
                    "threshold": 80,
                    "contactEmails": ["admin@company.com"]
                }
            }
        }
        return budget_config

    def optimize_recommendations(self):
        """Get cost optimization recommendations."""
        recommendations = []
        
        # Check for underutilized resources
        recommendations.append({
            "resource": "prsm-postgres",
            "current_tier": "Standard_D2s_v3",
            "recommended_tier": "Standard_B2s",
            "potential_savings": "40%"
        })
        
        recommendations.append({
            "resource": "prsm-redis",
            "current_tier": "Standard C1",
            "recommended_tier": "Basic C0",
            "potential_savings": "60%"
        })
        
        return recommendations

# Usage
cost_manager = AzureCostManager()
current_costs = cost_manager.get_current_month_costs()
print(f"Current month costs: {current_costs}")
```

## 🔧 Best Practices

### ARM Templates

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "location": {
      "type": "string",
      "defaultValue": "[resourceGroup().location]"
    },
    "appName": {
      "type": "string",
      "defaultValue": "prsm-api"
    }
  },
  "variables": {
    "containerAppEnvironmentName": "[concat(parameters('appName'), '-env')]",
    "storageAccountName": "[concat('prsmstorage', uniqueString(resourceGroup().id))]"
  },
  "resources": [
    {
      "type": "Microsoft.App/managedEnvironments",
      "apiVersion": "2022-03-01",
      "name": "[variables('containerAppEnvironmentName')]",
      "location": "[parameters('location')]",
      "properties": {
        "daprAIInstrumentationKey": "",
        "appLogsConfiguration": {
          "destination": "log-analytics"
        }
      }
    },
    {
      "type": "Microsoft.App/containerApps",
      "apiVersion": "2022-03-01",
      "name": "[parameters('appName')]",
      "location": "[parameters('location')]",
      "dependsOn": [
        "[resourceId('Microsoft.App/managedEnvironments', variables('containerAppEnvironmentName'))]"
      ],
      "properties": {
        "managedEnvironmentId": "[resourceId('Microsoft.App/managedEnvironments', variables('containerAppEnvironmentName'))]",
        "configuration": {
          "ingress": {
            "external": true,
            "targetPort": 8000
          }
        },
        "template": {
          "containers": [
            {
              "image": "prsmregistry.azurecr.io/prsm-api:latest",
              "name": "prsm-api",
              "resources": {
                "cpu": 2.0,
                "memory": "4Gi"
              }
            }
          ],
          "scale": {
            "minReplicas": 1,
            "maxReplicas": 10
          }
        }
      }
    }
  ],
  "outputs": {
    "fqdn": {
      "type": "string",
      "value": "[reference(resourceId('Microsoft.App/containerApps', parameters('appName'))).configuration.ingress.fqdn]"
    }
  }
}
```

### Security Configuration

```python
# prsm/security/azure_security.py
from azure.mgmt.security import SecurityCenter
from azure.identity import DefaultAzureCredential
import os

class AzureSecurity:
    def __init__(self):
        self.subscription_id = os.environ.get('AZURE_SUBSCRIPTION_ID')
        self.credential = DefaultAzureCredential()
        self.security_client = SecurityCenter(
            credential=self.credential,
            subscription_id=self.subscription_id
        )

    def configure_network_security(self):
        """Configure network security settings."""
        network_config = {
            "virtual_network": {
                "name": "prsm-vnet",
                "address_space": "10.0.0.0/16",
                "subnets": [
                    {
                        "name": "app-subnet",
                        "address_prefix": "10.0.1.0/24"
                    },
                    {
                        "name": "data-subnet",
                        "address_prefix": "10.0.2.0/24"
                    }
                ]
            },
            "network_security_groups": [
                {
                    "name": "prsm-app-nsg",
                    "rules": [
                        {
                            "name": "AllowHTTPS",
                            "priority": 100,
                            "direction": "Inbound",
                            "access": "Allow",
                            "protocol": "Tcp",
                            "source_port_range": "*",
                            "destination_port_range": "443",
                            "source_address_prefix": "*",
                            "destination_address_prefix": "*"
                        }
                    ]
                }
            ]
        }
        return network_config

    def setup_rbac_policies(self):
        """Setup Role-Based Access Control policies."""
        rbac_policies = {
            "roles": [
                {
                    "name": "PRSM Application Reader",
                    "permissions": [
                        "Microsoft.App/containerApps/read",
                        "Microsoft.KeyVault/vaults/secrets/read"
                    ]
                },
                {
                    "name": "PRSM Application Operator",
                    "permissions": [
                        "Microsoft.App/containerApps/*",
                        "Microsoft.KeyVault/vaults/secrets/*"
                    ]
                }
            ]
        }
        return rbac_policies

# Security configuration
security = AzureSecurity()
```

---

**Need help with Azure integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).