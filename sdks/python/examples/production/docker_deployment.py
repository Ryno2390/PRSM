#!/usr/bin/env python3
"""
PRSM Python SDK - Docker Deployment Example

This example demonstrates how to containerize and deploy PRSM-powered applications
using Docker, including multi-stage builds, health checks, and production configurations.
"""

import os
import sys
from pathlib import Path

# Create Docker configuration files
def create_dockerfile():
    """Create production Dockerfile for PRSM application"""
    dockerfile_content = """# Multi-stage Docker build for PRSM Python application
# Stage 1: Build dependencies
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Stage 2: Production runtime
FROM python:3.11-slim as runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r prsm && useradd -r -g prsm prsm

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Copy application code
COPY --chown=prsm:prsm . .

# Switch to non-root user
USER prsm

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    print("✅ Created Dockerfile")


def create_docker_compose():
    """Create Docker Compose configuration for development and production"""
    
    # Development compose file
    dev_compose = """version: '3.8'

services:
  prsm-app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - PRSM_API_KEY=${PRSM_API_KEY}
      - PRSM_BASE_URL=${PRSM_BASE_URL:-https://api.prsm.ai}
      - LOG_LEVEL=info
      - ENVIRONMENT=development
    volumes:
      - .:/app
      - /app/__pycache__
    command: >
      sh -c "pip install -e . &&
             python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
    depends_on:
      - redis
      - postgres
    networks:
      - prsm-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - prsm-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: prsm_app
      POSTGRES_USER: prsm
      POSTGRES_PASSWORD: prsm_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - prsm-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U prsm -d prsm_app"]
      interval: 10s
      timeout: 5s
      retries: 5

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - prsm-app
    networks:
      - prsm-network

volumes:
  redis_data:
  postgres_data:

networks:
  prsm-network:
    driver: bridge
"""
    
    # Production compose file
    prod_compose = """version: '3.8'

services:
  prsm-app:
    image: prsm-app:latest
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
    environment:
      - PRSM_API_KEY=${PRSM_API_KEY}
      - PRSM_BASE_URL=${PRSM_BASE_URL}
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - LOG_LEVEL=warning
      - ENVIRONMENT=production
      - WORKERS=4
    ports:
      - "8000"
    networks:
      - prsm-network
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.prod.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
      - ./logs:/var/log/nginx
    depends_on:
      - prsm-app
    networks:
      - prsm-network
    deploy:
      replicas: 2
      restart_policy:
        condition: on-failure

networks:
  prsm-network:
    external: true

secrets:
  prsm_api_key:
    external: true
"""
    
    with open("docker-compose.dev.yml", "w") as f:
        f.write(dev_compose)
    
    with open("docker-compose.prod.yml", "w") as f:
        f.write(prod_compose)
    
    print("✅ Created Docker Compose files")


def create_nginx_config():
    """Create Nginx configuration for load balancing and SSL termination"""
    
    nginx_config = """events {
    worker_connections 1024;
}

http {
    upstream prsm_app {
        least_conn;
        server prsm-app:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=5r/s;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;

    server {
        listen 80;
        server_name _;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name _;

        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
        ssl_prefer_server_ciphers off;

        # Client body size limit
        client_max_body_size 10M;

        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;

        # Health check endpoint (no rate limiting)
        location /health {
            proxy_pass http://prsm_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # API endpoints with rate limiting
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://prsm_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-Request-ID $request_id;
            
            # Handle streaming responses
            proxy_buffering off;
            proxy_cache off;
        }

        # Authentication endpoints with stricter rate limiting
        location /auth/ {
            limit_req zone=auth burst=10 nodelay;
            
            proxy_pass http://prsm_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Static files (if any)
        location /static/ {
            expires 30d;
            add_header Cache-Control "public, immutable";
        }

        # Default location
        location / {
            proxy_pass http://prsm_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
"""
    
    with open("nginx.conf", "w") as f:
        f.write(nginx_config)
    print("✅ Created Nginx configuration")


def create_requirements():
    """Create requirements.txt for Docker build"""
    requirements = """# PRSM SDK and core dependencies
prsm-python-sdk>=1.0.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0

# Production server
gunicorn>=21.2.0

# Database
sqlalchemy>=2.0.0
alembic>=1.12.0
psycopg2-binary>=2.9.0

# Caching
redis>=5.0.0
hiredis>=2.2.0

# Monitoring and logging
structlog>=23.1.0
prometheus-client>=0.19.0
sentry-sdk[fastapi]>=1.38.0

# Security
cryptography>=41.0.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4

# HTTP client
httpx>=0.25.0
aiofiles>=23.2.1

# Development tools (remove in production)
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.9.0
ruff>=0.1.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("✅ Created requirements.txt")


def create_deployment_scripts():
    """Create deployment and management scripts"""
    
    # Build script
    build_script = """#!/bin/bash
set -e

echo "🔨 Building PRSM application Docker image..."

# Build the image
docker build -t prsm-app:latest .

# Tag for different environments
docker tag prsm-app:latest prsm-app:$(git rev-parse --short HEAD)

echo "✅ Build completed successfully!"
echo "📋 Available tags:"
docker images prsm-app --format "table {{.Repository}}:{{.Tag}}\\t{{.CreatedAt}}\\t{{.Size}}"
"""
    
    # Deploy script
    deploy_script = """#!/bin/bash
set -e

ENVIRONMENT=${1:-development}
COMPOSE_FILE="docker-compose.${ENVIRONMENT}.yml"

echo "🚀 Deploying PRSM application to ${ENVIRONMENT}..."

# Check if compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "❌ Compose file $COMPOSE_FILE not found"
    exit 1
fi

# Load environment variables
if [ -f ".env.${ENVIRONMENT}" ]; then
    set -a
    source ".env.${ENVIRONMENT}"
    set +a
    echo "✅ Loaded environment variables from .env.${ENVIRONMENT}"
fi

# Deploy based on environment
case $ENVIRONMENT in
    "development")
        docker-compose -f $COMPOSE_FILE up --build -d
        ;;
    "production")
        # Pull latest images
        docker-compose -f $COMPOSE_FILE pull
        
        # Deploy with zero downtime
        docker-compose -f $COMPOSE_FILE up -d --remove-orphans
        
        # Clean up old images
        docker image prune -f
        ;;
    *)
        echo "❌ Unknown environment: $ENVIRONMENT"
        echo "Usage: $0 [development|production]"
        exit 1
        ;;
esac

echo "✅ Deployment completed!"

# Show running services
echo "📋 Running services:"
docker-compose -f $COMPOSE_FILE ps
"""
    
    # Health check script
    health_script = """#!/bin/bash
set -e

COMPOSE_FILE=${1:-docker-compose.dev.yml}
MAX_ATTEMPTS=30
ATTEMPT=1

echo "🔍 Checking application health..."

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo "Attempt $ATTEMPT/$MAX_ATTEMPTS..."
    
    # Check if services are running
    if docker-compose -f $COMPOSE_FILE ps | grep -q "Up"; then
        # Check application health endpoint
        if curl -f -s http://localhost:8000/health > /dev/null; then
            echo "✅ Application is healthy!"
            
            # Show service status
            echo "📋 Service status:"
            docker-compose -f $COMPOSE_FILE ps
            
            # Show logs for debugging
            echo "📝 Recent logs:"
            docker-compose -f $COMPOSE_FILE logs --tail=10 prsm-app
            
            exit 0
        fi
    fi
    
    sleep 5
    ATTEMPT=$((ATTEMPT + 1))
done

echo "❌ Health check failed after $MAX_ATTEMPTS attempts"
echo "📝 Service logs:"
docker-compose -f $COMPOSE_FILE logs prsm-app
exit 1
"""
    
    # Make scripts executable and save them
    scripts = {
        "build.sh": build_script,
        "deploy.sh": deploy_script,
        "health-check.sh": health_script
    }
    
    for filename, content in scripts.items():
        with open(filename, "w") as f:
            f.write(content)
        os.chmod(filename, 0o755)
    
    print("✅ Created deployment scripts")


def create_environment_files():
    """Create environment configuration files"""
    
    # Development environment
    dev_env = """# Development Environment Configuration
PRSM_API_KEY=your_prsm_api_key_here
PRSM_BASE_URL=https://api.prsm.ai
DATABASE_URL=postgresql://prsm:prsm_password@postgres:5432/prsm_app
REDIS_URL=redis://redis:6379/0

# Application settings
ENVIRONMENT=development
LOG_LEVEL=debug
DEBUG=true

# Security (use strong values in production)
SECRET_KEY=dev_secret_key_change_in_production
ALLOWED_HOSTS=localhost,127.0.0.1

# Feature flags
ENABLE_METRICS=true
ENABLE_TRACING=false
"""
    
    # Production environment template
    prod_env = """# Production Environment Configuration
# IMPORTANT: Replace all placeholder values with actual production values

PRSM_API_KEY=your_production_prsm_api_key
PRSM_BASE_URL=https://api.prsm.ai
DATABASE_URL=postgresql://user:pass@db-host:5432/dbname
REDIS_URL=redis://redis-host:6379/0

# Application settings
ENVIRONMENT=production
LOG_LEVEL=warning
DEBUG=false
WORKERS=4

# Security (MUST be changed for production)
SECRET_KEY=generate_a_secure_random_secret_key
ALLOWED_HOSTS=yourdomain.com,api.yourdomain.com

# SSL/TLS
SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
SSL_KEY_PATH=/etc/nginx/ssl/key.pem

# Monitoring
ENABLE_METRICS=true
ENABLE_TRACING=true
SENTRY_DSN=your_sentry_dsn_for_error_tracking

# Rate limiting
RATE_LIMIT_PER_MINUTE=100
BURST_LIMIT=20
"""
    
    with open(".env.development", "w") as f:
        f.write(dev_env)
    
    with open(".env.production.template", "w") as f:
        f.write(prod_env)
    
    print("✅ Created environment configuration files")


def create_kubernetes_manifests():
    """Create Kubernetes deployment manifests"""
    
    # Namespace
    namespace_yaml = """apiVersion: v1
kind: Namespace
metadata:
  name: prsm-app
  labels:
    name: prsm-app
"""
    
    # ConfigMap
    configmap_yaml = """apiVersion: v1
kind: ConfigMap
metadata:
  name: prsm-app-config
  namespace: prsm-app
data:
  PRSM_BASE_URL: "https://api.prsm.ai"
  ENVIRONMENT: "production"
  LOG_LEVEL: "info"
  WORKERS: "4"
"""
    
    # Secret (template)
    secret_yaml = """apiVersion: v1
kind: Secret
metadata:
  name: prsm-app-secrets
  namespace: prsm-app
type: Opaque
data:
  # Base64 encoded values
  # Use: echo -n 'your_secret' | base64
  PRSM_API_KEY: eW91cl9wcnNtX2FwaV9rZXk=  # Replace with actual base64 encoded key
  SECRET_KEY: eW91cl9zZWNyZXRfa2V5  # Replace with actual base64 encoded secret
"""
    
    # Deployment
    deployment_yaml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: prsm-app
  namespace: prsm-app
  labels:
    app: prsm-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: prsm-app
  template:
    metadata:
      labels:
        app: prsm-app
    spec:
      containers:
      - name: prsm-app
        image: prsm-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: PRSM_API_KEY
          valueFrom:
            secretKeyRef:
              name: prsm-app-secrets
              key: PRSM_API_KEY
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: prsm-app-secrets
              key: SECRET_KEY
        envFrom:
        - configMapRef:
            name: prsm-app-config
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
"""
    
    # Service
    service_yaml = """apiVersion: v1
kind: Service
metadata:
  name: prsm-app-service
  namespace: prsm-app
spec:
  selector:
    app: prsm-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP
"""
    
    # Ingress
    ingress_yaml = """apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: prsm-app-ingress
  namespace: prsm-app
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: prsm-app-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: prsm-app-service
            port:
              number: 80
"""
    
    # Create k8s directory and files
    k8s_dir = Path("k8s")
    k8s_dir.mkdir(exist_ok=True)
    
    k8s_files = {
        "namespace.yaml": namespace_yaml,
        "configmap.yaml": configmap_yaml,
        "secret.yaml": secret_yaml,
        "deployment.yaml": deployment_yaml,
        "service.yaml": service_yaml,
        "ingress.yaml": ingress_yaml
    }
    
    for filename, content in k8s_files.items():
        with open(k8s_dir / filename, "w") as f:
            f.write(content)
    
    print("✅ Created Kubernetes manifests in k8s/ directory")


def main():
    """Create all Docker deployment files"""
    print("🐳 Creating Docker deployment configuration for PRSM application...")
    print("=" * 60)
    
    try:
        create_dockerfile()
        create_docker_compose()
        create_nginx_config()
        create_requirements()
        create_deployment_scripts()
        create_environment_files()
        create_kubernetes_manifests()
        
        print("\n" + "=" * 60)
        print("✅ Docker deployment configuration created successfully!")
        print("\n📋 Next steps:")
        print("1. Set your PRSM_API_KEY in .env.development")
        print("2. Run: ./build.sh")
        print("3. Run: ./deploy.sh development")
        print("4. Check health: ./health-check.sh")
        print("\n🔒 For production:")
        print("1. Copy .env.production.template to .env.production")
        print("2. Update all production values in .env.production")
        print("3. Run: ./deploy.sh production")
        print("\n☸️ For Kubernetes:")
        print("1. Update k8s/secret.yaml with your base64 encoded secrets")
        print("2. Update k8s/ingress.yaml with your domain")
        print("3. Run: kubectl apply -f k8s/")
        
    except Exception as e:
        print(f"❌ Error creating deployment files: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()