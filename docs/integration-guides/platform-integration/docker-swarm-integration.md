# Docker Swarm Integration Guide

Deploy and scale PRSM using Docker Swarm for container orchestration and clustering.

## 🎯 Overview

This guide covers deploying PRSM on Docker Swarm, including cluster setup, service orchestration, load balancing, and production best practices.

## 📋 Prerequisites

- Docker 20.10+ installed
- Multiple nodes for cluster (minimum 3 for HA)
- PRSM Docker image built
- Basic knowledge of Docker Swarm concepts

## 🚀 Quick Start

### 1. Initialize Swarm Cluster

```bash
# On manager node
docker swarm init --advertise-addr <MANAGER-IP>

# Get join tokens
docker swarm join-token worker
docker swarm join-token manager

# On worker nodes
docker swarm join --token <WORKER-TOKEN> <MANAGER-IP>:2377

# On additional manager nodes
docker swarm join --token <MANAGER-TOKEN> <MANAGER-IP>:2377
```

### 2. Create Networks

```bash
# Create overlay network for PRSM services
docker network create --driver overlay --attachable prsm-network

# Create network for database services
docker network create --driver overlay --attachable prsm-db-network
```

### 3. Basic Stack Deployment

```yaml
# docker-compose.swarm.yml
version: '3.8'

services:
  prsm-api:
    image: prsm/api:latest
    ports:
      - "8000:8000"
    networks:
      - prsm-network
      - prsm-db-network
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/prsm
      - REDIS_URL=redis://redis:6379/0
      - ENVIRONMENT=production
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      placement:
        constraints:
          - node.role == worker
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'

  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: prsm
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - prsm-db-network
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.database == true
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'

  redis:
    image: redis:7-alpine
    networks:
      - prsm-db-network
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.cache == true
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'

networks:
  prsm-network:
    external: true
  prsm-db-network:
    external: true

volumes:
  postgres_data:
```

### 4. Deploy Stack

```bash
# Deploy the stack
docker stack deploy -c docker-compose.swarm.yml prsm

# Check services
docker service ls

# Check stack status
docker stack ps prsm
```

## 🏗️ Production Configuration

### Advanced Stack Configuration

```yaml
# docker-compose.production.yml
version: '3.8'

x-default-logging: &logging
  driver: "json-file"
  options:
    max-size: "50m"
    max-file: "3"

services:
  traefik:
    image: traefik:v2.10
    command:
      - --api.dashboard=true
      - --api.insecure=false
      - --providers.docker=true
      - --providers.docker.swarmMode=true
      - --providers.docker.exposedbydefault=false
      - --entrypoints.web.address=:80
      - --entrypoints.websecure.address=:443
      - --certificatesresolvers.letsencrypt.acme.email=admin@example.com
      - --certificatesresolvers.letsencrypt.acme.storage=/letsencrypt/acme.json
      - --certificatesresolvers.letsencrypt.acme.httpchallenge.entrypoint=web
      - --metrics.prometheus=true
      - --metrics.prometheus.addEntryPointsLabels=true
      - --metrics.prometheus.addServicesLabels=true
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - traefik_certs:/letsencrypt
    networks:
      - prsm-network
    deploy:
      mode: global
      placement:
        constraints:
          - node.role == manager
      labels:
        - traefik.enable=true
        - traefik.http.routers.traefik.rule=Host(`traefik.example.com`)
        - traefik.http.routers.traefik.tls.certresolver=letsencrypt
        - traefik.http.services.traefik.loadbalancer.server.port=8080
    logging: *logging

  prsm-api:
    image: prsm/api:latest
    networks:
      - prsm-network
      - prsm-db-network
    environment:
      - DATABASE_URL_FILE=/run/secrets/database_url
      - REDIS_URL_FILE=/run/secrets/redis_url
      - API_KEY_FILE=/run/secrets/api_key
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - WORKERS=4
    secrets:
      - database_url
      - redis_url
      - api_key
    deploy:
      replicas: 5
      update_config:
        parallelism: 2
        delay: 30s
        failure_action: rollback
        monitor: 60s
        max_failure_ratio: 0.2
      rollback_config:
        parallelism: 2
        delay: 30s
        monitor: 60s
      restart_policy:
        condition: on-failure
        delay: 10s
        max_attempts: 5
        window: 120s
      placement:
        constraints:
          - node.role == worker
          - node.labels.tier == application
        preferences:
          - spread: node.labels.zone
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
      labels:
        - traefik.enable=true
        - traefik.http.routers.prsm-api.rule=Host(`api.prsm.ai`)
        - traefik.http.routers.prsm-api.tls.certresolver=letsencrypt
        - traefik.http.services.prsm-api.loadbalancer.server.port=8000
        - traefik.http.services.prsm-api.loadbalancer.healthcheck.path=/health
        - traefik.http.services.prsm-api.loadbalancer.healthcheck.interval=30s
        - traefik.http.services.prsm-api.loadbalancer.healthcheck.timeout=10s
    logging: *logging
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: prsm
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
      POSTGRES_INITDB_ARGS: "--encoding=UTF8 --locale=C"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - postgres_backup:/backup
      - ./postgres/init:/docker-entrypoint-initdb.d:ro
    networks:
      - prsm-db-network
    secrets:
      - postgres_password
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.database == true
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
        reservations:
          memory: 4G
          cpus: '2.0'
    logging: *logging
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    networks:
      - prsm-db-network
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.cache == true
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
    logging: *logging
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - nginx_cache:/var/cache/nginx
    networks:
      - prsm-network
    deploy:
      replicas: 2
      placement:
        constraints:
          - node.role == worker
      labels:
        - traefik.enable=true
        - traefik.http.routers.nginx.rule=Host(`static.prsm.ai`)
        - traefik.http.routers.nginx.tls.certresolver=letsencrypt
        - traefik.http.services.nginx.loadbalancer.server.port=80
    logging: *logging

  prometheus:
    image: prom/prometheus:latest
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=15d'
      - '--web.enable-lifecycle'
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    networks:
      - prsm-network
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.monitoring == true
      labels:
        - traefik.enable=true
        - traefik.http.routers.prometheus.rule=Host(`prometheus.prsm.ai`)
        - traefik.http.routers.prometheus.tls.certresolver=letsencrypt
        - traefik.http.services.prometheus.loadbalancer.server.port=9090
    logging: *logging

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD_FILE=/run/secrets/grafana_password
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
    networks:
      - prsm-network
    secrets:
      - grafana_password
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.monitoring == true
      labels:
        - traefik.enable=true
        - traefik.http.routers.grafana.rule=Host(`grafana.prsm.ai`)
        - traefik.http.routers.grafana.tls.certresolver=letsencrypt
        - traefik.http.services.grafana.loadbalancer.server.port=3000
    logging: *logging

secrets:
  database_url:
    external: true
  redis_url:
    external: true
  api_key:
    external: true
  postgres_password:
    external: true
  grafana_password:
    external: true

networks:
  prsm-network:
    external: true
  prsm-db-network:
    external: true

volumes:
  postgres_data:
  postgres_backup:
  redis_data:
  nginx_cache:
  prometheus_data:
  grafana_data:
  traefik_certs:
```

## 🔐 Secrets Management

### Creating Secrets

```bash
# Create secrets for the stack
echo "postgresql://postgres:securepassword@postgres:5432/prsm" | docker secret create database_url -
echo "redis://redis:6379/0" | docker secret create redis_url -
echo "your-secure-api-key" | docker secret create api_key -
echo "secure-postgres-password" | docker secret create postgres_password -
echo "secure-grafana-password" | docker secret create grafana_password -

# List secrets
docker secret ls
```

### Secrets Rotation

```bash
#!/bin/bash
# scripts/rotate-secrets.sh

rotate_secret() {
    local secret_name=$1
    local new_value=$2
    
    echo "Rotating secret: $secret_name"
    
    # Create new secret with timestamp
    local new_secret_name="${secret_name}_$(date +%s)"
    echo "$new_value" | docker secret create "$new_secret_name" -
    
    # Update service to use new secret
    docker service update --secret-rm "$secret_name" --secret-add "$new_secret_name" prsm_prsm-api
    
    # Remove old secret
    docker secret rm "$secret_name"
    
    # Rename new secret
    docker secret create "$secret_name" < <(docker secret inspect "$new_secret_name" --format='{{.Spec.Data}}' | base64 -d)
    docker secret rm "$new_secret_name"
    
    echo "Secret $secret_name rotated successfully"
}

# Usage
# rotate_secret "api_key" "new-secure-api-key"
```

## 📊 Monitoring and Logging

### Prometheus Configuration

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'docker-swarm-nodes'
    dockerswarm_sd_configs:
      - host: unix:///var/run/docker.sock
        role: nodes
    relabel_configs:
      - source_labels: [__meta_dockerswarm_node_address]
        target_label: __address__
        replacement: ${1}:9100

  - job_name: 'docker-swarm-services'
    dockerswarm_sd_configs:
      - host: unix:///var/run/docker.sock
        role: services
    relabel_configs:
      - source_labels: [__meta_dockerswarm_service_label_prometheus_job]
        target_label: job
      - source_labels: [__meta_dockerswarm_service_label_prometheus_port]
        target_label: __address__
        replacement: ${__meta_dockerswarm_service_name}:${1}

  - job_name: 'traefik'
    static_configs:
      - targets: ['traefik:8080']

  - job_name: 'prsm-api'
    dns_sd_configs:
      - names: ['tasks.prsm_prsm-api']
        port: 8000
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### ELK Stack Integration

```yaml
# elk-stack.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    environment:
      - node.name=elasticsearch
      - cluster.name=prsm-logs
      - discovery.type=single-node
      - bootstrap.memory_lock=true
      - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
      - xpack.security.enabled=false
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    networks:
      - prsm-network
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.logging == true
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

  logstash:
    image: docker.elastic.co/logstash/logstash:8.5.0
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline:ro
      - ./logstash/config:/usr/share/logstash/config:ro
    networks:
      - prsm-network
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.logging == true
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.5.0
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    networks:
      - prsm-network
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.logging == true
      labels:
        - traefik.enable=true
        - traefik.http.routers.kibana.rule=Host(`kibana.prsm.ai`)
        - traefik.http.services.kibana.loadbalancer.server.port=5601
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:

networks:
  prsm-network:
    external: true
```

## 🔄 Auto-scaling and Load Balancing

### Service Auto-scaling

```bash
#!/bin/bash
# scripts/autoscale.sh

# Auto-scaling based on CPU and memory metrics
autoscale_service() {
    local service_name=$1
    local min_replicas=${2:-2}
    local max_replicas=${3:-10}
    local cpu_threshold=${4:-70}
    local memory_threshold=${5:-80}
    
    # Get current metrics
    local current_replicas=$(docker service inspect --format='{{.Spec.Mode.Replicated.Replicas}}' "$service_name")
    local avg_cpu=$(docker stats --no-stream --format "table {{.CPUPerc}}" | grep -v CPU | sed 's/%//' | awk '{sum+=$1} END {print sum/NR}')
    local avg_memory=$(docker stats --no-stream --format "table {{.MemPerc}}" | grep -v MEM | sed 's/%//' | awk '{sum+=$1} END {print sum/NR}')
    
    echo "Current replicas: $current_replicas"
    echo "Average CPU: $avg_cpu%"
    echo "Average Memory: $avg_memory%"
    
    # Scale up conditions
    if (( $(echo "$avg_cpu > $cpu_threshold" | bc -l) )) || (( $(echo "$avg_memory > $memory_threshold" | bc -l) )); then
        if [ "$current_replicas" -lt "$max_replicas" ]; then
            local new_replicas=$((current_replicas + 1))
            echo "Scaling up $service_name to $new_replicas replicas"
            docker service scale "$service_name=$new_replicas"
        fi
    # Scale down conditions
    elif (( $(echo "$avg_cpu < 30" | bc -l) )) && (( $(echo "$avg_memory < 40" | bc -l) )); then
        if [ "$current_replicas" -gt "$min_replicas" ]; then
            local new_replicas=$((current_replicas - 1))
            echo "Scaling down $service_name to $new_replicas replicas"
            docker service scale "$service_name=$new_replicas"
        fi
    fi
}

# Monitor and scale every 2 minutes
while true; do
    autoscale_service "prsm_prsm-api" 3 15 70 80
    sleep 120
done
```

### Load Balancer Configuration

```nginx
# nginx/nginx.conf
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log notice;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;
    
    upstream prsm_api {
        least_conn;
        server prsm_prsm-api:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }
    
    server {
        listen 80;
        server_name api.prsm.ai;
        
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
        
        location /api/v1/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://prsm_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 5s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
            
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
            
            proxy_http_version 1.1;
            proxy_set_header Connection "";
        }
        
        location /auth/ {
            limit_req zone=login burst=5 nodelay;
            
            proxy_pass http://prsm_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

## 🔧 Node Management

### Node Labeling

```bash
#!/bin/bash
# scripts/setup-nodes.sh

# Label nodes for specific workloads
docker node update --label-add tier=application worker-1
docker node update --label-add tier=application worker-2
docker node update --label-add tier=application worker-3

docker node update --label-add database=true worker-4
docker node update --label-add cache=true worker-5
docker node update --label-add monitoring=true worker-6
docker node update --label-add logging=true worker-6

# Set node availability
docker node update --availability active worker-1
docker node update --availability active worker-2
docker node update --availability active worker-3

# Configure resource limits
docker node update --label-add cpu=high worker-1
docker node update --label-add memory=high worker-2
docker node update --label-add storage=high worker-4
```

### Health Monitoring

```bash
#!/bin/bash
# scripts/monitor-cluster.sh

monitor_cluster_health() {
    echo "=== Docker Swarm Cluster Health ==="
    echo "Date: $(date)"
    echo
    
    # Node status
    echo "Node Status:"
    docker node ls
    echo
    
    # Service status
    echo "Service Status:"
    docker service ls
    echo
    
    # Stack status
    echo "Stack Status:"
    docker stack ps prsm --no-trunc
    echo
    
    # Resource usage
    echo "Resource Usage:"
    docker system df
    echo
    
    # Check for unhealthy services
    unhealthy_services=$(docker service ls --filter "health=unhealthy" --quiet)
    if [ ! -z "$unhealthy_services" ]; then
        echo "WARNING: Unhealthy services detected:"
        docker service ls --filter "health=unhealthy"
        
        # Send alert
        send_alert "Unhealthy services detected in Docker Swarm cluster"
    fi
    
    # Check node availability
    unavailable_nodes=$(docker node ls --filter "availability=drain" --quiet)
    if [ ! -z "$unavailable_nodes" ]; then
        echo "WARNING: Unavailable nodes detected:"
        docker node ls --filter "availability=drain"
    fi
}

send_alert() {
    local message=$1
    # Send to Slack, email, or other alerting system
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"$message\"}" \
        "$SLACK_WEBHOOK_URL"
}

# Run monitoring
monitor_cluster_health

# Log to file
monitor_cluster_health >> /var/log/swarm-health.log 2>&1
```

## 📋 Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# scripts/backup-database.sh

set -e

BACKUP_DIR="/backup"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="prsm_backup_$DATE.sql"

# Create backup
docker exec $(docker ps -q -f name=prsm_postgres) \
    pg_dump -U postgres prsm > "$BACKUP_DIR/$BACKUP_FILE"

# Compress backup
gzip "$BACKUP_DIR/$BACKUP_FILE"

# Upload to S3 (optional)
if [ ! -z "$AWS_S3_BUCKET" ]; then
    aws s3 cp "$BACKUP_DIR/$BACKUP_FILE.gz" \
        "s3://$AWS_S3_BUCKET/backups/database/$BACKUP_FILE.gz"
fi

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -name "prsm_backup_*.sql.gz" -mtime +7 -delete

echo "Database backup completed: $BACKUP_FILE.gz"
```

### Disaster Recovery

```bash
#!/bin/bash
# scripts/disaster-recovery.sh

restore_from_backup() {
    local backup_file=$1
    
    echo "Starting disaster recovery..."
    echo "Backup file: $backup_file"
    
    # Stop services
    docker service scale prsm_prsm-api=0
    
    # Restore database
    if [ -f "$backup_file" ]; then
        echo "Restoring database..."
        docker exec -i $(docker ps -q -f name=prsm_postgres) \
            psql -U postgres -d prsm < "$backup_file"
    else
        echo "Backup file not found: $backup_file"
        exit 1
    fi
    
    # Restart services
    docker service scale prsm_prsm-api=3
    
    # Wait for services to be healthy
    echo "Waiting for services to be healthy..."
    sleep 60
    
    # Verify recovery
    if curl -f http://localhost:8000/health; then
        echo "Disaster recovery completed successfully"
    else
        echo "Disaster recovery failed - services not healthy"
        exit 1
    fi
}

# Usage: ./disaster-recovery.sh /backup/prsm_backup_20240101_120000.sql
restore_from_backup "$1"
```

## 🚀 Deployment Scripts

### Production Deployment

```bash
#!/bin/bash
# scripts/deploy-production.sh

set -e

IMAGE_TAG=${1:-latest}
STACK_NAME="prsm"

echo "Deploying PRSM Stack to Production"
echo "Image tag: $IMAGE_TAG"
echo "Stack name: $STACK_NAME"

# Pre-deployment checks
check_cluster_health() {
    echo "Checking cluster health..."
    
    # Check if we have enough nodes
    active_nodes=$(docker node ls --filter "availability=active" --quiet | wc -l)
    if [ "$active_nodes" -lt 3 ]; then
        echo "Error: Not enough active nodes ($active_nodes). Need at least 3."
        exit 1
    fi
    
    # Check if all services are healthy
    unhealthy=$(docker service ls --filter "health=unhealthy" --quiet | wc -l)
    if [ "$unhealthy" -gt 0 ]; then
        echo "Warning: $unhealthy unhealthy services detected"
        docker service ls --filter "health=unhealthy"
    fi
}

# Update configuration
update_image_tag() {
    sed -i "s|image: prsm/api:.*|image: prsm/api:$IMAGE_TAG|g" docker-compose.production.yml
}

# Deploy stack
deploy_stack() {
    echo "Deploying stack..."
    docker stack deploy -c docker-compose.production.yml "$STACK_NAME"
    
    echo "Waiting for deployment to stabilize..."
    sleep 60
    
    # Check deployment status
    docker stack ps "$STACK_NAME" --no-trunc
}

# Health check
health_check() {
    echo "Performing health check..."
    
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s http://localhost:8000/health > /dev/null; then
            echo "Health check passed!"
            return 0
        fi
        
        echo "Health check attempt $attempt/$max_attempts failed, retrying..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    echo "Health check failed after $max_attempts attempts"
    return 1
}

# Main deployment flow
main() {
    check_cluster_health
    update_image_tag
    deploy_stack
    
    if health_check; then
        echo "✅ Deployment successful!"
        
        # Tag successful deployment
        git tag "production-$(date +%Y%m%d-%H%M%S)"
        
        # Send success notification
        send_notification "PRSM production deployment successful" "success"
    else
        echo "❌ Deployment failed!"
        
        # Rollback
        echo "Rolling back to previous version..."
        docker service rollback "${STACK_NAME}_prsm-api"
        
        # Send failure notification
        send_notification "PRSM production deployment failed" "error"
        exit 1
    fi
}

send_notification() {
    local message=$1
    local status=$2
    
    # Send to Slack or other notification system
    if [ ! -z "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\", \"username\":\"DeployBot\", \"icon_emoji\":\":robot_face:\"}" \
            "$SLACK_WEBHOOK_URL"
    fi
}

# Run deployment
main "$@"
```

---

**Need help with Docker Swarm integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).