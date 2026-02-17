# PRSM Production Dockerfile
# Multi-stage build for optimized production image

# ===================================
# Stage 1: Build Environment
# ===================================
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_ENV=production
ARG PRSM_VERSION=0.1.0
ARG BUILDKIT_INLINE_CACHE=1

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building (optimized layer)
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create build directory
WORKDIR /build

# Copy dependency files
COPY requirements.txt requirements-dev.txt pyproject.toml ./

# Install Python dependencies with cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Build the package
RUN pip install -e .

# ===================================
# Stage 2: Production Environment  
# ===================================
FROM python:3.11-slim as production

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PRSM_ENV=production \
    PRSM_LOG_LEVEL=INFO \
    PRSM_WORKERS=4

# Install runtime dependencies (optimized)
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y \
    libpq5 \
    curl \
    wget \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r prsm \
    && useradd -r -g prsm prsm

# Create application directories
RUN mkdir -p /app/logs /app/data /app/config && \
    chown -R prsm:prsm /app

# Copy built application from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build /app

# Set working directory
WORKDIR /app

# Copy configuration files
COPY config/ /app/config/
COPY alembic.ini /app/

# Create healthcheck script
COPY <<EOF /app/healthcheck.py
#!/usr/bin/env python3
"""Health check script for PRSM container"""
import asyncio
import sys
import httpx
import os

async def check_health():
    """Check if PRSM API is responding"""
    try:
        port = os.getenv('PRSM_PORT', '8000')
        timeout = float(os.getenv('HEALTHCHECK_TIMEOUT', '10'))
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"http://localhost:{port}/health")
            
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                print("✅ PRSM API is healthy")
                return 0
            else:
                print(f"❌ PRSM API unhealthy: {data}")
                return 1
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(check_health()))
EOF

# Make healthcheck script executable
RUN chmod +x /app/healthcheck.py

# Switch to non-root user
USER prsm

# Expose application port
EXPOSE 8000

# Define volumes for persistent data
VOLUME ["/app/logs", "/app/data"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/healthcheck.py

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command - start PRSM in production mode
CMD ["uvicorn", "prsm.interface.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ===================================
# Stage 3: Development Environment
# ===================================
FROM production as development

# Switch back to root for development setup
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    strace \
    iputils-ping \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
COPY requirements-dev.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt

# Set development environment
ENV PRSM_ENV=development \
    PRSM_LOG_LEVEL=DEBUG \
    PRSM_RELOAD=true

# Development command with auto-reload
CMD ["uvicorn", "prsm.interface.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ===================================
# Labels and Metadata
# ===================================
LABEL maintainer="PRSM Team <team@prsm.org>" \
      org.opencontainers.image.title="PRSM" \
      org.opencontainers.image.description="Protocol for Recursive Scientific Modeling" \
      org.opencontainers.image.version="${PRSM_VERSION}" \
      org.opencontainers.image.vendor="PRSM Organization" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/Ryno2390/PRSM" \
      org.opencontainers.image.documentation="https://docs.prsm.org"